"""Transformers-based activation distribution probe.

Loads model in main process (no vLLM multi-proc), attaches forward hooks to
every decoder layer's self_attn and mlp submodule, replays prompts from a
trace, and records output-tensor stats per (iteration, layer, kind).

Output: JSON with shape:
  {
    "model": str, "n_layers": int, "n_iterations": int,
    "iterations": [
      {
        "iter": int, "input_tokens": int,
        "stats": [
          {"layer": int, "kind": "attn"|"mlp",
           "mean": float, "std": float,
           "abs_mean": float, "abs_max": float,
           "l2_norm": float, "sparsity_1e-3": float}
        ]
      }, ...
    ]
  }
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Suppress tqdm/HF noise
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch import nn


def _msg_size(messages: list[dict[str, Any]]) -> int:
    return sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)


def _flatten_to_chat(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Convert trace-format messages to HF chat-template input. Tool messages
    are flattened to assistant-text wrapping."""
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(c.get("text", c)) for c in content)
        if role in ("system", "user", "assistant"):
            out.append({"role": role, "content": str(content)})
        else:
            out.append({"role": "assistant", "content": f"[{role}] {content}"})
    return out


def stats_from_tensor(x: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        x32 = x.detach().to(torch.float32)
        absx = x32.abs()
        return {
            "mean": float(x32.mean().item()),
            "std": float(x32.std(unbiased=False).item()),
            "abs_mean": float(absx.mean().item()),
            "abs_max": float(absx.max().item()),
            "l2_norm": float(x32.norm().item()),
            "sparsity_1e-3": float((absx < 1e-3).float().mean().item()),
        }


def attach_hooks(model: nn.Module) -> tuple[list[Any], dict[int, dict[str, dict[str, float]]]]:
    """Attach hooks. Returns (handles, capture-dict).

    capture[layer_idx] = {"attn": stats_dict, "mlp": stats_dict} populated by hooks.
    """
    handles: list[Any] = []
    capture: dict[int, dict[str, dict[str, float]]] = {}

    # Identify decoder layers by the presence of `self_attn` and `mlp`
    # submodules — model-architecture-agnostic enough for Qwen / Llama / Mistral
    layer_idx = 0
    for name, m in model.named_modules():
        if hasattr(m, "self_attn") and hasattr(m, "mlp"):
            attn = getattr(m, "self_attn")
            mlp = getattr(m, "mlp")
            idx = layer_idx
            layer_idx += 1

            def make_hook(li, kind):
                def hook(module, inputs, output):
                    # output may be tensor OR tuple (attn returns tuple in some impls)
                    out = output[0] if isinstance(output, (tuple, list)) else output
                    if isinstance(out, torch.Tensor):
                        capture.setdefault(li, {})[kind] = stats_from_tensor(out)
                return hook

            handles.append(attn.register_forward_hook(make_hook(idx, "attn")))
            handles.append(mlp.register_forward_hook(make_hook(idx, "mlp")))

    return handles, capture


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--trace", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--max-iters", type=int, default=10)
    p.add_argument("--max-input-tokens", type=int, default=8192,
                   help="Skip prompts whose token count exceeds this (avoids OOM).")
    p.add_argument("--dtype", default="float16")
    args = p.parse_args()

    print(f"loading {args.model}", file=sys.stderr)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=True
    ).to("cuda:0")
    model.eval()

    handles, capture = attach_hooks(model)
    n_layers = len({k for k in capture}) or sum(
        1 for _, m in model.named_modules() if hasattr(m, "self_attn") and hasattr(m, "mlp")
    )
    print(f"hooks attached on {n_layers} decoder layers", file=sys.stderr)

    iterations_out: list[dict[str, Any]] = []
    n_done = 0
    with open(args.trace, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") != "action" or r.get("action_type") != "llm_call":
                continue
            if n_done >= args.max_iters:
                break
            messages = r["data"].get("messages_in") or []
            if not messages:
                continue

            # Build chat-template input
            try:
                chat = _flatten_to_chat(messages)
                input_text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            except Exception as exc:
                print(f"iter {r.get('iteration')}: chat template failed: {exc}", file=sys.stderr)
                continue

            input_ids = tok(input_text, return_tensors="pt").input_ids
            n_tokens = input_ids.shape[1]
            if n_tokens > args.max_input_tokens:
                print(f"iter {r.get('iteration')}: skipping (token count {n_tokens} > {args.max_input_tokens})", file=sys.stderr)
                continue

            input_ids = input_ids.to("cuda:0")
            capture.clear()
            t0 = time.time()
            with torch.no_grad():
                # use_cache=False keeps KV from being stored across calls; we
                # only want the prefill activation distribution, not gen.
                _ = model(input_ids=input_ids, use_cache=False)
            torch.cuda.synchronize()
            forward_ms = (time.time() - t0) * 1000
            del input_ids
            torch.cuda.empty_cache()

            stats = []
            for li in sorted(capture):
                for kind, s in capture[li].items():
                    stats.append({"layer": li, "kind": kind, **s})
            iterations_out.append({
                "iter": int(r.get("iteration", n_done)),
                "input_tokens": int(n_tokens),
                "forward_ms": forward_ms,
                "stats": stats,
            })
            n_done += 1
            print(f"iter {r.get('iteration')} ({n_tokens} toks, {forward_ms:.0f}ms) → {len(stats)} stat rows", file=sys.stderr)

    for h in handles:
        h.remove()

    payload = {
        "model": args.model,
        "n_layers": n_layers,
        "n_iterations": len(iterations_out),
        "trace": str(args.trace),
        "iterations": iterations_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
