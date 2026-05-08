"""MoE expert activation probe.

For each decoder layer that has an `.mlp.gate` submodule (the typical MoE
routing pattern in HF transformers — OLMoE, Mixtral, Qwen2-MoE, DeepSeek-V2,
all expose this), hook the gate Linear and capture the router logits per
token. Compute:
  - per-expert activation count (number of tokens that placed this expert in
    their top-k)
  - per-expert probability mass (sum of softmax-normalized top-k probs
    assigned to this expert across all tokens)
  - per-token routing entropy (mean over tokens), measures load balance

Aggregated per (iteration, layer) and saved to JSON for plotting.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch import nn


def _flatten_to_chat(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
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


def find_gate_modules(model: nn.Module) -> list[tuple[int, str, nn.Module]]:
    """Return (layer_idx, full_path, gate_module) for every MoE routing layer.

    Heuristic: walk decoder layers (which have `self_attn` and `mlp`); within
    each, look for `mlp.gate` (OLMoE / Qwen2-MoE / DeepSeek-V2) or
    `block_sparse_moe.gate` (Mixtral). The gate is a Linear projecting hidden
    -> num_experts.
    """
    layers: list[tuple[int, str, nn.Module]] = []
    layer_idx = 0
    for name, m in model.named_modules():
        if not (hasattr(m, "self_attn") and hasattr(m, "mlp")):
            continue
        # decoder layer detected; probe for gate
        gate = None
        gate_path = None
        for candidate in ("mlp.gate", "mlp.router", "block_sparse_moe.gate"):
            obj = m
            ok = True
            for part in candidate.split("."):
                if not hasattr(obj, part):
                    ok = False
                    break
                obj = getattr(obj, part)
            # Accept any nn.Module that looks like a router: 1 weight matrix
            # whose out_features matches num_experts. We don't require nn.Linear
            # because some MoE implementations subclass it.
            if ok and isinstance(obj, nn.Module) and hasattr(obj, "weight"):
                gate = obj
                gate_path = f"{name}.{candidate}"
                break
        if gate is not None:
            layers.append((layer_idx, gate_path, gate))
        layer_idx += 1
    return layers


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--trace", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--max-iters", type=int, default=5)
    p.add_argument("--max-input-tokens", type=int, default=4096)
    p.add_argument("--top-k", type=int, default=0,
                   help="top-k for router (0 = read from model config)")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--quantize", choices=["none", "4bit"], default="none",
                   help="4bit uses bitsandbytes nf4 (weights 4-bit, compute fp16)")
    p.add_argument("--attn-impl", default="sdpa",
                   help="sdpa (default) or eager or flash_attention_2")
    args = p.parse_args()

    print(f"loading {args.model} quantize={args.quantize}", file=sys.stderr)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    load_kwargs = {"trust_remote_code": True, "attn_implementation": args.attn_impl}
    if args.quantize == "4bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        # bitsandbytes manages device placement itself via accelerate; need device_map
        load_kwargs["device_map"] = "cuda:0"
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    else:
        load_kwargs["dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).to("cuda:0")
    model.eval()

    cfg = model.config
    num_experts = getattr(cfg, "num_experts", None) or getattr(cfg, "num_local_experts", None)
    top_k = args.top_k or getattr(cfg, "num_experts_per_tok", None) or getattr(cfg, "moe_top_k", None)
    if num_experts is None or top_k is None:
        raise SystemExit(f"could not infer num_experts ({num_experts}) or top_k ({top_k}) from model config")
    print(f"model: num_experts={num_experts}, top_k={top_k}", file=sys.stderr)

    gate_layers = find_gate_modules(model)
    print(f"found {len(gate_layers)} MoE routing gates", file=sys.stderr)
    if not gate_layers:
        raise SystemExit("no MoE gates found — is this actually an MoE model?")

    # capture[layer_idx] = router_logits tensor [tokens, num_experts]
    capture: dict[int, torch.Tensor] = {}
    handles: list[Any] = []

    def make_hook(li: int):
        def hook(module, inputs, output):
            # gate Linear output: [batch * seq, num_experts] OR [batch, seq, num_experts]
            t = output if isinstance(output, torch.Tensor) else output[0]
            capture[li] = t.detach().to(torch.float32).reshape(-1, t.shape[-1])
        return hook

    for li, path, gate in gate_layers:
        handles.append(gate.register_forward_hook(make_hook(li)))

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

            try:
                chat = _flatten_to_chat(messages)
                input_text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            except Exception as exc:
                print(f"iter {r.get('iteration')}: chat template failed: {exc}", file=sys.stderr)
                continue

            input_ids = tok(input_text, return_tensors="pt").input_ids
            n_tokens = input_ids.shape[1]
            if n_tokens > args.max_input_tokens:
                print(f"iter {r.get('iteration')}: skipping ({n_tokens} > cap)", file=sys.stderr)
                continue

            input_ids = input_ids.to("cuda:0")
            capture.clear()
            t0 = time.time()
            with torch.no_grad():
                _ = model(input_ids=input_ids, use_cache=False)
            torch.cuda.synchronize()
            forward_ms = (time.time() - t0) * 1000

            # aggregate router stats per layer
            layers_summary: list[dict[str, Any]] = []
            for li in sorted(capture):
                logits = capture[li]  # [tokens, num_experts]
                probs = torch.softmax(logits, dim=-1)
                top_p, top_idx = torch.topk(probs, k=top_k, dim=-1)
                # renormalize top-k probs (actual gating is over top-k only)
                top_p_norm = top_p / top_p.sum(dim=-1, keepdim=True)

                # token counts per expert
                expert_counts = torch.zeros(num_experts, dtype=torch.long, device=logits.device)
                expert_counts.scatter_add_(0, top_idx.flatten(), torch.ones_like(top_idx.flatten()))

                # probability mass per expert (sum of normalized top-k probs)
                expert_prob_mass = torch.zeros(num_experts, dtype=torch.float32, device=logits.device)
                expert_prob_mass.scatter_add_(0, top_idx.flatten(), top_p_norm.flatten())

                # per-token routing entropy over the FULL distribution (not top-k)
                # entropy = -sum(p * log p), measures spread
                entropy_mean = float((-(probs * (probs + 1e-12).log()).sum(dim=-1)).mean().item())

                layers_summary.append({
                    "layer": li,
                    "expert_token_counts": expert_counts.tolist(),
                    "expert_prob_mass": expert_prob_mass.tolist(),
                    "routing_entropy_mean": entropy_mean,
                })

            iterations_out.append({
                "iter": int(r.get("iteration", n_done)),
                "input_tokens": int(n_tokens),
                "forward_ms": forward_ms,
                "layers": layers_summary,
            })
            n_done += 1
            print(f"iter {r.get('iteration')} ({n_tokens} toks, {forward_ms:.0f}ms) → {len(layers_summary)} layers", file=sys.stderr)

    for h in handles:
        h.remove()

    payload = {
        "model": args.model,
        "num_experts": num_experts,
        "top_k": top_k,
        "n_layers": len(gate_layers),
        "n_iterations": len(iterations_out),
        "trace": str(args.trace),
        "iterations": iterations_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
