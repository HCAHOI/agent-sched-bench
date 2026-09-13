"""OUTLETS-agent: an EAGLE-3 draft decoder over the target model's fused hidden states, fine-tuned to predict output
length for agent steps (PENDING §8b "OUTLETS-agent"; paper: arXiv 2609.01068).

Pipeline (all paths on the GPU host):
  extract   per example: render the prompt (chat template + generation prompt) and, for training, the completion
            (natural or recorded assistant message rendered by the same template, truncated to --completion-tokens);
            POST the text to the pooling server's token_embed task, whose aux patch returns fc(cat(h2, h24, h45)) per
            token; keep the last --prompt-tokens prompt positions and the completion positions; store float16 features,
            token ids (one extra for the EAGLE shift), the prompt boundary, the total length label and the tool.
  train     Eagle3Draft (pretrained SpecForge weights, target embedding table) + length heads; per example, pairs
            (h_i, embed(token_{i+1})) with remaining-length labels at every completion pair and the static labels at
            the last prompt pair; validation-selected; writes predictions on the test split (static, both heads) and
            dynamic MAE.
  predict   static predictions for another extracted set (replay steps) -> predictions.jsonl (same contract).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.evaluation.output_length_benchmark import _read_jsonl  # noqa: E402

BUCKETS = (128, 512)


# ----------------------------------------------------------------------------------------------------------- extract
def _render(tok, messages, tools, completion: dict | None):
    kw = {"tools": tools} if tools else {}
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
    if completion is None:
        return prompt, ""
    msg = {k: v for k, v in completion.items() if v is not None and k in ("role", "content", "tool_calls")}
    msg.setdefault("role", "assistant"); msg.setdefault("content", "")
    for tc in msg.get("tool_calls") or []:
        try:
            tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
        except (TypeError, ValueError, KeyError):
            pass
    full = tok.apply_chat_template(messages + [msg], tokenize=False, **kw)
    assert full.startswith(prompt)
    return prompt, full[len(prompt):]


def extract(a: argparse.Namespace) -> None:
    import base64
    import httpx
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    dataset = json.loads((a.dataset_dir / "dataset.json").read_text())
    tools = (dataset.get("request_options") or {}).get("tools")
    prefixes = _read_jsonl(a.dataset_dir / "prefixes.jsonl")
    completions: dict[str, dict] = {}
    labels: dict[str, int] = {}
    if a.completions:
        for r in _read_jsonl(a.completions):
            completions[r["sample_id"]] = r["message"]
            labels[r["sample_id"]] = int(r["actual_tokens"])
    a.out.mkdir(parents=True, exist_ok=True)
    index_path = a.out / "index.jsonl"
    done = {r["sample_id"] for r in _read_jsonl(index_path)} if index_path.exists() else set()
    todo = [p for p in prefixes if p["sample_id"] not in done and (not a.completions or p["sample_id"] in completions)]
    print(f"{len(done)} cached, {len(todo)} to extract", flush=True)

    def one(p: dict) -> dict:
        sid = p["sample_id"]
        prompt, comp = _render(tok, p["messages"], tools, completions.get(sid))
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        c_ids = tok(comp, add_special_tokens=False).input_ids[: a.completion_tokens] if comp else []
        text = prompt + (tok.decode(c_ids) if c_ids else "")
        ids = tok(text, add_special_tokens=False).input_ids
        r = client.post(f"{a.api_base.rstrip('/')}/pooling", json={"model": a.model, "input": text, "task": "token_embed",
                                                                 "encoding_format": "base64"})
        if r.status_code >= 400:
            raise RuntimeError(f"{sid}: HTTP {r.status_code}: {r.text[:200]}")
        body = r.json()
        n_server = body["usage"]["prompt_tokens"]
        feats = np.frombuffer(base64.b64decode(body["data"][0]["data"]), dtype=np.float32).reshape(n_server, -1)
        if n_server != len(ids):
            raise RuntimeError(f"{sid}: server tokens {n_server} != local {len(ids)}")
        n_p = len(p_ids)
        if ids[:n_p] != p_ids:
            raise RuntimeError(f"{sid}: prompt/full tokenisation mismatch at the boundary")
        n_c = len(ids) - n_p
        start = max(0, n_p - a.prompt_tokens)
        keep_feats = feats[start: n_p + n_c].astype(np.float16)           # positions [start, n_p + n_c)
        keep_ids = np.asarray(ids[start: n_p + n_c], dtype=np.int32)      # same positions; next tokens = shift by one
        tc = (completions.get(sid) or {}).get("tool_calls") or []
        tool = ((tc[0].get("function") or {}).get("name") if tc else None) or ("final" if sid in completions else None)
        np.savez(a.out / f"{sid.replace('/', '__')}.npz", feats=keep_feats, ids=keep_ids)
        return {"sample_id": sid, "split": p["split"], "file": f"{sid.replace('/', '__')}.npz", "n_prompt_kept": n_p - start,
                "n_completion": n_c, "prompt_tokens": n_p, "label_tokens": labels.get(sid), "tool": tool}

    with httpx.Client(timeout=a.timeout_s, trust_env=False) as client, index_path.open("a") as out, \
            ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        futures = [pool.submit(one, p) for p in todo]
        for n, f in enumerate(as_completed(futures), 1):
            out.write(json.dumps(f.result()) + "\n"); out.flush()
            if n % 100 == 0:
                print(f"extracted {n}/{len(todo)}", flush=True)
    (a.out / "protocol.json").write_text(json.dumps({"model": a.model, "features": "fc(cat(h2,h24,h45)) per token, float16",
                                                     "prompt_tokens_kept": a.prompt_tokens, "completion_tokens_max": a.completion_tokens,
                                                     "completions": str(a.completions) if a.completions else None}, indent=1))


# ------------------------------------------------------------------------------------------------------------- model
def build_draft(draft_dir: Path, target_dir: Path):
    """The SpecForge EAGLE-3 draft (one Llama-style layer over [embed | fc(fused)]) with the target's embedding table."""
    import torch
    from torch import nn
    from safetensors import safe_open

    cfg = json.loads((draft_dir / "config.json").read_text())
    H, NH, NKV, HD, I = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["intermediate_size"]
    eps, theta = cfg["rms_norm_eps"], cfg["rope_theta"]

    class RMSNorm(nn.Module):
        def __init__(self, d):
            super().__init__(); self.weight = nn.Parameter(torch.ones(d))

        def forward(self, x):
            xf = x.float(); return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * self.weight

    def rope(x, pos):  # x: [B, heads, T, HD]; rotate-half convention as in Llama
        inv = 1.0 / (theta ** (torch.arange(0, HD, 2, device=x.device).float() / HD))
        ang = pos.float()[:, None] * inv[None, :]                     # [T, HD/2]
        cos, sin = torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)
        x1, x2 = x[..., : HD // 2], x[..., HD // 2:]
        return (x * cos.to(x.dtype) + torch.cat([-x2, x1], -1) * sin.to(x.dtype))

    class Draft(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(cfg["vocab_size"], H)
            self.input_layernorm, self.hidden_norm, self.post_attention_layernorm, self.norm = RMSNorm(H), RMSNorm(H), RMSNorm(H), RMSNorm(H)
            self.q_proj, self.k_proj, self.v_proj = nn.Linear(2 * H, NH * HD, bias=False), nn.Linear(2 * H, NKV * HD, bias=False), nn.Linear(2 * H, NKV * HD, bias=False)
            self.o_proj = nn.Linear(NH * HD, H, bias=False)
            self.gate_proj, self.up_proj, self.down_proj = nn.Linear(H, I, bias=False), nn.Linear(H, I, bias=False), nn.Linear(I, H, bias=False)

        def forward(self, feats, next_ids):  # feats [B,T,H] = fc(fused) at positions i; next_ids [B,T] = token_{i+1}
            B, T, _ = feats.shape
            e = self.input_layernorm(self.embed(next_ids))
            residual = feats
            x = torch.cat([e, self.hidden_norm(feats)], -1)
            pos = torch.arange(T, device=feats.device)
            q = rope(self.q_proj(x).view(B, T, NH, HD).transpose(1, 2), pos)
            k = rope(self.k_proj(x).view(B, T, NKV, HD).transpose(1, 2), pos)
            v = self.v_proj(x).view(B, T, NKV, HD).transpose(1, 2)
            k, v = k.repeat_interleave(NH // NKV, 1), v.repeat_interleave(NH // NKV, 1)
            att = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            h = residual + self.o_proj(att.transpose(1, 2).reshape(B, T, NH * HD))
            y = self.post_attention_layernorm(h)
            h = h + self.down_proj(nn.functional.silu(self.gate_proj(y)) * self.up_proj(y))
            return self.norm(h)

    d = Draft()
    with safe_open(str(draft_dir / "model.safetensors"), "pt") as st:
        sd = {k: st.get_tensor(k) for k in st.keys()}
    own = {"midlayer.self_attn.q_proj.weight": "q_proj.weight", "midlayer.self_attn.k_proj.weight": "k_proj.weight",
           "midlayer.self_attn.v_proj.weight": "v_proj.weight", "midlayer.self_attn.o_proj.weight": "o_proj.weight",
           "midlayer.mlp.gate_proj.weight": "gate_proj.weight", "midlayer.mlp.up_proj.weight": "up_proj.weight",
           "midlayer.mlp.down_proj.weight": "down_proj.weight", "midlayer.input_layernorm.weight": "input_layernorm.weight",
           "midlayer.hidden_norm.weight": "hidden_norm.weight", "midlayer.post_attention_layernorm.weight": "post_attention_layernorm.weight",
           "norm.weight": "norm.weight"}
    state = {v: sd[k].float() for k, v in own.items()}
    idx = json.loads((target_dir / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(target_dir / idx["model.embed_tokens.weight"]), "pt") as st:
        state["embed.weight"] = st.get_tensor("model.embed_tokens.weight").float()
    missing, unexpected = d.load_state_dict(state, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    d.embed.weight.requires_grad_(False)
    return d, H


class Heads:
    """Length heads on the draft state: scalar remaining log-length; tool logits and per-tool log-length (static)."""

    def __init__(self, H: int, n_tools: int):
        import torch
        from torch import nn
        self.trunk = nn.Sequential(nn.Linear(H, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.scalar, self.tool, self.per_tool = nn.Linear(256, 1), nn.Linear(256, n_tools), nn.Linear(256, n_tools)
        self.module = nn.ModuleDict({"trunk": self.trunk, "scalar": self.scalar, "tool": self.tool, "per_tool": self.per_tool})
        self.torch = torch

    def __call__(self, h):
        z = self.trunk(h)
        return self.scalar(z).squeeze(-1), self.tool(z), self.per_tool(z)


def load_example(a, row) -> tuple:
    z = np.load(a.features / row["file"])
    feats, ids = z["feats"].astype(np.float32), z["ids"]
    n_pk, n_c = row["n_prompt_kept"], row["n_completion"]
    # pairs (h_i, token_{i+1}) for i in [0, n_pk + n_c - 1); the static pair is i = n_pk - 1 (needs the first completion token)
    T = n_pk + n_c - 1
    return feats[:T], ids[1: T + 1], n_pk - 1, n_c


def train(a: argparse.Namespace) -> None:
    import torch
    from torch import nn
    from scripts.evaluation.output_length_history_baselines import evaluate as tail_metrics

    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    rows = [r for r in _read_jsonl(a.features / "index.jsonl") if r["label_tokens"] and r["n_completion"] >= 1]
    excluded = set()
    if a.exclude_manifest:
        excluded = {t["label"] for t in __import__("yaml").safe_load(a.exclude_manifest.open())["traces"]}
        rows = [r for r in rows if r["split"] == "test" or r["sample_id"].split("/")[1] not in excluded]
    split = {s: [r for r in rows if r["split"] == s] for s in ("train", "validation", "test")}
    counts = {}
    for r in split["train"]:
        counts[r["tool"]] = counts.get(r["tool"], 0) + 1
    tools = sorted(t for t, c in counts.items() if c >= a.min_tool_count) + ["other"]
    tix = {t: i for i, t in enumerate(tools)}
    cls = lambda r: tix.get(r["tool"], tix["other"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    draft, H = build_draft(a.draft_dir, a.target_dir)
    draft.to(dev); heads = Heads(H, len(tools)); heads.module.to(dev)
    if a.freeze_backbone:
        for p in draft.parameters():
            p.requires_grad_(False)
    params = [{"params": [p for p in draft.parameters() if p.requires_grad], "lr": a.backbone_lr},
              {"params": heads.module.parameters(), "lr": a.head_lr}]
    opt = torch.optim.AdamW(params, weight_decay=a.weight_decay)

    def losses(r):
        feats, nxt, s_idx, n_c = load_example(a, r)
        f = torch.tensor(feats, device=dev)[None]; ni = torch.tensor(nxt, device=dev, dtype=torch.long)[None]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            h = draft(f, ni)
        scalar, tool_logits, per_tool = heads(h[0].float())
        L = float(r["label_tokens"])
        # static at s_idx: total length; dynamic at s_idx + t (t >= 1): remaining L - t
        t = torch.arange(0, n_c - 1, device=dev, dtype=torch.float32)                 # pairs s_idx .. s_idx + n_c - 2
        target = torch.log1p((L - t).clamp(min=1))
        pred = scalar[s_idx: s_idx + len(t)]
        l_dyn = nn.functional.mse_loss(pred, target) if len(t) > 1 else pred.new_zeros(())
        l_static = (scalar[s_idx] - math.log1p(L)) ** 2
        c = torch.tensor(cls(r), device=dev)
        l_tool = nn.functional.cross_entropy(tool_logits[s_idx][None], c[None])
        l_pt = (per_tool[s_idx][c] - math.log1p(L)) ** 2
        return l_static + a.dynamic_weight * l_dyn + (0.0 if a.scalar_only else (l_tool + l_pt))

    @torch.no_grad()
    def predict(r):
        feats, nxt, s_idx, n_c = load_example(a, r)
        f = torch.tensor(feats, device=dev)[None]; ni = torch.tensor(nxt, device=dev, dtype=torch.long)[None]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            h = draft(f, ni)
        scalar, tool_logits, per_tool = heads(h[0].float())
        p = torch.softmax(tool_logits[s_idx], -1)
        structured = float(torch.expm1((p * per_tool[s_idx]).sum().clamp(max=20)).clamp(min=1))
        static = float(torch.expm1(scalar[s_idx].clamp(max=20)).clamp(min=1))
        dyn = torch.expm1(scalar[s_idx: s_idx + max(0, n_c - 1)].clamp(max=20)).clamp(min=1).cpu().numpy()
        return static, structured, int(p.argmax()), dyn

    def val_loss():
        draft.eval(); heads.module.eval()
        with torch.no_grad():
            v = statistics.fmean(float(losses(r)) for r in split["validation"])
        draft.train(); heads.module.train()
        return v

    best, best_state, history = float("inf"), None, []
    order = list(split["train"])
    for epoch in range(a.epochs):
        random.shuffle(order); opt.zero_grad()
        for i, r in enumerate(order, 1):
            (losses(r) / a.accumulate).backward()
            if i % a.accumulate == 0:
                torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], 1.0)
                opt.step(); opt.zero_grad()
        v = val_loss(); history.append(round(v, 4))
        print(f"epoch {epoch + 1}: validation {v:.4f}", flush=True)
        if v < best:
            best = v
            best_state = ({k: t.detach().clone() for k, t in draft.state_dict().items() if k != "embed.weight"},
                          {k: t.detach().clone() for k, t in heads.module.state_dict().items()})
    draft.load_state_dict(best_state[0], strict=False); heads.module.load_state_dict(best_state[1])
    draft.eval(); heads.module.eval()
    a.out.mkdir(parents=True, exist_ok=True)
    static_pairs, struct_pairs, tool_hits, dyn_mae = [], [], 0, []
    with (a.out / "predictions.jsonl").open("w") as f_s, (a.out / "predictions-structured.jsonl").open("w") as f_t:
        for r in split["test"]:
            static, structured, tool_hat, dyn = predict(r)
            L = r["label_tokens"]
            f_s.write(json.dumps({"sample_id": r["sample_id"], "predicted_tokens": static}) + "\n")
            f_t.write(json.dumps({"sample_id": r["sample_id"], "predicted_tokens": structured, "tool_hat": tools[tool_hat]}) + "\n")
            static_pairs.append((static, L)); struct_pairs.append((structured, L)); tool_hits += tool_hat == cls(r)
            if len(dyn) > 1:
                dyn_mae.append(float(np.mean(np.abs((L - np.arange(1, len(dyn))) - dyn[1:]))) / L)  # paper: mean over t of |r_t - r_hat_t| / L
    ev = {"static_scalar": tail_metrics(static_pairs), "static_structured": tail_metrics(struct_pairs),
          "tool_accuracy": round(tool_hits / len(split["test"]), 3), "dynamic_mae_normalised": round(statistics.fmean(dyn_mae), 4),
          "tools": tools, "counts": {k: len(v) for k, v in split.items()}, "best_validation": round(best, 4), "validation": history,
          "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, "excluded_tasks": len(excluded)}
    (a.out / "evaluation.json").write_text(json.dumps(ev, indent=1))
    torch.save({"draft": best_state[0], "heads": best_state[1], "tools": tools}, a.out / "model.pt")
    print(json.dumps({k: ev[k] for k in ("static_scalar", "static_structured", "tool_accuracy", "dynamic_mae_normalised")}, indent=1), flush=True)


def predict_cmd(a: argparse.Namespace) -> None:
    """Static predictions for another extracted set (e.g. replay steps without completions): the static pair needs the
    first completion token, which is unknown at t = 0 for a bare prompt, so the last prompt pair (h_{n-2}, token_{n-1})
    is used, as for the probe."""
    import torch
    ck = torch.load(a.model, map_location="cpu")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    draft, H = build_draft(a.draft_dir, a.target_dir); draft.load_state_dict(ck["draft"], strict=False); draft.to(dev).eval()
    heads = Heads(H, len(ck["tools"])); heads.module.load_state_dict(ck["heads"]); heads.module.to(dev).eval()
    rows = _read_jsonl(a.features / "index.jsonl")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as f, torch.no_grad():
        for r in rows:
            z = np.load(a.features / r["file"]); feats, ids = z["feats"].astype(np.float32), z["ids"]
            T = len(ids) - 1
            fe = torch.tensor(feats[:T], device=dev)[None]; ni = torch.tensor(ids[1: T + 1], device=dev, dtype=torch.long)[None]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                h = draft(fe, ni)
            scalar, tool_logits, per_tool = heads(h[0].float())
            p = torch.softmax(tool_logits[-1], -1)
            out = per_tool[-1] if a.head == "structured" else scalar[-1]
            val = float(torch.expm1(((p * out).sum() if a.head == "structured" else out).clamp(max=20)).clamp(min=1))
            f.write(json.dumps({"sample_id": r["sample_id"], "predicted_tokens": val}) + "\n")
    print(f"predicted {len(rows)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--dataset-dir", type=Path, required=True); e.add_argument("--completions", type=Path, help="labels.jsonl with message + actual_tokens")
    e.add_argument("--api-base", required=True); e.add_argument("--model", required=True); e.add_argument("--tokenizer", required=True)
    e.add_argument("--out", type=Path, required=True); e.add_argument("--prompt-tokens", type=int, default=1024)
    e.add_argument("--completion-tokens", type=int, default=512); e.add_argument("--concurrency", type=int, default=4)
    e.add_argument("--timeout-s", type=float, default=900.0)
    t = sub.add_parser("train")
    t.add_argument("--features", type=Path, required=True); t.add_argument("--draft-dir", type=Path, required=True)
    t.add_argument("--target-dir", type=Path, required=True); t.add_argument("--out", type=Path, required=True)
    t.add_argument("--exclude-manifest", type=Path); t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--backbone-lr", type=float, default=1e-5); t.add_argument("--head-lr", type=float, default=1e-3)
    t.add_argument("--weight-decay", type=float, default=1e-2); t.add_argument("--accumulate", type=int, default=8)
    t.add_argument("--dynamic-weight", type=float, default=1.0); t.add_argument("--min-tool-count", type=int, default=20)
    t.add_argument("--freeze-backbone", action="store_true"); t.add_argument("--scalar-only", action="store_true")
    t.add_argument("--seed", type=int, default=42)
    pr = sub.add_parser("predict")
    pr.add_argument("--model", type=Path, required=True); pr.add_argument("--features", type=Path, required=True)
    pr.add_argument("--draft-dir", type=Path, required=True); pr.add_argument("--target-dir", type=Path, required=True)
    pr.add_argument("--out", type=Path, required=True); pr.add_argument("--head", choices=("scalar", "structured"), default="scalar")
    a = p.parse_args()
    {"extract": extract, "train": train, "predict": predict_cmd}[a.command](a)


if __name__ == "__main__":
    main()
