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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.evaluation.output_length_benchmark import _read_jsonl  # noqa: E402

BUCKETS = (128, 512)


# ----------------------------------------------------------------------------------------------------------- extract
def _render(tok, messages, tools, completion: dict | None, include_reasoning: bool = False):
    kw = {"tools": tools} if tools else {}
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
    if completion is None:
        return prompt, ""
    msg = {k: v for k, v in completion.items() if v is not None and k in ("role", "content", "tool_calls")}
    if include_reasoning:  # vLLM's reasoning parser stores the text under "reasoning"; the template reads reasoning_content
        msg["reasoning_content"] = completion.get("reasoning_content") or completion.get("reasoning") or ""
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
    wanted = [p for p in prefixes if p["sample_id"] not in done and (not a.completions or p["sample_id"] in completions)]
    tok_lock = threading.Lock()  # the fast tokenizer is not thread-safe ("Already borrowed")

    def row_of(p: dict, n_p: int, kept_ids: np.ndarray) -> dict:
        sid = p["sample_id"]
        start = max(0, n_p - a.prompt_tokens)
        tc = (completions.get(sid) or {}).get("tool_calls") or []
        tool = ((tc[0].get("function") or {}).get("name") if tc else None) or ("final" if sid in completions else None)
        return {"sample_id": sid, "split": p["split"], "file": f"{sid.replace('/', '__')}.npz", "n_prompt_kept": n_p - start,
                "n_completion": len(kept_ids) - (n_p - start), "prompt_tokens": n_p, "label_tokens": labels.get(sid), "tool": tool}

    # feature files from an interrupted run: rebuild their index rows without the server
    recovered, todo = 0, []
    with index_path.open("a") as out:
        for p in wanted:
            f = a.out / f"{p['sample_id'].replace('/', '__')}.npz"
            if f.exists():
                prompt, _ = _render(tok, p["messages"], tools, completions.get(p["sample_id"]))
                n_p = len(tok(prompt, add_special_tokens=False).input_ids)
                out.write(json.dumps(row_of(p, n_p, np.load(f)["ids"])) + "\n"); recovered += 1
            else:
                todo.append(p)
    print(f"{len(done)} cached, {recovered} recovered, {len(todo)} to extract", flush=True)

    def one(p: dict) -> dict:
        sid = p["sample_id"]
        with tok_lock:
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
        np.savez(a.out / f"{sid.replace('/', '__')}.npz", feats=keep_feats, ids=keep_ids)
        return row_of(p, n_p, keep_ids)

    rejected = 0
    with httpx.Client(timeout=a.timeout_s, trust_env=False) as client, index_path.open("a") as out, \
            (a.out / "rejected.jsonl").open("a") as rej, ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        futures = {pool.submit(one, p): p["sample_id"] for p in todo}
        for n, f in enumerate(as_completed(futures), 1):
            try:
                out.write(json.dumps(f.result()) + "\n"); out.flush()
            except Exception as e:  # one bad example must not end the run; it is recorded and skipped
                rej.write(json.dumps({"sample_id": futures[f], "error": f"{type(e).__name__}: {e}"[:300]}) + "\n"); rej.flush(); rejected += 1
            if n % 100 == 0:
                print(f"extracted {n}/{len(todo)} ({rejected} rejected)", flush=True)
    print(f"done: {len(todo) - rejected} extracted, {rejected} rejected", flush=True)
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


# ------------------------------------------------------------------------------------------------------ hazard (D)
THINK_END = 151668  # Qwen3 </think>


def extract_hazard(a: argparse.Namespace) -> None:
    """Per-position final-layer states along a teacher-forced completion (reasoning + visible), every --stride tokens,
    plus the last prompt position; the pooling server runs without the aux patch (token_embed = final layer)."""
    import base64
    import httpx
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    dataset = json.loads((a.dataset_dir / "dataset.json").read_text())
    tools = (dataset.get("request_options") or {}).get("tools")
    prefixes = _read_jsonl(a.dataset_dir / "prefixes.jsonl")
    completions = {r["sample_id"]: r for r in _read_jsonl(a.completions)}
    a.out.mkdir(parents=True, exist_ok=True)
    index_path = a.out / "index.jsonl"
    done = {r["sample_id"] for r in _read_jsonl(index_path)} if index_path.exists() else set()
    todo = [p for p in prefixes if p["sample_id"] in completions and p["sample_id"] not in done]
    print(f"{len(done)} cached, {len(todo)} to extract", flush=True)
    tok_lock = threading.Lock()

    def one(p: dict) -> dict:
        sid = p["sample_id"]; r = completions[sid]
        with tok_lock:
            prompt, comp = _render(tok, p["messages"], tools, r["message"], include_reasoning=True)
            p_ids = tok(prompt, add_special_tokens=False).input_ids
            c_ids = tok(comp, add_special_tokens=False).input_ids[: a.completion_tokens]
            text = prompt + tok.decode(c_ids)
            ids = tok(text, add_special_tokens=False).input_ids
        resp = client.post(f"{a.api_base.rstrip('/')}/pooling", json={"model": a.model, "input": text, "task": "token_embed",
                                                                    "encoding_format": "base64"})
        if resp.status_code >= 400:
            raise RuntimeError(f"{sid}: HTTP {resp.status_code}: {resp.text[:200]}")
        body = resp.json(); n_server = body["usage"]["prompt_tokens"]
        feats = np.frombuffer(base64.b64decode(body["data"][0]["data"]), dtype=np.float32).reshape(n_server, -1)
        if n_server != len(ids) or ids[: len(p_ids)] != p_ids:
            raise RuntimeError(f"{sid}: tokenisation mismatch ({n_server} vs {len(ids)})")
        n_p = len(p_ids); n_c = len(ids) - n_p
        comp_ids = ids[n_p:]
        think_end = comp_ids.index(THINK_END) if THINK_END in comp_ids else -1   # completion-relative position of </think>
        pos = [n_p - 1] + list(range(n_p, n_p + n_c, a.stride))                 # last prompt token, then every stride-th completion token
        np.savez(a.out / f"{sid.replace('/', '__')}.npz", feats=feats[pos].astype(np.float16), pos=np.asarray(pos, dtype=np.int32))
        return {"sample_id": sid, "split": p["split"], "file": f"{sid.replace('/', '__')}.npz", "n_prompt": n_p, "n_completion": n_c,
                "think_end": think_end, "label_total": int(r["actual_tokens"]), "truncated": n_c >= a.completion_tokens}

    rejected = 0
    with httpx.Client(timeout=a.timeout_s, trust_env=False) as client, index_path.open("a") as out, \
            (a.out / "rejected.jsonl").open("a") as rej, ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        futures = {pool.submit(one, p): p["sample_id"] for p in todo}
        for n, f in enumerate(as_completed(futures), 1):
            try:
                out.write(json.dumps(f.result()) + "\n"); out.flush()
            except Exception as e:
                rej.write(json.dumps({"sample_id": futures[f], "error": f"{type(e).__name__}: {e}"[:300]}) + "\n"); rej.flush(); rejected += 1
            if n % 100 == 0:
                print(f"extracted {n}/{len(todo)} ({rejected} rejected)", flush=True)
    print(f"done: {len(todo) - rejected} extracted, {rejected} rejected", flush=True)


def train_hazard(a: argparse.Namespace) -> None:
    """Per-position MLP: P(remaining <= X) for horizons X, for the whole output and for the reasoning part; evaluated on
    the test split per horizon and per progress quartile (t / L), plus the static position."""
    import torch
    from torch import nn

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    H = [int(x) for x in a.horizons.split(",")]
    rows = [{**r, "_dir": a.features, "_window": a.completion_window}
            for r in _read_jsonl(a.features / "index.jsonl") if r["n_completion"] > 0]
    if a.extra_features:  # e.g. the >512-token completions re-extracted to full length: replace the truncated rows
        extra = {r["sample_id"]: {**r, "_dir": a.extra_features, "_window": a.extra_window}
                 for r in _read_jsonl(a.extra_features / "index.jsonl") if r["n_completion"] > 0}
        rows = [extra.pop(r["sample_id"], r) for r in rows] + list(extra.values())

    def positions(r):
        z = np.load(r["_dir"] / r["file"]); X = z["feats"].astype(np.float32)
        if a.format == "outlets":  # OUTLETS-agent extraction: rows are [prompt window | completion], no </think>
            n_pk = r["n_prompt_kept"]; keep = np.arange(len(X)) >= n_pk - 1
            X = X[keep]; t = np.arange(len(X)) - 1                  # static row t = -1, then completion positions 0..n_c-1
            r = {**r, "n_prompt": n_pk, "think_end": -1, "label_total": r["label_tokens"],
                 "truncated": r["n_completion"] >= r["_window"]}
        else:
            pos = z["pos"].astype(np.int64); t = pos - r["n_prompt"]  # completion-relative; the static row has t = -1
        L = r["n_completion"] if not r["truncated"] else max(r["n_completion"], r["label_total"])  # rendered length (approx. label_total)
        rem_total = L - (t + 1)
        rem_reason = (r["think_end"] - (t + 1)) if r["think_end"] >= 0 else rem_total
        y = np.array([[rt <= h for h in H] + [rr <= h for h in H] for rt, rr in zip(rem_total, rem_reason)], dtype=np.float32)
        prog = np.clip((t + 1) / max(1, L), 0, 1)
        past_think = (t + 1) > r["think_end"] if r["think_end"] >= 0 else np.zeros_like(t, dtype=bool)
        long_step = np.full(len(t), L > 512)
        return X, y, prog, past_think, long_step

    data = {s: [] for s in ("train", "validation", "test")}
    for r in rows:
        data[r["split"]].append(positions(r))
    cat = lambda part, i: np.concatenate([d[i] for d in part])
    Xtr, ytr = cat(data["train"], 0), cat(data["train"], 1); Xva, yva = cat(data["validation"], 0), cat(data["validation"], 1)
    Xte, yte, pte, pastte, longte = (cat(data["test"], i) for i in range(5))
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    T = lambda X: torch.tensor((X - mu) / sd)
    xtr, xva, xte = T(Xtr), T(Xva), T(Xte); ytr_t, yva_t = torch.tensor(ytr), torch.tensor(yva)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 2 * len(H))).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    xtr, ytr_t, xva, yva_t = xtr.to(dev), ytr_t.to(dev), xva.to(dev), yva_t.to(dev)
    best, best_state = float("inf"), None
    for epoch in range(a.epochs):
        head.train(); perm = torch.randperm(len(xtr), device=dev)
        for i in range(0, len(xtr), 1024):
            b = perm[i:i + 1024]
            loss = nn.functional.binary_cross_entropy_with_logits(head(xtr[b]), ytr_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            v = nn.functional.binary_cross_entropy_with_logits(head(xva), yva_t).item()
        if v < best:
            best, best_state = v, {k: t.detach().clone() for k, t in head.state_dict().items()}
    head.load_state_dict(best_state); head.eval()
    with torch.no_grad():
        p = torch.sigmoid(head(xte.to(dev))).cpu().numpy()

    def score(pr, y):
        if y.sum() == 0 or y.sum() == len(y):
            return {"n": int(len(y)), "positives": int(y.sum())}
        auc = float((pr[y == 1][:, None] > pr[y == 0][None, :]).mean()) if len(y) < 20000 else float(np.mean([(pr[y == 1] > v).mean() for v in pr[y == 0][:2000]]))
        pred = pr >= 0.5; tp = float((pred & (y == 1)).sum())
        return {"n": int(len(y)), "positives": int(y.sum()), "auroc": round(auc, 3), "recall": round(float(tp / max(1, y.sum())), 3),
                "precision": round(float(tp / max(1, pred.sum())), 3), "brier": round(float(((pr - y) ** 2).mean()), 4)}

    out = {"horizons": H, "positions_test": int(len(yte)), "counts": {k: len(v) for k, v in data.items()}, "best_validation_bce": round(best, 4)}
    for j, target in enumerate(("total", "reasoning")):
        for k, h in enumerate(H):
            c = j * len(H) + k; res = {"all": score(p[:, c], yte[:, c]), "static_t0": score(p[pte == 0, c], yte[pte == 0, c])}
            for lo, hi in ((0, .25), (.25, .5), (.5, .75), (.75, 1.01)):
                m = (pte > lo) & (pte <= hi) & (pte > 0)
                res[f"progress_{lo}-{hi if hi < 1 else 1}"] = score(p[m, c], yte[m, c])
            if target == "reasoning":
                m = ~pastte & (pte > 0); res["during_reasoning"] = score(p[m, c], yte[m, c])
            m = longte & (pte > 0); res["long_steps_over_512"] = score(p[m, c], yte[m, c])
            for lo, hi in ((0, .5), (.5, 1.01)):
                m = longte & (pte > lo) & (pte <= hi); res[f"long_steps_progress_{lo}-{hi if hi < 1 else 1}"] = score(p[m, c], yte[m, c])
            out[f"{target}_within_{h}"] = res
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "evaluation.json").write_text(json.dumps(out, indent=1))
    torch.save({"state": best_state, "mu": mu, "sd": sd, "horizons": H}, a.out / "head.pt")
    print(json.dumps({k: v for k, v in out.items() if k.endswith(f"_within_{H[len(H) // 2]}")}, indent=1), flush=True)


def eval_policy(a: argparse.Namespace) -> None:
    """Per-step restore policy on a trained hazard head: fire the restore when P(remaining <= X) >= theta at the
    first position where that holds; a restore takes R tokens (R = restore seconds / TPOT). Late = the step ended
    before the restore was ready (fire + R > L); waste = tokens the restore sat ready (L - fire - R); never = no fire.
    Baselines: fire at the first completion token (never late, maximal waste) and fire at a fixed token count."""
    import torch
    from torch import nn

    ck = torch.load(a.head, map_location="cpu"); H = ck["horizons"]; mu, sd = ck["mu"], ck["sd"]
    head = nn.Sequential(nn.Linear(len(mu), 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 2 * len(H)))
    head.load_state_dict(ck["state"]); head.eval()
    rows = [{**r, "_dir": a.features, "_window": a.completion_window}
            for r in _read_jsonl(a.features / "index.jsonl") if r["n_completion"] > 0 and r["split"] == "test"]
    if a.extra_features:
        extra = {r["sample_id"]: {**r, "_dir": a.extra_features, "_window": a.extra_window}
                 for r in _read_jsonl(a.extra_features / "index.jsonl") if r["n_completion"] > 0 and r["split"] == "test"}
        rows = [extra.pop(r["sample_id"], r) for r in rows] + list(extra.values())
    steps = []
    with torch.no_grad():
        for r in rows:
            z = np.load(r["_dir"] / r["file"]); X = z["feats"].astype(np.float32)
            if a.format == "outlets":
                n_pk = r["n_prompt_kept"]; X = X[n_pk:]; t = np.arange(len(X))          # completion positions only
                L = r["label_tokens"] if r["n_completion"] >= r["_window"] else r["n_completion"]
            else:
                pos = z["pos"].astype(np.int64); t = pos - r["n_prompt"]; keep = t >= 0; X, t = X[keep], t[keep]
                L = r["n_completion"] if not r["truncated"] else max(r["n_completion"], r["label_total"])
            p = torch.sigmoid(head(torch.tensor((X - mu) / sd))).numpy()
            steps.append((t, p, int(L)))
    R = int(round(a.restore_s / a.tpot_s))
    out = {"steps": len(steps), "restore_tokens": R, "horizons": H, "policies": {}}

    def policy(fire_fn, name):
        late = waste = never = 0; wastes = []
        for t, p, L in steps:
            f = fire_fn(t, p, L)
            if f is None:
                never += 1; continue
            if f + R > L:
                late += 1
            else:
                wastes.append(L - f - R)
        n = len(steps)
        out["policies"][name] = {"late": round(late / n, 3), "never_fired": round(never / n, 3),
                                 "waste_tokens_p50": int(np.median(wastes)) if wastes else None,
                                 "waste_tokens_p90": int(np.percentile(wastes, 90)) if wastes else None,
                                 "ready_in_time": round((n - late - never) / n, 3)}

    policy(lambda t, p, L: 0, "fire_at_first_token")
    for k in (64, 256):
        policy(lambda t, p, L, k=k: k if L > k else None, f"fire_at_token_{k}")
    for j, target in enumerate(("total", "reasoning")):
        for i, h in enumerate(H):
            for theta in (0.5, 0.7, 0.9):
                c = j * len(H) + i

                def fire(t, p, L, c=c, theta=theta):
                    idx = np.nonzero(p[:, c] >= theta)[0]
                    return int(t[idx[0]]) if len(idx) else None
                policy(fire, f"{target}_within_{h}_theta_{theta}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    for k, v in out["policies"].items():
        print(f"{k:34s} late {v['late']:.3f} never {v['never_fired']:.3f} ready {v['ready_in_time']:.3f} waste p50/p90 {v['waste_tokens_p50']}/{v['waste_tokens_p90']}", flush=True)


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
    eh = sub.add_parser("extract-hazard")
    eh.add_argument("--dataset-dir", type=Path, required=True); eh.add_argument("--completions", type=Path, required=True)
    eh.add_argument("--api-base", required=True); eh.add_argument("--model", required=True); eh.add_argument("--tokenizer", required=True)
    eh.add_argument("--out", type=Path, required=True); eh.add_argument("--completion-tokens", type=int, default=2048)
    eh.add_argument("--stride", type=int, default=4); eh.add_argument("--concurrency", type=int, default=6); eh.add_argument("--timeout-s", type=float, default=900.0)
    th = sub.add_parser("train-hazard")
    th.add_argument("--features", type=Path, required=True); th.add_argument("--out", type=Path, required=True)
    th.add_argument("--horizons", default="64,256,1024"); th.add_argument("--epochs", type=int, default=30); th.add_argument("--seed", type=int, default=42)
    th.add_argument("--format", choices=("hazard", "outlets"), default="hazard"); th.add_argument("--completion-window", type=int, default=512)
    th.add_argument("--extra-features", type=Path); th.add_argument("--extra-window", type=int, default=2048)
    ep = sub.add_parser("eval-policy")
    ep.add_argument("--head", type=Path, required=True); ep.add_argument("--features", type=Path, required=True)
    ep.add_argument("--extra-features", type=Path); ep.add_argument("--extra-window", type=int, default=2048)
    ep.add_argument("--format", choices=("hazard", "outlets"), default="hazard"); ep.add_argument("--completion-window", type=int, default=512)
    ep.add_argument("--restore-s", type=float, default=1.8); ep.add_argument("--tpot-s", type=float, default=0.03)
    ep.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    {"extract": extract, "train": train, "predict": predict_cmd, "extract-hazard": extract_hazard, "train-hazard": train_hazard,
     "eval-policy": eval_policy}[a.command](a)


if __name__ == "__main__":
    main()
