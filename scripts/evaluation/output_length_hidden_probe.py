"""Internal-state probe for output length: last-layer, last-token hidden state of the target model -> MLP head.

This is the "shallow probe" baseline of the OUTLETS paper (arXiv 2609.01068), not OUTLETS itself: it uses the
final hidden state after prefill, obtained from a vLLM pooling server (`--runner pooling --convert embed`,
LAST pooling, no normalisation), and trains a small MLP on log(1 + tokens). Static prediction: prompt only.

  extract: render every prefix with the target tokenizer's chat template (with tools), POST to /v1/embeddings,
           store one vector per sample_id (float16 .npy plus an id list)
  train:   fit the head on the train split, select on validation, predict the test split -> predictions.jsonl
           (same contract as the other predictors); protocol.json records everything

Usage:
  output_length_hidden_probe.py extract --dataset-dir D --api-base http://127.0.0.1:8300/v1 --model M --out DIR
  output_length_hidden_probe.py train --dataset-dir D --labels L --features DIR --out DIR2 [--recorded-labels]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.evaluation.output_length_benchmark import _read_jsonl, _read_natural_labels  # noqa: E402


def extract(a: argparse.Namespace) -> None:
    import httpx
    from transformers import AutoTokenizer

    dataset = json.loads((a.dataset_dir / "dataset.json").read_text())
    tools = (dataset.get("request_options") or {}).get("tools")
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model)
    prefixes = _read_jsonl(a.dataset_dir / "prefixes.jsonl")
    a.out.mkdir(parents=True, exist_ok=True)
    done = {}
    part = a.out / "features.partial.jsonl"
    if part.exists():
        for row in _read_jsonl(part):
            done[row["sample_id"]] = row["embedding"]
    todo = [p for p in prefixes if p["sample_id"] not in done]
    print(f"{len(done)} cached, {len(todo)} to extract", flush=True)

    def one(prefix: dict) -> tuple[str, list[float]]:
        text = tok.apply_chat_template(prefix["messages"], tokenize=False, add_generation_prompt=True,
                                       **({"tools": tools} if tools else {}))
        r = client.post(f"{a.api_base.rstrip('/')}/embeddings", json={"model": a.model, "input": text})
        if r.status_code >= 400:
            raise RuntimeError(f"{prefix['sample_id']}: HTTP {r.status_code}: {r.text[:200]}")
        return prefix["sample_id"], r.json()["data"][0]["embedding"]

    with httpx.Client(timeout=a.timeout_s, trust_env=False) as client, part.open("a") as out, \
            ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        futures = [pool.submit(one, p) for p in todo]
        n = 0
        for f in as_completed(futures):
            sid, emb = f.result()
            out.write(json.dumps({"sample_id": sid, "embedding": emb}) + "\n")
            out.flush()
            done[sid] = emb
            n += 1
            if n % 200 == 0:
                print(f"extracted {n}/{len(todo)}", flush=True)
    ids = [p["sample_id"] for p in prefixes]
    X = np.asarray([done[s] for s in ids], dtype=np.float16)
    np.save(a.out / "features.npy", X)
    (a.out / "sample_ids.json").write_text(json.dumps(ids))
    (a.out / "protocol.json").write_text(json.dumps({"model": a.model, "api_base": a.api_base, "pooling": "LAST, final layer, no normalisation",
                                                     "input": "chat template with tools, generation prompt appended", "dim": int(X.shape[1]),
                                                     "samples": len(ids)}, indent=1))
    print("features", X.shape, flush=True)


def train(a: argparse.Namespace) -> None:
    import torch
    from torch import nn

    ids = json.loads((a.features / "sample_ids.json").read_text())
    X = np.load(a.features / "features.npy").astype(np.float32)
    index = {s: i for i, s in enumerate(ids)}
    prefixes = {p["sample_id"]: p for p in _read_jsonl(a.dataset_dir / "prefixes.jsonl")}
    rows, _ = _read_natural_labels(a.dataset_dir, a.labels, allow_recorded=a.recorded_labels)
    labels: dict[str, list[int]] = {}
    for r in rows:
        labels.setdefault(str(r["sample_id"]), []).append(int(r["actual_tokens"]))
    split = {s: [] for s in ("train", "validation", "test")}
    for sid, ys in labels.items():
        if sid in prefixes and sid in index:
            split[prefixes[sid]["split"]].append((index[sid], math.log1p(sum(ys) / len(ys)), sid))
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    mu, sd = X[[i for i, _, _ in split["train"]]].mean(0), X[[i for i, _, _ in split["train"]]].std(0) + 1e-6

    def tensors(part):
        xi = np.asarray([i for i, _, _ in part]); y = np.asarray([t for _, t, _ in part], dtype=np.float32)
        return torch.tensor((X[xi] - mu) / sd), torch.tensor(y)

    xtr, ytr = tensors(split["train"]); xva, yva = tensors(split["validation"]); xte, _ = tensors(split["test"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = nn.Sequential(nn.Linear(X.shape[1], a.hidden), nn.GELU(), nn.Dropout(a.dropout),
                         nn.Linear(a.hidden, a.hidden), nn.GELU(), nn.Dropout(a.dropout), nn.Linear(a.hidden, 1)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    xtr, ytr, xva, yva, xte = (t.to(dev) for t in (xtr, ytr, xva, yva, xte))
    best, best_state, history = float("inf"), None, []
    for epoch in range(a.epochs):
        head.train(); perm = torch.randperm(len(xtr), device=dev)
        for i in range(0, len(xtr), a.batch_size):
            b = perm[i:i + a.batch_size]
            loss = nn.functional.mse_loss(head(xtr[b]).squeeze(-1), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            va = nn.functional.mse_loss(head(xva).squeeze(-1), yva).item()
        history.append(round(va, 4))
        if va < best:
            best, best_state = va, {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state); head.eval()
    with torch.no_grad():
        pred = torch.expm1(head(xte).squeeze(-1).clamp(max=20)).clamp(min=1).cpu().numpy()
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "predictions.jsonl").open("w") as f:
        for (_, _, sid), p in zip(split["test"], pred):
            f.write(json.dumps({"sample_id": sid, "predicted_tokens": float(p)}) + "\n")
    torch.save({"state": best_state, "mu": torch.tensor(mu), "sd": torch.tensor(sd), "hidden": a.hidden, "dropout": a.dropout},
               a.out / "head.pt")
    (a.out / "protocol.json").write_text(json.dumps({
        "method": "hidden-state-probe (final layer, last token, MLP on log1p tokens)", "features": str(a.features.resolve()),
        "labels": str(a.labels.resolve()), "hidden": a.hidden, "dropout": a.dropout, "lr": a.lr, "weight_decay": a.weight_decay,
        "epochs": a.epochs, "batch_size": a.batch_size, "seed": a.seed, "validation_mse_log": history,
        "best_validation_mse_log": round(best, 4), "counts": {k: len(v) for k, v in split.items()}}, indent=1))
    print(f"best validation MSE (log space) {best:.4f}; test predictions {len(pred)}: mean {pred.mean():.1f} sd {pred.std():.1f}", flush=True)


def predict(a: argparse.Namespace) -> None:
    """Apply a trained head to another feature set (e.g. the steps of a replay run) -> predictions.jsonl."""
    import torch
    from torch import nn

    ck = torch.load(a.head, map_location="cpu")
    ids = json.loads((a.features / "sample_ids.json").read_text())
    X = np.load(a.features / "features.npy").astype(np.float32)
    x = torch.tensor((X - ck["mu"].numpy()) / ck["sd"].numpy())
    h = ck["hidden"]
    head = nn.Sequential(nn.Linear(X.shape[1], h), nn.GELU(), nn.Dropout(ck["dropout"]),
                         nn.Linear(h, h), nn.GELU(), nn.Dropout(ck["dropout"]), nn.Linear(h, 1))
    head.load_state_dict(ck["state"]); head.eval()
    with torch.no_grad():
        pred = torch.expm1(head(x).squeeze(-1).clamp(max=20)).clamp(min=1).numpy()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as f:
        for sid, v in zip(ids, pred):
            f.write(json.dumps({"sample_id": sid, "predicted_tokens": float(v)}) + "\n")
    print(f"predicted {len(ids)}: mean {pred.mean():.1f} sd {pred.std():.1f}", flush=True)


def replay_prefixes(a: argparse.Namespace) -> None:
    """Build a prefix dataset from a replay run's per-step prompts (messages_in), keyed job/action_id."""
    import shutil
    a.out.mkdir(parents=True, exist_ok=True)
    shutil.copy(a.dataset_dir / "dataset.json", a.out / "dataset.json")  # request options (tools) only
    n = 0
    with (a.out / "prefixes.jsonl").open("w") as f:
        for trace in sorted(a.run.glob("output/*/attempt_1/openclaw_host_replay.jsonl")):
            job = trace.parts[-3]
            if "__replacement-" in job:
                continue
            for line in trace.open():
                if '"action"' not in line[:40]:
                    continue
                r = json.loads(line)
                if r.get("type") != "action" or r["action_type"] != "llm_call":
                    continue
                f.write(json.dumps({"sample_id": f"{job}/{r['action_id']}", "split": "test",
                                    "messages": r["data"]["messages_in"]}) + "\n")
                n += 1
    print(f"{n} replay prefixes from {a.run.name}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--dataset-dir", type=Path, required=True); e.add_argument("--api-base", required=True)
    e.add_argument("--model", required=True); e.add_argument("--tokenizer"); e.add_argument("--out", type=Path, required=True)
    e.add_argument("--concurrency", type=int, default=16); e.add_argument("--timeout-s", type=float, default=600.0)
    t = sub.add_parser("train")
    t.add_argument("--dataset-dir", type=Path, required=True); t.add_argument("--labels", type=Path, required=True)
    t.add_argument("--features", type=Path, required=True); t.add_argument("--out", type=Path, required=True)
    t.add_argument("--recorded-labels", action="store_true")
    t.add_argument("--hidden", type=int, default=256); t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--lr", type=float, default=1e-3); t.add_argument("--weight-decay", type=float, default=1e-2)
    t.add_argument("--epochs", type=int, default=200); t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--seed", type=int, default=42)
    pr = sub.add_parser("predict")
    pr.add_argument("--head", type=Path, required=True); pr.add_argument("--features", type=Path, required=True)
    pr.add_argument("--out", type=Path, required=True)
    rp = sub.add_parser("replay-prefixes")
    rp.add_argument("--run", type=Path, required=True); rp.add_argument("--dataset-dir", type=Path, required=True)
    rp.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    {"extract": extract, "train": train, "predict": predict, "replay-prefixes": replay_prefixes}[a.command](a)


if __name__ == "__main__":
    main()
