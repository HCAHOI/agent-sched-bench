"""History-based output-length predictors on the frozen cross-benchmark test split.

Meeting question (2026-09-07 §2): does the same session's accumulated history of recorded output
lengths predict the next output better than a per-request content model or a constant? And what
does the tool-call format bound? Every predictor here sees only the session's earlier recorded
completion lengths (steps before the sample's step), so it is causal at inference time.

Inputs: the frozen dataset under analysis/results/output-length-source-labels-crossbench-20260904
(test sample ids from egtp-static/predictions.jsonl, labels from dataset/source_labels.jsonl) and
the trace pool under traces/exports (per-step recorded completion lengths and raw responses).

Usage:
  .venv/bin/python scripts/evaluation/output_length_history_baselines.py \
      --out analysis/results/output-length-source-labels-crossbench-20260904/history-baselines
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import tiktoken

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "analysis/results/output-length-source-labels-crossbench-20260904"
POOLS = {"swe": REPO / "traces/exports/swe-rebench-original-flat-644-20260904",
         "tb": REPO / "traces/exports/terminal-bench-original-flat-239-20260904"}
TRAIN_MEDIAN = 152  # frozen constant baseline (egtp-static/evaluation.json, train-median)
BUCKETS = (128, 512)  # <128 | 128-512 | >512 tokens


def steps_of(trace: Path) -> list[dict]:
    """Recorded LLM steps in action order: completion tokens plus a text/tool-argument character split."""
    out = []
    for line in trace.open():
        if '"action"' not in line[:40]:
            continue
        r = json.loads(line)
        if r.get("type") != "action" or r["action_type"] != "llm_call":
            continue
        raw = r["data"].get("raw_response")
        raw = json.loads(raw) if isinstance(raw, str) else (raw or {})
        msg = ((raw.get("choices") or [{}])[0].get("message") or {})
        text = msg.get("content") or ""
        args = "".join((tc.get("function") or {}).get("arguments") or "" for tc in (msg.get("tool_calls") or []))
        out.append({"action_id": r["action_id"], "completion_tokens": r["data"].get("completion_tokens") or 0,
                    "text": text, "args": args, "n_tool_calls": len(msg.get("tool_calls") or [])})
    return out


def qerr(pred: float, actual: float) -> float:
    pred, actual = max(pred, 1.0), max(actual, 1.0)
    return max(pred, actual) / min(pred, actual)


def bucket(x: float) -> int:
    return sum(x >= b for b in BUCKETS)


def evaluate(rows: list[tuple[float, float]]) -> dict:
    q = sorted(qerr(p, a) for p, a in rows)
    n = len(q)
    pick = lambda f: q[min(n - 1, int(f * (n - 1)))]
    return {"n": n, "q50": round(pick(0.5), 3), "q90": round(pick(0.9), 3), "q95": round(pick(0.95), 3),
            "q99": round(pick(0.99), 3), "mean_q": round(statistics.fmean(q), 3),
            "mae": round(statistics.fmean(abs(p - a) for p, a in rows), 1),
            "underprediction_rate": round(sum(p < a for p, a in rows) / n, 3),
            "bucket_accuracy": round(sum(bucket(p) == bucket(a) for p, a in rows) / n, 3),
            "long_recall": round(sum(p > BUCKETS[1] and a > BUCKETS[1] for p, a in rows)
                                 / max(1, sum(a > BUCKETS[1] for p, a in rows)), 3),
            "long_precision": round(sum(p > BUCKETS[1] and a > BUCKETS[1] for p, a in rows)
                                    / max(1, sum(p > BUCKETS[1] for p, a in rows)), 3)}


def predictors(history: list[int]) -> dict[str, float]:
    """Causal predictors from the session's earlier recorded lengths; constant when there is no history."""
    n = len(history)
    if n == 0:
        return {k: float(TRAIN_MEDIAN) for k in ("last", "running_median", "ewma", "shrunk_median")}
    ewma = history[0]
    for h in history[1:]:
        ewma = 0.5 * ewma + 0.5 * h
    med = statistics.median(history)
    return {"last": float(history[-1]), "running_median": float(med), "ewma": float(ewma),
            "shrunk_median": (n * med + 3 * TRAIN_MEDIAN) / (n + 3)}  # pseudo-count 3 toward the train median


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    test_ids = [json.loads(l)["sample_id"] for l in (DATASET / "egtp-static/predictions.jsonl").open()]
    labels = {}
    for l in (DATASET / "dataset/source_labels.jsonl").open():
        r = json.loads(l)
        if r["draw_id"] == 0:
            labels[r["sample_id"]] = r["actual_tokens"]

    # sample id: <model>/<task>/<session>/<action_id>; swe sessions are keyed by (instance_id, model), tb by task_id
    swe_files = {}  # the dataset sampled all 644 trajectories, completed or not
    for m in map(json.loads, (POOLS["swe"] / "MANIFEST.jsonl").open()):
        instance = m.get("instance_id") or m["flattened_name"].split("__", 1)[1].removesuffix(".trace.jsonl")
        swe_files[(instance, m["model"])] = POOLS["swe"] / m["flattened_name"]
    tb_files = {m["task_id"]: POOLS["tb"] / m["flattened_name"] for m in map(json.loads, (POOLS["tb"] / "MANIFEST.jsonl").open())}
    cache: dict[Path, list[dict]] = {}
    enc = tiktoken.get_encoding("o200k_base")  # token split of the recorded output is approximate: proportional to tiktoken counts

    per_pred: dict[str, list[tuple[float, float]]] = defaultdict(list)
    per_pred_hist: dict[str, list[tuple[float, float]]] = defaultdict(list)  # samples with at least one earlier step
    decomposition = []
    predictions = []
    for sid in test_ids:
        model, task, _session, action_id = sid.split("/")
        trace = swe_files.get((task, model)) if (task, model) in swe_files else tb_files[task]
        steps = cache.setdefault(trace, steps_of(trace))
        k = next(i for i, s in enumerate(steps) if s["action_id"] == action_id)
        actual = labels[sid]
        assert actual == steps[k]["completion_tokens"], (sid, actual, steps[k]["completion_tokens"])
        history = [s["completion_tokens"] for s in steps[:k]]
        preds = {"constant": float(TRAIN_MEDIAN), **predictors(history)}
        for name, v in preds.items():
            per_pred[name].append((v, actual))
            if history:
                per_pred_hist[name].append((v, actual))
        predictions.append({"sample_id": sid, "history_len": len(history), **{f"pred_{n}": round(v, 1) for n, v in preds.items()}})
        t_text, t_args = len(enc.encode(steps[k]["text"])), len(enc.encode(steps[k]["args"]))
        share_args = t_args / (t_text + t_args) if (t_text + t_args) else 0.0
        decomposition.append({"sample_id": sid, "model": model, "actual": actual, "n_tool_calls": steps[k]["n_tool_calls"],
                              "args_tokens_est": round(actual * share_args, 1), "text_tokens_est": round(actual * (1 - share_args), 1),
                              "measured_tokens": t_text + t_args})

    a.out.mkdir(parents=True, exist_ok=True)
    evaluation = {"all_test_samples": {n: evaluate(r) for n, r in per_pred.items()},
                  "samples_with_history": {n: evaluate(r) for n, r in per_pred_hist.items()}}
    args_share = [d["args_tokens_est"] / d["actual"] for d in decomposition if d["actual"] > 0]
    var_total = statistics.pvariance(d["actual"] for d in decomposition)
    var_args = statistics.pvariance(d["args_tokens_est"] for d in decomposition)
    var_text = statistics.pvariance(d["text_tokens_est"] for d in decomposition)
    by_model = defaultdict(list)
    for d in decomposition:
        by_model[d["model"]].append(d)
    evaluation["decomposition"] = {
        "note": "token split is proportional to tiktoken o200k counts of the recorded text and tool-call arguments; "
                "recorded completion tokens include hidden reasoning for gpt-5.6-sol, so measured_tokens / actual is the visible share",
        "args_share_mean": round(statistics.fmean(args_share), 3), "args_share_median": round(statistics.median(args_share), 3),
        "variance_share_args": round(var_args / var_total, 3), "variance_share_text": round(var_text / var_total, 3),
        "visible_share_by_model": {m: round(statistics.fmean(d["measured_tokens"] / max(1, d["actual"]) for d in ds), 3)
                                   for m, ds in by_model.items()},
        "actual_mean_by_model": {m: round(statistics.fmean(d["actual"] for d in ds), 1) for m, ds in by_model.items()}}
    (a.out / "evaluation.json").write_text(json.dumps(evaluation, indent=1) + "\n")
    (a.out / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in predictions))
    (a.out / "decomposition.jsonl").write_text("".join(json.dumps(r) + "\n" for r in decomposition))

    for scope in ("all_test_samples", "samples_with_history"):
        print(f"\n{scope}")
        print("| predictor | n | q50 | q90 | q95 | q99 | mean q | MAE | under | bucket acc | long recall | long prec |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for n, e in evaluation[scope].items():
            print(f"| {n} | {e['n']} | {e['q50']} | {e['q90']} | {e['q95']} | {e['q99']} | {e['mean_q']} | {e['mae']} | "
                  f"{e['underprediction_rate']} | {e['bucket_accuracy']} | {e['long_recall']} | {e['long_precision']} |")
    print("\ndecomposition", json.dumps(evaluation["decomposition"], indent=1))


if __name__ == "__main__":
    main()
