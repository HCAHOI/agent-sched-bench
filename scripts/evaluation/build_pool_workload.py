"""Build a distinct-trace replay workload from the exported trace pool.

Selects N trajectories (no replay copies) by proportional stratified sampling
over (corpus, agent model, peak-context bucket), writes the simulate manifest,
the task source, per-trace statistics and a provenance record that carries the
pre-registered pressure ratios:

  R_avg  = concurrency x mean prompt tokens per LLM step (step-weighted) / KV tokens
  R_peak = concurrency x mean peak prompt tokens per trace / KV tokens

Usage:
  .venv/bin/python scripts/evaluation/build_pool_workload.py --out analysis/development/pool64-distinct-v1 \
      --pool traces/exports/swe-rebench-original-flat-644-20260904 \
      --pool traces/exports/terminal-bench-original-flat-239-20260904 \
      --n 64 --seed 42 --concurrency 32 --capacity n2-full=1249760 --capacity capped=400000
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
BUCKETS = ((16_000, "<16K"), (32_000, "16-32K"), (64_000, "32-64K"), (float("inf"), ">=64K"))


def bucket(peak: int) -> str:
    return next(name for limit, name in BUCKETS if peak < limit)


def trace_stats(path: Path) -> dict:
    n_llm = n_tool = peak = prompt_sum = completion_sum = 0
    llm_s = tool_s = 0.0
    t0 = t1 = None
    metadata_id = None
    with path.open() as fh:
        for line in fh:
            if metadata_id is None and '"trace_metadata"' in line:  # metadata is the first record; its type key is not near the front
                metadata_id = json.loads(line).get("instance_id")
                continue
            if '"action"' not in line[:40]:
                continue
            r = json.loads(line)
            if r.get("type") != "action":
                continue
            t0 = r["ts_start"] if t0 is None else min(t0, r["ts_start"])
            t1 = r["ts_end"] if t1 is None else max(t1, r["ts_end"])
            if r["action_type"] == "llm_call":
                p = r["data"].get("prompt_tokens") or 0
                n_llm += 1
                peak = max(peak, p)
                prompt_sum += p
                completion_sum += r["data"].get("completion_tokens") or 0
                llm_s += r["ts_end"] - r["ts_start"]
            elif r["action_type"] == "tool_exec":
                n_tool += 1
                tool_s += r["ts_end"] - r["ts_start"]
    return {"metadata_instance_id": metadata_id, "n_llm": n_llm, "n_tool": n_tool, "peak_prompt_tokens": peak,
            "sum_prompt_tokens": prompt_sum, "sum_completion_tokens": completion_sum,
            "llm_s": round(llm_s, 1), "tool_s": round(tool_s, 1), "wall_s": round((t1 or 0) - (t0 or 0), 1)}


def load_pool(pool_dirs: list[Path]) -> list[dict]:
    rows = []
    for d in pool_dirs:
        for m in map(json.loads, (d / "MANIFEST.jsonl").open()):
            if m.get("status") not in (None, "completed"):
                continue
            path = d / m["flattened_name"]
            s = trace_stats(path)
            if s["n_llm"] < 2:
                continue
            corpus = "terminal-bench" if "terminal-bench" in d.name else "swe-rebench"
            rows.append({"corpus": corpus, "pool_dir": d.name, "file": m["flattened_name"], "trace": str(path.resolve()),
                         "instance_id": s["metadata_instance_id"] or m.get("instance_id") or m.get("task_id"), "model": m.get("model"), **s,
                         "bucket": bucket(s["peak_prompt_tokens"])})
    return rows


def allocate(pool: list[dict], n: int) -> dict[tuple, int]:
    """Largest-remainder allocation of n picks to (corpus, model, bucket) strata, proportional to pool size."""
    counts = Counter((r["corpus"], r["model"], r["bucket"]) for r in pool)
    total = sum(counts.values())
    quotas = {k: n * c / total for k, c in counts.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    for k in sorted(quotas, key=lambda k: quotas[k] - alloc[k], reverse=True)[: n - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", action="append", type=Path, required=True)
    p.add_argument("--tasks", type=Path, default=REPO / "data/swe-rebench/tasks.json", help="swe-rebench task records")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--capacity", action="append", default=[], help="NAME=total KV tokens, for the pressure table")
    p.add_argument("--docker-image", default="python:3.13-slim-bookworm")
    p.add_argument("--pool-stats", type=Path, help="reuse a pool-stats.csv from an earlier build instead of rescanning the traces")
    a = p.parse_args()

    if a.pool_stats:
        pool = []
        for r in csv.DictReader(a.pool_stats.open()):
            r.pop("selected", None)
            for k in ("n_llm", "n_tool", "peak_prompt_tokens", "sum_prompt_tokens", "sum_completion_tokens"):
                r[k] = int(r[k])
            for k in ("llm_s", "tool_s", "wall_s"):
                r[k] = float(r[k])
            pool.append(r)
        ids = {}
        for d in a.pool:
            for m in map(json.loads, (d / "MANIFEST.jsonl").open()):
                ids[m["flattened_name"]] = m.get("instance_id") or m.get("task_id")
        for r in pool:
            r["instance_id"] = r["instance_id"] or ids[r["file"]]
        assert all(r["instance_id"] for r in pool)
    else:
        pool = load_pool(a.pool)
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "pool-stats.csv").open("w") as fh:  # written before selection so a failed selection keeps the scan
        w = csv.DictWriter(fh, fieldnames=list(pool[0].keys()))
        w.writeheader()
        w.writerows(pool)
    alloc = allocate(pool, a.n)
    rng = random.Random(a.seed)
    chosen, used = [], set()
    for key, k in sorted(alloc.items()):
        members = sorted((r for r in pool if (r["corpus"], r["model"], r["bucket"]) == key), key=lambda r: r["file"])
        rng.shuffle(members)
        picks = [r for r in members if r["instance_id"] not in used][:k]  # one trace per task: two agents ran some instances
        assert len(picks) == k, f"stratum {key} exhausted"
        chosen.extend(picks)
        used.update(r["instance_id"] for r in picks)
    chosen.sort(key=lambda r: r["file"])
    assert len(chosen) == a.n and len({r["instance_id"] for r in chosen}) == a.n, "duplicate instance ids in selection"

    swe_tasks = {t["instance_id"]: t for t in json.loads(a.tasks.read_text())}
    task_source = []
    for r in chosen:
        if r["corpus"] == "swe-rebench":
            task_source.append(swe_tasks[r["instance_id"]])
        else:  # tool replay never runs the real task; the record only has to resolve the trace's instance id
            task_source.append({"instance_id": r["instance_id"], "task_id": r["instance_id"], "task_source_kind": "terminal-bench"})

    (a.out / "task-source.json").write_text(json.dumps(task_source, indent=1) + "\n")
    manifest = {"version": 1, "requires_trace_tool_replay": True,
                "defaults": {"task_source": "task-source.json", "docker_image": a.docker_image},
                "traces": [{"label": r["instance_id"], "trace": r["trace"]} for r in chosen]}
    (a.out / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    (a.out / "selected.txt").write_text("".join(r["file"] + "\n" for r in chosen))

    steps = sum(r["n_llm"] for r in chosen)
    mean_prompt_per_step = sum(r["sum_prompt_tokens"] for r in chosen) / steps
    mean_peak = statistics.fmean(r["peak_prompt_tokens"] for r in chosen)
    pressure = {}
    for spec in a.capacity:
        name, cap = spec.split("=")
        cap = int(cap)
        pressure[name] = {"kv_tokens": cap, "R_avg": round(a.concurrency * mean_prompt_per_step / cap, 3),
                          "R_peak": round(a.concurrency * mean_peak / cap, 3)}
    strata = defaultdict(int)
    for r in chosen:
        strata[f"{r['corpus']} | {r['model']} | {r['bucket']}"] += 1
    prov = {"pool_dirs": [str(d) for d in a.pool], "pool_size": len(pool), "n": a.n, "seed": a.seed,
            "selection": "proportional stratified over (corpus, agent model, peak-context bucket), largest-remainder "
                         "quotas, seeded sample within stratum; traces referenced in place, no replay copies",
            "strata": dict(sorted(strata.items())), "concurrency": a.concurrency,
            "selected_summary": {"steps_total": steps, "mean_prompt_tokens_per_step": round(mean_prompt_per_step),
                                 "mean_peak_prompt_tokens": round(mean_peak),
                                 "peak_p50": statistics.median(r["peak_prompt_tokens"] for r in chosen),
                                 "steps_p50": statistics.median(r["n_llm"] for r in chosen),
                                 "tool_s_p50": statistics.median(r["tool_s"] for r in chosen),
                                 "corpus": dict(Counter(r["corpus"] for r in chosen)),
                                 "model": dict(Counter(r["model"] for r in chosen))},
            "pressure": pressure}
    (a.out / "input-provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    print(json.dumps(prov, indent=1))


if __name__ == "__main__":
    main()
