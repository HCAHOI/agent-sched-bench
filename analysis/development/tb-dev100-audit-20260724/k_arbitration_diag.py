#!/usr/bin/env python3
"""EXPLORATORY dev diagnostic on the TB dev-100 split (seed=42 first 100).

Never reads the remaining 139 confirmation tasks. Compares two-layer lattice
repo/public arbitration k=1 (current champion rule) vs k=5 (proposed R1) on
TB dev-100, using the exact champion machinery. Task-clustered uncertainty is
reported but this is NOT a confirmatory read.
"""
from __future__ import annotations

import heapq
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("src").resolve()))
sys.path.insert(0, str(Path(".").resolve()))

from scripts.evaluation.evaluate_two_layer_resource import (  # noqa: E402
    BackoffLattice,
    _node_keys,
    _target_values,
    repo_key,
)
from tool_resource.labels import extract_resource_call_samples, load_resource_corpus  # noqa: E402
from tool_resource.metrics import ecdf_quantile, pinball_loss  # noqa: E402
from tool_time.statistics import resample_task_totals  # noqa: E402

P90 = 0.9
DEV100 = json.load(
    open(
        "/tmp/claude-1000/-home-chiyu-workspace-agent-sched-bench/"
        "20898024-ea7d-4f6a-88cd-d4d9fc0f8cc1/scratchpad/tb_dev100.json"
    )
)["dev_task_ids"]


def select_k(lattice: BackoffLattice, sample, target: str, k: int):
    """Champion select() plus a min-repo-count gate (k=1 == champion rule)."""
    repo = repo_key(sample.task_id)
    for key, granularity in _node_keys(sample):
        repo_values = lattice.repo_nodes.get((repo, target, key))
        if repo_values and len(repo_values) >= k:
            return repo_values, "repo", granularity
        public_values = lattice.public_nodes.get((target, key))
        if public_values:
            return public_values, "public", granularity
    return lattice.public_global[target], "public", "global"


def main() -> None:
    fit_by, _ = load_resource_corpus(
        Path("traces/swe-rebench/qwen3.7-max/offline-gated-confirm-100-v2"),
        Path("configs/corpora/swe-100.json"),
    )
    fit = [s for ss in fit_by.values() for s in ss]
    lattice0 = BackoffLattice.from_fit_samples(fit)

    manifest = json.load(open("traces/terminal-bench/tb-all/manifest.json"))
    by_id = {t["task_id"]: t for t in manifest["tasks"]}
    rows = []
    for tid in DEV100:
        trace = Path("traces/terminal-bench/tb-all") / by_id[tid]["trace_file"]
        rows.extend(extract_resource_call_samples(trace))
    rows.sort(key=lambda s: (s.tool_ts_start, s.sample_id))
    per_task = Counter(s.task_id for s in rows)
    print(f"TB dev-100 rows={len(rows)} tasks={len(per_task)} "
          f"calls/task p50={int(np.median(list(per_task.values())))} "
          f"p90={int(np.percentile(list(per_task.values()), 90))}")

    results: dict[int, dict] = {}
    for k in (1, 5):
        lattice = BackoffLattice()
        lattice.public_nodes = lattice0.public_nodes
        lattice.public_global = lattice0.public_global
        lattice.repo_nodes = defaultdict(list)
        pending: list[tuple[float, str, object]] = []
        loss_by_target = defaultdict(lambda: defaultdict(list))  # target -> task -> losses
        cover = Counter(); n_rows = Counter()
        bucket = defaultdict(lambda: [0, 0, 0.0])  # (target, callpos_bucket) -> [n, repo_used, loss_sum]
        call_index: Counter = Counter()
        for s in rows:
            while pending and pending[0][0] < s.tool_ts_start:
                _, _, done = heapq.heappop(pending)
                lattice.add_repo_observation(done)
            call_index[s.task_id] += 1
            pos = min(call_index[s.task_id], 6)
            for target, observed in _target_values(s).items():
                values, scope, _g = select_k(lattice, s, target, k)
                pred = ecdf_quantile(values, P90)
                loss = pinball_loss(observed, pred, P90)
                loss_by_target[target][s.task_id].append(loss)
                cover[target] += observed <= pred
                n_rows[target] += 1
                b = bucket[(target, pos)]
                b[0] += 1; b[1] += scope == "repo"; b[2] += loss
            heapq.heappush(pending, (s.tool_ts_end, s.sample_id, s))
        results[k] = {"loss": loss_by_target, "cover": cover, "n": n_rows, "bucket": bucket}

    for target in ("latency_ms", "peak_memory_mb", "peak_cpu_cores"):
        n = results[1]["n"][target]
        if n == 0:
            print(f"\n== {target}: no eligible rows =="); continue
        tasks = sorted(results[1]["loss"][target])
        totals = {k: np.array([sum(results[k]["loss"][target].get(t, [])) for t in tasks]) for k in (1, 5)}
        mean = {k: totals[k].sum() / n for k in (1, 5)}
        cov = {k: results[k]["cover"][target] / n for k in (1, 5)}
        contrib = np.stack([totals[5], totals[1]], axis=1)
        resampled = resample_task_totals(contrib, replicates=20_000, seed=0)
        skills = 1.0 - resampled[:, 0] / np.maximum(resampled[:, 1], 1e-12)
        ci_low, ci_high = np.percentile(skills, [2.5, 97.5])
        print(f"\n== {target} (n={n}, tasks={len(tasks)}) ==")
        print(f"  k=1: mean p90 pinball={mean[1]:.3f}  coverage={cov[1]:.3f}")
        print(f"  k=5: mean p90 pinball={mean[5]:.3f}  coverage={cov[5]:.3f}")
        print(f"  skill(k5 vs k1)={1 - mean[5]/mean[1]:+.4f}  "
              f"task-clustered 95% CI [{ci_low:+.4f}, {ci_high:+.4f}]  (EXPLORATORY)")
        print("  by call position (pos: n | repo-used% k1->k5 | mean loss k1->k5):")
        for pos in range(1, 7):
            b1 = results[1]["bucket"].get((target, pos)); b5 = results[5]["bucket"].get((target, pos))
            if not b1: continue
            print(f"    pos{'6+' if pos == 6 else pos}: n={b1[0]:5d} | "
                  f"{b1[1]/b1[0]:5.1%} -> {b5[1]/b5[0]:5.1%} | "
                  f"{b1[2]/b1[0]:9.2f} -> {b5[2]/b5[0]:9.2f}")


if __name__ == "__main__":
    main()
