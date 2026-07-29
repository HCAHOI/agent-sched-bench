#!/usr/bin/env python3
"""Is firing at T = kv the best CONSTANT trigger, or is the baseline suboptimal?

Criterion frozen in analysis/CLAIMS.md, "Open question -- is the deadline itself the
best constant policy?", commit 8d72166, before this existed.

Every comparison in this lane used one baseline: fire at trigger = threshold = kv.
The budget lies entirely in calls with kv < L < 2*kv, which a LOWER constant trigger
reaches earlier, at the price of firing on calls that end before kv and paying the
differential restore. This sweeps that one parameter.

  deadline_only     T = kv. The baseline everyone uses.
  best_constant     one global T fitted on the PROFILE folds by maximising net
                    utility, applied unchanged to the evaluation fold. Deployable:
                    it is prediction-free and fits a single scalar out-of-fold.
  ORACLE_constant   best T fitted IN-SAMPLE on the evaluation rows. ANALYSIS-ONLY,
                    reported to expose overfit, which f1cbee3 showed is essential.

The label threshold separating long from short stays at kv, the physical condition
for a call to be able to hide the swap. Only the trigger varies.

PREMISE. Forced eviction; with no memory pressure every arm including every oracle
scores zero. Fresh-277 cannot exhibit contention. Hidden-swap-milliseconds, not wall
clock.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from tool_time.offline_evaluation import balanced_task_folds  # noqa: E402
from tool_time.policy import trigger_policy_utility_ms  # noqa: E402
from trace_collect.tool_latency_dataset import load_tool_latency_corpus  # noqa: E402

KV_COSTS = (3500.0, 5000.0)
RESTORE_FRACTION = 0.94


def candidate_triggers(kv: float) -> list[float]:
    """Fixed grid, declared here rather than fitted, from 0 to 2*kv in 50 ms steps.

    The upper end is 2*kv because a trigger beyond that fires later than every call
    the budget lives in, and the lower end is 0, firing immediately on every call.
    """

    step = 50.0
    n = int((2.0 * kv) / step)
    return [i * step for i in range(n + 1)]


def total_utility(
    latencies: Sequence[float], trigger: float, *, th: float, kv: float, restore: float
) -> float:
    return sum(
        trigger_policy_utility_ms(
            L, trigger, threshold_ms=th, kv_cost_ms=kv, restore_cost_ms=restore
        )
        for L in latencies
    )


def best_trigger(
    latencies: Sequence[float], *, th: float, kv: float, restore: float
) -> tuple[float, float]:
    best_t, best_v = th, total_utility(latencies, th, th=th, kv=kv, restore=restore)
    for t in candidate_triggers(kv):
        v = total_utility(latencies, t, th=th, kv=kv, restore=restore)
        if v > best_v:
            best_v, best_t = v, t
    return best_t, best_v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()

    samples_by_task, task_ids, manifest = load_tool_latency_corpus(
        args.manifest, final=args.final
    )
    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        for s in samples_by_task[task_id]:
            rows.append({"task_id": task_id, "latency_ms": s.latency_ms})

    folds = balanced_task_folds(rows, fold_count=manifest["fold_count"])
    guard = float(manifest.get("guard_ms", 0.0))

    cells: dict[str, Any] = {}
    for kv in KV_COSTS:
        th = kv + guard
        restore = RESTORE_FRACTION * kv
        deadline = oracle_call = best_c = oracle_c = 0.0
        fitted: list[float] = []
        insample_t: list[float] = []

        for fold_tasks in folds:
            ev = [r["latency_ms"] for r in rows if r["task_id"] in fold_tasks]
            pr = [r["latency_ms"] for r in rows if r["task_id"] not in fold_tasks]
            t_fit, _ = best_trigger(pr, th=th, kv=kv, restore=restore)
            t_ins, _ = best_trigger(ev, th=th, kv=kv, restore=restore)
            fitted.append(t_fit)
            insample_t.append(t_ins)
            deadline += total_utility(ev, th, th=th, kv=kv, restore=restore)
            best_c += total_utility(ev, t_fit, th=th, kv=kv, restore=restore)
            oracle_c += total_utility(ev, t_ins, th=th, kv=kv, restore=restore)
            for L in ev:
                oracle_call += 0.0 if L <= th else trigger_policy_utility_ms(
                    L, max(0.0, L - kv), threshold_ms=th, kv_cost_ms=kv,
                    restore_cost_ms=restore,
                )

        budget = oracle_call - deadline
        cells[str(int(kv))] = {
            "kv_cost_ms": kv,
            "deadline_trigger_ms": th,
            "net_saved_s": {
                "deadline_only": deadline / 1000.0,
                "best_constant": best_c / 1000.0,
                "ORACLE_constant": oracle_c / 1000.0,
                "ORACLE_per_call": oracle_call / 1000.0,
            },
            "budget_s": budget / 1000.0,
            "best_constant_gain_s": (best_c - deadline) / 1000.0,
            "best_constant_gain_fraction_of_budget": (best_c - deadline) / budget,
            "oracle_constant_gain_fraction_of_budget": (oracle_c - deadline) / budget,
            "fitted_trigger_ms_per_fold": fitted,
            "in_sample_trigger_ms_per_fold": insample_t,
        }

    payload = {
        "criterion_frozen_in": "8d72166",
        "premise": "forced eviction; hidden-swap-ms not wall clock; Fresh-277 cannot "
                   "exhibit contention",
        "tasks": len(task_ids),
        "cells": cells,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(cells, indent=2))


if __name__ == "__main__":
    main()
