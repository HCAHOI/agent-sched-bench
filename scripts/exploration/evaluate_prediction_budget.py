#!/usr/bin/env python3
"""Bound the total value of duration prediction: ORACLE minus fixed deadline.

Criterion frozen in analysis/CLAIMS.md, section "Open question -- what is the total
budget available to any duration predictor?", commit f81890f, before this existed.

The mechanism is held fixed -- one stopping time scored by
tool_time.policy.trigger_policy_utility_ms -- and only the predictor varies:

  never_act       0 by construction
  deadline_only   fire at the threshold; no prediction
  mean_hazard     shipped predictor
  robust_clock    shipped predictor
  ORACLE          perfect foreknowledge of the realised latency L; fires only when
                  L > threshold, at max(0, L - kv), the latest trigger that still
                  hides the full swap. ANALYSIS-ONLY. Never deployable, and its
                  per-call triggers are never used as features by any other arm.

PREMISE. The functional credits min(kv, remaining) for hiding swap-out and charges
the differential restore only on short calls, which is coherent only under FORCED
EVICTION. With no memory pressure the optimal policy is never swap and every arm
scores zero. Fresh-277 cannot exhibit contention (CLOSED-QUESTIONS.md), so every
number here is hidden-swap-milliseconds under an assumed regime, not wall clock.

Causal contract: the deployable arms read profile-fold evidence only, with
profile/eval task disjointness asserted inside evaluate_utility_clock_policy. The
oracle reads the realised latency of the row it scores and is reported separately.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from tool_time.offline_evaluation import balanced_task_folds  # noqa: E402
from tool_time.policy import (  # noqa: E402
    evaluate_utility_clock_policy,
    trigger_policy_utility_ms,
)
from trace_collect.tool_latency_dataset import load_tool_latency_corpus  # noqa: E402

KV_COSTS = (3500.0, 5000.0)
RESTORE_FRACTION = 0.94
DRAWS = 10000
SEED = 0
_TRIGGER_FIELD = {
    "deadline_only": "deadline_trigger_ms",
    "mean_hazard": "mean_hazard_trigger_ms",
    "robust_clock": "robust_trigger_ms",
}


def oracle_trigger_ms(latency_ms: float, threshold_ms: float, kv_cost_ms: float) -> float | None:
    """Latest trigger that still hides the whole swap; None means do not fire.

    Firing is profitable only on a call that outlives the threshold. Given perfect
    knowledge of L, the best such trigger is L - kv when that is non-negative,
    which hides exactly kv with zero exposure; if L < kv the best is 0.
    """

    if latency_ms <= threshold_ms:
        return None
    return max(0.0, latency_ms - kv_cost_ms)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()

    samples_by_task, task_ids, manifest = load_tool_latency_corpus(
        args.manifest, final=args.final
    )
    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        for sample in samples_by_task[task_id]:
            row = sample.to_json_obj()
            row["task_id"] = task_id
            rows.append(row)

    folds = balanced_task_folds(rows, fold_count=manifest["fold_count"])
    pooled: list[dict[str, Any]] = []
    for fold_tasks in folds:
        eval_rows = [r for r in rows if r["task_id"] in fold_tasks]
        profile_rows = [r for r in rows if r["task_id"] not in fold_tasks]
        if not eval_rows or not profile_rows:
            raise ValueError("empty fold side")
        result = evaluate_utility_clock_policy(
            eval_rows,
            profile_rows=profile_rows,
            kv_costs_ms=list(KV_COSTS),
            guard_ms=float(manifest.get("guard_ms", 0.0)),
            min_tool_history=manifest["min_tool_history"],
            min_profile_tasks=manifest["min_profile_tasks"],
            command_field=manifest["command_field"],
            max_prefix_depth=manifest["max_prefix_depth"],
            skip_leading_cd=manifest["skip_leading_cd"],
            restore_cost_fraction=RESTORE_FRACTION,
        )
        pooled.extend(result["decisions"])

    cells: dict[str, Any] = {}
    for kv in KV_COSTS:
        restore = RESTORE_FRACTION * kv
        totals: dict[str, float] = defaultdict(float)
        by_task: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        n = 0
        for row in pooled:
            if row["kv_cost_ms"] != kv:
                continue
            n += 1
            th = row["threshold_ms"]
            lat = row["latency_ms"]
            for policy, field in _TRIGGER_FIELD.items():
                totals[policy] += trigger_policy_utility_ms(
                    lat, row[field], threshold_ms=th, kv_cost_ms=kv,
                    restore_cost_ms=restore,
                )
            ot = oracle_trigger_ms(lat, th, kv)
            u_oracle = 0.0 if ot is None else trigger_policy_utility_ms(
                lat, ot, threshold_ms=th, kv_cost_ms=kv, restore_cost_ms=restore
            )
            totals["ORACLE"] += u_oracle
            u_dead = trigger_policy_utility_ms(
                lat, row["deadline_trigger_ms"], threshold_ms=th, kv_cost_ms=kv,
                restore_cost_ms=restore,
            )
            by_task[row["task_id"]]["oracle_minus_deadline"] += u_oracle - u_dead
            by_task[row["task_id"]]["robust_minus_deadline"] += (
                trigger_policy_utility_ms(
                    lat, row["robust_trigger_ms"], threshold_ms=th, kv_cost_ms=kv,
                    restore_cost_ms=restore,
                ) - u_dead
            )

        tasks = sorted(by_task)
        rng = random.Random(SEED)
        def boot(field: str) -> dict[str, float]:
            vals = [by_task[t][field] for t in tasks]
            m = len(vals)
            draws = sorted(
                sum(vals[rng.randrange(m)] for _ in range(m)) / 1000.0
                for _ in range(DRAWS)
            )
            return {"ci_low_s": draws[int(0.025 * DRAWS)],
                    "ci_high_s": draws[int(0.975 * DRAWS)]}

        budget = (totals["ORACLE"] - totals["deadline_only"]) / 1000.0
        captured = (totals["robust_clock"] - totals["deadline_only"]) / 1000.0
        cells[str(int(kv))] = {
            "kv_cost_ms": kv, "rows": n,
            "net_saved_s": {k: v / 1000.0 for k, v in sorted(totals.items())},
            "never_act_s": 0.0,
            "budget_oracle_minus_deadline_s": budget,
            "budget_as_fraction_of_deadline_utility":
                budget / (totals["deadline_only"] / 1000.0)
                if totals["deadline_only"] else None,
            "robust_capture_s": captured,
            "robust_fraction_of_budget": captured / budget if budget else None,
            "bootstrap_oracle_minus_deadline": boot("oracle_minus_deadline"),
            "bootstrap_robust_minus_deadline": boot("robust_minus_deadline"),
        }

    payload = {
        "criterion_frozen_in": "f81890f",
        "premise": "forced eviction; hidden-swap-ms, not wall clock; Fresh-277 cannot "
                   "exhibit contention",
        "manifest": str(args.manifest),
        "tasks": len(task_ids),
        "restore_cost_fraction": RESTORE_FRACTION,
        "oracle_is_analysis_only": True,
        "cells": cells,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(cells, indent=2))


if __name__ == "__main__":
    main()
