#!/usr/bin/env python3
"""Are this lane's negative results an artifact of the restore charge?

Criterion frozen in analysis/CLAIMS.md, "Open question -- are the negative results an
artifact of the restore charge?", commit e99f439, before this existed.

Every negative result to date charged a flat restore_cost_fraction of 0.94, i.e.
always reload over PCIe. The repo's own prefill findings state the intended design is
the CHEAPER of reload or recompute, so 0.94 is an upper bound and every conclusion
sits under the most pessimistic restore assumption available.

This sweeps rho and locates where each conclusion flips:

  deadline_only   fire at T = kv, recomputed at each rho
  best_constant   one scalar fitted out-of-fold, as in 24d4344
  per_key_loo     cmd_depth_4 leave-one-out within the evaluation fold, as in f1cbee3

rho = 0 is included deliberately: a free restore is the most generous assumption
physically possible, so if nothing beats the deadline even there, the negative result
is unconditional in the restore charge.

PREMISE. Forced eviction; with no memory pressure every arm scores zero. Fresh-277
cannot exhibit contention. Hidden-swap-milliseconds, not wall clock. rho and kv are
swept parameters of the cost model, not derived from context lengths in this corpus.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts" / "exploration"))

from evaluate_key_ladder import build_keyers  # noqa: E402
from tool_time.offline_evaluation import balanced_task_folds  # noqa: E402
from tool_time.policy import trigger_policy_utility_ms  # noqa: E402
from tool_time.prior import hazard_recheck_ms  # noqa: E402
from trace_collect.tool_latency_dataset import load_tool_latency_corpus  # noqa: E402

KV_COSTS = (3500.0, 5000.0)
RHOS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.94)
KEY = "cmd_depth_4"


def utility(L: float, trig: float, *, th: float, kv: float, restore: float) -> float:
    return trigger_policy_utility_ms(
        L, trig, threshold_ms=th, kv_cost_ms=kv, restore_cost_ms=restore
    )


def best_constant(
    lat: Sequence[float], *, th: float, kv: float, restore: float
) -> float:
    step = 50.0
    best_t, best_v = th, sum(utility(L, th, th=th, kv=kv, restore=restore) for L in lat)
    for i in range(int(2.0 * kv / step) + 1):
        t = i * step
        v = sum(utility(L, t, th=th, kv=kv, restore=restore) for L in lat)
        if v > best_v:
            best_v, best_t = v, t
    return best_t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()

    sb, task_ids, manifest = load_tool_latency_corpus(args.manifest, final=args.final)
    rows: list[dict[str, Any]] = []
    for t in task_ids:
        for s in sb[t]:
            r = s.to_json_obj()
            r["task_id"] = t
            rows.append(r)
    keyer = build_keyers(manifest)[KEY]
    for r in rows:
        r["_k"] = keyer(r)

    folds = balanced_task_folds(rows, fold_count=manifest["fold_count"])
    guard = float(manifest.get("guard_ms", 0.0))

    cells: dict[str, Any] = {}
    for kv in KV_COSTS:
        th = kv + guard
        per_rho: dict[str, Any] = {}
        for rho in RHOS:
            restore = rho * kv
            dead = const = loo = oracle = 0.0
            for fold_tasks in folds:
                ev = [r for r in rows if r["task_id"] in fold_tasks]
                pr = [r["latency_ms"] for r in rows if r["task_id"] not in fold_tasks]
                t_const = best_constant(pr, th=th, kv=kv, restore=restore)

                groups: dict[str, list[float]] = defaultdict(list)
                for r in ev:
                    groups[r["_k"]].append(r["latency_ms"])
                loo_trig: dict[tuple[str, int], float] = {}
                for k, vals in groups.items():
                    if len(vals) == 1:
                        loo_trig[(k, 0)] = th
                        continue
                    for i in range(len(vals)):
                        loo_trig[(k, i)] = hazard_recheck_ms(
                            vals[:i] + vals[i + 1:], threshold_ms=th,
                            kv_cost_ms=kv, restore_cost_ms=restore,
                        )
                seen: dict[str, int] = defaultdict(int)
                for r in ev:
                    L = r["latency_ms"]
                    k = r["_k"]
                    i = seen[k]
                    seen[k] += 1
                    dead += utility(L, th, th=th, kv=kv, restore=restore)
                    const += utility(L, t_const, th=th, kv=kv, restore=restore)
                    loo += utility(L, loo_trig[(k, i)], th=th, kv=kv, restore=restore)
                    oracle += 0.0 if L <= th else utility(
                        L, max(0.0, L - kv), th=th, kv=kv, restore=restore
                    )
            budget = oracle - dead
            per_rho[f"{rho:.2f}"] = {
                "deadline_only_s": dead / 1000.0,
                "best_constant_s": const / 1000.0,
                "per_key_loo_s": loo / 1000.0,
                "oracle_s": oracle / 1000.0,
                "budget_s": budget / 1000.0,
                "best_constant_gain_s": (const - dead) / 1000.0,
                "per_key_loo_gain_s": (loo - dead) / 1000.0,
                "best_constant_beats_deadline": const > dead,
                "per_key_loo_beats_deadline": loo > dead,
            }
        flips = {
            arm: next(
                (r for r in (f"{x:.2f}" for x in RHOS)
                 if per_rho[r][f"{arm}_beats_deadline"]),
                None,
            )
            for arm in ("best_constant", "per_key_loo")
        }
        cells[str(int(kv))] = {"kv_cost_ms": kv, "by_rho": per_rho,
                               "lowest_rho_that_beats_deadline": flips}

    payload = {
        "criterion_frozen_in": "e99f439",
        "premise": "forced eviction; hidden-swap-ms not wall clock; Fresh-277 cannot "
                   "exhibit contention; rho and kv are swept cost-model parameters",
        "tasks": len(task_ids),
        "cells": cells,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    for kv, c in cells.items():
        print(f"kv={kv}: flips at {c['lowest_rho_that_beats_deadline']}")


if __name__ == "__main__":
    main()
