#!/usr/bin/env python3
"""Replicate the KV-lane arms on a second declared corpus.

Criterion frozen in analysis/CLAIMS.md, "Open question -- does the KV-lane negative
replicate on a second corpus?", commit 1b8a57d, before this existed.

Every result in the lane was computed on Fresh-277. This runs the identical arms and
cells against any declared corpus manifest so the two can be compared directly:

  deadline_only    fire at T = kv. No prediction.
  best_constant    one scalar fitted on profile folds, applied out-of-fold.
  per_key_loo      cmd_depth_4 leave-one-out within the evaluation fold.
  ORACLE           per-call perfect foreknowledge. ANALYSIS-ONLY.

PREMISE. Forced eviction; with no memory pressure every arm scores zero. Neither
declared corpus can exhibit contention. Hidden-swap-milliseconds, not wall clock. The
per-request kv values were derived from Fresh-277's context distribution and are held
fixed across corpora by construction rather than re-derived per corpus.
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

KV_COSTS = (56.0, 100.0, 160.0, 3500.0, 5000.0)
RESTORE_FRACTION = 0.94
KEY = "cmd_depth_4"


def utility(L: float, trig: float, *, th: float, kv: float, restore: float) -> float:
    return trigger_policy_utility_ms(
        L, trig, threshold_ms=th, kv_cost_ms=kv, restore_cost_ms=restore
    )


def best_constant(
    latencies: Sequence[float], *, th: float, kv: float, restore: float
) -> float:
    """Single global trigger maximising net utility on a declared grid."""

    step = max(5.0, kv / 70.0)
    best_t = th
    best_v = sum(utility(L, th, th=th, kv=kv, restore=restore) for L in latencies)
    for i in range(int(2.0 * kv / step) + 1):
        t = i * step
        v = sum(utility(L, t, th=th, kv=kv, restore=restore) for L in latencies)
        if v > best_v:
            best_v, best_t = v, t
    return best_t


def loo_triggers(
    groups: dict[str, list[float]], *, th: float, kv: float, restore: float
) -> dict[tuple[str, int], float]:
    """Leave-one-out trigger per (key, position). Singletons fall back to th."""

    out: dict[tuple[str, int], float] = {}
    for key, vals in groups.items():
        if len(vals) == 1:
            out[(key, 0)] = th
            continue
        for i in range(len(vals)):
            out[(key, i)] = hazard_recheck_ms(
                vals[:i] + vals[i + 1:], threshold_ms=th, kv_cost_ms=kv,
                restore_cost_ms=restore,
            )
    return out


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
        for sample in samples_by_task[task_id]:
            row = sample.to_json_obj()
            row["task_id"] = task_id
            rows.append(row)

    keyer = build_keyers(manifest)[KEY]
    for row in rows:
        row["_key"] = keyer(row)

    folds = balanced_task_folds(rows, fold_count=manifest["fold_count"])
    guard = float(manifest.get("guard_ms", 0.0))

    cells: dict[str, Any] = {}
    for kv in KV_COSTS:
        th = kv + guard
        restore = RESTORE_FRACTION * kv
        dead = const = loo = oracle = 0.0
        n_long = n_band = n = 0

        for fold_tasks in folds:
            ev = [r for r in rows if r["task_id"] in fold_tasks]
            profile = [
                r["latency_ms"] for r in rows if r["task_id"] not in fold_tasks
            ]
            t_const = best_constant(profile, th=th, kv=kv, restore=restore)

            groups: dict[str, list[float]] = defaultdict(list)
            for r in ev:
                groups[r["_key"]].append(r["latency_ms"])
            trig = loo_triggers(groups, th=th, kv=kv, restore=restore)

            seen: dict[str, int] = defaultdict(int)
            for r in ev:
                L = r["latency_ms"]
                key = r["_key"]
                i = seen[key]
                seen[key] += 1
                n += 1
                n_long += L > th
                n_band += th < L < 2.0 * kv
                dead += utility(L, th, th=th, kv=kv, restore=restore)
                const += utility(L, t_const, th=th, kv=kv, restore=restore)
                loo += utility(L, trig[(key, i)], th=th, kv=kv, restore=restore)
                oracle += 0.0 if L <= th else utility(
                    L, max(0.0, L - kv), th=th, kv=kv, restore=restore
                )

        budget = oracle - dead
        cells[str(int(kv))] = {
            "kv_ms": kv,
            "rows": n,
            "long_frac": n_long / n,
            "band_frac": n_band / n,
            "deadline_s": dead / 1000.0,
            "best_constant_s": const / 1000.0,
            "per_key_loo_s": loo / 1000.0,
            "oracle_s": oracle / 1000.0,
            "budget_s": budget / 1000.0,
            "budget_frac_of_deadline": budget / dead if dead else None,
            "const_gain_s": (const - dead) / 1000.0,
            "loo_gain_s": (loo - dead) / 1000.0,
            "const_beats": const > dead,
            "loo_beats": loo > dead,
        }

    payload = {
        "criterion_frozen_in": "1b8a57d",
        "premise": "forced eviction; hidden-swap-ms not wall clock; neither corpus "
                   "can exhibit contention",
        "corpus": str(args.manifest),
        "tasks": len(task_ids),
        "rows": len(rows),
        "cells": cells,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(cells, indent=2))


if __name__ == "__main__":
    main()
