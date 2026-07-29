#!/usr/bin/env python3
"""Does runtime CPU activity at a landmark beat survival alone on high-variance calls?

Criterion frozen in analysis/CLAIMS.md, section "Open question -- does runtime CPU
activity beat survival alone on high-variance calls?", commit b23b9d7, before this
existed.

Three arms on identical rows, scored with tool_time.policy.trigger_policy_utility_ms:

  deadline_only    fire at the threshold; structurally misses nothing
  elapsed_only@1s  fire at the landmark for every survivor; no feature
  cpu_state@1s     fire at the landmark only when a one-feature stump says long

Causal contract: only resource_timeline windows that have COMPLETELY ended before the
landmark are read (offset_s + dt_s <= landmark). The stump threshold is fitted on
profile tasks only, under five task-grouped folds, and never on the scored rows.
The realised duration is a label, never a feature.
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
from tool_time.policy import trigger_policy_utility_ms  # noqa: E402

LANDMARK_MS = 1000.0
KV_COSTS = (3500.0, 5000.0)
RESTORE_FRACTION = 0.94
ACCEPT_BAR = 0.94 / 1.94  # rho/(1+rho); frozen, derived
FOLDS = 5
DRAWS = 10000
SEED = 0


def load_rows(trace: Path, landmark_ms: float) -> list[dict[str, Any]]:
    """One row per exec call that is still running at the landmark."""

    rows: list[dict[str, Any]] = []
    with trace.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("action_type") != "tool_exec":
                continue
            data = record.get("data") or {}
            duration = data.get("duration_ms")
            timeline = data.get("resource_timeline")
            task_id = data.get("task_instance_id")
            if duration is None or not isinstance(timeline, dict) or not task_id:
                continue
            if duration <= landmark_ms:
                continue  # not alive at the landmark; nothing to decide
            # `offset_s` is the window's END (samples run 0.501, 1.002, 1.504 with
            # dt_s = 0.501 each), so a window is complete iff offset_s <= landmark.
            # At a 1000 ms landmark that admits exactly one ~501 ms window; the
            # frozen landmark and population are unchanged, only the prose
            # "two complete windows" was wrong about the sampler's phase.
            cpu_core_s = 0.0
            covered_s = 0.0
            for sample in timeline.get("samples") or ():
                end_s = sample["offset_s"]
                if end_s * 1000.0 > landmark_ms:
                    break  # window not complete before the decision
                cpu_core_s += sample["cpu_core_s"]
                covered_s = end_s
            if covered_s <= 0.0:
                continue  # no complete window: the feature is unavailable
            rows.append(
                {
                    "task_id": str(task_id),
                    "latency_ms": float(duration),
                    "feature": cpu_core_s / covered_s,  # mean active cores so far
                    "covered_s": covered_s,
                }
            )
    return rows


def fit_stump(profile: list[dict[str, Any]], *, kv: float) -> float:
    """Single threshold on the feature maximising net utility on profile rows.

    Fire when feature >= threshold. Candidates are the observed feature values plus
    +inf, which is the never-fire option.
    """

    threshold_ms = kv
    restore = RESTORE_FRACTION * kv
    candidates = sorted({row["feature"] for row in profile})
    candidates.append(float("inf"))
    best_cut, best_value = float("inf"), 0.0
    for cut in candidates:
        value = 0.0
        for row in profile:
            if row["feature"] >= cut:
                value += trigger_policy_utility_ms(
                    row["latency_ms"], LANDMARK_MS, threshold_ms=threshold_ms,
                    kv_cost_ms=kv, restore_cost_ms=restore,
                )
        if value > best_value:
            best_value, best_cut = value, cut
    return best_cut


def evaluate(rows: list[dict[str, Any]], *, kv: float) -> dict[str, Any]:
    threshold_ms = kv
    restore = RESTORE_FRACTION * kv
    folds = balanced_task_folds(rows, fold_count=FOLDS)

    totals: dict[str, float] = defaultdict(float)
    by_task: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    fired = fired_long = 0
    survivors_long = 0

    for fold_tasks in folds:
        profile = [r for r in rows if r["task_id"] not in fold_tasks]
        held = [r for r in rows if r["task_id"] in fold_tasks]
        if not profile or not held:
            raise ValueError("empty fold side")
        cut = fit_stump(profile, kv=kv)
        for row in held:
            is_long = row["latency_ms"] > threshold_ms
            survivors_long += is_long
            u_dead = trigger_policy_utility_ms(
                row["latency_ms"], threshold_ms, threshold_ms=threshold_ms,
                kv_cost_ms=kv, restore_cost_ms=restore,
            )
            u_elapsed = trigger_policy_utility_ms(
                row["latency_ms"], LANDMARK_MS, threshold_ms=threshold_ms,
                kv_cost_ms=kv, restore_cost_ms=restore,
            )
            if row["feature"] >= cut:
                u_cpu = u_elapsed
                fired += 1
                fired_long += is_long
            else:
                u_cpu = u_dead  # decline at the landmark, fall back to the deadline
            totals["deadline_only"] += u_dead
            totals["elapsed_only@1s"] += u_elapsed
            totals["cpu_state@1s"] += u_cpu
            t = by_task[row["task_id"]]
            t["cpu_minus_dead"] += u_cpu - u_dead
            t["cpu_minus_elapsed"] += u_cpu - u_elapsed

    def boot(field: str) -> dict[str, float]:
        tasks = sorted(by_task)
        vals = [by_task[t][field] for t in tasks]
        rng = random.Random(SEED)
        n = len(tasks)
        draws = sorted(
            sum(vals[rng.randrange(n)] for _ in range(n)) / 1000.0
            for _ in range(DRAWS)
        )
        return {
            "point_s": sum(vals) / 1000.0,
            "ci_low_s": draws[int(0.025 * DRAWS)],
            "ci_high_s": draws[int(0.975 * DRAWS)],
            "fraction_positive": sum(1 for v in draws if v > 0) / DRAWS,
        }

    return {
        "kv_cost_ms": kv,
        "survivors": len(rows),
        "survivor_p_long": survivors_long / len(rows),
        "acceptance_bar": ACCEPT_BAR,
        "net_saved_s": {k: v / 1000.0 for k, v in sorted(totals.items())},
        "fired": fired,
        "fired_long": fired_long,
        "fired_subgroup_p_long": (fired_long / fired) if fired else None,
        "fired_subgroup_clears_bar": bool(fired and fired_long / fired > ACCEPT_BAR),
        "margin_vs_deadline": boot("cpu_minus_dead"),
        "margin_vs_elapsed": boot("cpu_minus_elapsed"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.trace, LANDMARK_MS)
    payload = {
        "criterion_frozen_in": "b23b9d7",
        "trace": str(args.trace),
        "landmark_ms": LANDMARK_MS,
        "restore_cost_fraction": RESTORE_FRACTION,
        "folds": FOLDS,
        "rows_scored": len(rows),
        "cells": {str(int(kv)): evaluate(rows, kv=kv) for kv in KV_COSTS},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["cells"], indent=2))


if __name__ == "__main__":
    main()
