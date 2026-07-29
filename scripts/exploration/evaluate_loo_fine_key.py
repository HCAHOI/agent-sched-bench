#!/usr/bin/env python3
"""Fine-key gain under leave-one-out, removing the singleton-group degeneracy.

Criterion frozen in analysis/CLAIMS.md, "Open question -- the fine-key gain under an
unconfounded estimator", commit 82287e2, before this existed.

For each evaluation row, the trigger is fitted on the key's OTHER rows within the
same evaluation fold and scored on the held-out row. A singleton group has no other
rows and falls back to the deadline, contributing exactly zero -- which is the point:
in-sample fitting on a singleton is degenerate and supplied a quarter to a third of
the fine-key ceiling in 8e7ed28.

Scope is cmd_depth_4 and repo+cmd_depth_4 only. Coarse rungs are excluded because
their groups hold thousands of rows, making LOO quadratic and numerically pointless.

PREMISE. The functional is conditional on forced eviction; with no memory pressure
every arm including the oracle scores zero. Fresh-277 cannot exhibit contention.
Numbers are hidden-swap-milliseconds under an assumed regime, not wall clock.

The LOO ceiling is still an ORACLE: it uses evaluation-fold rows of the same key.
Analysis-only, never deployable.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts" / "exploration"))

from evaluate_key_ladder import build_keyers  # noqa: E402
from tool_time.offline_evaluation import balanced_task_folds  # noqa: E402
from tool_time.policy import trigger_policy_utility_ms  # noqa: E402
from tool_time.prior import hazard_recheck_ms  # noqa: E402
from trace_collect.tool_latency_dataset import load_tool_latency_corpus  # noqa: E402

KV_COSTS = (3500.0, 5000.0)
RESTORE_FRACTION = 0.94
KEYS = ("cmd_depth_4", "repo+cmd_depth_4")


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
            r = s.to_json_obj()
            r["task_id"] = task_id
            rows.append(r)

    keyers = build_keyers(manifest)
    for name in KEYS:
        fn = keyers[name]
        for r in rows:
            r[f"_k_{name}"] = fn(r)

    folds = balanced_task_folds(rows, fold_count=manifest["fold_count"])
    guard = float(manifest.get("guard_ms", 0.0))

    cells: dict[str, Any] = {}
    for kv in KV_COSTS:
        th = kv + guard
        restore = RESTORE_FRACTION * kv
        deadline = 0.0
        oracle = 0.0
        insample: dict[str, float] = defaultdict(float)
        loo: dict[str, float] = defaultdict(float)
        singleton_rows: dict[str, int] = defaultdict(int)

        for fold_tasks in folds:
            ev = [r for r in rows if r["task_id"] in fold_tasks]
            for r in ev:
                L = r["latency_ms"]
                deadline += trigger_policy_utility_ms(
                    L, th, threshold_ms=th, kv_cost_ms=kv, restore_cost_ms=restore
                )
                oracle += 0.0 if L <= th else trigger_policy_utility_ms(
                    L, max(0.0, L - kv), threshold_ms=th, kv_cost_ms=kv,
                    restore_cost_ms=restore,
                )
            for name in KEYS:
                groups: dict[str, list[float]] = defaultdict(list)
                for r in ev:
                    groups[r[f"_k_{name}"]].append(r["latency_ms"])
                fitted = {
                    k: hazard_recheck_ms(v, threshold_ms=th, kv_cost_ms=kv,
                                         restore_cost_ms=restore)
                    for k, v in groups.items()
                }
                # LOO by index so duplicate latencies are handled correctly.
                loo_trigger: dict[tuple[str, int], float] = {}
                for k, vals in groups.items():
                    if len(vals) == 1:
                        loo_trigger[(k, 0)] = th
                        continue
                    for i in range(len(vals)):
                        others = vals[:i] + vals[i + 1:]
                        loo_trigger[(k, i)] = hazard_recheck_ms(
                            others, threshold_ms=th, kv_cost_ms=kv,
                            restore_cost_ms=restore,
                        )
                seen: dict[str, int] = defaultdict(int)
                for r in ev:
                    L = r["latency_ms"]
                    k = r[f"_k_{name}"]
                    i = seen[k]
                    seen[k] += 1
                    if len(groups[k]) == 1:
                        singleton_rows[name] += 1
                    insample[name] += trigger_policy_utility_ms(
                        L, fitted[k], threshold_ms=th, kv_cost_ms=kv,
                        restore_cost_ms=restore,
                    )
                    loo[name] += trigger_policy_utility_ms(
                        L, loo_trigger[(k, i)], threshold_ms=th, kv_cost_ms=kv,
                        restore_cost_ms=restore,
                    )

        budget = oracle - deadline
        per_key = {}
        for name in KEYS:
            per_key[name] = {
                "in_sample_ceiling_s": (insample[name] - deadline) / 1000.0,
                "in_sample_fraction": (insample[name] - deadline) / budget,
                "loo_ceiling_s": (loo[name] - deadline) / 1000.0,
                "loo_fraction": (loo[name] - deadline) / budget,
                "overfit_s": (insample[name] - loo[name]) / 1000.0,
                "singleton_rows": singleton_rows[name],
            }
        gain_pp = (
            per_key["repo+cmd_depth_4"]["loo_fraction"]
            - per_key["cmd_depth_4"]["loo_fraction"]
        ) * 100.0
        cells[str(int(kv))] = {
            "kv_cost_ms": kv,
            "deadline_only_s": deadline / 1000.0,
            "oracle_s": oracle / 1000.0,
            "budget_s": budget / 1000.0,
            "keys": per_key,
            "repo_gain_pp_loo": gain_pp,
        }

    payload = {
        "criterion_frozen_in": "82287e2",
        "premise": "forced eviction; hidden-swap-ms not wall clock; Fresh-277 cannot "
                   "exhibit contention",
        "loo_ceiling_is_still_an_oracle": True,
        "tasks": len(task_ids),
        "cells": cells,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(cells, indent=2))


if __name__ == "__main__":
    main()
