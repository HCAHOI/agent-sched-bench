#!/usr/bin/env python3
"""Can ANY per-key empirical method reach the prediction budget?

Criterion frozen in analysis/CLAIMS.md, "Open question -- can any per-key empirical
method reach the prediction budget?", commit ecf19dd, before this existed.

Three trigger rules on identical rows, all scored by the one shared functional
tool_time.policy.trigger_policy_utility_ms:

  deadline_only         fire at the threshold. No prediction.
  profile_trigger       hazard_recheck_ms fitted on PROFILE-fold samples of the
                        row's command key. This is the shipped mean_hazard rule.
  eval_optimal_trigger  hazard_recheck_ms fitted IN-SAMPLE on the evaluation rows of
                        the same key. A per-key ORACLE: it is the best any per-key
                        method could do with perfect knowledge of the key's
                        distribution. ANALYSIS-ONLY, never deployable.
  ORACLE                per-call perfect foreknowledge, from 57cbbbb.

The gap splits as
    ORACLE - eval_optimal   = within-key variance, unreachable by ANY per-key method
    eval_optimal - profile  = estimation error, in principle recoverable

PREMISE. The functional is conditional on forced eviction; with no memory pressure
every arm including both oracles scores zero. Fresh-277 cannot exhibit contention.
Numbers are hidden-swap-milliseconds under an assumed regime, not wall clock.

Causal contract: profile_trigger reads profile-fold samples only, under the corpus's
five task-grouped folds; task disjointness is structural because folds partition
task ids. Both oracles are labelled and reported separately.
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

from tool_time.command import make_row_command_prefix_keys  # noqa: E402
from tool_time.offline_evaluation import balanced_task_folds  # noqa: E402
from tool_time.policy import trigger_policy_utility_ms  # noqa: E402
from tool_time.prior import hazard_recheck_ms  # noqa: E402
from trace_collect.tool_latency_dataset import load_tool_latency_corpus  # noqa: E402

KV_COSTS = (3500.0, 5000.0)
RESTORE_FRACTION = 0.94


def make_deepest_key(manifest: dict[str, Any]):
    """Most specific command-prefix key, using the shipped key factory.

    Rows without a parseable command yield no prefix keys; those fall back to
    tool-level grouping rather than being lumped into one bucket, which is what
    the shipped hierarchy does for them.
    """

    row_keys = make_row_command_prefix_keys(
        manifest["command_field"],
        max_depth=manifest["max_prefix_depth"],
        skip_leading_cd=manifest["skip_leading_cd"],
    )

    def deepest(row: dict[str, Any]) -> str:
        keys = row_keys(row)
        return keys[-1] if keys else f"tool:{row.get('tool_name')}"

    return deepest


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
    deepest = make_deepest_key(manifest)
    for r in rows:
        r["_key"] = deepest(r)

    folds = balanced_task_folds(rows, fold_count=manifest["fold_count"])
    guard = float(manifest.get("guard_ms", 0.0))

    cells: dict[str, Any] = {}
    for kv in KV_COSTS:
        th = kv + guard
        restore = RESTORE_FRACTION * kv
        tot: dict[str, float] = defaultdict(float)
        key_cov = {"profile_support": 0, "no_profile_support": 0}
        for fold_tasks in folds:
            ev = [r for r in rows if r["task_id"] in fold_tasks]
            pr = [r for r in rows if r["task_id"] not in fold_tasks]
            prof_by_key: dict[str, list[float]] = defaultdict(list)
            for r in pr:
                prof_by_key[r["_key"]].append(r["latency_ms"])
            eval_by_key: dict[str, list[float]] = defaultdict(list)
            for r in ev:
                eval_by_key[r["_key"]].append(r["latency_ms"])

            trig_p = {
                k: hazard_recheck_ms(v, threshold_ms=th, kv_cost_ms=kv,
                                     restore_cost_ms=restore)
                for k, v in prof_by_key.items()
            }
            trig_e = {
                k: hazard_recheck_ms(v, threshold_ms=th, kv_cost_ms=kv,
                                     restore_cost_ms=restore)
                for k, v in eval_by_key.items()
            }
            for r in ev:
                L = r["latency_ms"]
                k = r["_key"]
                if k in trig_p:
                    key_cov["profile_support"] += 1
                else:
                    key_cov["no_profile_support"] += 1
                u = lambda t: trigger_policy_utility_ms(  # noqa: E731
                    L, t, threshold_ms=th, kv_cost_ms=kv, restore_cost_ms=restore
                )
                tot["deadline_only"] += u(th)
                tot["profile_trigger"] += u(trig_p.get(k, th))
                tot["eval_optimal_trigger"] += u(trig_e.get(k, th))
                tot["ORACLE"] += 0.0 if L <= th else u(max(0.0, L - kv))

        budget = tot["ORACLE"] - tot["deadline_only"]
        reach = tot["eval_optimal_trigger"] - tot["deadline_only"]
        cells[str(int(kv))] = {
            "kv_cost_ms": kv,
            "net_saved_s": {k: v / 1000.0 for k, v in sorted(tot.items())},
            "budget_s": budget / 1000.0,
            "per_key_ceiling_s": reach / 1000.0,
            "per_key_ceiling_fraction_of_budget": reach / budget if budget else None,
            "unreachable_within_key_variance_s": (budget - reach) / 1000.0,
            "estimation_error_s": (
                tot["eval_optimal_trigger"] - tot["profile_trigger"]
            ) / 1000.0,
            "key_coverage": key_cov,
        }

    payload = {
        "criterion_frozen_in": "ecf19dd",
        "premise": "forced eviction; hidden-swap-ms not wall clock; Fresh-277 cannot "
                   "exhibit contention",
        "oracles_are_analysis_only": True,
        "tasks": len(task_ids),
        "cells": cells,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(cells, indent=2))


if __name__ == "__main__":
    main()
