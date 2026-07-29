#!/usr/bin/env python3
"""Is the within-key variance irreducible, or an artifact of key choice?

Criterion frozen in analysis/CLAIMS.md, "Open question -- is the within-key variance
irreducible, or an artifact of key choice?", commit 817edc7, before this existed.

3bc4906 attributed ~70% of the prediction budget to within-key variance, but computed
it against exactly one key. This walks a ladder from coarse to fine and reports the
per-key oracle ceiling for each:

    tool_name          Continuum's granularity
    cmd_depth_1        binary only
    cmd_depth_2
    cmd_depth_4        the shipped key
    repo+cmd_depth_4   richest key available offline

Every ceiling is an in-sample per-key ORACLE: within each evaluation fold, the
utility-maximising trigger is fitted on that fold's own rows for the key. It is
ANALYSIS-ONLY, optimistic by construction, and never deployable. The comparison
BETWEEN ceilings is the object of interest, not their absolute level.

PREMISE. The functional is conditional on forced eviction; with no memory pressure
every arm including every oracle scores zero. Fresh-277 cannot exhibit contention.
Numbers are hidden-swap-milliseconds under an assumed regime, not wall clock.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from tool_time.command import make_row_command_prefix_keys  # noqa: E402
from tool_time.offline_evaluation import balanced_task_folds  # noqa: E402
from tool_time.policy import trigger_policy_utility_ms  # noqa: E402
from tool_time.prior import hazard_recheck_ms  # noqa: E402
from trace_collect.tool_latency_dataset import load_tool_latency_corpus  # noqa: E402

KV_COSTS = (3500.0, 5000.0)
RESTORE_FRACTION = 0.94
_REPO_SUFFIX = re.compile(r"-\d+$")


def build_keyers(manifest: dict[str, Any]) -> dict[str, Callable[[dict], str]]:
    """One key function per rung. Depth-d keyers reuse the shipped key factory."""

    def prefix_keyer(depth: int) -> Callable[[dict], str]:
        row_keys = make_row_command_prefix_keys(
            manifest["command_field"],
            max_depth=depth,
            skip_leading_cd=manifest["skip_leading_cd"],
        )

        def key(row: dict) -> str:
            keys = row_keys(row)
            # Rows with no parseable command carry no prefix key; they group by
            # tool, which is what the shipped hierarchy does for them.
            return keys[-1] if keys else f"tool:{row.get('tool_name')}"

        return key

    depth4 = prefix_keyer(manifest["max_prefix_depth"])

    def repo_of(task_id: str) -> str:
        return _REPO_SUFFIX.sub("", task_id)

    return {
        "tool_name": lambda r: f"tool:{r.get('tool_name')}",
        "cmd_depth_1": prefix_keyer(1),
        "cmd_depth_2": prefix_keyer(2),
        "cmd_depth_4": depth4,
        "repo+cmd_depth_4": lambda r: f"{repo_of(r['task_id'])}\x00{depth4(r)}",
    }


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
    for name, fn in keyers.items():
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
        ceilings: dict[str, float] = defaultdict(float)
        distinct: dict[str, set] = defaultdict(set)

        for fold_tasks in folds:
            ev = [r for r in rows if r["task_id"] in fold_tasks]
            trig: dict[str, dict[str, float]] = {}
            for name in keyers:
                by_key: dict[str, list[float]] = defaultdict(list)
                for r in ev:
                    by_key[r[f"_k_{name}"]].append(r["latency_ms"])
                    distinct[name].add(r[f"_k_{name}"])
                trig[name] = {
                    k: hazard_recheck_ms(v, threshold_ms=th, kv_cost_ms=kv,
                                         restore_cost_ms=restore)
                    for k, v in by_key.items()
                }
            for r in ev:
                L = r["latency_ms"]
                deadline += trigger_policy_utility_ms(
                    L, th, threshold_ms=th, kv_cost_ms=kv, restore_cost_ms=restore
                )
                oracle += 0.0 if L <= th else trigger_policy_utility_ms(
                    L, max(0.0, L - kv), threshold_ms=th, kv_cost_ms=kv,
                    restore_cost_ms=restore,
                )
                for name in keyers:
                    ceilings[name] += trigger_policy_utility_ms(
                        L, trig[name][r[f"_k_{name}"]], threshold_ms=th,
                        kv_cost_ms=kv, restore_cost_ms=restore,
                    )

        budget = oracle - deadline
        cells[str(int(kv))] = {
            "kv_cost_ms": kv,
            "deadline_only_s": deadline / 1000.0,
            "oracle_s": oracle / 1000.0,
            "budget_s": budget / 1000.0,
            "ladder": {
                name: {
                    "ceiling_s": (ceilings[name] - deadline) / 1000.0,
                    "fraction_of_budget": (ceilings[name] - deadline) / budget
                    if budget else None,
                    "distinct_keys": len(distinct[name]),
                }
                for name in keyers
            },
        }

    payload = {
        "criterion_frozen_in": "817edc7",
        "premise": "forced eviction; hidden-swap-ms not wall clock; Fresh-277 cannot "
                   "exhibit contention",
        "all_ceilings_are_in_sample_oracles": True,
        "tasks": len(task_ids),
        "cells": cells,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(cells, indent=2))


if __name__ == "__main__":
    main()
