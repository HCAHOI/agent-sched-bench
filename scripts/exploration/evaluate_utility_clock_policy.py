#!/usr/bin/env python3
"""Attribute the priced-stopping advantage over the fixed deadline by command class.

Criterion frozen in analysis/CLAIMS.md, section "Open question -- is the advantage
over the fixed deadline a container artifact?", commit 69372e0, before this ran.

The policy arms already exist in tool_time.policy: ``deadline_only`` (fixed timeout,
no prediction), ``mean_hazard`` and ``robust_clock`` (predicted), with
``absorbed_if_oracle_ms`` as the analysis-only ceiling and never-act = 0 by
construction. This driver only restores a CLI over them and adds the per-call
attribution; it reimplements no policy semantics.

Causal contract: outer folds are task-grouped and task-disjoint, profile is every
other fold, and ``evaluate_utility_clock_policy`` asserts the disjointness itself.
All arms are scored from one pooled ``decisions`` list, so row identity across arms is
structural rather than reconciled.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shlex
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tool_time.offline_evaluation import balanced_task_folds  # noqa: E402
from tool_time.policy import (  # noqa: E402
    evaluate_utility_clock_policy,
    trigger_policy_utility_ms,
)
from trace_collect.tool_latency_dataset import (  # noqa: E402
    load_tool_latency_corpus,
)

# Frozen in analysis/CLAIMS.md before any result existed. Never tune these.
_SETUP_PREFIXES = ("apt-get", "apt", "pip", "conda", "apk", "yum", "dpkg")
_TESTBED = "/testbed"
_TRIGGER_FIELD = {
    "deadline_only": "deadline_trigger_ms",
    "mean_hazard": "mean_hazard_trigger_ms",
    "robust_clock": "robust_trigger_ms",
}


_QUOTED_SPAN = re.compile(r"""'[^']*'|"[^"]*\"""", re.VERBOSE)


def _strip_quoted_spans(command: str) -> str:
    """Remove single/double quoted spans so `outside a quoted string` is literal."""

    return _QUOTED_SPAN.sub(" ", command)


_UNPARSED_COMMANDS: list[str] = []


def is_container_setup(command: str) -> bool:
    """Frozen container-setup rule. Quote-aware via shlex, so /testbed inside a
    quoted argument does not count, as the criterion states.

    Agent shell lines legitimately contain heredocs and unbalanced quotes that
    shlex rejects. Those are classified on whitespace tokens instead, and every
    such command is recorded so the count is reported rather than hidden -- a
    silent difference in tokenization would change the frozen rule's meaning.
    """

    # The /testbed clause is on the COMMAND, before any stripping, and excludes
    # quoted spans. Stripping first would exclude `cd /testbed && ...`, which is
    # the family the criterion was frozen to catch.
    if _TESTBED in _strip_quoted_spans(command):
        return True
    try:
        tokens = shlex.split(command)
    except ValueError:
        _UNPARSED_COMMANDS.append(command)
        tokens = command.split()
    if not tokens:
        return False
    # strip a leading `cd <path> &&` before matching the package-manager head
    if tokens[0] == "cd" and len(tokens) >= 3 and tokens[2] == "&&":
        tokens = tokens[3:]
    if not tokens:
        return False
    head = tokens[0]
    if head in ("apt-get", "apk", "yum", "dpkg", "conda"):
        return True
    if head == "apt":
        return True
    # frozen list says `pip install`; `pip3` is NOT frozen and is excluded.
    return head == "pip" and len(tokens) >= 2 and tokens[1] == "install"


def command_of(row: dict[str, Any], command_field: str) -> str:
    args = row.get("tool_args")
    if not isinstance(args, dict):
        return ""
    value = args.get(command_field)
    return value if isinstance(value, str) else ""


def run_folds(
    samples_by_task: dict[str, list[Any]],
    task_ids: list[str],
    manifest: dict[str, Any],
    *,
    kv_costs: list[float],
    restore_cost_fraction: float,
    exclude_container_setup: bool,
) -> tuple[list[dict[str, Any]], dict[str, str], int]:
    """Pool out-of-fold decisions over the manifest's outer folds.

    Only the manifest's declared task ids are used. The trace root holds more
    directories than the corpus declares, so iterating samples_by_task directly
    would admit undeclared tasks.
    """

    command_field = manifest["command_field"]
    declared = list(task_ids)
    rows_all: list[dict[str, Any]] = []
    for task_id in declared:
        for sample in samples_by_task[task_id]:
            row = sample.to_json_obj()
            row["task_id"] = task_id
            rows_all.append(row)

    command_by_sample = {
        row["sample_id"]: command_of(row, command_field) for row in rows_all
    }
    # Fold assignment is derived from the FULL row set and then held fixed, so the
    # ablation changes only which rows are scored -- never which fold a task is in.
    # Deriving folds after dropping rows moved 239 of 277 tasks and shifted the
    # measured quantity by more than the quantity itself.
    folds = balanced_task_folds(rows_all, fold_count=manifest["fold_count"])

    dropped = 0
    if exclude_container_setup:
        kept = []
        for row in rows_all:
            if is_container_setup(command_by_sample[row["sample_id"]]):
                dropped += 1
            else:
                kept.append(row)
        rows_all = kept
    pooled: list[dict[str, Any]] = []
    for fold_tasks in folds:
        eval_rows = [r for r in rows_all if r["task_id"] in fold_tasks]
        profile_rows = [r for r in rows_all if r["task_id"] not in fold_tasks]
        if not eval_rows or not profile_rows:
            raise ValueError(
                "fold has empty eval or profile side after ablation; pooled totals "
                "would silently omit rows"
            )
        result = evaluate_utility_clock_policy(
            eval_rows,
            profile_rows=profile_rows,
            kv_costs_ms=kv_costs,
            guard_ms=float(manifest.get("guard_ms", 0.0)),
            min_tool_history=manifest["min_tool_history"],
            min_profile_tasks=manifest["min_profile_tasks"],
            command_field=command_field,
            max_prefix_depth=manifest["max_prefix_depth"],
            skip_leading_cd=manifest["skip_leading_cd"],
            restore_cost_fraction=restore_cost_fraction,
        )
        pooled.extend(result["decisions"])
    return pooled, command_by_sample, dropped


def attribute(
    decisions: list[dict[str, Any]],
    command_by_sample: dict[str, str],
    *,
    restore_cost_fraction: float,
) -> dict[str, Any]:
    """Per-call utility per arm, and the robust-minus-deadline delta by class."""

    by_cost: dict[float, dict[str, Any]] = defaultdict(
        lambda: {
            "totals": defaultdict(float),
            "delta_setup_ms": 0.0,
            "delta_other_ms": 0.0,
            "setup_calls": 0,
            "other_calls": 0,
            "setup_long_calls": 0,
            "delta_by_task": defaultdict(float),
        }
    )
    for row in decisions:
        kv = row["kv_cost_ms"]
        cell = by_cost[kv]
        util = {}
        for policy, field in _TRIGGER_FIELD.items():
            util[policy] = trigger_policy_utility_ms(
                row["latency_ms"],
                row[field],
                threshold_ms=row["threshold_ms"],
                kv_cost_ms=kv,
                restore_cost_ms=restore_cost_fraction * kv,
            )
            cell["totals"][policy] += util[policy]
        delta = util["robust_clock"] - util["deadline_only"]
        cell["delta_by_task"][row["task_id"]] += delta
        setup = is_container_setup(command_by_sample.get(row["sample_id"], ""))
        if setup:
            cell["delta_setup_ms"] += delta
            cell["setup_calls"] += 1
            cell["setup_long_calls"] += bool(row["label_exceeds_threshold"])
        else:
            cell["delta_other_ms"] += delta
            cell["other_calls"] += 1

    out = {}
    for kv, cell in sorted(by_cost.items()):
        total_delta = cell["delta_setup_ms"] + cell["delta_other_ms"]
        out[str(int(kv))] = {
            "kv_cost_ms": kv,
            "net_saved_s_per_277": {
                p: v / 1000.0 for p, v in sorted(cell["totals"].items())
            },
            "robust_minus_deadline_s": total_delta / 1000.0,
            "delta_from_container_setup_s": cell["delta_setup_ms"] / 1000.0,
            "delta_from_other_s": cell["delta_other_ms"] / 1000.0,
            "container_setup_share_of_advantage": (
                cell["delta_setup_ms"] / total_delta if total_delta else None
            ),
            "container_setup_calls": cell["setup_calls"],
            "other_calls": cell["other_calls"],
            "container_setup_long_calls": cell["setup_long_calls"],
            "delta_by_task_ms": dict(cell["delta_by_task"]),
            "clustered_bootstrap": clustered_interval(dict(cell["delta_by_task"])),
        }
    return out


def clustered_interval(
    delta_by_task: dict[str, float], *, draws: int = 10000, seed: int = 0
) -> dict[str, float]:
    """Task-clustered bootstrap of the delta, as C1's procedure does."""

    tasks = sorted(delta_by_task)
    values = [delta_by_task[t] for t in tasks]
    n = len(tasks)
    rng = random.Random(seed)
    totals = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / 1000.0 for _ in range(draws)
    )
    return {
        "point_s": sum(values) / 1000.0,
        "ci_low_s": totals[int(0.025 * draws)],
        "ci_high_s": totals[int(0.975 * draws)],
        "fraction_positive": sum(1 for v in totals if v > 0) / draws,
        "draws": draws,
        "clusters": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--kv-costs", default="3500,5000")
    parser.add_argument("--restore-cost-fraction", type=float, default=0.94)
    parser.add_argument("--limit-tasks", type=int, default=None)
    parser.add_argument("--final", action="store_true")
    parser.add_argument(
        "--exclude-container-setup",
        action="store_true",
        help="Ablation: drop container-setup calls from BOTH profile and eval.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    kv_costs = [float(v) for v in args.kv_costs.split(",")]
    samples_by_task, task_ids, manifest = load_tool_latency_corpus(
        args.manifest, limit_tasks=args.limit_tasks, final=args.final
    )
    decisions, command_by_sample, dropped = run_folds(
        samples_by_task,
        task_ids,
        manifest,
        kv_costs=kv_costs,
        restore_cost_fraction=args.restore_cost_fraction,
        exclude_container_setup=args.exclude_container_setup,
    )
    payload = {
        "criterion_frozen_in": "69372e0",
        "manifest": str(args.manifest),
        "declared_task_count": len(task_ids),
        "expected_task_count": manifest.get("expected_task_count"),
        "restore_cost_fraction": args.restore_cost_fraction,
        "guard_ms": float(manifest.get("guard_ms", 0.0)),
        "fold_count": manifest["fold_count"],
        "exclude_container_setup": args.exclude_container_setup,
        "container_setup_calls_dropped": dropped,
        "commands_shlex_unparseable": len(set(_UNPARSED_COMMANDS)),
        "pooled_decision_rows": len(decisions),
        "cells": attribute(
            decisions,
            command_by_sample,
            restore_cost_fraction=args.restore_cost_fraction,
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["cells"], indent=2))


if __name__ == "__main__":
    main()
