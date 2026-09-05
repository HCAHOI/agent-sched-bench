#!/usr/bin/env python3
"""Score the frozen causal full-command outcome memories."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Hashable, Sequence

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from scripts.evaluation.evaluate_clause_latency_buckets import (  # noqa: E402
    _accuracy_delta,
    _phase_changes,
)
from scripts.evaluation.evaluate_command_history_residual import (  # noqa: E402
    BUCKETS,
    DISK,
    SPLIT_MANIFEST,
    TARGETS,
    Row,
    _fail_closed_metrics,
    _load_rows,
    command_shape,
)
from tool_resource.runtime_kb import RESOURCE_BUCKET_LABELS  # noqa: E402

VERSION = "command-outcome-memory-v1"
ALL_TARGETS = (*TARGETS, DISK)
FROZEN_FIT_ROWS = (
    _ROOT
    / "analysis/results/tool-resource-5-3-3-3-20260804"
    / "sqlglot20-80-current-fit-v1/rows.jsonl"
)
FROZEN_VALIDATION_ROWS = (
    _ROOT
    / "analysis/results/tool-resource-5-3-3-3-20260804"
    / "sqlglot50-full-test-phase-validation-v1/rows.jsonl"
)
ARMS = ("exact", "shape", "hierarchy")


def _pmf(values: Sequence[int], buckets: int) -> list[float]:
    counts = Counter(values)
    return [counts[index] / len(values) for index in range(buckets)]


def _hard(target: str, pmf: Sequence[float]) -> int | str:
    bucket = max(range(len(pmf)), key=pmf.__getitem__)
    return bucket if target == "latency" else RESOURCE_BUCKET_LABELS[bucket]


def _task_groups(rows: Sequence[Row]) -> list[list[Row]]:
    groups: list[list[Row]] = []
    for row in rows:
        if not groups or groups[-1][0].task_id != row.task_id:
            groups.append([])
        groups[-1].append(row)
    return groups


def _shape_key(row: Row) -> Hashable:
    return command_shape(row.command)


def run(
    fit_rows: Sequence[Row], validation_rows: Sequence[Row]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    exact: dict[str, dict[str, list[int]]] = {
        target: defaultdict(list) for target in ALL_TARGETS
    }
    shape: dict[str, dict[Hashable, list[int]]] = {
        target: defaultdict(list) for target in ALL_TARGETS
    }

    def absorb(rows: Sequence[Row]) -> None:
        for row in rows:
            key = _shape_key(row)
            for target in ALL_TARGETS:
                label = row.labels[target]
                if label is not None:
                    exact[target][row.command].append(label)
                    shape[target][key].append(label)

    absorb(fit_rows)
    started = time.perf_counter()
    sidecar: list[dict[str, Any]] = []
    arm_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for task_rows in _task_groups(validation_rows):
        for row in task_rows:
            key = _shape_key(row)
            candidates = {arm: dict(row.current) for arm in ARMS}
            candidate_pmfs = {
                arm: {
                    target: None if pmf is None else list(pmf)
                    for target, pmf in row.pmfs.items()
                }
                for arm in ARMS
            }
            provenance: dict[str, dict[str, dict[str, Any]]] = {
                arm: {} for arm in ARMS
            }
            for target in ALL_TARGETS:
                exact_values = exact[target].get(row.command, [])
                shape_values = shape[target].get(key, [])
                selected = {
                    "exact": ("exact", exact_values),
                    "shape": ("shape", shape_values),
                    "hierarchy": (
                        ("exact", exact_values)
                        if exact_values
                        else ("shape", shape_values)
                    ),
                }
                for arm, (source, values) in selected.items():
                    if not values:
                        provenance[arm][target] = {
                            "source": "current",
                            "support": 0,
                        }
                        continue
                    pmf = _pmf(values, BUCKETS[target])
                    candidates[arm][target] = _hard(target, pmf)
                    candidate_pmfs[arm][target] = pmf
                    provenance[arm][target] = {
                        "source": source,
                        "support": len(values),
                    }
            base = {
                "sample_id": row.sample_id,
                "task_id": row.task_id,
                "command": row.command,
                "labels": dict(row.labels),
                "current_dynamic": dict(row.current),
                "current_probability_by_bucket": {
                    target: None if pmf is None else list(pmf)
                    for target, pmf in row.pmfs.items()
                },
            }
            sidecar.append(
                {
                    **base,
                    "arms": {
                        arm: {
                            "candidate": candidates[arm],
                            "candidate_probability_by_bucket": candidate_pmfs[arm],
                            "provenance": provenance[arm],
                        }
                        for arm in ARMS
                    },
                }
            )
            for arm in ARMS:
                arm_rows[arm].append(
                    {
                        **base,
                        "candidate": candidates[arm],
                        "candidate_probability_by_bucket": candidate_pmfs[arm],
                    }
                )
        absorb(task_rows)
    elapsed = time.perf_counter() - started

    current_rows = [
        {
            **row,
            "candidate": row["current_dynamic"],
            "candidate_probability_by_bucket": row[
                "current_probability_by_bucket"
            ],
        }
        for row in arm_rows["hierarchy"]
    ]
    current = {
        target: _fail_closed_metrics(current_rows, target)
        for target in ALL_TARGETS
    }
    metrics: dict[str, Any] = {}
    for arm in ARMS:
        arm_targets: dict[str, Any] = {}
        for target in ALL_TARGETS:
            candidate = _fail_closed_metrics(arm_rows[arm], target)
            changes = _phase_changes(arm_rows[arm], target)
            accuracy_key = (
                "exact_class_accuracy" if target == "latency" else "accuracy"
            )
            arm_targets[target] = {
                "current": current[target],
                "candidate": candidate,
                "delta_percentage_points": _accuracy_delta(
                    candidate[accuracy_key], current[target][accuracy_key]
                ),
                "changes": changes,
            }
        metrics[arm] = arm_targets

    primary = metrics["hierarchy"]
    gain_ok = all(
        primary[target]["delta_percentage_points"] >= 5.0
        for target in ALL_TARGETS
    )
    severe_ok = all(
        primary[target]["candidate"]["severe_underprediction_rate"]
        <= primary[target]["current"]["severe_underprediction_rate"]
        for target in ALL_TARGETS
    )
    helpful = sum(primary[target]["changes"]["helpful"] for target in ALL_TARGETS)
    harmful = sum(primary[target]["changes"]["harmful"] for target in ALL_TARGETS)
    helpful_tasks = {
        task_id
        for target in ALL_TARGETS
        for task_id in primary[target]["changes"]["helpful_task_ids"]
    }
    identity_ok = all(
        [row["sample_id"] for row in arm_rows[arm]]
        == [row["sample_id"] for row in current_rows]
        and [row["labels"] for row in arm_rows[arm]]
        == [row["labels"] for row in current_rows]
        and [row["current_probability_by_bucket"] for row in arm_rows[arm]]
        == [row["current_probability_by_bucket"] for row in current_rows]
        for arm in ARMS
    )
    go = (
        gain_ok
        and severe_ok
        and helpful > harmful
        and len(helpful_tasks) >= 10
        and identity_ok
    )
    result = {
        "schema": VERSION,
        "status": "development_validation_go" if go else "development_validation_no_go",
        "claim_bearing": False,
        "protocol": {
            "initial_evidence": "80 committed fit tasks",
            "update": "whole-task settlement",
            "primary_hierarchy": ["exact_command", "command_shape", "current"],
            "support_threshold": 1,
            "tie_break": "lower_bucket",
            "fit_updates": False,
        },
        "arms": metrics,
        "gate": {
            "go": go,
            "minimum_gain_percentage_points_each": 5.0,
            "all_targets_meet_gain": gain_ok,
            "no_severe_underprediction_regression": severe_ok,
            "helpful": helpful,
            "harmful": harmful,
            "helpful_tasks": len(helpful_tasks),
            "minimum_helpful_tasks": 10,
            "row_identity": identity_ok,
        },
        "cost": {
            "scoring_and_causal_updates_seconds": elapsed,
            "validation_rows": len(validation_rows),
        },
    }
    return result, sidecar


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-rows", type=Path, required=True)
    parser.add_argument("--validation-rows", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("output directory already exists")
    if (
        args.fit_rows.resolve() != FROZEN_FIT_ROWS.resolve()
        or args.validation_rows.resolve() != FROZEN_VALIDATION_ROWS.resolve()
    ):
        raise ValueError("row paths differ from the frozen development protocol")
    fit_rows = _load_rows(args.fit_rows)
    validation_rows = _load_rows(args.validation_rows)
    fit_tasks = list(dict.fromkeys(row.task_id for row in fit_rows))
    validation_tasks = list(dict.fromkeys(row.task_id for row in validation_rows))
    split = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    if (
        len(fit_rows) != 1420
        or len(fit_tasks) != 80
        or len(validation_rows) != 1044
        or len(validation_tasks) != 50
        or fit_tasks != split.get("development", [])[20:]
        or validation_tasks != split.get("validation")
    ):
        raise ValueError("rows differ from the frozen 80-fit/50-validation protocol")
    result, rows = run(fit_rows, validation_rows)
    result["inputs"] = {
        "fit_rows": str(args.fit_rows.resolve()),
        "validation_rows": str(args.validation_rows.resolve()),
        "git_sha": _git_sha(),
    }
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
