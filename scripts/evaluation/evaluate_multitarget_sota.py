#!/usr/bin/env python3
"""Compose the best exposed pre-command head for each resource target."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from scripts.evaluation.evaluate_clause_latency_buckets import (  # noqa: E402
    _accuracy_delta,
    _phase_changes,
)
from scripts.evaluation.evaluate_command_history_residual import (  # noqa: E402
    _fail_closed_metrics,
)
from scripts.evaluation.evaluate_command_outcome_memory import (  # noqa: E402
    ALL_TARGETS,
)
from tool_resource.runtime_kb import RESOURCE_BUCKET_LABELS  # noqa: E402

VERSION = "multitarget-sota-v1"
FROZEN_ROW_COUNT = 1044
PHASE_TARGETS = ("latency", "peak_cpu_cores", "sampled_peak_rss_mb")
DISK = "disk_read_write_bytes_total"
RESULTS = _ROOT / "analysis/results/tool-resource-5-3-3-3-20260804"
FROZEN_SEMANTIC_ROWS = RESULTS / "sqlglot50-semantic-work-units-v1/rows.jsonl"
FROZEN_PHASE_ROWS = RESULTS / "sqlglot50-full-test-phase-validation-v1/rows.jsonl"
FROZEN_EXACT_ROWS = RESULTS / "sqlglot50-command-outcome-memory-v1/rows.jsonl"


def _index(target: str, value: Any) -> int | None:
    if value is None:
        return None
    return int(value) if target == "latency" else RESOURCE_BUCKET_LABELS.index(str(value))


def _compose(
    semantic_row: Mapping[str, Any],
    phase_row: Mapping[str, Any],
    exact_row: Mapping[str, Any],
) -> dict[str, Any]:
    for key in (
        "sample_id",
        "task_id",
        "command",
        "labels",
        "current_dynamic",
        "current_probability_by_bucket",
    ):
        if semantic_row[key] != phase_row[key] or semantic_row[key] != exact_row[key]:
            raise ValueError(f"component rows disagree on {key}")

    candidate = deepcopy(semantic_row["arms"]["semantic_work_units"])
    applied: list[str] = []
    for target in PHASE_TARGETS:
        if target not in phase_row["phase_applied_targets"]:
            continue
        base_index = _index(target, candidate["candidate"][target])
        phase_index = _index(target, phase_row["candidate"][target])
        if base_index is None or phase_index is None or phase_index <= base_index:
            continue
        candidate["candidate"][target] = phase_row["candidate"][target]
        candidate["candidate_probability_by_bucket"][target] = list(
            phase_row["candidate_probability_by_bucket"][target]
        )
        candidate["provenance"][target] = {
            "source": "full_test_phase_monotone",
            "base_source": candidate["provenance"][target]["source"],
            "full_test_phase": phase_row["full_test_phase"],
        }
        applied.append(target)

    exact = exact_row["arms"]["exact"]
    candidate["candidate"][DISK] = exact["candidate"][DISK]
    candidate["candidate_probability_by_bucket"][DISK] = deepcopy(
        exact["candidate_probability_by_bucket"][DISK]
    )
    candidate["provenance"][DISK] = deepcopy(exact["provenance"][DISK])
    return {
        "sample_id": semantic_row["sample_id"],
        "task_id": semantic_row["task_id"],
        "command": semantic_row["command"],
        "labels": deepcopy(semantic_row["labels"]),
        "current_dynamic": deepcopy(semantic_row["current_dynamic"]),
        "current_probability_by_bucket": deepcopy(
            semantic_row["current_probability_by_bucket"]
        ),
        "candidate": candidate["candidate"],
        "candidate_probability_by_bucket": candidate[
            "candidate_probability_by_bucket"
        ],
        "provenance": candidate["provenance"],
        "phase_applied_targets": applied,
    }


def run(
    semantic_rows: Sequence[Mapping[str, Any]],
    phase_rows: Sequence[Mapping[str, Any]],
    exact_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len({len(semantic_rows), len(phase_rows), len(exact_rows)}) != 1:
        raise ValueError("component row counts differ")
    sample_ids = [row["sample_id"] for row in semantic_rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("component rows contain duplicate sample IDs")
    if len(sample_ids) != FROZEN_ROW_COUNT:
        raise ValueError(
            f"component rows differ from frozen {FROZEN_ROW_COUNT}-row population"
        )
    rows = [
        _compose(semantic, phase, exact)
        for semantic, phase, exact in zip(
            semantic_rows, phase_rows, exact_rows, strict=True
        )
    ]
    current_rows = [
        {
            **row,
            "candidate": row["current_dynamic"],
            "candidate_probability_by_bucket": row[
                "current_probability_by_bucket"
            ],
        }
        for row in rows
    ]
    metrics: dict[str, Any] = {}
    current_accuracies: list[float] = []
    candidate_accuracies: list[float] = []
    for target in ALL_TARGETS:
        current = _fail_closed_metrics(current_rows, target)
        candidate = _fail_closed_metrics(rows, target)
        accuracy_key = "exact_class_accuracy" if target == "latency" else "accuracy"
        current_accuracy = current[accuracy_key]
        candidate_accuracy = candidate[accuracy_key]
        current_accuracies.append(current_accuracy)
        candidate_accuracies.append(candidate_accuracy)
        metrics[target] = {
            "current": current,
            "candidate": candidate,
            "delta_percentage_points": _accuracy_delta(
                candidate_accuracy, current_accuracy
            ),
            "changes": _phase_changes(rows, target),
        }

    gains_pass = all(
        target["delta_percentage_points"] >= 5.0 for target in metrics.values()
    )
    return {
        "schema": VERSION,
        "status": "development_sota_selected_posthoc",
        "claim_bearing": False,
        "selection": "best_valid_begin_call_head_per_target_on_exposed_validation",
        "protocol": {
            "decision_time": "BeginCall",
            "cross_task_update": "whole_task_settlement",
            "current_task_outcomes_visible": False,
            "same_task_signal": "count_of_prior_completed_full_test_commands",
            "latency": "max(semantic_work_units, full_test_phase_monotone)",
            "peak_cpu_cores": "max(semantic_work_units, full_test_phase_monotone)",
            "sampled_peak_rss_mb": "max(semantic_work_units, full_test_phase_monotone)",
            DISK: "exact_complete_command",
        },
        "coverage": {
            "rows": len(rows),
            "phase_raised": {
                target: sum(target in row["phase_applied_targets"] for row in rows)
                for target in PHASE_TARGETS
            },
        },
        "metrics": metrics,
        "macro_equal_target_accuracy": {
            "current": sum(current_accuracies) / len(current_accuracies),
            "candidate": sum(candidate_accuracies) / len(candidate_accuracies),
            "delta_percentage_points": _accuracy_delta(
                sum(candidate_accuracies) / len(candidate_accuracies),
                sum(current_accuracies) / len(current_accuracies),
            ),
        },
        "research_gate": {
            "go": gains_pass,
            "minimum_gain_percentage_points_each": 5.0,
            "all_targets_meet_gain": gains_pass,
            "final_partition_authorized": False,
        },
        "integrity": {
            "development_exposed_selection": True,
            "common_decision_time": True,
            "identical_rows_labels_and_current": True,
            "early_execution_signals_used": False,
        },
    }, rows


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("output directory already exists")
    result, rows = run(
        _load(FROZEN_SEMANTIC_ROWS),
        _load(FROZEN_PHASE_ROWS),
        _load(FROZEN_EXACT_ROWS),
    )
    result["inputs"] = {
        "semantic_rows": str(FROZEN_SEMANTIC_ROWS),
        "phase_rows": str(FROZEN_PHASE_ROWS),
        "exact_rows": str(FROZEN_EXACT_ROWS),
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
