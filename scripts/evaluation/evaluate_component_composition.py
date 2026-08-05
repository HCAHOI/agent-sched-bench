#!/usr/bin/env python3
"""Compose the frozen phase, semantic, elapsed, and survival predictions."""

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

VERSION = "component-composition-v1"
ARMS = ("survival_ablation", "composition")
PHYSICAL_TARGETS = ("latency", "peak_cpu_cores", "sampled_peak_rss_mb")
FROZEN_PHASE_ROWS = (
    _ROOT
    / "analysis/results/tool-resource-5-3-3-3-20260804"
    / "sqlglot50-full-test-phase-validation-v1/rows.jsonl"
)
FROZEN_SURVIVAL_ROWS = (
    _ROOT
    / "analysis/results/tool-resource-5-3-3-3-20260804"
    / "sqlglot50-survival-disk-v1/rows.jsonl"
)


def _index(target: str, value: Any) -> int | None:
    if value is None:
        return None
    return int(value) if target == "latency" else RESOURCE_BUCKET_LABELS.index(str(value))


def _compose(
    survival_row: Mapping[str, Any], phase_row: Mapping[str, Any]
) -> dict[str, Any]:
    for key in ("sample_id", "task_id", "command", "labels", "current_dynamic"):
        if survival_row[key] != phase_row[key]:
            raise ValueError(f"component rows disagree on {key}")
    if (
        survival_row["current_probability_by_bucket"]
        != phase_row["current_probability_by_bucket"]
    ):
        raise ValueError("component rows disagree on Current PMFs")

    ablation = deepcopy(survival_row["arms"]["survival_disk"])
    candidate = deepcopy(ablation)
    applied: list[str] = []
    for target in PHYSICAL_TARGETS:
        if target not in phase_row["phase_applied_targets"]:
            continue
        base_index = _index(target, candidate["candidate"][target])
        phase_index = _index(target, phase_row["candidate"][target])
        if base_index is None or phase_index is None or phase_index <= base_index:
            continue
        base_source = candidate["provenance"][target]["source"]
        candidate["candidate"][target] = phase_row["candidate"][target]
        candidate["candidate_probability_by_bucket"][target] = list(
            phase_row["candidate_probability_by_bucket"][target]
        )
        candidate["provenance"][target] = {
            "source": "full_test_phase_monotone",
            "base_source": base_source,
            "full_test_phase": phase_row["full_test_phase"],
        }
        applied.append(target)
    return {
        "sample_id": survival_row["sample_id"],
        "task_id": survival_row["task_id"],
        "command": survival_row["command"],
        "labels": dict(survival_row["labels"]),
        "current_dynamic": dict(survival_row["current_dynamic"]),
        "current_probability_by_bucket": deepcopy(
            survival_row["current_probability_by_bucket"]
        ),
        "phase_applied_targets": applied,
        "arms": {"survival_ablation": ablation, "composition": candidate},
    }


def run(
    survival_rows: Sequence[Mapping[str, Any]],
    phase_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(survival_rows) != len(phase_rows):
        raise ValueError("component row counts differ")
    rows = [
        _compose(survival, phase)
        for survival, phase in zip(survival_rows, phase_rows, strict=True)
    ]
    arm_rows = {
        arm: [
            {
                **{key: value for key, value in row.items() if key != "arms"},
                "candidate": row["arms"][arm]["candidate"],
                "candidate_probability_by_bucket": row["arms"][arm][
                    "candidate_probability_by_bucket"
                ],
            }
            for row in rows
        ]
        for arm in ARMS
    }
    current_rows = [
        {
            **{key: value for key, value in row.items() if key != "arms"},
            "candidate": row["current_dynamic"],
            "candidate_probability_by_bucket": row["current_probability_by_bucket"],
        }
        for row in rows
    ]
    current = {
        target: _fail_closed_metrics(current_rows, target) for target in ALL_TARGETS
    }
    metrics: dict[str, Any] = {}
    for arm in ARMS:
        metrics[arm] = {}
        for target in ALL_TARGETS:
            candidate = _fail_closed_metrics(arm_rows[arm], target)
            accuracy_key = "exact_class_accuracy" if target == "latency" else "accuracy"
            metrics[arm][target] = {
                "current": current[target],
                "candidate": candidate,
                "delta_percentage_points": _accuracy_delta(
                    candidate[accuracy_key], current[target][accuracy_key]
                ),
                "changes": _phase_changes(arm_rows[arm], target),
            }

    primary = metrics["composition"]
    gain_ok = all(
        primary[target]["delta_percentage_points"] >= 5.0 for target in ALL_TARGETS
    )
    severe_ok = all(
        primary[target]["candidate"]["severe_underprediction_rate"]
        <= primary[target]["current"]["severe_underprediction_rate"]
        for target in ALL_TARGETS
    )
    helpful_ok = all(
        primary[target]["changes"]["helpful"]
        > primary[target]["changes"]["harmful"]
        for target in ALL_TARGETS
    )
    helpful_tasks = {
        task_id
        for target in ALL_TARGETS
        for task_id in primary[target]["changes"]["helpful_task_ids"]
    }
    row_identity = all(
        [row["sample_id"] for row in arm_rows[arm]]
        == [row["sample_id"] for row in current_rows]
        and [row["labels"] for row in arm_rows[arm]]
        == [row["labels"] for row in current_rows]
        and [row["current_dynamic"] for row in arm_rows[arm]]
        == [row["current_dynamic"] for row in current_rows]
        and [row["current_probability_by_bucket"] for row in arm_rows[arm]]
        == [row["current_probability_by_bucket"] for row in current_rows]
        for arm in ARMS
    )
    go = gain_ok and severe_ok and helpful_ok and len(helpful_tasks) >= 10 and row_identity
    return {
        "schema": VERSION,
        "status": "development_go" if go else "development_no_go",
        "claim_bearing": False,
        "protocol": {
            "fit": None,
            "validation_updates": False,
            "latency": "max(semantic, full_test_phase, elapsed_floor)",
            "cpu_rss": "max(semantic, full_test_phase)",
            "disk": "survival_disk_v1",
        },
        "coverage": {
            "rows": len(rows),
            "phase_raised": {
                target: sum(target in row["phase_applied_targets"] for row in rows)
                for target in PHYSICAL_TARGETS
            },
        },
        "arms": metrics,
        "gate": {
            "go": go,
            "minimum_gain_percentage_points": 5.0,
            "all_targets_meet_gain": gain_ok,
            "no_severe_underprediction_regression": severe_ok,
            "helpful_exceeds_harmful_each_target": helpful_ok,
            "helpful_tasks": len(helpful_tasks),
            "minimum_helpful_tasks": 10,
            "row_identity": row_identity,
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
    parser.add_argument("--phase-rows", type=Path, required=True)
    parser.add_argument("--survival-rows", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("output directory already exists")
    if (
        args.phase_rows.resolve() != FROZEN_PHASE_ROWS.resolve()
        or args.survival_rows.resolve() != FROZEN_SURVIVAL_ROWS.resolve()
    ):
        raise ValueError("inputs differ from the frozen composition protocol")
    result, rows = run(_load(args.survival_rows), _load(args.phase_rows))
    result["inputs"] = {
        "phase_rows": str(args.phase_rows.resolve()),
        "survival_rows": str(args.survival_rows.resolve()),
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
