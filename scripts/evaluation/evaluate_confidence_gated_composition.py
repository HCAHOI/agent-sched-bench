#!/usr/bin/env python3
"""Cross-fit scalar confidence gates over the frozen component composition."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from scripts.evaluation.evaluate_command_history_residual import DISK  # noqa: E402
from scripts.evaluation.evaluate_command_outcome_memory import (  # noqa: E402
    ALL_TARGETS,
)
from scripts.evaluation.evaluate_joint_prediction_state import (  # noqa: E402
    _class_id,
    _hard,
    _labels,
    _load,
    _metrics,
    _sha256,
    _validate_pmf,
)

VERSION = "confidence-gated-composition-v1"
FOLDS = 5
SELECTED_TARGETS = ("peak_cpu_cores", DISK)
UNCHANGED_TARGETS = ("latency", "sampled_peak_rss_mb")
BASE_ROWS = (
    _ROOT
    / "analysis/results/tool-resource-5-3-3-3-20260804"
    / "sqlglot50-component-composition-v1/rows.jsonl"
)
BASE_SHA256 = "f32f0fceabaf2983f67a85ef806cb9476733489bbaf73987a398fdff408e9dd6"


def _predictions(
    hard_value: Any, pmfs_value: Any
) -> tuple[dict[str, Any], dict[str, list[float] | None]]:
    if not isinstance(hard_value, Mapping) or not isinstance(pmfs_value, Mapping):
        raise ValueError("row lacks predictions or PMFs")
    hard = {target: hard_value.get(target) for target in ALL_TARGETS}
    pmfs = {
        target: (
            None
            if pmfs_value.get(target) is None
            else _validate_pmf(target, pmfs_value.get(target))
        )
        for target in ALL_TARGETS
    }
    for target in ALL_TARGETS:
        if pmfs[target] is None:
            if hard[target] is not None:
                raise ValueError(f"{target} hard prediction lacks PMF")
        elif hard[target] is None or _class_id(target, hard[target]) != _class_id(
            target, _hard(target, pmfs[target])
        ):
            raise ValueError(f"invalid {target} prediction")
    return hard, pmfs


def _current(
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, list[float] | None]]:
    hard = row.get("current_dynamic")
    nested = hard.get("probability_by_bucket") if isinstance(hard, Mapping) else None
    return _predictions(hard, nested or row.get("current_probability_by_bucket"))


def _composition(
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, list[float] | None]]:
    arm = row.get("arms", {}).get("composition")
    if not isinstance(arm, Mapping):
        raise ValueError("row lacks frozen composition")
    return _predictions(
        arm.get("candidate"), arm.get("candidate_probability_by_bucket")
    )


def _feature(row: Mapping[str, Any], target: str) -> float | None:
    _current_hard, current_pmfs = _current(row)
    _candidate, candidate_pmfs = _composition(row)
    if current_pmfs[target] is None or candidate_pmfs[target] is None:
        return None
    return max(candidate_pmfs[target]) - max(current_pmfs[target])


def _fit_threshold(rows: Sequence[Mapping[str, Any]], target: str) -> float:
    examples: list[tuple[float, bool, bool]] = []
    for row in rows:
        current, _current_pmfs = _current(row)
        candidate, _candidate_pmfs = _composition(row)
        label = _labels(row)[target]
        if label is None or candidate[target] == current[target]:
            continue
        truth = label
        feature = _feature(row, target)
        if feature is None:
            raise ValueError(f"changed {target} prediction lacks PMFs")
        examples.append(
            (
                feature,
                _class_id(target, current[target]) == truth,
                _class_id(target, candidate[target]) == truth,
            )
        )
    if not examples:
        raise ValueError(f"no labelled {target} disagreements")
    thresholds = [math.inf, *sorted({feature for feature, _, _ in examples})]

    def correct(threshold: float) -> int:
        return sum(
            candidate_correct if feature >= threshold else current_correct
            for feature, current_correct, candidate_correct in examples
        )

    return max(thresholds, key=lambda threshold: (correct(threshold), threshold))


def _apply(
    row: Mapping[str, Any], fold: int, thresholds: Mapping[str, float]
) -> dict[str, Any]:
    current, current_pmfs = _current(row)
    candidate, candidate_pmfs = _composition(row)
    output_hard = dict(candidate)
    output_pmfs = deepcopy(candidate_pmfs)
    selector = {}
    for target in SELECTED_TARGETS:
        feature = _feature(row, target)
        changed = candidate[target] != current[target]
        if changed and feature is None:
            raise ValueError(f"changed {target} prediction lacks PMFs")
        accepted = changed and feature >= thresholds[target]
        if changed and not accepted:
            output_hard[target] = current[target]
            output_pmfs[target] = list(current_pmfs[target])
        selector[target] = {
            "changed": changed,
            "feature": feature,
            "threshold": (
                "accept_none" if math.isinf(thresholds[target]) else thresholds[target]
            ),
            "accepted": accepted,
        }
    return {
        "sample_id": row["sample_id"],
        "task_id": row["task_id"],
        "command": row["command"],
        "labels": _labels(row),
        "current_dynamic": current,
        "current_probability_by_bucket": deepcopy(current_pmfs),
        "base": candidate,
        "base_probability_by_bucket": candidate_pmfs,
        "candidate": output_hard,
        "candidate_probability_by_bucket": output_pmfs,
        "fold": fold,
        "selector": selector,
    }


def _row_identity(
    source_rows: Sequence[Mapping[str, Any]],
    output_rows: Sequence[Mapping[str, Any]],
) -> bool:
    if len(source_rows) != len(output_rows):
        return False
    seen: set[Any] = set()
    for source, output in zip(source_rows, output_rows, strict=True):
        sample_id = source.get("sample_id")
        if sample_id in seen:
            return False
        seen.add(sample_id)
        current, pmfs = _current(source)
        if (
            output.get("sample_id") != sample_id
            or output.get("task_id") != source.get("task_id")
            or output.get("command") != source.get("command")
            or output.get("labels") != _labels(source)
            or output.get("current_dynamic") != current
            or output.get("current_probability_by_bucket") != pmfs
        ):
            return False
    return True


def evaluate(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task_order = list(dict.fromkeys(str(row["task_id"]) for row in rows))
    if len(task_order) != 50:
        raise ValueError("input differs from the frozen 50-task protocol")
    task_fold = {task_id: index % FOLDS for index, task_id in enumerate(task_order)}
    thresholds = {
        fold: {
            target: _fit_threshold(
                [row for row in rows if task_fold[str(row["task_id"])] != fold],
                target,
            )
            for target in SELECTED_TARGETS
        }
        for fold in range(FOLDS)
    }
    output = [
        _apply(
            row,
            task_fold[str(row["task_id"])],
            thresholds[task_fold[str(row["task_id"])]],
        )
        for row in rows
    ]
    identity = _row_identity(rows, output)
    unchanged = all(
        output_row["candidate"][target] == output_row["base"][target]
        and output_row["candidate_probability_by_bucket"][target]
        == output_row["base_probability_by_bucket"][target]
        for output_row in output
        for target in UNCHANGED_TARGETS
    )
    metrics = _metrics(output)
    gain = all(
        metrics[target]["delta_percentage_points"] >= 5.0 for target in ALL_TARGETS
    )
    severe = all(
        metrics[target]["candidate"]["severe_underprediction_rate"]
        <= metrics[target]["current"]["severe_underprediction_rate"]
        for target in ALL_TARGETS
    )
    helpful = all(
        metrics[target]["changes"]["helpful"] > metrics[target]["changes"]["harmful"]
        for target in ALL_TARGETS
    )
    helpful_tasks = {
        target: len(set(metrics[target]["changes"]["helpful_task_ids"]))
        for target in SELECTED_TARGETS
    }
    go = (
        gain
        and severe
        and helpful
        and all(count >= 5 for count in helpful_tasks.values())
        and unchanged
        and identity
    )
    return {
        "schema": VERSION,
        "status": "development_oof_go" if go else "development_oof_no_go",
        "claim_bearing": False,
        "protocol": {
            "folds": FOLDS,
            "task_assignment": "first_occurrence_index_modulo_5",
            "selected_targets": list(SELECTED_TARGETS),
            "unchanged_targets": list(UNCHANGED_TARGETS),
            "feature": "candidate_winning_probability_minus_current_winning_probability",
            "fit": "exact_empirical_risk_minimization_on_labelled_disagreements",
            "accept": "feature_greater_than_or_equal_to_threshold",
            "tie_break": "higher_threshold",
        },
        "fold_thresholds": {
            str(fold): {
                target: "accept_none" if math.isinf(value) else value
                for target, value in values.items()
            }
            for fold, values in thresholds.items()
        },
        "targets": metrics,
        "helpful_tasks": helpful_tasks,
        "latency_rss_bit_identical_to_composition": unchanged,
        "row_identity": identity,
        "gate": {
            "go": go,
            "minimum_gain_percentage_points_each": 5.0,
            "all_targets_meet_gain": gain,
            "no_severe_underprediction_regression": severe,
            "helpful_exceeds_harmful_each_target": helpful,
            "minimum_cpu_disk_helpful_tasks_each": 5,
            "cpu_disk_helpful_tasks_meet_minimum": all(
                count >= 5 for count in helpful_tasks.values()
            ),
            "latency_rss_bit_identical_to_composition": unchanged,
            "row_identity": identity,
        },
    }, output


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("output directory already exists")
    if args.rows.resolve() != BASE_ROWS.resolve() or _sha256(args.rows) != BASE_SHA256:
        raise ValueError("input differs from the frozen confidence-gate protocol")
    result, rows = evaluate(_load(args.rows))
    result["inputs"] = {
        "rows": str(args.rows.resolve()),
        "rows_sha256": BASE_SHA256,
        "evaluator_sha256": _sha256(Path(__file__)),
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
