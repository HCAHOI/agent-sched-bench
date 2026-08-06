#!/usr/bin/env python3
"""Calibrate CPU and Disk from the joint Current prediction state."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
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
    BUCKETS,
    DISK,
    _fail_closed_metrics,
)
from scripts.evaluation.evaluate_command_outcome_memory import (  # noqa: E402
    ALL_TARGETS,
)
from tool_resource.runtime_kb import RESOURCE_BUCKET_LABELS  # noqa: E402

VERSION = "joint-prediction-state-v1"
ALPHA = 16.0
CALIBRATION_TASKS = 40
AUDIT_TASKS = 40
MIN_COVERAGE = 0.95
CALIBRATED_TARGETS = ("peak_cpu_cores", DISK)
STATE_TARGETS = ("latency", "peak_cpu_cores", "sampled_peak_rss_mb", DISK)
FIT_ROWS = (
    _ROOT
    / "analysis/results/tool-resource-5-3-3-3-20260804"
    / "sqlglot20-80-current-fit-v1/rows.jsonl"
)
VALIDATION_ROWS = (
    _ROOT
    / "analysis/results/tool-resource-5-3-3-3-20260804"
    / "sqlglot50-component-composition-v1/rows.jsonl"
)
FIT_SHA256 = "21543e3dc02ba00e9b81c7e2c8c632bec48630fe1db701d309053473701b11e8"
VALIDATION_SHA256 = "f32f0fceabaf2983f67a85ef806cb9476733489bbaf73987a398fdff408e9dd6"


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _class_id(target: str, value: Any) -> int | None:
    if value is None:
        return None
    if target == "latency":
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < BUCKETS[target]
        ):
            raise ValueError("invalid latency hard prediction")
        return value
    return RESOURCE_BUCKET_LABELS.index(str(value))


def _hard(target: str, pmf: Sequence[float]) -> int | str:
    index = max(range(len(pmf)), key=lambda item: (pmf[item], -item))
    return index if target == "latency" else RESOURCE_BUCKET_LABELS[index]


def _validate_pmf(target: str, value: Any) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != BUCKETS[target]
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) < 0.0
            for item in value
        )
        or not math.isclose(sum(value), 1.0, abs_tol=1e-9)
    ):
        raise ValueError(f"invalid {target} PMF")
    return [float(item) for item in value]


def _state(row: Mapping[str, Any]) -> tuple[Any, ...] | None:
    current = row.get("current_dynamic")
    if not isinstance(current, Mapping):
        raise ValueError("row lacks Current predictions")
    state = tuple(current.get(target) for target in STATE_TARGETS)
    return None if any(value is None for value in state) else state


def _labels(row: Mapping[str, Any]) -> dict[str, int | None]:
    labels = row.get("labels")
    if labels is None:
        labels = {
            "latency": row.get("latency_label"),
            **dict(row.get("resource_labels") or {}),
        }
    if not isinstance(labels, Mapping):
        raise ValueError("row lacks labels")
    normalized = {target: labels.get(target) for target in ALL_TARGETS}
    for target, label in normalized.items():
        if label is not None and (
            isinstance(label, bool)
            or not isinstance(label, int)
            or not 0 <= label < BUCKETS[target]
        ):
            raise ValueError(f"invalid {target} label")
    return normalized


def _fit_counts(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[Any, ...], dict[str, Counter[int]]]:
    counts: dict[tuple[Any, ...], dict[str, Counter[int]]] = defaultdict(
        lambda: {target: Counter() for target in CALIBRATED_TARGETS}
    )
    for row in rows:
        _base(row, composition=False)
        state = _state(row)
        if state is None:
            continue
        labels = _labels(row)
        for target in CALIBRATED_TARGETS:
            label = labels.get(target)
            if label is not None:
                counts[state][target][label] += 1
    return dict(counts)


def _base(
    row: Mapping[str, Any], *, composition: bool
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    current = row["current_dynamic"]
    if composition:
        arm = row.get("arms", {}).get("composition")
        if not isinstance(arm, Mapping):
            raise ValueError("validation row lacks frozen composition")
        hard = dict(arm["candidate"])
        pmfs = {
            target: _validate_pmf(
                target, arm["candidate_probability_by_bucket"][target]
            )
            for target in ALL_TARGETS
        }
    else:
        hard = {target: current[target] for target in ALL_TARGETS}
        nested = current.get("probability_by_bucket")
        if not isinstance(nested, Mapping):
            nested = row.get("current_probability_by_bucket")
        if not isinstance(nested, Mapping):
            raise ValueError("fit row lacks Current PMFs")
        pmfs = {target: _validate_pmf(target, nested[target]) for target in ALL_TARGETS}
    for target in ALL_TARGETS:
        if hard[target] is not None and _class_id(target, hard[target]) != _class_id(
            target, _hard(target, pmfs[target])
        ):
            raise ValueError(f"{target} hard prediction differs from its PMF")
    return hard, pmfs


def _apply(
    rows: Sequence[Mapping[str, Any]],
    counts: Mapping[tuple[Any, ...], Mapping[str, Counter[int]]],
    *,
    composition: bool,
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        labels = _labels(row)
        hard, pmfs = _base(row, composition=composition)
        candidate = dict(hard)
        candidate_pmfs = deepcopy(pmfs)
        state = _state(row)
        support: dict[str, int] = {}
        for target in CALIBRATED_TARGETS:
            state_counts = (
                Counter()
                if state is None
                else counts.get(state, {}).get(target, Counter())
            )
            total = sum(state_counts.values())
            support[target] = total
            if total == 0:
                continue
            posterior = [
                (state_counts[bucket] + ALPHA * pmfs[target][bucket]) / (total + ALPHA)
                for bucket in range(BUCKETS[target])
            ]
            candidate_pmfs[target] = posterior
            candidate[target] = _hard(target, posterior)
        output.append(
            {
                "sample_id": row["sample_id"],
                "task_id": row["task_id"],
                "command": row["command"],
                "labels": dict(labels),
                "current_dynamic": {
                    target: row["current_dynamic"][target] for target in ALL_TARGETS
                },
                "current_probability_by_bucket": deepcopy(
                    _base(row, composition=False)[1]
                ),
                "base": hard,
                "base_probability_by_bucket": pmfs,
                "candidate": candidate,
                "candidate_probability_by_bucket": candidate_pmfs,
                "joint_state": list(state) if state is not None else None,
                "state_support": support,
            }
        )
    return output


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
        current, pmfs = _base(source, composition=False)
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


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    current_rows = [
        {
            **row,
            "candidate": row["current_dynamic"],
            "candidate_probability_by_bucket": row["current_probability_by_bucket"],
        }
        for row in rows
    ]
    targets = {}
    for target in ALL_TARGETS:
        current = _fail_closed_metrics(current_rows, target)
        candidate = _fail_closed_metrics(rows, target)
        accuracy = "exact_class_accuracy" if target == "latency" else "accuracy"
        targets[target] = {
            "current": current,
            "candidate": candidate,
            "delta_percentage_points": _accuracy_delta(
                candidate[accuracy], current[accuracy]
            ),
            "changes": _phase_changes(rows, target),
        }
    return targets


def forward_audit(
    fit_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task_order = list(dict.fromkeys(str(row["task_id"]) for row in fit_rows))
    if len(task_order) != CALIBRATION_TASKS + AUDIT_TASKS:
        raise ValueError("fit task split differs from the frozen 40/40 protocol")
    calibration_ids = set(task_order[:CALIBRATION_TASKS])
    audit_ids = set(task_order[CALIBRATION_TASKS:])
    calibration = [row for row in fit_rows if row["task_id"] in calibration_ids]
    audit = [row for row in fit_rows if row["task_id"] in audit_ids]
    rows = _apply(audit, _fit_counts(calibration), composition=False)
    row_identity = _row_identity(audit, rows)
    metrics = _metrics(rows)
    coverage = {
        target: sum(
            row["labels"][target] is not None and row["state_support"][target] > 0
            for row in rows
        )
        / sum(row["labels"][target] is not None for row in rows)
        for target in CALIBRATED_TARGETS
    }
    gain = all(
        metrics[target]["delta_percentage_points"] > 0.0
        for target in CALIBRATED_TARGETS
    )
    helpful = all(
        metrics[target]["changes"]["helpful"] > metrics[target]["changes"]["harmful"]
        for target in CALIBRATED_TARGETS
    )
    severe = all(
        metrics[target]["candidate"]["severe_underprediction_rate"]
        <= metrics[target]["current"]["severe_underprediction_rate"]
        for target in CALIBRATED_TARGETS
    )
    go = (
        gain
        and helpful
        and severe
        and all(value >= MIN_COVERAGE for value in coverage.values())
        and row_identity
    )
    return {
        "go": go,
        "calibration_tasks": task_order[:CALIBRATION_TASKS],
        "audit_tasks": task_order[CALIBRATION_TASKS:],
        "calibration_rows": len(calibration),
        "audit_rows": len(audit),
        "state_count": len(_fit_counts(calibration)),
        "target_coverage": coverage,
        "row_identity": row_identity,
        "targets": metrics,
        "gate": {
            "strict_gain_cpu_disk": gain,
            "helpful_exceeds_harmful_cpu_disk": helpful,
            "no_severe_underprediction_regression_cpu_disk": severe,
            "minimum_target_coverage": MIN_COVERAGE,
            "row_identity": row_identity,
        },
    }, rows


def validation(
    fit_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _apply(validation_rows, _fit_counts(fit_rows), composition=True)
    row_identity = _row_identity(validation_rows, rows)
    metrics = _metrics(rows)
    unchanged = all(
        row["candidate"][target] == row["base"][target]
        and row["candidate_probability_by_bucket"][target]
        == row["base_probability_by_bucket"][target]
        for row in rows
        for target in ("latency", "sampled_peak_rss_mb")
    )
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
        task_id
        for target in ALL_TARGETS
        for task_id in metrics[target]["changes"]["helpful_task_ids"]
    }
    go = (
        gain
        and severe
        and helpful
        and len(helpful_tasks) >= 10
        and unchanged
        and row_identity
    )
    return {
        "go": go,
        "targets": metrics,
        "latency_rss_bit_identical_to_composition": unchanged,
        "helpful_tasks": len(helpful_tasks),
        "row_identity": row_identity,
        "gate": {
            "minimum_gain_percentage_points_each": 5.0,
            "all_targets_meet_gain": gain,
            "no_severe_underprediction_regression": severe,
            "helpful_exceeds_harmful_each_target": helpful,
            "minimum_helpful_tasks": 10,
            "row_identity": row_identity,
        },
    }, rows


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _run(args: argparse.Namespace) -> None:
    if args.out_dir.exists():
        raise FileExistsError("output directory already exists")
    if (
        args.fit_rows.resolve() != FIT_ROWS.resolve()
        or _sha256(args.fit_rows) != FIT_SHA256
        or args.validation_rows.resolve() != VALIDATION_ROWS.resolve()
    ):
        raise ValueError("inputs differ from the frozen joint-state protocol")
    fit_rows = _load(args.fit_rows)
    audit, rows = forward_audit(fit_rows)
    result: dict[str, Any] = {
        "schema": VERSION,
        "status": "forward_go" if audit["go"] else "forward_no_go",
        "claim_bearing": False,
        "protocol": {
            "state": list(STATE_TARGETS),
            "calibrated_targets": list(CALIBRATED_TARGETS),
            "alpha": ALPHA,
            "fit_split_tasks": [CALIBRATION_TASKS, AUDIT_TASKS],
            "fallback": "base_pmf_bit_identical",
            "validation_base": "frozen_component_composition_v1",
            "validation_unchanged_targets": ["latency", "sampled_peak_rss_mb"],
        },
        "inputs": {
            "evaluator_sha256": _sha256(Path(__file__)),
            "fit_rows": str(args.fit_rows.resolve()),
            "fit_rows_sha256": FIT_SHA256,
            "validation_rows": "unopened"
            if not audit["go"]
            else str(args.validation_rows.resolve()),
            "validation_rows_sha256": None if not audit["go"] else VALIDATION_SHA256,
            "git_sha": _git_sha(),
        },
        "forward_audit": audit,
    }
    if audit["go"]:
        if _sha256(args.validation_rows) != VALIDATION_SHA256:
            raise ValueError("validation rows changed after preregistration")
        scored, rows = validation(fit_rows, _load(args.validation_rows))
        result["validation"] = scored
        result["status"] = "development_go" if scored["go"] else "development_no_go"
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-rows", type=Path, required=True)
    parser.add_argument("--validation-rows", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    _run(parser.parse_args())


if __name__ == "__main__":
    main()
