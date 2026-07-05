#!/usr/bin/env python3
"""Validate the offline predictive checkpoint policy on collected traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from trace_collect.predictive_policy import (
    CommandFamily,
    classify_tool_result,
    needs_checkpoint,
)


class CheckpointOutcome(Enum):
    CREATED = "created"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class PredictionStats:
    total_actions: int = 0
    predictions_made: int = 0
    ground_truth_available: int = 0
    predicted_checkpoint: int = 0
    predicted_skip: int = 0
    actual_checkpointed: int = 0
    actual_created: int = 0
    actual_skipped: int = 0
    actual_error: int = 0
    true_positive: int = 0
    true_negative: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def record(
        self,
        *,
        predicted_checkpoint: bool,
        actual: CheckpointOutcome | None,
    ) -> None:
        self.total_actions += 1
        self.predictions_made += 1
        if predicted_checkpoint:
            self.predicted_checkpoint += 1
        else:
            self.predicted_skip += 1

        if actual is None:
            return
        self.ground_truth_available += 1

        actual_checkpointed = actual in {
            CheckpointOutcome.CREATED,
            CheckpointOutcome.ERROR,
        }
        if actual == CheckpointOutcome.CREATED:
            self.actual_created += 1
        elif actual == CheckpointOutcome.SKIPPED:
            self.actual_skipped += 1
        elif actual == CheckpointOutcome.ERROR:
            self.actual_error += 1

        if actual_checkpointed:
            self.actual_checkpointed += 1

        if predicted_checkpoint and actual_checkpointed:
            self.true_positive += 1
        elif predicted_checkpoint and not actual_checkpointed:
            self.false_positive += 1
        elif not predicted_checkpoint and actual_checkpointed:
            self.false_negative += 1
        else:
            self.true_negative += 1

    def to_dict(self) -> dict[str, Any]:
        mispredictions = self.false_negative + self.false_positive
        return {
            "total_actions": self.total_actions,
            "predictions_made": self.predictions_made,
            "ground_truth_available": self.ground_truth_available,
            "predicted_checkpoint": self.predicted_checkpoint,
            "predicted_skip": self.predicted_skip,
            "actual_checkpointed": self.actual_checkpointed,
            "actual_created": self.actual_created,
            "actual_skipped": self.actual_skipped,
            "actual_error": self.actual_error,
            "true_negative": self.true_negative,
            "true_positive": self.true_positive,
            "false_negative": self.false_negative,
            "false_positive": self.false_positive,
            "misprediction_rate": _rate(
                mispredictions,
                self.ground_truth_available,
            ),
            "misprediction_rate_denominator": "ground_truth_available",
            "fn_rate": _rate(self.false_negative, self.actual_checkpointed),
            "fn_rate_denominator": "actual_checkpointed",
            "bias": _prediction_bias(self.false_positive, self.false_negative),
        }


def discover_trace_paths(trace_dir: Path) -> list[Path]:
    """Return trace JSONL files under a simulate/collect output directory."""
    if trace_dir.is_file():
        return [trace_dir]
    if not trace_dir.is_dir():
        raise FileNotFoundError(f"trace dir does not exist: {trace_dir}")

    trace_paths = sorted(path for path in trace_dir.rglob("trace.jsonl"))
    if trace_paths:
        return trace_paths
    return sorted(path for path in trace_dir.rglob("*.jsonl") if path.is_file())


def build_report(trace_dir: Path) -> dict[str, Any]:
    """Build a predictive-policy validation report from trace JSONL files."""
    input_trace_paths = discover_trace_paths(trace_dir)
    trace_paths = _validation_trace_paths(input_trace_paths)
    overall = PredictionStats()
    by_family: dict[CommandFamily, PredictionStats] = {}
    by_benchmark: dict[str, PredictionStats] = {}
    status_counts: Counter[str] = Counter()

    for trace_path in trace_paths:
        metadata, actions = _load_tool_actions(trace_path)
        benchmark = _benchmark_from_metadata(metadata)
        benchmark_stats = by_benchmark.setdefault(benchmark, PredictionStats())

        for action in actions:
            data = _tool_data(action, trace_path)
            family = classify_tool_result(
                _optional_str(data.get("tool_name")),
                _optional_str(data.get("tool_args")),
            )
            predicted_checkpoint = needs_checkpoint(family)
            actual = _checkpoint_outcome(data)
            if actual is not None:
                status_counts[actual.value] += 1

            overall.record(
                predicted_checkpoint=predicted_checkpoint,
                actual=actual,
            )
            by_family.setdefault(family, PredictionStats()).record(
                predicted_checkpoint=predicted_checkpoint,
                actual=actual,
            )
            benchmark_stats.record(
                predicted_checkpoint=predicted_checkpoint,
                actual=actual,
            )

    overall_dict = overall.to_dict()
    return {
        "trace_dir": str(trace_dir),
        "input_trace_count": len(input_trace_paths),
        "trace_count": len(trace_paths),
        "total_actions": overall.total_actions,
        "predictions_made": overall.predictions_made,
        "ground_truth_available": overall.ground_truth_available,
        "confusion": {
            "true_negative": overall.true_negative,
            "true_positive": overall.true_positive,
            "false_negative": overall.false_negative,
            "false_positive": overall.false_positive,
        },
        "misprediction_rate": overall_dict["misprediction_rate"],
        "misprediction_rate_denominator": overall_dict[
            "misprediction_rate_denominator"
        ],
        "fn_rate": overall_dict["fn_rate"],
        "fn_rate_denominator": overall_dict["fn_rate_denominator"],
        "ground_truth_status_counts": dict(sorted(status_counts.items())),
        "families": {
            family.value: by_family[family].to_dict()
            for family in CommandFamily
            if family in by_family
        },
        "benchmarks": {
            benchmark: stats.to_dict()
            for benchmark, stats in sorted(by_benchmark.items())
        },
    }


def print_report(report: dict[str, Any]) -> None:
    """Print the human-readable table and machine-readable JSON report."""
    print("=" * 72)
    print("PREDICTIVE CHECKPOINT POLICY VALIDATION")
    print("=" * 72)
    print(f"Trace dir: {report['trace_dir']}")
    print(f"Input traces: {report['input_trace_count']}")
    print(f"Validation traces: {report['trace_count']}")
    print(f"Total actions: {report['total_actions']}")
    print(f"Predictions made: {report['predictions_made']}")
    print(f"Ground truth available: {report['ground_truth_available']}")
    print(
        "Confusion: "
        f"TN={report['confusion']['true_negative']} "
        f"TP={report['confusion']['true_positive']} "
        f"FN={report['confusion']['false_negative']} "
        f"FP={report['confusion']['false_positive']}"
    )
    print(
        "Misprediction rate: "
        f"{_format_rate(report['misprediction_rate'])} "
        f"(denominator: {report['misprediction_rate_denominator']})"
    )
    print(
        "FN rate: "
        f"{_format_rate(report['fn_rate'])} "
        f"(denominator: {report['fn_rate_denominator']})"
    )
    print()
    _print_stats_table("By command family", report["families"])
    print()
    _print_stats_table("By benchmark", report["benchmarks"])
    print()
    print("JSON report:")
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the predictive checkpoint-skip classifier against "
            "checkpoint_after ground truth in collected trace JSONL files."
        )
    )
    parser.add_argument(
        "--trace-dir",
        required=True,
        type=Path,
        help=(
            "Directory containing per-task trace.jsonl files, or a single "
            "trace JSONL file."
        ),
    )
    args = parser.parse_args()

    if not args.trace_dir.exists():
        raise SystemExit(f"trace dir does not exist: {args.trace_dir}")
    print_report(build_report(args.trace_dir))


def _load_tool_actions(trace_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata: dict[str, Any] = {}
    actions: list[dict[str, Any]] = []
    with trace_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError(
                    f"trace record must be a JSON object: {trace_path}:{line_number}"
                )
            if record.get("type") == "trace_metadata":
                metadata.update(record)
            elif (
                record.get("type") == "action"
                and record.get("action_type") == "tool_exec"
            ):
                actions.append(record)
    return metadata, actions


def _validation_trace_paths(input_trace_paths: list[Path]) -> list[Path]:
    validation_paths: list[Path] = []
    seen: set[Path] = set()
    for trace_path in input_trace_paths:
        metadata, actions = _load_tool_actions(trace_path)
        if _has_checkpoint_ground_truth(actions):
            _append_unique_path(validation_paths, seen, trace_path)
            continue

        source_paths = _source_trace_paths_from_metadata(metadata, trace_path)
        if not source_paths:
            _append_unique_path(validation_paths, seen, trace_path)
            continue
        for source_path in source_paths:
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"source trace referenced by {trace_path} does not exist: "
                    f"{source_path}"
                )
            _append_unique_path(validation_paths, seen, source_path)
    return validation_paths


def _append_unique_path(paths: list[Path], seen: set[Path], path: Path) -> None:
    resolved = path.resolve()
    if resolved in seen:
        return
    seen.add(resolved)
    paths.append(path)


def _has_checkpoint_ground_truth(actions: list[dict[str, Any]]) -> bool:
    for action in actions:
        data = action.get("data")
        if not isinstance(data, dict):
            continue
        if "checkpoint_after" in data or "checkpoint_after_error" in data:
            return True
    return False


def _source_trace_paths_from_metadata(
    metadata: dict[str, Any],
    trace_path: Path,
) -> list[Path]:
    paths: list[Path] = []
    entries = metadata.get("source_trace_entries")
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            raw_path = entry.get("source_trace")
            if isinstance(raw_path, str) and raw_path:
                paths.append(_resolve_trace_reference(raw_path, trace_path))

    raw_paths = metadata.get("source_traces")
    if isinstance(raw_paths, list):
        for raw_path in raw_paths:
            if isinstance(raw_path, str) and raw_path:
                paths.append(_resolve_trace_reference(raw_path, trace_path))
    return _dedupe_paths(paths)


def _resolve_trace_reference(raw_path: str, trace_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return trace_path.parent / path


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        _append_unique_path(deduped, seen, path)
    return deduped


def _tool_data(action: dict[str, Any], trace_path: Path) -> dict[str, Any]:
    data = action.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"tool_exec action missing object data in {trace_path}")
    return data


def _checkpoint_outcome(data: dict[str, Any]) -> CheckpointOutcome | None:
    checkpoint_after = data.get("checkpoint_after")
    if isinstance(checkpoint_after, dict):
        if "error" in checkpoint_after:
            return CheckpointOutcome.ERROR
        if "skipped" in checkpoint_after:
            return CheckpointOutcome.SKIPPED
        return CheckpointOutcome.CREATED
    if isinstance(checkpoint_after, str) and checkpoint_after:
        return CheckpointOutcome.CREATED
    if checkpoint_after is not None:
        raise ValueError(f"invalid checkpoint_after payload: {checkpoint_after!r}")

    checkpoint_error = data.get("checkpoint_after_error")
    if isinstance(checkpoint_error, dict):
        return CheckpointOutcome.ERROR
    if checkpoint_error is not None:
        raise ValueError(
            f"invalid checkpoint_after_error payload: {checkpoint_error!r}"
        )
    return None


def _benchmark_from_metadata(metadata: dict[str, Any]) -> str:
    benchmark = metadata.get("benchmark") or metadata.get("benchmark_slug")
    if isinstance(benchmark, str) and benchmark:
        return benchmark
    run_config = metadata.get("run_config")
    if isinstance(run_config, dict):
        configured = run_config.get("benchmark") or run_config.get("benchmark_slug")
        if isinstance(configured, str) and configured:
            return configured
    return "unknown"


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _format_rate(rate: float | None) -> str:
    if rate is None:
        return "-"
    return f"{100 * rate:.2f}%"


def _prediction_bias(false_positive: int, false_negative: int) -> str:
    if false_positive > false_negative:
        return f"over_predicts_by_{false_positive - false_negative}"
    if false_negative > false_positive:
        return f"under_predicts_by_{false_negative - false_positive}"
    return "balanced"


def _print_stats_table(title: str, stats_by_key: dict[str, dict[str, Any]]) -> None:
    print(title)
    header = (
        f"{'Group':<18} {'Actions':>7} {'GT':>7} {'PredCP':>7} "
        f"{'ActualCP':>8} {'TN':>5} {'TP':>5} {'FN':>5} {'FP':>5} "
        f"{'Mis%':>8} {'FN%':>8} {'Bias':>22}"
    )
    print(header)
    print("-" * len(header))
    for key, stats in stats_by_key.items():
        print(
            f"{key:<18} "
            f"{stats['total_actions']:>7} "
            f"{stats['ground_truth_available']:>7} "
            f"{stats['predicted_checkpoint']:>7} "
            f"{stats['actual_checkpointed']:>8} "
            f"{stats['true_negative']:>5} "
            f"{stats['true_positive']:>5} "
            f"{stats['false_negative']:>5} "
            f"{stats['false_positive']:>5} "
            f"{_format_rate(stats['misprediction_rate']):>8} "
            f"{_format_rate(stats['fn_rate']):>8} "
            f"{stats['bias']:>22}"
        )


if __name__ == "__main__":
    main()
