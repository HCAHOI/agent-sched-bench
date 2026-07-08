"""Causal threshold-decision evaluation for tool latency labels."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

from trace_collect.tool_latency_dataset import read_tool_latency_jsonl
from trace_collect.latency_validation import (
    required_nonnegative_float,
    required_text,
    row_order_key,
)


@dataclass(frozen=True)
class ThresholdDecision:
    """One causal threshold decision for one tool latency row."""

    sample_id: str
    tool_name: str
    tool_ts_start: float
    latency_ms: float
    threshold_ms: float
    label_exceeds_threshold: bool
    predicted_exceeds_threshold: bool | None
    probability_exceeds_threshold: float | None
    prediction_source: str
    history_count: int

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "tool_name": self.tool_name,
            "tool_ts_start": self.tool_ts_start,
            "latency_ms": self.latency_ms,
            "threshold_ms": self.threshold_ms,
            "label_exceeds_threshold": self.label_exceeds_threshold,
            "predicted_exceeds_threshold": self.predicted_exceeds_threshold,
            "probability_exceeds_threshold": self.probability_exceeds_threshold,
            "prediction_source": self.prediction_source,
            "history_count": self.history_count,
        }


def evaluate_latency_thresholds(
    rows: Iterable[dict[str, Any]],
    *,
    thresholds_ms: Iterable[float],
    probability_cutoff: float = 0.5,
    min_tool_history: int = 1,
) -> dict[str, Any]:
    """Evaluate causal survival-probability decisions for latency thresholds.

    For each current tool action and threshold ``T``, this answers the only
    scheduler-relevant question: whether observed latency is greater than ``T``.
    The predictor is data-driven: it uses the empirical survival rate from completed
    prior observations of the same ``tool_name`` when enough history exists, and
    otherwise falls back to completed global history. No tool classes are
    hardcoded.
    """

    thresholds = _normalize_thresholds(thresholds_ms)
    if not math.isfinite(probability_cutoff) or not 0.0 <= probability_cutoff <= 1.0:
        raise ValueError(
            "probability_cutoff must be finite and in [0, 1], "
            f"got {probability_cutoff}"
        )
    if min_tool_history < 1:
        raise ValueError(f"min_tool_history must be >= 1, got {min_tool_history}")

    ordered_rows = sorted(list(rows), key=row_order_key)
    global_history: list[float] = []
    history_by_tool: dict[str, list[float]] = {}
    pending_updates: list[tuple[float, str, float]] = []
    current_source: str | None = None
    decisions: list[ThresholdDecision] = []

    row_index = 0
    while row_index < len(ordered_rows):
        bucket_start = row_index
        bucket_source = required_text(
            ordered_rows[bucket_start],
            "source_trace",
            source=f"row {bucket_start}",
        )
        if current_source is not None and bucket_source != current_source:
            for _, tool_name, latency_ms in pending_updates:
                global_history.append(latency_ms)
                history_by_tool.setdefault(tool_name, []).append(latency_ms)
            pending_updates = []
        current_source = bucket_source
        bucket_ts_start = required_nonnegative_float(
            ordered_rows[bucket_start],
            "tool_ts_start",
            source=f"row {bucket_start}",
        )
        ready_updates = [
            update for update in pending_updates if update[0] <= bucket_ts_start
        ]
        pending_updates = [
            update for update in pending_updates if update[0] > bucket_ts_start
        ]
        for _, tool_name, latency_ms in ready_updates:
            global_history.append(latency_ms)
            history_by_tool.setdefault(tool_name, []).append(latency_ms)

        while row_index < len(ordered_rows):
            row = ordered_rows[row_index]
            source_trace = required_text(
                row,
                "source_trace",
                source=f"row {row_index}",
            )
            tool_ts_start = required_nonnegative_float(
                row,
                "tool_ts_start",
                source=f"row {row_index}",
            )
            if source_trace != bucket_source or tool_ts_start != bucket_ts_start:
                break
            row_index += 1

        bucket_rows = ordered_rows[bucket_start:row_index]
        bucket_updates: list[tuple[float, str, float]] = []
        for scored_index, row in enumerate(bucket_rows, start=bucket_start):
            sample_id = required_text(row, "sample_id", source=f"row {scored_index}")
            tool_name = required_text(row, "tool_name", source=f"row {scored_index}")
            latency_ms = required_nonnegative_float(
                row,
                "latency_ms",
                source=f"row {scored_index}",
            )
            tool_ts_start = required_nonnegative_float(
                row,
                "tool_ts_start",
                source=f"row {scored_index}",
            )
            tool_ts_end = required_nonnegative_float(
                row,
                "tool_ts_end",
                source=f"row {scored_index}",
            )
            if tool_ts_end < tool_ts_start:
                raise ValueError(f"row {scored_index}: tool_ts_end < tool_ts_start")

            tool_history = history_by_tool.get(tool_name, [])
            if len(tool_history) >= min_tool_history:
                prediction_source = "tool_history"
                history = tool_history
            elif global_history:
                prediction_source = "global_history"
                history = global_history
            else:
                prediction_source = "cold_start"
                history = []

            for threshold_ms in thresholds:
                label = latency_ms > threshold_ms
                if history:
                    probability = _survival_probability(history, threshold_ms)
                    predicted = probability >= probability_cutoff
                    history_count = len(history)
                else:
                    probability = None
                    predicted = None
                    history_count = 0
                decisions.append(
                    ThresholdDecision(
                        sample_id=sample_id,
                        tool_name=tool_name,
                        tool_ts_start=tool_ts_start,
                        latency_ms=latency_ms,
                        threshold_ms=threshold_ms,
                        label_exceeds_threshold=label,
                        predicted_exceeds_threshold=predicted,
                        probability_exceeds_threshold=probability,
                        prediction_source=prediction_source,
                        history_count=history_count,
                    )
                )
            bucket_updates.append((tool_ts_end, tool_name, latency_ms))

        pending_updates.extend(bucket_updates)

    return {
        "predictor": "causal_empirical_survival",
        "thresholds_ms": thresholds,
        "probability_cutoff": probability_cutoff,
        "min_tool_history": min_tool_history,
        "row_count": len(ordered_rows),
        "decision_count": len(decisions),
        "metrics_by_threshold": _metrics_by_threshold(decisions),
        "metrics_by_tool_threshold": _metrics_by_tool_threshold(decisions),
        "decisions": [decision.to_json_obj() for decision in decisions],
    }


def load_and_evaluate_latency_thresholds(
    path: Path,
    *,
    thresholds_ms: Iterable[float],
    probability_cutoff: float = 0.5,
    min_tool_history: int = 1,
) -> dict[str, Any]:
    return evaluate_latency_thresholds(
        read_tool_latency_jsonl(path),
        thresholds_ms=thresholds_ms,
        probability_cutoff=probability_cutoff,
        min_tool_history=min_tool_history,
    )


def write_threshold_outputs(
    summary: dict[str, Any],
    *,
    summary_path: Path | None = None,
    decisions_path: Path | None = None,
) -> None:
    summary_without_decisions = {
        key: value for key, value in summary.items() if key != "decisions"
    }
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary_without_decisions, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    if decisions_path is not None:
        decisions_path.parent.mkdir(parents=True, exist_ok=True)
        with decisions_path.open("w", encoding="utf-8") as fh:
            for row in summary["decisions"]:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                fh.write("\n")


def _metrics_by_threshold(
    decisions: Iterable[ThresholdDecision],
) -> dict[str, dict[str, float | int | None]]:
    grouped: dict[float, list[ThresholdDecision]] = {}
    for decision in decisions:
        grouped.setdefault(decision.threshold_ms, []).append(decision)
    return {
        str(threshold): _classification_metrics(rows)
        for threshold, rows in sorted(grouped.items())
    }


def _metrics_by_tool_threshold(
    decisions: Iterable[ThresholdDecision],
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    grouped: dict[str, dict[float, list[ThresholdDecision]]] = {}
    for decision in decisions:
        grouped.setdefault(decision.tool_name, {}).setdefault(
            decision.threshold_ms,
            [],
        ).append(decision)
    return {
        tool_name: {
            str(threshold): _classification_metrics(rows)
            for threshold, rows in sorted(threshold_groups.items())
        }
        for tool_name, threshold_groups in sorted(grouped.items())
    }


def _classification_metrics(
    decisions: list[ThresholdDecision],
) -> dict[str, float | int | None]:
    evaluated = [d for d in decisions if d.predicted_exceeds_threshold is not None]
    positives = sum(d.label_exceeds_threshold for d in decisions)
    if not evaluated:
        return {
            "row_count": len(decisions),
            "evaluated_count": 0,
            "cold_start_count": len(decisions),
            "positive_count": positives,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "false_positive_rate": None,
            "false_negative_rate": None,
        }
    tp = sum(d.predicted_exceeds_threshold and d.label_exceeds_threshold for d in evaluated)
    fp = sum(d.predicted_exceeds_threshold and not d.label_exceeds_threshold for d in evaluated)
    tn = sum(
        (not d.predicted_exceeds_threshold) and (not d.label_exceeds_threshold)
        for d in evaluated
    )
    fn = sum((not d.predicted_exceeds_threshold) and d.label_exceeds_threshold for d in evaluated)
    return {
        "row_count": len(decisions),
        "evaluated_count": len(evaluated),
        "cold_start_count": len(decisions) - len(evaluated),
        "positive_count": positives,
        "accuracy": (tp + tn) / len(evaluated),
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "false_positive_rate": _safe_div(fp, fp + tn),
        "false_negative_rate": _safe_div(fn, fn + tp),
    }


def _survival_probability(history: list[float], threshold_ms: float) -> float:
    if not history:
        raise ValueError("cannot predict without history")
    return sum(value > threshold_ms for value in history) / len(history)


def _safe_div(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _normalize_thresholds(values: Iterable[float]) -> list[float]:
    thresholds = sorted({float(value) for value in values})
    if not thresholds:
        raise ValueError("at least one threshold is required")
    for threshold in thresholds:
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError(f"thresholds must be finite and positive, got {threshold}")
    return thresholds


__all__ = [
    "ThresholdDecision",
    "evaluate_latency_thresholds",
    "load_and_evaluate_latency_thresholds",
    "write_threshold_outputs",
]
