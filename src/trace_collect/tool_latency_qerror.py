"""Causal q-error evaluation for tool latency prediction baselines."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from trace_collect.latency_validation import (
    required_nonnegative_float,
    required_text,
    row_order_key,
)
from trace_collect.tool_latency_dataset import read_tool_latency_jsonl


def evaluate_latency_qerror(
    rows: Iterable[dict[str, Any]],
    *,
    quantile: float = 0.5,
    epsilon_ms: float = 1.0,
) -> dict[str, Any]:
    """Evaluate a causal historical-quantile latency predictor.

    Rows are sorted by ``source_trace`` then ``tool_ts_start`` before scoring.
    Each row is predicted using only rows already seen earlier in that causal
    order. The per-tool history is tried first; if absent, earlier global
    history is used. Rows with no prior history are cold starts and excluded
    from q-error aggregates.
    """

    if not math.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be finite and in [0, 1], got {quantile}")
    if not math.isfinite(epsilon_ms) or epsilon_ms <= 0.0:
        raise ValueError(f"epsilon_ms must be finite and positive, got {epsilon_ms}")

    global_history: list[float] = []
    history_by_tool: dict[str, list[float]] = {}
    predictions: list[dict[str, Any]] = []
    cold_start_count = 0
    clamped_count = 0
    ordered_rows = sorted(list(rows), key=row_order_key)
    pending_updates: list[tuple[float, str, float]] = []
    current_source: str | None = None



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
            tool_name = required_text(row, "tool_name", source=f"row {scored_index}")
            sample_id = required_text(row, "sample_id", source=f"row {scored_index}")
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
            if tool_history:
                prediction_source = "tool_history"
                predicted_ms = _quantile(tool_history, quantile)
                history_count = len(tool_history)
            elif global_history:
                prediction_source = "global_history"
                predicted_ms = _quantile(global_history, quantile)
                history_count = len(global_history)
            else:
                prediction_source = "cold_start"
                predicted_ms = None
                history_count = 0
                cold_start_count += 1

            if predicted_ms is None:
                qerror = None
            else:
                actual_for_qerror = max(latency_ms, epsilon_ms)
                predicted_for_qerror = max(predicted_ms, epsilon_ms)
                if actual_for_qerror != latency_ms or predicted_for_qerror != predicted_ms:
                    clamped_count += 1
                qerror = max(
                    predicted_for_qerror / actual_for_qerror,
                    actual_for_qerror / predicted_for_qerror,
                )

            predictions.append(
                {
                    "sample_id": sample_id,
                    "tool_name": tool_name,
                    "latency_ms": latency_ms,
                    "tool_ts_start": tool_ts_start,
                    "tool_ts_end": tool_ts_end,
                    "predicted_latency_ms": predicted_ms,
                    "qerror": qerror,
                    "prediction_source": prediction_source,
                    "history_count": history_count,
                }
            )
            bucket_updates.append((tool_ts_end, tool_name, latency_ms))

        pending_updates.extend(bucket_updates)

    qerrors = [p["qerror"] for p in predictions if p["qerror"] is not None]
    if not predictions:
        raise ValueError("no latency rows supplied")
    summary = {
        "predictor": "causal_historical_quantile",
        "quantile": quantile,
        "epsilon_ms": epsilon_ms,
        "row_count": len(predictions),
        "evaluated_count": len(qerrors),
        "cold_start_count": cold_start_count,
        "clamped_count": clamped_count,
        "qerror": _qerror_summary(qerrors),
        "by_tool": _summarize_by_tool(predictions),
        "predictions": predictions,
    }
    return summary


def load_and_evaluate_latency_qerror(
    path: Path,
    *,
    quantile: float = 0.5,
    epsilon_ms: float = 1.0,
) -> dict[str, Any]:
    return evaluate_latency_qerror(
        read_tool_latency_jsonl(path),
        quantile=quantile,
        epsilon_ms=epsilon_ms,
    )


def write_qerror_outputs(
    summary: dict[str, Any],
    *,
    summary_path: Path | None = None,
    predictions_path: Path | None = None,
) -> None:
    summary_without_predictions = {
        key: value for key, value in summary.items() if key != "predictions"
    }
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary_without_predictions, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    if predictions_path is not None:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with predictions_path.open("w", encoding="utf-8") as fh:
            for row in summary["predictions"]:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                fh.write("\n")


def _qerror_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p90": None, "p95": None, "max": None}
    return {
        "mean": fmean(values),
        "p50": _quantile(values, 0.50),
        "p90": _quantile(values, 0.90),
        "p95": _quantile(values, 0.95),
        "max": max(values),
    }


def _summarize_by_tool(predictions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    for row in predictions:
        tool_name = str(row["tool_name"])
        counts[tool_name] = counts.get(tool_name, 0) + 1
        qerror = row["qerror"]
        if qerror is not None:
            grouped.setdefault(tool_name, []).append(float(qerror))
    result: dict[str, dict[str, Any]] = {}
    for tool_name in sorted(counts):
        result[tool_name] = {"row_count": counts[tool_name], **_qerror_summary(grouped.get(tool_name, []))}
    return result


def _quantile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute quantile of empty values")
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction




__all__ = [
    "evaluate_latency_qerror",
    "load_and_evaluate_latency_qerror",
    "write_qerror_outputs",
]
