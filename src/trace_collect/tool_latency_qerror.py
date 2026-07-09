"""Causal q-error evaluation for tool latency prediction baselines."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from trace_collect.causal_history import iter_causal_latency_observations
from trace_collect.latency_outputs import write_summary_outputs
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

    predictions: list[dict[str, Any]] = []
    cold_start_count = 0
    clamped_count = 0
    for observation in iter_causal_latency_observations(rows, min_tool_history=1):
        history = observation.history
        latency_ms = observation.latency_ms
        if history:
            predicted_ms = _quantile(history, quantile)
            history_count = len(history)
        else:
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
                "sample_id": observation.sample_id,
                "tool_name": observation.tool_name,
                "latency_ms": latency_ms,
                "tool_ts_start": observation.tool_ts_start,
                "tool_ts_end": observation.tool_ts_end,
                "predicted_latency_ms": predicted_ms,
                "qerror": qerror,
                "prediction_source": observation.prediction_source,
                "history_count": history_count,
            }
        )

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
    write_summary_outputs(
        summary,
        detail_key="predictions",
        summary_path=summary_path,
        detail_path=predictions_path,
    )


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
