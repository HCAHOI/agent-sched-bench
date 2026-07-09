"""Causal bucket prediction evaluation for tool latency labels.

Bucket edges are strictly increasing latency thresholds, typically derived
from profiled KV swap costs (one edge per profiled KV size, plus guard time).
Edge ``e_i`` splits latency exactly like the binary threshold labels: bucket
``i`` contains latencies with ``e_{i-1} < latency_ms <= e_i`` (bucket 0 is
``[0, e_0]``, the last bucket is ``> e_{k-1}``), so "label bucket > i" is
equivalent to "latency exceeds edge i". Predicting the bucket therefore
answers every profiled threshold at once as the KV cache grows.

The predictor is data-driven: the empirical bucket distribution of completed
prior observations of the same ``tool_name`` (falling back to global history),
with no hardcoded tool classes.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable

from trace_collect.causal_history import iter_causal_latency_observations
from trace_collect.kv_profile_sweep import KVSwapProfileEntry
from trace_collect.latency_outputs import write_summary_outputs
from trace_collect.latency_validation import normalized_positive_floats
from trace_collect.tool_latency_dataset import read_tool_latency_jsonl


@dataclass(frozen=True)
class BucketDecision:
    """One causal bucket prediction for one tool latency row."""

    sample_id: str
    tool_name: str
    tool_ts_start: float
    latency_ms: float
    label_bucket: int
    predicted_bucket: int | None
    probability_by_bucket: tuple[float, ...] | None
    prediction_source: str
    history_count: int

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "tool_name": self.tool_name,
            "tool_ts_start": self.tool_ts_start,
            "latency_ms": self.latency_ms,
            "label_bucket": self.label_bucket,
            "predicted_bucket": self.predicted_bucket,
            "probability_by_bucket": (
                list(self.probability_by_bucket)
                if self.probability_by_bucket is not None
                else None
            ),
            "prediction_source": self.prediction_source,
            "history_count": self.history_count,
        }


def bucket_edges_from_profile(
    entries: Iterable[KVSwapProfileEntry],
    *,
    quantile: str,
    guard_ms: float = 0.0,
) -> list[float]:
    """Derive bucket edges from profiled KV swap costs plus guard time.

    Each profile entry contributes one edge (its quantile cost + guard), so
    the bucket structure follows the profiled KV sizes instead of any
    hand-picked grid. Duplicate edges collapse.
    """

    if not math.isfinite(guard_ms) or guard_ms < 0.0:
        raise ValueError(f"guard_ms must be finite and non-negative, got {guard_ms}")
    edges = sorted({entry.quantile_ms(quantile) + guard_ms for entry in entries})
    if not edges:
        raise ValueError("no profile entries to derive bucket edges from")
    # Normalized again in evaluate_latency_buckets; validating here too surfaces
    # degenerate profiles (e.g. zero-cost quantile with zero guard) at derivation.
    return _normalize_bucket_edges(edges)


def latency_bucket(latency_ms: float, bucket_edges_ms: list[float]) -> int:
    """Bucket index of a latency: number of edges it strictly exceeds."""

    return bisect_left(bucket_edges_ms, latency_ms)


def evaluate_latency_buckets(
    rows: Iterable[dict[str, Any]],
    *,
    bucket_edges_ms: Iterable[float],
    min_tool_history: int = 1,
) -> dict[str, Any]:
    """Evaluate causal empirical bucket predictions for tool latency rows.

    For each row, the predicted bucket is the mode of the empirical bucket
    distribution over causally prior completed observations (same tool first,
    global fallback). Ties resolve to the lowest bucket — the conservative
    assumption for swap scheduling, since underestimating the available window
    only forgoes a swap rather than stalling the next LLM call.
    """

    edges = _normalize_bucket_edges(bucket_edges_ms)
    bucket_count = len(edges) + 1

    decisions: list[BucketDecision] = []
    for observation in iter_causal_latency_observations(
        rows,
        min_tool_history=min_tool_history,
    ):
        history = observation.history
        label = latency_bucket(observation.latency_ms, edges)
        if history:
            counts = [0] * bucket_count
            for value in history:
                counts[latency_bucket(value, edges)] += 1
            probabilities = tuple(count / len(history) for count in counts)
            predicted = max(range(bucket_count), key=lambda i: (probabilities[i], -i))
            history_count = len(history)
        else:
            probabilities = None
            predicted = None
            history_count = 0
        decisions.append(
            BucketDecision(
                sample_id=observation.sample_id,
                tool_name=observation.tool_name,
                tool_ts_start=observation.tool_ts_start,
                latency_ms=observation.latency_ms,
                label_bucket=label,
                predicted_bucket=predicted,
                probability_by_bucket=probabilities,
                prediction_source=observation.prediction_source,
                history_count=history_count,
            )
        )

    return {
        "predictor": "causal_empirical_bucket",
        "bucket_edges_ms": edges,
        "bucket_count": bucket_count,
        "min_tool_history": min_tool_history,
        "row_count": len(decisions),
        "metrics": _bucket_metrics(decisions, bucket_count),
        "metrics_by_tool": {
            tool_name: _bucket_metrics(tool_decisions, bucket_count)
            for tool_name, tool_decisions in sorted(_group_by_tool(decisions).items())
        },
        "decisions": [decision.to_json_obj() for decision in decisions],
    }


def load_and_evaluate_latency_buckets(
    path: Path,
    *,
    bucket_edges_ms: Iterable[float],
    min_tool_history: int = 1,
) -> dict[str, Any]:
    return evaluate_latency_buckets(
        read_tool_latency_jsonl(path),
        bucket_edges_ms=bucket_edges_ms,
        min_tool_history=min_tool_history,
    )


def write_bucket_outputs(
    summary: dict[str, Any],
    *,
    summary_path: Path | None = None,
    decisions_path: Path | None = None,
) -> None:
    write_summary_outputs(
        summary,
        detail_key="decisions",
        summary_path=summary_path,
        detail_path=decisions_path,
    )


def _group_by_tool(decisions: list[BucketDecision]) -> dict[str, list[BucketDecision]]:
    grouped: dict[str, list[BucketDecision]] = {}
    for decision in decisions:
        grouped.setdefault(decision.tool_name, []).append(decision)
    return grouped


def _bucket_metrics(decisions: list[BucketDecision], bucket_count: int) -> dict[str, Any]:
    label_counts = [0] * bucket_count
    for decision in decisions:
        label_counts[decision.label_bucket] += 1
    evaluated = [d for d in decisions if d.predicted_bucket is not None]
    if not evaluated:
        return {
            "row_count": len(decisions),
            "evaluated_count": 0,
            "cold_start_count": len(decisions),
            "label_counts": label_counts,
            "accuracy": None,
            "mean_abs_bucket_error": None,
            "underestimate_rate": None,
            "overestimate_rate": None,
            "confusion": None,
        }
    correct = sum(d.predicted_bucket == d.label_bucket for d in evaluated)
    abs_errors = [abs(d.predicted_bucket - d.label_bucket) for d in evaluated]
    underestimates = sum(d.predicted_bucket < d.label_bucket for d in evaluated)
    overestimates = sum(d.predicted_bucket > d.label_bucket for d in evaluated)
    confusion = [[0] * bucket_count for _ in range(bucket_count)]
    for decision in evaluated:
        confusion[decision.label_bucket][decision.predicted_bucket] += 1
    return {
        "row_count": len(decisions),
        "evaluated_count": len(evaluated),
        "cold_start_count": len(decisions) - len(evaluated),
        "label_counts": label_counts,
        "accuracy": correct / len(evaluated),
        "mean_abs_bucket_error": sum(abs_errors) / len(evaluated),
        "underestimate_rate": underestimates / len(evaluated),
        "overestimate_rate": overestimates / len(evaluated),
        "confusion": confusion,
    }


def _normalize_bucket_edges(values: Iterable[float]) -> list[float]:
    return normalized_positive_floats(values, label="bucket edge")


__all__ = [
    "BucketDecision",
    "bucket_edges_from_profile",
    "evaluate_latency_buckets",
    "latency_bucket",
    "load_and_evaluate_latency_buckets",
    "write_bucket_outputs",
]
