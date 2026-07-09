"""Profiled-prior threshold decisions for tool latency on held-out traces.

Implements the tool side of the decision model ``D(t, s | a, P)``: the
action-side threshold ``T`` (KV swap cost plus guard, monotone in the
sequence state ``s``) is supplied by the caller, while this module
estimates the survival probability P(latency > T) for the current tool
action from a profiling prior ``P`` built on a disjoint, task-level split
of traces, optionally combined with causal online history from the
evaluated stream. Three predictors share one output schema so they can be
compared directly:

* ``prior_only``: empirical survival from the profile split alone
  (per-tool, falling back to the profile's global distribution).
* ``online_only``: causal within-stream history, mirroring
  tool_latency_threshold's fallback semantics.
* ``blended``: pseudo-count blend. The prior contributes
  ``prior_strength`` pseudo-observations (default: its true sample size,
  i.e. plain pooling) alongside the walk-selected online history, so cold
  starts vanish and online evidence dominates as it accrues. Each side
  keeps its own tool-to-global fallback hierarchy.

An optional Wilson-score abstain band turns the point estimate into a
selective decision: predict "exceeds" only when the lower confidence
bound clears the cutoff, "does not exceed" only when the upper bound
misses it, and abstain in between (policy-wise a conservative no-swap,
but reported separately). The band width shrinks with evidence, so
abstention concentrates on thinly observed tools.

Profile and eval rows must come from disjoint ``source_trace`` sets; any
overlap is rejected to prevent leakage of eval traces into the prior.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Iterable

from trace_collect.causal_history import iter_causal_latency_observations
from trace_collect.classification_metrics import binary_classification_metrics, safe_div
from trace_collect.command_features import make_row_command_key
from trace_collect.latency_outputs import write_summary_outputs
from trace_collect.latency_validation import (
    normalized_positive_floats,
    required_nonnegative_float,
    required_text,
)
from trace_collect.tool_latency_dataset import read_tool_latency_jsonl

_PREDICTORS = ("prior_only", "online_only", "blended")


@dataclass(frozen=True)
class LatencyPrior:
    """Per-group, per-tool, and global latency samples from the profile split."""

    values_by_tool: dict[str, list[float]]  # each sorted ascending
    global_values: list[float]  # sorted ascending
    source_traces: frozenset[str]
    values_by_group: dict[str, list[float]] | None = None  # each sorted ascending


@dataclass(frozen=True)
class ProfiledThresholdDecision:
    """One profiled threshold decision for one held-out tool latency row."""

    sample_id: str
    tool_name: str
    tool_ts_start: float
    latency_ms: float
    threshold_ms: float
    label_exceeds_threshold: bool
    predicted_exceeds_threshold: bool | None
    abstained: bool
    probability_exceeds_threshold: float | None
    ci_low: float | None
    ci_high: float | None
    prior_source: str | None
    online_source: str | None
    prior_count: int | None
    online_count: int | None
    effective_count: float | None
    group_key: str | None = None

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "tool_name": self.tool_name,
            "tool_ts_start": self.tool_ts_start,
            "latency_ms": self.latency_ms,
            "threshold_ms": self.threshold_ms,
            "label_exceeds_threshold": self.label_exceeds_threshold,
            "predicted_exceeds_threshold": self.predicted_exceeds_threshold,
            "abstained": self.abstained,
            "probability_exceeds_threshold": self.probability_exceeds_threshold,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "prior_source": self.prior_source,
            "online_source": self.online_source,
            "prior_count": self.prior_count,
            "online_count": self.online_count,
            "effective_count": self.effective_count,
            "group_key": self.group_key,
        }


def build_latency_prior(
    rows: Iterable[dict[str, Any]],
    *,
    row_group_key: Callable[[dict[str, Any]], str | None] | None = None,
) -> LatencyPrior:
    """Aggregate profile-split rows into group/tool/global latency samples."""

    values_by_tool: dict[str, list[float]] = {}
    values_by_group: dict[str, list[float]] | None = (
        {} if row_group_key is not None else None
    )
    global_values: list[float] = []
    source_traces: set[str] = set()
    for index, row in enumerate(rows):
        source = f"profile row {index}"
        tool_name = required_text(row, "tool_name", source=source)
        latency_ms = required_nonnegative_float(row, "latency_ms", source=source)
        source_traces.add(required_text(row, "source_trace", source=source))
        values_by_tool.setdefault(tool_name, []).append(latency_ms)
        global_values.append(latency_ms)
        if values_by_group is not None:
            group_key = row_group_key(row)
            if group_key is not None:
                values_by_group.setdefault(group_key, []).append(latency_ms)
    if not global_values:
        raise ValueError("empty latency prior: no profile rows supplied")
    for values in values_by_tool.values():
        values.sort()
    if values_by_group is not None:
        for values in values_by_group.values():
            values.sort()
    global_values.sort()
    return LatencyPrior(
        values_by_tool=values_by_tool,
        global_values=global_values,
        source_traces=frozenset(source_traces),
        values_by_group=values_by_group,
    )


def evaluate_profiled_latency_thresholds(
    eval_rows: Iterable[dict[str, Any]],
    *,
    profile_rows: Iterable[dict[str, Any]],
    thresholds_ms: Iterable[float],
    predictor: str,
    prior_strength: float | None = None,
    probability_cutoff: float = 0.5,
    abstain_confidence: float | None = None,
    min_tool_history: int = 1,
    command_field: str | None = None,
) -> dict[str, Any]:
    """Evaluate one predictor's threshold decisions on held-out eval rows.

    ``prior_strength`` (blended only) is the pseudo-observation weight of the
    prior; ``None`` pools raw counts. ``abstain_confidence`` enables the
    Wilson abstain band at that confidence level; ``None`` keeps plain
    point-estimate decisions. ``min_tool_history`` gates the group/tool vs
    global fallback on both the prior and online sides symmetrically.
    ``command_field`` enables data-derived command grouping: rows whose
    ``tool_args`` carry a shell command under that field are keyed by their
    command heads (see command_features), refining both prior and online
    history to group -> tool -> global.
    """

    if predictor not in _PREDICTORS:
        raise ValueError(
            f"unknown predictor {predictor!r}; choose one of {', '.join(_PREDICTORS)}"
        )
    thresholds = normalized_positive_floats(thresholds_ms, label="threshold")
    if not math.isfinite(probability_cutoff) or not 0.0 <= probability_cutoff <= 1.0:
        raise ValueError(
            "probability_cutoff must be finite and in [0, 1], "
            f"got {probability_cutoff}"
        )
    if prior_strength is not None and (
        not math.isfinite(prior_strength) or prior_strength <= 0.0
    ):
        raise ValueError(
            f"prior_strength must be finite and positive, got {prior_strength}"
        )
    z_score: float | None = None
    if abstain_confidence is not None:
        if not math.isfinite(abstain_confidence) or not 0.0 < abstain_confidence < 1.0:
            raise ValueError(
                "abstain_confidence must be finite and in (0, 1), "
                f"got {abstain_confidence}"
            )
        z_score = NormalDist().inv_cdf((1.0 + abstain_confidence) / 2.0)

    row_group_key = (
        make_row_command_key(command_field) if command_field is not None else None
    )
    prior = build_latency_prior(profile_rows, row_group_key=row_group_key)
    eval_list = list(eval_rows)
    if not eval_list:
        raise ValueError("no latency rows supplied")
    eval_traces = {
        required_text(row, "source_trace", source=f"eval row {index}")
        for index, row in enumerate(eval_list)
    }
    overlap = sorted(eval_traces & prior.source_traces)
    if overlap:
        raise ValueError(
            "profile and eval rows must come from disjoint traces; "
            f"shared source_trace values: {overlap}"
        )

    decisions: list[ProfiledThresholdDecision] = []
    row_count = 0
    for observation in iter_causal_latency_observations(
        eval_list,
        min_tool_history=min_tool_history,
        row_group_key=row_group_key,
    ):
        row_count += 1
        prior_values, prior_source = _select_prior(
            prior,
            observation.tool_name,
            observation.group_key,
            min_tool_history=min_tool_history,
        )
        online_history = observation.history
        for threshold_ms in thresholds:
            estimate = _estimate_survival(
                predictor,
                threshold_ms=threshold_ms,
                prior_values=prior_values,
                prior_source=prior_source,
                prior_strength=prior_strength,
                online_history=online_history,
                online_source=observation.prediction_source,
            )
            predicted, abstained, ci_low, ci_high = _decide(
                estimate["probability"],
                estimate["effective_count"],
                probability_cutoff=probability_cutoff,
                z_score=z_score,
            )
            decisions.append(
                ProfiledThresholdDecision(
                    sample_id=observation.sample_id,
                    tool_name=observation.tool_name,
                    tool_ts_start=observation.tool_ts_start,
                    latency_ms=observation.latency_ms,
                    threshold_ms=threshold_ms,
                    label_exceeds_threshold=observation.latency_ms > threshold_ms,
                    predicted_exceeds_threshold=predicted,
                    abstained=abstained,
                    probability_exceeds_threshold=estimate["probability"],
                    ci_low=ci_low,
                    ci_high=ci_high,
                    prior_source=estimate["prior_source"],
                    online_source=estimate["online_source"],
                    prior_count=estimate["prior_count"],
                    online_count=estimate["online_count"],
                    effective_count=estimate["effective_count"],
                    group_key=observation.group_key,
                )
            )

    return {
        "predictor": predictor,
        "prior_strength": prior_strength,
        "probability_cutoff": probability_cutoff,
        "abstain_confidence": abstain_confidence,
        "min_tool_history": min_tool_history,
        "command_field": command_field,
        "thresholds_ms": thresholds,
        "profile_row_count": len(prior.global_values),
        "profile_trace_count": len(prior.source_traces),
        "profile_tool_count": len(prior.values_by_tool),
        "profile_group_count": (
            len(prior.values_by_group) if prior.values_by_group is not None else None
        ),
        "row_count": row_count,
        "decision_count": len(decisions),
        "metrics_by_threshold": _metrics_by_threshold(decisions),
        "metrics_by_tool_threshold": _metrics_by_tool_threshold(decisions),
        "decisions": [decision.to_json_obj() for decision in decisions],
    }


def load_and_evaluate_profiled_latency_thresholds(
    eval_path: Path,
    *,
    profile_path: Path,
    thresholds_ms: Iterable[float],
    predictor: str,
    prior_strength: float | None = None,
    probability_cutoff: float = 0.5,
    abstain_confidence: float | None = None,
    min_tool_history: int = 1,
    command_field: str | None = None,
) -> dict[str, Any]:
    return evaluate_profiled_latency_thresholds(
        read_tool_latency_jsonl(eval_path),
        profile_rows=read_tool_latency_jsonl(profile_path),
        thresholds_ms=thresholds_ms,
        predictor=predictor,
        prior_strength=prior_strength,
        probability_cutoff=probability_cutoff,
        abstain_confidence=abstain_confidence,
        min_tool_history=min_tool_history,
        command_field=command_field,
    )


def write_profiled_outputs(
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


def _select_prior(
    prior: LatencyPrior,
    tool_name: str,
    group_key: str | None,
    *,
    min_tool_history: int,
) -> tuple[list[float], str]:
    if group_key is not None and prior.values_by_group is not None:
        group_values = prior.values_by_group.get(group_key, [])
        if len(group_values) >= min_tool_history:
            return group_values, "prior_group"
    tool_values = prior.values_by_tool.get(tool_name, [])
    if len(tool_values) >= min_tool_history:
        return tool_values, "prior_tool"
    return prior.global_values, "prior_global"


def _estimate_survival(
    predictor: str,
    *,
    threshold_ms: float,
    prior_values: list[float],
    prior_source: str,
    prior_strength: float | None,
    online_history: list[float],
    online_source: str,
) -> dict[str, Any]:
    prior_count = len(prior_values)
    prior_exceed = prior_count - bisect_right(prior_values, threshold_ms)
    online_count = len(online_history)
    online_exceed = sum(value > threshold_ms for value in online_history)

    if predictor == "prior_only":
        return {
            "probability": prior_exceed / prior_count,
            "effective_count": float(prior_count),
            "prior_source": prior_source,
            "online_source": None,
            "prior_count": prior_count,
            "online_count": None,
        }
    if predictor == "online_only":
        if online_count == 0:
            probability = None
            effective_count = None
        else:
            probability = online_exceed / online_count
            effective_count = float(online_count)
        return {
            "probability": probability,
            "effective_count": effective_count,
            "prior_source": None,
            "online_source": online_source,
            "prior_count": None,
            "online_count": online_count,
        }
    prior_weight = prior_strength if prior_strength is not None else float(prior_count)
    prior_rate = prior_exceed / prior_count
    effective_count = prior_weight + online_count
    probability = (prior_weight * prior_rate + online_exceed) / effective_count
    return {
        "probability": probability,
        "effective_count": effective_count,
        "prior_source": prior_source,
        "online_source": online_source,
        "prior_count": prior_count,
        "online_count": online_count,
    }


def _decide(
    probability: float | None,
    effective_count: float | None,
    *,
    probability_cutoff: float,
    z_score: float | None,
) -> tuple[bool | None, bool, float | None, float | None]:
    if probability is None:
        return None, False, None, None
    if z_score is None:
        return probability >= probability_cutoff, False, None, None
    ci_low, ci_high = _wilson_interval(probability, effective_count, z_score)
    if ci_low >= probability_cutoff:
        return True, False, ci_low, ci_high
    if ci_high < probability_cutoff:
        return False, False, ci_low, ci_high
    return None, True, ci_low, ci_high


def _wilson_interval(p_hat: float, n: float, z: float) -> tuple[float, float]:
    """Wilson score interval; ``n`` may be a fractional pseudo-count total."""

    denominator = 1.0 + z * z / n
    center = (p_hat + z * z / (2.0 * n)) / denominator
    half_width = (
        z * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n)) / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _metrics_by_threshold(
    decisions: list[ProfiledThresholdDecision],
) -> dict[str, dict[str, float | int | None]]:
    grouped: dict[float, list[ProfiledThresholdDecision]] = {}
    for decision in decisions:
        grouped.setdefault(decision.threshold_ms, []).append(decision)
    return {
        str(threshold): _selective_metrics(rows)
        for threshold, rows in sorted(grouped.items())
    }


def _metrics_by_tool_threshold(
    decisions: list[ProfiledThresholdDecision],
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    grouped: dict[str, dict[float, list[ProfiledThresholdDecision]]] = {}
    for decision in decisions:
        grouped.setdefault(decision.tool_name, {}).setdefault(
            decision.threshold_ms,
            [],
        ).append(decision)
    return {
        tool_name: {
            str(threshold): _selective_metrics(rows)
            for threshold, rows in sorted(threshold_groups.items())
        }
        for tool_name, threshold_groups in sorted(grouped.items())
    }


def _selective_metrics(
    decisions: list[ProfiledThresholdDecision],
) -> dict[str, float | int | None]:
    decided = [d for d in decisions if d.predicted_exceeds_threshold is not None]
    abstain_count = sum(d.abstained for d in decisions)
    cold_start_count = sum(
        d.predicted_exceeds_threshold is None and not d.abstained for d in decisions
    )
    core = binary_classification_metrics(
        (d.label_exceeds_threshold, d.predicted_exceeds_threshold) for d in decided
    )
    return {
        "row_count": len(decisions),
        "decided_count": len(decided),
        "abstain_count": abstain_count,
        "cold_start_count": cold_start_count,
        "positive_count": sum(d.label_exceeds_threshold for d in decisions),
        "abstain_rate": safe_div(abstain_count, len(decisions) - cold_start_count),
        "predicted_positive_rate": safe_div(
            sum(bool(d.predicted_exceeds_threshold) for d in decided),
            len(decided),
        ),
        "accuracy": core["accuracy"],
        "precision": core["precision"],
        "recall": core["recall"],
        "false_positive_rate": core["false_positive_rate"],
        "false_negative_rate": core["false_negative_rate"],
    }


__all__ = [
    "LatencyPrior",
    "ProfiledThresholdDecision",
    "build_latency_prior",
    "evaluate_profiled_latency_thresholds",
    "load_and_evaluate_profiled_latency_thresholds",
    "write_profiled_outputs",
]
