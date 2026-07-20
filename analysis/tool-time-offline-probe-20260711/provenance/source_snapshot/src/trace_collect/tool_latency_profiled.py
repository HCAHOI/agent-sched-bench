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
trace or logical-task overlap is rejected to prevent leakage of eval tasks
into the prior. Prior nodes can additionally require support from multiple
logical tasks. The default pooled-call ECDF preserves the system's per-call
estimand; an explicit task-balanced ECDF is available as a robustness arm.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from trace_collect.causal_history import iter_causal_latency_observations
from trace_collect.classification_metrics import binary_classification_metrics, safe_div
from trace_collect.command_features import (
    make_row_command_prefix_keys,
    segment_prefix_keys,
)
from trace_collect.segment_cost_model import SegmentCostModel, fit_segment_cost_model
from trace_collect.latency_outputs import write_summary_outputs
from trace_collect.latency_validation import (
    normalized_positive_floats,
    required_nonnegative_float,
    required_text,
)
from trace_collect.tool_latency_dataset import read_tool_latency_jsonl

_PREDICTORS = ("prior_only", "online_only", "blended")
_PRIOR_AGGREGATIONS = ("call", "task")


@dataclass(frozen=True)
class LatencyPrior:
    """Per-group, per-tool, and global latency samples from the profile split."""

    values_by_tool: dict[str, list[float]]  # each sorted ascending
    global_values: list[float]  # sorted ascending
    source_traces: frozenset[str]
    task_ids: frozenset[str]
    values_by_task: dict[str, list[float]]  # task_id -> sorted values
    values_by_task_by_tool: dict[str, dict[str, list[float]]]
    values_by_group: dict[str, list[float]] | None = None  # each sorted ascending
    values_by_task_by_group: dict[str, dict[str, list[float]]] | None = None


@dataclass(frozen=True)
class LatencyPriorNode:
    """One eligible level in the global -> tool -> prefix prior hierarchy."""

    values: list[float]
    values_by_task: dict[str, list[float]]
    source: str
    group_key: str | None


@dataclass(frozen=True)
class ProfiledThresholdDecision:
    """One profiled threshold decision for one held-out tool latency row."""

    sample_id: str
    task_id: str
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
    prior_task_count: int | None
    online_count: int | None
    effective_count: float | None
    prior_group_key: str | None = None
    online_group_key: str | None = None
    threshold_deduction_ms: float | None = None
    hazard_recheck_ms: float | None = None

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "task_id": self.task_id,
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
            "prior_task_count": self.prior_task_count,
            "online_count": self.online_count,
            "effective_count": self.effective_count,
            "prior_group_key": self.prior_group_key,
            "online_group_key": self.online_group_key,
            "threshold_deduction_ms": self.threshold_deduction_ms,
            "hazard_recheck_ms": self.hazard_recheck_ms,
        }


def build_latency_prior(
    rows: Iterable[dict[str, Any]],
    *,
    row_group_keys: Callable[[dict[str, Any]], tuple[str, ...]] | None = None,
    row_group_value: Callable[[dict[str, Any]], float] | None = None,
) -> LatencyPrior:
    """Aggregate profile-split rows into prefix-node/tool/global latency samples.

    ``row_group_value`` optionally maps a row to the value stored in its
    group nodes (e.g. a segment-attributed time); tool and global samples
    always store the raw latency.
    """

    values_by_tool: dict[str, list[float]] = {}
    values_by_task: dict[str, list[float]] = {}
    values_by_task_by_tool: dict[str, dict[str, list[float]]] = {}
    values_by_group: dict[str, list[float]] | None = (
        {} if row_group_keys is not None else None
    )
    values_by_task_by_group: dict[str, dict[str, list[float]]] | None = (
        {} if row_group_keys is not None else None
    )
    global_values: list[float] = []
    source_traces: set[str] = set()
    task_ids: set[str] = set()
    task_id_by_source: dict[str, str] = {}
    for index, row in enumerate(rows):
        source = f"profile row {index}"
        tool_name = required_text(row, "tool_name", source=source)
        latency_ms = required_nonnegative_float(row, "latency_ms", source=source)
        source_trace = required_text(row, "source_trace", source=source)
        task_id = _row_task_id(row, source=source)
        _record_source_task(
            task_id_by_source,
            source_trace=source_trace,
            task_id=task_id,
            source=source,
        )
        source_traces.add(source_trace)
        task_ids.add(task_id)
        values_by_tool.setdefault(tool_name, []).append(latency_ms)
        values_by_task.setdefault(task_id, []).append(latency_ms)
        values_by_task_by_tool.setdefault(tool_name, {}).setdefault(
            task_id,
            [],
        ).append(latency_ms)
        global_values.append(latency_ms)
        if values_by_group is not None:
            group_value = (
                row_group_value(row) if row_group_value is not None else latency_ms
            )
            for group_key in row_group_keys(row):
                values_by_group.setdefault(group_key, []).append(group_value)
                assert values_by_task_by_group is not None
                values_by_task_by_group.setdefault(group_key, {}).setdefault(
                    task_id,
                    [],
                ).append(group_value)
    if not global_values:
        raise ValueError("empty latency prior: no profile rows supplied")
    for values in values_by_tool.values():
        values.sort()
    for values in values_by_task.values():
        values.sort()
    for task_values in values_by_task_by_tool.values():
        for values in task_values.values():
            values.sort()
    if values_by_group is not None:
        for values in values_by_group.values():
            values.sort()
        assert values_by_task_by_group is not None
        for task_values in values_by_task_by_group.values():
            for values in task_values.values():
                values.sort()
    global_values.sort()
    return LatencyPrior(
        values_by_tool=values_by_tool,
        global_values=global_values,
        source_traces=frozenset(source_traces),
        task_ids=frozenset(task_ids),
        values_by_task=values_by_task,
        values_by_task_by_tool=values_by_task_by_tool,
        values_by_group=values_by_group,
        values_by_task_by_group=values_by_task_by_group,
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
    min_profile_tasks: int = 1,
    prior_aggregation: str = "call",
    command_field: str | None = None,
    max_prefix_depth: int = 4,
    skip_leading_cd: bool = False,
    segment_costs: bool = False,
    segment_fit: str = "nnls",
    hazard_kv_by_threshold: Mapping[float, float] | None = None,
) -> dict[str, Any]:
    """Evaluate one predictor's threshold decisions on held-out eval rows.

    ``prior_strength`` (blended only) is the pseudo-observation weight of the
    prior; ``None`` pools raw counts. ``abstain_confidence`` enables the
    Wilson abstain band at that confidence level; ``None`` keeps plain
    point-estimate decisions. ``min_tool_history`` gates every backoff level
    by row count on both the prior and online sides symmetrically.
    ``min_profile_tasks`` independently requires distinct logical-task support
    before a prior group/tool node is eligible; global fallback must also meet
    it. ``prior_aggregation=call`` estimates survival for a random profiled
    call. ``prior_aggregation=task`` averages within-task ECDF survival rates,
    making repeated calls within one task unable to dominate; this robustness
    arm is restricted to ``prior_only`` without Wilson abstention or hazard
    rechecks because those paths use raw-call evidence units. ``command_field``
    enables data-derived command prefix-tree grouping: rows whose
    ``tool_args`` carry a shell command under that field feed nested prefix
    nodes up to ``max_prefix_depth`` tokens (see command_features), and
    predictions use the deepest adequately observed node before backing off
    to tool then global. ``max_prefix_depth`` bounds node cardinality; the
    default of 4 covers program, subcommand, and primary flags, and deeper
    distinctions rarely accumulate ``min_tool_history`` samples at trace
    scale - it is configurable, not tuned to any dataset.
    ``skip_leading_cd`` applies command_features' leading-``cd`` segment
    stripping before keying, so the depth budget indexes the workload
    rather than the working directory.
    ``hazard_kv_by_threshold`` (policy support for swap_deadline_policy)
    maps each threshold to its KV cost; when provided, every decision also
    carries ``hazard_recheck_ms`` (see hazard_recheck_ms), computed from
    the selected prior node's samples (or, for ``online_only``, the causal
    online history - whose growth also defeats the per-node cache, an
    O(n) recompute per row that is honest but slow at scale). For
    ``prior_only`` the t0 decision is node-level, so optimizing over the
    full node distribution is exactly the optimum for t0-declined calls;
    for ``online_only``/``blended`` it is a parameter-free heuristic. Not
    supported together with ``segment_costs``, whose attributed group
    values answer deducted-budget questions rather than raw elapsed-time
    ones.
    ``segment_costs`` (requires ``command_field``) fits an additive
    per-segment cost model on the profile split (see segment_cost_model);
    each command is then keyed by its dominant segment's prefix chain, group
    nodes store segment-attributed times (latency minus the other segments'
    fitted costs), and group-node survival is queried at the deducted
    threshold. Tool/global backoff levels keep raw latencies and the raw
    threshold. Labels always use the raw latency and threshold.
    """

    if predictor not in _PREDICTORS:
        raise ValueError(
            f"unknown predictor {predictor!r}; choose one of {', '.join(_PREDICTORS)}"
        )
    if prior_aggregation not in _PRIOR_AGGREGATIONS:
        raise ValueError(
            f"unknown prior_aggregation {prior_aggregation!r}; choose one of "
            f"{', '.join(_PRIOR_AGGREGATIONS)}"
        )
    if min_tool_history < 1:
        raise ValueError(f"min_tool_history must be >= 1, got {min_tool_history}")
    if min_profile_tasks < 1:
        raise ValueError(f"min_profile_tasks must be >= 1, got {min_profile_tasks}")
    if prior_aggregation == "task" and predictor != "prior_only":
        raise ValueError("task prior aggregation is supported only for prior_only")
    if prior_aggregation == "task" and abstain_confidence is not None:
        raise ValueError(
            "task prior aggregation is not supported with Wilson abstention"
        )
    if prior_aggregation == "task" and hazard_kv_by_threshold is not None:
        raise ValueError("task prior aggregation is not supported with hazard recheck")
    thresholds = normalized_positive_floats(thresholds_ms, label="threshold")
    if not math.isfinite(probability_cutoff) or not 0.0 <= probability_cutoff <= 1.0:
        raise ValueError(
            f"probability_cutoff must be finite and in [0, 1], got {probability_cutoff}"
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

    if segment_costs and command_field is None:
        raise ValueError("segment_costs requires command_field")
    if segment_costs and skip_leading_cd:
        raise ValueError(
            "segment_costs and skip_leading_cd are alternative preamble "
            "treatments; enable only one"
        )
    if hazard_kv_by_threshold is not None and segment_costs:
        raise ValueError("hazard recheck is not supported with segment_costs")

    row_group_keys: Callable[[dict[str, Any]], tuple[str, ...]] | None = None
    row_group_value: Callable[[dict[str, Any]], float] | None = None
    row_deduction: Callable[[dict[str, Any]], float] | None = None
    segment_model: SegmentCostModel | None = None
    profile_list = list(profile_rows)
    if command_field is not None:
        row_group_keys = make_row_command_prefix_keys(
            command_field,
            max_depth=max_prefix_depth,
            skip_leading_cd=skip_leading_cd,
        )
        if segment_costs:
            segment_model = fit_segment_cost_model(
                profile_list,
                command_field=command_field,
                fit_method=segment_fit,
            )
            row_group_keys, row_group_value, row_deduction = _make_segment_keying(
                segment_model,
                command_field,
                fallback_keys=row_group_keys,
                max_prefix_depth=max_prefix_depth,
                min_tool_history=min_tool_history,
            )
    prior = build_latency_prior(
        profile_list,
        row_group_keys=row_group_keys,
        row_group_value=row_group_value,
    )
    if len(prior.task_ids) < min_profile_tasks:
        raise ValueError(
            "latency prior has fewer logical tasks than min_profile_tasks: "
            f"{len(prior.task_ids)} < {min_profile_tasks}"
        )
    eval_list = list(eval_rows)
    deduction_by_sample_id: dict[str, float] = {}
    task_id_by_sample_id = validate_profile_eval_disjoint(eval_list, prior=prior)
    if row_deduction is not None:
        for index, row in enumerate(eval_list):
            sample_id = required_text(row, "sample_id", source=f"eval row {index}")
            deduction = row_deduction(row)
            existing = deduction_by_sample_id.get(sample_id)
            if existing is not None and existing != deduction:
                raise ValueError(
                    f"duplicate sample_id {sample_id!r} with conflicting deductions"
                )
            deduction_by_sample_id[sample_id] = deduction
    decisions: list[ProfiledThresholdDecision] = []
    hazard_cache: dict[tuple[int, int, float], float] = {}
    row_count = 0
    for observation in iter_causal_latency_observations(
        eval_list,
        min_tool_history=min_tool_history,
        row_group_keys=row_group_keys,
        row_group_value=row_group_value,
    ):
        row_count += 1
        prior_values, prior_values_by_task, prior_source, prior_group_key = (
            _select_prior(
                prior,
                observation.tool_name,
                observation.group_keys,
                min_tool_history=min_tool_history,
                min_profile_tasks=min_profile_tasks,
            )
        )
        online_history = observation.history
        deduction = (
            deduction_by_sample_id.get(observation.sample_id, 0.0)
            if row_deduction is not None
            else None
        )
        for threshold_ms in thresholds:
            # Group nodes hold segment-attributed values, so they answer the
            # deducted-budget question; tool/global samples stay raw.
            deducted_ms = (
                max(0.0, threshold_ms - deduction) if deduction is not None else None
            )
            prior_query_ms = (
                deducted_ms
                if deducted_ms is not None and prior_source == "prior_group"
                else threshold_ms
            )
            online_query_ms = (
                deducted_ms
                if deducted_ms is not None
                and observation.prediction_source == "group_history"
                else threshold_ms
            )
            estimate = _estimate_survival(
                predictor,
                prior_threshold_ms=prior_query_ms,
                online_threshold_ms=online_query_ms,
                prior_values=prior_values,
                prior_values_by_task=prior_values_by_task,
                prior_source=prior_source,
                prior_strength=prior_strength,
                prior_aggregation=prior_aggregation,
                online_history=online_history,
                online_source=observation.prediction_source,
            )
            hazard_ms: float | None = None
            if hazard_kv_by_threshold is not None:
                kv_cost_ms = hazard_kv_by_threshold.get(threshold_ms)
                if kv_cost_ms is None:
                    raise ValueError(
                        f"hazard_kv_by_threshold is missing threshold {threshold_ms}"
                    )
                hazard_values = (
                    online_history if predictor == "online_only" else prior_values
                )
                cache_key = (id(hazard_values), len(hazard_values), threshold_ms)
                hazard_ms = hazard_cache.get(cache_key)
                if hazard_ms is None:
                    hazard_ms = hazard_recheck_ms(
                        hazard_values,
                        threshold_ms=threshold_ms,
                        kv_cost_ms=kv_cost_ms,
                    )
                    hazard_cache[cache_key] = hazard_ms
            predicted, abstained, ci_low, ci_high = _decide(
                estimate["probability"],
                estimate["effective_count"],
                probability_cutoff=probability_cutoff,
                z_score=z_score,
            )
            decisions.append(
                ProfiledThresholdDecision(
                    sample_id=observation.sample_id,
                    task_id=task_id_by_sample_id[observation.sample_id],
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
                    prior_task_count=estimate["prior_task_count"],
                    online_count=estimate["online_count"],
                    effective_count=estimate["effective_count"],
                    prior_group_key=(
                        prior_group_key
                        if estimate["prior_source"] is not None
                        else None
                    ),
                    online_group_key=(
                        observation.group_key
                        if estimate["online_source"] == "group_history"
                        else None
                    ),
                    threshold_deduction_ms=deduction,
                    hazard_recheck_ms=hazard_ms,
                )
            )

    return {
        "predictor": predictor,
        "prior_strength": prior_strength,
        "probability_cutoff": probability_cutoff,
        "abstain_confidence": abstain_confidence,
        "min_tool_history": min_tool_history,
        "min_profile_tasks": min_profile_tasks,
        "prior_aggregation": prior_aggregation,
        "command_field": command_field,
        "max_prefix_depth": max_prefix_depth if command_field is not None else None,
        "skip_leading_cd": skip_leading_cd if command_field is not None else None,
        "segment_costs": segment_costs,
        "segment_fit": segment_fit if segment_costs else None,
        "segment_cost_model": (
            segment_model.to_json_obj() if segment_model is not None else None
        ),
        "thresholds_ms": thresholds,
        "profile_row_count": len(prior.global_values),
        "profile_trace_count": len(prior.source_traces),
        "profile_task_count": len(prior.task_ids),
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
    min_profile_tasks: int = 1,
    prior_aggregation: str = "call",
    command_field: str | None = None,
    max_prefix_depth: int = 4,
    skip_leading_cd: bool = False,
    segment_costs: bool = False,
    segment_fit: str = "nnls",
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
        min_profile_tasks=min_profile_tasks,
        prior_aggregation=prior_aggregation,
        command_field=command_field,
        max_prefix_depth=max_prefix_depth,
        skip_leading_cd=skip_leading_cd,
        segment_costs=segment_costs,
        segment_fit=segment_fit,
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


def hazard_recheck_ms(
    values: list[float],
    *,
    threshold_ms: float,
    kv_cost_ms: float,
) -> float:
    """Expected-cost-optimal swap re-check time for a t0-declined call.

    The policy may start the swap at elapsed ``k`` while the call is still
    running. Under the node's sample distribution, a candidate ``k`` earns,
    per sample latency ``L``:

    * 0 when ``L <= k`` (the re-check never fires),
    * ``min(kv, L - k) - max(0, kv - (L - k))`` when ``L > threshold``
      (hiding on a truly long call minus its residual stall),
    * ``-max(0, kv - (L - k))`` when ``k < L <= threshold`` (a late swap on
      a short call only stalls).

    The expected benefit is piecewise linear in ``k`` with breakpoints only
    at sample-derived points (``L`` and ``L - kv``), so maximizing over
    those candidates plus ``{0, threshold}`` is exact - no grid and no
    tuning parameter. Ties resolve to the latest ``k``, so thin or
    ambiguous nodes degrade to the plain ``k = threshold`` re-check, which
    never fires on short calls. Empty ``values`` return ``threshold_ms``.
    """

    if not values:
        return threshold_ms
    samples = np.asarray(values, dtype=float)
    candidates = {0.0, threshold_ms}
    for value in values:
        if 0.0 < value < threshold_ms:
            candidates.add(float(value))
        edge = value - kv_cost_ms
        if 0.0 < edge < threshold_ms:
            candidates.add(float(edge))
    is_long = samples > threshold_ms
    best_k = threshold_ms
    best_benefit = -math.inf
    for k in sorted(candidates):
        window = samples - k
        fires = samples > k
        hidden_on_long = np.where(fires & is_long, np.minimum(kv_cost_ms, window), 0.0)
        exposed = np.where(fires, np.maximum(0.0, kv_cost_ms - window), 0.0)
        benefit = float(np.mean(hidden_on_long - exposed))
        if benefit >= best_benefit:
            best_benefit = benefit
            best_k = k
    return best_k


def _make_segment_keying(
    model: SegmentCostModel,
    command_field: str,
    *,
    fallback_keys: Callable[[dict[str, Any]], tuple[str, ...]],
    max_prefix_depth: int,
    min_tool_history: int,
) -> tuple[
    Callable[[dict[str, Any]], tuple[str, ...]],
    Callable[[dict[str, Any]], float],
    Callable[[dict[str, Any]], float],
]:
    """Row keying/value/deduction functions for the segment-cost model.

    A row is keyed by its dominant segment's prefix chain; its group nodes
    store the segment-attributed value ``latency - deduction`` (clipped at
    zero). Rows without a usable command or without any adequately observed
    segment head fall back to whole-command keying with zero deduction.
    Results are cached per row object, as the three callbacks are invoked
    on the same rows repeatedly. ``min_tool_history`` deliberately doubles
    as the segment-head evidence gate: both ask "how many observations
    before this estimate is trusted over its backoff", and a second knob
    would be an unjustified hyperparameter.
    """

    cache: dict[int, tuple[tuple[str, ...], float]] = {}

    def derived(row: dict[str, Any]) -> tuple[tuple[str, ...], float]:
        cache_id = id(row)
        found = cache.get(cache_id)
        if found is not None:
            return found
        tool_args = row.get("tool_args")
        command = tool_args.get(command_field) if isinstance(tool_args, dict) else None
        tool_name = row.get("tool_name")
        if (
            not isinstance(command, str)
            or not command.strip()
            or not isinstance(tool_name, str)
            or not tool_name
        ):
            result = fallback_keys(row), 0.0
        else:
            dominant, deduction = model.dominant_segment_and_deduction(
                tool_name,
                command,
                min_head_count=min_tool_history,
            )
            if dominant is None:
                result = fallback_keys(row), 0.0
            else:
                result = (
                    segment_prefix_keys(
                        tool_name, dominant, max_depth=max_prefix_depth
                    ),
                    deduction,
                )
        cache[cache_id] = result
        return result

    def row_keys(row: dict[str, Any]) -> tuple[str, ...]:
        return derived(row)[0]

    def row_value(row: dict[str, Any]) -> float:
        latency_ms = required_nonnegative_float(
            row,
            "latency_ms",
            source=f"row {row.get('sample_id', '<unknown>')}",
        )
        return max(0.0, latency_ms - derived(row)[1])

    def row_deduction(row: dict[str, Any]) -> float:
        return derived(row)[1]

    return row_keys, row_value, row_deduction


def _row_task_id(row: dict[str, Any], *, source: str) -> str:
    """Return the logical task identity, with legacy trace-level fallback."""

    if "task_id" in row:
        return required_text(row, "task_id", source=source)
    return required_text(row, "source_trace", source=source)


def _record_source_task(
    task_id_by_source: dict[str, str],
    *,
    source_trace: str,
    task_id: str,
    source: str,
) -> None:
    existing = task_id_by_source.get(source_trace)
    if existing is not None and existing != task_id:
        raise ValueError(
            f"{source}: source_trace {source_trace!r} maps to conflicting "
            f"task_id values {existing!r} and {task_id!r}"
        )
    task_id_by_source[source_trace] = task_id


def latency_prior_hierarchy(
    prior: LatencyPrior,
    tool_name: str,
    group_keys: tuple[str, ...],
    *,
    min_tool_history: int,
    min_profile_tasks: int,
) -> tuple[LatencyPriorNode, ...]:
    """Return eligible prior nodes ordered from global to most specific."""

    nodes = [
        LatencyPriorNode(
            values=prior.global_values,
            values_by_task=prior.values_by_task,
            source="prior_global",
            group_key=None,
        )
    ]
    tool_values = prior.values_by_tool.get(tool_name, [])
    tool_values_by_task = prior.values_by_task_by_tool.get(tool_name, {})
    if (
        len(tool_values) >= min_tool_history
        and len(tool_values_by_task) >= min_profile_tasks
    ):
        nodes.append(
            LatencyPriorNode(
                values=tool_values,
                values_by_task=tool_values_by_task,
                source="prior_tool",
                group_key=None,
            )
        )
    if prior.values_by_group is not None and prior.values_by_task_by_group is not None:
        for group_key in group_keys:
            group_values = prior.values_by_group.get(group_key, [])
            group_values_by_task = prior.values_by_task_by_group.get(group_key, {})
            if (
                len(group_values) >= min_tool_history
                and len(group_values_by_task) >= min_profile_tasks
            ):
                nodes.append(
                    LatencyPriorNode(
                        values=group_values,
                        values_by_task=group_values_by_task,
                        source="prior_group",
                        group_key=group_key,
                    )
                )
    return tuple(nodes)


def validate_profile_eval_disjoint(
    eval_rows: list[dict[str, Any]],
    *,
    prior: LatencyPrior,
) -> dict[str, str]:
    """Validate eval identity and return its sample-to-task mapping."""

    if not eval_rows:
        raise ValueError("no latency rows supplied")
    task_id_by_sample_id: dict[str, str] = {}
    eval_task_id_by_source: dict[str, str] = {}
    for index, row in enumerate(eval_rows):
        source = f"eval row {index}"
        sample_id = required_text(row, "sample_id", source=source)
        source_trace = required_text(row, "source_trace", source=source)
        task_id = _row_task_id(row, source=source)
        _record_source_task(
            eval_task_id_by_source,
            source_trace=source_trace,
            task_id=task_id,
            source=source,
        )
        existing_task_id = task_id_by_sample_id.get(sample_id)
        if existing_task_id is not None and existing_task_id != task_id:
            raise ValueError(
                f"duplicate sample_id {sample_id!r} with conflicting task_id values"
            )
        task_id_by_sample_id[sample_id] = task_id

    overlap = sorted(set(eval_task_id_by_source) & prior.source_traces)
    if overlap:
        raise ValueError(
            "profile and eval rows must come from disjoint traces; "
            f"shared source_trace values: {overlap}"
        )
    task_overlap = sorted(set(eval_task_id_by_source.values()) & prior.task_ids)
    if task_overlap:
        raise ValueError(
            "profile and eval rows must come from disjoint logical tasks; "
            f"shared task_id values: {task_overlap}"
        )
    return task_id_by_sample_id


def _select_prior(
    prior: LatencyPrior,
    tool_name: str,
    group_keys: tuple[str, ...],
    *,
    min_tool_history: int,
    min_profile_tasks: int,
) -> tuple[list[float], dict[str, list[float]], str, str | None]:
    node = latency_prior_hierarchy(
        prior,
        tool_name,
        group_keys,
        min_tool_history=min_tool_history,
        min_profile_tasks=min_profile_tasks,
    )[-1]
    return node.values, node.values_by_task, node.source, node.group_key


def _estimate_survival(
    predictor: str,
    *,
    prior_threshold_ms: float,
    online_threshold_ms: float,
    prior_values: list[float],
    prior_values_by_task: dict[str, list[float]],
    prior_source: str,
    prior_strength: float | None,
    prior_aggregation: str,
    online_history: list[float],
    online_source: str,
) -> dict[str, Any]:
    prior_count = len(prior_values)
    prior_task_count = len(prior_values_by_task)
    prior_exceed = prior_count - bisect_right(prior_values, prior_threshold_ms)
    online_count = len(online_history)
    online_exceed = sum(value > online_threshold_ms for value in online_history)

    if predictor == "prior_only":
        probability = (
            prior_exceed / prior_count
            if prior_aggregation == "call"
            else _task_balanced_survival(
                prior_values_by_task,
                threshold_ms=prior_threshold_ms,
            )
        )
        return {
            "probability": probability,
            "effective_count": float(
                prior_count if prior_aggregation == "call" else prior_task_count
            ),
            "prior_source": prior_source,
            "online_source": None,
            "prior_count": prior_count,
            "prior_task_count": prior_task_count,
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
            "prior_task_count": None,
            "online_count": online_count,
        }
    prior_weight = prior_strength if prior_strength is not None else float(prior_count)
    prior_rate = prior_exceed / prior_count
    effective_count = prior_weight + online_count
    # Each side may condition at a different level (deducted group query vs
    # raw tool/global query), but under the additive model both estimate the
    # same P(latency > threshold), so pooling them stays coherent.
    probability = (prior_weight * prior_rate + online_exceed) / effective_count
    return {
        "probability": probability,
        "effective_count": effective_count,
        "prior_source": prior_source,
        "online_source": online_source,
        "prior_count": prior_count,
        "prior_task_count": prior_task_count,
        "online_count": online_count,
    }


def _task_balanced_survival(
    values_by_task: Mapping[str, list[float]],
    *,
    threshold_ms: float,
) -> float:
    """Average within-task ECDF survival so call repetition has no task weight."""

    rates = [
        (len(values) - bisect_right(values, threshold_ms)) / len(values)
        for values in values_by_task.values()
    ]
    if not rates:
        raise ValueError("task-balanced prior node has no task samples")
    return float(sum(rates) / len(rates))


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
    probability_rows = [
        d for d in decisions if d.probability_exceeds_threshold is not None
    ]
    abstain_count = sum(d.abstained for d in decisions)
    cold_start_count = sum(
        d.predicted_exceeds_threshold is None and not d.abstained for d in decisions
    )
    core = binary_classification_metrics(
        (d.label_exceeds_threshold, d.predicted_exceeds_threshold) for d in decided
    )
    brier_score = (
        sum(
            (float(d.probability_exceeds_threshold) - float(d.label_exceeds_threshold))
            ** 2
            for d in probability_rows
        )
        / len(probability_rows)
        if probability_rows
        else None
    )
    brier_by_task: dict[str, list[float]] = {}
    for decision in probability_rows:
        error = (
            float(decision.probability_exceeds_threshold)
            - float(decision.label_exceeds_threshold)
        ) ** 2
        brier_by_task.setdefault(decision.task_id, []).append(error)
    task_macro_brier_score = (
        sum(sum(errors) / len(errors) for errors in brier_by_task.values())
        / len(brier_by_task)
        if brier_by_task
        else None
    )
    return {
        "row_count": len(decisions),
        "task_count": len({d.task_id for d in decisions}),
        "probability_count": len(probability_rows),
        "probability_task_count": len(brier_by_task),
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
        "brier_score": brier_score,
        "task_macro_brier_score": task_macro_brier_score,
    }


__all__ = [
    "LatencyPrior",
    "LatencyPriorNode",
    "ProfiledThresholdDecision",
    "build_latency_prior",
    "evaluate_profiled_latency_thresholds",
    "hazard_recheck_ms",
    "latency_prior_hierarchy",
    "load_and_evaluate_profiled_latency_thresholds",
    "validate_profile_eval_disjoint",
    "write_profiled_outputs",
]
