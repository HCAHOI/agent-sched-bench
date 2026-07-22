"""Classifier-free utility-clock policies for profiled tool latency.

Each policy schedules one action trigger while the tool is still running:

* ``deadline_only`` fires at the fixed action threshold.
* ``mean_hazard`` maximizes empirical action utility at one prior node.
* ``robust_clock`` fires early only when every leave-one-task-out selected-node
  and parent model prefers that trigger to every later trigger. Otherwise it
  falls back to the fixed deadline.

No policy consumes a binary long/short prediction or a probability cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from trace_collect.classification_metrics import safe_div
from trace_collect.command_features import make_row_command_prefix_keys
from trace_collect.latency_validation import (
    normalized_positive_floats,
    required_nonnegative_float,
    required_text,
)
from trace_collect.tool_latency_dataset import read_tool_latency_jsonl
from trace_collect.tool_latency_profiled import (
    LatencyPriorNode,
    build_latency_prior,
    hazard_recheck_ms,
    latency_prior_hierarchy,
    validate_profile_eval_disjoint,
)


_POLICIES = ("deadline_only", "mean_hazard", "robust_clock")
_TRIGGER_FIELDS = {
    "deadline_only": "deadline_trigger_ms",
    "mean_hazard": "mean_hazard_trigger_ms",
    "robust_clock": "robust_trigger_ms",
}


@dataclass(frozen=True)
class UtilityClockDecision:
    """One held-out call and its classifier-free trigger times."""

    sample_id: str
    task_id: str
    tool_name: str
    latency_ms: float
    kv_cost_ms: float
    threshold_ms: float
    label_exceeds_threshold: bool
    prior_source: str
    prior_group_key: str | None
    prior_task_count: int
    robust_source: str
    robust_group_key: str | None
    robust_task_count: int
    deadline_trigger_ms: float
    mean_hazard_trigger_ms: float
    robust_trigger_ms: float

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "task_id": self.task_id,
            "tool_name": self.tool_name,
            "latency_ms": self.latency_ms,
            "kv_cost_ms": self.kv_cost_ms,
            "threshold_ms": self.threshold_ms,
            "label_exceeds_threshold": self.label_exceeds_threshold,
            "prior_source": self.prior_source,
            "prior_group_key": self.prior_group_key,
            "prior_task_count": self.prior_task_count,
            "robust_source": self.robust_source,
            "robust_group_key": self.robust_group_key,
            "robust_task_count": self.robust_task_count,
            "deadline_trigger_ms": self.deadline_trigger_ms,
            "mean_hazard_trigger_ms": self.mean_hazard_trigger_ms,
            "robust_trigger_ms": self.robust_trigger_ms,
        }


@dataclass(frozen=True)
class RobustUtilityTriggerStats:
    """Robust trigger and its weakest normalized advantage over waiting."""

    trigger_ms: float
    normalized_advantage: float


def evaluate_utility_clock_policy(
    eval_rows: Iterable[dict[str, Any]],
    *,
    profile_rows: Iterable[dict[str, Any]],
    kv_costs_ms: Iterable[float],
    guard_ms: float,
    min_tool_history: int = 1,
    min_profile_tasks: int = 1,
    command_field: str | None = None,
    max_prefix_depth: int = 4,
    skip_leading_cd: bool = False,
    transparent_wrappers: frozenset[str] = frozenset(),
    restore_cost_fraction: float = 0.0,
) -> dict[str, Any]:
    """Evaluate fixed-deadline, mean-hazard, and robust-clock triggers.

    ``restore_cost_fraction`` charges each fire on a short call a swap-back
    of that fraction of its kv cost; swap-in scales with the swapped KV
    footprint the same way swap-out does, so the knob is dimensionless.

    ``transparent_wrappers`` threads a learned wrapper-transparency set into
    the command-prefix keying (see command_features); the empty default is a
    no-op reproducing the frozen keys byte-for-byte.
    """

    kv_costs = normalized_positive_floats(kv_costs_ms, label="kv cost")
    if not math.isfinite(guard_ms) or guard_ms < 0.0:
        raise ValueError(f"guard_ms must be finite and non-negative, got {guard_ms}")
    validate_restore_cost(restore_cost_fraction, label="restore_cost_fraction")
    if min_tool_history < 1:
        raise ValueError(f"min_tool_history must be >= 1, got {min_tool_history}")
    if min_profile_tasks < 1:
        raise ValueError(f"min_profile_tasks must be >= 1, got {min_profile_tasks}")
    if max_prefix_depth < 1:
        raise ValueError(f"max_prefix_depth must be >= 1, got {max_prefix_depth}")

    row_group_keys = (
        make_row_command_prefix_keys(
            command_field,
            max_depth=max_prefix_depth,
            skip_leading_cd=skip_leading_cd,
            transparent_wrappers=transparent_wrappers,
        )
        if command_field is not None
        else None
    )
    profile_list = list(profile_rows)
    prior = build_latency_prior(profile_list, row_group_keys=row_group_keys)
    if len(prior.task_ids) < min_profile_tasks:
        raise ValueError(
            "latency prior has fewer logical tasks than min_profile_tasks: "
            f"{len(prior.task_ids)} < {min_profile_tasks}"
        )
    eval_list = list(eval_rows)
    task_id_by_sample = validate_profile_eval_disjoint(eval_list, prior=prior)
    row_by_sample: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(eval_list):
        sample_id = required_text(row, "sample_id", source=f"eval row {index}")
        if sample_id in row_by_sample:
            raise ValueError(f"duplicate eval sample_id: {sample_id!r}")
        row_by_sample[sample_id] = row

    decisions: list[UtilityClockDecision] = []
    trigger_cache: dict[tuple[int, int | None, float, float], float] = {}
    for sample_id, row in row_by_sample.items():
        source = f"eval sample {sample_id}"
        tool_name = required_text(row, "tool_name", source=source)
        latency_ms = required_nonnegative_float(row, "latency_ms", source=source)
        group_keys = row_group_keys(row) if row_group_keys is not None else ()
        hierarchy = latency_prior_hierarchy(
            prior,
            tool_name,
            group_keys,
            min_tool_history=min_tool_history,
            min_profile_tasks=min_profile_tasks,
        )
        selected = hierarchy[-1]
        robust_node, robust_parent = robust_prior_nodes(hierarchy)
        for kv_cost_ms in kv_costs:
            threshold_ms = kv_cost_ms + guard_ms
            restore_cost_ms = restore_cost_fraction * kv_cost_ms
            mean_hazard_ms = hazard_recheck_ms(
                selected.values,
                threshold_ms=threshold_ms,
                kv_cost_ms=kv_cost_ms,
                restore_cost_ms=restore_cost_ms,
            )
            cache_key = (
                id(robust_node.values),
                id(robust_parent.values) if robust_parent is not None else None,
                threshold_ms,
                kv_cost_ms,
            )
            robust_ms = trigger_cache.get(cache_key)
            if robust_ms is None:
                robust_ms = robust_utility_trigger_ms(
                    robust_node,
                    parent=robust_parent,
                    threshold_ms=threshold_ms,
                    kv_cost_ms=kv_cost_ms,
                    restore_cost_ms=restore_cost_ms,
                )
                trigger_cache[cache_key] = robust_ms
            decisions.append(
                UtilityClockDecision(
                    sample_id=sample_id,
                    task_id=task_id_by_sample[sample_id],
                    tool_name=tool_name,
                    latency_ms=latency_ms,
                    kv_cost_ms=kv_cost_ms,
                    threshold_ms=threshold_ms,
                    label_exceeds_threshold=latency_ms > threshold_ms,
                    prior_source=selected.source,
                    prior_group_key=selected.group_key,
                    prior_task_count=len(selected.values_by_task),
                    robust_source=robust_node.source,
                    robust_group_key=robust_node.group_key,
                    robust_task_count=len(robust_node.values_by_task),
                    deadline_trigger_ms=threshold_ms,
                    mean_hazard_trigger_ms=mean_hazard_ms,
                    robust_trigger_ms=robust_ms,
                )
            )

    decision_rows = [decision.to_json_obj() for decision in decisions]
    points = []
    for kv_cost_ms in kv_costs:
        threshold_ms = kv_cost_ms + guard_ms
        restore_cost_ms = restore_cost_fraction * kv_cost_ms
        rows = [row for row in decision_rows if row["kv_cost_ms"] == kv_cost_ms]
        positive_count = sum(row["label_exceeds_threshold"] for row in rows)
        oracle_ms = positive_count * kv_cost_ms
        points.append(
            {
                "kv_cost_ms": kv_cost_ms,
                "guard_ms": guard_ms,
                "threshold_ms": threshold_ms,
                "restore_cost_ms": restore_cost_ms,
                "row_count": len(rows),
                "positive_count": positive_count,
                "absorbed_if_oracle_ms": oracle_ms,
                "policies": {
                    policy: _accumulate_trigger_policy(
                        rows,
                        trigger_field=_TRIGGER_FIELDS[policy],
                        kv_cost_ms=kv_cost_ms,
                        threshold_ms=threshold_ms,
                        oracle_ms=oracle_ms,
                        restore_cost_ms=restore_cost_ms,
                    )
                    for policy in _POLICIES
                },
            }
        )
    return {
        "policies": list(_POLICIES),
        "min_tool_history": min_tool_history,
        "min_profile_tasks": min_profile_tasks,
        "command_field": command_field,
        "max_prefix_depth": max_prefix_depth,
        "skip_leading_cd": skip_leading_cd,
        "kv_costs_ms": kv_costs,
        "guard_ms": guard_ms,
        "restore_cost_fraction": restore_cost_fraction,
        "row_count": len(eval_list),
        "profile_row_count": len(profile_list),
        "profile_trace_count": len(prior.source_traces),
        "profile_task_count": len(prior.task_ids),
        "points": points,
        "decisions": decision_rows,
    }


def load_and_evaluate_utility_clock_policy(
    eval_path: Path,
    *,
    profile_path: Path,
    kv_costs_ms: Iterable[float],
    guard_ms: float,
    min_tool_history: int = 1,
    min_profile_tasks: int = 1,
    command_field: str | None = None,
    max_prefix_depth: int = 4,
    skip_leading_cd: bool = False,
    restore_cost_fraction: float = 0.0,
) -> dict[str, Any]:
    return evaluate_utility_clock_policy(
        read_tool_latency_jsonl(eval_path),
        profile_rows=read_tool_latency_jsonl(profile_path),
        kv_costs_ms=kv_costs_ms,
        guard_ms=guard_ms,
        min_tool_history=min_tool_history,
        min_profile_tasks=min_profile_tasks,
        command_field=command_field,
        max_prefix_depth=max_prefix_depth,
        skip_leading_cd=skip_leading_cd,
        restore_cost_fraction=restore_cost_fraction,
    )


def robust_utility_trigger_ms(
    node: LatencyPriorNode,
    *,
    parent: LatencyPriorNode | None,
    threshold_ms: float,
    kv_cost_ms: float,
    restore_cost_ms: float = 0.0,
) -> float:
    """Choose the earliest trigger unanimously preferred to every later one.

    The model family contains the full selected node, every non-empty
    leave-one-task-out selected-node fit, and the same curves for its parent.
    A strict comparison makes utility ties wait. If there is no unanimous
    positive early trigger, the fixed threshold is returned.
    """

    return robust_utility_trigger_stats(
        node,
        parent=parent,
        threshold_ms=threshold_ms,
        kv_cost_ms=kv_cost_ms,
        restore_cost_ms=restore_cost_ms,
    ).trigger_ms


def robust_utility_trigger_stats(
    node: LatencyPriorNode,
    *,
    parent: LatencyPriorNode | None,
    threshold_ms: float,
    kv_cost_ms: float,
    restore_cost_ms: float = 0.0,
) -> RobustUtilityTriggerStats:
    """Return the robust trigger and minimum curve advantage over waiting."""

    if not math.isfinite(threshold_ms) or threshold_ms <= 0.0:
        raise ValueError(
            f"threshold_ms must be finite and positive, got {threshold_ms}"
        )
    if not math.isfinite(kv_cost_ms) or kv_cost_ms <= 0.0:
        raise ValueError(f"kv_cost_ms must be finite and positive, got {kv_cost_ms}")
    validate_restore_cost(restore_cost_ms)
    if len(node.values_by_task) < 2:
        return RobustUtilityTriggerStats(threshold_ms, 0.0)
    model_nodes = [node]
    if parent is not None:
        model_nodes.append(parent)
    candidates = _utility_candidates(model_nodes, threshold_ms, kv_cost_ms)
    curves = [
        curve
        for model_node in model_nodes
        for curve in _node_utility_curves(
            model_node,
            candidates,
            threshold_ms=threshold_ms,
            kv_cost_ms=kv_cost_ms,
            restore_cost_ms=restore_cost_ms,
        )
    ]
    if not curves:
        return RobustUtilityTriggerStats(threshold_ms, 0.0)
    curve_matrix = np.asarray(curves)
    future_best = np.maximum.accumulate(curve_matrix[:, ::-1], axis=1)[:, ::-1]
    minimum_advantages = np.min(
        curve_matrix[:, :-1] - np.maximum(future_best[:, 1:], 0.0),
        axis=0,
    )
    positive = np.flatnonzero(minimum_advantages > 0.0)
    if positive.size:
        index = int(positive[0])
        minimum_advantage = float(minimum_advantages[index])
        return RobustUtilityTriggerStats(
            trigger_ms=float(candidates[index]),
            normalized_advantage=minimum_advantage / kv_cost_ms,
        )
    return RobustUtilityTriggerStats(threshold_ms, 0.0)


def robust_prior_nodes(
    hierarchy: tuple[LatencyPriorNode, ...],
) -> tuple[LatencyPriorNode, LatencyPriorNode | None]:
    """Select the deepest multi-task node and its immediate parent."""

    if not hierarchy:
        raise ValueError("robust prior hierarchy must be non-empty")
    for index in range(len(hierarchy) - 1, -1, -1):
        if len(hierarchy[index].values_by_task) >= 2:
            return hierarchy[index], hierarchy[index - 1] if index > 0 else None
    return hierarchy[0], None


def _utility_candidates(
    nodes: list[LatencyPriorNode],
    threshold_ms: float,
    kv_cost_ms: float,
) -> np.ndarray:
    candidates = {0.0, threshold_ms}
    for node in nodes:
        for value in node.values:
            if 0.0 < value < threshold_ms:
                candidates.add(float(value))
            edge = value - kv_cost_ms
            if 0.0 < edge < threshold_ms:
                candidates.add(float(edge))
    return np.asarray(sorted(candidates), dtype=float)


def _node_utility_curves(
    node: LatencyPriorNode,
    candidates: np.ndarray,
    *,
    threshold_ms: float,
    kv_cost_ms: float,
    restore_cost_ms: float,
) -> list[np.ndarray]:
    total_sum = np.zeros(len(candidates), dtype=float)
    task_sums: list[tuple[int, np.ndarray]] = []
    total_count = 0
    for values in node.values_by_task.values():
        task_sum = _utility_sum(
            values,
            candidates,
            threshold_ms=threshold_ms,
            kv_cost_ms=kv_cost_ms,
            restore_cost_ms=restore_cost_ms,
        )
        task_sums.append((len(values), task_sum))
        total_sum += task_sum
        total_count += len(values)
    if total_count != len(node.values):
        raise ValueError("prior node task partition does not match its call samples")
    curves = [total_sum / total_count]
    for task_count, task_sum in task_sums:
        remaining_count = total_count - task_count
        if remaining_count:
            curves.append((total_sum - task_sum) / remaining_count)
    return curves


def _utility_sum(
    values: list[float],
    candidates: np.ndarray,
    *,
    threshold_ms: float,
    kv_cost_ms: float,
    restore_cost_ms: float,
) -> np.ndarray:
    utility = _utility_matrix(
        np.asarray(values, dtype=float),
        candidates,
        threshold_ms=threshold_ms,
        kv_cost_ms=kv_cost_ms,
        restore_cost_ms=restore_cost_ms,
    )
    return np.sum(utility, axis=0)


def validate_restore_cost(value: float, *, label: str = "restore_cost_ms") -> None:
    """Reject non-finite or negative differential restore costs."""
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} must be finite and non-negative, got {value}")


def trigger_policy_utility_ms(
    latency_ms: float,
    trigger_ms: float,
    *,
    threshold_ms: float,
    kv_cost_ms: float,
    restore_cost_ms: float = 0.0,
) -> float:
    """Return the utility-clock value for one observed call and trigger."""
    values = (latency_ms, trigger_ms, threshold_ms, kv_cost_ms)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("latency, trigger, threshold, and cost must be finite")
    if latency_ms < 0.0 or trigger_ms < 0.0:
        raise ValueError("latency_ms and trigger_ms must be non-negative")
    if threshold_ms <= 0.0 or kv_cost_ms <= 0.0:
        raise ValueError("threshold_ms and kv_cost_ms must be positive")
    validate_restore_cost(restore_cost_ms)
    utility = _utility_matrix(
        np.asarray([latency_ms], dtype=float),
        np.asarray([trigger_ms], dtype=float),
        threshold_ms=threshold_ms,
        kv_cost_ms=kv_cost_ms,
        restore_cost_ms=restore_cost_ms,
    )
    return float(utility[0, 0])


def utility_matrix(
    samples: np.ndarray,
    candidates: np.ndarray,
    *,
    threshold_ms: float,
    kv_cost_ms: float,
    restore_cost_ms: float = 0.0,
) -> np.ndarray:
    """Public passthrough to the shared utility functional.

    Exposes ``_utility_matrix`` unchanged so learned-predictor lanes (e.g.
    the discrete-time hazard model) apply the identical
    ``hidden_on_long - exposed - rho*restore`` semantics to interval masses
    instead of raw samples. Additive, non-behavioral: the empirical policies
    keep calling ``_utility_matrix`` directly.
    """

    return _utility_matrix(
        samples,
        candidates,
        threshold_ms=threshold_ms,
        kv_cost_ms=kv_cost_ms,
        restore_cost_ms=restore_cost_ms,
    )


def _utility_matrix(
    samples: np.ndarray,
    candidates: np.ndarray,
    *,
    threshold_ms: float,
    kv_cost_ms: float,
    restore_cost_ms: float = 0.0,
) -> np.ndarray:
    sample_column = samples[:, None]
    remaining = sample_column - candidates[None, :]
    fires = remaining > 0.0
    is_long = sample_column > threshold_ms
    hidden_on_long = np.where(
        fires & is_long,
        np.minimum(kv_cost_ms, remaining),
        0.0,
    )
    exposed = np.where(fires, np.maximum(0.0, kv_cost_ms - remaining), 0.0)
    # A fire on a short call swaps state the deadline policy never touches;
    # the swap-back lands on the critical path when the call returns. Long
    # calls pay the same restore under the deadline policy, so no differential
    # charge applies there.
    restore = np.where(fires & ~is_long, restore_cost_ms, 0.0)
    return hidden_on_long - exposed - restore


def _accumulate_trigger_policy(
    rows: list[dict[str, Any]],
    *,
    trigger_field: str,
    kv_cost_ms: float,
    threshold_ms: float,
    oracle_ms: float,
    restore_cost_ms: float = 0.0,
) -> dict[str, Any]:
    trigger_count = 0
    early_trigger_count = 0
    early_trigger_on_short_count = 0
    deadline_trigger_count = 0
    absorbed_ms = 0.0
    absorbed_on_long_ms = 0.0
    exposed_ms = 0.0
    missed_positive_count = 0
    trigger_times: list[float] = []
    for row in rows:
        latency_ms = float(row["latency_ms"])
        trigger_ms = float(row[trigger_field])
        label = bool(row["label_exceeds_threshold"])
        if latency_ms > trigger_ms:
            trigger_count += 1
            trigger_times.append(trigger_ms)
            remaining_ms = latency_ms - trigger_ms
            hidden = min(kv_cost_ms, remaining_ms)
            absorbed_ms += hidden
            if label:
                absorbed_on_long_ms += hidden
            exposed_ms += max(0.0, kv_cost_ms - remaining_ms)
            if trigger_ms < threshold_ms:
                early_trigger_count += 1
                if not label:
                    early_trigger_on_short_count += 1
            else:
                deadline_trigger_count += 1
        elif label:
            missed_positive_count += 1
    restore_ms_total = early_trigger_on_short_count * restore_cost_ms
    return {
        "trigger_count": trigger_count,
        "early_trigger_count": early_trigger_count,
        "early_trigger_on_short_count": early_trigger_on_short_count,
        "deadline_trigger_count": deadline_trigger_count,
        "absorbed_ms_total": absorbed_ms,
        "absorbed_on_long_ms_total": absorbed_on_long_ms,
        "exposed_ms_total": exposed_ms,
        "missed_ms_total": missed_positive_count * kv_cost_ms,
        "restore_ms_total": restore_ms_total,
        "net_saved_ms": absorbed_on_long_ms - exposed_ms - restore_ms_total,
        "hidden_fraction_of_oracle": safe_div(absorbed_on_long_ms, oracle_ms),
        "mean_fired_trigger_ms": (
            sum(trigger_times) / len(trigger_times) if trigger_times else None
        ),
    }


__all__ = [
    "UtilityClockDecision",
    "evaluate_utility_clock_policy",
    "load_and_evaluate_utility_clock_policy",
    "robust_utility_trigger_ms",
    "trigger_policy_utility_ms",
    "utility_matrix",
    "validate_restore_cost",
]
