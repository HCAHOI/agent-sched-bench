"""Paired task-cluster uncertainty for tool-latency policy confirmation."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable, Mapping

import numpy as np

from trace_collect.latency_validation import normalized_positive_floats, required_text
from trace_collect.tool_latency_utility_clock import (
    trigger_policy_utility_ms,
    validate_restore_cost,
)


def paired_task_cluster_bootstrap(
    decisions: Iterable[Mapping[str, Any]],
    *,
    costs_ms: Iterable[float],
    replicates: int,
    confidence_level: float,
    seed: int,
    baseline_trigger_field: str = "robust_trigger_ms",
    treatment_trigger_field: str = "offline_gated_robust_trigger_ms",
    restore_cost_fraction: float = 0.0,
    baseline_restore_cost_ms_field: str | None = None,
    treatment_restore_cost_ms_field: str | None = None,
    enforce_gated_treatment: bool = True,
) -> dict[str, Any]:
    """Bootstrap paired workload-total utility by resampling logical tasks.

    ``restore_cost_fraction`` charges each fire on a short call a swap-back
    of that fraction of the row's kv cost (swap-in scales with the swapped
    KV footprint like swap-out does). ``enforce_gated_treatment`` asserts
    the gated-policy invariant that every treatment trigger equals either
    the baseline trigger or the deadline; disable it only for comparisons
    whose baseline is not the gate's fallback pair (e.g. treatment vs. the
    fixed deadline itself).

    ``baseline_restore_cost_ms_field`` / ``treatment_restore_cost_ms_field``
    optionally override the restore charge on each side with a per-row
    absolute cost (in ms) read from that decision field, instead of
    ``restore_cost_fraction * kv_cost_ms``. This lets one policy pay a
    context-dependent restore (e.g. min(swap-in, recompute); see
    tool_latency_recompute) while another pays the plain swap-in, with both
    firing on the same trigger. ``None`` (default) keeps the scalar swap
    charge, reproducing the frozen numerics exactly.

    ``restore_cost_fraction`` and ``enforce_gated_treatment`` are additive
    output keys on top of schema_version 1; defaults reproduce the frozen
    confirmation numerics exactly.
    """

    costs = normalized_positive_floats(costs_ms, label="confirmation cost")
    validate_restore_cost(restore_cost_fraction, label="restore_cost_fraction")
    if (
        not isinstance(replicates, int)
        or isinstance(replicates, bool)
        or replicates < 1
    ):
        raise ValueError("replicates must be a positive integer")
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be finite and between zero and one")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    rows = list(decisions)
    if not rows:
        raise ValueError("confirmation decisions must be non-empty")
    expected_costs = set(costs)
    rows_by_sample: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    task_by_sample: dict[str, str] = {}
    fold_by_task: dict[str, str | None] = {}
    for index, row in enumerate(rows):
        source = f"confirmation decision {index}"
        sample_id = required_text(row, "sample_id", source=source)
        task_id = required_text(row, "task_id", source=source)
        rows_by_sample[sample_id].append(row)
        existing_task = task_by_sample.setdefault(sample_id, task_id)
        if existing_task != task_id:
            raise ValueError(f"sample {sample_id!r} maps to multiple tasks")
        fold = row.get("outer_fold")
        if fold is not None and (not isinstance(fold, str) or not fold.strip()):
            raise ValueError(f"{source} has an invalid outer_fold")
        existing_fold = fold_by_task.setdefault(task_id, fold)
        if existing_fold != fold:
            raise ValueError(f"task {task_id!r} maps to multiple outer folds")

    for sample_id, panel in rows_by_sample.items():
        panel_costs = [float(row["kv_cost_ms"]) for row in panel]
        if (
            len(panel_costs) != len(set(panel_costs))
            or set(panel_costs) != expected_costs
        ):
            raise ValueError(f"sample {sample_id!r} has an incomplete cost panel")

    task_ids = sorted(set(task_by_sample.values()))
    task_index = {task_id: index for index, task_id in enumerate(task_ids)}
    cost_index = {cost: index for index, cost in enumerate(costs)}
    contributions = np.zeros((len(task_ids), len(costs)), dtype=float)
    counts = {
        cost: {
            "baseline_early_fire_count": 0,
            "treatment_early_fire_count": 0,
            "baseline_early_short_fire_count": 0,
            "treatment_early_short_fire_count": 0,
        }
        for cost in costs
    }
    thresholds: dict[float, float] = {}
    for index, row in enumerate(rows):
        source = f"confirmation decision {index}"
        task_id = required_text(row, "task_id", source=source)
        cost = float(row["kv_cost_ms"])
        threshold = float(row["threshold_ms"])
        latency = float(row["latency_ms"])
        baseline_trigger = float(row[baseline_trigger_field])
        treatment_trigger = float(row[treatment_trigger_field])
        values = (cost, threshold, latency, baseline_trigger, treatment_trigger)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{source} has a non-finite policy value")
        if (
            cost <= 0.0
            or threshold <= 0.0
            or latency < 0.0
            or not 0.0 <= baseline_trigger <= threshold
            or not 0.0 <= treatment_trigger <= threshold
        ):
            raise ValueError(f"{source} has an invalid cost, latency, or trigger")
        existing_threshold = thresholds.setdefault(cost, threshold)
        if not math.isclose(existing_threshold, threshold, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"cost {cost} maps to multiple thresholds")
        if enforce_gated_treatment and not (
            math.isclose(
                treatment_trigger,
                baseline_trigger,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or math.isclose(
                treatment_trigger,
                threshold,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ValueError(f"{source} treatment is not baseline or deadline")

        scalar_restore_cost_ms = restore_cost_fraction * cost
        baseline_restore_cost_ms = _resolve_restore_cost(
            row, baseline_restore_cost_ms_field, scalar_restore_cost_ms, source=source
        )
        treatment_restore_cost_ms = _resolve_restore_cost(
            row, treatment_restore_cost_ms_field, scalar_restore_cost_ms, source=source
        )
        baseline_utility = trigger_policy_utility_ms(
            latency,
            baseline_trigger,
            threshold_ms=threshold,
            kv_cost_ms=cost,
            restore_cost_ms=baseline_restore_cost_ms,
        )
        treatment_utility = trigger_policy_utility_ms(
            latency,
            treatment_trigger,
            threshold_ms=threshold,
            kv_cost_ms=cost,
            restore_cost_ms=treatment_restore_cost_ms,
        )
        contributions[task_index[task_id], cost_index[cost]] += (
            treatment_utility - baseline_utility
        )
        baseline_fires = latency > baseline_trigger and baseline_trigger < threshold
        treatment_fires = latency > treatment_trigger and treatment_trigger < threshold
        if baseline_fires:
            counts[cost]["baseline_early_fire_count"] += 1
            if latency <= threshold:
                counts[cost]["baseline_early_short_fire_count"] += 1
        if treatment_fires:
            counts[cost]["treatment_early_fire_count"] += 1
            if latency <= threshold:
                counts[cost]["treatment_early_short_fire_count"] += 1

    bootstrap_totals = _resample_task_totals(
        contributions,
        replicates=replicates,
        seed=seed,
    )
    alpha = 1.0 - confidence_level
    point_quantiles = np.quantile(
        bootstrap_totals,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        axis=0,
        method="linear",
    )
    family_tail = alpha / (2.0 * len(costs))
    simultaneous_quantiles = np.quantile(
        bootstrap_totals,
        [family_tail, 1.0 - family_tail],
        axis=0,
        method="linear",
    )
    observed = np.sum(contributions, axis=0)

    points: dict[str, Any] = {}
    for column, cost in enumerate(costs):
        simultaneous_low = float(simultaneous_quantiles[0, column])
        simultaneous_high = float(simultaneous_quantiles[1, column])
        if simultaneous_low > 0.0:
            label = "positive"
        elif simultaneous_high < 0.0:
            label = "harmful"
        else:
            label = "inconclusive"
        column_values = contributions[:, column]
        positive_index = int(np.argmax(column_values))
        negative_index = int(np.argmin(column_values))
        points[str(cost)] = {
            "kv_cost_ms": cost,
            "threshold_ms": thresholds[cost],
            "paired_delta_ms": float(observed[column]),
            "paired_delta_normalized": float(observed[column] / cost),
            "pointwise_interval_ms": {
                "low": float(point_quantiles[0, column]),
                "high": float(point_quantiles[1, column]),
            },
            "simultaneous_interval_ms": {
                "low": simultaneous_low,
                "high": simultaneous_high,
            },
            "simultaneous_label": label,
            "largest_positive_task": {
                "task_id": task_ids[positive_index],
                "paired_delta_ms": float(column_values[positive_index]),
            },
            "largest_negative_task": {
                "task_id": task_ids[negative_index],
                "paired_delta_ms": float(column_values[negative_index]),
            },
            **counts[cost],
        }

    task_contributions = [
        {
            "task_id": task_id,
            "outer_fold": fold_by_task[task_id],
            "paired_delta_ms_by_cost": {
                str(cost): float(contributions[row_index, column])
                for column, cost in enumerate(costs)
            },
        }
        for row_index, task_id in enumerate(task_ids)
    ]
    fold_paired_delta_ms_by_cost: dict[str, dict[str, float]] = {}
    for fold in sorted({fold for fold in fold_by_task.values() if fold is not None}):
        fold_rows = [
            index
            for index, task_id in enumerate(task_ids)
            if fold_by_task[task_id] == fold
        ]
        fold_totals = np.sum(contributions[fold_rows], axis=0)
        fold_paired_delta_ms_by_cost[fold] = {
            str(cost): float(fold_totals[column]) for column, cost in enumerate(costs)
        }
    return {
        "schema_version": 1,
        "baseline_trigger_field": baseline_trigger_field,
        "treatment_trigger_field": treatment_trigger_field,
        "restore_cost_fraction": restore_cost_fraction,
        "baseline_restore_cost_ms_field": baseline_restore_cost_ms_field,
        "treatment_restore_cost_ms_field": treatment_restore_cost_ms_field,
        "enforce_gated_treatment": enforce_gated_treatment,
        "sample_count": len(rows_by_sample),
        "task_count": len(task_ids),
        "costs_ms": costs,
        "bootstrap": {
            "unit": "task_id",
            "replicates": replicates,
            "confidence_level": confidence_level,
            "seed": seed,
            "bit_generator": "PCG64",
            "pointwise_method": "percentile",
            "simultaneous_method": "bonferroni_percentile",
            "simultaneous_family_size": len(costs),
            "simultaneous_tail_probability": family_tail,
            "conditional_on_fitted_outer_policies": True,
        },
        "points": points,
        "fold_paired_delta_ms_by_cost": fold_paired_delta_ms_by_cost,
        "task_contributions": task_contributions,
    }


def _resolve_restore_cost(
    row: Mapping[str, Any],
    field: str | None,
    scalar_restore_cost_ms: float,
    *,
    source: str,
) -> float:
    """Per-row restore cost: a decision field when named, else the scalar."""

    if field is None:
        return scalar_restore_cost_ms
    value = row.get(field)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{source} lacks a numeric restore field {field!r}")
    restore_cost_ms = float(value)
    validate_restore_cost(restore_cost_ms, label=field)
    return restore_cost_ms


def _resample_task_totals(
    contributions: np.ndarray,
    *,
    replicates: int,
    seed: int,
    batch_size: int = 4096,
) -> np.ndarray:
    """Resample complete task contribution vectors in bounded memory."""

    task_count, cost_count = contributions.shape
    if task_count < 1 or cost_count < 1:
        raise ValueError("task contribution matrix must be non-empty")
    rng = np.random.Generator(np.random.PCG64(seed))
    probabilities = np.full(task_count, 1.0 / task_count, dtype=float)
    totals = np.empty((replicates, cost_count), dtype=float)
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        multiplicities = rng.multinomial(
            task_count,
            probabilities,
            size=stop - start,
        )
        totals[start:stop] = multiplicities @ contributions
    return totals


__all__ = ["paired_task_cluster_bootstrap"]
