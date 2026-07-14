"""Exact metric-v1 headroom diagnostics for utility-clock decisions."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from trace_collect.tool_latency_utility_clock import trigger_policy_utility_ms


POLICY_TRIGGER_FIELDS = {
    "deadline_only": "deadline_trigger_ms",
    "mean_hazard": "mean_hazard_trigger_ms",
    "robust_clock": "robust_trigger_ms",
}


@dataclass(frozen=True)
class UtilityDecision:
    """Validated OOF decision required by the headroom analysis."""

    sample_id: str
    task_id: str
    tool_name: str
    latency_ms: float
    kv_cost_ms: float
    threshold_ms: float
    label_exceeds_threshold: bool
    triggers_ms: Mapping[str, float]

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> UtilityDecision:
        """Parse and validate one serialized utility-clock decision."""
        required = {
            "sample_id",
            "task_id",
            "tool_name",
            "latency_ms",
            "kv_cost_ms",
            "threshold_ms",
            "label_exceeds_threshold",
            *POLICY_TRIGGER_FIELDS.values(),
        }
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"decision row is missing fields: {missing}")

        sample_id = str(row["sample_id"])
        task_id = str(row["task_id"])
        tool_name = str(row["tool_name"])
        if not sample_id or not task_id or not tool_name:
            raise ValueError("sample_id, task_id, and tool_name must be non-empty")
        if type(row["label_exceeds_threshold"]) is not bool:
            raise ValueError("label_exceeds_threshold must be a JSON boolean")

        latency_ms = float(row["latency_ms"])
        kv_cost_ms = float(row["kv_cost_ms"])
        threshold_ms = float(row["threshold_ms"])
        triggers_ms = {
            policy: float(row[field]) for policy, field in POLICY_TRIGGER_FIELDS.items()
        }
        numeric_values = (
            latency_ms,
            kv_cost_ms,
            threshold_ms,
            *triggers_ms.values(),
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError(f"decision {sample_id!r} has non-finite numeric data")
        if latency_ms < 0.0:
            raise ValueError(f"decision {sample_id!r} has negative latency")
        if kv_cost_ms <= 0.0 or threshold_ms <= 0.0:
            raise ValueError(f"decision {sample_id!r} has non-positive cost/threshold")
        for policy, trigger_ms in triggers_ms.items():
            if not 0.0 <= trigger_ms <= threshold_ms:
                raise ValueError(
                    f"decision {sample_id!r} has invalid {policy} trigger "
                    f"{trigger_ms}; expected [0, {threshold_ms}]"
                )

        label = row["label_exceeds_threshold"]
        if label != (latency_ms > threshold_ms):
            raise ValueError(f"decision {sample_id!r} has an inconsistent label")
        if not math.isclose(
            triggers_ms["deadline_only"],
            threshold_ms,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"decision {sample_id!r} has a non-deadline anchor")
        return cls(
            sample_id=sample_id,
            task_id=task_id,
            tool_name=tool_name,
            latency_ms=latency_ms,
            kv_cost_ms=kv_cost_ms,
            threshold_ms=threshold_ms,
            label_exceeds_threshold=label,
            triggers_ms=triggers_ms,
        )


def load_decision_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Load JSONL decision rows without dropping serialized fields."""
    if not paths:
        raise ValueError("at least one decisions path is required")
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"decisions file does not exist: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                rows.append(value)
    if not rows:
        raise ValueError("decisions inputs contain no rows")
    return rows


def analyze_utility_headroom(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_costs_ms: Sequence[float],
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Compute exact deadline headroom and task-cluster bootstrap intervals."""
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    expected_costs = tuple(float(cost) for cost in expected_costs_ms)
    if not expected_costs or len(set(expected_costs)) != len(expected_costs):
        raise ValueError("expected_costs_ms must be non-empty and unique")
    if not all(math.isfinite(cost) and cost > 0.0 for cost in expected_costs):
        raise ValueError("expected_costs_ms must contain positive finite values")

    decisions = [UtilityDecision.from_mapping(row) for row in rows]
    by_cost = _validate_decision_panel(decisions, expected_costs=expected_costs)
    rng = np.random.default_rng(bootstrap_seed)
    points: dict[str, Any] = {}
    for cost_ms in sorted(by_cost):
        point = _analyze_cost(
            by_cost[cost_ms],
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
            rng=rng,
        )
        points[str(cost_ms)] = point

    return {
        "schema_version": 1,
        "row_count": len(decisions),
        "sample_count": len({row.sample_id for row in decisions}),
        "task_count": len({row.task_id for row in decisions}),
        "costs_ms": sorted(by_cost),
        "bootstrap": {
            "unit": "task_id",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "confidence_level": confidence_level,
            "interval": "percentile",
        },
        "points": points,
    }


def _validate_decision_panel(
    decisions: Sequence[UtilityDecision],
    *,
    expected_costs: Sequence[float],
) -> dict[float, list[UtilityDecision]]:
    if not decisions:
        raise ValueError("decision rows must be non-empty")
    expected_cost_set = set(expected_costs)
    actual_cost_set = {row.kv_cost_ms for row in decisions}
    if actual_cost_set != expected_cost_set:
        raise ValueError(
            f"decision costs {sorted(actual_cost_set)} do not match expected "
            f"{sorted(expected_cost_set)}"
        )

    seen: set[tuple[str, float]] = set()
    sample_panels: dict[str, list[UtilityDecision]] = defaultdict(list)
    by_cost: dict[float, list[UtilityDecision]] = defaultdict(list)
    for row in decisions:
        key = (row.sample_id, row.kv_cost_ms)
        if key in seen:
            raise ValueError(f"duplicate sample/cost decision: {key}")
        seen.add(key)
        if not math.isclose(
            row.threshold_ms,
            row.kv_cost_ms,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"decision {row.sample_id!r} is not from the fixed zero-guard probe"
            )
        sample_panels[row.sample_id].append(row)
        by_cost[row.kv_cost_ms].append(row)

    for sample_id, panel in sample_panels.items():
        if {row.kv_cost_ms for row in panel} != expected_cost_set:
            raise ValueError(f"sample {sample_id!r} is missing one or more costs")
        reference = panel[0]
        for row in panel[1:]:
            if (
                row.task_id != reference.task_id
                or row.tool_name != reference.tool_name
                or not math.isclose(
                    row.latency_ms,
                    reference.latency_ms,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise ValueError(f"sample {sample_id!r} metadata changes across costs")
    return dict(by_cost)


def _analyze_cost(
    rows: Sequence[UtilityDecision],
    *,
    bootstrap_replicates: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cost point contains no decisions")
    cost_ms = rows[0].kv_cost_ms
    threshold_ms = rows[0].threshold_ms
    if any(row.threshold_ms != threshold_ms for row in rows):
        raise ValueError(f"cost {cost_ms} has inconsistent thresholds")

    positive_rows = [row for row in rows if row.label_exceeds_threshold]
    band_rows = [
        row for row in positive_rows if row.latency_ms < threshold_ms + cost_ms
    ]
    far_tail_rows = [
        row for row in positive_rows if row.latency_ms >= threshold_ms + cost_ms
    ]
    oracle_ms = len(positive_rows) * cost_ms
    headroom_ms = math.fsum(
        2.0 * (cost_ms - (row.latency_ms - threshold_ms)) for row in band_rows
    )
    policy_net_ms = {
        policy: math.fsum(
            trigger_policy_utility_ms(
                row.latency_ms,
                row.triggers_ms[policy],
                threshold_ms=row.threshold_ms,
                kv_cost_ms=row.kv_cost_ms,
            )
            for row in rows
        )
        for policy in POLICY_TRIGGER_FIELDS
    }
    deadline_ms = policy_net_ms["deadline_only"]
    identity_residual_ms = oracle_ms - deadline_ms - headroom_ms
    if not math.isclose(
        identity_residual_ms,
        0.0,
        rel_tol=1e-12,
        abs_tol=1e-6,
    ):
        raise AssertionError(
            f"deadline headroom identity failed at cost {cost_ms}: "
            f"residual={identity_residual_ms}ms"
        )

    policy_metrics = {
        policy: _policy_metrics(
            net_ms=net_ms,
            deadline_ms=deadline_ms,
            oracle_ms=oracle_ms,
            headroom_ms=headroom_ms,
        )
        for policy, net_ms in policy_net_ms.items()
    }
    bootstrap = _task_cluster_bootstrap(
        rows,
        replicates=bootstrap_replicates,
        confidence_level=confidence_level,
        rng=rng,
    )
    return {
        "kv_cost_ms": cost_ms,
        "threshold_ms": threshold_ms,
        "call_count": len(rows),
        "task_count": len({row.task_id for row in rows}),
        "positive_count": len(positive_rows),
        "band_count": len(band_rows),
        "far_tail_count": len(far_tail_rows),
        "band_fraction_of_calls": _safe_ratio(len(band_rows), len(rows)),
        "band_fraction_of_positives": _safe_ratio(len(band_rows), len(positive_rows)),
        "oracle_ms": oracle_ms,
        "deadline_headroom_ms": headroom_ms,
        "rho_headroom_over_oracle": _safe_ratio(headroom_ms, oracle_ms),
        "identity_residual_ms": identity_residual_ms,
        "policies": policy_metrics,
        "task_cluster_bootstrap": bootstrap,
    }


def _policy_metrics(
    *,
    net_ms: float,
    deadline_ms: float,
    oracle_ms: float,
    headroom_ms: float,
) -> dict[str, float | None]:
    delta_ms = net_ms - deadline_ms
    return {
        "net_saved_ms": net_ms,
        "delta_vs_deadline_ms": delta_ms,
        "captured_fraction_of_deadline_headroom": _safe_ratio(delta_ms, headroom_ms),
        "remaining_gap_to_oracle_ms": oracle_ms - net_ms,
        "net_fraction_of_oracle": _safe_ratio(net_ms, oracle_ms),
    }


def _task_cluster_bootstrap(
    rows: Sequence[UtilityDecision],
    *,
    replicates: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    task_ids = sorted({row.task_id for row in rows})
    task_index = {task_id: index for index, task_id in enumerate(task_ids)}
    # Columns: oracle, headroom, deadline, mean_hazard, robust_clock.
    task_totals = np.zeros((len(task_ids), 5), dtype=float)
    for row in rows:
        index = task_index[row.task_id]
        if row.label_exceeds_threshold:
            task_totals[index, 0] += row.kv_cost_ms
            if row.latency_ms < row.threshold_ms + row.kv_cost_ms:
                task_totals[index, 1] += 2.0 * (
                    row.kv_cost_ms - (row.latency_ms - row.threshold_ms)
                )
        for column, policy in enumerate(POLICY_TRIGGER_FIELDS, start=2):
            task_totals[index, column] += trigger_policy_utility_ms(
                row.latency_ms,
                row.triggers_ms[policy],
                threshold_ms=row.threshold_ms,
                kv_cost_ms=row.kv_cost_ms,
            )

    weights = rng.multinomial(
        len(task_ids),
        np.full(len(task_ids), 1.0 / len(task_ids)),
        size=replicates,
    )
    totals = weights @ task_totals
    oracle = totals[:, 0]
    headroom = totals[:, 1]
    deadline = totals[:, 2]
    mean_delta = totals[:, 3] - deadline
    robust_delta = totals[:, 4] - deadline
    return {
        "deadline_headroom_ms": _percentile_interval(
            headroom, confidence_level=confidence_level
        ),
        "rho_headroom_over_oracle": _percentile_interval(
            _finite_ratio(headroom, oracle), confidence_level=confidence_level
        ),
        "mean_hazard_delta_vs_deadline_ms": _percentile_interval(
            mean_delta, confidence_level=confidence_level
        ),
        "robust_clock_delta_vs_deadline_ms": _percentile_interval(
            robust_delta, confidence_level=confidence_level
        ),
        "mean_hazard_captured_fraction": _percentile_interval(
            _finite_ratio(mean_delta, headroom), confidence_level=confidence_level
        ),
        "robust_clock_captured_fraction": _percentile_interval(
            _finite_ratio(robust_delta, headroom), confidence_level=confidence_level
        ),
    }


def _finite_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    mask = denominator != 0.0
    return numerator[mask] / denominator[mask]


def _percentile_interval(
    samples: np.ndarray,
    *,
    confidence_level: float,
) -> dict[str, float | int | None]:
    if samples.size == 0:
        return {
            "lower": None,
            "upper": None,
            "valid_replicates": 0,
        }
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(samples, [tail, 1.0 - tail])
    return {
        "lower": float(lower),
        "upper": float(upper),
        "valid_replicates": int(samples.size),
    }


def _safe_ratio(numerator: float | int, denominator: float | int) -> float | None:
    return float(numerator / denominator) if denominator else None


__all__ = [
    "POLICY_TRIGGER_FIELDS",
    "UtilityDecision",
    "analyze_utility_headroom",
    "load_decision_rows",
]
