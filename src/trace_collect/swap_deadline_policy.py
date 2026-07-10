"""Deadline-triggered KV swap policy evaluation.

The t=0 survival decision only determines how *early* hiding starts: if it
says "no swap" (including cold starts) and the tool is still running at
elapsed ``k = threshold``, the policy starts the swap then. At ``k`` the
exceedance label is proven - the call has already run longer than the
threshold - so deadline swaps are never wrong; their only cost is the
reduced remaining window ``latency - k`` available for hiding. Missed
opportunities are structurally zero: every call longer than the threshold
is swapped either immediately or at the deadline. The re-check time is the
threshold itself, not a tuned parameter.

For each KV cost this module reports both policies side by side:

* ``t0_only`` - the plain threshold decision (the existing method),
* ``deadline_recheck`` - the same t=0 decisions plus the deadline swap.

``absorbed_on_long_ms_total`` counts hiding achieved on calls truly longer
than the threshold (each contributes at most one KV cost, so its
``hidden_fraction_of_oracle`` is <= 1). ``absorbed_ms_total`` additionally
includes partial hides on early-returning swaps, mirroring
swap_cutoff_sweep's accounting; ``exposed_ms_total`` counts all stall time
regardless of label. The Wilson abstain band is deliberately not exposed
here: the deadline re-check already converts uncertain "no swap" decisions
into proven late swaps, which dominates abstention.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

from trace_collect.classification_metrics import safe_div
from trace_collect.latency_validation import normalized_positive_floats
from trace_collect.tool_latency_dataset import read_tool_latency_jsonl
from trace_collect.tool_latency_profiled import evaluate_profiled_latency_thresholds


def evaluate_deadline_policy(
    eval_rows: Iterable[dict[str, Any]],
    *,
    profile_rows: Iterable[dict[str, Any]],
    kv_costs_ms: Iterable[float],
    guard_ms: float,
    predictor: str,
    probability_cutoff: float = 0.5,
    prior_strength: float | None = None,
    min_tool_history: int = 1,
    command_field: str | None = None,
    max_prefix_depth: int = 4,
    skip_leading_cd: bool = False,
    segment_costs: bool = False,
    segment_fit: str = "nnls",
) -> dict[str, Any]:
    """Compare the t0-only and deadline-recheck swap policies per KV cost."""

    kv_costs = normalized_positive_floats(kv_costs_ms, label="kv cost")
    if not math.isfinite(guard_ms) or guard_ms < 0.0:
        raise ValueError(f"guard_ms must be finite and non-negative, got {guard_ms}")

    kv_cost_by_threshold = {kv_cost + guard_ms: kv_cost for kv_cost in kv_costs}
    inner = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=profile_rows,
        thresholds_ms=kv_cost_by_threshold.keys(),
        predictor=predictor,
        prior_strength=prior_strength,
        probability_cutoff=probability_cutoff,
        min_tool_history=min_tool_history,
        command_field=command_field,
        max_prefix_depth=max_prefix_depth,
        skip_leading_cd=skip_leading_cd,
        segment_costs=segment_costs,
        segment_fit=segment_fit,
    )
    decisions_by_threshold: dict[float, list[dict[str, Any]]] = {}
    for decision in inner["decisions"]:
        decisions_by_threshold.setdefault(decision["threshold_ms"], []).append(decision)

    points = [
        _policy_point(
            decisions_by_threshold[threshold_ms],
            kv_cost_ms=kv_cost_ms,
            guard_ms=guard_ms,
            threshold_ms=threshold_ms,
        )
        for threshold_ms, kv_cost_ms in sorted(kv_cost_by_threshold.items())
    ]

    return {
        "predictor": inner["predictor"],
        "probability_cutoff": probability_cutoff,
        "prior_strength": prior_strength,
        "min_tool_history": min_tool_history,
        "command_field": command_field,
        "max_prefix_depth": inner["max_prefix_depth"],
        "skip_leading_cd": inner["skip_leading_cd"],
        "segment_costs": inner["segment_costs"],
        "segment_fit": inner["segment_fit"],
        "segment_cost_model": inner["segment_cost_model"],
        "kv_costs_ms": kv_costs,
        "guard_ms": guard_ms,
        "recheck_at": "threshold",
        "row_count": inner["row_count"],
        "profile_row_count": inner["profile_row_count"],
        "profile_trace_count": inner["profile_trace_count"],
        "points": points,
    }


def load_and_evaluate_deadline_policy(
    eval_path: Path,
    *,
    profile_path: Path,
    kv_costs_ms: Iterable[float],
    guard_ms: float,
    predictor: str,
    probability_cutoff: float = 0.5,
    prior_strength: float | None = None,
    min_tool_history: int = 1,
    command_field: str | None = None,
    max_prefix_depth: int = 4,
    skip_leading_cd: bool = False,
    segment_costs: bool = False,
    segment_fit: str = "nnls",
) -> dict[str, Any]:
    return evaluate_deadline_policy(
        read_tool_latency_jsonl(eval_path),
        profile_rows=read_tool_latency_jsonl(profile_path),
        kv_costs_ms=kv_costs_ms,
        guard_ms=guard_ms,
        predictor=predictor,
        probability_cutoff=probability_cutoff,
        prior_strength=prior_strength,
        min_tool_history=min_tool_history,
        command_field=command_field,
        max_prefix_depth=max_prefix_depth,
        skip_leading_cd=skip_leading_cd,
        segment_costs=segment_costs,
        segment_fit=segment_fit,
    )


def _policy_point(
    decisions: list[dict[str, Any]],
    *,
    kv_cost_ms: float,
    guard_ms: float,
    threshold_ms: float,
) -> dict[str, Any]:
    positive_count = sum(d["label_exceeds_threshold"] for d in decisions)
    cold_start_count = sum(
        d["predicted_exceeds_threshold"] is None for d in decisions
    )
    oracle_ms = positive_count * kv_cost_ms

    t0_only = _accumulate_policy(
        decisions,
        kv_cost_ms=kv_cost_ms,
        recheck_at_ms=None,
        oracle_ms=oracle_ms,
    )
    deadline = _accumulate_policy(
        decisions,
        kv_cost_ms=kv_cost_ms,
        recheck_at_ms=threshold_ms,
        oracle_ms=oracle_ms,
    )

    return {
        "kv_cost_ms": kv_cost_ms,
        "guard_ms": guard_ms,
        "threshold_ms": threshold_ms,
        "row_count": len(decisions),
        "positive_count": positive_count,
        "cold_start_count": cold_start_count,
        "absorbed_if_oracle_ms": oracle_ms,
        "policies": {
            "t0_only": t0_only,
            "deadline_recheck": deadline,
        },
    }


def _accumulate_policy(
    decisions: list[dict[str, Any]],
    *,
    kv_cost_ms: float,
    recheck_at_ms: float | None,
    oracle_ms: float,
) -> dict[str, Any]:
    swap_count = 0
    deadline_swap_count = 0
    absorbed_ms = 0.0
    absorbed_on_long_ms = 0.0
    exposed_ms = 0.0
    missed_positive_count = 0
    for decision in decisions:
        latency_ms = decision["latency_ms"]
        label = decision["label_exceeds_threshold"]
        if decision["predicted_exceeds_threshold"] is True:
            swap_count += 1
            hidden = min(kv_cost_ms, latency_ms)
            absorbed_ms += hidden
            if label:
                absorbed_on_long_ms += hidden
            exposed_ms += max(0.0, kv_cost_ms - latency_ms)
        elif recheck_at_ms is not None and latency_ms > recheck_at_ms:
            # The tool outlived the deadline, so the label is proven long.
            deadline_swap_count += 1
            remaining_ms = latency_ms - recheck_at_ms
            hidden = min(kv_cost_ms, remaining_ms)
            absorbed_ms += hidden
            absorbed_on_long_ms += hidden
            exposed_ms += max(0.0, kv_cost_ms - remaining_ms)
        elif label:
            missed_positive_count += 1
    return {
        "swap_count": swap_count,
        "deadline_swap_count": deadline_swap_count,
        "absorbed_ms_total": absorbed_ms,
        "absorbed_on_long_ms_total": absorbed_on_long_ms,
        "exposed_ms_total": exposed_ms,
        "missed_ms_total": missed_positive_count * kv_cost_ms,
        "hidden_fraction_of_oracle": safe_div(absorbed_on_long_ms, oracle_ms),
    }


__all__ = [
    "evaluate_deadline_policy",
    "load_and_evaluate_deadline_policy",
]
