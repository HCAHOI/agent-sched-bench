"""Risk-coverage sweep of probability cutoffs for KV swap decisions.

The swap decision at tool start is asymmetric. Swapping when the tool
returns early stalls the next LLM call by the unfinished swap remainder
(exposed latency), while declining when the window was long enough only
forgoes hideable swap time (missed opportunity). A fixed
``probability_cutoff = 0.5`` implicitly assumes those costs are equal, so
this sweep re-thresholds the causal survival probabilities across cutoffs
and reports, per (KV cost, cutoff) point, the realized coverage, stall
rate, and exposed / absorbed / missed milliseconds. The cutoff is then
chosen from the measured cost trade-off instead of an assumed one.

Rows without causal history (cold starts) always decline to swap - the
conservative default, since declining cannot stall the next LLM call -
and are reported separately.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

from trace_collect.latency_validation import normalized_positive_floats
from trace_collect.tool_latency_dataset import read_tool_latency_jsonl
from trace_collect.tool_latency_threshold import evaluate_latency_thresholds


def evaluate_swap_cutoff_sweep(
    rows: Iterable[dict[str, Any]],
    *,
    kv_costs_ms: Iterable[float],
    guard_ms: float,
    probability_cutoffs: Iterable[float],
    min_tool_history: int = 1,
) -> dict[str, Any]:
    """Sweep swap-decision cutoffs over causal survival probabilities.

    For each KV cost ``c`` the decision threshold is ``c + guard_ms`` and the
    policy swaps when the causal survival probability of the current tool
    action meets the cutoff. Costs per decision:

    * exposed: ``max(0, c - latency)`` when swapping (stall on the critical
      path; the guard region absorbs early returns within it),
    * absorbed: ``min(c, latency)`` when swapping (swap time hidden behind
      the tool),
    * missed: ``c`` when declining (or cold-starting) on a row whose latency
      exceeded the threshold.

    ``absorbed_if_oracle_ms`` matches the kv_profile_sweep convention: an
    oracle that swaps exactly on rows exceeding the threshold, not the
    cost-minimizing optimum. A permissive policy can therefore absorb more
    raw milliseconds than this oracle: partially on unsafe rows below the KV
    cost (paying exposed milliseconds), and in full at zero exposure on
    unsafe rows inside the guard region (``kv_cost <= latency <= threshold``).
    """

    kv_costs = normalized_positive_floats(kv_costs_ms, label="kv cost")
    if not math.isfinite(guard_ms) or guard_ms < 0.0:
        raise ValueError(f"guard_ms must be finite and non-negative, got {guard_ms}")
    cutoffs = sorted({float(value) for value in probability_cutoffs})
    if not cutoffs:
        raise ValueError("at least one probability cutoff is required")
    for cutoff in cutoffs:
        if not math.isfinite(cutoff) or not 0.0 <= cutoff <= 1.0:
            raise ValueError(
                f"probability cutoffs must be finite and in [0, 1], got {cutoff}"
            )

    kv_cost_by_threshold = {kv_cost + guard_ms: kv_cost for kv_cost in kv_costs}
    inner = evaluate_latency_thresholds(
        rows,
        thresholds_ms=kv_cost_by_threshold.keys(),
        min_tool_history=min_tool_history,
    )
    if inner["row_count"] == 0:
        raise ValueError("no latency rows supplied")
    decisions_by_threshold: dict[float, list[dict[str, Any]]] = {}
    for decision in inner["decisions"]:
        decisions_by_threshold.setdefault(decision["threshold_ms"], []).append(decision)

    sweep = [
        _sweep_point(
            decisions_by_threshold[threshold_ms],
            kv_cost_ms=kv_cost_ms,
            guard_ms=guard_ms,
            threshold_ms=threshold_ms,
            cutoff=cutoff,
        )
        for threshold_ms, kv_cost_ms in sorted(kv_cost_by_threshold.items())
        for cutoff in cutoffs
    ]

    return {
        "predictor": inner["predictor"],
        "kv_costs_ms": kv_costs,
        "guard_ms": guard_ms,
        "probability_cutoffs": cutoffs,
        "min_tool_history": min_tool_history,
        "cold_start_policy": "no_swap",
        "row_count": inner["row_count"],
        "sweep": sweep,
    }


def load_and_evaluate_swap_cutoff_sweep(
    path: Path,
    *,
    kv_costs_ms: Iterable[float],
    guard_ms: float,
    probability_cutoffs: Iterable[float],
    min_tool_history: int = 1,
) -> dict[str, Any]:
    return evaluate_swap_cutoff_sweep(
        read_tool_latency_jsonl(path),
        kv_costs_ms=kv_costs_ms,
        guard_ms=guard_ms,
        probability_cutoffs=probability_cutoffs,
        min_tool_history=min_tool_history,
    )


def _sweep_point(
    decisions: list[dict[str, Any]],
    *,
    kv_cost_ms: float,
    guard_ms: float,
    threshold_ms: float,
    cutoff: float,
) -> dict[str, Any]:
    evaluated = [d for d in decisions if d["probability_exceeds_threshold"] is not None]
    cold_start = [d for d in decisions if d["probability_exceeds_threshold"] is None]
    swaps = [d for d in evaluated if d["probability_exceeds_threshold"] >= cutoff]
    declines = [d for d in evaluated if d["probability_exceeds_threshold"] < cutoff]

    tp = sum(d["label_exceeds_threshold"] for d in swaps)
    fp = len(swaps) - tp
    fn = sum(d["label_exceeds_threshold"] for d in declines)
    tn = len(declines) - fn

    exposed_ms = sum(max(0.0, kv_cost_ms - d["latency_ms"]) for d in swaps)
    absorbed_ms = sum(min(kv_cost_ms, d["latency_ms"]) for d in swaps)
    missed_positive_count = fn + sum(d["label_exceeds_threshold"] for d in cold_start)
    positive_count = sum(d["label_exceeds_threshold"] for d in decisions)

    return {
        "kv_cost_ms": kv_cost_ms,
        "guard_ms": guard_ms,
        "threshold_ms": threshold_ms,
        "probability_cutoff": cutoff,
        "row_count": len(decisions),
        "evaluated_count": len(evaluated),
        "cold_start_count": len(cold_start),
        "positive_count": positive_count,
        "swap_count": len(swaps),
        "coverage": _safe_div(len(swaps), len(evaluated)),
        "true_positive_count": tp,
        "false_positive_count": fp,
        "true_negative_count": tn,
        "false_negative_count": fn,
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "stall_rate": _safe_div(fp, len(swaps)),
        "exposed_ms_total": exposed_ms,
        "absorbed_ms_total": absorbed_ms,
        "missed_ms_total": missed_positive_count * kv_cost_ms,
        "absorbed_if_oracle_ms": positive_count * kv_cost_ms,
    }


def _safe_div(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


__all__ = [
    "evaluate_swap_cutoff_sweep",
    "load_and_evaluate_swap_cutoff_sweep",
]
