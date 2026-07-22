"""CPU-only pre-restore stopping-time optimization.

``E[u(s)]`` over a node's samples is piecewise linear, with endpoints and jumps
at ``{g} union {L - R > g} union {L > g}``. Enumerating that grid is exact.
"""

from __future__ import annotations

import numpy as np

from trace_collect.tool_latency_profiled import LatencyPriorNode


def _expected_prerestore_utility(
    samples: np.ndarray, start_ms: float, restore_cost_ms: float
) -> float:
    """Mean per-call pre-restore utility of a start over a node sample set."""

    lead = samples - start_ms
    fires = lead > 0.0
    hidden = np.where(fires & (lead <= restore_cost_ms), lead, 0.0)
    wasted = np.where(fires & (lead > restore_cost_ms), restore_cost_ms, 0.0)
    return float(np.mean(hidden - wasted))


def prerestore_start_ms(
    node: LatencyPriorNode, *, swap_trigger_ms: float, restore_cost_ms: float
) -> float | None:
    """Exact-optimal pre-restore start ``s* >= swap_trigger_ms``, or ``None``.

    ``None`` = no pre-restore (the conservative floor): returned when there is
    nothing to overlap (``R <= 0``), the node is thin/degenerate (``< 2`` logical
    tasks, mirroring ``robust_utility_trigger_stats``), or no start beats the
    zero-utility no-fire baseline. The candidate grid ``{g} u {L - R > g} u
    {L > g}`` is exact for the piecewise-linear objective (module docstring);
    ties resolve to the LATEST (smallest-window, least-speculative) start.
    """

    if restore_cost_ms <= 0.0:
        return None
    if len(node.values_by_task) < 2:
        return None
    values = node.values
    if not values:
        return None
    samples = np.asarray(values, dtype=float)
    candidates = {float(swap_trigger_ms)}
    for latency in values:
        edge = latency - restore_cost_ms
        if edge > swap_trigger_ms:
            candidates.add(float(edge))
        if latency > swap_trigger_ms:
            candidates.add(float(latency))
    best_start: float | None = None
    best_utility = 0.0  # no-pre-restore floor
    for start in sorted(candidates):
        utility = _expected_prerestore_utility(samples, start, restore_cost_ms)
        if utility > 0.0 and utility >= best_utility:
            best_utility = utility
            best_start = start
    return best_start
