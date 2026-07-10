"""Additive per-segment command cost model for tool latency.

A compound shell command executes its sequential segments one after
another, so its latency is (approximately) the sum of per-segment costs:
``latency(cmd) ~= sum over sequential segments of c(head(segment))`` with
``c >= 0``. Costs are fit jointly over the profile split with non-negative
least squares - segments almost never appear standalone in agent traces
(e.g. ``cd`` never does), so per-segment costs must be attributed from
compound observations; they are identifiable because segments co-occur in
varying combinations. Preamble transparency (``cd``, ``export``, ...) is
therefore *learned* as a near-zero coefficient, not declared.

The model is deliberately used only for two ranking/shift roles, never as
the latency predictor itself:

* pick a command's *dominant* segment (largest fitted cost), whose
  empirical prefix-node distribution supplies the survival estimate, and
* *deduct* the other segments' fitted costs from the decision threshold,
  so the dominant node is queried at the time budget that actually remains
  for it.

Costs are fit at segment-head granularity (``tool:head``) for
identifiability; the dominant segment's distribution still comes from its
full depth-capped prefix nodes. Two established estimators are offered:
``nnls`` minimizes squared error, so heavy-tail rows dominate the fit
(e.g. a ubiquitous preamble can absorb shared variance from minute-long
outliers); ``lad`` is non-negative median regression (least absolute
deviations via the standard linear-programming encoding, solved with
HiGHS), which is outlier-robust and parameter-free. Fitted costs only
rank segments and shift thresholds either way - distributions stay
empirical.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np
from scipy import sparse
from scipy.optimize import linprog, nnls

from trace_collect.command_features import shell_command_segments
from trace_collect.latency_validation import required_nonnegative_float, required_text

_FIT_METHODS = ("nnls", "lad")


@dataclass(frozen=True)
class SegmentCostModel:
    """Fitted non-negative additive segment costs, keyed by ``tool:head``."""

    costs_by_head: dict[str, float]
    counts_by_head: dict[str, int]
    fitted_row_count: int
    residual_rms_ms: float

    def head_cost(self, tool_name: str, head: str, *, min_head_count: int) -> float | None:
        """Fitted cost of one segment head, or None below the evidence gate."""

        key = f"{tool_name}:{head}"
        if self.counts_by_head.get(key, 0) < min_head_count:
            return None
        return self.costs_by_head[key]

    def dominant_segment_and_deduction(
        self,
        tool_name: str,
        command: str,
        *,
        min_head_count: int,
    ) -> tuple[list[str] | None, float]:
        """Dominant segment tokens and the other segments' summed costs.

        Returns ``(None, 0.0)`` when the command has no segment with an
        adequately observed head - the caller falls back to whole-command
        keying. Single-segment commands are their own dominant segment with
        zero deduction. Unknown-head segments contribute no deduction (a
        conservative under-deduction).
        """

        segments = shell_command_segments(command)
        if not segments:
            return None, 0.0
        if len(segments) == 1:
            return segments[0], 0.0
        costs = [
            self.head_cost(tool_name, segment[0], min_head_count=min_head_count)
            for segment in segments
        ]
        known = [
            (cost, index) for index, cost in enumerate(costs) if cost is not None
        ]
        if not known:
            return None, 0.0
        _, dominant_index = max(known, key=lambda pair: (pair[0], -pair[1]))
        deduction = sum(
            cost
            for index, cost in enumerate(costs)
            if index != dominant_index and cost is not None
        )
        return segments[dominant_index], deduction

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "head_count": len(self.costs_by_head),
            "fitted_row_count": self.fitted_row_count,
            "residual_rms_ms": self.residual_rms_ms,
        }


def fit_segment_cost_model(
    rows: Iterable[dict[str, Any]],
    *,
    command_field: str,
    fit_method: str = "nnls",
) -> SegmentCostModel:
    """Fit non-negative additive segment-head costs on profile rows.

    Rows without a parseable command under ``command_field`` are skipped
    (they carry no segment structure). Raises when no command rows exist.
    ``fit_method`` selects the estimator: ``nnls`` (squared error) or
    ``lad`` (median regression, outlier-robust).
    """

    if fit_method not in _FIT_METHODS:
        raise ValueError(
            f"unknown fit_method {fit_method!r}; choose one of {', '.join(_FIT_METHODS)}"
        )

    row_heads: list[list[str]] = []
    latencies: list[float] = []
    head_index: dict[str, int] = {}
    counts: dict[str, int] = {}
    for index, row in enumerate(rows):
        tool_args = row.get("tool_args")
        if not isinstance(tool_args, dict):
            continue
        command = tool_args.get(command_field)
        if not isinstance(command, str) or not command.strip():
            continue
        segments = shell_command_segments(command)
        if not segments:
            continue
        source = f"profile row {index}"
        tool_name = required_text(row, "tool_name", source=source)
        latency_ms = required_nonnegative_float(row, "latency_ms", source=source)
        heads = [f"{tool_name}:{segment[0]}" for segment in segments]
        row_heads.append(heads)
        latencies.append(latency_ms)
        for head in heads:
            head_index.setdefault(head, len(head_index))
        # Evidence gate counts observing commands, not occurrences: a head
        # repeated within one command is still one observation of it.
        for head in set(heads):
            counts[head] = counts.get(head, 0) + 1
    if not row_heads:
        raise ValueError("no command rows to fit segment costs from")

    design = np.zeros((len(row_heads), len(head_index)))
    for row_number, heads in enumerate(row_heads):
        for head in heads:
            design[row_number, head_index[head]] += 1.0
    target = np.asarray(latencies)
    if fit_method == "nnls":
        costs, residual_norm = nnls(design, target)
        residual_rms = float(residual_norm) / math.sqrt(len(row_heads))
    else:
        costs = _nonnegative_lad(design, target)
        # L2 RMS is reported for comparability across fit methods, but for
        # lad it is a diagnostic dominated by the deliberately un-fit tail
        # rows - it is not the fitted loss.
        residual_rms = float(np.sqrt(np.mean((design @ costs - target) ** 2)))
    return SegmentCostModel(
        costs_by_head={head: float(costs[column]) for head, column in head_index.items()},
        counts_by_head=counts,
        fitted_row_count=len(row_heads),
        residual_rms_ms=residual_rms,
    )


def _nonnegative_lad(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Non-negative least-absolute-deviations fit via the standard LP encoding.

    Minimize ``sum(u + v)`` subject to ``design @ c + u - v = target`` with
    ``c, u, v >= 0`` - the textbook LP form of median regression, solved
    with scipy's HiGHS backend. Parameter-free.
    """

    row_count, head_count = design.shape
    objective = np.concatenate(
        [np.zeros(head_count), np.ones(row_count), np.ones(row_count)]
    )
    identity = sparse.eye(row_count, format="csr")
    equality = sparse.hstack([sparse.csr_matrix(design), identity, -identity])
    result = linprog(
        objective,
        A_eq=equality,
        b_eq=target,
        bounds=(0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"LAD fit failed: {result.message}")
    return np.asarray(result.x[:head_count])


__all__ = ["SegmentCostModel", "fit_segment_cost_model"]
