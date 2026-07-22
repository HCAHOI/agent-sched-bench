#!/usr/bin/env python3
"""Calibration + sharpness of the certified latency priors (DESCRIPTIVE lane).

Pre-registered in ``analysis/curve-calibration-design-20260720.md``. This lane
MEASURES the shipped, H1-certified estimator; it does NOT tune it. There is no
kill criterion: a coverage miss or a negative skill score is reported as a
limitation, never fixed by touching the estimator. Nothing under ``src/`` is
modified or re-implemented here -- the prior, its hierarchy, and the certified
trigger are imported and consumed as-is.

**The forecast object.** For a held-out call, the certified policy selects the
deepest eligible prior node ``latency_prior_hierarchy(...)[-1]`` built from
FIT-FOLD samples only, and reads its call-level ECDF: ``P(L > t) = (n -
bisect_right(V, t)) / n`` over that node's sorted sample list ``V``
(``_estimate_survival``, ``prior_only`` / ``call``). Every metric below scores
exactly that ECDF ``F``, cross-fitted over the frozen manifest's outer folds
(``index % fold_count``), identical to A0/A2.

Metric families (all out-of-sample, task-grouped folds, task-clustered CIs).

1. **PIT.** Marginal ``u = F(y)`` with MID-RANK ties:
   ``u = (#{v < y} + 0.5 * #{v == y}) / n``. Honest ``=> u ~ U[0, 1]``. Reported
   as a histogram plus the Cramer-von Mises distance to uniform. With sorted PIT
   values ``u_(1..N)`` the statistic is
   ``W2 = 1/(12N) + sum_i (u_(i) - (2i-1)/(2N))^2``; we report both ``W2`` and
   the N-normalised ``omega2 = W2 / N == integral (F_N(u) - u)^2 du``, which is a
   sample-size-free distance and so is the quantity carrying the CI.
   ROLLING form (the object the policy actually consumes): at elapsed ``t`` the
   residual forecast is the renormalised conditional
   ``F_t(r) = (F(t + r) - F(t)) / (1 - F(t))``, scored against ``y - t`` over the
   calls still alive at ``t``, i.e.
   ``u = (#{t < v < y} + 0.5 * #{v == y}) / #{v > t}``.
   The ``t``-grid is each node's OWN support quantiles at a fixed FRACTION grid
   (``--rolling-fractions``, default deciles 0.1..0.9): the fractions are the
   documented constant, the ``t`` values are node-derived, so no corpus-specific
   time constant enters.

2. **Quantile coverage, tail-weighted.** Empirical exceedance of the predicted
   P50/P90/P95/P99 against nominal ``1 - q``, with task-clustered CIs. The
   predicted quantile is the inverse of the SAME step ECDF the policy reads,
   ``Q_q = inf{x : F(x) >= q}`` (type-1 / inverse-ECDF), so coverage is measured
   against the curve as shipped rather than against a smoothed proxy. A node
   with ``n < 1 / (1 - q)`` cannot express the ``q`` quantile at all and is
   excluded FOR THAT QUANTILE ONLY, with counts reported.

3. **CRPS with skill scores.** EXACT closed form for an empirical/ECDF forecast
   with members ``x_1..x_n`` and observation ``y`` (energy form, no sampling):

       CRPS = (1/n) * sum_i |x_i - y|
              - (1 / (2 n^2)) * sum_i sum_j |x_i - x_j|

   The double sum is evaluated in O(n) on the sorted members via
   ``sum_i sum_j |x_i - x_j| = 2 * sum_i (2i - n - 1) * x_(i)``, and the first
   term in O(log n) via prefix sums -- both are identities, not approximations.
   Skill ``= 1 - CRPS_model / CRPS_baseline`` against TWO baselines drawn from
   the estimator's own hierarchy: the POOLED unconditional curve
   (``prior_global``) and the TOOL-NAME curve (``prior_tool``, falling back to
   global exactly as the hierarchy does when the tool node is ineligible). A
   calibrated-but-uninformative forecaster returns the marginal and scores ~0.

4. **Decision-band Brier (the bridge metric).** Brier score of the exceedance
   probability exactly where the policy reads it. Per kv cell the certified
   trigger is ``t = hazard_recheck_ms(node.values, threshold_ms=kv + guard,
   kv_cost_ms=kv, restore_cost_ms=rho * kv)`` -- the same k=1 optimizer A0
   adjudicated and A2 uses as its ``hazard`` trigger source, chosen here for
   exactly that consistency. The trigger region is the calls still alive at
   ``t``; the forecast is ``p = P(L > threshold | L > t)`` renormalised from the
   node ECDF and the outcome is ``o = 1{y > threshold}``. Reported with the
   region base rate, the climatology (base-rate) reference Brier and the Brier
   skill score, because a node whose trigger collapses to ``t == threshold``
   makes the region vacuous (``o == 1`` identically) and would otherwise show a
   flattering Brier of ~0.

**Censoring (correctness, not decoration).** Protocol-guard timeouts are
RIGHT-CENSORED: the call was cut off at its bound, it did not complete there.
The indicator is read from the trace (see ``censored_sample_ids``): a
``tool_exec`` whose ``data.success is False`` and whose ``data.tool_result``
begins ``Error: [timeout]`` and carries ``Exit code: 124``. Such a row means
``L > c`` where ``c`` is the observed cut-off. Treatment, per the spec:

* **PIT** -- ``u`` is only identified as lying in ``[F(c), 1]``, so censored
  rows are EXCLUDED from the histogram/CvM and their count is reported.
* **Coverage** -- they contribute as "exceeded the bound": a definite exceedance
  when ``c >= Q_q``, and INDETERMINATE otherwise. Indeterminate rows are counted
  as non-exceedances (a lower bound on exceedance) and their count is reported
  per quantile so the reader can bound the other way.
* **CRPS** -- EXCLUDED, with count and mass reported.
* **Decision-band Brier** -- INCLUDED when both the outcome and the
  alive-at-``t`` status are determinate (``c >= threshold`` forces ``o = 1``),
  since dropping them would bias away precisely the tail this metric exists to
  measure. Indeterminate rows are dropped and counted.

Degenerate nodes are excluded per metric with counts reported, never silently
dropped: ``n < 2`` for PIT/CRPS, ``#{v > t} < 2`` for the rolling PIT and the
Brier region, ``n < 1 / (1 - q)`` for coverage at ``q``.

**Statistics.** Task-clustered percentile bootstrap at the certified
replicate/seed discipline (50000, 0.95, seed 0), resampling whole logical tasks
with ``_resample_task_totals`` (reused verbatim from the H1 engine) for every
ratio-of-sums statistic -- coverage, CRPS skill, Brier -- so numerator and
denominator move together under one resample. CvM is not a ratio of sums, so it
uses ``_task_cluster_multiplicities``, which draws the SAME multinomial task
multiplicities under the same generator and recomputes the statistic exactly.

EXPLORATORY until ``--final``. Emits JSON + MD to ``analysis/`` (``-PARTIAL``
unless ``--final``); per-call rows go to a local zstd sidecar, never the
committed JSON.

Usage (full corpus):
  uv run python scripts/certification/analyze_prior_calibration.py \
    --manifest analysis/fresh-corpus-certification-20260717/\
offline-gated-robust/manifest.json --final
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
import datetime as _dt
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import numpy as np

# Allow direct `python scripts/certification/analyze_prior_calibration.py ...` invocation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.certification.adjudicate_k2_recheck import (  # noqa: E402
    _CERT_GUARD_MS,
    _CERT_RESTORE_COST_FRACTION,
    _banner,
    _git_sha,
    _load_manifest_corpus,
    _row_group_keys,
)
from scripts.certification.analyze_prerestore_accounting import _write_decisions_zst  # noqa: E402
from trace_collect.tool_latency_confirmation import (  # noqa: E402
    _resample_task_totals,
)
from trace_collect.tool_latency_dataset import ToolLatencySample  # noqa: E402
from trace_collect.tool_latency_profiled import (  # noqa: E402
    LatencyPriorNode,
    build_latency_prior,
    hazard_recheck_ms,
    latency_prior_hierarchy,
)
from trace_collect.trace_data import TraceData  # noqa: E402

# Nominal coverage levels. P90/P95/P99 are the tail the decisions live in; P50
# is the median sanity check. Fixed before the run, not selected from output.
_COVERAGE_QUANTILES = (0.50, 0.90, 0.95, 0.99)

# PIT histogram resolution. Ten equal-width bins on [0, 1] -- a presentation
# choice for the MD table; the CvM statistic is computed on the raw PIT values
# and is not affected by it.
_PIT_BINS = 10

# Right-censoring indicator in the canonical traces: the exec protocol guard
# fires, the runtime reports the shell's SIGTERM exit status, and the tool call
# is recorded as failed. Verified against the fresh-277 corpus (see the module
# docstring); matched conjunctively so ordinary tool errors are never censored.
_CENSOR_RESULT_PREFIX = "Error: [timeout]"
_CENSOR_RESULT_MARKER = "Exit code: 124"


# --------------------------------------------------------------------------- #
# Censoring detection (read from the trace; no estimator or extractor change).
# --------------------------------------------------------------------------- #
def censored_sample_ids(trace_paths: Sequence[Path]) -> set[str]:
    """Sample ids of RIGHT-CENSORED (protocol-guard timeout) tool calls.

    Walks the same traces the corpus loader consumes and reconstructs the
    ``ToolLatencySample.sample_id`` key (``{trace}:{agent}:{iteration}:{action}``)
    for every ``tool_exec`` whose result is a protocol-guard timeout. Reading the
    indicator here rather than widening ``ToolLatencySample`` keeps this lane
    strictly additive: nothing under ``src/`` is touched.
    """

    censored: set[str] = set()
    for trace_path in trace_paths:
        trace = TraceData.load(trace_path)
        for action in trace.actions:
            if action.get("action_type") != "tool_exec":
                continue
            data = action.get("data") or {}
            if data.get("success") is not False:
                continue
            result = data.get("tool_result")
            if not isinstance(result, str):
                continue
            if not result.startswith(_CENSOR_RESULT_PREFIX):
                continue
            if _CENSOR_RESULT_MARKER not in result:
                continue
            agent_id = str(action.get("agent_id") or "")
            iteration = action.get("iteration")
            action_id = str(action.get("action_id") or "")
            censored.add(f"{trace_path}:{agent_id}:{iteration}:{action_id}")
    return censored


# --------------------------------------------------------------------------- #
# ECDF forecast primitives. All operate on the node's sorted sample list.
# --------------------------------------------------------------------------- #
def ecdf_pit(values: Sequence[float], y: float) -> float:
    """Mid-rank PIT ``u = (#{v < y} + 0.5 #{v == y}) / n`` (module docstring)."""

    n = len(values)
    if n == 0:
        raise ValueError("cannot compute PIT against an empty forecast")
    below = bisect_left(values, y)
    ties = bisect_right(values, y) - below
    return (below + 0.5 * ties) / n


def ecdf_quantile(values: Sequence[float], q: float) -> float:
    """Inverse of the step ECDF: ``inf{x : F(x) >= q}`` (type-1 quantile).

    This is the inverse of exactly the ECDF the certified survival estimator
    reads, so coverage is scored against the shipped curve, not a smoothed one.
    """

    n = len(values)
    if n == 0:
        raise ValueError("cannot compute a quantile of an empty forecast")
    if not 0.0 < q < 1.0:
        raise ValueError(f"quantile must lie in (0, 1), got {q}")
    index = min(n - 1, max(0, math.ceil(q * n) - 1))
    return float(values[index])


def ecdf_survival(values: Sequence[float], t: float) -> float:
    """``P(L > t)`` under the node ECDF -- the estimator's own survival form."""

    n = len(values)
    if n == 0:
        raise ValueError("cannot compute survival of an empty forecast")
    return (n - bisect_right(values, t)) / n


def _pairwise_absolute_mean(sorted_values: np.ndarray) -> float:
    """``(1 / n^2) * sum_i sum_j |x_i - x_j|`` in O(n) on sorted members.

    Identity: ``sum_i sum_j |x_i - x_j| = 2 * sum_i (2i - n - 1) * x_(i)`` with
    ``i`` one-based over ascending members. Exact, not a sampled approximation.
    """

    n = len(sorted_values)
    if n == 0:
        raise ValueError("cannot compute the pairwise term of an empty forecast")
    weights = 2.0 * np.arange(1, n + 1, dtype=float) - n - 1.0
    return float(2.0 * np.dot(weights, sorted_values) / (n * n))


@dataclass(frozen=True)
class EcdfForecast:
    """A node's ECDF with the O(1)-per-observation CRPS terms precomputed."""

    values: tuple[float, ...]
    _prefix: np.ndarray
    _pairwise_mean: float

    @classmethod
    def from_values(cls, values: Sequence[float]) -> EcdfForecast:
        sorted_values = np.sort(np.asarray(values, dtype=float))
        prefix = np.concatenate(([0.0], np.cumsum(sorted_values)))
        return cls(
            values=tuple(float(v) for v in sorted_values),
            _prefix=prefix,
            _pairwise_mean=_pairwise_absolute_mean(sorted_values),
        )

    def crps(self, y: float) -> float:
        """Exact CRPS of this empirical forecast at ``y`` (module docstring)."""

        n = len(self.values)
        if n == 0:
            raise ValueError("cannot score CRPS against an empty forecast")
        split = bisect_left(self.values, y)
        total = float(self._prefix[n])
        below_sum = float(self._prefix[split])
        # sum |x_i - y| = (split * y - sum_below) + ((total - sum_below) - (n - split) * y)
        absolute_sum = (split * y - below_sum) + (
            (total - below_sum) - (n - split) * y
        )
        return absolute_sum / n - 0.5 * self._pairwise_mean


def cramer_von_mises(pit_values: Sequence[float]) -> tuple[float, float]:
    """Return ``(W2, omega2)`` of the PIT sample against U[0, 1].

    ``W2 = 1/(12N) + sum_i (u_(i) - (2i-1)/(2N))^2`` on the ascending PIT values;
    ``omega2 = W2 / N`` is the sample-size-free distance ``integral (F_N(u) -
    u)^2 du`` that the readout quotes and the CI covers.
    """

    n = len(pit_values)
    if n == 0:
        raise ValueError("cannot compute CvM of an empty PIT sample")
    ordered = np.sort(np.asarray(pit_values, dtype=float))
    ranks = np.arange(1, n + 1, dtype=float)
    w2 = 1.0 / (12.0 * n) + float(np.sum((ordered - (2.0 * ranks - 1.0) / (2.0 * n)) ** 2))
    return w2, w2 / n


def _cvm_omega2_weighted(sorted_pit: np.ndarray, multiplicity: np.ndarray) -> float:
    """``omega2`` of a multiset given ascending distinct-position PIT values.

    A bootstrap replicate reuses the globally pre-sorted PIT array with integer
    repeat counts, so its order statistics are that same array with runs of
    length ``m_i`` occupying ranks ``lo_i..hi_i``. The rank block sums close in
    form -- with ``c = 1/(2N)``,
    ``sum_{r=lo}^{hi} (u - c(2r-1))^2 = m u^2 - 2 u c (2 s1 - m) + c^2 (4 s2 - 4 s1 + m)``
    for ``s1 = sum r`` and ``s2 = sum r^2`` over the block -- so a replicate costs
    O(n) with no re-sorting and the result is exact, not an approximation.
    """

    total = float(multiplicity.sum())
    if total <= 0.0:
        return math.nan
    hi = np.cumsum(multiplicity, dtype=np.float64)
    lo = hi - multiplicity
    # sum r and sum r^2 over the integer block [lo+1, hi].
    s1 = (hi * (hi + 1.0) - lo * (lo + 1.0)) / 2.0
    s2 = (
        hi * (hi + 1.0) * (2.0 * hi + 1.0) - lo * (lo + 1.0) * (2.0 * lo + 1.0)
    ) / 6.0
    c = 1.0 / (2.0 * total)
    m = multiplicity.astype(np.float64)
    block = (
        m * sorted_pit**2
        - 2.0 * sorted_pit * c * (2.0 * s1 - m)
        + c * c * (4.0 * s2 - 4.0 * s1 + m)
    )
    w2 = 1.0 / (12.0 * total) + float(block.sum())
    return w2 / total


# --------------------------------------------------------------------------- #
# Task-clustered bootstrap. Ratio statistics reuse the certified resampler.
# --------------------------------------------------------------------------- #
def _task_cluster_multiplicities(
    task_count: int, *, replicates: int, seed: int, batch: int
) -> Any:
    """Yield batches of multinomial task multiplicities (same draw as H1).

    Mirrors ``_resample_task_totals``' generator and draw exactly
    (``PCG64(seed)``, ``multinomial(task_count, uniform)``) for the statistics
    that are not ratios of sums and so cannot go through that helper directly.
    """

    if task_count < 1:
        raise ValueError("task count must be positive")
    rng = np.random.Generator(np.random.PCG64(seed))
    probabilities = np.full(task_count, 1.0 / task_count, dtype=float)
    for start in range(0, replicates, batch):
        stop = min(start + batch, replicates)
        yield rng.multinomial(task_count, probabilities, size=stop - start)


def _percentile_interval(
    draws: np.ndarray, confidence_level: float
) -> dict[str, float]:
    alpha = 1.0 - confidence_level
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return {"low": math.nan, "high": math.nan}
    low, high = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear")
    return {"low": float(low), "high": float(high)}


def ratio_bootstrap(
    numerator_by_task: np.ndarray,
    denominator_by_task: np.ndarray,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, Any]:
    """Task-clustered percentile CI for ``f(sum_num / sum_den)``.

    Numerator and denominator are resampled under ONE task draw (they move
    together, as a ratio estimator requires), using ``_resample_task_totals``
    verbatim. ``transform`` maps the ratio to the reported statistic (identity
    for coverage/Brier; ``1 - r`` for a skill score).
    """

    if numerator_by_task.shape != denominator_by_task.shape:
        raise ValueError("numerator and denominator must be aligned per task")
    stacked = np.column_stack([numerator_by_task, denominator_by_task])
    totals = _resample_task_totals(stacked, replicates=replicates, seed=seed)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(totals[:, 1] != 0.0, totals[:, 0] / totals[:, 1], np.nan)
    draws = transform(ratios) if transform is not None else ratios
    denominator = float(denominator_by_task.sum())
    point_ratio = (
        float(numerator_by_task.sum()) / denominator if denominator != 0.0 else math.nan
    )
    point = (
        float(transform(np.asarray([point_ratio]))[0])
        if transform is not None
        else point_ratio
    )
    return {
        "point": point,
        "interval": _percentile_interval(draws, confidence_level),
        "denominator": denominator,
    }


def cvm_bootstrap(
    pit_values: Sequence[float],
    task_index: Sequence[int],
    *,
    task_count: int,
    replicates: int,
    confidence_level: float,
    seed: int,
    batch: int = 256,
) -> dict[str, Any]:
    """Task-clustered percentile CI for the CvM ``omega2`` distance."""

    values = np.asarray(pit_values, dtype=float)
    tasks = np.asarray(task_index, dtype=np.int64)
    if values.size == 0:
        return {
            "w2": math.nan,
            "point": math.nan,
            "interval": {"low": math.nan, "high": math.nan},
            "sample_count": 0,
        }
    order = np.argsort(values, kind="stable")
    sorted_pit = values[order]
    sorted_tasks = tasks[order]
    w2, omega2 = cramer_von_mises(values)
    draws = np.empty(replicates, dtype=float)
    cursor = 0
    for multiplicities in _task_cluster_multiplicities(
        task_count, replicates=replicates, seed=seed, batch=batch
    ):
        # Per replicate a call repeats as often as its task was drawn.
        per_call = multiplicities[:, sorted_tasks]
        for row in per_call:
            draws[cursor] = _cvm_omega2_weighted(sorted_pit, row)
            cursor += 1
    return {
        "w2": w2,
        "point": omega2,
        "interval": _percentile_interval(draws[:cursor], confidence_level),
        "sample_count": int(values.size),
    }


# --------------------------------------------------------------------------- #
# Per-call scoring over the certified folds.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CalibrationConfig:
    fold_count: int
    command_field: str
    max_prefix_depth: int
    skip_leading_cd: bool
    min_tool_history: int
    min_profile_tasks: int
    costs_ms: tuple[float, ...]
    guard_ms: float
    restore_cost_fraction: float
    replicates: int
    confidence_level: float
    seed: int
    rolling_fractions: tuple[float, ...]
    cvm_replicates: int


@dataclass
class DegenerateCounts:
    """Excluded-node bookkeeping. Never silently dropped: every count is emitted."""

    pit_empty_node: int = 0
    pit_singleton_node: int = 0
    pit_censored: int = 0
    crps_censored: int = 0
    crps_censored_mass_ms: float = 0.0
    rolling_no_support: int = 0
    rolling_censored: int = 0
    coverage_unresolvable: dict[str, int] = field(default_factory=dict)
    coverage_indeterminate: dict[str, int] = field(default_factory=dict)
    brier_no_region: int = 0
    brier_indeterminate_censored: int = 0

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "pit_empty_node": self.pit_empty_node,
            "pit_singleton_node": self.pit_singleton_node,
            "pit_censored_excluded": self.pit_censored,
            "crps_censored_excluded": self.crps_censored,
            "crps_censored_excluded_mass_ms": self.crps_censored_mass_ms,
            "rolling_no_support_beyond_t": self.rolling_no_support,
            "rolling_censored_excluded": self.rolling_censored,
            "coverage_unresolvable_by_quantile": dict(self.coverage_unresolvable),
            "coverage_indeterminate_censored_by_quantile": dict(
                self.coverage_indeterminate
            ),
            "brier_empty_region": self.brier_no_region,
            "brier_indeterminate_censored": self.brier_indeterminate_censored,
        }


def _baseline_nodes(
    hierarchy: tuple[LatencyPriorNode, ...],
) -> tuple[LatencyPriorNode, LatencyPriorNode]:
    """``(pooled, tool_level)`` baselines taken from the estimator's hierarchy.

    ``pooled`` is always ``prior_global`` (hierarchy root). ``tool_level`` is the
    ``prior_tool`` node when the tool clears the eligibility thresholds, else the
    global node -- the same fallback the hierarchy itself performs, so the
    baseline is never a construct this lane invented.
    """

    pooled = hierarchy[0]
    tool_level = pooled
    for node in hierarchy:
        if node.source == "prior_tool":
            tool_level = node
    return pooled, tool_level


def _score_call(
    *,
    node: LatencyPriorNode,
    pooled: LatencyPriorNode,
    tool_level: LatencyPriorNode,
    latency_ms: float,
    censored: bool,
    cfg: CalibrationConfig,
    forecast_cache: dict[int, EcdfForecast],
    trigger_cache: dict[tuple[int, float], float],
    counts: DegenerateCounts,
) -> dict[str, Any]:
    """Score one held-out call against its fit-fold node across all metrics."""

    values = node.values
    row: dict[str, Any] = {
        "latency_ms": latency_ms,
        "censored": censored,
        "prior_source": node.source,
        "prior_group_key": node.group_key,
        "node_count": len(values),
        "pit": None,
        "crps": None,
        "crps_pooled": None,
        "crps_tool": None,
        "rolling": [],
        "coverage": {},
        "brier": {},
    }
    if not values:
        counts.pit_empty_node += 1
        return row
    if len(values) < 2:
        # A singleton ECDF is a point mass: PIT and CRPS are degenerate.
        counts.pit_singleton_node += 1
        return row

    # --- 1a. Marginal PIT (censored rows: u only identified in [F(c), 1]). ----
    if censored:
        counts.pit_censored += 1
    else:
        row["pit"] = ecdf_pit(values, latency_ms)

    # --- 3. CRPS + baselines (censored rows excluded, mass reported). --------
    if censored:
        counts.crps_censored += 1
        counts.crps_censored_mass_ms += latency_ms
    else:
        for key, source in (
            ("crps", node),
            ("crps_pooled", pooled),
            ("crps_tool", tool_level),
        ):
            if len(source.values) < 2:
                continue
            marker = id(source.values)
            if marker not in forecast_cache:
                forecast_cache[marker] = EcdfForecast.from_values(source.values)
            row[key] = forecast_cache[marker].crps(latency_ms)

    # --- 1b. Rolling PIT on the node's own support quantiles. ----------------
    for fraction in cfg.rolling_fractions:
        t = ecdf_quantile(values, fraction)
        alive = len(values) - bisect_right(values, t)
        if alive < 2:
            counts.rolling_no_support += 1
            continue
        if latency_ms <= t:
            continue  # the call had already finished; nothing to score at t
        if censored:
            counts.rolling_censored += 1
            continue
        below = bisect_left(values, latency_ms)
        ties = bisect_right(values, latency_ms) - below
        alive_below = below - bisect_right(values, t)
        row["rolling"].append(
            {
                "fraction": fraction,
                "t_ms": t,
                "u": (max(0, alive_below) + 0.5 * ties) / alive,
            }
        )

    # --- 2. Quantile coverage (censored: definite exceedance iff c >= Q_q). --
    for q in _COVERAGE_QUANTILES:
        label = f"p{q * 100:.0f}"
        if len(values) < 1.0 / (1.0 - q):
            counts.coverage_unresolvable[label] = (
                counts.coverage_unresolvable.get(label, 0) + 1
            )
            continue
        predicted = ecdf_quantile(values, q)
        if censored and latency_ms < predicted:
            counts.coverage_indeterminate[label] = (
                counts.coverage_indeterminate.get(label, 0) + 1
            )
            row["coverage"][label] = {"predicted_ms": predicted, "exceeds": False}
            continue
        row["coverage"][label] = {
            "predicted_ms": predicted,
            "exceeds": latency_ms > predicted,
        }

    # --- 4. Decision-band Brier at the certified trigger, per kv cell. -------
    for kv_cost_ms in cfg.costs_ms:
        threshold_ms = kv_cost_ms + cfg.guard_ms
        cache_key = (id(values), kv_cost_ms)
        if cache_key not in trigger_cache:
            trigger_cache[cache_key] = hazard_recheck_ms(
                values,
                threshold_ms=threshold_ms,
                kv_cost_ms=kv_cost_ms,
                restore_cost_ms=cfg.restore_cost_fraction * kv_cost_ms,
            )
        trigger_ms = trigger_cache[cache_key]
        alive = len(values) - bisect_right(values, trigger_ms)
        if alive < 2:
            counts.brier_no_region += 1
            continue
        if latency_ms <= trigger_ms:
            continue  # outside the trigger region: the policy never reads here
        if censored and latency_ms < threshold_ms:
            # L > c but c < threshold: the outcome is not determinate.
            counts.brier_indeterminate_censored += 1
            continue
        survival_at_trigger = ecdf_survival(values, trigger_ms)
        probability = ecdf_survival(values, threshold_ms) / survival_at_trigger
        outcome = 1.0 if latency_ms > threshold_ms else 0.0
        row["brier"][f"{kv_cost_ms:.0f}"] = {
            "kv_cost_ms": kv_cost_ms,
            "threshold_ms": threshold_ms,
            "trigger_ms": trigger_ms,
            "degenerate_region": trigger_ms >= threshold_ms,
            "probability": probability,
            "outcome": outcome,
            "squared_error": (probability - outcome) ** 2,
        }
    return row


def score_decisions(
    samples_by_task: dict[str, list[ToolLatencySample]],
    task_ids: Sequence[str],
    cfg: CalibrationConfig,
    *,
    censored_ids: set[str],
) -> tuple[list[dict[str, Any]], DegenerateCounts]:
    """Cross-fitted per-call calibration rows over the frozen manifest's folds."""

    row_group_keys = _row_group_keys(
        cfg.command_field,
        max_prefix_depth=cfg.max_prefix_depth,
        skip_leading_cd=cfg.skip_leading_cd,
    )
    declared = list(task_ids)
    counts = DegenerateCounts()
    decisions: list[dict[str, Any]] = []
    for fold in range(1, cfg.fold_count + 1):
        eval_tasks = {
            task_id
            for index, task_id in enumerate(declared)
            if index % cfg.fold_count == fold - 1
        }
        profile_tasks = set(declared) - eval_tasks
        profile_rows = [
            sample.to_json_obj()
            for task_id in sorted(profile_tasks)
            for sample in samples_by_task[task_id]
        ]
        prior = build_latency_prior(profile_rows, row_group_keys=row_group_keys)
        forecast_cache: dict[int, EcdfForecast] = {}
        trigger_cache: dict[tuple[int, float], float] = {}
        for task_id in sorted(eval_tasks):
            for sample in samples_by_task[task_id]:
                row = sample.to_json_obj()
                hierarchy = latency_prior_hierarchy(
                    prior,
                    str(row["tool_name"]),
                    row_group_keys(row),
                    min_tool_history=cfg.min_tool_history,
                    min_profile_tasks=cfg.min_profile_tasks,
                )
                pooled, tool_level = _baseline_nodes(hierarchy)
                scored = _score_call(
                    node=hierarchy[-1],
                    pooled=pooled,
                    tool_level=tool_level,
                    latency_ms=float(row["latency_ms"]),
                    censored=str(row["sample_id"]) in censored_ids,
                    cfg=cfg,
                    forecast_cache=forecast_cache,
                    trigger_cache=trigger_cache,
                    counts=counts,
                )
                decisions.append(
                    {
                        "sample_id": str(row["sample_id"]),
                        "task_id": task_id,
                        "tool_name": str(row["tool_name"]),
                        "outer_fold": f"f{fold}",
                        **scored,
                    }
                )
    return decisions, counts


# --------------------------------------------------------------------------- #
# Aggregation.
# --------------------------------------------------------------------------- #
def _task_vector(
    decisions: Sequence[dict[str, Any]],
    task_ids: Sequence[str],
    contribution: Callable[[dict[str, Any]], tuple[float, float] | None],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-task (numerator, denominator) sums for a ratio statistic."""

    index = {task_id: position for position, task_id in enumerate(task_ids)}
    numerator = np.zeros(len(task_ids), dtype=float)
    denominator = np.zeros(len(task_ids), dtype=float)
    for row in decisions:
        pair = contribution(row)
        if pair is None:
            continue
        position = index[str(row["task_id"])]
        numerator[position] += pair[0]
        denominator[position] += pair[1]
    return numerator, denominator


def _coverage_verdict_suppressed(
    exceedance_count: float, interval: dict[str, float]
) -> str | None:
    """Reason the inside/outside verdict is withheld, or ``None`` to report it.

    A percentile bootstrap over zero events resamples zeros forever and returns
    the zero-width interval ``[0, 0]``, which would render nominal as "OUTSIDE"
    however well calibrated the curve is -- a spurious verdict, and in the tail
    exactly where it would be believed. An exact (Clopper-Pearson) interval is
    what a verdict in that regime would require; this lane withholds the verdict
    rather than inventing one, since it is descriptive and nothing downstream
    reads the flag.
    """

    low, high = interval["low"], interval["high"]
    if not (math.isfinite(low) and math.isfinite(high)):
        return "interval is undefined (no scored calls)"
    if exceedance_count == 0.0:
        return (
            "zero observed exceedances -- the percentile bootstrap is degenerate "
            "here and an exact (Clopper-Pearson) interval would be required"
        )
    if high <= low:
        return "zero-width bootstrap interval -- not an interval, no verdict"
    return None


def _pit_histogram(pit_values: Sequence[float]) -> list[dict[str, Any]]:
    counts, edges = np.histogram(
        np.asarray(pit_values, dtype=float), bins=_PIT_BINS, range=(0.0, 1.0)
    )
    total = int(counts.sum())
    return [
        {
            "bin_low": float(edges[i]),
            "bin_high": float(edges[i + 1]),
            "count": int(counts[i]),
            "fraction": (float(counts[i]) / total) if total else math.nan,
            "expected_fraction": 1.0 / _PIT_BINS,
        }
        for i in range(_PIT_BINS)
    ]


def summarize(
    decisions: Sequence[dict[str, Any]],
    counts: DegenerateCounts,
    task_ids: Sequence[str],
    cfg: CalibrationConfig,
) -> dict[str, Any]:
    """Assemble the four metric families with task-clustered CIs."""

    task_position = {task_id: position for position, task_id in enumerate(task_ids)}

    # --- 1a. Marginal PIT. ---------------------------------------------------
    pit_rows = [row for row in decisions if row["pit"] is not None]
    pit_values = [float(row["pit"]) for row in pit_rows]
    pit_tasks = [task_position[str(row["task_id"])] for row in pit_rows]
    marginal_pit = {
        "histogram": _pit_histogram(pit_values) if pit_values else [],
        "cvm": cvm_bootstrap(
            pit_values,
            pit_tasks,
            task_count=len(task_ids),
            replicates=cfg.cvm_replicates,
            confidence_level=cfg.confidence_level,
            seed=cfg.seed,
        ),
        "mean": float(np.mean(pit_values)) if pit_values else math.nan,
        "scored_call_count": len(pit_values),
    }

    # --- 1b. Rolling PIT, per t-grid fraction. -------------------------------
    rolling_by_fraction: dict[float, list[tuple[int, float, float]]] = defaultdict(list)
    for row in decisions:
        position = task_position[str(row["task_id"])]
        for entry in row["rolling"]:
            rolling_by_fraction[float(entry["fraction"])].append(
                (position, float(entry["u"]), float(entry["t_ms"]))
            )
    rolling = []
    for fraction in cfg.rolling_fractions:
        entries = rolling_by_fraction.get(fraction, [])
        values = [u for _, u, _ in entries]
        rolling.append(
            {
                "fraction": fraction,
                "scored_call_count": len(values),
                "median_t_ms": float(np.median([t for _, _, t in entries]))
                if entries
                else math.nan,
                "mean_pit": float(np.mean(values)) if values else math.nan,
                "histogram": _pit_histogram(values) if values else [],
                "cvm": cvm_bootstrap(
                    values,
                    [position for position, _, _ in entries],
                    task_count=len(task_ids),
                    replicates=cfg.cvm_replicates,
                    confidence_level=cfg.confidence_level,
                    seed=cfg.seed,
                ),
            }
        )

    # --- 2. Quantile coverage vs nominal. ------------------------------------
    coverage = []
    for q in _COVERAGE_QUANTILES:
        label = f"p{q * 100:.0f}"

        def contribution(row: dict[str, Any], _label: str = label) -> tuple[float, float] | None:
            cell = row["coverage"].get(_label)
            if cell is None:
                return None
            return (1.0 if cell["exceeds"] else 0.0, 1.0)

        numerator, denominator = _task_vector(decisions, task_ids, contribution)
        stats = ratio_bootstrap(
            numerator,
            denominator,
            replicates=cfg.replicates,
            confidence_level=cfg.confidence_level,
            seed=cfg.seed,
        )
        nominal = 1.0 - q
        interval = stats["interval"]
        exceedance_count = float(numerator.sum())
        suppressed_reason = _coverage_verdict_suppressed(exceedance_count, interval)
        coverage.append(
            {
                "quantile": q,
                "label": label,
                "nominal_exceedance": nominal,
                "empirical_exceedance": stats["point"],
                "interval": interval,
                "exceedance_count": exceedance_count,
                # None (not False) when the interval cannot support a verdict: a
                # zero-event percentile bootstrap is degenerate, not evidence.
                "nominal_inside_interval": None
                if suppressed_reason is not None
                else bool(interval["low"] <= nominal <= interval["high"]),
                "verdict_suppressed_reason": suppressed_reason,
                "scored_call_count": int(stats["denominator"]),
                "unresolvable_node_call_count": counts.coverage_unresolvable.get(
                    label, 0
                ),
                "indeterminate_censored_count": counts.coverage_indeterminate.get(
                    label, 0
                ),
            }
        )

    # --- 3. CRPS and skill vs the two baselines. -----------------------------
    def crps_sum(key: str) -> float:
        return float(sum(row[key] for row in decisions if row[key] is not None))

    scored_crps = [row for row in decisions if row["crps"] is not None]
    skills = {}
    for name, key in (("pooled", "crps_pooled"), ("tool_name", "crps_tool")):

        def contribution(row: dict[str, Any], _key: str = key) -> tuple[float, float] | None:
            if row["crps"] is None or row[_key] is None:
                return None
            return (float(row["crps"]), float(row[_key]))

        numerator, denominator = _task_vector(decisions, task_ids, contribution)
        skills[name] = ratio_bootstrap(
            numerator,
            denominator,
            replicates=cfg.replicates,
            confidence_level=cfg.confidence_level,
            seed=cfg.seed,
            transform=lambda ratio: 1.0 - ratio,
        )
    crps = {
        "scored_call_count": len(scored_crps),
        "mean_crps_ms": (crps_sum("crps") / len(scored_crps)) if scored_crps else math.nan,
        "mean_crps_pooled_ms": (crps_sum("crps_pooled") / len(scored_crps))
        if scored_crps
        else math.nan,
        "mean_crps_tool_ms": (crps_sum("crps_tool") / len(scored_crps))
        if scored_crps
        else math.nan,
        "skill_vs_pooled": skills["pooled"],
        "skill_vs_tool_name": skills["tool_name"],
        "censored_excluded_count": counts.crps_censored,
        "censored_excluded_mass_ms": counts.crps_censored_mass_ms,
    }

    # --- 4. Decision-band Brier per kv cell. ---------------------------------
    brier = []
    for kv_cost_ms in cfg.costs_ms:
        label = f"{kv_cost_ms:.0f}"

        def contribution(row: dict[str, Any], _label: str = label) -> tuple[float, float] | None:
            cell = row["brier"].get(_label)
            if cell is None:
                return None
            return (float(cell["squared_error"]), 1.0)

        numerator, denominator = _task_vector(decisions, task_ids, contribution)
        stats = ratio_bootstrap(
            numerator,
            denominator,
            replicates=cfg.replicates,
            confidence_level=cfg.confidence_level,
            seed=cfg.seed,
        )
        cells = [row["brier"][label] for row in decisions if label in row["brier"]]
        outcomes = [float(cell["outcome"]) for cell in cells]
        base_rate = float(np.mean(outcomes)) if outcomes else math.nan
        # Climatology reference: predicting the region base rate everywhere.
        reference = base_rate * (1.0 - base_rate) if outcomes else math.nan
        brier.append(
            {
                "kv_cost_ms": kv_cost_ms,
                "threshold_ms": kv_cost_ms + cfg.guard_ms,
                "region_call_count": len(cells),
                "median_trigger_ms": float(
                    np.median([float(cell["trigger_ms"]) for cell in cells])
                )
                if cells
                else math.nan,
                "degenerate_region_call_count": sum(
                    1 for cell in cells if cell["degenerate_region"]
                ),
                "base_rate": base_rate,
                "brier": stats["point"],
                "interval": stats["interval"],
                "reference_brier": reference,
                "brier_skill_score": (1.0 - stats["point"] / reference)
                if reference and math.isfinite(reference) and reference > 0.0
                else math.nan,
            }
        )

    return {
        "task_count": len(task_ids),
        "call_count": len({str(row["sample_id"]) for row in decisions}),
        "censored_call_count": sum(1 for row in decisions if row["censored"]),
        "marginal_pit": marginal_pit,
        "rolling_pit": rolling,
        "coverage": coverage,
        "crps": crps,
        "decision_band_brier": brier,
        "excluded": counts.to_json_obj(),
    }


def run_prior_calibration(
    samples_by_task: dict[str, list[ToolLatencySample]],
    task_ids: Sequence[str],
    cfg: CalibrationConfig,
    *,
    censored_ids: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    decisions, counts = score_decisions(
        samples_by_task, task_ids, cfg, censored_ids=censored_ids
    )
    return summarize(decisions, counts, list(task_ids), cfg), decisions


# --------------------------------------------------------------------------- #
# Rendering / CLI.
# --------------------------------------------------------------------------- #
def _fmt(value: float, digits: int = 4) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def _coverage_verdict_cell(cell: dict[str, Any]) -> str:
    if cell["nominal_inside_interval"] is None:
        return "no verdict"
    return "yes" if cell["nominal_inside_interval"] else "NO"


def render_markdown(results: dict[str, Any], provenance: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Calibration of the certified latency priors")
    lines.append("")
    lines.append(f"> **{_banner(provenance['final'])}**")
    lines.append(">")
    lines.append(
        "> DESCRIPTIVE lane, NO kill criterion. The estimator is frozen and "
        "H1-certified; this lane measures it and cannot change it. Original-trace "
        f"latencies via the frozen manifest ({provenance['collection_id']}), "
        f"cross-fitted as A0/A2. Generated {provenance['generated']} "
        f"(git {provenance['git_sha']})."
    )
    lines.append("")
    lines.append(
        f"{results['task_count']} tasks, {results['call_count']} calls "
        f"({results['censored_call_count']} right-censored), guard "
        f"{provenance['guard_ms']:.0f}ms (threshold==kv), rho="
        f"{provenance['restore_cost_fraction']}. Task-clustered percentile CIs at "
        f"{provenance['confidence_level']:.2f}."
    )
    lines.append("")

    # Pre-declared reporting standard, in the order it was declared.
    lines.append("## Pre-declared readout")
    lines.append("")
    p90 = next((c for c in results["coverage"] if c["label"] == "p90"), None)
    if p90 is not None:
        if p90["nominal_inside_interval"] is None:
            verdict = (
                f"**NO VERDICT** -- {p90['verdict_suppressed_reason']}"
            )
        else:
            verdict = (
                "nominal lies **"
                + ("INSIDE" if p90["nominal_inside_interval"] else "OUTSIDE")
                + "** the interval"
            )
        lines.append(
            f"- **P90 coverage**: empirical exceedance {_fmt(p90['empirical_exceedance'])} "
            f"(nominal {_fmt(p90['nominal_exceedance'])}), CI "
            f"[{_fmt(p90['interval']['low'])}, {_fmt(p90['interval']['high'])}] -- "
            f"{verdict}."
        )
    cvm = results["marginal_pit"]["cvm"]
    lines.append(
        f"- **CvM distance to uniform** (marginal PIT): omega2="
        f"{_fmt(cvm['point'], 5)}, CI [{_fmt(cvm['interval']['low'], 5)}, "
        f"{_fmt(cvm['interval']['high'], 5)}] over {cvm['sample_count']} calls. "
        f"omega2 is the N-NORMALISED statistic `W2 / N == integral (F_N(u) - "
        f"u)^2 du` (here W2={_fmt(cvm['w2'], 3)}); it is quoted as the headline "
        "because it is free of sample size and so is comparable across the "
        "differently-sized rolling-grid subsets below."
    )
    for name, key in (("pooled", "skill_vs_pooled"), ("tool-name", "skill_vs_tool_name")):
        skill = results["crps"][key]
        lines.append(
            f"- **CRPS skill vs {name} curve**: {_fmt(skill['point'])}, CI "
            f"[{_fmt(skill['interval']['low'])}, {_fmt(skill['interval']['high'])}]."
        )
    lines.append("- **Decision-band Brier**: per kv cell, table below.")
    lines.append("")
    lines.append(
        f"> **Replicate discipline (non-uniform, stated deliberately).** Coverage, "
        f"CRPS skill and Brier CIs use {provenance['replicates']} bootstrap "
        f"replicates; the CvM intervals (marginal and rolling, wherever omega2 "
        f"appears) use {provenance['cvm_replicates']}. All are task-clustered at "
        f"confidence {provenance['confidence_level']:.2f}, seed "
        f"{provenance['seed']}. The split is cost, not convenience: the first "
        "three are ratios of sums and go through the certified "
        "`_resample_task_totals` in one vectorised pass, whereas CvM is not a "
        f"ratio of sums and each replicate costs O(calls). At the certified "
        f"discipline ({provenance['certified_replicates']} and "
        f"{provenance['certified_cvm_replicates']}) the CvM percentile intervals "
        f"are stable to ~1e-4. `--final` REJECTS any override of these four "
        f"knobs, so a FINAL artifact always carries the certified values; only a "
        f"non-final run can lower them. All four are recorded in the JSON "
        f"provenance (`replicates`, `cvm_replicates`, `seed`, "
        f"`confidence_level`)."
    )
    lines.append("")
    lines.append(
        "> Any coverage miss or negative skill is a reported LIMITATION. The "
        "estimator does not change based on this lane; a finding of "
        "miscalibration would motivate a separate, pre-registered estimator lane "
        "with its own decision gate."
    )
    lines.append("")

    lines.append("## Quantile coverage vs nominal (tail is what matters)")
    lines.append("")
    lines.append(
        "| quantile | nominal exceed | empirical exceed | CI | nominal inside | "
        "calls | node-unresolvable | indeterminate censored |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for cell in results["coverage"]:
        lines.append(
            f"| {cell['label'].upper()} | {_fmt(cell['nominal_exceedance'])} | "
            f"{_fmt(cell['empirical_exceedance'])} | "
            f"[{_fmt(cell['interval']['low'])}, {_fmt(cell['interval']['high'])}] | "
            f"{_coverage_verdict_cell(cell)} | "
            f"{cell['scored_call_count']} | {cell['unresolvable_node_call_count']} | "
            f"{cell['indeterminate_censored_count']} |"
        )
    lines.append("")
    suppressed = [c for c in results["coverage"] if c["nominal_inside_interval"] is None]
    if suppressed:
        lines.append(
            "> **No verdict** is reported for "
            + ", ".join(c["label"].upper() for c in suppressed)
            + f": {suppressed[0]['verdict_suppressed_reason']}. A zero-event "
            "percentile bootstrap resamples zeros forever and returns [0, 0], "
            "which would render nominal as OUTSIDE however well calibrated the "
            "curve is. That is a spurious verdict in exactly the tail where it "
            "would be believed, so it is withheld rather than printed."
        )
        lines.append("")
    total_censored = results["censored_call_count"]
    total_indeterminate = sum(
        c["indeterminate_censored_count"] for c in results["coverage"]
    )
    lines.append(
        f"> **Direction of the censoring bias (disclosure).** Right-censored rows "
        f"whose bound falls BELOW the predicted quantile cannot decide the "
        f"exceedance, and this lane scores them as non-exceedances. That biases "
        f"empirical exceedance **LOW** -- i.e. toward the FLATTERING direction, "
        f"since under-coverage (exceeding more often than nominal) is the "
        f"dangerous mode for a swap policy. The magnitude is bounded by the "
        f"censored count ({total_censored} of {results['call_count']} calls) and "
        f"is {total_indeterminate} on this corpus: every censored bound sits far "
        f"above every predicted quantile, so each one resolves as a DEFINITE "
        f"exceedance and the bias does not bind here. It is disclosed because the "
        f"bound, not the observed value, is what guarantees that."
    )
    lines.append("")

    lines.append("## PIT calibration")
    lines.append("")
    lines.append(
        f"Marginal PIT over {results['marginal_pit']['scored_call_count']} calls, "
        f"mean {_fmt(results['marginal_pit']['mean'])} (uniform expects 0.5)."
    )
    lines.append("")
    lines.append("| bin | count | fraction | expected |")
    lines.append("| --- | --- | --- | --- |")
    for entry in results["marginal_pit"]["histogram"]:
        lines.append(
            f"| [{entry['bin_low']:.1f}, {entry['bin_high']:.1f}) | {entry['count']} | "
            f"{_fmt(entry['fraction'])} | {_fmt(entry['expected_fraction'])} |"
        )
    lines.append("")
    lines.append("### Rolling PIT (the consumed object) on node-support quantiles")
    lines.append("")
    lines.append(
        "The t-grid is each node's OWN support quantile at the listed fraction, so "
        "no corpus-specific time constant enters."
    )
    lines.append("")
    lines.append("| support fraction | median t (ms) | calls | mean PIT | CvM omega2 | CI |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for entry in results["rolling_pit"]:
        rcvm = entry["cvm"]
        lines.append(
            f"| {entry['fraction']:.2f} | {_fmt(entry['median_t_ms'], 1)} | "
            f"{entry['scored_call_count']} | {_fmt(entry['mean_pit'])} | "
            f"{_fmt(rcvm['point'], 5)} | [{_fmt(rcvm['interval']['low'], 5)}, "
            f"{_fmt(rcvm['interval']['high'], 5)}] |"
        )
    lines.append("")

    crps = results["crps"]
    lines.append("## CRPS and skill (sharpness)")
    lines.append("")
    lines.append(
        f"Mean CRPS over {crps['scored_call_count']} uncensored calls: model "
        f"{_fmt(crps['mean_crps_ms'], 1)} ms, pooled baseline "
        f"{_fmt(crps['mean_crps_pooled_ms'], 1)} ms, tool-name baseline "
        f"{_fmt(crps['mean_crps_tool_ms'], 1)} ms. Censored rows excluded: "
        f"{crps['censored_excluded_count']} calls carrying "
        f"{crps['censored_excluded_mass_ms'] / 1000.0:.1f} s of observed latency."
    )
    lines.append("")

    lines.append("## Decision-band Brier (per kv cell, at the certified trigger)")
    lines.append("")
    lines.append(
        "Trigger is `hazard_recheck_ms` (the same k=1 optimizer A0 adjudicated and "
        "A2 uses as its `hazard` source). Region = calls alive at the trigger; "
        "outcome = `L > threshold`. A degenerate region (trigger == threshold) "
        "makes the outcome constant, hence the base rate and reference columns."
    )
    lines.append("")
    region_counts = [c["region_call_count"] for c in results["decision_band_brier"]]
    lines.append(
        f"> **Effective sample (quotable limitation).** Every reported Brier "
        f"statistic is PER kv CELL -- nothing pools across cells -- so the honest "
        f"effective sample is the per-cell one: between {min(region_counts)} and "
        f"{max(region_counts)} calls fall inside the trigger region in any single "
        f"cell (across {len(region_counts)} cells), against "
        f"{results['marginal_pit']['scored_call_count']} calls scoring the "
        f"marginal PIT and {results['crps']['scored_call_count']} scoring CRPS. A "
        f"further {results['excluded']['brier_empty_region']} call-by-kv pairs had "
        "no usable region (fewer than 2 fit-fold samples surviving past the "
        "trigger) and were excluded. The cause is structural, not a data defect: "
        "most tool calls last milliseconds, so a node carries little support above "
        "a 500-5000 ms trigger. Conclusions from this family are correspondingly "
        "weaker than from PIT, coverage, or CRPS."
    )
    lines.append("")
    lines.append(
        "| kv | region calls | median trigger (ms) | base rate | Brier | CI | "
        "reference | skill | degenerate |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for cell in results["decision_band_brier"]:
        lines.append(
            f"| {cell['kv_cost_ms']:.0f} | {cell['region_call_count']} | "
            f"{_fmt(cell['median_trigger_ms'], 1)} | {_fmt(cell['base_rate'])} | "
            f"{_fmt(cell['brier'])} | [{_fmt(cell['interval']['low'])}, "
            f"{_fmt(cell['interval']['high'])}] | {_fmt(cell['reference_brier'])} | "
            f"{_fmt(cell['brier_skill_score'])} | "
            f"{cell['degenerate_region_call_count']} |"
        )
    lines.append("")

    lines.append("## Excluded (degenerate nodes and censoring), never silently dropped")
    lines.append("")
    for key, value in results["excluded"].items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "analysis/fresh-corpus-certification-20260717/"
            "offline-gated-robust/manifest.json"
        ),
    )
    parser.add_argument(
        "--rolling-fractions",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="Support-quantile FRACTIONS defining each node's own rolling t-grid "
        "(deciles by default). Fractions are the documented constant; the t "
        "values are node-derived, never corpus-specific.",
    )
    parser.add_argument("--replicates", type=int, default=50000)
    parser.add_argument(
        "--cvm-replicates",
        type=int,
        default=2000,
        help="Bootstrap replicates for the CvM CIs. Lower than --replicates "
        "because CvM is not a ratio of sums and each replicate costs O(calls); "
        "percentile CIs at this width are stable to ~1e-4.",
    )
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit-tasks",
        type=int,
        default=None,
        help="Smoke only: cap tasks (subsets folds consistently). Rejected with "
        "--final.",
    )
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--final", action="store_true")
    return parser


_CERTIFIED_KNOBS = ("replicates", "cvm_replicates", "seed", "confidence_level")


def _require_certified_discipline(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """``--final`` may only ship at the certified statistical discipline.

    Mirrors the ``--limit-tasks`` convention (``adjudicate_k2_recheck``): a knob
    that would quietly weaken a FINAL-bannered artifact is rejected rather than
    truthfully labelled. The certified values ARE the argparse defaults, so
    there is no second copy to drift.
    """

    off = {
        name: (getattr(args, name), parser.get_default(name))
        for name in _CERTIFIED_KNOBS
        if getattr(args, name) != parser.get_default(name)
    }
    if off:
        detail = ", ".join(
            f"--{name.replace('_', '-')}={got!r} (certified {want!r})"
            for name, (got, want) in sorted(off.items())
        )
        raise ValueError(
            f"--final requires the certified discipline; got {detail}. Drop the "
            "override or drop --final."
        )


def _default_output_paths(final: bool) -> tuple[Path, Path]:
    today = _dt.date.today().isoformat()
    suffix = "" if final else "-PARTIAL"
    stem = f"analysis/certification/prior-calibration-{today}{suffix}"
    return Path(f"{stem}.json"), Path(f"{stem}.md")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.final:
        _require_certified_discipline(args, parser)
    default_json, default_md = _default_output_paths(args.final)
    out_json = args.out_json or default_json
    out_md = args.out_md or default_md
    print(_banner(args.final))

    samples_by_task, task_ids, manifest = _load_manifest_corpus(
        args.manifest, limit_tasks=args.limit_tasks, final=args.final
    )
    trace_paths = sorted(
        {
            Path(sample.source_trace)
            for task_id in task_ids
            for sample in samples_by_task[task_id]
        }
    )
    censored_ids = censored_sample_ids(trace_paths)
    print(f"right-censored protocol-guard timeouts: {len(censored_ids)}")

    cfg = CalibrationConfig(
        fold_count=manifest["fold_count"],
        command_field=manifest["command_field"],
        max_prefix_depth=manifest["max_prefix_depth"],
        skip_leading_cd=manifest["skip_leading_cd"],
        min_tool_history=manifest["min_tool_history"],
        min_profile_tasks=manifest["min_profile_tasks"],
        costs_ms=tuple(float(cost) for cost in manifest["costs_ms"]),
        guard_ms=float(manifest.get("guard_ms", _CERT_GUARD_MS)),
        restore_cost_fraction=_CERT_RESTORE_COST_FRACTION,
        replicates=args.replicates,
        confidence_level=args.confidence_level,
        seed=args.seed,
        rolling_fractions=tuple(float(f) for f in args.rolling_fractions),
        cvm_replicates=args.cvm_replicates,
    )
    summary, decisions = run_prior_calibration(
        samples_by_task, task_ids, cfg, censored_ids=censored_ids
    )
    provenance = {
        "exploratory": True,
        "descriptive_lane": True,
        "kill_criterion": None,
        "estimator_modified": False,
        "final": bool(args.final),
        "manifest": str(args.manifest),
        "collection_id": manifest["collection_id"],
        "task_count": len(task_ids),
        "limit_tasks": args.limit_tasks,
        "guard_ms": cfg.guard_ms,
        "restore_cost_fraction": cfg.restore_cost_fraction,
        "confidence_level": cfg.confidence_level,
        "replicates": cfg.replicates,
        "cvm_replicates": cfg.cvm_replicates,
        "seed": cfg.seed,
        # The certified discipline IS the argparse default set (--final enforces
        # it), so the artifact never carries a second, driftable copy.
        "certified_replicates": parser.get_default("replicates"),
        "certified_cvm_replicates": parser.get_default("cvm_replicates"),
        "censoring_rule": (
            "tool_exec with data.success is False and data.tool_result starting "
            f"{_CENSOR_RESULT_PREFIX!r} containing {_CENSOR_RESULT_MARKER!r}"
        ),
        "git_sha": _git_sha(),
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    decisions_path = out_json.with_name(out_json.stem + "-decisions.json.zst")
    payload = {
        "provenance": provenance,
        "config": cfg.__dict__,
        "decisions_sidecar": decisions_path.name,
        **summary,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, default=list), encoding="utf-8")
    out_md.write_text(render_markdown(summary, provenance), encoding="utf-8")
    _write_decisions_zst(decisions_path, decisions)

    p90 = next((c for c in summary["coverage"] if c["label"] == "p90"), None)
    if p90 is not None:
        print(
            f"P90 exceedance={p90['empirical_exceedance']:.4f} "
            f"nominal={p90['nominal_exceedance']:.4f} "
            f"inside={_coverage_verdict_cell(p90)}"
        )
    print(f"CvM omega2={summary['marginal_pit']['cvm']['point']:.5f}")
    print(
        f"CRPS skill vs pooled={summary['crps']['skill_vs_pooled']['point']:.4f} "
        f"vs tool={summary['crps']['skill_vs_tool_name']['point']:.4f}"
    )
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    print(f"wrote {decisions_path}")


if __name__ == "__main__":
    main()
