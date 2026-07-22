#!/usr/bin/env python3
"""Candidate C Stage-1 kill test: does atom-boundary identity add information
beyond elapsed time alone, for the residual-survival re-check decision?

Pre-registered in ``analysis/CLOSED-QUESTIONS.md`` (Candidate C,
"Stage 1 (2 days, the kill switch)"). The estimator machinery is NOT
reinvented here: the conditional residual sample sets feed the EXISTING
``hazard_recheck_ms`` optimizer from
``trace_collect.tool_latency_profiled`` at the certified operating point, and
segment parsing / atom extraction / task-grouped folds are REUSED verbatim
from ``scripts.analyze_segment_variance`` (the five-model atom study).

The falsifiable claim (memo §Candidate C): elapsed time alone is already
implicit conditioning (survival of the residual given t); C's extra claim is
that *which atom finished, at which boundary index* refines the residual
distribution beyond elapsed time. Two conditioning arms at each observed
sequential boundary:

* arm ``E``   - condition on elapsed time only: the residual-survival sample
  set is every fit-fold chain still running at the boundary's elapsed time
  ``t`` (``total_i > t``), residual ``total_i - t``. No structure.
* arm ``E+B`` - condition on elapsed time AND boundary identity/index: the
  fit-fold chains that, at elapsed ``t``, are in the SAME boundary state -
  their ``k``-th boundary exists, its atom equals the observed atom, that
  boundary has been reached (``t_end(k) <= t``) and is the most recent one
  completed (no later boundary reached by ``t``), and the chain is still
  running (``total_i > t``); residual ``total_i - t``. ``E+B`` is thus a
  refinement of ``E`` by the discrete event "atom ``a`` completed as boundary
  ``k``". Because same-atom-same-index boundaries cluster in time (an
  ``apt-get update`` completing means several seconds elapsed), temporal
  proximity is carried structurally by the shared boundary state rather than
  by a soft kernel weight; this is the one deviation from the memo's
  "reweighted by proximity of their t_k" phrasing, taken because the required
  existing ``hazard_recheck_ms`` optimizer consumes an UNWEIGHTED sample list
  (soft weights cannot flow through it without fabricating pseudo-counts), and
  a ``--proximity-bandwidth-ms`` hard window is exposed for sensitivity.

NO-MIXING RULE (memo, non-negotiable): both the call total AND every boundary
time come from the SAME segment-timeline replay re-run - the total is
``raw_total_ms`` (the xtrace re-run's measured wall time) and boundaries are
that run's segment ``t_end_ms``. The original collected-trace call durations
(different hardware/provider) are NEVER used, and the replayed total is never
combined with a boundary *fraction* of a different-hardware duration. Chains
lacking ``raw_total_ms`` are excluded and counted.

NO ORACLE LEAKAGE: every conditioning variable (elapsed ``t``, boundary index
``k``, completed atom) is known at the boundary event time. The observed
residual ``r = total - t`` is used ONLY as the held-out label to score, never
as a feature. Conditional sample sets are built from FIT-fold chains only
(task-disjoint from the scored event).

Two pre-registered metrics, cross-fitted over task-grouped folds:

  (i)  paired log-score gain of ``E+B`` over ``E`` on the held-out residual,
       with a task-clustered bootstrap CI. Log-score is the log density a
       Gaussian-KDE (Scott's rule, an established estimator - scipy) fit on
       the arm's residual samples (in ``log1p`` ms, a standard
       variance-stabilizing transform for heavy-tailed durations; the
       transform's Jacobian is identical for both arms at the same residual
       and cancels in the paired difference) assigns to the observed residual.
  (ii) decision divergence: the fraction of re-check decisions that change
       between arms, where each decision is
       ``hazard_recheck_ms(residuals, threshold_ms=kv, kv_cost_ms=kv,
       restore_cost_ms=rho*kv)`` swept over the certified kv-cost grid at the
       certified operating point (rho=0.94, guard 0 so threshold==kv).

Kill readout (printed explicitly): KILL if pooled decision divergence is below
``--kill-divergence-frac`` (~1%) OR the log-score gain CI covers zero; SURVIVE
only if divergence clears the bar AND the gain CI lies strictly above zero.

Exploratory; durations replayed on our own hardware. Emits JSON + MD to
``analysis/`` (``-PARTIAL`` suffix unless ``--final``).

Usage:
  uv run python scripts/exploration/analyze_boundary_evidence_stage1.py \
    --traces-dir traces/fresh-277-segtimeline --fold-count 5 --final
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from dataclasses import dataclass
import datetime as _dt
import functools
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterator, Sequence

import numpy as np
from scipy.stats import gaussian_kde

# Allow `python scripts/exploration/analyze_boundary_evidence_stage1.py ...` to import the
# sibling study module as a package (pytest adds the repo root itself).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.exploration.analyze_segment_variance import Config, _task_folds, atom_key  # noqa: E402
from scripts.certification.run_offline_gated_robust_confirmation import (  # noqa: E402
    resolve_worker_count,
)
from trace_collect.tool_latency_dataset import (
    SegmentLatencySample,
    extract_many_segment_latency_samples,
)
from trace_collect.tool_latency_profiled import hazard_recheck_ms

# Certified operating point (fresh-corpus certification manifest,
# analysis/fresh-corpus-certification-20260717/offline-gated-robust/
# manifest.json): kv-cost grid, guard 0 (so threshold == kv), and the measured
# restore-cost fraction rho=0.94 used by scripts/serving/export_trigger_table.py. These
# are the operating point, not tunable knobs - CLI overrides exist only for the
# smoke path, and the defaults are the certified values.
_CERT_KV_COSTS_MS: tuple[float, ...] = (
    500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 4500.0, 5000.0,
)
_CERT_GUARD_MS = 0.0
_CERT_RESTORE_COST_FRACTION = 0.94  # rho; matches export_trigger_table.py


@dataclass(frozen=True)
class TimedBoundary:
    """One segment's completion: its atom and its replay-run t_end (elapsed)."""

    atom: str
    t_end_ms: float


@dataclass(frozen=True)
class TimedChain:
    """One exec call keeping the replay boundary times build_chains discards.

    The study's ``Chain``/``Segment`` keep only per-segment DURATIONS, but the
    no-mixing rule needs the absolute boundary times ``t_end_ms`` (cumulative
    elapsed, including inter-segment gaps) from the SAME replay run as the
    total. Parsing / atom extraction / pipe-loop exclusion are still reused
    (``extract_many_segment_latency_samples`` + ``atom_key``); only the
    aggregation differs, keeping the timing the study threw away.
    """

    task_id: str
    source_trace: str
    action_id: str
    raw_total_ms: float | None  # same-run total (segment_timeline raw_total_ms)
    boundaries: tuple[TimedBoundary, ...]  # ordered by segment_index


@dataclass(frozen=True)
class BoundaryEvent:
    """One observed sequential boundary at which a re-check could be re-timed."""

    task_id: str
    boundary_index: int
    atom: str
    elapsed_ms: float  # boundary time t (from the same replay run as total)
    total_ms: float  # raw_total_ms of the same replay run
    residual_ms: float  # observed label = total - elapsed (never a feature)


@dataclass(frozen=True)
class _FitBoundary:
    """A fit-fold chain's j-th boundary, for E+B state selection."""

    t_end_ms: float
    next_end_ms: float  # (j+1)-th boundary time, or +inf if j is the last
    total_ms: float


def build_timed_chains(samples: Sequence[SegmentLatencySample]) -> list[TimedChain]:
    """Group segment samples into TimedChains keyed by (trace, action).

    Segments are ordered by ``segment_index``; the boundary atom reuses the
    study's ``atom_key``; ``raw_total_ms`` is the segment-timeline re-run's
    measured wall time (same run as the boundary ``t_end_ms``).
    """

    grouped: dict[tuple[str, str], list[SegmentLatencySample]] = defaultdict(list)
    for sample in samples:
        grouped[(sample.source_trace, sample.action_id)].append(sample)
    chains: list[TimedChain] = []
    for (source_trace, action_id), rows in grouped.items():
        rows.sort(key=lambda s: s.segment_index)
        first = rows[0]
        boundaries = tuple(
            TimedBoundary(atom=atom_key(row.segment_command), t_end_ms=row.t_end_ms)
            for row in rows
        )
        chains.append(
            TimedChain(
                task_id=first.task_id,
                source_trace=source_trace,
                action_id=action_id,
                raw_total_ms=first.parent_raw_total_ms,
                boundaries=boundaries,
            )
        )
    return chains


def analysable_chains(chains: Sequence[TimedChain]) -> list[TimedChain]:
    """Multi-segment sequential chains with a replayed total (raw_total_ms).

    Pipe/loop parents are already dropped upstream (``skip_concurrent=True``).
    A boundary decision needs at least one interior boundary, hence >= 2
    segments; the no-mixing rule needs ``raw_total_ms`` (same-run total).
    """

    return [
        chain
        for chain in chains
        if len(chain.boundaries) >= 2 and chain.raw_total_ms is not None
    ]


def boundary_events(chain: TimedChain) -> Iterator[BoundaryEvent]:
    """Yield interior boundary events (index 0..N-2) of one analysable chain."""

    total = chain.raw_total_ms
    assert total is not None  # analysable_chains guarantees this
    for index in range(len(chain.boundaries) - 1):
        elapsed = chain.boundaries[index].t_end_ms
        residual = total - elapsed
        if elapsed <= 0.0 or residual <= 0.0:
            continue
        yield BoundaryEvent(
            task_id=chain.task_id,
            boundary_index=index,
            atom=chain.boundaries[index].atom,
            elapsed_ms=elapsed,
            total_ms=total,
            residual_ms=residual,
        )


def _fit_index(
    chains: Sequence[TimedChain],
) -> tuple[np.ndarray, dict[tuple[int, str], list[_FitBoundary]]]:
    """Precompute the E population totals and the E+B (index, atom) nodes."""

    totals = np.asarray([chain.raw_total_ms for chain in chains], dtype=float)
    nodes: dict[tuple[int, str], list[_FitBoundary]] = defaultdict(list)
    for chain in chains:
        total = chain.raw_total_ms
        assert total is not None
        boundaries = chain.boundaries
        for j, boundary in enumerate(boundaries):
            next_end = (
                boundaries[j + 1].t_end_ms if j + 1 < len(boundaries) else math.inf
            )
            nodes[(j, boundary.atom)].append(
                _FitBoundary(
                    t_end_ms=boundary.t_end_ms, next_end_ms=next_end, total_ms=total
                )
            )
    return totals, nodes


def elapsed_only_residuals(fit_totals: np.ndarray, elapsed_ms: float) -> np.ndarray:
    """Arm E: residuals of every fit chain still running at ``elapsed_ms``."""

    running = fit_totals[fit_totals > elapsed_ms]
    return running - elapsed_ms


def boundary_state_residuals(
    node: Sequence[_FitBoundary], elapsed_ms: float
) -> np.ndarray:
    """Arm E+B: residuals of fit chains in the same (index, atom) state at t.

    Same state = boundary reached (``t_end <= t``), most recent one completed
    (next boundary not yet reached: ``next_end > t``), still running
    (``total > t``). Residual measured from the common elapsed ``t`` so E and
    E+B are paired at the same clock reading, isolating the boundary evidence.
    """

    if not node:
        return np.empty(0, dtype=float)
    t_end = np.fromiter((b.t_end_ms for b in node), dtype=float, count=len(node))
    next_end = np.fromiter((b.next_end_ms for b in node), dtype=float, count=len(node))
    total = np.fromiter((b.total_ms for b in node), dtype=float, count=len(node))
    mask = (t_end <= elapsed_ms) & (next_end > elapsed_ms) & (total > elapsed_ms)
    return total[mask] - elapsed_ms


def kde_log_score(samples: np.ndarray, observed_ms: float) -> float | None:
    """Log density a Scott's-rule Gaussian KDE (log1p ms) gives ``observed``.

    Returns ``None`` when the KDE is undefined (fewer than two samples or zero
    spread), so the event is dropped from the log-score metric rather than
    scored on a degenerate density.
    """

    if samples.size < 2:
        return None
    transformed = np.log1p(samples)
    if float(np.ptp(transformed)) == 0.0:
        return None
    kde = gaussian_kde(transformed)  # Scott's rule bandwidth (established tool)
    return float(kde.logpdf(math.log1p(observed_ms))[0])


def hazard_decision_ms(
    residuals: np.ndarray, *, kv_cost_ms: float, restore_cost_fraction: float
) -> float:
    """Re-check time from the EXISTING optimizer on the conditional residuals.

    Threshold == kv (guard 0) and restore == rho*kv, the certified operating
    point. The residual is treated as the remaining call; the optimizer picks
    the expected-cost-optimal re-check offset within it.
    """

    threshold_ms = kv_cost_ms + _CERT_GUARD_MS
    return hazard_recheck_ms(
        residuals.tolist(),
        threshold_ms=threshold_ms,
        kv_cost_ms=kv_cost_ms,
        restore_cost_ms=restore_cost_fraction * kv_cost_ms,
    )


def task_clustered_bootstrap_ci(
    values_by_task: dict[str, list[float]],
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> tuple[float, float, float]:
    """Point mean + percentile CI of a per-event scalar, resampling tasks.

    A focused clustered bootstrap of a plain per-event mean; the repo's
    ``paired_task_cluster_bootstrap`` is bound to the trigger/utility decision
    schema and does not fit an arbitrary per-event scalar, so reusing it would
    mean fabricating trigger fields. Tasks (clusters) are resampled with
    replacement; every event keeps its task's weight.
    """

    if not values_by_task:
        raise ValueError("no per-task values to bootstrap")
    if replicates < 1:
        raise ValueError(f"replicates must be >= 1, got {replicates}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    per_task = [np.asarray(v, dtype=float) for v in values_by_task.values()]
    point = float(np.concatenate(per_task).mean())
    rng = np.random.default_rng(seed)
    n = len(per_task)
    means = np.empty(replicates, dtype=float)
    for b in range(replicates):
        idx = rng.integers(0, n, n)
        means[b] = np.concatenate([per_task[i] for i in idx]).mean()
    alpha = (1.0 - confidence) / 2.0
    return point, float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


@dataclass(frozen=True)
class StageOneConfig:
    fold_count: int
    kv_costs_ms: tuple[float, ...]
    restore_cost_fraction: float
    proximity_bandwidth_ms: float | None
    min_conditional_samples: int
    kill_divergence_frac: float
    bootstrap_replicates: int
    bootstrap_confidence: float
    bootstrap_seed: int
    decision_tolerance_ms: float


def _apply_proximity(
    residuals: np.ndarray,
    node: Sequence[_FitBoundary],
    elapsed_ms: float,
    bandwidth_ms: float | None,
) -> np.ndarray:
    """Optional hard temporal window on the E+B state (sensitivity knob).

    ``None`` (default) keeps the structural state selection only. A finite
    bandwidth additionally requires the fit boundary time within
    ``elapsed +/- bandwidth`` - a hard window (not a soft weight) so the
    residuals stay an unweighted list the hazard optimizer accepts.
    """

    if bandwidth_ms is None:
        return residuals
    # Recompute the mask with the extra proximity constraint. The residual
    # order matches boundary_state_residuals' internal mask, so re-derive from
    # the node to keep alignment explicit.
    t_end = np.fromiter((b.t_end_ms for b in node), dtype=float, count=len(node))
    next_end = np.fromiter((b.next_end_ms for b in node), dtype=float, count=len(node))
    total = np.fromiter((b.total_ms for b in node), dtype=float, count=len(node))
    mask = (
        (t_end <= elapsed_ms)
        & (next_end > elapsed_ms)
        & (total > elapsed_ms)
        & (np.abs(t_end - elapsed_ms) <= bandwidth_ms)
    )
    return total[mask] - elapsed_ms


@dataclass(frozen=True)
class _StageOneFoldResult:
    gains_by_task: dict[str, list[float]]
    divergence_by_kv: dict[float, list[bool]]
    event_total: int
    event_supported: int
    event_scored: int
    thin_e: int
    thin_eb: int


def _run_stage_one_fold(
    fold: tuple[Sequence[TimedChain], Sequence[TimedChain]],
    *,
    cfg: StageOneConfig,
) -> _StageOneFoldResult:
    train, test = fold
    fit_totals, fit_nodes = _fit_index(train)
    gains_by_task: dict[str, list[float]] = defaultdict(list)
    divergence_by_kv: dict[float, list[bool]] = {kv: [] for kv in cfg.kv_costs_ms}
    event_total = event_supported = event_scored = thin_e = thin_eb = 0
    for chain in test:
        for event in boundary_events(chain):
            event_total += 1
            e_res = elapsed_only_residuals(fit_totals, event.elapsed_ms)
            node = fit_nodes.get((event.boundary_index, event.atom), [])
            eb_res = _apply_proximity(
                boundary_state_residuals(node, event.elapsed_ms),
                node,
                event.elapsed_ms,
                cfg.proximity_bandwidth_ms,
            )
            if e_res.size < cfg.min_conditional_samples:
                thin_e += 1
                continue
            if eb_res.size < cfg.min_conditional_samples:
                thin_eb += 1
                continue
            event_supported += 1
            for kv in cfg.kv_costs_ms:
                k_e = hazard_decision_ms(
                    e_res,
                    kv_cost_ms=kv,
                    restore_cost_fraction=cfg.restore_cost_fraction,
                )
                k_eb = hazard_decision_ms(
                    eb_res,
                    kv_cost_ms=kv,
                    restore_cost_fraction=cfg.restore_cost_fraction,
                )
                divergence_by_kv[kv].append(
                    not math.isclose(
                        k_e,
                        k_eb,
                        abs_tol=cfg.decision_tolerance_ms,
                        rel_tol=0.0,
                    )
                )
            score_e = kde_log_score(e_res, event.residual_ms)
            score_eb = kde_log_score(eb_res, event.residual_ms)
            if score_e is not None and score_eb is not None:
                event_scored += 1
                gains_by_task[event.task_id].append(score_eb - score_e)
    return _StageOneFoldResult(
        gains_by_task=dict(gains_by_task),
        divergence_by_kv=divergence_by_kv,
        event_total=event_total,
        event_supported=event_supported,
        event_scored=event_scored,
        thin_e=thin_e,
        thin_eb=thin_eb,
    )


def run_stage_one(
    chains: Sequence[TimedChain],
    cfg: StageOneConfig,
    *,
    workers: int = 1,
) -> dict[str, Any]:
    """Cross-fitted Stage-1 information test over task-grouped folds."""

    workers = resolve_worker_count(workers)
    usable = analysable_chains(chains)
    fold_cfg = Config(
        fold_count=cfg.fold_count,
        prefix_depth=4,
        min_prefix_evidence=1,
        atom_depth=4,
        token_bin_count=3,
        min_atom_count=10,
        min_family_count=20,
        tail_percentile=90.0,
    )
    worker = functools.partial(_run_stage_one_fold, cfg=cfg)
    folds = list(_task_folds(usable, fold_cfg))
    if workers == 1 or len(folds) < 2:
        per_fold = list(map(worker, folds))
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(folds))) as pool:
            per_fold = list(pool.map(worker, folds))

    gains_by_task: dict[str, list[float]] = defaultdict(list)
    divergence_by_kv: dict[float, list[bool]] = {kv: [] for kv in cfg.kv_costs_ms}
    event_total = event_supported = event_scored = thin_e = thin_eb = 0
    for result in per_fold:
        for task_id, gains in result.gains_by_task.items():
            gains_by_task[task_id].extend(gains)
        for kv, flags in result.divergence_by_kv.items():
            divergence_by_kv[kv].extend(flags)
        event_total += result.event_total
        event_supported += result.event_supported
        event_scored += result.event_scored
        thin_e += result.thin_e
        thin_eb += result.thin_eb

    per_kv = {
        f"{kv:.0f}": {
            "decision_count": len(flags),
            "divergence_fraction": float(np.mean(flags)) if flags else None,
        }
        for kv, flags in divergence_by_kv.items()
    }
    pooled_flags = [flag for flags in divergence_by_kv.values() for flag in flags]
    pooled_divergence = float(np.mean(pooled_flags)) if pooled_flags else None

    if gains_by_task:
        gain_point, gain_lo, gain_hi = task_clustered_bootstrap_ci(
            gains_by_task,
            replicates=cfg.bootstrap_replicates,
            confidence=cfg.bootstrap_confidence,
            seed=cfg.bootstrap_seed,
        )
    else:
        gain_point = gain_lo = gain_hi = None

    ci_covers_zero = (
        gain_lo is None or gain_hi is None or (gain_lo <= 0.0 <= gain_hi)
    )
    # SURVIVE demands the gain CI lie STRICTLY above zero: a CI that covers zero
    # (no evidence) OR sits below it (boundary evidence strictly hurts) both
    # kill. `ci_covers_zero` alone would wrongly pass a negative-gain result.
    gain_ci_above_zero = gain_lo is not None and gain_lo > 0.0
    divergence_below_bar = (
        pooled_divergence is None or pooled_divergence < cfg.kill_divergence_frac
    )
    killed = bool(divergence_below_bar or not gain_ci_above_zero)

    return {
        "census": {
            "input_chains": len(chains),
            "analysable_chains": len(usable),
            "tasks": len({c.task_id for c in usable}),
            "boundary_events": event_total,
            "events_supported": event_supported,
            "events_scored": event_scored,
            "events_thin_E": thin_e,
            "events_thin_EplusB": thin_eb,
            "support_coverage_fraction": (
                event_supported / event_total if event_total else None
            ),
        },
        "decision_divergence": {
            "per_kv": per_kv,
            "pooled_decision_count": len(pooled_flags),
            "pooled_divergence_fraction": pooled_divergence,
        },
        "log_score_gain": {
            "scored_event_count": event_scored,
            "scored_task_count": len(gains_by_task),
            "mean_gain": gain_point,
            "ci_low": gain_lo,
            "ci_high": gain_hi,
            "confidence": cfg.bootstrap_confidence,
            "ci_covers_zero": ci_covers_zero,
        },
        "kill_readout": {
            "kill_divergence_frac": cfg.kill_divergence_frac,
            "divergence_below_bar": divergence_below_bar,
            "ci_covers_zero": ci_covers_zero,
            "gain_ci_above_zero": gain_ci_above_zero,
            "verdict": "KILL" if killed else "SURVIVE",
        },
    }


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def _banner(final: bool) -> str:
    if final:
        return "FINAL - complete corpus"
    return (
        "PARTIAL - not final: validated against a subset; numbers are for "
        "script validation only, not findings."
    )


def render_markdown(results: dict[str, Any], provenance: dict[str, Any]) -> str:
    census = results["census"]
    div = results["decision_divergence"]
    gain = results["log_score_gain"]
    kill = results["kill_readout"]
    lines: list[str] = []
    lines.append("# Candidate C Stage-1 boundary-evidence kill test")
    lines.append("")
    lines.append(f"> **{_banner(provenance['final'])}**")
    lines.append(">")
    lines.append(
        "> EXPLORATORY. Durations replayed on our own hardware "
        f"({provenance['replayed_on']}); totals and boundary times both from "
        f"the segment-timeline re-run (no-mixing rule). Generated "
        f"{provenance['generated']}."
    )
    lines.append("")
    lines.append(f"**Verdict: {kill['verdict']}**")
    lines.append("")
    lines.append("## Census")
    lines.append("")
    lines.append("| quantity | value |")
    lines.append("| --- | --- |")
    for label, key in [
        ("input chains", "input_chains"),
        ("analysable chains (>=2 seg, has raw_total)", "analysable_chains"),
        ("tasks", "tasks"),
        ("boundary events", "boundary_events"),
        ("events supported (both arms)", "events_supported"),
        ("events scored (log-score defined)", "events_scored"),
        ("events dropped: thin E", "events_thin_E"),
        ("events dropped: thin E+B", "events_thin_EplusB"),
    ]:
        lines.append(f"| {label} | {census[key]} |")
    cov = census["support_coverage_fraction"]
    lines.append(
        f"| support coverage | {cov:.1%} |" if cov is not None else "| support coverage | - |"
    )
    lines.append("")
    lines.append("## Metric (ii): decision divergence via hazard_recheck_ms")
    lines.append("")
    lines.append(
        f"Certified operating point: rho={_CERT_RESTORE_COST_FRACTION}, guard "
        f"{_CERT_GUARD_MS:.0f}ms (threshold==kv). Pooled divergence: "
        + (
            f"{div['pooled_divergence_fraction']:.3%}"
            if div["pooled_divergence_fraction"] is not None
            else "-"
        )
        + f" over {div['pooled_decision_count']} decisions."
    )
    lines.append("")
    lines.append("| kv cost ms | decisions | divergence |")
    lines.append("| --- | --- | --- |")
    for kv, block in div["per_kv"].items():
        frac = block["divergence_fraction"]
        lines.append(
            f"| {kv} | {block['decision_count']} | "
            + (f"{frac:.3%}" if frac is not None else "-")
            + " |"
        )
    lines.append("")
    lines.append("## Metric (i): paired log-score gain (E+B over E)")
    lines.append("")
    if gain["mean_gain"] is None:
        lines.append("No scored events.")
    else:
        lines.append(
            f"Mean gain {gain['mean_gain']:.4f} nats; "
            f"{gain['confidence']:.0%} task-clustered bootstrap CI "
            f"[{gain['ci_low']:.4f}, {gain['ci_high']:.4f}] over "
            f"{gain['scored_task_count']} tasks / {gain['scored_event_count']} "
            f"events. CI covers zero: {gain['ci_covers_zero']}."
        )
    lines.append("")
    lines.append("## Kill readout")
    lines.append("")
    lines.append(
        f"KILL if pooled divergence < {kill['kill_divergence_frac']:.1%} OR the "
        "log-score gain CI is not strictly above zero; SURVIVE only if both hold."
    )
    lines.append("")
    lines.append(f"- divergence below bar: {kill['divergence_below_bar']}")
    lines.append(f"- CI covers zero: {kill['ci_covers_zero']}")
    lines.append(f"- gain CI strictly above zero: {kill['gain_ci_above_zero']}")
    lines.append(f"- **verdict: {kill['verdict']}**")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-dir", type=Path, required=True)
    parser.add_argument("--glob", default="*.wave_*.worker_*.jsonl")
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Smoke only: cap the number of wave files loaded.",
    )
    parser.add_argument(
        "--proximity-bandwidth-ms",
        type=float,
        default=None,
        help="Sensitivity knob: hard temporal window (+/- ms) on the E+B "
        "boundary state. Default None keeps structural state selection only.",
    )
    parser.add_argument(
        "--min-conditional-samples",
        type=int,
        default=5,
        help="Minimum fit samples per arm before an event is scored (a "
        "hazard/KDE estimate on fewer is noise). Default 5.",
    )
    parser.add_argument(
        "--kill-divergence-frac",
        type=float,
        default=0.01,
        help="Pooled decision-divergence kill threshold (memo ~1%%).",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--decision-tolerance-ms",
        type=float,
        default=1e-6,
        help="Two re-check times within this tolerance count as unchanged.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=resolve_worker_count(),
        help="CPU processes for independent folds; 1 is sequential.",
    )
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--final", action="store_true")
    return parser


def _default_output_paths(final: bool) -> tuple[Path, Path]:
    today = _dt.date.today().isoformat()
    suffix = "" if final else "-PARTIAL"
    stem = f"analysis/offline/boundary-evidence-stage1-{today}{suffix}"
    return Path(f"{stem}.json"), Path(f"{stem}.md")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    default_json, default_md = _default_output_paths(args.final)
    out_json = args.out_json or default_json
    out_md = args.out_md or default_md
    print(_banner(args.final))

    files = sorted(args.traces_dir.glob(args.glob))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise ValueError(f"no trace files under {args.traces_dir}/{args.glob}")
    samples = extract_many_segment_latency_samples(files, skip_concurrent=True)
    chains = build_timed_chains(samples)

    cfg = StageOneConfig(
        fold_count=args.fold_count,
        kv_costs_ms=_CERT_KV_COSTS_MS,
        restore_cost_fraction=_CERT_RESTORE_COST_FRACTION,
        proximity_bandwidth_ms=args.proximity_bandwidth_ms,
        min_conditional_samples=args.min_conditional_samples,
        kill_divergence_frac=args.kill_divergence_frac,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_confidence=args.bootstrap_confidence,
        bootstrap_seed=args.bootstrap_seed,
        decision_tolerance_ms=args.decision_tolerance_ms,
    )
    results = run_stage_one(chains, cfg, workers=args.workers)
    provenance = {
        "exploratory": True,
        "final": bool(args.final),
        "replayed_on": "our_hardware",
        "traces_dir": str(args.traces_dir),
        "corpus_file_count": len(files),
        "workers": args.workers,
        "git_sha": _git_sha(),
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "operating_point": {
            "kv_costs_ms": list(_CERT_KV_COSTS_MS),
            "guard_ms": _CERT_GUARD_MS,
            "restore_cost_fraction": _CERT_RESTORE_COST_FRACTION,
        },
    }
    payload = {"provenance": provenance, "config": cfg.__dict__, **results}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, default=list), encoding="utf-8")
    out_md.write_text(render_markdown(results, provenance), encoding="utf-8")

    verdict = results["kill_readout"]["verdict"]
    div = results["decision_divergence"]["pooled_divergence_fraction"]
    gain = results["log_score_gain"]
    print(
        f"verdict={verdict} pooled_divergence="
        + (f"{div:.3%}" if div is not None else "-")
        + " mean_gain="
        + (f"{gain['mean_gain']:.4f}" if gain["mean_gain"] is not None else "-")
        + " ci=["
        + (
            f"{gain['ci_low']:.4f},{gain['ci_high']:.4f}"
            if gain["ci_low"] is not None
            else "-"
        )
        + "]"
    )
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
