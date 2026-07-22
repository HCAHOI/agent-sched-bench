"""Unit tests for the Candidate C Stage-1 boundary-evidence kill test.

Fixtures are hand-built TimedChains with known boundary times so the
no-mixing rule, fold logic, conditional-subset selection, and kill-criterion
arithmetic are checkable without any replay corpus on disk.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.exploration.analyze_boundary_evidence_stage1 import (
    StageOneConfig,
    TimedBoundary,
    TimedChain,
    _CERT_KV_COSTS_MS,
    _CERT_RESTORE_COST_FRACTION,
    analysable_chains,
    boundary_events,
    boundary_state_residuals,
    build_timed_chains,
    elapsed_only_residuals,
    hazard_decision_ms,
    kde_log_score,
    run_stage_one,
    task_clustered_bootstrap_ci,
)
from scripts.exploration.analyze_segment_variance import Config, _task_folds
from trace_collect.tool_latency_dataset import SegmentLatencySample
from trace_collect.tool_latency_profiled import hazard_recheck_ms


def _chain(task: str, atoms_ends: list[tuple[str, float]], raw_total: float | None) -> TimedChain:
    return TimedChain(
        task_id=task,
        source_trace=f"trace-{task}",
        action_id=f"act-{task}",
        raw_total_ms=raw_total,
        boundaries=tuple(TimedBoundary(atom=a, t_end_ms=t) for a, t in atoms_ends),
    )


def test_build_timed_chains_keeps_boundary_times_and_orders_segments() -> None:
    samples = [
        SegmentLatencySample(
            sample_id="s1", source_trace="tr", task_id="task-a", agent_id="ag",
            action_id="a1", tool_name="exec", segment_index=1,
            segment_command="python3 x.py", segment_ms=10.0, t_start_ms=2.0,
            t_end_ms=12.0, parent_chain_command="cd /t && python3 x.py",
            parent_total_ms=15.0, parent_raw_total_ms=20.0,
        ),
        SegmentLatencySample(
            sample_id="s0", source_trace="tr", task_id="task-a", agent_id="ag",
            action_id="a1", tool_name="exec", segment_index=0,
            segment_command="cd /t", segment_ms=1.0, t_start_ms=0.0, t_end_ms=1.0,
            parent_chain_command="cd /t && python3 x.py", parent_total_ms=15.0,
            parent_raw_total_ms=20.0,
        ),
    ]
    (chain,) = build_timed_chains(samples)
    # Ordered by segment_index; boundary atoms via atom_key; t_end kept verbatim.
    assert [b.atom for b in chain.boundaries] == ["cd", "python3"]
    assert [b.t_end_ms for b in chain.boundaries] == [1.0, 12.0]
    # raw_total is the segment-timeline re-run wall time, NOT parent_total_ms.
    assert chain.raw_total_ms == pytest.approx(20.0)


def test_boundary_events_use_raw_total_not_original_duration() -> None:
    # No-mixing rule: residual is raw_total - boundary_t_end, never the bogus
    # (original-trace) total. Here raw_total=100; a mixing bug using 999 would
    # give residual 959 at boundary 0.
    chain = _chain("t", [("cd", 40.0), ("python3", 90.0)], raw_total=100.0)
    events = list(boundary_events(chain))
    assert len(events) == 1  # only interior boundary 0
    (event,) = events
    assert event.boundary_index == 0
    assert event.atom == "cd"
    assert event.elapsed_ms == pytest.approx(40.0)
    assert event.total_ms == pytest.approx(100.0)
    assert event.residual_ms == pytest.approx(60.0)  # 100 - 40, not 999 - 40


def test_boundary_events_skip_nonpositive_residual_and_elapsed() -> None:
    # Boundary at/after the total leaves no residual -> skipped.
    chain = _chain("t", [("cd", 100.0), ("git", 150.0)], raw_total=100.0)
    assert list(boundary_events(chain)) == []


def test_analysable_chains_filters_short_and_missing_total() -> None:
    good = _chain("t1", [("cd", 1.0), ("git", 5.0)], raw_total=10.0)
    single = _chain("t2", [("cd", 1.0)], raw_total=10.0)
    no_total = _chain("t3", [("cd", 1.0), ("git", 5.0)], raw_total=None)
    assert analysable_chains([good, single, no_total]) == [good]


def test_elapsed_only_residuals_conditions_on_running_chains() -> None:
    totals = np.array([50.0, 120.0, 200.0])
    # at t=100 only 120 and 200 are still running -> residuals 20, 100.
    res = elapsed_only_residuals(totals, 100.0)
    assert sorted(res.tolist()) == pytest.approx([20.0, 100.0])


def test_boundary_state_residuals_matches_only_the_shared_state() -> None:
    # Fit node (index k, atom a) entries: (t_end, next_end, total). The E+B set
    # keeps chains whose boundary is reached, is the most recent one, and are
    # still running, residual = total - t.
    from scripts.exploration.analyze_boundary_evidence_stage1 import _FitBoundary

    node = [
        _FitBoundary(t_end_ms=40.0, next_end_ms=200.0, total_ms=300.0),  # in-state at t=100
        _FitBoundary(t_end_ms=40.0, next_end_ms=80.0, total_ms=300.0),   # next boundary passed -> out
        _FitBoundary(t_end_ms=150.0, next_end_ms=math.inf, total_ms=300.0),  # boundary not reached
        _FitBoundary(t_end_ms=40.0, next_end_ms=200.0, total_ms=90.0),   # already finished (<t)
    ]
    res = boundary_state_residuals(node, 100.0)
    assert res.tolist() == pytest.approx([200.0])  # only the first: 300 - 100


def test_kde_log_score_guards_degenerate_samples() -> None:
    assert kde_log_score(np.array([5.0]), 5.0) is None  # < 2 samples
    assert kde_log_score(np.array([5.0, 5.0, 5.0]), 5.0) is None  # zero spread
    score = kde_log_score(np.array([10.0, 20.0, 30.0, 40.0]), 25.0)
    assert isinstance(score, float) and math.isfinite(score)


def test_hazard_decision_delegates_to_existing_optimizer() -> None:
    residuals = np.array([100.0, 2000.0, 3000.0, 6000.0])
    kv = 2000.0
    got = hazard_decision_ms(
        residuals, kv_cost_ms=kv, restore_cost_fraction=_CERT_RESTORE_COST_FRACTION
    )
    want = hazard_recheck_ms(
        residuals.tolist(),
        threshold_ms=kv,  # guard 0 -> threshold == kv
        kv_cost_ms=kv,
        restore_cost_ms=_CERT_RESTORE_COST_FRACTION * kv,
    )
    assert got == pytest.approx(want)


def test_task_clustered_bootstrap_ci_point_and_bracket() -> None:
    gains = {"t1": [1.0, 1.0], "t2": [1.0], "t3": [1.0, 1.0]}
    point, lo, hi = task_clustered_bootstrap_ci(
        gains, replicates=200, confidence=0.95, seed=0
    )
    # All gains are 1.0 -> point mean 1.0 and a degenerate CI at 1.0.
    assert point == pytest.approx(1.0)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)
    # Determinism under a fixed seed.
    again = task_clustered_bootstrap_ci(gains, replicates=200, confidence=0.95, seed=0)
    assert (point, lo, hi) == again


def test_task_folds_reused_are_task_disjoint() -> None:
    chains = [_chain(f"t{i}", [("cd", 1.0), ("git", 5.0)], 10.0) for i in range(4)]
    cfg = Config(
        fold_count=2, prefix_depth=4, min_prefix_evidence=1, atom_depth=4,
        token_bin_count=2, min_atom_count=1, min_family_count=1, tail_percentile=90.0,
    )
    for train, test in _task_folds(chains, cfg):
        train_tasks = {c.task_id for c in train}
        test_tasks = {c.task_id for c in test}
        assert not (train_tasks & test_tasks)  # no task leaks across the split


def _homogeneous_corpus() -> list[TimedChain]:
    # Every chain shares family cd>>python3 with the same boundary geometry, so
    # at any elapsed t the E population and the E+B (k, atom) state coincide ->
    # identical residual sets -> zero divergence and zero log-score gain.
    chains: list[TimedChain] = []
    for i in range(8):
        for j in range(6):  # 6 calls per task -> ample conditional support
            total = 3000.0 + 10.0 * j
            chains.append(
                TimedChain(
                    task_id=f"task{i}",
                    source_trace=f"tr{i}-{j}",
                    action_id=f"a{i}-{j}",
                    raw_total_ms=total,
                    boundaries=(
                        TimedBoundary("cd", 50.0),
                        TimedBoundary("python3", 1500.0),
                    ),
                )
            )
    return chains


def test_run_stage_one_identical_arms_kill() -> None:
    cfg = StageOneConfig(
        fold_count=4,
        kv_costs_ms=_CERT_KV_COSTS_MS,
        restore_cost_fraction=_CERT_RESTORE_COST_FRACTION,
        proximity_bandwidth_ms=None,
        min_conditional_samples=5,
        kill_divergence_frac=0.01,
        bootstrap_replicates=200,
        bootstrap_confidence=0.95,
        bootstrap_seed=0,
        decision_tolerance_ms=1e-6,
    )
    results = run_stage_one(_homogeneous_corpus(), cfg)
    assert run_stage_one(_homogeneous_corpus(), cfg, workers=2) == results
    # E and E+B are the same set on every event -> no decision ever changes and
    # the paired log-score gain is exactly zero.
    assert results["decision_divergence"]["pooled_divergence_fraction"] == pytest.approx(0.0)
    assert results["log_score_gain"]["mean_gain"] == pytest.approx(0.0, abs=1e-9)
    kill = results["kill_readout"]
    assert kill["divergence_below_bar"] is True
    assert kill["ci_covers_zero"] is True
    assert kill["verdict"] == "KILL"
    assert results["census"]["boundary_events"] > 0
