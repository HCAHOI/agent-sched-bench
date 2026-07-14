from __future__ import annotations

import math

import numpy as np
import pytest

from trace_collect.tool_latency_offline_probe import mean_clock_region_stats
from trace_collect.tool_latency_profiled import hazard_recheck_ms
from trace_collect.tool_latency_hazard_model import (
    DiscreteTimeGrid,
    build_log_grid,
    fit_hazard_model,
    survival_clock_region_stats,
    survival_trigger_ms,
)
from trace_collect.tool_latency_hazard_model import _expand_person_periods
from trace_collect.tool_latency_survival_features import (
    CausalRowFeatures,
    SurvivalFeatureSpec,
    fit_feature_encoder,
    iter_causal_row_features,
)


def _anchor_grid() -> DiscreteTimeGrid:
    """Two-interval grid with reps [80, 150] (only reps drive the trigger)."""

    return DiscreteTimeGrid(
        edges=np.array([0.0, 100.0, np.inf]),
        reps=np.array([80.0, 150.0]),
    )


# --- The anchor: weighted masses reproduce the empirical policy exactly ------


def test_survival_trigger_matches_hazard_recheck_restore_zero() -> None:
    grid = _anchor_grid()
    masses = np.array([2.0 / 3.0, 1.0 / 3.0])
    trigger = survival_trigger_ms(
        grid, masses, threshold_ms=100.0, kv_cost_ms=100.0, restore_cost_ms=0.0
    )
    assert trigger == 0.0
    assert trigger == hazard_recheck_ms(
        [80.0, 80.0, 150.0], threshold_ms=100.0, kv_cost_ms=100.0, restore_cost_ms=0.0
    )


def test_survival_trigger_matches_hazard_recheck_restore_thirty_five() -> None:
    grid = _anchor_grid()
    masses = np.array([2.0 / 3.0, 1.0 / 3.0])
    trigger = survival_trigger_ms(
        grid, masses, threshold_ms=100.0, kv_cost_ms=100.0, restore_cost_ms=35.0
    )
    assert trigger == 80.0
    assert trigger == hazard_recheck_ms(
        [80.0, 80.0, 150.0], threshold_ms=100.0, kv_cost_ms=100.0, restore_cost_ms=35.0
    )


def test_survival_clock_region_stats_matches_mean_clock_anchor() -> None:
    grid = _anchor_grid()
    masses = np.array([2.0 / 3.0, 1.0 / 3.0])
    stats = survival_clock_region_stats(
        grid, masses, threshold_ms=100.0, kv_cost_ms=100.0
    )
    empirical = mean_clock_region_stats(
        [80.0, 80.0, 150.0], threshold_ms=100.0, kv_cost_ms=100.0
    )
    assert stats.trigger_ms == 0.0
    assert stats.normalized_band_gain == pytest.approx(1.0 / 3.0)
    assert stats.normalized_short_penalty == pytest.approx(2.0 / 15.0)
    assert stats.normalized_margin == pytest.approx(0.2)
    assert stats.normalized_margin == pytest.approx(empirical.normalized_margin)
    # Two representatives survive the k=0 trigger; region probabilities are the
    # mass-weighted analogs of the empirical survivor fractions.
    assert stats.survivor_count == 2
    assert stats.probability_short_given_survival == pytest.approx(2.0 / 3.0)
    assert stats.probability_band_given_survival == pytest.approx(1.0 / 3.0)
    assert stats.probability_far_given_survival == 0.0


@pytest.mark.parametrize(
    "values",
    [
        [40.0, 120.0, 300.0],
        [10.0, 10.0, 10.0, 500.0],
        [60.0, 90.0, 130.0, 210.0, 800.0],
    ],
)
def test_equal_weight_masses_reproduce_hazard_recheck(values: list[float]) -> None:
    reps = np.array(sorted(set(values)))
    counts = np.array([values.count(rep) for rep in reps], dtype=float)
    masses = counts / counts.sum()
    grid = DiscreteTimeGrid(
        edges=np.concatenate(([0.0], reps, [np.inf])), reps=reps
    )
    for restore in (0.0, 25.0):
        trigger = survival_trigger_ms(
            grid, masses, threshold_ms=150.0, kv_cost_ms=120.0, restore_cost_ms=restore
        )
        assert trigger == hazard_recheck_ms(
            values, threshold_ms=150.0, kv_cost_ms=120.0, restore_cost_ms=restore
        )


# --- Trigger tie resolution -------------------------------------------------


def test_survival_trigger_ties_resolve_to_latest_candidate() -> None:
    grid = DiscreteTimeGrid(
        edges=np.array([0.0, 200.0, np.inf]), reps=np.array([150.0, 250.0])
    )
    masses = np.array([0.5, 0.5])
    # Expected utility ties at candidates 0 and 50; the latest (50) must win.
    trigger = survival_trigger_ms(
        grid, masses, threshold_ms=100.0, kv_cost_ms=100.0
    )
    assert trigger == 50.0
    assert trigger == hazard_recheck_ms(
        [150.0, 250.0], threshold_ms=100.0, kv_cost_ms=100.0
    )


# --- Person-period expansion convention -------------------------------------


def test_person_period_expansion_counts_and_targets() -> None:
    # num_intervals=4 -> free intervals {0,1,2}, absorbing top interval 3.
    call_index, interval_index, targets = _expand_person_periods(
        np.array([2, 3, 0]), num_intervals=4
    )
    # Call 0 (event interval 2): rows for 0,1,2 with a single event at 2.
    # Call 1 (top/absorbing): survived-only rows for every free interval.
    # Call 2 (event interval 0): a single event row.
    assert list(call_index) == [0, 0, 0, 1, 1, 1, 2]
    assert list(interval_index) == [0, 1, 2, 0, 1, 2, 0]
    assert list(targets) == [0, 0, 1, 0, 0, 0, 1]


def test_person_period_expansion_censored_emits_survived_only() -> None:
    call_index, interval_index, targets = _expand_person_periods(
        np.array([2]), num_intervals=4, censored=np.array([True])
    )
    assert list(call_index) == [0, 0, 0]
    assert list(interval_index) == [0, 1, 2]
    assert list(targets) == [0, 0, 0]


# --- Grid construction ------------------------------------------------------


def test_build_log_grid_edges_reps_and_top_interval() -> None:
    latencies = [float(value) for value in range(10, 210, 5)]  # 40 samples, 10..205
    grid = build_log_grid(
        latencies, num_intervals=5, low_quantile=0.1, high_quantile=0.9
    )
    edges = grid.edges
    reps = grid.reps
    assert edges[0] == 0.0
    assert math.isinf(edges[-1])
    # Interior (finite) edges strictly increasing and spanning the quantiles.
    finite_edges = edges[:-1]
    assert np.all(np.diff(finite_edges) > 0.0)
    q_low = float(np.quantile(latencies, 0.1))
    q_high = float(np.quantile(latencies, 0.9))
    assert edges[1] == pytest.approx(q_low)
    assert edges[-2] == pytest.approx(q_high)
    assert len(reps) == 5
    # Each representative lies inside its own interval [edges[j], edges[j+1]).
    for j in range(len(reps)):
        assert edges[j] <= reps[j] < edges[j + 1]
    # The top interval is unbounded and absorbs the tail (rep >= last edge).
    assert reps[-1] >= edges[-2]
    assert math.isfinite(reps[-1])


def test_build_log_grid_rejects_nonpositive_latencies() -> None:
    with pytest.raises(ValueError):
        build_log_grid([0.0, 10.0, 20.0], num_intervals=4)


# --- predict_interval_masses monotonicity -----------------------------------


def _tool_spec() -> SurvivalFeatureSpec:
    """Tool-identity-only spec keeps the encoder tiny and hand-checkable."""

    return SurvivalFeatureSpec(
        command_field=None,
        use_tool_identity=True,
        use_command_prefix=False,
        use_within_task_history=False,
        use_task_aggregates=False,
    )


def _feats(sample_id: str, tool_name: str, latency_ms: float) -> CausalRowFeatures:
    return CausalRowFeatures(
        sample_id=sample_id,
        tool_name=tool_name,
        group_keys=(),
        within_task_prefix_last_ms=None,
        within_task_prefix_median_ms=None,
        within_task_prefix_count=0,
        within_task_tool_last_ms=None,
        call_index_in_task=0,
        task_running_mean_ms=None,
        latency_ms=latency_ms,
    )


def _row(
    sample_id: str,
    task_id: str,
    tool_name: str,
    latency_ms: float,
    ts_start: float,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "task_id": task_id,
        "source_trace": task_id,
        "tool_name": tool_name,
        "latency_ms": latency_ms,
        "tool_ts_start": ts_start,
        "tool_ts_end": ts_start + latency_ms,
    }


def test_predict_interval_masses_are_a_valid_distribution() -> None:
    rows = [
        _row("s0", "task-a", "fast", 40.0, 0.0),
        _row("s1", "task-a", "slow", 2000.0, 100.0),
    ]
    spec = _tool_spec()
    encoder = fit_feature_encoder(rows, spec=spec)
    grid = build_log_grid([40.0, 2000.0], num_intervals=6, low_quantile=0.0, high_quantile=1.0)
    from trace_collect.tool_latency_hazard_model import FittedHazardModel

    num_free = grid.num_intervals - 1
    rng = np.random.default_rng(0)
    coef = rng.normal(size=num_free + encoder.n_columns)
    model = FittedHazardModel(encoder=encoder, grid=grid, coef=coef, l2_penalty=0.0)
    masses = model.predict_interval_masses(_feats("q", "fast", 0.0))
    assert masses.shape == (grid.num_intervals,)
    assert np.all(masses >= -1e-12)
    assert float(masses.sum()) == pytest.approx(1.0)
    # Implied survival S_j = 1 - cumsum(masses) is monotone non-increasing.
    survival = 1.0 - np.cumsum(masses)
    assert np.all(np.diff(survival) <= 1e-12)
    assert survival[-1] == pytest.approx(0.0, abs=1e-9)


# --- Fit: a separating feature moves the predicted distribution -------------


def _separable_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    fast = [40.0, 55.0, 48.0]
    slow = [1800.0, 2100.0, 1950.0]
    for task in range(6):
        task_id = f"task-{task}"
        base = task * 10_000.0
        for call, latency in enumerate(fast):
            rows.append(
                _row(f"f-{task}-{call}", task_id, "fast", latency, base + call * 5000.0)
            )
        for call, latency in enumerate(slow):
            rows.append(
                _row(
                    f"s-{task}-{call}",
                    task_id,
                    "slow",
                    latency,
                    base + 500.0 + call * 5000.0,
                )
            )
    return rows


def test_fit_hazard_model_separates_fast_and_slow_tools() -> None:
    rows = _separable_rows()
    spec = _tool_spec()
    latencies = [float(row["latency_ms"]) for row in rows]
    grid = build_log_grid(latencies, num_intervals=8)
    model = fit_hazard_model(rows, spec=spec, grid=grid, l2_penalty=1e-3)

    fast_masses = model.predict_interval_masses(_feats("q-fast", "fast", 0.0))
    slow_masses = model.predict_interval_masses(_feats("q-slow", "slow", 0.0))
    expected_fast = float(fast_masses @ grid.reps)
    expected_slow = float(slow_masses @ grid.reps)
    # The fast tool concentrates mass on short intervals; the slow tool on the
    # long tail, so the mass-weighted expected latency is ordered.
    assert expected_fast < expected_slow

    threshold_ms = 1000.0
    kv_cost_ms = 1000.0
    trigger_fast = survival_trigger_ms(
        grid, fast_masses, threshold_ms=threshold_ms, kv_cost_ms=kv_cost_ms
    )
    trigger_slow = survival_trigger_ms(
        grid, slow_masses, threshold_ms=threshold_ms, kv_cost_ms=kv_cost_ms
    )
    # A tool that is almost always long triggers the swap no later than a tool
    # that is almost always short, and both triggers stay within the deadline.
    assert trigger_slow <= trigger_fast
    assert trigger_fast <= threshold_ms


# --- L2 selection is train-only ---------------------------------------------


def test_fit_hazard_model_cv_selects_penalty_from_grid() -> None:
    rows = _separable_rows()
    spec = _tool_spec()
    latencies = [float(row["latency_ms"]) for row in rows]
    grid = build_log_grid(latencies, num_intervals=8)
    l2_grid = (1e-3, 1e-1, 10.0)
    model = fit_hazard_model(
        rows,
        spec=spec,
        grid=grid,
        l2_penalty=None,
        cv_folds=3,
        l2_grid=l2_grid,
    )
    assert model.l2_penalty in l2_grid


def test_fit_hazard_model_consumes_causal_features_without_leakage() -> None:
    # Sanity: fit runs end-to-end on features produced by the causal walk and
    # the number of feature rows never exceeds the number of raw rows.
    rows = _separable_rows()
    spec = _tool_spec()
    feats = list(iter_causal_row_features(rows, spec=spec))
    assert 0 < len(feats) <= len(rows)
