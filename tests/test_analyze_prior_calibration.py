from __future__ import annotations

import json
import math

import numpy as np
import pytest

from scripts.certification.analyze_prior_calibration import (
    CalibrationConfig,
    EcdfForecast,
    _coverage_verdict_suppressed,
    _cvm_omega2_weighted,
    _require_certified_discipline,
    build_parser,
    censored_sample_ids,
    cramer_von_mises,
    cvm_bootstrap,
    ecdf_pit,
    ecdf_quantile,
    ecdf_survival,
    ratio_bootstrap,
    run_prior_calibration,
    score_decisions,
)
from trace_collect.tool_latency_dataset import ToolLatencySample
from trace_collect.trace_data import CURRENT_TRACE_FORMAT_VERSION


# --------------------------------------------------------------------------- #
# ECDF primitives.
# --------------------------------------------------------------------------- #
def test_ecdf_pit_mid_rank_ties() -> None:
    # Mid-rank: a tied observation sits at the MIDPOINT of the jump, not either
    # edge. With [1, 2, 2, 3] and y=2 the ECDF jumps from 1/4 to 3/4, so u=1/2.
    assert ecdf_pit([1.0, 2.0, 2.0, 3.0], 2.0) == pytest.approx(0.5)
    assert ecdf_pit([1.0, 2.0, 2.0, 3.0], 0.5) == pytest.approx(0.0)
    assert ecdf_pit([1.0, 2.0, 2.0, 3.0], 9.0) == pytest.approx(1.0)


def test_ecdf_quantile_is_the_step_ecdf_inverse() -> None:
    # Q_q = inf{x : F(x) >= q}: the inverse of the same step curve the certified
    # survival estimator reads, so F(Q_q) >= q must hold exactly.
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    for q in (0.5, 0.9, 0.95, 0.99):
        predicted = ecdf_quantile(values, q)
        cdf = 1.0 - ecdf_survival(values, predicted)
        assert cdf >= q - 1e-12
    assert ecdf_quantile(values, 0.5) == 30.0
    assert ecdf_quantile(values, 0.9) == 50.0


def test_ecdf_survival_matches_the_estimator_form() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert ecdf_survival(values, 2.0) == pytest.approx(0.5)
    assert ecdf_survival(values, 4.0) == pytest.approx(0.0)
    assert ecdf_survival(values, 0.0) == pytest.approx(1.0)


def test_empty_forecast_fails_fast() -> None:
    with pytest.raises(ValueError):
        ecdf_pit([], 1.0)
    with pytest.raises(ValueError):
        ecdf_quantile([], 0.9)
    with pytest.raises(ValueError):
        ecdf_survival([], 1.0)


# --------------------------------------------------------------------------- #
# CRPS: the closed form against hand computation.
# --------------------------------------------------------------------------- #
def test_crps_closed_form_two_members_hand_computed() -> None:
    # x = [1, 3], y = 2.
    #   (1/n) sum|x_i - y|          = (1/2)(1 + 1)            = 1
    #   (1/(2 n^2)) sum sum|x_i-x_j| = (1/8)(0 + 2 + 2 + 0)   = 0.5
    #   CRPS                                                  = 0.5
    assert EcdfForecast.from_values([1.0, 3.0]).crps(2.0) == pytest.approx(0.5)


def test_crps_closed_form_three_members_hand_computed() -> None:
    # x = [0, 1, 4], y = 2.
    #   (1/3)(2 + 1 + 2)                    = 5/3
    #   (1/(2*9)) * 2*(1 + 4 + 3)           = 16/18 = 8/9
    #   CRPS = 5/3 - 8/9                    = 7/9
    assert EcdfForecast.from_values([0.0, 1.0, 4.0]).crps(2.0) == pytest.approx(7.0 / 9.0)


def test_crps_matches_brute_force_double_sum() -> None:
    # The O(n) prefix/sorted-weight identities must equal the literal definition.
    rng = np.random.default_rng(7)
    members = rng.gamma(2.0, 500.0, size=64)
    forecast = EcdfForecast.from_values(members)
    n = len(members)
    for y in (0.0, 100.0, 750.0, 5000.0):
        brute = float(np.mean(np.abs(members - y))) - float(
            np.sum(np.abs(members[:, None] - members[None, :]))
        ) / (2.0 * n * n)
        assert forecast.crps(y) == pytest.approx(brute, rel=1e-12, abs=1e-9)


def test_crps_of_a_point_mass_is_absolute_error() -> None:
    # Degenerate ECDF: the spread term vanishes and CRPS collapses to |x - y|.
    assert EcdfForecast.from_values([5.0, 5.0]).crps(9.0) == pytest.approx(4.0)


def test_crps_is_unsorted_input_invariant() -> None:
    a = EcdfForecast.from_values([4.0, 1.0, 9.0, 2.0]).crps(3.0)
    b = EcdfForecast.from_values([1.0, 2.0, 4.0, 9.0]).crps(3.0)
    assert a == pytest.approx(b)


# --------------------------------------------------------------------------- #
# PIT uniformity: calibrated fixture passes, miscalibrated fixture is detected.
# --------------------------------------------------------------------------- #
def _pit_sample(node_values, observations) -> list[float]:
    ordered = sorted(node_values)
    return [ecdf_pit(ordered, y) for y in observations]


def test_pit_is_uniform_for_a_perfectly_calibrated_forecast() -> None:
    # Forecast and observations drawn from the SAME law => u ~ U[0, 1]. We check
    # the mean, the bin balance, and the CvM distance the readout quotes.
    rng = np.random.default_rng(0)
    node = rng.exponential(1000.0, size=4000)
    observations = rng.exponential(1000.0, size=4000)
    pit = _pit_sample(node, observations)

    assert float(np.mean(pit)) == pytest.approx(0.5, abs=0.02)
    counts, _ = np.histogram(pit, bins=10, range=(0.0, 1.0))
    assert counts.min() > 0.7 * len(pit) / 10
    assert counts.max() < 1.3 * len(pit) / 10
    _, omega2 = cramer_von_mises(pit)
    # omega2 = integral (F_n(u) - u)^2 du; ~1e-4 for a calibrated sample of this
    # size, orders of magnitude below the miscalibrated case asserted below.
    assert omega2 < 1e-3


def test_pit_detects_a_miscalibrated_forecast() -> None:
    # Same observations, but the forecast is scale-shifted (predicts 3x too
    # short). PIT must pile up at the top of [0, 1] and CvM must blow up.
    rng = np.random.default_rng(0)
    node = rng.exponential(1000.0, size=4000) / 3.0
    observations = rng.exponential(1000.0, size=4000)
    pit = _pit_sample(node, observations)

    assert float(np.mean(pit)) > 0.65
    _, omega2_bad = cramer_von_mises(pit)
    _, omega2_good = cramer_von_mises(
        _pit_sample(rng.exponential(1000.0, size=4000), observations)
    )
    assert omega2_bad > 20.0 * omega2_good
    assert omega2_bad > 1e-2


def test_cvm_of_an_exactly_uniform_grid_is_minimal() -> None:
    # The order statistics of a perfect uniform grid sit exactly on the (2i-1)/2n
    # midpoints, so W2 attains its floor 1/(12N).
    n = 500
    grid = [(2 * i - 1) / (2 * n) for i in range(1, n + 1)]
    w2, omega2 = cramer_von_mises(grid)
    assert w2 == pytest.approx(1.0 / (12.0 * n))
    assert omega2 == pytest.approx(1.0 / (12.0 * n * n))


def test_cvm_weighted_rank_block_algebra_matches_brute_force() -> None:
    # The rank-block closed form is the only bespoke math in this lane, so it is
    # brute-forced rather than compared against itself: expand the multiset with
    # np.repeat and run the plain statistic over it. Trials deliberately include
    # repeats (multiplicity > 1) and dropped points (multiplicity 0), which is
    # exactly what a task-cluster resample produces.
    rng = np.random.default_rng(11)
    for _ in range(200):
        size = int(rng.integers(1, 40))
        pit = np.sort(rng.random(size))
        multiplicity = rng.integers(0, 4, size=size)
        if multiplicity.sum() == 0:
            multiplicity[rng.integers(0, size)] = 1
        expanded = np.repeat(pit, multiplicity)
        _, expected = cramer_von_mises(expanded)
        assert _cvm_omega2_weighted(pit, multiplicity) == pytest.approx(
            expected, rel=1e-12, abs=1e-12
        )


def test_cvm_weighted_form_handles_an_all_zero_multiplicity() -> None:
    # A replicate that drops every point has no statistic; it must return nan
    # rather than dividing by zero.
    assert math.isnan(_cvm_omega2_weighted(np.array([0.2, 0.7]), np.array([0, 0])))


def test_cvm_bootstrap_point_matches_the_direct_statistic() -> None:
    rng = np.random.default_rng(3)
    pit = rng.random(300)
    result = cvm_bootstrap(
        pit,
        list(range(300)),
        task_count=300,
        replicates=32,
        confidence_level=0.95,
        seed=0,
    )
    _, omega2 = cramer_von_mises(pit)
    assert result["point"] == pytest.approx(omega2)
    assert result["interval"]["low"] <= result["point"] <= result["interval"]["high"]


def test_cvm_bootstrap_returns_the_full_key_set_when_empty() -> None:
    # The renderer reads w2 and sample_count; an empty PIT sample must not turn a
    # degenerate run into a KeyError inside markdown rendering.
    empty = cvm_bootstrap(
        [], [], task_count=4, replicates=8, confidence_level=0.95, seed=0
    )
    populated = cvm_bootstrap(
        [0.1, 0.6], [0, 1], task_count=2, replicates=8, confidence_level=0.95, seed=0
    )
    assert set(empty) == set(populated)
    assert empty["sample_count"] == 0
    assert math.isnan(empty["w2"])
    assert math.isnan(empty["point"])


# --------------------------------------------------------------------------- #
# Coverage arithmetic and skill scores.
# --------------------------------------------------------------------------- #
def test_ratio_bootstrap_recovers_the_point_ratio() -> None:
    numerator = np.array([1.0, 3.0, 0.0, 2.0])
    denominator = np.array([4.0, 4.0, 4.0, 4.0])
    stats = ratio_bootstrap(
        numerator, denominator, replicates=1000, confidence_level=0.95, seed=0
    )
    assert stats["point"] == pytest.approx(6.0 / 16.0)
    assert stats["denominator"] == 16.0
    assert stats["interval"]["low"] <= stats["point"] <= stats["interval"]["high"]


def test_skill_is_zero_when_model_equals_baseline() -> None:
    # skill = 1 - CRPS_model / CRPS_baseline; identical forecasts must score 0
    # exactly, and the whole CI must collapse onto 0.
    values = np.array([3.0, 5.0, 11.0, 2.0])
    stats = ratio_bootstrap(
        values,
        values.copy(),
        replicates=1000,
        confidence_level=0.95,
        seed=0,
        transform=lambda ratio: 1.0 - ratio,
    )
    assert stats["point"] == pytest.approx(0.0)
    assert stats["interval"]["low"] == pytest.approx(0.0)
    assert stats["interval"]["high"] == pytest.approx(0.0)


def test_crps_skill_is_zero_when_the_node_is_the_pooled_node() -> None:
    # End-to-end: one tool and no command grouping means the deepest node IS the
    # pooled node, so both CRPS columns coincide and the skill must be exactly 0.
    samples_by_task = {
        f"t{i}": [_sample(f"t{i}", f"t{i}:0", 100.0 + 10.0 * i)] for i in range(6)
    }
    cfg = _cfg(fold_count=2)
    summary, decisions = run_prior_calibration(
        samples_by_task, sorted(samples_by_task), cfg, censored_ids=set()
    )
    for row in decisions:
        assert row["crps"] == pytest.approx(row["crps_pooled"])
    assert summary["crps"]["skill_vs_pooled"]["point"] == pytest.approx(0.0)
    assert summary["crps"]["skill_vs_tool_name"]["point"] == pytest.approx(0.0)


def test_coverage_arithmetic_against_a_hand_counted_fixture() -> None:
    # 4 eval calls, 2 folds, one tool. Coverage is exceedance count / call count;
    # here we recount it by hand from the emitted per-call rows.
    samples_by_task = {
        "t0": [_sample("t0", "t0:0", 100.0), _sample("t0", "t0:1", 8000.0)],
        "t1": [_sample("t1", "t1:0", 200.0), _sample("t1", "t1:1", 9000.0)],
    }
    cfg = _cfg(fold_count=2)
    summary, decisions = run_prior_calibration(
        samples_by_task, ["t0", "t1"], cfg, censored_ids=set()
    )
    p50 = next(c for c in summary["coverage"] if c["label"] == "p50")
    exceed = sum(
        1 for row in decisions if row["coverage"].get("p50", {}).get("exceeds")
    )
    scored = sum(1 for row in decisions if "p50" in row["coverage"])
    assert p50["scored_call_count"] == scored
    assert p50["empirical_exceedance"] == pytest.approx(exceed / scored)
    assert p50["nominal_exceedance"] == pytest.approx(0.5)


def test_final_rejects_any_weakened_statistical_knob() -> None:
    # --final must not be able to ship a FINAL-bannered artifact resting on a
    # weakened bootstrap. Each of the four certified knobs is rejected on its
    # own, and the message names the knob and the certified value.
    parser = build_parser()
    for flag, value in (
        ("--replicates", "100"),
        ("--cvm-replicates", "10"),
        ("--seed", "7"),
        ("--confidence-level", "0.8"),
    ):
        args = parser.parse_args(["--final", flag, value])
        with pytest.raises(ValueError, match="certified discipline") as excinfo:
            _require_certified_discipline(args, parser)
        assert flag in str(excinfo.value)


def test_final_accepts_the_certified_defaults() -> None:
    parser = build_parser()
    _require_certified_discipline(parser.parse_args(["--final"]), parser)


def test_zero_event_coverage_suppresses_the_verdict() -> None:
    # A zero-event percentile bootstrap returns the zero-width [0, 0], which
    # would render nominal as "OUTSIDE" no matter how well calibrated the curve
    # is. The verdict must be withheld, with a stated reason, not printed.
    assert _coverage_verdict_suppressed(0.0, {"low": 0.0, "high": 0.0}) is not None
    assert "Clopper-Pearson" in _coverage_verdict_suppressed(
        0.0, {"low": 0.0, "high": 0.0}
    )
    # Zero-width with events present is still not an interval.
    assert _coverage_verdict_suppressed(5.0, {"low": 0.2, "high": 0.2}) is not None
    # Undefined (no scored calls at all).
    assert _coverage_verdict_suppressed(
        0.0, {"low": math.nan, "high": math.nan}
    ) is not None
    # A genuine interval reports normally.
    assert _coverage_verdict_suppressed(5.0, {"low": 0.05, "high": 0.2}) is None


def test_zero_event_tail_reports_no_verdict_end_to_end() -> None:
    # Tight, short latencies: nothing exceeds the predicted P99, so that cell has
    # zero events and must come back with a suppressed verdict rather than a
    # spurious "outside".
    samples_by_task = {
        f"t{i:02d}": [
            _sample(f"t{i:02d}", f"t{i:02d}:{j}", 100.0 + j) for j in range(30)
        ]
        for i in range(10)
    }
    cfg = _cfg(fold_count=2)
    summary, _ = run_prior_calibration(
        samples_by_task, sorted(samples_by_task), cfg, censored_ids=set()
    )
    p99 = next(c for c in summary["coverage"] if c["label"] == "p99")
    assert p99["scored_call_count"] > 0  # the cell was scored, not skipped
    assert p99["exceedance_count"] == 0.0
    assert p99["nominal_inside_interval"] is None
    assert p99["verdict_suppressed_reason"] is not None


def test_coverage_verdict_is_reported_when_the_interval_is_genuine() -> None:
    samples_by_task = {
        f"t{i:02d}": [
            _sample(f"t{i:02d}", f"t{i:02d}:{j}", 100.0 * (j + 1) + i) for j in range(8)
        ]
        for i in range(12)
    }
    cfg = _cfg(fold_count=2)
    summary, _ = run_prior_calibration(
        samples_by_task, sorted(samples_by_task), cfg, censored_ids=set()
    )
    p50 = next(c for c in summary["coverage"] if c["label"] == "p50")
    assert p50["exceedance_count"] > 0
    assert isinstance(p50["nominal_inside_interval"], bool)
    assert p50["verdict_suppressed_reason"] is None


def test_coverage_excludes_quantiles_the_node_cannot_express() -> None:
    # A 2-sample node cannot resolve P99 (needs n >= 100): that quantile must be
    # excluded FOR THAT QUANTILE ONLY and counted, never silently scored.
    samples_by_task = {
        "t0": [_sample("t0", "t0:0", 100.0)],
        "t1": [_sample("t1", "t1:0", 200.0)],
        "t2": [_sample("t2", "t2:0", 300.0)],
    }
    cfg = _cfg(fold_count=3)
    summary, _ = run_prior_calibration(
        samples_by_task, ["t0", "t1", "t2"], cfg, censored_ids=set()
    )
    p99 = next(c for c in summary["coverage"] if c["label"] == "p99")
    assert p99["scored_call_count"] == 0
    assert p99["unresolvable_node_call_count"] == 3
    p50 = next(c for c in summary["coverage"] if c["label"] == "p50")
    assert p50["scored_call_count"] == 3


# --------------------------------------------------------------------------- #
# Censoring.
# --------------------------------------------------------------------------- #
def test_censored_row_raises_the_upper_tail_and_leaves_crps() -> None:
    # The SAME corpus scored twice: once with the long call treated as a
    # completion, once as a right-censored protocol-guard timeout. Censoring must
    # (a) keep it as an exceedance in the tail, (b) drop it from CRPS with its
    # mass reported, (c) drop it from PIT.
    # 12 tasks x 4 calls: fit folds hold 24 samples, enough for P90/P95 to be
    # expressible so the tail assertions below are not vacuous.
    samples_by_task = {
        f"t{i:02d}": [
            _sample(f"t{i:02d}", f"t{i:02d}:{j}", 100.0 + 10.0 * i + j)
            for j in range(4)
        ]
        for i in range(12)
    }
    samples_by_task["t11"][3] = _sample("t11", "t11:3", 300_000.0)
    cfg = _cfg(fold_count=2)
    task_ids = sorted(samples_by_task)

    uncensored, _ = run_prior_calibration(
        samples_by_task, task_ids, cfg, censored_ids=set()
    )
    censored, _ = run_prior_calibration(
        samples_by_task, task_ids, cfg, censored_ids={"t11:3"}
    )

    # (a) Still counted as an exceedance in the tail -- the bound (300 s) is far
    # above any predicted quantile, so it is a DEFINITE exceedance.
    for label in ("p50", "p90", "p95"):
        before = next(c for c in uncensored["coverage"] if c["label"] == label)
        after = next(c for c in censored["coverage"] if c["label"] == label)
        assert after["empirical_exceedance"] == pytest.approx(
            before["empirical_exceedance"]
        )
        assert after["indeterminate_censored_count"] == 0
        assert after["empirical_exceedance"] > 0.0

    # (b) Excluded from CRPS, count and mass reported.
    assert censored["crps"]["scored_call_count"] == (
        uncensored["crps"]["scored_call_count"] - 1
    )
    assert censored["crps"]["censored_excluded_count"] == 1
    assert censored["crps"]["censored_excluded_mass_ms"] == pytest.approx(300_000.0)

    # (c) Excluded from PIT (u is only identified as lying in [F(c), 1]).
    assert censored["marginal_pit"]["scored_call_count"] == (
        uncensored["marginal_pit"]["scored_call_count"] - 1
    )
    assert censored["excluded"]["pit_censored_excluded"] == 1


def test_censoring_below_a_predicted_quantile_is_indeterminate() -> None:
    # A bound BELOW the predicted quantile cannot decide the exceedance: it must
    # be counted indeterminate (and conservatively scored as non-exceedance),
    # not silently asserted either way.
    samples_by_task = {
        "t0": [_sample("t0", "t0:0", 10.0)],
        "t1": [_sample("t1", "t1:0", 20.0)],
        "t2": [_sample("t2", "t2:0", 30.0)],
        "t3": [_sample("t3", "t3:0", 5_000.0)],
    }
    cfg = _cfg(fold_count=4)
    # t0 is censored at 10 ms while its fit-fold P50 is well above that.
    summary, decisions = run_prior_calibration(
        samples_by_task, ["t0", "t1", "t2", "t3"], cfg, censored_ids={"t0:0"}
    )
    p50 = next(c for c in summary["coverage"] if c["label"] == "p50")
    assert p50["indeterminate_censored_count"] == 1
    row = next(r for r in decisions if r["sample_id"] == "t0:0")
    assert row["coverage"]["p50"]["exceeds"] is False


def test_censored_sample_ids_matches_only_protocol_guard_timeouts(tmp_path) -> None:
    # The indicator is conjunctive: failure AND the timeout prefix AND exit 124.
    # An ordinary tool error, and a SUCCESSFUL call whose output merely mentions
    # a timeout, must both stay uncensored.
    trace = tmp_path / "trace.jsonl"
    records = [
        {
            "type": "trace_metadata",
            "instance_id": "task-a",
            "trace_format_version": CURRENT_TRACE_FORMAT_VERSION,
        },
        _tool_exec("a1", success=False, result="Error: [timeout]\n\nExit code: 124\n"),
        _tool_exec("a2", success=False, result="Error: old_text not found in /x"),
        _tool_exec("a3", success=True, result="see docs about timeout Exit code: 124"),
        _tool_exec("a4", success=True, result="ok"),
    ]
    trace.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    found = censored_sample_ids([trace])
    assert found == {f"{trace}:agent:0:a1"}


# --------------------------------------------------------------------------- #
# Cross-fit isolation and determinism.
# --------------------------------------------------------------------------- #
def test_cross_fit_never_scores_a_call_against_its_own_task() -> None:
    # fold_count=2 with per-task-unique latencies: the node an eval call is
    # scored against must never contain that call's own latency, otherwise the
    # PIT/CRPS would be contaminated by the held-out label.
    samples_by_task = {
        "t0": [_sample("t0", "t0:0", 1000.0)],
        "t1": [_sample("t1", "t1:0", 2000.0)],
        "t2": [_sample("t2", "t2:0", 3000.0)],
        "t3": [_sample("t3", "t3:0", 4000.0)],
    }
    cfg = _cfg(fold_count=2)
    decisions, _ = score_decisions(
        samples_by_task, ["t0", "t1", "t2", "t3"], cfg, censored_ids=set()
    )
    assert len(decisions) == 4
    # Folds alternate by index: {t0, t2} eval in f1, {t1, t3} in f2.
    assert {row["sample_id"]: row["outer_fold"] for row in decisions} == {
        "t0:0": "f1",
        "t2:0": "f1",
        "t1:0": "f2",
        "t3:0": "f2",
    }
    # A held-out call at the top of its fit fold gets PIT 1.0 -- only possible
    # because its own (larger) latency is absent from the node.
    by_id = {row["sample_id"]: row for row in decisions}
    assert by_id["t3:0"]["pit"] == pytest.approx(1.0)
    assert by_id["t0:0"]["pit"] == pytest.approx(0.0)
    assert by_id["t0:0"]["node_count"] == 2


def test_scoring_is_deterministic() -> None:
    samples_by_task = {
        f"t{i}": [_sample(f"t{i}", f"t{i}:0", 100.0 * (i + 1))] for i in range(6)
    }
    cfg = _cfg(fold_count=3)
    task_ids = sorted(samples_by_task)
    first, first_rows = run_prior_calibration(
        samples_by_task, task_ids, cfg, censored_ids=set()
    )
    second, second_rows = run_prior_calibration(
        samples_by_task, task_ids, cfg, censored_ids=set()
    )
    assert json.dumps(first, sort_keys=True, default=list) == json.dumps(
        second, sort_keys=True, default=list
    )
    assert first_rows == second_rows


def test_summary_reports_every_exclusion_bucket() -> None:
    # Degenerate nodes and censoring are never silently dropped: the excluded
    # block must always be present with the full set of counters.
    samples_by_task = {
        f"t{i}": [_sample(f"t{i}", f"t{i}:0", 100.0 * (i + 1))] for i in range(4)
    }
    cfg = _cfg(fold_count=2)
    summary, _ = run_prior_calibration(
        samples_by_task, sorted(samples_by_task), cfg, censored_ids=set()
    )
    excluded = summary["excluded"]
    for key in (
        "pit_empty_node",
        "pit_singleton_node",
        "pit_censored_excluded",
        "crps_censored_excluded",
        "crps_censored_excluded_mass_ms",
        "rolling_no_support_beyond_t",
        "coverage_unresolvable_by_quantile",
        "brier_empty_region",
    ):
        assert key in excluded


def test_rolling_pit_grid_is_node_derived_not_a_time_constant() -> None:
    # Two corpora identical up to a 1000x time rescaling must produce IDENTICAL
    # rolling PIT values: the t-grid comes from each node's own support, so no
    # absolute millisecond constant can leak in.
    # Nodes must be wide enough that every grid fraction leaves >= 2 samples
    # alive, or the assertions below would be vacuous (see scored_call_count).
    def corpus(scale: float):
        return {
            f"t{i:02d}": [
                _sample(f"t{i:02d}", f"t{i:02d}:{j}", scale * (10.0 * (i + 1) + j))
                for j in range(6)
            ]
            for i in range(10)
        }

    cfg = _cfg(fold_count=2)
    task_ids = [f"t{i:02d}" for i in range(10)]
    small, _ = run_prior_calibration(corpus(1.0), task_ids, cfg, censored_ids=set())
    large, _ = run_prior_calibration(corpus(1000.0), task_ids, cfg, censored_ids=set())
    for a, b in zip(small["rolling_pit"], large["rolling_pit"], strict=True):
        # Without this the nan_ok comparisons below pass vacuously wherever a
        # grid fraction happens to score nothing.
        assert a["scored_call_count"] > 0
        assert a["scored_call_count"] == b["scored_call_count"]
        assert a["mean_pit"] == pytest.approx(b["mean_pit"], nan_ok=True)
        assert a["cvm"]["point"] == pytest.approx(b["cvm"]["point"], nan_ok=True)


def test_rolling_pit_is_the_renormalized_conditional_curve() -> None:
    # The spec calls the rolling curve "the object actually consumed": at elapsed
    # t the forecast must be F_t(r) = (F(t+r) - F(t)) / (1 - F(t)) evaluated at
    # r = y - t. Recompute it here straight from that definition, against the
    # node the call was actually scored on, and require an exact match.
    samples_by_task = {
        f"t{i:02d}": [
            _sample(f"t{i:02d}", f"t{i:02d}:{j}", 10.0 * (i + 1) + 3.0 * j)
            for j in range(4)
        ]
        for i in range(10)
    }
    cfg = _cfg(fold_count=2)
    decisions, _ = score_decisions(
        samples_by_task, sorted(samples_by_task), cfg, censored_ids=set()
    )
    node_values = _fit_fold_node_values(samples_by_task, cfg)
    checked = 0
    for row in decisions:
        values = sorted(node_values[row["outer_fold"]])
        y = float(row["latency_ms"])
        for entry in row["rolling"]:
            t = float(entry["t_ms"])
            f_t = 1.0 - ecdf_survival(values, t)
            # Mid-rank at y, expressed through the marginal ECDF, then
            # renormalized onto the survivors of t -- the definition, verbatim.
            f_y = ecdf_pit(values, y)
            expected = (f_y - f_t) / (1.0 - f_t)
            assert entry["u"] == pytest.approx(expected, abs=1e-12)
            checked += 1
    assert checked > 0


def test_decision_band_brier_flags_a_degenerate_region() -> None:
    # When the certified trigger collapses onto the threshold the region outcome
    # is constant (o == 1) and the Brier is vacuously ~0; the run must SAY so via
    # the degenerate counter rather than present it as skill.
    # A wide latency spread so some calls are genuinely alive at the trigger.
    samples_by_task = {
        f"t{i:02d}": [
            _sample(f"t{i:02d}", f"t{i:02d}:{j}", 50.0 * (i + 1) * (j + 1))
            for j in range(4)
        ]
        for i in range(12)
    }
    cfg = _cfg(fold_count=2, costs_ms=(100.0,))
    summary, _ = run_prior_calibration(
        samples_by_task, sorted(samples_by_task), cfg, censored_ids=set()
    )
    cell = summary["decision_band_brier"][0]
    assert cell["kv_cost_ms"] == 100.0
    # The region must be non-empty, or the assertions below are vacuous.
    assert cell["region_call_count"] > 0
    # Some calls sit in a collapsed region (trigger == threshold) and are
    # counted as such rather than being folded silently into the score.
    assert 0 < cell["degenerate_region_call_count"] <= cell["region_call_count"]
    # Brier is a mean squared error on [0, 1] and the climatology reference is
    # reported alongside, so a low score cannot be read as skill on its own.
    assert 0.0 <= cell["brier"] <= 1.0
    assert cell["reference_brier"] == pytest.approx(
        cell["base_rate"] * (1.0 - cell["base_rate"])
    )
    assert cell["interval"]["low"] <= cell["brier"] <= cell["interval"]["high"]


# --------------------------------------------------------------------------- #
# Fixtures.
# --------------------------------------------------------------------------- #
def _sample(task_id: str, sample_id: str, latency_ms: float) -> ToolLatencySample:
    return ToolLatencySample(
        sample_id=sample_id,
        source_trace=f"/traces/{task_id}.jsonl",
        task_id=task_id,
        agent_id="agent",
        instance_id=task_id,
        iteration=0,
        action_id=sample_id,
        tool_name="bash",
        tool_call_id=sample_id,
        tool_ts_start=0.0,
        tool_ts_end=latency_ms,
        latency_ms=latency_ms,
        success=True,
        reported_duration_ms=None,
        tool_args={"command": "run"},
    )


def _fit_fold_node_values(
    samples_by_task: dict, cfg: CalibrationConfig
) -> dict[str, list[float]]:
    """Per-fold profile-split latencies, i.e. the node every eval call is scored
    against when the corpus uses a single tool and a single command."""

    declared = sorted(samples_by_task)
    values: dict[str, list[float]] = {}
    for fold in range(1, cfg.fold_count + 1):
        profile = [
            sample.latency_ms
            for index, task_id in enumerate(declared)
            if index % cfg.fold_count != fold - 1
            for sample in samples_by_task[task_id]
        ]
        values[f"f{fold}"] = sorted(profile)
    return values


def _tool_exec(action_id: str, *, success: bool, result: str) -> dict:
    return {
        "type": "action",
        "action_type": "tool_exec",
        "action_id": action_id,
        "agent_id": "agent",
        "instance_id": "task-a",
        "iteration": 0,
        "ts_start": 0.0,
        "ts_end": 1.0,
        "data": {
            "tool_name": "exec",
            "tool_call_id": action_id,
            "tool_args": {"command": "run"},
            "tool_result": result,
            "duration_ms": 1000.0,
            "success": success,
        },
    }


def _cfg(
    *,
    fold_count: int = 2,
    costs_ms: tuple[float, ...] = (1000.0, 3500.0),
    replicates: int = 500,
    cvm_replicates: int = 64,
) -> CalibrationConfig:
    return CalibrationConfig(
        fold_count=fold_count,
        command_field="command",
        max_prefix_depth=4,
        skip_leading_cd=False,
        min_tool_history=1,
        min_profile_tasks=1,
        costs_ms=costs_ms,
        guard_ms=0.0,
        restore_cost_fraction=0.94,
        replicates=replicates,
        confidence_level=0.95,
        seed=0,
        rolling_fractions=(0.25, 0.5, 0.75),
        cvm_replicates=cvm_replicates,
    )
