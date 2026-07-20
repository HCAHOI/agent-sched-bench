"""Anytime-validity, revocation, and agreement arithmetic for the online gate."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from trace_collect.tool_latency_confirmation import paired_task_cluster_bootstrap
from trace_collect.tool_latency_online_gate import (
    BET_RULE,
    CERTIFIED,
    CONTINUE,
    HARMFUL,
    replay_online_gate,
    symmetry_log_martingales,
)

_OFFLINE_LABELS = {"positive", "harmful", "inconclusive"}


def _heavy_tailed_null(rng: np.random.Generator, size: int) -> np.ndarray:
    """Mean-zero, heavy-tailed per-task deltas: lognormal magnitude x random sign.

    Sign symmetry is exactly the null the offline permutation gate assumes, and
    the lognormal magnitude reproduces the skew of real per-task summed paired
    regret. A light-tailed Gaussian null would flatter the construction; this is
    the shape that already broke the percentile bootstrap on this project.
    """

    magnitudes = rng.lognormal(mean=0.0, sigma=1.5, size=size)
    signs = rng.choice(np.array([-1.0, 1.0]), size=size)
    return magnitudes * signs


# --------------------------------------------------------------------------- #
# The load-bearing property: no peeking penalty.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("task_count", [50, 277])
@pytest.mark.parametrize("one_sided_tail", [0.025, 0.0025])
def test_null_false_certification_rate_at_or_below_alpha(
    task_count: int, one_sided_tail: float
) -> None:
    """Under a mean-zero null the gate must rarely EVER leave CONTINUE.

    The verdict is read after every task, so a construction with a peeking
    penalty blows through the level here. 0.0025 is the tail the deliverable
    actually runs at (0.05 / (2 x 10 kv cells)); 50 and 277 tasks bracket the
    corpora. Ville's inequality makes this exact, so the measured rate must sit
    at or below the two-sided nominal with no asymptotic excuse.
    """

    # 2000 runs cannot resolve a nominal of 0.005 -- the 3-sigma Monte-Carlo
    # slack is as large as the quantity under test, and a validity-voiding
    # off-by-one measured 0.0090 would slip through. The deliverable runs at
    # 0.0025, so that case gets the run count it needs.
    runs = 50000 if one_sided_tail <= 0.0025 else 5000
    rng = np.random.Generator(np.random.PCG64(20260720))
    certifications = 0
    harmful = 0
    for _ in range(runs):
        replay = replay_online_gate(
            _heavy_tailed_null(rng, task_count), one_sided_tail=one_sided_tail
        )
        certifications += int(replay["ever_certified"])
        harmful += int(replay["ever_harmful"])
    nominal = 2.0 * one_sided_tail
    slack = 3.0 * (nominal * (1.0 - nominal) / runs) ** 0.5
    exits = certifications + harmful
    assert exits / runs <= nominal + slack, (
        f"null exit rate {exits / runs:.5f} exceeds nominal {nominal} "
        f"(+{slack:.5f} MC slack) at {task_count} tasks; "
        f"certified={certifications}, harmful={harmful}"
    )


def test_null_martingale_has_unit_mean() -> None:
    """The direct martingale property the validity argument rests on."""

    rng = np.random.Generator(np.random.PCG64(99))
    finals = [
        math.exp(symmetry_log_martingales(_heavy_tailed_null(rng, 60))[0][-1])
        for _ in range(20000)
    ]
    # E[M_t] == 1 exactly; heavy tails make the sample mean noisy, so this is a
    # loose sanity band, not a precision check.
    assert 0.7 < float(np.mean(finals)) < 1.4


def test_certifies_a_clearly_positive_effect() -> None:
    """Power sanity: a real effect must be detected, not merely be safe."""

    rng = np.random.Generator(np.random.PCG64(7))
    deltas = _heavy_tailed_null(rng, 277) + 3.0
    replay = replay_online_gate(deltas, one_sided_tail=0.0025)
    assert replay["instantaneous_final_label"] == CERTIFIED
    assert replay["final_lifecycle_label"] == CERTIFIED
    assert replay["first_certified_at_task"] is not None
    assert replay["bet_rule"] == BET_RULE


def test_declares_a_clearly_negative_effect_harmful() -> None:
    rng = np.random.Generator(np.random.PCG64(8))
    replay = replay_online_gate(
        _heavy_tailed_null(rng, 277) - 3.0, one_sided_tail=0.0025
    )
    assert replay["instantaneous_final_label"] == HARMFUL
    assert replay["final_lifecycle_label"] == HARMFUL
    assert replay["first_harmful_at_task"] is not None
    assert not replay["ever_certified"]


# --------------------------------------------------------------------------- #
# Bets must be predictable, or the guarantee is void.
# --------------------------------------------------------------------------- #
def test_bets_are_predictable() -> None:
    """Bet i must not see delta i. If it ever does, the guarantee is void.

    Asserted on the BETS directly, at every index. Asserting on the cumulative
    log-martingale instead cannot detect this: increments 0..i-1 involve only
    deltas 0..i-1 whether or not bet i peeks at its own observation, so a
    prefix-equality check passes on the broken implementation. That off-by-one
    is genuinely anticonservative (measured null exit 0.0105 against a nominal
    0.005 at 277 tasks), so it has to be caught here.
    """

    from trace_collect.tool_latency_online_gate import _kelly_bets

    rng = np.random.Generator(np.random.PCG64(21))
    deltas = _heavy_tailed_null(rng, 40)
    base = _kelly_bets(deltas)
    assert base[0] == 0.0, "the first task has no past, so it must not bet"
    for index in range(deltas.size):
        perturbed = deltas.copy()
        perturbed[index] += 1000.0
        other = _kelly_bets(perturbed)
        assert np.array_equal(base[: index + 1], other[: index + 1]), (
            f"bet {index} changed when delta {index} changed -- the bet is "
            "looking at its own observation"
        )
    # And the perturbation must actually reach the later bets, or the loop above
    # would pass vacuously on an implementation that never bets at all.
    tail = _kelly_bets(np.concatenate([deltas[:20], deltas[20:] + 1000.0]))
    assert not np.allclose(base[21:], tail[21:])


def test_first_task_places_no_bet() -> None:
    """With no past there is nothing to bet on, so M_1 must be exactly 1."""

    log_plus, log_minus = symmetry_log_martingales(np.array([37.0, -2.0]))
    assert log_plus[0] == pytest.approx(0.0)
    assert log_minus[0] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Interval, determinism, validation.
# --------------------------------------------------------------------------- #


def test_replay_is_deterministic() -> None:
    rng = np.random.Generator(np.random.PCG64(11))
    deltas = _heavy_tailed_null(rng, 120) + 1.0
    first = replay_online_gate(deltas, one_sided_tail=0.0025)
    second = replay_online_gate(deltas, one_sided_tail=0.0025)
    assert first == second


@pytest.mark.parametrize("tail", [0.0, 1.0, -0.1, float("nan")])
def test_replay_rejects_an_invalid_tail(tail: float) -> None:
    with pytest.raises(ValueError, match="one_sided_tail"):
        replay_online_gate([1.0, 2.0], one_sided_tail=tail)


@pytest.mark.parametrize("deltas", [[], [1.0, float("nan")], [1.0, float("inf")]])
def test_martingales_reject_invalid_deltas(deltas: list[float]) -> None:
    with pytest.raises(ValueError):
        symmetry_log_martingales(np.asarray(deltas, dtype=float))


def test_log_cosh_survives_a_huge_delta() -> None:
    """Naive cosh overflows past ~710; the gate must not return NaN there."""

    log_plus, log_minus = symmetry_log_martingales(np.array([1.0, 1.0, 1e12, 1.0]))
    assert np.all(np.isfinite(log_plus))
    assert np.all(np.isfinite(log_minus))


# --------------------------------------------------------------------------- #
# Revocation is a first-class output.
# --------------------------------------------------------------------------- #
def test_certification_is_revoked_when_evidence_truly_reverses() -> None:
    """A sustained reversal must produce the CALIBRATED revocation.

    A reversing run passes through CONTINUE for many tasks before harmful
    evidence accumulates, so revocation cannot be read off the lapse step
    alone -- that bug reported zero revocations on exactly this sequence.
    """

    rng = np.random.Generator(np.random.PCG64(5))
    deltas = np.concatenate(
        [_heavy_tailed_null(rng, 100) + 6.0, _heavy_tailed_null(rng, 600) - 6.0]
    )
    replay = replay_online_gate(deltas, one_sided_tail=0.0025)
    assert replay["ever_certified"]
    assert replay["revoked"]
    assert replay["instantaneous_final_label"] == HARMFUL
    assert replay["final_lifecycle_label"] == HARMFUL
    assert replay["revoked_at_task"] > replay["first_certified_at_task"]
    assert replay["revocation_lag_tasks"] == (
        replay["revoked_at_task"] - replay["first_certified_at_task"]
    )
    assert replay["labels"][replay["revoked_at_task"] - 1] == HARMFUL


def test_lapse_bookkeeping_matches_the_label_stream() -> None:
    rng = np.random.Generator(np.random.PCG64(6))
    deltas = np.concatenate(
        [_heavy_tailed_null(rng, 150) + 6.0, _heavy_tailed_null(rng, 300) - 6.0]
    )
    replay = replay_online_gate(deltas, one_sided_tail=0.0025)
    labels = replay["labels"]
    assert replay["lapses"], "test needs at least one lapse to check"
    for lapse in replay["lapses"]:
        certified_at = lapse["certified_at_task"]
        lapsed_at = lapse["lapsed_at_task"]
        assert labels[lapsed_at - 1] == lapse["lapsed_to"] != CERTIFIED
        # The episode is contiguous: CERTIFIED throughout, until the lapse.
        assert set(labels[certified_at - 1 : lapsed_at - 1]) == {CERTIFIED}
        assert lapse["lag_tasks"] == lapsed_at - certified_at
    assert replay["first_certified_at_task"] == labels.index(CERTIFIED) + 1


def test_a_run_that_never_turns_harmful_is_not_revoked() -> None:
    """Lapsing is not revoking -- the whole point of separating the two."""

    rng = np.random.Generator(np.random.PCG64(41))
    deltas = np.concatenate(
        [_heavy_tailed_null(rng, 120) + 6.0, _heavy_tailed_null(rng, 200)]
    )
    replay = replay_online_gate(deltas, one_sided_tail=0.0025)
    assert replay["lapse_count"] > 0
    assert replay["instantaneous_final_label"] == CONTINUE
    assert replay["final_lifecycle_label"] == CERTIFIED
    assert replay["ever_certified"]
    assert not replay["revoked"]
    assert replay["revocation_lag_tasks"] is None


def test_lapse_is_common_under_a_sustained_true_effect() -> None:
    """Pins the number that stops a lapse being read as a finding.

    Ville bounds the probability of ever CROSSING, not of crossing back, so a
    lapse has no type-I control. Under a genuine sustained positive effect a
    large minority of runs certify and then lapse; if that rate ever drifts,
    the docstring in ``Lapse`` and the artifact's footnote go stale with it.
    """

    runs = 500
    rng = np.random.Generator(np.random.PCG64(1234))
    certified_runs = 0
    lapsed_runs = 0
    revoked_runs = 0
    for _ in range(runs):
        deltas = _heavy_tailed_null(rng, 277) + 1.2
        replay = replay_online_gate(deltas, one_sided_tail=0.0025)
        if not replay["ever_certified"]:
            continue
        certified_runs += 1
        lapsed_runs += int(replay["lapse_count"] > 0)
        revoked_runs += int(replay["revoked"])
    lapse_rate = lapsed_runs / certified_runs
    # Measured ~0.72 at this tail: at 0.0025 certification happens late and
    # marginally, so falling back below threshold is the NORM, not the
    # exception. That is precisely why a lapse must never be reported as
    # evidence against a policy.
    assert 0.55 < lapse_rate < 0.90, (
        f"lapse rate under a true effect is {lapse_rate:.3f}; the figure "
        "quoted in Lapse's docstring and the artifact footnote needs updating"
    )
    # The calibrated event, by contrast, must essentially never fire when the
    # truth is positive -- that is what makes it reportable.
    assert revoked_runs / certified_runs < 0.01


def test_revocation_implies_a_prior_certification_and_a_lapse() -> None:
    rng = np.random.Generator(np.random.PCG64(77))
    deltas = np.concatenate(
        [_heavy_tailed_null(rng, 100) + 6.0, _heavy_tailed_null(rng, 600) - 6.0]
    )
    replay = replay_online_gate(deltas, one_sided_tail=0.0025)
    assert replay["revoked"]
    assert replay["ever_certified"]
    assert replay["lapse_count"] >= 1


def test_no_revocation_recorded_when_never_certified() -> None:
    replay = replay_online_gate(np.full(50, -1.0), one_sided_tail=0.0025)
    assert not replay["ever_certified"]
    assert not replay["revoked"]
    assert replay["lapse_count"] == 0
    assert replay["revocation_lag_tasks"] is None
    assert replay["max_lapse_lag_tasks"] is None


def test_the_two_sides_can_never_fire_together() -> None:
    """``_label``'s priority order is only safe because this holds.

    ``log M+ + log M- = -2 * sum(log cosh(...)) <= 0``, so at most one side can
    exceed a threshold above 1. If that ever broke, CERTIFIED would silently
    mask a simultaneous HARMFUL.
    """

    rng = np.random.Generator(np.random.PCG64(64))
    for shift in (-5.0, 0.0, 5.0):
        deltas = _heavy_tailed_null(rng, 200) + shift
        log_plus, log_minus = symmetry_log_martingales(deltas)
        assert np.all(log_plus + log_minus <= 1e-12)


def test_bet_truncation_does_not_affect_calibration() -> None:
    """Validity holds for ANY predictable bet -- so truncation is not a knob.

    Pins the claim that ``_BET_TRUNCATION`` trades tightness only. If a future
    change made calibration depend on it, it would become a tunable parameter
    and a research-integrity problem.
    """

    import trace_collect.tool_latency_online_gate as gate

    original = gate._BET_TRUNCATION
    try:
        for truncation in (0.5, 1.0, 4.0):
            gate._BET_TRUNCATION = truncation
            rng = np.random.Generator(np.random.PCG64(808))
            exits = sum(
                replay_online_gate(_heavy_tailed_null(rng, 100), one_sided_tail=0.025)[
                    "ever_certified"
                ]
                for _ in range(4000)
            )
            assert exits / 4000 <= 0.025 + 0.008, (
                f"truncation {truncation} broke calibration: {exits / 4000:.4f}"
            )
    finally:
        gate._BET_TRUNCATION = original


def test_labels_are_one_per_task_and_from_the_declared_set() -> None:
    replay = replay_online_gate(np.arange(1.0, 31.0), one_sided_tail=0.0025)
    assert len(replay["labels"]) == 30
    assert set(replay["labels"]) <= {CERTIFIED, HARMFUL, CONTINUE}
    assert len(replay["log_e_positive"]) == len(replay["log_e_harmful"]) == 30


# --------------------------------------------------------------------------- #
# Agreement arithmetic (the deliverable's headline column).
# --------------------------------------------------------------------------- #
def test_agreement_mapping_covers_every_offline_label() -> None:
    from scripts.replay_online_gate import _OFFLINE_TO_ONLINE

    assert set(_OFFLINE_TO_ONLINE) == {"positive", "harmful", "inconclusive"}
    assert set(_OFFLINE_TO_ONLINE.values()) == {CERTIFIED, HARMFUL, CONTINUE}


def test_ordered_task_deltas_follow_the_declared_order() -> None:
    from scripts.replay_online_gate import _ordered_task_deltas

    offline = {
        "task_contributions": [
            {"task_id": "b", "paired_delta_ms_by_cost": {"3500.0": 2.0}},
            {"task_id": "a", "paired_delta_ms_by_cost": {"3500.0": 1.0}},
        ]
    }
    assert list(_ordered_task_deltas(offline, ["a", "b"], 3500.0)) == [1.0, 2.0]
    assert list(_ordered_task_deltas(offline, ["b", "a"], 3500.0)) == [2.0, 1.0]


def test_ordered_task_deltas_reject_a_missing_task() -> None:
    from scripts.replay_online_gate import _ordered_task_deltas

    offline = {
        "task_contributions": [
            {"task_id": "a", "paired_delta_ms_by_cost": {"3500.0": 1.0}},
        ]
    }
    with pytest.raises(ValueError, match="omit tasks"):
        _ordered_task_deltas(offline, ["a", "missing"], 3500.0)


def test_outer_fold_parallelism_is_byte_order_equivalent() -> None:
    from scripts.replay_online_gate import ReplayConfig, cross_fitted_decisions
    from trace_collect.tool_latency_dataset import ToolLatencySample

    samples = {
        f"task-{index}": [
            ToolLatencySample(
                sample_id=f"task-{index}:0",
                source_trace=f"/tmp/task-{index}/trace.jsonl",
                task_id=f"task-{index}",
                agent_id="agent",
                instance_id=f"task-{index}",
                iteration=0,
                action_id=f"action-{index}",
                tool_name="exec",
                tool_call_id=f"call-{index}",
                tool_ts_start=0.0,
                tool_ts_end=latency_ms / 1000.0,
                latency_ms=latency_ms,
                success=True,
                reported_duration_ms=None,
                tool_args={"command": "echo ok"},
            )
        ]
        for index, latency_ms in enumerate([100.0, 300.0, 700.0, 1200.0, 2500.0])
    }
    task_ids = sorted(samples)
    cfg = ReplayConfig(
        fold_count=5,
        inner_folds=4,
        command_field="command",
        max_prefix_depth=4,
        skip_leading_cd=False,
        min_tool_history=1,
        min_profile_tasks=1,
        costs_ms=(500.0,),
        guard_ms=0.0,
        restore_cost_fraction=0.94,
        replicates=10,
        confidence_level=0.95,
        seed=0,
        order_seeds=(0,),
    )
    serial = cross_fitted_decisions(samples, task_ids, cfg, workers=1)
    parallel = cross_fitted_decisions(samples, task_ids, cfg, workers=8)
    assert parallel == serial
    with pytest.raises(ValueError, match="workers must be positive"):
        cross_fitted_decisions(samples, task_ids, cfg, workers=0)


def test_order_sensitivity_reports_a_lag_spread_not_a_verdict_flip() -> None:
    """Order may move the LAG; on a clear effect it must not move the verdict."""

    from scripts.replay_online_gate import _order_sensitivity

    rng = np.random.Generator(np.random.PCG64(3))
    deltas = _heavy_tailed_null(rng, 277) + 4.0
    sweep = _order_sensitivity(deltas, one_sided_tail=0.0025, seeds=range(20))
    assert sweep["final_lifecycle_label_counts"] == {CERTIFIED: 20}
    assert sweep["never_certified_seed_count"] == 0
    spread = sweep["first_certified_at_task"]
    assert spread["min"] <= spread["median"] <= spread["max"]


def test_order_sensitivity_reports_harmful_detection_lag() -> None:
    from scripts.replay_online_gate import _order_sensitivity

    rng = np.random.Generator(np.random.PCG64(9))
    deltas = _heavy_tailed_null(rng, 277) - 4.0
    sweep = _order_sensitivity(deltas, one_sided_tail=0.0025, seeds=range(20))
    assert sweep["final_lifecycle_label_counts"] == {HARMFUL: 20}
    assert sweep["never_harmful_seed_count"] == 0
    spread = sweep["first_harmful_at_task"]
    assert spread["min"] <= spread["median"] <= spread["max"]


def test_order_sensitivity_exposes_partial_crossings_and_verdict_flips() -> None:
    from scripts.replay_online_gate import (
        _format_crossing_sweep,
        _format_lifecycle_counts,
        _order_sensitivity,
    )

    rng = np.random.Generator(np.random.PCG64(0))
    sweep = _order_sensitivity(
        _heavy_tailed_null(rng, 277) + 1.0,
        one_sided_tail=0.0025,
        seeds=range(50),
    )
    detected = 50 - sweep["never_certified_seed_count"]
    assert 0 < detected < 50
    assert _format_crossing_sweep(
        sweep,
        summary_key="first_certified_at_task",
        never_key="never_certified_seed_count",
    ).startswith(f"{detected}/50; ")
    counts = sweep["final_lifecycle_label_counts"]
    assert _format_lifecycle_counts(sweep) == (
        f"{counts.get(CERTIFIED, 0)}/{counts.get(HARMFUL, 0)}/{counts.get(CONTINUE, 0)}"
    )


def _decision_rows(
    task_deltas_ms: dict[str, float], costs_ms: tuple[float, ...]
) -> list[dict[str, Any]]:
    """Minimal gated decision rows realising a target per-task paired delta.

    One call per task. The baseline swaps at ``trigger``; the treatment waits to
    the deadline (the gated fallback), so the paired delta is whatever the cost
    functional makes of that pair -- the point is to exercise the real
    ``paired_task_cluster_bootstrap`` path, not to hand-compute utilities.
    """

    rows: list[dict[str, Any]] = []
    for task_id, latency_ms in task_deltas_ms.items():
        for cost in costs_ms:
            rows.append(
                {
                    # One call, scored at every cost: the sample_id is shared
                    # across the cost panel, as the confirmation engine requires.
                    "sample_id": f"{task_id}-call0",
                    "task_id": task_id,
                    "outer_fold": "f1",
                    "kv_cost_ms": cost,
                    "threshold_ms": cost,
                    "latency_ms": latency_ms,
                    "robust_trigger_ms": cost / 2.0,
                    "offline_gated_robust_trigger_ms": cost,
                }
            )
    return rows


def test_compare_gates_runs_end_to_end_and_reports_both_verdicts() -> None:
    """Guards the driver's wiring: every key the table reads must exist.

    A full-corpus run takes ~15 minutes, so a key-name drift between the gate
    module and its driver must be caught here instead.
    """

    from scripts.replay_online_gate import ReplayConfig, compare_gates

    rng = np.random.Generator(np.random.PCG64(31))
    costs = (3500.0, 5000.0)
    latencies = {
        f"task-{index:03d}": float(value)
        for index, value in enumerate(rng.lognormal(8.0, 1.0, 40))
    }
    cfg = ReplayConfig(
        fold_count=5,
        inner_folds=4,
        command_field="command",
        max_prefix_depth=4,
        skip_leading_cd=False,
        min_tool_history=1,
        min_profile_tasks=1,
        costs_ms=costs,
        guard_ms=0.0,
        restore_cost_fraction=0.94,
        replicates=500,
        confidence_level=0.95,
        seed=0,
        order_seeds=(0, 1, 2),
    )
    task_ids = sorted(latencies)
    comparison = compare_gates(_decision_rows(latencies, costs), task_ids, cfg)

    # The load-bearing property of the whole replay: both gates must consume the
    # SAME per-task deltas, or "agreement" compares two different quantities.
    offline = paired_task_cluster_bootstrap(
        _decision_rows(latencies, costs),
        costs_ms=costs,
        replicates=cfg.replicates,
        confidence_level=cfg.confidence_level,
        seed=cfg.seed,
        baseline_trigger_field="robust_trigger_ms",
        treatment_trigger_field="offline_gated_robust_trigger_ms",
        restore_cost_fraction=cfg.restore_cost_fraction,
        permutation_draws=cfg.replicates,
    )
    for cell in comparison["cells"]:
        offline_deltas = [
            entry["paired_delta_ms_by_cost"][str(cell["kv_cost_ms"])]
            for entry in sorted(
                offline["task_contributions"], key=lambda e: e["task_id"]
            )
        ]
        assert cell["_replay_trace"]["task_delta_ms"] == offline_deltas
        assert cell["total_delta_ms"] == pytest.approx(sum(offline_deltas))

    assert comparison["one_sided_tail"] == pytest.approx(0.05 / (2 * len(costs)))
    assert comparison["task_count"] == len(task_ids)
    assert len(comparison["cells"]) == len(costs)
    for cell in comparison["cells"]:
        assert cell["offline_permutation_label"] in _OFFLINE_LABELS
        assert 0.0 <= cell["offline_permutation_p_positive"] <= 1.0
        assert 0.0 <= cell["offline_permutation_p_harmful"] <= 1.0
        assert cell["online_instantaneous_final_label"] in {
            CERTIFIED,
            HARMFUL,
            CONTINUE,
        }
        assert cell["online_lifecycle_label"] in {CERTIFIED, HARMFUL, CONTINUE}
        assert isinstance(cell["agree"], bool)
        assert set(cell["_replay_trace"]) == {
            "task_delta_ms",
            "labels",
            "lifecycle_labels",
            "log_e_positive",
            "log_e_harmful",
        }
        assert len(cell["_replay_trace"]["task_delta_ms"]) == len(task_ids)
        assert cell["order_sensitivity"]["seeds"] == [0, 1, 2]
    assert 0 <= comparison["agreement_count"] <= len(costs)
    # Agreement must be decomposed: "both certified" and "both said nothing"
    # are different evidence and must never collapse into one number.
    assert (
        comparison["concordant_positive"]
        + comparison["concordant_harmful"]
        + comparison["concordant_null"]
        == comparison["agreement_count"]
    )
    for cell in comparison["cells"]:
        assert "online_final_interval_ms" not in cell, (
            "the mean interval was deleted: inverting a sign-symmetry test "
            "targets the symmetry centre, not the mean"
        )
        assert (
            cell["effective_task_count"] + cell["zero_delta_task_count"]
            == (cell["task_count"])
        )
        assert cell["effective_task_count"] == sum(
            1 for value in cell["_replay_trace"]["task_delta_ms"] if value != 0.0
        )
        # The first nonzero delta gets a zero bet; every later one contributes
        # strictly less than log 2. This is a one-way upper-bound diagnostic,
        # never a claim that clearing it makes certification attainable.
        assert cell["online_log_e_upper_bound"] == pytest.approx(
            max(cell["effective_task_count"] - 1, 0) * math.log(2.0)
        )
        assert cell["online_log_e_positive"] <= cell["online_log_e_upper_bound"] + 1e-9
        assert cell["online_log_e_harmful"] <= cell["online_log_e_upper_bound"] + 1e-9
        ruled_out = cell["online_log_e_upper_bound"] <= cell["online_log_e_threshold"]
        assert cell["online_certification_ruled_out_by_upper_bound"] == ruled_out
        if ruled_out:
            assert not cell["online_ever_certified"]


def test_zero_delta_tasks_are_excluded_from_effective_n() -> None:
    """Effective n drives whether an e-value of 1/tail is reachable at all."""

    from scripts.replay_online_gate import compare_gates

    assert compare_gates is not None  # imported for symmetry with the pin above
    replay = replay_online_gate(
        np.array([0.0, 0.0, 5.0, 0.0, -5.0]), one_sided_tail=0.0025
    )
    # Zero deltas move neither martingale: the bet times zero is zero, and
    # log cosh(0) is zero, so they contribute exactly nothing to the evidence.
    assert replay["log_e_positive"][0] == pytest.approx(0.0)
    assert replay["log_e_positive"][1] == pytest.approx(0.0)


def test_replay_no_longer_exposes_an_interval() -> None:
    assert "final_interval_ms" not in replay_online_gate(
        np.arange(1.0, 20.0), one_sided_tail=0.0025
    )


def test_markdown_renders_from_a_comparison() -> None:
    from scripts.replay_online_gate import build_parser, render_markdown

    results = {
        "corpora": [
            {"name": "missing-corpus", "unavailable": "no traces on disk"},
        ]
    }
    provenance = {
        "final": False,
        "generated": "2026-07-20T00:00:00",
        "git_sha": "deadbeef",
        "one_sided_tail": 0.0025,
        "alpha_family": 0.05,
        "family_size": 10,
        "restore_cost_fraction": 0.94,
        "guard_ms": 0.0,
        "task_order": "unicode_lexicographic_task_id",
        "baseline_trigger_field": "robust_trigger_ms",
        "treatment_trigger_field": "offline_gated_robust_trigger_ms",
        "bet_rule": BET_RULE,
    }
    rendered = render_markdown(results, provenance)
    assert "NOT RUN" in rendered
    assert "no traces on disk" in rendered
    assert BET_RULE in rendered
    assert "does NOT establish earlier offline-verdict reproduction" in rendered
    assert "reproducing the offline verdict earlier" not in rendered
    help_text = build_parser().format_help()
    assert (
        "does not establish earlier offline-verdict reproduction" in help_text.lower()
    )
    assert "reproduces the offline verdict" not in help_text.lower()


def test_guard_rejects_the_superseded_swe_rebench_root() -> None:
    """The exact path that was wrongly used as a dev corpus must be refused.

    ``traces/swe-rebench/qwen3.7-max/20260624T162037`` holds 50 traces and looks
    like a usable corpus on disk, but ``offline-gated-confirm-100-v2`` supersedes
    it with a committed manifest. Note the test is NOT that the root is on some
    exclusion list -- that list is scoped and marks development data rather than
    forbidden data -- it is that a canonical corpus exists for that benchmark.
    """

    from scripts.replay_online_gate import _reject_superseded_roots

    repo_root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="manifest-defined corpus already exists"):
        _reject_superseded_roots(
            {"bad": Path("traces/swe-rebench/qwen3.7-max/20260624T162037")},
            repo_root,
        )


def test_guard_admits_a_dev_root_with_no_canonical_rival() -> None:
    """Terminal-Bench is on exclusion lists yet nothing supersedes it.

    The regression test for the earlier over-broad rule, which rejected every
    development corpus and so would have deleted this deliverable.
    """

    from scripts.replay_online_gate import _DEV_CORPORA, _reject_superseded_roots

    repo_root = Path(__file__).resolve().parents[1]
    terminal_bench = _DEV_CORPORA["terminal-bench"]
    assert terminal_bench.trace_root is not None
    _reject_superseded_roots({"terminal-bench": terminal_bench.trace_root}, repo_root)


def test_configured_dev_corpora_pin_a_frozen_task_list() -> None:
    """Every corpus pins its task set; the manifest form also pins a count."""

    from scripts.replay_online_gate import _DEV_CORPORA, read_pinned_task_ids

    repo_root = Path(__file__).resolve().parents[1]
    assert set(_DEV_CORPORA) == {"swe-rebench-100", "terminal-bench"}

    swe = _DEV_CORPORA["swe-rebench-100"]
    assert swe.manifest is not None
    manifest = json.loads((repo_root / swe.manifest).read_text(encoding="utf-8"))
    assert manifest["expected_task_count"] == 100
    assert manifest["trace_root"].endswith("offline-gated-confirm-100-v2")
    task_ids = Path(manifest["task_ids_file"]).read_text(encoding="utf-8")
    assert len(task_ids.split()) == 100

    terminal_bench = _DEV_CORPORA["terminal-bench"]
    assert terminal_bench.manifest is None
    assert terminal_bench.task_ids_file is not None
    assert terminal_bench.provenance_note, "weaker provenance must be recorded"
    pinned = read_pinned_task_ids(repo_root / terminal_bench.task_ids_file)
    assert len(pinned) == len(set(pinned)) == 83


def test_dev_corpus_requires_exactly_one_source_and_a_pinned_list() -> None:
    from scripts.replay_online_gate import DevCorpus

    with pytest.raises(ValueError, match="exactly one"):
        DevCorpus()
    with pytest.raises(ValueError, match="exactly one"):
        DevCorpus(manifest=Path("m.json"), trace_root=Path("traces/x"))
    with pytest.raises(ValueError, match="must pin a task_ids_file"):
        DevCorpus(trace_root=Path("traces/x"))


def test_fold_eval_splits_that_overlap_are_refused(tmp_path: Path) -> None:
    """A non-partition makes the recovered task set ambiguous, so it must fail."""

    from scripts.replay_online_gate import read_pinned_task_ids

    (tmp_path / "f1_eval.txt").write_text("a\nb\n", encoding="utf-8")
    (tmp_path / "f2_eval.txt").write_text("b\nc\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a partition"):
        read_pinned_task_ids(tmp_path)

    (tmp_path / "f2_eval.txt").write_text("c\nd\n", encoding="utf-8")
    assert read_pinned_task_ids(tmp_path) == ["a", "b", "c", "d"]


def test_final_requires_the_certified_discipline() -> None:
    from scripts.replay_online_gate import (
        _require_certified_discipline,
        build_parser,
    )

    parser = build_parser()
    weakened = parser.parse_args(["--final", "--replicates", "10"])
    with pytest.raises(ValueError, match="certified discipline"):
        _require_certified_discipline(weakened, parser)
    _require_certified_discipline(parser.parse_args(["--final"]), parser)


@pytest.mark.parametrize(
    ("args", "flag"),
    [
        (["--skip-fresh"], "--skip-fresh"),
        (["--manifest", "other.json"], "--manifest"),
    ],
)
def test_corpus_identity_is_not_cli_overridable(args: list[str], flag: str) -> None:
    from scripts.replay_online_gate import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(args)


def test_frozen_reference_manifest_pins_277_tasks() -> None:
    from scripts.replay_online_gate import (
        _CERTIFIED_REFERENCE_TASK_COUNT,
        _FROZEN_MANIFEST,
        _preflight_fresh_pin,
    )

    repo_root = Path(__file__).resolve().parents[1]
    manifest = json.loads((repo_root / _FROZEN_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["expected_task_count"] == _CERTIFIED_REFERENCE_TASK_COUNT == 277
    assert len(_preflight_fresh_pin(manifest)) == 277


def test_artifact_write_failure_publishes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts.replay_online_gate import _publish_artifacts_and_results

    out_json = tmp_path / "result.json"
    out_md = tmp_path / "result.md"
    sidecar_path = tmp_path / "result-decisions.json.zst"
    rendered = [
        {
            "name": "fresh-secret",
            "comparison": {"agreement_count": 1, "family_size": 1},
        }
    ]
    with pytest.raises(TypeError):
        _publish_artifacts_and_results(
            out_json,
            out_md,
            sidecar_path,
            {"valid": True},
            "valid markdown",
            [{"not_json_serializable": object()}],
            rendered,
        )
    assert not out_json.exists()
    assert not out_md.exists()
    assert not sidecar_path.exists()
    captured = capsys.readouterr().out
    assert "agreement" not in captured
    assert "fresh-secret" not in captured
    assert list(tmp_path.iterdir()) == []


def test_fresh_pin_drift_fails_before_trace_loading(tmp_path: Path) -> None:
    from scripts.replay_online_gate import _preflight_fresh_pin

    task_ids_file = tmp_path / "task_ids.txt"
    task_ids_file.write_text(
        "".join(f"task-{index}\n" for index in range(276)), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="pins 276 task IDs"):
        _preflight_fresh_pin(
            {"expected_task_count": 277, "task_ids_file": str(task_ids_file)}
        )


def test_failed_rerun_removes_stale_json_anchor(tmp_path: Path) -> None:
    from scripts.replay_online_gate import _write_artifacts

    out_json = tmp_path / "result.json"
    out_md = tmp_path / "result.md"
    sidecar_path = tmp_path / "result-decisions.json.zst"
    out_json.write_text('{"stale": true}', encoding="utf-8")
    sidecar_path.write_bytes(b"stale")
    out_md.mkdir()  # Makes the second publish rename fail after sidecar replace.

    with pytest.raises(IsADirectoryError):
        _write_artifacts(
            out_json,
            out_md,
            sidecar_path,
            {"current": True},
            "current markdown",
            [],
        )
    assert not out_json.exists(), "no stale completion anchor may survive"
    assert sidecar_path.exists()


def test_serialized_artifacts_record_directional_bet_rule(tmp_path: Path) -> None:
    from scripts.replay_online_gate import _write_artifacts

    out_json = tmp_path / "result.json"
    out_md = tmp_path / "result.md"
    sidecar_path = tmp_path / "result-decisions.json.zst"
    markdown = f"Directional bet rule: `{BET_RULE}`"
    _write_artifacts(
        out_json,
        out_md,
        sidecar_path,
        {"provenance": {"bet_rule": BET_RULE}},
        markdown,
        [],
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["provenance"]["bet_rule"] == BET_RULE
    assert BET_RULE in out_md.read_text(encoding="utf-8")
