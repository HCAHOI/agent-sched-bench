from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_prerestore_accounting import (
    PrerestoreConfig,
    _expected_prerestore_utility,
    _robust_swap_triggers,
    prerestore_components,
    prerestore_start_ms,
    score_decisions,
    summarize,
)
from trace_collect.tool_latency_dataset import ToolLatencySample
from trace_collect.tool_latency_profiled import LatencyPriorNode, hazard_recheck_ms


# --------------------------------------------------------------------------- #
# Per-call accounting identity (docstring contract).
# --------------------------------------------------------------------------- #
def test_components_hidden_is_min_lead_restore_in_window() -> None:
    # s < L <= s + R: restore overlaps the tail, hidden == L - s == min(L-s, R).
    hidden, wasted = prerestore_components(120.0, 50.0, restore_cost_ms=100.0)
    assert hidden == pytest.approx(70.0)  # == min(120-50, 100)
    assert wasted == 0.0


def test_components_waste_when_call_outlives_window() -> None:
    # L > s + R: the swap-in completed with the call still running -> wasted R,
    # no hidden lead. Net utility -R (a genuine downside, the gamble).
    hidden, wasted = prerestore_components(300.0, 50.0, restore_cost_ms=100.0)
    assert hidden == 0.0
    assert wasted == pytest.approx(100.0)
    assert hidden - wasted == pytest.approx(-100.0)


def test_components_zero_when_completed_before_start() -> None:
    # L <= s (covers "never swapped", L <= g <= s): baseline, differential zero.
    assert prerestore_components(40.0, 50.0, restore_cost_ms=100.0) == (0.0, 0.0)
    assert prerestore_components(40.0, None, restore_cost_ms=100.0) == (0.0, 0.0)


def test_components_boundary_full_hide_at_window_edge() -> None:
    # L == s + R exactly: fully overlapped, hidden R, no waste (the +R side of
    # the cliff). One epsilon later would flip to waste.
    hidden, wasted = prerestore_components(150.0, 50.0, restore_cost_ms=100.0)
    assert (hidden, wasted) == pytest.approx((100.0, 0.0))
    hidden2, wasted2 = prerestore_components(150.0001, 50.0, restore_cost_ms=100.0)
    assert (hidden2, wasted2) == pytest.approx((0.0, 100.0))


def test_expected_utility_matches_scalar_components() -> None:
    values = np.asarray([40.0, 120.0, 300.0], dtype=float)
    manual = np.mean(
        [
            (h - w)
            for h, w in (
                prerestore_components(float(v), 50.0, restore_cost_ms=100.0)
                for v in values
            )
        ]
    )
    assert _expected_prerestore_utility(values, 50.0, 100.0) == pytest.approx(manual)


# --------------------------------------------------------------------------- #
# Stopping-time optimizer.
# --------------------------------------------------------------------------- #
def _node(values_by_task: dict[str, list[float]]) -> LatencyPriorNode:
    flat = [v for values in values_by_task.values() for v in values]
    return LatencyPriorNode(
        values=flat, values_by_task=values_by_task, source="prior_global", group_key=None
    )


def test_optimizer_lands_on_long_call_lead_minus_restore() -> None:
    # values=[120,300,40], R=100, g=0. Hand optimum: start at 300-100=200 catches
    # the one long call just-in-time (mean +100/3), beating every other start.
    node = _node({"a": [120.0, 40.0], "b": [300.0]})
    start = prerestore_start_ms(node, swap_trigger_ms=0.0, restore_cost_ms=100.0)
    assert start == pytest.approx(200.0)
    assert _expected_prerestore_utility(
        np.asarray(node.values), start, 100.0
    ) == pytest.approx(100.0 / 3.0)


def test_optimizer_never_before_swap_trigger() -> None:
    # s* must be >= g (nothing to restore before the swap fires).
    node = _node({"a": [120.0, 300.0], "b": [500.0, 250.0]})
    for guard in (0.0, 100.0, 260.0, 1000.0):
        start = prerestore_start_ms(node, swap_trigger_ms=guard, restore_cost_ms=100.0)
        assert start is None or start >= guard


def test_optimizer_no_pre_restore_floor() -> None:
    # A node whose best start cannot beat the zero-utility no-fire baseline
    # returns None (conservative). All-short-relative-to-window mass here yields
    # only waste, never a positive window.
    node = _node({"a": [10.0, 12.0], "b": [11.0]})
    assert prerestore_start_ms(node, swap_trigger_ms=100.0, restore_cost_ms=100.0) is None


def test_optimizer_thin_and_degenerate_inherit_no_pre_restore() -> None:
    single_task = _node({"a": [120.0, 300.0]})  # one logical task -> conservative
    assert prerestore_start_ms(
        single_task, swap_trigger_ms=0.0, restore_cost_ms=100.0
    ) is None
    two_task = _node({"a": [120.0], "b": [300.0]})
    assert prerestore_start_ms(two_task, swap_trigger_ms=0.0, restore_cost_ms=0.0) is None
    empty = _node({"a": [], "b": []})
    assert prerestore_start_ms(empty, swap_trigger_ms=0.0, restore_cost_ms=100.0) is None


# --------------------------------------------------------------------------- #
# Cross-fit isolation over folds.
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


def _cfg(
    costs_ms: tuple[float, ...],
    *,
    fold_count: int = 2,
    trigger_source: str = "hazard",
    inner_folds: int = 2,
) -> PrerestoreConfig:
    return PrerestoreConfig(
        fold_count=fold_count,
        command_field="command",
        max_prefix_depth=4,
        skip_leading_cd=False,
        min_tool_history=1,
        min_profile_tasks=1,
        costs_ms=costs_ms,
        guard_ms=0.0,
        restore_cost_fraction=0.94,
        replicates=2000,
        confidence_level=0.95,
        seed=0,
        trigger_source=trigger_source,
        inner_folds=inner_folds,
    )


def test_cross_fit_start_uses_fit_fold_only() -> None:
    # fold_count=2: task index 0 -> eval fold 1 (profile = task index 1), and vice
    # versa. Each eval call's s* must be computed from the OTHER task's samples,
    # never its own held-out latency.
    kv = 1000.0
    restore = 0.94 * kv
    samples_by_task = {
        "t0": [_sample("t0", "t0:0", 5000.0)],
        "t1": [_sample("t1", "t1:0", 3000.0)],
    }
    cfg = _cfg((kv,))
    decisions = score_decisions(samples_by_task, ["t0", "t1"], cfg)
    by_task = {row["task_id"]: row for row in decisions}
    # Eval t0 is scored with the prior built from t1's single sample [3000].
    expected_start_t0 = prerestore_start_ms(
        _node({"t1": [3000.0]}), swap_trigger_ms=by_task["t0"]["swap_trigger_ms"],
        restore_cost_ms=restore,
    )
    # A single-task fit node is thin -> conservative None -> no pre-restore.
    assert expected_start_t0 is None
    assert by_task["t0"]["prerestore_start_ms"] is None
    assert by_task["t0"]["utility_ms"] == 0.0
    # Swap trigger is the fit-fold hazard optimum, not derived from eval latency.
    assert by_task["t0"]["swap_trigger_ms"] == pytest.approx(
        hazard_recheck_ms([3000.0], threshold_ms=kv, kv_cost_ms=kv, restore_cost_ms=restore)
    )


def test_cross_fit_real_positive_start_from_fit_samples() -> None:
    # 4 tasks, fold_count=2: eval fold for t0 has profile {t1, t3} (2 logical
    # tasks -> not thin), which yields a genuine positive s*. The eval call's own
    # latency (4500) never enters s*; the hidden lead is scored against it.
    kv = 1000.0
    restore = 0.94 * kv
    samples_by_task = {
        "t0": [_sample("t0", "t0:0", 4500.0)],
        "t1": [_sample("t1", "t1:0", 5000.0), _sample("t1", "t1:1", 200.0)],
        "t2": [_sample("t2", "t2:0", 100.0)],
        "t3": [_sample("t3", "t3:0", 5000.0), _sample("t3", "t3:1", 200.0)],
    }
    cfg = _cfg((kv,))
    decisions = score_decisions(samples_by_task, ["t0", "t1", "t2", "t3"], cfg)
    assert score_decisions(
        samples_by_task, ["t0", "t1", "t2", "t3"], cfg, workers=2
    ) == decisions
    row = next(r for r in decisions if r["task_id"] == "t0")
    fit_node = _node({"t1": [5000.0, 200.0], "t3": [5000.0, 200.0]})
    expected_g = hazard_recheck_ms(
        fit_node.values, threshold_ms=kv, kv_cost_ms=kv, restore_cost_ms=restore
    )
    expected_start = prerestore_start_ms(
        fit_node, swap_trigger_ms=expected_g, restore_cost_ms=restore
    )
    assert expected_start is not None and expected_start > 0.0  # real positive s*
    assert row["swap_trigger_ms"] == pytest.approx(expected_g)
    assert row["prerestore_start_ms"] == pytest.approx(expected_start)
    assert row["hidden_lead_ms"] == pytest.approx(4500.0 - expected_start)
    assert row["wasted_restore_ms"] == 0.0


# --------------------------------------------------------------------------- #
# Robust trigger source (shipped offline-gated robust clock).
# --------------------------------------------------------------------------- #
def _fold_rows(samples_by_task, task_ids, fold, fold_count):
    eval_tasks = {t for i, t in enumerate(task_ids) if i % fold_count == fold - 1}
    profile = [t for t in task_ids if t not in eval_tasks]
    to = lambda tasks: [s.to_json_obj() for t in sorted(tasks) for s in samples_by_task[t]]
    return to(eval_tasks), to(profile)


def _hetero_corpus() -> tuple[dict[str, list], list[str]]:
    # Mixed short/long: the robust clock falls back to the deadline everywhere.
    task_ids = [f"t{i}" for i in range(6)]
    samples = {
        "t0": [_sample("t0", "t0:0", 8000.0), _sample("t0", "t0:1", 300.0)],
        "t1": [_sample("t1", "t1:0", 200.0)],
        "t2": [_sample("t2", "t2:0", 5000.0), _sample("t2", "t2:1", 400.0)],
        "t3": [_sample("t3", "t3:0", 150.0)],
        "t4": [_sample("t4", "t4:0", 6000.0), _sample("t4", "t4:1", 250.0)],
        "t5": [_sample("t5", "t5:0", 350.0)],
    }
    return samples, task_ids


def _band_corpus() -> tuple[dict[str, list], list[str]]:
    # Calls in the (kv, 2kv) band: hazard and the robust clock both fire early but
    # at different times (hazard 400ms, robust 450ms) -> a genuine differ case.
    task_ids = [f"t{i}" for i in range(6)]
    samples = {
        "t0": [_sample("t0", "t0:0", 1500.0), _sample("t0", "t0:1", 1600.0)],
        "t1": [_sample("t1", "t1:0", 1400.0)],
        "t2": [_sample("t2", "t2:0", 1700.0), _sample("t2", "t2:1", 1550.0)],
        "t3": [_sample("t3", "t3:0", 1450.0)],
        "t4": [_sample("t4", "t4:0", 1650.0), _sample("t4", "t4:1", 1500.0)],
        "t5": [_sample("t5", "t5:0", 1480.0)],
    }
    return samples, task_ids


def test_robust_source_uses_shipped_clock_and_differs_from_hazard() -> None:
    kv = 1000.0
    samples_by_task, task_ids = _band_corpus()
    hz = _cfg((kv,), fold_count=2, trigger_source="hazard")
    rb = _cfg((kv,), fold_count=2, trigger_source="robust", inner_folds=2)
    hz_dec = score_decisions(samples_by_task, task_ids, hz)
    rb_dec = score_decisions(samples_by_task, task_ids, rb)

    # Plumbing: each robust decision's swap trigger IS the shipped clock's trigger.
    reproduced: dict[tuple[str, float], float] = {}
    for fold in (1, 2):
        eval_rows, profile_rows = _fold_rows(samples_by_task, task_ids, fold, 2)
        reproduced.update(_robust_swap_triggers(eval_rows, profile_rows, rb))
    for d in rb_dec:
        key = (d["sample_id"], d["kv_cost_ms"])
        assert d["swap_trigger_ms"] == pytest.approx(reproduced[key])

    # Source matters: hazard fires earlier than the (deadline-falling-back) robust
    # clock on >= 1 call, so the accounting follows the selected source.
    hz_g = {(d["sample_id"], d["kv_cost_ms"]): d["swap_trigger_ms"] for d in hz_dec}
    assert any(
        abs(d["swap_trigger_ms"] - hz_g[(d["sample_id"], d["kv_cost_ms"])]) > 1e-9
        for d in rb_dec
    )


def test_robust_no_swap_fallback_inherits_no_earlier_trigger() -> None:
    # Where the robust clock falls back to the deadline (g == threshold), the
    # pre-restore start must never precede it (no earlier trigger invented).
    kv = 1000.0
    samples_by_task, task_ids = _hetero_corpus()
    rb = _cfg((kv,), fold_count=2, trigger_source="robust", inner_folds=2)
    rb_dec = score_decisions(samples_by_task, task_ids, rb)
    fell_back = [d for d in rb_dec if d["swap_trigger_ms"] == pytest.approx(d["threshold_ms"])]
    assert fell_back  # this small heterogeneous corpus does fall back
    for d in fell_back:
        start = d["prerestore_start_ms"]
        assert start is None or start >= d["threshold_ms"] - 1e-9


def test_config_rejects_unknown_trigger_source() -> None:
    with pytest.raises(ValueError, match="trigger_source"):
        _cfg((3500.0, 5000.0), trigger_source="bogus")


# --------------------------------------------------------------------------- #
# Verdict / kill arithmetic (both directions) + determinism + identity.
# --------------------------------------------------------------------------- #
def _decision(task_id: str, cost: float, utility: float) -> dict[str, object]:
    # Hidden/wasted chosen to satisfy the identity utility == hidden - wasted so
    # summarize's per-cell assertion is exercised on crafted rows.
    hidden = max(utility, 0.0)
    wasted = max(-utility, 0.0)
    return {
        "sample_id": f"{task_id}:{cost}",
        "task_id": task_id,
        "tool_name": "bash",
        "outer_fold": "f1",
        "kv_cost_ms": cost,
        "latency_ms": 9999.0,
        "hidden_lead_ms": hidden,
        "wasted_restore_ms": wasted,
        "utility_ms": utility,
        "fired": utility != 0.0,
    }


def _panel(utils_by_cost: dict[float, float], n_tasks: int) -> list[dict[str, object]]:
    # Same per-task utility at each cost (full headline panel present).
    return [
        _decision(f"t{i}", cost, util)
        for i in range(n_tasks)
        for cost, util in utils_by_cost.items()
    ]


def test_verdict_survive_when_headline_positive() -> None:
    # 8 tasks strictly positive at headline kv 3500 -> permutation p_positive
    # hits the 2^-8 floor (< family tail) -> SURVIVE. kv5000 zero -> inconclusive,
    # so one positive headline cell is enough.
    cfg = _cfg((3500.0, 5000.0), fold_count=1)
    summary = summarize(_panel({3500.0: 100.0, 5000.0: 0.0}, 8), cfg, task_count=8)
    assert summary["verdict"] == "SURVIVE"
    assert summary["survive_positive_headline_costs_ms"] == [3500.0]
    cell = next(c for c in summary["cells"] if c["kv_cost_ms"] == 3500.0)
    assert cell["permutation_label"] == "positive"
    assert cell["net_ms_per_277"] == pytest.approx(800.0)
    assert cell["hidden_lead_ms_total"] == pytest.approx(800.0)
    assert cell["wasted_restore_ms_total"] == 0.0


def test_verdict_kill_when_mixed_signs() -> None:
    cfg = _cfg((3500.0, 5000.0), fold_count=1)
    decisions = [
        _decision(f"t{i}", cost, (100.0 if i % 2 == 0 else -100.0))
        for i in range(8)
        for cost in (3500.0, 5000.0)
    ]
    summary = summarize(decisions, cfg, task_count=8)
    assert summary["verdict"] == "KILL"
    assert summary["survive_positive_headline_costs_ms"] == []
    assert summary["cells"][0]["permutation_label"] == "inconclusive"


def test_verdict_kill_ignores_positive_non_headline_cell() -> None:
    # A positive NON-headline cell (kv 1000) must not flip the verdict; only
    # 3500/5000 count toward SURVIVE.
    cfg = _cfg((1000.0, 3500.0, 5000.0), fold_count=1)
    summary = summarize(
        _panel({1000.0: 100.0, 3500.0: 0.0, 5000.0: 0.0}, 8), cfg, task_count=8
    )
    non_headline = next(c for c in summary["cells"] if c["kv_cost_ms"] == 1000.0)
    assert non_headline["permutation_label"] == "positive"
    assert summary["verdict"] == "KILL"


def test_summarize_requires_headline_costs() -> None:
    # A cost panel missing a headline cell must fail fast, not yield a vacuous KILL.
    cfg = _cfg((3500.0,), fold_count=1)
    with pytest.raises(ValueError, match="headline"):
        summarize(_panel({3500.0: 100.0}, 8), cfg, task_count=8)


def test_summarize_deterministic() -> None:
    cfg = _cfg((3500.0, 5000.0), fold_count=1)
    decisions = _panel({3500.0: 100.0, 5000.0: 0.0}, 8)
    a = summarize(decisions, cfg, task_count=8)["cells"][0]
    b = summarize(decisions, cfg, task_count=8)["cells"][0]
    assert a["permutation_p_positive"] == b["permutation_p_positive"]
    assert a["simultaneous_interval_ms"] == b["simultaneous_interval_ms"]


def test_summarize_identity_assertion_rejects_inconsistent_rows() -> None:
    cfg = _cfg((3500.0, 5000.0), fold_count=1)
    bad = _decision("t0", 3500.0, 100.0)
    bad["wasted_restore_ms"] = 50.0  # breaks utility == hidden - wasted
    rows = [bad] + _panel({3500.0: 0.0, 5000.0: 0.0}, 8)[1:]
    with pytest.raises(AssertionError):
        summarize(rows, cfg, task_count=8)
