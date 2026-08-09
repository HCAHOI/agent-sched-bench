from __future__ import annotations

import pytest

import scripts.evaluation.evaluate_gpu_tool_gap_actions as gpu_actions
from scripts.evaluation.evaluate_gpu_tool_gap_actions import (
    TransferPoint,
    _summarize,
    choose_transfer_point,
    score_gap_action,
)
from spike.multitenant import ToolSpan


def test_transfer_point_uses_conservative_ceiling_without_extrapolation() -> None:
    points = (
        TransferPoint(1024, 1.0, 7.0, 5.0),
        TransferPoint(2048, 2.0, 14.0, 10.0),
    )

    assert choose_transfer_point(points, 1024) == points[0]
    assert choose_transfer_point(points, 1025) == points[1]
    with pytest.raises(ValueError, match="exceeds measured maximum"):
        choose_transfer_point(points, 2049)


def test_gap_action_prices_earlier_offload_and_prerestore() -> None:
    tool = ToolSpan("exec", "pytest -q", 0.0, 10_000.0)

    deadline = score_gap_action(
        gap_ms=10_000.0,
        tools=(tool,),
        triggers_ms=(5000.0,),
        prerestore_starts_ms=(None,),
        size_gib=2.0,
        swap_out_ms=1000.0,
        swap_in_ms=500.0,
    )
    robust = score_gap_action(
        gap_ms=10_000.0,
        tools=(tool,),
        triggers_ms=(2000.0,),
        prerestore_starts_ms=(None,),
        size_gib=2.0,
        swap_out_ms=1000.0,
        swap_in_ms=500.0,
    )
    hidden = score_gap_action(
        gap_ms=10_000.0,
        tools=(tool,),
        triggers_ms=(5000.0,),
        prerestore_starts_ms=(9500.0,),
        size_gib=2.0,
        swap_out_ms=1000.0,
        swap_in_ms=500.0,
    )
    too_early = score_gap_action(
        gap_ms=10_000.0,
        tools=(tool,),
        triggers_ms=(5000.0,),
        prerestore_starts_ms=(8000.0,),
        size_gib=2.0,
        swap_out_ms=1000.0,
        swap_in_ms=500.0,
    )

    assert deadline["released_gib_s"] == pytest.approx(8.0)
    assert deadline["critical_path_stall_ms"] == pytest.approx(500.0)
    assert robust["released_gib_s"] == pytest.approx(14.0)
    assert robust["critical_path_stall_ms"] == pytest.approx(500.0)
    assert hidden["released_gib_s"] == pytest.approx(7.0)
    assert hidden["critical_path_stall_ms"] == pytest.approx(0.0)
    assert hidden["hidden_reload_ms"] == pytest.approx(500.0)
    assert too_early["critical_path_stall_ms"] == pytest.approx(500.0)
    assert too_early["wasted_reload_ms"] == pytest.approx(500.0)


def test_gap_action_requires_a_tool_to_survive_its_trigger() -> None:
    short = ToolSpan("exec", "true", 0.0, 100.0)
    later = ToolSpan("exec", "pytest", 500.0, 3000.0)

    none = score_gap_action(
        gap_ms=4000.0,
        tools=(short,),
        triggers_ms=(500.0,),
        prerestore_starts_ms=(None,),
        size_gib=1.0,
        swap_out_ms=100.0,
        swap_in_ms=50.0,
    )
    fired = score_gap_action(
        gap_ms=4000.0,
        tools=(short, later),
        triggers_ms=(500.0, 1000.0),
        prerestore_starts_ms=(None, None),
        size_gib=1.0,
        swap_out_ms=100.0,
        swap_in_ms=50.0,
    )

    assert none["offloaded"] is False
    assert none["released_gib_s"] == 0.0
    assert fired["offload_tool_index"] == 1
    assert fired["offload_start_ms"] == pytest.approx(1500.0)


def test_gap_action_never_fires_after_the_next_turn_arrives() -> None:
    overlapping = ToolSpan("exec", "pytest", 0.0, 10_000.0)

    result = score_gap_action(
        gap_ms=6000.0,
        tools=(overlapping,),
        triggers_ms=(9000.0,),
        prerestore_starts_ms=(None,),
        size_gib=1.0,
        swap_out_ms=100.0,
        swap_in_ms=50.0,
    )

    assert result["offloaded"] is False
    assert result["critical_path_stall_ms"] == 0.0


def test_summary_distinguishes_planned_from_fired_prerestores() -> None:
    tool = ToolSpan("exec", "pytest", 0.0, 10_000.0)
    late = score_gap_action(
        gap_ms=10_000.0,
        tools=(tool,),
        triggers_ms=(5000.0,),
        prerestore_starts_ms=(12_000.0,),
        size_gib=1.0,
        swap_out_ms=100.0,
        swap_in_ms=50.0,
    )
    fired = score_gap_action(
        gap_ms=10_000.0,
        tools=(tool,),
        triggers_ms=(5000.0,),
        prerestore_starts_ms=(9975.0,),
        size_gib=1.0,
        swap_out_ms=100.0,
        swap_in_ms=50.0,
    )

    summary = _summarize(
        [{"arms": {"candidate": late}}, {"arms": {"candidate": fired}}],
        "candidate",
    )

    assert summary["prerestore_plan_count"] == 2
    assert summary["prerestore_fired_count"] == 1


def test_task_counts_fail_closed_against_configured_replay_and_profile_sizes() -> None:
    workload = {"expected_task_count": 2, "expected_profile_task_count": 1}

    gpu_actions._validate_task_counts(workload, {"r1", "r2"}, {"p1"})
    with pytest.raises(ValueError, match="expected 2 replay tasks"):
        gpu_actions._validate_task_counts(workload, {"r1"}, {"p1"})
    with pytest.raises(ValueError, match="expected 1 profile tasks"):
        gpu_actions._validate_task_counts(workload, {"r1", "r2"}, set())
