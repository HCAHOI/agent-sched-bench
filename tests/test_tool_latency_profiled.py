from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.exploration.evaluate_profiled_latency_thresholds import main as evaluate_profiled_main
from trace_collect.tool_latency_profiled import (
    build_latency_prior,
    evaluate_profiled_latency_thresholds,
    hazard_recheck_ms,
)
from trace_collect.tool_latency_threshold import evaluate_latency_thresholds


def test_hazard_recheck_restore_cost_shifts_trigger_past_short_calls() -> None:
    values = [80.0, 80.0, 150.0]
    kwargs = {"threshold_ms": 100.0, "kv_cost_ms": 100.0}

    assert hazard_recheck_ms(values, **kwargs) == 0.0
    # Charging the swap-back of fires on short calls makes k=0 lose to k=80,
    # which only the long call survives.
    assert hazard_recheck_ms(values, restore_cost_ms=35.0, **kwargs) == 80.0
    with pytest.raises(ValueError, match="restore_cost_ms"):
        hazard_recheck_ms(values, restore_cost_ms=float("nan"), **kwargs)


def _profile_rows() -> list[dict[str, object]]:
    # Per-tool prior: probe -> [900, 900]; other -> [50]; global -> [50, 900, 900].
    return [
        _latency_row("p1", "probe", 900.0, tool_ts_start=0.0, source_trace="trace-p"),
        _latency_row("p2", "probe", 900.0, tool_ts_start=1.0, source_trace="trace-p"),
        _latency_row("p3", "other", 50.0, tool_ts_start=2.0, source_trace="trace-p"),
    ]


def test_prior_only_uses_tool_prior_with_global_fallback_and_no_cold_start() -> None:
    eval_rows = [
        _latency_row("probe-hit", "probe", 900.0, tool_ts_start=0.0),
        _latency_row("unseen", "newtool", 50.0, tool_ts_start=1.0),
    ]

    summary = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=_profile_rows(),
        thresholds_ms=[100.0],
        predictor="prior_only",
    )

    decisions = {row["sample_id"]: row for row in summary["decisions"]}
    assert summary["profile_row_count"] == 3
    assert summary["profile_tool_count"] == 2

    probe_hit = decisions["probe-hit"]
    assert probe_hit["prior_source"] == "prior_tool"
    assert probe_hit["prior_count"] == 2
    assert probe_hit["probability_exceeds_threshold"] == 1.0
    assert probe_hit["predicted_exceeds_threshold"] is True
    assert probe_hit["online_source"] is None

    unseen = decisions["unseen"]
    assert unseen["prior_source"] == "prior_global"
    assert unseen["prior_count"] == 3
    assert unseen["probability_exceeds_threshold"] == pytest.approx(2 / 3)
    assert unseen["predicted_exceeds_threshold"] is True
    assert unseen["label_exceeds_threshold"] is False

    metrics = summary["metrics_by_threshold"]["100.0"]
    assert metrics["decided_count"] == 2
    assert metrics["cold_start_count"] == 0
    assert metrics["abstain_count"] == 0


def test_online_only_matches_unprofiled_threshold_eval() -> None:
    eval_rows = [
        _latency_row("seed-global", "seed-tool", 100.0, tool_ts_start=0.0),
        _latency_row("first-zircon", "zircon-saw", 900.0, tool_ts_start=1.0),
        _latency_row("second-zircon", "zircon-saw", 700.0, tool_ts_start=2.0),
    ]

    profiled = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=_profile_rows(),
        thresholds_ms=[500.0],
        predictor="online_only",
    )
    baseline = evaluate_latency_thresholds(eval_rows, thresholds_ms=[500.0])

    profiled_by_id = {row["sample_id"]: row for row in profiled["decisions"]}
    for base_row in baseline["decisions"]:
        row = profiled_by_id[base_row["sample_id"]]
        assert row["predicted_exceeds_threshold"] == base_row["predicted_exceeds_threshold"]
        assert row["probability_exceeds_threshold"] == base_row["probability_exceeds_threshold"]
        assert row["label_exceeds_threshold"] == base_row["label_exceeds_threshold"]
        assert row["online_source"] == base_row["prediction_source"]
        assert row["prior_source"] is None


def test_blended_pools_prior_and_online_counts() -> None:
    profile_rows = [
        _latency_row("p1", "probe", 900.0, tool_ts_start=0.0, source_trace="trace-p"),
        _latency_row("p2", "probe", 900.0, tool_ts_start=1.0, source_trace="trace-p"),
    ]
    eval_rows = [
        _latency_row("no-online", "probe", 50.0, tool_ts_start=0.0),
        _latency_row("one-online", "probe", 900.0, tool_ts_start=1.0),
    ]

    summary = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=profile_rows,
        thresholds_ms=[100.0],
        predictor="blended",
        probability_cutoff=0.6,
    )

    decisions = {row["sample_id"]: row for row in summary["decisions"]}
    # Prior alone: 2 pseudo-observations at survival 1.0.
    no_online = decisions["no-online"]
    assert no_online["probability_exceeds_threshold"] == 1.0
    assert no_online["effective_count"] == 2.0
    assert no_online["predicted_exceeds_threshold"] is True
    assert no_online["online_count"] == 0
    # Pooled with one online miss (50ms): (2 * 1.0 + 0) / 3.
    one_online = decisions["one-online"]
    assert one_online["probability_exceeds_threshold"] == pytest.approx(2 / 3)
    assert one_online["effective_count"] == 3.0
    assert one_online["predicted_exceeds_threshold"] is True

    # No cold starts for blended even with empty online history.
    metrics = summary["metrics_by_threshold"]["100.0"]
    assert metrics["cold_start_count"] == 0


def test_blended_prior_strength_caps_prior_weight() -> None:
    profile_rows = [
        _latency_row("p1", "probe", 900.0, tool_ts_start=0.0, source_trace="trace-p"),
        _latency_row("p2", "probe", 900.0, tool_ts_start=1.0, source_trace="trace-p"),
    ]
    eval_rows = [
        _latency_row("no-online", "probe", 50.0, tool_ts_start=0.0),
        _latency_row("one-online", "probe", 900.0, tool_ts_start=1.0),
    ]

    summary = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=profile_rows,
        thresholds_ms=[100.0],
        predictor="blended",
        prior_strength=1.0,
        probability_cutoff=0.6,
    )

    decisions = {row["sample_id"]: row for row in summary["decisions"]}
    # Prior scaled to 1 pseudo-observation: (1 * 1.0 + 0) / (1 + 1) = 0.5 < 0.6.
    one_online = decisions["one-online"]
    assert one_online["probability_exceeds_threshold"] == 0.5
    assert one_online["effective_count"] == 2.0
    assert one_online["predicted_exceeds_threshold"] is False


def test_wilson_abstain_band_abstains_on_thin_evidence() -> None:
    thin_profile = [
        _latency_row("p1", "probe", 900.0, tool_ts_start=0.0, source_trace="trace-p"),
        _latency_row("p2", "probe", 900.0, tool_ts_start=1.0, source_trace="trace-p"),
    ]
    rich_profile = [
        _latency_row(f"p{i}", "probe", 900.0, tool_ts_start=float(i), source_trace="trace-p")
        for i in range(20)
    ]
    eval_rows = [_latency_row("scored", "probe", 900.0, tool_ts_start=0.0)]

    thin = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=thin_profile,
        thresholds_ms=[100.0],
        predictor="prior_only",
        abstain_confidence=0.95,
    )
    rich = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=rich_profile,
        thresholds_ms=[100.0],
        predictor="prior_only",
        abstain_confidence=0.95,
    )

    # Survival estimate is 1.0 both times; only the evidence differs.
    (thin_decision,) = thin["decisions"]
    assert thin_decision["probability_exceeds_threshold"] == 1.0
    assert thin_decision["predicted_exceeds_threshold"] is None
    assert thin_decision["abstained"] is True
    assert thin_decision["ci_low"] < 0.5 <= thin_decision["ci_high"]
    thin_metrics = thin["metrics_by_threshold"]["100.0"]
    assert thin_metrics["abstain_count"] == 1
    assert thin_metrics["abstain_rate"] == 1.0
    assert thin_metrics["decided_count"] == 0

    (rich_decision,) = rich["decisions"]
    assert rich_decision["predicted_exceeds_threshold"] is True
    assert rich_decision["abstained"] is False
    assert rich_decision["ci_low"] >= 0.5


def test_fractional_prior_strength_yields_valid_wilson_band() -> None:
    profile_rows = [
        _latency_row("p1", "probe", 900.0, tool_ts_start=0.0, source_trace="trace-p"),
        _latency_row("p2", "probe", 900.0, tool_ts_start=1.0, source_trace="trace-p"),
    ]
    eval_rows = [_latency_row("scored", "probe", 900.0, tool_ts_start=0.0)]

    summary = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=profile_rows,
        thresholds_ms=[100.0],
        predictor="blended",
        prior_strength=0.5,
        abstain_confidence=0.95,
    )

    (decision,) = summary["decisions"]
    assert decision["probability_exceeds_threshold"] == 1.0
    assert decision["effective_count"] == 0.5
    assert 0.0 <= decision["ci_low"] <= decision["ci_high"] <= 1.0
    # Half a pseudo-observation is far too thin to clear the cutoff.
    assert decision["abstained"] is True


def test_wilson_band_predicts_false_when_upper_bound_misses_cutoff() -> None:
    profile_rows = [
        _latency_row(f"p{i}", "probe", 50.0, tool_ts_start=float(i), source_trace="trace-p")
        for i in range(20)
    ]
    eval_rows = [_latency_row("scored", "probe", 900.0, tool_ts_start=0.0)]

    summary = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=profile_rows,
        thresholds_ms=[100.0],
        predictor="prior_only",
        abstain_confidence=0.95,
    )

    (decision,) = summary["decisions"]
    assert decision["probability_exceeds_threshold"] == 0.0
    assert decision["ci_high"] < 0.5
    assert decision["predicted_exceeds_threshold"] is False
    assert decision["abstained"] is False


def test_min_tool_history_gates_prior_and_online_fallback_symmetrically() -> None:
    profile_rows = [
        _latency_row("p1", "probe", 900.0, tool_ts_start=0.0, source_trace="trace-p"),
        _latency_row("p2", "other", 50.0, tool_ts_start=1.0, source_trace="trace-p"),
        _latency_row("p3", "other", 50.0, tool_ts_start=2.0, source_trace="trace-p"),
    ]
    eval_rows = [
        _latency_row("first", "probe", 900.0, tool_ts_start=0.0),
        _latency_row("second", "probe", 900.0, tool_ts_start=1.0),
    ]

    summary = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=profile_rows,
        thresholds_ms=[100.0],
        predictor="blended",
        min_tool_history=2,
    )

    decisions = {row["sample_id"]: row for row in summary["decisions"]}
    # Prior has one probe sample (< 2) and online has at most one (< 2):
    # both sides fall back to their global distributions together.
    assert decisions["first"]["prior_source"] == "prior_global"
    assert decisions["first"]["online_source"] == "cold_start"
    assert decisions["second"]["prior_source"] == "prior_global"
    assert decisions["second"]["online_source"] == "global_history"


def test_min_profile_tasks_rejects_repetitive_group_and_backs_off() -> None:
    profile_rows = [
        _latency_row(
            f"pytest-{index}", "exec", 900.0, tool_ts_start=float(index),
            source_trace="trace-task-a", task_id="task-a",
            tool_args={"command": "pytest -q"},
        )
        for index in range(4)
    ]
    profile_rows.append(
        _latency_row(
            "ls-task-b", "exec", 50.0, tool_ts_start=5.0,
            source_trace="trace-task-b", task_id="task-b",
            tool_args={"command": "ls"},
        )
    )
    eval_rows = [
        _latency_row(
            "scored", "exec", 900.0, tool_ts_start=0.0,
            task_id="task-eval", tool_args={"command": "pytest -x"},
        )
    ]

    summary = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=profile_rows,
        thresholds_ms=[100.0],
        predictor="prior_only",
        command_field="command",
        min_profile_tasks=2,
    )

    (decision,) = summary["decisions"]
    assert decision["prior_source"] == "prior_tool"
    assert decision["prior_group_key"] is None
    assert decision["prior_count"] == 5
    assert decision["prior_task_count"] == 2
    assert summary["profile_task_count"] == 2
    assert summary["min_profile_tasks"] == 2


def test_min_profile_tasks_backs_off_from_tool_to_global() -> None:
    profile_rows = [
        _latency_row(
            "probe-a", "probe", 900.0, tool_ts_start=0.0,
            source_trace="trace-task-a", task_id="task-a",
        ),
        _latency_row(
            "other-b", "other", 50.0, tool_ts_start=0.0,
            source_trace="trace-task-b", task_id="task-b",
        ),
    ]
    eval_rows = [
        _latency_row(
            "scored", "probe", 900.0, tool_ts_start=0.0,
            task_id="task-eval",
        )
    ]

    summary = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=profile_rows,
        thresholds_ms=[100.0],
        predictor="prior_only",
        min_profile_tasks=2,
    )

    (decision,) = summary["decisions"]
    assert decision["prior_source"] == "prior_global"
    assert decision["prior_count"] == 2
    assert decision["prior_task_count"] == 2


def test_min_profile_tasks_selects_group_represented_by_two_tasks() -> None:
    profile_rows = [
        _latency_row(
            "p-a", "exec", 900.0, tool_ts_start=0.0,
            source_trace="trace-task-a", task_id="task-a",
            tool_args={"command": "pytest -q"},
        ),
        _latency_row(
            "p-b", "exec", 700.0, tool_ts_start=0.0,
            source_trace="trace-task-b", task_id="task-b",
            tool_args={"command": "pytest -x"},
        ),
    ]
    eval_rows = [
        _latency_row(
            "scored", "exec", 800.0, tool_ts_start=0.0,
            task_id="task-eval", tool_args={"command": "pytest tests/"},
        )
    ]

    summary = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=profile_rows,
        thresholds_ms=[100.0, 750.0, 1_000.0],
        predictor="prior_only",
        command_field="command",
        min_profile_tasks=2,
    )

    decisions = summary["decisions"]
    assert {row["prior_source"] for row in decisions} == {"prior_group"}
    assert {row["prior_group_key"] for row in decisions} == {"exec:pytest"}
    assert {row["prior_task_count"] for row in decisions} == {2}
    probabilities = [row["probability_exceeds_threshold"] for row in decisions]
    assert probabilities == sorted(probabilities, reverse=True)


def test_task_aggregation_is_invariant_to_repeated_calls_within_task() -> None:
    base_profile = [
        _latency_row(
            "slow-a", "probe", 900.0, tool_ts_start=0.0,
            source_trace="trace-task-a", task_id="task-a",
        ),
        _latency_row(
            "fast-b", "probe", 50.0, tool_ts_start=0.0,
            source_trace="trace-task-b", task_id="task-b",
        ),
    ]
    repeated_profile = base_profile + [
        _latency_row(
            f"slow-a-{index}", "probe", 900.0,
            tool_ts_start=float(index + 1),
            source_trace="trace-task-a", task_id="task-a",
        )
        for index in range(8)
    ]
    eval_rows = [
        _latency_row(
            "scored", "probe", 900.0, tool_ts_start=0.0,
            task_id="task-eval",
        )
    ]

    task_base = evaluate_profiled_latency_thresholds(
        eval_rows, profile_rows=base_profile, thresholds_ms=[100.0],
        predictor="prior_only", min_profile_tasks=2, prior_aggregation="task",
    )
    task_repeated = evaluate_profiled_latency_thresholds(
        eval_rows, profile_rows=repeated_profile, thresholds_ms=[100.0],
        predictor="prior_only", min_profile_tasks=2, prior_aggregation="task",
    )
    call_repeated = evaluate_profiled_latency_thresholds(
        eval_rows, profile_rows=repeated_profile, thresholds_ms=[100.0],
        predictor="prior_only", min_profile_tasks=2, prior_aggregation="call",
    )
    task_curve = evaluate_profiled_latency_thresholds(
        eval_rows, profile_rows=repeated_profile,
        thresholds_ms=[10.0, 100.0, 1_000.0], predictor="prior_only",
        min_profile_tasks=2, prior_aggregation="task",
    )

    assert task_base["decisions"][0]["probability_exceeds_threshold"] == 0.5
    assert task_repeated["decisions"][0]["probability_exceeds_threshold"] == 0.5
    assert call_repeated["decisions"][0]["probability_exceeds_threshold"] == 0.9
    assert task_repeated["decisions"][0]["effective_count"] == 2.0
    curve = [row["probability_exceeds_threshold"] for row in task_curve["decisions"]]
    assert curve == [1.0, 0.5, 0.0]


def test_brier_metrics_report_call_and_task_macro_coverage() -> None:
    profile_rows = [
        _latency_row(
            "p-a", "probe", 900.0, tool_ts_start=0.0,
            source_trace="trace-profile-a", task_id="profile-a",
        ),
        _latency_row(
            "p-b", "probe", 900.0, tool_ts_start=0.0,
            source_trace="trace-profile-b", task_id="profile-b",
        ),
    ]
    eval_rows = [
        _latency_row(
            f"correct-{index}", "probe", 900.0, tool_ts_start=float(index),
            source_trace="trace-eval-a", task_id="eval-a",
        )
        for index in range(3)
    ]
    eval_rows.append(
        _latency_row(
            "wrong", "probe", 50.0, tool_ts_start=0.0,
            source_trace="trace-eval-b", task_id="eval-b",
        )
    )

    summary = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=profile_rows,
        thresholds_ms=[100.0],
        predictor="prior_only",
    )

    metrics = summary["metrics_by_threshold"]["100.0"]
    assert metrics["probability_count"] == 4
    assert metrics["probability_task_count"] == 2
    assert metrics["brier_score"] == 0.25
    assert metrics["task_macro_brier_score"] == 0.5


def test_command_grouping_separates_commands_within_one_tool() -> None:
    # Tool-level exec prior is a degenerate 50/50 mix; command groups separate it.
    profile_rows = [
        _latency_row("p1", "exec", 900.0, tool_ts_start=0.0, source_trace="trace-p",
                     tool_args={"command": "pytest -x"}),
        _latency_row("p2", "exec", 900.0, tool_ts_start=1.0, source_trace="trace-p",
                     tool_args={"command": "pytest tests/"}),
        _latency_row("p3", "exec", 50.0, tool_ts_start=2.0, source_trace="trace-p",
                     tool_args={"command": "ls -la"}),
        _latency_row("p4", "exec", 50.0, tool_ts_start=3.0, source_trace="trace-p",
                     tool_args={"command": "ls /tmp"}),
    ]
    eval_rows = [
        _latency_row("slow", "exec", 900.0, tool_ts_start=0.0,
                     tool_args={"command": "pytest -q"}),
        _latency_row("fast", "exec", 50.0, tool_ts_start=1.0,
                     tool_args={"command": "ls"}),
    ]

    grouped = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=profile_rows,
        thresholds_ms=[100.0],
        predictor="prior_only",
        command_field="command",
    )
    ungrouped = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=profile_rows,
        thresholds_ms=[100.0],
        predictor="prior_only",
    )

    assert grouped["command_field"] == "command"
    assert grouped["max_prefix_depth"] == 4
    # Nodes: exec:pytest(+2 leaves), exec:ls(+2 leaves) -> 6 prefix nodes.
    assert grouped["profile_group_count"] == 6
    decisions = {row["sample_id"]: row for row in grouped["decisions"]}
    slow, fast = decisions["slow"], decisions["fast"]
    # "pytest -q" is unseen at depth 2; backs off to the exec:pytest node.
    assert slow["prior_group_key"] == "exec:pytest"
    assert slow["prior_source"] == "prior_group"
    assert slow["probability_exceeds_threshold"] == 1.0
    assert slow["predicted_exceeds_threshold"] is True
    assert fast["prior_group_key"] == "exec:ls"
    assert fast["probability_exceeds_threshold"] == 0.0
    assert fast["predicted_exceeds_threshold"] is False

    metrics = grouped["metrics_by_threshold"]["100.0"]
    assert metrics["accuracy"] == 1.0
    assert metrics["predicted_positive_rate"] == 0.5

    # Without grouping the tool-level prior is exactly at the 0.5 cutoff for
    # every exec row: both rows predicted True (a degenerate constant answer).
    flat = {row["sample_id"]: row for row in ungrouped["decisions"]}
    assert flat["slow"]["probability_exceeds_threshold"] == 0.5
    assert flat["fast"]["probability_exceeds_threshold"] == 0.5
    assert ungrouped["metrics_by_threshold"]["100.0"]["predicted_positive_rate"] == 1.0


def test_online_history_prefers_command_group_over_tool() -> None:
    eval_rows = [
        _latency_row("e1", "exec", 50.0, tool_ts_start=0.0,
                     tool_args={"command": "ls"}),
        _latency_row("e2", "exec", 900.0, tool_ts_start=1.0,
                     tool_args={"command": "pytest -x"}),
        _latency_row("e3", "exec", 900.0, tool_ts_start=2.0,
                     tool_args={"command": "pytest tests/"}),
    ]

    summary = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=_profile_rows(),
        thresholds_ms=[100.0],
        predictor="online_only",
        command_field="command",
    )

    decisions = {row["sample_id"]: row for row in summary["decisions"]}
    # e2: no pytest-node history yet, backs off to same-tool history [50].
    assert decisions["e2"]["online_source"] == "tool_history"
    assert decisions["e2"]["online_group_key"] is None
    assert decisions["e2"]["probability_exceeds_threshold"] == 0.0
    # e3: the exec:pytest node history [900] beats the mixed tool history.
    assert decisions["e3"]["online_source"] == "group_history"
    assert decisions["e3"]["online_group_key"] == "exec:pytest"
    assert decisions["e3"]["probability_exceeds_threshold"] == 1.0
    assert decisions["e3"]["predicted_exceeds_threshold"] is True


def test_prefix_depth_differentiates_flag_values_with_backoff() -> None:
    profile_rows = [
        _latency_row("p1", "exec", 50.0, tool_ts_start=0.0, source_trace="trace-p",
                     tool_args={"command": "make -j2"}),
        _latency_row("p2", "exec", 50.0, tool_ts_start=1.0, source_trace="trace-p",
                     tool_args={"command": "make -j2"}),
        _latency_row("p3", "exec", 900.0, tool_ts_start=2.0, source_trace="trace-p",
                     tool_args={"command": "make -j12"}),
        _latency_row("p4", "exec", 900.0, tool_ts_start=3.0, source_trace="trace-p",
                     tool_args={"command": "make -j12"}),
    ]
    eval_rows = [
        _latency_row("fast", "exec", 50.0, tool_ts_start=0.0,
                     tool_args={"command": "make -j2"}),
        _latency_row("slow", "exec", 900.0, tool_ts_start=1.0,
                     tool_args={"command": "make -j12"}),
        _latency_row("unseen", "exec", 900.0, tool_ts_start=2.0,
                     tool_args={"command": "make -j99"}),
    ]

    summary = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=profile_rows,
        thresholds_ms=[100.0],
        predictor="prior_only",
        command_field="command",
    )

    decisions = {row["sample_id"]: row for row in summary["decisions"]}
    # Same program, different flag values: separated at depth 2.
    fast, slow, unseen = decisions["fast"], decisions["slow"], decisions["unseen"]
    assert fast["prior_group_key"] == "exec:make -j2"
    assert fast["probability_exceeds_threshold"] == 0.0
    assert fast["predicted_exceeds_threshold"] is False
    assert slow["prior_group_key"] == "exec:make -j12"
    assert slow["probability_exceeds_threshold"] == 1.0
    assert slow["predicted_exceeds_threshold"] is True
    # Unseen flag value backs off to the shared exec:make node (mixed 50/50).
    assert unseen["prior_group_key"] == "exec:make"
    assert unseen["probability_exceeds_threshold"] == 0.5
    assert unseen["prior_count"] == 4


def test_skip_leading_cd_groups_across_working_directories() -> None:
    profile_rows = [
        _latency_row("p1", "exec", 900.0, tool_ts_start=0.0, source_trace="trace-p",
                     tool_args={"command": "cd /repo-a && pytest -x"}),
        _latency_row("p2", "exec", 900.0, tool_ts_start=1.0, source_trace="trace-p",
                     tool_args={"command": "cd /repo-a && pytest tests/"}),
    ]
    eval_rows = [
        _latency_row("scored", "exec", 900.0, tool_ts_start=0.0,
                     tool_args={"command": "cd /repo-b && pytest -q"}),
    ]

    summary = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=profile_rows,
        thresholds_ms=[100.0],
        predictor="prior_only",
        command_field="command",
        skip_leading_cd=True,
    )

    assert summary["skip_leading_cd"] is True
    (decision,) = summary["decisions"]
    # Different working directories share the workload node.
    assert decision["prior_group_key"] == "exec:pytest"
    assert decision["prior_count"] == 2
    assert decision["probability_exceeds_threshold"] == 1.0


def _segment_profile_rows() -> list[dict[str, object]]:
    # Model fit: c(prepA)=c(prepB)=200 from singles, c(work)=500-200=300
    # attributed. The exec:work prior node stores attributed values [300]*4.
    rows = []
    for index, command in enumerate(["prepA", "prepA", "prepB", "prepB"]):
        rows.append(
            _latency_row(f"p{index}", "exec", 200.0, tool_ts_start=float(index),
                         source_trace="trace-p", tool_args={"command": command})
        )
    for index, command in enumerate(
        ["prepA && work", "prepA && work", "prepB && work", "prepB && work"]
    ):
        rows.append(
            _latency_row(f"c{index}", "exec", 500.0, tool_ts_start=4.0 + index,
                         source_trace="trace-p", tool_args={"command": command})
        )
    return rows


def test_segment_costs_deduct_preamble_from_threshold() -> None:
    # A preamble combination never seen as a whole command in the profile.
    eval_rows = [
        _latency_row("compound", "exec", 680.0, tool_ts_start=0.0,
                     tool_args={"command": "prepA && prepB && work"}),
    ]

    plain = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=_segment_profile_rows(),
        thresholds_ms=[650.0],
        predictor="prior_only",
        command_field="command",
    )
    attributed = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=_segment_profile_rows(),
        thresholds_ms=[650.0],
        predictor="prior_only",
        command_field="command",
        segment_costs=True,
    )

    # Whole-command keying backs off to a prepA-prefixed node holding raw
    # 500ms compounds and misses the true-long call.
    (plain_decision,) = plain["decisions"]
    assert plain_decision["label_exceeds_threshold"] is True
    assert plain_decision["predicted_exceeds_threshold"] is False
    assert plain_decision["threshold_deduction_ms"] is None

    # Dominant-segment keying queries exec:work (attributed [300]*4) at the
    # deducted budget 650 - 400 = 250 and catches it.
    assert attributed["segment_costs"] is True
    assert attributed["segment_cost_model"]["head_count"] == 3
    (decision,) = attributed["decisions"]
    assert decision["prior_group_key"] == "exec:work"
    assert decision["threshold_deduction_ms"] == pytest.approx(400.0, abs=1e-6)
    assert decision["probability_exceeds_threshold"] == 1.0
    assert decision["predicted_exceeds_threshold"] is True


def test_segment_costs_attribute_group_node_values() -> None:
    # exec:work holds attributed values [300, 300], not raw [500, 500]: a
    # bare "work" call queried at 400ms must see survival 0.
    eval_rows = [
        _latency_row("bare-work", "exec", 280.0, tool_ts_start=0.0,
                     tool_args={"command": "work"}),
    ]

    summary = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=_segment_profile_rows(),
        thresholds_ms=[400.0],
        predictor="prior_only",
        command_field="command",
        segment_costs=True,
    )

    (decision,) = summary["decisions"]
    assert decision["prior_group_key"] == "exec:work"
    assert decision["threshold_deduction_ms"] == 0.0
    assert decision["probability_exceeds_threshold"] == 0.0
    assert decision["predicted_exceeds_threshold"] is False
    assert decision["label_exceeds_threshold"] is False


def test_segment_costs_clamp_deducted_threshold_at_zero() -> None:
    eval_rows = [
        _latency_row("tiny-budget", "exec", 480.0, tool_ts_start=0.0,
                     tool_args={"command": "prepA && work"}),
    ]

    summary = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=_segment_profile_rows(),
        thresholds_ms=[150.0],
        predictor="prior_only",
        command_field="command",
        segment_costs=True,
    )

    # Deduction 200 exceeds the 150ms threshold: query clamps to 0 and any
    # positive attributed sample counts as exceeding.
    (decision,) = summary["decisions"]
    assert decision["threshold_deduction_ms"] == pytest.approx(200.0, abs=1e-6)
    assert decision["probability_exceeds_threshold"] == 1.0
    assert decision["predicted_exceeds_threshold"] is True


def test_segment_costs_attribute_online_history_values() -> None:
    eval_rows = [
        _latency_row("e1", "exec", 500.0, tool_ts_start=0.0,
                     tool_args={"command": "prepA && work"}),
        _latency_row("e2", "exec", 450.0, tool_ts_start=1.0,
                     tool_args={"command": "work"}),
    ]

    summary = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=_segment_profile_rows(),
        thresholds_ms=[400.0],
        predictor="online_only",
        command_field="command",
        segment_costs=True,
    )

    decisions = {row["sample_id"]: row for row in summary["decisions"]}
    # e1's completed observation enters exec:work as 500 - 200 = 300, so
    # e2's survival at 400ms is 0 (raw storage would wrongly give 1.0).
    e2 = decisions["e2"]
    assert e2["online_source"] == "group_history"
    assert e2["online_group_key"] == "exec:work"
    assert e2["probability_exceeds_threshold"] == 0.0
    assert e2["predicted_exceeds_threshold"] is False


def test_segment_costs_require_command_field() -> None:
    with pytest.raises(ValueError, match="segment_costs requires command_field"):
        evaluate_profiled_latency_thresholds(
            [_latency_row("good", "exec", 100.0, tool_ts_start=0.0)],
            profile_rows=_profile_rows(),
            thresholds_ms=[100.0],
            predictor="prior_only",
            segment_costs=True,
        )


def test_segment_costs_and_skip_leading_cd_are_exclusive() -> None:
    with pytest.raises(ValueError, match="alternative preamble treatments"):
        evaluate_profiled_latency_thresholds(
            [_latency_row("good", "exec", 100.0, tool_ts_start=0.0)],
            profile_rows=_segment_profile_rows(),
            thresholds_ms=[100.0],
            predictor="prior_only",
            command_field="command",
            segment_costs=True,
            skip_leading_cd=True,
        )


def test_conflicting_deductions_for_duplicate_sample_ids_are_rejected() -> None:
    eval_rows = [
        _latency_row("dup", "exec", 480.0, tool_ts_start=0.0,
                     tool_args={"command": "prepA && work"}),
        _latency_row("dup", "exec", 480.0, tool_ts_start=1.0,
                     tool_args={"command": "work"}),
    ]

    with pytest.raises(ValueError, match="duplicate sample_id .* conflicting"):
        evaluate_profiled_latency_thresholds(
            eval_rows,
            profile_rows=_segment_profile_rows(),
            thresholds_ms=[100.0],
            predictor="prior_only",
            command_field="command",
            segment_costs=True,
        )


def test_max_prefix_depth_caps_trie_nodes_end_to_end() -> None:
    profile_rows = [
        _latency_row("p1", "exec", 900.0, tool_ts_start=0.0, source_trace="trace-p",
                     tool_args={"command": "make -j12 all install"}),
        _latency_row("p2", "exec", 900.0, tool_ts_start=1.0, source_trace="trace-p",
                     tool_args={"command": "make -j12 all clean"}),
    ]
    eval_rows = [
        _latency_row("scored", "exec", 900.0, tool_ts_start=0.0,
                     tool_args={"command": "make -j12 all verify"}),
    ]

    summary = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=profile_rows,
        thresholds_ms=[100.0],
        predictor="prior_only",
        command_field="command",
        max_prefix_depth=2,
    )

    # Depth 2 cap: only exec:make and exec:make -j12 nodes exist; the depth-3
    # "all" node that both profile commands share is never created.
    assert summary["max_prefix_depth"] == 2
    assert summary["profile_group_count"] == 2
    (decision,) = summary["decisions"]
    assert decision["prior_group_key"] == "exec:make -j12"
    assert decision["prior_count"] == 2


def test_empty_eval_rows_are_rejected() -> None:
    with pytest.raises(ValueError, match="no latency rows supplied"):
        evaluate_profiled_latency_thresholds(
            [],
            profile_rows=_profile_rows(),
            thresholds_ms=[100.0],
            predictor="prior_only",
        )


def test_shared_traces_between_profile_and_eval_are_rejected() -> None:
    eval_rows = [
        _latency_row("leak", "probe", 900.0, tool_ts_start=0.0, source_trace="trace-p"),
    ]

    with pytest.raises(ValueError, match="disjoint traces.*trace-p"):
        evaluate_profiled_latency_thresholds(
            eval_rows,
            profile_rows=_profile_rows(),
            thresholds_ms=[100.0],
            predictor="prior_only",
        )


def test_shared_logical_task_between_different_traces_is_rejected() -> None:
    profile_rows = [
        _latency_row(
            "profile", "probe", 900.0, tool_ts_start=0.0,
            source_trace="trace-profile-attempt", task_id="same-task",
        )
    ]
    eval_rows = [
        _latency_row(
            "eval", "probe", 900.0, tool_ts_start=0.0,
            source_trace="trace-eval-attempt", task_id="same-task",
        )
    ]

    with pytest.raises(ValueError, match="disjoint logical tasks.*same-task"):
        evaluate_profiled_latency_thresholds(
            eval_rows,
            profile_rows=profile_rows,
            thresholds_ms=[100.0],
            predictor="prior_only",
        )


def test_conflicting_task_ids_within_one_trace_are_rejected() -> None:
    profile_rows = [
        _latency_row(
            "first", "probe", 900.0, tool_ts_start=0.0,
            source_trace="same-trace", task_id="task-a",
        ),
        _latency_row(
            "second", "probe", 900.0, tool_ts_start=1.0,
            source_trace="same-trace", task_id="task-b",
        ),
    ]

    with pytest.raises(ValueError, match="maps to conflicting task_id values"):
        build_latency_prior(profile_rows)


def test_empty_profile_rows_are_rejected() -> None:
    with pytest.raises(ValueError, match="empty latency prior"):
        build_latency_prior([])


def test_profiled_eval_rejects_invalid_parameters() -> None:
    eval_rows = [_latency_row("good", "probe", 100.0, tool_ts_start=0.0)]

    with pytest.raises(ValueError, match="unknown predictor"):
        evaluate_profiled_latency_thresholds(
            eval_rows,
            profile_rows=_profile_rows(),
            thresholds_ms=[100.0],
            predictor="oracle",
        )
    for strength in [0.0, -1.0, float("inf"), float("nan")]:
        with pytest.raises(ValueError, match="prior_strength must be finite and positive"):
            evaluate_profiled_latency_thresholds(
                eval_rows,
                profile_rows=_profile_rows(),
                thresholds_ms=[100.0],
                predictor="blended",
                prior_strength=strength,
            )
    for confidence in [0.0, 1.0, 1.5, float("nan")]:
        with pytest.raises(ValueError, match="abstain_confidence must be finite"):
            evaluate_profiled_latency_thresholds(
                eval_rows,
                profile_rows=_profile_rows(),
                thresholds_ms=[100.0],
                predictor="prior_only",
                abstain_confidence=confidence,
            )
    with pytest.raises(ValueError, match="min_profile_tasks must be >= 1"):
        evaluate_profiled_latency_thresholds(
            eval_rows,
            profile_rows=_profile_rows(),
            thresholds_ms=[100.0],
            predictor="prior_only",
            min_profile_tasks=0,
        )
    with pytest.raises(ValueError, match="fewer logical tasks"):
        evaluate_profiled_latency_thresholds(
            eval_rows,
            profile_rows=_profile_rows(),
            thresholds_ms=[100.0],
            predictor="prior_only",
            min_profile_tasks=2,
        )
    with pytest.raises(ValueError, match="supported only for prior_only"):
        evaluate_profiled_latency_thresholds(
            eval_rows,
            profile_rows=_profile_rows(),
            thresholds_ms=[100.0],
            predictor="blended",
            prior_aggregation="task",
        )
    with pytest.raises(ValueError, match="not supported with Wilson"):
        evaluate_profiled_latency_thresholds(
            eval_rows,
            profile_rows=_profile_rows(),
            thresholds_ms=[100.0],
            predictor="prior_only",
            prior_aggregation="task",
            abstain_confidence=0.95,
        )
    with pytest.raises(ValueError, match="not supported with hazard"):
        evaluate_profiled_latency_thresholds(
            eval_rows,
            profile_rows=_profile_rows(),
            thresholds_ms=[100.0],
            predictor="prior_only",
            prior_aggregation="task",
            hazard_kv_by_threshold={100.0: 100.0},
        )


def test_evaluate_profiled_cli_compares_predictors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path = tmp_path / "profile.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    summary_path = tmp_path / "summary.json"
    decisions_path = tmp_path / "decisions.jsonl"
    _write_jsonl(profile_path, _profile_rows())
    _write_jsonl(
        eval_path,
        [
            _latency_row("first", "probe", 900.0, tool_ts_start=1.0),
            _latency_row("second", "probe", 50.0, tool_ts_start=2.0),
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_profiled_latency_thresholds.py",
            "--profile-latencies",
            str(profile_path),
            "--eval-latencies",
            str(eval_path),
            "--thresholds-ms",
            "100",
            "--output",
            str(summary_path),
            "--decisions-output",
            str(decisions_path),
        ],
    )

    evaluate_profiled_main()

    assert f"Evaluated 3 predictors, 6 decisions -> {summary_path}" in capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert sorted(summary["by_predictor"]) == ["blended", "online_only", "prior_only"]
    for predictor_summary in summary["by_predictor"].values():
        assert "decisions" not in predictor_summary
        assert predictor_summary["row_count"] == 2

    decisions = _read_jsonl(decisions_path)
    assert len(decisions) == 6
    assert {row["predictor"] for row in decisions} == {
        "prior_only",
        "online_only",
        "blended",
    }
    # online_only has one cold start; the profiled predictors decide every row.
    online_first = next(
        row
        for row in decisions
        if row["predictor"] == "online_only" and row["sample_id"] == "first"
    )
    assert online_first["predicted_exceeds_threshold"] is None
    prior_first = next(
        row
        for row in decisions
        if row["predictor"] == "prior_only" and row["sample_id"] == "first"
    )
    assert prior_first["predicted_exceeds_threshold"] is True


def test_evaluate_profiled_cli_forwards_task_prior_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_path = tmp_path / "profile.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_jsonl(
        profile_path,
        [
            _latency_row(
                "p-a", "probe", 900.0, tool_ts_start=0.0,
                source_trace="trace-a", task_id="task-a",
            ),
            _latency_row(
                "p-b", "probe", 50.0, tool_ts_start=0.0,
                source_trace="trace-b", task_id="task-b",
            ),
        ],
    )
    _write_jsonl(
        eval_path,
        [_latency_row(
            "eval", "probe", 900.0, tool_ts_start=0.0,
            task_id="task-eval",
        )],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_profiled_latency_thresholds.py",
            "--profile-latencies", str(profile_path),
            "--eval-latencies", str(eval_path),
            "--thresholds-ms", "100",
            "--predictors", "prior_only",
            "--min-profile-tasks", "2",
            "--prior-aggregation", "task",
            "--output", str(summary_path),
        ],
    )

    evaluate_profiled_main()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))["by_predictor"]
    prior_summary = summary["prior_only"]
    assert prior_summary["min_profile_tasks"] == 2
    assert prior_summary["prior_aggregation"] == "task"
    assert prior_summary["profile_task_count"] == 2


def _latency_row(
    sample_id: str,
    tool_name: str,
    latency_ms: float,
    *,
    tool_ts_start: float,
    source_trace: str = "trace-e",
    task_id: str | None = None,
    tool_args: dict[str, object] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_id": sample_id,
        "source_trace": source_trace,
        "tool_name": tool_name,
        "latency_ms": latency_ms,
        "tool_ts_start": tool_ts_start,
        "tool_ts_end": tool_ts_start + latency_ms / 1000.0,
    }
    if task_id is not None:
        row["task_id"] = task_id
    if tool_args is not None:
        row["tool_args"] = tool_args
    return row


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
