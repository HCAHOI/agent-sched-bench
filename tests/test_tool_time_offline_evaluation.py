from __future__ import annotations

import pytest

from tool_time.offline_evaluation import (
    evaluate_offline_probe_clock,
    mean_clock_region_stats,
    select_probe_guard,
)


def test_mean_clock_region_stats_partition_surviving_calls() -> None:
    stats = mean_clock_region_stats(
        [80.0, 80.0, 150.0],
        threshold_ms=100.0,
        kv_cost_ms=100.0,
    )

    assert stats.trigger_ms == 0.0
    assert stats.normalized_band_gain == pytest.approx(1.0 / 3.0)
    assert stats.normalized_short_penalty == pytest.approx(2.0 / 15.0)
    assert stats.normalized_margin == pytest.approx(0.2)
    assert stats.survivor_count == 3
    assert stats.probability_short_given_survival == pytest.approx(2.0 / 3.0)
    assert stats.probability_band_given_survival == pytest.approx(1.0 / 3.0)
    assert stats.probability_far_given_survival == 0.0


def test_select_probe_guard_uses_observed_utility_breakpoints() -> None:
    calibration = select_probe_guard(
        [
            _probe_decision("short", "task-a", latency_ms=80.0, score=0.1),
            _probe_decision("band", "task-b", latency_ms=150.0, score=0.5),
        ]
    )

    assert calibration["selected_guard_normalized"] == 0.1
    assert calibration["probe_objective_normalized"] == 1.0
    assert calibration["accepted_early_decision_count"] == 1
    assert calibration["candidate_guard_count"] == 4


def test_select_probe_guard_can_disable_all_early_actions() -> None:
    calibration = select_probe_guard(
        [
            _probe_decision("short-a", "task-a", latency_ms=80.0, score=0.1),
            _probe_decision("short-b", "task-b", latency_ms=90.0, score=0.5),
        ]
    )

    assert calibration["selected_guard_normalized"] is None
    assert calibration["probe_objective_normalized"] == 0.0
    assert calibration["accepted_early_decision_count"] == 0


def test_offline_probe_cross_fits_profile_tasks_and_applies_learned_guard() -> None:
    result = evaluate_offline_probe_clock(
        _eval_rows(),
        profile_rows=_profile_rows(),
        kv_costs_ms=[100.0],
        guard_ms=0.0,
        inner_folds=4,
    )

    assert result["profile_task_count"] == 8
    assert result["calibration"]["probe_task_count"] == 8
    assert result["calibration"]["selected_guard_normalized"] == 0.0
    assert result["robust_calibration"]["probe_task_count"] == 8
    assert result["robust_calibration"]["selected_guard_normalized"] == 0.0
    assert "dataset_name" not in result
    point = result["points"]["100.0"]
    calibrated = point["policies"]["offline_probe_guard"]
    assert calibrated["delta_vs_deadline_ms"] == 80.0
    assert calibrated["early_trigger_count"] == 2
    gated = point["policies"]["offline_gated_robust_clock"]
    assert gated["delta_vs_deadline_ms"] == 80.0
    assert gated["early_trigger_count"] == 2
    for decision in result["decisions"]:
        assert decision["offline_probe_trigger_ms"] == 0.0
        assert decision["offline_gated_robust_trigger_ms"] in {
            decision["robust_trigger_ms"],
            decision["threshold_ms"],
        }
        assert decision["probe_robust_margin_normalized"] > 0.0
        probabilities = [
            decision["probe_probability_short_given_survival"],
            decision["probe_probability_band_given_survival"],
            decision["probe_probability_far_given_survival"],
        ]
        assert sum(probabilities) == pytest.approx(1.0)


def test_offline_probe_refit_with_restore_cost_moves_triggers_later() -> None:
    result = evaluate_offline_probe_clock(
        _eval_rows(),
        profile_rows=_profile_rows(),
        kv_costs_ms=[100.0],
        guard_ms=0.0,
        inner_folds=4,
        restore_cost_fraction=0.35,
    )

    assert result["restore_cost_fraction"] == 0.35
    # At restore 35 the k=0 clock loses to k=80 on every [80,80,150] node, so
    # both refit policies wait past the short calls and never fire on them.
    point = result["points"]["100.0"]
    for policy in ("offline_probe_guard", "offline_gated_robust_clock"):
        row = point["policies"][policy]
        assert row["early_trigger_count"] == 1
        assert row["early_trigger_on_short_count"] == 0
        assert row["delta_vs_deadline_ms"] == 40.0
    for decision in result["decisions"]:
        assert decision["offline_probe_trigger_ms"] == 80.0
        assert decision["offline_gated_robust_trigger_ms"] == 80.0


def test_offline_robust_guard_filters_low_margin_and_retains_high_margin() -> None:
    profile_by_task = {
        "p0": {"alpha": [180.0, 220.0, 120.0], "beta": [220.0, 80.0, 80.0]},
        "p1": {"alpha": [80.0, 150.0], "beta": [80.0]},
        "p2": {"alpha": [220.0, 20.0], "beta": [60.0, 60.0]},
        "p3": {"alpha": [180.0, 20.0, 80.0], "beta": [80.0, 80.0, 180.0]},
        "p4": {"alpha": [180.0, 80.0, 180.0], "beta": [180.0]},
        "p5": {"alpha": [60.0], "beta": [80.0]},
        "p6": {"alpha": [180.0, 120.0, 120.0], "beta": [80.0, 20.0, 60.0]},
        "p7": {"alpha": [60.0, 150.0], "beta": [20.0, 20.0, 20.0]},
    }
    result = evaluate_offline_probe_clock(
        [
            _row(
                "alpha-band",
                150.0,
                task_id="eval-a",
                source_trace="eval-a",
                command="alpha",
            ),
            _row(
                "beta-band",
                150.0,
                task_id="eval-b",
                source_trace="eval-b",
                command="beta",
            ),
            _row(
                "alpha-short",
                80.0,
                task_id="eval-c",
                source_trace="eval-c",
                command="alpha",
            ),
            _row(
                "beta-short",
                80.0,
                task_id="eval-d",
                source_trace="eval-d",
                command="beta",
            ),
        ],
        profile_rows=_grouped_profile_rows(profile_by_task),
        kv_costs_ms=[100.0],
        guard_ms=0.0,
        inner_folds=4,
        command_field="command",
    )

    calibration = result["robust_calibration"]
    assert calibration["selected_guard_normalized"] == pytest.approx(
        0.03636363636363638
    )
    assert calibration["eligible_early_decision_count"] == 28
    assert calibration["accepted_early_decision_count"] == 19
    decisions = {row["sample_id"]: row for row in result["decisions"]}
    for sample_id in ("alpha-band", "alpha-short"):
        assert decisions[sample_id]["robust_trigger_ms"] == 80.0
        assert decisions[sample_id]["offline_gated_robust_trigger_ms"] == 80.0
    for sample_id in ("beta-band", "beta-short"):
        assert decisions[sample_id]["robust_trigger_ms"] == 80.0
        assert decisions[sample_id]["offline_gated_robust_trigger_ms"] == 100.0
    policies = result["points"]["100.0"]["policies"]
    assert policies["robust_clock"]["delta_vs_deadline_ms"] == 80.0
    assert policies["offline_gated_robust_clock"]["delta_vs_deadline_ms"] == 40.0


def test_offline_robust_guard_can_select_never_early() -> None:
    latencies_by_task = {
        "p0": [220.0, 20.0],
        "p1": [20.0, 20.0, 20.0, 220.0],
        "p2": [60.0, 60.0, 60.0, 20.0],
        "p3": [60.0, 220.0],
        "p4": [80.0, 180.0],
        "p5": [150.0],
        "p6": [60.0],
        "p7": [20.0, 220.0, 20.0, 80.0],
    }
    profile_by_task = {
        task_id: {"alpha": latencies}
        for task_id, latencies in latencies_by_task.items()
    }
    result = evaluate_offline_probe_clock(
        [
            _row(
                "band", 150.0, task_id="eval-a", source_trace="eval-a", command="alpha"
            ),
            _row(
                "short", 80.0, task_id="eval-b", source_trace="eval-b", command="alpha"
            ),
        ],
        profile_rows=_grouped_profile_rows(profile_by_task),
        kv_costs_ms=[100.0],
        guard_ms=0.0,
        inner_folds=4,
        command_field="command",
    )

    calibration = result["robust_calibration"]
    assert calibration["selected_guard_normalized"] is None
    assert calibration["eligible_early_decision_count"] == 9
    assert calibration["accepted_early_decision_count"] == 0
    for decision in result["decisions"]:
        assert decision["robust_trigger_ms"] == 80.0
        assert decision["offline_gated_robust_trigger_ms"] == 100.0
    policies = result["points"]["100.0"]["policies"]
    assert policies["robust_clock"]["delta_vs_deadline_ms"] == 40.0
    assert policies["offline_gated_robust_clock"]["delta_vs_deadline_ms"] == 0.0


def test_offline_probe_rejects_outer_task_overlap() -> None:
    overlapping_eval = [
        _row(
            "eval-overlap",
            150.0,
            task_id="profile-0",
            source_trace="eval-overlap",
        )
    ]

    with pytest.raises(ValueError, match="disjoint logical tasks"):
        evaluate_offline_probe_clock(
            overlapping_eval,
            profile_rows=_profile_rows(),
            kv_costs_ms=[100.0],
            guard_ms=0.0,
            inner_folds=4,
        )


def _probe_decision(
    sample_id: str,
    task_id: str,
    *,
    latency_ms: float,
    score: float,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "task_id": task_id,
        "latency_ms": latency_ms,
        "kv_cost_ms": 100.0,
        "threshold_ms": 100.0,
        "probe_candidate_trigger_ms": 0.0,
        "probe_margin_normalized": score,
    }


def _profile_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task_index in range(8):
        task_id = f"profile-{task_index}"
        for call_index, latency_ms in enumerate((80.0, 80.0, 150.0)):
            rows.append(
                _row(
                    f"{task_id}-{call_index}",
                    latency_ms,
                    task_id=task_id,
                    source_trace=f"trace-{task_id}",
                )
            )
    return rows


def _eval_rows() -> list[dict[str, object]]:
    return [
        _row("eval-short", 80.0, task_id="eval-a", source_trace="eval-a"),
        _row("eval-band", 150.0, task_id="eval-b", source_trace="eval-b"),
    ]


def _row(
    sample_id: str,
    latency_ms: float,
    *,
    task_id: str,
    source_trace: str,
    command: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_id": sample_id,
        "source_trace": source_trace,
        "task_id": task_id,
        "tool_name": "exec",
        "tool_ts_start": 0.0,
        "tool_ts_end": latency_ms / 1000.0,
        "latency_ms": latency_ms,
    }
    if command is not None:
        row["tool_args"] = {"command": command}
    return row


def _grouped_profile_rows(
    profile_by_task: dict[str, dict[str, list[float]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task_id, groups in profile_by_task.items():
        call_index = 0
        for command, latencies in groups.items():
            for latency_ms in latencies:
                rows.append(
                    _row(
                        f"{task_id}-{call_index}",
                        latency_ms,
                        task_id=task_id,
                        source_trace=f"trace-{task_id}",
                        command=command,
                    )
                )
                call_index += 1
    return rows
