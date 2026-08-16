import numpy as np

from scripts.evaluation import evaluate_pennylane_causal_joint_admission as evaluation
from scripts.evaluation.evaluate_pennylane_causal_joint_admission import (
    Program,
    Segment,
    _gate,
    _segments,
    simulate,
)
from scripts.evaluation.evaluate_pennylane_joint_phase_packing import TaskProfile


def _programs() -> list[Program]:
    return [
        Program(
            "task-a",
            (
                Segment(1.0, "exec", "a", 1.0, 100.0),
                Segment(4.0, "exec", "a", 1.0, 100.0),
            ),
        ),
        Program("task-b", (Segment(2.0, "exec", "b", 2.0, 100.0),)),
    ]


def test_feedback_releases_capacity_only_after_observation() -> None:
    reservations = {
        ("task-a", "a"): (43.0, 500.0),
        ("task-b", "b"): (2.0, 500.0),
    }

    static = simulate(_programs(), "task_aware_static", reservations)
    feedback = simulate(_programs(), "task_aware_feedback", reservations)
    serial = simulate(_programs(), "serial_tool", reservations)

    assert static["makespan_s"] == serial["makespan_s"] == 7.0
    assert feedback["makespan_s"] == 5.0
    assert feedback["mean_task_completion_s"] == 4.0
    assert feedback["tasks_with_overlapped_exec"] == 2
    assert feedback["capacity_violation_s"] == {"gpu": 0.0, "cpu": 0.0, "rss": 0.0}


def test_missing_prediction_falls_back_to_serial_high_page() -> None:
    result = simulate(_programs(), "task_aware_feedback", {})

    assert result["completed"] is True
    assert result["makespan_s"] == 7.0
    assert result["tasks_with_overlapped_exec"] == 0
    assert result["deadlock"] is None


def test_first_segment_rss_is_not_visible_before_it_runs() -> None:
    programs = [
        Program("task-a", (Segment(1.0, "exec", "a", 1.0, 79_900.0),)),
        Program("task-b", (Segment(1.0, "exec", "b", 1.0, 200.0),)),
    ]
    reservations = {
        ("task-a", "a"): (2.0, 500.0),
        ("task-b", "b"): (2.0, 500.0),
    }

    result = simulate(programs, "task_aware_feedback", reservations)

    assert result["completed"] is True
    assert result["capacity_violation_s"]["rss"] == 1.0


def test_segmentation_keeps_equal_sample_boundaries(monkeypatch) -> None:
    action = {
        "action_type": "tool_exec",
        "action_id": "tool-a",
        "ts_start": 0.0,
        "ts_end": 4.0,
        "data": {
            "tool_name": "exec",
            "tool_call_id": "a",
            "resource_timeline": {"samples": [{"dt_s": 0.5}]},
        },
    }
    monkeypatch.setattr(
        evaluation, "_actions_and_origin", lambda _task, _bins: ([action], 0.0)
    )
    profile = TaskProfile(
        "task-a",
        np.zeros(2),
        np.ones(2),
        np.full(2, 100.0),
    )

    program = _segments(profile)

    assert [segment.duration_s for segment in program.segments] == [2.0, 2.0]


def test_unsafe_serial_baseline_cannot_support_go() -> None:
    candidate = {
        "completed": True,
        "mean_task_completion_s": 9.0,
        "makespan_s": 9.0,
        "capacity_violation_s": {"gpu": 0.0, "cpu": 0.0, "rss": 0.0},
        "tasks_with_overlapped_exec": 20,
    }
    baseline = {
        "completed": True,
        "mean_task_completion_s": 10.0,
        "makespan_s": 10.0,
        "capacity_violation_s": {"gpu": 0.0, "cpu": 1.0, "rss": 0.0},
    }

    gate = _gate({"task_aware_feedback": candidate, "serial_tool": baseline})

    assert gate["status"] == "no_go"
    assert gate["mean_completion_reduction_vs_serial_tool"] is None
    assert gate["checks"]["serial_tool_completed_and_safe"] is False
