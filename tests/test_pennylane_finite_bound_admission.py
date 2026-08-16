from scripts.evaluation import evaluate_pennylane_finite_bound_admission as evaluation
from scripts.evaluation.evaluate_clause_resource_classes import CommandRow
from scripts.evaluation.evaluate_pennylane_causal_joint_admission import (
    Program,
    Segment,
)
from scripts.evaluation.evaluate_pennylane_finite_bound_admission import (
    _fit_bounds,
    _finite_class_bounds,
    _finite_requests,
    _gate,
)


def test_finite_bounds_keep_edges_and_monotonicity() -> None:
    bounds = _finite_class_bounds(
        {0: [1.5], 1: [7.0], 2: [5.0]},
        (2.0, 4.0, 43.0),
        43.0,
    )

    assert bounds == (1.5, 7.0, 7.0)


def test_unsupported_high_class_fails_back_to_capacity() -> None:
    bounds = _finite_class_bounds(
        {0: [1.0], 1: [3.0], 2: []},
        (500.0, 2_000.0, 80_000.0),
        80_000.0,
    )

    assert bounds == (1.0, 3.0, 80_000.0)


def test_fit_bounds_use_only_observed_labels(monkeypatch) -> None:
    row = CommandRow("task-a", "repo", 0, 0, "call-a", "cmd", 1_000.0, ())
    program = Program(
        "task-a",
        (Segment(1.0, "exec", "call-a", 1.5, 700.0),),
    )

    monkeypatch.setattr(
        evaluation,
        "command_resource_bucket_label",
        lambda _row, target: (
            (0, "observed_composed_low")
            if target == "peak_cpu_cores"
            else (0, "short_null_composed_low")
        ),
    )

    bounds, evidence = _fit_bounds([row], [program])

    assert bounds["peak_cpu_cores"] == (1.5, 4.0, 43.0)
    assert evidence["peak_cpu_cores"]["support_commands"] == [1, 0, 0]
    assert evidence["sampled_peak_rss_mb"]["support_commands"] == [0, 0, 0]


def test_missing_prediction_falls_back_per_target() -> None:
    hard = {("task-a", "call-a"): (4.0, 2_000.0)}
    bounds = {
        "peak_cpu_cores": (1.0, 3.0, 10.0),
        "sampled_peak_rss_mb": (100.0, 1_000.0, 30_000.0),
    }

    cpu_missing = _finite_requests(
        hard,
        {
            "cpu_prediction_unavailable_command_ids": ["task-a:call-a"],
            "rss_prediction_unavailable_command_ids": [],
        },
        bounds,
    )
    rss_missing = _finite_requests(
        hard,
        {
            "cpu_prediction_unavailable_command_ids": [],
            "rss_prediction_unavailable_command_ids": ["task-a:call-a"],
        },
        bounds,
    )

    assert cpu_missing[("task-a", "call-a")] == (43.0, 1_000.0)
    assert rss_missing[("task-a", "call-a")] == (3.0, 80_000.0)


def test_finite_gate_requires_safe_baseline() -> None:
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

    gate = _gate({"finite_fit_feedback": candidate, "serial_tool": baseline})

    assert gate["status"] == "no_go"
    assert gate["mean_completion_reduction_vs_serial_tool"] is None
