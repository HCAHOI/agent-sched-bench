import pytest

from scripts.evaluation.evaluate_clause_resource_classes import CommandRow, Row
from scripts.evaluation.evaluate_resource_admission_oracle import _reservation
from tool_resource_eval.resource_admission import (
    AdmissionCommand,
    AdmissionProgram,
    simulate_admission,
)


def _program(task_id: str) -> AdmissionProgram:
    return AdmissionProgram(
        task_id,
        initial_delay_s=0.0,
        commands=(AdmissionCommand(f"{task_id}:0", 10.0, 4.0, 500.0, 0.0),),
        tail_s=0.0,
    )


def test_admission_packs_safe_commands_and_serializes_fixed_high() -> None:
    programs = [_program("a"), _program("b")]

    fixed = simulate_admission(
        programs,
        cpu_capacity=8.0,
        rss_capacity_mb=16_000.0,
        fixed_high=True,
    )
    packed = simulate_admission(
        programs,
        cpu_capacity=8.0,
        rss_capacity_mb=16_000.0,
        fixed_high=False,
    )

    assert fixed["makespan_s"] == 20.0
    assert packed["makespan_s"] == 10.0
    assert packed["overlapped_command_ids"] == ["b:0"]
    assert not packed["capacity_violation"]


def test_admission_rejects_zero_duration_overlap() -> None:
    program = AdmissionProgram(
        "zero",
        0.0,
        (AdmissionCommand("zero:0", 0.0, 1.0, 1.0, 0.0),),
        0.0,
    )

    with pytest.raises(ValueError, match="duration must be positive"):
        simulate_admission(
            [program],
            cpu_capacity=8.0,
            rss_capacity_mb=16_000.0,
            fixed_high=False,
        )


def test_explicit_underreservation_reports_modeled_capacity_exposure() -> None:
    programs = [_program("a"), _program("b")]
    reservations = {"a:0": (1.0, 1.0), "b:0": (1.0, 1.0)}

    result = simulate_admission(
        programs,
        cpu_capacity=4.0,
        rss_capacity_mb=16_000.0,
        fixed_high=False,
        requested_reservations=reservations,
    )

    assert result["modeled_capacity_exposure_events"] == 1
    assert result["modeled_capacity_exposure_command_ids"] == ["b:0"]
    assert result["max_modeled_cpu_demand_cores"] == 8.0


def test_reservation_sums_pipeline() -> None:
    clauses = (
        Row("task", "repo", 0, "left", ("left",), 100.0, 1.0, 100.0, 0.0, in_pipe=True, pipeline_position=0),
        Row("task", "repo", 0, "right", ("right",), 100.0, 2.0, 500.0, 0.0, in_pipe=True, pipeline_position=1),
    )
    row = CommandRow("task", "repo", 0, 0, "call_0", "left | right", 100.0, clauses)

    assert _reservation(row) == (3.0, 600.0, "observed_upper_bound")


def test_null_falls_back_only_for_its_resource() -> None:
    clause = Row(
        "task",
        "repo",
        0,
        "cmd",
        ("cmd",),
        1_000.0,
        None,
        750.0,
        0.0,
    )
    row = CommandRow("task", "repo", 0, 0, "call_0", "cmd", 1_000.0, (clause,))

    assert _reservation(row) == (8.0, 750.0, "cpu_null_target_fallback")


def test_unknown_structure_falls_back_to_full_host() -> None:
    clause = Row(
        "task",
        "repo",
        0,
        "cmd",
        ("cmd",),
        10.0,
        1.0,
        10.0,
        0.0,
        structure_known=False,
    )
    row = CommandRow("task", "repo", 0, 0, "call_0", "cmd", 10.0, (clause,))

    assert _reservation(row) == (8.0, 16_000.0, "structure_full_fallback")
