import pytest

from scripts.evaluation.evaluate_cpu_work_admission import (
    _clause_cpu_work,
    _cpu_floor_programs,
)
from scripts.evaluation.evaluate_cpu_throughput_oracle import _bucket_cpu
from scripts.evaluation.evaluate_clause_resource_classes import CommandRow, Row
from scripts.evaluation.evaluate_resource_admission_oracle import _reservation
from tool_resource_eval.resource_admission import (
    AdmissionCommand,
    AdmissionProgram,
    simulate_admission,
    simulate_burstable_admission,
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


def test_burstable_requests_share_spare_cpu_and_conserve_work() -> None:
    programs = [
        AdmissionProgram(
            task_id,
            0.0,
            (AdmissionCommand(f"{task_id}:0", 2.0, 4.0, 1.0, 0.0),),
            0.0,
        )
        for task_id in ("a", "b", "c")
    ]
    requests = {f"{task_id}:0": (2.0, 1.0) for task_id in ("a", "b", "c")}
    work = {f"{task_id}:0": 8.0 for task_id in ("a", "b", "c")}
    demand = {f"{task_id}:0": 4.0 for task_id in ("a", "b", "c")}

    result = simulate_burstable_admission(
        programs,
        cpu_capacity=8.0,
        rss_capacity_mb=16_000.0,
        requested_reservations=requests,
        cpu_work_core_s=work,
        max_cpu_cores=demand,
    )

    assert result["makespan_s"] == pytest.approx(3.0)
    assert result["total_cpu_work_core_s"] == 24.0
    assert result["served_cpu_work_core_s"] == pytest.approx(24.0)
    assert result["modeled_capacity_exposure_events"] == 1
    assert result["contended_command_ids"] == ["a:0", "b:0", "c:0"]


def test_burstable_throughput_requests_reproduce_recorded_duration() -> None:
    programs = [
        AdmissionProgram(
            task_id,
            0.0,
            (AdmissionCommand(f"{task_id}:0", 2.0, 4.0, 1.0, 0.0),),
            0.0,
        )
        for task_id in ("a", "b", "c")
    ]
    requests = {f"{task_id}:0": (4.0, 1.0) for task_id in ("a", "b", "c")}
    work = {f"{task_id}:0": 8.0 for task_id in ("a", "b", "c")}
    demand = {f"{task_id}:0": 4.0 for task_id in ("a", "b", "c")}

    result = simulate_burstable_admission(
        programs,
        cpu_capacity=8.0,
        rss_capacity_mb=16_000.0,
        requested_reservations=requests,
        cpu_work_core_s=work,
        max_cpu_cores=demand,
    )

    assert result["makespan_s"] == 4.0
    assert result["added_service_s"] == 0.0
    assert result["contended_command_ids"] == []


def test_burstable_cpu_redistributes_share_after_demand_cap() -> None:
    programs = [
        AdmissionProgram(
            "capped",
            0.0,
            (AdmissionCommand("capped:0", 2.0, 2.0, 1.0, 0.0),),
            0.0,
        ),
        AdmissionProgram(
            "elastic",
            0.0,
            (AdmissionCommand("elastic:0", 2.0, 6.0, 1.0, 0.0),),
            0.0,
        ),
    ]

    result = simulate_burstable_admission(
        programs,
        cpu_capacity=8.0,
        rss_capacity_mb=16_000.0,
        requested_reservations={"capped:0": (2.0, 1.0), "elastic:0": (1.0, 1.0)},
        cpu_work_core_s={"capped:0": 4.0, "elastic:0": 12.0},
        max_cpu_cores={"capped:0": 2.0, "elastic:0": 6.0},
    )

    assert result["makespan_s"] == 2.0
    assert result["served_cpu_work_core_s"] == 16.0


def test_burstable_exposure_excludes_work_completed_before_wall_floor() -> None:
    programs = [
        AdmissionProgram(
            "waiting",
            0.0,
            (AdmissionCommand("waiting:0", 10.0, 2.0, 1.0, 0.0),),
            0.0,
        ),
        AdmissionProgram(
            "later",
            2.0,
            (AdmissionCommand("later:0", 2.0, 4.0, 1.0, 0.0),),
            0.0,
        ),
    ]

    result = simulate_burstable_admission(
        programs,
        cpu_capacity=4.0,
        rss_capacity_mb=16_000.0,
        requested_reservations={"waiting:0": (2.0, 1.0), "later:0": (2.0, 1.0)},
        cpu_work_core_s={"waiting:0": 2.0, "later:0": 8.0},
        max_cpu_cores={"waiting:0": 2.0, "later:0": 4.0},
    )

    assert result["max_modeled_cpu_demand_cores"] == 4.0
    assert result["modeled_capacity_exposure_events"] == 0


def test_reservation_sums_pipeline() -> None:
    clauses = (
        Row(
            "task",
            "repo",
            0,
            "left",
            ("left",),
            100.0,
            1.0,
            100.0,
            0.0,
            in_pipe=True,
            pipeline_position=0,
        ),
        Row(
            "task",
            "repo",
            0,
            "right",
            ("right",),
            100.0,
            2.0,
            500.0,
            0.0,
            in_pipe=True,
            pipeline_position=1,
        ),
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


def test_cpu_work_floor_changes_only_physically_impossible_duration() -> None:
    programs = {
        "task": AdmissionProgram(
            "task",
            0.0,
            (
                AdmissionCommand("task:cpu", 2.0, 8.0, 100.0, 1.0),
                AdmissionCommand("task:wait", 5.0, 8.0, 100.0, 0.0),
                AdmissionCommand("task:missing", 3.0, 8.0, 100.0, 0.0),
            ),
            0.0,
        )
    }
    adjusted, summary = _cpu_floor_programs(
        programs,
        {
            "task:cpu": (2.0, 100.0),
            "task:wait": (2.0, 100.0),
            "task:missing": (2.0, 100.0),
        },
        {"task:cpu": 8.0, "task:wait": 1.0},
    )

    assert [command.duration_s for command in adjusted["task"].commands] == [
        4.0,
        5.0,
        3.0,
    ]
    assert summary["dilated_commands"] == 1
    assert summary["added_service_s"] == 2.0


def test_clause_cpu_work_rejects_duplicate_identity() -> None:
    clause = {
        "bin": "pytest",
        "argv": ["pytest"],
        "ts_start": 1.0,
        "ts_end": 2.0,
        "pipeline_position": -1,
        "in_loop": False,
        "in_pipe": False,
        "in_subst": False,
        "cpu_ns_cumulative": 500_000_000,
    }

    assert _clause_cpu_work([clause]) == 0.5
    with pytest.raises(ValueError, match="duplicate clause CPU-work identity"):
        _clause_cpu_work([clause, dict(clause)])


def test_cpu_throughput_target_uses_existing_request_classes() -> None:
    assert [_bucket_cpu(value) for value in (0.0, 2.0, 2.01, 4.0, 4.01, 8.1)] == [
        2.0,
        2.0,
        4.0,
        4.0,
        8.0,
        8.0,
    ]
