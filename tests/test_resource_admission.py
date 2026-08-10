import json

import pytest

import scripts.evaluation.evaluate_cpu_idle_rss_safety as idle_rss_safety

from scripts.evaluation.evaluate_cpu_idle_backfill_oracle import (
    ACTION_MINIMUM_REDUCTION,
    ARMS as IDLE_BACKFILL_ARMS,
    PROTOCOL as IDLE_BACKFILL_PROTOCOL,
    SELECTION_MINIMUM_REDUCTION,
    VERSION as IDLE_BACKFILL_VERSION,
    _committed_input_paths as _idle_backfill_committed_input_paths,
    _gate as _idle_backfill_gate,
)
from scripts.evaluation.evaluate_cpu_idle_rss_safety import (
    ARMS as IDLE_RSS_ARMS,
    PROTOCOL as IDLE_RSS_PROTOCOL,
    VERSION as IDLE_RSS_VERSION,
    _arm_specs as _idle_rss_arm_specs,
    _gate as _idle_rss_gate,
)
from scripts.evaluation.evaluate_cpu_feedback_borrowing import (
    PROTOCOL as BORROWING_PROTOCOL,
    VERSION as BORROWING_VERSION,
    WORK_CONSERVING_CPU,
)
from scripts.evaluation.evaluate_cpu_work_admission import (
    _clause_cpu_work,
    _cpu_floor_programs,
)
from scripts.evaluation.evaluate_cpu_throughput_oracle import _bucket_cpu
from scripts.evaluation.evaluate_pennylane_pairwise_cpu_backfill import (
    _bucket_upper,
    _command_signatures,
    _profile_envelope,
)
from scripts.evaluation.evaluate_cpu_feedback_admission import (
    _assert_frozen_protocol,
    _file_identities,
    _fixed_cpu_requests,
    _gate,
    _mean_relative_reduction,
    _service_inflation,
)
from scripts.evaluation.evaluate_clause_resource_classes import CommandRow, Row
from scripts.evaluation.evaluate_resource_admission_oracle import _reservation
from tool_resource_eval.resource_admission import (
    AdmissionCommand,
    AdmissionProgram,
    simulate_admission,
    simulate_burstable_admission,
    simulate_feedback_admission,
    simulate_idle_backfill,
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


def test_feedback_shrink_admits_a_waiting_command() -> None:
    programs = [
        AdmissionProgram(
            "a",
            0.0,
            (AdmissionCommand("a", 3.0, 1.0, 100.0, 0.0),),
            0.0,
        ),
        AdmissionProgram(
            "b",
            0.0,
            (AdmissionCommand("b", 1.0, 1.0, 100.0, 0.0),),
            0.0,
        ),
    ]
    kwargs = {
        "cpu_capacity": 8.0,
        "rss_capacity_mb": 1_000.0,
        "requested_reservations": {"a": (8.0, 100.0), "b": (4.0, 100.0)},
        "cpu_work_profiles": {
            "a": ((0.5, 0.5),) * 6,
        },
        "sample_interval_s": 0.5,
        "update_delay_s": 0.1,
        "cpu_pages": (2.0, 4.0, 8.0),
    }

    static = simulate_feedback_admission(programs, **kwargs, feedback=False)
    feedback = simulate_feedback_admission(programs, **kwargs, feedback=True)

    assert static["start_s_by_command"]["b"] == 3.0
    assert feedback["start_s_by_command"]["b"] == pytest.approx(0.6)
    assert feedback["mean_task_completion_s"] < static["mean_task_completion_s"]
    assert feedback["total_command_queue_s"] < static["total_command_queue_s"]
    assert feedback["reservation_shrinks"] == 1


def test_feedback_denies_expansion_that_exceeds_capacity() -> None:
    programs = [
        AdmissionProgram(
            task_id,
            0.0,
            (AdmissionCommand(task_id, 1.0, 6.0, 100.0, 0.0),),
            0.0,
        )
        for task_id in ("a", "b")
    ]

    result = simulate_feedback_admission(
        programs,
        cpu_capacity=8.0,
        rss_capacity_mb=1_000.0,
        requested_reservations={"a": (4.0, 100.0), "b": (4.0, 100.0)},
        cpu_work_profiles={
            "a": ((0.5, 3.0), (0.5, 3.0)),
            "b": ((0.5, 3.0), (0.5, 3.0)),
        },
        feedback=True,
        sample_interval_s=0.5,
        update_delay_s=0.1,
        cpu_pages=(2.0, 4.0, 8.0),
    )

    assert result["denied_expansions"] >= 2
    assert result["reservation_expansions"] == 0
    assert not result["capacity_violation"]


def test_feedback_without_profile_matches_static_service() -> None:
    programs = [_program("a"), _program("b")]
    kwargs = {
        "cpu_capacity": 8.0,
        "rss_capacity_mb": 16_000.0,
        "requested_reservations": {"a:0": (4.0, 500.0), "b:0": (4.0, 500.0)},
        "cpu_work_profiles": {},
        "sample_interval_s": 0.5,
        "update_delay_s": 0.1,
        "cpu_pages": (2.0, 4.0, 8.0),
    }

    static = simulate_feedback_admission(programs, **kwargs, feedback=False)
    feedback = simulate_feedback_admission(programs, **kwargs, feedback=True)

    assert feedback == static


def test_work_conserving_feedback_borrows_above_logical_reservation() -> None:
    program = AdmissionProgram(
        "task",
        0.0,
        (AdmissionCommand("cmd", 1.0, 8.0, 100.0, 0.0),),
        0.0,
    )
    kwargs = {
        "cpu_capacity": 8.0,
        "rss_capacity_mb": 1_000.0,
        "requested_reservations": {"cmd": (2.0, 100.0)},
        "cpu_work_profiles": {"cmd": ((1.0, 8.0),)},
        "feedback": False,
        "sample_interval_s": 0.5,
        "update_delay_s": 0.1,
        "cpu_pages": (2.0, 4.0, 8.0),
    }

    hard = simulate_feedback_admission([program], **kwargs)
    borrowing = simulate_feedback_admission(
        [program], **kwargs, work_conserving_cpu=True
    )

    assert hard["total_command_service_s"] == 4.0
    assert borrowing["total_command_service_s"] == 1.0
    assert borrowing["served_cpu_work_core_s"] == 8.0


def test_work_conserving_feedback_charges_contention_and_denied_expansion() -> None:
    programs = [
        AdmissionProgram(
            task_id,
            0.0,
            (AdmissionCommand(task_id, 1.0, 8.0, 100.0, 0.0),),
            0.0,
        )
        for task_id in ("a", "b")
    ]

    result = simulate_feedback_admission(
        programs,
        cpu_capacity=8.0,
        rss_capacity_mb=1_000.0,
        requested_reservations={"a": (4.0, 100.0), "b": (4.0, 100.0)},
        cpu_work_profiles={
            "a": ((1.0, 8.0),),
            "b": ((1.0, 8.0),),
        },
        feedback=True,
        sample_interval_s=0.5,
        update_delay_s=0.1,
        cpu_pages=(2.0, 4.0, 8.0),
        work_conserving_cpu=True,
    )

    assert result["makespan_s"] == 2.0
    assert result["total_command_service_s"] == 4.0
    assert result["served_cpu_work_core_s"] == 16.0
    assert result["denied_expansions"] >= 2
    assert not result["capacity_violation"]


def test_work_conserving_cpu_uses_equal_weights_not_logical_pages() -> None:
    specifications = {
        "a": (4.0, 4.0),
        "b": (2.0, 8.0),
        "c": (2.0, 8.0),
    }
    programs = [
        AdmissionProgram(
            task_id,
            0.0,
            (AdmissionCommand(task_id, 1.0, demand, 100.0, 0.0),),
            0.0,
        )
        for task_id, (_request, demand) in specifications.items()
    ]

    result = simulate_feedback_admission(
        programs,
        cpu_capacity=8.0,
        rss_capacity_mb=1_000.0,
        requested_reservations={
            task_id: (request, 100.0)
            for task_id, (request, _demand) in specifications.items()
        },
        cpu_work_profiles={
            task_id: ((1.0, demand),)
            for task_id, (_request, demand) in specifications.items()
        },
        feedback=True,
        sample_interval_s=0.5,
        update_delay_s=0.1,
        cpu_pages=(2.0, 4.0, 8.0),
        work_conserving_cpu=True,
    )

    assert result["service_s_by_command"]["a"] == pytest.approx(1.5)
    assert result["service_s_by_command"]["b"] == pytest.approx(2.5)
    assert result["service_s_by_command"]["c"] == pytest.approx(2.5)
    assert result["served_cpu_work_core_s"] == 20.0


def test_work_conserving_feedback_requires_every_cpu_profile() -> None:
    with pytest.raises(ValueError, match="complete CPU work profiles"):
        simulate_feedback_admission(
            [_program("task")],
            cpu_capacity=8.0,
            rss_capacity_mb=16_000.0,
            requested_reservations={"task:0": (2.0, 500.0)},
            cpu_work_profiles={},
            feedback=True,
            sample_interval_s=0.5,
            update_delay_s=0.1,
            cpu_pages=(2.0, 4.0, 8.0),
            work_conserving_cpu=True,
        )


def test_idle_backfill_uses_only_cpu_left_by_normal_command() -> None:
    programs = [
        AdmissionProgram(
            "normal",
            0.0,
            (AdmissionCommand("normal", 2.0, 4.0, 300.0, 0.0),),
            0.0,
        ),
        AdmissionProgram(
            "speculative",
            0.0,
            (AdmissionCommand("speculative", 1.0, 4.0, 300.0, 0.0),),
            0.0,
        ),
    ]

    result = simulate_idle_backfill(
        programs,
        cpu_capacity=8.0,
        rss_capacity_mb=1_000.0,
        cpu_work_profiles={
            "normal": ((2.0, 8.0),),
            "speculative": ((1.0, 4.0),),
        },
        speculative_eligible_command_ids={"normal", "speculative"},
        selection="fcfs",
    )

    assert result["makespan_s"] == 2.0
    assert result["total_command_service_s"] == 3.0
    assert result["speculative_cpu_work_core_s"] == 4.0
    assert result["speculative_completions"] == 1
    assert not result["physical_capacity_violation"]


def test_idle_backfill_skips_pairwise_cpu_incompatible_candidate() -> None:
    programs = [
        AdmissionProgram(
            task_id,
            0.0,
            (AdmissionCommand(task_id, 2.0, 4.0, 100.0, 0.0),),
            0.0,
        )
        for task_id in ("normal", "too_large", "fits")
    ]
    profiles = {task_id: ((2.0, 4.0),) for task_id in ("normal", "too_large", "fits")}

    result = simulate_idle_backfill(
        programs,
        cpu_capacity=8.0,
        rss_capacity_mb=1_000.0,
        cpu_work_profiles=profiles,
        speculative_eligible_command_ids=set(profiles),
        pairwise_cpu_demands={"normal": 6.0, "too_large": 3.0, "fits": 2.0},
        selection="fcfs",
    )

    assert result["speculative_start_ids"][0] == "fits"


def test_idle_backfill_uses_remaining_phase_compatibility() -> None:
    programs = [
        AdmissionProgram(
            task_id,
            0.0,
            (AdmissionCommand(task_id, 2.0, 8.0, 100.0, 0.0),),
            0.0,
        )
        for task_id in ("normal", "candidate")
    ]
    staggered = {
        "normal": ((1.0, 8.0), (1.0, 0.0)),
        "candidate": ((1.0, 0.0), (1.0, 8.0)),
    }

    admitted = simulate_idle_backfill(
        programs,
        cpu_capacity=8.0,
        rss_capacity_mb=1_000.0,
        cpu_work_profiles=staggered,
        speculative_eligible_command_ids=set(staggered),
        require_pairwise_profile_compatibility=True,
        selection="fcfs",
    )
    rejected = simulate_idle_backfill(
        programs,
        cpu_capacity=8.0,
        rss_capacity_mb=1_000.0,
        cpu_work_profiles={
            "normal": ((1.0, 8.0), (1.0, 0.0)),
            "candidate": ((1.0, 8.0), (1.0, 0.0)),
        },
        speculative_eligible_command_ids={"normal", "candidate"},
        require_pairwise_profile_compatibility=True,
        selection="fcfs",
    )
    predicted = simulate_idle_backfill(
        programs,
        cpu_capacity=8.0,
        rss_capacity_mb=1_000.0,
        cpu_work_profiles=staggered,
        speculative_eligible_command_ids={"candidate"},
        pairwise_candidate_profiles={
            "candidate": ((1.0, 8.0), (1.0, 0.0)),
        },
        require_pairwise_profile_compatibility=True,
        selection="fcfs",
    )
    two_sided = simulate_idle_backfill(
        programs,
        cpu_capacity=8.0,
        rss_capacity_mb=1_000.0,
        cpu_work_profiles=staggered,
        speculative_eligible_command_ids={"candidate"},
        pairwise_candidate_profiles={"candidate": staggered["candidate"]},
        pairwise_foreground_profiles={
            "normal": ((1.0, 0.0), (1.0, 8.0)),
        },
        require_pairwise_profile_compatibility=True,
        selection="fcfs",
    )

    assert admitted["speculative_start_ids"] == ["candidate"]
    assert admitted["start_s_by_command"]["candidate"] == 0.0
    assert admitted["total_command_service_s"] == 4.0
    assert rejected["start_s_by_command"]["candidate"] == 1.0
    assert predicted["start_s_by_command"]["candidate"] == 1.0
    assert two_sided["start_s_by_command"]["candidate"] == 1.0


def test_idle_backfill_waits_for_causal_foreground_sample() -> None:
    programs = [
        AdmissionProgram(
            "normal",
            0.0,
            (AdmissionCommand("normal", 2.0, 4.0, 100.0, 0.0),),
            0.0,
        ),
        AdmissionProgram(
            "candidate",
            0.0,
            (AdmissionCommand("candidate", 1.0, 4.0, 100.0, 0.0),),
            0.0,
        ),
    ]
    profiles = {
        "normal": ((0.5, 2.0),) * 4,
        "candidate": ((0.5, 2.0),) * 2,
    }

    result = simulate_idle_backfill(
        programs,
        cpu_capacity=8.0,
        rss_capacity_mb=1_000.0,
        cpu_work_profiles=profiles,
        speculative_eligible_command_ids={"candidate"},
        pairwise_cpu_demands={"normal": 4.0, "candidate": 4.0},
        reactive_foreground_observation_delay_s=0.14132007875,
        selection="fcfs",
    )

    assert result["start_s_by_command"]["candidate"] == 1.0
    assert result["total_command_service_s"] == 3.0


def test_idle_backfill_promotion_preserves_partial_cpu_work() -> None:
    programs = [
        AdmissionProgram(
            "normal",
            0.0,
            (AdmissionCommand("normal", 1.0, 4.0, 300.0, 0.0),),
            0.0,
        ),
        AdmissionProgram(
            "speculative",
            0.0,
            (AdmissionCommand("speculative", 2.0, 8.0, 300.0, 0.0),),
            0.0,
        ),
    ]

    result = simulate_idle_backfill(
        programs,
        cpu_capacity=8.0,
        rss_capacity_mb=1_000.0,
        cpu_work_profiles={
            "normal": ((1.0, 4.0),),
            "speculative": ((2.0, 16.0),),
        },
        speculative_eligible_command_ids={"normal", "speculative"},
        selection="fcfs",
    )

    assert result["makespan_s"] == 2.5
    assert result["promotions"] == 1
    assert result["speculative_cpu_work_core_s"] == 4.0
    assert result["served_cpu_work_core_s"] == 20.0


def test_idle_backfill_shortest_selection_differs_from_fcfs() -> None:
    programs = [
        AdmissionProgram(
            task_id,
            0.0,
            (AdmissionCommand(task_id, duration, 4.0, 100.0, 0.0),),
            0.0,
        )
        for task_id, duration in (("normal", 4.0), ("older", 3.0), ("short", 1.0))
    ]
    profiles = {
        task_id: ((duration, 4.0 * duration),)
        for task_id, duration in (("normal", 4.0), ("older", 3.0), ("short", 1.0))
    }

    fcfs = simulate_idle_backfill(
        programs,
        cpu_capacity=8.0,
        rss_capacity_mb=1_000.0,
        cpu_work_profiles=profiles,
        speculative_eligible_command_ids=set(profiles),
        selection="fcfs",
    )
    shortest = simulate_idle_backfill(
        programs,
        cpu_capacity=8.0,
        rss_capacity_mb=1_000.0,
        cpu_work_profiles=profiles,
        speculative_eligible_command_ids=set(profiles),
        selection="shortest",
    )

    assert fcfs["speculative_start_ids"][0] == "older"
    assert shortest["speculative_start_ids"][0] == "short"


def test_idle_backfill_rejects_speculation_that_does_not_fit_rss() -> None:
    programs = [
        AdmissionProgram(
            task_id,
            0.0,
            (AdmissionCommand(task_id, 1.0, 4.0, rss, 0.0),),
            0.0,
        )
        for task_id, rss in (("normal", 900.0), ("waiting", 200.0))
    ]

    result = simulate_idle_backfill(
        programs,
        cpu_capacity=8.0,
        rss_capacity_mb=1_000.0,
        cpu_work_profiles={
            "normal": ((1.0, 4.0),),
            "waiting": ((1.0, 4.0),),
        },
        speculative_eligible_command_ids={"normal", "waiting"},
        selection="fcfs",
    )

    assert result["start_s_by_command"] == {"normal": 0.0, "waiting": 1.0}
    assert result["speculative_starts"] == 0


def test_idle_backfill_requires_every_cpu_profile() -> None:
    with pytest.raises(ValueError, match="complete CPU work profiles"):
        simulate_idle_backfill(
            [_program("task")],
            cpu_capacity=8.0,
            rss_capacity_mb=16_000.0,
            cpu_work_profiles={},
            speculative_eligible_command_ids=set(),
            selection="serial",
        )


def test_idle_backfill_excludes_commands_without_rss_evidence() -> None:
    programs = [
        AdmissionProgram(
            task_id,
            0.0,
            (AdmissionCommand(task_id, 1.0, 4.0, rss, 0.0),),
            0.0,
        )
        for task_id, rss in (("normal", 100.0), ("unknown", 0.0))
    ]

    result = simulate_idle_backfill(
        programs,
        cpu_capacity=8.0,
        rss_capacity_mb=1_000.0,
        cpu_work_profiles={
            "normal": ((1.0, 4.0),),
            "unknown": ((1.0, 4.0),),
        },
        speculative_eligible_command_ids={"normal"},
        selection="fcfs",
    )

    assert result["speculative_starts"] == 0


def test_idle_backfill_separates_predicted_fit_from_source_rss_exposure() -> None:
    programs = [
        AdmissionProgram(
            task_id,
            0.0,
            (AdmissionCommand(task_id, 1.0, 4.0, 900.0, 0.0),),
            0.0,
        )
        for task_id in ("normal", "speculative")
    ]

    result = simulate_idle_backfill(
        programs,
        cpu_capacity=8.0,
        rss_capacity_mb=1_000.0,
        cpu_work_profiles={
            "normal": ((1.0, 4.0),),
            "speculative": ((1.0, 4.0),),
        },
        speculative_eligible_command_ids={"normal", "speculative"},
        rss_reservations={"normal": 500.0, "speculative": 500.0},
        selection="fcfs",
    )

    assert result["speculative_starts"] == 1
    assert not result["capacity_violation"]
    assert result["modeled_capacity_exposure_events"] == 1
    assert result["max_modeled_rss_demand_mb"] == 1_800.0


def test_idle_backfill_reports_overlap_involving_unverified_rss() -> None:
    programs = [
        AdmissionProgram(
            task_id,
            0.0,
            (AdmissionCommand(task_id, 1.0, 4.0, 500.0, 0.0),),
            0.0,
        )
        for task_id in ("normal", "speculative")
    ]

    result = simulate_idle_backfill(
        programs,
        cpu_capacity=8.0,
        rss_capacity_mb=1_000.0,
        cpu_work_profiles={
            "normal": ((1.0, 4.0),),
            "speculative": ((1.0, 4.0),),
        },
        speculative_eligible_command_ids={"normal", "speculative"},
        rss_unverified_command_ids={"normal"},
        selection="fcfs",
    )

    assert result["rss_unverified_overlap_events"] == 1
    assert result["rss_unverified_overlap_command_ids"] == [
        "normal",
        "speculative",
    ]


def test_idle_backfill_guard_covers_split_and_profile_definitions() -> None:
    names = {path.name for path in _idle_backfill_committed_input_paths()}

    assert "evaluate_kv_prediction_actionability.py" in names
    assert "early_cpu_reservation.py" in names


def test_idle_backfill_evaluator_locks_arms_and_protocol() -> None:
    assert IDLE_BACKFILL_VERSION == "cpu-idle-backfill-oracle-v1"
    assert IDLE_BACKFILL_PROTOCOL.name == (
        "cpu-idle-speculative-backfill-protocol.md"
    )
    assert IDLE_BACKFILL_ARMS == ("serial8", "fcfs_idle", "oracle_idle")
    assert ACTION_MINIMUM_REDUCTION == 0.05
    assert SELECTION_MINIMUM_REDUCTION == 0.10


def test_idle_backfill_gate_requires_every_frozen_condition() -> None:
    passing = _idle_backfill_gate(
        reduction=0.10,
        minimum_reduction=0.10,
        bootstrap_high=-1.0,
        service_inflation=0.05,
        makespan_regression=0.01,
        violation=False,
    )
    assert passing["go"]

    for field, value in (
        ("reduction", 0.099),
        ("bootstrap_high", 0.0),
        ("service_inflation", 0.051),
        ("makespan_regression", 0.011),
        ("violation", True),
    ):
        values = {
            "reduction": 0.10,
            "minimum_reduction": 0.10,
            "bootstrap_high": -1.0,
            "service_inflation": 0.05,
            "makespan_regression": 0.01,
            "violation": False,
        }
        values[field] = value
        assert not _idle_backfill_gate(**values)["go"]


def test_idle_rss_evaluator_locks_arms_and_protocol() -> None:
    assert IDLE_RSS_VERSION == "cpu-idle-rss-safety-v1"
    assert IDLE_RSS_PROTOCOL.name == "cpu-idle-rss-safety-protocol.md"
    assert IDLE_RSS_ARMS == (
        "serial8",
        "oracle_rss_fcfs",
        "clause_kb_rss_fcfs",
        "task_aware_rss_fcfs",
    )
    command_ids = {"observed", "unavailable"}
    oracle_rss = {"observed": 500.0, "unavailable": 16_000.0}
    arm_specs = _idle_rss_arm_specs(
        command_ids,
        oracle_rss,
        oracle_rss,
        oracle_rss,
    )
    assert arm_specs["oracle_rss_fcfs"][1] == command_ids
    assert arm_specs["oracle_rss_fcfs"][2]["unavailable"] == 16_000.0


def test_short_null_policy_imputes_only_strictly_short_insufficient_samples(
    tmp_path,
) -> None:
    clauses = (
        Row("task", "repo", 0, "measured", ("measured",), 100.0, 1.0, 700.0, 0.0),
        Row("task", "repo", 0, "short", ("short",), 499.0, None, None, 0.0),
        Row("task", "repo", 0, "boundary", ("boundary",), 500.0, None, None, 0.0),
        Row("task", "repo", 0, "missing", ("missing",), 100.0, None, None, 0.0),
    )
    command_rows = {
        ("task", "call"): CommandRow(
            "task", "repo", 0, 0, "call", "commands", 500.0, clauses
        )
    }
    attempt = tmp_path / "attempt_1"
    attempt.mkdir()
    trace = attempt / "trace.jsonl"
    trace.write_text("", encoding="utf-8")
    (attempt / "resource_observations.json").write_text(
        json.dumps(
            {
                "calls": [
                    {
                        "tool_call_id": "call",
                        "clauses": [
                            {
                                "bin": "measured",
                                "argv": ["measured"],
                                "availability": {"memory": "ok"},
                            },
                            {
                                "bin": "short",
                                "argv": ["short"],
                                "availability": {
                                    "memory": "unknown:insufficient_rss_samples"
                                }
                            },
                            {
                                "bin": "boundary",
                                "argv": ["boundary"],
                                "availability": {
                                    "memory": "unknown:insufficient_rss_samples"
                                }
                            },
                            {
                                "bin": "missing",
                                "argv": ["missing"],
                                "availability": {
                                    "memory": "unknown:missing_rss_profile"
                                }
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    adjusted, imputed_ids, metadata = idle_rss_safety._short_null_command_rows(
        command_rows, {"task": trace}
    )

    assert [clause.sampled_peak_rss_mb for clause in adjusted[("task", "call")].clauses] == [
        700.0,
        500.0,
        None,
        None,
    ]
    assert imputed_ids == {"task:call"}
    assert metadata == {"imputed_clauses": 1, "imputed_commands": 1}


def test_short_null_policy_rejects_equal_length_clause_reordering(tmp_path) -> None:
    clauses = (
        Row("task", "repo", 0, "first", ("first",), 100.0, None, None, 0.0),
        Row("task", "repo", 0, "second", ("second",), 100.0, None, None, 0.0),
    )
    command_rows = {
        ("task", "call"): CommandRow(
            "task", "repo", 0, 0, "call", "first; second", 200.0, clauses
        )
    }
    attempt = tmp_path / "attempt_1"
    attempt.mkdir()
    trace = attempt / "trace.jsonl"
    trace.write_text("", encoding="utf-8")
    (attempt / "resource_observations.json").write_text(
        json.dumps(
            {
                "calls": [
                    {
                        "tool_call_id": "call",
                        "clauses": [
                            {
                                "bin": "second",
                                "argv": ["second"],
                                "availability": {
                                    "memory": "unknown:insufficient_rss_samples"
                                },
                            },
                            {
                                "bin": "first",
                                "argv": ["first"],
                                "availability": {
                                    "memory": "unknown:insufficient_rss_samples"
                                },
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="identity differs"):
        idle_rss_safety._short_null_command_rows(
            command_rows, {"task": trace}
        )


def test_short_null_policy_composes_imputed_pipeline_rss(tmp_path) -> None:
    clauses = tuple(
        Row(
            "task",
            "repo",
            0,
            name,
            (name,),
            100.0,
            None,
            None,
            0.0,
            in_pipe=True,
            pipeline_position=position,
        )
        for position, name in enumerate(("left", "right"))
    )
    command_rows = {
        ("task", "call"): CommandRow(
            "task", "repo", 0, 0, "call", "left | right", 100.0, clauses
        )
    }
    attempt = tmp_path / "attempt_1"
    attempt.mkdir()
    trace = attempt / "trace.jsonl"
    trace.write_text("", encoding="utf-8")
    (attempt / "resource_observations.json").write_text(
        json.dumps(
            {
                "calls": [
                    {
                        "tool_call_id": "call",
                        "clauses": [
                            {
                                "bin": name,
                                "argv": [name],
                                "availability": {
                                    "memory": "unknown:insufficient_rss_samples"
                                },
                            }
                            for name in ("left", "right")
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    adjusted, _imputed_ids, _metadata = idle_rss_safety._short_null_command_rows(
        command_rows, {"task": trace}
    )

    assert _reservation(adjusted[("task", "call")]) == (
        8.0,
        1_000.0,
        "cpu_null_target_fallback",
    )


def test_idle_rss_gate_requires_utility_contribution_and_safety() -> None:
    passing = _idle_rss_gate(
        candidate_reduction=0.08,
        oracle_reduction=0.14,
        clause_reduction=0.06,
        bootstrap_high=-1.0,
        service_inflation=0.05,
        makespan_regression=0.01,
        exposure_events=0,
        violation=False,
    )
    assert passing["go"]

    for field, value in (
        ("candidate_reduction", 0.049),
        ("oracle_reduction", 0.17),
        ("clause_reduction", 0.071),
        ("bootstrap_high", 0.0),
        ("service_inflation", 0.051),
        ("makespan_regression", 0.011),
        ("exposure_events", 1),
        ("violation", True),
    ):
        values = {
            "candidate_reduction": 0.08,
            "oracle_reduction": 0.14,
            "clause_reduction": 0.06,
            "bootstrap_high": -1.0,
            "service_inflation": 0.05,
            "makespan_regression": 0.01,
            "exposure_events": 0,
            "violation": False,
        }
        values[field] = value
        assert not _idle_rss_gate(**values)["go"]


def test_short_null_gate_requires_every_absolute_condition() -> None:
    passing = idle_rss_safety._short_null_gate(
        candidate_reduction=0.08,
        oracle_reduction=0.14,
        bootstrap_high=-1.0,
        service_inflation=0.05,
        makespan_regression=0.01,
        exposure_events=0,
        violation=False,
    )
    assert passing["go"]
    assert idle_rss_safety.SHORT_NULL_VERSION == "cpu-idle-short-null-v1"
    assert idle_rss_safety.SHORT_NULL_PROTOCOL.name == (
        "cpu-idle-short-null-amendment.md"
    )

    for field, value in (
        ("candidate_reduction", 0.049),
        ("oracle_reduction", 0.17),
        ("bootstrap_high", 0.0),
        ("service_inflation", 0.051),
        ("makespan_regression", 0.011),
        ("exposure_events", 1),
        ("violation", True),
    ):
        values = {
            "candidate_reduction": 0.08,
            "oracle_reduction": 0.14,
            "bootstrap_high": -1.0,
            "service_inflation": 0.05,
            "makespan_regression": 0.01,
            "exposure_events": 0,
            "violation": False,
        }
        values[field] = value
        assert not idle_rss_safety._short_null_gate(**values)["go"]


def test_short_null_evaluator_rejects_unknown_source_policy() -> None:
    with pytest.raises(ValueError, match="source policy"):
        idle_rss_safety.run(seeds=(0,), source_policy="unknown")


def test_rss_safety_source_policy_selects_its_frozen_protocol() -> None:
    assert idle_rss_safety._protocol_for("conservative") == (
        idle_rss_safety.PROTOCOL
    )
    assert idle_rss_safety._protocol_for("short-null-low") == (
        idle_rss_safety.SHORT_NULL_PROTOCOL
    )


def test_feedback_admission_gate_requires_every_frozen_condition() -> None:
    passing = _gate(
        relative_mean_completion_reduction=0.05,
        paired_bootstrap_high=-1e-9,
        service_inflation=0.05,
        capacity_violation=False,
    )

    assert passing["go"]
    assert not _gate(0.049, -1e-9, 0.05, False)["go"]
    assert not _gate(0.05, 0.0, 0.05, False)["go"]
    assert not _gate(0.05, -1e-9, 0.051, False)["go"]
    assert not _gate(0.05, -1e-9, 0.05, True)["go"]


def test_feedback_admission_averages_per_order_relative_reductions() -> None:
    rows = [
        {
            "arms": {
                "task_aware_static": {"mean_task_completion_s": 100.0},
                "task_aware_feedback": {"mean_task_completion_s": 96.0},
            }
        },
        {
            "arms": {
                "task_aware_static": {"mean_task_completion_s": 10.0},
                "task_aware_feedback": {"mean_task_completion_s": 9.0},
            }
        },
    ]

    assert _mean_relative_reduction(rows) == pytest.approx(0.07)


def test_feedback_admission_rejects_protocol_constant_drift() -> None:
    _assert_frozen_protocol(40, tuple(range(32)), 8.0, 16_000.0)

    with pytest.raises(ValueError, match="frozen protocol constants changed"):
        _assert_frozen_protocol(39, tuple(range(32)), 8.0, 16_000.0)


def test_feedback_admission_keeps_same_named_input_identities(tmp_path) -> None:
    first = tmp_path / "first" / "rows.jsonl"
    second = tmp_path / "second" / "rows.jsonl"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("first\n")
    second.write_text("second\n")

    identities = _file_identities((first, second))

    assert set(identities) == {str(first.resolve()), str(second.resolve())}


def test_borrowing_evaluator_has_a_separate_frozen_entrypoint() -> None:
    assert BORROWING_VERSION == "cpu-feedback-borrowing-v1"
    assert BORROWING_PROTOCOL.name == "cpu-feedback-borrowing-protocol.md"
    assert WORK_CONSERVING_CPU is True


def test_borrowing_fixed8_context_preserves_task_aware_rss() -> None:
    requests = {"a": (2.0, 500.0), "b": (8.0, 2_000.0)}

    assert _fixed_cpu_requests(requests) == {
        "a": (8.0, 500.0),
        "b": (8.0, 2_000.0),
    }


def test_feedback_evaluator_reports_service_inflation() -> None:
    assert _service_inflation(
        {"total_command_service_s": 11.0, "recorded_command_service_s": 10.0}
    ) == pytest.approx(0.1)


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


def test_burstable_share_weights_change_completion_order_not_admission() -> None:
    programs = [
        AdmissionProgram(
            task_id,
            0.0,
            (AdmissionCommand(f"{task_id}:cmd", 1.0, 4.0, 100.0, 0.0),),
            0.0,
        )
        for task_id in ("a", "b")
    ]
    reservations = {
        "a:cmd": (2.0, 100.0),
        "b:cmd": (2.0, 100.0),
    }
    work = {"a:cmd": 4.0, "b:cmd": 4.0}
    demand = {"a:cmd": 4.0, "b:cmd": 4.0}

    equal = simulate_burstable_admission(
        programs,
        cpu_capacity=4.0,
        rss_capacity_mb=1_000.0,
        requested_reservations=reservations,
        cpu_work_core_s=work,
        max_cpu_cores=demand,
    )
    weighted = simulate_burstable_admission(
        programs,
        cpu_capacity=4.0,
        rss_capacity_mb=1_000.0,
        requested_reservations=reservations,
        cpu_work_core_s=work,
        max_cpu_cores=demand,
        cpu_share_weights={"a:cmd": 1.0, "b:cmd": 3.0},
    )
    scaled_equal = [
        simulate_burstable_admission(
            programs,
            cpu_capacity=4.0,
            rss_capacity_mb=1_000.0,
            requested_reservations=reservations,
            cpu_work_core_s=work,
            max_cpu_cores=demand,
            cpu_share_weights={"a:cmd": scale, "b:cmd": scale},
        )
        for scale in (5.0, 1e-320, 1e308)
    ]

    assert weighted["makespan_s"] == equal["makespan_s"] == 2.0
    assert weighted["mean_task_completion_s"] < equal["mean_task_completion_s"]
    assert weighted["max_concurrent_commands"] == equal["max_concurrent_commands"]
    assert weighted["service_s_by_command"] == pytest.approx(
        {"a:cmd": 2.0, "b:cmd": 4.0 / 3.0}
    )
    assert all(
        result["service_s_by_command"] == equal["service_s_by_command"]
        for result in scaled_equal
    )


def test_burstable_default_shares_follow_unequal_requests() -> None:
    programs = [
        AdmissionProgram(
            task_id,
            0.0,
            (AdmissionCommand(f"{task_id}:cmd", 1.0, 4.0, 100.0, 0.0),),
            0.0,
        )
        for task_id in ("a", "b")
    ]
    reservations = {"a:cmd": (1.0, 100.0), "b:cmd": (3.0, 100.0)}
    kwargs = {
        "cpu_capacity": 4.0,
        "rss_capacity_mb": 1_000.0,
        "requested_reservations": reservations,
        "cpu_work_core_s": {"a:cmd": 4.0, "b:cmd": 4.0},
        "max_cpu_cores": {"a:cmd": 4.0, "b:cmd": 4.0},
    }

    default = simulate_burstable_admission(programs, **kwargs)
    equal = simulate_burstable_admission(
        programs,
        **kwargs,
        cpu_share_weights={"a:cmd": 1.0, "b:cmd": 1.0},
    )

    assert default["mean_task_completion_s"] == pytest.approx(5.0 / 3.0)
    assert equal["mean_task_completion_s"] == 2.0


def test_burstable_rejects_numerically_unusable_share_ratio() -> None:
    programs = [
        AdmissionProgram(
            task_id,
            0.0,
            (AdmissionCommand(f"{task_id}:cmd", 1.0, 4.0, 100.0, 0.0),),
            0.0,
        )
        for task_id in ("a", "b")
    ]
    with pytest.raises(ValueError, match="dynamic range"):
        simulate_burstable_admission(
            programs,
            cpu_capacity=4.0,
            rss_capacity_mb=1_000.0,
            requested_reservations={
                "a:cmd": (2.0, 100.0),
                "b:cmd": (2.0, 100.0),
            },
            cpu_work_core_s={"a:cmd": 4.0, "b:cmd": 4.0},
            max_cpu_cores={"a:cmd": 4.0, "b:cmd": 4.0},
            cpu_share_weights={"a:cmd": 1e-300, "b:cmd": 1e300},
        )


def test_burstable_admission_priority_reorders_ready_commands() -> None:
    durations = {"a": 3.0, "b": 1.0, "c": 1.0}
    programs = [
        AdmissionProgram(
            task_id,
            0.0,
            (
                AdmissionCommand(
                    f"{task_id}:cmd", duration, 1.0, 100.0, 0.0
                ),
            ),
            0.0,
        )
        for task_id, duration in durations.items()
    ]
    reservations = {f"{task_id}:cmd": (1.0, 100.0) for task_id in durations}
    work = {f"{task_id}:cmd": duration for task_id, duration in durations.items()}
    demand = {f"{task_id}:cmd": 1.0 for task_id in durations}
    kwargs = {
        "cpu_capacity": 1.0,
        "rss_capacity_mb": 1_000.0,
        "requested_reservations": reservations,
        "cpu_work_core_s": work,
        "max_cpu_cores": demand,
    }

    baseline = simulate_burstable_admission(programs, **kwargs)
    prioritized = simulate_burstable_admission(
        programs,
        **kwargs,
        admission_priorities={"a:cmd": 2.0, "b:cmd": 0.0, "c:cmd": 1.0},
    )

    assert baseline["start_s_by_command"] == {
        "a:cmd": 0.0,
        "b:cmd": 3.0,
        "c:cmd": 4.0,
    }
    assert prioritized["start_s_by_command"] == {
        "a:cmd": 2.0,
        "b:cmd": 0.0,
        "c:cmd": 1.0,
    }
    assert prioritized["mean_task_completion_s"] == pytest.approx(8.0 / 3.0)
    assert baseline["mean_task_completion_s"] == 4.0
    assert prioritized["makespan_s"] == baseline["makespan_s"] == 5.0


def test_burstable_priority_cannot_see_future_ready_command() -> None:
    programs = [
        AdmissionProgram(
            "ready",
            0.0,
            (AdmissionCommand("ready:cmd", 2.0, 1.0, 100.0, 0.0),),
            0.0,
        ),
        AdmissionProgram(
            "future",
            1.0,
            (AdmissionCommand("future:cmd", 1.0, 1.0, 100.0, 0.0),),
            0.0,
        ),
    ]
    result = simulate_burstable_admission(
        programs,
        cpu_capacity=1.0,
        rss_capacity_mb=1_000.0,
        requested_reservations={
            "ready:cmd": (1.0, 100.0),
            "future:cmd": (1.0, 100.0),
        },
        cpu_work_core_s={"ready:cmd": 2.0, "future:cmd": 1.0},
        max_cpu_cores={"ready:cmd": 1.0, "future:cmd": 1.0},
        admission_priorities={"ready:cmd": 1.0, "future:cmd": 0.0},
    )

    assert result["start_s_by_command"] == {
        "future:cmd": 2.0,
        "ready:cmd": 0.0,
    }


def test_burstable_priority_requires_complete_finite_values() -> None:
    programs = [
        AdmissionProgram(
            "task",
            0.0,
            (AdmissionCommand("task:cmd", 1.0, 1.0, 100.0, 0.0),),
            0.0,
        )
    ]
    kwargs = {
        "cpu_capacity": 1.0,
        "rss_capacity_mb": 1_000.0,
        "requested_reservations": {"task:cmd": (1.0, 100.0)},
        "cpu_work_core_s": {"task:cmd": 1.0},
        "max_cpu_cores": {"task:cmd": 1.0},
    }

    with pytest.raises(ValueError, match="complete admission priorities"):
        simulate_burstable_admission(programs, **kwargs, admission_priorities={})
    with pytest.raises(ValueError, match="priority aging"):
        simulate_burstable_admission(
            programs,
            **kwargs,
            age_admission_priorities=True,
        )
    with pytest.raises(ValueError, match="finite"):
        simulate_burstable_admission(
            programs,
            **kwargs,
            admission_priorities={"task:cmd": float("nan")},
        )


def test_burstable_priority_aging_bounds_short_job_overtaking() -> None:
    specs = {
        "long": (0.0, 3.0),
        "short0": (0.0, 1.0),
        "short1": (1.0, 1.0),
        "short2": (2.0, 1.0),
        "short3": (3.0, 1.0),
    }
    programs = [
        AdmissionProgram(
            task_id,
            ready_s,
            (AdmissionCommand(f"{task_id}:cmd", duration, 1.0, 100.0, 0.0),),
            0.0,
        )
        for task_id, (ready_s, duration) in specs.items()
    ]
    command_ids = {f"{task_id}:cmd" for task_id in specs}
    kwargs = {
        "cpu_capacity": 1.0,
        "rss_capacity_mb": 1_000.0,
        "requested_reservations": {cid: (1.0, 100.0) for cid in command_ids},
        "cpu_work_core_s": {
            f"{task_id}:cmd": duration
            for task_id, (_ready_s, duration) in specs.items()
        },
        "max_cpu_cores": {cid: 1.0 for cid in command_ids},
        "admission_priorities": {
            f"{task_id}:cmd": duration
            for task_id, (_ready_s, duration) in specs.items()
        },
    }

    pure = simulate_burstable_admission(programs, **kwargs)
    aged = simulate_burstable_admission(
        programs,
        **kwargs,
        age_admission_priorities=True,
    )

    assert pure["start_s_by_command"]["long:cmd"] == 4.0
    assert aged["start_s_by_command"]["long:cmd"] == 2.0


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


def test_pairwise_peak_uses_canonical_cpu_upper_bounds() -> None:
    assert [_bucket_upper(value) for value in (0.0, 2.0, 2.01, 4.0, 4.01, 8.0)] == [
        2.0,
        2.0,
        4.0,
        4.0,
        8.0,
        8.0,
    ]


def test_phase_envelope_reuses_clause_prefixes_and_max_rates() -> None:
    direct = _command_signatures("python -m pytest tests/test_a.py")
    venv = _command_signatures("/opt/venv/bin/python -m pytest tests/test_b.py")

    assert direct[0] != venv[0]
    assert direct[2] == venv[2]
    assert _profile_envelope(
        (
            ((1.0, 8.0), (1.0, 0.0)),
            ((0.5, 0.0), (1.0, 4.0), (0.5, 0.0)),
        )
    ) == (
        (0.5, 4.0),
        (0.5, 4.0),
        (0.5, 2.0),
        (0.5, 0.0),
    )
