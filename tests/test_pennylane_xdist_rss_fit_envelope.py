from copy import deepcopy

from scripts.evaluation.evaluate_pennylane_xdist_rss_fit_envelope import (
    _gate,
    continuous_reservations,
)


def test_fit_envelope_changes_only_supported_high_carriers() -> None:
    rows = [
        {
            "task_id": "task",
            "call_id": call_id,
            "carrier": carrier,
            "workers": 4,
            "scope": scope,
            "clause_kb": pmf,
            "count_unconditioned": pmf,
            "scope_conditioned": pmf,
        }
        for call_id, carrier, scope, pmf in (
            ("high", True, "broad", [0.0, 0.0, 1.0]),
            ("medium", True, "narrow", [0.0, 1.0, 0.0]),
            ("plain", False, None, [1.0, 0.0, 0.0]),
        )
    ]
    reservations, activated = continuous_reservations(
        {"task:high", "task:medium", "task:plain", "task:unknown"},
        rows,
        {(4, "broad"): 11_000.0, (4, "narrow"): 12_000.0},
    )
    assert reservations == {
        "task:high": 11_000.0,
        "task:medium": 2_000.0,
        "task:plain": 500.0,
        "task:unknown": 16_000.0,
    }
    assert activated["command_ids"] == ["task:high"]


def test_fit_envelope_gate_locks_every_condition() -> None:
    serial = {
        "mean_task_completion_s": 100.0,
        "makespan_s": 100.0,
        "recorded_command_service_s": 100.0,
        "total_command_service_s": 100.0,
    }
    scope = serial | {"mean_task_completion_s": 90.0}
    candidate = serial | {
        "mean_task_completion_s": 80.0,
        "makespan_s": 90.0,
        "modeled_capacity_exposure_events": 0,
        "capacity_violation": False,
        "physical_capacity_violation": False,
        "total_cpu_work_core_s": 8.0,
        "served_cpu_work_core_s": 8.0,
    }
    changes = {"commands": 10, "tasks": 3}
    assert _gate(serial, scope, candidate, changes)["go"]

    failures = (
        ("candidate", "modeled_capacity_exposure_events", 1),
        ("candidate", "capacity_violation", True),
        ("candidate", "physical_capacity_violation", True),
        ("candidate", "served_cpu_work_core_s", 7.0),
        ("candidate", "mean_task_completion_s", 96.0),
        ("candidate", "makespan_s", 100.0),
        ("candidate", "total_command_service_s", 106.0),
        ("scope", "mean_task_completion_s", 80.5),
        ("changes", "commands", 9),
        ("changes", "tasks", 2),
    )
    for owner, field, value in failures:
        changed_scope = deepcopy(scope)
        changed_candidate = deepcopy(candidate)
        changed_decisions = deepcopy(changes)
        target = {
            "scope": changed_scope,
            "candidate": changed_candidate,
            "changes": changed_decisions,
        }[owner]
        target[field] = value
        assert not _gate(
            serial, changed_scope, changed_candidate, changed_decisions
        )["go"], field
