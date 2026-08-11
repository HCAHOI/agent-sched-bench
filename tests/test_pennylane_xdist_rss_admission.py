from copy import deepcopy

from scripts.evaluation.evaluate_pennylane_xdist_rss_admission import (
    _gate,
    decision_changes,
    prediction_reservations,
)


def test_prediction_reservations_and_decision_gate() -> None:
    reservations = prediction_reservations(
        {"task:a", "task:b"},
        [
            {
                "task_id": "task",
                "call_id": "a",
                "clause_kb": [1.0, 0.0, 0.0],
                "count_unconditioned": [0.0, 1.0, 0.0],
                "scope_conditioned": [0.0, 0.0, 1.0],
            }
        ],
    )
    assert reservations["clause_kb"] == {"task:a": 500.0, "task:b": 16_000.0}
    assert reservations["count_unconditioned"]["task:a"] == 2_000.0
    assert reservations["scope_conditioned"]["task:a"] == 16_000.0

    changes = decision_changes(["a:1", "b:1"], ["b:1", "c:1"])
    assert changes == {"commands": 2, "tasks": 2, "command_ids": ["a:1", "c:1"]}

    base = {
        "mean_task_completion_s": 100.0,
        "makespan_s": 100.0,
        "recorded_command_service_s": 100.0,
        "total_command_service_s": 100.0,
        "modeled_capacity_exposure_events": 2,
        "capacity_violation": False,
        "physical_capacity_violation": False,
        "total_cpu_work_core_s": 8.0,
        "served_cpu_work_core_s": 8.0,
        "speculative_starts": 20,
        "speculative_task_count": 5,
    }
    arms = {
        "serial8": base,
        "clause_kb": base,
        "count_unconditioned": base,
        "scope_conditioned": base
        | {
            "mean_task_completion_s": 90.0,
            "makespan_s": 90.0,
            "modeled_capacity_exposure_events": 0,
        },
    }
    assert _gate(arms, {"commands": 10, "tasks": 3})["go"]

    failures = (
        ("candidate", "modeled_capacity_exposure_events", 1),
        ("candidate", "capacity_violation", True),
        ("candidate", "physical_capacity_violation", True),
        ("candidate", "served_cpu_work_core_s", 7.0),
        ("candidate", "mean_task_completion_s", 96.0),
        ("candidate", "makespan_s", 100.0),
        ("candidate", "total_command_service_s", 106.0),
        ("clause_kb", "modeled_capacity_exposure_events", 0),
        ("count_unconditioned", "modeled_capacity_exposure_events", 0),
        ("candidate", "speculative_starts", 19),
        ("candidate", "speculative_task_count", 4),
        ("changes", "commands", 9),
        ("changes", "tasks", 2),
    )
    for owner, field, value in failures:
        changed_arms = deepcopy(arms)
        changed_decisions = {"commands": 10, "tasks": 3}
        if owner == "changes":
            target = changed_decisions
        else:
            target = changed_arms[
                "scope_conditioned" if owner == "candidate" else owner
            ]
        target[field] = value
        assert not _gate(changed_arms, changed_decisions)["go"], field

    arms["exact_rss_reference"] = deepcopy(base) | {
        "mean_task_completion_s": 1.0,
        "modeled_capacity_exposure_events": 999,
    }
    assert _gate(arms, {"commands": 10, "tasks": 3})["go"]
