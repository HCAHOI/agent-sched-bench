from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluation.evaluate_pennylane_joint_phase_packing import (
    Capacities,
    TaskProfile,
    _held_samples,
    best_feasible,
    simulate,
)


def _profile(task_id: str) -> TaskProfile:
    return TaskProfile(
        task_id,
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 1.0, 0.0]),
        np.array([0.0, 1.0, 1.0, 0.0]),
    )


def test_phase_gate_reconsiders_admission_before_task_completion() -> None:
    result = simulate(
        [_profile("a"), _profile("b")],
        "gpu",
        active_cap=None,
        capacities=Capacities(1.0, 2.0, 2.0),
    )

    assert result["start_s_by_task"] == {"a": 0.0, "b": 2.0}
    assert result["maximum_active_tasks"] == 2
    assert result["feasible"]


def test_best_single_resource_cap_is_feasible_on_all_resources() -> None:
    best, search = best_feasible(
        [_profile("a"), _profile("b")],
        "gpu",
        Capacities(1.0, 1.0, 1.0),
    )

    assert [row["active_cap"] for row in search] == [1, 2]
    assert not search[1]["feasible"]
    assert best["active_cap"] == 1
    assert best["feasible"]


def test_same_bin_peak_does_not_become_the_next_hold_value() -> None:
    cpu, rss = _held_samples(
        [
            {"epoch": 0.1, "cpu_percent": "1000%", "mem_usage": "10MiB"},
            {"epoch": 1.9, "cpu_percent": "100%", "mem_usage": "1MiB"},
        ],
        0.0,
        3,
    )

    assert cpu.tolist() == [10.0, 1.0, 1.0]
    assert rss.tolist() == [10.0, 1.0, 1.0]


def test_non_finite_resource_sample_fails_closed() -> None:
    with pytest.raises(ValueError, match="invalid task-resource sample"):
        _held_samples(
            [{"epoch": 0.1, "cpu_percent": "NaN%", "mem_usage": "1MiB"}],
            0.0,
            1,
        )
