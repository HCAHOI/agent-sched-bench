from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluation.evaluate_pennylane_joint_phase_packing import (
    Capacities,
    TaskProfile,
)
from scripts.evaluation.evaluate_pennylane_perfect_container_parking import (
    _evaluate_profiles,
    _gate,
    _park,
)


def test_perfect_parking_releases_only_llm_bins_and_advances_admission() -> None:
    foreground = TaskProfile(
        "foreground",
        gpu=np.array([1.0, 1.0, 0.0, 0.0]),
        cpu=np.array([1.0, 1.0, 0.5, 0.5]),
        rss=np.array([1.0, 1.0, 0.5, 0.5]),
    )
    tool = TaskProfile(
        "tool",
        gpu=np.zeros(2),
        cpu=np.ones(2),
        rss=np.ones(2),
    )

    parked = _park(foreground)
    assert parked.cpu.tolist() == [0.0, 0.0, 0.5, 0.5]
    assert parked.rss.tolist() == [0.0, 0.0, 0.5, 0.5]
    assert parked.gpu.tolist() == foreground.gpu.tolist()

    result = _evaluate_profiles(
        [foreground, tool], Capacities(gpu_slots=1.0, cpu_cores=1.0, rss_mb=1.0)
    )

    assert result["status"] == "go"
    assert result["arms"]["always_resident"]["start_s_by_task"]["tool"] == 8.0
    assert result["arms"]["perfect_parking"]["start_s_by_task"]["tool"] == 0.0
    assert result["comparison"]["mean_task_completion_reduction"] == pytest.approx(0.4)
    assert result["comparison"]["total_start_advance_s"] == 8.0
    assert result["comparison"]["total_completion_advance_s"] == 8.0
    assert result["comparison"]["removed_cpu_core_s"] == 4.0
    assert result["comparison"]["removed_rss_mib_s"] == 4.0
    assert result["gate"]["checks"] == {
        "perfect_parking_zero_capacity_violations": True,
        "at_least_one_task_starts_earlier": True,
        "mean_task_completion_improves_at_least_5pct": True,
    }


def test_parking_gate_includes_exact_five_percent_and_rejects_failures() -> None:
    always = {"mean_task_completion_s": 100.0}

    assert (
        _gate(always, {"mean_task_completion_s": 95.0, "feasible": True}, 1)["status"]
        == "go"
    )
    assert (
        _gate(always, {"mean_task_completion_s": 95.0001, "feasible": True}, 1)[
            "status"
        ]
        == "no_go"
    )
    assert (
        _gate(always, {"mean_task_completion_s": 95.0, "feasible": True}, 0)["status"]
        == "no_go"
    )
    assert (
        _gate(always, {"mean_task_completion_s": 95.0, "feasible": False}, 1)["status"]
        == "no_go"
    )
