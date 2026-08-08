from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.evaluation.evaluate_cpu_feedback_generality import (
    _decision,
    _eligible_status,
    _task_row,
)


def test_feedback_generality_keeps_valid_task_without_exec() -> None:
    with patch(
        "scripts.evaluation.evaluate_cpu_feedback_generality.TraceData.load",
        return_value=SimpleNamespace(actions=[]),
    ):
        row = _task_row("swe100", "owner__repo-1", Path("unused.jsonl"))

    assert row["exec_commands"] == 0
    assert row["recorded_service_s"] == 0.0


def test_feedback_generality_requires_valid_source_and_all_frozen_gates() -> None:
    valid = {
        "success": True,
        "collection_validity": "valid",
        "telemetry_quality": "ok",
        "telemetry_integrity_failed": False,
        "replay_execution": "completed",
    }
    assert _eligible_status(valid)
    invalid_values = {
        "success": False,
        "collection_validity": "invalid",
        "telemetry_quality": "invalid",
        "telemetry_integrity_failed": True,
        "replay_execution": "failed",
    }
    for field, value in invalid_values.items():
        assert not _eligible_status({**valid, field: value})

    passing = {
        "reservation_reduction": 0.30,
        "service_inflation": 0.04,
    }
    decision = _decision(
        pooled=passing,
        corpora={"swe100": passing, "swe277": passing},
        pooled_intervals={
            "reservation_reduction_ci95": [0.25, 0.35],
            "service_inflation_ci95": [0.03, 0.045],
        },
        task_count=100,
        repo_count=10,
    )

    assert decision["go"] is True
    cases = (
        {"pooled": {**passing, "reservation_reduction": 0.24}},
        {"pooled": {**passing, "service_inflation": 0.06}},
        {
            "corpora": {
                "swe100": {**passing, "reservation_reduction": 0.24},
                "swe277": passing,
            }
        },
        {
            "corpora": {
                "swe100": {**passing, "service_inflation": 0.06},
                "swe277": passing,
            }
        },
        {
            "pooled_intervals": {
                "reservation_reduction_ci95": [0.19, 0.35],
                "service_inflation_ci95": [0.03, 0.045],
            }
        },
        {
            "pooled_intervals": {
                "reservation_reduction_ci95": [0.25, 0.35],
                "service_inflation_ci95": [0.03, 0.051],
            }
        },
        {"task_count": 99},
        {"repo_count": 9},
    )
    defaults = {
        "pooled": passing,
        "corpora": {"swe100": passing, "swe277": passing},
        "pooled_intervals": {
            "reservation_reduction_ci95": [0.25, 0.35],
            "service_inflation_ci95": [0.03, 0.045],
        },
        "task_count": 100,
        "repo_count": 10,
    }
    for override in cases:
        assert _decision(**{**defaults, **override})["go"] is False
