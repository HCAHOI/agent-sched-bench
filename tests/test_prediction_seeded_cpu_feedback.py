import pytest

from scripts.evaluation.evaluate_prediction_seeded_cpu_feedback import (
    _assert_alignment,
    _decision,
    _summarize,
)


def _row(duration: float, arms: dict[str, tuple[float, float]]) -> dict:
    return {
        "recorded_duration_s": duration,
        "arms": {
            name: {"service_s": service, "reserved_cpu_core_s": reserved}
            for name, (service, reserved) in arms.items()
        },
    }


def test_summary_selects_lowest_reservation_arm_within_service_budget() -> None:
    rows = [
        _row(
            10.0,
            {
                "fixed8": (10.0, 80.0),
                "feedback8": (10.2, 44.0),
                "clause_kb_feedback": (10.3, 42.0),
                "task_aware_feedback": (10.4, 40.0),
                "task_aware_static": (11.0, 24.0),
            },
        ),
        _row(
            10.0,
            {
                "fixed8": (10.0, 80.0),
                "feedback8": (10.2, 44.0),
                "clause_kb_feedback": (10.3, 42.0),
                "task_aware_feedback": (10.4, 40.0),
                "task_aware_static": (11.0, 24.0),
            },
        ),
    ]

    result = _summarize(rows)

    assert result["selected_arm"] == "task_aware_feedback"
    assert result["arms"]["task_aware_feedback"]["service_inflation"] == pytest.approx(0.04)
    assert result["arms"]["task_aware_feedback"]["reservation_reduction"] == pytest.approx(0.50)
    assert "task_aware_static" not in result["eligible_feedback_arms"]


def test_summary_rejects_feedback_arm_above_five_percent_service_inflation() -> None:
    rows = [
        _row(
            10.0,
            {
                "fixed8": (10.0, 80.0),
                "feedback8": (10.2, 44.0),
                "clause_kb_feedback": (10.4, 42.0),
                "task_aware_feedback": (10.6, 20.0),
            },
        )
    ]

    result = _summarize(rows)

    assert result["selected_arm"] == "clause_kb_feedback"
    assert result["arms"]["task_aware_feedback"]["service_budget_pass"] is False


def test_decision_requires_incremental_prediction_value() -> None:
    feedback_only = {
        "selected_arm": "feedback8",
        "arms": {
            "fixed8": {"service_inflation": 0.0, "reservation_reduction": 0.0},
            "feedback8": {
                "service_inflation": 0.02,
                "reservation_reduction": 0.40,
            },
            "clause_kb_feedback": {
                "service_inflation": 0.022,
                "reservation_reduction": 0.405,
            },
            "task_aware_feedback": {
                "service_inflation": 0.021,
                "reservation_reduction": 0.408,
            },
        },
    }
    seeded = {
        **feedback_only,
        "selected_arm": "task_aware_feedback",
        "arms": {
            **feedback_only["arms"],
            "task_aware_feedback": {
                "service_inflation": 0.024,
                "reservation_reduction": 0.42,
            },
        },
    }

    assert _decision(feedback_only)["prediction_seed_go"] is False
    assert _decision(feedback_only)["status"] == "development_promising_feedback_only"
    assert _decision(seeded)["prediction_seed_go"] is True
    assert _decision(seeded)["status"] == "development_promising_prediction_seed"


def test_alignment_rejects_prediction_or_trace_id_drift() -> None:
    expected = {"task:call-1", "task:call-2"}
    predictions = {command_id: (2.0, 500.0) for command_id in expected}
    _assert_alignment(expected, predictions, predictions, expected | {"task:other"})

    with pytest.raises(ValueError, match="Clause-KB"):
        _assert_alignment(expected, {"task:call-1": (2.0, 500.0)}, predictions, expected)
    with pytest.raises(ValueError, match="absent from traces"):
        _assert_alignment(expected, predictions, predictions, {"task:call-1"})
