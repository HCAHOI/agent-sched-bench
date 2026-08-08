from scripts.evaluation.evaluate_prediction_weighted_cpu_shares import (
    _gate,
    _priority_weight,
    _service_changes,
)


def test_priority_weight_favors_predicted_short_commands() -> None:
    assert _priority_weight(None) == 1.0
    assert _priority_weight((0.9, 0.1, 0.0, 0.0, 0.0)) == 5.0
    assert _priority_weight((0.0, 0.0, 0.0, 0.1, 0.9)) == 1.0


def test_service_changes_ignore_action_equivalent_weight_scaling() -> None:
    assert _service_changes({"a": 2.0, "b": 3.0}, {"a": 2.0, "b": 3.0}) == set()
    assert _service_changes({"a": 1.0, "b": 3.0}, {"a": 2.0, "b": 3.0}) == {
        "a"
    }


def test_weight_gate_requires_useful_prediction_without_makespan_harm() -> None:
    passing = _gate(
        predicted_reduction=0.03,
        predicted_ci=(-4.0, -1.0),
        makespan_regression=0.005,
        oracle_reduction=0.06,
        changed_commands=20,
        changed_tasks=10,
    )
    harmful = _gate(
        predicted_reduction=0.03,
        predicted_ci=(-4.0, -1.0),
        makespan_regression=0.02,
        oracle_reduction=0.06,
        changed_commands=20,
        changed_tasks=10,
    )

    assert passing["go"] is True
    assert harmful["go"] is False
