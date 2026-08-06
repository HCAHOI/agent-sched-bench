from collections import Counter

import pytest

from scripts.evaluation.evaluate_pytest_transition_residual import (
    _correct,
    _posterior,
    _states,
    _transition,
    _validate_fit,
)


def test_transition_categories_and_task_reset() -> None:
    assert _transition(frozenset({"file:a.py"}), frozenset({"file:a.py"})) == "exact_repeat"
    assert _transition(frozenset({"file:a.py"}), frozenset({"file:a.py", "file:b.py"})) == "expand"
    assert _transition(frozenset(), frozenset({"file:a.py"})) == "full_to_targeted"
    assert _transition(frozenset({"file:a.py"}), frozenset()) == "targeted_to_full"

    rows = [
        {"task_id": "a", "command": "pytest tests/a.py"},
        {"task_id": "a", "command": "pytest tests/a.py tests/b.py"},
        {"task_id": "b", "command": "pytest tests/a.py tests/b.py"},
    ]
    assert _states(rows) == [
        ("first", "no_comparable_previous"),
        ("second", "expand"),
        ("first", "no_comparable_previous"),
    ]


def test_posterior_correction_shrinks_and_normalizes() -> None:
    population = _posterior(Counter({0: 8, 1: 2}), [0.5, 0.5])
    repeated = _posterior(Counter({0: 1, 1: 4}), population)
    corrected = _correct([0.8, 0.2], 5.0, population, repeated)

    assert sum(corrected) == pytest.approx(1.0)
    assert corrected[1] > 0.2


def test_fit_population_fails_closed() -> None:
    with pytest.raises(ValueError, match="frozen population"):
        _validate_fit([])
