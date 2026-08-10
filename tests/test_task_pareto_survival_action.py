from __future__ import annotations

from pathlib import Path

import pytest

from scripts.evaluation.evaluate_task_pareto_survival_action import (
    _SPLIT_REPO_PATH,
    _gate,
    _git_blob_oid,
    _require_frozen_file,
    _require_closed_tokenizer_cache,
    _retained_tokens,
    _select,
    task_unanimous_pareto_trigger_ms,
)
from spike.multitenant import TraceTurn


_PHYSICAL = {
    "deadline_ms": 5000.0,
    "size_gib": 1.0,
    "swap_out_ms": 100.0,
    "swap_in_ms": 80.0,
}


def test_task_unanimous_trigger_uses_largest_observed_short_duration() -> None:
    assert task_unanimous_pareto_trigger_ms(
        {"a": (1000.0, 8000.0), "b": (2000.0, 9000.0)},
        **_PHYSICAL,
    ) == 2000.0


def test_task_unanimous_trigger_requires_two_independent_beneficial_tasks() -> None:
    assert task_unanimous_pareto_trigger_ms(
        {"a": (8000.0,)}, **_PHYSICAL
    ) == 5000.0
    assert task_unanimous_pareto_trigger_ms(
        {"a": (8000.0,), "b": (2000.0,)}, **_PHYSICAL
    ) == 5000.0


def test_task_unanimous_trigger_releases_immediately_for_all_long_histories() -> None:
    assert task_unanimous_pareto_trigger_ms(
        {"a": (8000.0,), "b": (9000.0,)}, **_PHYSICAL
    ) == 0.0


def test_retained_tokens_include_the_finished_completion() -> None:
    class Tokenizer:
        @staticmethod
        def apply_chat_template(*args: object, **kwargs: object) -> list[int]:
            return list(range(10))

    turn = TraceTurn(
        messages=({"role": "user", "content": "x"},),
        recorded_prompt_tokens=1,
        completion_tokens=7,
        gap_ms=1.0,
        tools=(),
    )
    assert _retained_tokens(turn, Tokenizer()) == 16


def test_frozen_file_rejects_an_alternate_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frozen input path changed"):
        _require_frozen_file(tmp_path / "split.json", _SPLIT_REPO_PATH)


def test_tokenizer_blob_identity_uses_git_blob_oid() -> None:
    assert _git_blob_oid(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


def test_tokenizer_cache_rejects_unverified_template(tmp_path: Path) -> None:
    (tmp_path / "chat_template.jinja").write_text("override")
    with pytest.raises(ValueError, match="unexpected tokenizer cache entries"):
        _require_closed_tokenizer_cache(tmp_path)


def test_gate_and_selection_keep_work_as_primary() -> None:
    passing = {
        "released_gib_s_delta": 1.0,
        "critical_path_stall_ms_delta": 0.0,
        "changed_task_count": 8,
    }
    assert all(_gate(passing).values())

    comparisons = {
        "exact_task_pareto": {
            "released_gib_s_delta": 1.0,
            "critical_path_stall_ms_delta": 0.0,
        },
        "work_task_pareto": {
            "released_gib_s_delta": 2.0,
            "critical_path_stall_ms_delta": 0.0,
        },
    }
    assert _select(
        {"exact_task_pareto": True, "work_task_pareto": False}, comparisons
    ) == (False, True, None)
    assert _select(
        {"exact_task_pareto": True, "work_task_pareto": True}, comparisons
    ) == (True, True, "work_task_pareto")

    comparisons["work_task_pareto"]["released_gib_s_delta"] = 1.0
    assert _select(
        {"exact_task_pareto": True, "work_task_pareto": True}, comparisons
    ) == (True, False, "exact_task_pareto")
