"""Tests for SWE-bench data loading and task selection."""

from __future__ import annotations

import json
from typing import Any

from agents.swebench_data import (
    _count_fail_to_pass,
    select_tool_intensive_tasks,
)


def _make_task(
    instance_id: str = "django__django-12345",
    repo: str = "django/django",
    fail_to_pass: list[str] | None = None,
    problem_statement: str = "A" * 200,
) -> dict[str, Any]:
    """Create a minimal task dict for testing."""
    if fail_to_pass is None:
        fail_to_pass = ["tests/test_foo.py::test_bar"]
    return {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": "abc123",
        "problem_statement": problem_statement,
        "FAIL_TO_PASS": json.dumps(fail_to_pass),
        "PASS_TO_PASS": "[]",
        "patch": "",
        "test_patch": "",
        "version": "4.0",
    }


def test_count_fail_to_pass_string_field() -> None:
    task = {"FAIL_TO_PASS": '["a", "b"]'}
    assert _count_fail_to_pass(task) == 2


def test_select_tool_intensive_tasks_returns_n() -> None:
    tasks = [
        _make_task(f"django__django-{i:05d}", "django/django", ["t1", "t2"])
        for i in range(20)
    ] + [_make_task(f"sympy__sympy-{i:05d}", "sympy/sympy", ["t1"]) for i in range(15)]
    selected = select_tool_intensive_tasks(tasks, n=10, seed=42)
    assert len(selected) == 10


def test_select_excludes_trivial_tasks() -> None:
    good = _make_task("django__django-00001", fail_to_pass=["t1"])
    trivial_short = _make_task(
        "django__django-00002",
        problem_statement="Fix typo",  # Too short (< 100 chars)
    )
    trivial_empty = _make_task(
        "django__django-00003",
        fail_to_pass=[],  # No tests to run
    )
    selected = select_tool_intensive_tasks(
        [good, trivial_short, trivial_empty],
        n=10,
        seed=42,
    )
    ids = {t["instance_id"] for t in selected}
    assert "django__django-00001" in ids
    assert "django__django-00002" not in ids
    assert "django__django-00003" not in ids


def test_select_is_deterministic() -> None:
    tasks = [
        _make_task(
            f"django__django-{i:05d}", "django/django", [f"t{j}" for j in range(i % 5)]
        )
        for i in range(1, 50)
    ]
    a = select_tool_intensive_tasks(tasks, n=10, seed=123)
    b = select_tool_intensive_tasks(tasks, n=10, seed=123)
    assert [t["instance_id"] for t in a] == [t["instance_id"] for t in b]


def test_select_sorted_by_instance_id() -> None:
    tasks = [
        _make_task(f"django__django-{i:05d}", "django/django", ["t1"])
        for i in range(20)
    ]
    selected = select_tool_intensive_tasks(tasks, n=5, seed=42)
    ids = [t["instance_id"] for t in selected]
    assert ids == sorted(ids)
