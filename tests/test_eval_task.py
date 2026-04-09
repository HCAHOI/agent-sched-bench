"""Tests for :class:`agents.openclaw.eval.types.EvalTask`.

Covers both the existing SWE-patch extraction path and the new
function-call fields for non-repo benchmarks.
"""

from __future__ import annotations

from pathlib import Path

from agents.openclaw.eval.types import EvalTask


# ── Function-call fields (US-002) ─────────────────────────────────────


def test_from_benchmark_instance_populates_function_call_fields(
    tmp_path: Path,
) -> None:
    """A function-call row (no repo, with tools/question/ground_truth) produces
    an EvalTask carrying all three function-call fields verbatim."""
    row = {
        "instance_id": "fc_simple_0",
        "problem_statement": "What is 2 + 2?",
        "category": "simple",
        "tools": [
            {
                "name": "add",
                "description": "Adds two numbers",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["a", "b"],
                },
            }
        ],
        "question": [[{"role": "user", "content": "What is 2 + 2?"}]],
        "ground_truth": [{"add": {"a": [2], "b": [2]}}],
    }
    task = EvalTask.from_benchmark_instance(row, tmp_path)

    assert task.instance_id == "fc_simple_0"
    assert task.problem_statement == "What is 2 + 2?"
    assert task.category == "simple"
    assert len(task.tools) == 1
    assert task.tools[0]["name"] == "add"
    assert task.question == [[{"role": "user", "content": "What is 2 + 2?"}]]
    assert task.ground_truth == [{"add": {"a": [2], "b": [2]}}]

    # SWE-patch fields are empty / None for function-call rows.
    assert task.repo is None
    assert task.base_commit is None
    assert task.fail_to_pass == []
    assert task.pass_to_pass == []


def test_needs_prepare_false_for_function_call_task(tmp_path: Path) -> None:
    """needs_prepare must return False when repo is None, even if tools
    are populated (guarantees non-repo tasks skip the git clone phase)."""
    row = {
        "instance_id": "fc_simple_0",
        "problem_statement": "x",
        "tools": [{"name": "f", "description": "", "parameters": {}}],
        "question": [[{"role": "user", "content": "x"}]],
    }
    task = EvalTask.from_benchmark_instance(row, tmp_path)
    assert task.needs_prepare is False


def test_from_benchmark_instance_missing_problem_statement_defaults_to_empty(
    tmp_path: Path,
) -> None:
    """Function-call rows may not have a flat ``problem_statement`` —
    the extraction must default to an empty string, not raise KeyError."""
    row = {
        "instance_id": "fc_mt_0",
        "question": [[{"role": "user", "content": "hi"}]],
        "tools": [],
    }
    task = EvalTask.from_benchmark_instance(row, tmp_path)
    assert task.problem_statement == ""
    assert task.question == [[{"role": "user", "content": "hi"}]]


# ── SWE-patch backward compatibility ──────────────────────────────────


def test_from_benchmark_instance_swe_patch_unchanged(tmp_path: Path) -> None:
    """Classic SWE-bench row path still extracts repo/base_commit/FAIL_TO_PASS
    correctly and leaves function-call fields as empty defaults."""
    row = {
        "instance_id": "django__django-11001",
        "problem_statement": "Fix the bug in Model.save",
        "repo": "django/django",
        "base_commit": "abc123def",
        "FAIL_TO_PASS": '["tests/test_models.py::test_save"]',
        "PASS_TO_PASS": "[]",
        "test_patch": "--- a/test\n+++ b/test\n",
        "image_name": "swebench/sweb.eval.x86_64.django__django-11001",
    }
    task = EvalTask.from_benchmark_instance(row, tmp_path)

    assert task.repo == "django/django"
    assert task.base_commit == "abc123def"
    assert task.fail_to_pass == ["tests/test_models.py::test_save"]
    assert task.pass_to_pass == []
    assert task.image_name == "swebench/sweb.eval.x86_64.django__django-11001"
    assert task.needs_prepare is True

    # New function-call fields are empty defaults for SWE-patch rows.
    assert task.tools == []
    assert task.question == []
    assert task.ground_truth == []
    assert task.category is None
