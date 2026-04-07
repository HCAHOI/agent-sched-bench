"""Tests for :class:`agents.benchmarks.bfcl_runner.BFCLRunner`.

Covers:
- ``_ast_match`` rules across all single-turn BFCL category types
- End-to-end ``run_task`` with a mocked LLM provider
- Trace file shape (metadata + llm_call + summary records)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from agents.benchmarks.bfcl_runner import BFCLRunner
from agents.openclaw.eval.types import EvalTask
from agents.openclaw.providers.base import LLMProvider, LLMResponse, ToolCallRequest


# ── _ast_match rules ───────────────────────────────────────────────────


def test_ast_match_simple_exact_call() -> None:
    predicted = [{"name": "add", "arguments": {"a": 2, "b": 3}}]
    gt = [{"add": {"a": [2], "b": [3]}}]
    assert BFCLRunner._ast_match(predicted, gt, category="simple") is True


def test_ast_match_wrong_function_name() -> None:
    predicted = [{"name": "subtract", "arguments": {"a": 2, "b": 3}}]
    gt = [{"add": {"a": [2], "b": [3]}}]
    assert BFCLRunner._ast_match(predicted, gt, category="simple") is False


def test_ast_match_wrong_arg_value() -> None:
    predicted = [{"name": "add", "arguments": {"a": 2, "b": 99}}]
    gt = [{"add": {"a": [2], "b": [3]}}]
    assert BFCLRunner._ast_match(predicted, gt, category="simple") is False


def test_ast_match_accepts_any_alternative_value() -> None:
    """Ground truth lists multiple acceptable values — any one matches."""
    predicted = [{"name": "greet", "arguments": {"name": "Alice", "lang": "en"}}]
    gt = [{"greet": {"name": ["Alice", "Bob"], "lang": ["en", "english"]}}]
    assert BFCLRunner._ast_match(predicted, gt, category="simple") is True


def test_ast_match_extra_optional_arg_allowed() -> None:
    """Predicted args may include keys not in the ground-truth spec."""
    predicted = [
        {"name": "add", "arguments": {"a": 2, "b": 3, "rounding": "ceil"}}
    ]
    gt = [{"add": {"a": [2], "b": [3]}}]
    assert BFCLRunner._ast_match(predicted, gt, category="simple") is True


def test_ast_match_missing_required_arg() -> None:
    predicted = [{"name": "add", "arguments": {"a": 2}}]
    gt = [{"add": {"a": [2], "b": [3]}}]
    assert BFCLRunner._ast_match(predicted, gt, category="simple") is False


def test_ast_match_irrelevance_requires_empty_predicted() -> None:
    """The irrelevance category is correct iff the model emits no calls."""
    assert (
        BFCLRunner._ast_match(predicted=[], ground_truth=[], category="irrelevance")
        is True
    )
    assert (
        BFCLRunner._ast_match(
            predicted=[{"name": "f", "arguments": {}}],
            ground_truth=[],
            category="irrelevance",
        )
        is False
    )


def test_ast_match_parallel_matches_as_set() -> None:
    """Parallel category: two calls in any order must both match."""
    predicted = [
        {"name": "sub", "arguments": {"a": 5, "b": 2}},
        {"name": "add", "arguments": {"a": 1, "b": 2}},
    ]
    gt = [
        {"add": {"a": [1], "b": [2]}},
        {"sub": {"a": [5], "b": [2]}},
    ]
    assert BFCLRunner._ast_match(predicted, gt, category="parallel") is True


def test_ast_match_parallel_mismatch_length() -> None:
    predicted = [{"name": "add", "arguments": {"a": 1, "b": 2}}]
    gt = [
        {"add": {"a": [1], "b": [2]}},
        {"sub": {"a": [5], "b": [2]}},
    ]
    assert BFCLRunner._ast_match(predicted, gt, category="parallel") is False


# ── run_task end-to-end with a mocked provider ────────────────────────


class _StubProvider(LLMProvider):
    """Deterministic provider returning a canned LLMResponse."""

    def __init__(self, response: LLMResponse) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def get_default_model(self) -> str:
        return "stub-model"

    async def chat(  # type: ignore[override]
        self,
        messages,
        tools=None,
        model=None,
        max_tokens=4096,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "model": model,
                "tool_choice": tool_choice,
            }
        )
        return self._response


def _make_task(tmp_path: Path, *, category: str, ground_truth: list) -> EvalTask:
    return EvalTask(
        instance_id=f"bfcl_{category}_0",
        problem_statement="What is 2 + 3?",
        workspace_dir=tmp_path / f"bfcl_{category}_0",
        tools=[
            {
                "name": "add",
                "description": "Adds two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
            }
        ],
        question=[[{"role": "user", "content": "What is 2 + 3?"}]],
        ground_truth=ground_truth,
        category=category,
    )


def test_run_task_populates_official_resolved_true(tmp_path: Path) -> None:
    """A correct model prediction yields official_resolved=True."""
    provider = _StubProvider(
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(id="c0", name="add", arguments={"a": 2, "b": 3})
            ],
            finish_reason="tool_calls",
            usage={"prompt_tokens": 50, "completion_tokens": 10},
        )
    )
    task = _make_task(
        tmp_path, category="simple", ground_truth=[{"add": {"a": [2], "b": [3]}}]
    )
    runner = BFCLRunner(
        provider=provider,
        workspace_base=tmp_path,
        model="stub-model",
    )
    result = asyncio.run(runner.run_task(task))

    assert result.official_resolved is True
    assert result.evaluation_report is not None
    assert result.evaluation_report["score"] == 1.0
    assert result.evaluation_report["category"] == "simple"
    assert result.tools_used == ["add"]
    assert result.usage == {"prompt_tokens": 50, "completion_tokens": 10}
    assert result.stop_reason == "completed"


def test_run_task_populates_official_resolved_false(tmp_path: Path) -> None:
    """A wrong prediction yields official_resolved=False with score=0.0."""
    provider = _StubProvider(
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(id="c0", name="add", arguments={"a": 2, "b": 99})
            ],
            finish_reason="tool_calls",
            usage={"prompt_tokens": 50, "completion_tokens": 10},
        )
    )
    task = _make_task(
        tmp_path, category="simple", ground_truth=[{"add": {"a": [2], "b": [3]}}]
    )
    runner = BFCLRunner(provider=provider, workspace_base=tmp_path, model="stub-model")
    result = asyncio.run(runner.run_task(task))

    assert result.official_resolved is False
    assert result.evaluation_report is not None
    assert result.evaluation_report["score"] == 0.0


def test_run_task_emits_trace_metadata_and_llm_call_action(tmp_path: Path) -> None:
    """Trace file contains exactly one trace_metadata, one llm_call
    action, and one summary — the expected v5 shape for single-turn BFCL."""
    provider = _StubProvider(
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(id="c0", name="add", arguments={"a": 2, "b": 3})
            ],
            finish_reason="tool_calls",
            usage={"prompt_tokens": 42, "completion_tokens": 7},
        )
    )
    task = _make_task(
        tmp_path, category="simple", ground_truth=[{"add": {"a": [2], "b": [3]}}]
    )
    runner = BFCLRunner(provider=provider, workspace_base=tmp_path, model="stub-model")
    asyncio.run(runner.run_task(task))

    trace_file = task.workspace_dir / "trace.jsonl"
    assert trace_file.exists()

    records = [
        json.loads(line)
        for line in trace_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 3

    meta = records[0]
    assert meta["type"] == "trace_metadata"
    assert meta["trace_format_version"] == 5
    assert meta["scaffold"] == "openclaw"
    assert meta["benchmark"] == "bfcl-v4"  # plugin default when self.benchmark is None
    assert meta["category"] == "simple"

    action = records[1]
    assert action["type"] == "action"
    assert action["action_type"] == "llm_call"
    assert action["data"]["prompt_tokens"] == 42
    assert action["data"]["completion_tokens"] == 7
    assert action["data"]["llm_latency_ms"] > 0

    summary = records[2]
    assert summary["type"] == "summary"
    assert summary["success"] is True
    assert summary["total_tokens"] == 49


def test_run_task_irrelevance_with_empty_predicted(tmp_path: Path) -> None:
    """Irrelevance category is correct when the model emits no calls."""
    provider = _StubProvider(
        LLMResponse(
            content="I cannot answer with the provided tools.",
            tool_calls=[],
            finish_reason="stop",
            usage={"prompt_tokens": 20, "completion_tokens": 8},
        )
    )
    task = _make_task(tmp_path, category="irrelevance", ground_truth=[])
    runner = BFCLRunner(provider=provider, workspace_base=tmp_path, model="stub-model")
    result = asyncio.run(runner.run_task(task))
    assert result.official_resolved is True


def test_run_task_llm_error_returns_structured_failure(tmp_path: Path) -> None:
    """When provider.chat raises, the runner returns a structured EvalResult
    with stop_reason='llm_error' and a written trace summary."""

    class _BrokenProvider(LLMProvider):
        def get_default_model(self) -> str:
            return "broken"

        async def chat(  # type: ignore[override]
            self,
            messages,
            tools=None,
            model=None,
            max_tokens=4096,
            temperature=0.7,
            reasoning_effort=None,
            tool_choice=None,
        ):
            raise RuntimeError("upstream down")

    task = _make_task(
        tmp_path, category="simple", ground_truth=[{"add": {"a": [2], "b": [3]}}]
    )
    runner = BFCLRunner(provider=_BrokenProvider(), workspace_base=tmp_path, model="broken")
    result = asyncio.run(runner.run_task(task))

    assert result.stop_reason == "llm_error"
    assert result.official_resolved is False
    assert result.error is not None
    assert "upstream down" in result.error
    assert result.evaluation_report is not None
    assert result.evaluation_report["score"] == 0.0
