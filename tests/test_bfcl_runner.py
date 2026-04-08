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


def test_ast_match_live_irrelevance_requires_empty_predicted() -> None:
    """live_irrelevance must use the same abstention rule as irrelevance."""
    # reviewer C2: live_irrelevance previously fell through to generic length-match
    assert (
        BFCLRunner._ast_match(
            predicted=[], ground_truth=[], category="live_irrelevance"
        )
        is True
    )
    assert (
        BFCLRunner._ast_match(
            predicted=[{"name": "f", "arguments": {}}],
            ground_truth=[],
            category="live_irrelevance",
        )
        is False
    )


def test_ast_match_irrelevance_with_nonempty_ground_truth_fails_loudly() -> None:
    """Schema-drift guard: if BFCL ever ships an 'irrelevance' task with
    non-empty ground truth, treat it as a mismatch rather than silently
    returning True. The runner asserts both predicted and ground_truth
    are empty for the shortcut to fire."""
    assert (
        BFCLRunner._ast_match(
            predicted=[],
            ground_truth=[{"foo": {"x": [1]}}],
            category="irrelevance",
        )
        is False
    )


def test_ast_match_empty_alternatives_is_a_known_limitation() -> None:
    """BFCL's full spec includes a '[don't care]' wildcard / empty-list
    alternative that accepts any predicted value. The v1 runner does NOT
    implement this wildcard — this test pins the current (limited)
    behavior so future upgrades notice the regression when they lift the
    limitation. See bfcl_runner.py module docstring 'Known limitations'.
    """
    # With the v1 implementation, empty alternatives can never match.
    assert (
        BFCLRunner._ast_match(
            predicted=[{"name": "f", "arguments": {"x": 42}}],
            ground_truth=[{"f": {"x": []}}],
            category="simple_python",
        )
        is False
    ), (
        "If this assertion starts failing, the dont_care wildcard "
        "has been implemented — update the runner docstring and "
        "delete this test in favor of the positive-match version."
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
        # Calling the real base __init__ wires up api_key, api_base,
        # and self.generation — without this the test is a time bomb
        # the moment any code path reads those attributes.
        super().__init__(api_key="test", api_base="http://test")
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
        tmp_path, category="simple_python", ground_truth=[{"add": {"a": [2], "b": [3]}}]
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
    assert result.evaluation_report["category"] == "simple_python"
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
        tmp_path, category="simple_python", ground_truth=[{"add": {"a": [2], "b": [3]}}]
    )
    runner = BFCLRunner(provider=provider, workspace_base=tmp_path, model="stub-model")
    result = asyncio.run(runner.run_task(task))

    assert result.official_resolved is False
    assert result.evaluation_report is not None
    assert result.evaluation_report["score"] == 0.0


def test_run_task_evaluation_report_round_trips(tmp_path: Path) -> None:
    """evaluation_report must carry category + per-task score breakdown
    so downstream analysis can read it from results.jsonl without re-walking traces.

    Previously dropped at the collector boundary (reviewer C3).
    """
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
        tmp_path,
        category="simple_python",
        ground_truth=[{"add": {"a": [2], "b": [3]}}],
    )
    runner = BFCLRunner(provider=provider, workspace_base=tmp_path, model="stub-model")
    result = asyncio.run(runner.run_task(task))

    report = result.evaluation_report
    assert report is not None
    assert report["category"] == "simple_python"
    assert report["score"] == 1.0
    assert report["predicted_calls"] == [{"name": "add", "arguments": {"a": 2, "b": 3}}]
    assert report["ground_truth"] == [{"add": {"a": [2], "b": [3]}}]


def test_run_task_emits_scheduling_events_via_session_runner(
    tmp_path: Path,
) -> None:
    """v2 trace shape: routing through SessionRunner produces MORE than
    the v1 baseline of 3 records (metadata + single llm_call + summary).
    Must contain trace_metadata with the custom-registry sentinel, at
    least one llm_call action (with usage populated), at least one
    llm_call_start event, and at least one tool_exec_start event when
    the model dispatches a tool. This pins the architectural contract
    that single-turn BFCL now walks the full openclaw scheduling path.
    """
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
        tmp_path,
        category="simple_python",
        ground_truth=[{"add": {"a": [2], "b": [3]}}],
    )
    runner = BFCLRunner(
        provider=provider, workspace_base=tmp_path, model="stub-model"
    )
    asyncio.run(runner.run_task(task))

    trace_file = task.workspace_dir / "trace.jsonl"
    assert trace_file.exists()

    records = [
        json.loads(line)
        for line in trace_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # v1 baseline was exactly 3 records. v2 emits many more (events +
    # actions) because it walks the openclaw scheduling path.
    assert len(records) > 3, (
        f"v2 trace should have >3 records, got {len(records)}"
    )

    meta = records[0]
    assert meta["type"] == "trace_metadata"
    assert meta["trace_format_version"] == 5
    assert meta["scaffold"] == "openclaw"
    caps = meta["scaffold_capabilities"]
    # Custom-registry auto-derive sentinel (Phase 0B contract).
    assert caps.get("source") == "custom_registry"
    # The BFCL tool is registered, not bash/file_read.
    assert "add" in caps["tools"]
    assert "bash" not in caps["tools"]

    # Find the llm_call action and assert usage was captured.
    llm_actions = [
        r for r in records
        if r.get("type") == "action" and r.get("action_type") == "llm_call"
    ]
    assert len(llm_actions) >= 1, "expected at least one llm_call action"
    llm_action = llm_actions[0]
    assert llm_action["data"]["prompt_tokens"] == 42
    assert llm_action["data"]["completion_tokens"] == 7

    # Scheduling events from SessionRunner / TraceCollectorHook.
    event_names = [r.get("event") for r in records if r.get("type") == "event"]
    assert "llm_call_start" in event_names, (
        f"expected llm_call_start event; got events={event_names}"
    )
    # The model emitted a tool_call → dispatch fired → tool_exec event.
    assert "tool_exec_start" in event_names, (
        f"expected tool_exec_start event; got events={event_names}"
    )


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


def test_run_task_llm_error_yields_score_zero_via_absorbed_error(
    tmp_path: Path,
) -> None:
    """When provider.chat raises, SessionRunner's AgentLoop catches the
    exception internally and surfaces it as an "LLM returned error"
    trace event — the session completes normally with an empty recorder.

    In v2 this manifests as stop_reason='completed', official_resolved=
    False, and predicted_calls=[]. This is the correct behavior: a
    provider error is semantically "the model couldn't produce a
    prediction", which scores False under _ast_match. The error itself
    still lands in the trace file for debugging.
    """

    class _BrokenProvider(LLMProvider):
        def __init__(self) -> None:
            super().__init__(api_key="test", api_base="http://test")

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
        tmp_path,
        category="simple_python",
        ground_truth=[{"add": {"a": [2], "b": [3]}}],
    )
    runner = BFCLRunner(
        provider=_BrokenProvider(), workspace_base=tmp_path, model="broken"
    )
    result = asyncio.run(runner.run_task(task))

    # v2: session absorbs the provider error; stop_reason is "completed"
    # (the session itself ran to completion, it just produced no tool
    # calls) and the score is 0 because the recorder is empty.
    assert result.stop_reason == "completed"
    assert result.official_resolved is False
    assert result.evaluation_report is not None
    assert result.evaluation_report["score"] == 0.0
    # predicted_calls empty because the recorder never got hit.
    assert json.loads(result.content) == []
    # Debuggability: EvalResult.error must carry the absorbed LLM error
    # message so downstream analysis can distinguish "wrong answer" from
    # "model crashed" without re-walking the trace file (reviewer v2-M3).
    assert result.error is not None
    assert "upstream down" in result.error
