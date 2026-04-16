"""Integration tests for ResearchAgentRunner."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agents.base import TraceAction
from agents.research_agent.runner import ResearchAgentRunner
from trace_collect.attempt_pipeline import AttemptContext, AttemptResult

# ---------------------------------------------------------------------------
# Mock streaming LLM client (same pattern as phases test)
# ---------------------------------------------------------------------------


class _MockDelta:
    def __init__(self, content: str | None = None) -> None:
        self.content = content


class _MockChoice:
    def __init__(self, delta: _MockDelta, finish_reason: str | None = None) -> None:
        self.delta = delta
        self.finish_reason = finish_reason


class _MockUsage:
    def __init__(self, prompt_tokens: int = 10, completion_tokens: int = 5) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _MockChunk:
    def __init__(
        self,
        choices: list[_MockChoice] | None = None,
        usage: _MockUsage | None = None,
    ) -> None:
        self.choices = choices or []
        self.usage = usage


class _AsyncStream:
    def __init__(self, chunks: list[_MockChunk]) -> None:
        self._chunks = chunks
        self._idx = 0

    def __aiter__(self) -> _AsyncStream:
        return self

    async def __anext__(self) -> _MockChunk:
        if self._idx >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._idx]
        self._idx += 1
        return chunk


class _SequentialCompletions:
    """Return different responses for sequential LLM calls."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._call_idx = 0
        self.call_kwargs: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _AsyncStream:
        self.call_kwargs.append(kwargs)
        idx = min(self._call_idx, len(self._responses) - 1)
        text = self._responses[idx]
        self._call_idx += 1

        chunks: list[_MockChunk] = []
        for char in text:
            chunks.append(
                _MockChunk(choices=[_MockChoice(_MockDelta(content=char))])
            )
        chunks.append(
            _MockChunk(
                choices=[_MockChoice(_MockDelta(content=None), finish_reason="stop")]
            )
        )
        chunks.append(_MockChunk(usage=_MockUsage()))
        return _AsyncStream(chunks)


class _MockChat:
    def __init__(self, completions: _SequentialCompletions) -> None:
        self.completions = completions


class _MockClient:
    def __init__(self, responses: list[str]) -> None:
        self.chat = _MockChat(_SequentialCompletions(responses))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PLAN_RESPONSE = "what is quantum computing\nquantum computing applications"
_EXTRACT_RESPONSE = (
    '{"source_url": "https://example.com/q1", "passage": "QC is fast", "relevance_note": "core"}\n'
)
_SYNTH_RESPONSE = "Quantum computing uses qubits."


def _make_attempt_ctx(tmp_path: Path, instance_id: str = "test-instance") -> AttemptContext:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = AttemptContext(
        run_dir=run_dir,
        instance_id=instance_id,
        attempt=1,
        task={},
        model="test-model",
        scaffold="research-agent",
        source_image=None,
        execution_environment="host",
    )
    ctx.attempt_dir.mkdir(parents=True, exist_ok=True)
    return ctx


def _make_task(
    problem_statement: str = "What is quantum computing?",
    reference_answer: str | None = None,
) -> dict[str, Any]:
    task: dict[str, Any] = {"problem_statement": problem_statement}
    if reference_answer is not None:
        task["reference_answer"] = reference_answer
    return task


def _build_runner(
    client: Any,
    benchmark_slug: str = "browsecomp",
) -> ResearchAgentRunner:
    return ResearchAgentRunner(
        model="test-model",
        api_base="http://localhost:8000/v1",
        api_key="test-key",
        max_iterations=5,
        benchmark_slug=benchmark_slug,
        client=client,
    )


def _patch_tools_no_network():
    """Patch TracedWebSearch and TracedWebFetch to avoid real network calls."""
    from unittest.mock import patch

    async def _fake_search_execute(
        self: Any,
        query: str,
        *,
        action_id: str = "tool_search_0",
        agent_id: str = "",
        instance_id: str = "",
        iteration: int = 0,
    ) -> TraceAction:
        return TraceAction(
            action_type="tool_exec",
            action_id=action_id,
            agent_id=agent_id,
            instance_id=instance_id,
            iteration=iteration,
            ts_start=1.0,
            ts_end=2.0,
            data={
                "tool_name": "web_search",
                "args": {"query": query},
                "result": f"Results for: {query}\n\n1. QC Page\n   https://example.com/{query.replace(' ', '_')}",
                "duration_ms": 50.0,
                "error": None,
            },
        )

    async def _fake_fetch_execute(
        self: Any,
        url: str,
        *,
        action_id: str = "tool_fetch_0",
        agent_id: str = "",
        instance_id: str = "",
        iteration: int = 0,
    ) -> TraceAction:
        return TraceAction(
            action_type="tool_exec",
            action_id=action_id,
            agent_id=agent_id,
            instance_id=instance_id,
            iteration=iteration,
            ts_start=2.0,
            ts_end=3.0,
            data={
                "tool_name": "web_fetch",
                "args": {"url": url},
                "result": json.dumps({"text": f"Content from {url}", "url": url, "status": 200}),
                "duration_ms": 80.0,
                "error": None,
            },
        )

    return (
        patch.object(
            __import__("agents.research_agent.tools", fromlist=["TracedWebSearch"]).TracedWebSearch,
            "execute",
            _fake_search_execute,
        ),
        patch.object(
            __import__("agents.research_agent.tools", fromlist=["TracedWebFetch"]).TracedWebFetch,
            "execute",
            _fake_fetch_execute,
        ),
    )


def _read_trace(attempt_dir: Path) -> list[dict[str, Any]]:
    trace_path = attempt_dir / "trace.jsonl"
    records: list[dict[str, Any]] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_research_agent_runner_writes_v5_trace(tmp_path: Path) -> None:
    client = _MockClient([_PLAN_RESPONSE, _EXTRACT_RESPONSE, _SYNTH_RESPONSE])
    runner = _build_runner(client)
    ctx = _make_attempt_ctx(tmp_path)
    task = _make_task()

    search_patch, fetch_patch = _patch_tools_no_network()
    with search_patch, fetch_patch:
        result = asyncio.run(
            runner.run_task(task, attempt_ctx=ctx, prompt_template="default")
        )

    assert result.success
    assert result.exit_status == "completed"
    assert result.trace_path.exists()

    records = _read_trace(ctx.attempt_dir)

    # First record is trace_metadata
    assert records[0]["type"] == "trace_metadata"
    assert records[0]["scaffold"] == "research-agent"
    assert records[0]["trace_format_version"] == 5

    # Should have LLM calls (plan, extract, synthesize = 3)
    llm_calls = [r for r in records if r.get("action_type") == "llm_call"]
    assert len(llm_calls) >= 3

    # Should have tool_exec actions (search + fetch)
    tool_execs = [r for r in records if r.get("action_type") == "tool_exec"]
    assert len(tool_execs) >= 2

    # Should have a summary
    summaries = [r for r in records if r.get("type") == "summary"]
    assert len(summaries) == 1


def test_research_agent_no_reference_leak(tmp_path: Path) -> None:
    """reference_answer must NEVER appear in any messages sent to the LLM."""
    client = _MockClient([_PLAN_RESPONSE, _EXTRACT_RESPONSE, _SYNTH_RESPONSE])
    runner = _build_runner(client)
    ctx = _make_attempt_ctx(tmp_path)
    task = _make_task(reference_answer="SECRET_ANSWER_42")

    search_patch, fetch_patch = _patch_tools_no_network()
    with search_patch, fetch_patch:
        result = asyncio.run(
            runner.run_task(task, attempt_ctx=ctx, prompt_template="default")
        )

    assert result.success
    records = _read_trace(ctx.attempt_dir)

    # Check every LLM call's messages_in
    for record in records:
        if record.get("action_type") != "llm_call":
            continue
        data = record.get("data", {})
        messages_in = data.get("messages_in", [])
        for msg in messages_in:
            content = msg.get("content", "")
            assert "SECRET_ANSWER_42" not in content, (
                f"reference_answer leaked into messages_in of {record.get('action_id')}"
            )
        # Also check content field
        assert "SECRET_ANSWER_42" not in str(data.get("content", ""))


def test_research_agent_empty_search_graceful(tmp_path: Path) -> None:
    """If search returns empty results, runner should skip fetch/extract."""
    client = _MockClient([_PLAN_RESPONSE, _SYNTH_RESPONSE])
    runner = _build_runner(client)
    ctx = _make_attempt_ctx(tmp_path)
    task = _make_task()

    # Patch search to return no URLs
    async def _empty_search(
        self: Any, query: str, *, action_id: str = "", agent_id: str = "",
        instance_id: str = "", iteration: int = 0,
    ) -> TraceAction:
        return TraceAction(
            action_type="tool_exec",
            action_id=action_id,
            agent_id=agent_id,
            instance_id=instance_id,
            iteration=iteration,
            ts_start=1.0, ts_end=1.5,
            data={
                "tool_name": "web_search",
                "args": {"query": query},
                "result": f"No results for: {query}",
                "duration_ms": 20.0,
                "error": None,
            },
        )

    from unittest.mock import patch
    from agents.research_agent.tools import TracedWebSearch

    with patch.object(TracedWebSearch, "execute", _empty_search):
        result = asyncio.run(
            runner.run_task(task, attempt_ctx=ctx, prompt_template="default")
        )

    assert result.success
    records = _read_trace(ctx.attempt_dir)

    # Should still have synthesize LLM call
    llm_calls = [r for r in records if r.get("action_type") == "llm_call"]
    assert len(llm_calls) >= 2  # plan + synthesize at minimum

    # No fetch actions
    fetch_actions = [
        r for r in records
        if r.get("action_type") == "tool_exec"
        and r.get("data", {}).get("tool_name") == "web_fetch"
    ]
    assert len(fetch_actions) == 0


def test_research_agent_fetch_failure_graceful(tmp_path: Path) -> None:
    """If fetch tool errors, runner should still continue to synthesize."""
    client = _MockClient([_PLAN_RESPONSE, _EXTRACT_RESPONSE, _SYNTH_RESPONSE])
    runner = _build_runner(client)
    ctx = _make_attempt_ctx(tmp_path)
    task = _make_task()

    # Search returns URLs but fetch errors
    async def _error_fetch(
        self: Any, url: str, *, action_id: str = "", agent_id: str = "",
        instance_id: str = "", iteration: int = 0,
    ) -> TraceAction:
        return TraceAction(
            action_type="tool_exec",
            action_id=action_id,
            agent_id=agent_id,
            instance_id=instance_id,
            iteration=iteration,
            ts_start=2.0, ts_end=2.5,
            data={
                "tool_name": "web_fetch",
                "args": {"url": url},
                "result": "Error: connection refused",
                "duration_ms": 10.0,
                "error": "connection refused",
            },
        )

    search_patch, _ = _patch_tools_no_network()
    from agents.research_agent.tools import TracedWebFetch
    from unittest.mock import patch

    with search_patch, patch.object(TracedWebFetch, "execute", _error_fetch):
        result = asyncio.run(
            runner.run_task(task, attempt_ctx=ctx, prompt_template="default")
        )

    assert result.success
    records = _read_trace(ctx.attempt_dir)

    # Should have a synthesize call even though fetch failed
    synth_actions = [
        r for r in records
        if r.get("action_type") == "llm_call"
        and r.get("action_id", "").startswith("llm_synthesize")
    ]
    assert len(synth_actions) == 1


def test_research_agent_iteration_numbering(tmp_path: Path) -> None:
    """Verify plan=0, search=1, fetch=2, extract=3, synthesize=4."""
    client = _MockClient([_PLAN_RESPONSE, _EXTRACT_RESPONSE, _SYNTH_RESPONSE])
    runner = _build_runner(client)
    ctx = _make_attempt_ctx(tmp_path)
    task = _make_task()

    search_patch, fetch_patch = _patch_tools_no_network()
    with search_patch, fetch_patch:
        result = asyncio.run(
            runner.run_task(task, attempt_ctx=ctx, prompt_template="default")
        )

    assert result.success
    records = _read_trace(ctx.attempt_dir)

    action_records = [r for r in records if r.get("type") == "action"]
    iteration_by_id: dict[str, int] = {}
    for r in action_records:
        iteration_by_id[r["action_id"]] = r["iteration"]

    # Plan LLM call at iteration 0
    assert iteration_by_id.get("llm_plan_0") == 0
    # Search tool calls at iteration 1
    for aid, it in iteration_by_id.items():
        if aid.startswith("tool_search_"):
            assert it == 1, f"{aid} has iteration {it}, expected 1"
    # Fetch tool calls at iteration 2
    for aid, it in iteration_by_id.items():
        if aid.startswith("tool_fetch_"):
            assert it == 2, f"{aid} has iteration {it}, expected 2"
    # Extract LLM call at iteration 3
    assert iteration_by_id.get("llm_extract_0") == 3
    # Synthesize LLM call at iteration 4
    assert iteration_by_id.get("llm_synthesize_0") == 4


def test_research_agent_action_id_uniqueness(tmp_path: Path) -> None:
    """All action_ids must be unique within a trace."""
    client = _MockClient([_PLAN_RESPONSE, _EXTRACT_RESPONSE, _SYNTH_RESPONSE])
    runner = _build_runner(client)
    ctx = _make_attempt_ctx(tmp_path)
    task = _make_task()

    search_patch, fetch_patch = _patch_tools_no_network()
    with search_patch, fetch_patch:
        result = asyncio.run(
            runner.run_task(task, attempt_ctx=ctx, prompt_template="default")
        )

    assert result.success
    records = _read_trace(ctx.attempt_dir)

    action_ids = [
        r["action_id"]
        for r in records
        if r.get("type") == "action" and "action_id" in r
    ]
    assert len(action_ids) == len(set(action_ids)), (
        f"Duplicate action_ids found: {[x for x in action_ids if action_ids.count(x) > 1]}"
    )


def test_research_agent_summary_aggregation(tmp_path: Path) -> None:
    """Summary totals must match action-level sums."""
    client = _MockClient([_PLAN_RESPONSE, _EXTRACT_RESPONSE, _SYNTH_RESPONSE])
    runner = _build_runner(client)
    ctx = _make_attempt_ctx(tmp_path)
    task = _make_task()

    search_patch, fetch_patch = _patch_tools_no_network()
    with search_patch, fetch_patch:
        result = asyncio.run(
            runner.run_task(task, attempt_ctx=ctx, prompt_template="default")
        )

    assert result.success
    records = _read_trace(ctx.attempt_dir)

    summary = [r for r in records if r.get("type") == "summary"][0]
    action_records = [r for r in records if r.get("type") == "action"]

    # Verify total_tokens matches sum from llm_call actions
    llm_actions = [r for r in action_records if r.get("action_type") == "llm_call"]
    expected_tokens = sum(
        (r["data"].get("prompt_tokens") or 0) + (r["data"].get("completion_tokens") or 0)
        for r in llm_actions
    )
    assert summary["total_tokens"] == expected_tokens

    # Verify total_tool_ms matches sum from tool_exec actions
    tool_actions = [r for r in action_records if r.get("action_type") == "tool_exec"]
    expected_tool_ms = sum(r["data"].get("duration_ms", 0) for r in tool_actions)
    assert abs(summary["total_tool_ms"] - expected_tool_ms) < 0.01

    # Verify n_iterations
    iterations = {r["iteration"] for r in action_records}
    assert summary["n_iterations"] == len(iterations)


def test_research_agent_runner_marks_failed_task_unsuccessful(tmp_path: Path) -> None:
    """Regression: failures must set summary.success=False (was hardcoded True)."""
    # Mock client that raises on the first LLM call -> plan phase explodes
    from unittest.mock import MagicMock
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()

    async def _boom(**_kwargs: Any) -> Any:
        raise RuntimeError("upstream LLM failure")

    client.chat.completions.create = _boom

    runner = _build_runner(client)
    ctx = _make_attempt_ctx(tmp_path)
    task = _make_task()

    result = asyncio.run(
        runner.run_task(task, attempt_ctx=ctx, prompt_template="default")
    )

    assert result.success is False
    assert result.exit_status == "error"

    records = _read_trace(ctx.attempt_dir)
    summary = next(r for r in records if r.get("type") == "summary")
    # The bug: summary["success"] used to be hardcoded True even on failure.
    assert summary["success"] is False
    assert summary["final_answer"] == ""


def test_research_agent_runner_uses_rendered_prompt_template(tmp_path: Path) -> None:
    """Regression: phases must see the rendered benchmark prompt, not raw problem_statement."""
    # browsecomp default.md contains the unique string "browsing-comprehension"
    # which is only present after render_research_prompt() runs. If we see it
    # in messages_in, we know the rendered prompt reached the phase.
    client = _MockClient([_PLAN_RESPONSE, _EXTRACT_RESPONSE, _SYNTH_RESPONSE])
    runner = _build_runner(client, benchmark_slug="browsecomp")
    ctx = _make_attempt_ctx(tmp_path)
    task = _make_task(problem_statement="What is the capital of France?")

    search_patch, fetch_patch = _patch_tools_no_network()
    with search_patch, fetch_patch:
        asyncio.run(
            runner.run_task(task, attempt_ctx=ctx, prompt_template="default")
        )

    records = _read_trace(ctx.attempt_dir)
    llm_actions = [r for r in records if r.get("action_type") == "llm_call"]
    assert llm_actions, "expected at least one LLM call"

    # The browsecomp default prompt template contains "browsing-comprehension".
    # If phases received only the raw problem_statement, this template text
    # would never appear in messages_in.
    found_template_text = False
    for rec in llm_actions:
        for msg in rec["data"].get("messages_in", []):
            content = msg.get("content", "") or ""
            if "browsing-comprehension" in content:
                found_template_text = True
                break
        if found_template_text:
            break
    assert found_template_text, (
        "Rendered benchmark prompt template must appear in at least one "
        "phase's messages_in (otherwise prompt_template is a no-op)."
    )
