"""Unit tests for research-agent phases and traced tool wrappers."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from agents.base import TraceAction
from agents.research_agent.evidence import Evidence
from agents.research_agent.phases import (
    ExtractPhase,
    FetchPhase,
    PlanPhase,
    SearchPhase,
    SynthesizePhase,
)
from agents.research_agent.tools import TracedWebFetch, TracedWebSearch

# ---------------------------------------------------------------------------
# Mock streaming LLM client
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
    """Simulate an async iterator of SSE chunks."""

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


class _MockCompletions:
    def __init__(self, response_text: str, usage: _MockUsage | None = None) -> None:
        self._response_text = response_text
        self._usage = usage or _MockUsage()

    async def create(self, **kwargs: Any) -> _AsyncStream:
        # Build chunks: one per token (character) + usage-only final chunk
        chunks: list[_MockChunk] = []
        for i, char in enumerate(self._response_text):
            chunks.append(
                _MockChunk(
                    choices=[_MockChoice(_MockDelta(content=char))],
                )
            )
        # Final chunk with finish_reason
        chunks.append(
            _MockChunk(
                choices=[_MockChoice(_MockDelta(content=None), finish_reason="stop")],
            )
        )
        # Usage-only chunk
        chunks.append(_MockChunk(usage=self._usage))
        return _AsyncStream(chunks)


class _MockChat:
    def __init__(self, completions: _MockCompletions) -> None:
        self.completions = completions


class _MockClient:
    def __init__(self, response_text: str, usage: _MockUsage | None = None) -> None:
        self.chat = _MockChat(_MockCompletions(response_text, usage))


# ---------------------------------------------------------------------------
# PlanPhase
# ---------------------------------------------------------------------------


def test_plan_phase_generates_queries() -> None:
    client = _MockClient("query1\nquery2\n")
    phase = PlanPhase(client, "test-model", agent_id="a", instance_id="i")
    queries, actions = asyncio.run(phase.execute("What is X?"))
    assert queries == ["query1", "query2"]
    assert len(actions) == 1
    assert actions[0].action_type == "llm_call"
    assert actions[0].action_id == "llm_plan_0"
    assert actions[0].iteration == 0


# ---------------------------------------------------------------------------
# SearchPhase
# ---------------------------------------------------------------------------


def test_search_phase_traces_tool_exec() -> None:
    mock_search = TracedWebSearch.__new__(TracedWebSearch)

    async def _fake_execute(
        query: str,
        *,
        action_id: str = "",
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
                "result": f"Results for: {query}\n\n1. Title\n   https://example.com/{query}",
                "duration_ms": 100.0,
                "error": None,
            },
        )

    mock_search.execute = _fake_execute  # type: ignore[assignment]
    phase = SearchPhase(mock_search, agent_id="a", instance_id="i")
    results, actions = asyncio.run(phase.execute(["q1", "q2"]))
    assert len(actions) == 2
    assert all(a.action_type == "tool_exec" for a in actions)
    assert actions[0].action_id == "tool_search_0"
    assert actions[1].action_id == "tool_search_1"
    assert all(a.iteration == 1 for a in actions)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# FetchPhase
# ---------------------------------------------------------------------------


def test_fetch_phase_traces_tool_exec() -> None:
    mock_fetch = TracedWebFetch.__new__(TracedWebFetch)

    async def _fake_execute(
        url: str,
        *,
        action_id: str = "",
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
                "tool_name": "web_fetch",
                "args": {"url": url},
                "result": '{"text": "page content", "url": "' + url + '", "status": 200}',
                "duration_ms": 200.0,
                "error": None,
            },
        )

    mock_fetch.execute = _fake_execute  # type: ignore[assignment]
    phase = FetchPhase(mock_fetch, agent_id="a", instance_id="i")
    pages, actions = asyncio.run(phase.execute(["https://a.com", "https://b.com"]))
    assert len(actions) == 2
    assert all(a.action_type == "tool_exec" for a in actions)
    assert actions[0].action_id == "tool_fetch_0"
    assert actions[1].action_id == "tool_fetch_1"
    assert all(a.iteration == 2 for a in actions)
    assert len(pages) == 2
    assert pages[0]["text"] == "page content"


# ---------------------------------------------------------------------------
# ExtractPhase
# ---------------------------------------------------------------------------


def test_extract_phase_produces_evidence() -> None:
    response = (
        '{"source_url": "https://a.com", "passage": "fact A", "relevance_note": "relevant"}\n'
        '{"source_url": "https://b.com", "passage": "fact B", "relevance_note": "somewhat relevant"}\n'
    )
    client = _MockClient(response)
    phase = ExtractPhase(client, "test-model", agent_id="a", instance_id="i")
    fetched = [
        {"url": "https://a.com", "text": "content A", "fetch_timestamp": 100.0},
        {"url": "https://b.com", "text": "content B", "fetch_timestamp": 200.0},
    ]
    evidence, actions = asyncio.run(phase.execute("What is X?", fetched))
    assert len(evidence) == 2
    assert isinstance(evidence[0], Evidence)
    assert evidence[0].source_url == "https://a.com"
    assert evidence[0].passage == "fact A"
    assert evidence[0].fetch_timestamp == 100.0
    assert evidence[1].fetch_timestamp == 200.0
    assert len(actions) == 1
    assert actions[0].action_type == "llm_call"
    assert actions[0].iteration == 3


# ---------------------------------------------------------------------------
# SynthesizePhase
# ---------------------------------------------------------------------------


def test_synthesize_phase_uses_evidence() -> None:
    client = _MockClient("The answer is 42.")
    phase = SynthesizePhase(client, "test-model", agent_id="a", instance_id="i")
    evidence = [
        Evidence(
            source_url="https://a.com",
            passage="fact A",
            relevance_note="key",
            search_query="q",
            fetch_timestamp=100.0,
        ),
    ]
    answer, actions = asyncio.run(phase.execute("What is X?", evidence))
    assert answer == "The answer is 42."
    assert len(actions) == 1
    assert actions[0].action_type == "llm_call"
    assert actions[0].iteration == 4
    # Verify evidence appears in the messages sent to the LLM
    messages_in = actions[0].data.get("messages_in", [])
    user_msg = messages_in[-1]["content"]
    assert "fact A" in user_msg
    assert "https://a.com" in user_msg


# ---------------------------------------------------------------------------
# TracedWebSearch exception handling
# ---------------------------------------------------------------------------


def test_traced_web_search_catches_exceptions() -> None:
    tool = TracedWebSearch.__new__(TracedWebSearch)
    tool._timeout_s = 5.0  # type: ignore[attr-defined]
    # Create a mock inner tool that raises
    mock_inner = AsyncMock(side_effect=RuntimeError("network down"))
    tool._tool = type("FakeTool", (), {"execute": mock_inner})()  # type: ignore[assignment]
    action = asyncio.run(
        tool.execute("test query", action_id="tool_search_err")
    )
    assert isinstance(action, TraceAction)
    assert action.action_type == "tool_exec"
    assert action.data["error"] == "network down"
    assert "Error" in action.data["result"]
    # No exception propagated


# ---------------------------------------------------------------------------
# TracedWebFetch exception handling
# ---------------------------------------------------------------------------


def test_traced_web_fetch_catches_exceptions() -> None:
    tool = TracedWebFetch.__new__(TracedWebFetch)
    tool._timeout_s = 5.0  # type: ignore[attr-defined]
    mock_inner = AsyncMock(side_effect=RuntimeError("fetch failed"))
    tool._tool = type("FakeTool", (), {"execute": mock_inner})()  # type: ignore[assignment]
    action = asyncio.run(
        tool.execute("https://bad.com", action_id="tool_fetch_err")
    )
    assert isinstance(action, TraceAction)
    assert action.action_type == "tool_exec"
    assert action.data["error"] == "fetch failed"
    assert "Error" in action.data["result"]
