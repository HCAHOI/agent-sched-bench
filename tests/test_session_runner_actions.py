"""Regression test for the OpenClaw TraceCollectorHook v4 emission bug.

US-010: Earlier the hook defined ``after_llm_response`` to emit the
``llm_call`` action, but ``AgentLoop``'s CompositeHook never invokes
that method — it only calls ``before_iteration``, ``before_execute_tools``
and ``after_iteration``. The result was traces with ``tool_exec`` actions
but ZERO ``llm_call`` actions, which broke Gantt rendering and the
simulator's iteration grouping.

This test drives ``TraceCollectorHook`` with synthetic ``AgentHookContext``
inputs through the realistic hook order and asserts that the JSONL trace
contains BOTH action types after a single iteration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# Skip the entire module if openclaw / minisweagent deps are unavailable.
pytest.importorskip("agents.openclaw._session_runner")

from agents.openclaw._session_runner import TraceCollectorHook, _resolve_run_outcome


class _StubResponse:
    def __init__(self, content: str = "", finish_reason: str = "stop") -> None:
        self.content = content
        self.finish_reason = finish_reason
        self.reasoning_content: str | None = None
        self.extra: dict[str, Any] | None = None


class _StubToolCall:
    def __init__(self, name: str, arguments: dict[str, Any]) -> None:
        self.name = name
        self.arguments = arguments


class _StubContext:
    """Mimics the bits of AgentHookContext the trace hook reads."""

    def __init__(
        self,
        iteration: int,
        messages: list[dict[str, Any]],
        tool_calls: list[_StubToolCall] | None = None,
        usage: dict[str, int] | None = None,
        response: _StubResponse | None = None,
    ) -> None:
        self.iteration = iteration
        self.messages = messages
        self.tool_calls = tool_calls or []
        self.usage = usage or {}
        self.response = response


def test_trace_collector_emits_llm_call_action(tmp_path: Path) -> None:
    import asyncio
    asyncio.run(_drive_emits_llm_call_action(tmp_path))


async def _drive_emits_llm_call_action(tmp_path: Path) -> None:
    """One iteration with one tool call must produce ONE llm_call action
    AND ONE tool_exec action — in that chronological order."""
    trace_file = tmp_path / "trace.jsonl"
    hook = TraceCollectorHook(trace_file, instance_id="test-1")

    # ── Iteration 0 ──────────────────────────────────────────────
    msgs_in = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Write hello world."},
    ]
    ctx_before = _StubContext(iteration=0, messages=msgs_in)
    await hook.before_iteration(ctx_before)

    # Simulate LLM response producing a tool call
    msgs_after_llm = msgs_in + [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "write_file", "arguments": "{\"path\":\"a.py\"}"}}
        ]}
    ]
    ctx_before_tools = _StubContext(
        iteration=0,
        messages=msgs_after_llm,
        tool_calls=[_StubToolCall("write_file", {"path": "a.py"})],
        usage={"prompt_tokens": 100, "completion_tokens": 20},
    )
    await hook.before_execute_tools(ctx_before_tools)

    # Simulate tool result appended to messages
    msgs_after_tool = msgs_after_llm + [
        {"role": "tool", "name": "write_file", "content": "wrote a.py"}
    ]
    ctx_after = _StubContext(
        iteration=0,
        messages=msgs_after_tool,
        tool_calls=[_StubToolCall("write_file", {"path": "a.py"})],
        usage={"prompt_tokens": 100, "completion_tokens": 20},
        response=_StubResponse(content="", finish_reason="tool_calls"),
    )
    await hook.after_iteration(ctx_after)
    hook.close()

    # ── Verify ──────────────────────────────────────────────────
    lines = trace_file.read_text().strip().splitlines()
    records = [json.loads(line) for line in lines]
    actions = [r for r in records if r.get("type") == "action"]

    llm_calls = [a for a in actions if a.get("action_type") == "llm_call"]
    tool_execs = [a for a in actions if a.get("action_type") == "tool_exec"]

    assert len(llm_calls) == 1, (
        f"Expected exactly 1 llm_call action, got {len(llm_calls)}. "
        f"All action types: {[a.get('action_type') for a in actions]}"
    )
    assert len(tool_execs) == 1, f"Expected 1 tool_exec, got {len(tool_execs)}"

    # Order: llm_call must come BEFORE tool_exec in the file
    llm_idx = next(i for i, r in enumerate(records)
                   if r.get("type") == "action" and r.get("action_type") == "llm_call")
    tool_idx = next(i for i, r in enumerate(records)
                    if r.get("type") == "action" and r.get("action_type") == "tool_exec")
    assert llm_idx < tool_idx, "llm_call action must precede tool_exec in trace"

    # Verify llm_call action carries the snapshotted messages_in (NOT the
    # post-tool-result messages — that would be a leak from after_iteration)
    llm = llm_calls[0]
    assert llm["data"]["messages_in"] == msgs_in
    assert llm["data"]["prompt_tokens"] == 100
    assert llm["data"]["completion_tokens"] == 20
    assert llm["iteration"] == 0
    assert llm["ts_start"] <= llm["ts_end"]


def test_trace_collector_llm_only_iteration(tmp_path: Path) -> None:
    import asyncio
    asyncio.run(_drive_llm_only_iteration(tmp_path))


async def _drive_llm_only_iteration(tmp_path: Path) -> None:
    """An iteration with NO tool calls (final answer) still emits llm_call."""
    trace_file = tmp_path / "trace.jsonl"
    hook = TraceCollectorHook(trace_file, instance_id="test-2")

    msgs_in = [{"role": "user", "content": "Say hi."}]
    await hook.before_iteration(_StubContext(iteration=0, messages=msgs_in))
    # Note: before_execute_tools is NOT called when there are no tool calls
    msgs_after_llm = msgs_in + [{"role": "assistant", "content": "hi"}]
    await hook.after_iteration(
        _StubContext(
            iteration=0,
            messages=msgs_after_llm,
            tool_calls=[],
            usage={"prompt_tokens": 5, "completion_tokens": 1},
            response=_StubResponse(content="hi", finish_reason="stop"),
        )
    )
    hook.close()

    records = [json.loads(line) for line in trace_file.read_text().strip().splitlines()]
    llm_calls = [r for r in records
                 if r.get("type") == "action" and r.get("action_type") == "llm_call"]
    tool_execs = [r for r in records
                  if r.get("type") == "action" and r.get("action_type") == "tool_exec"]

    assert len(llm_calls) == 1
    assert len(tool_execs) == 0
    # ts_end falls back to "now" when before_execute_tools was never called
    assert llm_calls[0]["ts_end"] >= llm_calls[0]["ts_start"]


def test_resolve_run_outcome_uses_trace_llm_error_event(tmp_path: Path) -> None:
    trace_file = tmp_path / "trace.jsonl"
    trace_file.write_text(
        json.dumps(
            {
                "type": "event",
                "event": "llm_error",
                "category": "LLM",
                "data": {"error_message": "credits exhausted"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    stop_reason, error = _resolve_run_outcome(
        outcome={},
        content='Error: {"error":{"message":"This request requires more credits"}}',
        trace_file=trace_file,
    )

    assert stop_reason == "error"
    assert "requires more credits" in (error or "")
