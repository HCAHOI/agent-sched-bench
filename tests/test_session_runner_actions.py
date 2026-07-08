"""Regression tests for OpenClaw TraceCollectorHook action emission.

The hook emits one ``llm_call`` action from ``after_llm_response`` for each
model response and emits only iteration totals/tool actions from
``after_iteration``.  These tests drive synthetic ``AgentHookContext`` inputs
through the same hook order as the runner so regressions show up in the JSONL
trace rather than in hook internals.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

# Skip the entire module if OpenClaw deps are unavailable.
pytest.importorskip("agents.openclaw._session_runner")

from agents.deep_research.web_tools import BackendFailureResult
from agents.openclaw._hook import AgentHook, AgentHookContext
from agents.openclaw._loop import AgentLoop
from agents.openclaw._session_runner import (
    TraceCollectorHook,
    _resolve_run_outcome,
)
from agents.openclaw._subagent import SubagentManager
from agents.openclaw.bus.queue import MessageBus
from agents.openclaw.session.manager import Session
from agents.openclaw.tools.base import Tool
from llm_call.provider_base import LLMProvider, LLMResponse, ToolCallRequest


class _StubResponse:
    def __init__(
        self,
        content: str = "",
        finish_reason: str = "stop",
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.content = content
        self.finish_reason = finish_reason
        self.reasoning_content: str | None = None
        self.extra: dict[str, Any] | None = extra


class _StubToolCall:
    _counter = 0

    def __init__(self, name: str, arguments: dict[str, Any]) -> None:
        _StubToolCall._counter += 1
        self.id = f"call_{_StubToolCall._counter}"
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
        tool_resource_timelines: dict[str, dict[str, Any]] | None = None,
        tool_timings: dict[str, dict[str, float]] | None = None,
        model_messages: list[dict[str, Any]] | None = None,
        llm_call_start_ts: float | None = None,
    ) -> None:
        self.iteration = iteration
        self.messages = messages
        self.model_messages = messages if model_messages is None else model_messages
        self.llm_call_start_ts = llm_call_start_ts
        self.tool_calls = tool_calls or []
        self.usage = usage or {}
        self.response = response
        self.tool_resource_timelines = tool_resource_timelines or {}
        self.tool_timings = tool_timings or {}
        self.malformed_retry_count = 0


class _LoopPathProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__(api_key="test", api_base="http://test")
        self.responses = responses
        self.requests: list[list[dict[str, Any]]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        del tools, model, max_tokens, temperature, reasoning_effort, tool_choice
        self.requests.append([dict(message) for message in messages])
        return self.responses.pop(0)

    def get_default_model(self) -> str:
        return "fake-model"


class _CountingAfterLLMHook(AgentHook):
    def __init__(self) -> None:
        self.calls = 0
        self.contents: list[str | None] = []
        self.usages: list[dict[str, int]] = []

    async def after_llm_response(self, context: AgentHookContext) -> None:
        self.calls += 1
        self.contents.append(context.response.content if context.response else None)
        self.usages.append(dict(context.usage))


class _ToolEventCaptureHook(AgentHook):
    def __init__(self) -> None:
        self.iterations: list[list[dict[str, str]]] = []

    async def after_iteration(self, context: AgentHookContext) -> None:
        self.iterations.append([dict(event) for event in context.tool_events])


class _TerminalBackendFailureTool(Tool):
    @property
    def name(self) -> str:
        return "terminal_backend_failure"

    @property
    def description(self) -> str:
        return "Return the same terminal-yield failure shape as web backend exhaustion."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, **kwargs: Any) -> BackendFailureResult:
        del kwargs
        return BackendFailureResult("web backend failed")


class _BlockingProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__(api_key="test", api_base="http://test")
        self.started: asyncio.Queue[str] = asyncio.Queue()
        self.release = asyncio.Event()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        del tools, model, max_tokens, temperature, reasoning_effort, tool_choice
        self.started.put_nowait(str(messages[-1].get("content", "")))
        await self.release.wait()
        return LLMResponse(content="done")

    def get_default_model(self) -> str:
        return "fake-model"


async def _seed_active_subagent(
    manager: SubagentManager, session_key: str
) -> asyncio.Task[bool]:
    gate = asyncio.Event()
    task = asyncio.create_task(gate.wait())
    task_id = f"active-{session_key}"
    manager._running_tasks[task_id] = task
    manager._session_tasks.setdefault(session_key, set()).add(task_id)
    manager._sessions_with_subagents.add(session_key)
    return task


def test_agent_loop_waits_for_runtime_event_when_yield_succeeds_with_recoverable_tool_error(
    tmp_path: Path,
) -> None:
    asyncio.run(
        _drive_yield_succeeds_with_recoverable_tool_error_waits_for_runtime(tmp_path)
    )


async def _drive_yield_succeeds_with_recoverable_tool_error_waits_for_runtime(
    tmp_path: Path,
) -> None:
    session = Session(key="cli:yield-recoverable")
    capture = _ToolEventCaptureHook()
    provider = _LoopPathProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="yield-call",
                        name="sessions_yield",
                        arguments={},
                    ),
                    ToolCallRequest(
                        id="spawn-call",
                        name="spawn",
                        arguments={
                            "task": "this spawn should exceed the active budget",
                            "label": "overflow",
                        },
                    ),
                ],
                finish_reason="tool_calls",
            )
        ]
    )
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path / "state",
        tool_workspace=tmp_path / "worktree",
        model="fake-model",
        max_iterations=1,
        max_tool_result_chars=1000,
        hooks=[capture],
        subagent_max_active_per_session=0,
    )
    await _seed_active_subagent(loop.subagents, session.key)

    try:
        final_content, _tools_used, _messages = await loop._run_agent_loop(
            [{"role": "user", "content": "yield and hit a recoverable spawn error"}],
            session=session,
            channel="cli",
            chat_id="yield-recoverable",
        )

        assert final_content is None
        assert loop._last_run_outcomes[session.key] == {
            "stop_reason": "yielded",
            "error": None,
            "waiting_for_runtime_event": True,
        }
        assert capture.iterations
        events = {event["name"]: event for event in capture.iterations[-1]}
        assert events["sessions_yield"]["status"] == "ok"
        assert events["sessions_yield"]["should_yield"] == "true"
        assert events["spawn"]["status"] == "error"
        assert events["spawn"]["should_yield"] == "false"
    finally:
        await loop.subagents.cancel_session(session.key)


def test_agent_loop_does_not_wait_when_terminal_yield_error_shares_turn_with_yield(
    tmp_path: Path,
) -> None:
    asyncio.run(_drive_terminal_yield_error_suppresses_runtime_wait(tmp_path))


async def _drive_terminal_yield_error_suppresses_runtime_wait(tmp_path: Path) -> None:
    session = Session(key="cli:yield-terminal")
    capture = _ToolEventCaptureHook()
    provider = _LoopPathProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="yield-call",
                        name="sessions_yield",
                        arguments={},
                    ),
                    ToolCallRequest(
                        id="backend-failure-call",
                        name="terminal_backend_failure",
                        arguments={},
                    ),
                ],
                finish_reason="tool_calls",
            )
        ]
    )
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path / "state",
        tool_workspace=tmp_path / "worktree",
        model="fake-model",
        max_iterations=1,
        max_tool_result_chars=1000,
        hooks=[capture],
        tool_overrides=[_TerminalBackendFailureTool()],
    )
    await _seed_active_subagent(loop.subagents, session.key)

    try:
        final_content, _tools_used, _messages = await loop._run_agent_loop(
            [{"role": "user", "content": "yield and hit a terminal backend failure"}],
            session=session,
            channel="cli",
            chat_id="yield-terminal",
        )

        assert final_content is None
        assert loop._last_run_outcomes[session.key] == {
            "stop_reason": "yielded",
            "error": None,
            "waiting_for_runtime_event": False,
        }
        assert capture.iterations
        events = {event["name"]: event for event in capture.iterations[-1]}
        assert events["sessions_yield"]["status"] == "ok"
        assert events["sessions_yield"]["should_yield"] == "true"
        assert events["terminal_backend_failure"]["status"] == "error"
        assert events["terminal_backend_failure"]["should_yield"] == "true"
    finally:
        await loop.subagents.cancel_session(session.key)


def test_subagent_manager_cancel_session_drains_only_target_session(
    tmp_path: Path,
) -> None:
    asyncio.run(_drive_cancel_session_drains_only_target_session(tmp_path))


async def _drive_cancel_session_drains_only_target_session(tmp_path: Path) -> None:
    provider = _BlockingProvider()
    manager = SubagentManager(
        provider=provider,
        workspace=tmp_path / "workspace",
        bus=MessageBus(),
        max_tool_result_chars=1000,
    )

    await manager.spawn("target one", label="target-one", session_key="session-a")
    await manager.spawn("target two", label="target-two", session_key="session-a")
    await manager.spawn("unrelated", label="unrelated", session_key="session-b")
    started = [await provider.started.get() for _ in range(3)]
    assert sorted(started) == ["target one", "target two", "unrelated"]
    target_ids = set(manager._session_tasks["session-a"])
    unrelated_ids = set(manager._session_tasks["session-b"])

    try:
        await manager.cancel_session("session-a")

        assert manager.has_active("session-a") is False
        assert "session-a" not in manager._session_tasks
        assert all(task_id not in manager._running_tasks for task_id in target_ids)
        assert manager.has_active("session-b") is True
        assert manager._session_tasks["session-b"] == unrelated_ids
        assert all(
            not manager._running_tasks[task_id].done() for task_id in unrelated_ids
        )
    finally:
        await manager.cancel_session("session-b")


def test_agent_loop_extra_hooks_receive_after_llm_response_from_runner(
    tmp_path: Path,
) -> None:
    asyncio.run(_drive_agent_loop_extra_hooks_receive_after_llm_response(tmp_path))


async def _drive_agent_loop_extra_hooks_receive_after_llm_response(
    tmp_path: Path,
) -> None:
    trace_file = tmp_path / "trace.jsonl"
    trace_hook = TraceCollectorHook(trace_file, instance_id="loop-extra")
    counter_hook = _CountingAfterLLMHook()
    provider = _LoopPathProvider(
        [
            LLMResponse(
                content="normal path final",
                usage={"prompt_tokens": 9, "completion_tokens": 4},
            )
        ]
    )
    prompt = {"role": "user", "content": "Say hi through the normal loop."}
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path / "state",
        tool_workspace=tmp_path / "worktree",
        model="fake-model",
        max_iterations=1,
        max_tool_result_chars=1000,
        hooks=[trace_hook, counter_hook],
    )

    final_content, tools_used, _messages = await loop._run_agent_loop(
        [prompt],
        channel="cli",
        chat_id="trace-test",
    )
    trace_hook.close()

    records = [json.loads(line) for line in trace_file.read_text().splitlines()]
    llm_calls = [
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "llm_call"
    ]

    assert final_content == "normal path final"
    assert tools_used == []
    assert provider.requests == [[prompt]]
    assert counter_hook.calls == 1
    assert counter_hook.contents == ["normal path final"]
    assert counter_hook.usages == [{"prompt_tokens": 9, "completion_tokens": 4}]
    assert len(llm_calls) == 1
    assert llm_calls[0]["agent_id"] == "loop-extra"
    assert llm_calls[0]["iteration"] == 0
    assert llm_calls[0]["data"]["messages_in"] == [prompt]
    assert llm_calls[0]["data"]["raw_response"]["choices"][0]["message"] == {
        "role": "assistant",
        "content": "normal path final",
    }
    assert llm_calls[0]["data"]["raw_response"]["usage"] == {
        "prompt_tokens": 9,
        "completion_tokens": 4,
    }

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
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": '{"path":"a.py"}'},
                }
            ],
        }
    ]
    stub_tc = _StubToolCall("write_file", {"path": "a.py"})
    response = _StubResponse(
        content="",
        finish_reason="tool_calls",
        extra={"llm_wall_ts_end": 1000.25},
    )
    ctx_before_tools = _StubContext(
        iteration=0,
        messages=msgs_after_llm,
        model_messages=msgs_in,
        llm_call_start_ts=1000.0,
        tool_calls=[stub_tc],
        usage={"prompt_tokens": 100, "completion_tokens": 20},
        response=response,
    )
    await hook.after_llm_response(ctx_before_tools)
    await hook.before_execute_tools(ctx_before_tools)

    # Simulate tool result appended to messages
    msgs_after_tool = msgs_after_llm + [
        {"role": "tool", "tool_call_id": stub_tc.id, "name": "write_file", "content": "wrote a.py"}
    ]
    ctx_after = _StubContext(
        iteration=0,
        messages=msgs_after_tool,
        model_messages=msgs_in,
        llm_call_start_ts=1000.0,
        tool_calls=[stub_tc],
        usage={"prompt_tokens": 100, "completion_tokens": 20},
        response=response,
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
    llm_idx = next(
        i
        for i, r in enumerate(records)
        if r.get("type") == "action" and r.get("action_type") == "llm_call"
    )
    tool_idx = next(
        i
        for i, r in enumerate(records)
        if r.get("type") == "action" and r.get("action_type") == "tool_exec"
    )
    assert llm_idx < tool_idx, "llm_call action must precede tool_exec in trace"

    # Verify llm_call action carries the snapshotted messages_in (NOT the
    # post-tool-result messages — that would be a leak from after_iteration)
    llm = llm_calls[0]
    assert llm["data"]["messages_in"] == msgs_in
    assert llm["data"]["prompt_tokens"] == 100
    assert llm["data"]["completion_tokens"] == 20
    assert llm["iteration"] == 0
    assert llm["ts_start"] == 1000.0
    assert llm["ts_end"] == 1000.25


@pytest.mark.parametrize(
    ("tool_content", "expected_success"),
    [
        ("Spawned subagent child-task", True),
        ("Error: active subagent budget exceeded", False),
    ],
)
def test_trace_collector_names_spawn_tool_result_event(
    tmp_path: Path,
    tool_content: str,
    expected_success: bool,
) -> None:
    asyncio.run(
        _drive_trace_collector_names_spawn_tool_result_event(
            tmp_path,
            tool_content,
            expected_success,
        )
    )


async def _drive_trace_collector_names_spawn_tool_result_event(
    tmp_path: Path,
    tool_content: str,
    expected_success: bool,
) -> None:
    trace_file = tmp_path / "trace.jsonl"
    hook = TraceCollectorHook(trace_file, instance_id="spawn-trace")

    messages_in = [{"role": "user", "content": "Start a child worker."}]
    await hook.before_iteration(_StubContext(iteration=0, messages=messages_in))

    spawn_call = _StubToolCall("spawn", {"task": "inspect logs", "label": "child-task"})
    response = _StubResponse(
        content="",
        finish_reason="tool_calls",
        extra={"llm_wall_ts_end": 1700.25},
    )
    messages_after_llm = messages_in + [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": spawn_call.id,
                    "type": "function",
                    "function": {
                        "name": "spawn",
                        "arguments": '{"task":"inspect logs","label":"child-task"}',
                    },
                }
            ],
        }
    ]
    ctx_before_tools = _StubContext(
        iteration=0,
        messages=messages_after_llm,
        model_messages=messages_in,
        llm_call_start_ts=1700.0,
        tool_calls=[spawn_call],
        response=response,
    )
    await hook.after_llm_response(ctx_before_tools)
    await hook.before_execute_tools(ctx_before_tools)

    messages_after_tool = messages_after_llm + [
        {
            "role": "tool",
            "tool_call_id": spawn_call.id,
            "name": "spawn",
            "content": tool_content,
        }
    ]
    await hook.after_iteration(
        _StubContext(
            iteration=0,
            messages=messages_after_tool,
            model_messages=messages_in,
            llm_call_start_ts=1700.0,
            tool_calls=[spawn_call],
            response=response,
            tool_timings={
                spawn_call.id: {
                    "ts_start": 1700.25,
                    "ts_end": 1700.3,
                    "duration_ms": 50.0,
                }
            },
        )
    )
    hook.close()

    records = [json.loads(line) for line in trace_file.read_text().splitlines()]
    subagent_events = [
        record
        for record in records
        if record.get("type") == "event" and record.get("category") == "SUBAGENT"
    ]

    assert [event["event"] for event in subagent_events] == ["subagent_spawn_result"]
    assert subagent_events[0]["data"] == {
        "success": expected_success,
        "result_preview": tool_content,
    }
    assert "subagent_complete" not in {
        record.get("event") for record in records if record.get("type") == "event"
    }


def test_trace_collector_records_model_visible_messages_not_full_context(
    tmp_path: Path,
) -> None:
    asyncio.run(_drive_records_model_visible_messages_not_full_context(tmp_path))


async def _drive_records_model_visible_messages_not_full_context(
    tmp_path: Path,
) -> None:
    trace_file = tmp_path / "trace.jsonl"
    hook = TraceCollectorHook(trace_file, instance_id="test-snipped-messages")

    model_messages = [
        {"role": "system", "content": "visible system instruction"},
        {"role": "user", "content": "VISIBLE_QUESTION_SENTINEL"},
    ]
    full_context_messages = [
        {
            "role": "system",
            "content": "full context with SECRET_SYSTEM_SENTINEL",
        },
        {"role": "user", "content": "VISIBLE_QUESTION_SENTINEL"},
        {"role": "user", "content": "SECRET_REFERENCE_SENTINEL"},
    ]
    await hook.before_iteration(
        _StubContext(
            iteration=0,
            messages=full_context_messages,
            model_messages=model_messages,
        )
    )

    response = _StubResponse(
        content="answer",
        finish_reason="stop",
        extra={"llm_wall_ts_end": 2000.25},
    )
    ctx_after_llm = _StubContext(
        iteration=0,
        messages=full_context_messages + [{"role": "assistant", "content": "answer"}],
        model_messages=model_messages,
        llm_call_start_ts=2000.0,
        usage={"prompt_tokens": 7, "completion_tokens": 1},
        response=response,
    )
    await hook.after_llm_response(ctx_after_llm)
    await hook.after_iteration(ctx_after_llm)
    hook.close()

    records = [json.loads(line) for line in trace_file.read_text().splitlines()]
    llm_call = next(
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "llm_call"
    )

    assert llm_call["data"]["messages_in"] == model_messages
    serialized_messages = json.dumps(llm_call["data"]["messages_in"])
    assert "VISIBLE_QUESTION_SENTINEL" in serialized_messages
    assert "SECRET_SYSTEM_SENTINEL" not in serialized_messages
    assert "SECRET_REFERENCE_SENTINEL" not in serialized_messages


def test_trace_collector_records_empty_final_retry_as_distinct_llm_calls(
    tmp_path: Path,
) -> None:
    asyncio.run(_drive_records_empty_final_retry_as_distinct_llm_calls(tmp_path))


async def _drive_records_empty_final_retry_as_distinct_llm_calls(
    tmp_path: Path,
) -> None:
    trace_file = tmp_path / "trace.jsonl"
    hook = TraceCollectorHook(trace_file, instance_id="test-empty-final-retry")

    first_model_messages = [{"role": "user", "content": "first final attempt"}]
    await hook.before_iteration(
        _StubContext(iteration=0, messages=first_model_messages)
    )
    first_response = _StubResponse(
        content="",
        finish_reason="stop",
        extra={"llm_wall_ts_end": 3000.1},
    )
    first_context = _StubContext(
        iteration=0,
        messages=first_model_messages + [{"role": "assistant", "content": ""}],
        model_messages=first_model_messages,
        llm_call_start_ts=3000.0,
        usage={"prompt_tokens": 3, "completion_tokens": 0},
        response=first_response,
    )
    await hook.after_llm_response(first_context)

    second_model_messages = [
        {"role": "user", "content": "first final attempt"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "retry because final answer was empty"},
    ]
    second_response = _StubResponse(
        content="retry succeeded",
        finish_reason="stop",
        extra={"llm_wall_ts_end": 3001.2},
    )
    second_context = _StubContext(
        iteration=0,
        messages=second_model_messages
        + [{"role": "assistant", "content": "retry succeeded"}],
        model_messages=second_model_messages,
        llm_call_start_ts=3001.0,
        usage={"prompt_tokens": 9, "completion_tokens": 2},
        response=second_response,
    )
    await hook.after_llm_response(second_context)
    await hook.after_iteration(second_context)
    hook.close()

    records = [json.loads(line) for line in trace_file.read_text().splitlines()]
    llm_calls = [
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "llm_call"
    ]

    assert [record["action_id"] for record in llm_calls] == ["llm_0", "llm_1"]
    assert [record["iteration"] for record in llm_calls] == [0, 0]
    assert [record["ts_start"] for record in llm_calls] == [3000.0, 3001.0]
    assert [record["ts_end"] for record in llm_calls] == [3000.1, 3001.2]
    assert llm_calls[0]["data"]["messages_in"] == first_model_messages
    assert llm_calls[1]["data"]["messages_in"] == second_model_messages
    assert llm_calls[0]["data"]["raw_response"]["choices"][0]["message"]["content"] == ""
    assert (
        llm_calls[1]["data"]["raw_response"]["choices"][0]["message"]["content"]
        == "retry succeeded"
    )


def test_trace_collector_after_iteration_does_not_duplicate_llm_call(
    tmp_path: Path,
) -> None:
    asyncio.run(_drive_after_iteration_does_not_duplicate_llm_call(tmp_path))


async def _drive_after_iteration_does_not_duplicate_llm_call(tmp_path: Path) -> None:
    trace_file = tmp_path / "trace.jsonl"
    hook = TraceCollectorHook(trace_file, instance_id="test-no-duplicate")

    model_messages = [{"role": "user", "content": "single call"}]
    await hook.before_iteration(_StubContext(iteration=0, messages=model_messages))
    response = _StubResponse(
        content="done",
        finish_reason="stop",
        extra={"llm_wall_ts_end": 4000.25},
    )
    ctx_after_llm = _StubContext(
        iteration=0,
        messages=model_messages + [{"role": "assistant", "content": "done"}],
        model_messages=model_messages,
        llm_call_start_ts=4000.0,
        usage={"prompt_tokens": 2, "completion_tokens": 1},
        response=response,
    )
    await hook.after_llm_response(ctx_after_llm)
    await hook.after_iteration(ctx_after_llm)
    hook.close()

    records = [json.loads(line) for line in trace_file.read_text().splitlines()]
    llm_calls = [
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "llm_call"
    ]

    assert len(llm_calls) == 1
    assert llm_calls[0]["action_id"] == "llm_0"
    assert llm_calls[0]["data"]["messages_in"] == model_messages


def test_trace_collector_emits_tool_resource_timeline(tmp_path: Path) -> None:
    asyncio.run(_drive_emits_tool_resource_timeline(tmp_path))


async def _drive_emits_tool_resource_timeline(tmp_path: Path) -> None:
    trace_file = tmp_path / "trace.jsonl"
    hook = TraceCollectorHook(trace_file, instance_id="test-resource")

    msgs_in = [{"role": "user", "content": "Run test."}]
    await hook.before_iteration(_StubContext(iteration=0, messages=msgs_in))
    stub_tc = _StubToolCall("exec", {"command": "pytest"})
    msgs_after_llm = msgs_in + [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": stub_tc.id,
                    "type": "function",
                    "function": {
                        "name": "exec",
                        "arguments": '{"command":"pytest"}',
                    },
                }
            ],
        }
    ]
    response = _StubResponse(
        content="",
        finish_reason="tool_calls",
        extra={"llm_wall_ts_end": 1100.25},
    )
    ctx_before_tools = _StubContext(
        iteration=0,
        messages=msgs_after_llm,
        model_messages=msgs_in,
        llm_call_start_ts=1100.0,
        tool_calls=[stub_tc],
        response=response,
    )
    await hook.after_llm_response(ctx_before_tools)
    await hook.before_execute_tools(ctx_before_tools)
    resource_timeline = {
        "version": 1,
        "source": "cgroup_cpu_proc_net",
        "scope": "openclaw_exec_tool_interval",
        "samples": [
            {
                "offset_s": 0.5,
                "dt_s": 0.5,
                "cpu_core_s": 1.0,
                "net_rx_bytes": 128,
                "net_tx_bytes": 64,
            }
        ],
        "summary": {
            "sample_count": 1,
            "wall_s": 0.5,
            "cpu_core_s": 1.0,
            "net_rx_bytes": 128,
            "net_tx_bytes": 64,
        },
    }
    msgs_after_tool = msgs_after_llm + [
        {
            "role": "tool",
            "tool_call_id": stub_tc.id,
            "name": "exec",
            "content": "ok",
        }
    ]
    await hook.after_iteration(
        _StubContext(
            iteration=0,
            messages=msgs_after_tool,
            model_messages=msgs_in,
            llm_call_start_ts=1100.0,
            tool_calls=[stub_tc],
            response=response,
            tool_resource_timelines={stub_tc.id: resource_timeline},
        )
    )
    hook.close()

    records = [json.loads(line) for line in trace_file.read_text().splitlines()]
    tool_exec = next(record for record in records if record.get("action_type") == "tool_exec")
    assert tool_exec["data"]["resource_timeline"] == resource_timeline


def test_trace_collector_uses_runner_tool_timings(tmp_path: Path) -> None:
    asyncio.run(_drive_uses_runner_tool_timings(tmp_path))


async def _drive_uses_runner_tool_timings(tmp_path: Path) -> None:
    trace_file = tmp_path / "trace.jsonl"
    hook = TraceCollectorHook(trace_file, instance_id="test-timing")

    msgs_in = [{"role": "user", "content": "Run two tools."}]
    await hook.before_iteration(_StubContext(iteration=0, messages=msgs_in))
    first = _StubToolCall("read_file", {"path": "a.py"})
    second = _StubToolCall("list_dir", {"path": "."})
    msgs_after_llm = msgs_in + [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": first.id,
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"a.py"}'},
                },
                {
                    "id": second.id,
                    "type": "function",
                    "function": {"name": "list_dir", "arguments": '{"path":"."}'},
                },
            ],
        }
    ]
    response = _StubResponse(
        content="",
        finish_reason="tool_calls",
        extra={"llm_wall_ts_end": 1200.25},
    )
    ctx_before_tools = _StubContext(
        iteration=0,
        messages=msgs_after_llm,
        model_messages=msgs_in,
        llm_call_start_ts=1200.0,
        tool_calls=[first, second],
        response=response,
    )
    await hook.after_llm_response(ctx_before_tools)
    await hook.before_execute_tools(ctx_before_tools)
    msgs_after_tool = msgs_after_llm + [
        {
            "role": "tool",
            "tool_call_id": first.id,
            "name": "read_file",
            "content": "file",
        },
        {
            "role": "tool",
            "tool_call_id": second.id,
            "name": "list_dir",
            "content": "dir",
        },
    ]
    await hook.after_iteration(
        _StubContext(
            iteration=0,
            messages=msgs_after_tool,
            model_messages=msgs_in,
            llm_call_start_ts=1200.0,
            tool_calls=[first, second],
            response=response,
            tool_timings={
                first.id: {"ts_start": 1000.0, "ts_end": 1000.2, "duration_ms": 200.0},
                second.id: {"ts_start": 1000.0, "ts_end": 1000.3, "duration_ms": 300.0},
            },
        )
    )
    hook.close()

    records = [json.loads(line) for line in trace_file.read_text().splitlines()]
    tools = [
        record
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "tool_exec"
    ]
    by_name = {record["data"]["tool_name"]: record for record in tools}

    assert by_name["read_file"]["ts_start"] == 1000.0
    assert by_name["read_file"]["ts_end"] == 1000.2
    assert by_name["read_file"]["data"]["duration_ms"] == 200.0
    assert by_name["list_dir"]["ts_start"] == 1000.0
    assert by_name["list_dir"]["ts_end"] == 1000.3
    assert by_name["list_dir"]["data"]["duration_ms"] == 300.0


def test_trace_collector_subagent_hooks_share_trace_file(tmp_path: Path) -> None:
    asyncio.run(_drive_subagent_hooks_share_trace_file(tmp_path))


async def _drive_subagent_hooks_share_trace_file(tmp_path: Path) -> None:
    trace_file = tmp_path / "trace.jsonl"
    parent_hook = TraceCollectorHook(trace_file, instance_id="parent")
    child_hook = parent_hook.for_subagent("child-task")

    parent_messages = [{"role": "user", "content": "spawn"}]
    await parent_hook.before_iteration(
        _StubContext(iteration=0, messages=parent_messages)
    )
    parent_response = _StubResponse(
        content="parent",
        finish_reason="stop",
        extra={"llm_wall_ts_end": 1300.25},
    )
    await parent_hook.after_llm_response(
        _StubContext(
            iteration=0,
            messages=parent_messages + [{"role": "assistant", "content": "parent"}],
            model_messages=parent_messages,
            llm_call_start_ts=1300.0,
            response=parent_response,
        )
    )
    await parent_hook.after_iteration(
        _StubContext(
            iteration=0,
            messages=parent_messages + [{"role": "assistant", "content": "parent"}],
            model_messages=parent_messages,
            llm_call_start_ts=1300.0,
            response=parent_response,
        )
    )

    child_messages = [{"role": "user", "content": "child work"}]
    await child_hook.before_iteration(_StubContext(iteration=0, messages=child_messages))
    child_response = _StubResponse(
        content="child",
        finish_reason="stop",
        extra={"llm_wall_ts_end": 1400.25},
    )
    await child_hook.after_llm_response(
        _StubContext(
            iteration=0,
            messages=child_messages + [{"role": "assistant", "content": "child"}],
            model_messages=child_messages,
            llm_call_start_ts=1400.0,
            response=child_response,
        )
    )
    await child_hook.after_iteration(
        _StubContext(
            iteration=0,
            messages=child_messages + [{"role": "assistant", "content": "child"}],
            model_messages=child_messages,
            llm_call_start_ts=1400.0,
            response=child_response,
        )
    )
    parent_hook.close()

    records = [json.loads(line) for line in trace_file.read_text().splitlines()]
    action_agent_ids = {
        record["agent_id"]
        for record in records
        if record.get("type") == "action" and record.get("action_type") == "llm_call"
    }

    assert "parent" in action_agent_ids
    assert "parent:subagent:child-task" in action_agent_ids


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
    response = _StubResponse(
        content="hi",
        finish_reason="stop",
        extra={"llm_wall_ts_end": 1500.25},
    )
    ctx_after_llm = _StubContext(
        iteration=0,
        messages=msgs_after_llm,
        model_messages=msgs_in,
        llm_call_start_ts=1500.0,
        tool_calls=[],
        usage={"prompt_tokens": 5, "completion_tokens": 1},
        response=response,
    )
    await hook.after_llm_response(ctx_after_llm)
    await hook.after_iteration(ctx_after_llm)
    hook.close()

    records = [json.loads(line) for line in trace_file.read_text().strip().splitlines()]
    llm_calls = [
        r
        for r in records
        if r.get("type") == "action" and r.get("action_type") == "llm_call"
    ]
    tool_execs = [
        r
        for r in records
        if r.get("type") == "action" and r.get("action_type") == "tool_exec"
    ]

    assert len(llm_calls) == 1
    assert len(tool_execs) == 0
    assert llm_calls[0]["ts_start"] == 1500.0
    assert llm_calls[0]["ts_end"] == 1500.25


def test_trace_collector_records_openrouter_latency_fields(tmp_path: Path) -> None:
    import asyncio

    asyncio.run(_drive_openrouter_latency_fields(tmp_path))


async def _drive_openrouter_latency_fields(tmp_path: Path) -> None:
    trace_file = tmp_path / "trace.jsonl"
    hook = TraceCollectorHook(trace_file, instance_id="test-openrouter")

    model_messages = [{"role": "user", "content": "Ping"}]
    await hook.before_iteration(_StubContext(iteration=0, messages=model_messages))
    hook._iter_start_wall = 100.0
    hook._before_exec_wall = 0.0

    openrouter_metadata = {
        "generation_id": "gen-123",
        "request_id": "req-123",
        "provider_name": "Z.AI",
        "latency_ms": 7000.0,
        "generation_time_ms": 6500.0,
        "provider_latency_ms": 6800.0,
        "upstream_id": "up-123",
        "provider_responses": [
            {"provider_name": "Z.AI", "latency_ms": 6800.0, "status": 200}
        ],
    }
    response = _StubResponse(
        content="pong",
        finish_reason="stop",
        extra={
            "llm_wall_ts_end": 115.0,
            "llm_call_time_ms": 6500.0,
            "llm_timing_source": "openrouter_generation_time_ms",
            "openrouter_generation_id": "gen-123",
            "openrouter_request_id": "req-123",
            "openrouter_latency_ms": 7000.0,
            "openrouter_generation_time_ms": 6500.0,
            "openrouter_provider_latency_ms": 6800.0,
            "openrouter_provider_name": "Z.AI",
            "openrouter_upstream_id": "up-123",
            "openrouter_metadata": openrouter_metadata,
        },
    )
    ctx_after_llm = _StubContext(
        iteration=0,
        messages=[
            {"role": "user", "content": "Ping"},
            {"role": "assistant", "content": "pong"},
        ],
        model_messages=model_messages,
        llm_call_start_ts=100.0,
        usage={"prompt_tokens": 12, "completion_tokens": 3},
        response=response,
    )
    await hook.after_llm_response(ctx_after_llm)
    await hook.after_iteration(ctx_after_llm)
    await hook.write_summary(success=True, elapsed_s=15.0)

    records = [json.loads(line) for line in trace_file.read_text().strip().splitlines()]
    llm_call = next(
        r
        for r in records
        if r.get("type") == "action" and r.get("action_type") == "llm_call"
    )
    llm_event = next(
        r
        for r in records
        if r.get("type") == "event" and r.get("event") == "llm_call_end"
    )
    summary = next(r for r in records if r.get("type") == "summary")

    assert llm_call["data"]["llm_latency_ms"] == 6500.0
    assert llm_call["data"]["llm_call_time_ms"] == 6500.0
    assert llm_call["data"]["llm_wall_latency_ms"] == 15000.0
    assert llm_call["data"]["llm_timing_source"] == "openrouter_generation_time_ms"
    assert llm_call["data"]["openrouter_latency_ms"] == 7000.0
    assert llm_call["data"]["openrouter_generation_time_ms"] == 6500.0
    assert llm_call["data"]["openrouter_provider_latency_ms"] == 6800.0
    assert llm_call["data"]["openrouter_generation_id"] == "gen-123"
    assert (
        llm_call["data"]["raw_response"]["openrouter_metadata"] == openrouter_metadata
    )
    assert llm_event["data"]["llm_latency_ms"] == 6500.0
    assert llm_event["data"]["openrouter_request_id"] == "req-123"
    assert "openrouter_metadata" not in llm_event["data"]
    assert summary["total_llm_ms"] == 6500.0
    assert summary["total_llm_call_time_ms"] == 6500.0
    assert summary["llm_call_time_count"] == 1
    assert summary["llm_timing_source"] == "openrouter_generation_time_ms"
    assert summary["total_llm_wall_ms"] == 15000.0


def test_trace_collector_refetches_late_openrouter_metadata(tmp_path: Path) -> None:
    import asyncio

    asyncio.run(_drive_refetches_late_openrouter_metadata(tmp_path))


async def _drive_refetches_late_openrouter_metadata(tmp_path: Path) -> None:
    trace_file = tmp_path / "trace.jsonl"
    hook = TraceCollectorHook(trace_file, instance_id="test-openrouter-late")

    model_messages = [{"role": "user", "content": "Ping"}]
    await hook.before_iteration(_StubContext(iteration=0, messages=model_messages))
    hook._iter_start_wall = 100.0
    hook._before_exec_wall = 115.0

    extra: dict[str, Any] = {
        "llm_wall_ts_end": 115.0,
        "openrouter_generation_id": "gen-late",
        "openrouter_metadata_fetch_status": "pending",
        "openrouter_metadata_fetch_ms": 1.0,
    }

    async def initial_fetch() -> dict[str, Any]:
        result = {
            "openrouter_metadata_fetch_status": "unavailable",
            "openrouter_metadata_fetch_ms": 2.0,
            "openrouter_metadata_fetch_attempt_count": 1,
            "openrouter_metadata_fetch_last_status_code": 404,
            "openrouter_metadata_fetch_last_reason": "not_found",
        }
        extra.update(result)
        return result

    async def refetch() -> dict[str, Any]:
        return {
            "openrouter_metadata_fetch_status": "success",
            "openrouter_metadata_fetch_ms": 3.0,
            "openrouter_metadata_fetch_attempt_count": 1,
            "openrouter_metadata_fetch_last_status_code": 200,
            "openrouter_metadata_fetch_last_reason": "success",
            "openrouter_metadata": {
                "generation_id": "gen-late",
                "generation_time_ms": 4321.0,
                "latency_ms": 5000.0,
            },
            "openrouter_generation_time_ms": 4321.0,
            "openrouter_latency_ms": 5000.0,
            "llm_call_time_ms": 4321.0,
            "llm_timing_source": "openrouter_generation_time_ms",
        }

    extra["_openrouter_metadata_task"] = asyncio.create_task(initial_fetch())
    extra["_openrouter_metadata_refetcher"] = refetch
    response = _StubResponse(content="pong", finish_reason="stop", extra=extra)
    ctx_after_llm = _StubContext(
        iteration=0,
        messages=[
            {"role": "user", "content": "Ping"},
            {"role": "assistant", "content": "pong"},
        ],
        model_messages=model_messages,
        llm_call_start_ts=100.0,
        usage={"prompt_tokens": 12, "completion_tokens": 3},
        response=response,
    )

    await hook.after_llm_response(ctx_after_llm)
    await hook.after_iteration(ctx_after_llm)
    await hook.write_summary(success=True, elapsed_s=15.0)

    records = [json.loads(line) for line in trace_file.read_text().strip().splitlines()]
    llm_call = next(
        r
        for r in records
        if r.get("type") == "action" and r.get("action_type") == "llm_call"
    )
    llm_event = next(
        r
        for r in records
        if r.get("type") == "event" and r.get("event") == "llm_call_end"
    )
    summary = next(r for r in records if r.get("type") == "summary")

    assert llm_call["data"]["openrouter_metadata_fetch_status"] == "success"
    assert llm_call["data"]["openrouter_metadata_refetch_attempted"] is True
    assert llm_call["data"]["openrouter_metadata_initial_fetch_status"] == "unavailable"
    assert llm_call["data"]["openrouter_generation_time_ms"] == 4321.0
    assert llm_call["data"]["llm_call_time_ms"] == 4321.0
    assert llm_call["data"]["llm_timing_source"] == "openrouter_generation_time_ms"
    assert "_openrouter_metadata_task" not in llm_call["data"]
    assert "_openrouter_metadata_refetcher" not in llm_call["data"]
    assert llm_event["data"]["openrouter_metadata_fetch_status"] == "success"
    assert summary["total_llm_call_time_ms"] == 4321.0
    assert summary["llm_timing_source"] == "openrouter_generation_time_ms"


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
