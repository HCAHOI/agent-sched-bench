"""Regression test for the segment_timeline response-marshaling drop bug.

Live repro (mini manifest, real docker replay) proved: the container agent's
JSON response DID carry ``segment_timeline`` (confirmed via /proc/<pid>/environ
showing OPENCLAW_SEGMENT_TIMELINE=1 reaching the container), but the emitted
trace's ``tool_exec`` actions had zero ``segment_timeline`` keys. Root cause:
``ContainerExecTool.execute()`` (agents/openclaw/tools/container.py) — the tool
implementation used by the real OpenClaw ``SessionRunner`` replay path
(``simulate_openclaw.py``'s ``openclaw_host_worker`` mode, which is the ONLY
replay path for scaffold="openclaw", i.e. every benchmark in this repo) —
returns only a plain ``str`` built from ``result``/``returncode`` and silently
discards every other key in the container agent's response dict, including
``segment_timeline``. A unit test of the container-side xtrace parser alone
would never have caught this: the parser was correct, the response body was
correct, but the host-side tool wrapper threw it away before it ever reached
the trace writer.

This test drives the REAL chain end-to-end: a fake ``ContainerAgent.execute()``
returns a response shaped like the container script's actual reply (including
resource-timeout-path fields, to prove the fix is not path-specific) ->
``ContainerExecTool.execute()`` -> ``AgentRunner._execute_tools()`` -> the
``(tool_call_id -> segment_timeline)`` dict that ``_session_runner.py`` reads
onto ``AgentHookContext.tool_segment_timelines`` and attaches to the trace.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("agents.openclaw._runner")

from agents.openclaw._runner import AgentRunner, AgentRunSpec
from agents.openclaw.tools.container import ContainerExecTool
from agents.openclaw.tools.registry import ToolRegistry
from llm_call.provider_base import ToolCallRequest


class _FakeContainerAgent:
    """Stands in for trace_collect.openclaw_tools.ContainerAgent.

    Returns a response dict shaped exactly like the real replay container
    script's handle_exec reply when OPENCLAW_SEGMENT_TIMELINE=1: segment
    telemetry plus (to cover "the resource-timeout path" specifically)
    resource-timeout-style fields that _run_shell_command_with_resource_timeout
    also emits alongside segment_timeline.
    """

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.requests: list[dict[str, Any]] = []

    async def execute(
        self, request: dict[str, Any], *, timeout_s: float | None = 600.0
    ) -> dict[str, Any]:
        self.requests.append(request)
        return self.response


def _segment_timeline() -> dict[str, Any]:
    return {
        "version": 2,
        "source": "bash_xtrace_epochrealtime",
        "segments": [
            {
                "segment_index": 0,
                "command_text": "cd /tmp",
                "t_start_ms": 0.0,
                "t_end_ms": 2.0,
            },
            {
                "segment_index": 1,
                "command_text": "sleep 0.2",
                "t_start_ms": 2.0,
                "t_end_ms": 202.0,
            },
        ],
        "segment_count": 2,
        "raw_total_ms": 203.0,
    }


def _container_response(segment_timeline: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "result": "ok\n",
        "returncode": 0,
        "inner_duration_ms": 203.5,
        # Fields the resource-timeout path (_run_shell_command_with_resource_timeout)
        # emits alongside segment_timeline; must not gate the marshaling.
        "resource_timeout_policy": "resource_integrated",
        "resource_virtual_time_s": 0.203,
        "segment_timeline": segment_timeline,
        # Per-binary process-accounting rows ride the same response->tool->
        # runner->trace side-channel and must marshal alongside segment_timeline.
        "per_process": [
            {"comm": "sleep", "pid": 11, "ppid": 10, "utime_s": 0.0,
             "stime_s": 0.0, "avg_mem_kb": 640, "exitcode": 0},
        ],
    }


def test_container_exec_tool_surfaces_segment_timeline_without_leaking_into_result() -> None:
    """Layer 1: the drop point itself. execute() keeps returning a plain str
    (the LLM-visible tool result) but must stash segment_timeline for the
    caller to retrieve -- and never print it into the command's own output.
    """
    timeline = _segment_timeline()
    agent = _FakeContainerAgent(_container_response(timeline))
    tool = ContainerExecTool(agent)

    result = asyncio.run(tool.execute(command="cd /tmp && sleep 0.2"))

    assert result == "ok\n\n\nExit code: 0"
    assert "segment_timeline" not in result
    assert "bash_xtrace_epochrealtime" not in result
    assert tool.last_segment_timeline == timeline


def test_container_exec_tool_resets_stale_segment_timeline_on_next_call() -> None:
    """A call whose response omits segment_timeline must not leak the
    previous call's telemetry (e.g. bash unavailable mid-session)."""
    agent = _FakeContainerAgent(_container_response(_segment_timeline()))
    tool = ContainerExecTool(agent)
    asyncio.run(tool.execute(command="cd /tmp && sleep 0.2"))
    assert tool.last_segment_timeline is not None

    agent.response = {"ok": True, "result": "ok\n", "returncode": 0}
    asyncio.run(tool.execute(command="echo ok"))

    assert tool.last_segment_timeline is None


def test_guard_blocked_exec_does_not_inherit_previous_segment_timeline() -> None:
    """Reviewer HIGH finding: the guard early-return used to skip the reset,
    so a deny-pattern-blocked exec inherited the PREVIOUS exec's timeline —
    stale-but-valid telemetry that extraction cannot detect. The reset must
    run before any early return."""
    agent = _FakeContainerAgent(_container_response(_segment_timeline()))
    tool = ContainerExecTool(agent)
    asyncio.run(tool.execute(command="cd /tmp && sleep 0.2"))
    assert tool.last_segment_timeline is not None

    # Trips the default deny_patterns guard -> early return, no request sent.
    result = asyncio.run(tool.execute(command="rm -rf /"))
    assert "blocked" in result.lower() or "denied" in result.lower() or result

    assert tool.last_segment_timeline is None


def test_execute_tools_end_to_end_marshals_segment_timeline_to_tool_call_id() -> None:
    """Layer 2: the full AgentRunner._execute_tools() chain that
    _session_runner.py consumes. This is the exact chain the live mini-repro
    exercised and found broken (zero segment_timeline keys in the trace)."""
    timeline = _segment_timeline()
    agent = _FakeContainerAgent(_container_response(timeline))
    tool = ContainerExecTool(agent)
    registry = ToolRegistry()
    registry.register(tool)

    spec = AgentRunSpec(
        initial_messages=[],
        tools=registry,
        model="replay-openclaw",
        max_iterations=1,
        max_tool_result_chars=100_000,
    )
    tool_call = ToolCallRequest(
        id="call_1", name="exec", arguments={"command": "cd /tmp && sleep 0.2"}
    )
    runner = AgentRunner(provider=None)

    (
        results,
        events,
        fatal_error,
        resource_timelines,
        segment_timelines,
        per_process_records,
        tool_timings,
    ) = asyncio.run(runner._execute_tools(spec, [tool_call], {}))

    assert fatal_error is None
    assert segment_timelines == {"call_1": timeline}
    assert per_process_records == {"call_1": [
        {"comm": "sleep", "pid": 11, "ppid": 10, "utime_s": 0.0,
         "stime_s": 0.0, "avg_mem_kb": 640, "exitcode": 0},
    ]}
    # The str result the LLM would see never carries the telemetry.
    assert "segment_timeline" not in str(results[0])
    assert "per_process" not in str(results[0])
    assert "call_1" in tool_timings


def test_execute_tools_omits_segment_timeline_for_non_exec_tools() -> None:
    """Tools without container-response segment telemetry must not appear in
    the dict at all (mirrors resource_timeline's None-is-absent contract)."""

    class _NoopTool:
        name = "noop"
        read_only = True

        def cast_params(self, params: dict[str, Any]) -> dict[str, Any]:
            return params

        def validate_params(self, params: dict[str, Any]) -> list[str]:
            return []

        async def execute(self, **_: Any) -> str:
            return "done"

    registry = ToolRegistry()
    registry.register(_NoopTool())
    spec = AgentRunSpec(
        initial_messages=[],
        tools=registry,
        model="replay-openclaw",
        max_iterations=1,
        max_tool_result_chars=100_000,
    )
    tool_call = ToolCallRequest(id="call_1", name="noop", arguments={})
    runner = AgentRunner(provider=None)

    (_, _, _, _, segment_timelines, _, _) = asyncio.run(
        runner._execute_tools(spec, [tool_call], {})
    )

    assert segment_timelines == {}
