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

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

# Skip the entire module if OpenClaw deps are unavailable.
pytest.importorskip("agents.openclaw._session_runner")

import agents.openclaw._session_runner as session_runner
from agents.openclaw._runner import AgentRunner, AgentRunSpec
from agents.openclaw._session_runner import (
    TraceCollectorHook,
    _any_file_newer_than,
    _resolve_run_outcome,
)
from agents.openclaw.tools.registry import ToolRegistry
from agents.openclaw.tools.shell import ExecTool
from llm_call.provider_base import ToolCallRequest


@pytest.fixture(autouse=True)
def _isolate_checkpoint_cas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        session_runner,
        "_CHECKPOINT_CAS_ROOT",
        tmp_path / "cas",
    )


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
        tool_structured_results: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.iteration = iteration
        self.messages = messages
        self.tool_calls = tool_calls or []
        self.usage = usage or {}
        self.response = response
        self.tool_resource_timelines = tool_resource_timelines or {}
        self.tool_structured_results = tool_structured_results or {}
        self.malformed_retry_count = 0


def test_trace_collector_emits_llm_call_action(tmp_path: Path) -> None:
    import asyncio

    asyncio.run(_drive_emits_llm_call_action(tmp_path))


def test_trace_collector_delta_mode_emits_message_deltas(tmp_path: Path) -> None:
    asyncio.run(_drive_delta_mode_emits_message_deltas(tmp_path))


def test_trace_collector_rejects_invalid_message_recording_mode(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="message_recording_mode"):
        TraceCollectorHook(
            tmp_path / "trace.jsonl",
            instance_id="test-invalid-mode",
            message_recording_mode="compact",
        )


def test_trace_collector_delta_mode_requires_append_only_messages(
    tmp_path: Path,
) -> None:
    asyncio.run(_drive_delta_mode_requires_append_only_messages(tmp_path))


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
    ctx_before_tools = _StubContext(
        iteration=0,
        messages=msgs_after_llm,
        tool_calls=[stub_tc],
        usage={"prompt_tokens": 100, "completion_tokens": 20},
    )
    await hook.before_execute_tools(ctx_before_tools)

    # Simulate tool result appended to messages
    msgs_after_tool = msgs_after_llm + [
        {"role": "tool", "tool_call_id": stub_tc.id, "name": "write_file", "content": "wrote a.py"}
    ]
    ctx_after = _StubContext(
        iteration=0,
        messages=msgs_after_tool,
        tool_calls=[stub_tc],
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
    assert llm["ts_start"] <= llm["ts_end"]


async def _drive_delta_mode_emits_message_deltas(tmp_path: Path) -> None:
    trace_file = tmp_path / "trace.jsonl"
    hook = TraceCollectorHook(
        trace_file,
        instance_id="test-delta",
        message_recording_mode="delta",
    )

    msgs_initial = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Run tests."},
    ]
    await hook.before_iteration(_StubContext(iteration=0, messages=msgs_initial))

    stub_tc = _StubToolCall("exec", {"command": "pytest"})
    assistant_tool_call = {
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
    msgs_after_llm = msgs_initial + [assistant_tool_call]
    await hook.before_execute_tools(
        _StubContext(iteration=0, messages=msgs_after_llm, tool_calls=[stub_tc])
    )
    tool_result = {
        "role": "tool",
        "tool_call_id": stub_tc.id,
        "name": "exec",
        "content": "ok",
    }
    msgs_after_tool = msgs_after_llm + [tool_result]
    await hook.after_iteration(
        _StubContext(
            iteration=0,
            messages=msgs_after_tool,
            tool_calls=[stub_tc],
            usage={"prompt_tokens": 100, "completion_tokens": 20},
            response=_StubResponse(content="", finish_reason="tool_calls"),
        )
    )

    await hook.before_iteration(_StubContext(iteration=1, messages=msgs_after_tool))
    await hook.after_iteration(
        _StubContext(
            iteration=1,
            messages=msgs_after_tool
            + [{"role": "assistant", "content": "Done."}],
            usage={"prompt_tokens": 120, "completion_tokens": 10},
            response=_StubResponse(content="Done.", finish_reason="stop"),
        )
    )
    hook.close()

    records = [json.loads(line) for line in trace_file.read_text().splitlines()]
    llm_calls = [
        record
        for record in records
        if record.get("type") == "action"
        and record.get("action_type") == "llm_call"
    ]
    assert len(llm_calls) == 2

    first_data = llm_calls[0]["data"]
    assert "messages_in" not in first_data
    assert first_data["messages_delta"] == msgs_initial
    assert first_data["is_delta"] is True

    second_data = llm_calls[1]["data"]
    assert "messages_in" not in second_data
    assert second_data["messages_delta"] == [assistant_tool_call, tool_result]
    assert second_data["is_delta"] is True

    start_events = [
        record
        for record in records
        if record.get("type") == "event" and record.get("event") == "llm_call_start"
    ]
    assert start_events[0]["data"]["messages_delta"] == msgs_initial
    assert start_events[1]["data"]["messages_delta"] == [
        assistant_tool_call,
        tool_result,
    ]


async def _drive_delta_mode_requires_append_only_messages(tmp_path: Path) -> None:
    hook = TraceCollectorHook(
        tmp_path / "trace.jsonl",
        instance_id="test-delta-prefix",
        message_recording_mode="delta",
    )
    try:
        await hook.before_iteration(
            _StubContext(
                iteration=0,
                messages=[{"role": "user", "content": "first"}],
            )
        )
        await hook.after_iteration(
            _StubContext(
                iteration=0,
                messages=[{"role": "user", "content": "first"}],
                response=_StubResponse(content="", finish_reason="stop"),
            )
        )
        with pytest.raises(ValueError, match="append-only"):
            await hook.before_iteration(
                _StubContext(
                    iteration=1,
                    messages=[{"role": "user", "content": "rewritten"}],
                )
            )
    finally:
        hook.close()


def test_trace_collector_emits_tool_resource_timeline(tmp_path: Path) -> None:
    asyncio.run(_drive_emits_tool_resource_timeline(tmp_path))


def test_agent_runner_carries_exec_structured_result(tmp_path: Path) -> None:
    asyncio.run(_drive_agent_runner_carries_exec_structured_result(tmp_path))


async def _drive_agent_runner_carries_exec_structured_result(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(ExecTool(working_dir=str(tmp_path)))
    runner = AgentRunner(provider=object())
    spec = AgentRunSpec(
        initial_messages=[],
        tools=registry,
        model="test",
        max_iterations=1,
        max_tool_result_chars=10_000,
    )

    command = "printf '%s\\n%s\\n' 'Error: literal stdout' 'Exit code: 7'"
    (
        results,
        events,
        fatal_error,
        _resource_timelines,
        structured_results,
    ) = await runner._execute_tools(
        spec,
        [
            ToolCallRequest(
                id="call_exec",
                name="exec",
                arguments={"command": command},
            )
        ],
        {},
    )

    assert fatal_error is None
    assert events[0]["status"] == "ok"
    assert "[Analyze the error above" not in str(results[0])
    assert structured_results == {
        "call_exec": {"returncode": 0, "timed_out": False}
    }


def test_trace_collector_emits_structured_exec_result(tmp_path: Path) -> None:
    asyncio.run(_drive_emits_structured_exec_result(tmp_path))


async def _drive_emits_structured_exec_result(tmp_path: Path) -> None:
    trace_file = tmp_path / "trace.jsonl"
    hook = TraceCollectorHook(trace_file, instance_id="test-structured")

    msgs_in = [{"role": "user", "content": "Run command."}]
    await hook.before_iteration(_StubContext(iteration=0, messages=msgs_in))
    stub_tc = _StubToolCall("exec", {"command": "printf"})
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
                        "arguments": '{"command":"printf"}',
                    },
                }
            ],
        }
    ]
    await hook.before_execute_tools(
        _StubContext(iteration=0, messages=msgs_after_llm, tool_calls=[stub_tc])
    )
    msgs_after_tool = msgs_after_llm + [
        {
            "role": "tool",
            "tool_call_id": stub_tc.id,
            "name": "exec",
            "content": "Error: literal stdout\nExit code: 7\n\nExit code: 0",
        }
    ]
    await hook.after_iteration(
        _StubContext(
            iteration=0,
            messages=msgs_after_tool,
            tool_calls=[stub_tc],
            response=_StubResponse(content="", finish_reason="tool_calls"),
            tool_structured_results={
                stub_tc.id: {"returncode": 0, "timed_out": False}
            },
        )
    )
    hook.close()

    records = [json.loads(line) for line in trace_file.read_text().splitlines()]
    tool_exec = next(
        record for record in records if record.get("action_type") == "tool_exec"
    )

    assert tool_exec["data"]["success"] is True
    assert tool_exec["data"]["success_source"] == "structured"
    assert tool_exec["data"]["returncode"] == 0
    assert tool_exec["data"]["timed_out"] is False


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
    await hook.before_execute_tools(
        _StubContext(iteration=0, messages=msgs_after_llm, tool_calls=[stub_tc])
    )
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
            tool_calls=[stub_tc],
            response=_StubResponse(content="", finish_reason="tool_calls"),
            tool_resource_timelines={stub_tc.id: resource_timeline},
        )
    )
    hook.close()

    records = [json.loads(line) for line in trace_file.read_text().splitlines()]
    tool_exec = next(record for record in records if record.get("action_type") == "tool_exec")
    assert tool_exec["data"]["resource_timeline"] == resource_timeline


def test_trace_collector_emits_exec_checkpoint_after(tmp_path: Path) -> None:
    asyncio.run(_drive_emits_exec_checkpoint_after(tmp_path))


def test_trace_collector_deferred_checkpoint_captures_on_next_tool_gate(
    tmp_path: Path,
) -> None:
    asyncio.run(_drive_deferred_checkpoint_captures_on_next_tool_gate(tmp_path))


def test_trace_collector_deferred_predictive_skip_records_verified_decision(
    tmp_path: Path,
) -> None:
    asyncio.run(_drive_deferred_predictive_skip_records_verified_decision(tmp_path))


async def _drive_emits_exec_checkpoint_after(tmp_path: Path) -> None:
    trace_file = tmp_path / "trace.jsonl"
    testbed = tmp_path / "testbed"
    checkpoint_dir = tmp_path / "runtime" / "checkpoints"
    testbed.mkdir()
    (testbed / "result.txt").write_text("source state\n", encoding="utf-8")
    hook = TraceCollectorHook(
        trace_file,
        instance_id="test-checkpoint",
        checkpoint_root=testbed,
        checkpoint_dir=checkpoint_dir,
        checkpoint_root_label="/testbed",
    )

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
                    "function": {"name": "exec", "arguments": '{"command":"pytest"}'},
                }
            ],
        }
    ]
    await hook.before_execute_tools(
        _StubContext(iteration=0, messages=msgs_after_llm, tool_calls=[stub_tc])
    )
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
            tool_calls=[stub_tc],
            response=_StubResponse(content="", finish_reason="tool_calls"),
        )
    )
    hook.close()

    records = [json.loads(line) for line in trace_file.read_text().splitlines()]
    tool_exec = next(record for record in records if record.get("action_type") == "tool_exec")
    checkpoint_after = tool_exec["data"]["checkpoint_after"]
    checkpoint_path = trace_file.parent / checkpoint_after["path"]

    assert checkpoint_after["kind"] == "cas_manifest_full"
    assert checkpoint_after["incremental"] is False
    assert checkpoint_after["incremental_since_ns"] is None
    assert checkpoint_after["root"] == "/testbed"
    assert checkpoint_after["overhead_excluded"] is True
    assert checkpoint_after["elapsed_ms"] >= 0
    assert checkpoint_after["size_bytes"] == checkpoint_path.stat().st_size
    assert checkpoint_after["chain_bytes"] == len("source state\n")
    manifest = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert sorted(manifest) == ["deleted_paths", "entries"]
    assert manifest["deleted_paths"] == []
    result_entry = manifest["entries"]["result.txt"]
    assert result_entry["size"] == len("source state\n")
    assert result_entry["mode"] == (testbed / "result.txt").stat().st_mode & 0o777
    blob_path = (
        session_runner._CHECKPOINT_CAS_ROOT
        / "blobs"
        / result_entry["hash"][:2]
        / result_entry["hash"][2:]
    )
    assert blob_path.exists()


async def _drive_deferred_checkpoint_captures_on_next_tool_gate(
    tmp_path: Path,
) -> None:
    trace_file = tmp_path / "trace.jsonl"
    testbed = tmp_path / "testbed"
    checkpoint_dir = tmp_path / "runtime" / "checkpoints"
    testbed.mkdir()
    (testbed / "result.txt").write_text("source state\n", encoding="utf-8")
    hook = TraceCollectorHook(
        trace_file,
        instance_id="test-deferred-checkpoint",
        checkpoint_root=testbed,
        checkpoint_dir=checkpoint_dir,
        checkpoint_root_label="/testbed",
        checkpoint_scheduling="deferred",
    )

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
                    "function": {"name": "exec", "arguments": '{"command":"pytest"}'},
                }
            ],
        }
    ]
    await hook.before_execute_tools(
        _StubContext(iteration=0, messages=msgs_after_llm, tool_calls=[stub_tc])
    )
    await hook.after_iteration(
        _StubContext(
            iteration=0,
            messages=msgs_after_llm
            + [
                {
                    "role": "tool",
                    "tool_call_id": stub_tc.id,
                    "name": "exec",
                    "content": "ok",
                }
            ],
            tool_calls=[stub_tc],
            response=_StubResponse(content="", finish_reason="tool_calls"),
        )
    )

    next_tc = _StubToolCall("exec", {"command": "cat result.txt"})
    await hook.before_execute_tools(
        _StubContext(iteration=1, messages=[], tool_calls=[next_tc])
    )
    hook.close()

    records = [json.loads(line) for line in trace_file.read_text().splitlines()]
    tool_exec = next(record for record in records if record.get("action_type") == "tool_exec")
    checkpoint_after = tool_exec["data"]["checkpoint_after"]
    checkpoint_path = trace_file.parent / checkpoint_after["path"]

    assert checkpoint_after["kind"] == "cas_manifest_full"
    assert checkpoint_after["incremental"] is False
    assert checkpoint_after["probe_result"] == "initial"
    assert checkpoint_after["overhead_excluded"] is True
    assert checkpoint_after["size_bytes"] == checkpoint_path.stat().st_size
    assert tool_exec["data"]["checkpoint_exposed_ms"] >= 0


async def _drive_deferred_predictive_skip_records_verified_decision(
    tmp_path: Path,
) -> None:
    trace_file = tmp_path / "trace.jsonl"
    testbed = tmp_path / "testbed"
    checkpoint_dir = tmp_path / "runtime" / "checkpoints"
    testbed.mkdir()
    (testbed / "result.txt").write_text("source state\n", encoding="utf-8")
    hook = TraceCollectorHook(
        trace_file,
        instance_id="test-deferred-skip",
        checkpoint_root=testbed,
        checkpoint_dir=checkpoint_dir,
        checkpoint_root_label="/testbed",
        checkpoint_scheduling="deferred",
    )

    first_action_data: dict[str, Any] = {}
    first = await hook._checkpoint_after_tool_deferred(
        tool_call_id="call_first",
        tool_name="exec",
        tool_args_json='{"command":"pytest"}',
        action_data=first_action_data,
        source_concurrent_execs=False,
    )
    assert first is None
    await hook._await_pending_checkpoint_captures()
    assert first_action_data["checkpoint_after"]["kind"] == "cas_manifest_full"

    read_only_action_data: dict[str, Any] = {}
    read_only = await hook._checkpoint_after_tool_deferred(
        tool_call_id="call_read",
        tool_name="exec",
        tool_args_json='{"command":"ls"}',
        action_data=read_only_action_data,
        source_concurrent_execs=False,
    )
    assert read_only is not None
    assert read_only["skipped"] == "predicted_read_only"
    assert read_only["probe_result"] == "skipped"
    assert read_only["checkpoint_decision"] == "predicted_read_only"
    assert read_only["predicted_family"] == "read_only"

    unknown_action_data: dict[str, Any] = {}
    unknown = await hook._checkpoint_after_tool_deferred(
        tool_call_id="call_unknown",
        tool_name="exec",
        tool_args_json='{"command":"python -m pytest"}',
        action_data=unknown_action_data,
        source_concurrent_execs=False,
    )
    assert unknown is not None
    assert unknown["skipped"] == "no filesystem changes since last checkpoint"
    assert unknown["probe_result"] == "unchanged"
    assert unknown["checkpoint_decision"] == "predicted_unknown_probe_verified"
    assert unknown["predicted_family"] == "unknown"
    hook.close()


def test_trace_collector_checkpoints_testbed_symlink(
    tmp_path: Path,
) -> None:
    asyncio.run(_drive_checkpoints_testbed_symlink(tmp_path))


def test_any_file_newer_than_detects_changes(tmp_path: Path) -> None:
    testbed = tmp_path / "testbed"
    testbed.mkdir()
    marker_mtime_ns = time.time_ns()
    mtime_delta_ns = 1_000_000_000
    old_mtime_ns = marker_mtime_ns - mtime_delta_ns
    new_mtime_ns = marker_mtime_ns + mtime_delta_ns

    assert _any_file_newer_than(testbed, marker_mtime_ns) is False

    old_file = testbed / "old.txt"
    old_file.write_text("old\n", encoding="utf-8")
    os.utime(old_file, ns=(old_mtime_ns, old_mtime_ns))
    os.utime(testbed, ns=(old_mtime_ns, old_mtime_ns))
    assert _any_file_newer_than(testbed, marker_mtime_ns) is False

    os.utime(old_file, ns=(new_mtime_ns, new_mtime_ns))
    assert _any_file_newer_than(testbed, marker_mtime_ns) is True

    os.utime(old_file, ns=(old_mtime_ns, old_mtime_ns))
    os.utime(testbed, ns=(old_mtime_ns, old_mtime_ns))
    new_file = testbed / "new.txt"
    new_file.write_text("new\n", encoding="utf-8")
    os.utime(new_file, ns=(new_mtime_ns, new_mtime_ns))
    assert _any_file_newer_than(testbed, marker_mtime_ns) is True

    new_file.unlink()
    os.utime(testbed, ns=(old_mtime_ns, old_mtime_ns))
    empty_dir = testbed / "empty-dir"
    empty_dir.mkdir()
    os.utime(empty_dir, ns=(new_mtime_ns, new_mtime_ns))
    assert _any_file_newer_than(testbed, marker_mtime_ns) is True

    empty_dir.rmdir()
    os.utime(testbed, ns=(old_mtime_ns, old_mtime_ns))
    deleted_file = testbed / "deleted.txt"
    deleted_file.write_text("deleted\n", encoding="utf-8")
    os.utime(deleted_file, ns=(old_mtime_ns, old_mtime_ns))
    os.utime(testbed, ns=(old_mtime_ns, old_mtime_ns))
    deleted_file.unlink()
    os.utime(testbed, ns=(new_mtime_ns, new_mtime_ns))
    assert _any_file_newer_than(testbed, marker_mtime_ns) is True


def test_write_cas_manifest_reuses_hash_cache_and_deduplicates_blobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    testbed = tmp_path / "testbed"
    testbed.mkdir()
    first_file = testbed / "first.txt"
    second_file = testbed / "second.txt"
    content = b"shared content\n"
    first_file.write_bytes(content)
    second_file.write_bytes(content)
    first_stat = os.lstat(first_file)
    second_stat = os.lstat(second_file)
    digest = hashlib.sha256(content).hexdigest()
    hash_cache = {
        "first.txt": (first_stat.st_size, first_stat.st_mtime_ns, digest),
        "second.txt": (second_stat.st_size, second_stat.st_mtime_ns, digest),
    }

    def fail_sha256(data: bytes) -> Any:
        raise AssertionError(f"unexpected hash cache miss for {len(data)} bytes")

    monkeypatch.setattr(session_runner.hashlib, "sha256", fail_sha256)

    manifest_path = tmp_path / "checkpoints" / "manifest.json"
    chain_bytes = session_runner._write_cas_manifest(
        root=testbed,
        manifest_path=manifest_path,
        hash_cache=hash_cache,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest["entries"]) == {"first.txt", "second.txt"}
    assert manifest["entries"]["first.txt"]["hash"] == digest
    assert manifest["entries"]["second.txt"]["hash"] == digest
    assert chain_bytes == len(content)
    blob_path = (
        session_runner._CHECKPOINT_CAS_ROOT / "blobs" / digest[:2] / digest[2:]
    )
    assert blob_path.read_bytes() == content


def test_trace_collector_skips_checkpoint_when_testbed_unchanged(
    tmp_path: Path,
) -> None:
    trace_file = tmp_path / "trace.jsonl"
    testbed = tmp_path / "testbed"
    checkpoint_dir = tmp_path / "runtime" / "checkpoints"
    testbed.mkdir()
    (testbed / "result.txt").write_text("source state\n", encoding="utf-8")
    hook = TraceCollectorHook(
        trace_file,
        instance_id="test-smart-checkpoint",
        checkpoint_root=testbed,
        checkpoint_dir=checkpoint_dir,
        checkpoint_root_label="/testbed",
    )

    first = hook._checkpoint_after_tool(
        tool_call_id="call_first",
        tool_name="exec",
        tool_args_json='{"command":"ls"}',
    )
    assert first is not None
    assert first["kind"] == "cas_manifest_full"
    assert first["incremental"] is False
    assert (trace_file.parent / first["path"]).exists()

    second = hook._checkpoint_after_tool(
        tool_call_id="call_second",
        tool_name="exec",
        tool_args_json='{"command":"cat result.txt"}',
    )
    assert second is not None
    assert second["skipped"] == "no filesystem changes since last checkpoint"
    assert second["overhead_excluded"] is True
    assert second["elapsed_ms"] >= 0
    assert not (checkpoint_dir / "call_second-manifest.json").exists()

    new_file = testbed / "new.txt"
    new_file.write_text("new state\n", encoding="utf-8")
    assert hook._last_incremental_checkpoint_ns is not None
    new_mtime_ns = time.time_ns()
    assert new_mtime_ns > hook._last_incremental_checkpoint_ns
    os.utime(new_file, ns=(new_mtime_ns, new_mtime_ns))
    third = hook._checkpoint_after_tool(
        tool_call_id="call_third",
        tool_name="exec",
        tool_args_json='{"command":"echo new > new.txt"}',
    )
    assert third is not None
    assert third["kind"] == "cas_manifest_incremental"
    assert third["incremental"] is True
    assert (trace_file.parent / third["path"]).exists()
    hook.close()


def test_trace_collector_incremental_checkpoint_only_manifests_changed_files(
    tmp_path: Path,
) -> None:
    trace_file = tmp_path / "trace.jsonl"
    testbed = tmp_path / "testbed"
    checkpoint_dir = tmp_path / "runtime" / "checkpoints"
    testbed.mkdir()
    (testbed / "large.bin").write_bytes(b"x" * 1024 * 1024)
    changed_file = testbed / "result.txt"
    changed_file.write_text("source state\n", encoding="utf-8")
    hook = TraceCollectorHook(
        trace_file,
        instance_id="test-incremental-checkpoint",
        checkpoint_root=testbed,
        checkpoint_dir=checkpoint_dir,
        checkpoint_root_label="/testbed",
    )

    first = hook._checkpoint_after_tool(
        tool_call_id="call_first",
        tool_name="exec",
        tool_args_json='{"command":"pytest"}',
    )
    assert first is not None
    assert first["kind"] == "cas_manifest_full"
    previous_marker_ns = hook._last_incremental_checkpoint_ns
    assert previous_marker_ns is not None

    changed_file.write_text("changed state\n", encoding="utf-8")
    changed_mtime_ns = max(time.time_ns(), previous_marker_ns + 1)
    os.utime(changed_file, ns=(changed_mtime_ns, changed_mtime_ns))
    second = hook._checkpoint_after_tool(
        tool_call_id="call_second",
        tool_name="exec",
        tool_args_json='{"command":"pytest"}',
    )
    assert second is not None
    assert second["kind"] == "cas_manifest_incremental"
    assert second["incremental"] is True
    assert second["incremental_since_ns"] == previous_marker_ns
    second_path = trace_file.parent / second["path"]

    assert second["size_bytes"] == second_path.stat().st_size
    assert second["chain_bytes"] == len("changed state\n")
    manifest = json.loads(second_path.read_text(encoding="utf-8"))
    assert set(manifest["entries"]) == {"result.txt"}
    assert manifest["entries"]["result.txt"]["size"] == len("changed state\n")
    hook.close()


def test_trace_collector_incremental_checkpoint_detects_backdated_size_change(
    tmp_path: Path,
) -> None:
    trace_file = tmp_path / "trace.jsonl"
    testbed = tmp_path / "testbed"
    checkpoint_dir = tmp_path / "runtime" / "checkpoints"
    testbed.mkdir()
    backdated_file = testbed / "backdated.txt"
    trigger_file = testbed / "trigger.txt"
    backdated_file.write_text("old\n", encoding="utf-8")
    trigger_file.write_text("old trigger\n", encoding="utf-8")
    original_mtime_ns = os.lstat(backdated_file).st_mtime_ns
    hook = TraceCollectorHook(
        trace_file,
        instance_id="test-backdated-size-change",
        checkpoint_root=testbed,
        checkpoint_dir=checkpoint_dir,
        checkpoint_root_label="/testbed",
    )

    first = hook._checkpoint_after_tool(
        tool_call_id="call_first",
        tool_name="exec",
        tool_args_json='{"command":"pytest"}',
    )
    assert first is not None
    assert first["kind"] == "cas_manifest_full"
    previous_marker_ns = hook._last_incremental_checkpoint_ns
    assert previous_marker_ns is not None

    changed_content = b"changed with different size\n"
    backdated_file.write_bytes(changed_content)
    os.utime(backdated_file, ns=(original_mtime_ns, original_mtime_ns))
    trigger_file.write_text("trigger changed\n", encoding="utf-8")
    trigger_mtime_ns = max(time.time_ns(), previous_marker_ns + 1)
    os.utime(trigger_file, ns=(trigger_mtime_ns, trigger_mtime_ns))

    second = hook._checkpoint_after_tool(
        tool_call_id="call_second",
        tool_name="exec",
        tool_args_json='{"command":"pytest"}',
    )
    assert second is not None
    assert second["kind"] == "cas_manifest_incremental"
    second_path = trace_file.parent / second["path"]
    manifest = json.loads(second_path.read_text(encoding="utf-8"))
    backdated_entry = manifest["entries"]["backdated.txt"]
    assert set(manifest["entries"]) == {"backdated.txt", "trigger.txt"}
    assert backdated_entry["size"] == len(changed_content)
    assert backdated_entry["mtime_ns"] == original_mtime_ns
    assert backdated_entry["hash"] == hashlib.sha256(changed_content).hexdigest()
    hook.close()


def test_trace_collector_incremental_checkpoint_detects_backdated_size_change_without_trigger(
    tmp_path: Path,
) -> None:
    trace_file = tmp_path / "trace.jsonl"
    testbed = tmp_path / "testbed"
    checkpoint_dir = tmp_path / "runtime" / "checkpoints"
    testbed.mkdir()
    backdated_file = testbed / "backdated.txt"
    backdated_file.write_text("old\n", encoding="utf-8")
    original_file_mtime_ns = os.lstat(backdated_file).st_mtime_ns
    original_dir_mtime_ns = os.lstat(testbed).st_mtime_ns
    hook = TraceCollectorHook(
        trace_file,
        instance_id="test-backdated-size-change-without-trigger",
        checkpoint_root=testbed,
        checkpoint_dir=checkpoint_dir,
        checkpoint_root_label="/testbed",
    )

    first = hook._checkpoint_after_tool(
        tool_call_id="call_first",
        tool_name="exec",
        tool_args_json='{"command":"pytest"}',
    )
    assert first is not None
    assert first["kind"] == "cas_manifest_full"

    changed_content = b"changed with different size\n"
    backdated_file.write_bytes(changed_content)
    os.utime(backdated_file, ns=(original_file_mtime_ns, original_file_mtime_ns))
    os.utime(testbed, ns=(original_dir_mtime_ns, original_dir_mtime_ns))

    second = hook._checkpoint_after_tool(
        tool_call_id="call_second",
        tool_name="exec",
        tool_args_json='{"command":"pytest"}',
    )
    assert second is not None
    assert second["kind"] == "cas_manifest_incremental"
    second_path = trace_file.parent / second["path"]
    manifest = json.loads(second_path.read_text(encoding="utf-8"))
    backdated_entry = manifest["entries"]["backdated.txt"]
    assert set(manifest["entries"]) == {"backdated.txt"}
    assert backdated_entry["size"] == len(changed_content)
    assert backdated_entry["mtime_ns"] == original_file_mtime_ns
    assert backdated_entry["hash"] == hashlib.sha256(changed_content).hexdigest()
    hook.close()


async def _drive_checkpoints_testbed_symlink(tmp_path: Path) -> None:
    trace_file = tmp_path / "trace.jsonl"
    testbed = tmp_path / "testbed"
    testbed.mkdir()
    (testbed / "target.txt").write_text("target\n", encoding="utf-8")
    (testbed / "link.txt").symlink_to("target.txt")
    hook = TraceCollectorHook(
        trace_file,
        instance_id="test-symlink",
        checkpoint_root=testbed,
        checkpoint_dir=tmp_path / "runtime" / "checkpoints",
        checkpoint_root_label="/testbed",
    )

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
                    "function": {"name": "exec", "arguments": '{"command":"pytest"}'},
                }
            ],
        }
    ]
    await hook.before_execute_tools(
        _StubContext(iteration=0, messages=msgs_after_llm, tool_calls=[stub_tc])
    )
    await hook.after_iteration(
        _StubContext(
            iteration=0,
            messages=msgs_after_llm
            + [
                {
                    "role": "tool",
                    "tool_call_id": stub_tc.id,
                    "name": "exec",
                    "content": "ok",
                }
            ],
            tool_calls=[stub_tc],
            response=_StubResponse(content="", finish_reason="tool_calls"),
        )
    )
    hook.close()

    records = [json.loads(line) for line in trace_file.read_text().splitlines()]
    tool_exec = next(record for record in records if record.get("action_type") == "tool_exec")
    assert "checkpoint_after_error" not in tool_exec["data"]
    checkpoint_after = tool_exec["data"]["checkpoint_after"]
    assert checkpoint_after["kind"] == "cas_manifest_full"
    assert checkpoint_after["overhead_excluded"] is True
    assert checkpoint_after["elapsed_ms"] >= 0
    checkpoint_path = trace_file.parent / checkpoint_after["path"]
    manifest = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert manifest["entries"]["link.txt"] == {
        "type": "symlink",
        "target": "target.txt",
    }


def test_trace_collector_checkpoints_exec_in_multi_tool_iteration(
    tmp_path: Path,
) -> None:
    asyncio.run(_drive_checkpoints_exec_in_multi_tool_iteration(tmp_path))


async def _drive_checkpoints_exec_in_multi_tool_iteration(tmp_path: Path) -> None:
    trace_file = tmp_path / "trace.jsonl"
    testbed = tmp_path / "testbed"
    testbed.mkdir()
    (testbed / "result.txt").write_text("source state\n", encoding="utf-8")
    hook = TraceCollectorHook(
        trace_file,
        instance_id="test-multi-tool",
        checkpoint_root=testbed,
        checkpoint_dir=tmp_path / "runtime" / "checkpoints",
        checkpoint_root_label="/testbed",
    )

    msgs_in = [{"role": "user", "content": "Run test."}]
    await hook.before_iteration(_StubContext(iteration=0, messages=msgs_in))
    exec_tc = _StubToolCall("exec", {"command": "pytest"})
    exec_tc_2 = _StubToolCall("exec", {"command": "python -m pytest"})
    batched_exec_tc = _StubToolCall("exec", {"commands": ["pytest", "ruff check ."]})
    read_tc = _StubToolCall("read_file", {"path": "x.txt"})
    msgs_after_llm = msgs_in + [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": exec_tc.id,
                    "type": "function",
                    "function": {"name": "exec", "arguments": '{"command":"pytest"}'},
                },
                {
                    "id": exec_tc_2.id,
                    "type": "function",
                    "function": {
                        "name": "exec",
                        "arguments": '{"command":"python -m pytest"}',
                    },
                },
                {
                    "id": batched_exec_tc.id,
                    "type": "function",
                    "function": {
                        "name": "exec",
                        "arguments": '{"commands":["pytest","ruff check ."]}',
                    },
                },
                {
                    "id": read_tc.id,
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"x.txt"}'},
                },
            ],
        }
    ]
    await hook.before_execute_tools(
        _StubContext(
            iteration=0,
            messages=msgs_after_llm,
            tool_calls=[exec_tc, exec_tc_2, batched_exec_tc, read_tc],
        )
    )
    await hook.after_iteration(
        _StubContext(
            iteration=0,
            messages=msgs_after_llm
            + [
                {
                    "role": "tool",
                    "tool_call_id": exec_tc.id,
                    "name": "exec",
                    "content": "ok",
                },
                {
                    "role": "tool",
                    "tool_call_id": exec_tc_2.id,
                    "name": "exec",
                    "content": "ok 2",
                },
                {
                    "role": "tool",
                    "tool_call_id": batched_exec_tc.id,
                    "name": "exec",
                    "content": "batched ok",
                },
                {
                    "role": "tool",
                    "tool_call_id": read_tc.id,
                    "name": "read_file",
                    "content": "1| x",
                },
            ],
            tool_calls=[exec_tc, exec_tc_2, batched_exec_tc, read_tc],
            response=_StubResponse(content="", finish_reason="tool_calls"),
        )
    )
    hook.close()

    records = [json.loads(line) for line in trace_file.read_text().splitlines()]
    records_by_tool_id = {
        record["data"]["tool_call_id"]: record
        for record in records
        if record.get("type") == "action"
        and (record.get("data") or {}).get("tool_call_id") is not None
    }
    checkpoint_after = records_by_tool_id[exec_tc.id]["data"]["checkpoint_after"]
    checkpoint_path = trace_file.parent / checkpoint_after["path"]
    assert records_by_tool_id[exec_tc.id]["data"]["source_concurrent_execs"] is True
    assert records_by_tool_id[exec_tc.id]["data"]["smeared_checkpoint"] is True
    assert checkpoint_after["kind"] == "cas_manifest_full"
    assert checkpoint_after["incremental"] is False
    assert checkpoint_after["root"] == "/testbed"
    assert checkpoint_after["overhead_excluded"] is True
    assert checkpoint_after["elapsed_ms"] >= 0
    assert checkpoint_after["size_bytes"] == checkpoint_path.stat().st_size
    manifest = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert "result.txt" in manifest["entries"]

    skipped_checkpoint = records_by_tool_id[exec_tc_2.id]["data"]["checkpoint_after"]
    assert records_by_tool_id[exec_tc_2.id]["data"]["source_concurrent_execs"] is True
    assert records_by_tool_id[exec_tc_2.id]["data"]["smeared_checkpoint"] is True
    assert skipped_checkpoint["skipped"] == "no filesystem changes since last checkpoint"
    assert skipped_checkpoint["overhead_excluded"] is True
    assert skipped_checkpoint["elapsed_ms"] >= 0

    for tool_call_id in (batched_exec_tc.id, read_tc.id):
        tool_data = records_by_tool_id[tool_call_id]["data"]
        assert tool_data["source_concurrent_execs"] is True
        assert "checkpoint_after" not in tool_data
        assert "checkpoint_after_error" not in tool_data
        assert "smeared_checkpoint" not in tool_data


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
    # ts_end falls back to "now" when before_execute_tools was never called
    assert llm_calls[0]["ts_end"] >= llm_calls[0]["ts_start"]


def test_trace_collector_records_openrouter_latency_fields(tmp_path: Path) -> None:
    import asyncio

    asyncio.run(_drive_openrouter_latency_fields(tmp_path))


async def _drive_openrouter_latency_fields(tmp_path: Path) -> None:
    trace_file = tmp_path / "trace.jsonl"
    hook = TraceCollectorHook(trace_file, instance_id="test-openrouter")

    await hook.before_iteration(
        _StubContext(iteration=0, messages=[{"role": "user", "content": "Ping"}])
    )
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
    await hook.after_iteration(
        _StubContext(
            iteration=0,
            messages=[
                {"role": "user", "content": "Ping"},
                {"role": "assistant", "content": "pong"},
            ],
            usage={"prompt_tokens": 12, "completion_tokens": 3},
            response=response,
        )
    )
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

    await hook.before_iteration(
        _StubContext(iteration=0, messages=[{"role": "user", "content": "Ping"}])
    )
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

    await hook.after_iteration(
        _StubContext(
            iteration=0,
            messages=[
                {"role": "user", "content": "Ping"},
                {"role": "assistant", "content": "pong"},
            ],
            usage={"prompt_tokens": 12, "completion_tokens": 3},
            response=response,
        )
    )
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
