"""Tests for trace_collect.trace_inspector (v4 action/event format)."""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from trace_collect.trace_inspector import (
    TraceData,
    cmd_overview,
    cmd_step,
    cmd_messages,
    cmd_response,
    cmd_events,
    cmd_tools,
    cmd_search,
)


def _llm_call_action(iteration: int, ts_start: float, ts_end: float,
                     prompt_tokens: int, completion_tokens: int,
                     content: str, agent_id: str = "django__django-11734",
                     messages: list | None = None) -> dict:
    return {
        "type": "action",
        "action_type": "llm_call",
        "action_id": f"llm_{iteration}",
        "agent_id": agent_id,
        "program_id": agent_id,
        "iteration": iteration,
        "ts_start": ts_start,
        "ts_end": ts_end,
        "data": {
            "messages_in": messages or [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Fix the bug"},
            ],
            "raw_response": {
                "choices": [{"message": {"content": content, "tool_calls": []}}]
            },
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "llm_latency_ms": (ts_end - ts_start) * 1000,
        },
    }


def _tool_exec_action(iteration: int, ts_start: float, ts_end: float,
                      tool_name: str, tool_args: str, result: str,
                      duration_ms: float, agent_id: str = "django__django-11734") -> dict:
    return {
        "type": "action",
        "action_type": "tool_exec",
        "action_id": f"tool_{iteration}_{tool_name}",
        "agent_id": agent_id,
        "program_id": agent_id,
        "iteration": iteration,
        "ts_start": ts_start,
        "ts_end": ts_end,
        "data": {
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_result": result,
            "duration_ms": duration_ms,
            "success": True,
        },
    }


@pytest.fixture
def sample_trace(tmp_path: Path) -> Path:
    """Minimal v4 trace: 2 iterations, each with llm_call + tool_exec actions."""
    trace_file = tmp_path / "test_trace.jsonl"
    records = [
        {
            "type": "trace_metadata",
            "scaffold": "mini-swe-agent",
            "trace_format_version": 5,
            "mode": "collect",
            "model": "test-model",
        },
        _llm_call_action(0, 1000.0, 1001.5, 100, 50, "Let me search"),
        _tool_exec_action(0, 1001.5, 1001.55, "bash",
                          '{"command": "find . -name models.py"}',
                          "./models.py", 50.0),
        _llm_call_action(
            1, 1002.0, 1004.0, 200, 80, "Reading file",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Fix the bug"},
                {"role": "assistant", "content": "Let me search"},
            ],
        ),
        _tool_exec_action(1, 1004.0, 1004.03, "bash",
                          '{"command": "cat models.py"}',
                          "class Model:\n    pass", 30.0),
        {
            "type": "summary",
            "agent_id": "django__django-11734",
            "n_iterations": 2,
            "elapsed_s": 4.0,
            "total_llm_ms": 3500,
            "total_tool_ms": 80,
            "total_tokens": 430,
            "success": True,
        },
    ]
    with open(trace_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return trace_file


# ── TraceData.load ─────────────────────────────────────────────────


def test_load_trace(sample_trace: Path) -> None:
    data = TraceData.load(sample_trace)
    assert len(data.actions) == 4  # 2 llm_call + 2 tool_exec
    assert len(data.events) == 0
    assert len(data.summaries) == 1
    assert data.metadata["scaffold"] == "mini-swe-agent"
    assert data.metadata["model"] == "test-model"
    # Sorted by (iteration, ts_start)
    assert data.actions[0]["iteration"] == 0
    assert data.actions[0]["action_type"] == "llm_call"
    assert data.actions[1]["iteration"] == 0
    assert data.actions[1]["action_type"] == "tool_exec"


def test_load_with_agent_filter(sample_trace: Path) -> None:
    data = TraceData.load(sample_trace, agent_filter="django__django")
    assert len(data.actions) == 4
    assert "django__django-11734" in data.agents

    data_empty = TraceData.load(sample_trace, agent_filter="nonexistent_agent")
    assert len(data_empty.actions) == 0


def test_load_skips_malformed_lines(tmp_path: Path) -> None:
    trace_file = tmp_path / "malformed.jsonl"
    trace_file.write_text(
        '{"type": "trace_metadata", "scaffold": "mini-swe-agent", "trace_format_version": 5}\n'
        "NOT VALID JSON {{{\n"
        '{"type": "action", "action_type": "llm_call", "action_id": "llm_0",'
        ' "agent_id": "a1", "iteration": 0, "data": {}}\n'
    )
    data = TraceData.load(trace_file)
    assert len(data.actions) == 1


# ── cmd_overview ───────────────────────────────────────────────────


def test_cmd_overview(sample_trace: Path, capsys: pytest.CaptureFixture) -> None:
    data = TraceData.load(sample_trace)
    cmd_overview(data)
    out = capsys.readouterr().out
    assert "mini-swe-agent" in out
    assert "test-model" in out
    assert "bash" in out


def test_cmd_overview_json(sample_trace: Path, capsys: pytest.CaptureFixture) -> None:
    data = TraceData.load(sample_trace)
    cmd_overview(data, as_json=True)
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["scaffold"] == "mini-swe-agent"
    assert parsed["model"] == "test-model"
    assert parsed["tool_counts"]["bash"] == 2
    assert parsed["total_tokens"] == 430  # 100+50 + 200+80


# ── cmd_step (now lookup by iteration) ─────────────────────────────


def test_cmd_step_valid(sample_trace: Path, capsys: pytest.CaptureFixture) -> None:
    data = TraceData.load(sample_trace)
    cmd_step(data, 0)
    out = capsys.readouterr().out
    assert "Step 0" in out


def test_cmd_step_invalid(sample_trace: Path, capsys: pytest.CaptureFixture) -> None:
    data = TraceData.load(sample_trace)
    cmd_step(data, 99)
    out = capsys.readouterr().out
    assert "ERROR" in out
    assert "99" in out


def test_cmd_step_json(sample_trace: Path, capsys: pytest.CaptureFixture) -> None:
    data = TraceData.load(sample_trace)
    cmd_step(data, 0, as_json=True)
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["iteration"] == 0


# ── cmd_messages ───────────────────────────────────────────────────


def test_cmd_messages(sample_trace: Path, capsys: pytest.CaptureFixture) -> None:
    data = TraceData.load(sample_trace)
    cmd_messages(data, 0)
    out = capsys.readouterr().out
    assert "system" in out
    assert "user" in out
    assert "helpful assistant" in out


def test_cmd_messages_role_filter(
    sample_trace: Path, capsys: pytest.CaptureFixture
) -> None:
    data = TraceData.load(sample_trace)
    cmd_messages(data, 0, role_filter="user")
    out = capsys.readouterr().out
    assert "user" in out
    assert "system" not in out


def test_cmd_messages_json(sample_trace: Path, capsys: pytest.CaptureFixture) -> None:
    data = TraceData.load(sample_trace)
    cmd_messages(data, 0, as_json=True)
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert parsed[0]["role"] == "system"


# ── cmd_response ───────────────────────────────────────────────────


def test_cmd_response(sample_trace: Path, capsys: pytest.CaptureFixture) -> None:
    data = TraceData.load(sample_trace)
    cmd_response(data, 0)
    out = capsys.readouterr().out
    assert "choices" in out
    assert "Let me search" in out


def test_cmd_response_json(sample_trace: Path, capsys: pytest.CaptureFixture) -> None:
    data = TraceData.load(sample_trace)
    cmd_response(data, 0, as_json=True)
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert "raw_response" in parsed


# ── cmd_tools ──────────────────────────────────────────────────────


def test_cmd_tools(sample_trace: Path, capsys: pytest.CaptureFixture) -> None:
    data = TraceData.load(sample_trace)
    cmd_tools(data)
    out = capsys.readouterr().out
    assert "bash" in out
    assert "2" in out


def test_cmd_tools_json(sample_trace: Path, capsys: pytest.CaptureFixture) -> None:
    data = TraceData.load(sample_trace)
    cmd_tools(data, as_json=True)
    out = capsys.readouterr().out
    parsed = json.loads(out)
    bash_row = next(r for r in parsed if r["tool_name"] == "bash")
    assert bash_row["count"] == 2
    assert bash_row["success_rate"] == 1.0
    assert bash_row["total_duration_ms"] == 80.0


def test_cmd_tools_step_filter(
    sample_trace: Path, capsys: pytest.CaptureFixture
) -> None:
    data = TraceData.load(sample_trace)
    cmd_tools(data, step_idx=0, as_json=True)
    out = capsys.readouterr().out
    parsed = json.loads(out)
    bash_row = next(r for r in parsed if r["tool_name"] == "bash")
    assert bash_row["count"] == 1
    assert bash_row["total_duration_ms"] == 50.0


# ── cmd_search ─────────────────────────────────────────────────────


def test_cmd_search_match(sample_trace: Path, capsys: pytest.CaptureFixture) -> None:
    data = TraceData.load(sample_trace)
    cmd_search(data, "search")
    out = capsys.readouterr().out
    assert "iter 0" in out.lower() or "step 0" in out.lower()
    assert "search" in out.lower()


def test_cmd_search_no_match(sample_trace: Path, capsys: pytest.CaptureFixture) -> None:
    data = TraceData.load(sample_trace)
    cmd_search(data, "zzz_no_match_xyz")
    out = capsys.readouterr().out
    assert "No matches" in out


def test_cmd_search_json(sample_trace: Path, capsys: pytest.CaptureFixture) -> None:
    data = TraceData.load(sample_trace)
    cmd_search(data, "Reading", as_json=True)
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert len(parsed) == 1
    assert parsed[0]["iteration"] == 1


# ── Events fixture ─────────────────────────────────────────────────


@pytest.fixture
def trace_with_events(tmp_path: Path) -> Path:
    """v4 trace with envelope events for testing cmd_events."""
    trace_file = tmp_path / "trace_events.jsonl"
    records = [
        {"type": "trace_metadata", "scaffold": "openclaw", "trace_format_version": 5},
        {
            "type": "event", "agent_id": "task-1", "category": "SCHEDULING",
            "event": "message_dispatch", "iteration": 0, "ts": 1000.0,
            "data": {"channel": "cli"},
        },
        {
            "type": "event", "agent_id": "task-1", "category": "TOOL",
            "event": "tool_exec_start", "iteration": 0, "ts": 1001.0,
            "data": {"tool_name": "bash"},
        },
    ]
    with open(trace_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return trace_file


def test_cmd_events(trace_with_events: Path, capsys: pytest.CaptureFixture) -> None:
    data = TraceData.load(trace_with_events)
    cmd_events(data)
    out = capsys.readouterr().out
    assert "message_dispatch" in out
    assert "tool_exec_start" in out


def test_cmd_events_category_filter(
    trace_with_events: Path, capsys: pytest.CaptureFixture
) -> None:
    data = TraceData.load(trace_with_events)
    cmd_events(data, category="TOOL")
    out = capsys.readouterr().out
    assert "tool_exec_start" in out
    assert "message_dispatch" not in out


def test_cmd_events_json(trace_with_events: Path, capsys: pytest.CaptureFixture) -> None:
    data = TraceData.load(trace_with_events)
    cmd_events(data, as_json=True)
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert len(parsed) == 2
