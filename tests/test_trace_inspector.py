"""Tests for trace_collect.trace_inspector."""

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


@pytest.fixture
def sample_trace(tmp_path: Path) -> Path:
    """Create a minimal v3-format JSONL trace for testing.

    v3 traces store messages_in/raw_response in llm_call_start/end events,
    not in step records.
    """
    trace_file = tmp_path / "test_trace.jsonl"
    records = [
        {
            "type": "trace_metadata",
            "scaffold": "mini-swe-agent",
            "trace_format_version": 3,
            "mode": "collect",
            "model": "test-model",
        },
        # Step 0: LLM events
        {
            "type": "llm_call_start",
            "agent_id": "django__django-11734",
            "step_idx": 0,
            "ts": 1000.0,
            "messages_in": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Fix the bug"},
            ],
        },
        {
            "type": "llm_call_end",
            "agent_id": "django__django-11734",
            "step_idx": 0,
            "ts": 1001.5,
            "raw_response": {
                "choices": [{"message": {"content": "Let me search", "tool_calls": []}}]
            },
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "llm_latency_ms": 1500,
        },
        # Step 0: tool events
        {
            "type": "tool_start",
            "agent_id": "django__django-11734",
            "step_idx": 0,
            "ts": 1001.5,
            "tool_name": "bash",
            "tool_args": '{"command": "find . -name models.py"}',
        },
        {
            "type": "tool_end",
            "agent_id": "django__django-11734",
            "step_idx": 0,
            "ts": 1001.6,
            "tool_name": "bash",
            "duration_ms": 50,
            "success": True,
            "timeout": False,
        },
        # Step 0: slim step record (no messages_in, raw_response, llm_output)
        {
            "type": "step",
            "agent_id": "django__django-11734",
            "step_idx": 0,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "llm_latency_ms": 1500,
            "ttft_ms": 200,
            "tpot_ms": 26.5,
            "tool_name": "bash",
            "tool_args": '{"command": "find . -name models.py"}',
            "tool_result": "./models.py",
            "tool_duration_ms": 50,
            "tool_success": True,
            "ts_start": 1000.0,
            "ts_end": 1001.5,
            "tool_ts_start": 1001.5,
            "tool_ts_end": 1001.55,
            "phase": "acting",
        },
        # Step 1: LLM events
        {
            "type": "llm_call_start",
            "agent_id": "django__django-11734",
            "step_idx": 1,
            "ts": 1002.0,
            "messages_in": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Fix the bug"},
                {"role": "assistant", "content": "Let me search"},
            ],
        },
        {
            "type": "llm_call_end",
            "agent_id": "django__django-11734",
            "step_idx": 1,
            "ts": 1004.0,
            "raw_response": {
                "choices": [{"message": {"content": "Reading file", "tool_calls": []}}]
            },
            "prompt_tokens": 200,
            "completion_tokens": 80,
            "llm_latency_ms": 2000,
        },
        # Step 1: slim step
        {
            "type": "step",
            "agent_id": "django__django-11734",
            "step_idx": 1,
            "prompt_tokens": 200,
            "completion_tokens": 80,
            "llm_latency_ms": 2000,
            "ttft_ms": 300,
            "tpot_ms": 21.5,
            "tool_name": "bash",
            "tool_args": '{"command": "cat models.py"}',
            "tool_result": "class Model:\n    pass",
            "tool_duration_ms": 30,
            "tool_success": True,
            "ts_start": 1002.0,
            "ts_end": 1004.0,
            "tool_ts_start": 1004.0,
            "tool_ts_end": 1004.03,
            "phase": "acting",
        },
        {
            "type": "summary",
            "agent_id": "django__django-11734",
            "n_steps": 2,
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


# ---------------------------------------------------------------------------
# TraceData.load tests
# ---------------------------------------------------------------------------


def test_load_trace(sample_trace: Path) -> None:
    data = TraceData.load(sample_trace)
    assert len(data.steps) == 2
    # 2x llm_call_start + 2x llm_call_end + 2x tool_start + 2x tool_end = 8
    assert len(data.events) == 6
    assert len(data.summaries) == 1
    assert data.metadata["scaffold"] == "mini-swe-agent"
    assert data.metadata["model"] == "test-model"
    # Steps sorted by step_idx
    assert data.steps[0]["step_idx"] == 0
    assert data.steps[1]["step_idx"] == 1
    # Events sorted by ts
    assert data.events[0]["ts"] <= data.events[1]["ts"]


def test_load_with_agent_filter(sample_trace: Path) -> None:
    # Filter by partial agent_id
    data = TraceData.load(sample_trace, agent_filter="django__django")
    assert len(data.steps) == 2
    assert len(data.agents) == 1
    assert "django__django-11734" in data.agents

    # Filter that matches nothing
    data_empty = TraceData.load(sample_trace, agent_filter="nonexistent_agent")
    assert len(data_empty.steps) == 0
    assert len(data_empty.events) == 0


def test_load_skips_malformed_lines(tmp_path: Path) -> None:
    trace_file = tmp_path / "malformed.jsonl"
    trace_file.write_text(
        '{"type": "trace_metadata", "scaffold": "mini-swe-agent"}\n'
        "NOT VALID JSON {{{\n"
        '{"type": "step", "agent_id": "a1", "step_idx": 0}\n'
    )
    data = TraceData.load(trace_file)
    assert len(data.steps) == 1


# ---------------------------------------------------------------------------
# cmd_overview tests
# ---------------------------------------------------------------------------


def test_cmd_overview(sample_trace: Path, capsys: pytest.CaptureFixture) -> None:
    data = TraceData.load(sample_trace)
    cmd_overview(data)
    out = capsys.readouterr().out
    assert "mini-swe-agent" in out
    assert "test-model" in out
    assert "2" in out  # n_steps
    assert "bash" in out  # tool name


def test_cmd_overview_json(sample_trace: Path, capsys: pytest.CaptureFixture) -> None:
    data = TraceData.load(sample_trace)
    cmd_overview(data, as_json=True)
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["n_steps"] == 2
    assert parsed["scaffold"] == "mini-swe-agent"
    assert parsed["model"] == "test-model"
    assert parsed["tool_counts"]["bash"] == 2
    assert parsed["total_tokens"] == 430  # 100+50 + 200+80


# ---------------------------------------------------------------------------
# cmd_step tests
# ---------------------------------------------------------------------------


def test_cmd_step_valid(sample_trace: Path, capsys: pytest.CaptureFixture) -> None:
    data = TraceData.load(sample_trace)
    cmd_step(data, 0)
    out = capsys.readouterr().out
    assert "Step 0" in out
    assert "bash" in out
    assert "1500" in out  # llm_latency_ms


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
    assert parsed["step_idx"] == 0
    assert parsed["tool_name"] == "bash"


# ---------------------------------------------------------------------------
# cmd_messages tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# cmd_response tests
# ---------------------------------------------------------------------------


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
    assert parsed["step_idx"] == 0
    assert "raw_response" in parsed


# ---------------------------------------------------------------------------
# cmd_events tests
# ---------------------------------------------------------------------------


def test_cmd_events(sample_trace: Path, capsys: pytest.CaptureFixture) -> None:
    data = TraceData.load(sample_trace)
    cmd_events(data)
    out = capsys.readouterr().out
    assert "llm_call_start" in out
    assert "tool_exec_start" in out


def test_cmd_events_category_filter(
    sample_trace: Path, capsys: pytest.CaptureFixture
) -> None:
    data = TraceData.load(sample_trace)
    cmd_events(data, category="TOOL")
    out = capsys.readouterr().out
    assert "tool_exec_start" in out
    assert "llm_call_start" not in out


def test_cmd_events_iteration_filter(
    sample_trace: Path, capsys: pytest.CaptureFixture
) -> None:
    data = TraceData.load(sample_trace)
    cmd_events(data, iteration=0)
    out = capsys.readouterr().out
    assert "llm_call_start" in out
    # Step 1 events should be excluded
    assert out.count("llm_call_start") == 1


def test_cmd_events_json(sample_trace: Path, capsys: pytest.CaptureFixture) -> None:
    data = TraceData.load(sample_trace)
    cmd_events(data, as_json=True)
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert len(parsed) == 6


# ---------------------------------------------------------------------------
# cmd_tools tests
# ---------------------------------------------------------------------------


def test_cmd_tools(sample_trace: Path, capsys: pytest.CaptureFixture) -> None:
    data = TraceData.load(sample_trace)
    cmd_tools(data)
    out = capsys.readouterr().out
    assert "bash" in out
    # 2 calls total
    assert "2" in out


def test_cmd_tools_json(sample_trace: Path, capsys: pytest.CaptureFixture) -> None:
    data = TraceData.load(sample_trace)
    cmd_tools(data, as_json=True)
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert isinstance(parsed, list)
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


# ---------------------------------------------------------------------------
# cmd_search tests
# ---------------------------------------------------------------------------


def test_cmd_search_match(sample_trace: Path, capsys: pytest.CaptureFixture) -> None:
    data = TraceData.load(sample_trace)
    # "Let me search" is in step 0's raw_response -> enriched llm_output
    cmd_search(data, "search")
    out = capsys.readouterr().out
    assert "step 0" in out
    assert "search" in out.lower()


def test_cmd_search_no_match(sample_trace: Path, capsys: pytest.CaptureFixture) -> None:
    data = TraceData.load(sample_trace)
    cmd_search(data, "zzz_no_match_xyz")
    out = capsys.readouterr().out
    assert "No matches" in out


def test_cmd_search_json(sample_trace: Path, capsys: pytest.CaptureFixture) -> None:
    data = TraceData.load(sample_trace)
    # "Reading file" is in step 1's raw_response -> enriched llm_output
    cmd_search(data, "Reading", as_json=True)
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert len(parsed) == 1
    assert parsed[0]["step_idx"] == 1


# ---------------------------------------------------------------------------
# Legacy event normalization tests
# ---------------------------------------------------------------------------


@pytest.fixture
def miniswe_trace(tmp_path: Path) -> Path:
    """Trace with legacy mini-swe flat events."""
    trace_file = tmp_path / "miniswe.jsonl"
    records = [
        {"type": "trace_metadata", "scaffold": "mini-swe-agent", "model": "test"},
        {
            "type": "llm_start",
            "agent_id": "task-1",
            "step_idx": 0,
            "ts": 1000.0,
        },
        {
            "type": "llm_end",
            "agent_id": "task-1",
            "step_idx": 0,
            "ts": 1001.5,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "latency_ms": 1500,
        },
        {
            "type": "tool_start",
            "agent_id": "task-1",
            "step_idx": 0,
            "ts": 1001.5,
            "tool_name": "bash",
            "tool_args": '{"command": "ls"}',
        },
        {
            "type": "tool_end",
            "agent_id": "task-1",
            "step_idx": 0,
            "ts": 1001.6,
            "tool_name": "bash",
            "duration_ms": 100,
            "success": True,
            "timeout": False,
        },
        {
            "type": "action",
            "agent_id": "task-1",
            "step_idx": 0,
            "ts": 1001.0,
        },
        {"type": "summary", "agent_id": "task-1", "n_steps": 1, "success": True},
    ]
    with open(trace_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return trace_file


def test_legacy_events_normalized(miniswe_trace: Path) -> None:
    """Legacy mini-swe flat events are normalized into unified envelope."""
    data = TraceData.load(miniswe_trace)
    assert len(data.events) == 5  # llm_start, llm_end, tool_start, tool_end, action

    # All events should have category and event fields
    for ev in data.events:
        assert "category" in ev, f"Missing category in {ev}"
        assert "event" in ev, f"Missing event in {ev}"
        assert ev["type"] == "event"

    # Check specific mappings
    by_event = {ev["event"]: ev for ev in data.events}
    assert "llm_call_start" in by_event
    assert "llm_call_end" in by_event
    assert "tool_exec_start" in by_event
    assert "tool_exec_end" in by_event
    assert "llm_action" in by_event

    assert by_event["llm_call_start"]["category"] == "LLM"
    assert by_event["llm_call_end"]["category"] == "LLM"
    assert by_event["tool_exec_start"]["category"] == "TOOL"
    assert by_event["tool_exec_end"]["category"] == "TOOL"
    assert by_event["llm_action"]["category"] == "LLM"


def test_legacy_events_category_filter(
    miniswe_trace: Path, capsys: pytest.CaptureFixture
) -> None:
    """Category filter works on normalized legacy events."""
    data = TraceData.load(miniswe_trace)
    cmd_events(data, category="LLM")
    out = capsys.readouterr().out
    assert "llm_call_start" in out
    assert "llm_call_end" in out
    assert "llm_action" in out
    assert "tool_exec_start" not in out


def test_legacy_events_step_idx_preserved(miniswe_trace: Path) -> None:
    """step_idx is preserved through normalization."""
    data = TraceData.load(miniswe_trace)
    for ev in data.events:
        assert ev["step_idx"] == 0


def test_legacy_events_data_fields(miniswe_trace: Path) -> None:
    """Event-specific payload fields are collected in data dict."""
    data = TraceData.load(miniswe_trace)
    by_event = {ev["event"]: ev for ev in data.events}

    llm_end = by_event["llm_call_end"]
    assert llm_end["data"]["prompt_tokens"] == 100
    assert llm_end["data"]["completion_tokens"] == 50
    assert llm_end["data"]["latency_ms"] == 1500

    tool_start = by_event["tool_exec_start"]
    assert tool_start["data"]["tool_name"] == "bash"


def test_openclaw_iteration_compat(tmp_path: Path) -> None:
    """Openclaw events with 'iteration' field are normalized to 'step_idx'."""
    trace_file = tmp_path / "openclaw.jsonl"
    records = [
        {"type": "trace_metadata", "scaffold": "openclaw", "model": "test"},
        {
            "type": "event",
            "agent_id": "task-1",
            "event": "tool_exec_start",
            "category": "TOOL",
            "data": {"tool_name": "bash"},
            "iteration": 3,
            "ts": 1000.0,
        },
    ]
    with open(trace_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    data = TraceData.load(trace_file)
    assert len(data.events) == 1
    assert data.events[0]["step_idx"] == 3


def test_v3_step_enrichment(tmp_path: Path) -> None:
    """v3 slim steps are enriched with messages_in/raw_response from events."""
    trace_file = tmp_path / "v3.jsonl"
    records = [
        {"type": "trace_metadata", "scaffold": "openclaw", "trace_format_version": 3},
        {
            "type": "event",
            "agent_id": "t1",
            "event": "llm_call_start",
            "category": "LLM",
            "data": {"messages_in": [{"role": "user", "content": "hello"}]},
            "step_idx": 0,
            "ts": 100.0,
        },
        {
            "type": "event",
            "agent_id": "t1",
            "event": "llm_call_end",
            "category": "LLM",
            "data": {
                "raw_response": {"choices": [{"message": {"content": "hi"}}]},
                "prompt_tokens": 10,
                "completion_tokens": 5,
            },
            "step_idx": 0,
            "ts": 101.0,
        },
        {
            "type": "step",
            "agent_id": "t1",
            "step_idx": 0,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "ts_start": 100.0,
            "ts_end": 101.0,
        },
    ]
    with open(trace_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    data = TraceData.load(trace_file)
    step = data.steps[0]
    # Step should be enriched from events
    assert step["messages_in"] == [{"role": "user", "content": "hello"}]
    assert step["raw_response"]["choices"][0]["message"]["content"] == "hi"
