from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.extract_tool_gaps import build_parser, discover_trace_files
from trace_collect.tool_gap_extractor import extract_tool_gap_windows


def test_extract_tool_gap_windows_uses_next_llm_gap_and_groups_by_agent(tmp_path: Path) -> None:
    trace_path = _write_canonical_trace(tmp_path / "trace.jsonl")

    windows = extract_tool_gap_windows(trace_path)

    assert len(windows) == 1
    window = windows[0]
    assert window.agent_id == "agent-a"
    assert window.instance_id == "task-123"
    assert window.iteration == 0
    assert window.llm_action_id == "agent-a-llm-0"
    assert window.next_llm_action_id == "agent-a-llm-1"
    assert window.available_gap_ms == 2250.0


def test_extract_tool_gap_windows_ignores_interleaved_reset_iteration_llm_same_agent(
    tmp_path: Path,
) -> None:
    trace_path = _write_interleaved_same_agent_trace(tmp_path / "trace.jsonl")

    windows = extract_tool_gap_windows(trace_path)

    assert len(windows) == 1
    window = windows[0]
    assert window.llm_action_id == "agent-a-main-llm-0"
    assert window.next_llm_action_id == "agent-a-main-llm-1"
    assert window.tool_call_ids == ("main-call-read", "main-call-bash")
    assert window.available_gap_ms == 2250.0
    assert set(window.to_json_obj()) == {
        "sample_id",
        "source_trace",
        "agent_id",
        "instance_id",
        "iteration",
        "llm_action_id",
        "next_llm_action_id",
        "available_gap_ms",
        "tool_count",
        "tool_names",
        "tool_call_ids",
    }


def test_extract_tool_gap_windows_captures_same_iteration_batch_without_duration_feature(tmp_path: Path) -> None:
    trace_path = _write_canonical_trace(tmp_path / "trace.jsonl")

    [window] = extract_tool_gap_windows(trace_path)
    row = window.to_json_obj()

    assert row["tool_count"] == 2
    assert row["tool_names"] == ["read", "bash"]
    assert row["tool_call_ids"] == ["call-read", "call-bash"]
    leaked_timing_features = [
        key for key in row if "duration" in key or "makespan" in key
    ]
    assert leaked_timing_features == []


def test_extract_tool_gap_windows_rejects_tool_batch_crossing_next_same_agent_llm(
    tmp_path: Path,
) -> None:
    trace_path = _write_canonical_trace(tmp_path / "trace.jsonl")
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    for record in records:
        if record.get("action_id") == "agent-a-tool-bash":
            record["ts_end"] = 103.3
            break
    trace_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="tool batch crosses next LLM start"):
        extract_tool_gap_windows(trace_path)


def test_extract_tool_gaps_script_parser_discovers_trace_jsonl_under_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run-1"
    trace_path = _write_canonical_trace(run_dir / "trace.jsonl")
    ignored_path = tmp_path / "runs" / "not_a_trace.jsonl"
    ignored_path.write_text("", encoding="utf-8")
    output_path = tmp_path / "tool_gaps.jsonl"
    parser = build_parser()

    args = parser.parse_args([str(tmp_path / "runs"), "--output", str(output_path)])
    discovered = discover_trace_files(args.paths)

    assert args.output == output_path
    assert discovered == [trace_path.resolve()]


def _write_interleaved_same_agent_trace(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "type": "trace_metadata",
            "trace_format_version": 5,
            "scaffold": "openclaw",
            "instance_id": "browsecomp-task-7",
            "model": "fixture-model",
        },
        {
            "type": "action",
            "action_type": "llm_call",
            "action_id": "agent-a-main-llm-0",
            "agent_id": "agent-a",
            "iteration": 0,
            "ts_start": 100.0,
            "ts_end": 101.0,
            "data": {"llm_latency_ms": 1000.0},
        },
        {
            "type": "action",
            "action_type": "tool_exec",
            "action_id": "agent-a-main-tool-read",
            "agent_id": "agent-a",
            "iteration": 0,
            "ts_start": 101.1,
            "ts_end": 101.4,
            "data": {
                "tool_name": "read",
                "tool_call_id": "main-call-read",
                "duration_ms": 300.0,
            },
        },
        {
            "type": "action",
            "action_type": "llm_call",
            "action_id": "agent-a-side-llm-0",
            "agent_id": "agent-a",
            "iteration": 0,
            "ts_start": 101.15,
            "ts_end": 101.6,
            "data": {"llm_latency_ms": 450.0},
        },
        {
            "type": "action",
            "action_type": "tool_exec",
            "action_id": "agent-a-main-tool-bash",
            "agent_id": "agent-a",
            "iteration": 0,
            "ts_start": 101.2,
            "ts_end": 102.5,
            "data": {
                "tool_name": "bash",
                "tool_call_id": "main-call-bash",
                "duration_ms": 1300.0,
            },
        },
        {
            "type": "action",
            "action_type": "llm_call",
            "action_id": "agent-a-main-llm-1",
            "agent_id": "agent-a",
            "iteration": 1,
            "ts_start": 103.25,
            "ts_end": 104.0,
            "data": {"llm_latency_ms": 750.0},
        },
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _write_canonical_trace(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "type": "trace_metadata",
            "trace_format_version": 5,
            "scaffold": "openclaw",
            "instance_id": "task-123",
            "model": "fixture-model",
        },
        {
            "type": "action",
            "action_type": "llm_call",
            "action_id": "agent-a-llm-0",
            "agent_id": "agent-a",
            "iteration": 0,
            "ts_start": 100.0,
            "ts_end": 101.0,
            "data": {"llm_latency_ms": 1000.0},
        },
        {
            "type": "action",
            "action_type": "tool_exec",
            "action_id": "agent-a-tool-read",
            "agent_id": "agent-a",
            "iteration": 0,
            "ts_start": 101.1,
            "ts_end": 101.4,
            "data": {
                "tool_name": "read",
                "tool_call_id": "call-read",
                "duration_ms": 999999.0,
            },
        },
        {
            "type": "action",
            "action_type": "tool_exec",
            "action_id": "agent-a-tool-bash",
            "agent_id": "agent-a",
            "iteration": 0,
            "ts_start": 101.2,
            "ts_end": 102.5,
            "data": {
                "tool_name": "bash",
                "tool_call_id": "call-bash",
                "duration_ms": 999999.0,
            },
        },
        {
            "type": "action",
            "action_type": "llm_call",
            "action_id": "agent-b-llm-0",
            "agent_id": "agent-b",
            "iteration": 0,
            "ts_start": 101.8,
            "ts_end": 102.0,
            "data": {"llm_latency_ms": 200.0},
        },
        {
            "type": "action",
            "action_type": "tool_exec",
            "action_id": "agent-b-final-tool",
            "agent_id": "agent-b",
            "iteration": 0,
            "ts_start": 102.1,
            "ts_end": 102.2,
            "data": {"tool_name": "search", "tool_call_id": "call-search"},
        },
        {
            "type": "action",
            "action_type": "llm_call",
            "action_id": "agent-a-llm-1",
            "agent_id": "agent-a",
            "iteration": 1,
            "ts_start": 103.25,
            "ts_end": 104.0,
            "data": {"llm_latency_ms": 750.0},
        },
        {
            "type": "action",
            "action_type": "tool_exec",
            "action_id": "agent-a-final-tool",
            "agent_id": "agent-a",
            "iteration": 1,
            "ts_start": 104.1,
            "ts_end": 104.9,
            "data": {"tool_name": "edit", "tool_call_id": "call-edit"},
        },
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path
