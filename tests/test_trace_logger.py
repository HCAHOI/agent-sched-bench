from __future__ import annotations

import json
from pathlib import Path
from agents.base import TraceAction
from harness.trace_logger import TraceLogger


def test_trace_logger_writes_action_entries(tmp_path: Path) -> None:
    run_id = "demo_run"
    logger = TraceLogger(tmp_path, run_id)
    action = TraceAction(
        action_type="llm_call",
        action_id="llm_0",
        agent_id="agent-0001",
        program_id="agent-0001",
        iteration=0,
        ts_start=1.0,
        ts_end=2.0,
        data={"prompt_tokens": 10, "completion_tokens": 3, "llm_latency_ms": 11.0},
    )
    logger.log_trace_action("agent-0001", action)
    logger.log_summary("agent-0001", {"task_id": "t-1", "success": True})
    logger.close()

    lines = (tmp_path / f"{run_id}.jsonl").read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["type"] == "action"
    assert parsed[0]["action_type"] == "llm_call"
    assert parsed[0]["agent_id"] == "agent-0001"
    assert parsed[1]["type"] == "summary"
    assert parsed[1]["success"] is True


def test_trace_logger_appends_on_resume(tmp_path: Path) -> None:
    run_id = "resume"
    logger1 = TraceLogger(tmp_path, run_id)
    logger1.log_event("agent-1", "LLM", "llm_call_start", {}, iteration=0, ts=1.0)
    logger1.close()

    logger2 = TraceLogger(tmp_path, run_id)
    logger2.log_event("agent-1", "LLM", "llm_call_end", {}, iteration=0, ts=2.0)
    logger2.close()

    lines = (tmp_path / f"{run_id}.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    e0 = json.loads(lines[0])
    e1 = json.loads(lines[1])
    assert e0["type"] == "event"
    assert e0["event"] == "llm_call_start"
    assert e1["event"] == "llm_call_end"


def test_trace_logger_log_event_writes_correct_record(tmp_path: Path) -> None:
    logger = TraceLogger(tmp_path, "evt_run")
    logger.log_event("agent-42", "LLM", "llm_call_start", {"prompt_tokens": 10},
                     iteration=0, ts=1.0)
    logger.close()

    lines = (tmp_path / "evt_run.jsonl").read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[0])
    assert entry["type"] == "event"
    assert entry["event"] == "llm_call_start"
    assert entry["category"] == "LLM"
    assert entry["agent_id"] == "agent-42"
    assert entry["iteration"] == 0
    assert entry["ts"] == 1.0
    assert entry["data"]["prompt_tokens"] == 10


