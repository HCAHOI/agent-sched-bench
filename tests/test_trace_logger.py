from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from agents.base import AgentBase, StepRecord
from harness.trace_logger import TraceLogger, build_run_id


def test_build_run_id_uses_expected_shape() -> None:
    run_id = build_run_id(
        system="vllm",
        workload="code",
        concurrency=4,
        started_at=datetime(2026, 3, 31, 12, 0, 0, tzinfo=timezone.utc),
    )
    assert run_id == "vllm_code_4_20260331T120000000Z"


def test_build_run_id_normalizes_non_utc_datetime() -> None:
    run_id = build_run_id(
        system="vllm",
        workload="code",
        concurrency=4,
        started_at=datetime(2026, 3, 31, 20, 0, 0, tzinfo=timezone(timedelta(hours=8))),
    )
    assert run_id == "vllm_code_4_20260331T120000000Z"


def test_trace_logger_writes_jsonl_entries(tmp_path: Path) -> None:
    run_id = "demo_run"
    logger = TraceLogger(tmp_path, run_id)
    record = StepRecord(
        step_idx=0,
        phase="reasoning",
        program_id="agent-0001",
        prompt_tokens=10,
        completion_tokens=3,
        llm_latency_ms=11.0,
        llm_output="hello",
        raw_response={"id": "resp-1"},
        ts_start=1.0,
        ts_end=2.0,
    )
    logger.log_step("agent-0001", record)
    logger.log_summary("agent-0001", {"task_id": "t-1", "success": True})
    logger.close()

    lines = (tmp_path / f"{run_id}.jsonl").read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["type"] == "step"
    assert parsed[0]["program_id"] == "agent-0001"
    assert parsed[1]["type"] == "summary"
    assert parsed[1]["success"] is True


def test_trace_logger_appends_on_resume(tmp_path: Path) -> None:
    # Append mode supports resume: opening the same run_id twice should
    # append records without raising or overwriting.
    run_id = "resume"
    logger1 = TraceLogger(tmp_path, run_id)
    logger1.log_event("agent-1", "llm_start", {"step_idx": 0, "ts": 1.0})
    logger1.close()

    logger2 = TraceLogger(tmp_path, run_id)
    logger2.log_event("agent-1", "llm_end", {"step_idx": 0, "ts": 2.0})
    logger2.close()

    lines = (tmp_path / f"{run_id}.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["type"] == "llm_start"
    assert json.loads(lines[1])["type"] == "llm_end"


def test_trace_logger_log_event_writes_correct_record(tmp_path: Path) -> None:
    logger = TraceLogger(tmp_path, "evt_run")
    logger.log_event("agent-42", "llm_start", {"step_idx": 0, "ts": 1.0})
    logger.close()

    lines = (tmp_path / "evt_run.jsonl").read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[0])
    assert entry["type"] == "llm_start"
    assert entry["agent_id"] == "agent-42"
    assert entry["step_idx"] == 0
    assert entry["ts"] == 1.0


def _make_record(step_idx: int = 0) -> StepRecord:
    return StepRecord(
        step_idx=step_idx,
        phase="acting",
        program_id="agent-1",
        prompt_tokens=5,
        completion_tokens=2,
        llm_latency_ms=100.0,
        ts_start=1.0,
        ts_end=2.0,
        tool_name="bash",
        tool_duration_ms=300.0,
        tool_timeout=False,
    )


def _make_concrete_agent(api_base: str = "http://localhost") -> AgentBase:
    """Return a minimal concrete AgentBase subclass for testing."""

    class _ConcreteAgent(AgentBase):
        async def run(self, task):  # type: ignore[override]
            return False

    return _ConcreteAgent(agent_id="agent-1", api_base=api_base, model="test-model")


def test_emit_event_noop_when_no_logger() -> None:
    agent = _make_concrete_agent()
    # Must not raise even without a logger injected
    agent._emit_event("llm_start", {"step_idx": 0, "ts": 1.0})
    assert len(agent.trace) == 0


def test_emit_step_appends_and_merges_metadata(tmp_path: Path) -> None:
    logger = TraceLogger(tmp_path, "step_run")
    agent = _make_concrete_agent()
    agent._trace_logger = logger
    agent.run_metadata = {"model": "qwen-plus", "prepare_ms": 500.0}

    record = _make_record(step_idx=0)
    agent._emit_step(record)
    logger.close()

    # Record is appended to trace
    assert len(agent.trace) == 1
    assert agent.trace[0].step_idx == 0

    # run_metadata merged into extra
    assert agent.trace[0].extra["model"] == "qwen-plus"
    assert agent.trace[0].extra["prepare_ms"] == 500.0

    # Written to JSONL
    lines = (tmp_path / "step_run.jsonl").read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[0])
    assert entry["type"] == "step"
    assert entry["extra"]["model"] == "qwen-plus"


def test_emit_step_noop_on_logger_when_none() -> None:
    agent = _make_concrete_agent()
    record = _make_record()
    agent._emit_step(record)
    # trace still gets the record; no logger means no file writes
    assert len(agent.trace) == 1


def test_summary_tool_stats() -> None:
    agent = _make_concrete_agent()
    r1 = _make_record(step_idx=0)
    r1.tool_name = "bash"
    r1.tool_duration_ms = 200.0
    r1.tool_timeout = False

    r2 = _make_record(step_idx=1)
    r2.tool_name = "bash"
    r2.tool_duration_ms = 100.0
    r2.tool_timeout = True

    r3 = _make_record(step_idx=2)
    r3.tool_name = "submit"
    r3.tool_duration_ms = 50.0
    r3.tool_timeout = False

    for r in (r1, r2, r3):
        agent.trace.append(r)

    s = agent.summary()
    assert s["tool_ms_by_name"]["bash"] == 300.0
    assert s["tool_ms_by_name"]["submit"] == 50.0
    assert s["tool_timeouts"]["bash"] == 1
    assert "submit" not in s["tool_timeouts"]
