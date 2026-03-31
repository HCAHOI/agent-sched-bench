from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agents.base import StepRecord
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


def test_trace_logger_rejects_duplicate_run_file(tmp_path: Path) -> None:
    run_id = "duplicate"
    logger = TraceLogger(tmp_path, run_id)
    logger.close()
    try:
        TraceLogger(tmp_path, run_id)
    except FileExistsError:
        return
    raise AssertionError("expected duplicate trace logger creation to fail")
