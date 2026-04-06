"""Tests for the simulate module (trace_collect.simulator)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from agents.base import ActionRecord, StepRecord


class TestActionRecord:
    def test_asdict_round_trip(self) -> None:
        record = ActionRecord(
            step_idx=0,
            program_id="test-001",
            tool_name="bash",
            tool_args='{"command": "ls"}',
            prompt_tokens=100,
            completion_tokens=20,
            llm_latency_ms=500.0,
            ttft_ms=50.0,
            ts=1000.0,
        )
        d = asdict(record)
        assert d["step_idx"] == 0
        assert d["tool_name"] == "bash"
        assert d["ttft_ms"] == 50.0
        assert d["ts"] == 1000.0
        # Serializable
        assert json.loads(json.dumps(d)) == d

    def test_none_ttft_for_non_simulate(self) -> None:
        record = ActionRecord(
            step_idx=0,
            program_id="test-001",
            tool_name="bash",
            tool_args="{}",
            prompt_tokens=100,
            completion_tokens=20,
            llm_latency_ms=500.0,
        )
        assert record.ttft_ms is None


class TestStepRecordNewFields:
    def test_ttft_tpot_default_none(self) -> None:
        record = StepRecord(
            step_idx=0,
            phase="acting",
            program_id="test-001",
            prompt_tokens=100,
            completion_tokens=20,
            llm_latency_ms=500.0,
        )
        assert record.ttft_ms is None
        assert record.tpot_ms is None

    def test_ttft_tpot_set(self) -> None:
        record = StepRecord(
            step_idx=0,
            phase="acting",
            program_id="test-001",
            prompt_tokens=100,
            completion_tokens=20,
            llm_latency_ms=500.0,
            ttft_ms=50.0,
            tpot_ms=23.7,
        )
        d = asdict(record)
        assert d["ttft_ms"] == 50.0
        assert d["tpot_ms"] == 23.7


class TestDetectAgentId:
    def test_finds_first_step_agent_id(self, tmp_path: Path) -> None:
        from trace_collect.simulator import _detect_agent_id  # no minisweagent needed

        trace = tmp_path / "test.jsonl"
        records = [
            {"type": "action", "agent_id": "task-001", "step_idx": 0},
            {
                "type": "step",
                "agent_id": "task-001",
                "step_idx": 0,
                "tool_name": "bash",
            },
        ]
        trace.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        assert _detect_agent_id(trace) == "task-001"


class TestTraceLoggerAction:
    def test_log_action_writes_jsonl(self, tmp_path: Path) -> None:
        from harness.trace_logger import TraceLogger

        tl = TraceLogger(tmp_path, "test_run")
        action = ActionRecord(
            step_idx=0,
            program_id="test-001",
            tool_name="bash",
            tool_args='{"command": "echo hi"}',
            prompt_tokens=50,
            completion_tokens=10,
            llm_latency_ms=200.0,
            ts=1000.0,
        )
        tl.log_action("test-001", action)
        tl.close()

        lines = (
            (tmp_path / "test_run.jsonl")
            .read_text(encoding="utf-8")
            .strip()
            .splitlines()
        )
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["type"] == "action"
        assert record["agent_id"] == "test-001"
        assert record["tool_name"] == "bash"
        assert record["llm_latency_ms"] == 200.0


class TestEmitStepEmitsAction:
    """Verify that AgentBase._emit_step also emits an action record."""

    def test_emit_step_writes_action_and_step(self, tmp_path: Path) -> None:
        from harness.trace_logger import TraceLogger

        tl = TraceLogger(tmp_path, "test_run")

        # Create a minimal concrete agent
        from agents.base import AgentBase

        class DummyAgent(AgentBase):
            async def run(self, task: dict) -> bool:
                return True

        agent = DummyAgent(
            agent_id="test-001",
            api_base="http://localhost:8000/v1",
            model="test-model",
        )
        agent._trace_logger = tl

        record = StepRecord(
            step_idx=0,
            phase="acting",
            program_id="test-001",
            prompt_tokens=100,
            completion_tokens=20,
            llm_latency_ms=500.0,
            tool_name="bash",
            tool_args='{"command": "ls"}',
            tool_result="file1.py\n",
            ts_start=1000.0,
            ts_end=1001.0,
        )
        agent._emit_step(record)
        tl.close()

        lines = (
            (tmp_path / "test_run.jsonl")
            .read_text(encoding="utf-8")
            .strip()
            .splitlines()
        )
        assert len(lines) == 2
        action = json.loads(lines[0])
        step = json.loads(lines[1])

        assert action["type"] == "action"
        assert action["tool_name"] == "bash"
        assert action["ts"] == 1000.0 + 500.0 / 1000  # ts_start + llm_latency_ms/1000

        assert step["type"] == "step"
        assert step["tool_result"] == "file1.py\n"
