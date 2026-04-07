"""Tests for the simulate module (trace_collect.simulator)."""

from __future__ import annotations

import json
from pathlib import Path

from agents.base import TraceAction


class TestTraceAction:
    def test_to_dict_round_trip(self) -> None:
        action = TraceAction(
            action_type="tool_exec",
            action_id="tool_0_bash",
            agent_id="test-001",
            program_id="test-001",
            iteration=0,
            ts_start=1000.0,
            ts_end=1001.0,
            data={
                "tool_name": "bash",
                "tool_args": '{"command": "ls"}',
                "duration_ms": 500.0,
            },
        )
        d = action.to_dict()
        assert d["action_type"] == "tool_exec"
        assert d["action_id"] == "tool_0_bash"
        assert d["iteration"] == 0
        assert d["data"]["tool_name"] == "bash"
        # Serializable
        assert json.loads(json.dumps(d)) == d

    def test_llm_call_action(self) -> None:
        action = TraceAction(
            action_type="llm_call",
            action_id="llm_0",
            iteration=0,
            ts_start=100.0,
            ts_end=102.5,
            data={
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "llm_latency_ms": 2500.0,
            },
        )
        d = action.to_dict()
        assert d["type"] == "action"
        assert d["action_type"] == "llm_call"
        assert d["data"]["llm_latency_ms"] == 2500.0


class TestDetectAgentId:
    def test_finds_first_action_agent_id(self, tmp_path: Path) -> None:
        from trace_collect.simulator import _detect_agent_id

        trace = tmp_path / "test.jsonl"
        records = [
            {"type": "action", "action_type": "llm_call", "agent_id": "task-001",
             "action_id": "llm_0", "iteration": 0},
        ]
        trace.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        assert _detect_agent_id(trace) == "task-001"



class TestTraceLoggerAction:
    def test_log_trace_action_writes_jsonl(self, tmp_path: Path) -> None:
        from harness.trace_logger import TraceLogger

        tl = TraceLogger(tmp_path, "test_run")
        action = TraceAction(
            action_type="tool_exec",
            action_id="tool_0_bash",
            agent_id="test-001",
            program_id="test-001",
            iteration=0,
            ts_start=1000.0,
            ts_end=1001.0,
            data={
                "tool_name": "bash",
                "tool_args": '{"command": "echo hi"}',
                "duration_ms": 200.0,
            },
        )
        tl.log_trace_action("test-001", action)
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
        assert record["action_type"] == "tool_exec"
        assert record["data"]["tool_name"] == "bash"


class TestEmitAction:
    def test_emit_action_writes_to_logger(self, tmp_path: Path) -> None:
        from harness.trace_logger import TraceLogger
        from agents.base import AgentBase

        class DummyAgent(AgentBase):
            async def run(self, task: dict) -> bool:
                return True

        tl = TraceLogger(tmp_path, "test_run")
        agent = DummyAgent(
            agent_id="test-001",
            api_base="http://localhost:8000/v1",
            model="test-model",
        )
        agent._trace_logger = tl

        action = TraceAction(
            action_type="llm_call",
            action_id="llm_0",
            agent_id="test-001",
            iteration=0,
            ts_start=1000.0,
            ts_end=1001.0,
            data={"prompt_tokens": 100, "completion_tokens": 20, "llm_latency_ms": 500.0},
        )
        agent._emit_action(action)
        tl.close()

        assert len(agent.actions) == 1
        assert agent.actions[0].action_type == "llm_call"

        lines = (
            (tmp_path / "test_run.jsonl")
            .read_text(encoding="utf-8")
            .strip()
            .splitlines()
        )
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["type"] == "action"
        assert entry["action_type"] == "llm_call"


class TestLoadTraceActions:
    def test_loads_v4_actions_grouped_by_iteration(self, tmp_path: Path) -> None:
        from trace_collect.simulator import load_trace_actions

        trace = tmp_path / "test.jsonl"
        records = [
            {"type": "trace_metadata", "scaffold": "openclaw", "trace_format_version": 5},
            {"type": "action", "action_type": "llm_call", "action_id": "llm_0",
             "agent_id": "t1", "iteration": 0, "ts_start": 100, "ts_end": 102,
             "data": {"messages_in": [{"role": "user", "content": "hi"}],
                      "completion_tokens": 10}},
            {"type": "action", "action_type": "tool_exec", "action_id": "tool_0_bash",
             "agent_id": "t1", "iteration": 0, "ts_start": 102, "ts_end": 103,
             "data": {"tool_name": "bash", "tool_args": "{}"}},
            {"type": "summary", "agent_id": "t1", "n_steps": 1},
        ]
        trace.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

        iterations, summary = load_trace_actions(trace, "t1")
        assert 0 in iterations
        assert iterations[0]["llm"] is not None
        assert iterations[0]["llm"]["action_type"] == "llm_call"
        assert len(iterations[0]["tools"]) == 1
        assert iterations[0]["tools"][0]["data"]["tool_name"] == "bash"
        assert summary is not None
