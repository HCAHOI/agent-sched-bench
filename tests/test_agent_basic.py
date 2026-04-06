from __future__ import annotations

import asyncio
import json
from typing import Any

from agents.base import AgentBase, LLMCallResult, StepRecord


class DummyAgent(AgentBase):
    async def run(self, task: dict[str, Any]) -> bool:  # pragma: no cover - trivial
        self.task_id = task["task_id"]
        self.task_success = True
        return True


def test_step_record_export_preserves_program_id_and_llm_fields() -> None:
    agent = DummyAgent(
        agent_id="agent-0001", api_base="http://localhost:8000/v1", model="mock"
    )
    record = StepRecord(
        step_idx=1,
        phase="reasoning",
        program_id="agent-0001",
        prompt_tokens=10,
        completion_tokens=5,
        llm_latency_ms=12.5,
        llm_output="analysis text",
        raw_response={"id": "resp-1"},
        extra={"reasoning_depth": "medium"},
    )
    agent.trace.append(record)
    exported = agent.get_trace()
    assert exported[0]["program_id"] == "agent-0001"
    assert exported[0]["llm_output"] == "analysis text"
    assert exported[0]["raw_response"]["id"] == "resp-1"
    json.dumps(exported)


def test_agent_summary_aggregates_trace_fields() -> None:
    agent = DummyAgent(
        agent_id="agent-0002", api_base="http://localhost:8000/v1", model="mock"
    )
    agent.task_id = "task-1"
    agent.task_success = True
    agent.trace = [
        StepRecord(
            step_idx=0,
            phase="reasoning",
            program_id="agent-0002",
            prompt_tokens=12,
            completion_tokens=3,
            llm_latency_ms=10.0,
        ),
        StepRecord(
            step_idx=1,
            phase="acting",
            program_id="agent-0002",
            prompt_tokens=0,
            completion_tokens=0,
            llm_latency_ms=0.0,
            tool_name="bash",
            tool_duration_ms=25.0,
            tool_success=True,
        ),
    ]
    summary = agent.summary()
    assert summary["program_id"] == "agent-0002"
    assert summary["n_steps"] == 2
    assert summary["total_tokens"] == 15
    assert summary["total_tool_ms"] == 25.0


def test_build_step_record_uses_llm_result_fields() -> None:
    agent = DummyAgent(
        agent_id="agent-0003", api_base="http://localhost:8000/v1", model="mock"
    )
    llm_result = LLMCallResult(
        content="ok",
        prompt_tokens=7,
        completion_tokens=2,
        llm_latency_ms=9.5,
        raw_response={"id": "resp-1"},
    )
    record = agent.build_step_record(
        step_idx=0,
        phase="reasoning",
        llm_result=llm_result,
        ts_start=1.0,
        ts_end=2.0,
        extra={"raw_id": "resp-1"},
    )
    assert record.program_id == "agent-0003"
    assert record.prompt_tokens == 7
    assert record.completion_tokens == 2
    assert record.llm_output == "ok"
    assert record.raw_response["id"] == "resp-1"
    assert record.extra["raw_id"] == "resp-1"


class FakeUsage:
    prompt_tokens = 11
    completion_tokens = 4


class FakeMessage:
    content = "synthetic reply"
    tool_calls = None


class FakeChoice:
    message = FakeMessage()


class FakeResponse:
    choices = [FakeChoice()]
    usage = FakeUsage()

    def model_dump(self) -> dict[str, Any]:
        return {"id": "resp-2"}


class RecordingCompletions:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> FakeResponse:
        self.kwargs = kwargs
        return FakeResponse()


class RecordingChat:
    def __init__(self) -> None:
        self.completions = RecordingCompletions()


class RecordingClient:
    def __init__(self) -> None:
        self.chat = RecordingChat()


def test_call_llm_attaches_program_id_and_returns_normalized_fields() -> None:
    agent = DummyAgent(
        agent_id="agent-0004", api_base="http://localhost:8000/v1", model="mock"
    )
    recording_client = RecordingClient()
    agent._client = recording_client  # type: ignore[assignment]
    result = asyncio.run(agent._call_llm([{"role": "user", "content": "hi"}]))
    assert recording_client.chat.completions.kwargs is not None
    assert (
        recording_client.chat.completions.kwargs["extra_body"]["program_id"]
        == "agent-0004"
    )
    assert result.content == "synthetic reply"
    assert result.prompt_tokens == 11
    assert result.raw_response["id"] == "resp-2"
