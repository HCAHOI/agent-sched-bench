from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from openai import AsyncOpenAI


def _message_content_to_text(content: Any) -> str:
    """Normalize OpenAI response content into a single plain-text string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


@dataclass(slots=True)
class StepRecord:
    """Full record for one reasoning or acting step in an agent run."""

    step_idx: int
    phase: str
    program_id: str
    prompt_tokens: int
    completion_tokens: int
    llm_latency_ms: float
    llm_output: str = ""
    raw_response: dict[str, Any] = field(default_factory=dict)
    tool_name: str | None = None
    tool_args: str | None = None
    tool_result: str | None = None
    tool_duration_ms: float | None = None
    tool_success: bool | None = None
    ts_start: float = 0.0
    ts_end: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LLMCallResult:
    """Normalized result of one OpenAI-compatible chat-completions call."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    llm_latency_ms: float
    raw_response: dict[str, Any]


class AgentBase(ABC):
    """Shared base class for all benchmark agents."""

    def __init__(
        self,
        agent_id: str,
        api_base: str,
        model: str,
        *,
        api_key: str = "EMPTY",
        request_timeout_s: float = 180.0,
    ) -> None:
        self.agent_id = agent_id
        self.api_base = api_base
        self.model = model
        self.trace: list[StepRecord] = []
        self.task_id: str = ""
        self.task_success: bool | None = None
        self._client = AsyncOpenAI(
            base_url=api_base,
            api_key=api_key,
            timeout=request_timeout_s,
        )

    async def _call_llm(self, messages: list[dict[str, Any]]) -> LLMCallResult:
        """Call the OpenAI-compatible endpoint and normalize the response."""
        started = time.monotonic()
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            extra_body={"program_id": self.agent_id},
        )
        elapsed_ms = (time.monotonic() - started) * 1000
        usage = getattr(response, "usage", None)
        return LLMCallResult(
            content=_message_content_to_text(response.choices[0].message.content),
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            llm_latency_ms=elapsed_ms,
            raw_response=response.model_dump(),
        )

    def build_step_record(
        self,
        *,
        step_idx: int,
        phase: str,
        llm_result: LLMCallResult,
        ts_start: float,
        ts_end: float,
        extra: dict[str, Any] | None = None,
    ) -> StepRecord:
        """Construct a typed step record from an LLM call result."""
        return StepRecord(
            step_idx=step_idx,
            phase=phase,
            program_id=self.agent_id,
            prompt_tokens=llm_result.prompt_tokens,
            completion_tokens=llm_result.completion_tokens,
            llm_latency_ms=llm_result.llm_latency_ms,
            llm_output=llm_result.content,
            raw_response=llm_result.raw_response,
            ts_start=ts_start,
            ts_end=ts_end,
            extra=extra or {},
        )

    @abstractmethod
    async def run(self, task: dict[str, Any]) -> bool:
        """Run the agent on one task and return success/failure."""

    def get_trace(self) -> list[dict[str, Any]]:
        """Export the trace as JSON-serializable dictionaries."""
        return [asdict(record) for record in self.trace]

    def summary(self) -> dict[str, Any]:
        """Aggregate per-agent trace statistics for downstream analysis."""
        total_llm_ms = sum(record.llm_latency_ms for record in self.trace)
        total_tool_ms = sum(record.tool_duration_ms or 0.0 for record in self.trace)
        total_tokens = sum(
            record.prompt_tokens + record.completion_tokens for record in self.trace
        )
        return {
            "agent_id": self.agent_id,
            "program_id": self.agent_id,
            "task_id": self.task_id,
            "n_steps": len(self.trace),
            "total_llm_ms": total_llm_ms,
            "total_tool_ms": total_tool_ms,
            "total_tokens": total_tokens,
            "success": self.task_success,
        }
