from __future__ import annotations

import asyncio
import random
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from openai import AsyncOpenAI


class ToolLatencySimulator:
    """Simulate realistic tool-call latency to keep KV cache occupied."""

    PROFILES: dict[str, Any] = {
        "local": (0, 0),
        "realistic": {
            "grep": (0.05, 0.1),
            "cat": (0.05, 0.1),
            "find": (0.05, 0.2),
            "sed": (0.05, 0.2),
            "bash": (0.1, 2.0),
            "pytest": (2.0, 15.0),
            "python": (0.5, 5.0),
            "git": (0.1, 1.0),
            "schema_inspect": (0.05, 0.1),
            "sql_execute": (0.2, 5.0),
            "web_search": (0.5, 3.0),
            "read_page": (1.0, 5.0),
            "default": (0.1, 3.0),
        },
        "heavy": (1.0, 10.0),
    }

    def __init__(self, profile: str = "realistic") -> None:
        if profile not in self.PROFILES:
            raise ValueError(f"Unknown latency profile: {profile}")
        self.profile = profile

    def _classify_bash_command(self, command: str) -> str:
        """Extract the base command name for latency lookup."""
        token = command.strip().split()[0] if command.strip() else "bash"
        return token.split("/")[-1]

    async def wrap(self, tool_name: str, real_duration_ms: float, command: str = "") -> float:
        """Sleep simulated delay and return total duration in ms."""
        if self.profile == "local":
            return real_duration_ms
        profile_data = self.PROFILES[self.profile]
        if isinstance(profile_data, dict):
            key = tool_name
            if tool_name == "bash" and command:
                key = self._classify_bash_command(command)
            lo, hi = profile_data.get(key, profile_data["default"])
        else:
            lo, hi = profile_data
        simulated_s = random.uniform(lo, hi)
        await asyncio.sleep(simulated_s)
        return real_duration_ms + simulated_s * 1000


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
class ToolCallResult:
    """A single tool call extracted from the LLM response."""

    id: str
    name: str
    arguments: str


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
    tool_calls: list[ToolCallResult] = field(default_factory=list)


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

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMCallResult:
        """Call the OpenAI-compatible endpoint with optional tool definitions."""
        extra_body: dict[str, Any] = {
            "program_id": self.agent_id,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        started = time.monotonic()
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            extra_body=extra_body,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        response = await self._client.chat.completions.create(**kwargs)
        elapsed_ms = (time.monotonic() - started) * 1000
        if not response.choices:
            raise RuntimeError(f"LLM returned empty choices: {response.model_dump()}")

        message = response.choices[0].message
        usage = getattr(response, "usage", None)

        # Extract structured tool calls if present
        parsed_tool_calls: list[ToolCallResult] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                parsed_tool_calls.append(
                    ToolCallResult(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    )
                )

        return LLMCallResult(
            content=_message_content_to_text(message.content),
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            llm_latency_ms=elapsed_ms,
            raw_response=response.model_dump(),
            tool_calls=parsed_tool_calls,
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
