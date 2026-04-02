from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from harness.trace_logger import TraceLogger

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
    messages_in: list[dict[str, Any]] | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)
    tool_name: str | None = None
    tool_args: str | None = None
    tool_result: str | None = None
    tool_duration_ms: float | None = None
    tool_success: bool | None = None
    tool_timeout: bool | None = None
    tool_ts_start: float | None = None
    tool_ts_end: float | None = None
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
        self._trace_logger: TraceLogger | None = None
        self.run_metadata: dict[str, Any] = {}
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
        messages_in: list[dict[str, Any]] | None = None,
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
            messages_in=messages_in,
            raw_response=llm_result.raw_response,
            ts_start=ts_start,
            ts_end=ts_end,
            extra=extra or {},
        )

    async def prepare(self, task: dict[str, Any]) -> None:
        """Prepare the agent's environment before the main loop.

        Called during the setup phase so that expensive operations
        (container creation, dependency installation) complete before
        all agents start competing for the GPU simultaneously.

        The default implementation is a no-op.  Subclasses that need
        heavyweight setup (e.g. Podman containers) should override this.
        """

    def _emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit a fine-grained event if a logger is injected. No-op otherwise."""
        if self._trace_logger is not None:
            self._trace_logger.log_event(self.agent_id, event_type, data)

    def _emit_step(self, record: StepRecord) -> None:
        """Append record to self.trace and write to logger immediately (if injected)."""
        record.extra.update(self.run_metadata)
        self.trace.append(record)
        if self._trace_logger is not None:
            self._trace_logger.log_step(self.agent_id, record)

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
        tool_ms_by_name: dict[str, float] = {}
        tool_timeouts: dict[str, int] = {}
        for record in self.trace:
            if record.tool_name is None:
                continue
            tool_ms_by_name[record.tool_name] = (
                tool_ms_by_name.get(record.tool_name, 0.0)
                + (record.tool_duration_ms or 0.0)
            )
            if record.tool_timeout:
                tool_timeouts[record.tool_name] = (
                    tool_timeouts.get(record.tool_name, 0) + 1
                )
        return {
            "agent_id": self.agent_id,
            "program_id": self.agent_id,
            "task_id": self.task_id,
            "n_steps": len(self.trace),
            "total_llm_ms": total_llm_ms,
            "total_tool_ms": total_tool_ms,
            "total_tokens": total_tokens,
            "tool_ms_by_name": tool_ms_by_name,
            "tool_timeouts": tool_timeouts,
            "success": self.task_success,
        }
