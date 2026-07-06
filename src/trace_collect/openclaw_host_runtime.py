from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from agents.openclaw.eval.types import EvalResult
from agents.openclaw.tools.container import build_container_tool_overrides
from llm_call.provider_base import GenerationSettings, LLMProvider, LLMResponse, ToolCallRequest
from trace_collect.openclaw_tools import ContainerAgent


@dataclass(slots=True)
class ReplaySleepRecord:
    phase: str
    expected_s: float
    actual_s: float

    @property
    def drift_s(self) -> float:
        return self.actual_s - self.expected_s

    def to_dict(self) -> dict[str, float | str]:
        return {
            "phase": self.phase,
            "expected_s": round(self.expected_s, 6),
            "actual_s": round(self.actual_s, 6),
            "drift_s": round(self.drift_s, 6),
        }


class OpenClawReplayProvider(LLMProvider):
    """LLMProvider that replays source OpenClaw LLM responses and sleeps in-process."""

    preserve_openclaw_message_ids = True

    def __init__(
        self,
        *,
        llm_actions: list[dict[str, Any]],
        replay_speed: float,
        timing_mode: str,
        llm_ttft_ms: float | None = None,
        llm_tpot_ms: float | None = None,
        model: str = "replay-openclaw",
    ) -> None:
        super().__init__(api_key=None, api_base=None)
        if replay_speed <= 0:
            raise ValueError("replay_speed must be > 0")
        self._llm_actions = list(llm_actions)
        self._replay_speed = replay_speed
        self._timing_mode = timing_mode
        self._llm_ttft_ms = llm_ttft_ms
        self._llm_tpot_ms = llm_tpot_ms
        self._model = model
        self._index = 0
        self.sleep_records: list[ReplaySleepRecord] = []
        self.generation = GenerationSettings()

    def get_default_model(self) -> str:
        return self._model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        del messages, tools, model, max_tokens, temperature, reasoning_effort, tool_choice
        if self._index >= len(self._llm_actions):
            return LLMResponse(
                content="Replay trace exhausted before OpenClaw produced a final response.",
                finish_reason="error",
                extra={"replay_failure_kind": "llm_trace_exhausted"},
            )

        action = self._llm_actions[self._index]
        self._index += 1
        data = dict(action.get("data") or {})
        start_s = float(action.get("ts_start", 0.0) or 0.0)
        end_s = float(action.get("ts_end", start_s) or start_s)
        source_duration_s = max(0.0, end_s - start_s)
        sleep_s, timing_fields = self._duration_s(data, source_duration_s)
        wall_start = time.time()
        sleep_record = await self._sleep(sleep_s, phase="llm_replay")
        wall_end = time.time()

        raw_response = data.get("raw_response") if isinstance(data.get("raw_response"), dict) else {}
        message = self._raw_message(raw_response)
        tool_calls = self._tool_calls(message)
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        usage = self._usage(raw_response, data)
        finish_reason = self._finish_reason(raw_response, default="tool_calls" if tool_calls else "stop")
        extra: dict[str, Any] = {
            "llm_call_time_ms": round((wall_end - wall_start) * 1000, 3),
            "llm_latency_ms": round((wall_end - wall_start) * 1000, 3),
            "llm_wall_ts_end": wall_end,
            "llm_timing_source": "openclaw_replay_provider_sleep",
            "source_llm_latency_ms": data.get("llm_latency_ms"),
            "replay_speed": self._replay_speed,
            **timing_fields,
        }
        if sleep_record is not None:
            extra["replay_sleep"] = sleep_record.to_dict()
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            reasoning_content=message.get("reasoning_content") if isinstance(message.get("reasoning_content"), str) else None,
            extra=extra,
        )

    def _duration_s(
        self,
        data: dict[str, Any],
        source_duration_s: float,
    ) -> tuple[float, dict[str, Any]]:
        if self._timing_mode == "source_scaled":
            return source_duration_s / self._replay_speed, {
                "llm_timing_mode": "source_scaled",
            }
        if self._timing_mode != "ttft_tpot":
            raise ValueError(f"Unsupported llm_timing_mode: {self._timing_mode}")
        if self._llm_ttft_ms is None:
            raise ValueError("llm_ttft_ms is required when llm_timing_mode='ttft_tpot'")
        if self._llm_tpot_ms is None:
            raise ValueError("llm_tpot_ms is required when llm_timing_mode='ttft_tpot'")
        completion_tokens = _coerce_nonnegative_int(data.get("completion_tokens", 0))
        simulated_ms = self._llm_ttft_ms + max(0, completion_tokens - 1) * self._llm_tpot_ms
        return simulated_ms / 1000.0, {
            "llm_timing_mode": "ttft_tpot",
            "simulated_ttft_ms": self._llm_ttft_ms,
            "simulated_tpot_ms": self._llm_tpot_ms,
            "simulated_llm_latency_ms": simulated_ms,
            "source_ttft_ms": data.get("ttft_ms"),
            "source_tpot_ms": data.get("tpot_ms"),
        }

    async def _sleep(self, expected_s: float, *, phase: str) -> ReplaySleepRecord | None:
        if expected_s <= 0:
            return None
        import asyncio

        start = time.monotonic()
        await asyncio.sleep(expected_s)
        actual = time.monotonic() - start
        record = ReplaySleepRecord(phase=phase, expected_s=expected_s, actual_s=actual)
        self.sleep_records.append(record)
        return record

    @staticmethod
    def _raw_message(raw_response: dict[str, Any]) -> dict[str, Any]:
        choices = raw_response.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict) and isinstance(choice.get("message"), dict):
                return dict(choice["message"])
        return {"role": "assistant", "content": ""}

    @staticmethod
    def _finish_reason(raw_response: dict[str, Any], *, default: str) -> str:
        choices = raw_response.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict) and isinstance(choice.get("finish_reason"), str):
                return choice["finish_reason"]
        return default

    @staticmethod
    def _tool_calls(message: dict[str, Any]) -> list[ToolCallRequest]:
        raw_calls = message.get("tool_calls")
        if not isinstance(raw_calls, list):
            return []
        calls: list[ToolCallRequest] = []
        for index, raw in enumerate(raw_calls):
            if not isinstance(raw, dict):
                continue
            fn = raw.get("function")
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if not isinstance(name, str) or not name:
                continue
            raw_args = fn.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    arguments = json.loads(raw_args or "{}")
                except json.JSONDecodeError:
                    arguments = {}
            elif isinstance(raw_args, dict):
                arguments = raw_args
            else:
                arguments = {}
            calls.append(
                ToolCallRequest(
                    id=str(raw.get("id") or f"replay_call_{index}"),
                    name=name,
                    arguments=arguments,
                )
            )
        return calls

    @staticmethod
    def _usage(raw_response: dict[str, Any], data: dict[str, Any]) -> dict[str, int]:
        usage = raw_response.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        return {
            "prompt_tokens": _coerce_nonnegative_int(
                usage.get("prompt_tokens", data.get("prompt_tokens", 0))
            ),
            "completion_tokens": _coerce_nonnegative_int(
                usage.get("completion_tokens", data.get("completion_tokens", 0))
            ),
        }


def _coerce_nonnegative_int(value: Any) -> int:
    try:
        result = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, result)


async def extract_container_patch(
    agent: ContainerAgent,
    *,
    base_commit: str | None,
    timeout_s: float = 180.0,
) -> str | None:
    """Extract a SWE patch from /testbed through the container wrapper."""

    response = await agent.execute(
        {
            "tool": "extract_patch",
            "args": {
                "base_commit": base_commit or "HEAD",
                "exclude_pathspecs": EvalResult.exclude_pathspecs(),
            },
        },
        timeout_s=timeout_s,
    )
    if not response.get("ok", False):
        return None
    result = str(response.get("result", "")).strip()
    if not result:
        return None
    if not result.lstrip().startswith("diff --git"):
        return None
    return result


def container_runtime_proof(*, container_id: str, mode: str) -> dict[str, Any]:
    return {
        "agent_execution_environment": "host",
        "tool_execution_environment": "task_container",
        "tool_container_id": container_id,
        "tool_container_user": "root_or_image_default",
        "tool_runtime": "ContainerAgent",
        "openclaw_host_pid": __import__("os").getpid(),
        "mode": mode,
    }


def build_container_tools_for_agent(
    agent: ContainerAgent,
    *,
    exec_timeout: int,
    exec_path_append: str = "",
) -> list[Any]:
    return build_container_tool_overrides(
        agent,
        exec_timeout=exec_timeout,
        exec_path_append=exec_path_append,
    )
