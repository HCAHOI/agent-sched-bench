from __future__ import annotations

import asyncio
import fcntl
import hashlib
import ipaddress
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, TextIO
from urllib.parse import urlparse

import httpx

from agents.openclaw.eval.types import EvalResult
from agents.openclaw.tools.container import build_container_tool_overrides
from llm_call.provider_base import (
    GenerationSettings,
    LLMProvider,
    LLMResponse,
    ToolCallRequest,
)
from trace_collect.openclaw_tools import ContainerAgent
from trace_collect.tool_gap_loan import (
    ToolGapLoanConfig,
    ToolGapLoanRuntime,
    ToolGapPrediction,
)


@dataclass(slots=True)
class ReplaySleepRecord:
    phase: str
    expected_s: float
    actual_s: float
    source: str
    pid: int

    @property
    def drift_s(self) -> float:
        return self.actual_s - self.expected_s

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "phase": self.phase,
            "expected_s": round(self.expected_s, 6),
            "actual_s": round(self.actual_s, 6),
            "drift_s": round(self.drift_s, 6),
            "source": self.source,
            "pid": self.pid,
        }


@dataclass(frozen=True, slots=True)
class ReplayActionFailureCounts:
    emitted_actions: int
    source_failed_actions: int
    replay_failed_actions: int
    unexpected_replay_failed_actions: int
    action_sequence_matches: bool


@dataclass(frozen=True, slots=True)
class ShadowGenerationConfig:
    """Loopback vLLM settings for fixed-trajectory shadow generation."""

    api_base: str
    model: str
    timeout_s: float
    seed: int
    admission_slot_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        parsed = urlparse(self.api_base)
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("shadow api_base must not contain credentials")
        host = (parsed.hostname or "").lower().rstrip(".")
        try:
            is_loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            is_loopback = host == "localhost"
        if parsed.scheme not in {"http", "https"} or not is_loopback:
            raise ValueError("shadow api_base must be an http(s) loopback URL")
        if not self.model:
            raise ValueError("shadow model must be non-empty")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("shadow timeout_s must be finite and > 0")
        slot_paths = tuple(self.admission_slot_paths)
        if any(not path for path in slot_paths) or len(set(slot_paths)) != len(
            slot_paths
        ):
            raise ValueError("shadow admission slot paths must be unique and non-empty")
        object.__setattr__(self, "admission_slot_paths", slot_paths)


@dataclass(slots=True)
class _ShadowRequestSlotLease:
    slot_index: int
    _handle: TextIO | None

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


class _ShadowRequestSlotPool:
    """Cross-process request slots backed by kernel-released file locks."""

    def __init__(self, slot_paths: tuple[str, ...]) -> None:
        if not slot_paths:
            raise ValueError("shadow request slot pool requires at least one slot")
        self._slot_paths = slot_paths

    async def acquire(self) -> _ShadowRequestSlotLease:
        # ponytail: polling is bounded to staged replay workers; add a broker
        # only if measured starvation makes lock ordering insufficient.
        while True:
            for slot_index, slot_path in enumerate(self._slot_paths):
                handle = open(slot_path, "r+", encoding="utf-8")
                try:
                    fcntl.flock(
                        handle.fileno(),
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                except BlockingIOError:
                    handle.close()
                    continue
                return _ShadowRequestSlotLease(slot_index, handle)
            await asyncio.sleep(0.005)


def _tool_name(record: dict[str, Any]) -> str | None:
    data = record.get("data")
    if isinstance(data, dict) and data.get("tool_name"):
        return str(data["tool_name"])
    return None


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _replayable_source_actions(
    source_actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        action
        for action in source_actions
        if action.get("action_type") in {"llm_call", "tool_exec"}
    ]


def _resource_expected_calls(
    source_actions: list[dict[str, Any]],
) -> list[dict[str, str]]:
    expected = []
    for action in source_actions:
        data = action.get("data")
        if (
            action.get("action_type") != "tool_exec"
            or not isinstance(data, dict)
            or data.get("tool_name") != "exec"
        ):
            continue
        raw_args = data.get("tool_args")
        try:
            args = (
                raw_args
                if isinstance(raw_args, dict)
                else json.loads(str(raw_args or "{}"))
            )
        except json.JSONDecodeError:
            args = {}
        expected.append(
            {
                "source_tool_call_id": str(data.get("tool_call_id") or ""),
                "source_command": str(args.get("command") or ""),
                "source_tool_result": str(
                    data.get("tool_result", data.get("result", "")) or ""
                ),
            }
        )
    return expected


def _action_matches_source(
    replay_record: dict[str, Any],
    source_action: dict[str, Any],
    *,
    require_exact_tool_calls: bool = False,
) -> bool:
    if replay_record.get("action_type") != source_action.get("action_type"):
        return False
    replay_tool = _tool_name(replay_record)
    source_tool = _tool_name(source_action)
    if replay_tool != source_tool:
        return False
    if not require_exact_tool_calls or source_action.get("action_type") != "tool_exec":
        return True
    replay_data = replay_record.get("data") or {}
    source_data = source_action.get("data") or {}
    return (
        replay_record.get("action_id") == source_action.get("action_id")
        and replay_data.get("tool_call_id") == source_data.get("tool_call_id")
        and replay_data.get("tool_args") == source_data.get("tool_args")
    )


def _action_failed(record: dict[str, Any]) -> bool:
    data = record.get("data")
    return isinstance(data, dict) and data.get("success") is False


def replay_action_failure_counts(
    source_actions: list[dict[str, Any]],
    replay_records: list[dict[str, Any]],
    *,
    require_exact_tool_calls: bool = False,
) -> ReplayActionFailureCounts:
    """Compare replay failures against source actions aligned by replay order.

    A replay failure is unexpected only when the corresponding source action
    at the same replay position, action type, and tool name was not already a
    recorded source failure.
    """
    source_replay_actions = _replayable_source_actions(source_actions)
    source_failed_actions = sum(
        1 for action in source_replay_actions if _action_failed(action)
    )

    emitted_actions = 0
    replay_failed_actions = 0
    unexpected_replay_failed_actions = 0
    action_sequence_matches = True
    for record in replay_records:
        if record.get("type") != "action":
            continue
        source_action = (
            source_replay_actions[emitted_actions]
            if emitted_actions < len(source_replay_actions)
            else None
        )
        emitted_actions += 1
        if source_action is None or not _action_matches_source(
            record,
            source_action,
            require_exact_tool_calls=require_exact_tool_calls,
        ):
            action_sequence_matches = False
        if not _action_failed(record):
            continue
        replay_failed_actions += 1
        if (
            source_action is None
            or not _action_matches_source(
                record,
                source_action,
                require_exact_tool_calls=require_exact_tool_calls,
            )
            or not _action_failed(source_action)
        ):
            unexpected_replay_failed_actions += 1

    return ReplayActionFailureCounts(
        emitted_actions=emitted_actions,
        source_failed_actions=source_failed_actions,
        replay_failed_actions=replay_failed_actions,
        unexpected_replay_failed_actions=unexpected_replay_failed_actions,
        action_sequence_matches=(
            action_sequence_matches and emitted_actions == len(source_replay_actions)
        ),
    )


def replay_framework_failure_count(
    action_counts: ReplayActionFailureCounts,
    *,
    missing_actions: int,
    require_source_outcome_match: bool,
) -> int:
    """Count structural failures, plus outcome drift only when required."""
    failed_actions = missing_actions
    if require_source_outcome_match:
        failed_actions += action_counts.unexpected_replay_failed_actions
    if not action_counts.action_sequence_matches:
        failed_actions = max(1, failed_actions)
    return failed_actions


def _replay_execution_completed(
    *,
    action_counts: ReplayActionFailureCounts,
    expected_actions: int,
    stop_reason: str,
    error: str | None,
    source_terminal_reason: str,
    source_terminal_boundary_reached: bool,
    provider_request_sequence_matches: bool,
    require_source_outcome_match: bool = True,
) -> bool:
    if (
        action_counts.emitted_actions != expected_actions
        or not action_counts.action_sequence_matches
        or (
            require_source_outcome_match
            and action_counts.unexpected_replay_failed_actions
        )
        or not provider_request_sequence_matches
    ):
        return False
    if source_terminal_reason == "completed":
        return stop_reason == "completed" and error is None
    if source_terminal_reason == "llm_error":
        return stop_reason == "error"
    if source_terminal_reason in {"max_iterations", "trace_ended_after_tools"}:
        return stop_reason == "max_iterations"
    if source_terminal_reason == "trace_ended_before_tools":
        return source_terminal_boundary_reached and stop_reason == "error"
    return False


def validate_llm_replay_timing(
    *,
    replay_speed: float,
    timing_mode: str,
    llm_ttft_ms: float | None = None,
    llm_tpot_ms: float | None = None,
) -> None:
    """Validate exclusive LLM replay duration modes."""
    if replay_speed <= 0:
        raise ValueError("replay_speed must be > 0")
    if timing_mode == "source_scaled":
        if llm_ttft_ms is not None or llm_tpot_ms is not None:
            raise ValueError(
                "llm_ttft_ms/llm_tpot_ms require llm_timing_mode='ttft_tpot'"
            )
        return
    if timing_mode != "ttft_tpot":
        raise ValueError(f"Unsupported llm_timing_mode: {timing_mode}")
    if replay_speed != 1.0:
        raise ValueError(
            "replay_speed acceleration is exclusive with llm_timing_mode='ttft_tpot'"
        )
    if llm_ttft_ms is None:
        raise ValueError("llm_ttft_ms is required when llm_timing_mode='ttft_tpot'")
    if llm_tpot_ms is None:
        raise ValueError("llm_tpot_ms is required when llm_timing_mode='ttft_tpot'")
    if llm_ttft_ms < 0:
        raise ValueError("llm_ttft_ms must be non-negative")
    if llm_tpot_ms < 0:
        raise ValueError("llm_tpot_ms must be non-negative")


def llm_replay_duration_s(
    *,
    data: dict[str, Any],
    source_duration_s: float,
    replay_speed: float,
    timing_mode: str,
    llm_ttft_ms: float | None = None,
    llm_tpot_ms: float | None = None,
) -> tuple[float, dict[str, Any]]:
    """Return one replay LLM sleep duration and timing audit fields."""
    validate_llm_replay_timing(
        replay_speed=replay_speed,
        timing_mode=timing_mode,
        llm_ttft_ms=llm_ttft_ms,
        llm_tpot_ms=llm_tpot_ms,
    )
    if timing_mode == "source_scaled":
        return source_duration_s / replay_speed, {"llm_timing_mode": "source_scaled"}
    assert llm_ttft_ms is not None
    assert llm_tpot_ms is not None
    completion_tokens = _coerce_completion_tokens(data.get("completion_tokens", 0))
    simulated_ms = llm_ttft_ms + max(0, completion_tokens - 1) * llm_tpot_ms
    return simulated_ms / 1000.0, {
        "llm_timing_mode": "ttft_tpot",
        "simulated_ttft_ms": llm_ttft_ms,
        "simulated_tpot_ms": llm_tpot_ms,
        "simulated_llm_latency_ms": simulated_ms,
        "source_ttft_ms": data.get("ttft_ms"),
        "source_tpot_ms": data.get("tpot_ms"),
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
        stop_before_final_tool_calls: bool = False,
        shadow_generation: ShadowGenerationConfig | None = None,
        tool_gap_loan: ToolGapLoanRuntime | None = None,
    ) -> None:
        super().__init__(api_key=None, api_base=None)
        validate_llm_replay_timing(
            replay_speed=replay_speed,
            timing_mode=timing_mode,
            llm_ttft_ms=llm_ttft_ms,
            llm_tpot_ms=llm_tpot_ms,
        )
        self._llm_actions = list(llm_actions)
        self._replay_speed = replay_speed
        self._timing_mode = timing_mode
        self._llm_ttft_ms = llm_ttft_ms
        self._llm_tpot_ms = llm_tpot_ms
        self._model = model
        self._index = 0
        self._stop_before_final_tool_calls = stop_before_final_tool_calls
        self._shadow_generation = shadow_generation
        self._tool_gap_loan = tool_gap_loan
        self._shadow_client = (
            httpx.AsyncClient(timeout=shadow_generation.timeout_s, trust_env=False)
            if shadow_generation is not None
            else None
        )
        self._shadow_slot_pool = (
            _ShadowRequestSlotPool(shadow_generation.admission_slot_paths)
            if shadow_generation is not None
            and shadow_generation.admission_slot_paths
            else None
        )
        self.source_terminal_boundary_reached = False
        self.unexpected_request_count = 0
        self._replayed_action_count = 0
        self.sleep_records: list[ReplaySleepRecord] = []
        self.generation = GenerationSettings()

    def get_default_model(self) -> str:
        return self._model

    @classmethod
    def _is_transient_error(cls, content: str | None) -> bool:
        return False

    @staticmethod
    def _strip_image_content(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        return None

    @property
    def request_sequence_matches(self) -> bool:
        return (
            self._index == len(self._llm_actions)
            and self._replayed_action_count == len(self._llm_actions)
            and self.unexpected_request_count == 0
        )

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
        del (
            messages,
            tools,
            model,
            max_tokens,
            temperature,
            reasoning_effort,
            tool_choice,
        )
        if self._index >= len(self._llm_actions):
            self.unexpected_request_count += 1
            return LLMResponse(
                content="Replay trace exhausted before OpenClaw produced a final response.",
                finish_reason="error",
                extra={"replay_failure_kind": "llm_trace_exhausted"},
            )

        action = self._llm_actions[self._index]
        source_action_index = action.get("_source_action_index", self._index)
        self._index += 1
        data = dict(action.get("data") or {})
        start_s = float(action.get("ts_start", 0.0) or 0.0)
        end_s = float(action.get("ts_end", start_s) or start_s)
        source_duration_s = max(0.0, end_s - start_s)
        wall_start = time.time()
        shadow_metrics: dict[str, Any] | None = None
        if self._shadow_generation is None:
            sleep_s, timing_fields = self._duration_s(data, source_duration_s)
            sleep_record = await self._sleep(sleep_s, phase="llm_replay")
        else:
            source_action_id = action.get("action_id")
            if not isinstance(source_action_id, str) or not source_action_id:
                raise RuntimeError("shadow generation source action has no action_id")
            if (
                not isinstance(source_action_index, int)
                or isinstance(source_action_index, bool)
                or source_action_index < 0
            ):
                raise RuntimeError("shadow generation source action has no valid index")
            request_ready_monotonic = time.monotonic()
            request_ready_wall_time_s = time.time()
            lease = (
                await self._shadow_slot_pool.acquire()
                if self._shadow_slot_pool is not None
                else None
            )
            slot_acquired_monotonic = time.monotonic()
            slot_acquired_wall_time_s = time.time()
            try:
                shadow_metrics = await self._shadow_generate(
                    data,
                    source_action_id=source_action_id,
                    source_action_index=source_action_index,
                )
            finally:
                slot_released_wall_time_s = time.time()
                if lease is not None:
                    lease.release()
            if lease is not None:
                admission_wait_ms = round(
                    (slot_acquired_monotonic - request_ready_monotonic) * 1000.0,
                    3,
                )
                shadow_metrics.update(
                    {
                        "admission_slot_index": lease.slot_index,
                        "admission_wait_ms": admission_wait_ms,
                        "end_to_end_ttft_ms": round(
                            admission_wait_ms + float(shadow_metrics["ttft_ms"]),
                            3,
                        ),
                        "request_ready_wall_time_s": request_ready_wall_time_s,
                        "slot_acquired_wall_time_s": slot_acquired_wall_time_s,
                        "slot_released_wall_time_s": slot_released_wall_time_s,
                    }
                )
            if self._tool_gap_loan is not None:
                self._tool_gap_loan.record_llm_response(
                    source_action_id,
                    float(shadow_metrics["latency_ms"]),
                    wall_end_s=slot_released_wall_time_s,
                )
            sleep_record = None
            timing_fields = {"llm_timing_mode": "shadow_generation"}
        wall_end = time.time()

        raw_response = (
            data.get("raw_response")
            if isinstance(data.get("raw_response"), dict)
            else {}
        )
        message = self._raw_message(raw_response)
        tool_calls = self._tool_calls(message)
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        usage = self._usage(raw_response, data)
        finish_reason = self._finish_reason(
            raw_response, default="tool_calls" if tool_calls else "stop"
        )
        source_finish_reason = finish_reason
        if (
            self._stop_before_final_tool_calls
            and self._index == len(self._llm_actions)
            and tool_calls
        ):
            self.source_terminal_boundary_reached = True
            tool_calls = []
            finish_reason = "error"
            if not content:
                content = "Source trace ended before its final tool call executed."
        extra: dict[str, Any] = {
            "llm_call_time_ms": round((wall_end - wall_start) * 1000, 3),
            "llm_latency_ms": round((wall_end - wall_start) * 1000, 3),
            "llm_wall_ts_end": wall_end,
            "llm_timing_source": (
                "shadow_generation"
                if shadow_metrics is not None
                else "openclaw_replay_provider_sleep"
            ),
            "source_llm_latency_ms": data.get("llm_latency_ms"),
            "replay_speed": self._replay_speed,
            **timing_fields,
        }
        if self.source_terminal_boundary_reached:
            extra.update(
                {
                    "replay_failure_kind": "source_trace_terminal_boundary",
                    "source_finish_reason": source_finish_reason,
                }
            )
        if sleep_record is not None:
            extra["replay_sleep"] = sleep_record.to_dict()
        if shadow_metrics is not None:
            extra["shadow_generation"] = shadow_metrics
        response = LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            reasoning_content=(
                message.get("reasoning_content")
                if isinstance(message.get("reasoning_content"), str)
                else None
            ),
            extra=extra,
        )
        self._replayed_action_count += 1
        return response

    async def aclose(self) -> None:
        if self._shadow_client is not None:
            await self._shadow_client.aclose()

    async def _shadow_generate(
        self,
        data: dict[str, Any],
        *,
        source_action_id: str,
        source_action_index: int,
    ) -> dict[str, Any]:
        assert self._shadow_generation is not None
        assert self._shadow_client is not None
        requested_tokens = _coerce_completion_tokens(data.get("completion_tokens", 0))
        messages = self._shadow_messages(data.get("messages_in"))
        started_at = time.monotonic()
        first_token_at: float | None = None
        token_ids: list[int] = []
        prompt_token_ids: list[int] | None = None
        request_id: str | None = None
        finish_reason: str | None = None
        usage: dict[str, Any] = {}
        observed_usage_values: dict[str, int] = {}
        request = {
            "model": self._shadow_generation.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": requested_tokens,
            "ignore_eos": True,
            "return_token_ids": True,
            "seed": self._shadow_generation.seed,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if (
            self._tool_gap_loan is not None
            and not self._tool_gap_loan.config.can_lend
            and self._tool_gap_loan.config.borrower_priority is not None
        ):
            request["priority"] = self._tool_gap_loan.config.borrower_priority
        url = f"{self._shadow_generation.api_base.rstrip('/')}/chat/completions"
        async with self._shadow_client.stream("POST", url, json=request) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise RuntimeError("shadow generation returned invalid SSE JSON") from exc
                if not isinstance(chunk, dict):
                    raise RuntimeError("shadow generation returned a non-object SSE chunk")
                raw_request_id = chunk.get("id")
                if raw_request_id is not None:
                    if not isinstance(raw_request_id, str) or not raw_request_id:
                        raise RuntimeError(
                            "shadow generation returned a malformed request ID"
                        )
                    if request_id is None:
                        request_id = raw_request_id
                    elif request_id != raw_request_id:
                        raise RuntimeError(
                            "shadow generation returned conflicting request IDs: "
                            f"{request_id!r} and {raw_request_id!r}"
                        )
                raw_prompt_token_ids = chunk.get("prompt_token_ids")
                if raw_prompt_token_ids is not None:
                    if not isinstance(raw_prompt_token_ids, list) or any(
                        not isinstance(token_id, int) or isinstance(token_id, bool)
                        for token_id in raw_prompt_token_ids
                    ):
                        raise RuntimeError(
                            "shadow generation returned malformed prompt token IDs"
                        )
                    if (
                        prompt_token_ids is not None
                        and prompt_token_ids != raw_prompt_token_ids
                    ):
                        raise RuntimeError(
                            "shadow generation returned conflicting prompt token IDs"
                        )
                    prompt_token_ids = list(raw_prompt_token_ids)
                raw_usage = chunk.get("usage")
                if raw_usage is not None:
                    if not isinstance(raw_usage, dict):
                        raise RuntimeError(
                            "shadow generation returned malformed usage"
                        )
                    raw_prompt_tokens = raw_usage.get("prompt_tokens")
                    prompt_token_details = raw_usage.get("prompt_tokens_details")
                    if (
                        prompt_token_details is not None
                        and not isinstance(prompt_token_details, dict)
                    ):
                        raise RuntimeError(
                            "shadow generation returned malformed prompt token details"
                        )
                    raw_cached_prompt_tokens = (
                        prompt_token_details.get("cached_tokens")
                        if isinstance(prompt_token_details, dict)
                        else None
                    )
                    for usage_name, usage_value in (
                        ("prompt_tokens", raw_prompt_tokens),
                        ("cached_prompt_tokens", raw_cached_prompt_tokens),
                    ):
                        if usage_value is None:
                            continue
                        if (
                            not isinstance(usage_value, int)
                            or isinstance(usage_value, bool)
                            or usage_value < 0
                        ):
                            raise RuntimeError(
                                "shadow generation returned malformed usage "
                                f"{usage_name}"
                            )
                        previous_value = observed_usage_values.get(usage_name)
                        if (
                            previous_value is not None
                            and previous_value != usage_value
                        ):
                            raise RuntimeError(
                                "shadow generation returned conflicting usage "
                                f"{usage_name}: {previous_value} and {usage_value}"
                            )
                        observed_usage_values[usage_name] = usage_value
                    usage = raw_usage
                choices = chunk.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0]
                if not isinstance(choice, dict):
                    raise RuntimeError("shadow generation returned a malformed choice")
                raw_ids = choice.get("token_ids")
                delta = choice.get("delta")
                if isinstance(delta, dict):
                    raw_ids = delta.get("token_ids", raw_ids)
                if raw_ids is not None:
                    if not isinstance(raw_ids, list) or any(
                        not isinstance(token_id, int) or isinstance(token_id, bool)
                        for token_id in raw_ids
                    ):
                        raise RuntimeError("shadow generation returned malformed token IDs")
                    if raw_ids and first_token_at is None:
                        first_token_at = time.monotonic()
                    token_ids.extend(raw_ids)
                if isinstance(choice.get("finish_reason"), str):
                    finish_reason = choice["finish_reason"]
        if len(token_ids) != requested_tokens:
            raise RuntimeError(
                "shadow generation token count mismatch: "
                f"expected {requested_tokens}, got {len(token_ids)}"
            )
        if request_id is None:
            raise RuntimeError("shadow generation returned no request ID")
        if prompt_token_ids is None:
            raise RuntimeError("shadow generation returned no prompt token IDs")
        raw_prompt_tokens = usage.get("prompt_tokens")
        if raw_prompt_tokens is None:
            raise RuntimeError("shadow generation returned no valid prompt token count")
        if raw_prompt_tokens != len(prompt_token_ids):
            raise RuntimeError(
                "shadow generation prompt token count mismatch: "
                f"usage reported {raw_prompt_tokens}, got {len(prompt_token_ids)} IDs"
            )
        prompt_token_details = usage.get("prompt_tokens_details")
        cached_prompt_tokens = (
            prompt_token_details.get("cached_tokens")
            if isinstance(prompt_token_details, dict)
            else None
        )
        finished_at = time.monotonic()
        return {
            "model": self._shadow_generation.model,
            "seed": self._shadow_generation.seed,
            "request_priority": int(request.get("priority", 0)),
            "source_action_id": source_action_id,
            "source_action_index": source_action_index,
            "request_id": request_id,
            "messages_sha256": _canonical_json_sha256(request["messages"]),
            "prompt_tokens": raw_prompt_tokens,
            "prompt_token_ids_sha256": _canonical_json_sha256(prompt_token_ids),
            "cached_prompt_tokens": cached_prompt_tokens,
            "requested_completion_tokens": requested_tokens,
            "returned_completion_tokens": len(token_ids),
            "completion_token_ids": token_ids,
            "completion_token_ids_sha256": _canonical_json_sha256(token_ids),
            "finish_reason": finish_reason,
            "ttft_ms": round(
                ((first_token_at or finished_at) - started_at) * 1000.0, 3
            ),
            "latency_ms": round((finished_at - started_at) * 1000.0, 3),
        }

    @staticmethod
    def _shadow_messages(raw_messages: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_messages, list):
            raise RuntimeError("shadow generation source messages_in is missing")
        messages: list[dict[str, Any]] = []
        for raw_message in raw_messages:
            if not isinstance(raw_message, dict):
                raise RuntimeError("shadow generation source message is malformed")
            role = raw_message.get("role")
            if not isinstance(role, str):
                raise RuntimeError("shadow generation source message has no role")
            if role == "tool":
                tool_call_id = raw_message.get("tool_call_id")
                if not isinstance(tool_call_id, str) or not tool_call_id:
                    raise RuntimeError("shadow generation tool message has no tool_call_id")
                content = raw_message.get("content")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": (
                            content
                            if isinstance(content, str)
                            else json.dumps(
                                content, ensure_ascii=False, separators=(",", ":")
                            )
                        ),
                    }
                )
                continue
            message: dict[str, Any] = {"role": role, "content": raw_message.get("content")}
            if role == "assistant" and isinstance(raw_message.get("tool_calls"), list):
                message["tool_calls"] = raw_message["tool_calls"]
                if not message["content"]:
                    message["content"] = None
            messages.append(message)
        return messages

    def _duration_s(
        self,
        data: dict[str, Any],
        source_duration_s: float,
    ) -> tuple[float, dict[str, Any]]:
        return llm_replay_duration_s(
            data=data,
            source_duration_s=source_duration_s,
            replay_speed=self._replay_speed,
            timing_mode=self._timing_mode,
            llm_ttft_ms=self._llm_ttft_ms,
            llm_tpot_ms=self._llm_tpot_ms,
        )

    async def _sleep(
        self, expected_s: float, *, phase: str
    ) -> ReplaySleepRecord | None:
        if expected_s <= 0:
            return None
        import asyncio

        start = time.monotonic()
        await asyncio.sleep(expected_s)
        actual = time.monotonic() - start
        record = ReplaySleepRecord(
            phase=phase,
            expected_s=expected_s,
            actual_s=actual,
            source="openclaw_replay_provider_sleep",
            pid=os.getpid(),
        )
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
            if isinstance(choice, dict) and isinstance(
                choice.get("finish_reason"), str
            ):
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


def _coerce_completion_tokens(value: Any) -> int:
    if value is None or value == "":
        return 0
    tokens = int(value)
    if tokens < 0:
        raise ValueError(f"completion_tokens must be non-negative, got {value!r}")
    return tokens


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


_RUNTIME_LABEL_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.+-]{1,40}$")
_RUNTIME_LABEL_PATH_RE = re.compile(r"^/[A-Za-z0-9_./:@+-]{0,159}$")
_RUNTIME_LABEL_UID_RE = re.compile(r"^[0-9]{1,10}$")
_PYTHON_VERSION_LABEL_RE = re.compile(
    r"^Python [0-9]+(?:\.[0-9]+){1,3}[A-Za-z0-9.+_-]*$"
)


def _safe_runtime_label_token(value: object, *, default: str = "unknown") -> str:
    text = ("" if value is None else str(value)).strip()
    if _RUNTIME_LABEL_TOKEN_RE.fullmatch(text):
        return text
    return default


def _safe_runtime_label_path(value: object, *, default: str = "unknown") -> str:
    text = ("" if value is None else str(value)).strip()
    if _RUNTIME_LABEL_PATH_RE.fullmatch(text):
        return text
    return default


def _safe_runtime_label_uid(value: object, *, default: str = "unknown") -> str:
    text = ("" if value is None else str(value)).strip()
    if _RUNTIME_LABEL_UID_RE.fullmatch(text):
        return text
    return default


def _safe_python_version_label(value: object) -> str:
    text = ("" if value is None else str(value)).strip()
    if "\n" in text or "\r" in text:
        return "unknown"
    if _PYTHON_VERSION_LABEL_RE.fullmatch(text):
        return text
    return "unknown"


def container_runtime_label(proof: dict[str, Any]) -> str:
    """Describe the tool execution environment shown to OpenClaw."""

    workdir = _safe_runtime_label_path(proof.get("tool_container_workdir"))
    user = _safe_runtime_label_token(proof.get("tool_container_user"))
    uid = _safe_runtime_label_uid(proof.get("tool_container_user_id"))
    os_name = _safe_runtime_label_token(proof.get("tool_container_os"))
    arch = _safe_runtime_label_token(proof.get("tool_container_arch"))
    python = _safe_python_version_label(proof.get("tool_container_python"))
    return (
        f"Shell/file tools runtime: {os_name} {arch}\n"
        f"Shell/file tools workdir: {workdir}\n"
        f"Shell/file tools user: {user} (uid {uid})\n"
        f"Shell/file tools `python3`: {python}"
    )


async def container_runtime_proof(
    agent: ContainerAgent,
    *,
    container_id: str,
    mode: str,
    expected_workdir: str,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    probe_command = (
        "id -u && pwd && "
        "(uname -s 2>/dev/null || printf 'unknown\\n') && "
        "(uname -m 2>/dev/null || printf 'unknown\\n') && "
        "(if command -v python3 >/dev/null 2>&1; then "
        "python3 --version 2>&1; else printf 'python3 unavailable\\n'; fi)"
    )
    response = await agent.execute(
        {"tool": "exec", "args": {"command": probe_command, "timeout": timeout_s}},
        timeout_s=timeout_s + 5.0,
    )
    if not response.get("ok", False):
        raise RuntimeError(f"container root proof failed: {response.get('result', '')}")
    if int(response.get("returncode", 1)) != 0:
        raise RuntimeError(
            "container root proof command failed with returncode "
            f"{response.get('returncode')}: {response.get('result', '')}"
        )
    lines = str(response.get("result", "")).strip().splitlines()
    if len(lines) < 5:
        raise RuntimeError(
            f"container root proof returned malformed output: {response!r}"
        )
    uid_text = lines[0].strip()
    observed_workdir = lines[1].strip()
    observed_os = lines[2].strip()
    observed_arch = lines[3].strip()
    observed_python = lines[4].strip()
    if uid_text != "0":
        raise RuntimeError(f"container tool bridge is not root: uid={uid_text!r}")
    if observed_workdir != expected_workdir:
        raise RuntimeError(
            "container tool bridge workdir mismatch: "
            f"expected {expected_workdir!r}, observed {observed_workdir!r}"
        )
    return {
        "agent_execution_environment": "host",
        "tool_execution_environment": "task_container",
        "tool_container_id": container_id,
        "tool_container_user": "root",
        "tool_container_user_id": 0,
        "tool_container_workdir": observed_workdir,
        "tool_container_os": observed_os,
        "tool_container_arch": observed_arch,
        "tool_container_python": observed_python,
        "tool_runtime": "ContainerAgent",
        "openclaw_host_pid": os.getpid(),
        "mode": mode,
    }


def build_container_tools_for_agent(
    agent: ContainerAgent,
    *,
    exec_timeout: int,
    exec_timeout_floor: int | None = None,
    exec_path_append: str = "",
    workspace: str = "/testbed",
    resource_trace: Any | None = None,
    tool_gap_loan: ToolGapLoanRuntime | None = None,
    runtime_artifact_root_map: dict[str, str] | None = None,
    exec_timeout_floor_exempt_call_ids: set[str] | frozenset[str] = frozenset(),
) -> list[Any]:
    return build_container_tool_overrides(
        agent,
        exec_timeout=exec_timeout,
        exec_timeout_floor=exec_timeout_floor,
        exec_path_append=exec_path_append,
        workspace=workspace,
        resource_trace=resource_trace,
        tool_gap_loan=tool_gap_loan,
        runtime_artifact_root_map=runtime_artifact_root_map,
        exec_timeout_floor_exempt_call_ids=exec_timeout_floor_exempt_call_ids,
    )


def _attach_resource_observations(
    trace_path: Path,
    calls: list[dict[str, Any]],
    source_actions: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    duplicate_call_ids: set[str] = set()
    for call in calls:
        tool_call_id = str(call.get("tool_call_id") or "")
        if not tool_call_id:
            errors.append("resource observation has no tool_call_id")
        elif tool_call_id in by_id:
            duplicate_call_ids.add(tool_call_id)
        else:
            by_id[tool_call_id] = call
    for tool_call_id in sorted(duplicate_call_ids):
        errors.append(f"duplicate resource observation tool_call_id {tool_call_id}")
        by_id.pop(tool_call_id)

    seen: set[str] = set()
    if not trace_path.exists():
        return [*errors, "resource observation trace is missing"]
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    exec_action_ids: list[str] = []
    for record in records:
        if record.get("type") != "action" or record.get("action_type") != "tool_exec":
            continue
        data = record.get("data")
        if isinstance(data, dict) and data.get("tool_name") == "exec":
            exec_action_ids.append(str(data.get("tool_call_id") or ""))
    duplicate_action_ids = {
        tool_call_id
        for tool_call_id in exec_action_ids
        if tool_call_id and exec_action_ids.count(tool_call_id) > 1
    }
    for tool_call_id in sorted(duplicate_action_ids):
        errors.append(f"duplicate exec action tool_call_id {tool_call_id}")

    updated: list[str] = []
    source_replay_actions = _replayable_source_actions(source_actions)
    replay_action_index = 0
    for record in records:
        source_action: dict[str, Any] | None = None
        if record.get("type") == "action":
            if replay_action_index < len(source_replay_actions):
                source_action = source_replay_actions[replay_action_index]
            replay_action_index += 1
            if source_action is not None and not _action_matches_source(
                record, source_action
            ):
                source_action = None
        if record.get("type") == "action" and record.get("action_type") == "tool_exec":
            data = record.get("data")
            if isinstance(data, dict) and data.get("tool_name") == "exec":
                tool_call_id = str(data.get("tool_call_id") or "")
                summary = (
                    None
                    if tool_call_id in duplicate_action_ids
                    else by_id.get(tool_call_id)
                )
                if summary is None:
                    errors.append(
                        f"exec action {tool_call_id or '<missing>'} has no "
                        "resource observation"
                    )
                else:
                    raw_tool_args = data.get("tool_args")
                    try:
                        tool_args = (
                            raw_tool_args
                            if isinstance(raw_tool_args, dict)
                            else json.loads(str(raw_tool_args or "{}"))
                        )
                    except json.JSONDecodeError:
                        tool_args = {}
                    command = tool_args.get("command")
                    if command != summary.get("command"):
                        errors.append(
                            f"exec action {tool_call_id} command does not match "
                            "resource observation"
                        )
                    else:
                        data["resource_observation"] = summary
                        seen.add(tool_call_id)
                source_data = (
                    source_action.get("data")
                    if isinstance(source_action, dict)
                    else None
                )
                source_result = (
                    source_data.get("tool_result", source_data.get("result", ""))
                    if isinstance(source_data, dict)
                    else ""
                )
                source_exit = _command_exit_code(str(source_result))
                replay_exit = _command_exit_code(str(data.get("tool_result") or ""))
                available = source_exit is not None and replay_exit is not None
                data["exit_code_agreement"] = {
                    "source": source_exit,
                    "replay": replay_exit,
                    "available": available,
                    "matches": (source_exit == replay_exit if available else None),
                }
        updated.append(json.dumps(record, ensure_ascii=False))
    for tool_call_id in sorted(set(by_id) - seen):
        errors.append(
            f"resource observation {tool_call_id} has no matching exec action"
        )
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=trace_path.parent,
            prefix=f".{trace_path.name}.",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write("\n".join(updated) + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, trace_path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return errors


def _command_exit_code(tool_result: str) -> int | None:
    marker = "Exit code:"
    if marker not in tool_result:
        return None
    value = tool_result.rsplit(marker, 1)[1].strip().splitlines()[0].strip()
    try:
        return int(value)
    except ValueError:
        return None


def _update_trace_metadata(trace_path: Path, extra: dict[str, Any]) -> None:
    if not trace_path.exists():
        return
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    updated: list[str] = []
    replaced = False
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        if not replaced and record.get("type") == "trace_metadata":
            record.update(extra)
            replaced = True
        updated.append(json.dumps(record, ensure_ascii=False))
    if not replaced:
        updated.insert(
            0, json.dumps({"type": "trace_metadata", **extra}, ensure_ascii=False)
        )
    trace_path.write_text("\n".join(updated) + "\n", encoding="utf-8")


def _worker_trace_action_counts(
    trace_path: Path,
    source_actions: list[dict[str, Any]],
    *,
    require_exact_tool_calls: bool = False,
) -> ReplayActionFailureCounts:
    if not trace_path.exists():
        return replay_action_failure_counts(
            source_actions,
            [],
            require_exact_tool_calls=require_exact_tool_calls,
        )
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return replay_action_failure_counts(
        source_actions,
        records,
        require_exact_tool_calls=require_exact_tool_calls,
    )


def _finalized_resource_status(
    collector: Any,
    *,
    replay_execution: str,
) -> tuple[dict[str, Any], list[str]]:
    try:
        finalize_error = collector.finalize(replay_execution=replay_execution)
    except BaseException as exc:
        finalize_error = f"telemetry finalize failed: {type(exc).__name__}: {exc}"
    if finalize_error is not None:
        return (
            {
                "telemetry_quality": "unavailable",
                "formal_completeness": "unavailable",
                "call_coverage": None,
                "collection_validity": "invalid",
            },
            [str(finalize_error)],
        )
    try:
        telemetry_payload = collector.final_artifact
        if not isinstance(telemetry_payload, dict):
            raise ValueError("resource-agentd returned no finalized artifact")
        return (
            {
                "telemetry_quality": telemetry_payload["telemetry_quality"],
                "formal_completeness": telemetry_payload["formal_completeness"],
                "call_coverage": telemetry_payload["call_coverage"],
                "collection_validity": telemetry_payload["collection_validity"],
            },
            [],
        )
    except BaseException as exc:
        return (
            {
                "telemetry_quality": "unavailable",
                "formal_completeness": "unavailable",
                "call_coverage": None,
                "collection_validity": "invalid",
            },
            [f"telemetry artifact unavailable: {type(exc).__name__}: {exc}"],
        )


async def run_openclaw_host_replay_request(request: dict[str, Any]) -> dict[str, Any]:
    """Run one OpenClaw replay in this host process and write structured status."""

    from agents.openclaw._session_runner import SessionRunner

    source_actions = list(request["source_actions"])
    llm_actions = [
        {**action, "_source_action_index": action_index}
        for action_index, action in enumerate(source_actions)
        if action.get("action_type") == "llm_call"
    ]
    container_id = str(request["container_id"])
    container_executable = str(request["container_executable"])
    output_trace = Path(request["output_trace"])
    runtime_dir = Path(request["runtime_dir"])
    workspace = Path(request["workspace"])
    status_path = Path(request["status_path"])
    replay_speed = float(request["replay_speed"])
    llm_timing = dict(request["llm_timing"])
    shadow_payload = request.get("shadow_generation")
    shadow_generation = (
        ShadowGenerationConfig(**dict(shadow_payload))
        if shadow_payload is not None
        else None
    )
    tool_gap_payload = request.get("tool_gap_loan")
    tool_gap_loan = None
    tool_gap_status = None
    if tool_gap_payload is not None:
        payload = dict(tool_gap_payload)
        predictions = tuple(
            ToolGapPrediction(
                sample_id=str(item["sample_id"]),
                command=str(item["command"]),
                probability_by_bucket=tuple(item["probability_by_bucket"]),
                hard_bucket=int(item["hard_bucket"]),
                provenance=dict(item["provenance"]),
            )
            for item in payload.get("predictions", [])
        )
        tool_gap_loan = ToolGapLoanRuntime(
            ToolGapLoanConfig(
                arm=payload["arm"],
                state_dir=str(payload["state_dir"]),
                task_id=str(payload["task_id"]),
                foreground_task_ids=tuple(payload["foreground_task_ids"]),
                can_lend=bool(payload["can_lend"]),
                predictions=predictions,
                borrower_priority=payload.get("borrower_priority"),
            )
        )
        tool_gap_status = {
            "arm": tool_gap_loan.config.arm,
            "state_dir": str(tool_gap_loan.state_dir),
            "task_id": tool_gap_loan.config.task_id,
            "can_lend": tool_gap_loan.config.can_lend,
            "prediction_count": len(tool_gap_loan.config.predictions),
            **(
                {"borrower_priority": tool_gap_loan.config.borrower_priority}
                if tool_gap_loan.config.borrower_priority is not None
                else {}
            ),
        }
    tool_resource_profile = request.get("tool_resource_profile")
    if tool_resource_profile is not None:
        tool_resource_profile = str(tool_resource_profile)
    tool_resource_run_token = request.get("tool_resource_run_token")
    if tool_resource_run_token is not None:
        tool_resource_run_token = str(tool_resource_run_token)
    command_timeout_s = float(request["command_timeout_s"])
    exec_timeout_floor_s = request.get("exec_timeout_floor_s")
    if exec_timeout_floor_s is not None:
        exec_timeout_floor_s = int(exec_timeout_floor_s)
    replay_action_contract = dict(request.get("replay_action_contract") or {})
    timeout_floor_exempt_call_ids = {
        str(value)
        for value in replay_action_contract.get(
            "exec_timeout_floor_exempt_call_ids", []
        )
    }
    require_source_outcome_match = bool(
        replay_action_contract.get("require_source_outcome_match", True)
    )
    require_exact_tool_calls = bool(
        replay_action_contract.get("require_exact_tool_calls", False)
    )
    run_instance_id = str(request["run_instance_id"])
    prompt = str(request["prompt"])
    container_workdir = str(request.get("container_workdir") or "/testbed")
    container_python_runtime = request.get("container_python_runtime")
    if container_python_runtime is not None:
        container_python_runtime = str(container_python_runtime)
    container_pythonpath = request.get("container_pythonpath")
    if container_pythonpath is not None:
        container_pythonpath = str(container_pythonpath)
    source_terminal_reason = str(request.get("source_terminal_reason") or "completed")

    agent = ContainerAgent(
        container_id,
        container_executable,
        python_runtime=container_python_runtime,
        pythonpath=container_pythonpath,
        workdir=container_workdir,
    )
    status: dict[str, Any]
    resource_trace: Any | None = None
    resource_trace_finalized = False
    telemetry_run_status = {
        "replay_execution": "completed",
        "telemetry_quality": "ok" if not tool_resource_profile else "unavailable",
        "formal_completeness": (
            "not_requested" if not tool_resource_profile else "unavailable"
        ),
        "call_coverage": None,
        "collection_validity": (
            "not_requested" if not tool_resource_profile else "invalid"
        ),
    }
    telemetry_errors: list[str] = []
    provider: OpenClawReplayProvider | None = None

    def finalize_resource_trace(replay_execution: str) -> None:
        nonlocal resource_trace_finalized
        if resource_trace is None or resource_trace_finalized:
            return
        try:
            if output_trace.exists():
                for error in _attach_resource_observations(
                    output_trace,
                    resource_trace.calls,
                    source_actions,
                ):
                    resource_trace.add_integrity_error(error)
            else:
                resource_trace.add_integrity_error("resource trace is missing")
        except BaseException as exc:
            telemetry_errors.append(
                f"resource attachment failed: {type(exc).__name__}: {exc}"
            )
            try:
                resource_trace.add_integrity_error(telemetry_errors[-1])
            except BaseException:
                pass
        try:
            final_status, final_errors = _finalized_resource_status(
                resource_trace,
                replay_execution=replay_execution,
            )
            telemetry_run_status.update(final_status)
            telemetry_errors.extend(final_errors)
            if resource_trace.final_artifact is not None and output_trace.exists():
                telemetry_errors.extend(
                    _attach_resource_observations(
                        output_trace,
                        resource_trace.calls,
                        source_actions,
                    )
                )
        finally:
            resource_trace_finalized = True

    if tool_resource_profile:
        from tool_resource.client import ResourceTrace

        resource_trace = ResourceTrace.open(
            tool_resource_profile,
            run_token=tool_resource_run_token or "",
            trace_id=run_instance_id,
            container_runtime=Path(container_executable).name,
            container_id=container_id,
            artifact_path=Path(request["resource_artifact_path"]),
            expected_calls=_resource_expected_calls(source_actions),
        )
        setup_error = await asyncio.to_thread(resource_trace.wait_ready)
        if setup_error is not None:
            telemetry_errors.append(setup_error)
            resource_trace.add_integrity_error(setup_error)

    wall_start = time.time()
    try:
        await agent.start()
        proof = await container_runtime_proof(
            agent,
            container_id=container_id,
            mode="replay",
            expected_workdir=container_workdir,
        )
        runtime_label = container_runtime_label(proof)
        provider = OpenClawReplayProvider(
            llm_actions=llm_actions,
            replay_speed=replay_speed,
            timing_mode=str(llm_timing["mode"]),
            llm_ttft_ms=llm_timing.get("ttft_ms"),
            llm_tpot_ms=llm_timing.get("tpot_ms"),
            model=str(request.get("source_model") or "replay-openclaw"),
            stop_before_final_tool_calls=(
                source_terminal_reason == "trace_ended_before_tools"
            ),
            shadow_generation=shadow_generation,
            tool_gap_loan=tool_gap_loan,
        )
        runner = SessionRunner(
            provider,
            model=provider.get_default_model(),
            max_iterations=max(1, len(llm_actions)),
            context_window_tokens=int(request.get("context_window_tokens") or 65536),
            tool_overrides=build_container_tools_for_agent(
                agent,
                exec_timeout=int(command_timeout_s),
                exec_timeout_floor=exec_timeout_floor_s,
                workspace=container_workdir,
                resource_trace=resource_trace,
                tool_gap_loan=tool_gap_loan,
                runtime_artifact_root_map=dict(
                    request.get("runtime_artifact_root_map") or {}
                ),
                exec_timeout_floor_exempt_call_ids=(timeout_floor_exempt_call_ids),
            ),
        )
        metadata_extra = {
            **proof,
            "task_instance_id": request["task_instance_id"],
            "source_action_agent_id": request["source_action_agent_id"],
            "source_agent_id": request["source_action_agent_id"],
            "run_instance_id": run_instance_id,
            "replay_mode": "openclaw_host_worker",
            "exec_timeout_floor_s": exec_timeout_floor_s,
            "paired_workload_contract": bool(
                request.get("paired_workload_contract", False)
            ),
            "replay_action_contract": replay_action_contract,
            **(
                {
                    "shadow_generation": {
                        "api_base": shadow_generation.api_base,
                        "model": shadow_generation.model,
                        "timeout_s": shadow_generation.timeout_s,
                        "seed": shadow_generation.seed,
                    }
                }
                if shadow_generation is not None
                else {}
            ),
            "tool_resource": {
                "profile": tool_resource_profile,
                "service_enabled": bool(tool_resource_profile),
            },
            **(
                {"tool_gap_loan": tool_gap_status}
                if tool_gap_status is not None
                else {}
            ),
        }
        result = await runner.run(
            prompt=prompt,
            workspace=workspace,
            tool_workspace=Path(container_workdir),
            project_workspace=Path(container_workdir),
            session_key=f"simulate:{run_instance_id}",
            trace_file=output_trace,
            runtime_dir=runtime_dir,
            instance_id=run_instance_id,
            channel="simulate",
            prepare_ms=None,
            runtime_label=runtime_label,
        )
        _update_trace_metadata(output_trace, metadata_extra)
        sleep_records = [record.to_dict() for record in provider.sleep_records]
        action_counts = _worker_trace_action_counts(
            output_trace,
            source_actions,
            require_exact_tool_calls=require_exact_tool_calls,
        )
        expected_actions = int(request.get("expected_action_count") or 0)
        missing_actions = max(0, expected_actions - action_counts.emitted_actions)
        failed_actions = replay_framework_failure_count(
            action_counts,
            missing_actions=missing_actions,
            require_source_outcome_match=require_source_outcome_match,
        )
        success = _replay_execution_completed(
            action_counts=action_counts,
            expected_actions=expected_actions,
            stop_reason=result.stop_reason,
            error=result.error,
            source_terminal_reason=source_terminal_reason,
            source_terminal_boundary_reached=(
                provider.source_terminal_boundary_reached
            ),
            provider_request_sequence_matches=(provider.request_sequence_matches),
            require_source_outcome_match=require_source_outcome_match,
        )
        finalize_resource_trace("completed" if success else "failed")
        telemetry_run_status["replay_execution"] = "completed" if success else "failed"
        wall_end = time.time()
        status = {
            "success": success,
            "stop_reason": result.stop_reason,
            "error": result.error,
            "elapsed_s": wall_end - wall_start,
            "output_trace": str(output_trace),
            "sleep_records": sleep_records,
            "emitted_actions": action_counts.emitted_actions,
            "failed_actions": failed_actions,
            "source_failed_actions": action_counts.source_failed_actions,
            "replay_failed_actions": action_counts.replay_failed_actions,
            "unexpected_replay_failed_actions": (
                action_counts.unexpected_replay_failed_actions
            ),
            "expected_actions": expected_actions,
            "missing_source_action_count": missing_actions,
            "action_sequence_matches": action_counts.action_sequence_matches,
            "source_success": request.get("source_success"),
            "source_terminal_reason": source_terminal_reason,
            "provider_request_sequence_matches": (provider.request_sequence_matches),
            "telemetry_integrity_failed": (
                telemetry_run_status["collection_validity"] == "invalid"
            ),
            "telemetry_errors": telemetry_errors,
            **telemetry_run_status,
            **metadata_extra,
        }
    except BaseException as exc:
        wall_end = time.time()
        status = {
            "success": False,
            "stop_reason": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": wall_end - wall_start,
            "output_trace": str(output_trace),
            "sleep_records": [],
            "agent_execution_environment": "host",
            "tool_execution_environment": "task_container",
            "tool_container_id": container_id,
            "tool_container_user": "unknown",
            "openclaw_host_pid": os.getpid(),
            "task_instance_id": request.get("task_instance_id"),
            "source_action_agent_id": request.get("source_action_agent_id"),
            "run_instance_id": run_instance_id,
            "replay_mode": "openclaw_host_worker",
            "replay_execution": "failed",
            "telemetry_quality": telemetry_run_status["telemetry_quality"],
            "formal_completeness": telemetry_run_status["formal_completeness"],
            "call_coverage": telemetry_run_status["call_coverage"],
            "collection_validity": telemetry_run_status["collection_validity"],
            "telemetry_integrity_failed": (
                bool(tool_resource_profile)
                and (
                    resource_trace is None
                    or type(exc).__name__ == "ClauseTelemetryIntegrityError"
                )
            ),
            "telemetry_errors": telemetry_errors,
            "tool_resource": {
                "profile": tool_resource_profile,
                "service_enabled": bool(tool_resource_profile),
            },
            **(
                {"tool_gap_loan": tool_gap_status}
                if tool_gap_status is not None
                else {}
            ),
        }
    finally:
        try:
            if resource_trace is not None and not resource_trace_finalized:
                finalize_resource_trace("failed")
        finally:
            try:
                if provider is not None:
                    await provider.aclose()
            finally:
                await agent.stop()
                status["telemetry_integrity_failed"] = bool(
                    status.get("telemetry_integrity_failed")
                    or telemetry_errors
                    or telemetry_run_status["collection_validity"] == "invalid"
                )
                status["telemetry_quality"] = telemetry_run_status["telemetry_quality"]
                status["formal_completeness"] = telemetry_run_status[
                    "formal_completeness"
                ]
                status["call_coverage"] = telemetry_run_status["call_coverage"]
                status["collection_validity"] = telemetry_run_status[
                    "collection_validity"
                ]
                status["telemetry_errors"] = telemetry_errors
                status_path.parent.mkdir(parents=True, exist_ok=True)
                status_path.write_text(
                    json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True)
                    + "\n",
                    encoding="utf-8",
                )
    return status
