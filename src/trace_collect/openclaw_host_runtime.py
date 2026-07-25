from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
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


def _tool_name(record: dict[str, Any]) -> str | None:
    data = record.get("data")
    if isinstance(data, dict) and data.get("tool_name"):
        return str(data["tool_name"])
    return None


def _replayable_source_actions(
    source_actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        action
        for action in source_actions
        if action.get("action_type") in {"llm_call", "tool_exec"}
    ]


def _action_matches_source(
    replay_record: dict[str, Any],
    source_action: dict[str, Any],
) -> bool:
    if replay_record.get("action_type") != source_action.get("action_type"):
        return False
    replay_tool = _tool_name(replay_record)
    source_tool = _tool_name(source_action)
    if replay_tool is not None and source_tool is not None:
        return replay_tool == source_tool
    return True


def _action_failed(record: dict[str, Any]) -> bool:
    data = record.get("data")
    return isinstance(data, dict) and data.get("success") is False


def replay_action_failure_counts(
    source_actions: list[dict[str, Any]],
    replay_records: list[dict[str, Any]],
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
    for record in replay_records:
        if record.get("type") != "action":
            continue
        source_action = (
            source_replay_actions[emitted_actions]
            if emitted_actions < len(source_replay_actions)
            else None
        )
        emitted_actions += 1
        if not _action_failed(record):
            continue
        replay_failed_actions += 1
        if (
            source_action is None
            or not _action_matches_source(record, source_action)
            or not _action_failed(source_action)
        ):
            unexpected_replay_failed_actions += 1

    return ReplayActionFailureCounts(
        emitted_actions=emitted_actions,
        source_failed_actions=source_failed_actions,
        replay_failed_actions=replay_failed_actions,
        unexpected_replay_failed_actions=unexpected_replay_failed_actions,
    )


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

        raw_response = (
            data.get("raw_response") if isinstance(data.get("raw_response"), dict) else {}
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
            reasoning_content=(
                message.get("reasoning_content")
                if isinstance(message.get("reasoning_content"), str)
                else None
            ),
            extra=extra,
        )

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

    async def _sleep(self, expected_s: float, *, phase: str) -> ReplaySleepRecord | None:
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
        raise RuntimeError(f"container root proof returned malformed output: {response!r}")
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
    exec_path_append: str = "",
    workspace: str = "/testbed",
    clause_telemetry: Any | None = None,
) -> list[Any]:
    return build_container_tool_overrides(
        agent,
        exec_timeout=exec_timeout,
        exec_path_append=exec_path_append,
        workspace=workspace,
        clause_telemetry=clause_telemetry,
    )


def _attach_clause_telemetry(
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
            errors.append("clause telemetry has no tool_call_id")
        elif tool_call_id in by_id:
            duplicate_call_ids.add(tool_call_id)
        else:
            by_id[tool_call_id] = call
    for tool_call_id in sorted(duplicate_call_ids):
        errors.append(f"duplicate clause telemetry tool_call_id {tool_call_id}")
        by_id.pop(tool_call_id)

    seen: set[str] = set()
    if not trace_path.exists():
        return [*errors, "clause telemetry trace is missing"]
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
                        "clause telemetry"
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
                            "clause telemetry"
                        )
                    else:
                        data["clause_telemetry"] = summary
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
                    "matches": (
                        source_exit == replay_exit if available else None
                    ),
                }
        updated.append(json.dumps(record, ensure_ascii=False))
    for tool_call_id in sorted(set(by_id) - seen):
        errors.append(
            f"clause telemetry {tool_call_id} has no matching exec action"
        )
    trace_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
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
        updated.insert(0, json.dumps({"type": "trace_metadata", **extra}, ensure_ascii=False))
    trace_path.write_text("\n".join(updated) + "\n", encoding="utf-8")


def _worker_trace_action_counts(
    trace_path: Path,
    source_actions: list[dict[str, Any]],
) -> ReplayActionFailureCounts:
    if not trace_path.exists():
        return replay_action_failure_counts(source_actions, [])
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return replay_action_failure_counts(source_actions, records)


async def run_openclaw_host_replay_request(request: dict[str, Any]) -> dict[str, Any]:
    """Run one OpenClaw replay in this host process and write structured status."""

    from agents.openclaw._session_runner import SessionRunner

    source_actions = list(request["source_actions"])
    llm_actions = [
        action for action in source_actions if action.get("action_type") == "llm_call"
    ]
    container_id = str(request["container_id"])
    container_executable = str(request["container_executable"])
    output_trace = Path(request["output_trace"])
    runtime_dir = Path(request["runtime_dir"])
    workspace = Path(request["workspace"])
    status_path = Path(request["status_path"])
    replay_speed = float(request["replay_speed"])
    llm_timing = dict(request["llm_timing"])
    tool_resource_telemetry = str(
        request.get("tool_resource_telemetry") or "command"
    )
    if tool_resource_telemetry not in {"off", "command", "clause"}:
        raise ValueError(
            "tool_resource_telemetry must be one of: off, command, clause"
        )
    os.environ["OPENCLAW_TOOL_RESOURCE_TELEMETRY"] = tool_resource_telemetry
    repo = str(request.get("repo") or "")
    if tool_resource_telemetry == "clause" and not repo:
        raise ValueError("clause telemetry requires task repository identity")
    command_timeout_s = float(request["command_timeout_s"])
    run_instance_id = str(request["run_instance_id"])
    prompt = str(request["prompt"])
    container_workdir = str(request.get("container_workdir") or "/testbed")
    container_python_runtime = request.get("container_python_runtime")
    if container_python_runtime is not None:
        container_python_runtime = str(container_python_runtime)
    container_pythonpath = request.get("container_pythonpath")
    if container_pythonpath is not None:
        container_pythonpath = str(container_pythonpath)

    agent = ContainerAgent(
        container_id,
        container_executable,
        python_runtime=container_python_runtime,
        pythonpath=container_pythonpath,
        workdir=container_workdir,
    )
    status: dict[str, Any]
    clause_collector: Any | None = None
    clause_collector_finalized = False
    wall_start = time.time()
    try:
        if tool_resource_telemetry == "clause":
            from trace_collect.clause_telemetry import ClauseTelemetryCollector

            clause_collector = ClauseTelemetryCollector(
                container_id=container_id,
                container_executable=container_executable,
                repo=repo,
                artifact_path=Path(request["clause_telemetry_path"]),
                source_actions=source_actions,
            )
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
        )
        runner = SessionRunner(
            provider,
            model=provider.get_default_model(),
            max_iterations=max(1, len(llm_actions)),
            context_window_tokens=int(request.get("context_window_tokens") or 65536),
            tool_overrides=build_container_tools_for_agent(
                agent,
                exec_timeout=int(command_timeout_s),
                workspace=container_workdir,
                clause_telemetry=clause_collector,
            ),
        )
        metadata_extra = {
            **proof,
            "task_instance_id": request["task_instance_id"],
            "source_action_agent_id": request["source_action_agent_id"],
            "source_agent_id": request["source_action_agent_id"],
            "run_instance_id": run_instance_id,
            "replay_mode": "openclaw_host_worker",
            "tool_resource_telemetry": {
                "mode": tool_resource_telemetry,
                "command_envelope_enabled": tool_resource_telemetry != "off",
                "clause_observations_enabled": (
                    tool_resource_telemetry == "clause"
                ),
                "segment_timeline_requested": bool(
                    request.get("segment_timeline_requested", True)
                ),
                "segment_timeline_enabled": bool(
                    request.get("segment_timeline_enabled", True)
                ),
                "segment_timeline_decision": request.get(
                    "segment_timeline_decision", "as_requested"
                ),
            },
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
        if clause_collector is not None:
            for error in _attach_clause_telemetry(
                output_trace,
                clause_collector.calls,
                source_actions,
            ):
                clause_collector.add_integrity_error(error)
            try:
                clause_collector.finalize()
            finally:
                clause_collector_finalized = True
        sleep_records = [record.to_dict() for record in provider.sleep_records]
        action_counts = _worker_trace_action_counts(output_trace, source_actions)
        expected_actions = int(request.get("expected_action_count") or 0)
        missing_actions = max(0, expected_actions - action_counts.emitted_actions)
        failed_actions = action_counts.unexpected_replay_failed_actions
        success = (
            result.stop_reason == "completed"
            and result.error is None
            and failed_actions == 0
            and missing_actions == 0
        )
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
            "telemetry_integrity_failed": False,
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
            "telemetry_integrity_failed": (
                tool_resource_telemetry == "clause"
                and (
                    clause_collector is None
                    or type(exc).__name__ == "ClauseTelemetryIntegrityError"
                )
            ),
            "tool_resource_telemetry": {
                "mode": tool_resource_telemetry,
                "command_envelope_enabled": tool_resource_telemetry != "off",
                "clause_observations_enabled": (
                    tool_resource_telemetry == "clause"
                ),
                "segment_timeline_requested": bool(
                    request.get("segment_timeline_requested", True)
                ),
                "segment_timeline_enabled": bool(
                    request.get("segment_timeline_enabled", True)
                ),
                "segment_timeline_decision": request.get(
                    "segment_timeline_decision", "as_requested"
                ),
            },
        }
    finally:
        try:
            await agent.stop()
        finally:
            if clause_collector is not None and not clause_collector_finalized:
                try:
                    if output_trace.exists():
                        for error in _attach_clause_telemetry(
                            output_trace,
                            clause_collector.calls,
                            source_actions,
                        ):
                            clause_collector.add_integrity_error(error)
                    try:
                        clause_collector.finalize()
                    finally:
                        clause_collector_finalized = True
                except BaseException as telemetry_exc:
                    status["success"] = False
                    status["stop_reason"] = "error"
                    status["telemetry_integrity_failed"] = True
                    telemetry_error = (
                        f"{type(telemetry_exc).__name__}: {telemetry_exc}"
                    )
                    prior_error = status.get("error")
                    status["error"] = (
                        f"{prior_error}; {telemetry_error}"
                        if prior_error
                        else telemetry_error
                    )
            status_path.parent.mkdir(parents=True, exist_ok=True)
            status_path.write_text(
                json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    return status
