from __future__ import annotations

import asyncio
import dataclasses
import functools
import hashlib
import json
import logging
import multiprocessing
import os
import re
import subprocess
import shutil
import stat
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import BrokenBarrierError
from typing import Any, Callable

import yaml

from agents.base import TraceAction
from agents.sandbox_runtime import (
    AgentTransportRequest,
    DockerBackend,
    FCBackend,
    SandboxBackend,
    agent_response_dict_from_transport,
    get_sandbox_backend_class,
    validate_checkpoint_backend,
)
from harness.container_image_prep import (
    ensure_fixed_image,
    ensure_source_image,
    fixed_image_name_for,
    normalize_image_reference,
    remove_image,
)
from harness.container_stats_sampler import (
    ContainerResourceRecorder,
    ContainerStatsSampler,
    summarize_samples,
)
from harness.trace_logger import TraceLogger
from trace_collect import attempt_layout
from trace_collect.mismatch import MismatchOracle
from trace_collect.output_normalize import normalize_tool_output
from trace_collect.resource_timeline import valid_resource_timeline
from trace_collect.monitoring import MonitoringMode, resolve_simulate_monitoring
from trace_collect.attempt_pipeline import (
    configure_task_container_apt_mirror,
    next_attempt_number_in,
    sanitize_path_segment,
    start_task_container,
    stop_task_container,
)
logger = logging.getLogger(__name__)
CasManifestValue = str | dict[str, Any]
CasManifestEntries = dict[str, CasManifestValue]
GLOBAL_CONTAINER_RESOURCE_SAMPLE_INTERVAL_S = 1.0
_DEFAULT_PREP_CONCURRENCY = 20
_SHARED_SEMAPHORE_POLL_S = 0.05
_REPLAY_START_DELAY_S = 0.1
_SOURCE_EXEC_TIMEOUT_REPLAY_FLOOR_S = 5.0
_PREP_ERROR_MAX_CHARS = 500
_CHECKPOINT_CAS_ROOT = os.path.expanduser("~/.cache/agent-checkpoint-cas")
_CHECKPOINT_SKIP_DIRS = frozenset({".git"})
_TRACE_REPLAY_TOOL_NAMES = frozenset({"spawn", "web_search", "web_fetch"})
_MISMATCH_ORACLE = MismatchOracle()


class SimulateError(Exception):
    """Raised when simulation encounters a fatal issue."""


class ReplayPreparationError(Exception):
    """Raised when one replay session cannot be prepared."""

    def __init__(
        self,
        *,
        loaded: "LoadedTraceSession",
        original: BaseException,
        prepared: "PreparedTraceSession | None",
    ) -> None:
        super().__init__(f"{type(original).__name__}: {original}")
        self.loaded = loaded
        self.original = original
        self.prepared = prepared


@dataclass(frozen=True, slots=True)
class TraceManifestEntry:
    """One resolved trace entry from a simulate manifest."""

    index: int
    trace: Path
    task_source: Path
    docker_image: str | None = None
    label: str | None = None
    sandbox_backend: str = "docker"
    checkpoint_backend: str = "walk"


@dataclass(frozen=True, slots=True)
class ReplayTaskStats:
    """Per-trace throughput accounting for a simulate run."""

    agent_id: str
    run_instance_id: str
    source_agent_id: str
    manifest_index: int
    label: str | None
    source_trace: str
    success: bool
    elapsed_s: float
    action_count: int
    llm_call_count: int
    tool_exec_count: int
    failed_action_count: int = 0
    replay_env_parity: str = "default_env"
    prep_error: str | None = None
    # STORY-3: scheduler metric aggregates
    total_checkpoint_exposed_ms: float = 0.0
    total_capture_elapsed_ms: float = 0.0
    total_probe_elapsed_ms: float = 0.0
    total_compare_elapsed_ms: float = 0.0
    captures_fully_absorbed: int = 0
    boundaries_total: int = 0
    overlap_fraction_avg: float = 0.0


@dataclass(frozen=True, slots=True)
class LLMTimingConfig:
    """LLM duration model for cloud replay."""

    mode: str = "source_scaled"
    ttft_ms: float | None = None
    tpot_ms: float | None = None


@dataclass(frozen=True, slots=True)
class ReplaySchedulerConfig:
    """Replay-side checkpoint scheduler configuration.

    Two scheduling modes:

    | checkpoint_scheduling | Behavior                         |
    |-----------------------|----------------------------------|
    | sync                  | Inline blocking (today's mode)   |
    | deferred              | Probe-scheduled overlap behind LLM sleep |

    Probe-only decision: after every tool execution the scheduler runs
    ``probe_changes_since`` on the backend.  ``changed`` triggers a
    checkpoint capture; ``unchanged`` (or no backend) skips.  The
    ``gate`` / ``speculative`` skip modes (formerly controlled by
    ``predictive_skip``) are removed — the whitelist-based command
    classifier was unreliable (CRAB §4 fig.4) and the probe dominates
    the signal.
    """

    checkpoint_scheduling: str = "sync"   # {"sync", "deferred"}

    def __post_init__(self) -> None:
        """Validate config values."""
        if self.checkpoint_scheduling not in ("sync", "deferred"):
            raise ValueError(
                f"checkpoint_scheduling must be 'sync' or 'deferred', "
                f"got {self.checkpoint_scheduling!r}"
            )


@dataclass(frozen=True, slots=True)
class SleepDrift:
    """Expected-vs-observed asyncio sleep timing for replay diagnostics."""

    phase: str
    expected_s: float
    actual_s: float

    @property
    def drift_s(self) -> float:
        return self.actual_s - self.expected_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "expected_s": round(self.expected_s, 6),
            "actual_s": round(self.actual_s, 6),
            "drift_s": round(self.drift_s, 6),
            "drift_ms": round(self.drift_s * 1000.0, 3),
        }


@dataclass(slots=True)
class LoadedTraceSession:
    """Resolved replay inputs for one source trace."""

    source_trace: Path
    task_source: Path
    source_agent_id: str
    run_instance_id: str
    manifest_index: int
    scaffold: str
    metadata: dict[str, Any] | None
    summary: dict[str, Any] | None
    task: dict[str, Any]
    actions: list[dict[str, Any]]
    iterations: dict[int, dict[str, Any]]
    docker_image_override: str | None = None
    label: str | None = None
    sandbox_backend: str = "docker"
    checkpoint_backend: str = "walk"

    @property
    def agent_id(self) -> str:
        return self.run_instance_id


@dataclass(frozen=True, slots=True)
class WorkerTraceInput:
    """Picklable replay input for a subprocess worker."""

    source_trace: str
    task_source: str
    manifest_index: int
    docker_image_override: str | None
    label: str | None
    run_instance_id: str
    sandbox_backend: str = "docker"
    checkpoint_backend: str = "walk"


@dataclass(frozen=True, slots=True)
class WorkerReplayResult:
    """One subprocess worker's replay outputs."""

    wave_index: int
    worker_index: int
    trace_file: str
    task_stats: list[ReplayTaskStats]
    task_output_dirs: dict[str, str]


@dataclass(slots=True)
class PreparedContainer:
    """Container prepared for trace replay."""

    container_id: str
    container_executable: str
    docker_image: str
    agent: Any  # ContainerAgent
    fixed_image: str | None = None
    cleanup_fixed_image: bool = True
    backend: SandboxBackend | None = None
    # FCBackend paired snapshots indexed by action_index for forced-sync restore.
    replay_snapshots: dict[int, Any] = dataclasses.field(default_factory=dict)


@dataclass(slots=True)
class PreparedTraceSession:
    """Container plus the loaded source-trace context."""

    loaded: LoadedTraceSession
    container: PreparedContainer | None = None
    container_resource_recorder: ContainerResourceRecorder | None = None
    sampler: ContainerStatsSampler | None = None
    task_output_dir: Path | None = None
    resources_written: bool = False
    resource_monitoring_enabled: bool = True
    memory_bandwidth_enabled: bool = True
    monitoring_policy: dict[str, object] | None = None
    runtime_artifact_root_map: dict[str, str] = dataclasses.field(default_factory=dict)
    replay_exec_env_parity: str = "default_env"
    replay_task_env_parity: str = "default_env"


@dataclass(frozen=True, slots=True)
class FailedPreparedTraceSession:
    """Preparation failure plus the task output context kept for reporting."""

    loaded: LoadedTraceSession
    prepared: PreparedTraceSession
    error: BaseException
    elapsed_s: float


class ContainerStartupRecorder:
    """Collect and persist one task's container startup facts."""

    def __init__(
        self,
        *,
        loaded: LoadedTraceSession,
        task_output_dir: Path,
        container_executable: str | None,
        network_mode: str,
        source_image: str | None,
    ) -> None:
        self.loaded = loaded
        self.task_output_dir = task_output_dir
        self.container_executable = container_executable
        self.network_mode = network_mode
        self.source_image = source_image
        self.fixed_image: str | None = None
        self.container_id: str | None = None
        self._started_monotonic = time.monotonic()
        self._started_at = _utc_now_iso()
        self._phases: list[dict[str, Any]] = []
        self._resources: dict[str, Any] = {
            "samples": [],
            "summary": summarize_samples([]),
        }
        self._written = False

    def start_phase(self, name: str) -> dict[str, Any]:
        phase = {
            "name": name,
            "started_at": _utc_now_iso(),
            "_started_monotonic": time.monotonic(),
        }
        return phase

    def finish_phase(
        self,
        phase: dict[str, Any],
        *,
        status: str = "success",
        error: BaseException | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        started_monotonic = float(phase.pop("_started_monotonic"))
        phase["ended_at"] = _utc_now_iso()
        phase["elapsed_s"] = time.monotonic() - started_monotonic
        phase["status"] = status
        if error is not None:
            phase["error"] = _exception_payload(error)
        if extra:
            phase.update(extra)
        self._phases.append(phase)

    def set_resources(self, samples: list[dict[str, Any]]) -> None:
        self._resources = {
            "samples": samples,
            "summary": summarize_samples(samples),
        }

    def write(
        self,
        *,
        status: str,
        reason: str | None = None,
        error: BaseException | None = None,
    ) -> None:
        if self._written:
            return
        payload: dict[str, Any] = {
            "status": status,
            "agent_id": self.loaded.agent_id,
            "run_instance_id": self.loaded.run_instance_id,
            "source_agent_id": self.loaded.source_agent_id,
            "task_id": self.loaded.source_agent_id,
            "manifest_index": self.loaded.manifest_index,
            "label": self.loaded.label,
            "source_trace": str(self.loaded.source_trace),
            "container_executable": self.container_executable,
            "network_mode": self.network_mode,
            "source_image": self.source_image,
            "fixed_image": self.fixed_image,
            "container_id": self.container_id,
            "started_at": self._started_at,
            "ended_at": _utc_now_iso(),
            "elapsed_s": time.monotonic() - self._started_monotonic,
            "phases": self._phases,
            "resources": self._resources,
        }
        if reason is not None:
            payload["reason"] = reason
        if error is not None:
            payload["error"] = _exception_payload(error)
        attempt_layout.write_container_startup_json(self.task_output_dir, payload)
        self._written = True


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _exception_payload(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
    }



async def _exec_tool(
    agent: Any,
    tool_name: str | None,
    tool_args_json: str,
    command_timeout_s: float,
    source_exec_timeout_s: float | None = None,
    allow_source_runtime_artifacts: bool = False,
    source_resource_timeline: dict[str, Any] | None = None,
) -> tuple[str, float, bool, dict[str, Any]]:
    """Execute one source-trace tool call via the persistent container agent.

    Returns:
        (tool_result, tool_duration_ms, tool_success, replay_metadata)
    """
    from trace_collect.openclaw_tools import execute_trace_tool_detailed

    t0 = time.monotonic()
    request_executor = None
    raw_agent = agent
    if isinstance(agent, SandboxBackend):
        backend = agent
        raw_agent = None

        async def request_executor(
            request: dict[str, Any],
            timeout_s: float | None,
        ) -> dict[str, Any]:
            response = await backend.execute(
                AgentTransportRequest(
                    tool=str(request.get("tool", "")),
                    args=dict(request.get("args") or {}),
                ),
                timeout_s=timeout_s,
            )
            return agent_response_dict_from_transport(response)

    (
        tool_result,
        tool_success,
        inner_duration_ms,
        tool_metadata,
    ) = await execute_trace_tool_detailed(
        agent=raw_agent,
        request_executor=request_executor,
        tool_name=tool_name,
        tool_args_json=tool_args_json,
        command_timeout_s=command_timeout_s,
        source_exec_timeout_s=source_exec_timeout_s,
        allow_source_runtime_artifacts=allow_source_runtime_artifacts,
        source_resource_timeline=source_resource_timeline,
    )
    wall_duration_ms = (time.monotonic() - t0) * 1000
    # Prefer agent-side timing to exclude pipe transfer overhead
    duration_ms = inner_duration_ms if inner_duration_ms is not None else wall_duration_ms
    return tool_result, duration_ms, tool_success, tool_metadata


def _unpack_exec_tool_result(
    result: tuple[Any, ...],
) -> tuple[str, float, bool, dict[str, Any]]:
    if len(result) == 3:
        tool_result, duration_ms, tool_success = result
        return str(tool_result), float(duration_ms), bool(tool_success), {}
    if len(result) == 4:
        tool_result, duration_ms, tool_success, metadata = result
        return (
            str(tool_result),
            float(duration_ms),
            bool(tool_success),
            metadata if isinstance(metadata, dict) else {},
        )
    raise ValueError(f"unexpected _exec_tool result shape: {len(result)}")


def _group_actions_by_iteration(
    actions: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Group loaded trace actions into per-iteration replay buckets."""

    iterations: dict[int, dict[str, Any]] = {}
    for action in actions:
        it = int(action.get("iteration", 0))
        if it not in iterations:
            iterations[it] = {"llms": [], "tools": []}
        if action.get("action_type") == "llm_call":
            iterations[it]["llms"].append(action)
        elif action.get("action_type") == "tool_exec":
            iterations[it]["tools"].append(action)
    return iterations


def _source_tool_success(data: dict[str, Any]) -> bool:
    raw_success = data.get("success")
    if isinstance(raw_success, bool):
        return raw_success
    if raw_success is None:
        return not bool(data.get("error"))
    raise ValueError(f"tool success must be boolean when present, got {raw_success!r}")


def _source_llm_message_payload(data: dict[str, Any]) -> dict[str, Any]:
    if "messages_delta" in data:
        return {
            "messages_delta": data.get("messages_delta"),
            "is_delta": data.get("is_delta", True),
        }
    return {"messages_in": data.get("messages_in")}


def _structured_returncode(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"structured returncode must be int, got {value!r}")
    return value


def _structured_timed_out(value: Any) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"structured timed_out must be boolean, got {value!r}")
    return value


def _command_exit_code(
    tool_result: str,
    structured_returncode: Any = None,
) -> int | None:
    returncode = _structured_returncode(structured_returncode)
    if returncode is not None:
        return returncode
    marker = "Exit code:"
    if marker not in tool_result:
        return None
    suffix = tool_result.rsplit(marker, 1)[1].strip().splitlines()[0].strip()
    if suffix == "<missing>":
        return None
    try:
        return int(suffix)
    except ValueError:
        return None


def _compute_output_diff_snippet(
    source: str,
    replay: str,
    *,
    max_lines: int = 5,
    context_lines: int = 1,
    max_line_chars: int = 240,
) -> str | None:
    """Return a bounded unified-diff-style snippet of the first raw divergence."""
    if source == replay:
        return None

    line_limit = max(0, max_line_chars)

    def truncate_line(line: str) -> str:
        if len(line) <= line_limit:
            return line
        omitted = len(line) - line_limit
        return f"{line[:line_limit]}...<truncated {omitted} chars>"

    source_lines = source.splitlines()
    replay_lines = replay.splitlines()
    total_lines = max(len(source_lines), len(replay_lines))
    for i in range(total_lines):
        source_line = source_lines[i] if i < len(source_lines) else "<missing>"
        replay_line = replay_lines[i] if i < len(replay_lines) else "<missing>"
        if source_line == replay_line:
            continue
        start = max(0, i - context_lines)
        end = min(total_lines, i + max_lines)
        snippet: list[str] = []
        for j in range(start, end):
            s = source_lines[j] if j < len(source_lines) else "<missing>"
            r = replay_lines[j] if j < len(replay_lines) else "<missing>"
            if s == r:
                snippet.append(f"  {truncate_line(s)}")
            else:
                snippet.append(f"- {truncate_line(s)}")
                snippet.append(f"+ {truncate_line(r)}")
        return "\n".join(snippet)

    return "raw output differs without line-content difference"


def _exec_semantics_payload(
    tool_name: str | None,
    tool_args_json: Any,
) -> dict[str, Any] | None:
    if isinstance(tool_args_json, str):
        try:
            parsed = json.loads(tool_args_json or "{}")
        except json.JSONDecodeError:
            return None
    elif isinstance(tool_args_json, dict):
        parsed = tool_args_json
    else:
        return None
    if not isinstance(parsed, dict):
        return None
    if tool_name == "exec" and isinstance(parsed, dict):
        return parsed.get("exec") if isinstance(parsed.get("exec"), dict) else parsed
    payload = parsed.get("exec") if isinstance(parsed.get("exec"), dict) else parsed
    return payload if isinstance(payload, dict) else None


def _tool_uses_exec_semantics(tool_name: str | None, tool_args_json: Any) -> bool:
    if tool_name == "exec":
        payload = _exec_semantics_payload(tool_name, tool_args_json)
        return payload is None or "command" in payload or "commands" in payload
    payload = _exec_semantics_payload(tool_name, tool_args_json)
    return isinstance(payload, dict) and (
        "command" in payload or "commands" in payload
    )


def _tool_uses_single_exec_command_semantics(
    tool_name: str | None,
    tool_args_json: Any,
) -> bool:
    payload = _exec_semantics_payload(tool_name, tool_args_json)
    if payload is None:
        return False
    return "command" in payload and "commands" not in payload


def _source_container_exec_env(
    metadata: dict[str, Any] | None,
) -> dict[str, str] | None:
    if metadata is None:
        return None
    raw_run_config = metadata.get("run_config")
    if raw_run_config is None:
        return None
    if not isinstance(raw_run_config, dict):
        raise ValueError("trace metadata run_config must be a dict when present")
    raw_env = raw_run_config.get("container_exec_env")
    if raw_env is None:
        return None
    if not isinstance(raw_env, dict):
        raise ValueError("run_config.container_exec_env must be a dict")

    normalized: dict[str, str] = {}
    for key in ("pythonpath", "path", "pythonuserbase", "bootstrap_site_dir"):
        value = raw_env.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(
                f"run_config.container_exec_env.{key} must be a string"
            )
        if value:
            normalized[key] = value
    return normalized


def _path_under(path: Path, root: Path) -> bool:
    resolved_path = path.expanduser().resolve()
    resolved_root = root.expanduser().resolve()
    return resolved_path == resolved_root or resolved_path.is_relative_to(
        resolved_root,
    )


def _bootstrap_site_dir_from_exec_env(
    container_exec_env: dict[str, str] | None,
) -> Path | None:
    if container_exec_env is None:
        return None
    from trace_collect.runtime.task_container import _SHARED_BOOTSTRAP_CACHE

    cache_root = _SHARED_BOOTSTRAP_CACHE.expanduser()
    raw_site_dir = container_exec_env.get("bootstrap_site_dir")
    if raw_site_dir:
        site_dir = Path(raw_site_dir).expanduser()
        return site_dir if _path_under(site_dir, cache_root) else None

    pythonpath = container_exec_env.get("pythonpath")
    if not pythonpath:
        return None
    for raw_entry in pythonpath.split(os.pathsep):
        if not raw_entry:
            continue
        entry = Path(raw_entry).expanduser()
        if _path_under(entry, cache_root):
            return entry
    return None


def _bootstrap_cache_mount_args(
    container_exec_env: dict[str, str] | None,
) -> tuple[list[str], str | None]:
    site_dir = _bootstrap_site_dir_from_exec_env(container_exec_env)
    if site_dir is None:
        return [], None

    from trace_collect.runtime.task_container import _SHARED_BOOTSTRAP_CACHE

    cache_root = _SHARED_BOOTSTRAP_CACHE.expanduser().resolve()
    if not site_dir.exists():
        logger.warning(
            "Replay source references missing task-container bootstrap site dir: %s",
            site_dir,
        )
        if not cache_root.exists():
            return [], "bootstrap_cache_missing"
        return ["-v", f"{cache_root}:{cache_root}:ro"], "bootstrap_cache_missing"
    return ["-v", f"{cache_root}:{cache_root}:ro"], None


def _container_agent_env_kwargs(
    container_exec_env: dict[str, str] | None,
) -> dict[str, str]:
    if container_exec_env is None:
        return {}
    return {
        key: container_exec_env[key]
        for key in ("pythonpath", "path", "pythonuserbase")
        if key in container_exec_env
    }


def _denied_exec_command(
    *,
    tool_name: str | None,
    tool_args_json: Any,
) -> str | None:
    payload = _exec_semantics_payload(tool_name, tool_args_json)
    if payload is None:
        return None

    commands: list[str] = []
    command = payload.get("command")
    if isinstance(command, str):
        commands.append(command)
    raw_commands = payload.get("commands")
    if isinstance(raw_commands, list):
        commands.extend(command for command in raw_commands if isinstance(command, str))

    if not commands:
        return None

    from agents.openclaw.tools.shell import EXEC_TOOL_DENY_PATTERNS

    for command_text in commands:
        lower = command_text.strip().lower()
        if any(re.search(pattern, lower) for pattern in EXEC_TOOL_DENY_PATTERNS):
            return command_text
    return None


def _source_tool_exec_metadata(data: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    returncode = data.get("returncode")
    if isinstance(returncode, int) and not isinstance(returncode, bool):
        metadata["returncode"] = returncode
    timed_out = data.get("timed_out")
    if isinstance(timed_out, bool):
        metadata["timed_out"] = timed_out
    return metadata


def _exec_normalized_output_match(
    *,
    tool_name: str | None,
    tool_args_json: Any,
    source_tool_result: Any,
    replay_tool_result: Any,
) -> bool | None:
    if not _tool_uses_exec_semantics(tool_name, tool_args_json):
        return None
    source_text = "" if source_tool_result is None else str(source_tool_result)
    replay_text = "" if replay_tool_result is None else str(replay_tool_result)
    return normalize_tool_output(source_text) == normalize_tool_output(replay_text)


def _tool_mismatch_reason(
    *,
    source_success: bool,
    tool_success: bool,
    replay_source: str,
    source_tool_result: Any,
    replay_tool_result: Any,
    tool_name: str | None,
    tool_args_json: Any,
    source_returncode: Any = None,
    replay_returncode: Any = None,
    source_timed_out: Any = None,
    replay_timed_out: Any = None,
) -> str | None:
    if replay_source == "source_artifact_unavailable":
        return "source_artifact_unavailable"
    uses_exec_semantics = _tool_uses_exec_semantics(tool_name, tool_args_json)
    source_timeout = False
    replay_timeout = False
    if uses_exec_semantics:
        source_timeout = _tool_result_indicates_wrapper_timeout(
            source_tool_result,
            structured_timed_out=source_timed_out,
        )
        replay_timeout = _tool_result_indicates_wrapper_timeout(
            replay_tool_result,
            structured_timed_out=replay_timed_out,
        )
        if source_timeout != replay_timeout:
            return "timeout_mismatch"
    if source_success != tool_success:
        return "tool_success_mismatch"
    if uses_exec_semantics:
        source_exit = _command_exit_code(
            str(source_tool_result or ""),
            source_returncode,
        )
        replay_exit = _command_exit_code(
            str(replay_tool_result or ""),
            replay_returncode,
        )
        if (
            source_exit is not None
            and replay_exit is not None
            and source_exit != replay_exit
        ):
            return "command_exit_code_mismatch"
    return None


def _command_metadata(
    *,
    tool_name: str | None,
    tool_args_json: Any,
    tool_result: str,
    tool_success: bool,
    returncode: Any = None,
) -> dict[str, Any]:
    if not _tool_uses_exec_semantics(tool_name, tool_args_json):
        return {}
    exit_code = _command_exit_code(tool_result, returncode)
    if exit_code is None:
        return {}
    return {
        "command_exit_code": exit_code,
        "command_success": exit_code == 0,
        "replay_transport_success": tool_success,
    }


def _is_replay_wrapper_timeout_result(tool_result: str) -> bool:
    timeout_markers = {
        "[timeout]",
        "[resource_timeout]",
        "[resource_stall_timeout]",
    }
    return any(line.strip() in timeout_markers for line in tool_result.splitlines())


def _tool_result_indicates_wrapper_timeout(
    tool_result: Any,
    structured_timed_out: Any = None,
) -> bool:
    timed_out = _structured_timed_out(structured_timed_out)
    if timed_out is not None:
        return timed_out
    text = str(tool_result or "")
    return (
        "Error: Command timed out after " in text
        or _is_replay_wrapper_timeout_result(text)
    )


def _source_exec_timeout_s(
    *,
    tool_name: str | None,
    tool_args_json: Any,
    source_duration_ms: float,
    source_success: bool,
    source_tool_result: Any,
    source_timed_out: Any = None,
) -> float | None:
    if source_duration_ms <= 0:
        return None
    if not _tool_uses_exec_semantics(tool_name, tool_args_json):
        return None
    timed_out = _structured_timed_out(source_timed_out)
    if timed_out is not None:
        return max(0.001, source_duration_ms / 1000.0) if timed_out else None
    if source_success:
        return None

    source_tool_result_text = str(source_tool_result or "")
    if (
        "Error: Command timed out after " not in source_tool_result_text
        and not _is_replay_wrapper_timeout_result(source_tool_result_text)
    ):
        return None
    return max(0.001, source_duration_ms / 1000.0)


def _effective_source_exec_timeout_s(
    *,
    source_exec_timeout_s: float | None,
    replay_speed: float,
) -> float | None:
    if source_exec_timeout_s is None:
        return None
    if replay_speed <= 0:
        raise ValueError("replay_speed must be > 0")
    return max(
        _SOURCE_EXEC_TIMEOUT_REPLAY_FLOOR_S,
        source_exec_timeout_s / replay_speed,
    )


def _remap_runtime_artifact_tool_args(
    *,
    tool_name: str | None,
    tool_args_json: Any,
    runtime_root_map: dict[str, str],
) -> tuple[Any, str | None, str | None, bool]:
    from trace_collect.openclaw_tools import remap_source_runtime_artifact_tool_args

    if not isinstance(tool_args_json, str):
        return tool_args_json, None, None, False
    mapped_args, source_path, mapped_path = remap_source_runtime_artifact_tool_args(
        tool_name=tool_name,
        tool_args_json=tool_args_json,
        runtime_root_map=runtime_root_map,
    )
    mapped_exists = mapped_path is not None and Path(mapped_path).is_file()
    return mapped_args, source_path, mapped_path, mapped_exists


def _artifact_unavailable_result(source_path: str) -> str:
    return (
        "Error: source trace references an OpenClaw runtime artifact "
        "that is unavailable in the simulator runtime: "
        f"{source_path}"
    )


def _container_tool_runtime_args(
    *,
    tool_name: str | None,
    tool_args_json: Any,
    runtime_root_map: dict[str, str],
) -> tuple[Any, str | None, str | None, bool]:
    mapped_tool_args, original_artifact_path, mapped_artifact_path, mapped_exists = (
        _remap_runtime_artifact_tool_args(
            tool_name=tool_name,
            tool_args_json=tool_args_json,
            runtime_root_map=runtime_root_map,
        )
    )
    if original_artifact_path is None and isinstance(tool_args_json, str):
        from trace_collect.openclaw_tools import (
            source_runtime_artifact_path_from_tool_call,
        )

        original_artifact_path = source_runtime_artifact_path_from_tool_call(
            tool_name=tool_name,
            tool_args_json=tool_args_json,
        )
    return mapped_tool_args, original_artifact_path, mapped_artifact_path, mapped_exists


async def _execute_container_tool_call(
    *,
    agent: Any,
    tool_name: str | None,
    mapped_tool_args: Any,
    command_timeout_s: float,
    source_exec_timeout: float | None,
    mapped_artifact_path: str | None,
    exec_resource_timeline: dict[str, Any] | None,
) -> tuple[str, float, bool, dict[str, Any]]:
    if exec_resource_timeline is None:
        if mapped_artifact_path is not None:
            return _unpack_exec_tool_result(
                await _exec_tool(
                    agent,
                    tool_name,
                    mapped_tool_args,
                    command_timeout_s,
                    source_exec_timeout,
                    True,
                )
            )
        return _unpack_exec_tool_result(
            await _exec_tool(
                agent,
                tool_name,
                mapped_tool_args,
                command_timeout_s,
                source_exec_timeout,
            )
        )
    if mapped_artifact_path is not None:
        return _unpack_exec_tool_result(
            await _exec_tool(
                agent,
                tool_name,
                mapped_tool_args,
                command_timeout_s,
                source_exec_timeout,
                True,
                exec_resource_timeline,
            )
        )
    return _unpack_exec_tool_result(
        await _exec_tool(
            agent,
            tool_name,
            mapped_tool_args,
            command_timeout_s,
            source_exec_timeout,
            False,
            exec_resource_timeline,
        )
    )


def _checkpoint_after_spec(
    *,
    action_data: dict[str, Any],
    source_trace: Path,
) -> dict[str, Any] | None:
    raw = action_data.get("checkpoint_after")
    if raw is None:
        return None
    if isinstance(raw, str):
        spec: dict[str, Any] = {"path": raw}
    elif isinstance(raw, dict):
        spec = dict(raw)
    else:
        return None
    raw_path = spec.get("path")
    if not raw_path:
        return None
    checkpoint_path = Path(str(raw_path))
    if not checkpoint_path.is_absolute():
        checkpoint_path = source_trace.parent / checkpoint_path
    restore_root = str(spec.get("root") or "/testbed")
    if restore_root != "/testbed":
        return None
    spec["path"] = str(checkpoint_path)
    kind = str(spec.setdefault("kind", "cas_manifest"))
    if "incremental" not in spec:
        spec["incremental"] = kind in {
            "cas_manifest_incremental",
        }
    spec["root"] = restore_root
    return spec


def _copy_checkpoint_archive_to_container(
    *,
    checkpoint_path: Path,
    container_id: str,
    container_executable: str,
    container_archive_path: str,
) -> None:
    _run_checked_container_command(
        [
            container_executable,
            "cp",
            str(checkpoint_path.resolve()),
            f"{container_id}:{container_archive_path}",
        ],
        timeout=600,
    )


def _restore_cas_manifest_in_container(
    *,
    container_id: str,
    container_executable: str,
    container_manifest_path: str,
    restore_root: str,
    clear_root: bool = True,
) -> None:
    """Restore files from a CAS manifest inside the task container.

    The manifest JSON was copied into the container at *container_manifest_path*.
    Blobs are read from the host-mounted CAS store.
    """
    script = r'''
import hashlib, json, logging, os, shutil, stat
logger = logging.getLogger("trace_collect.restore_cas_manifest")
manifest_path = os.environ["CAS_MANIFEST_PATH"]
cas_root = os.environ["CAS_ROOT"]
root = os.path.abspath(os.environ["CHECKPOINT_ROOT"])
clear_root = os.environ.get("CHECKPOINT_CLEAR_ROOT") == "1"
preserved_top_level_dirs = set(json.loads(os.environ["CHECKPOINT_PRESERVE_TOP_LEVEL_DIRS"]))
if os.path.lexists(root):
    if os.path.islink(root):
        os.unlink(root)
        os.makedirs(root, exist_ok=True)
    elif not os.path.isdir(root):
        os.unlink(root)
        os.makedirs(root, exist_ok=True)
else:
    os.makedirs(root, exist_ok=True)
root_real = os.path.realpath(root)
if root_real != root:
    raise RuntimeError(f"checkpoint root symlinks are unsupported: {root}")

with open(manifest_path, "r") as f:
    manifest = json.load(f)

entries = manifest.get("entries", {})
deleted = manifest.get("deleted_paths", [])
if not isinstance(entries, dict):
    raise RuntimeError("checkpoint manifest missing 'entries' dict")
if not isinstance(deleted, list):
    raise RuntimeError("checkpoint manifest has invalid 'deleted_paths'")

created_dirs = set()

def safe_target(relpath):
    if not isinstance(relpath, str) or relpath == "":
        raise RuntimeError(f"unsafe checkpoint path: {relpath}")
    if os.path.isabs(relpath) or ".." in relpath.split(os.sep):
        raise RuntimeError(f"unsafe checkpoint path: {relpath}")
    target = os.path.abspath(os.path.join(root, relpath))
    if target == root or not target.startswith(root + os.sep):
        raise RuntimeError(f"unsafe checkpoint path: {relpath}")
    return target

def safe_symlink_target(relpath, link_target):
    if not isinstance(link_target, str) or link_target == "":
        raise RuntimeError(f"unsafe checkpoint symlink target: {relpath}")
    link_path = safe_target(relpath)
    if os.path.isabs(link_target):
        resolved = os.path.abspath(link_target)
    else:
        resolved = os.path.abspath(os.path.join(os.path.dirname(link_path), link_target))
    if resolved != root and not resolved.startswith(root + os.sep):
        logger.debug(
            "checkpoint symlink target resolves outside restore root: %s -> %s",
            relpath,
            link_target,
        )
    return link_path

def ensure_parent_dir(target):
    parent = os.path.dirname(target)
    rel_parent = os.path.relpath(parent, root)
    current = root
    if rel_parent == ".":
        return
    for part in rel_parent.split(os.sep):
        current = os.path.join(current, part)
        if os.path.lexists(current):
            if os.path.islink(current) or not os.path.isdir(current):
                raise RuntimeError(f"unsafe checkpoint parent path: {current}")
        else:
            os.mkdir(current)
            created_dirs.add(current)

def is_preserved_relpath(relpath):
    parts = relpath.split(os.sep)
    return bool(parts) and parts[0] in preserved_top_level_dirs

for relpath in list(entries) + deleted:
    safe_target(relpath)

if clear_root:
    for name in os.listdir(root):
        if name in preserved_top_level_dirs:
            continue
        path = os.path.join(root, name)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.unlink(path)
else:
    for del_path in sorted(deleted, key=lambda p: p.count(os.sep), reverse=True):
        if is_preserved_relpath(del_path):
            continue
        target = safe_target(del_path)
        if os.path.lexists(target):
            if os.path.isdir(target) and not os.path.islink(target):
                shutil.rmtree(target)
            else:
                os.unlink(target)

file_entries = []
symlink_entries = []
for relpath, entry in entries.items():
    if is_preserved_relpath(relpath):
        continue
    if not isinstance(entry, dict):
        raise RuntimeError(f"invalid checkpoint manifest entry: {relpath}")
    entry_type = entry.get("type", "file")
    if entry_type == "symlink":
        symlink_entries.append((relpath, entry))
        continue
    if entry_type != "file":
        raise RuntimeError(f"unsupported checkpoint manifest entry type: {relpath}")
    file_entries.append((relpath, entry))

for relpath, entry in file_entries:
    hash_val = entry["hash"]
    blob_path = os.path.join(cas_root, "blobs", hash_val[:2], hash_val[2:])
    with open(blob_path, "rb") as f:
        content = f.read()
    actual_hash = hashlib.sha256(content).hexdigest()
    if actual_hash != hash_val:
        raise RuntimeError(
            f"checkpoint blob digest mismatch for {relpath}: "
            f"expected {hash_val}, got {actual_hash}"
        )
    target = safe_target(relpath)
    ensure_parent_dir(target)
    if os.path.lexists(target):
        if os.path.islink(target):
            raise RuntimeError(f"checkpoint target symlinks are unsupported: {relpath}")
        if os.path.isdir(target):
            shutil.rmtree(target)
        elif not stat.S_ISREG(os.stat(target).st_mode):
            raise RuntimeError(f"checkpoint special targets are unsupported: {relpath}")
    with open(target, "wb") as f:
        f.write(content)
    os.chmod(target, entry.get("mode", 0o644))
    mtime_ns = entry.get("mtime_ns")
    if mtime_ns is not None:
        try:
            os.utime(target, ns=(mtime_ns, mtime_ns))
        except OSError:
            pass
for relpath, entry in symlink_entries:
    link_target = entry.get("target")
    target = safe_symlink_target(relpath, link_target)
    ensure_parent_dir(target)
    if os.path.lexists(target):
        if os.path.isdir(target) and not os.path.islink(target):
            shutil.rmtree(target)
        else:
            os.unlink(target)
    os.symlink(link_target, target)
if not os.path.isdir(root):
    raise RuntimeError(f"checkpoint root missing after restore: {root}")
if os.path.exists(manifest_path):
    os.unlink(manifest_path)
'''
    _run_checked_container_command(
        [
            container_executable,
            "exec",
            "-e",
            f"CAS_MANIFEST_PATH={container_manifest_path}",
            "-e",
            "CAS_ROOT=" + _CHECKPOINT_CAS_ROOT,
            "-e",
            f"CHECKPOINT_ROOT={restore_root}",
            "-e",
            f"CHECKPOINT_CLEAR_ROOT={'1' if clear_root else '0'}",
            "-e",
            "CHECKPOINT_PRESERVE_TOP_LEVEL_DIRS="
            + json.dumps(sorted(_CHECKPOINT_SKIP_DIRS)),
            container_id,
            "python3",
            "-c",
            script,
        ],
        timeout=600,
    )


def _cas_manifest_entry_hash(entry: CasManifestValue) -> str:
    if isinstance(entry, str):
        return entry
    entry_type = entry.get("type", "file")
    if entry_type == "symlink":
        target = entry.get("target")
        if not isinstance(target, str):
            raise ValueError(f"CAS manifest symlink entry missing target: {entry!r}")
        return json.dumps({"type": "symlink", "target": target}, sort_keys=True)
    if entry_type != "file":
        raise ValueError(f"unsupported CAS manifest entry type: {entry_type!r}")
    hash_value = entry.get("hash")
    if not isinstance(hash_value, str):
        raise ValueError(f"CAS manifest entry missing string hash: {entry!r}")
    return hash_value


def _cas_manifest_entry_mode(entry: CasManifestValue) -> int | None:
    if isinstance(entry, str):
        return None
    if entry.get("type", "file") == "symlink":
        return None
    mode = entry.get("mode")
    if mode is None:
        return None
    if isinstance(mode, bool) or not isinstance(mode, int):
        raise ValueError(f"CAS manifest entry mode must be int, got {mode!r}")
    return stat.S_IMODE(mode)


def _source_cas_manifest_entry(entry: Any) -> CasManifestValue | None:
    if not isinstance(entry, dict):
        return None
    entry_type = entry.get("type", "file")
    if entry_type == "symlink":
        target = entry.get("target")
        if not isinstance(target, str):
            raise ValueError(f"CAS source symlink entry missing target: {entry!r}")
        return {"type": "symlink", "target": target}
    if entry_type != "file":
        raise ValueError(f"unsupported CAS source manifest entry type: {entry_type!r}")
    hash_value = entry.get("hash")
    if not isinstance(hash_value, str):
        return None
    mode = entry.get("mode")
    if mode is None:
        return hash_value
    if isinstance(mode, bool) or not isinstance(mode, int):
        raise ValueError(f"CAS source manifest entry mode must be int, got {mode!r}")
    return {"hash": hash_value, "mode": stat.S_IMODE(mode)}


def _capture_snapshot_manifest(
    *,
    container_id: str,
    container_executable: str,
    root: str = "/testbed",
    previous_manifest: CasManifestEntries | None = None,
) -> CasManifestEntries | None:
    """Walk /testbed inside a live container, return CAS manifest entries.

    When *previous_manifest* is provided, uses ``find -newer`` with a
    marker file inside the container to only hash files changed since
    the last snapshot. Unchanged files are inherited from the previous
    manifest. A full ``all_paths`` listing (stat only, no hash) is always
    collected for deletion detection.

    Marker protocol (race-resistant):
      1. touch temp_marker (capture start timestamp)
      2. find -newer old_marker → hash changed files
      3. walk full tree for all_paths (stat only)
      4. rename temp_marker → old_marker (atomic success signal)

    Skips .git directory. Uses same hash convention as _write_cas_manifest.
    Returns None on any error (logged as warning), ``{}`` on successful
    capture of an empty tree — caller must use ``is not None`` to
    distinguish failure from empty success.
    """
    import json as _json_module


    if previous_manifest is not None:
        script = r"""
import hashlib, json, os, stat, subprocess

root = os.environ.get("SNAPSHOT_ROOT", "/testbed")
marker = "/tmp/.cas_marker"
temp_marker = "/tmp/.cas_marker_new"
skip_dirs = set(json.loads(os.environ["SNAPSHOT_SKIP_DIRS"]))

# Record capture start timestamp
subprocess.run(["touch", temp_marker], capture_output=True)

# Find files changed since last snapshot
result = subprocess.run(
    ["find", root, "-newer", marker, "-type", "f"],
    capture_output=True, text=True, timeout=30,
)
changed = None  # None = find-not-run-or-failed -> full hash
if result.returncode == 0:
    changed = set()
    for p in result.stdout.strip().splitlines():
        if not p:
            continue
        parts = p.split(os.sep)
        if any(part in skip_dirs for part in parts):
            continue
        rel = os.path.relpath(p, root)
        changed.add(rel)
else:
    # find failed — fall back to full hash (changed stays None)
    pass

entries = {}
all_paths = []
def record_symlink(fpath, rel):
    try:
        target = os.readlink(fpath)
    except OSError:
        return
    all_paths.append(rel)
    entries[rel] = {"type": "symlink", "target": target}

for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
    dirnames[:] = [d for d in dirnames if d not in skip_dirs]
    dirnames.sort()
    filenames.sort()
    for dname in dirnames:
        dpath = os.path.join(dirpath, dname)
        rel = os.path.relpath(dpath, root)
        try:
            st = os.lstat(dpath)
        except OSError:
            continue
        if stat.S_ISLNK(st.st_mode):
            record_symlink(dpath, rel)
    for fname in filenames:
        fpath = os.path.join(dirpath, fname)
        try:
            st = os.lstat(fpath)
        except OSError:
            continue
        rel = os.path.relpath(fpath, root)
        if stat.S_ISLNK(st.st_mode):
            record_symlink(fpath, rel)
            continue
        if not stat.S_ISREG(st.st_mode):
            continue
        all_paths.append(rel)
        if changed is not None and rel not in changed:
            continue
        try:
            with open(fpath, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
        except OSError:
            continue
        entries[rel] = {"hash": digest, "mode": stat.S_IMODE(st.st_mode)}

os.rename(temp_marker, marker)
print(json.dumps({"entries": entries, "all_paths": all_paths}))
"""
    else:
        script = r"""
import hashlib, json, os, stat, subprocess

root = os.environ.get("SNAPSHOT_ROOT", "/testbed")
marker = "/tmp/.cas_marker"
temp_marker = "/tmp/.cas_marker_new"
skip_dirs = set(json.loads(os.environ["SNAPSHOT_SKIP_DIRS"]))

# Record capture start timestamp (sets baseline for next incremental)
subprocess.run(["touch", temp_marker], capture_output=True)

entries = {}
all_paths = []
def record_symlink(fpath, rel):
    try:
        target = os.readlink(fpath)
    except OSError:
        return
    all_paths.append(rel)
    entries[rel] = {"type": "symlink", "target": target}

for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
    dirnames[:] = [d for d in dirnames if d not in skip_dirs]
    dirnames.sort()
    filenames.sort()
    for dname in dirnames:
        dpath = os.path.join(dirpath, dname)
        rel = os.path.relpath(dpath, root)
        try:
            st = os.lstat(dpath)
        except OSError:
            continue
        if stat.S_ISLNK(st.st_mode):
            record_symlink(dpath, rel)
    for fname in filenames:
        fpath = os.path.join(dirpath, fname)
        try:
            st = os.lstat(fpath)
        except OSError:
            continue
        rel = os.path.relpath(fpath, root)
        if stat.S_ISLNK(st.st_mode):
            record_symlink(fpath, rel)
            continue
        if not stat.S_ISREG(st.st_mode):
            continue
        all_paths.append(rel)
        try:
            with open(fpath, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
        except OSError:
            continue
        entries[rel] = {"hash": digest, "mode": stat.S_IMODE(st.st_mode)}

os.rename(temp_marker, marker)
print(json.dumps({"entries": entries, "all_paths": all_paths}))
"""
    result = subprocess.run(
        [
            container_executable,
            "exec",
            "-i",
            "-e",
            "SNAPSHOT_SKIP_DIRS=" + json.dumps(sorted(_CHECKPOINT_SKIP_DIRS)),
            container_id,
            "python3",
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        logger.warning(
            "Snapshot manifest failed (cid=%s): %s",
            container_id[:12],
            (result.stderr or result.stdout).strip()[:200],
        )
        return None
    try:
        delta = _json_module.loads(result.stdout.strip())
    except (_json_module.JSONDecodeError, ValueError) as exc:
        logger.warning("Snapshot manifest parse error: %s", exc)
        return None

    delta_entries: CasManifestEntries = delta.get("entries", {})
    all_paths: list[str] = delta.get("all_paths", [])

    if previous_manifest is not None:
        merged = dict(previous_manifest)
        merged.update(delta_entries)
        all_paths_set = set(all_paths)
        for path in list(merged):
            if path not in all_paths_set:
                del merged[path]
        return merged

    if not delta_entries and all_paths:
        return {}
    return delta_entries


def _capture_snapshot_manifest_diagnostic(
    *,
    container_id: str,
    container_executable: str,
    root: str = "/testbed",
    previous_manifest: CasManifestEntries | None = None,
    context: str,
) -> tuple[CasManifestEntries | None, str | None]:
    try:
        return (
            _capture_snapshot_manifest(
                container_id=container_id,
                container_executable=container_executable,
                root=root,
                previous_manifest=previous_manifest,
            ),
            None,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "Snapshot manifest diagnostic capture failed during %s (cid=%s): %s",
            context,
            container_id[:12],
            error,
        )
        return None, error


def _sandbox_snapshot_entries_to_cas_entries(
    snapshot_entries: dict[str, Any],
) -> CasManifestEntries:
    cas_entries: CasManifestEntries = {}
    for relpath, entry in snapshot_entries.items():
        if not isinstance(relpath, str):
            raise ValueError(f"sandbox snapshot path must be a string: {relpath!r}")
        if isinstance(entry, str):
            cas_entries[relpath] = entry
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"sandbox snapshot entry must be a dict: {relpath}")
        entry_type = entry.get("type", "file")
        if entry_type == "symlink":
            target = entry.get("target")
            if not isinstance(target, str):
                raise ValueError(f"sandbox snapshot symlink missing target: {relpath}")
            cas_entries[relpath] = {"type": "symlink", "target": target}
            continue
        if entry_type != "file":
            raise ValueError(f"unsupported sandbox snapshot entry type: {entry_type!r}")
        hash_value = entry.get("hash")
        if not isinstance(hash_value, str):
            raise ValueError(f"sandbox snapshot file missing hash: {relpath}")
        mode = entry.get("mode")
        cas_entries[relpath] = (
            {"hash": hash_value, "mode": stat.S_IMODE(mode)}
            if isinstance(mode, int) and not isinstance(mode, bool)
            else hash_value
        )
    return cas_entries


async def _capture_replay_snapshot_manifest_diagnostic(
    *,
    container: PreparedContainer,
    root: str,
    previous_manifest: CasManifestEntries | None,
    context: str,
) -> tuple[CasManifestEntries | None, str | None]:
    if container.backend is not None:
        try:
            snapshot = await container.backend.capture_snapshot()
            entries = snapshot.disk_state.get("entries")
            if not isinstance(entries, dict):
                raise ValueError("sandbox snapshot missing entries")
            replay_entries = _sandbox_snapshot_entries_to_cas_entries(entries)
            deleted_paths = snapshot.disk_state.get("deleted_paths", [])
            if not isinstance(deleted_paths, list):
                raise ValueError("sandbox snapshot deleted_paths must be a list")
            if snapshot.disk_state.get("incremental") is True and previous_manifest is not None:
                merged_entries = dict(previous_manifest)
                for deleted_path in deleted_paths:
                    if not isinstance(deleted_path, str):
                        raise ValueError(
                            f"sandbox snapshot deleted path must be a string: {deleted_path!r}"
                        )
                    merged_entries.pop(deleted_path, None)
                merged_entries.update(replay_entries)
                replay_entries = merged_entries
            return replay_entries, None
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Snapshot manifest diagnostic capture failed during %s "
                "(backend=%s): %s",
                context,
                container.docker_image,
                error,
            )
            return None, error
    return await asyncio.to_thread(
        _capture_snapshot_manifest_diagnostic,
        container_id=container.container_id,
        container_executable=container.container_executable,
        root=root,
        previous_manifest=previous_manifest,
        context=context,
    )


def _forced_sync_verification_unavailable_fields(
    error: str | None = None,
) -> dict[str, Any]:
    verification: dict[str, Any] = {"snapshot_captured": False}
    if error is not None:
        verification["error"] = error
    return {
        "forced_sync_verified": None,
        "forced_sync_verification": verification,
    }


def _cas_manifest_comparison_fields(
    *,
    source_entries: CasManifestEntries,
    replay_entries: CasManifestEntries,
) -> dict[str, Any]:
    source_keys = set(source_entries.keys())
    replay_keys = set(replay_entries.keys())
    common = source_keys & replay_keys
    modified = [
        k
        for k in common
        if _cas_manifest_entry_hash(source_entries[k])
        != _cas_manifest_entry_hash(replay_entries[k])
    ]
    mode_mismatches: list[dict[str, Any]] = []
    mode_comparison_available = False
    for path in sorted(common):
        source_mode = _cas_manifest_entry_mode(source_entries[path])
        replay_mode = _cas_manifest_entry_mode(replay_entries[path])
        if source_mode is None or replay_mode is None:
            continue
        mode_comparison_available = True
        if source_mode != replay_mode:
            mode_mismatches.append(
                {
                    "path": path,
                    "source_mode": format(source_mode, "o"),
                    "replay_mode": format(replay_mode, "o"),
                }
            )
    added = sorted(replay_keys - source_keys)
    removed = sorted(source_keys - replay_keys)
    fields: dict[str, Any] = {
        "cas_manifest_match": (
            not modified and not removed and not added and not mode_mismatches
        ),
        "cas_source_entries": len(source_entries),
        "cas_replay_entries": len(replay_entries),
        "cas_modified_count": len(modified),
        "cas_added_count": len(added),
        "cas_removed_count": len(removed),
    }
    if modified:
        fields["cas_modified_examples"] = modified[:10]
    if mode_comparison_available:
        fields["cas_mode_mismatch_count"] = len(mode_mismatches)
        if mode_mismatches:
            fields["cas_mode_mismatch_examples"] = mode_mismatches[:10]
    return fields


async def _verify_forced_sync_restore_state(
    *,
    container: PreparedContainer,
    checkpoint_spec: dict[str, Any],
    source_entries: CasManifestEntries,
    context: str,
) -> tuple[dict[str, Any], CasManifestEntries | None]:
    replay_entries, capture_error = await _capture_replay_snapshot_manifest_diagnostic(
        container=container,
        root=str(checkpoint_spec.get("root") or "/testbed"),
        previous_manifest=None,
        context=context,
    )
    if replay_entries is None:
        return _forced_sync_verification_unavailable_fields(capture_error), None

    verification_fields = _cas_manifest_comparison_fields(
        source_entries=source_entries,
        replay_entries=replay_entries,
    )
    return (
        {
            "forced_sync_verified": verification_fields["cas_manifest_match"],
            "forced_sync_verification": verification_fields,
        },
        replay_entries,
    )


def _checkpoint_relpath_is_skipped(relpath: str) -> bool:
    return any(part in _CHECKPOINT_SKIP_DIRS for part in relpath.split("/"))


def _tool_executes_in_container(tool_name: str | None) -> bool:
    return (
        bool(tool_name)
        and tool_name != "message"
        and not str(tool_name).startswith("mcp_")
        and tool_name not in _TRACE_REPLAY_TOOL_NAMES
    )


def _fold_source_checkpoint_entries_through_action(
    *,
    actions: list[dict[str, Any]],
    target_index: int,
    source_trace: Path,
) -> CasManifestEntries:
    folded: CasManifestEntries = {}
    if target_index < 0:
        return folded
    for action in actions[: target_index + 1]:
        data = action.get("data") or {}
        checkpoint_spec = _checkpoint_after_spec(
            action_data=data,
            source_trace=source_trace,
        )
        if checkpoint_spec is None:
            continue
        folded = _fold_source_checkpoint_entries(
            checkpoint_spec=checkpoint_spec,
            prev_folded=folded,
        )
    return folded


def _fold_source_checkpoint_entries(
    *,
    checkpoint_spec: dict[str, Any],
    prev_folded: CasManifestEntries,
) -> CasManifestEntries:
    """Fold a source checkpoint spec into the accumulated entries state.

    For full checkpoints (``cas_manifest_full`` or ``filesystem_tar``),
    replaces *prev_folded* entirely. For incremental CAS manifests, updates
    *prev_folded* with the new entries and removes entries listed in
    ``deleted_paths``.
    """
    manifest_path = Path(checkpoint_spec["path"])
    if not manifest_path.is_file():
        return prev_folded

    is_incremental = _checkpoint_spec_is_incremental(checkpoint_spec)

    # filesystem_tar: treat as full replacement
    if manifest_path.suffix != ".json":
        entries = _load_source_manifest_entries(str(manifest_path))
        return entries if entries is not None else prev_folded

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return prev_folded

    raw_entries = data.get("entries", {})
    if not isinstance(raw_entries, dict):
        return prev_folded

    entries: CasManifestEntries = {}
    for rel, entry in raw_entries.items():
        if not isinstance(rel, str) or _checkpoint_relpath_is_skipped(rel):
            continue
        normalized_entry = _source_cas_manifest_entry(entry)
        if normalized_entry is not None:
            entries[rel] = normalized_entry

    if not is_incremental:
        return entries

    folded = dict(prev_folded)
    folded.update(entries)
    deleted = data.get("deleted_paths", [])
    if isinstance(deleted, list):
        for dpath in deleted:
            if not isinstance(dpath, str) or _checkpoint_relpath_is_skipped(dpath):
                continue
            folded.pop(dpath, None)
    return folded


def _load_source_manifest_entries(manifest_path: str) -> CasManifestEntries | None:
    """Load source checkpoint entries, return CAS manifest entries or None.

    Handles both CAS manifest (JSON with ``entries`` dict) and
    ``filesystem_tar`` checkpoints. For tars, extracts files and hashes them.
    """
    mpath = Path(manifest_path)
    if not mpath.is_file():
        return None

    # CAS manifest (JSON)
    if mpath.suffix == ".json":
        try:
            data = json.loads(mpath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("Failed to load source manifest %s: %s", manifest_path, exc)
            return None
        entries = data.get("entries", {})
        if not isinstance(entries, dict):
            return None
        result: CasManifestEntries = {}
        for rel, entry in entries.items():
            if not isinstance(rel, str) or _checkpoint_relpath_is_skipped(rel):
                continue
            normalized_entry = _source_cas_manifest_entry(entry)
            if normalized_entry is not None:
                result[rel] = normalized_entry
        # Remove entries for paths that were deleted between checkpoints.
        deleted = data.get("deleted_paths", [])
        if isinstance(deleted, list):
            for dpath in deleted:
                if not isinstance(dpath, str) or _checkpoint_relpath_is_skipped(dpath):
                    continue
                result.pop(dpath, None)
        return result

    # filesystem_tar — read archive, hash files
    import tarfile as _tarfile_mod
    try:
        entries: CasManifestEntries = {}
        with _tarfile_mod.open(str(mpath), "r:*") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                f = tf.extractfile(member)
                if f is None:
                    continue
                digest = hashlib.sha256(f.read()).hexdigest()
                if _checkpoint_relpath_is_skipped(member.name):
                    continue
                entries[member.name] = digest
        return entries
    except Exception as exc:
        logger.debug("Failed to read tar checkpoint %s: %s", manifest_path, exc)
        return None


def _checkpoint_restore_base_fields(
    *,
    checkpoint_path: Path,
    kind: str,
    restore_root: str,
    size_bytes: int | None = None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "checkpoint_kind": kind,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_root": restore_root,
        "restore_overhead_excluded": True,
        "forced_sync_checkpoint_kind": kind,
        "forced_sync_checkpoint": str(checkpoint_path),
        "forced_sync_root": restore_root,
    }
    if size_bytes is not None:
        fields["checkpoint_size_bytes"] = size_bytes
    return fields


def _checkpoint_restore_failed_fields(
    *,
    checkpoint_path: Path,
    kind: str,
    restore_root: str,
    status: str,
    error: str,
    started: float | None = None,
    archive_exists: bool | None = None,
    size_bytes: int | None = None,
) -> dict[str, Any]:
    fields = _checkpoint_restore_base_fields(
        checkpoint_path=checkpoint_path,
        kind=kind,
        restore_root=restore_root,
        size_bytes=size_bytes,
    )
    fields.update(
        {
            "forced_sync_success": False,
            "forced_sync_status": status,
            "forced_sync_error": error,
            "forced_sync_continued": False,
        }
    )
    if started is not None:
        elapsed_ms = round((time.monotonic() - started) * 1000, 3)
        fields["restore_elapsed_ms"] = elapsed_ms
    if archive_exists is not None:
        fields["checkpoint_archive_exists"] = archive_exists
    return fields


def _restore_checkpoint_to_container(
    *,
    checkpoint_spec: dict[str, Any],
    container: PreparedContainer,
    clear_root: bool = True,
) -> dict[str, Any]:
    checkpoint_path = Path(str(checkpoint_spec["path"]))
    kind = str(checkpoint_spec.get("kind") or "cas_manifest")
    restore_root = str(checkpoint_spec.get("root") or "/testbed")
    started = time.monotonic()
    if kind not in {"cas_manifest_full", "cas_manifest_incremental"}:
        return _checkpoint_restore_failed_fields(
            checkpoint_path=checkpoint_path,
            kind=kind,
            restore_root=restore_root,
            status="checkpoint_restore_failed",
            error=f"unsupported checkpoint kind: {kind}",
            started=started,
            archive_exists=checkpoint_path.is_file(),
        )
    if not checkpoint_path.is_file():
        return _checkpoint_restore_failed_fields(
            checkpoint_path=checkpoint_path,
            kind=kind,
            restore_root=restore_root,
            status="checkpoint_missing",
            error=f"checkpoint not found: {checkpoint_path}",
            started=started,
            archive_exists=False,
        )
    size_bytes = checkpoint_path.stat().st_size
    try:
        manifest = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return _checkpoint_restore_failed_fields(
            checkpoint_path=checkpoint_path,
            kind=kind,
            restore_root=restore_root,
            status="checkpoint_restore_failed",
            error=f"invalid checkpoint manifest: {exc}",
            started=started,
            archive_exists=True,
            size_bytes=size_bytes,
        )
    if not isinstance(manifest.get("entries"), dict):
        return _checkpoint_restore_failed_fields(
            checkpoint_path=checkpoint_path,
            kind=kind,
            restore_root=restore_root,
            status="checkpoint_restore_failed",
            error="checkpoint manifest missing 'entries' dict",
            started=started,
            archive_exists=True,
            size_bytes=size_bytes,
        )

    container_manifest_path = f"/tmp/agent_sched_manifest_{uuid.uuid4().hex}.json"
    try:
        _copy_checkpoint_archive_to_container(
            checkpoint_path=checkpoint_path,
            container_id=container.container_id,
            container_executable=container.container_executable,
            container_archive_path=container_manifest_path,
        )
        _restore_cas_manifest_in_container(
            container_id=container.container_id,
            container_executable=container.container_executable,
            container_manifest_path=container_manifest_path,
            restore_root=restore_root,
            clear_root=clear_root,
        )
    except Exception as exc:
        return _checkpoint_restore_failed_fields(
            checkpoint_path=checkpoint_path,
            kind=kind,
            restore_root=restore_root,
            status="checkpoint_restore_failed",
            error=f"{type(exc).__name__}: {exc}",
            started=started,
            archive_exists=True,
            size_bytes=size_bytes,
        )
    elapsed_ms = round((time.monotonic() - started) * 1000, 3)
    fields = _checkpoint_restore_base_fields(
        checkpoint_path=checkpoint_path,
        kind=kind,
        restore_root=restore_root,
        size_bytes=size_bytes,
    )
    fields.update(
        {
            "forced_sync_success": True,
            "forced_sync_status": "checkpoint_restored_continuation",
            "forced_sync_continued": True,
            "checkpoint_archive_exists": True,
            "restore_elapsed_ms": elapsed_ms,
            "restore_root_exists": True,
            "tar_extraction_returncode": 0,
        }
    )
    return fields


def _checkpoint_spec_is_incremental(checkpoint_spec: dict[str, Any]) -> bool:
    kind = str(checkpoint_spec.get("kind") or "cas_manifest")
    return (
        checkpoint_spec.get("incremental") is True
        or kind == "cas_manifest_incremental"
    )


def _checkpoint_chain_specs_for_action(
    *,
    actions: list[dict[str, Any]],
    target_index: int,
    source_trace: Path,
) -> list[dict[str, Any]] | None:
    if target_index < 0 or target_index >= len(actions):
        return None
    target_data = actions[target_index].get("data") or {}
    target_spec = _checkpoint_after_spec(
        action_data=target_data,
        source_trace=source_trace,
    )
    if target_spec is None:
        return None
    if not _checkpoint_spec_is_incremental(target_spec):
        return [target_spec]

    reversed_chain: list[dict[str, Any]] = []
    for index in range(target_index, -1, -1):
        data = actions[index].get("data") or {}
        checkpoint_spec = _checkpoint_after_spec(
            action_data=data,
            source_trace=source_trace,
        )
        if checkpoint_spec is None:
            continue
        reversed_chain.append(checkpoint_spec)
        if not _checkpoint_spec_is_incremental(checkpoint_spec):
            return list(reversed(reversed_chain))
    return None


def _restore_checkpoint_chain_to_container(
    *,
    checkpoint_specs: list[dict[str, Any]],
    container: PreparedContainer,
) -> dict[str, Any]:
    if not checkpoint_specs:
        return _checkpoint_restore_failed_fields(
            checkpoint_path=Path(""),
            kind="cas_manifest_incremental",
            restore_root="/testbed",
            status="checkpoint_missing",
            error="no checkpoint chain available",
            started=time.monotonic(),
        )

    started = time.monotonic()
    chain_paths = [str(Path(str(spec["path"]))) for spec in checkpoint_specs]
    chain_kinds = [str(spec.get("kind") or "cas_manifest") for spec in checkpoint_specs]
    total_size_bytes = 0
    last_result: dict[str, Any] | None = None
    for index, checkpoint_spec in enumerate(checkpoint_specs):
        restore_kwargs: dict[str, Any] = {
            "checkpoint_spec": checkpoint_spec,
            "container": container,
        }
        if index != 0:
            restore_kwargs["clear_root"] = False
        restore_result = _restore_checkpoint_to_container(**restore_kwargs)
        last_result = restore_result
        size_bytes = restore_result.get("checkpoint_size_bytes")
        if isinstance(size_bytes, int):
            total_size_bytes += size_bytes
        if restore_result.get("forced_sync_success") is not True:
            fields = dict(restore_result)
            break
    else:
        assert last_result is not None
        fields = dict(last_result)

    elapsed_ms = round((time.monotonic() - started) * 1000, 3)
    fields["restore_elapsed_ms"] = elapsed_ms
    fields["checkpoint_restore_chain_length"] = len(checkpoint_specs)
    fields["checkpoint_restore_chain_paths"] = chain_paths
    fields["checkpoint_restore_chain_kinds"] = chain_kinds
    fields["checkpoint_restore_chain_size_bytes"] = total_size_bytes
    fields["forced_sync_checkpoint_chain_length"] = len(checkpoint_specs)
    fields["forced_sync_checkpoint_chain"] = chain_paths
    return fields


async def _reapply_forced_sync_actions(
    *,
    prepared_session: PreparedTraceSession,
    start_index: int,
    end_index: int,
    replay_speed: float,
    command_timeout_s: float,
) -> dict[str, Any]:
    container = prepared_session.container
    if container is None:
        raise ValueError("cannot reapply forced-sync actions without a container")

    reapplied_action_ids: list[str] = []
    reapply_errors: list[str] = []
    for index in range(start_index, end_index + 1):
        action = prepared_session.loaded.actions[index]
        if action.get("action_type") != "tool_exec":
            continue
        data = action.get("data") or {}
        tool_name = data.get("tool_name")
        if not _tool_executes_in_container(tool_name):
            continue
        tool_args = data.get("tool_args", "{}")
        source_duration_ms = float(data.get("duration_ms") or 0.0)
        source_success = _source_tool_success(data)
        source_tool_result = data.get("tool_result", data.get("result", ""))
        source_exec_timeout = _source_exec_timeout_s(
            tool_name=tool_name,
            tool_args_json=tool_args,
            source_duration_ms=source_duration_ms,
            source_success=source_success,
            source_tool_result=source_tool_result,
            source_timed_out=data.get("timed_out"),
        )
        replay_exec_timeout = _effective_source_exec_timeout_s(
            source_exec_timeout_s=source_exec_timeout,
            replay_speed=replay_speed,
        )
        mapped_tool_args, original_artifact_path, mapped_artifact_path, mapped_exists = (
            _container_tool_runtime_args(
                tool_name=tool_name,
                tool_args_json=tool_args,
                runtime_root_map=prepared_session.runtime_artifact_root_map,
            )
        )
        if original_artifact_path is not None and not mapped_exists:
            continue
        source_resource_timeline = valid_resource_timeline(
            data.get("resource_timeline")
        )
        exec_resource_timeline = (
            source_resource_timeline
            if _tool_uses_single_exec_command_semantics(tool_name, mapped_tool_args)
            else None
        )
        action_id = str(action.get("action_id") or f"action-{index}")
        if _denied_exec_command(tool_name=tool_name, tool_args_json=tool_args):
            continue
        reapplied_action_ids.append(action_id)
        try:
            await _execute_container_tool_call(
                agent=container.backend or container.agent,
                tool_name=tool_name,
                mapped_tool_args=mapped_tool_args,
                command_timeout_s=command_timeout_s,
                source_exec_timeout=replay_exec_timeout,
                mapped_artifact_path=mapped_artifact_path,
                exec_resource_timeline=exec_resource_timeline,
            )
        except Exception as exc:
            reapply_errors.append(
                f"action_index={index} action_id={action_id}: "
                f"{type(exc).__name__}: {exc}"
            )
            break

    return {
        "forced_sync_reapplied_action_count": len(reapplied_action_ids),
        "forced_sync_reapplied_action_ids": reapplied_action_ids,
        "forced_sync_reapply_errors": reapply_errors,
    }


async def _run_deferred_forced_sync(
    *,
    prepared_session: PreparedTraceSession,
    effective_mismatch_reason: str,
    checkpoint_spec: dict[str, Any],
    checkpoint_action_index: int,
    ctr: PreparedContainer,
    scheduler: "ReplayCheckpointScheduler",
    lane_induced: bool = False,
    replay_speed: float = 1.0,
    command_timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Run forced-sync for a deferred CAS mismatch found during LLM sleep.

    Called from Hook 2 when a background boundary task completed during an
    LLM sleep window and found a CAS state mismatch.  Runs the same inline
    restore logic as the tool-result-mismatch path, but uses the stored
    boundary context rather than the current loop iteration's variables.
    """
    import time as _time

    forced_sync_fields: dict[str, Any] = {
        "forced_sync_attempted": True,
        "forced_sync_reason": effective_mismatch_reason,
        "forced_sync_overhead_excluded": True,
    }
    if lane_induced:
        forced_sync_fields["lane_induced_mismatch_candidate"] = True

    forced_sync_started = _time.monotonic()

    if isinstance(ctr.backend, FCBackend):
        fc_restore_snapshot_index: int | None = None
        available = sorted(
            idx for idx in ctr.replay_snapshots
            if idx <= checkpoint_action_index
        )
        if available:
            fc_restore_snapshot_index = available[-1]
            fc_snapshot = ctr.replay_snapshots[fc_restore_snapshot_index]
        if fc_restore_snapshot_index is not None:
            try:
                restored = await ctr.backend.restore_snapshot(fc_snapshot)
                scheduler.set_prev_manifest(None)
                restore_result = {
                    "forced_sync_success": restored,
                    "forced_sync_status": (
                        "fc_paired_restored_continuation"
                        if restored
                        else "fc_paired_restore_failed"
                    ),
                    "forced_sync_continued": restored,
                    "forced_sync_overhead_excluded": True,
                    "forced_sync_fc_snapshot_index": fc_restore_snapshot_index,
                    "forced_sync_fc_mem_version": (
                        fc_snapshot.process_state.get("mem_version")
                        if fc_snapshot.process_state
                        else None
                    ),
                    "forced_sync_fc_disk_version": (
                        fc_snapshot.disk_state.get("disk_version")
                    ),
                    "restore_root_exists": True,
                }
            except Exception as exc:
                restore_result = {
                    "forced_sync_success": False,
                    "forced_sync_status": "fc_paired_restore_failed",
                    "forced_sync_continued": False,
                    "forced_sync_error": f"{type(exc).__name__}: {exc}",
                }
        else:
            restore_result = {
                "forced_sync_success": False,
                "forced_sync_status": "fc_no_paired_snapshot",
                "forced_sync_continued": False,
                "forced_sync_error": (
                    "no FC paired snapshot available for "
                    f"checkpoint_action_index={checkpoint_action_index}"
                ),
            }
        forced_sync_fields.update(restore_result)
        fc_reapply_start = (
            fc_restore_snapshot_index + 1
            if fc_restore_snapshot_index is not None
            and fc_restore_snapshot_index < checkpoint_action_index
            else None
        )
        if (
            restore_result.get("forced_sync_success") is True
            and fc_reapply_start is not None
        ):
            reapply_fields = await _reapply_forced_sync_actions(
                prepared_session=prepared_session,
                start_index=fc_reapply_start,
                end_index=checkpoint_action_index,
                replay_speed=replay_speed,
                command_timeout_s=command_timeout_s,
            )
            forced_sync_fields.update(reapply_fields)
            if reapply_fields["forced_sync_reapply_errors"]:
                forced_sync_fields.update({
                    "forced_sync_success": False,
                    "forced_sync_continued": False,
                    "forced_sync_status": "fc_reapply_failed",
                    "forced_sync_error": "; ".join(
                        reapply_fields["forced_sync_reapply_errors"]
                    ),
                })
    else:
        checkpoint_chain = _checkpoint_chain_specs_for_action(
            actions=prepared_session.loaded.actions,
            target_index=checkpoint_action_index,
            source_trace=prepared_session.loaded.source_trace,
        )
        try:
            if checkpoint_chain is None:
                restore_result = _checkpoint_restore_failed_fields(
                    checkpoint_path=Path(str(checkpoint_spec["path"])),
                    kind=str(
                        checkpoint_spec.get("kind")
                        or "cas_manifest_incremental"
                    ),
                    restore_root=str(
                        checkpoint_spec.get("root") or "/testbed"
                    ),
                    status="checkpoint_full_missing",
                    error=(
                        "incremental checkpoint has no preceding full "
                        "checkpoint"
                    ),
                    started=_time.monotonic(),
                    archive_exists=Path(
                        str(checkpoint_spec["path"])
                    ).is_file(),
                )
            else:
                restore_result = await asyncio.to_thread(
                    _restore_checkpoint_chain_to_container,
                    checkpoint_specs=checkpoint_chain,
                    container=ctr,
                )
            forced_sync_fields.update(restore_result)
            if forced_sync_fields.get("forced_sync_success") is True:
                scheduler.set_prev_manifest(None)
        except Exception as exc:
            forced_sync_fields.update({
                "forced_sync_success": False,
                "forced_sync_continued": False,
                "forced_sync_status": "checkpoint_restore_failed",
                "forced_sync_error": f"{type(exc).__name__}: {exc}",
            })
    forced_sync_success = forced_sync_fields.get("forced_sync_success") is True
    forced_sync_fields.setdefault("forced_sync_success", False)
    forced_sync_fields.setdefault(
        "forced_sync_status",
        "checkpoint_restored_continuation"
        if forced_sync_success
        else "checkpoint_restore_failed",
    )
    forced_sync_fields.setdefault(
        "forced_sync_continued",
        forced_sync_success
        and forced_sync_fields.get("forced_sync_status")
        == "checkpoint_restored_continuation",
    )
    forced_sync_fields["forced_sync_elapsed_ms"] = round(
        (_time.monotonic() - forced_sync_started) * 1000, 3,
    )
    return forced_sync_fields


def _source_openclaw_tool_results_dir(source_trace: Path) -> Path | None:
    attempt_dir = source_trace.parent
    candidates: list[Path] = []
    manifest_path = attempt_dir / attempt_layout.RUN_MANIFEST_FILENAME
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact_value = (manifest.get("artifacts") or {}).get(
            "openclaw_tool_results_dir"
        )
        if artifact_value:
            artifact_path = Path(str(artifact_value))
            if not artifact_path.is_absolute():
                artifact_path = attempt_dir / artifact_path
            candidates.append(artifact_path)
    candidates.append(attempt_dir / "openclaw-runtime" / "tool-results")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _source_runtime_artifact_roots(actions: list[dict[str, Any]]) -> set[str]:
    from trace_collect.openclaw_tools import (
        source_runtime_artifact_path_from_tool_call,
        source_runtime_artifact_root_from_path,
    )

    roots: set[str] = set()
    for action in actions:
        if action.get("action_type") != "tool_exec":
            continue
        data = action.get("data") or {}
        tool_name = data.get("tool_name")
        tool_args = data.get("tool_args", "{}")
        if not tool_name:
            continue
        try:
            artifact_path = source_runtime_artifact_path_from_tool_call(
                tool_name=tool_name,
                tool_args_json=tool_args,
            )
        except (TypeError, ValueError):
            continue
        if artifact_path is None:
            continue
        artifact_root = source_runtime_artifact_root_from_path(artifact_path)
        if artifact_root is not None:
            roots.add(artifact_root)
    return roots


def _run_checked_container_command(cmd: list[str], *, timeout: float) -> None:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if result.returncode == 0:
        return
    message = result.stderr.strip() or result.stdout.strip()
    raise RuntimeError(
        f"container command failed ({result.returncode}): {' '.join(cmd)}"
        + (f": {message}" if message else "")
    )




def _copy_source_runtime_artifacts_to_container(
    *,
    source_dir: Path,
    container_id: str,
    container_executable: str,
    destination_dir: str,
) -> None:
    _run_checked_container_command(
        [container_executable, "exec", container_id, "mkdir", "-p", destination_dir],
        timeout=120,
    )
    _run_checked_container_command(
        [
            container_executable,
            "cp",
            f"{source_dir.resolve()}/.",
            f"{container_id}:{destination_dir}",
        ],
        timeout=600,
    )


async def _restore_source_runtime_artifacts(
    prepared: PreparedTraceSession,
) -> None:
    container = prepared.container
    if container is None or prepared.task_output_dir is None:
        return
    artifact_roots = _source_runtime_artifact_roots(prepared.loaded.actions)
    if not artifact_roots:
        return
    source_dir = _source_openclaw_tool_results_dir(prepared.loaded.source_trace)
    if source_dir is None:
        logger.warning(
            "Source trace references OpenClaw runtime artifacts but no "
            "tool-results directory is available: %s",
            prepared.loaded.source_trace,
        )
        return
    simulator_dir = prepared.task_output_dir / "openclaw-runtime" / "tool-results"
    if source_dir.resolve() != simulator_dir.resolve():
        if simulator_dir.exists():
            shutil.rmtree(simulator_dir)
        shutil.copytree(source_dir, simulator_dir)
    simulator_root = str(simulator_dir.resolve())
    for destination_dir in sorted(artifact_roots):
        await asyncio.to_thread(
            _copy_source_runtime_artifacts_to_container,
            source_dir=simulator_dir,
            container_id=container.container_id,
            container_executable=container.container_executable,
            destination_dir=simulator_root,
        )
        prepared.runtime_artifact_root_map[destination_dir] = simulator_root
    logger.info(
        "Restored OpenClaw runtime artifacts for %s into simulator runtime %s",
        prepared.loaded.agent_id,
        simulator_root,
    )


def _parse_trace_session_file(
    trace_path: Path,
) -> tuple[str, dict[str, Any] | None, list[dict[str, Any]], dict[str, Any] | None]:
    """Read one canonical trace once and extract primary plus subagent lanes."""

    metadata: dict[str, Any] | None = None
    first_agent_id: str | None = None
    all_actions: list[dict[str, Any]] = []
    actions_by_agent: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}

    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            record_type = record.get("type")
            if record_type == "trace_metadata":
                metadata = record
                continue

            agent_id = record.get("agent_id")
            if record_type == "action" and agent_id:
                agent_id = str(agent_id)
                if first_agent_id is None:
                    first_agent_id = agent_id
                all_actions.append(record)
                actions_by_agent.setdefault(agent_id, []).append(record)
                continue

            if record_type == "summary" and agent_id:
                summaries[str(agent_id)] = record

    if first_agent_id is None or not all_actions:
        raise SimulateError(f"No action records with agent_id found in {trace_path}")

    metadata_agent_id = metadata.get("instance_id") if metadata else None
    if isinstance(metadata_agent_id, str) and metadata_agent_id in actions_by_agent:
        primary_agent_id = metadata_agent_id
    elif first_agent_id and "/" in first_agent_id:
        primary_agent_id = first_agent_id.split("/", 1)[0]
        if primary_agent_id not in actions_by_agent:
            primary_agent_id = first_agent_id
    else:
        primary_agent_id = first_agent_id

    subagent_prefix = f"{primary_agent_id}/"
    actions = [
        action
        for action in all_actions
        if action.get("agent_id") == primary_agent_id
        or str(action.get("agent_id", "")).startswith(subagent_prefix)
    ]
    if not actions_by_agent.get(primary_agent_id):
        raise SimulateError(
            f"No primary action records for agent_id {primary_agent_id!r} in {trace_path}"
        )

    actions.sort(
        key=lambda action: (
            float(action.get("ts_start", 0.0)),
            float(action.get("ts_end", 0.0)),
            int(action.get("iteration", 0)),
            str(action.get("action_id", "")),
        )
    )
    return primary_agent_id, metadata, actions, summaries.get(primary_agent_id)


def _source_action_agent_id(action: dict[str, Any]) -> str:
    return str(action.get("agent_id") or "")


def _is_subagent_lane_action(
    action: dict[str, Any],
    *,
    source_agent_id: str,
) -> bool:
    return _source_action_agent_id(action).startswith(f"{source_agent_id}/")


def _replay_action_id(
    action: dict[str, Any],
    *,
    source_agent_id: str,
    fallback: str,
) -> str:
    action_id = str(action.get("action_id") or fallback)
    lane_agent_id = _source_action_agent_id(action)
    if lane_agent_id and lane_agent_id != source_agent_id:
        return f"{lane_agent_id}:{action_id}"
    return action_id


def _find_task(task_source: Path, agent_id: str) -> dict[str, Any]:
    tasks = json.loads(task_source.read_text(encoding="utf-8"))
    for task in tasks:
        if task["instance_id"] == agent_id:
            return task
    raise SimulateError(f"Task {agent_id!r} not found in {task_source}")


def _iteration_count(actions: list[dict[str, Any]]) -> int:
    return len({int(action.get("iteration", 0)) for action in actions})


def _sanitize_run_label(value: str) -> str:
    return sanitize_path_segment(value).replace(" ", "-")


def _replay_fixed_image_name(
    *,
    source_image: str,
    agent_id: str,
    task_output_dir: Path,
) -> str:
    label = _sanitize_run_label(agent_id).lower()[:64]
    digest = hashlib.sha1(str(task_output_dir.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{fixed_image_name_for(source_image)}:simulate-{label}-{digest}"


def _sweep_fixed_image_name(
    *,
    source_image: str,
    output_path: Path,
    sweep_id: str,
) -> str:
    digest_source = f"{source_image}\0{output_path.resolve()}\0{sweep_id}"
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:12]
    return f"{fixed_image_name_for(source_image)}:simulate-sweep-{digest}"


def _structured_output_subdir(
    sessions: list["LoadedTraceSession"],
    *,
    concurrency: int,
    workers: int = 1,
) -> Path:
    primary = sessions[0].metadata or {}
    benchmark = str(primary.get("benchmark") or "unknown")
    model = str(primary.get("model") or "unknown")
    scaffold = str(primary.get("scaffold") or sessions[0].scaffold or "unknown")
    for session in sessions[1:]:
        other = session.metadata or {}
        if (
            other.get("benchmark") != primary.get("benchmark")
            or other.get("model") != primary.get("model")
            or other.get("scaffold") != primary.get("scaffold")
        ):
            logger.warning(
                "Heterogeneous trace metadata in manifest — primary "
                "benchmark/model/scaffold=%s/%s/%s but %s has %s/%s/%s; "
                "using primary for output path.",
                benchmark, model, scaffold, session.agent_id,
                other.get("benchmark"), other.get("model"), other.get("scaffold"),
            )
            break
    scheduler_dir = "bounded_queue" if workers == 1 else "multi_process_workers"
    leaf = (
        f"concurrency_{concurrency}"
        if workers == 1
        else f"concurrency_{concurrency}_workers_{workers}"
    )
    return (
        Path(_sanitize_run_label(benchmark))
        / _sanitize_run_label(model)
        / _sanitize_run_label(scaffold)
        / scheduler_dir
        / leaf
    )


def _build_run_id(*, mode: str, model: str | None, concurrency: int) -> str:
    label = model if model else mode
    now = datetime.now(tz=timezone.utc)
    ts = now.strftime("%Y%m%dT%H%M%S") + f"{now.microsecond // 1000:03d}"
    return f"simulate_{_sanitize_run_label(label)}_c{concurrency}_{ts}"


def _coerce_timestamp(
    value: Any,
    *,
    field: str,
    source_trace: Path,
    action_id: str,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SimulateError(
            f"{source_trace} action {action_id!r} is missing a numeric {field}"
        ) from exc


def _coerce_action_bounds(
    action: dict[str, Any],
    *,
    source_trace: Path,
) -> tuple[float, float]:
    action_id = str(action.get("action_id", ""))
    ts_start = _coerce_timestamp(
        action.get("ts_start"),
        field="ts_start",
        source_trace=source_trace,
        action_id=action_id,
    )
    ts_end = _coerce_timestamp(
        action.get("ts_end"),
        field="ts_end",
        source_trace=source_trace,
        action_id=action_id,
    )
    return ts_start, ts_end


def _load_trace_session(
    source_trace: Path,
    task_source: Path,
    manifest_index: int,
    docker_image_override: str | None = None,
    label: str | None = None,
    sandbox_backend: str = "docker",
    checkpoint_backend: str = "walk",
) -> LoadedTraceSession:
    source_agent_id, metadata, actions, summary = _parse_trace_session_file(source_trace)
    scaffold = metadata.get("scaffold", "unknown") if metadata else "unknown"
    task = _find_task(task_source, source_agent_id)
    return LoadedTraceSession(
        source_trace=source_trace,
        task_source=task_source,
        source_agent_id=source_agent_id,
        run_instance_id=source_agent_id,
        manifest_index=manifest_index,
        scaffold=scaffold,
        metadata=metadata,
        summary=summary,
        task=task,
        actions=actions,
        iterations=_group_actions_by_iteration(actions),
        docker_image_override=docker_image_override,
        label=label,
        sandbox_backend=sandbox_backend,
        checkpoint_backend=checkpoint_backend,
    )


def _assign_replay_instance_ids(sessions: list[LoadedTraceSession]) -> None:
    source_counts: dict[str, int] = {}
    for session in sessions:
        source_counts[session.source_agent_id] = (
            source_counts.get(session.source_agent_id, 0) + 1
        )

    reserved_source_ids = set(source_counts)
    used_ids: set[str] = set()
    source_occurrences: dict[str, int] = {}

    for session in sessions:
        source_agent_id = session.source_agent_id
        if source_counts[source_agent_id] == 1:
            candidate = source_agent_id
        else:
            occurrence = source_occurrences.get(source_agent_id, 0) + 1
            source_occurrences[source_agent_id] = occurrence
            base = f"{source_agent_id}__replica-{occurrence:03d}"
            candidate = base
            if candidate in reserved_source_ids or candidate in used_ids:
                candidate = f"{base}__entry-{session.manifest_index:04d}"
                suffix = 2
                while candidate in reserved_source_ids or candidate in used_ids:
                    candidate = (
                        f"{base}__entry-{session.manifest_index:04d}-{suffix}"
                    )
                    suffix += 1

        session.run_instance_id = candidate
        used_ids.add(candidate)


def _worker_trace_input(session: LoadedTraceSession) -> WorkerTraceInput:
    return WorkerTraceInput(
        source_trace=str(session.source_trace),
        task_source=str(session.task_source),
        manifest_index=session.manifest_index,
        docker_image_override=session.docker_image_override,
        label=session.label,
        run_instance_id=session.run_instance_id,
        sandbox_backend=session.sandbox_backend,
        checkpoint_backend=session.checkpoint_backend,
    )


def _load_worker_trace_inputs(inputs: list[WorkerTraceInput]) -> list[LoadedTraceSession]:
    sessions: list[LoadedTraceSession] = []
    for entry in inputs:
        session = _load_trace_session(
            Path(entry.source_trace),
            Path(entry.task_source),
            manifest_index=entry.manifest_index,
            docker_image_override=entry.docker_image_override,
            label=entry.label,
            sandbox_backend=entry.sandbox_backend,
            checkpoint_backend=entry.checkpoint_backend,
        )
        session.run_instance_id = entry.run_instance_id
        sessions.append(session)
    return sessions


def _resolve_prep_concurrency(requested: int, num_sessions: int) -> int:
    """Resolve the system-wide concurrent container preparation limit."""
    if requested < 0:
        raise ValueError("prep_concurrency must be >= 0")
    if num_sessions < 1:
        raise ValueError("num_sessions must be >= 1")
    return min(requested or _DEFAULT_PREP_CONCURRENCY, num_sessions)


def _partition_worker_inputs(
    inputs: list[WorkerTraceInput],
    workers: int,
) -> list[list[WorkerTraceInput]]:
    """Split worker inputs into non-empty contiguous chunks without reordering."""
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if not inputs:
        raise ValueError("inputs must not be empty")
    partition_count = min(workers, len(inputs))
    chunk_size, remainder = divmod(len(inputs), partition_count)
    chunks: list[list[WorkerTraceInput]] = []
    start = 0
    for worker_index in range(partition_count):
        size = chunk_size + (1 if worker_index < remainder else 0)
        stop = start + size
        chunks.append(inputs[start:stop])
        start = stop
    return chunks


def _chunk_worker_inputs_by_concurrency(
    inputs: list[WorkerTraceInput],
    concurrency: int,
) -> list[list[WorkerTraceInput]]:
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    return [
        inputs[index : index + concurrency]
        for index in range(0, len(inputs), concurrency)
    ]


def _abort_global_replay_start(barrier: Any, start_event: Any) -> None:
    """Best-effort release of peers waiting for a failed global start."""
    try:
        barrier.abort()
    except Exception:
        logger.debug("Failed to abort replay-start barrier", exc_info=True)
    try:
        start_event.set()
    except Exception:
        logger.debug("Failed to set replay-start event", exc_info=True)


async def _wait_for_global_replay_start(
    barrier: Any,
    start_event: Any,
    start_wall_time: Any,
    *,
    coordinator: bool,
) -> float:
    """Wait until every worker is prepared and return shared monotonic time zero."""
    try:
        await asyncio.to_thread(barrier.wait)
    except BrokenBarrierError as exc:
        raise SimulateError("Global replay start was aborted") from exc

    if coordinator:
        # Give peer processes a short, fixed grace window to return from the
        # manager barrier and enter their local replay coroutines before time zero.
        start_wall_time.value = time.time() + _REPLAY_START_DELAY_S
        start_event.set()
    else:
        await asyncio.to_thread(start_event.wait)

    shared_wall_zero = float(start_wall_time.value)
    if shared_wall_zero <= 0:
        raise SimulateError("Global replay start has no valid shared time zero")
    return time.monotonic() + (shared_wall_zero - time.time())


async def _acquire_shared_semaphore(semaphore: Any) -> None:
    """Acquire a multiprocessing-manager semaphore without blocking the event loop."""
    while True:
        acquired = await asyncio.to_thread(semaphore.acquire, False)
        if acquired:
            return
        await asyncio.sleep(_SHARED_SEMAPHORE_POLL_S)


async def _sleep_until_monotonic(target_s: float) -> SleepDrift | None:
    delay_s = target_s - time.monotonic()
    if delay_s <= 0:
        return None
    return await _sleep_and_measure(delay_s, phase="worker_replay_start")


async def _sleep_and_measure(expected_s: float, *, phase: str) -> SleepDrift | None:
    if expected_s <= 0:
        return None
    start = time.monotonic()
    await asyncio.sleep(expected_s)
    actual_s = time.monotonic() - start
    return SleepDrift(phase=phase, expected_s=expected_s, actual_s=actual_s)


def _sleep_drift_metrics(
    *,
    source_gap: SleepDrift | None,
    action_sleep: SleepDrift | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if source_gap is not None:
        payload["source_gap_sleep"] = source_gap.to_dict()
    if action_sleep is not None:
        payload["action_sleep"] = action_sleep.to_dict()
    return payload


def _summarize_sleep_drifts(drifts: list[SleepDrift]) -> dict[str, Any]:
    if not drifts:
        return {
            "sample_count": 0,
            "expected_total_s": 0.0,
            "actual_total_s": 0.0,
            "drift_s": {"min": 0.0, "max": 0.0, "avg": 0.0, "p50": 0.0, "p95": 0.0},
            "by_phase": {},
        }
    drift_values = [drift.drift_s for drift in drifts]
    by_phase: dict[str, list[SleepDrift]] = {}
    for drift in drifts:
        by_phase.setdefault(drift.phase, []).append(drift)
    return {
        "sample_count": len(drifts),
        "expected_total_s": round(sum(drift.expected_s for drift in drifts), 6),
        "actual_total_s": round(sum(drift.actual_s for drift in drifts), 6),
        "drift_s": _summarize_float_values(drift_values),
        "by_phase": {
            phase: {
                "sample_count": len(items),
                "expected_total_s": round(sum(item.expected_s for item in items), 6),
                "actual_total_s": round(sum(item.actual_s for item in items), 6),
                "drift_s": _summarize_float_values([item.drift_s for item in items]),
            }
            for phase, items in sorted(by_phase.items())
        },
    }


def _check_sleep_drift_tolerance(
    sync_drift_summary: dict[str, Any],
    deferred_drift_summary: dict[str, Any],
    *,
    tolerance_factor: float = 1.5,
) -> dict[str, Any]:
    """Check that deferred-arm sleep drift is within tolerance of sync-arm.

    Rule: deferred-arm drift p95 <= tolerance_factor * sync-arm drift p95.
    A zero sync-arm p95 is treated as within tolerance (no meaningful drift
    to compare against). The A/B harness (STORY-5) uses this as a regression
    gate: drift regression in the deferred arm means boundary work leaked
    onto the event loop.

    Returns a dict with ``within_tolerance`` (bool), both p95 values, and
    the effective bound.
    """
    sync_drift_stats = sync_drift_summary.get("drift_s", {})
    deferred_drift_stats = deferred_drift_summary.get("drift_s", {})
    sync_p95 = float(sync_drift_stats.get("p95", 0.0))
    deferred_p95 = float(deferred_drift_stats.get("p95", 0.0))
    if sync_p95 <= 0.0:
        return {
            "within_tolerance": True,
            "reason": "sync_arm_no_drift",
            "sync_p95": sync_p95,
            "deferred_p95": deferred_p95,
            "bound": None,
            "tolerance_factor": tolerance_factor,
        }
    bound = tolerance_factor * sync_p95
    return {
        "within_tolerance": deferred_p95 <= bound,
        "sync_p95": round(sync_p95, 6),
        "deferred_p95": round(deferred_p95, 6),
        "bound": round(bound, 6),
        "tolerance_factor": tolerance_factor,
    }


def _summarize_float_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "avg": 0.0, "p50": 0.0, "p95": 0.0}
    sorted_values = sorted(values)
    return {
        "min": round(sorted_values[0], 6),
        "max": round(sorted_values[-1], 6),
        "avg": round(sum(sorted_values) / len(sorted_values), 6),
        "p50": round(_nearest_rank_percentile(sorted_values, 50), 6),
        "p95": round(_nearest_rank_percentile(sorted_values, 95), 6),
    }


def _nearest_rank_percentile(sorted_values: list[float], percentile: int) -> float:
    if not sorted_values:
        raise ValueError("sorted_values must not be empty")
    index = max(0, min(len(sorted_values) - 1, (percentile * len(sorted_values) + 99) // 100 - 1))
    return sorted_values[index]


def _resolve_manifest_path(
    value: Any,
    *,
    base_dir: Path,
    field: str,
    require_absolute: bool = False,
) -> Path:
    if not isinstance(value, str) or not value:
        raise SimulateError(f"manifest {field} must be a non-empty string")
    path = Path(value)
    if require_absolute and not path.is_absolute():
        raise SimulateError(f"manifest {field} must be an absolute path: {value}")
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _resolve_manifest_sandbox_backend(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SimulateError(f"manifest {field} must be a non-empty string")
    try:
        get_sandbox_backend_class(value)
    except ValueError as exc:
        raise SimulateError(f"manifest {field} is invalid: {exc}") from exc
    return value


def _resolve_manifest_checkpoint_backend(value: Any, *, field: str) -> str:
    if value is None:
        return "walk"
    if not isinstance(value, str) or not value:
        raise SimulateError(f"manifest {field} must be a non-empty string")
    try:
        return validate_checkpoint_backend(value)
    except ValueError as exc:
        raise SimulateError(f"manifest {field} is invalid: {exc}") from exc


def _load_simulate_manifest(
    manifest: Path,
    *,
    default_task_source: Path,
    default_sandbox_backend: str = "docker",
    default_checkpoint_backend: str | None = None,
) -> list[TraceManifestEntry]:
    try:
        raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SimulateError(f"Invalid simulate manifest YAML: {manifest}") from exc

    base_dir = manifest.parent
    manifest_default_task_source: Path | None = None
    manifest_default_sandbox_backend = _resolve_manifest_sandbox_backend(
        default_sandbox_backend,
        field="sandbox_backend",
    )
    manifest_default_checkpoint_backend = _resolve_manifest_checkpoint_backend(
        default_checkpoint_backend,
        field="checkpoint_backend",
    )
    raw_traces: Any

    if isinstance(raw, list):
        raw_traces = raw
    elif isinstance(raw, dict):
        allowed_manifest_keys = {"version", "defaults", "traces"}
        unknown_manifest_keys = set(raw) - allowed_manifest_keys
        if unknown_manifest_keys:
            keys = ", ".join(sorted(str(key) for key in unknown_manifest_keys))
            raise SimulateError(f"simulate manifest has unsupported top-level keys: {keys}")
        version = raw.get("version", 1)
        if version != 1:
            raise SimulateError(f"simulate manifest version must be 1, got {version!r}")
        defaults = raw.get("defaults") or {}
        if not isinstance(defaults, dict):
            raise SimulateError("simulate manifest defaults must be an object")
        unknown_default_keys = set(defaults) - {
            "task_source",
            "sandbox_backend",
            "checkpoint_backend",
        }
        if unknown_default_keys:
            keys = ", ".join(sorted(str(key) for key in unknown_default_keys))
            raise SimulateError(f"simulate manifest defaults has unsupported keys: {keys}")
        if "task_source" in defaults:
            manifest_default_task_source = _resolve_manifest_path(
                defaults["task_source"],
                base_dir=base_dir,
                field="defaults.task_source",
            )
        if "sandbox_backend" in defaults:
            manifest_default_sandbox_backend = _resolve_manifest_sandbox_backend(
                defaults["sandbox_backend"],
                field="defaults.sandbox_backend",
            )
        if "checkpoint_backend" in defaults:
            manifest_default_checkpoint_backend = _resolve_manifest_checkpoint_backend(
                defaults["checkpoint_backend"],
                field="defaults.checkpoint_backend",
            )
        raw_traces = raw.get("traces")
    else:
        raise SimulateError("simulate manifest must be a YAML list or object with traces")

    if not isinstance(raw_traces, list) or not raw_traces:
        raise SimulateError("simulate manifest traces must be a non-empty list")

    entries: list[TraceManifestEntry] = []
    for index, entry in enumerate(raw_traces):
        trace_value: Any
        task_value: Any | None = None
        docker_image: str | None = None
        label: str | None = None
        sandbox_backend = manifest_default_sandbox_backend
        checkpoint_backend = manifest_default_checkpoint_backend

        if isinstance(entry, str):
            trace_value = entry
        elif isinstance(entry, dict):
            allowed_entry_keys = {
                "trace",
                "task_source",
                "docker_image",
                "label",
                "sandbox_backend",
                "checkpoint_backend",
            }
            unknown_entry_keys = set(entry) - allowed_entry_keys
            if unknown_entry_keys:
                keys = ", ".join(sorted(str(key) for key in unknown_entry_keys))
                raise SimulateError(
                    f"simulate manifest trace entry {index} has unsupported keys: {keys}"
                )
            if "trace" not in entry:
                raise SimulateError(f"simulate manifest trace entry {index} is missing trace")
            trace_value = entry["trace"]
            task_value = entry.get("task_source")
            docker_value = entry.get("docker_image")
            label_value = entry.get("label")
            sandbox_backend_value = entry.get("sandbox_backend")
            checkpoint_backend_value = entry.get("checkpoint_backend")
            if docker_value is not None:
                if not isinstance(docker_value, str) or not docker_value:
                    raise SimulateError(
                        f"simulate manifest trace entry {index} docker_image must be a non-empty string"
                    )
                docker_image = docker_value
            if label_value is not None:
                if not isinstance(label_value, str) or not label_value:
                    raise SimulateError(
                        f"simulate manifest trace entry {index} label must be a non-empty string"
                    )
                label = label_value
            if sandbox_backend_value is not None:
                sandbox_backend = _resolve_manifest_sandbox_backend(
                    sandbox_backend_value,
                    field=f"traces[{index}].sandbox_backend",
                )
            if checkpoint_backend_value is not None:
                checkpoint_backend = _resolve_manifest_checkpoint_backend(
                    checkpoint_backend_value,
                    field=f"traces[{index}].checkpoint_backend",
                )
        else:
            raise SimulateError(
                f"simulate manifest trace entry {index} must be a string or object"
            )

        trace_path = _resolve_manifest_path(
            trace_value,
            base_dir=base_dir,
            field=f"traces[{index}].trace",
            require_absolute=True,
        )
        task_path = (
            _resolve_manifest_path(
                task_value,
                base_dir=base_dir,
                field=f"traces[{index}].task_source",
            )
            if task_value is not None
            else manifest_default_task_source or default_task_source
        )
        if not trace_path.exists():
            raise SimulateError(f"simulate manifest trace entry {index} does not exist: {trace_path}")
        if not task_path.exists():
            raise SimulateError(
                f"simulate manifest trace entry {index} task_source does not exist: {task_path}"
            )
        entries.append(
            TraceManifestEntry(
                index=index,
                trace=trace_path,
                task_source=task_path,
                docker_image=docker_image,
                label=label,
                sandbox_backend=sandbox_backend,
                checkpoint_backend=checkpoint_backend,
            )
        )
    return entries


def _resolve_docker_image(loaded: LoadedTraceSession) -> str | None:
    """Resolve docker image: manifest override > task[image_name] > task[docker_image]."""
    return (
        loaded.docker_image_override
        or loaded.task.get("image_name")
        or loaded.task.get("docker_image")
    )


def _execution_environment(loaded: LoadedTraceSession) -> str:
    metadata = loaded.metadata or {}
    value = metadata.get("execution_environment")
    if value is None or value == "":
        # Backward compat for legacy traces that predate the
        # execution_environment field: host_controller agents always ran on
        # the host, so infer "host" from agent_runtime_mode before falling
        # back to the container default.
        if metadata.get("agent_runtime_mode") == "host_controller":
            logger.warning(
                "%s has no execution_environment metadata; inferring host "
                "from agent_runtime_mode=host_controller",
                loaded.source_trace,
            )
            return "host"
        logger.warning(
            "%s has no execution_environment metadata; assuming container",
            loaded.source_trace,
        )
        return "container"
    return str(value)


def _is_host_mode(loaded: LoadedTraceSession) -> bool:
    return _execution_environment(loaded) == "host"


def _validate_loaded_sessions(
    sessions: list[LoadedTraceSession],
    *,
    mode: str,
    replay_speed: float,
    llm_timing: LLMTimingConfig,
) -> None:
    if replay_speed <= 0:
        raise ValueError("replay_speed must be > 0")
    if not sessions:
        raise SimulateError("No trace sessions were loaded")
    _validate_llm_timing_config(llm_timing)
    for session in sessions:
        get_sandbox_backend_class(session.sandbox_backend)
        validate_checkpoint_backend(session.checkpoint_backend)
        if _is_host_mode(session):
            continue
        if session.sandbox_backend == "fake":
            continue
        docker_image = _resolve_docker_image(session)
        if not docker_image:
            raise SimulateError(
                f"Task {session.source_agent_id!r} has no resolvable docker_image "
                "(set docker_image in manifest or ensure task has image_name)"
            )

    seen_run_instance_ids: set[str] = set()
    for session in sessions:
        if session.run_instance_id in seen_run_instance_ids:
            raise SimulateError(
                "Duplicate run_instance_id across replay sessions: "
                f"{session.run_instance_id!r}"
            )
        seen_run_instance_ids.add(session.run_instance_id)

        for action in session.actions:
            action_id = str(action.get("action_id", ""))
            ts_start, ts_end = _coerce_action_bounds(action, source_trace=session.source_trace)
            if ts_end < ts_start:
                raise SimulateError(
                    f"{session.source_trace} action {action_id!r} has ts_end < ts_start"
                )


def _validate_container_runtime(
    sessions: list[LoadedTraceSession],
    *,
    container_executable: str | None,
) -> None:
    container_sessions = [
        session.agent_id
        for session in sessions
        if not _is_host_mode(session) and session.sandbox_backend != "fake"
    ]
    if container_sessions and container_executable is None:
        sample = ", ".join(container_sessions[:3])
        suffix = "..." if len(container_sessions) > 3 else ""
        raise ValueError(
            "container_executable is required for container-mode traces "
            f"({sample}{suffix})"
        )


def _container_source_images(sessions: list[LoadedTraceSession]) -> list[str]:
    images: set[str] = set()
    for session in sessions:
        if _is_host_mode(session) or session.sandbox_backend == "fake":
            continue
        docker_image = _resolve_docker_image(session)
        if docker_image is None:
            continue
        images.add(normalize_image_reference(docker_image))
    return sorted(images)


def _has_container_mode_sessions(sessions: list[LoadedTraceSession]) -> bool:
    return any(
        not _is_host_mode(session) and session.sandbox_backend != "fake"
        for session in sessions
    )


async def _prefetch_container_images(
    sessions: list[LoadedTraceSession],
    *,
    container_executable: str | None,
) -> None:
    if container_executable is None:
        return
    images = _container_source_images(sessions)
    if not images:
        return
    logger.info("Prefetching %d container source image(s)", len(images))
    for image in images:
        logger.info("Prefetching container source image: %s", image)
        await asyncio.to_thread(
            ensure_source_image,
            image,
            container_executable=container_executable,
        )


async def _prebuild_sweep_fixed_images(
    sessions: list[LoadedTraceSession],
    *,
    output_path: Path,
    container_executable: str | None,
) -> dict[str, str]:
    if container_executable is None:
        return {}
    images = _container_source_images(sessions)
    if not images:
        return {}
    logger.info("Prebuilding %d sweep fixed image(s)", len(images))
    fixed_images: dict[str, str] = {}
    sweep_id = uuid.uuid4().hex
    for source_image in images:
        fixed_image_name = _sweep_fixed_image_name(
            source_image=source_image,
            output_path=output_path,
            sweep_id=sweep_id,
        )
        logger.info(
            "Prebuilding sweep fixed image: source=%s fixed=%s",
            source_image,
            fixed_image_name,
        )
        fixed_name, elapsed_s = await asyncio.to_thread(
            ensure_fixed_image,
            source_image,
            container_executable=container_executable,
            fixed_image_name=fixed_image_name,
            rebuild=True,
        )
        fixed_images[source_image] = fixed_name
        logger.info(
            "Prebuilt sweep fixed image: source=%s fixed=%s elapsed=%.3fs",
            source_image,
            fixed_name,
            elapsed_s,
        )
    return fixed_images


async def _cleanup_sweep_fixed_images(
    fixed_images: dict[str, str],
    *,
    container_executable: str | None,
) -> None:
    if container_executable is None:
        return
    cleanup_error: BaseException | None = None
    for source_image, fixed_image in fixed_images.items():
        try:
            removed = await asyncio.to_thread(
                remove_image,
                fixed_image,
                container_executable=container_executable,
            )
            if removed:
                logger.info(
                    "Removed sweep fixed image: source=%s fixed=%s",
                    source_image,
                    fixed_image,
                )
        except (Exception, asyncio.CancelledError) as exc:
            logger.exception(
                "Failed to remove sweep fixed image: source=%s fixed=%s",
                source_image,
                fixed_image,
            )
            if cleanup_error is None:
                cleanup_error = exc
    if cleanup_error is not None:
        raise cleanup_error


_FC_KERNEL_PATH = os.environ.get(
    "FC_KERNEL_PATH",
    "/tmp/fc-cache/vmlinux-5.10.225",
)


async def _prepare_fc_session(
    loaded: LoadedTraceSession,
    *,
    task_output_dir: Path,
    network_mode: str,
    container_executable: str | None,
    replay_exec_env_parity: str,
    replay_task_env_parity: str,
) -> PreparedTraceSession:
    """Prepare an FCBackend and start its Firecracker microVM.

    Falls back to FakeBackend when /dev/kvm is absent (CI environments).
    """
    docker_image = _resolve_docker_image(loaded)
    if not docker_image:
        raise SimulateError(
            f"Task {loaded.source_agent_id!r} has no resolvable docker_image"
        )
    normalized = normalize_image_reference(docker_image)
    kernel_path = Path(_FC_KERNEL_PATH)

    backend = FCBackend(
        source_image=normalized,
        kernel_path=kernel_path,
        checkpoint_dir=task_output_dir / "checkpoints",
        container_executable=container_executable or "docker",
    )
    await backend.start()

    container = PreparedContainer(
        container_id=f"fc-{loaded.agent_id}",
        container_executable=container_executable or "docker",
        docker_image=normalized,
        agent=backend,
        fixed_image=None,
        cleanup_fixed_image=False,
        backend=backend,
    )
    return PreparedTraceSession(
        loaded=loaded,
        container=container,
        replay_exec_env_parity=replay_exec_env_parity,
        replay_task_env_parity=replay_task_env_parity,
    )


async def _prepare_container_session(
    loaded: LoadedTraceSession,
    *,
    task_output_dir: Path,
    container_executable: str | None,
    network_mode: str = "host",
    fixed_images_by_source: dict[str, str] | None = None,
) -> PreparedTraceSession:
    """Prepare a sandbox backend and start its persistent replay transport."""
    container_exec_env = _source_container_exec_env(loaded.metadata)
    agent_env_kwargs = _container_agent_env_kwargs(container_exec_env)
    replay_exec_env_parity = (
        "source_env" if container_exec_env is not None else "default_env"
    )
    replay_task_env_parity = replay_exec_env_parity
    bootstrap_mount_args, bootstrap_env_status = _bootstrap_cache_mount_args(
        container_exec_env,
    )
    if bootstrap_env_status is not None:
        replay_task_env_parity = bootstrap_env_status

    backend_name = loaded.sandbox_backend or "docker"
    backend_cls = get_sandbox_backend_class(backend_name)
    if backend_name == "fake":
        backend = backend_cls()
        await backend.start()
        container = PreparedContainer(
            container_id=f"fake-{loaded.agent_id}",
            container_executable=container_executable or "fake",
            docker_image="fake",
            agent=backend,
            fixed_image=None,
            cleanup_fixed_image=False,
            backend=backend,
        )
        return PreparedTraceSession(
            loaded=loaded,
            container=container,
            replay_exec_env_parity=replay_exec_env_parity,
            replay_task_env_parity=replay_task_env_parity,
        )

    if backend_cls is not DockerBackend and backend_cls is not FCBackend:
        raise SimulateError(f"unsupported sandbox backend for simulator: {backend_name}")
    if backend_cls is FCBackend:
        return await _prepare_fc_session(
            loaded=loaded,
            task_output_dir=task_output_dir,
            network_mode=network_mode,
            container_executable=container_executable,
            replay_exec_env_parity=replay_exec_env_parity,
            replay_task_env_parity=replay_task_env_parity,
        )
    if container_executable is None:
        raise ValueError("container_executable is required for docker sandbox backend")

    docker_image = _resolve_docker_image(loaded)
    if not docker_image:
        raise SimulateError(
            f"Task {loaded.source_agent_id!r} has no resolvable docker_image"
        )
    normalized = normalize_image_reference(docker_image)
    recorder = ContainerStartupRecorder(
        loaded=loaded,
        task_output_dir=task_output_dir,
        container_executable=container_executable,
        network_mode=network_mode,
        source_image=normalized,
    )
    backend = DockerBackend(
        source_image=normalized,
        fixed_image_name=_replay_fixed_image_name(
            source_image=normalized,
            agent_id=loaded.agent_id,
            task_output_dir=task_output_dir,
        ),
        agent_id=loaded.agent_id,
        source_agent_id=loaded.source_agent_id,
        manifest_index=loaded.manifest_index,
        task_output_dir=task_output_dir,
        container_executable=container_executable,
        network_mode=network_mode,
        fixed_images_by_source=fixed_images_by_source,
        bootstrap_mount_args=bootstrap_mount_args,
        agent_env_kwargs=agent_env_kwargs,
        startup_recorder=recorder,
        ensure_fixed_image_fn=ensure_fixed_image,
        start_task_container_fn=start_task_container,
        configure_apt_mirror_fn=configure_task_container_apt_mirror,
        stop_task_container_fn=stop_task_container,
        remove_image_fn=remove_image,
        copy_checkpoint_archive_to_container_fn=_copy_checkpoint_archive_to_container,
        restore_cas_manifest_in_container_fn=_restore_cas_manifest_in_container,
        checkpoint_backend=loaded.checkpoint_backend,
    )
    await backend.start()

    container = PreparedContainer(
        container_id=backend.container_id,
        container_executable=container_executable,
        docker_image=normalized,
        agent=backend.agent,
        fixed_image=backend.fixed_image,
        cleanup_fixed_image=backend.cleanup_fixed_image,
        backend=backend,
    )
    return PreparedTraceSession(
        loaded=loaded,
        container=container,
        replay_exec_env_parity=replay_exec_env_parity,
        replay_task_env_parity=replay_task_env_parity,
    )


def _log_trace_metadata(
    *,
    trace_logger: TraceLogger,
    mode: str,
    sessions: list[LoadedTraceSession],
    replay_speed: float,
    llm_timing: LLMTimingConfig,
    manifest: Path,
    concurrency: int,
    scheduler_mode: str,
    api_base: str | None,
    model: str | None,
    network_mode: str = "host",
    extra: dict[str, Any] | None = None,
) -> None:
    scaffolds = {session.scaffold for session in sessions}
    sandbox_backends = {session.sandbox_backend for session in sessions}
    checkpoint_backends = {session.checkpoint_backend for session in sessions}
    source_models = [
        (session.summary or {}).get("model", "unknown") for session in sessions
    ]
    metadata: dict[str, Any] = {
        "scaffold": sessions[0].scaffold if len(scaffolds) == 1 else "mixed",
        "execution_environment": (
            _execution_environment(sessions[0])
            if len({_execution_environment(session) for session in sessions}) == 1
            else "mixed"
        ),
        "mode": "simulate",
        "simulate_mode": mode,
        "sandbox_backend": (
            sessions[0].sandbox_backend
            if len(sandbox_backends) == 1
            else "mixed"
        ),
        "checkpoint_backend": (
            sessions[0].checkpoint_backend
            if len(checkpoint_backends) == 1
            else "mixed"
        ),
        "replay_speed": replay_speed,
        "llm_timing_mode": llm_timing.mode,
        "source_trace_count": len(sessions),
        "source_traces": [str(session.source_trace) for session in sessions],
        "source_trace_entries": [
            {
                "manifest_index": session.manifest_index,
                "source_trace": str(session.source_trace),
                "source_agent_id": session.source_agent_id,
                "run_instance_id": session.run_instance_id,
                "label": session.label,
                "checkpoint_backend": session.checkpoint_backend,
            }
            for session in sessions
        ],
        "source_agent_ids": [session.source_agent_id for session in sessions],
        "run_instance_ids": [session.run_instance_id for session in sessions],
        "source_models": source_models,
        "manifest": str(manifest),
        "concurrency": concurrency,
        "effective_concurrency": min(concurrency, len(sessions)),
        "scheduler_mode": scheduler_mode,
        "network_mode": network_mode,
    }
    if llm_timing.mode == "ttft_tpot":
        metadata["llm_ttft_ms"] = llm_timing.ttft_ms
        metadata["llm_tpot_ms"] = llm_timing.tpot_ms
    metadata["source_model"] = (
        source_models[0] if len(set(source_models)) == 1 else "multiple"
    )
    metadata["replay_target"] = "cloud_replay"
    if extra:
        metadata.update(extra)
    trace_logger.log_metadata(**metadata)


def _make_trace_action(
    *,
    loaded: LoadedTraceSession,
    action_type: str,
    action_id: str,
    iteration: int,
    ts_start: float,
    ts_end: float,
    data: dict[str, Any],
) -> TraceAction:
    action_data = {
        **data,
        "run_instance_id": loaded.run_instance_id,
        "source_agent_id": loaded.source_agent_id,
        "manifest_index": loaded.manifest_index,
    }
    if loaded.label is not None:
        action_data["label"] = loaded.label
    return TraceAction(
        action_type=action_type,
        action_id=action_id,
        agent_id=loaded.run_instance_id,
        program_id=loaded.run_instance_id,
        instance_id=loaded.run_instance_id,
        iteration=iteration,
        ts_start=ts_start,
        ts_end=ts_end,
        data=action_data,
    )


def _make_trace_summary(
    *,
    loaded: LoadedTraceSession,
    success: bool,
    elapsed_s: float,
    source_model: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    summary = {
        "agent_id": loaded.run_instance_id,
        "run_instance_id": loaded.run_instance_id,
        "source_agent_id": loaded.source_agent_id,
        "task_id": loaded.source_agent_id,
        "manifest_index": loaded.manifest_index,
        "label": loaded.label,
        "success": success,
        "source_success": (loaded.summary or {}).get("success"),
        "n_iterations": _iteration_count(loaded.actions),
        "elapsed_s": elapsed_s,
        "source_trace": str(loaded.source_trace),
        "source_model": source_model,
    }
    summary.update(extra)
    return summary


def _make_task_stats(
    *,
    loaded: LoadedTraceSession,
    success: bool,
    elapsed_s: float,
    failed_action_count: int = 0,
    replay_env_parity: str = "default_env",
    prep_error: str | None = None,
    scheduler_metrics: dict[str, Any] | None = None,
) -> ReplayTaskStats:
    llm_call_count = sum(1 for action in loaded.actions if action.get("action_type") == "llm_call")
    tool_exec_count = sum(1 for action in loaded.actions if action.get("action_type") == "tool_exec")
    sm: dict[str, Any] = scheduler_metrics or {}
    boundaries_total = sm.get("boundaries_total", 0)
    total_work = (
        sm.get("capture_elapsed_ms", 0.0)
        + sm.get("probe_elapsed_ms", 0.0)
        + sm.get("compare_elapsed_ms", 0.0)
    )
    overlap_fraction_avg: float = 0.0
    if boundaries_total > 0 and total_work > 0:
        total_absorbed = max(0.0, total_work - sm.get("checkpoint_exposed_ms", 0.0))
        overlap_fraction_avg = round(total_absorbed / total_work, 6)
    return ReplayTaskStats(
        agent_id=loaded.run_instance_id,
        run_instance_id=loaded.run_instance_id,
        source_agent_id=loaded.source_agent_id,
        manifest_index=loaded.manifest_index,
        label=loaded.label,
        source_trace=str(loaded.source_trace),
        success=success,
        elapsed_s=elapsed_s,
        action_count=len(loaded.actions),
        llm_call_count=llm_call_count,
        tool_exec_count=tool_exec_count,
        failed_action_count=failed_action_count,
        replay_env_parity=replay_env_parity,
        prep_error=prep_error,
        total_checkpoint_exposed_ms=sm.get("checkpoint_exposed_ms", 0.0),
        total_capture_elapsed_ms=sm.get("capture_elapsed_ms", 0.0),
        total_probe_elapsed_ms=sm.get("probe_elapsed_ms", 0.0),
        total_compare_elapsed_ms=sm.get("compare_elapsed_ms", 0.0),
        captures_fully_absorbed=sm.get("captures_fully_absorbed", 0),
        boundaries_total=boundaries_total,
        overlap_fraction_avg=overlap_fraction_avg,
    )


def _prep_error_text(error: BaseException) -> str:
    if isinstance(error, ReplayPreparationError):
        error = error.original
    text = f"{type(error).__name__}: {error}"
    return text[:_PREP_ERROR_MAX_CHARS]


def _prepared_session_for_prep_error(
    *,
    loaded: LoadedTraceSession,
    error: BaseException,
    output_path: Path,
) -> PreparedTraceSession:
    if isinstance(error, ReplayPreparationError) and error.prepared is not None:
        prepared = error.prepared
    else:
        prepared = PreparedTraceSession(loaded=loaded)
    if prepared.task_output_dir is None:
        _assign_task_output_dir(prepared, output_path)
    return prepared


def _record_prep_failure(
    *,
    trace_logger: TraceLogger,
    loaded: LoadedTraceSession,
    error: BaseException,
    elapsed_s: float,
    replay_speed: float,
    llm_timing: LLMTimingConfig,
) -> ReplayTaskStats:
    prep_error = _prep_error_text(error)
    source_model = (loaded.summary or {}).get("model", "unknown")
    trace_logger.log_summary(
        loaded.agent_id,
        _make_trace_summary(
            loaded=loaded,
            success=False,
            elapsed_s=elapsed_s,
            source_model=source_model,
            extra={
                "replay_mode": "cloud_model",
                "replay_speed": replay_speed,
                "llm_timing_mode": llm_timing.mode,
                "prep_error": prep_error,
                "error": prep_error,
                "failed_actions": 0,
                "fatal_replay_errors": 0,
                "replay_action_errors": 0,
            },
        ),
    )
    return _make_task_stats(
        loaded=loaded,
        success=False,
        elapsed_s=elapsed_s,
        prep_error=prep_error,
    )


def _write_throughput_summary(
    *,
    output_path: Path,
    run_id: str,
    manifest: Path,
    mode: str,
    concurrency: int,
    scheduler_mode: str,
    llm_timing: LLMTimingConfig,
    workers: int = 1,
    prep_concurrency: int = 0,
    trace_file: Path,
    wall_time_s: float,
    task_stats: list[ReplayTaskStats],
    container_resources: dict[str, Any] | None = None,
    monitoring_policy: dict[str, object] | None = None,
) -> Path:
    attempted = len(task_stats)
    completed = sum(1 for stat in task_stats if stat.success)
    failed = attempted - completed
    effective_concurrency = min(concurrency, attempted)
    safe_wall_time_s = max(wall_time_s, 1e-9)
    payload = {
        "run_id": run_id,
        "mode": mode,
        "manifest": str(manifest),
        "trace_file": str(trace_file),
        "concurrency": concurrency,
        "effective_concurrency": effective_concurrency,
        "workers": workers,
        "effective_workers": min(workers, attempted) if attempted else 0,
        "prep_concurrency": prep_concurrency,
        "effective_prep_concurrency": (
            _resolve_prep_concurrency(prep_concurrency, attempted)
            if workers > 1 and attempted
            else None
        ),
        "scheduler_mode": scheduler_mode,
        "monitoring": monitoring_policy or {},
        "llm_timing_mode": llm_timing.mode,
        "wall_time_s": wall_time_s,
        "attempted_traces": attempted,
        "completed_traces": completed,
        "failed_traces": failed,
        "traces_per_s": attempted / safe_wall_time_s,
        "successful_traces_per_s": completed / safe_wall_time_s,
        "action_count": sum(stat.action_count for stat in task_stats),
        "llm_call_count": sum(stat.llm_call_count for stat in task_stats),
        "tool_exec_count": sum(stat.tool_exec_count for stat in task_stats),
        "tasks": [dataclasses.asdict(stat) for stat in task_stats],
    }
    if llm_timing.mode == "ttft_tpot":
        payload["llm_ttft_ms"] = llm_timing.ttft_ms
        payload["llm_tpot_ms"] = llm_timing.tpot_ms
    if container_resources is not None:
        payload["container_resources"] = {
            "status": container_resources.get("status", "collected"),
            "reason": container_resources.get("reason"),
            "jsonl_path": container_resources.get("jsonl_path"),
            "summary_path": container_resources.get("summary_path"),
            "sample_count": container_resources.get("sample_count", 0),
            "monitoring": container_resources.get("monitoring", {}),
            "sampling": container_resources.get("sampling", {}),
            "errors": container_resources.get("errors", []),
        }
    summary_path = output_path / "throughput_summary.json"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run_summary_path = output_path / f"{run_id}.throughput_summary.json"
    run_summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_path


def _assign_task_output_dir(prepared: PreparedTraceSession, output_path: Path) -> None:
    instance_dir = output_path / prepared.loaded.agent_id
    attempt_n = next_attempt_number_in(instance_dir)
    task_dir = instance_dir / f"attempt_{attempt_n}"
    task_dir.mkdir(parents=True, exist_ok=True)
    prepared.task_output_dir = task_dir


def _write_prepared_resources(
    prepared: PreparedTraceSession,
    samples: list[dict[str, Any]],
    *,
    monitoring_enabled: bool,
) -> None:
    if prepared.task_output_dir is None:
        return
    summary = summarize_samples(samples)
    if samples:
        monitoring_status = "collected"
    elif monitoring_enabled:
        monitoring_status = "enabled_no_samples"
    else:
        monitoring_status = "disabled"
    summary["monitoring_disabled"] = not monitoring_enabled
    summary["monitoring"] = {
        **(prepared.monitoring_policy or {}),
        "status": monitoring_status,
    }
    attempt_layout.write_resources_json(
        prepared.task_output_dir,
        samples,
        summary,
    )
    prepared.resources_written = True
    logger.info(
        "Wrote %d resource samples → %s",
        len(samples),
        prepared.task_output_dir / "resources.json",
    )


async def _finalize_prepared_session(prepared: PreparedTraceSession) -> None:
    resource_write_error: BaseException | None = None
    resource_samples: list[dict[str, Any]] | None = None
    resource_monitoring_enabled = prepared.resource_monitoring_enabled
    try:
        if prepared.sampler is not None:
            resource_samples = prepared.sampler.stop()
            prepared.sampler = None
        elif prepared.task_output_dir is not None and not prepared.resources_written:
            resource_samples = []
    except (Exception, asyncio.CancelledError) as exc:
        resource_write_error = exc

    ctr = prepared.container
    if ctr is None:
        if (
            resource_write_error is None
            and resource_samples is not None
            and prepared.task_output_dir is not None
            and not prepared.resources_written
        ):
            try:
                _write_prepared_resources(
                    prepared,
                    resource_samples,
                    monitoring_enabled=resource_monitoring_enabled,
                )
            except (Exception, asyncio.CancelledError) as exc:
                resource_write_error = exc
        if resource_write_error is not None:
            logger.exception(
                "Failed to write resource artifact for %s",
                prepared.loaded.agent_id,
                exc_info=(
                    type(resource_write_error),
                    resource_write_error,
                    resource_write_error.__traceback__,
                ),
            )
            raise resource_write_error
        return
    prepared.container = None
    agent_stop_error: BaseException | None = None
    container_stop_error: BaseException | None = None
    fixed_image_cleanup_error: BaseException | None = None
    container_stopped = False
    if ctr.backend is not None:
        try:
            await ctr.backend.stop()
            container_stopped = True
        except (Exception, asyncio.CancelledError) as exc:
            container_stop_error = exc
    else:
        try:
            if ctr.agent is not None:
                await ctr.agent.stop()
        except (Exception, asyncio.CancelledError) as exc:
            agent_stop_error = exc

        try:
            await asyncio.to_thread(
                stop_task_container,
                ctr.container_id,
                executable=ctr.container_executable,
            )
            container_stopped = True
        except (Exception, asyncio.CancelledError) as exc:
            container_stop_error = exc

    if container_stopped and prepared.container_resource_recorder is not None:
        prepared.container_resource_recorder.unregister_container(ctr.container_id)
        prepared.container_resource_recorder = None

    if (
        ctr.backend is None
        and container_stopped
        and ctr.fixed_image
        and ctr.cleanup_fixed_image
    ):
        try:
            removed_fixed = await asyncio.to_thread(
                remove_image,
                ctr.fixed_image,
                container_executable=ctr.container_executable,
            )
            if removed_fixed:
                logger.info(
                    "Removed fixed replay image for %s: %s",
                    prepared.loaded.agent_id,
                    ctr.fixed_image,
                )
        except (Exception, asyncio.CancelledError) as exc:
            fixed_image_cleanup_error = exc

    if (
        resource_write_error is None
        and resource_samples is not None
        and prepared.task_output_dir is not None
        and not prepared.resources_written
    ):
        try:
            _write_prepared_resources(
                prepared,
                resource_samples,
                monitoring_enabled=resource_monitoring_enabled,
            )
        except (Exception, asyncio.CancelledError) as exc:
            resource_write_error = exc

    if container_stop_error is not None and agent_stop_error is not None:
        logger.exception(
            "Failed to stop task container after agent stop failure for %s",
            prepared.loaded.agent_id,
            exc_info=(
                type(container_stop_error),
                container_stop_error,
                container_stop_error.__traceback__,
            ),
        )
    if fixed_image_cleanup_error is not None:
        logger.exception(
            "Failed to remove fixed replay image for %s",
            prepared.loaded.agent_id,
            exc_info=(
                type(fixed_image_cleanup_error),
                fixed_image_cleanup_error,
                fixed_image_cleanup_error.__traceback__,
            ),
        )
    if resource_write_error is not None:
        logger.exception(
            "Failed to write resource artifact for %s",
            prepared.loaded.agent_id,
            exc_info=(
                type(resource_write_error),
                resource_write_error,
                resource_write_error.__traceback__,
            ),
        )
    if resource_write_error is not None:
        raise resource_write_error
    if agent_stop_error is not None:
        raise agent_stop_error
    if container_stop_error is not None:
        raise container_stop_error
    if fixed_image_cleanup_error is not None:
        raise fixed_image_cleanup_error



def _source_action_excluded_overhead_s(action: dict[str, Any]) -> float:
    data = action.get("data") or {}
    checkpoint_after = data.get("checkpoint_after")
    if not isinstance(checkpoint_after, dict):
        checkpoint_after = data.get("checkpoint_after_error")
    if not isinstance(checkpoint_after, dict):
        return 0.0
    if checkpoint_after.get("overhead_excluded") is not True:
        return 0.0
    try:
        elapsed_ms = float(checkpoint_after.get("elapsed_ms") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, elapsed_ms / 1000.0)


async def _sleep_source_gap(
    *,
    previous_source_end: float | None,
    action_source_start: float,
    replay_speed: float,
) -> SleepDrift | None:
    if previous_source_end is None:
        return None
    gap_s = max(0.0, action_source_start - previous_source_end)
    return await _sleep_and_measure(gap_s / replay_speed, phase="source_gap")


def _coerce_completion_tokens(value: Any) -> int:
    if value is None or value == "":
        return 0
    tokens = int(value)
    if tokens < 0:
        raise ValueError(f"completion_tokens must be non-negative, got {value!r}")
    return tokens


def _validate_llm_timing_config(config: LLMTimingConfig) -> None:
    if config.mode not in {"source_scaled", "ttft_tpot"}:
        raise ValueError(f"Unsupported llm_timing_mode: {config.mode}")
    if config.mode == "source_scaled":
        return
    if config.ttft_ms is None:
        raise ValueError("llm_ttft_ms is required when llm_timing_mode='ttft_tpot'")
    if config.tpot_ms is None:
        raise ValueError("llm_tpot_ms is required when llm_timing_mode='ttft_tpot'")
    if config.ttft_ms < 0:
        raise ValueError("llm_ttft_ms must be non-negative")
    if config.tpot_ms < 0:
        raise ValueError("llm_tpot_ms must be non-negative")


def _llm_replay_duration_s(
    *,
    data: dict[str, Any],
    source_duration_s: float,
    replay_speed: float,
    timing: LLMTimingConfig,
) -> tuple[float, dict[str, Any]]:
    if timing.mode == "source_scaled":
        return source_duration_s / replay_speed, {
            "llm_timing_mode": "source_scaled",
        }

    completion_tokens = _coerce_completion_tokens(data.get("completion_tokens", 0))
    assert timing.ttft_ms is not None
    assert timing.tpot_ms is not None
    simulated_ms = timing.ttft_ms + max(0, completion_tokens - 1) * timing.tpot_ms
    return simulated_ms / 1000.0, {
        "llm_timing_mode": "ttft_tpot",
        "simulated_ttft_ms": timing.ttft_ms,
        "simulated_tpot_ms": timing.tpot_ms,
        "simulated_llm_latency_ms": simulated_ms,
        "source_ttft_ms": data.get("ttft_ms"),
        "source_tpot_ms": data.get("tpot_ms"),
    }


async def _prepare_replay_session(
    loaded: LoadedTraceSession,
    *,
    output_path: Path,
    container_executable: str | None,
    network_mode: str,
    container_resource_recorder: ContainerResourceRecorder | None = None,
    fixed_images_by_source: dict[str, str] | None = None,
    resource_monitoring_enabled: bool = True,
    memory_bandwidth_enabled: bool = True,
    monitoring_policy: dict[str, object] | None = None,
) -> PreparedTraceSession:
    prepared: PreparedTraceSession | None = None
    session_resource_monitoring_enabled = (
        resource_monitoring_enabled and not _is_host_mode(loaded)
    )
    try:
        prepared = PreparedTraceSession(
            loaded=loaded,
            resource_monitoring_enabled=session_resource_monitoring_enabled,
            memory_bandwidth_enabled=memory_bandwidth_enabled,
            monitoring_policy=monitoring_policy,
        )
        _assign_task_output_dir(prepared, output_path)
        assert prepared.task_output_dir is not None
        task_output_dir = prepared.task_output_dir
        if _is_host_mode(loaded):
            recorder = ContainerStartupRecorder(
                loaded=loaded,
                task_output_dir=task_output_dir,
                container_executable=container_executable,
                network_mode=network_mode,
                source_image=None,
            )
            recorder.write(
                status="skipped",
                reason="host_execution_environment",
            )
        else:
            if loaded.sandbox_backend != "fake" and container_executable is None:
                raise ValueError("container_executable is required for container-mode traces")
            prepare_kwargs: dict[str, Any] = {}
            if fixed_images_by_source:
                prepare_kwargs["fixed_images_by_source"] = fixed_images_by_source
            prepared = await _prepare_container_session(
                loaded,
                task_output_dir=task_output_dir,
                container_executable=container_executable,
                network_mode=network_mode,
                **prepare_kwargs,
            )
            prepared.task_output_dir = task_output_dir
            if loaded.sandbox_backend != "fake":
                await _restore_source_runtime_artifacts(prepared)
        if prepared.container is not None:
            sandbox_monitoring_enabled = (
                session_resource_monitoring_enabled
                and loaded.sandbox_backend != "fake"
            )
            prepared.resource_monitoring_enabled = sandbox_monitoring_enabled
            prepared.memory_bandwidth_enabled = memory_bandwidth_enabled
            prepared.monitoring_policy = monitoring_policy
            prepared.container_resource_recorder = (
                container_resource_recorder if sandbox_monitoring_enabled else None
            )
            if container_resource_recorder is not None and sandbox_monitoring_enabled:
                container_resource_recorder.register_container(
                    prepared.container.container_id
                )
            if sandbox_monitoring_enabled:
                sampler = ContainerStatsSampler(
                    container_id=prepared.container.container_id,
                    interval_s=1.0,
                    executable=prepared.container.container_executable,
                    enable_memory_bandwidth=memory_bandwidth_enabled,
                )
                sampler.start()
                prepared.sampler = sampler
        return prepared
    except (Exception, asyncio.CancelledError) as exc:
        if prepared is not None:
            await _finalize_prepared_session(prepared)
        if isinstance(exc, asyncio.CancelledError):
            raise
        raise ReplayPreparationError(
            loaded=loaded,
            original=exc,
            prepared=prepared,
        ) from exc


async def _run_cloud_model_queue(
    loaded_sessions: list[LoadedTraceSession],
    *,
    output_path: Path,
    trace_logger: TraceLogger,
    concurrency: int,
    container_executable: str | None,
    network_mode: str,
    container_resource_recorder: ContainerResourceRecorder | None,
    replay_speed: float,
    llm_timing: LLMTimingConfig,
    command_timeout_s: float,
    warmup_skip_iterations: int,
    fixed_images_by_source: dict[str, str] | None = None,
    resource_monitoring_enabled: bool = True,
    memory_bandwidth_enabled: bool = True,
    monitoring_policy: dict[str, object] | None = None,
    replay_scheduler_config: ReplaySchedulerConfig | None = None,
) -> tuple[list[PreparedTraceSession], list[ReplayTaskStats]]:
    if replay_scheduler_config is None:
        replay_scheduler_config = ReplaySchedulerConfig()
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    queue: asyncio.Queue[LoadedTraceSession] = asyncio.Queue()
    for loaded in loaded_sessions:
        queue.put_nowait(loaded)

    prepared_sessions: list[PreparedTraceSession] = []
    task_stats: list[ReplayTaskStats] = []
    result_lock = asyncio.Lock()
    worker_count = min(concurrency, len(loaded_sessions))
    first_error: BaseException | None = None

    async def worker(worker_index: int) -> None:
        nonlocal first_error
        while True:
            if first_error is not None:
                return
            try:
                loaded = queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            prepared: PreparedTraceSession | None = None
            stats: ReplayTaskStats | None = None
            try:
                task_started = time.monotonic()
                logger.info(
                    "Worker %d replaying %s (%d queued)",
                    worker_index,
                    loaded.agent_id,
                    queue.qsize(),
                )
                try:
                    prepared = await _prepare_replay_session(
                        loaded,
                        output_path=output_path,
                        container_executable=container_executable,
                        network_mode=network_mode,
                        container_resource_recorder=container_resource_recorder,
                        fixed_images_by_source=fixed_images_by_source,
                        resource_monitoring_enabled=resource_monitoring_enabled,
                        memory_bandwidth_enabled=memory_bandwidth_enabled,
                        monitoring_policy=monitoring_policy,
                    )
                except Exception as exc:
                    failed_prepared = _prepared_session_for_prep_error(
                        loaded=loaded,
                        error=exc,
                        output_path=output_path,
                    )
                    stats = _record_prep_failure(
                        trace_logger=trace_logger,
                        loaded=loaded,
                        error=exc,
                        elapsed_s=time.monotonic() - task_started,
                        replay_speed=replay_speed,
                        llm_timing=llm_timing,
                    )
                    async with result_lock:
                        prepared_sessions.append(failed_prepared)
                        task_stats.append(stats)
                    continue
                stats = await _replay_cloud_model_session(
                    prepared,
                    trace_logger=trace_logger,
                    replay_speed=replay_speed,
                    llm_timing=llm_timing,
                    command_timeout_s=command_timeout_s,
                    warmup_skip_iterations=warmup_skip_iterations,
                    replay_scheduler_config=replay_scheduler_config,
                )
            except Exception as exc:
                async with result_lock:
                    if first_error is None:
                        first_error = exc
            finally:
                if prepared is not None:
                    await _finalize_prepared_session(prepared)
                    async with result_lock:
                        prepared_sessions.append(prepared)
                        if stats is not None:
                            task_stats.append(stats)
                queue.task_done()

    worker_results = await asyncio.gather(
        *(worker(index) for index in range(worker_count)),
        return_exceptions=True,
    )
    for result in worker_results:
        if isinstance(result, asyncio.CancelledError):
            raise result
        if isinstance(result, Exception) and first_error is None:
            first_error = result
    if first_error is not None:
        raise first_error
    prep_failure_count = sum(1 for stat in task_stats if stat.prep_error is not None)
    if prep_failure_count:
        logger.warning(
            "Simulate continued after %d/%d preparation failure(s)",
            prep_failure_count,
            len(loaded_sessions),
        )
    return prepared_sessions, task_stats


async def _prepare_replay_session_with_shared_limit(
    loaded: LoadedTraceSession,
    *,
    output_path: Path,
    container_executable: str | None,
    network_mode: str,
    prep_semaphore: Any,
    fixed_images_by_source: dict[str, str] | None,
    resource_monitoring_enabled: bool,
    memory_bandwidth_enabled: bool,
    monitoring_policy: dict[str, object] | None,
) -> PreparedTraceSession:
    await _acquire_shared_semaphore(prep_semaphore)
    try:
        return await _prepare_replay_session(
            loaded,
            output_path=output_path,
            container_executable=container_executable,
            network_mode=network_mode,
            container_resource_recorder=None,
            fixed_images_by_source=fixed_images_by_source,
            resource_monitoring_enabled=resource_monitoring_enabled,
            memory_bandwidth_enabled=memory_bandwidth_enabled,
            monitoring_policy=monitoring_policy,
        )
    finally:
        prep_semaphore.release()


async def _run_prepared_cloud_model_sessions(
    prepared_sessions: list[PreparedTraceSession],
    *,
    trace_logger: TraceLogger,
    replay_zero_monotonic: float,
    replay_speed: float,
    llm_timing: LLMTimingConfig,
    command_timeout_s: float,
    warmup_skip_iterations: int,
    replay_scheduler_config: ReplaySchedulerConfig | None = None,
) -> list[ReplayTaskStats]:
    if replay_scheduler_config is None:
        replay_scheduler_config = ReplaySchedulerConfig()
    results = await asyncio.gather(
        *(
            _replay_cloud_model_session(
                prepared,
                trace_logger=trace_logger,
                replay_zero_monotonic=replay_zero_monotonic,
                replay_speed=replay_speed,
                llm_timing=llm_timing,
                command_timeout_s=command_timeout_s,
                warmup_skip_iterations=warmup_skip_iterations,
                replay_scheduler_config=replay_scheduler_config,
            )
            for prepared in prepared_sessions
        ),
        return_exceptions=True,
    )
    failures = [result for result in results if isinstance(result, BaseException)]
    if failures:
        for failure in failures:
            logger.error("Worker replay session failed: %s", failure)
        raise SimulateError(
            f"{len(failures)}/{len(results)} worker replay sessions failed"
        ) from failures[0]
    return [result for result in results if isinstance(result, ReplayTaskStats)]


async def _run_worker_wave_async(
    *,
    worker_inputs: list[WorkerTraceInput],
    output_path: Path,
    worker_run_id: str,
    global_run_id: str,
    global_concurrency: int,
    wave_index: int,
    worker_index: int,
    worker_count: int,
    container_executable: str | None,
    network_mode: str,
    replay_speed: float,
    llm_timing: LLMTimingConfig,
    command_timeout_s: float,
    warmup_skip_iterations: int,
    fixed_images_by_source: dict[str, str] | None,
    resource_monitoring_enabled: bool,
    memory_bandwidth_enabled: bool,
    monitoring_policy: dict[str, object] | None,
    prep_semaphore: Any,
    replay_start_barrier: Any,
    replay_start_event: Any,
    replay_start_wall_time: Any,
    replay_scheduler_config: ReplaySchedulerConfig | None = None,
) -> WorkerReplayResult:
    if replay_scheduler_config is None:
        replay_scheduler_config = ReplaySchedulerConfig()
    loaded_sessions = _load_worker_trace_inputs(worker_inputs)
    prepared_sessions: list[PreparedTraceSession] = []
    trace_logger: TraceLogger | None = None
    replay_started = False
    try:
        logger.info(
            "Worker %d/%d wave %d preparing %d session(s)",
            worker_index + 1,
            worker_count,
            wave_index,
            len(loaded_sessions),
        )

        async def prepare_one(
            loaded: LoadedTraceSession,
        ) -> PreparedTraceSession | FailedPreparedTraceSession:
            started = time.monotonic()
            try:
                return await _prepare_replay_session_with_shared_limit(
                    loaded,
                    output_path=output_path,
                    container_executable=container_executable,
                    network_mode=network_mode,
                    prep_semaphore=prep_semaphore,
                    fixed_images_by_source=fixed_images_by_source,
                    resource_monitoring_enabled=resource_monitoring_enabled,
                    memory_bandwidth_enabled=memory_bandwidth_enabled,
                    monitoring_policy=monitoring_policy,
                )
            except Exception as exc:
                return FailedPreparedTraceSession(
                    loaded=loaded,
                    prepared=_prepared_session_for_prep_error(
                        loaded=loaded,
                        error=exc,
                        output_path=output_path,
                    ),
                    error=exc,
                    elapsed_s=time.monotonic() - started,
                )

        prep_results = await asyncio.gather(
            *(prepare_one(loaded) for loaded in loaded_sessions),
            return_exceptions=True,
        )
        prep_failures: list[FailedPreparedTraceSession] = []
        for result in prep_results:
            if isinstance(result, BaseException):
                raise result
            if isinstance(result, FailedPreparedTraceSession):
                prep_failures.append(result)
            else:
                prepared_sessions.append(result)
        if prep_failures:
            logger.warning(
                "Worker %d/%d wave %d continuing after %d/%d preparation failure(s)",
                worker_index + 1,
                worker_count,
                wave_index,
                len(prep_failures),
                len(prep_results),
            )
        worker_trace_path = output_path / f"{worker_run_id}.jsonl"
        if worker_trace_path.exists():
            worker_trace_path.unlink()
        trace_logger = TraceLogger(output_path, worker_run_id)
        _log_trace_metadata(
            trace_logger=trace_logger,
            mode="cloud_model",
            sessions=loaded_sessions,
            replay_speed=replay_speed,
            llm_timing=llm_timing,
            manifest=Path("<worker>"),
            concurrency=global_concurrency,
            scheduler_mode="multi_process_workers",
            api_base=None,
            model=None,
            network_mode=network_mode,
            extra={
                "global_run_id": global_run_id,
                "worker_run_id": worker_run_id,
                "wave_index": wave_index,
                "worker_index": worker_index,
                "worker_count": worker_count,
                "worker_chunk_size": len(worker_inputs),
                "replay_start_delay_s": _REPLAY_START_DELAY_S,
                "monitoring": monitoring_policy or {},
                "checkpoint_scheduling": replay_scheduler_config.checkpoint_scheduling,
            },
        )
        task_stats = [
            _record_prep_failure(
                trace_logger=trace_logger,
                loaded=failure.loaded,
                error=failure.error,
                elapsed_s=failure.elapsed_s,
                replay_speed=replay_speed,
                llm_timing=llm_timing,
            )
            for failure in prep_failures
        ]
        replay_zero_monotonic = await _wait_for_global_replay_start(
            replay_start_barrier,
            replay_start_event,
            replay_start_wall_time,
            coordinator=worker_index == 0,
        )
        replay_started = True
        task_stats.extend(
            await _run_prepared_cloud_model_sessions(
                prepared_sessions,
                trace_logger=trace_logger,
                replay_zero_monotonic=replay_zero_monotonic,
                replay_speed=replay_speed,
                llm_timing=llm_timing,
                command_timeout_s=command_timeout_s,
                warmup_skip_iterations=warmup_skip_iterations,
                replay_scheduler_config=replay_scheduler_config,
            )
        )
        task_stats.sort(key=lambda stat: stat.manifest_index)
        trace_logger.close()
        task_output_sessions = [
            *prepared_sessions,
            *(failure.prepared for failure in prep_failures),
        ]
        return WorkerReplayResult(
            wave_index=wave_index,
            worker_index=worker_index,
            trace_file=str(trace_logger.path),
            task_stats=task_stats,
            task_output_dirs={
                session.loaded.run_instance_id: str(session.task_output_dir)
                for session in task_output_sessions
                if session.task_output_dir is not None
            },
        )
    except BaseException:
        if not replay_started:
            _abort_global_replay_start(replay_start_barrier, replay_start_event)
        raise
    finally:
        if trace_logger is not None:
            trace_logger.close()
        for prepared in prepared_sessions:
            await _finalize_prepared_session(prepared)


def _run_worker_wave_sync(
    *,
    worker_inputs: list[WorkerTraceInput],
    output_path: str,
    worker_run_id: str,
    global_run_id: str,
    global_concurrency: int,
    wave_index: int,
    worker_index: int,
    worker_count: int,
    container_executable: str | None,
    network_mode: str,
    replay_speed: float,
    llm_timing: LLMTimingConfig,
    command_timeout_s: float,
    warmup_skip_iterations: int,
    fixed_images_by_source: dict[str, str] | None,
    resource_monitoring_enabled: bool,
    memory_bandwidth_enabled: bool,
    monitoring_policy: dict[str, object] | None,
    prep_semaphore: Any,
    replay_start_barrier: Any,
    replay_start_event: Any,
    replay_start_wall_time: Any,
    replay_scheduler_config: ReplaySchedulerConfig | None = None,
) -> WorkerReplayResult:
    if replay_scheduler_config is None:
        replay_scheduler_config = ReplaySchedulerConfig()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return asyncio.run(
        _run_worker_wave_async(
            worker_inputs=worker_inputs,
            output_path=Path(output_path),
            worker_run_id=worker_run_id,
            global_run_id=global_run_id,
            global_concurrency=global_concurrency,
            wave_index=wave_index,
            worker_index=worker_index,
            worker_count=worker_count,
            container_executable=container_executable,
            network_mode=network_mode,
            replay_speed=replay_speed,
            llm_timing=llm_timing,
            command_timeout_s=command_timeout_s,
            warmup_skip_iterations=warmup_skip_iterations,
            fixed_images_by_source=fixed_images_by_source,
            resource_monitoring_enabled=resource_monitoring_enabled,
            memory_bandwidth_enabled=memory_bandwidth_enabled,
            monitoring_policy=monitoring_policy,
            prep_semaphore=prep_semaphore,
            replay_start_barrier=replay_start_barrier,
            replay_start_event=replay_start_event,
            replay_start_wall_time=replay_start_wall_time,
            replay_scheduler_config=replay_scheduler_config,
        )
    )


async def _run_cloud_model_worker_waves(
    worker_inputs: list[WorkerTraceInput],
    *,
    output_path: Path,
    run_id: str,
    concurrency: int,
    workers: int,
    prep_concurrency: int,
    container_executable: str | None,
    network_mode: str,
    replay_speed: float,
    llm_timing: LLMTimingConfig,
    command_timeout_s: float,
    warmup_skip_iterations: int,
    fixed_images_by_source: dict[str, str] | None,
    resource_monitoring_enabled: bool,
    memory_bandwidth_enabled: bool,
    monitoring_policy: dict[str, object] | None,
    replay_scheduler_config: ReplaySchedulerConfig | None = None,
) -> tuple[list[WorkerReplayResult], list[ReplayTaskStats]]:
    if replay_scheduler_config is None:
        replay_scheduler_config = ReplaySchedulerConfig()
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    prep_limit = _resolve_prep_concurrency(prep_concurrency, len(worker_inputs))
    wave_inputs = _chunk_worker_inputs_by_concurrency(worker_inputs, concurrency)
    replay_results: list[WorkerReplayResult] = []
    task_stats: list[ReplayTaskStats] = []

    loop = asyncio.get_running_loop()
    with multiprocessing.Manager() as sync_manager:
        prep_semaphore = sync_manager.Semaphore(prep_limit)
        for wave_index, wave in enumerate(wave_inputs):
            chunks = _partition_worker_inputs(wave, workers)
            worker_count = len(chunks)
            replay_start_barrier = sync_manager.Barrier(worker_count)
            replay_start_event = sync_manager.Event()
            replay_start_wall_time = sync_manager.Value("d", 0.0)
            logger.info(
                "Starting simulate wave %d/%d: sessions=%d workers=%d prep_limit=%d",
                wave_index + 1,
                len(wave_inputs),
                len(wave),
                worker_count,
                prep_limit,
            )
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    loop.run_in_executor(
                        executor,
                        functools.partial(
                            _run_worker_wave_sync,
                            worker_inputs=chunk,
                            output_path=str(output_path),
                            worker_run_id=(
                                f"{run_id}.wave_{wave_index:04d}.worker_{worker_index:04d}"
                            ),
                            global_run_id=run_id,
                            global_concurrency=concurrency,
                            wave_index=wave_index,
                            worker_index=worker_index,
                            worker_count=worker_count,
                            container_executable=container_executable,
                            network_mode=network_mode,
                            replay_speed=replay_speed,
                            llm_timing=llm_timing,
                            command_timeout_s=command_timeout_s,
                            warmup_skip_iterations=warmup_skip_iterations,
                            fixed_images_by_source=fixed_images_by_source,
                            resource_monitoring_enabled=resource_monitoring_enabled,
                            memory_bandwidth_enabled=memory_bandwidth_enabled,
                            monitoring_policy=monitoring_policy,
                            prep_semaphore=prep_semaphore,
                            replay_start_barrier=replay_start_barrier,
                            replay_start_event=replay_start_event,
                            replay_start_wall_time=replay_start_wall_time,
                            replay_scheduler_config=replay_scheduler_config,
                        ),
                    )
                    for worker_index, chunk in enumerate(chunks)
                ]
                wave_results = await asyncio.gather(*futures)
            replay_results.extend(sorted(wave_results, key=lambda item: item.worker_index))
            for result in sorted(wave_results, key=lambda item: item.worker_index):
                task_stats.extend(result.task_stats)
    task_stats.sort(key=lambda stat: stat.manifest_index)
    replay_results.sort(key=lambda item: (item.wave_index, item.worker_index))
    prep_failure_count = sum(1 for stat in task_stats if stat.prep_error is not None)
    if prep_failure_count:
        logger.warning(
            "Simulate worker waves continued after %d/%d preparation failure(s)",
            prep_failure_count,
            len(worker_inputs),
        )
    return replay_results, task_stats


class ReplayCheckpointScheduler:
    """Replay-side checkpoint work scheduler with deferred capture/compare.

    States: IDLE -> DECIDING -> CAPTURING -> COMPARING -> {IDLE|RESTORING} -> IDLE

    Invariant: at most one boundary's work in flight (single lane, not configurable).
    Container mutation requires serialization, and prev_cas_manifest incremental
    chaining requires captures to resolve in boundary order.
    """

    def __init__(
        self,
        config: ReplaySchedulerConfig,
        container: PreparedContainer | None,
        source_trace: Path,
        log_action: Callable[[str, dict[str, Any]], None],
    ):
        self._state = "IDLE"
        self._config = config
        self._container = container
        self._source_trace = source_trace
        self._log_action = log_action  # (agent_id, record) -> None
        self._in_flight_task: asyncio.Task | None = None
        self._deferred_records: list[dict[str, Any]] = []
        self._pending_boundary_record: dict[str, Any] | None = None
        self._prev_cas_manifest: CasManifestEntries | None = None
        self._prev_timestamp_ns: int | None = None
        self._folded_source_entries: CasManifestEntries = {}
        # Per-boundary metric accumulators
        self._pending_boundary_agent_id: str | None = None
        self._boundary_probe_elapsed_ms: float = 0.0
        self._boundary_capture_elapsed_ms: float = 0.0
        self._boundary_compare_elapsed_ms: float = 0.0
        self._boundary_exposed_ms: float = 0.0
        self._boundary_scheduler_overhead_ms: float = 0.0
        self._boundary_cas_compare_source: str = ""
        # Pending forced-sync context (preserved across iterations for deferred
        # CAS mismatches resolved at the next Hook 2 gate).
        self._pending_action_index: int | None = None
        self._pending_cas_spec: dict[str, Any] | None = None
        self._pending_lane_active: bool = False
        self._pending_tool_mismatch_reason: str | None = None
        self._pending_iteration: int = 0
        self._pending_warmup: bool = False
        # Session-aggregate accumulators
        self._capture_elapsed_ms = 0.0
        self._probe_elapsed_ms = 0.0
        self._compare_elapsed_ms = 0.0
        self._exposed_ms = 0.0
        self._scheduler_overhead_ms = 0.0
        self._boundaries_total = 0
        self._captures_fully_absorbed = 0
        # LLM sleep window accumulators (note_window)
        self._window_total_s = 0.0
        self._window_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def on_boundary(
        self,
        action_index: int,
        cas_spec: dict[str, Any] | None,
        tool_name: str | None,
        tool_args_json: str | None,
        tool_mismatch_reason: str | None,
        record_slot: dict[str, Any],
        agent_id: str,
        lane_active_at_parent: bool = False,
        iteration: int = 0,
        warmup: bool = False,
    ) -> None:
        """Called when a checkpoint boundary is reached.

        Drains in-flight work first (per Critic condition 1: drain FIRST, then
        check pending_forced_sync), then starts new probe/capture chain.

        In sync mode, runs the capture/compare inline so drain is a no-op
        and record_slot is fully populated by the time this returns.
        """
        self._boundaries_total += 1

        # Drain in-flight work before starting new boundary.
        # The wait time is charged to the in-flight boundary's
        # _boundary_exposed_ms (which has not been reset yet).
        if self._state != "IDLE" and self._in_flight_task is not None and not self._in_flight_task.done():
            exposed = await self._drain_internal()
            self._boundary_exposed_ms += exposed
        # Handle the back-to-back boundary case where the previous boundary's
        # task completed but its record was never flushed (gate-less path:
        # denied-command, artifact-unavailable, trace-replayed tools).  Flush
        # the pending record now so the new boundary starts with a clean
        # slot.
        if self._state == "IDLE" and self._pending_boundary_record is not None:
            self._finalize_and_accumulate()
            self.flush_pending()

        # Reset per-boundary accumulators for the new boundary
        self._boundary_probe_elapsed_ms = 0.0
        self._boundary_capture_elapsed_ms = 0.0
        self._boundary_compare_elapsed_ms = 0.0
        self._boundary_exposed_ms = 0.0
        self._boundary_scheduler_overhead_ms = 0.0
        self._boundary_cas_compare_source = ""

        # Fold source entries for this boundary.
        # Runs manifest JSON reads via asyncio.to_thread so filesystem I/O
        # never blocks the event loop (Critic condition 2).
        if cas_spec is not None:
            self._folded_source_entries = await asyncio.to_thread(
                _fold_source_checkpoint_entries,
                checkpoint_spec=cas_spec,
                prev_folded=self._folded_source_entries,
            )

        # Scheduling overhead: time spent on bookkeeping after drain + fold
        sched_t0 = time.monotonic()

        if self._scheduling == "sync":
            self._state = "COMPARING"
            work_t0 = time.monotonic()
            await self._run_capture_compare_inline(
                action_index=action_index,
                cas_spec=cas_spec,
                record_slot=record_slot,
                tool_name=tool_name,
                tool_args_json=tool_args_json,
            )
            work_elapsed_ms = (time.monotonic() - work_t0) * 1000
            # Sync mode: all work runs inline, no separate probe or compare phase
            self._boundary_capture_elapsed_ms = work_elapsed_ms
            self._boundary_exposed_ms = work_elapsed_ms
            self._boundary_cas_compare_source = record_slot.get("cas_compare_source", "")
            self._boundary_scheduler_overhead_ms = (time.monotonic() - sched_t0) * 1000
            self._finalize_boundary_record(record_slot)
            self._state = "IDLE"
            return

        # Deferred mode: start probe/capture as background task
        self._state = "DECIDING"
        self._pending_boundary_record = record_slot
        self._pending_boundary_agent_id = agent_id
        # Store forced-sync context for deferred CAS-mismatch resolution
        self._pending_action_index = action_index
        self._pending_cas_spec = cas_spec
        self._pending_lane_active = lane_active_at_parent
        self._pending_tool_mismatch_reason = tool_mismatch_reason
        self._pending_iteration = iteration
        self._pending_warmup = warmup
        self._in_flight_task = asyncio.create_task(
            self._boundary_work(
                action_index=action_index,
                cas_spec=cas_spec,
                tool_name=tool_name,
                tool_args_json=tool_args_json,
                tool_mismatch_reason=tool_mismatch_reason,
                record_slot=record_slot,
                lane_active_at_parent=lane_active_at_parent,
                iteration=iteration,
                warmup=warmup,
            )
        )
        self._boundary_scheduler_overhead_ms += (time.monotonic() - sched_t0) * 1000

    async def drain(self) -> float:
        """Wait for in-flight work, measure exposed_ms. Returns exposed_ms.

        After the in-flight task resolves, calls _finalize_boundary_record to
        compute per-boundary metrics (including checkpoint_exposed_ms) and
        accumulate session totals.  The record stays pending until
        flush_pending() is called — the caller may need to run forced-sync
        first when pending_forced_sync is True.
        """
        if self._in_flight_task is None:
            return 0.0
        # If the task already completed (state is IDLE) there is nothing to
        # wait for, but the pending record may still need finalization.
        if self._state == "IDLE" and self._in_flight_task.done():
            self._finalize_and_accumulate()
            return 0.0
        if self._state == "IDLE":
            return 0.0
        t0 = time.monotonic()
        exposed = await self._drain_internal()
        self._boundary_exposed_ms += exposed
        drain_overhead_ms = (time.monotonic() - t0) * 1000
        self._boundary_scheduler_overhead_ms += drain_overhead_ms
        # Finalize the pending record now that exposed_ms has been charged.
        self._finalize_and_accumulate()
        return exposed

    def log_or_defer(self, agent_id: str, record: dict[str, Any]) -> None:
        """Log immediately unless a boundary record is pending."""
        if self._pending_boundary_record is not None and self._state not in ("IDLE",):
            self._deferred_records.append(record)
        else:
            self._log_action(agent_id, record)

    def note_window(self, sleep_s: float) -> None:
        """Record that an LLM sleep window occurred (metric purposes).

        Accumulates total window time and count so overlap/absorbed metrics
        have a real denominator.  No behavioral coupling (PR1 semantics).
        """
        self._window_total_s += sleep_s
        self._window_count += 1

    def set_prev_manifest(self, entries: CasManifestEntries | None) -> None:
        """Explicit setter for prev_cas_manifest (used by forced-sync callers)."""
        self._prev_cas_manifest = entries

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pending_forced_sync(self) -> bool:
        """True when the resolved boundary requires forced-sync."""
        return bool(
            self._pending_boundary_record is not None
            and self._pending_boundary_record.get("checkpoint_pending_forced_sync")
        )

    @property
    def pending_action_index(self) -> int | None:
        """Action index of the pending boundary (for forced-sync context)."""
        return self._pending_action_index

    @property
    def pending_cas_spec(self) -> dict[str, Any] | None:
        """CAS spec of the pending boundary (for forced-sync context)."""
        return self._pending_cas_spec

    @property
    def pending_lane_active(self) -> bool:
        """Whether a subagent lane was active before the pending boundary."""
        return self._pending_lane_active

    @property
    def pending_tool_mismatch_reason(self) -> str | None:
        """Tool-level mismatch reason stored with the pending boundary."""
        return self._pending_tool_mismatch_reason

    @property
    def pending_iteration(self) -> int:
        return self._pending_iteration

    @property
    def pending_warmup(self) -> bool:
        return self._pending_warmup

    @property
    def state(self) -> str:
        return self._state

    @property
    def prev_cas_manifest(self) -> CasManifestEntries | None:
        return self._prev_cas_manifest

    @property
    def _scheduling(self) -> str:
        return self._config.checkpoint_scheduling

    # ------------------------------------------------------------------
    # Internal: drain
    # ------------------------------------------------------------------

    async def _drain_internal(self) -> float:
        t0 = time.monotonic()
        try:
            if self._in_flight_task and not self._in_flight_task.done():
                await asyncio.wait_for(self._in_flight_task, timeout=None)
        except asyncio.CancelledError:
            pass
        return (time.monotonic() - t0) * 1000

    # ------------------------------------------------------------------
    # Internal: inline capture/compare (sync mode)
    # ------------------------------------------------------------------

    async def _run_capture_compare_inline(
        self,
        action_index: int,
        cas_spec: dict[str, Any] | None,
        record_slot: dict[str, Any],
        tool_name: str | None = None,
        tool_args_json: str | None = None,
    ) -> None:
        """Run capture + compare synchronously (sync mode path).

        The decision is probe-only: ``backend.probe_changes_since`` runs
        after each tool execution and determines whether a checkpoint
        capture is needed.  No whitelist-based command classifier is
        consulted (removed per ADR: unreliable syntactic classification
        that the probe already dominates).
        """
        if self._container is None or cas_spec is None:
            record_slot.setdefault("cas_compare_source", "skip_audit")
            return

        # FCBackend captures paired snapshots
        if isinstance(self._container.backend, FCBackend):
            try:
                fc_snapshot = await self._container.backend.capture_snapshot()
                self._container.replay_snapshots[action_index] = fc_snapshot
                self._prev_timestamp_ns = getattr(fc_snapshot, "timestamp_ns", time.time_ns())
            except Exception:
                pass
            record_slot.setdefault("cas_compare_source", "skip_audit")
            return

        replay_entries, _ = await _capture_replay_snapshot_manifest_diagnostic(
            container=self._container,
            root=cas_spec.get("root", "/testbed"),
            previous_manifest=self._prev_cas_manifest,
            context="checkpoint_boundary",
        )
        if replay_entries is not None:
            self._prev_cas_manifest = replay_entries
            self._prev_timestamp_ns = time.time_ns()
            source_entries = self._folded_source_entries
            if source_entries:
                cas_fields = _cas_manifest_comparison_fields(
                    source_entries=source_entries,
                    replay_entries=replay_entries,
                )
                record_slot.update(cas_fields)
                record_slot["cas_compare_source"] = "normal"
            else:
                record_slot.setdefault("cas_compare_source", "normal")
        else:
            record_slot.setdefault("cas_compare_source", "skip_audit")

    # ------------------------------------------------------------------
    # Internal: boundary work task (deferred mode)
    # ------------------------------------------------------------------

    async def _boundary_work(
        self,
        action_index: int,
        cas_spec: dict[str, Any] | None,
        tool_name: str | None,
        tool_args_json: str | None,
        tool_mismatch_reason: str | None,
        record_slot: dict[str, Any],
        lane_active_at_parent: bool = False,
        iteration: int = 0,
        warmup: bool = False,
    ) -> None:
        """Run probe -> classify -> capture -> compare chain as background task.

        Runs as an asyncio task created by on_boundary. The replay-side
        _capture_replay_snapshot_manifest_diagnostic already dispatches walk
        operations via asyncio.to_thread internally; backend.capture_snapshot()
        is an async method so it naturally yields the event loop during I/O.

        Stores per-boundary durations in _boundary_* accumulators and records
        the oracle verdict in *record_slot*.  Does NOT flush the pending record
        — finalization (including checkpoint_exposed_ms charging) is deferred
        to drain() / close(), so the gate wait is measured correctly.
        """
        import time as _time

        if self._container is None or cas_spec is None:
            self._boundary_cas_compare_source = "skip_audit"
            self._state = "IDLE"
            return

        self._state = "CAPTURING"
        self._boundary_probe_elapsed_ms = 0.0
        cap_t0 = _time.monotonic()

        try:
            # FCBackend captures paired snapshots
            if isinstance(self._container.backend, FCBackend):
                try:
                    fc_snapshot = await self._container.backend.capture_snapshot()
                    self._container.replay_snapshots[action_index] = fc_snapshot
                    self._prev_timestamp_ns = getattr(fc_snapshot, "timestamp_ns", time.time_ns())
                except Exception:
                    pass
                self._boundary_capture_elapsed_ms = (_time.monotonic() - cap_t0) * 1000
                self._boundary_cas_compare_source = "skip_audit"
            else:
                # CAS backend: use the diagnostic capture (handles to_thread internally)
                replay_entries, _ = await _capture_replay_snapshot_manifest_diagnostic(
                    container=self._container,
                    root=cas_spec.get("root", "/testbed"),
                    previous_manifest=self._prev_cas_manifest,
                    context="checkpoint_boundary",
                )
                self._boundary_capture_elapsed_ms = (_time.monotonic() - cap_t0) * 1000

                # Step 2: oracle compare
                self._state = "COMPARING"
                comp_t0 = _time.monotonic()
                if replay_entries is not None and self._folded_source_entries:
                    fields = _cas_manifest_comparison_fields(
                        source_entries=self._folded_source_entries,
                        replay_entries=replay_entries,
                    )
                    record_slot.update(fields)
                    record_slot["cas_compare_source"] = "normal"
                    if not fields.get("cas_manifest_match", True):
                        record_slot["checkpoint_pending_forced_sync"] = True
                elif replay_entries is not None:
                    record_slot["cas_compare_source"] = "normal"
                else:
                    record_slot["cas_compare_source"] = "skip_audit"
                if replay_entries is not None:
                    self._prev_cas_manifest = replay_entries
                    # Set prev_timestamp_ns from capture time for future probe baseline
                    self._prev_timestamp_ns = time.time_ns()
                self._boundary_compare_elapsed_ms = (_time.monotonic() - comp_t0) * 1000
                self._boundary_cas_compare_source = record_slot.get("cas_compare_source", "")
        except Exception as exc:
            record_slot["checkpoint_after_error"] = {"error": str(exc)}
            record_slot.setdefault("cas_compare_source", "skip_audit")
            self._boundary_cas_compare_source = "skip_audit"
            self._state = "IDLE"
            return

        # Task complete — record stays pending.  drain() / close() will call
        # _finalize_boundary_record after charging exposed_ms, then flush.
        self._state = "IDLE"

    # ------------------------------------------------------------------
    # Internal: CAS oracle compare (used by inline sync path)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Internal: deferred record management
    # ------------------------------------------------------------------

    def _finalize_and_accumulate(self) -> None:
        """Finalize the pending boundary record (compute per-boundary metrics).

        Called by drain() / close() after exposed_ms has been charged, so
        checkpoint_exposed_ms reflects the actual gate wait time.  Does NOT
        flush the record — the caller may need to run forced-sync first when
        pending_forced_sync is True.
        """
        pending = self._pending_boundary_record
        if pending is None:
            return
        # Decouple finalization from flushing so exposed_ms is charged first.
        self._finalize_boundary_record(pending)

    def flush_pending(self) -> None:
        """Emit the pending boundary record and any deferred records.

        Must be called after drain() (and, when applicable, after forced-sync
        has resolved a pending mismatch).  Idempotent — a no-op when there is
        no pending record.
        """
        self._flush_deferred()

    def _finalize_boundary_record(self, record_slot: dict[str, Any]) -> None:
        """Compute derived per-boundary metrics and accumulate session totals.

        Writes per-boundary timing fields into *record_slot* and increments
        session-aggregate accumulators. Must be called exactly once per boundary.
        """
        probe = self._boundary_probe_elapsed_ms
        capture = self._boundary_capture_elapsed_ms
        compare = self._boundary_compare_elapsed_ms
        exposed = self._boundary_exposed_ms
        total_work = probe + capture + compare
        absorbed = max(0.0, total_work - exposed)
        overlap_fraction = absorbed / total_work if total_work > 0 else 1.0

        record_slot["checkpoint_exposed_ms"] = round(exposed, 3)
        record_slot["probe_elapsed_ms"] = round(probe, 3)
        record_slot["capture_elapsed_ms"] = round(capture, 3)
        record_slot["compare_elapsed_ms"] = round(compare, 3)
        record_slot["capture_absorbed_ms"] = round(absorbed, 3)
        record_slot["overlap_fraction"] = round(overlap_fraction, 3)
        record_slot["scheduler_overhead_ms"] = round(
            self._boundary_scheduler_overhead_ms, 3
        )

        # Ensure cas_compare_source is always populated
        if not record_slot.get("cas_compare_source"):
            record_slot["cas_compare_source"] = (
                self._boundary_cas_compare_source or "skip_audit"
            )

        # Accumulate into session totals
        self._probe_elapsed_ms += probe
        self._capture_elapsed_ms += capture
        self._compare_elapsed_ms += compare
        self._exposed_ms += exposed
        self._scheduler_overhead_ms += self._boundary_scheduler_overhead_ms
        if exposed == 0.0 and total_work > 0:
            self._captures_fully_absorbed += 1

    def _flush_deferred(self) -> None:
        """Flush all deferred records through the logging callback.

        Emits the pending boundary record (after finalization has already been
        applied by drain() / close()) followed by any queued records.
        """
        records = self._deferred_records[:]
        self._deferred_records.clear()
        pending = self._pending_boundary_record
        agent_id = self._pending_boundary_agent_id
        if pending is None:
            # No pending record — just emit any deferred records.
            for r in records:
                record_agent_id = (
                    r.agent_id
                    if isinstance(r, TraceAction)
                    else r.get("agent_id", "unknown")
                )
                self._log_action(record_agent_id, r)
            return
        # Clear pending state BEFORE logging so log_or_defer sees no pending
        # record and writes immediately.
        self._pending_boundary_record = None
        self._pending_boundary_agent_id = None
        # The pending record is the tool_exec record slot (cas_manifest_fields
        # in the loop), already populated with CAS and timing fields.  It is
        # emitted as part of the normal tool_exec logging from the loop, not
        # here — so we only log the deferred records queued behind it.
        for r in records:
            record_agent_id = (
                r.agent_id
                if isinstance(r, TraceAction)
                else r.get("agent_id", agent_id or "unknown")
            )
            self._log_action(record_agent_id, r)

    # ------------------------------------------------------------------
    # Internal: previous manifest info for incremental captures
    # ------------------------------------------------------------------

    def _prev_manifest_info(self) -> dict[str, Any] | None:
        """Return info for incremental capture."""
        if self._prev_cas_manifest is not None:
            return {
                "manifest": self._prev_cas_manifest,
                "timestamp_ns": self._prev_timestamp_ns,
            }
        return None

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict[str, Any]:
        return {
            "checkpoint_exposed_ms": round(self._exposed_ms, 3),
            "capture_elapsed_ms": round(self._capture_elapsed_ms, 3),
            "probe_elapsed_ms": round(self._probe_elapsed_ms, 3),
            "compare_elapsed_ms": round(self._compare_elapsed_ms, 3),
            "scheduler_overhead_ms": round(self._scheduler_overhead_ms, 3),
            "boundaries_total": self._boundaries_total,
            "captures_fully_absorbed": self._captures_fully_absorbed,
            "llm_sleep_window_total_s": round(self._window_total_s, 3),
            "llm_sleep_window_count": self._window_count,
        }

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    async def close(self) -> dict[str, Any]:
        """Drain in-flight work, flush deferred records. Returns aggregates.

        Any pending boundary record that has not yet been flushed (e.g. a
        deferred CAS mismatch whose forced-sync could not be invoked because
        no further container-touching op occurred) is emitted here so the
        session JSONL is always complete.
        """
        await self.drain()
        if self._deferred_records or self._pending_boundary_record is not None:
            self.flush_pending()
        return self.get_metrics()


async def _replay_cloud_model_session(
    prepared_session: PreparedTraceSession,
    *,
    trace_logger: TraceLogger,
    replay_zero_monotonic: float | None = None,
    replay_speed: float,
    llm_timing: LLMTimingConfig,
    command_timeout_s: float,
    warmup_skip_iterations: int,
    replay_scheduler_config: ReplaySchedulerConfig | None = None,
) -> ReplayTaskStats:
    if replay_scheduler_config is None:
        replay_scheduler_config = ReplaySchedulerConfig()
    loaded = prepared_session.loaded
    ctr = prepared_session.container
    source_model = (loaded.summary or {}).get("model", "unknown")
    logger.info(
        "Replaying %s [scaffold=%s]: %d actions from %s at %.2fx (llm_timing=%s)",
        loaded.agent_id,
        loaded.scaffold,
        len(loaded.actions),
        source_model,
        replay_speed,
        llm_timing.mode,
    )

    wall_start = time.time()
    succeeded_actions = 0
    outcome_mismatches = 0
    unresolved_mismatches = 0
    replay_action_errors = 0
    fatal_replay_errors = 0
    forced_sync_actions = 0
    forced_sync_attempts = 0
    forced_sync_successes = 0
    forced_sync_continued_actions = 0
    source_failed_actions = 0
    replay_failed_actions = 0
    matched_failed_actions = 0
    previous_source_end: float | None = None
    sleep_drifts: list[SleepDrift] = []

    if replay_zero_monotonic is not None:
        start_drift = await _sleep_until_monotonic(replay_zero_monotonic)
        if start_drift is not None:
            sleep_drifts.append(start_drift)

    # Create replay checkpoint scheduler (Hook 0 — session initialization).
    # The scheduler owns prev_cas_manifest, folded_source_entries, and the
    # pending-background-work state. In sync mode it runs capture/compare
    # inline; in deferred mode it overlaps them with LLM sleep windows.
    scheduler = ReplayCheckpointScheduler(
        config=replay_scheduler_config,
        container=ctr,
        source_trace=loaded.source_trace,
        log_action=trace_logger.log_trace_action,
    )
    lane_active_since_last_boundary = False

    for action_index, action in enumerate(loaded.actions):
        action_id = str(action.get("action_id", ""))
        action_type = str(action.get("action_type", ""))
        iteration = int(action.get("iteration", 0))
        data = action.get("data", {})
        source_action_agent_id = _source_action_agent_id(action)
        source_is_subagent_lane = _is_subagent_lane_action(
            action,
            source_agent_id=loaded.source_agent_id,
        )
        if source_is_subagent_lane:
            lane_active_since_last_boundary = True
        action_ts_start, action_ts_end = _coerce_action_bounds(action, source_trace=loaded.source_trace)
        source_duration_s = max(0.0, action_ts_end - action_ts_start)

        source_gap_sleep = await _sleep_source_gap(
            previous_source_end=previous_source_end,
            action_source_start=action_ts_start,
            replay_speed=replay_speed,
        )
        if source_gap_sleep is not None:
            sleep_drifts.append(source_gap_sleep)
        action_excluded_overhead_s = _source_action_excluded_overhead_s(action)
        effective_action_source_end = action_ts_end + action_excluded_overhead_s
        previous_source_end = max(
            effective_action_source_end,
            previous_source_end
            if previous_source_end is not None
            else effective_action_source_end,
        )

        try:
            if action_type == "llm_call":
                record_ts_start = time.time()
                sleep_s, llm_timing_fields = _llm_replay_duration_s(
                    data=data,
                    source_duration_s=source_duration_s,
                    replay_speed=replay_speed,
                    timing=llm_timing,
                )
                action_sleep = await _sleep_and_measure(
                    sleep_s,
                    phase="llm_replay",
                )
                scheduler.note_window(sleep_s)  # Hook 3: metrics (no behavioral coupling)
                if action_sleep is not None:
                    sleep_drifts.append(action_sleep)
                record_ts_end = time.time()
                record = _make_trace_action(
                    loaded=loaded,
                    action_type="llm_call",
                    action_id=_replay_action_id(
                        action,
                        source_agent_id=loaded.source_agent_id,
                        fallback=f"llm_{iteration}",
                    ),
                    iteration=iteration,
                    ts_start=record_ts_start,
                    ts_end=record_ts_end,
                    data={
                        **_source_llm_message_payload(data),
                        "raw_response": data.get("raw_response", {}),
                        "prompt_tokens": data.get("prompt_tokens", 0),
                        "completion_tokens": data.get("completion_tokens", 0),
                        "llm_latency_ms": (record_ts_end - record_ts_start) * 1000,
                        "simulate_source": str(loaded.source_trace),
                        "source_llm_latency_ms": data.get("llm_latency_ms"),
                        "replay_mode": "cloud_model",
                        "replay_speed": replay_speed,
                        **llm_timing_fields,
                        **(
                            {
                                "replay_source": "subagent_lane_replay",
                                "source_lane_agent_id": source_action_agent_id,
                            }
                            if source_is_subagent_lane
                            else {}
                        ),
                        "sim_metrics": {
                            "warmup": iteration < warmup_skip_iterations,
                            **_sleep_drift_metrics(
                                source_gap=source_gap_sleep,
                                action_sleep=action_sleep,
                            ),
                        },
                    },
                )
                scheduler.log_or_defer(loaded.agent_id, record)  # Hook 4
                succeeded_actions += 1
                continue

            if action_type != "tool_exec":
                logger.warning(
                    "Skipping unsupported action_type=%s in %s",
                    action_type,
                    loaded.source_trace,
                )
                continue

            tool_name = data.get("tool_name")
            tool_args = data.get("tool_args", "{}")
            if not tool_name:
                logger.warning(
                    "Skipping tool action without tool_name in %s",
                    loaded.source_trace,
                )
                continue

            record_ts_start = time.time()
            source_duration_ms = float(data.get("duration_ms") or 0.0)
            action_sleep: SleepDrift | None = None
            source_success = _source_tool_success(data)
            source_tool_result = data.get("tool_result", data.get("result", ""))
            source_exec_timeout = _source_exec_timeout_s(
                tool_name=tool_name,
                tool_args_json=tool_args,
                source_duration_ms=source_duration_ms,
                source_success=source_success,
                source_tool_result=source_tool_result,
                source_timed_out=data.get("timed_out"),
            )
            replay_exec_timeout = _effective_source_exec_timeout_s(
                source_exec_timeout_s=source_exec_timeout,
                replay_speed=replay_speed,
            )
            source_resource_timeline = valid_resource_timeline(
                data.get("resource_timeline")
            )
            original_artifact_path: str | None = None
            mapped_artifact_path: str | None = None
            exec_resource_timeline: dict[str, Any] | None = None
            tool_exec_metadata: dict[str, Any] = {}
            if not source_success:
                source_failed_actions += 1
            if source_is_subagent_lane:
                action_sleep = await _sleep_and_measure(
                    source_duration_ms / 1000 / replay_speed,
                    phase="tool_trace_replay",
                )
                if action_sleep is not None:
                    sleep_drifts.append(action_sleep)
                tool_result = data.get("tool_result", data.get("result", ""))
                tool_success = source_success
                duration_ms = (time.time() - record_ts_start) * 1000
                replay_source = "subagent_lane_replay"
            elif ctr is None:
                logger.info(
                    "Skipping host-mode tool action for %s action=%s tool=%s",
                    loaded.agent_id,
                    action_id,
                    tool_name,
                )
                replay_source = "skipped_host_mode"
                tool_result = data.get("tool_result", data.get("result", ""))
                tool_success = source_success
                action_sleep = await _sleep_and_measure(
                    source_duration_ms / 1000 / replay_speed,
                    phase="tool_trace_replay",
                )
                if action_sleep is not None:
                    sleep_drifts.append(action_sleep)
                duration_ms = (time.time() - record_ts_start) * 1000
            elif tool_name == "message":
                action_sleep = await _sleep_and_measure(
                    source_duration_ms / 1000 / replay_speed,
                    phase="tool_trace_replay",
                )
                if action_sleep is not None:
                    sleep_drifts.append(action_sleep)
                tool_result = data.get("tool_result", data.get("result", ""))
                if not tool_result:
                    tool_result = "Message replayed as no-op"
                tool_success = source_success
                duration_ms = (time.time() - record_ts_start) * 1000
                replay_source = "message_noop"
            elif tool_name.startswith("mcp_") or tool_name in _TRACE_REPLAY_TOOL_NAMES:
                action_sleep = await _sleep_and_measure(
                    source_duration_ms / 1000 / replay_speed,
                    phase="tool_trace_replay",
                )
                if action_sleep is not None:
                    sleep_drifts.append(action_sleep)
                tool_result = data.get("tool_result", "")
                tool_success = source_success
                duration_ms = (time.time() - record_ts_start) * 1000
                replay_source = "replayed_from_trace"
            else:
                denied_command = _denied_exec_command(
                    tool_name=tool_name,
                    tool_args_json=tool_args,
                )
                if denied_command is not None:
                    action_sleep = await _sleep_and_measure(
                        source_duration_ms / 1000 / replay_speed,
                        phase="tool_trace_replay",
                    )
                    if action_sleep is not None:
                        sleep_drifts.append(action_sleep)
                    duration_ms = (time.time() - record_ts_start) * 1000
                    tool_result = source_tool_result
                    tool_success = source_success
                    replay_source = "denied_command_replayed_from_trace"
                    tool_exec_metadata = _source_tool_exec_metadata(data)
                    tool_exec_metadata["replay_denied_command"] = denied_command
                else:
                    (
                        mapped_tool_args,
                        original_artifact_path,
                        mapped_artifact_path,
                        mapped_exists,
                    ) = _container_tool_runtime_args(
                        tool_name=tool_name,
                        tool_args_json=tool_args,
                        runtime_root_map=prepared_session.runtime_artifact_root_map,
                    )
                    if original_artifact_path is not None and not mapped_exists:
                        action_sleep = await _sleep_and_measure(
                            source_duration_ms / 1000 / replay_speed,
                            phase="tool_trace_replay",
                        )
                        if action_sleep is not None:
                            sleep_drifts.append(action_sleep)
                        tool_result = _artifact_unavailable_result(
                            original_artifact_path
                        )
                        tool_success = False
                        duration_ms = (time.time() - record_ts_start) * 1000
                        replay_source = "source_artifact_unavailable"
                    else:
                        exec_resource_timeline = (
                            source_resource_timeline
                            if _tool_uses_single_exec_command_semantics(
                                tool_name,
                                mapped_tool_args,
                            )
                            else None
                        )
                        # Hook 2 — gate: drain pending boundary work before container mutation
                        if replay_scheduler_config.checkpoint_scheduling == "deferred":
                            await scheduler.drain()
                            # Check for a pending CAS mismatch from a previous
                            # boundary whose work completed during the last LLM
                            # sleep.  If set, run forced-sync now (inline) before
                            # the next container-touching tool executes.
                            if scheduler.pending_forced_sync:
                                _fs_index = scheduler.pending_action_index
                                _fs_spec = scheduler.pending_cas_spec
                                if _fs_spec is not None and _fs_index is not None:
                                    # Build a minimal forced-sync record for the
                                    # pending boundary.  Uses stored context from
                                    # the mismatch boundary.
                                    _fs_data = loaded.actions[_fs_index].get("data") or {}
                                    _fs_checkpoint_spec = _checkpoint_after_spec(
                                        action_data=_fs_data,
                                        source_trace=loaded.source_trace,
                                    )
                                    if _fs_checkpoint_spec is not None:
                                        _fs_pending = scheduler._pending_boundary_record
                                        _fs_fields = await _run_deferred_forced_sync(
                                            prepared_session=prepared_session,
                                            effective_mismatch_reason="cas_state_mismatch",
                                            checkpoint_spec=_fs_checkpoint_spec,
                                            checkpoint_action_index=_fs_index,
                                            ctr=ctr,
                                            scheduler=scheduler,
                                            lane_induced=(
                                                scheduler.pending_lane_active
                                            ),
                                            replay_speed=replay_speed,
                                            command_timeout_s=command_timeout_s,
                                        )
                                        if _fs_pending is not None:
                                            _fs_pending.update(_fs_fields)
                                scheduler.flush_pending()
                            elif scheduler.state == "IDLE":
                                # Clean boundary (no mismatch) — the record has
                                # been finalized by drain(); flush it now.
                                scheduler.flush_pending()
                        (
                            tool_result,
                            duration_ms,
                            tool_success,
                            tool_exec_metadata,
                        ) = await _execute_container_tool_call(
                            agent=ctr.backend or ctr.agent,
                            tool_name=tool_name,
                            mapped_tool_args=mapped_tool_args,
                            command_timeout_s=command_timeout_s,
                            source_exec_timeout=replay_exec_timeout,
                            mapped_artifact_path=mapped_artifact_path,
                            exec_resource_timeline=exec_resource_timeline,
                        )
                        replay_source = (
                            "restored_runtime_artifact"
                            if mapped_artifact_path is not None
                            else "executed_in_container"
                        )
            if not tool_success:
                replay_failed_actions += 1
            source_returncode = data.get("returncode")
            replay_returncode = tool_exec_metadata.get("returncode")
            source_timed_out = data.get("timed_out")
            replay_timed_out = tool_exec_metadata.get("timed_out")
            mismatch_reason = _tool_mismatch_reason(
                source_success=source_success,
                tool_success=tool_success,
                replay_source=replay_source,
                source_tool_result=source_tool_result,
                replay_tool_result=tool_result,
                tool_name=tool_name,
                tool_args_json=tool_args,
                source_returncode=source_returncode,
                replay_returncode=replay_returncode,
                source_timed_out=source_timed_out,
                replay_timed_out=replay_timed_out,
            )
            effective_mismatch_reason = mismatch_reason
            normalized_output_match = _exec_normalized_output_match(
                tool_name=tool_name,
                tool_args_json=tool_args,
                source_tool_result=source_tool_result,
                replay_tool_result=tool_result,
            )
            record_ts_end = time.time()
            # === CAS manifest comparison at checkpoint boundaries ===
            cas_manifest_fields: dict[str, Any] = {}
            cas_spec = _checkpoint_after_spec(
                action_data=data,
                source_trace=loaded.source_trace,
            )
            lane_induced_mismatch_candidate = False
            if (
                cas_spec is not None
                and ctr is not None
                and not source_is_subagent_lane
            ):
                lane_active_at_parent_boundary = lane_active_since_last_boundary
                lane_active_since_last_boundary = False

                # Hook 1: hand off to scheduler
                # Sync mode: capture/compare runs inline, cas_manifest_fields populated
                # Deferred mode: background task started, fields filled asynchronously
                await scheduler.on_boundary(
                    action_index=action_index,
                    cas_spec=cas_spec,
                    tool_name=tool_name,
                    tool_args_json=tool_args,
                    tool_mismatch_reason=mismatch_reason,
                    record_slot=cas_manifest_fields,
                    agent_id=loaded.agent_id,
                    lane_active_at_parent=lane_active_at_parent_boundary,
                    iteration=iteration,
                    warmup=iteration < warmup_skip_iterations,
                )
                if replay_scheduler_config.checkpoint_scheduling == "sync":
                    # Sync mode: cas_manifest_fields already populated
                    if (
                        effective_mismatch_reason is None
                        and not cas_manifest_fields.get("cas_manifest_match", True)
                    ):
                        effective_mismatch_reason = "cas_state_mismatch"
                        lane_induced_mismatch_candidate = (
                            lane_active_at_parent_boundary
                        )
            # Output content mismatch: transport tiers matched (same timeout,
            # success, exit code) but normalized output differs.  Exclude
            # web_search and web_fetch whose content legitimately varies.
            # Checked after CAS mismatch so cas_state_mismatch (root cause)
            # takes priority over output_content_mismatch (symptom).
            if effective_mismatch_reason is None and normalized_output_match is False:
                if tool_name not in ("web_search", "web_fetch"):
                    effective_mismatch_reason = "output_content_mismatch"
            replay_outcome_match = mismatch_reason is None
            output_diff_snippet: str | None = None
            if effective_mismatch_reason is not None or normalized_output_match is False:
                source_raw = (
                    "" if source_tool_result is None else str(source_tool_result)
                )
                replay_raw = "" if tool_result is None else str(tool_result)
                output_diff_snippet = _compute_output_diff_snippet(
                    source_raw,
                    replay_raw,
                )
            oracle_verdict = _MISMATCH_ORACLE.verdict(
                source_returncode=source_returncode,
                replay_returncode=replay_returncode,
                source_timed_out=source_timed_out,
                replay_timed_out=replay_timed_out,
                normalized_output_match=normalized_output_match,
                output_diff_snippet=output_diff_snippet,
                cas_manifest_match=cas_manifest_fields.get("cas_manifest_match"),
                cas_modified_count=cas_manifest_fields.get("cas_modified_count"),
                cas_removed_count=cas_manifest_fields.get("cas_removed_count"),
                cas_added_count=cas_manifest_fields.get("cas_added_count"),
                mismatch_reason=effective_mismatch_reason,
            )
            if (not tool_success) and replay_outcome_match:
                matched_failed_actions += 1
            forced_sync_fields: dict[str, Any] = {}
            # Hook 2 — second gate: drain before forced-sync.
            # Only drain when a tool-result mismatch is already known
            # (effective_mismatch_reason is set before the gate).  For clean
            # boundaries the background task overlaps with the upcoming LLM
            # sleep and checkpoint_exposed_ms is measured at the next
            # Hook 2 drain — this is the correct exposure measurement point.
            if (
                replay_scheduler_config.checkpoint_scheduling == "deferred"
                and effective_mismatch_reason is not None
            ):
                _exposed = await scheduler.drain()
                # Upgrade effective_mismatch_reason from record_slot if the
                # deferred CAS compare found a mismatch.
                if cas_manifest_fields.get(
                    "checkpoint_pending_forced_sync"
                ):
                    effective_mismatch_reason = "cas_state_mismatch"
            if (
                effective_mismatch_reason is not None
                and ctr is not None
                and not source_is_subagent_lane
            ):
                checkpoint_spec = _checkpoint_after_spec(
                    action_data=data,
                    source_trace=loaded.source_trace,
                )
                checkpoint_action_index: int | None = (
                    action_index if checkpoint_spec is not None else None
                )
                fallback_reapply_start_index: int | None = None
                if checkpoint_spec is None:
                    fallback_spec: dict[str, Any] | None = None
                    fallback_action_index: int | None = None
                    for prev_index in range(action_index - 1, -1, -1):
                        prev_action = loaded.actions[prev_index]
                        prev_data = prev_action.get("data") or {}
                        candidate = _checkpoint_after_spec(
                            action_data=prev_data,
                            source_trace=loaded.source_trace,
                        )
                        if candidate is not None:
                            fallback_spec = candidate
                            fallback_action_index = prev_index
                            break

                    if fallback_spec is None:
                        forced_sync_fields = {
                            "forced_sync_attempted": True,
                            "forced_sync_success": False,
                            "forced_sync_continued": False,
                            "forced_sync_reason": effective_mismatch_reason,
                            "forced_sync_status": "checkpoint_missing",
                            "forced_sync_error": (
                                "no checkpoint available (searched entire trace history)"
                            ),
                        }
                    else:
                        assert fallback_action_index is not None
                        checkpoint_spec = fallback_spec
                        checkpoint_action_index = fallback_action_index
                        fallback_reapply_start_index = fallback_action_index + 1
                        forced_sync_fields = {
                            "forced_sync_attempted": True,
                            "forced_sync_reason": effective_mismatch_reason,
                            "forced_sync_overhead_excluded": True,
                            "forced_sync_fallback": True,
                            "forced_sync_fallback_from_action_index": (
                                fallback_action_index
                            ),
                            "forced_sync_fallback_from_action_id": str(
                                loaded.actions[fallback_action_index].get(
                                    "action_id",
                                    "",
                                )
                            ),
                        }
                else:
                    forced_sync_fields = {
                        "forced_sync_attempted": True,
                        "forced_sync_reason": effective_mismatch_reason,
                        "forced_sync_overhead_excluded": True,
                    }
                if lane_induced_mismatch_candidate:
                    forced_sync_fields["lane_induced_mismatch_candidate"] = True

                if checkpoint_spec is not None:
                    assert checkpoint_action_index is not None
                    forced_sync_started = time.monotonic()

                    if isinstance(ctr.backend, FCBackend):
                        # FC forced-sync: restore from the most recent paired
                        # snapshot captured during replay at or before the
                        # checkpoint action index.
                        fc_restore_snapshot_index: int | None = None
                        available = sorted(
                            idx for idx in ctr.replay_snapshots
                            if idx <= checkpoint_action_index
                        )
                        if available:
                            fc_restore_snapshot_index = available[-1]
                            fc_snapshot = ctr.replay_snapshots[
                                fc_restore_snapshot_index
                            ]
                        if fc_restore_snapshot_index is not None:
                            try:
                                restored = await ctr.backend.restore_snapshot(
                                    fc_snapshot,
                                )
                                scheduler.set_prev_manifest(None)  # FC restore
                                restore_result = {
                                    "forced_sync_success": restored,
                                    "forced_sync_status": (
                                        "fc_paired_restored_continuation"
                                        if restored
                                        else "fc_paired_restore_failed"
                                    ),
                                    "forced_sync_continued": restored,
                                    "forced_sync_overhead_excluded": True,
                                    "forced_sync_fc_snapshot_index": (
                                        fc_restore_snapshot_index
                                    ),
                                    "forced_sync_fc_mem_version": (
                                        fc_snapshot.process_state.get(
                                            "mem_version"
                                        )
                                        if fc_snapshot.process_state
                                        else None
                                    ),
                                    "forced_sync_fc_disk_version": (
                                        fc_snapshot.disk_state.get(
                                            "disk_version"
                                        )
                                    ),
                                    "restore_root_exists": True,
                                }
                            except Exception as exc:
                                restore_result = {
                                    "forced_sync_success": False,
                                    "forced_sync_status": "fc_paired_restore_failed",
                                    "forced_sync_continued": False,
                                    "forced_sync_error": (
                                        f"{type(exc).__name__}: {exc}"
                                    ),
                                }
                        else:
                            restore_result = {
                                "forced_sync_success": False,
                                "forced_sync_status": "fc_no_paired_snapshot",
                                "forced_sync_continued": False,
                                "forced_sync_error": (
                                    "no FC paired snapshot available "
                                    "for checkpoint_action_index="
                                    f"{checkpoint_action_index}"
                                ),
                            }
                        forced_sync_fields.update(restore_result)
                        skip_post_restore_verification = True
                        # Reapply actions from the snapshot index to the
                        # mismatch point when snapshot was from an earlier action.
                        fc_reapply_start = (
                            fc_restore_snapshot_index + 1
                            if fc_restore_snapshot_index is not None
                            and fc_restore_snapshot_index < action_index
                            else None
                        )
                        if (
                            restore_result.get("forced_sync_success") is True
                            and fc_reapply_start is not None
                        ):
                            reapply_fields = await _reapply_forced_sync_actions(
                                prepared_session=prepared_session,
                                start_index=fc_reapply_start,
                                end_index=action_index,
                                replay_speed=replay_speed,
                                command_timeout_s=command_timeout_s,
                            )
                            forced_sync_fields.update(reapply_fields)
                            if reapply_fields["forced_sync_reapply_errors"]:
                                forced_sync_fields.update(
                                    {
                                        "forced_sync_success": False,
                                        "forced_sync_continued": False,
                                        "forced_sync_status": "fc_reapply_failed",
                                        "forced_sync_error": "; ".join(
                                            reapply_fields[
                                                "forced_sync_reapply_errors"
                                            ]
                                        ),
                                    }
                                )
                    else:
                        checkpoint_chain = _checkpoint_chain_specs_for_action(
                            actions=loaded.actions,
                            target_index=checkpoint_action_index,
                            source_trace=loaded.source_trace,
                        )
                        try:
                            if checkpoint_chain is None:
                                restore_result = _checkpoint_restore_failed_fields(
                                    checkpoint_path=Path(str(checkpoint_spec["path"])),
                                    kind=str(
                                        checkpoint_spec.get("kind")
                                        or "cas_manifest_incremental"
                                    ),
                                    restore_root=str(
                                        checkpoint_spec.get("root") or "/testbed"
                                    ),
                                    status="checkpoint_full_missing",
                                    error=(
                                        "incremental checkpoint has no preceding full "
                                        "checkpoint"
                                    ),
                                    started=time.monotonic(),
                                    archive_exists=Path(
                                        str(checkpoint_spec["path"])
                                    ).is_file(),
                                )
                            else:
                                restore_result = await asyncio.to_thread(
                                    _restore_checkpoint_chain_to_container,
                                    checkpoint_specs=checkpoint_chain,
                                    container=ctr,
                                )
                            forced_sync_fields.update(restore_result)
                            if forced_sync_fields.get("forced_sync_success") is True:
                                scheduler.set_prev_manifest(None)  # FC restore
                            skip_post_restore_verification = False
                            if (
                                forced_sync_fields.get("forced_sync_success") is True
                                and fallback_reapply_start_index is not None
                            ):
                                restore_source_entries = (
                                    _fold_source_checkpoint_entries_through_action(
                                        actions=loaded.actions,
                                        target_index=checkpoint_action_index,
                                        source_trace=loaded.source_trace,
                                    )
                                )
                                restore_verification_fields, _ = (
                                    await _verify_forced_sync_restore_state(
                                        container=ctr,
                                        checkpoint_spec=checkpoint_spec,
                                        source_entries=restore_source_entries,
                                        context="forced_sync_restore_pre_reapply",
                                    )
                                )
                                forced_sync_fields.update(
                                    {
                                        "forced_sync_restore_verified": (
                                            restore_verification_fields[
                                                "forced_sync_verified"
                                            ]
                                        ),
                                        "forced_sync_restore_verification": (
                                            restore_verification_fields[
                                                "forced_sync_verification"
                                            ]
                                        ),
                                    }
                                )
                                reapply_fields = await _reapply_forced_sync_actions(
                                    prepared_session=prepared_session,
                                    start_index=fallback_reapply_start_index,
                                    end_index=action_index,
                                    replay_speed=replay_speed,
                                    command_timeout_s=command_timeout_s,
                                )
                                forced_sync_fields.update(reapply_fields)
                                skip_post_restore_verification = (
                                    reapply_fields[
                                        "forced_sync_reapplied_action_count"
                                    ]
                                    > 0
                                )
                                if reapply_fields["forced_sync_reapply_errors"]:
                                    forced_sync_fields.update(
                                        {
                                            "forced_sync_success": False,
                                            "forced_sync_continued": False,
                                            "forced_sync_status": "reapply_failed",
                                            "forced_sync_error": "; ".join(
                                                reapply_fields[
                                                    "forced_sync_reapply_errors"
                                                ]
                                            ),
                                        }
                                    )
                                elif skip_post_restore_verification:
                                    scheduler.set_prev_manifest(None)  # FC restore
                                    forced_sync_fields.update(
                                        {
                                            "forced_sync_verified": None,
                                            "forced_sync_verify_reason": (
                                                "reapplied_actions_unverifiable"
                                            ),
                                            "forced_sync_verification": {
                                                "restore_verified": (
                                                    restore_verification_fields[
                                                        "forced_sync_verified"
                                                    ]
                                                ),
                                                "restore_verification": (
                                                    restore_verification_fields[
                                                        "forced_sync_verification"
                                                    ]
                                                ),
                                                "reason": (
                                                    "reapplied_actions_unverifiable"
                                                ),
                                            },
                                        }
                                    )
                                else:
                                    forced_sync_fields.update(
                                        restore_verification_fields
                                    )
                            if (
                                forced_sync_fields.get("forced_sync_success") is True
                                and not skip_post_restore_verification
                            ):
                                try:
                                    source_entries = (
                                        _fold_source_checkpoint_entries_through_action(
                                            actions=loaded.actions,
                                            target_index=action_index,
                                            source_trace=loaded.source_trace,
                                        )
                                    )
                                    verification_fields, replay_entries = (
                                        await _verify_forced_sync_restore_state(
                                            container=ctr,
                                            checkpoint_spec=checkpoint_spec,
                                            source_entries=source_entries,
                                            context="forced_sync_verification",
                                        )
                                    )
                                    if replay_entries is None:
                                        scheduler.set_prev_manifest(None)  # FC restore
                                        forced_sync_fields.update(verification_fields)
                                    else:
                                        scheduler.set_prev_manifest(replay_entries)
                                        forced_sync_fields.update(verification_fields)
                                        if not verification_fields[
                                            "forced_sync_verification"
                                        ]["cas_manifest_match"]:
                                            forced_sync_fields.update(
                                                {
                                                    "forced_sync_success": False,
                                                    "forced_sync_continued": False,
                                                    "forced_sync_status": (
                                                        "verification_failed"
                                                    ),
                                                }
                                            )
                                except Exception as exc:
                                    logger.warning(
                                        "Forced sync verification failed for %s "
                                        "action=%s: %s",
                                        loaded.agent_id,
                                        action_id,
                                        exc,
                                    )
                                    scheduler.set_prev_manifest(None)  # FC restore
                                    forced_sync_fields.update(
                                        _forced_sync_verification_unavailable_fields(
                                            f"{type(exc).__name__}: {exc}",
                                        )
                                    )
                        except Exception as exc:
                            logger.exception(
                                "Forced sync failed for %s action=%s",
                                loaded.agent_id,
                                action_id,
                            )
                            forced_sync_fields.update(
                                {
                                    "forced_sync_success": False,
                                    "forced_sync_continued": False,
                                    "forced_sync_status": "checkpoint_restore_failed",
                                    "forced_sync_error": f"{type(exc).__name__}: {exc}",
                                }
                            )
                    forced_sync_success = (
                        forced_sync_fields.get("forced_sync_success") is True
                    )
                    forced_sync_fields.setdefault("forced_sync_success", False)
                    forced_sync_fields.setdefault(
                        "forced_sync_status",
                        "checkpoint_restored_continuation"
                        if forced_sync_success
                        else "checkpoint_restore_failed",
                    )
                    forced_sync_fields.setdefault(
                        "forced_sync_continued",
                        forced_sync_success
                        and forced_sync_fields.get("forced_sync_status")
                        == "checkpoint_restored_continuation",
                    )
                    forced_sync_fields["forced_sync_elapsed_ms"] = round(
                        (time.monotonic() - forced_sync_started) * 1000,
                        3,
                    )
            # Flush the pending boundary record now that forced-sync has
            # resolved (tool-result mismatch path).  For clean boundaries
            # the flush happens at the next Hook 2 drain or close().
            if (
                replay_scheduler_config.checkpoint_scheduling == "deferred"
                and forced_sync_fields
            ):
                scheduler.flush_pending()
            extra_tool_fields = _command_metadata(
                tool_name=tool_name,
                tool_args_json=tool_args,
                tool_result=str(tool_result),
                tool_success=tool_success,
                returncode=tool_exec_metadata.get("returncode"),
            )
            extra_tool_fields["source_action_id"] = action_id
            if isinstance(source_returncode, int) and not isinstance(
                source_returncode,
                bool,
            ):
                extra_tool_fields["source_returncode"] = source_returncode
            if isinstance(replay_returncode, int) and not isinstance(
                replay_returncode,
                bool,
            ):
                extra_tool_fields["replay_returncode"] = replay_returncode
            if isinstance(source_timed_out, bool):
                extra_tool_fields["source_timed_out"] = source_timed_out
            if isinstance(replay_timed_out, bool):
                extra_tool_fields["replay_timed_out"] = replay_timed_out
            if oracle_verdict is not None:
                extra_tool_fields["oracle_semantic_match"] = (
                    oracle_verdict.semantic_match
                )
                extra_tool_fields["oracle_tier"] = oracle_verdict.tier
                extra_tool_fields["oracle_category"] = oracle_verdict.category
                extra_tool_fields["mismatch_oracle"] = {
                    "tier": oracle_verdict.tier,
                    "category": oracle_verdict.category,
                    "semantic_match": oracle_verdict.semantic_match,
                    "evidence": oracle_verdict.evidence,
                }
            # Disk-hash comparison: diagnostic signal for FC backends.
            # Stores the replay-side disk hash for post-hoc analysis.
            # Source comparision is future work once collection-side
            # disk_hash recording is implemented.
            if (
                ctr is not None
                and isinstance(ctr.backend, FCBackend)
                and action_index in ctr.replay_snapshots
            ):
                try:
                    replay_snap = ctr.replay_snapshots[action_index]
                    replay_hash = (
                        replay_snap.process_state.get("disk_hash")
                        if replay_snap.process_state
                        else None
                    )
                    if replay_hash is not None:
                        extra_tool_fields["replay_disk_hash"] = replay_hash[:16]
                except Exception:
                    pass
            if output_diff_snippet is not None:
                extra_tool_fields["output_diff_snippet"] = output_diff_snippet
            if _tool_uses_exec_semantics(tool_name, tool_args):
                extra_tool_fields["replay_env_parity"] = (
                    prepared_session.replay_exec_env_parity
                )
                assert normalized_output_match is not None
                extra_tool_fields["normalized_output_match"] = normalized_output_match
            if original_artifact_path is not None:
                extra_tool_fields["source_artifact_path"] = original_artifact_path
            if mapped_artifact_path is not None:
                extra_tool_fields["simulator_artifact_path"] = mapped_artifact_path
            if effective_mismatch_reason is not None:
                extra_tool_fields["mismatch_reason"] = effective_mismatch_reason
            extra_tool_fields.update(tool_exec_metadata)
            extra_tool_fields.update(cas_manifest_fields)
            extra_tool_fields.update(forced_sync_fields)
            if source_exec_timeout is not None:
                extra_tool_fields["source_exec_timeout_s"] = source_exec_timeout
                extra_tool_fields["replay_exec_timeout_s"] = replay_exec_timeout
            if source_resource_timeline is not None:
                extra_tool_fields["source_resource_timeline"] = source_resource_timeline
                extra_tool_fields["resource_timeout_policy"] = (
                    "resource_integrated"
                    if exec_resource_timeline is not None
                    else "wall_clock"
                )
            if source_is_subagent_lane:
                extra_tool_fields["source_lane_agent_id"] = source_action_agent_id
            sim_metrics: dict[str, Any] = {
                "warmup": iteration < warmup_skip_iterations,
                "source": replay_source,
                "sim_tool_format": replay_source
                if replay_source
                in {
                    "skipped_host_mode",
                    "message_noop",
                    "replayed_from_trace",
                    "denied_command_replayed_from_trace",
                    "source_artifact_unavailable",
                    "restored_runtime_artifact",
                    "subagent_lane_replay",
                }
                else "container_exec",
                **_sleep_drift_metrics(
                    source_gap=source_gap_sleep,
                    action_sleep=action_sleep,
                ),
            }
            if data.get("source_concurrent_execs") is True:
                sim_metrics["concurrent_source_execs"] = True
            tool_record = _make_trace_action(
                loaded=loaded,
                action_type="tool_exec",
                action_id=_replay_action_id(
                    action,
                    source_agent_id=loaded.source_agent_id,
                    fallback=f"tool_{iteration}_{tool_name}",
                ),
                iteration=iteration,
                ts_start=record_ts_start,
                ts_end=record_ts_end,
                data={
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "tool_result": tool_result,
                    "duration_ms": duration_ms,
                    "success": tool_success,
                    "source_success": source_success,
                    "replay_outcome_match": replay_outcome_match,
                    **extra_tool_fields,
                    "simulate_source": str(loaded.source_trace),
                    "source_duration_ms": source_duration_ms,
                    "replay_mode": "cloud_model",
                    "replay_speed": replay_speed,
                    "replay_source": replay_source,
                    "sim_metrics": sim_metrics,
                },
            )
            scheduler.log_or_defer(loaded.agent_id, tool_record)  # Hook 4
            forced_sync_attempted = (
                forced_sync_fields.get("forced_sync_attempted") is True
            )
            forced_sync_success = forced_sync_fields.get("forced_sync_success") is True
            forced_sync_continued = (
                forced_sync_fields.get("forced_sync_continued") is True
            )
            if forced_sync_attempted:
                forced_sync_attempts += 1
            if forced_sync_success:
                forced_sync_actions += 1
                forced_sync_successes += 1
            if forced_sync_continued:
                forced_sync_continued_actions += 1
            if replay_outcome_match:
                if tool_success:
                    succeeded_actions += 1
                else:
                    logger.info(
                        "Replay tool action matched source failure for %s action=%s tool=%s",
                        loaded.agent_id,
                        action_id,
                        tool_name,
                    )
            else:
                outcome_mismatches += 1
                if not forced_sync_continued:
                    unresolved_mismatches += 1
                if forced_sync_continued:
                    logger.warning(
                        "Replay mismatch checkpoint-restored for %s action=%s "
                        "tool=%s reason=%s status=%s",
                        loaded.agent_id,
                        action_id,
                        tool_name,
                        effective_mismatch_reason,
                        forced_sync_fields.get("forced_sync_status"),
                    )
                else:
                    logger.error(
                        "Replay tool outcome mismatch for %s action=%s tool=%s "
                        "source_success=%s replay_success=%s status=%s",
                        loaded.agent_id,
                        action_id,
                        tool_name,
                        source_success,
                        tool_success,
                        forced_sync_fields.get("forced_sync_status"),
                    )
                if replay_source == "source_artifact_unavailable":
                    fatal_replay_errors += 1
        except Exception as exc:
            logger.error(
                "Replay action failed for %s action=%s: %s",
                loaded.agent_id,
                action_id,
                exc,
            )
            replay_action_errors += 1

    wall_end = time.time()
    failed_actions = outcome_mismatches + replay_action_errors
    success = failed_actions == 0 and fatal_replay_errors == 0

    # Session end: drain in-flight work, flush deferred records
    scheduler_aggregates = await scheduler.close()

    trace_logger.log_summary(
        loaded.agent_id,
        _make_trace_summary(
            loaded=loaded,
            success=success,
            elapsed_s=wall_end - wall_start,
            source_model=source_model,
            extra={
                "replay_mode": "cloud_model",
                "replay_speed": replay_speed,
                "llm_timing_mode": llm_timing.mode,
                "replay_env_parity": prepared_session.replay_task_env_parity,
                "succeeded_actions": succeeded_actions,
                "failed_actions": failed_actions,
                "source_failed_actions": source_failed_actions,
                "replay_failed_actions": replay_failed_actions,
                "matched_failed_actions": matched_failed_actions,
                "fatal_replay_errors": fatal_replay_errors,
                "replay_action_errors": replay_action_errors,
                "forced_sync_actions": forced_sync_actions,
                "forced_sync_attempts": forced_sync_attempts,
                "forced_sync_successes": forced_sync_successes,
                "forced_sync_continued": forced_sync_continued_actions,
                "outcome_mismatches": outcome_mismatches,
                "unresolved_mismatches": unresolved_mismatches,
                "sleep_drift": _summarize_sleep_drifts(sleep_drifts),
                "checkpoint_scheduling": {
                    "mode": replay_scheduler_config.checkpoint_scheduling,
                },
                "scheduler_metrics": scheduler_aggregates,
            },
        ),
    )
    return _make_task_stats(
        loaded=loaded,
        success=success,
        elapsed_s=wall_end - wall_start,
        failed_action_count=failed_actions,
        replay_env_parity=prepared_session.replay_task_env_parity,
        scheduler_metrics=scheduler_aggregates,
    )


def _split_trace_by_agent(
    combined_path: Path,
    sessions: list[PreparedTraceSession],
) -> None:
    """Write per-task trace.jsonl from the combined JSONL, filtered by replay id."""
    agent_dirs = {
        s.loaded.run_instance_id: s.task_output_dir
        for s in sessions
        if s.task_output_dir is not None
    }
    sessions_by_agent = {s.loaded.run_instance_id: s for s in sessions}
    if not agent_dirs:
        return

    per_agent: dict[str, list[str]] = {aid: [] for aid in agent_dirs}
    metadata_line: str | None = None

    with combined_path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            rtype = record.get("type")
            if rtype == "trace_metadata":
                metadata_line = stripped
                continue
            agent_id = record.get("agent_id")
            if agent_id in per_agent:
                per_agent[agent_id].append(stripped)

    for agent_id, lines in per_agent.items():
        out_dir = agent_dirs[agent_id]
        out_path = out_dir / "trace.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            if metadata_line:
                metadata = json.loads(metadata_line)
                session = sessions_by_agent[agent_id].loaded
                metadata["scaffold"] = session.scaffold
                metadata["execution_environment"] = _execution_environment(session)
                metadata["instance_id"] = session.run_instance_id
                metadata["run_instance_id"] = session.run_instance_id
                metadata["source_agent_id"] = session.source_agent_id
                metadata["task_id"] = session.source_agent_id
                metadata["manifest_index"] = session.manifest_index
                metadata["label"] = session.label
                metadata["source_trace"] = str(session.source_trace)
                metadata["source_trace_count"] = 1
                metadata["source_traces"] = [str(session.source_trace)]
                metadata["source_trace_entries"] = [
                    {
                        "manifest_index": session.manifest_index,
                        "source_trace": str(session.source_trace),
                        "source_agent_id": session.source_agent_id,
                        "run_instance_id": session.run_instance_id,
                        "label": session.label,
                    }
                ]
                metadata["source_agent_ids"] = [session.source_agent_id]
                metadata["run_instance_ids"] = [session.run_instance_id]
                source_model = (session.summary or {}).get("model", "unknown")
                metadata["source_models"] = [source_model]
                metadata["source_model"] = source_model
                fh.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            for ln in lines:
                fh.write(ln + "\n")
        logger.info("Wrote per-task trace (%d records) → %s", len(lines), out_path)


def _worker_task_output_dirs(
    worker_results: list[WorkerReplayResult],
) -> dict[str, Path]:
    task_dirs: dict[str, Path] = {}
    for result in worker_results:
        for agent_id, path in result.task_output_dirs.items():
            task_dirs[agent_id] = Path(path)
    return task_dirs


def _split_combined_worker_trace_by_agent(
    *,
    combined_path: Path,
    sessions: list[LoadedTraceSession],
    worker_results: list[WorkerReplayResult],
) -> None:
    task_dirs = _worker_task_output_dirs(worker_results)
    prepared_sessions: list[PreparedTraceSession] = []
    for session in sessions:
        task_output_dir = task_dirs.get(session.run_instance_id)
        if task_output_dir is None:
            continue
        prepared_sessions.append(
            PreparedTraceSession(loaded=session, task_output_dir=task_output_dir)
        )
    _split_trace_by_agent(combined_path, prepared_sessions)


def _write_combined_worker_trace(
    *,
    trace_file: Path,
    worker_results: list[WorkerReplayResult],
    sessions: list[LoadedTraceSession],
    mode: str,
    replay_speed: float,
    llm_timing: LLMTimingConfig,
    manifest: Path,
    concurrency: int,
    workers: int,
    prep_concurrency: int,
    network_mode: str,
    model: str | None,
    monitoring_policy: dict[str, object] | None,
) -> None:
    """Concatenate worker JSONL files behind one global metadata header."""
    if trace_file.exists():
        trace_file.unlink()
    trace_logger = TraceLogger(trace_file.parent, trace_file.stem)
    try:
        _log_trace_metadata(
            trace_logger=trace_logger,
            mode=mode,
            sessions=sessions,
            replay_speed=replay_speed,
            llm_timing=llm_timing,
            manifest=manifest,
            concurrency=concurrency,
            scheduler_mode="multi_process_workers",
            api_base=None,
            model=model,
            network_mode=network_mode,
            extra={
                "workers": workers,
                "prep_concurrency": prep_concurrency,
                "effective_workers": min(workers, len(sessions)),
                "worker_trace_files": [result.trace_file for result in worker_results],
                "monitoring": monitoring_policy or {},
            },
        )
    finally:
        trace_logger.close()

    records: list[tuple[tuple[float, int, int], dict[str, Any]]] = []
    sequence = 0
    for result in worker_results:
        worker_path = Path(result.trace_file)
        if not worker_path.exists():
            raise SimulateError(f"worker trace does not exist: {worker_path}")
        with worker_path.open(encoding="utf-8") as in_fh:
            for line in in_fh:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SimulateError(
                        f"invalid worker trace JSONL: {worker_path}"
                    ) from exc
                if record.get("type") == "trace_metadata":
                    continue
                records.append((_combined_trace_sort_key(record, sequence), record))
                sequence += 1

    with trace_file.open("a", encoding="utf-8") as out_fh:
        for _sort_key, record in sorted(records, key=lambda item: item[0]):
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _combined_trace_sort_key(record: dict[str, Any], sequence: int) -> tuple[float, int, int]:
    rtype = record.get("type")
    if rtype == "action":
        return (_float_sort_value(record.get("ts_start"), default=float("inf")), 0, sequence)
    if rtype == "event":
        return (_float_sort_value(record.get("ts")), 1, sequence)
    if rtype == "summary":
        return (_float_sort_value(record.get("ts"), default=float("inf")), 2, sequence)
    return (_float_sort_value(record.get("ts"), default=float("inf")), 3, sequence)


def _float_sort_value(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


async def simulate(
    *,
    manifest: Path,
    task_source: Path,
    output_dir: Path,
    mode: str = "cloud_model",
    concurrency: int = 1,
    workers: int = 1,
    prep_concurrency: int = 0,
    sandbox_backend: str = "docker",
    checkpoint_backend: str | None = None,
    container_executable: str | None = None,
    network_mode: str = "host",
    api_base: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    command_timeout_s: float = 120.0,
    warmup_skip_iterations: int = 0,
    replay_speed: float = 1.0,
    resource_monitoring: MonitoringMode = "auto",
    pmu_monitoring: MonitoringMode = "auto",
    memory_bandwidth_monitoring: MonitoringMode = "auto",
    llm_timing_mode: str = "source_scaled",
    llm_ttft_ms: float | None = None,
    llm_tpot_ms: float | None = None,
    checkpoint_scheduling: str = "sync",
    structured_output: bool = False,
) -> Path:
    if mode != "cloud_model":
        raise ValueError(f"Unsupported simulate mode: {mode}")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if prep_concurrency < 0:
        raise ValueError("prep_concurrency must be >= 0")
    if checkpoint_scheduling not in ("sync", "deferred"):
        raise ValueError(
            f"checkpoint_scheduling must be sync or deferred, "
            f"got {checkpoint_scheduling!r}"
        )
    replay_scheduler_config = ReplaySchedulerConfig(
        checkpoint_scheduling=checkpoint_scheduling,
    )
    if mode != "cloud_model":
        raise ValueError(f"Unsupported simulate mode: {mode}")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if prep_concurrency < 0:
        raise ValueError("prep_concurrency must be >= 0")
    llm_timing = LLMTimingConfig(
        mode=llm_timing_mode,
        ttft_ms=llm_ttft_ms,
        tpot_ms=llm_tpot_ms,
    )
    _validate_llm_timing_config(llm_timing)

    manifest_entries = _load_simulate_manifest(
        manifest,
        default_task_source=task_source.resolve(),
        default_sandbox_backend=sandbox_backend,
        default_checkpoint_backend=checkpoint_backend,
    )

    loaded_sessions = [
        _load_trace_session(
            entry.trace,
            entry.task_source,
            manifest_index=entry.index,
            docker_image_override=entry.docker_image,
            label=entry.label,
            sandbox_backend=entry.sandbox_backend,
            checkpoint_backend=entry.checkpoint_backend,
        )
        for entry in manifest_entries
    ]
    _assign_replay_instance_ids(loaded_sessions)
    _validate_loaded_sessions(
        loaded_sessions,
        mode=mode,
        replay_speed=replay_speed,
        llm_timing=llm_timing,
    )
    _validate_container_runtime(
        loaded_sessions,
        container_executable=container_executable,
    )
    monitoring_policy = resolve_simulate_monitoring(
        resource=resource_monitoring,
        pmu=pmu_monitoring,
        memory_bandwidth=memory_bandwidth_monitoring,
        concurrency=concurrency,
        workers=workers,
        has_container_session=_has_container_mode_sessions(loaded_sessions),
        has_host_session=any(_is_host_mode(session) for session in loaded_sessions),
    )
    monitoring_policy_dict = monitoring_policy.to_dict()
    await _prefetch_container_images(
        loaded_sessions,
        container_executable=container_executable,
    )

    output_path = Path(output_dir)
    if structured_output:
        output_path = output_path / _structured_output_subdir(
            loaded_sessions,
            concurrency=concurrency,
            workers=workers,
        )

    prepared_sessions: list[PreparedTraceSession] = []
    task_stats: list[ReplayTaskStats] = []
    trace_logger: TraceLogger | None = None
    container_resource_recorder: ContainerResourceRecorder | None = None
    container_resource_summary: dict[str, Any] | None = None
    sweep_fixed_images: dict[str, str] = {}
    run_completed_for_fixed_cleanup = False
    run_wall_start: float | None = None
    run_wall_end: float | None = None
    output_path.mkdir(parents=True, exist_ok=True)
    scheduler_mode = "bounded_queue" if workers == 1 else "multi_process_workers"

    try:
        sweep_fixed_images = await _prebuild_sweep_fixed_images(
            loaded_sessions,
            output_path=output_path,
            container_executable=container_executable,
        )
        run_wall_start = time.monotonic()
        run_id = _build_run_id(mode=mode, model=model, concurrency=concurrency)
        if workers == 1:
            trace_path = output_path / f"{run_id}.jsonl"
            if trace_path.exists():
                trace_path.unlink()
            trace_logger = TraceLogger(output_path, run_id)
            _log_trace_metadata(
                trace_logger=trace_logger,
                mode=mode,
                sessions=loaded_sessions,
                replay_speed=replay_speed,
                llm_timing=llm_timing,
                manifest=manifest,
                concurrency=concurrency,
                scheduler_mode=scheduler_mode,
                api_base=None,
                model=model,
                network_mode=network_mode,
                extra={
                    "workers": workers,
                    "prep_concurrency": prep_concurrency,
                    "monitoring": monitoring_policy_dict,
                },
            )
            if monitoring_policy.global_container_resource_enabled:
                if container_executable is None:
                    raise AssertionError("container_executable required for monitoring")
                container_resource_recorder = ContainerResourceRecorder(
                    output_dir=output_path,
                    run_id=run_id,
                    interval_s=GLOBAL_CONTAINER_RESOURCE_SAMPLE_INTERVAL_S,
                    executable=container_executable,
                    sample_all_containers=False,
                    collect_cgroup_memory_access=monitoring_policy.pmu_enabled,
                    monitoring_policy=monitoring_policy_dict,
                )
                container_resource_recorder.start()

            assert trace_logger is not None
            prepared_sessions, task_stats = await _run_cloud_model_queue(
                loaded_sessions,
                output_path=output_path,
                trace_logger=trace_logger,
                concurrency=concurrency,
                container_executable=container_executable,
                network_mode=network_mode,
                container_resource_recorder=container_resource_recorder,
                replay_speed=replay_speed,
                llm_timing=llm_timing,
                command_timeout_s=command_timeout_s,
                warmup_skip_iterations=warmup_skip_iterations,
                fixed_images_by_source=sweep_fixed_images,
                resource_monitoring_enabled=monitoring_policy.per_task_resource_enabled,
                memory_bandwidth_enabled=monitoring_policy.memory_bandwidth_enabled,
                monitoring_policy=monitoring_policy_dict,
                replay_scheduler_config=replay_scheduler_config,
            )
        else:
            worker_results, task_stats = await _run_cloud_model_worker_waves(
                [_worker_trace_input(session) for session in loaded_sessions],
                output_path=output_path,
                run_id=run_id,
                concurrency=concurrency,
                workers=workers,
                prep_concurrency=prep_concurrency,
                container_executable=container_executable,
                network_mode=network_mode,
                replay_speed=replay_speed,
                llm_timing=llm_timing,
                command_timeout_s=command_timeout_s,
                warmup_skip_iterations=warmup_skip_iterations,
                fixed_images_by_source=sweep_fixed_images,
                resource_monitoring_enabled=monitoring_policy.per_task_resource_enabled,
                memory_bandwidth_enabled=monitoring_policy.memory_bandwidth_enabled,
                monitoring_policy=monitoring_policy_dict,
                replay_scheduler_config=replay_scheduler_config,
            )
            container_resource_summary = {
                "status": "disabled",
                "reason": "multi_process_workers_use_per_task_resources"
                if monitoring_policy.resource_enabled
                else "disabled_by_monitoring_policy",
                "sample_count": 0,
                "monitoring": monitoring_policy_dict,
            }
            combined_trace_file = output_path / f"{run_id}.jsonl"
            _write_combined_worker_trace(
                trace_file=combined_trace_file,
                worker_results=worker_results,
                sessions=loaded_sessions,
                mode=mode,
                replay_speed=replay_speed,
                llm_timing=llm_timing,
                manifest=manifest,
                concurrency=concurrency,
                workers=workers,
                prep_concurrency=prep_concurrency,
                network_mode=network_mode,
                model=model,
                monitoring_policy=monitoring_policy_dict,
            )
            _split_combined_worker_trace_by_agent(
                combined_path=combined_trace_file,
                sessions=loaded_sessions,
                worker_results=worker_results,
            )
        run_completed_for_fixed_cleanup = True
    finally:
        finalization_error: BaseException | None = None
        try:
            try:
                if trace_logger is not None:
                    trace_logger.close()
                    _split_trace_by_agent(trace_logger.path, prepared_sessions)
                for prepared in prepared_sessions:
                    await _finalize_prepared_session(prepared)
            except (Exception, asyncio.CancelledError) as exc:
                finalization_error = exc
            if finalization_error is None:
                run_wall_end = time.monotonic()
            if container_resource_recorder is not None:
                container_resource_summary = container_resource_recorder.stop()
            if run_completed_for_fixed_cleanup and finalization_error is None:
                await _cleanup_sweep_fixed_images(
                    sweep_fixed_images,
                    container_executable=container_executable,
                )
            elif sweep_fixed_images:
                logger.warning(
                    "Skipping sweep fixed image cleanup because simulate did not "
                    "complete cleanly; images=%s",
                    sorted(sweep_fixed_images.values()),
                )
        finally:
            if finalization_error is not None:
                raise finalization_error

    if run_wall_start is None or run_wall_end is None:
        raise AssertionError("simulate wall-clock measurement was not recorded")
    trace_file = output_path / f"{run_id}.jsonl"
    _write_throughput_summary(
        output_path=output_path,
        run_id=run_id,
        manifest=manifest,
        mode=mode,
        concurrency=concurrency,
        scheduler_mode=scheduler_mode,
        llm_timing=llm_timing,
        workers=workers,
        prep_concurrency=prep_concurrency,
        trace_file=trace_file,
        wall_time_s=run_wall_end - run_wall_start,
        task_stats=task_stats,
        container_resources=container_resource_summary,
        monitoring_policy=monitoring_policy_dict,
    )
    logger.info("Simulate complete [%s] -> %s", mode, trace_file)
    return trace_file
