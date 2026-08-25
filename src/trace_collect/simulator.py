from __future__ import annotations

import asyncio
import dataclasses
import functools
import hashlib
import json
import logging
import multiprocessing
import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import BrokenBarrierError
from typing import Any

from harness.container_image_prep import (
    drop_cached_fixed_image,
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
from trace_collect.resource_timeline import valid_resource_timeline
from trace_collect.monitoring import MonitoringMode, resolve_simulate_monitoring
from trace_collect.attempt_pipeline import (
    configure_task_container_apt_mirror,
    start_task_container,
    stop_task_container,
)
from trace_collect.runtime.task_container import (
    resolve_running_container_exec_config,
    resolve_task_container_exec_config,
)
from trace_collect.openclaw_host_runtime import (
    ShadowGenerationConfig,
    ShadowGenerationMode,
    llm_replay_duration_s,
    shadow_generation_payload,
    validate_llm_replay_timing,
)
from trace_collect.simulate_manifest import (
    _assign_replay_instance_ids,
    _load_simulate_manifest,
    _load_trace_session,
    _load_worker_trace_inputs,
    _worker_trace_input,
)
from trace_collect.simulate_openclaw import (
    _run_openclaw_replay_session,
    replay_exec_timeout_floor_s,
    replay_trace_tools_enabled,
)
from trace_collect.simulate_outputs import (
    _assign_task_output_dir,
    _build_run_id,
    _log_trace_metadata,
    _make_task_stats,
    _make_trace_action,
    _make_trace_summary,
    _replay_agent_id_for_action,
    _split_combined_worker_trace_by_agent,
    _split_trace_by_agent,
    _structured_output_subdir,
    _write_combined_worker_trace,
    _write_prepared_resources,
    _write_throughput_summary,
)
from trace_collect.simulate_types import (
    LLMTimingConfig,
    LoadedTraceSession,
    PreparedContainer,
    PreparedTraceSession,
    ReplayTaskStats,
    SimulateError,
    SleepDrift,
    TraceManifestEntry,
    WorkerReplayResult,
    WorkerTraceInput,
)
from trace_collect.simulate_utils import (
    _coerce_action_bounds,
    _exception_payload,
    _is_host_mode,
    _is_terminal_bench_registry_task,
    _requires_task_container,
    _resolve_docker_image,
    _resolve_prep_concurrency,
    _sanitize_run_label,
    _sleep_drift_metrics,
    _source_model,
    _summarize_sleep_drifts,
    _utc_now_iso,
)
from trace_collect.tool_gap_loan import (
    ToolGapLoanConfig,
    ToolGapPrediction,
)

_TOOL_RESOURCE_RUN_TOKENS_ENV = "TOOL_RESOURCE_RUN_TOKENS"


def _source_exec_commands(session: LoadedTraceSession) -> list[str]:
    commands: list[str] = []
    for action in session.actions:
        if action.get("action_type") != "tool_exec":
            continue
        data = action.get("data")
        if not isinstance(data, dict) or data.get("tool_name") != "exec":
            continue
        arguments = data.get("tool_args")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        if isinstance(arguments, dict) and isinstance(arguments.get("command"), str):
            commands.append(arguments["command"])
    return commands


def _load_tool_gap_predictions(
    path: Path,
    sessions: list[LoadedTraceSession],
) -> dict[str, tuple[ToolGapPrediction, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("tool-gap prediction file must be a task-keyed object")
    expected_ids = {session.task_instance_id for session in sessions}
    if set(payload) != expected_ids:
        raise ValueError("tool-gap prediction task IDs must exactly match the manifest")
    allowed_fields = {
        "sample_id",
        "command",
        "probability_by_bucket",
        "hard_bucket",
        "provenance",
    }
    result: dict[str, tuple[ToolGapPrediction, ...]] = {}
    for session in sessions:
        raw_predictions = payload[session.task_instance_id]
        if not isinstance(raw_predictions, list):
            raise ValueError("tool-gap task predictions must be a list")
        predictions: list[ToolGapPrediction] = []
        for raw in raw_predictions:
            if not isinstance(raw, dict) or set(raw) != allowed_fields:
                raise ValueError("tool-gap prediction fields do not match the schema")
            if (
                not isinstance(raw["sample_id"], str)
                or not isinstance(raw["command"], str)
                or not isinstance(raw["probability_by_bucket"], list)
                or not isinstance(raw["hard_bucket"], int)
                or isinstance(raw["hard_bucket"], bool)
                or not isinstance(raw["provenance"], dict)
            ):
                raise ValueError("tool-gap prediction has invalid field types")
            predictions.append(
                ToolGapPrediction(
                    sample_id=raw["sample_id"],
                    command=raw["command"],
                    probability_by_bucket=tuple(raw["probability_by_bucket"]),
                    hard_bucket=raw["hard_bucket"],
                    provenance=raw["provenance"],
                )
            )
        source_commands = _source_exec_commands(session)
        cursor = 0
        for prediction in predictions:
            while cursor < len(source_commands) and source_commands[cursor] != prediction.command:
                cursor += 1
            if cursor == len(source_commands):
                raise ValueError(
                    "tool-gap prediction commands must be an ordered source-exec subsequence"
                )
            cursor += 1
        result[session.task_instance_id] = tuple(predictions)
    return result


def _tool_resource_scope(loaded: LoadedTraceSession) -> str:
    repo = loaded.task.get("repo")
    if isinstance(repo, str) and repo.strip():
        return repo.strip()
    return f"task:{loaded.task_instance_id}"


logger = logging.getLogger(__name__)
GLOBAL_CONTAINER_RESOURCE_SAMPLE_INTERVAL_S = 1.0
_SHARED_SEMAPHORE_POLL_S = 0.05
_REPLAY_START_DELAY_S = 0.1
__all__ = (
    "LLMTimingConfig",
    "LoadedTraceSession",
    "PreparedContainer",
    "PreparedTraceSession",
    "ReplayTaskStats",
    "SimulateError",
    "SleepDrift",
    "TraceManifestEntry",
    "WorkerReplayResult",
    "WorkerTraceInput",
    "simulate",
)


class _ReplayPreparationError(RuntimeError):
    def __init__(
        self,
        prepared: PreparedTraceSession,
        reason: str,
        cause: RuntimeError,
        cleanup_error: BaseException | None = None,
    ) -> None:
        super().__init__(str(cause))
        self.prepared = prepared
        self.reason = reason
        self.cause = cause
        self.cleanup_error = cleanup_error


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
            "task_instance_id": self.loaded.task_instance_id,
            "source_action_agent_id": self.loaded.source_action_agent_id,
            "source_agent_id": self.loaded.source_action_agent_id,
            "task_id": self.loaded.task_instance_id,
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
    (
        tool_result,
        tool_success,
        inner_duration_ms,
        tool_metadata,
    ) = await execute_trace_tool_detailed(
        agent=agent,
        tool_name=tool_name,
        tool_args_json=tool_args_json,
        command_timeout_s=command_timeout_s,
        source_exec_timeout_s=source_exec_timeout_s,
        allow_source_runtime_artifacts=allow_source_runtime_artifacts,
        source_resource_timeline=source_resource_timeline,
    )
    wall_duration_ms = (time.monotonic() - t0) * 1000
    # Prefer agent-side timing to exclude pipe transfer overhead
    duration_ms = (
        inner_duration_ms if inner_duration_ms is not None else wall_duration_ms
    )
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


_REPLAY_EXECUTION_FAILURE_KINDS = frozenset(
    {
        "malformed_replay_exec_response",
        "source_runtime_artifact_unavailable",
        "unsupported_replay_tool",
    }
)
_CONTROL_PLANE_NOOP_TOOLS = frozenset({"message", "spawn", "sessions_yield"})
_CONTROL_PLANE_NOOP_RESULTS = {
    "message": "Message replayed as no-op",
    "spawn": "Subagent spawn replayed as no-op",
    "sessions_yield": "Session yield replayed as no-op",
}


@dataclass(slots=True)
class _ReplayActionOutcome:
    action_id: str
    succeeded_actions: int = 0
    replay_action_errors: int = 0
    fatal_replay_errors: int = 0
    source_failed_actions: int = 0
    replay_failed_actions: int = 0
    replay_execution_errors: int = 0
    unexpected_replay_failed_actions: int = 0
    sleep_drifts: list[SleepDrift] = dataclasses.field(default_factory=list)


def _replay_failure_kind(metadata: dict[str, Any]) -> str | None:
    value = metadata.get("replay_failure_kind")
    if isinstance(value, str) and value:
        return value
    return None


def _source_tool_success(data: dict[str, Any]) -> bool:
    raw_success = data.get("success")
    if isinstance(raw_success, bool):
        return raw_success
    if raw_success is None:
        return not bool(data.get("error"))
    raise ValueError(f"tool success must be boolean when present, got {raw_success!r}")


def _command_exit_code(tool_result: str) -> int | None:
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
    return isinstance(payload, dict) and ("command" in payload or "commands" in payload)


def _tool_uses_single_exec_command_semantics(
    tool_name: str | None,
    tool_args_json: Any,
) -> bool:
    payload = _exec_semantics_payload(tool_name, tool_args_json)
    if payload is None:
        return False
    return "command" in payload and "commands" not in payload


def _command_metadata(
    *,
    tool_name: str | None,
    tool_args_json: Any,
    tool_result: str,
    tool_success: bool,
) -> dict[str, Any]:
    if not _tool_uses_exec_semantics(tool_name, tool_args_json):
        return {}
    exit_code = _command_exit_code(tool_result)
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


def _source_exec_timeout_s(
    *,
    tool_name: str | None,
    tool_args_json: Any,
    source_duration_ms: float,
    source_success: bool,
    source_tool_result: Any,
) -> float | None:
    if source_success or source_duration_ms <= 0:
        return None
    if not _tool_uses_exec_semantics(tool_name, tool_args_json):
        return None

    source_tool_result_text = str(source_tool_result or "")
    if (
        "Error: Command timed out after " not in source_tool_result_text
        and not _is_replay_wrapper_timeout_result(source_tool_result_text)
    ):
        return None
    return max(0.001, source_duration_ms / 1000.0)


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


def _replay_fixed_image_name(
    *,
    source_image: str,
    agent_id: str,
    task_output_dir: Path,
) -> str:
    label = _sanitize_run_label(agent_id).lower()[:64]
    digest = hashlib.sha1(str(task_output_dir.resolve()).encode("utf-8")).hexdigest()[
        :12
    ]
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


def _has_session_dependencies(sessions: list[LoadedTraceSession]) -> bool:
    return any(session.depends_on for session in sessions)


def _validate_session_dependencies(sessions: list[LoadedTraceSession]) -> None:
    if not _has_session_dependencies(sessions):
        return

    task_counts: dict[str, int] = {}
    for session in sessions:
        task_counts[session.task_instance_id] = (
            task_counts.get(session.task_instance_id, 0) + 1
        )
    duplicate_task_ids = sorted(
        task_id for task_id, count in task_counts.items() if count > 1
    )
    if duplicate_task_ids:
        raise SimulateError(
            "Dependency metadata is ambiguous with duplicate task ids: "
            + ", ".join(repr(task_id) for task_id in duplicate_task_ids)
        )

    known_task_ids = set(task_counts)
    graph: dict[str, tuple[str, ...]] = {}
    for session in sessions:
        missing = [dep for dep in session.depends_on if dep not in known_task_ids]
        if missing:
            raise SimulateError(
                f"Task {session.task_instance_id!r} depends on missing task ids: "
                + ", ".join(repr(dep) for dep in missing)
            )
        if session.task_instance_id in session.depends_on:
            raise SimulateError(f"Task {session.task_instance_id!r} depends on itself")
        graph[session.task_instance_id] = session.depends_on

    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []

    def visit(task_id: str) -> None:
        if task_id in visited:
            return
        if task_id in visiting:
            cycle_start = stack.index(task_id)
            cycle = stack[cycle_start:] + [task_id]
            raise SimulateError("Dependency cycle detected: " + " -> ".join(cycle))
        visiting.add(task_id)
        stack.append(task_id)
        for dependency in graph[task_id]:
            visit(dependency)
        stack.pop()
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in graph:
        visit(task_id)


def _ready_worker_input_batches(
    worker_inputs: list[WorkerTraceInput],
    *,
    batch_size: int,
) -> list[list[WorkerTraceInput]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not any(entry.depends_on for entry in worker_inputs):
        return [
            worker_inputs[index : index + batch_size]
            for index in range(0, len(worker_inputs), batch_size)
        ]

    remaining = list(worker_inputs)
    completed: set[str] = set()
    batches: list[list[WorkerTraceInput]] = []
    while remaining:
        ready = [
            entry for entry in remaining if set(entry.depends_on).issubset(completed)
        ]
        if not ready:
            raise SimulateError(
                "No dependency-ready worker inputs remain; "
                "dependency validation should have failed"
            )
        batch = ready[:batch_size]
        batches.append(batch)
        batch_ids = {id(entry) for entry in batch}
        remaining = [entry for entry in remaining if id(entry) not in batch_ids]
        completed.update(entry.task_instance_id for entry in batch)
    return batches


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
    _validate_llm_timing_config(llm_timing, replay_speed=replay_speed)
    _validate_session_dependencies(sessions)
    for session in sessions:
        if not _requires_task_container(session):
            continue
        if _is_terminal_bench_registry_task(session):
            if not session.task.get("task_source_path"):
                raise SimulateError(
                    f"Terminal-Bench task {session.task_instance_id!r} has no task_source_path"
                )
            continue
        docker_image = _resolve_docker_image(session)
        if not docker_image:
            raise SimulateError(
                f"Task {session.task_instance_id!r} has no resolvable docker_image "
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
            ts_start, ts_end = _coerce_action_bounds(
                action, source_trace=session.source_trace
            )
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
        session.agent_id for session in sessions if _requires_task_container(session)
    ]
    if container_sessions and container_executable is None:
        sample = ", ".join(container_sessions[:3])
        suffix = "..." if len(container_sessions) > 3 else ""
        raise ValueError(
            "container_executable is required for replay sessions with task containers "
            f"({sample}{suffix})"
        )


def _replay_source_image(session: LoadedTraceSession) -> str | None:
    """Normalized source image a container-mode replay session pulls, if any.

    Terminal-bench registry tasks resolve their image inside the registry
    harness, so they never expose a host-pullable source image here.
    """
    if not _requires_task_container(session):
        return None
    if _is_terminal_bench_registry_task(session):
        return None
    docker_image = _resolve_docker_image(session)
    if docker_image is None:
        return None
    return normalize_image_reference(docker_image)


def _container_source_images(sessions: list[LoadedTraceSession]) -> list[str]:
    images = {
        image
        for session in sessions
        if (image := _replay_source_image(session)) is not None
    }
    return sorted(images)


@dataclass
class _ImageCleanupState:
    """Refcount bookkeeping for ``--cleanup-images`` source-image removal.

    ``refcounts`` maps a normalized source image to the number of pending
    replay sessions that still reference it. Each session decrements its image
    on finalize; the image is removed once the count reaches zero so per-task
    unique images do not accumulate on disk during a replay.
    """

    image_by_instance: dict[str, str]
    refcounts: dict[str, int]
    lock: asyncio.Lock
    container_executable: str | None
    skipped: int = 0


def _build_image_cleanup_state(
    sessions: list[LoadedTraceSession],
    *,
    container_executable: str | None,
) -> _ImageCleanupState:
    image_by_instance: dict[str, str] = {}
    refcounts: dict[str, int] = {}
    for session in sessions:
        image = _replay_source_image(session)
        if image is None:
            continue
        image_by_instance[session.run_instance_id] = image
        refcounts[image] = refcounts.get(image, 0) + 1
    return _ImageCleanupState(
        image_by_instance=image_by_instance,
        refcounts=refcounts,
        lock=asyncio.Lock(),
        container_executable=container_executable,
    )


async def _release_source_image(
    state: _ImageCleanupState,
    run_instance_id: str,
) -> None:
    """Decrement an image's pending refcount and remove it best-effort at zero.

    Callers must invoke this only after the session's container is finalized,
    so the image is no longer referenced by a running container. Removal
    failures are logged and counted, never raised (matches the collector's
    best-effort cleanup semantics).
    """
    image = state.image_by_instance.get(run_instance_id)
    if image is None or state.container_executable is None:
        return
    async with state.lock:
        remaining = state.refcounts.get(image, 0) - 1
        state.refcounts[image] = remaining
        should_remove = remaining <= 0
    if not should_remove:
        return
    try:
        removed = await asyncio.to_thread(
            remove_image,
            image,
            container_executable=state.container_executable,
        )
        drop_cached_fixed_image(image)
        if removed:
            logger.info("cleanup-images: removed source image %s", image)
    except Exception as exc:
        state.skipped += 1
        logger.warning(
            "cleanup-images: failed to remove source image %s: %s", image, exc
        )


def _has_container_mode_sessions(sessions: list[LoadedTraceSession]) -> bool:
    return any(_requires_task_container(session) for session in sessions)


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
        # Passthrough image prep returns the source image itself (no fixed
        # derivative); never delete the shared source image out from under
        # other concurrent replay sessions.
        if fixed_image == source_image:
            continue
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


def _terminal_bench_compose_file(task_dir: Path) -> Path:
    for name in ("docker-compose.yaml", "docker-compose.yml"):
        candidate = task_dir / name
        if candidate.exists():
            return candidate
    raise SimulateError(
        f"Terminal-Bench task has no docker-compose.yaml/yml: {task_dir}"
    )


def _run_terminal_bench_compose(
    *,
    container_executable: str,
    project: str,
    compose_file: Path,
    env: dict[str, str],
    args: list[str],
) -> str:
    cmd = [
        container_executable,
        "compose",
        "-p",
        project,
        "-f",
        str(compose_file),
    ]
    cmd.extend(args)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=compose_file.parent,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Terminal-Bench compose {' '.join(args)} failed "
            f"(returncode={result.returncode}). stdout tail:\n"
            f"{result.stdout[-2000:]}\n--- stderr tail:\n{result.stderr[-2000:]}"
        )
    return result.stdout.strip()


def _inspect_container_workdir(
    *,
    container_executable: str,
    container_id: str,
) -> str:
    result = subprocess.run(
        [
            container_executable,
            "inspect",
            "--format",
            "{{.Config.WorkingDir}}",
            container_id,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"docker inspect failed for {container_id[:12]} "
            f"(returncode={result.returncode}). stdout tail:\n"
            f"{result.stdout[-2000:]}\n--- stderr tail:\n{result.stderr[-2000:]}"
        )
    return result.stdout.strip() or "/"


def _inspect_container_cpu_controls(
    *,
    container_executable: str,
    container_id: str,
) -> dict[str, object]:
    result = subprocess.run(
        [
            container_executable,
            "inspect",
            "--format",
            "{{json .HostConfig}}",
            container_id,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"docker inspect failed for {container_id[:12]} "
            f"(returncode={result.returncode}). stdout tail:\n"
            f"{result.stdout[-2000:]}\n--- stderr tail:\n{result.stderr[-2000:]}"
        )
    host_config = json.loads(result.stdout)
    return {
        "nano_cpus": host_config.get("NanoCpus"),
        "cpu_shares": host_config.get("CpuShares"),
        "cpuset_cpus": host_config.get("CpusetCpus"),
    }


def _inspect_container_final_state(
    *,
    container_executable: str,
    container_id: str,
) -> dict[str, object]:
    state_result = subprocess.run(
        [
            container_executable,
            "inspect",
            "--format",
            "{{json .State}}",
            container_id,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if state_result.returncode != 0:
        raise RuntimeError(
            f"final container inspection failed for {container_id[:12]}: "
            f"state_rc={state_result.returncode}"
        )
    state = json.loads(state_result.stdout)
    memory_events: dict[str, int] | None = None
    if state.get("Running") is True:
        memory_result = subprocess.run(
            [
                container_executable,
                "exec",
                container_id,
                "cat",
                "/sys/fs/cgroup/memory.events",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        if memory_result.returncode != 0:
            raise RuntimeError(
                f"final container inspection failed for {container_id[:12]}: "
                f"memory_rc={memory_result.returncode}"
            )
        memory_events = {
            key: int(value)
            for line in memory_result.stdout.splitlines()
            for key, value in [line.split()]
        }
    return {
        "status": state.get("Status"),
        "running": state.get("Running"),
        "oom_killed": state.get("OOMKilled"),
        "exit_code": state.get("ExitCode"),
        "memory_events": memory_events,
    }


def _validate_container_workdir(
    *,
    container_executable: str,
    container_id: str,
    workdir: str,
) -> str:
    result = subprocess.run(
        [
            container_executable,
            "exec",
            "-i",
            "--user",
            "0",
            "-w",
            workdir,
            container_id,
            "/bin/sh",
            "-c",
            "pwd",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"container workdir {workdir!r} is not executable in {container_id[:12]} "
            f"(returncode={result.returncode}). stdout tail:\n"
            f"{result.stdout[-2000:]}\n--- stderr tail:\n{result.stderr[-2000:]}"
        )
    resolved = result.stdout.strip()
    if not resolved:
        raise RuntimeError(
            f"container workdir probe returned no pwd for {container_id[:12]}"
        )
    return resolved


def _install_terminal_bench_python_dependencies(
    *,
    container_executable: str,
    container_id: str,
    workdir: str,
) -> dict[str, str | int]:
    from agents.terminal_bench.openclaw_agent import TerminalBenchOpenClawAgent

    result = subprocess.run(
        [
            container_executable,
            "exec",
            "-i",
            "--user",
            "0",
            "-w",
            workdir,
            container_id,
            "/bin/bash",
            "-lc",
            TerminalBenchOpenClawAgent._bootstrap_dependencies_command(),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Terminal-Bench python dependency bootstrap failed "
            f"(returncode={result.returncode}). stdout tail:\n"
            f"{result.stdout[-2000:]}\n--- stderr tail:\n{result.stderr[-2000:]}"
        )
    return {
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


async def _prepare_terminal_bench_container_session(
    loaded: LoadedTraceSession,
    *,
    task_output_dir: Path,
    container_executable: str,
) -> PreparedTraceSession:
    """Prepare a real Terminal-Bench client container for host OpenClaw replay."""

    from agents.terminal_bench.runner import TerminalBenchRunner

    source_raw = loaded.task.get("task_source_path")
    if not isinstance(source_raw, str) or not source_raw:
        raise SimulateError(
            f"Terminal-Bench task {loaded.task_instance_id!r} has no task_source_path"
        )
    source_dir = Path(source_raw).expanduser().resolve()
    if not source_dir.is_dir():
        raise SimulateError(
            f"Terminal-Bench task_source_path is not a directory: {source_dir}"
        )
    runtime_dir = task_output_dir / "terminal-bench-runtime"
    task_runtime_dir = runtime_dir / "task"
    run_root = runtime_dir / "tb-run"
    task_id = str(loaded.task.get("task_id") or loaded.task_instance_id)
    run_id = _sanitize_run_label(loaded.run_instance_id).lower()[:48]
    if not run_id:
        run_id = "openclaw-replay"
    project = TerminalBenchRunner._expected_client_container_name(
        task_id=task_id,
        run_id=run_id,
    )
    env_values = TerminalBenchRunner._terminal_bench_compose_env(
        task_id=task_id,
        run_id=run_id,
        run_root=run_root,
    )
    compose_env = os.environ.copy()
    compose_env.update(env_values)
    recorder = ContainerStartupRecorder(
        loaded=loaded,
        task_output_dir=task_output_dir,
        container_executable=container_executable,
        network_mode="terminal_bench_compose",
        source_image=None,
    )
    cleanup_callback: Callable[[], None] | None = None
    container_workdir = "/testbed"
    container_python_runtime: str | None = None
    container_pythonpath: str | None = None
    try:
        phase = recorder.start_phase("materialize_terminal_bench_task")
        try:
            if task_runtime_dir.exists():
                shutil.rmtree(task_runtime_dir)
            task_runtime_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_dir, task_runtime_dir)
            compose_file = _terminal_bench_compose_file(task_runtime_dir).resolve()
            dockerfile = task_runtime_dir / "Dockerfile"
            if not dockerfile.exists():
                raise SimulateError(
                    f"Terminal-Bench task has no Dockerfile: {task_runtime_dir}"
                )
            recorder.finish_phase(
                phase,
                extra={
                    "source_dir": str(source_dir),
                    "runtime_dir": str(task_runtime_dir),
                    "compose_file": str(compose_file),
                    "dockerfile": str(dockerfile),
                    "compose_project": project,
                    "compose_env": env_values,
                },
            )
        except (Exception, asyncio.CancelledError) as exc:
            recorder.finish_phase(phase, status="failed", error=exc)
            raise

        cleanup_callback = functools.partial(
            _run_terminal_bench_compose,
            container_executable=container_executable,
            project=project,
            compose_file=compose_file,
            env=compose_env,
            args=["down", "--volumes", "--remove-orphans"],
        )
        phase = recorder.start_phase("terminal_bench_compose_build")
        try:
            stdout = await asyncio.to_thread(
                _run_terminal_bench_compose,
                container_executable=container_executable,
                project=project,
                compose_file=compose_file,
                env=compose_env,
                args=["build"],
            )
            recorder.finish_phase(phase, extra={"stdout_tail": stdout[-2000:]})
        except (Exception, asyncio.CancelledError) as exc:
            recorder.finish_phase(phase, status="failed", error=exc)
            raise

        phase = recorder.start_phase("terminal_bench_compose_up")
        try:
            stdout = await asyncio.to_thread(
                _run_terminal_bench_compose,
                container_executable=container_executable,
                project=project,
                compose_file=compose_file,
                env=compose_env,
                args=["up", "-d"],
            )
            recorder.finish_phase(phase, extra={"stdout_tail": stdout[-2000:]})
        except (Exception, asyncio.CancelledError) as exc:
            recorder.finish_phase(phase, status="failed", error=exc)
            raise

        phase = recorder.start_phase("terminal_bench_resolve_client_container")
        try:
            container_id = await asyncio.to_thread(
                _run_terminal_bench_compose,
                container_executable=container_executable,
                project=project,
                compose_file=compose_file,
                env=compose_env,
                args=["ps", "-q", "client"],
            )
            if not container_id:
                raise RuntimeError(
                    "docker compose ps -q client returned no container id"
                )
            recorder.container_id = container_id
            recorder.finish_phase(
                phase,
                extra={
                    "container_id": container_id,
                    "client_image": env_values["T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME"],
                    "client_container_name": env_values[
                        "T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME"
                    ],
                },
            )
        except (Exception, asyncio.CancelledError) as exc:
            recorder.finish_phase(phase, status="failed", error=exc)
            raise

        phase = recorder.start_phase("terminal_bench_resolve_container_workdir")
        try:
            inspected_workdir = await asyncio.to_thread(
                _inspect_container_workdir,
                container_executable=container_executable,
                container_id=container_id,
            )
            container_workdir = await asyncio.to_thread(
                _validate_container_workdir,
                container_executable=container_executable,
                container_id=container_id,
                workdir=inspected_workdir,
            )
            recorder.finish_phase(
                phase,
                extra={
                    "inspected_workdir": inspected_workdir,
                    "tool_container_workdir": container_workdir,
                },
            )
        except (Exception, asyncio.CancelledError) as exc:
            recorder.finish_phase(phase, status="failed", error=exc)
            raise

        phase = recorder.start_phase("terminal_bench_resolve_python_runtime")
        try:
            base_exec_config = resolve_task_container_exec_config(
                attempt_dir=task_output_dir,
                image=env_values["T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME"],
                container_executable=container_executable,
            )
            dependency_bootstrap: dict[str, str | int] | None = None
            try:
                running_exec_config = await asyncio.to_thread(
                    resolve_running_container_exec_config,
                    container_id=container_id,
                    exec_config=base_exec_config,
                    container_executable=container_executable,
                    cwd=container_workdir,
                )
            except RuntimeError:
                dependency_bootstrap = await asyncio.to_thread(
                    _install_terminal_bench_python_dependencies,
                    container_executable=container_executable,
                    container_id=container_id,
                    workdir=container_workdir,
                )
                running_exec_config = await asyncio.to_thread(
                    resolve_running_container_exec_config,
                    container_id=container_id,
                    exec_config=base_exec_config,
                    container_executable=container_executable,
                    cwd=container_workdir,
                )
            container_python_runtime = running_exec_config.runtime
            container_pythonpath = None
            recorder.finish_phase(
                phase,
                extra={
                    "python_runtime": container_python_runtime,
                    "pythonpath": container_pythonpath,
                    "dependency_bootstrap": dependency_bootstrap,
                },
            )
        except (Exception, asyncio.CancelledError) as exc:
            recorder.finish_phase(phase, status="failed", error=exc)
            raise

        recorder.write(status="success")
    except (Exception, asyncio.CancelledError) as exc:
        try:
            recorder.write(status="failed", error=exc)
        except (Exception, asyncio.CancelledError):
            logger.exception(
                "Failed to write Terminal-Bench startup failure artifact for %s",
                loaded.agent_id,
            )
        if cleanup_callback is not None:
            try:
                await asyncio.to_thread(cleanup_callback)
            except (Exception, asyncio.CancelledError):
                logger.exception(
                    "Failed to clean Terminal-Bench compose project for %s",
                    loaded.agent_id,
                )
        raise

    container = PreparedContainer(
        container_id=container_id,
        container_executable=container_executable,
        docker_image=env_values["T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME"],
        agent=None,
        fixed_image=None,
        python_runtime=container_python_runtime,
        pythonpath=container_pythonpath,
        workdir=container_workdir,
        cleanup_fixed_image=False,
        cleanup_callback=cleanup_callback,
    )
    return PreparedTraceSession(loaded=loaded, container=container)


async def _prepare_container_session(
    loaded: LoadedTraceSession,
    *,
    task_output_dir: Path,
    container_executable: str,
    network_mode: str = "host",
    fixed_images_by_source: dict[str, str] | None = None,
    start_agent: bool = True,
    start_extra_args: list[str] | None = None,
) -> PreparedTraceSession:
    """Prepare a Docker/Podman container and start a persistent replay agent."""
    from trace_collect.openclaw_tools import ContainerAgent

    docker_image = _resolve_docker_image(loaded)
    if not docker_image:
        raise SimulateError(
            f"Task {loaded.task_instance_id!r} has no resolvable docker_image"
        )
    normalized = normalize_image_reference(docker_image)
    recorder = ContainerStartupRecorder(
        loaded=loaded,
        task_output_dir=task_output_dir,
        container_executable=container_executable,
        network_mode=network_mode,
        source_image=normalized,
    )
    container_id: str | None = None
    agent: Any | None = None
    cleanup_fixed_image = True
    try:
        phase = recorder.start_phase("ensure_fixed_image")
        try:
            fixed_name = (fixed_images_by_source or {}).get(normalized)
            if fixed_name is not None:
                cleanup_fixed_image = False
                fixed_elapsed_s = 0.0
                extra = {
                    "fixed_image": fixed_name,
                    "reported_elapsed_s": fixed_elapsed_s,
                    "prebuilt": True,
                }
            else:
                fixed_image_name = _replay_fixed_image_name(
                    source_image=normalized,
                    agent_id=loaded.agent_id,
                    task_output_dir=task_output_dir,
                )
                fixed_name, fixed_elapsed_s = await asyncio.to_thread(
                    ensure_fixed_image,
                    normalized,
                    container_executable=container_executable,
                    fixed_image_name=fixed_image_name,
                    rebuild=True,
                )
                extra = {
                    "fixed_image": fixed_name,
                    "reported_elapsed_s": fixed_elapsed_s,
                    "prebuilt": False,
                }
            recorder.fixed_image = fixed_name
            recorder.finish_phase(
                phase,
                extra=extra,
            )
        except (Exception, asyncio.CancelledError) as exc:
            recorder.finish_phase(phase, status="failed", error=exc)
            raise

        phase = recorder.start_phase("start_task_container")
        try:
            extra_args = [
                "--label",
                "agent-sched-bench.component=simulate-replay",
                "--label",
                f"agent-sched-bench.run_instance_id={loaded.agent_id}",
                "--label",
                f"agent-sched-bench.source_action_agent_id={loaded.source_action_agent_id}",
                "--label",
                f"agent-sched-bench.task_instance_id={loaded.task_instance_id}",
                "--label",
                f"agent-sched-bench.manifest_index={loaded.manifest_index}",
                "--label",
                f"agent-sched-bench.output_dir={task_output_dir}",
            ]
            extra_args.extend(start_extra_args or [])
            container_id = await asyncio.to_thread(
                start_task_container,
                fixed_name,
                executable=container_executable,
                run_as_host_user=False,
                mount_host_home=False,
                container_home="/root",
                extra_args=extra_args,
                network_mode=network_mode,
            )
            recorder.container_id = container_id
            cpu_controls = (
                await asyncio.to_thread(
                    _inspect_container_cpu_controls,
                    container_executable=container_executable,
                    container_id=container_id,
                )
                if start_extra_args
                else None
            )
            recorder.finish_phase(
                phase,
                extra={
                    "container_id": container_id,
                    "start_extra_args": list(start_extra_args or ()),
                    "cpu_controls": cpu_controls,
                },
            )
        except (Exception, asyncio.CancelledError) as exc:
            recorder.finish_phase(phase, status="failed", error=exc)
            raise

        phase = recorder.start_phase("configure_apt_mirror")
        try:
            mirror_info = await asyncio.to_thread(
                configure_task_container_apt_mirror,
                container_id,
                executable=container_executable,
            )
            mirror_status = (
                "skipped"
                if mirror_info is None or mirror_info.get("configured") == "false"
                else "success"
            )
            recorder.finish_phase(
                phase,
                status=mirror_status,
                extra=mirror_info or {"reason": "TASK_CONTAINER_APT_MIRROR unset"},
            )
        except (Exception, asyncio.CancelledError) as exc:
            recorder.finish_phase(phase, status="failed", error=exc)
            raise

        if start_agent:
            agent = ContainerAgent(
                container_id,
                container_executable,
                python_runtime=None,
                pythonpath=None,
                workdir="/testbed",
            )
            phase = recorder.start_phase("container_agent_start")
            try:
                await agent.start()
                recorder.finish_phase(phase)
            except (Exception, asyncio.CancelledError) as exc:
                recorder.finish_phase(phase, status="failed", error=exc)
                raise
        else:
            phase = recorder.start_phase("container_agent_start")
            recorder.finish_phase(
                phase,
                status="skipped",
                extra={"reason": "openclaw_host_worker_owns_tool_agent"},
            )

        recorder.write(status="success")
    except (Exception, asyncio.CancelledError) as exc:
        cleanup_errors: list[BaseException] = []
        try:
            recorder.write(status="failed", error=exc)
        except (Exception, asyncio.CancelledError):
            logger.exception(
                "Failed to write container startup failure artifact for %s",
                loaded.agent_id,
            )
        if agent is not None:
            try:
                await agent.stop()
            except (Exception, asyncio.CancelledError):
                logger.exception(
                    "Failed to stop container agent for %s", loaded.agent_id
                )
        container_stopped = False
        if container_id is not None:
            try:
                await asyncio.to_thread(
                    stop_task_container,
                    container_id,
                    executable=container_executable,
                )
                container_stopped = True
            except (Exception, asyncio.CancelledError) as cleanup_exc:
                cleanup_errors.append(cleanup_exc)
                logger.exception(
                    "Failed to stop startup container for %s", loaded.agent_id
                )
        if (
            cleanup_fixed_image
            and recorder.fixed_image is not None
            and recorder.fixed_image != recorder.source_image
            and (container_id is None or container_stopped)
        ):
            try:
                await asyncio.to_thread(
                    remove_image,
                    recorder.fixed_image,
                    container_executable=container_executable,
                )
            except (Exception, asyncio.CancelledError) as cleanup_exc:
                cleanup_errors.append(cleanup_exc)
                logger.exception(
                    "Failed to remove startup fixed image for %s",
                    loaded.agent_id,
                )
        if cleanup_errors:
            cleanup_error = cleanup_errors[0]
            if cleanup_error is not exc:
                cleanup_error.__context__ = exc
            raise cleanup_error
        raise

    assert container_id is not None

    container = PreparedContainer(
        container_id=container_id,
        container_executable=container_executable,
        docker_image=normalized,
        agent=agent,
        fixed_image=recorder.fixed_image,
        cleanup_fixed_image=cleanup_fixed_image,
        inspect_final_state=bool(start_extra_args),
    )
    return PreparedTraceSession(
        loaded=loaded,
        container=container,
        task_output_dir=task_output_dir,
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
    final_state_error: BaseException | None = None
    container_stop_error: BaseException | None = None
    fixed_image_cleanup_error: BaseException | None = None
    container_stopped = False
    try:
        agents = [*ctr.extra_agents]
        if ctr.agent is not None:
            agents.append(ctr.agent)
        for agent in agents:
            try:
                await agent.stop()
            except (Exception, asyncio.CancelledError) as exc:
                if agent_stop_error is None:
                    agent_stop_error = exc
                else:
                    logger.exception(
                        "Failed to stop additional container agent for %s",
                        prepared.loaded.agent_id,
                    )
        ctr.extra_agents.clear()
    except (Exception, asyncio.CancelledError) as exc:
        agent_stop_error = exc

    if ctr.inspect_final_state:
        try:
            prepared.final_container_state = await asyncio.to_thread(
                _inspect_container_final_state,
                container_executable=ctr.container_executable,
                container_id=ctr.container_id,
            )
        except (Exception, asyncio.CancelledError) as exc:
            final_state_error = exc

    try:
        if ctr.cleanup_callback is not None:
            await asyncio.to_thread(ctr.cleanup_callback)
        else:
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
        container_stopped
        and ctr.fixed_image
        and ctr.cleanup_fixed_image
        and ctr.fixed_image != ctr.docker_image
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
    if final_state_error is not None:
        raise final_state_error
    if fixed_image_cleanup_error is not None:
        raise fixed_image_cleanup_error


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


def _replay_action_bounds_key(
    action: dict[str, Any],
    *,
    source_trace: Path,
) -> tuple[float, float, str]:
    start, end = _coerce_action_bounds(action, source_trace=source_trace)
    return start, end, str(action.get("action_id", ""))


def _replay_action_batches(
    actions: list[dict[str, Any]],
    *,
    source_trace: Path,
) -> list[list[dict[str, Any]]]:
    """Group source-overlapping actions for concurrent replay."""

    sorted_actions = sorted(
        actions,
        key=lambda action: _replay_action_bounds_key(
            action,
            source_trace=source_trace,
        ),
    )
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_end: float | None = None
    for action in sorted_actions:
        start, end = _coerce_action_bounds(action, source_trace=source_trace)
        if not current:
            current = [action]
            current_end = end
            continue
        assert current_end is not None
        if start < current_end:
            current.append(action)
            current_end = max(current_end, end)
            continue
        batches.append(current)
        current = [action]
        current_end = end
    if current:
        batches.append(current)
    return batches


def _action_may_use_container_agent(action: dict[str, Any]) -> bool:
    if action.get("action_type") != "tool_exec":
        return False
    data = action.get("data") or {}
    tool_name = data.get("tool_name")
    if not isinstance(tool_name, str) or not tool_name:
        return False
    return tool_name not in _CONTROL_PLANE_NOOP_TOOLS and not tool_name.startswith(
        "mcp_"
    )


async def _prewarm_replay_agents_for_batch(
    prepared_session: PreparedTraceSession,
    ordered_batch: list[dict[str, Any]],
) -> dict[int, Any]:
    ctr = prepared_session.container
    if ctr is None:
        return {}

    runnable_actions = [
        action for action in ordered_batch if _action_may_use_container_agent(action)
    ]
    if not runnable_actions:
        return {}

    from trace_collect.openclaw_tools import ContainerAgent

    assignments: dict[int, Any] = {id(runnable_actions[0]): ctr.agent}
    extra_agents = [
        ContainerAgent(
            ctr.container_id,
            ctr.container_executable,
            python_runtime=ctr.python_runtime,
            pythonpath=ctr.pythonpath,
            workdir=ctr.workdir,
        )
        for _ in runnable_actions[1:]
    ]
    if extra_agents:
        ctr.extra_agents.extend(extra_agents)
        await asyncio.gather(*(agent.start() for agent in extra_agents))
        assignments.update(
            {
                id(action): agent
                for action, agent in zip(runnable_actions[1:], extra_agents)
            }
        )
    return assignments


def _validate_llm_timing_config(
    config: LLMTimingConfig,
    *,
    replay_speed: float,
) -> None:
    validate_llm_replay_timing(
        replay_speed=replay_speed,
        timing_mode=config.mode,
        llm_ttft_ms=config.ttft_ms,
        llm_tpot_ms=config.tpot_ms,
    )


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
    container_start_extra_args: tuple[str, ...] = (),
) -> PreparedTraceSession:
    prepared: PreparedTraceSession | None = None
    session_resource_monitoring_enabled = (
        resource_monitoring_enabled and _requires_task_container(loaded)
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
        if not _requires_task_container(loaded):
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
            if container_executable is None:
                raise ValueError(
                    "container_executable is required for replay task containers"
                )
            if _is_terminal_bench_registry_task(loaded):
                prepared = await _prepare_terminal_bench_container_session(
                    loaded,
                    task_output_dir=task_output_dir,
                    container_executable=container_executable,
                )
            else:
                prepare_kwargs: dict[str, Any] = {}
                if loaded.scaffold == "openclaw":
                    prepare_kwargs["start_agent"] = False
                if fixed_images_by_source:
                    prepare_kwargs["fixed_images_by_source"] = fixed_images_by_source
                if container_start_extra_args:
                    prepare_kwargs["start_extra_args"] = list(
                        container_start_extra_args
                    )
                prepared = await _prepare_container_session(
                    loaded,
                    task_output_dir=task_output_dir,
                    container_executable=container_executable,
                    network_mode=network_mode,
                    **prepare_kwargs,
                )
                prepared.task_output_dir = task_output_dir
                await _restore_source_runtime_artifacts(prepared)
            prepared.task_output_dir = task_output_dir
        if prepared.container is not None:
            prepared.resource_monitoring_enabled = session_resource_monitoring_enabled
            prepared.memory_bandwidth_enabled = memory_bandwidth_enabled
            prepared.monitoring_policy = monitoring_policy
            prepared.container_resource_recorder = container_resource_recorder
            if container_resource_recorder is not None:
                container_resource_recorder.register_container(
                    prepared.container.container_id
                )
            if session_resource_monitoring_enabled:
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
        cleanup_error: BaseException | None = None
        if prepared is not None:
            try:
                await _finalize_prepared_session(prepared)
            except (Exception, asyncio.CancelledError) as finalize_exc:
                cleanup_error = finalize_exc
        if isinstance(exc, RuntimeError) and prepared is not None:
            raise _ReplayPreparationError(
                prepared,
                _container_prep_failure_reason(exc),
                exc,
                cleanup_error,
            ) from exc
        if cleanup_error is not None:
            raise cleanup_error from exc
        raise


def _container_prep_failure_reason(error: RuntimeError) -> str:
    message = str(error).lower()
    if "python probe failed" in message or "no python >=3.11" in message:
        return "container_python_unavailable"
    return "container_prep_failed"


def _failed_prepared_session(
    loaded: LoadedTraceSession,
    *,
    output_path: Path,
    container_executable: str | None,
    network_mode: str,
    reason: str,
    error: RuntimeError,
) -> PreparedTraceSession:
    prepared = PreparedTraceSession(loaded=loaded)
    _assign_task_output_dir(prepared, output_path)
    assert prepared.task_output_dir is not None
    recorder = ContainerStartupRecorder(
        loaded=loaded,
        task_output_dir=prepared.task_output_dir,
        container_executable=container_executable,
        network_mode=network_mode,
        source_image=None,
    )
    recorder.write(status="failed", reason=reason, error=error)
    return prepared


def _record_prepare_failure(
    prepared: PreparedTraceSession,
    *,
    trace_logger: TraceLogger,
    reason: str,
    error: RuntimeError,
    cleanup_error: BaseException | None = None,
) -> ReplayTaskStats:
    loaded = prepared.loaded
    extra = {
        "replay_mode": "cloud_model",
        "status": "failed",
        "reason": reason,
        "failure_phase": "container_prep",
        "failed_actions": 1,
        "error": _exception_payload(error),
    }
    if cleanup_error is not None:
        extra["cleanup_error"] = _exception_payload(cleanup_error)
    trace_logger.log_summary(
        loaded.agent_id,
        _make_trace_summary(
            loaded=loaded,
            success=False,
            elapsed_s=0.0,
            source_model=_source_model(loaded),
            extra=extra,
        ),
    )
    return _make_task_stats(
        loaded=loaded,
        success=False,
        elapsed_s=0.0,
        failed_action_count=1,
    )


async def _cleanup_staged_sessions(
    prepared_sessions: list[PreparedTraceSession],
    cleanup_state: _ImageCleanupState | None,
    *,
    release_only: tuple[str, ...] = (),
) -> None:
    async def finalize(prepared: PreparedTraceSession) -> None:
        try:
            await _finalize_prepared_session(prepared)
        finally:
            if cleanup_state is not None:
                await _release_source_image(
                    cleanup_state,
                    prepared.loaded.run_instance_id,
                )

    operations = [finalize(prepared) for prepared in prepared_sessions]
    if cleanup_state is not None:
        operations.extend(
            _release_source_image(cleanup_state, run_instance_id)
            for run_instance_id in release_only
        )
    cleanup_results = await asyncio.gather(*operations, return_exceptions=True)
    cleanup_failures = [
        result for result in cleanup_results if isinstance(result, BaseException)
    ]
    if cleanup_failures:
        raise SimulateError(
            f"{len(cleanup_failures)}/{len(cleanup_results)} staged cleanups failed"
        ) from cleanup_failures[0]


async def _run_staged_cloud_model_queue(
    loaded_sessions: list[LoadedTraceSession],
    *,
    output_path: Path,
    trace_logger: TraceLogger,
    concurrency: int,
    prep_concurrency: int,
    container_executable: str | None,
    network_mode: str,
    container_resource_recorder: ContainerResourceRecorder | None,
    replay_speed: float,
    shadow_generation: ShadowGenerationConfig | None = None,
    tool_gap_arm: str | None = None,
    tool_gap_predictions: dict[str, tuple[ToolGapPrediction, ...]] | None = None,
    tool_gap_borrower_priority: int | None = None,
    llm_timing: LLMTimingConfig,
    command_timeout_s: float,
    warmup_skip_iterations: int,
    fixed_images_by_source: dict[str, str] | None = None,
    resource_monitoring_enabled: bool = True,
    memory_bandwidth_enabled: bool = True,
    monitoring_policy: dict[str, object] | None = None,
    container_start_extra_args: tuple[str, ...] = (),
    cleanup_state: _ImageCleanupState | None = None,
) -> tuple[list[PreparedTraceSession], list[ReplayTaskStats], float]:
    """Prepare all sessions, then replay them FIFO from one ready time."""

    state_dir = output_path / ".tool-gap-loan"
    if tool_gap_arm is not None:
        if state_dir.exists() and any(state_dir.iterdir()):
            raise SimulateError(f"tool-gap state directory is not empty: {state_dir}")
        state_dir.mkdir(parents=True, exist_ok=True)
    prep_limit = _resolve_prep_concurrency(prep_concurrency, len(loaded_sessions))
    prep_semaphore = asyncio.Semaphore(prep_limit)

    async def prepare(loaded: LoadedTraceSession) -> PreparedTraceSession:
        async with prep_semaphore:
            return await _prepare_replay_session(
                loaded,
                output_path=output_path,
                container_executable=container_executable,
                network_mode=network_mode,
                container_resource_recorder=container_resource_recorder,
                fixed_images_by_source=fixed_images_by_source,
                resource_monitoring_enabled=resource_monitoring_enabled,
                memory_bandwidth_enabled=memory_bandwidth_enabled,
                monitoring_policy=monitoring_policy,
                container_start_extra_args=container_start_extra_args,
            )

    prep_results = await asyncio.gather(
        *(prepare(loaded) for loaded in loaded_sessions),
        return_exceptions=True,
    )
    prepared_sessions: list[PreparedTraceSession] = []
    successful_preparations: list[PreparedTraceSession] = []
    prep_failures: list[BaseException] = []
    failed_run_instance_ids: list[str] = []
    for loaded, result in zip(loaded_sessions, prep_results, strict=True):
        if isinstance(result, _ReplayPreparationError):
            prepared_sessions.append(result.prepared)
            prep_failures.append(result)
            failed_run_instance_ids.append(loaded.run_instance_id)
        elif isinstance(result, BaseException):
            prep_failures.append(result)
            failed_run_instance_ids.append(loaded.run_instance_id)
        else:
            prepared_sessions.append(result)
            successful_preparations.append(result)
    if prep_failures:
        await _cleanup_staged_sessions(
            successful_preparations,
            cleanup_state,
            release_only=tuple(failed_run_instance_ids),
        )
        raise SimulateError(
            f"{len(prep_failures)}/{len(prep_results)} staged preparations failed"
        ) from prep_failures[0]

    common_ready_monotonic = time.monotonic() + _REPLAY_START_DELAY_S
    common_ready_wall_time_s = time.time() + _REPLAY_START_DELAY_S
    if tool_gap_arm is not None:
        foreground_ids = tuple(
            prepared.loaded.task_instance_id
            for prepared in prepared_sessions[:concurrency]
        )
        predictions_by_task = tool_gap_predictions or {}
        task_stats: dict[str, ReplayTaskStats] = {}
        active: dict[
            asyncio.Task[ReplayTaskStats],
            tuple[PreparedTraceSession, float, str, str | None],
        ] = {}
        admissions: list[dict[str, object]] = []
        loan_releases: list[dict[str, object]] = []
        next_index = 0
        max_active = 0
        observed_lenders: set[str] = set()
        consumed_lenders: set[str] = set()
        loan_records: dict[str, dict[str, object]] = {}

        def refresh_loans() -> None:
            loan_dir = state_dir / "loans"
            if not loan_dir.exists():
                return
            for loan_path in loan_dir.glob("*.json"):
                record = json.loads(loan_path.read_text(encoding="utf-8"))
                task_id = record.get("task_id")
                if task_id not in foreground_ids:
                    raise SimulateError(f"invalid tool-gap lender: {task_id!r}")
                decision_wall_time_s = record.get("decision_wall_time_s")
                if (
                    not isinstance(decision_wall_time_s, (int, float))
                    or isinstance(decision_wall_time_s, bool)
                ):
                    raise SimulateError("tool-gap loan has no valid decision time")
                lender = str(task_id)
                observed_lenders.add(lender)
                loan_records[lender] = record

        async def replay_one(
            prepared: PreparedTraceSession,
            admitted_monotonic: float,
            config: ToolGapLoanConfig,
        ) -> ReplayTaskStats:
            try:
                stats = await _replay_cloud_model_session(
                    prepared,
                    trace_logger=trace_logger,
                    replay_zero_monotonic=common_ready_monotonic,
                    replay_speed=replay_speed,
                    shadow_generation=shadow_generation,
                    tool_gap_loan=config,
                    llm_timing=llm_timing,
                    command_timeout_s=command_timeout_s,
                    warmup_skip_iterations=warmup_skip_iterations,
                )
                terminal_monotonic = time.monotonic()
            finally:
                await _finalize_prepared_session(prepared)
            return dataclasses.replace(
                stats,
                admission_wait_s=max(
                    0.0,
                    admitted_monotonic - common_ready_monotonic,
                ),
                ready_to_terminal_s=max(
                    0.0,
                    terminal_monotonic - common_ready_monotonic,
                ),
            )

        def admit_to_capacity() -> None:
            nonlocal next_index, max_active
            while next_index < len(prepared_sessions):
                base_active = sum(slot == "base" for _, _, slot, _ in active.values())
                lender: str | None = None
                if base_active < concurrency:
                    admission_kind = "base"
                else:
                    pending_lenders = sorted(
                        observed_lenders - consumed_lenders,
                        key=lambda task_id: (
                            float(loan_records[task_id]["decision_wall_time_s"]),
                            task_id,
                        ),
                    )
                    if not pending_lenders:
                        break
                    lender = pending_lenders[0]
                    consumed_lenders.add(lender)
                    admission_kind = "loan"
                prepared = prepared_sessions[next_index]
                task_id = prepared.loaded.task_instance_id
                admitted_monotonic = time.monotonic()
                admitted_wall_time_s = time.time()
                config = ToolGapLoanConfig(
                    arm=tool_gap_arm,
                    state_dir=str(state_dir),
                    task_id=task_id,
                    foreground_task_ids=foreground_ids,
                    can_lend=next_index < concurrency,
                    predictions=predictions_by_task.get(task_id, ()),
                    borrower_priority=tool_gap_borrower_priority,
                )
                task = asyncio.create_task(
                    replay_one(prepared, admitted_monotonic, config)
                )
                active[task] = (
                    prepared,
                    admitted_monotonic,
                    admission_kind,
                    lender,
                )
                admissions.append(
                    {
                        "task_id": task_id,
                        "run_instance_id": prepared.loaded.run_instance_id,
                        "admitted_wall_time_s": admitted_wall_time_s,
                        "admission_kind": admission_kind,
                        "lender_task_id": lender,
                        "active_after": len(active),
                        "observed_lenders": sorted(observed_lenders),
                    }
                )
                if lender is not None:
                    lender_record = loan_records[lender]
                    loan_releases.append(
                        {
                            "lender_task_id": lender,
                            "borrower_task_id": task_id,
                            "lender_call_id": lender_record.get("call_id"),
                            "trigger": lender_record.get("trigger"),
                            "lender_decision_wall_time_s": lender_record[
                                "decision_wall_time_s"
                            ],
                            "borrower_admitted_wall_time_s": admitted_wall_time_s,
                        }
                    )
                next_index += 1
                max_active = max(max_active, len(active))

        await _sleep_until_monotonic(common_ready_monotonic)
        admit_to_capacity()
        failure: BaseException | None = None
        try:
            while active:
                done, _ = await asyncio.wait(
                    active,
                    timeout=0.01,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                try:
                    refresh_loans()
                except BaseException as exc:
                    failure = exc
                    break
                for task in done:
                    prepared, _admitted, _slot, _lender = active.pop(task)
                    try:
                        stats = task.result()
                    except BaseException as exc:
                        failure = exc
                        break
                    task_stats[prepared.loaded.run_instance_id] = stats
                if failure is not None:
                    break
                admit_to_capacity()
        except BaseException:
            for task in active:
                task.cancel()
            await asyncio.gather(*active, return_exceptions=True)
            raise
        if failure is not None:
            for task in active:
                task.cancel()
            await asyncio.gather(*active, return_exceptions=True)
            await _cleanup_staged_sessions(prepared_sessions, cleanup_state)
            raise SimulateError("staged tool-gap replay failed") from failure
        summary = {
            "arm": tool_gap_arm,
            "state_dir": str(state_dir),
            "foreground_task_ids": list(foreground_ids),
            "waiting_task_ids": [
                prepared.loaded.task_instance_id
                for prepared in prepared_sessions[concurrency:]
            ],
            "observed_loan_count": len(observed_lenders),
            "observed_lender_task_ids": sorted(observed_lenders),
            "consumed_lender_task_ids": sorted(consumed_lenders),
            "effective_max_concurrency": max_active,
            "admissions": admissions,
            "loan_releases": loan_releases,
        }
        (state_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        ordered_stats = [
            task_stats[prepared.loaded.run_instance_id]
            for prepared in prepared_sessions
        ]
        return prepared_sessions, ordered_stats, common_ready_wall_time_s

    queue: asyncio.Queue[PreparedTraceSession] = asyncio.Queue()
    for prepared in prepared_sessions:
        queue.put_nowait(prepared)
    task_stats: dict[str, ReplayTaskStats] = {}

    async def worker() -> None:
        await _sleep_until_monotonic(common_ready_monotonic)
        while True:
            try:
                prepared = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            admitted_monotonic = time.monotonic()
            try:
                try:
                    stats = await _replay_cloud_model_session(
                        prepared,
                        trace_logger=trace_logger,
                        replay_zero_monotonic=common_ready_monotonic,
                        replay_speed=replay_speed,
                        shadow_generation=shadow_generation,
                        llm_timing=llm_timing,
                        command_timeout_s=command_timeout_s,
                        warmup_skip_iterations=warmup_skip_iterations,
                    )
                    terminal_monotonic = time.monotonic()
                finally:
                    await _finalize_prepared_session(prepared)
                task_stats[prepared.loaded.run_instance_id] = dataclasses.replace(
                    stats,
                    admission_wait_s=max(
                        0.0,
                        admitted_monotonic - common_ready_monotonic,
                    ),
                    ready_to_terminal_s=max(
                        0.0,
                        terminal_monotonic - common_ready_monotonic,
                    ),
                )
            finally:
                queue.task_done()

    worker_results = await asyncio.gather(
        *(worker() for _ in range(min(concurrency, len(prepared_sessions)))),
        return_exceptions=True,
    )
    failures = [result for result in worker_results if isinstance(result, BaseException)]
    if failures:
        await _cleanup_staged_sessions(prepared_sessions, cleanup_state)
        raise SimulateError(
            f"{len(failures)}/{len(worker_results)} staged replay workers failed"
        ) from failures[0]
    ordered_stats = [
        task_stats[prepared.loaded.run_instance_id]
        for prepared in prepared_sessions
    ]
    return prepared_sessions, ordered_stats, common_ready_wall_time_s


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
    shadow_generation: ShadowGenerationConfig | None = None,
    llm_timing: LLMTimingConfig,
    command_timeout_s: float,
    warmup_skip_iterations: int,
    fixed_images_by_source: dict[str, str] | None = None,
    resource_monitoring_enabled: bool = True,
    memory_bandwidth_enabled: bool = True,
    monitoring_policy: dict[str, object] | None = None,
    cleanup_state: _ImageCleanupState | None = None,
    container_start_extra_args: tuple[str, ...] = (),
) -> tuple[list[PreparedTraceSession], list[ReplayTaskStats], float | None]:
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    queue: asyncio.Queue[LoadedTraceSession | None] = asyncio.Queue()
    released_session_ids: set[int] = set()
    completed_task_ids: set[str] = set()
    completed_session_count = 0
    children_by_dependency: dict[str, list[LoadedTraceSession]] = {}
    scheduled_arrivals = any(loaded.arrival_s > 0 for loaded in loaded_sessions)
    arrival_zero_monotonic = time.monotonic()
    arrival_zero_wall_time_s = time.time()
    for loaded in loaded_sessions:
        for dependency in loaded.depends_on:
            children_by_dependency.setdefault(dependency, []).append(loaded)
        if not scheduled_arrivals and not loaded.depends_on:
            queue.put_nowait(loaded)
            released_session_ids.add(id(loaded))

    prepared_sessions: list[PreparedTraceSession] = []
    task_stats: list[ReplayTaskStats] = []
    result_lock = asyncio.Lock()
    worker_count = min(concurrency, len(loaded_sessions))
    first_error: BaseException | None = None

    async def release_scheduled_arrivals() -> None:
        for loaded in sorted(
            loaded_sessions,
            key=lambda item: (item.arrival_s, item.manifest_index),
        ):
            await _sleep_until_monotonic(
                arrival_zero_monotonic + loaded.arrival_s
            )
            if first_error is not None:
                return
            queue.put_nowait(loaded)

    def stop_workers() -> None:
        for _ in range(worker_count):
            queue.put_nowait(None)

    def release_ready_children(parent_task_id: str) -> None:
        for child in children_by_dependency.get(parent_task_id, []):
            if id(child) in released_session_ids:
                continue
            if all(dependency in completed_task_ids for dependency in child.depends_on):
                queue.put_nowait(child)
                released_session_ids.add(id(child))

    async def worker(worker_index: int) -> None:
        nonlocal completed_session_count, first_error
        while True:
            loaded = await queue.get()
            try:
                if loaded is None:
                    return
                if first_error is not None:
                    continue

                admitted_monotonic = time.monotonic()
                prepared: PreparedTraceSession | None = None
                stats: ReplayTaskStats | None = None
                session_error: BaseException | None = None
                preparation_already_finalized = False
                try:
                    logger.info(
                        "Worker %d replaying %s (%d ready)",
                        worker_index,
                        loaded.agent_id,
                        queue.qsize(),
                    )
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
                        container_start_extra_args=container_start_extra_args,
                    )
                except _ReplayPreparationError as exc:
                    prepared = exc.prepared
                    preparation_already_finalized = True
                    stats = _record_prepare_failure(
                        prepared,
                        trace_logger=trace_logger,
                        reason=exc.reason,
                        error=exc.cause,
                        cleanup_error=exc.cleanup_error,
                    )
                except RuntimeError as exc:
                    reason = _container_prep_failure_reason(exc)
                    prepared = _failed_prepared_session(
                        loaded,
                        output_path=output_path,
                        container_executable=container_executable,
                        network_mode=network_mode,
                        reason=reason,
                        error=exc,
                    )
                    stats = _record_prepare_failure(
                        prepared,
                        trace_logger=trace_logger,
                        reason=reason,
                        error=exc,
                    )
                except Exception as exc:
                    session_error = exc
                else:
                    try:
                        stats = await _replay_cloud_model_session(
                            prepared,
                            trace_logger=trace_logger,
                            replay_speed=replay_speed,
                            shadow_generation=shadow_generation,
                            llm_timing=llm_timing,
                            command_timeout_s=command_timeout_s,
                            warmup_skip_iterations=warmup_skip_iterations,
                        )
                    except Exception as exc:
                        session_error = exc
                terminal_monotonic = time.monotonic()
                try:
                    if prepared is not None and not preparation_already_finalized:
                        await _finalize_prepared_session(prepared)
                except Exception as exc:
                    if session_error is None:
                        session_error = exc

                async with result_lock:
                    if prepared is not None:
                        prepared_sessions.append(prepared)
                        if stats is not None:
                            if scheduled_arrivals:
                                planned_arrival = (
                                    arrival_zero_monotonic + loaded.arrival_s
                                )
                                stats = dataclasses.replace(
                                    stats,
                                    admission_wait_s=max(
                                        0.0,
                                        admitted_monotonic - planned_arrival,
                                    ),
                                    ready_to_terminal_s=max(
                                        0.0,
                                        terminal_monotonic - planned_arrival,
                                    ),
                                )
                            task_stats.append(stats)
                    if session_error is not None:
                        if first_error is None:
                            first_error = session_error
                            stop_workers()
                        continue
                    if first_error is not None:
                        continue
                    completed_task_ids.add(loaded.task_instance_id)
                    completed_session_count += 1
                    release_ready_children(loaded.task_instance_id)
                    if completed_session_count == len(loaded_sessions):
                        stop_workers()
            finally:
                # Release the source image after finalize (container stopped),
                # so per-task unique images are removed as soon as no pending
                # session references them.
                if cleanup_state is not None and loaded is not None:
                    await _release_source_image(cleanup_state, loaded.run_instance_id)
                queue.task_done()

    arrival_task = (
        asyncio.create_task(release_scheduled_arrivals())
        if scheduled_arrivals
        else None
    )
    worker_results = await asyncio.gather(
        *(worker(index) for index in range(worker_count)),
        return_exceptions=True,
    )
    if arrival_task is not None:
        if not arrival_task.done():
            arrival_task.cancel()
        await asyncio.gather(arrival_task, return_exceptions=True)
    for result in worker_results:
        if isinstance(result, asyncio.CancelledError):
            raise result
        if isinstance(result, Exception) and first_error is None:
            first_error = result
    if first_error is not None:
        raise first_error
    return (
        prepared_sessions,
        task_stats,
        arrival_zero_wall_time_s if scheduled_arrivals else None,
    )


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
    container_start_extra_args: tuple[str, ...] = (),
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
            container_start_extra_args=container_start_extra_args,
        )
    finally:
        prep_semaphore.release()


async def _run_prepared_cloud_model_sessions(
    prepared_sessions: list[PreparedTraceSession],
    *,
    trace_logger: TraceLogger,
    replay_zero_monotonic: float,
    replay_speed: float,
    shadow_generation: ShadowGenerationConfig | None = None,
    llm_timing: LLMTimingConfig,
    command_timeout_s: float,
    warmup_skip_iterations: int,
) -> list[ReplayTaskStats]:
    results = await asyncio.gather(
        *(
            _replay_cloud_model_session(
                prepared,
                trace_logger=trace_logger,
                replay_zero_monotonic=replay_zero_monotonic,
                replay_speed=replay_speed,
                shadow_generation=shadow_generation,
                llm_timing=llm_timing,
                command_timeout_s=command_timeout_s,
                warmup_skip_iterations=warmup_skip_iterations,
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
    shadow_generation: ShadowGenerationConfig | None = None,
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
    container_start_extra_args: tuple[str, ...] = (),
) -> WorkerReplayResult:
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
        prep_results = await asyncio.gather(
            *(
                _prepare_replay_session_with_shared_limit(
                    loaded,
                    output_path=output_path,
                    container_executable=container_executable,
                    network_mode=network_mode,
                    prep_semaphore=prep_semaphore,
                    fixed_images_by_source=fixed_images_by_source,
                    resource_monitoring_enabled=resource_monitoring_enabled,
                    memory_bandwidth_enabled=memory_bandwidth_enabled,
                    monitoring_policy=monitoring_policy,
                    container_start_extra_args=container_start_extra_args,
                )
                for loaded in loaded_sessions
            ),
            return_exceptions=True,
        )
        replay_sessions: list[PreparedTraceSession] = []
        prep_failures: list[
            tuple[PreparedTraceSession, str, RuntimeError, BaseException | None]
        ] = []
        prep_errors: list[BaseException] = []
        for loaded, result in zip(loaded_sessions, prep_results, strict=True):
            if isinstance(result, _ReplayPreparationError):
                prepared_sessions.append(result.prepared)
                prep_failures.append(
                    (
                        result.prepared,
                        result.reason,
                        result.cause,
                        result.cleanup_error,
                    )
                )
            elif isinstance(result, RuntimeError):
                reason = _container_prep_failure_reason(result)
                prepared = _failed_prepared_session(
                    loaded,
                    output_path=output_path,
                    container_executable=container_executable,
                    network_mode=network_mode,
                    reason=reason,
                    error=result,
                )
                prepared_sessions.append(prepared)
                prep_failures.append((prepared, reason, result, None))
            elif isinstance(result, BaseException):
                prep_errors.append(result)
            else:
                prepared_sessions.append(result)
                replay_sessions.append(result)
        if prep_errors:
            raise SimulateError(
                f"{len(prep_errors)}/{len(prep_results)} worker preparations failed"
            ) from prep_errors[0]
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
                "container_start_extra_args": list(container_start_extra_args),
                "exec_timeout_floor_s": replay_exec_timeout_floor_s(),
                **(
                    {"shadow_generation": shadow_generation_payload(shadow_generation)}
                    if shadow_generation is not None
                    else {}
                ),
            },
        )
        replay_zero_monotonic = await _wait_for_global_replay_start(
            replay_start_barrier,
            replay_start_event,
            replay_start_wall_time,
            coordinator=worker_index == 0,
        )
        replay_started = True
        task_stats = [
            _record_prepare_failure(
                prepared,
                trace_logger=trace_logger,
                reason=reason,
                error=error,
                cleanup_error=cleanup_error,
            )
            for prepared, reason, error, cleanup_error in prep_failures
        ]
        task_stats.extend(
            await _run_prepared_cloud_model_sessions(
                replay_sessions,
                trace_logger=trace_logger,
                replay_zero_monotonic=replay_zero_monotonic,
                replay_speed=replay_speed,
                shadow_generation=shadow_generation,
                llm_timing=llm_timing,
                command_timeout_s=command_timeout_s,
                warmup_skip_iterations=warmup_skip_iterations,
            )
        )
        trace_logger.close()
        return WorkerReplayResult(
            wave_index=wave_index,
            worker_index=worker_index,
            trace_file=str(trace_logger.path),
            task_stats=task_stats,
            task_output_dirs={
                prepared.loaded.run_instance_id: str(prepared.task_output_dir)
                for prepared in prepared_sessions
                if prepared.task_output_dir is not None
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
    shadow_generation: ShadowGenerationConfig | None = None,
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
    container_start_extra_args: tuple[str, ...] = (),
) -> WorkerReplayResult:
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
            shadow_generation=shadow_generation,
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
            container_start_extra_args=container_start_extra_args,
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
    shadow_generation: ShadowGenerationConfig | None = None,
    llm_timing: LLMTimingConfig,
    command_timeout_s: float,
    warmup_skip_iterations: int,
    fixed_images_by_source: dict[str, str] | None,
    resource_monitoring_enabled: bool,
    memory_bandwidth_enabled: bool,
    monitoring_policy: dict[str, object] | None,
    cleanup_state: _ImageCleanupState | None = None,
    container_start_extra_args: tuple[str, ...] = (),
) -> tuple[list[WorkerReplayResult], list[ReplayTaskStats]]:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if any(entry.depends_on for entry in worker_inputs):
        raise SimulateError(
            "Dependency-aware simulation currently requires workers=1; "
            "multi-process worker waves cannot release children immediately "
            "after each parent finishes"
        )
    prep_limit = _resolve_prep_concurrency(prep_concurrency, len(worker_inputs))
    wave_inputs = _ready_worker_input_batches(worker_inputs, batch_size=concurrency)
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
                            shadow_generation=shadow_generation,
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
                            container_start_extra_args=container_start_extra_args,
                        ),
                    )
                    for worker_index, chunk in enumerate(chunks)
                ]
                wave_results = await asyncio.gather(*futures)
            replay_results.extend(
                sorted(wave_results, key=lambda item: item.worker_index)
            )
            for result in sorted(wave_results, key=lambda item: item.worker_index):
                task_stats.extend(result.task_stats)
            # The wave's containers are finalized in their subprocesses before
            # the wave returns, so it is safe to release each session's source
            # image now; refcounts keep images shared across waves alive until
            # their last wave completes.
            if cleanup_state is not None:
                for entry in wave:
                    await _release_source_image(cleanup_state, entry.run_instance_id)
    task_stats.sort(key=lambda stat: stat.manifest_index)
    replay_results.sort(key=lambda item: (item.wave_index, item.worker_index))
    return replay_results, task_stats


async def _replay_cloud_model_action(
    prepared_session: PreparedTraceSession,
    action: dict[str, Any],
    *,
    trace_logger: TraceLogger,
    replay_speed: float,
    llm_timing: LLMTimingConfig,
    command_timeout_s: float,
    warmup_skip_iterations: int,
    source_start_offset_s: float,
    source_gap_sleep: SleepDrift | None,
    replay_agent: Any | None,
) -> _ReplayActionOutcome:
    loaded = prepared_session.loaded
    ctr = prepared_session.container
    action_id = str(action.get("action_id", ""))
    action_type = str(action.get("action_type", ""))
    iteration = int(action.get("iteration", 0))
    data = action.get("data", {})
    action_ts_start, action_ts_end = _coerce_action_bounds(
        action,
        source_trace=loaded.source_trace,
    )
    source_duration_s = max(0.0, action_ts_end - action_ts_start)
    outcome = _ReplayActionOutcome(action_id=action_id)
    action_start_sleep = source_gap_sleep
    if source_gap_sleep is not None:
        outcome.sleep_drifts.append(source_gap_sleep)
    if source_start_offset_s > 0:
        action_start_sleep = await _sleep_and_measure(
            source_start_offset_s / replay_speed,
            phase="source_gap",
        )
        if action_start_sleep is not None:
            outcome.sleep_drifts.append(action_start_sleep)

    try:
        if action_type == "llm_call":
            record_ts_start = time.time()
            sleep_s, llm_timing_fields = llm_replay_duration_s(
                data=data,
                source_duration_s=source_duration_s,
                replay_speed=replay_speed,
                timing_mode=llm_timing.mode,
                llm_ttft_ms=llm_timing.ttft_ms,
                llm_tpot_ms=llm_timing.tpot_ms,
            )
            action_sleep = await _sleep_and_measure(
                sleep_s,
                phase="llm_replay",
            )
            if action_sleep is not None:
                outcome.sleep_drifts.append(action_sleep)
            record_ts_end = time.time()
            record = _make_trace_action(
                loaded=loaded,
                action_type="llm_call",
                action_id=action_id or f"llm_{iteration}",
                iteration=iteration,
                ts_start=record_ts_start,
                ts_end=record_ts_end,
                agent_id=_replay_agent_id_for_action(
                    loaded,
                    action.get("agent_id"),
                ),
                data={
                    "messages_in": data.get("messages_in"),
                    "raw_response": data.get("raw_response", {}),
                    "prompt_tokens": data.get("prompt_tokens", 0),
                    "completion_tokens": data.get("completion_tokens", 0),
                    "llm_latency_ms": (record_ts_end - record_ts_start) * 1000,
                    "simulate_source": str(loaded.source_trace),
                    "source_action_agent_id": action.get("agent_id"),
                    "source_llm_latency_ms": data.get("llm_latency_ms"),
                    "replay_mode": "cloud_model",
                    "replay_speed": replay_speed,
                    **llm_timing_fields,
                    "sim_metrics": {
                        "warmup": iteration < warmup_skip_iterations,
                        **_sleep_drift_metrics(
                            source_gap=action_start_sleep,
                            action_sleep=action_sleep,
                        ),
                    },
                },
            )
            trace_logger.log_trace_action(loaded.agent_id, record)
            outcome.succeeded_actions += 1
            return outcome

        if action_type != "tool_exec":
            logger.warning(
                "Skipping unsupported action_type=%s in %s",
                action_type,
                loaded.source_trace,
            )
            return outcome

        tool_name = data.get("tool_name")
        tool_args = data.get("tool_args", "{}")
        if not tool_name:
            logger.warning(
                "Skipping tool action without tool_name in %s",
                loaded.source_trace,
            )
            return outcome

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
        )
        source_resource_timeline = valid_resource_timeline(
            data.get("resource_timeline")
        )
        original_artifact_path: str | None = None
        mapped_artifact_path: str | None = None
        exec_resource_timeline: dict[str, Any] | None = None
        tool_exec_metadata: dict[str, Any] = {}
        if not source_success:
            outcome.source_failed_actions += 1
        if ctr is None:
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
                outcome.sleep_drifts.append(action_sleep)
            duration_ms = (time.time() - record_ts_start) * 1000
        elif tool_name in _CONTROL_PLANE_NOOP_TOOLS:
            action_sleep = await _sleep_and_measure(
                source_duration_ms / 1000 / replay_speed,
                phase="tool_trace_replay",
            )
            if action_sleep is not None:
                outcome.sleep_drifts.append(action_sleep)
            tool_result = data.get("tool_result", data.get("result", ""))
            if not tool_result:
                tool_result = _CONTROL_PLANE_NOOP_RESULTS[tool_name]
            tool_success = source_success
            duration_ms = (time.time() - record_ts_start) * 1000
            replay_source = "message_noop" if tool_name == "message" else "control_noop"
        elif tool_name.startswith("mcp_"):
            action_sleep = await _sleep_and_measure(
                source_duration_ms / 1000 / replay_speed,
                phase="tool_trace_replay",
            )
            if action_sleep is not None:
                outcome.sleep_drifts.append(action_sleep)
            tool_result = data.get("tool_result", "")
            tool_success = source_success
            duration_ms = (time.time() - record_ts_start) * 1000
            replay_source = "replayed_from_trace"
        else:
            (
                mapped_tool_args,
                original_artifact_path,
                mapped_artifact_path,
                mapped_exists,
            ) = _remap_runtime_artifact_tool_args(
                    tool_name=tool_name,
                    tool_args_json=tool_args,
                    runtime_root_map=prepared_session.runtime_artifact_root_map,
                )
            if original_artifact_path is None and isinstance(tool_args, str):
                from trace_collect.openclaw_tools import (
                    source_runtime_artifact_path_from_tool_call,
                )

                original_artifact_path = source_runtime_artifact_path_from_tool_call(
                    tool_name=tool_name,
                    tool_args_json=tool_args,
                )
            if original_artifact_path is not None and not mapped_exists:
                action_sleep = await _sleep_and_measure(
                    source_duration_ms / 1000 / replay_speed,
                    phase="tool_trace_replay",
                )
                if action_sleep is not None:
                    outcome.sleep_drifts.append(action_sleep)
                tool_result = _artifact_unavailable_result(original_artifact_path)
                tool_success = False
                duration_ms = (time.time() - record_ts_start) * 1000
                replay_source = "source_artifact_unavailable"
            else:
                if replay_agent is None:
                    raise RuntimeError(
                        f"No replay agent assigned for tool action {action_id!r}"
                    )
                agent = replay_agent
                exec_resource_timeline = (
                    source_resource_timeline
                    if _tool_uses_single_exec_command_semantics(
                        tool_name,
                        mapped_tool_args,
                    )
                    else None
                )
                if exec_resource_timeline is None:
                    if mapped_artifact_path is not None:
                        (
                            tool_result,
                            duration_ms,
                            tool_success,
                            tool_exec_metadata,
                        ) = _unpack_exec_tool_result(
                            await _exec_tool(
                                agent,
                                tool_name,
                                mapped_tool_args,
                                command_timeout_s,
                                source_exec_timeout,
                                True,
                            )
                        )
                    else:
                        (
                            tool_result,
                            duration_ms,
                            tool_success,
                            tool_exec_metadata,
                        ) = _unpack_exec_tool_result(
                            await _exec_tool(
                                agent,
                                tool_name,
                                mapped_tool_args,
                                command_timeout_s,
                                source_exec_timeout,
                            )
                        )
                elif mapped_artifact_path is not None:
                    (
                        tool_result,
                        duration_ms,
                        tool_success,
                        tool_exec_metadata,
                    ) = _unpack_exec_tool_result(
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
                else:
                    (
                        tool_result,
                        duration_ms,
                        tool_success,
                        tool_exec_metadata,
                    ) = _unpack_exec_tool_result(
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
                replay_source = (
                    "restored_runtime_artifact"
                    if mapped_artifact_path is not None
                    else "executed_in_container"
                )
        replay_failure_kind = _replay_failure_kind(tool_exec_metadata)
        source_artifact_unavailable = replay_source == "source_artifact_unavailable"
        if not tool_success:
            outcome.replay_failed_actions += 1
            if not source_artifact_unavailable:
                if replay_failure_kind in _REPLAY_EXECUTION_FAILURE_KINDS:
                    outcome.replay_execution_errors += 1
                elif source_success:
                    outcome.unexpected_replay_failed_actions += 1
        if source_artifact_unavailable:
            outcome.fatal_replay_errors += 1
        record_ts_end = time.time()
        extra_tool_fields = _command_metadata(
            tool_name=tool_name,
            tool_args_json=tool_args,
            tool_result=str(tool_result),
            tool_success=tool_success,
        )
        if original_artifact_path is not None:
            extra_tool_fields["source_artifact_path"] = original_artifact_path
        if mapped_artifact_path is not None:
            extra_tool_fields["simulator_artifact_path"] = mapped_artifact_path
        extra_tool_fields.update(tool_exec_metadata)
        if source_exec_timeout is not None:
            extra_tool_fields["source_exec_timeout_s"] = source_exec_timeout
        if source_resource_timeline is not None:
            extra_tool_fields["source_resource_timeline"] = source_resource_timeline
            extra_tool_fields["resource_timeout_policy"] = (
                "resource_integrated"
                if exec_resource_timeline is not None
                else "wall_clock"
            )
        tool_record = _make_trace_action(
            loaded=loaded,
            action_type="tool_exec",
            action_id=action_id or f"tool_{iteration}_{tool_name}",
            iteration=iteration,
            ts_start=record_ts_start,
            ts_end=record_ts_end,
            agent_id=_replay_agent_id_for_action(
                loaded,
                action.get("agent_id"),
            ),
            data={
                "tool_name": tool_name,
                "tool_args": tool_args,
                "tool_result": tool_result,
                "duration_ms": duration_ms,
                "success": tool_success,
                "source_success": source_success,
                **extra_tool_fields,
                "simulate_source": str(loaded.source_trace),
                "source_action_agent_id": action.get("agent_id"),
                "source_duration_ms": source_duration_ms,
                "replay_mode": "cloud_model",
                "replay_speed": replay_speed,
                "replay_source": replay_source,
                "sim_metrics": {
                    "warmup": iteration < warmup_skip_iterations,
                    "source": replay_source,
                    "sim_tool_format": replay_source
                    if replay_source
                    in {
                        "skipped_host_mode",
                        "message_noop",
                        "control_noop",
                        "replayed_from_trace",
                        "source_artifact_unavailable",
                        "restored_runtime_artifact",
                    }
                    else "container_exec",
                    **_sleep_drift_metrics(
                        source_gap=action_start_sleep,
                        action_sleep=action_sleep,
                    ),
                },
            },
        )
        trace_logger.log_trace_action(loaded.agent_id, tool_record)
        if tool_success:
            outcome.succeeded_actions += 1
    except Exception as exc:
        logger.error(
            "Replay action failed for %s action=%s: %s",
            loaded.agent_id,
            action_id,
            exc,
        )
        outcome.replay_action_errors += 1
    return outcome


async def _replay_cloud_model_session(
    prepared_session: PreparedTraceSession,
    *,
    trace_logger: TraceLogger,
    replay_zero_monotonic: float | None = None,
    replay_speed: float,
    shadow_generation: ShadowGenerationConfig | None = None,
    tool_gap_loan: ToolGapLoanConfig | None = None,
    llm_timing: LLMTimingConfig,
    command_timeout_s: float,
    warmup_skip_iterations: int,
) -> ReplayTaskStats:
    loaded = prepared_session.loaded
    if loaded.scaffold == "openclaw":
        if prepared_session.container is None:
            raise SimulateError(
                f"OpenClaw replay for {loaded.task_instance_id!r} requires a task container"
            )
        if not any(
            action.get("action_type") == "llm_call" for action in loaded.actions
        ):
            raise SimulateError(
                f"OpenClaw replay for {loaded.task_instance_id!r} has no source llm_call actions"
            )
        if replay_zero_monotonic is not None:
            await _sleep_until_monotonic(replay_zero_monotonic)
        return await _run_openclaw_replay_session(
            prepared_session,
            trace_logger=trace_logger,
            replay_speed=replay_speed,
            shadow_generation=shadow_generation,
            tool_gap_loan=tool_gap_loan,
            llm_timing=llm_timing,
            command_timeout_s=command_timeout_s,
            warmup_skip_iterations=warmup_skip_iterations,
        )
    source_model = _source_model(loaded)
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
    replay_action_errors = 0
    fatal_replay_errors = 0
    source_failed_actions = 0
    replay_failed_actions = 0
    replay_execution_errors = 0
    unexpected_replay_failed_actions = 0
    previous_source_end: float | None = None
    sleep_drifts: list[SleepDrift] = []

    if replay_zero_monotonic is not None:
        start_drift = await _sleep_until_monotonic(replay_zero_monotonic)
        if start_drift is not None:
            sleep_drifts.append(start_drift)

    for batch in _replay_action_batches(
        loaded.actions,
        source_trace=loaded.source_trace,
    ):
        bounds = [
            _coerce_action_bounds(action, source_trace=loaded.source_trace)
            for action in batch
        ]
        batch_start = min(start for start, _ in bounds)
        batch_end = max(end for _, end in bounds)

        ordered_batch = sorted(
            batch,
            key=lambda action: _replay_action_bounds_key(
                action,
                source_trace=loaded.source_trace,
            ),
        )
        replay_agents = await _prewarm_replay_agents_for_batch(
            prepared_session,
            ordered_batch,
        )
        source_gap_sleep = await _sleep_source_gap(
            previous_source_end=previous_source_end,
            action_source_start=batch_start,
            replay_speed=replay_speed,
        )
        previous_source_end = max(
            batch_end,
            previous_source_end if previous_source_end is not None else batch_end,
        )
        outcomes = await asyncio.gather(
            *(
                _replay_cloud_model_action(
                    prepared_session,
                    action,
                    trace_logger=trace_logger,
                    replay_speed=replay_speed,
                    llm_timing=llm_timing,
                    command_timeout_s=command_timeout_s,
                    warmup_skip_iterations=warmup_skip_iterations,
                    source_start_offset_s=max(
                        0.0,
                        _coerce_action_bounds(
                            action,
                            source_trace=loaded.source_trace,
                        )[0]
                        - batch_start,
                    ),
                    source_gap_sleep=source_gap_sleep if index == 0 else None,
                    replay_agent=replay_agents.get(id(action)),
                )
                for index, action in enumerate(ordered_batch)
            )
        )
        for outcome in outcomes:
            succeeded_actions += outcome.succeeded_actions
            replay_action_errors += outcome.replay_action_errors
            fatal_replay_errors += outcome.fatal_replay_errors
            source_failed_actions += outcome.source_failed_actions
            replay_failed_actions += outcome.replay_failed_actions
            replay_execution_errors += outcome.replay_execution_errors
            unexpected_replay_failed_actions += outcome.unexpected_replay_failed_actions
            sleep_drifts.extend(outcome.sleep_drifts)

    wall_end = time.time()
    failed_actions = (
        replay_action_errors
        + fatal_replay_errors
        + replay_execution_errors
        + unexpected_replay_failed_actions
    )
    success = failed_actions == 0
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
                "succeeded_actions": succeeded_actions,
                "failed_actions": failed_actions,
                "source_failed_actions": source_failed_actions,
                "replay_failed_actions": replay_failed_actions,
                "replay_execution_errors": replay_execution_errors,
                "unexpected_replay_failed_actions": unexpected_replay_failed_actions,
                "fatal_replay_errors": fatal_replay_errors,
                "replay_action_errors": replay_action_errors,
                "sleep_drift": _summarize_sleep_drifts(sleep_drifts),
            },
        ),
    )
    return _make_task_stats(
        loaded=loaded,
        success=success,
        elapsed_s=wall_end - wall_start,
        failed_action_count=failed_actions,
    )


async def simulate(
    *,
    manifest: Path,
    task_source: Path | None = None,
    output_dir: Path,
    mode: str = "cloud_model",
    concurrency: int = 1,
    workers: int = 1,
    prep_concurrency: int = 0,
    container_executable: str | None = None,
    network_mode: str = "host",
    api_base: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    command_timeout_s: float = 120.0,
    warmup_skip_iterations: int = 0,
    replay_speed: float = 1.0,
    shadow_llm_api_base: str | None = None,
    shadow_llm_model: str | None = None,
    shadow_llm_timeout_s: float = 120.0,
    shadow_llm_seed: int = 0,
    shadow_llm_max_concurrency: int | None = None,
    shadow_llm_mode: ShadowGenerationMode = "vllm",
    shadow_llm_cachewise_predictor_checkout: Path | None = None,
    shadow_llm_cachewise_models_dir: Path | None = None,
    tool_gap_loan_arm: str | None = None,
    tool_gap_predictions: Path | None = None,
    tool_gap_borrower_priority: int | None = None,
    resource_monitoring: MonitoringMode = "auto",
    pmu_monitoring: MonitoringMode = "auto",
    memory_bandwidth_monitoring: MonitoringMode = "auto",
    llm_timing_mode: str = "source_scaled",
    llm_ttft_ms: float | None = None,
    llm_tpot_ms: float | None = None,
    structured_output: bool = False,
    tool_resource_profile: Path | None = None,
    cleanup_images: bool = False,
    container_start_extra_args: tuple[str, ...] = (),
    stage_all_before_replay: bool = False,
) -> Path:
    if mode != "cloud_model":
        raise ValueError(f"Unsupported simulate mode: {mode}")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if stage_all_before_replay and workers != 1:
        raise ValueError("stage_all_before_replay requires workers=1")
    if prep_concurrency < 0:
        raise ValueError("prep_concurrency must be >= 0")
    if (shadow_llm_api_base is None) != (shadow_llm_model is None):
        raise ValueError("shadow_llm_api_base and shadow_llm_model must be supplied together")
    if shadow_llm_max_concurrency is not None:
        if (
            isinstance(shadow_llm_max_concurrency, bool)
            or shadow_llm_max_concurrency < 1
        ):
            raise ValueError("shadow_llm_max_concurrency must be >= 1")
        if shadow_llm_api_base is None:
            raise ValueError("shadow_llm_max_concurrency requires shadow generation")
    if shadow_llm_mode != "vllm" and shadow_llm_api_base is None:
        raise ValueError("non-vLLM shadow mode requires shadow generation")
    cachewise_paths = (
        shadow_llm_cachewise_predictor_checkout,
        shadow_llm_cachewise_models_dir,
    )
    if shadow_llm_mode == "cachewise":
        if any(path is None for path in cachewise_paths):
            raise ValueError("CacheWise mode requires predictor and model paths")
    elif any(path is not None for path in cachewise_paths):
        raise ValueError("CacheWise paths require CacheWise mode")
    if tool_gap_loan_arm not in {None, "fixed", "feedback", "predictor"}:
        raise ValueError(f"unknown tool-gap loan arm: {tool_gap_loan_arm}")
    if tool_gap_loan_arm == "predictor" and tool_gap_predictions is None:
        raise ValueError("predictor tool-gap loan requires a prediction file")
    if tool_gap_loan_arm != "predictor" and tool_gap_predictions is not None:
        raise ValueError("tool-gap predictions are only valid for the predictor arm")
    if tool_gap_borrower_priority is not None:
        if (
            not isinstance(tool_gap_borrower_priority, int)
            or isinstance(tool_gap_borrower_priority, bool)
            or tool_gap_borrower_priority < 1
        ):
            raise ValueError("tool-gap borrower priority must be a positive integer")
        if tool_gap_loan_arm is None:
            raise ValueError("tool-gap borrower priority requires a tool-gap loan arm")
    shadow_generation = None
    if shadow_llm_api_base is not None:
        if replay_speed != 1.0:
            raise ValueError("shadow generation requires replay_speed=1.0")
        shadow_generation = ShadowGenerationConfig(
            api_base=shadow_llm_api_base,
            model=shadow_llm_model,
            timeout_s=shadow_llm_timeout_s,
            seed=shadow_llm_seed,
            mode=shadow_llm_mode,
            cachewise_predictor_checkout=(
                str(shadow_llm_cachewise_predictor_checkout.resolve())
                if shadow_llm_cachewise_predictor_checkout is not None
                else None
            ),
            cachewise_models_dir=(
                str(shadow_llm_cachewise_models_dir.resolve())
                if shadow_llm_cachewise_models_dir is not None
                else None
            ),
        )
    exec_timeout_floor_s = replay_exec_timeout_floor_s()
    resolved_tool_resource_profile: Path | None = None
    if tool_resource_profile is None:
        os.environ.pop("TOOL_RESOURCE_PROFILE", None)
    else:
        from tool_resource.profile import ResourceProfile

        resolved_tool_resource_profile = tool_resource_profile.resolve()
        ResourceProfile.load(resolved_tool_resource_profile)
        os.environ["TOOL_RESOURCE_PROFILE"] = str(resolved_tool_resource_profile)
    llm_timing = LLMTimingConfig(
        mode=llm_timing_mode,
        ttft_ms=llm_ttft_ms,
        tpot_ms=llm_tpot_ms,
    )
    _validate_llm_timing_config(llm_timing, replay_speed=replay_speed)

    manifest_entries = _load_simulate_manifest(
        manifest,
        default_task_source=task_source.resolve() if task_source is not None else None,
    )
    if any(entry.requires_trace_tool_replay for entry in manifest_entries):
        if not replay_trace_tools_enabled():
            raise ValueError(
                "simulate manifest requires OPENCLAW_REPLAY_TRACE_TOOLS=1"
            )

    loaded_sessions = [
        _load_trace_session(
            entry.trace,
            entry.task_source,
            manifest_index=entry.index,
            docker_image_override=entry.docker_image,
            label=entry.label,
            manifest_depends_on=entry.depends_on,
            arrival_s=entry.arrival_s,
        )
        for entry in manifest_entries
    ]
    _assign_replay_instance_ids(loaded_sessions)
    non_openclaw_sessions = [
        session.task_instance_id
        for session in loaded_sessions
        if session.scaffold != "openclaw"
    ]
    if any(entry.requires_trace_tool_replay for entry in manifest_entries):
        if non_openclaw_sessions:
            raise ValueError(
                "trace-tool-replay-only manifest requires OpenClaw for every trace; "
                "non-OpenClaw tasks: " + ", ".join(non_openclaw_sessions)
            )
    tool_gap_prediction_map: dict[str, tuple[ToolGapPrediction, ...]] | None = None
    if tool_gap_loan_arm is not None:
        if workers != 1 or not stage_all_before_replay:
            raise ValueError(
                "tool-gap loan requires workers=1 and stage_all_before_replay"
            )
        if concurrency != 4 or len(loaded_sessions) != 8:
            raise ValueError("tool-gap loan requires concurrency=4 and exactly 8 traces")
        if shadow_generation is None:
            raise ValueError("tool-gap loan requires shadow generation")
        if shadow_llm_max_concurrency is not None:
            raise ValueError("tool-gap loan forbids shadow request admission caps")
        if tool_gap_loan_arm == "predictor":
            assert tool_gap_predictions is not None
            tool_gap_prediction_map = _load_tool_gap_predictions(
                tool_gap_predictions,
                loaded_sessions,
            )
        else:
            tool_gap_prediction_map = {}
    _validate_loaded_sessions(
        loaded_sessions,
        mode=mode,
        replay_speed=replay_speed,
        llm_timing=llm_timing,
    )
    has_dependencies = _has_session_dependencies(loaded_sessions)
    has_scheduled_arrivals = any(
        session.arrival_s > 0 for session in loaded_sessions
    )
    if has_scheduled_arrivals and stage_all_before_replay:
        raise ValueError("scheduled arrivals do not support stage_all_before_replay")
    if has_scheduled_arrivals and workers != 1:
        raise ValueError("scheduled arrivals require workers=1")
    if has_scheduled_arrivals and has_dependencies:
        raise ValueError("scheduled arrivals do not support task dependencies")
    if stage_all_before_replay and has_dependencies:
        raise ValueError(
            "stage_all_before_replay does not support task dependencies"
        )
    if shadow_generation is not None and non_openclaw_sessions:
        raise ValueError(
            "shadow generation requires OpenClaw for every selected trace; "
            "non-OpenClaw tasks: " + ", ".join(non_openclaw_sessions)
        )
    if shadow_generation is not None and shadow_generation.mode == "continuum_public":
        missing_step_limits = [
            session.task_instance_id
            for session in loaded_sessions
            if not isinstance((session.metadata or {}).get("max_iterations"), int)
            or isinstance((session.metadata or {}).get("max_iterations"), bool)
            or int((session.metadata or {})["max_iterations"]) < 1
        ]
        if missing_step_limits:
            raise ValueError(
                "Continuum public mode requires causal max_iterations metadata: "
                + ", ".join(missing_step_limits[:4])
            )
    if {"--cpus", "--cpuset-cpus"}.intersection(container_start_extra_args):
        if non_openclaw_sessions:
            raise ValueError(
                "container CPU controls require OpenClaw for every selected trace; "
                "non-OpenClaw tasks: " + ", ".join(non_openclaw_sessions)
            )
        terminal_bench_sessions = [
            session.task_instance_id
            for session in loaded_sessions
            if _is_terminal_bench_registry_task(session)
        ]
        if terminal_bench_sessions:
            raise ValueError(
                "container CPU controls do not support Terminal-Bench compose tasks: "
                + ", ".join(terminal_bench_sessions)
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
    # When --cleanup-images is on, skip the global prefetch: each task pulls its
    # source image on demand (ensure_fixed_image -> ensure_source_image) and the
    # image is removed once no pending session references it. Prefetching all
    # per-task-unique images up front can exceed disk before any task replays.
    cleanup_state: _ImageCleanupState | None = None
    if cleanup_images:
        cleanup_state = _build_image_cleanup_state(
            loaded_sessions,
            container_executable=container_executable,
        )
        logger.info(
            "cleanup-images: on-demand pulls with per-task removal for "
            "%d source image(s) across %d session(s)",
            len(cleanup_state.refcounts),
            len(cleanup_state.image_by_instance),
        )
    else:
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
    resource_runs: dict[str, Any] = {}
    run_wall_start: float | None = None
    run_wall_end: float | None = None
    common_ready_wall_time_s: float | None = None
    arrival_zero_wall_time_s: float | None = None
    output_path.mkdir(parents=True, exist_ok=True)
    if shadow_generation is not None and shadow_llm_max_concurrency is not None:
        slot_dir = output_path / ".shadow-llm-admission"
        slot_dir.mkdir(exist_ok=True)
        slot_paths: list[str] = []
        for slot_index in range(shadow_llm_max_concurrency):
            slot_path = slot_dir / f"slot-{slot_index:03d}.lock"
            slot_path.touch(exist_ok=True)
            slot_paths.append(str(slot_path.resolve()))
        shadow_generation = dataclasses.replace(
            shadow_generation,
            admission_slot_paths=tuple(slot_paths),
        )
    if has_dependencies and workers > 1:
        raise SimulateError(
            "Dependency-aware simulation currently requires workers=1; "
            "multi-process worker waves cannot release children immediately "
            "after each parent finishes"
        )
    if tool_gap_loan_arm is not None:
        scheduler_mode = "staged_tool_gap_loan"
    elif stage_all_before_replay:
        scheduler_mode = "staged_bounded_queue"
    elif has_dependencies:
        scheduler_mode = "dependency_queue"
    elif has_scheduled_arrivals:
        scheduler_mode = "scheduled_arrival_queue"
    else:
        scheduler_mode = "bounded_queue" if workers == 1 else "multi_process_workers"

    try:
        if cleanup_images:
            # Prebuilding sweep fixed images pulls every source image up front
            # (ensure_fixed_image is a pull-through passthrough), which defeats
            # the on-demand cleanup budget; leave it empty so each task pulls
            # and releases its own image.
            sweep_fixed_images = {}
        else:
            sweep_fixed_images = await _prebuild_sweep_fixed_images(
                loaded_sessions,
                output_path=output_path,
                container_executable=container_executable,
            )
        run_wall_start = time.monotonic()
        run_id = _build_run_id(mode=mode, model=model, concurrency=concurrency)
        if resolved_tool_resource_profile is not None:
            from tool_resource.client import ResourceRun

            manifest_dir = output_path / "tool_resource_runs" / run_id
            for scope in sorted(
                {_tool_resource_scope(item) for item in loaded_sessions}
            ):
                scope_digest = hashlib.sha256(scope.encode()).hexdigest()[:16]
                resource_runs[scope] = ResourceRun.open(
                    resolved_tool_resource_profile,
                    run_id=f"simulate:{run_id}:{scope_digest}",
                    workspace_scope=scope,
                    manifest_path=manifest_dir / f"{scope_digest}.json",
                )
            os.environ[_TOOL_RESOURCE_RUN_TOKENS_ENV] = json.dumps(
                {
                    scope: resource_run.run_token or ""
                    for scope, resource_run in resource_runs.items()
                },
                sort_keys=True,
            )
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
                    "stage_all_before_replay": stage_all_before_replay,
                    "monitoring": monitoring_policy_dict,
                    "container_start_extra_args": list(container_start_extra_args),
                    "exec_timeout_floor_s": exec_timeout_floor_s,
                    **(
                        {
                            "shadow_generation": shadow_generation_payload(
                                shadow_generation
                            )
                        }
                        if shadow_generation is not None
                        else {}
                    ),
                    **(
                        {
                            "tool_gap_loan": {
                                "arm": tool_gap_loan_arm,
                                "state_dir": str(output_path / ".tool-gap-loan"),
                                "foreground_task_ids": [
                                    item.task_instance_id for item in loaded_sessions[:4]
                                ],
                                "waiting_task_ids": [
                                    item.task_instance_id for item in loaded_sessions[4:]
                                ],
                                "prediction_file": (
                                    str(tool_gap_predictions)
                                    if tool_gap_predictions is not None
                                    else None
                                ),
                                "borrower_priority": tool_gap_borrower_priority,
                            }
                        }
                        if tool_gap_loan_arm is not None
                        else {}
                    ),
                    "tool_resource": {
                        "profile": (
                            str(tool_resource_profile.resolve())
                            if tool_resource_profile is not None
                            else None
                        ),
                        "service_enabled": tool_resource_profile is not None,
                    },
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
            queue_kwargs = {
                "output_path": output_path,
                "trace_logger": trace_logger,
                "concurrency": concurrency,
                "container_executable": container_executable,
                "network_mode": network_mode,
                "container_resource_recorder": container_resource_recorder,
                "replay_speed": replay_speed,
                "shadow_generation": shadow_generation,
                "llm_timing": llm_timing,
                "command_timeout_s": command_timeout_s,
                "warmup_skip_iterations": warmup_skip_iterations,
                "fixed_images_by_source": sweep_fixed_images,
                "resource_monitoring_enabled": (
                    monitoring_policy.per_task_resource_enabled
                ),
                "memory_bandwidth_enabled": (
                    monitoring_policy.memory_bandwidth_enabled
                ),
                "monitoring_policy": monitoring_policy_dict,
                "container_start_extra_args": container_start_extra_args,
            }
            if stage_all_before_replay:
                (
                    prepared_sessions,
                    task_stats,
                    common_ready_wall_time_s,
                ) = await _run_staged_cloud_model_queue(
                    loaded_sessions,
                    prep_concurrency=prep_concurrency,
                    cleanup_state=cleanup_state,
                    tool_gap_arm=tool_gap_loan_arm,
                    tool_gap_predictions=tool_gap_prediction_map,
                    tool_gap_borrower_priority=tool_gap_borrower_priority,
                    **queue_kwargs,
                )
            else:
                (
                    prepared_sessions,
                    task_stats,
                    arrival_zero_wall_time_s,
                ) = await _run_cloud_model_queue(
                    loaded_sessions,
                    cleanup_state=cleanup_state,
                    **queue_kwargs,
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
                shadow_generation=shadow_generation,
                llm_timing=llm_timing,
                command_timeout_s=command_timeout_s,
                warmup_skip_iterations=warmup_skip_iterations,
                fixed_images_by_source=sweep_fixed_images,
                resource_monitoring_enabled=monitoring_policy.per_task_resource_enabled,
                memory_bandwidth_enabled=monitoring_policy.memory_bandwidth_enabled,
                monitoring_policy=monitoring_policy_dict,
                cleanup_state=cleanup_state,
                container_start_extra_args=container_start_extra_args,
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
                exec_timeout_floor_s=exec_timeout_floor_s,
                shadow_generation=(
                    shadow_generation_payload(shadow_generation)
                    if shadow_generation is not None
                    else None
                ),
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
            if stage_all_before_replay:
                try:
                    if trace_logger is not None:
                        trace_logger.close()
                        _split_trace_by_agent(trace_logger.path, prepared_sessions)
                except (Exception, asyncio.CancelledError) as exc:
                    finalization_error = exc
                try:
                    await _cleanup_staged_sessions(
                        prepared_sessions,
                        cleanup_state,
                    )
                except (Exception, asyncio.CancelledError) as exc:
                    if finalization_error is None:
                        finalization_error = exc
                    else:
                        logger.error("Staged cleanup also failed: %s", exc)
            else:
                try:
                    if trace_logger is not None:
                        trace_logger.close()
                        _split_trace_by_agent(trace_logger.path, prepared_sessions)
                    for prepared in prepared_sessions:
                        await _finalize_prepared_session(prepared)
                except (Exception, asyncio.CancelledError) as exc:
                    finalization_error = exc
            for resource_run in resource_runs.values():
                resource_error = resource_run.finalize(
                    workload_status=(
                        "completed"
                        if run_completed_for_fixed_cleanup
                        and finalization_error is None
                        else "failed"
                    )
                )
                if resource_error is not None:
                    logger.error("%s", resource_error)
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
            os.environ.pop(_TOOL_RESOURCE_RUN_TOKENS_ENV, None)
            if finalization_error is not None:
                raise finalization_error

    if run_wall_start is None or run_wall_end is None:
        raise AssertionError("simulate wall-clock measurement was not recorded")
    trace_file = output_path / f"{run_id}.jsonl"
    tool_gap_summary = None
    if tool_gap_loan_arm is not None:
        tool_gap_summary = json.loads(
            (output_path / ".tool-gap-loan" / "summary.json").read_text(
                encoding="utf-8"
            )
        )
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
        arrival_zero_wall_time_s=arrival_zero_wall_time_s,
        common_ready_wall_time_s=common_ready_wall_time_s,
        tool_gap_loan=tool_gap_summary,
    )
    if cleanup_state is not None:
        logger.info(
            "cleanup-images: %d source image removal(s) skipped due to errors",
            cleanup_state.skipped,
        )
    logger.info("Simulate complete [%s] -> %s", mode, trace_file)
    return trace_file
