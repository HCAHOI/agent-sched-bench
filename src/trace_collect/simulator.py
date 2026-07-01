from __future__ import annotations

import asyncio
import dataclasses
import functools
import hashlib
import json
import logging
import multiprocessing
import subprocess
import shutil
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import BrokenBarrierError
from typing import Any

import yaml

from agents.base import TraceAction
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
GLOBAL_CONTAINER_RESOURCE_SAMPLE_INTERVAL_S = 1.0
_DEFAULT_PREP_CONCURRENCY = 20
_SHARED_SEMAPHORE_POLL_S = 0.05
_REPLAY_START_DELAY_S = 0.1


class SimulateError(Exception):
    """Raised when simulation encounters a fatal issue."""


@dataclass(frozen=True, slots=True)
class TraceManifestEntry:
    """One resolved trace entry from a simulate manifest."""

    index: int
    trace: Path
    task_source: Path
    docker_image: str | None = None
    label: str | None = None


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


@dataclass(frozen=True, slots=True)
class LLMTimingConfig:
    """LLM duration model for cloud replay."""

    mode: str = "source_scaled"
    ttft_ms: float | None = None
    tpot_ms: float | None = None


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


def _tool_mismatch_reason(
    *,
    source_success: bool,
    tool_success: bool,
    replay_source: str,
    source_tool_result: Any,
    replay_tool_result: Any,
    tool_name: str | None,
    tool_args_json: Any,
) -> str | None:
    if replay_source == "source_artifact_unavailable":
        return "source_artifact_unavailable"
    if source_success != tool_success:
        return "tool_success_mismatch"
    if _tool_uses_exec_semantics(tool_name, tool_args_json):
        source_exit = _command_exit_code(str(source_tool_result or ""))
        replay_exit = _command_exit_code(str(replay_tool_result or ""))
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
    spec.setdefault("kind", "filesystem_tar")
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


def _restore_checkpoint_archive_in_container(
    *,
    container_id: str,
    container_executable: str,
    container_archive_path: str,
    restore_root: str,
) -> None:
    script = r'''
import os, shutil, tarfile
archive = os.environ["CHECKPOINT_ARCHIVE"]
root = os.path.abspath(os.environ["CHECKPOINT_ROOT"])
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
try:
    with tarfile.open(archive, "r:*") as tf:
        members = tf.getmembers()
        for member in members:
            if member.issym() or member.islnk():
                raise RuntimeError(f"checkpoint links are unsupported: {member.name}")
            target = os.path.abspath(os.path.join(root, member.name))
            if target != root and not target.startswith(root + os.sep):
                raise RuntimeError(f"unsafe checkpoint member: {member.name}")
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path)
            else:
                os.unlink(path)
        for member in members:
            tf.extract(member, root)
finally:
    if os.path.exists(archive):
        os.unlink(archive)
'''
    _run_checked_container_command(
        [
            container_executable,
            "exec",
            "-e",
            f"CHECKPOINT_ARCHIVE={container_archive_path}",
            "-e",
            f"CHECKPOINT_ROOT={restore_root}",
            container_id,
            "python3",
            "-c",
            script,
        ],
        timeout=600,
    )


def _restore_checkpoint_to_container(
    *,
    checkpoint_spec: dict[str, Any],
    container: PreparedContainer,
) -> dict[str, Any]:
    checkpoint_path = Path(str(checkpoint_spec["path"]))
    if not checkpoint_path.is_file():
        return {
            "forced_sync_success": False,
            "forced_sync_error": f"checkpoint not found: {checkpoint_path}",
        }
    kind = str(checkpoint_spec.get("kind") or "filesystem_tar")
    if kind not in {"filesystem_tar", "filesystem_tar_gz", "tar", "tar_gz"}:
        return {
            "forced_sync_success": False,
            "forced_sync_error": f"unsupported checkpoint kind: {kind}",
        }
    restore_root = str(checkpoint_spec.get("root") or "/testbed")
    container_archive_path = f"/tmp/agent_sched_checkpoint_{uuid.uuid4().hex}.tar"
    started = time.monotonic()
    _copy_checkpoint_archive_to_container(
        checkpoint_path=checkpoint_path,
        container_id=container.container_id,
        container_executable=container.container_executable,
        container_archive_path=container_archive_path,
    )
    _restore_checkpoint_archive_in_container(
        container_id=container.container_id,
        container_executable=container.container_executable,
        container_archive_path=container_archive_path,
        restore_root=restore_root,
    )
    return {
        "forced_sync_success": True,
        "forced_sync_elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        "forced_sync_checkpoint": str(checkpoint_path),
        "forced_sync_checkpoint_kind": kind,
        "forced_sync_root": restore_root,
    }


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
    """Read one canonical trace once and extract the primary replay lane."""

    metadata: dict[str, Any] | None = None
    first_agent_id: str | None = None
    actions: list[dict[str, Any]] = []
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
                if first_agent_id is None:
                    first_agent_id = agent_id
                if agent_id == first_agent_id:
                    actions.append(record)
                continue

            if record_type == "summary" and agent_id:
                summaries[agent_id] = record

    if first_agent_id is None or not actions:
        raise SimulateError(f"No action records with agent_id found in {trace_path}")

    actions.sort(
        key=lambda action: (
            float(action.get("ts_start", 0.0)),
            float(action.get("ts_end", 0.0)),
            int(action.get("iteration", 0)),
            str(action.get("action_id", "")),
        )
    )
    return first_agent_id, metadata, actions, summaries.get(first_agent_id)


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


def _load_simulate_manifest(
    manifest: Path,
    *,
    default_task_source: Path,
) -> list[TraceManifestEntry]:
    try:
        raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SimulateError(f"Invalid simulate manifest YAML: {manifest}") from exc

    base_dir = manifest.parent
    manifest_default_task_source: Path | None = None
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
        unknown_default_keys = set(defaults) - {"task_source"}
        if unknown_default_keys:
            keys = ", ".join(sorted(str(key) for key in unknown_default_keys))
            raise SimulateError(f"simulate manifest defaults has unsupported keys: {keys}")
        if "task_source" in defaults:
            manifest_default_task_source = _resolve_manifest_path(
                defaults["task_source"],
                base_dir=base_dir,
                field="defaults.task_source",
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

        if isinstance(entry, str):
            trace_value = entry
        elif isinstance(entry, dict):
            allowed_entry_keys = {"trace", "task_source", "docker_image", "label"}
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
        if _is_host_mode(session):
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
    container_sessions = [session.agent_id for session in sessions if not _is_host_mode(session)]
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
        if _is_host_mode(session):
            continue
        docker_image = _resolve_docker_image(session)
        if docker_image is None:
            continue
        images.add(normalize_image_reference(docker_image))
    return sorted(images)


def _has_container_mode_sessions(sessions: list[LoadedTraceSession]) -> bool:
    return any(not _is_host_mode(session) for session in sessions)


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


async def _prepare_container_session(
    loaded: LoadedTraceSession,
    *,
    task_output_dir: Path,
    container_executable: str,
    network_mode: str = "host",
    fixed_images_by_source: dict[str, str] | None = None,
) -> PreparedTraceSession:
    """Prepare a Docker/Podman container and start a persistent replay agent."""
    from trace_collect.openclaw_tools import ContainerAgent

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
    container_id: str | None = None
    agent: Any | None = None
    cleanup_fixed_image = True
    try:
        phase = recorder.start_phase("ensure_fixed_image")
        try:
            fixed_name = (
                fixed_images_by_source or {}
            ).get(normalized)
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
            container_id = await asyncio.to_thread(
                start_task_container,
                fixed_name,
                executable=container_executable,
                run_as_host_user=False,
                mount_host_home=False,
                container_home="/root",
                extra_args=[
                    "--label",
                    "agent-sched-bench.component=simulate-replay",
                    "--label",
                    f"agent-sched-bench.run_instance_id={loaded.agent_id}",
                    "--label",
                    f"agent-sched-bench.source_agent_id={loaded.source_agent_id}",
                    "--label",
                    f"agent-sched-bench.manifest_index={loaded.manifest_index}",
                    "--label",
                    f"agent-sched-bench.output_dir={task_output_dir}",
                ],
                network_mode=network_mode,
            )
            recorder.container_id = container_id
            recorder.finish_phase(
                phase,
                extra={
                    "container_id": container_id,
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

        agent = ContainerAgent(container_id, container_executable)
        phase = recorder.start_phase("container_agent_start")
        try:
            await agent.start()
            recorder.finish_phase(phase)
        except (Exception, asyncio.CancelledError) as exc:
            recorder.finish_phase(phase, status="failed", error=exc)
            raise

        recorder.write(status="success")
    except (Exception, asyncio.CancelledError) as exc:
        cleanup_errors: list[BaseException] = []
        try:
            recorder.write(status="failed", error=exc)
        except (Exception, asyncio.CancelledError):
            logger.exception("Failed to write container startup failure artifact for %s", loaded.agent_id)
        if agent is not None:
            try:
                await agent.stop()
            except (Exception, asyncio.CancelledError):
                logger.exception("Failed to stop container agent for %s", loaded.agent_id)
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
                logger.exception("Failed to stop startup container for %s", loaded.agent_id)
        if cleanup_fixed_image and recorder.fixed_image is not None and (
            container_id is None or container_stopped
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
    assert agent is not None

    container = PreparedContainer(
        container_id=container_id,
        container_executable=container_executable,
        docker_image=normalized,
        agent=agent,
        fixed_image=recorder.fixed_image,
        cleanup_fixed_image=cleanup_fixed_image,
    )
    return PreparedTraceSession(loaded=loaded, container=container)


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
) -> ReplayTaskStats:
    llm_call_count = sum(1 for action in loaded.actions if action.get("action_type") == "llm_call")
    tool_exec_count = sum(1 for action in loaded.actions if action.get("action_type") == "tool_exec")
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

    if container_stopped and ctr.fixed_image and ctr.cleanup_fixed_image:
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
            if container_executable is None:
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
            await _restore_source_runtime_artifacts(prepared)
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
    except (Exception, asyncio.CancelledError):
        if prepared is not None:
            await _finalize_prepared_session(prepared)
        raise


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
) -> tuple[list[PreparedTraceSession], list[ReplayTaskStats]]:
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
                logger.info(
                    "Worker %d replaying %s (%d queued)",
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
                )
                stats = await _replay_cloud_model_session(
                    prepared,
                    trace_logger=trace_logger,
                    replay_speed=replay_speed,
                    llm_timing=llm_timing,
                    command_timeout_s=command_timeout_s,
                    warmup_skip_iterations=warmup_skip_iterations,
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
) -> list[ReplayTaskStats]:
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
                )
                for loaded in loaded_sessions
            ),
            return_exceptions=True,
        )
        prep_errors: list[BaseException] = []
        for result in prep_results:
            if isinstance(result, BaseException):
                prep_errors.append(result)
            else:
                prepared_sessions.append(result)
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
            },
        )
        replay_zero_monotonic = await _wait_for_global_replay_start(
            replay_start_barrier,
            replay_start_event,
            replay_start_wall_time,
            coordinator=worker_index == 0,
        )
        replay_started = True
        task_stats = await _run_prepared_cloud_model_sessions(
            prepared_sessions,
            trace_logger=trace_logger,
            replay_zero_monotonic=replay_zero_monotonic,
            replay_speed=replay_speed,
            llm_timing=llm_timing,
            command_timeout_s=command_timeout_s,
            warmup_skip_iterations=warmup_skip_iterations,
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
) -> tuple[list[WorkerReplayResult], list[ReplayTaskStats]]:
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
    return replay_results, task_stats


async def _replay_cloud_model_session(
    prepared_session: PreparedTraceSession,
    *,
    trace_logger: TraceLogger,
    replay_zero_monotonic: float | None = None,
    replay_speed: float,
    llm_timing: LLMTimingConfig,
    command_timeout_s: float,
    warmup_skip_iterations: int,
) -> ReplayTaskStats:
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
    failed_actions = 0
    fatal_replay_errors = 0
    forced_sync_actions = 0
    source_failed_actions = 0
    replay_failed_actions = 0
    matched_failed_actions = 0
    previous_source_end: float | None = None
    sleep_drifts: list[SleepDrift] = []

    if replay_zero_monotonic is not None:
        start_drift = await _sleep_until_monotonic(replay_zero_monotonic)
        if start_drift is not None:
            sleep_drifts.append(start_drift)

    for action in loaded.actions:
        action_id = str(action.get("action_id", ""))
        action_type = str(action.get("action_type", ""))
        iteration = int(action.get("iteration", 0))
        data = action.get("data", {})
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
                if action_sleep is not None:
                    sleep_drifts.append(action_sleep)
                record_ts_end = time.time()
                record = _make_trace_action(
                    loaded=loaded,
                    action_type="llm_call",
                    action_id=action_id or f"llm_{iteration}",
                    iteration=iteration,
                    ts_start=record_ts_start,
                    ts_end=record_ts_end,
                    data={
                        "messages_in": data.get("messages_in"),
                        "raw_response": data.get("raw_response", {}),
                        "prompt_tokens": data.get("prompt_tokens", 0),
                        "completion_tokens": data.get("completion_tokens", 0),
                        "llm_latency_ms": (record_ts_end - record_ts_start) * 1000,
                        "simulate_source": str(loaded.source_trace),
                        "source_llm_latency_ms": data.get("llm_latency_ms"),
                        "replay_mode": "cloud_model",
                        "replay_speed": replay_speed,
                        **llm_timing_fields,
                        "sim_metrics": {
                            "warmup": iteration < warmup_skip_iterations,
                            **_sleep_drift_metrics(
                                source_gap=source_gap_sleep,
                                action_sleep=action_sleep,
                            ),
                        },
                    },
                )
                trace_logger.log_trace_action(loaded.agent_id, record)
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
            elif tool_name.startswith("mcp_"):
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
                mapped_tool_args, original_artifact_path, mapped_artifact_path, mapped_exists = (
                    _remap_runtime_artifact_tool_args(
                        tool_name=tool_name,
                        tool_args_json=tool_args,
                        runtime_root_map=prepared_session.runtime_artifact_root_map,
                    )
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
                        sleep_drifts.append(action_sleep)
                    tool_result = _artifact_unavailable_result(original_artifact_path)
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
                    if exec_resource_timeline is None:
                        if mapped_artifact_path is not None:
                            (
                                tool_result,
                                duration_ms,
                                tool_success,
                                tool_exec_metadata,
                            ) = _unpack_exec_tool_result(
                                await _exec_tool(
                                    ctr.agent,
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
                                    ctr.agent,
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
                                ctr.agent,
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
                                ctr.agent,
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
            if not tool_success:
                replay_failed_actions += 1
            mismatch_reason = _tool_mismatch_reason(
                source_success=source_success,
                tool_success=tool_success,
                replay_source=replay_source,
                source_tool_result=source_tool_result,
                replay_tool_result=tool_result,
                tool_name=tool_name,
                tool_args_json=tool_args,
            )
            replay_outcome_match = mismatch_reason is None
            if (not tool_success) and replay_outcome_match:
                matched_failed_actions += 1
            record_ts_end = time.time()
            forced_sync_fields: dict[str, Any] = {}
            if mismatch_reason is not None and ctr is not None:
                checkpoint_spec = _checkpoint_after_spec(
                    action_data=data,
                    source_trace=loaded.source_trace,
                )
                if checkpoint_spec is None:
                    forced_sync_fields = {
                        "forced_sync_attempted": False,
                        "forced_sync_success": False,
                        "forced_sync_resolved": False,
                        "forced_sync_reason": mismatch_reason,
                        "forced_sync_error": "checkpoint_after missing",
                    }
                else:
                    forced_sync_fields = {
                        "forced_sync_attempted": True,
                        "forced_sync_reason": mismatch_reason,
                        "forced_sync_overhead_excluded": True,
                    }
                    try:
                        restore_result = await asyncio.to_thread(
                            _restore_checkpoint_to_container,
                            checkpoint_spec=checkpoint_spec,
                            container=ctr,
                        )
                        forced_sync_fields.update(restore_result)
                        forced_sync_fields["forced_sync_resolved"] = (
                            restore_result.get("forced_sync_success") is True
                            and mismatch_reason != "source_artifact_unavailable"
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
                                "forced_sync_resolved": False,
                                "forced_sync_error": f"{type(exc).__name__}: {exc}",
                            }
                        )
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
            if mismatch_reason is not None:
                extra_tool_fields["mismatch_reason"] = mismatch_reason
            extra_tool_fields.update(tool_exec_metadata)
            extra_tool_fields.update(forced_sync_fields)
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
                    "sim_metrics": {
                        "warmup": iteration < warmup_skip_iterations,
                        "source": replay_source,
                        "sim_tool_format": replay_source
                        if replay_source
                        in {
                            "skipped_host_mode",
                            "message_noop",
                            "replayed_from_trace",
                            "source_artifact_unavailable",
                            "restored_runtime_artifact",
                        }
                        else "container_exec",
                        **_sleep_drift_metrics(
                            source_gap=source_gap_sleep,
                            action_sleep=action_sleep,
                        ),
                    },
                },
            )
            trace_logger.log_trace_action(loaded.agent_id, tool_record)
            forced_sync_success = forced_sync_fields.get("forced_sync_success") is True
            forced_sync_resolved = (
                forced_sync_success and mismatch_reason != "source_artifact_unavailable"
            )
            if forced_sync_success:
                forced_sync_actions += 1
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
            elif forced_sync_resolved:
                logger.warning(
                    "Replay mismatch forced-synced for %s action=%s tool=%s reason=%s",
                    loaded.agent_id,
                    action_id,
                    tool_name,
                    mismatch_reason,
                )
            else:
                logger.error(
                    "Replay tool outcome mismatch for %s action=%s tool=%s "
                    "source_success=%s replay_success=%s",
                    loaded.agent_id,
                    action_id,
                    tool_name,
                    source_success,
                    tool_success,
                )
                failed_actions += 1
                if replay_source == "source_artifact_unavailable":
                    fatal_replay_errors += 1
        except Exception as exc:
            logger.error(
                "Replay action failed for %s action=%s: %s",
                loaded.agent_id,
                action_id,
                exc,
            )
            failed_actions += 1

    wall_end = time.time()
    success = failed_actions == 0 and fatal_replay_errors == 0
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
                "matched_failed_actions": matched_failed_actions,
                "fatal_replay_errors": fatal_replay_errors,
                "forced_sync_actions": forced_sync_actions,
                "outcome_mismatches": failed_actions,
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
    llm_timing = LLMTimingConfig(
        mode=llm_timing_mode,
        ttft_ms=llm_ttft_ms,
        tpot_ms=llm_tpot_ms,
    )
    _validate_llm_timing_config(llm_timing)

    manifest_entries = _load_simulate_manifest(
        manifest,
        default_task_source=task_source.resolve(),
    )

    loaded_sessions = [
        _load_trace_session(
            entry.trace,
            entry.task_source,
            manifest_index=entry.index,
            docker_image_override=entry.docker_image,
            label=entry.label,
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
