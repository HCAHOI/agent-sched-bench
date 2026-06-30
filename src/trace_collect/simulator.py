from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import subprocess
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
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
from harness.gpu_resource_sampler import GpuResourceSampler
from harness.metrics_client import VLLMMetricsClient
from harness.scheduler_hooks import GpuBaseline
from harness.trace_logger import TraceLogger
from llm_call import create_async_openai_client
from trace_collect import attempt_layout
from trace_collect.attempt_pipeline import (
    configure_task_container_apt_mirror,
    next_attempt_number_in,
    sanitize_path_segment,
    start_task_container,
    stop_task_container,
)

logger = logging.getLogger(__name__)
GLOBAL_CONTAINER_RESOURCE_SAMPLE_INTERVAL_S = 1.0


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


def validate_gpu_tracking_args(args: Any) -> None:
    """Validate GPU tracking CLI args. Raises ValueError with a clear message on failure.

    Designed to be called from _run_simulate before any work begins,
    so failures are fast and explicit (CLAUDE.md no-silent-fallback rule).
    """
    if args.gpu_tracking != "on":
        return

    if args.mode == "cloud_model":
        raise ValueError("--gpu-tracking on is forbidden in cloud_model mode")

    if not args.metrics_url:
        raise ValueError("--gpu-tracking on requires --metrics-url")

    if args.vllm_pid is None:
        raise ValueError("--gpu-tracking on requires --vllm-pid")

    if args.vllm_startup_log is None:
        raise ValueError("--gpu-tracking on requires --vllm-startup-log")


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


async def _call_local_model_streaming(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    n_tokens: int,
) -> tuple[float, float, float]:
    """Send *messages* to the local model and force exactly *n_tokens* of output.

    Returns:
        (ttft_ms, tpot_ms, total_latency_ms)
    """
    t0 = time.monotonic()
    first_token_ts: float | None = None

    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=n_tokens,
        stream=True,
        temperature=0.0,
        extra_body={"min_tokens": n_tokens},
    )
    async for chunk in stream:
        if first_token_ts is None and chunk.choices and chunk.choices[0].delta.content:
            first_token_ts = time.monotonic()

    t_end = time.monotonic()
    total_ms = (t_end - t0) * 1000
    ttft_ms = (first_token_ts - t0) * 1000 if first_token_ts else total_ms
    gen_ms = total_ms - ttft_ms
    tpot_ms = gen_ms / max(1, n_tokens - 1) if n_tokens > 1 else 0.0
    return ttft_ms, tpot_ms, total_ms


async def _exec_tool(
    agent: Any,
    tool_name: str | None,
    tool_args_json: str,
    command_timeout_s: float,
    source_exec_timeout_s: float | None = None,
    allow_source_runtime_artifacts: bool = False,
) -> tuple[str, float, bool]:
    """Execute one source-trace tool call via the persistent container agent.

    Returns:
        (tool_result, tool_duration_ms, tool_success)
    """
    from trace_collect.openclaw_tools import execute_trace_tool

    t0 = time.monotonic()
    tool_result, tool_success, inner_duration_ms = await execute_trace_tool(
        agent=agent,
        tool_name=tool_name,
        tool_args_json=tool_args_json,
        command_timeout_s=command_timeout_s,
        source_exec_timeout_s=source_exec_timeout_s,
        allow_source_runtime_artifacts=allow_source_runtime_artifacts,
    )
    wall_duration_ms = (time.monotonic() - t0) * 1000
    # Prefer agent-side timing to exclude pipe transfer overhead
    duration_ms = inner_duration_ms if inner_duration_ms is not None else wall_duration_ms
    return tool_result, duration_ms, tool_success


def _group_actions_by_iteration(
    actions: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Group loaded trace actions into the local-model iteration shape."""

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


def _tool_uses_exec_semantics(tool_name: str | None, tool_args_json: Any) -> bool:
    if tool_name == "exec":
        return True
    if isinstance(tool_args_json, str):
        try:
            parsed = json.loads(tool_args_json or "{}")
        except json.JSONDecodeError:
            return False
    elif isinstance(tool_args_json, dict):
        parsed = tool_args_json
    else:
        return False
    if not isinstance(parsed, dict):
        return False
    payload = parsed.get("exec") if isinstance(parsed.get("exec"), dict) else parsed
    return isinstance(payload, dict) and (
        "command" in payload or "commands" in payload
    )


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
    for line in tool_result.splitlines():
        if not line.strip():
            continue
        return line.strip() == "[timeout]"
    return False


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
    return (
        Path(_sanitize_run_label(benchmark))
        / _sanitize_run_label(model)
        / _sanitize_run_label(scaffold)
        / "bounded_queue"
        / f"concurrency_{concurrency}"
    )


def _build_run_id(*, mode: str, model: str | None, concurrency: int) -> str:
    label = model if model else mode
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
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
) -> None:
    if replay_speed <= 0:
        raise ValueError("replay_speed must be > 0")
    if not sessions:
        raise SimulateError("No trace sessions were loaded")
    if mode == "local_model" and len(sessions) != 1:
        raise SimulateError("local_model mode supports exactly one source trace")
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
    manifest: Path,
    concurrency: int,
    scheduler_mode: str,
    api_base: str | None,
    model: str | None,
    network_mode: str = "host",
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
    if mode == "local_model":
        metadata["source_trace"] = str(sessions[0].source_trace)
        metadata["source_model"] = source_models[0]
        metadata["local_model"] = model
        metadata["local_api_base"] = api_base
        metadata["n_source_iterations"] = _iteration_count(sessions[0].actions)
    else:
        metadata["source_model"] = (
            source_models[0] if len(set(source_models)) == 1 else "multiple"
        )
        metadata["replay_target"] = "cloud_replay"
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
    trace_file: Path,
    wall_time_s: float,
    task_stats: list[ReplayTaskStats],
    container_resources: dict[str, Any] | None = None,
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
        "scheduler_mode": scheduler_mode,
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
    if container_resources is not None:
        payload["container_resources"] = {
            "jsonl_path": container_resources.get("jsonl_path"),
            "summary_path": container_resources.get("summary_path"),
            "sample_count": container_resources.get("sample_count", 0),
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
) -> None:
    if prepared.task_output_dir is None:
        return
    summary = summarize_samples(samples)
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
                _write_prepared_resources(prepared, resource_samples)
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
            _write_prepared_resources(prepared, resource_samples)
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


async def _run_local_model_simulation(
    prepared_session: PreparedTraceSession,
    *,
    trace_logger: TraceLogger,
    replay_speed: float,
    api_base: str,
    api_key: str,
    model: str,
    command_timeout_s: float,
    metrics_url: str | None,
    warmup_skip_iterations: int,
    gpu_baseline: GpuBaseline | None = None,
    vllm_pid: int | None = None,
    gpu_sample_hz: float = 10.0,
    gpu_output_path: Path | None = None,
) -> ReplayTaskStats:
    loaded = prepared_session.loaded
    iterations = loaded.iterations
    source_model = (loaded.summary or {}).get("model", "unknown")
    logger.info(
        "Simulating %s [scaffold=%s]: %d iterations from %s, local model=%s",
        loaded.agent_id,
        loaded.scaffold,
        len(iterations),
        source_model,
        model,
    )

    metrics_client = VLLMMetricsClient(
        metrics_url=metrics_url,
        gpu_baseline=gpu_baseline,
        vllm_pid=vllm_pid,
    )
    logger.info(
        "vLLM metrics client: %s",
        f"enabled (url={metrics_url})" if metrics_client.is_enabled else "disabled",
    )

    gpu_sampler: GpuResourceSampler | None = None
    if gpu_baseline is not None and vllm_pid is not None and metrics_url and gpu_output_path:
        gpu_sampler = GpuResourceSampler(
            metrics_url=metrics_url,
            gpu_baseline=gpu_baseline,
            vllm_pid=vllm_pid,
            output_path=gpu_output_path,
            sample_hz=gpu_sample_hz,
        )
        await gpu_sampler.start()
        logger.info("GPU resource sampler started (%.1f Hz) → %s", gpu_sample_hz, gpu_output_path)

    client = None

    wall_start = time.time()
    total_iters = len(iterations)
    succeeded_iters = 0
    failed_iters = 0
    fatal_replay_errors = 0
    sorted_iters = sorted(iterations.keys())

    try:
        for i, it_num in enumerate(sorted_iters):
            it_group = iterations[it_num]
            llm_actions = it_group.get("llms", [])
            tool_actions = it_group.get("tools", [])

            if not llm_actions:
                logger.warning("Iteration %d: no LLM actions, skipping", it_num)
                continue

            iter_failed = False
            for llm_idx, llm_action in enumerate(llm_actions):
                llm_data = llm_action.get("data", {})
                messages_in = llm_data.get("messages_in")
                n_tokens = llm_data.get("completion_tokens", 1) or 1

                if llm_data.get("transport_retry_terminal"):
                    ts_now = time.time()
                    llm_record = _make_trace_action(
                        loaded=loaded,
                        action_type="llm_call",
                        action_id=f"llm_{it_num}_{llm_idx}",
                        iteration=it_num,
                        ts_start=ts_now,
                        ts_end=ts_now,
                        data={
                            "messages_in": messages_in,
                            "raw_response": llm_data.get("raw_response", {}),
                            "prompt_tokens": llm_data.get("prompt_tokens", 0),
                            "completion_tokens": 0,
                            "llm_latency_ms": 0.0,
                            "simulate_source": str(loaded.source_trace),
                            "source_llm_latency_ms": llm_data.get("llm_latency_ms"),
                            "transport_retry": True,
                            "transport_retry_terminal": True,
                            "error": llm_data.get("error"),
                            "sim_metrics": {
                                "warmup": i < warmup_skip_iterations,
                                "failed": True,
                            },
                        },
                    )
                    trace_logger.log_trace_action(loaded.agent_id, llm_record)
                    iter_failed = True
                    break

                if not messages_in:
                    logger.warning("Iteration %d llm %d: no messages_in, skipping", it_num, llm_idx)
                    continue

                ts_start = time.time()

                try:
                    if client is None:
                        client = create_async_openai_client(
                            api_base=api_base,
                            api_key=api_key,
                            timeout=180.0,
                        )
                    ttft_ms, tpot_ms, llm_latency_ms = await _call_local_model_streaming(
                        client, model, messages_in, n_tokens
                    )
                except Exception as exc:
                    logger.error("Iteration %d llm %d: LLM call failed: %s", it_num, llm_idx, exc)
                    iter_failed = True
                    break

                ts_after_llm = time.time()

                scheduler_snapshot = metrics_client.get_snapshot()

                llm_record = _make_trace_action(
                    loaded=loaded,
                    action_type="llm_call",
                    action_id=f"llm_{it_num}_{llm_idx}",
                    iteration=it_num,
                    ts_start=ts_start,
                    ts_end=ts_after_llm,
                    data={
                        "messages_in": messages_in,
                        "raw_response": llm_data.get("raw_response", {}),
                        "prompt_tokens": llm_data.get("prompt_tokens", 0),
                        "completion_tokens": llm_data.get("completion_tokens", 0),
                        "llm_latency_ms": llm_latency_ms,
                        "ttft_ms": ttft_ms,
                        "tpot_ms": tpot_ms,
                        "simulate_source": str(loaded.source_trace),
                        "source_llm_latency_ms": llm_data.get("llm_latency_ms"),
                        "sim_metrics": {
                            "timing": {
                                "ttft_ms": ttft_ms,
                                "tpot_ms": tpot_ms,
                                "total_ms": llm_latency_ms,
                            },
                            "vllm_scheduler_snapshot": dataclasses.asdict(
                                scheduler_snapshot
                            ),
                            "warmup": i < warmup_skip_iterations,
                        },
                    },
                )
                trace_logger.log_trace_action(loaded.agent_id, llm_record)

            if iter_failed:
                failed_iters += 1
                continue

            ctr = prepared_session.container
            total_tool_ms = 0.0
            failed_tool_actions = 0
            for tool_act in tool_actions:
                td = tool_act.get("data", {})
                tool_name = td.get("tool_name")
                tool_args = td.get("tool_args", "{}")
                if not tool_name:
                    continue

                tool_ts_start = time.time()
                source_duration_ms = float(td.get("duration_ms") or 0.0)
                source_success = _source_tool_success(td)
                source_tool_result = td.get("tool_result", td.get("result", ""))
                source_exec_timeout = _source_exec_timeout_s(
                    tool_name=tool_name,
                    tool_args_json=tool_args,
                    source_duration_ms=source_duration_ms,
                    source_success=source_success,
                    source_tool_result=source_tool_result,
                )
                original_artifact_path: str | None = None
                mapped_artifact_path: str | None = None
                if ctr is None:
                    tool_result = td.get("tool_result", td.get("result", ""))
                    tool_success = source_success
                    await asyncio.sleep(max(0.0, source_duration_ms / 1000.0 / replay_speed))
                    tool_ts_end = time.time()
                    tool_duration_ms = source_duration_ms
                    sim_provenance = "replayed_from_trace"
                elif tool_name == "message":
                    tool_result = td.get("tool_result", td.get("result", ""))
                    if not tool_result:
                        tool_result = "Message replayed as no-op"
                    tool_success = source_success
                    await asyncio.sleep(max(0.0, source_duration_ms / 1000.0 / replay_speed))
                    tool_ts_end = time.time()
                    tool_duration_ms = source_duration_ms
                    sim_provenance = "message_noop"
                elif tool_name.startswith("mcp_"):
                    tool_result = td.get("tool_result", "")
                    tool_success = source_success
                    await asyncio.sleep(max(0.0, source_duration_ms / 1000.0 / replay_speed))
                    tool_ts_end = time.time()
                    tool_duration_ms = source_duration_ms
                    sim_provenance = "replayed_from_trace"
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
                        await asyncio.sleep(max(0.0, source_duration_ms / 1000.0 / replay_speed))
                        tool_result = _artifact_unavailable_result(original_artifact_path)
                        tool_success = False
                        tool_ts_end = time.time()
                        tool_duration_ms = (tool_ts_end - tool_ts_start) * 1000
                        sim_provenance = "source_artifact_unavailable"
                        fatal_replay_errors += 1
                    elif mapped_artifact_path is not None:
                        tool_result, tool_duration_ms, tool_success = await _exec_tool(
                            ctr.agent,
                            tool_name,
                            mapped_tool_args,
                            command_timeout_s,
                            source_exec_timeout,
                            True,
                        )
                        tool_ts_end = time.time()
                        sim_provenance = "restored_runtime_artifact"
                    else:
                        tool_result, tool_duration_ms, tool_success = await _exec_tool(
                            ctr.agent,
                            tool_name,
                            mapped_tool_args,
                            command_timeout_s,
                            source_exec_timeout,
                        )
                        tool_ts_end = time.time()
                        sim_provenance = "executed_in_container"
                total_tool_ms += tool_duration_ms
                replay_outcome_match = (
                    tool_success == source_success
                    and sim_provenance != "source_artifact_unavailable"
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
                if source_exec_timeout is not None:
                    extra_tool_fields["source_exec_timeout_s"] = source_exec_timeout

                tool_record = _make_trace_action(
                    loaded=loaded,
                    action_type="tool_exec",
                    action_id=f"tool_{it_num}_{tool_name}",
                    iteration=it_num,
                    ts_start=tool_ts_start,
                    ts_end=tool_ts_end,
                    data={
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "tool_result": tool_result,
                        "duration_ms": tool_duration_ms,
                        "success": tool_success,
                        "source_success": source_success,
                        "replay_outcome_match": replay_outcome_match,
                        **extra_tool_fields,
                        "replay_source": sim_provenance,
                        "sim_metrics": {
                            "source": sim_provenance,
                            "sim_tool_format": (
                                sim_provenance
                                if sim_provenance
                                in {
                                    "replayed_from_trace",
                                    "message_noop",
                                    "source_artifact_unavailable",
                                    "restored_runtime_artifact",
                                }
                                else "container_exec"
                            ),
                            "warmup": i < warmup_skip_iterations,
                        },
                    },
                )
                trace_logger.log_trace_action(loaded.agent_id, tool_record)
                if not replay_outcome_match:
                    failed_tool_actions += 1

            if failed_tool_actions:
                logger.error(
                    "Local-model replay failed %d tool action(s) for %s iteration=%s",
                    failed_tool_actions,
                    loaded.agent_id,
                    it_num,
                )
                failed_iters += 1
                continue

            succeeded_iters += 1

            logger.info(
                "[%d/%d] iter %d: %d llm calls, tool=%.0fms",
                i + 1,
                total_iters,
                it_num,
                len(llm_actions),
                total_tool_ms,
            )
    finally:
        if gpu_sampler is not None:
            await gpu_sampler.stop()
            logger.info("GPU resource sampler stopped → %s", gpu_output_path)

        wall_end = time.time()

        success = failed_iters == 0 and succeeded_iters == total_iters
        elapsed_s = wall_end - wall_start
        simulate_summary = _make_trace_summary(
            loaded=loaded,
            success=success,
            elapsed_s=elapsed_s,
            source_model=source_model,
            extra={
                "local_model": model,
                "local_api_base": api_base,
                "succeeded_iterations": succeeded_iters,
                "failed_iterations": failed_iters,
                "fatal_replay_errors": fatal_replay_errors,
            },
        )
        trace_logger.log_summary(loaded.agent_id, simulate_summary)
    return _make_task_stats(
        loaded=loaded,
        success=success,
        elapsed_s=elapsed_s,
        failed_action_count=failed_iters,
    )


async def _sleep_until_offset(
    *,
    replay_zero_monotonic: float,
    target_offset_s: float,
) -> None:
    delay_s = target_offset_s - (time.monotonic() - replay_zero_monotonic)
    if delay_s > 0:
        await asyncio.sleep(delay_s)


async def _prepare_replay_session(
    loaded: LoadedTraceSession,
    *,
    output_path: Path,
    container_executable: str | None,
    network_mode: str,
    container_resource_recorder: ContainerResourceRecorder | None = None,
    fixed_images_by_source: dict[str, str] | None = None,
) -> PreparedTraceSession:
    prepared: PreparedTraceSession | None = None
    try:
        prepared = PreparedTraceSession(loaded=loaded)
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
            prepared.container_resource_recorder = container_resource_recorder
            if container_resource_recorder is not None:
                container_resource_recorder.register_container(
                    prepared.container.container_id
                )
            sampler = ContainerStatsSampler(
                container_id=prepared.container.container_id,
                interval_s=1.0,
                executable=prepared.container.container_executable,
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
    command_timeout_s: float,
    warmup_skip_iterations: int,
    fixed_images_by_source: dict[str, str] | None = None,
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
                )
                stats = await _replay_cloud_model_session(
                    prepared,
                    trace_logger=trace_logger,
                    replay_zero_monotonic=time.monotonic(),
                    replay_speed=replay_speed,
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


async def _replay_cloud_model_session(
    prepared_session: PreparedTraceSession,
    *,
    trace_logger: TraceLogger,
    replay_zero_monotonic: float,
    replay_speed: float,
    command_timeout_s: float,
    warmup_skip_iterations: int,
) -> ReplayTaskStats:
    loaded = prepared_session.loaded
    ctr = prepared_session.container
    source_model = (loaded.summary or {}).get("model", "unknown")
    source_zero = _coerce_timestamp(
        loaded.actions[0].get("ts_start"),
        field="ts_start",
        source_trace=loaded.source_trace,
        action_id=str(loaded.actions[0].get("action_id", "")),
    )

    logger.info(
        "Replaying %s [scaffold=%s]: %d actions from %s at %.2fx",
        loaded.agent_id,
        loaded.scaffold,
        len(loaded.actions),
        source_model,
        replay_speed,
    )

    wall_start = time.time()
    succeeded_actions = 0
    failed_actions = 0
    fatal_replay_errors = 0
    source_failed_actions = 0
    replay_failed_actions = 0
    matched_failed_actions = 0

    for action in loaded.actions:
        action_id = str(action.get("action_id", ""))
        action_type = str(action.get("action_type", ""))
        iteration = int(action.get("iteration", 0))
        data = action.get("data", {})
        action_ts_start, action_ts_end = _coerce_action_bounds(action, source_trace=loaded.source_trace)
        source_duration_s = max(0.0, action_ts_end - action_ts_start)

        await _sleep_until_offset(
            replay_zero_monotonic=replay_zero_monotonic,
            target_offset_s=(action_ts_start - source_zero) / replay_speed,
        )

        try:
            if action_type == "llm_call":
                record_ts_start = time.time()
                if source_duration_s > 0:
                    await asyncio.sleep(source_duration_s / replay_speed)
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
                        "sim_metrics": {
                            "warmup": iteration < warmup_skip_iterations,
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
            source_success = _source_tool_success(data)
            source_tool_result = data.get("tool_result", data.get("result", ""))
            source_exec_timeout = _source_exec_timeout_s(
                tool_name=tool_name,
                tool_args_json=tool_args,
                source_duration_ms=source_duration_ms,
                source_success=source_success,
                source_tool_result=source_tool_result,
            )
            original_artifact_path: str | None = None
            mapped_artifact_path: str | None = None
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
                if source_duration_ms > 0:
                    await asyncio.sleep(source_duration_ms / 1000 / replay_speed)
                duration_ms = (time.time() - record_ts_start) * 1000
            elif tool_name == "message":
                if source_duration_ms > 0:
                    await asyncio.sleep(source_duration_ms / 1000 / replay_speed)
                tool_result = data.get("tool_result", data.get("result", ""))
                if not tool_result:
                    tool_result = "Message replayed as no-op"
                tool_success = source_success
                duration_ms = (time.time() - record_ts_start) * 1000
                replay_source = "message_noop"
            elif tool_name.startswith("mcp_"):
                if source_duration_ms > 0:
                    await asyncio.sleep(source_duration_ms / 1000 / replay_speed)
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
                    if source_duration_ms > 0:
                        await asyncio.sleep(source_duration_ms / 1000 / replay_speed)
                    tool_result = _artifact_unavailable_result(original_artifact_path)
                    tool_success = False
                    duration_ms = (time.time() - record_ts_start) * 1000
                    replay_source = "source_artifact_unavailable"
                    fatal_replay_errors += 1
                else:
                    if mapped_artifact_path is not None:
                        tool_result, duration_ms, tool_success = await _exec_tool(
                            ctr.agent,
                            tool_name,
                            mapped_tool_args,
                            command_timeout_s,
                            source_exec_timeout,
                            True,
                        )
                    else:
                        tool_result, duration_ms, tool_success = await _exec_tool(
                            ctr.agent,
                            tool_name,
                            mapped_tool_args,
                            command_timeout_s,
                            source_exec_timeout,
                        )
                    replay_source = (
                        "restored_runtime_artifact"
                        if mapped_artifact_path is not None
                        else "executed_in_container"
                    )
            if not tool_success:
                replay_failed_actions += 1
            replay_outcome_match = (
                tool_success == source_success
                and replay_source != "source_artifact_unavailable"
            )
            if (not tool_success) and replay_outcome_match:
                matched_failed_actions += 1
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
            if source_exec_timeout is not None:
                extra_tool_fields["source_exec_timeout_s"] = source_exec_timeout
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
                    },
                },
            )
            trace_logger.log_trace_action(loaded.agent_id, tool_record)
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
                "succeeded_actions": succeeded_actions,
                "failed_actions": failed_actions,
                "source_failed_actions": source_failed_actions,
                "replay_failed_actions": replay_failed_actions,
                "matched_failed_actions": matched_failed_actions,
                "fatal_replay_errors": fatal_replay_errors,
                "outcome_mismatches": failed_actions,
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


async def simulate(
    *,
    manifest: Path,
    task_source: Path,
    output_dir: Path,
    mode: str = "local_model",
    concurrency: int = 1,
    container_executable: str | None = None,
    network_mode: str = "host",
    api_base: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    command_timeout_s: float = 120.0,
    metrics_url: str | None = None,
    warmup_skip_iterations: int = 0,
    replay_speed: float = 1.0,
    structured_output: bool = False,
    gpu_baseline: GpuBaseline | None = None,
    vllm_pid: int | None = None,
    gpu_sample_hz: float = 10.0,
) -> Path:
    if mode not in {"local_model", "cloud_model"}:
        raise ValueError(f"Unsupported simulate mode: {mode}")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if mode == "local_model" and concurrency != 1:
        raise ValueError("local_model simulate requires concurrency=1")

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
    )
    _validate_container_runtime(
        loaded_sessions,
        container_executable=container_executable,
    )
    await _prefetch_container_images(
        loaded_sessions,
        container_executable=container_executable,
    )

    output_path = Path(output_dir)
    if structured_output:
        output_path = output_path / _structured_output_subdir(
            loaded_sessions,
            concurrency=concurrency,
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
    scheduler_mode = "bounded_queue"

    try:
        sweep_fixed_images = await _prebuild_sweep_fixed_images(
            loaded_sessions,
            output_path=output_path,
            container_executable=container_executable,
        )
        run_wall_start = time.monotonic()
        run_id = _build_run_id(mode=mode, model=model, concurrency=concurrency)
        trace_logger = TraceLogger(output_path, run_id)
        _log_trace_metadata(
            trace_logger=trace_logger,
            mode=mode,
            sessions=loaded_sessions,
            replay_speed=replay_speed,
            manifest=manifest,
            concurrency=concurrency,
            scheduler_mode=scheduler_mode,
            api_base=api_base,
            model=model,
            network_mode=network_mode,
        )
        if container_executable is not None and _has_container_mode_sessions(
            loaded_sessions
        ):
            container_resource_recorder = ContainerResourceRecorder(
                output_dir=output_path,
                run_id=run_id,
                interval_s=GLOBAL_CONTAINER_RESOURCE_SAMPLE_INTERVAL_S,
                executable=container_executable,
                sample_all_containers=False,
            )
            container_resource_recorder.start()

        if mode == "local_model":
            prepared = await _prepare_replay_session(
                loaded_sessions[0],
                output_path=output_path,
                container_executable=container_executable,
                network_mode=network_mode,
                container_resource_recorder=container_resource_recorder,
                fixed_images_by_source=sweep_fixed_images,
            )
            prepared_sessions.append(prepared)
            assert trace_logger is not None
            assert api_base is not None
            assert api_key is not None
            assert model is not None
            # Compute gpu_output_path from the single session's attempt dir
            gpu_output_path: Path | None = None
            if gpu_baseline is not None and vllm_pid is not None and metrics_url:
                task_dir = prepared_sessions[0].task_output_dir
                if task_dir is not None:
                    gpu_output_path = task_dir / "gpu_resources.json"
            local_stats = await _run_local_model_simulation(
                prepared_sessions[0],
                trace_logger=trace_logger,
                replay_speed=replay_speed,
                api_base=api_base,
                api_key=api_key,
                model=model,
                command_timeout_s=command_timeout_s,
                metrics_url=metrics_url,
                warmup_skip_iterations=warmup_skip_iterations,
                gpu_baseline=gpu_baseline,
                vllm_pid=vllm_pid,
                gpu_sample_hz=gpu_sample_hz,
                gpu_output_path=gpu_output_path,
            )
            task_stats = [local_stats]
        else:
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
                command_timeout_s=command_timeout_s,
                warmup_skip_iterations=warmup_skip_iterations,
                fixed_images_by_source=sweep_fixed_images,
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
        trace_file=trace_file,
        wall_time_s=run_wall_end - run_wall_start,
        task_stats=task_stats,
        container_resources=container_resource_summary,
    )
    logger.info("Simulate complete [%s] -> %s", mode, trace_file)
    return trace_file
