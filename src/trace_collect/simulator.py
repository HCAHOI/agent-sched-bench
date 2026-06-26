from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agents.base import TraceAction
from harness.container_image_prep import (
    ensure_fixed_image,
    ensure_source_image,
    normalize_image_reference,
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

    trace: Path
    task_source: Path
    docker_image: str | None = None
    label: str | None = None


@dataclass(frozen=True, slots=True)
class ReplayTaskStats:
    """Per-trace throughput accounting for a simulate run."""

    agent_id: str
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
    agent_id: str
    scaffold: str
    metadata: dict[str, Any] | None
    summary: dict[str, Any] | None
    task: dict[str, Any]
    actions: list[dict[str, Any]]
    iterations: dict[int, dict[str, Any]]
    docker_image_override: str | None = None
    label: str | None = None


@dataclass(slots=True)
class PreparedContainer:
    """Container prepared for trace replay."""

    container_id: str
    container_executable: str
    docker_image: str
    agent: Any  # ContainerAgent


@dataclass(slots=True)
class PreparedTraceSession:
    """Container plus the loaded source-trace context."""

    loaded: LoadedTraceSession
    container: PreparedContainer | None = None
    container_resource_recorder: ContainerResourceRecorder | None = None
    sampler: ContainerStatsSampler | None = None
    task_output_dir: Path | None = None
    resources_written: bool = False


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
    docker_image_override: str | None = None,
    label: str | None = None,
) -> LoadedTraceSession:
    agent_id, metadata, actions, summary = _parse_trace_session_file(source_trace)
    scaffold = metadata.get("scaffold", "unknown") if metadata else "unknown"
    task = _find_task(task_source, agent_id)
    return LoadedTraceSession(
        source_trace=source_trace,
        task_source=task_source,
        agent_id=agent_id,
        scaffold=scaffold,
        metadata=metadata,
        summary=summary,
        task=task,
        actions=actions,
        iterations=_group_actions_by_iteration(actions),
        docker_image_override=docker_image_override,
        label=label,
    )


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
                f"Task {session.agent_id!r} has no resolvable docker_image "
                "(set docker_image in manifest or ensure task has image_name)"
            )

    seen_agent_ids: set[str] = set()
    for session in sessions:
        if session.agent_id in seen_agent_ids:
            raise SimulateError(
                f"Duplicate agent_id across replay sessions: {session.agent_id!r}"
            )
        seen_agent_ids.add(session.agent_id)

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


async def _prepare_container_session(
    loaded: LoadedTraceSession,
    *,
    task_output_dir: Path,
    container_executable: str,
    network_mode: str = "host",
) -> PreparedTraceSession:
    """Prepare a Docker/Podman container and start a persistent replay agent."""
    from trace_collect.openclaw_tools import ContainerAgent

    docker_image = _resolve_docker_image(loaded)
    if not docker_image:
        raise SimulateError(
            f"Task {loaded.agent_id!r} has no resolvable docker_image"
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
    try:
        phase = recorder.start_phase("ensure_fixed_image")
        try:
            fixed_name, fixed_elapsed_s = await asyncio.to_thread(
                ensure_fixed_image,
                normalized,
                container_executable=container_executable,
            )
            recorder.fixed_image = fixed_name
            recorder.finish_phase(
                phase,
                extra={
                    "fixed_image": fixed_name,
                    "reported_elapsed_s": fixed_elapsed_s,
                },
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
        try:
            recorder.write(status="failed", error=exc)
        except (Exception, asyncio.CancelledError):
            logger.exception("Failed to write container startup failure artifact for %s", loaded.agent_id)
        if agent is not None:
            try:
                await agent.stop()
            except (Exception, asyncio.CancelledError):
                logger.exception("Failed to stop container agent for %s", loaded.agent_id)
        if container_id is not None:
            try:
                await asyncio.to_thread(
                    stop_task_container,
                    container_id,
                    executable=container_executable,
                )
            except (Exception, asyncio.CancelledError):
                logger.exception("Failed to stop startup container for %s", loaded.agent_id)
        raise

    assert container_id is not None
    assert agent is not None

    container = PreparedContainer(
        container_id=container_id,
        container_executable=container_executable,
        docker_image=normalized,
        agent=agent,
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
        "source_models": source_models,
        "manifest": str(manifest),
        "concurrency": concurrency,
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
    return TraceAction(
        action_type=action_type,
        action_id=action_id,
        agent_id=loaded.agent_id,
        program_id=loaded.agent_id,
        iteration=iteration,
        ts_start=ts_start,
        ts_end=ts_end,
        data=data,
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
        "agent_id": loaded.agent_id,
        "task_id": loaded.agent_id,
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
        agent_id=loaded.agent_id,
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
    safe_wall_time_s = max(wall_time_s, 1e-9)
    payload = {
        "run_id": run_id,
        "mode": mode,
        "manifest": str(manifest),
        "trace_file": str(trace_file),
        "concurrency": concurrency,
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


async def _finalize_prepared_session(prepared: PreparedTraceSession) -> None:
    if prepared.sampler is not None:
        samples = prepared.sampler.stop()
        prepared.sampler = None
        if prepared.task_output_dir is not None:
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
    elif prepared.task_output_dir is not None and not prepared.resources_written:
        attempt_layout.write_resources_json(
            prepared.task_output_dir,
            samples=[],
            summary=summarize_samples([]),
        )
        prepared.resources_written = True

    ctr = prepared.container
    if ctr is None:
        return
    prepared.container = None
    agent_stop_error: BaseException | None = None
    container_stop_error: BaseException | None = None
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

    if container_stopped:
        if prepared.container_resource_recorder is not None:
            prepared.container_resource_recorder.unregister_container(ctr.container_id)
            prepared.container_resource_recorder = None

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
    if agent_stop_error is not None:
        raise agent_stop_error
    if container_stop_error is not None:
        raise container_stop_error


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
            for tool_act in tool_actions:
                td = tool_act.get("data", {})
                tool_name = td.get("tool_name")
                tool_args = td.get("tool_args", "{}")
                if not tool_name:
                    continue

                tool_ts_start = time.time()
                if ctr is None:
                    # Host-mode: replay source-trace timing verbatim.
                    tool_result = td.get("tool_result", td.get("result", ""))
                    tool_duration_ms = float(td.get("duration_ms") or 0.0)
                    tool_success = bool(td.get("success", not td.get("error")))
                    await asyncio.sleep(max(0.0, tool_duration_ms / 1000.0 / replay_speed))
                    tool_ts_end = time.time()
                    sim_provenance = "replayed_from_trace"
                elif tool_name.startswith("mcp_"):
                    tool_result = td.get("tool_result", "")
                    tool_duration_ms = float(td.get("duration_ms") or 0.0)
                    tool_success = bool(td.get("success", True))
                    await asyncio.sleep(max(0.0, tool_duration_ms / 1000.0 / replay_speed))
                    tool_ts_end = time.time()
                    sim_provenance = "replayed_from_trace"
                else:
                    tool_result, tool_duration_ms, tool_success = await _exec_tool(
                        ctr.agent,
                        tool_name,
                        tool_args,
                        command_timeout_s,
                    )
                    tool_ts_end = time.time()
                    sim_provenance = "executed_in_container"
                total_tool_ms += tool_duration_ms

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
                        "sim_metrics": {
                            "source": sim_provenance,
                            "sim_tool_format": (
                                "replayed_from_trace"
                                if sim_provenance == "replayed_from_trace"
                                else "container_exec"
                            ),
                            "warmup": i < warmup_skip_iterations,
                        },
                    },
                )
                trace_logger.log_trace_action(loaded.agent_id, tool_record)

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
            prepared = await _prepare_container_session(
                loaded,
                task_output_dir=task_output_dir,
                container_executable=container_executable,
                network_mode=network_mode,
            )
            prepared.task_output_dir = task_output_dir
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
            if ctr is None:
                logger.info(
                    "Skipping host-mode tool action for %s action=%s tool=%s",
                    loaded.agent_id,
                    action_id,
                    tool_name,
                )
                replay_source = "skipped_host_mode"
                tool_result = data.get("tool_result", data.get("result", ""))
                # Research-agent tools may omit "success"; infer from absence-of-error.
                tool_success = bool(data.get("success", not data.get("error")))
                if source_duration_ms > 0:
                    await asyncio.sleep(source_duration_ms / 1000 / replay_speed)
                duration_ms = (time.time() - record_ts_start) * 1000
            elif tool_name.startswith("mcp_"):
                if source_duration_ms > 0:
                    await asyncio.sleep(source_duration_ms / 1000 / replay_speed)
                tool_result = data.get("tool_result", "")
                tool_success = bool(data.get("success", True))
                duration_ms = (time.time() - record_ts_start) * 1000
                replay_source = "replayed_from_trace"
            else:
                tool_result, duration_ms, tool_success = await _exec_tool(
                    ctr.agent,
                    tool_name,
                    tool_args,
                    command_timeout_s,
                )
                replay_source = "executed_in_container"
            record_ts_end = time.time()
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
                    "simulate_source": str(loaded.source_trace),
                    "source_duration_ms": source_duration_ms,
                    "replay_mode": "cloud_model",
                    "replay_speed": replay_speed,
                    "replay_source": replay_source,
                    "sim_metrics": {
                        "warmup": iteration < warmup_skip_iterations,
                        "source": replay_source,
                        "sim_tool_format": replay_source
                        if replay_source == "skipped_host_mode"
                        else "container_exec",
                    },
                },
            )
            trace_logger.log_trace_action(loaded.agent_id, tool_record)
            succeeded_actions += 1
        except Exception as exc:
            logger.error(
                "Replay action failed for %s action=%s: %s",
                loaded.agent_id,
                action_id,
                exc,
            )
            failed_actions += 1

    wall_end = time.time()
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
                "succeeded_actions": succeeded_actions,
                "failed_actions": failed_actions,
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
    """Write per-task trace.jsonl from the combined JSONL, filtered by agent_id."""
    agent_dirs = {
        s.loaded.agent_id: s.task_output_dir
        for s in sessions
        if s.task_output_dir is not None
    }
    sessions_by_agent = {s.loaded.agent_id: s for s in sessions}
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
                metadata["instance_id"] = session.agent_id
                metadata["source_trace"] = str(session.source_trace)
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
            docker_image_override=entry.docker_image,
            label=entry.label,
        )
        for entry in manifest_entries
    ]
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
    output_path.mkdir(parents=True, exist_ok=True)
    run_wall_start = time.monotonic()
    scheduler_mode = "bounded_queue"

    try:
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
            )
    finally:
        try:
            if trace_logger is not None:
                trace_logger.close()
                _split_trace_by_agent(trace_logger.path, prepared_sessions)
            for prepared in prepared_sessions:
                await _finalize_prepared_session(prepared)
        finally:
            if container_resource_recorder is not None:
                container_resource_summary = container_resource_recorder.stop()

    trace_file = output_path / f"{run_id}.jsonl"
    _write_throughput_summary(
        output_path=output_path,
        run_id=run_id,
        manifest=manifest,
        mode=mode,
        concurrency=concurrency,
        scheduler_mode=scheduler_mode,
        trace_file=trace_file,
        wall_time_s=time.monotonic() - run_wall_start,
        task_stats=task_stats,
        container_resources=container_resource_summary,
    )
    logger.info("Simulate complete [%s] -> %s", mode, trace_file)
    return trace_file
