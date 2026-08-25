from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml

from trace_collect.simulate_types import (
    LoadedTraceSession,
    SimulateError,
    TraceManifestEntry,
    WorkerTraceInput,
)

SIMULATE_MANIFEST_SCHEMA_VERSION = 1


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


def _parse_trace_session_file(
    trace_path: Path,
) -> tuple[str, str, dict[str, Any] | None, list[dict[str, Any]], dict[str, Any] | None]:
    """Read one canonical trace and keep task id separate from action owner id."""

    metadata: dict[str, Any] | None = None
    all_actions: list[dict[str, Any]] = []
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
                all_actions.append(record)
                continue

            if record_type == "summary" and agent_id:
                summaries[str(agent_id)] = record

    if not all_actions:
        raise SimulateError(f"No action records with agent_id found in {trace_path}")

    metadata_task_id = (metadata or {}).get("instance_id")
    if isinstance(metadata_task_id, str) and metadata_task_id:
        task_instance_id = metadata_task_id
    else:
        first_task_id = all_actions[0].get("instance_id")
        if isinstance(first_task_id, str) and first_task_id:
            task_instance_id = first_task_id
        else:
            task_instance_id = str(all_actions[0]["agent_id"])

    action_owner_ids = {
        str(action.get("agent_id"))
        for action in all_actions
        if action.get("agent_id")
        and ":subagent:" not in str(action.get("agent_id"))
    }
    if len(summaries) == 1:
        source_action_agent_id = next(iter(summaries))
    elif len(action_owner_ids) == 1:
        source_action_agent_id = next(iter(action_owner_ids))
    elif task_instance_id in summaries:
        source_action_agent_id = task_instance_id
    elif task_instance_id in action_owner_ids:
        source_action_agent_id = task_instance_id
    else:
        raise SimulateError(
            "Ambiguous action owner for task "
            f"{task_instance_id!r} in {trace_path}: "
            f"owners={sorted(action_owner_ids)!r}, summaries={sorted(summaries)!r}"
        )

    source_subagent_prefix = f"{source_action_agent_id}:subagent:"
    actions = [
        action
        for action in all_actions
        if action.get("agent_id") == source_action_agent_id
        or str(action.get("agent_id", "")).startswith(source_subagent_prefix)
    ]
    if not actions:
        raise SimulateError(
            "No action records for task "
            f"{task_instance_id!r} and action owner {source_action_agent_id!r} "
            f"found in {trace_path}"
        )

    actions.sort(
        key=lambda action: (
            float(action.get("ts_start", 0.0)),
            float(action.get("ts_end", 0.0)),
            int(action.get("iteration", 0)),
            str(action.get("action_id", "")),
        )
    )
    return (
        task_instance_id,
        source_action_agent_id,
        metadata,
        actions,
        summaries.get(source_action_agent_id),
    )


def _find_task(task_source: Path, task_instance_id: str) -> dict[str, Any]:
    tasks = json.loads(task_source.read_text(encoding="utf-8"))
    for task in tasks:
        if task["instance_id"] == task_instance_id:
            return task
    raise SimulateError(f"Task {task_instance_id!r} not found in {task_source}")

def _parse_depends_on(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SimulateError(f"{field} must be a list of strings")
    depends_on: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise SimulateError(f"{field}[{index}] must be a non-empty string")
        if item in seen:
            raise SimulateError(f"{field} contains duplicate dependency {item!r}")
        seen.add(item)
        depends_on.append(item)
    return tuple(depends_on)


def _combine_depends_on(*values: tuple[str, ...]) -> tuple[str, ...]:
    combined: list[str] = []
    seen: set[str] = set()
    for depends_on in values:
        for dependency in depends_on:
            if dependency not in seen:
                seen.add(dependency)
                combined.append(dependency)
    return tuple(combined)


def _load_trace_session(
    source_trace: Path,
    task_source: Path,
    manifest_index: int,
    docker_image_override: str | None = None,
    label: str | None = None,
    manifest_depends_on: tuple[str, ...] = (),
    arrival_s: float = 0.0,
) -> LoadedTraceSession:
    (
        task_instance_id,
        source_action_agent_id,
        metadata,
        actions,
        summary,
    ) = _parse_trace_session_file(source_trace)
    scaffold = metadata.get("scaffold", "unknown") if metadata else "unknown"
    task = _find_task(task_source, task_instance_id)
    task_depends_on = (
        _parse_depends_on(
            task["depends_on"],
            field=f"task {task_instance_id!r} depends_on",
        )
        if "depends_on" in task
        else ()
    )
    return LoadedTraceSession(
        source_trace=source_trace,
        task_source=task_source,
        task_instance_id=task_instance_id,
        source_action_agent_id=source_action_agent_id,
        run_instance_id=source_action_agent_id,
        manifest_index=manifest_index,
        scaffold=scaffold,
        metadata=metadata,
        summary=summary,
        task=task,
        actions=actions,
        iterations=_group_actions_by_iteration(actions),
        docker_image_override=docker_image_override,
        label=label,
        depends_on=_combine_depends_on(manifest_depends_on, task_depends_on),
        arrival_s=arrival_s,
    )


def _assign_replay_instance_ids(sessions: list[LoadedTraceSession]) -> None:
    source_counts: dict[str, int] = {}
    for session in sessions:
        source_counts[session.task_instance_id] = (
            source_counts.get(session.task_instance_id, 0) + 1
        )

    reserved_source_ids = set(source_counts)
    used_ids: set[str] = set()
    source_occurrences: dict[str, int] = {}

    for session in sessions:
        task_instance_id = session.task_instance_id
        if source_counts[task_instance_id] == 1:
            candidate = task_instance_id
        else:
            occurrence = source_occurrences.get(task_instance_id, 0) + 1
            source_occurrences[task_instance_id] = occurrence
            base = f"{task_instance_id}__replica-{occurrence:03d}"
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
        task_instance_id=session.task_instance_id,
        source_action_agent_id=session.source_action_agent_id,
        depends_on=session.depends_on,
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
            manifest_depends_on=entry.depends_on,
        )
        if session.task_instance_id != entry.task_instance_id:
            raise SimulateError(
                f"Worker task id changed while reloading {entry.source_trace}: "
                f"{session.task_instance_id!r} != {entry.task_instance_id!r}"
            )
        if session.source_action_agent_id != entry.source_action_agent_id:
            raise SimulateError(
                f"Worker action owner changed while reloading {entry.source_trace}: "
                f"{session.source_action_agent_id!r} != {entry.source_action_agent_id!r}"
            )
        if session.depends_on != entry.depends_on:
            raise SimulateError(
                f"Worker dependency metadata changed while reloading {entry.source_trace}: "
                f"{session.depends_on!r} != {entry.depends_on!r}"
            )
        session.run_instance_id = entry.run_instance_id
        sessions.append(session)
    return sessions


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
    default_task_source: Path | None,
) -> list[TraceManifestEntry]:
    try:
        raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SimulateError(f"Invalid simulate manifest YAML: {manifest}") from exc

    base_dir = manifest.parent
    manifest_default_task_source: Path | None = None
    manifest_default_docker_image: str | None = None
    manifest_requires_trace_tool_replay = False
    raw_traces: Any

    if isinstance(raw, list):
        raw_traces = raw
    elif isinstance(raw, dict):
        allowed_manifest_keys = {
            "version",
            "defaults",
            "traces",
            "requires_trace_tool_replay",
        }
        unknown_manifest_keys = set(raw) - allowed_manifest_keys
        if unknown_manifest_keys:
            keys = ", ".join(sorted(str(key) for key in unknown_manifest_keys))
            raise SimulateError(f"simulate manifest has unsupported top-level keys: {keys}")
        version = raw.get("version", SIMULATE_MANIFEST_SCHEMA_VERSION)
        if version != SIMULATE_MANIFEST_SCHEMA_VERSION:
            raise SimulateError(
                "simulate manifest version must be "
                f"{SIMULATE_MANIFEST_SCHEMA_VERSION}, got {version!r}"
            )
        replay_requirement = raw.get("requires_trace_tool_replay", False)
        if not isinstance(replay_requirement, bool):
            raise SimulateError(
                "simulate manifest requires_trace_tool_replay must be a boolean"
            )
        manifest_requires_trace_tool_replay = replay_requirement
        defaults = raw.get("defaults") or {}
        if not isinstance(defaults, dict):
            raise SimulateError("simulate manifest defaults must be an object")
        unknown_default_keys = set(defaults) - {"task_source", "docker_image"}
        if unknown_default_keys:
            keys = ", ".join(sorted(str(key) for key in unknown_default_keys))
            raise SimulateError(f"simulate manifest defaults has unsupported keys: {keys}")
        if "task_source" in defaults:
            manifest_default_task_source = _resolve_manifest_path(
                defaults["task_source"],
                base_dir=base_dir,
                field="defaults.task_source",
            )
        if "docker_image" in defaults:
            docker_value = defaults["docker_image"]
            if not isinstance(docker_value, str) or not docker_value:
                raise SimulateError(
                    "simulate manifest defaults.docker_image must be a non-empty string"
                )
            manifest_default_docker_image = docker_value
        raw_traces = raw.get("traces")
    else:
        raise SimulateError("simulate manifest must be a YAML list or object with traces")

    if not isinstance(raw_traces, list) or not raw_traces:
        raise SimulateError("simulate manifest traces must be a non-empty list")

    entries: list[TraceManifestEntry] = []
    for index, entry in enumerate(raw_traces):
        trace_value: Any
        task_value: Any | None = None
        docker_image = manifest_default_docker_image
        label: str | None = None
        depends_on: tuple[str, ...] = ()
        arrival_s = 0.0

        if isinstance(entry, str):
            trace_value = entry
        elif isinstance(entry, dict):
            allowed_entry_keys = {
                "trace",
                "task_source",
                "docker_image",
                "label",
                "depends_on",
                "arrival_s",
            }
            unknown_entry_keys = set(entry) - allowed_entry_keys
            if unknown_entry_keys:
                keys = ", ".join(sorted(str(key) for key in unknown_entry_keys))
                raise SimulateError(
                    f"simulate manifest trace entry {index} has unsupported keys: {keys}"
                )
            if "trace" not in entry:
                raise SimulateError(
                    f"simulate manifest trace entry {index} is missing trace"
                )
            trace_value = entry["trace"]
            task_value = entry.get("task_source")
            docker_value = entry.get("docker_image")
            label_value = entry.get("label")
            depends_value = entry.get("depends_on")
            arrival_value = entry.get("arrival_s", 0.0)
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
            if "depends_on" in entry:
                depends_on = _parse_depends_on(
                    depends_value,
                    field=f"simulate manifest trace entry {index} depends_on",
                )
            if (
                not isinstance(arrival_value, (int, float))
                or isinstance(arrival_value, bool)
                or not math.isfinite(arrival_value)
                or arrival_value < 0
            ):
                raise SimulateError(
                    f"simulate manifest trace entry {index} arrival_s "
                    "must be a finite non-negative number"
                )
            arrival_s = float(arrival_value)
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
        if task_value is not None:
            task_path = _resolve_manifest_path(
                task_value,
                base_dir=base_dir,
                field=f"traces[{index}].task_source",
            )
        elif manifest_default_task_source is not None:
            task_path = manifest_default_task_source
        elif default_task_source is not None:
            task_path = default_task_source
        else:
            raise SimulateError(
                f"simulate manifest trace entry {index} needs task_source; "
                "set defaults.task_source, traces[].task_source, or --task-source"
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
                depends_on=depends_on,
                arrival_s=arrival_s,
                requires_trace_tool_replay=manifest_requires_trace_tool_replay,
            )
        )
    return entries
