from __future__ import annotations

import dataclasses
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.base import TraceAction
from harness.container_stats_sampler import summarize_samples
from harness.trace_logger import TraceLogger
from trace_collect import attempt_layout
from trace_collect.attempt_pipeline import next_attempt_number_in
from trace_collect.simulate_types import (
    LLMTimingConfig,
    LoadedTraceSession,
    PreparedTraceSession,
    ReplayTaskStats,
    SimulateError,
    WorkerReplayResult,
)
from trace_collect.simulate_utils import (
    _execution_environment,
    _iteration_count,
    _resolve_prep_concurrency,
    _sanitize_run_label,
    _source_model,
)

logger = logging.getLogger(__name__)


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
                benchmark,
                model,
                scaffold,
                session.agent_id,
                other.get("benchmark"),
                other.get("model"),
                other.get("scaffold"),
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
    source_models = [_source_model(session) for session in sessions]
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
                "task_instance_id": session.task_instance_id,
                "source_action_agent_id": session.source_action_agent_id,
                "source_agent_id": session.source_action_agent_id,
                "run_instance_id": session.run_instance_id,
                "label": session.label,
                "source_model": _source_model(session),
                "depends_on": list(session.depends_on),
            }
            for session in sessions
        ],
        "task_instance_ids": [session.task_instance_id for session in sessions],
        "source_action_agent_ids": [
            session.source_action_agent_id for session in sessions
        ],
        "source_agent_ids": [session.source_action_agent_id for session in sessions],
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
    agent_id: str | None = None,
) -> TraceAction:
    replay_agent_id = agent_id or loaded.run_instance_id
    per_action_source_agent_id = data.get(
        "source_action_agent_id",
        loaded.source_action_agent_id,
    )
    action_data = {
        **data,
        "run_instance_id": loaded.run_instance_id,
        "task_instance_id": loaded.task_instance_id,
        "source_action_agent_id": per_action_source_agent_id,
        "source_agent_id": loaded.source_action_agent_id,
        "manifest_index": loaded.manifest_index,
    }
    if loaded.label is not None:
        action_data["label"] = loaded.label
    return TraceAction(
        action_type=action_type,
        action_id=action_id,
        agent_id=replay_agent_id,
        program_id=replay_agent_id,
        instance_id=loaded.run_instance_id,
        iteration=iteration,
        ts_start=ts_start,
        ts_end=ts_end,
        data=action_data,
    )


def _replay_agent_id_for_action(
    loaded: LoadedTraceSession,
    source_action_agent_id: Any,
) -> str:
    if not isinstance(source_action_agent_id, str) or not source_action_agent_id:
        return loaded.run_instance_id
    subagent_prefix = f"{loaded.source_action_agent_id}:subagent:"
    if source_action_agent_id == loaded.source_action_agent_id:
        return loaded.run_instance_id
    if source_action_agent_id.startswith(subagent_prefix):
        return (
            f"{loaded.run_instance_id}:subagent:"
            f"{source_action_agent_id[len(subagent_prefix):]}"
        )
    return loaded.run_instance_id


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
        "source_agent_id": loaded.source_action_agent_id,
        "source_action_agent_id": loaded.source_action_agent_id,
        "task_instance_id": loaded.task_instance_id,
        "task_id": loaded.task_instance_id,
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
    llm_call_count = sum(
        1 for action in loaded.actions if action.get("action_type") == "llm_call"
    )
    tool_exec_count = sum(
        1 for action in loaded.actions if action.get("action_type") == "tool_exec"
    )
    return ReplayTaskStats(
        agent_id=loaded.run_instance_id,
        run_instance_id=loaded.run_instance_id,
        source_agent_id=loaded.source_action_agent_id,
        manifest_index=loaded.manifest_index,
        label=loaded.label,
        source_trace=str(loaded.source_trace),
        success=success,
        elapsed_s=elapsed_s,
        action_count=len(loaded.actions),
        llm_call_count=llm_call_count,
        tool_exec_count=tool_exec_count,
        failed_action_count=failed_action_count,
        depends_on=loaded.depends_on,
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

    def owning_agent_id(agent_id: Any) -> str | None:
        if not isinstance(agent_id, str) or not agent_id:
            return None
        if agent_id in per_agent:
            return agent_id
        for run_instance_id in per_agent:
            if agent_id.startswith(f"{run_instance_id}:subagent:"):
                return run_instance_id
        return None

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
            owner = owning_agent_id(record.get("agent_id"))
            if owner is not None:
                per_agent[owner].append(stripped)

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
                metadata["task_instance_id"] = session.task_instance_id
                metadata["source_action_agent_id"] = session.source_action_agent_id
                metadata["source_agent_id"] = session.source_action_agent_id
                metadata["task_id"] = session.task_instance_id
                metadata["manifest_index"] = session.manifest_index
                metadata["label"] = session.label
                metadata["source_trace"] = str(session.source_trace)
                metadata["source_trace_count"] = 1
                metadata["source_traces"] = [str(session.source_trace)]
                metadata["source_trace_entries"] = [
                    {
                        "manifest_index": session.manifest_index,
                        "source_trace": str(session.source_trace),
                        "task_instance_id": session.task_instance_id,
                        "source_action_agent_id": session.source_action_agent_id,
                        "source_agent_id": session.source_action_agent_id,
                        "run_instance_id": session.run_instance_id,
                        "label": session.label,
                        "source_model": _source_model(session),
                    }
                ]
                metadata["task_instance_ids"] = [session.task_instance_id]
                metadata["source_action_agent_ids"] = [session.source_action_agent_id]
                metadata["source_agent_ids"] = [session.source_action_agent_id]
                metadata["run_instance_ids"] = [session.run_instance_id]
                source_model = _source_model(session)
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
    _split_trace_by_agent(
        combined_path,
        prepared_sessions,
    )


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


def _combined_trace_sort_key(
    record: dict[str, Any], sequence: int
) -> tuple[float, int, int]:
    rtype = record.get("type")
    if rtype == "action":
        return (
            _float_sort_value(record.get("ts_start"), default=float("inf")),
            0,
            sequence,
        )
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
