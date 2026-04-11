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

from agents.base import TraceAction
from harness.metrics_client import VLLMMetricsClient
from harness.trace_logger import TraceLogger
from llm_call import create_async_openai_client
from trace_collect.scaffold_registry import (
    PreparedWorkspace,
    SimulatePrepareConfig,
    SimulateLLMConfig,
    get_prepare,
)

logger = logging.getLogger(__name__)


class SimulateError(Exception):
    """Raised when simulation encounters a fatal issue."""


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


@dataclass(slots=True)
class PreparedTraceSession:
    """Prepared workspace plus the loaded source-trace context."""

    loaded: LoadedTraceSession
    prepared: PreparedWorkspace


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
    agent_id: str,
    repo_dir: Path,
    tool_name: str | None,
    tool_args_json: str,
    command_timeout_s: float,
) -> tuple[str, float, bool]:
    """Execute one source-trace tool call in *repo_dir*.

    Returns:
        (tool_result, tool_duration_ms, tool_success)
    """
    from trace_collect.openclaw_tools import execute_trace_tool

    t0 = time.monotonic()
    tool_result, tool_success = await execute_trace_tool(
        agent_id=agent_id,
        tool_name=tool_name,
        tool_args_json=tool_args_json,
        repo_dir=repo_dir,
        command_timeout_s=command_timeout_s,
        command_output_style="raw",
    )
    duration_ms = (time.monotonic() - t0) * 1000
    return tool_result, duration_ms, tool_success


def _group_actions_by_iteration(
    actions: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Group loaded trace actions into the local-model iteration shape."""

    iterations: dict[int, dict[str, Any]] = {}
    for action in actions:
        it = int(action.get("iteration", 0))
        if it not in iterations:
            iterations[it] = {"llm": None, "tools": []}
        if action.get("action_type") == "llm_call":
            iterations[it]["llm"] = action
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
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

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
    return value.replace("/", "-").replace(":", "-").replace(" ", "-")


def _build_run_id(*, mode: str, model: str | None) -> str:
    label = model if model else mode
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"simulate_{_sanitize_run_label(label)}_{ts}"


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


def _load_trace_session(source_trace: Path, task_source: Path) -> LoadedTraceSession:
    agent_id, metadata, actions, summary = _parse_trace_session_file(source_trace)
    _validate_simulation_source_trace(metadata)
    scaffold = metadata.get("scaffold", "unknown") if metadata else "unknown"
    task = _find_task(task_source, agent_id)
    _validate_simulation_task(scaffold, task)
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
    )


def _load_trace_manifest(
    trace_manifest: Path,
    *,
    default_task_source: Path,
) -> list[tuple[Path, Path]]:
    try:
        raw = json.loads(trace_manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SimulateError(f"Invalid trace manifest JSON: {trace_manifest}") from exc
    if not isinstance(raw, list) or not raw:
        raise SimulateError("trace manifest must be a non-empty JSON array")

    base_dir = trace_manifest.parent
    entries: list[tuple[Path, Path]] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise SimulateError(
                f"trace manifest entry {index} must be an object with source_trace"
            )
        source_value = entry.get("source_trace")
        if not source_value:
            raise SimulateError(f"trace manifest entry {index} is missing source_trace")
        task_value = entry.get("task_source")
        source_path = Path(source_value)
        task_path = Path(task_value) if task_value else default_task_source
        if not source_path.is_absolute():
            source_path = (base_dir / source_path).resolve()
        if task_value and not task_path.is_absolute():
            task_path = (base_dir / task_path).resolve()
        entries.append((source_path, task_path))
    return entries


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
    if mode == "cloud_model":
        unsupported = sorted(
            {session.scaffold for session in sessions if session.scaffold != "openclaw"}
        )
        if unsupported:
            raise NotImplementedError(
                "cloud_model replay currently supports only repo-backed "
                f"OpenClaw traces, got: {unsupported}"
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
            ts_start = _coerce_timestamp(
                action.get("ts_start"),
                field="ts_start",
                source_trace=session.source_trace,
                action_id=action_id,
            )
            ts_end = _coerce_timestamp(
                action.get("ts_end"),
                field="ts_end",
                source_trace=session.source_trace,
                action_id=action_id,
            )
            if ts_end < ts_start:
                raise SimulateError(
                    f"{session.source_trace} action {action_id!r} has ts_end < ts_start"
                )


def _build_prepare_config(
    *,
    agent_id: str,
    api_base: str | None,
    api_key: str | None,
    model: str | None,
    command_timeout_s: float,
    task_timeout_s: float,
    repos_root: Path,
    max_context_tokens: int,
) -> SimulatePrepareConfig:
    llm = None
    if api_base is not None or api_key is not None or model is not None:
        if api_base is None or api_key is None or model is None:
            raise ValueError("prepare LLM config must be all-set or all-None")
        llm = SimulateLLMConfig(api_base=api_base, api_key=api_key, model=model)
    return SimulatePrepareConfig(
        agent_id=agent_id,
        llm=llm,
        command_timeout_s=command_timeout_s,
        task_timeout_s=task_timeout_s,
        repos_root=Path(repos_root) if repos_root else None,
        max_context_tokens=max_context_tokens,
    )


async def _prepare_trace_session(
    loaded: LoadedTraceSession,
    *,
    api_base: str | None,
    api_key: str | None,
    model: str | None,
    command_timeout_s: float,
    task_timeout_s: float,
    repos_root: Path,
    max_context_tokens: int,
) -> PreparedTraceSession:
    prepare_callable = get_prepare(loaded.scaffold)
    prepare_config = _build_prepare_config(
        agent_id=loaded.agent_id,
        api_base=api_base,
        api_key=api_key,
        model=model,
        command_timeout_s=command_timeout_s,
        task_timeout_s=task_timeout_s,
        repos_root=repos_root,
        max_context_tokens=max_context_tokens,
    )
    prepared = await prepare_callable(loaded.task, prepare_config)
    return PreparedTraceSession(loaded=loaded, prepared=prepared)


def _log_trace_metadata(
    *,
    trace_logger: TraceLogger,
    mode: str,
    sessions: list[LoadedTraceSession],
    replay_speed: float,
    source_trace: Path | None,
    trace_manifest: Path | None,
    api_base: str | None,
    model: str | None,
) -> None:
    scaffolds = {session.scaffold for session in sessions}
    source_models = [
        (session.summary or {}).get("model", "unknown") for session in sessions
    ]
    metadata: dict[str, Any] = {
        "scaffold": sessions[0].scaffold if len(scaffolds) == 1 else "mixed",
        "mode": "simulate",
        "simulate_mode": mode,
        "replay_speed": replay_speed,
        "source_trace_count": len(sessions),
        "source_models": source_models,
    }
    if source_trace is not None:
        metadata["source_trace"] = str(source_trace)
    if trace_manifest is not None:
        metadata["trace_manifest"] = str(trace_manifest)
        metadata["source_traces"] = [str(session.source_trace) for session in sessions]
    if mode == "local_model":
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


async def _run_local_model_simulation(
    prepared_session: PreparedTraceSession,
    *,
    trace_logger: TraceLogger,
    api_base: str,
    api_key: str,
    model: str,
    command_timeout_s: float,
    metrics_url: str | None,
    warmup_skip_iterations: int,
) -> None:
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

    metrics_client = VLLMMetricsClient(metrics_url=metrics_url)
    logger.info(
        "vLLM metrics client: %s",
        f"enabled (url={metrics_url})" if metrics_client.is_enabled else "disabled",
    )

    client = create_async_openai_client(
        api_base=api_base,
        api_key=api_key,
        timeout=180.0,
    )

    wall_start = time.time()
    total_iters = len(iterations)
    succeeded_iters = 0
    failed_iters = 0
    sorted_iters = sorted(iterations.keys())

    try:
        for i, it_num in enumerate(sorted_iters):
            it_group = iterations[it_num]
            llm_action = it_group.get("llm") or {}
            llm_data = llm_action.get("data", {})
            tool_actions = it_group.get("tools", [])

            messages_in = llm_data.get("messages_in")
            n_tokens = llm_data.get("completion_tokens", 1) or 1

            if not messages_in:
                logger.warning("Iteration %d: no messages_in, skipping", it_num)
                continue

            ts_start = time.time()

            try:
                ttft_ms, tpot_ms, llm_latency_ms = await _call_local_model_streaming(
                    client, model, messages_in, n_tokens
                )
            except Exception as exc:
                logger.error("Iteration %d: LLM call failed: %s", it_num, exc)
                failed_iters += 1
                continue

            ts_after_llm = time.time()

            scheduler_snapshot = metrics_client.get_snapshot()

            llm_record = _make_trace_action(
                loaded=loaded,
                action_type="llm_call",
                action_id=f"llm_{it_num}",
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

            from agents.openclaw.simulate_adapter import is_mcp_tool_call

            total_tool_ms = 0.0
            for tool_act in tool_actions:
                td = tool_act.get("data", {})
                tool_name = td.get("tool_name")
                tool_args = td.get("tool_args", "{}")
                if not tool_name:
                    continue

                tool_ts_start = time.time()
                if is_mcp_tool_call(tool_name):
                    tool_result = td.get("tool_result", "")
                    tool_duration_ms = float(td.get("duration_ms") or 0.0)
                    tool_success = bool(td.get("success", True))
                    tool_ts_end = tool_ts_start
                    sim_provenance = "replayed_from_trace"
                else:
                    tool_result, tool_duration_ms, tool_success = await _exec_tool(
                        loaded.agent_id,
                        prepared_session.prepared.repo_dir,
                        tool_name,
                        tool_args,
                        command_timeout_s,
                    )
                    tool_ts_end = time.time()
                    sim_provenance = "executed_locally"
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
                            "warmup": i < warmup_skip_iterations,
                        },
                    },
                )
                trace_logger.log_trace_action(loaded.agent_id, tool_record)

            succeeded_iters += 1

            logger.info(
                "[%d/%d] iter %d: ttft=%.1fms tpot=%.2fms llm=%.0fms tool=%.0fms",
                i + 1,
                total_iters,
                it_num,
                ttft_ms,
                tpot_ms,
                llm_latency_ms,
                total_tool_ms,
            )
    finally:
        wall_end = time.time()

        simulate_summary = _make_trace_summary(
            loaded=loaded,
            success=failed_iters == 0 and succeeded_iters == total_iters,
            elapsed_s=wall_end - wall_start,
            source_model=source_model,
            extra={
                "local_model": model,
                "local_api_base": api_base,
                "succeeded_iterations": succeeded_iters,
                "failed_iterations": failed_iters,
            },
        )
        trace_logger.log_summary(loaded.agent_id, simulate_summary)


async def _sleep_until_offset(
    *,
    replay_zero_monotonic: float,
    target_offset_s: float,
) -> None:
    delay_s = target_offset_s - (time.monotonic() - replay_zero_monotonic)
    if delay_s > 0:
        await asyncio.sleep(delay_s)


async def _run_cloud_model_replay(
    prepared_sessions: list[PreparedTraceSession],
    *,
    trace_logger: TraceLogger,
    replay_speed: float,
    command_timeout_s: float,
    warmup_skip_iterations: int,
) -> None:
    replay_zero_monotonic = time.monotonic()
    await asyncio.gather(
        *[
            _replay_cloud_model_session(
                prepared_session,
                trace_logger=trace_logger,
                replay_zero_monotonic=replay_zero_monotonic,
                replay_speed=replay_speed,
                command_timeout_s=command_timeout_s,
                warmup_skip_iterations=warmup_skip_iterations,
            )
            for prepared_session in prepared_sessions
        ]
    )


async def _replay_cloud_model_session(
    prepared_session: PreparedTraceSession,
    *,
    trace_logger: TraceLogger,
    replay_zero_monotonic: float,
    replay_speed: float,
    command_timeout_s: float,
    warmup_skip_iterations: int,
) -> None:
    from agents.openclaw.simulate_adapter import is_mcp_tool_call

    loaded = prepared_session.loaded
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
        action_ts_start = _coerce_timestamp(
            action.get("ts_start"),
            field="ts_start",
            source_trace=loaded.source_trace,
            action_id=action_id,
        )
        action_ts_end = _coerce_timestamp(
            action.get("ts_end"),
            field="ts_end",
            source_trace=loaded.source_trace,
            action_id=action_id,
        )
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
            if is_mcp_tool_call(tool_name):
                if source_duration_ms > 0:
                    await asyncio.sleep(source_duration_ms / 1000 / replay_speed)
                tool_result = data.get("tool_result", "")
                tool_success = bool(data.get("success", True))
                duration_ms = (time.time() - record_ts_start) * 1000
                replay_source = "replayed_from_trace"
            else:
                tool_result, duration_ms, tool_success = await _exec_tool(
                    loaded.agent_id,
                    prepared_session.prepared.repo_dir,
                    tool_name,
                    tool_args,
                    command_timeout_s,
                )
                replay_source = "executed_locally"
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
    trace_logger.log_summary(
        loaded.agent_id,
        _make_trace_summary(
            loaded=loaded,
            success=failed_actions == 0,
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


async def simulate(
    *,
    source_trace: Path | None = None,
    trace_manifest: Path | None = None,
    task_source: Path,
    repos_root: Path,
    output_dir: Path,
    mode: str = "local_model",
    api_base: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    command_timeout_s: float = 120.0,
    task_timeout_s: float = 1200.0,
    max_context_tokens: int = 256_000,
    metrics_url: str | None = None,
    warmup_skip_iterations: int = 0,
    replay_speed: float = 1.0,
) -> Path:
    if source_trace is not None and trace_manifest is not None:
        raise ValueError("source_trace and trace_manifest are mutually exclusive")
    if source_trace is None and trace_manifest is None:
        raise ValueError("simulate requires source_trace or trace_manifest")
    if mode not in {"local_model", "cloud_model"}:
        raise ValueError(f"Unsupported simulate mode: {mode}")

    if trace_manifest is not None:
        trace_inputs = _load_trace_manifest(
            trace_manifest,
            default_task_source=task_source.resolve(),
        )
    else:
        assert source_trace is not None
        trace_inputs = [(source_trace, task_source)]

    loaded_sessions = [
        _load_trace_session(source_path, task_path)
        for source_path, task_path in trace_inputs
    ]
    _validate_loaded_sessions(
        loaded_sessions,
        mode=mode,
        replay_speed=replay_speed,
    )

    prepared_sessions: list[PreparedTraceSession] = []
    trace_logger: TraceLogger | None = None
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        for loaded in loaded_sessions:
            prepared_sessions.append(
                await _prepare_trace_session(
                    loaded,
                    api_base=api_base,
                    api_key=api_key,
                    model=model,
                    command_timeout_s=command_timeout_s,
                    task_timeout_s=task_timeout_s,
                    repos_root=repos_root,
                    max_context_tokens=max_context_tokens,
                )
            )

        run_id = _build_run_id(mode=mode, model=model)
        trace_logger = TraceLogger(output_path, run_id)
        _log_trace_metadata(
            trace_logger=trace_logger,
            mode=mode,
            sessions=loaded_sessions,
            replay_speed=replay_speed,
            source_trace=source_trace,
            trace_manifest=trace_manifest,
            api_base=api_base,
            model=model,
        )

        if mode == "local_model":
            assert trace_logger is not None
            assert api_base is not None
            assert api_key is not None
            assert model is not None
            await _run_local_model_simulation(
                prepared_sessions[0],
                trace_logger=trace_logger,
                api_base=api_base,
                api_key=api_key,
                model=model,
                command_timeout_s=command_timeout_s,
                metrics_url=metrics_url,
                warmup_skip_iterations=warmup_skip_iterations,
            )
        else:
            assert trace_logger is not None
            await _run_cloud_model_replay(
                prepared_sessions,
                trace_logger=trace_logger,
                replay_speed=replay_speed,
                command_timeout_s=command_timeout_s,
                warmup_skip_iterations=warmup_skip_iterations,
            )
    finally:
        if trace_logger is not None:
            trace_logger.close()
        for prepared in prepared_sessions:
            prepared.prepared.cleanup()

    trace_file = output_path / f"{run_id}.jsonl"
    logger.info("Simulate complete [%s] -> %s", mode, trace_file)
    return trace_file


def _validate_simulation_task(scaffold: str, task: dict[str, Any]) -> None:
    if scaffold == "openclaw" and (not task.get("repo") or not task.get("base_commit")):
        raise NotImplementedError(
            "Simulate mode requires repo-backed OpenClaw tasks with repo and "
            "base_commit."
        )


def _validate_simulation_source_trace(metadata: dict[str, Any] | None) -> None:
    if metadata is None or metadata.get("scaffold") != "openclaw":
        return
    if metadata.get("needs_prepare") is False:
        raise NotImplementedError("Simulate mode requires repo-backed OpenClaw traces.")
    if metadata.get("task_shape") not in (None, "swe_patch"):
        raise NotImplementedError("Simulate mode requires repo-backed OpenClaw traces.")
