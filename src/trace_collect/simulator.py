
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

import dataclasses

from agents.base import TraceAction
from harness.metrics_client import VLLMMetricsClient
from harness.trace_logger import TraceLogger
from trace_collect.openclaw_tools import execute_trace_tool
from trace_collect.scaffold_registry import (
    PreparedWorkspace,
    SimulatePrepareConfig,
    get_prepare,
)

logger = logging.getLogger(__name__)

class SimulateError(Exception):
    """Raised when simulation encounters a fatal issue."""

async def _call_local_model_streaming(
    client: AsyncOpenAI,
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
    raw_response: dict[str, Any] | None = None,
) -> tuple[str, float, bool]:
    """Execute one source-trace tool call in *repo_dir*.

    Returns:
        (tool_result, tool_duration_ms, tool_success)
    """
    t0 = time.monotonic()
    tool_result, tool_success = await execute_trace_tool(
        agent_id=agent_id,
        tool_name=tool_name,
        tool_args_json=tool_args_json,
        repo_dir=repo_dir,
        command_timeout_s=command_timeout_s,
        command_output_style="raw",
        raw_response=raw_response,
    )
    duration_ms = (time.monotonic() - t0) * 1000
    return tool_result, duration_ms, tool_success

def load_trace_actions(
    trace_path: Path,
    agent_id: str,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any] | None]:
    """Load canonical trace actions grouped by iteration, plus optional summary.

    Returns:
        (iterations dict {iteration_num: {"llm": action_dict, "tools": [action_dict, ...]}},
         summary or None)

    Raises:
        SimulateError: if agent_id not found or no action records exist.
    """
    actions: list[dict[str, Any]] = []
    summary: dict[str, Any] | None = None

    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("agent_id") != agent_id:
                continue
            rtype = record.get("type")
            if rtype == "action":
                actions.append(record)
            elif rtype == "summary":
                summary = record

    if not actions:
        raise SimulateError(f"No action records found for agent_id={agent_id!r}")

    iterations: dict[int, dict[str, Any]] = {}
    for a in actions:
        it = a.get("iteration", 0)
        if it not in iterations:
            iterations[it] = {"llm": None, "tools": []}
        if a.get("action_type") == "llm_call":
            iterations[it]["llm"] = a
        elif a.get("action_type") == "tool_exec":
            iterations[it]["tools"].append(a)
    return iterations, summary

def _find_task(task_source: Path, agent_id: str) -> dict[str, Any]:
    tasks = json.loads(task_source.read_text(encoding="utf-8"))
    for task in tasks:
        if task["instance_id"] == agent_id:
            return task
    raise SimulateError(f"Task {agent_id!r} not found in {task_source}")

async def simulate(
    *,
    source_trace: Path,
    task_source: Path,
    repos_root: Path,
    output_dir: Path,
    api_base: str,
    api_key: str,
    model: str,
    command_timeout_s: float = 120.0,
    task_timeout_s: float = 1200.0,
    max_context_tokens: int = 256_000,
    metrics_url: str | None = None,
    warmup_skip_iterations: int = 0,
) -> Path:
    metadata = _load_trace_metadata(source_trace)
    _validate_trace_for_simulation(metadata)

    first_agent_id = _detect_agent_id(source_trace)
    iterations, summary = load_trace_actions(source_trace, first_agent_id)
    agent_id = first_agent_id
    scaffold = _detect_scaffold(source_trace)

    task = _find_task(task_source, agent_id)
    source_model = (summary or {}).get("model", "unknown")
    logger.info(
        "Simulating %s [scaffold=%s]: %d iterations from %s, local model=%s",
        agent_id,
        scaffold,
        len(iterations),
        source_model,
        model,
    )

    prepare_callable = get_prepare(scaffold)
    prepare_config = SimulatePrepareConfig(
        agent_id=agent_id,
        api_base=api_base,
        model=model,
        api_key=api_key,
        command_timeout_s=command_timeout_s,
        task_timeout_s=task_timeout_s,
        repos_root=Path(repos_root) if repos_root else None,
        max_context_tokens=max_context_tokens,
    )
    prepared: PreparedWorkspace = await prepare_callable(task, prepare_config)
    repo_dir = prepared.repo_dir

    metrics_client = VLLMMetricsClient(metrics_url=metrics_url)
    logger.info(
        "vLLM metrics client: %s",
        f"enabled (url={metrics_url})" if metrics_client.is_enabled else "disabled",
    )

    run_id = f"simulate_{model.replace('/', '-').replace(':', '-')}_{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trace_logger = TraceLogger(output_path, run_id)

    trace_logger.log_metadata(
        scaffold=_detect_scaffold(source_trace),
        mode="simulate",
        source_trace=str(source_trace),
        source_model=source_model,
        local_model=model,
        local_api_base=api_base,
        n_source_iterations=len(iterations),
    )

    client = AsyncOpenAI(
        base_url=api_base,
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
                    client, model, messages_in, n_tokens,
                )
            except Exception as exc:
                logger.error("Iteration %d: LLM call failed: %s", it_num, exc)
                failed_iters += 1
                continue

            ts_after_llm = time.time()

            scheduler_snapshot = metrics_client.get_snapshot()

            llm_record = TraceAction(
                action_type="llm_call",
                action_id=f"llm_{it_num}",
                agent_id=agent_id,
                program_id=agent_id,
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
                    "simulate_source": str(source_trace),
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
            trace_logger.log_trace_action(agent_id, llm_record)

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
                        agent_id, repo_dir, tool_name, tool_args,
                        command_timeout_s, llm_data.get("raw_response"),
                    )
                    tool_ts_end = time.time()
                    sim_provenance = "executed_locally"
                total_tool_ms += tool_duration_ms

                tool_record = TraceAction(
                    action_type="tool_exec",
                    action_id=f"tool_{it_num}_{tool_name}",
                    agent_id=agent_id,
                    program_id=agent_id,
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
                trace_logger.log_trace_action(agent_id, tool_record)

            succeeded_iters += 1

            logger.info(
                "[%d/%d] iter %d: ttft=%.1fms tpot=%.2fms llm=%.0fms tool=%.0fms",
                i + 1, total_iters, it_num,
                ttft_ms, tpot_ms, llm_latency_ms, total_tool_ms,
            )

    finally:
        wall_end = time.time()

        simulate_summary: dict[str, Any] = {
            "agent_id": agent_id,
            "task_id": agent_id,
            "n_iterations": total_iters,
            "elapsed_s": wall_end - wall_start,
            "source_trace": str(source_trace),
            "source_model": source_model,
            "local_model": model,
            "local_api_base": api_base,
            "succeeded_iterations": succeeded_iters,
            "failed_iterations": failed_iters,
        }
        trace_logger.log_summary(agent_id, simulate_summary)
        trace_logger.close()

        prepared.cleanup()

    trace_file = output_path / f"{run_id}.jsonl"
    logger.info(
        "Simulate complete: %s iterations=%d/%d elapsed=%.1fs -> %s",
        agent_id,
        succeeded_iters,
        total_iters,
        wall_end - wall_start,
        trace_file,
    )
    return trace_file

def _detect_scaffold(trace_path: Path) -> str:
    metadata = _load_trace_metadata(trace_path)
    return metadata.get("scaffold", "unknown") if metadata else "unknown"

def _load_trace_metadata(trace_path: Path) -> dict[str, Any] | None:
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") == "trace_metadata":
                return record
            if record.get("type") in ("step", "summary", "action"):
                break
    return None

def _validate_trace_for_simulation(metadata: dict[str, Any] | None) -> None:
    """Reject trace types the replay path cannot prepare into a workspace.

    The simulator only supports traces whose scaffold can materialize a real
    repo workspace for local tool replay. The check happens before scaffold
    lookup so unrelated ``agents.*`` packages stay unloaded.
    """
    if metadata is None:
        return

    if metadata.get("needs_prepare") is False:
        raise NotImplementedError(
            "Simulate mode requires a prepare-able workspace trace; "
            "metadata.needs_prepare was false."
        )

    task_shape = metadata.get("task_shape")
    if task_shape not in (None, "swe_patch"):
        raise NotImplementedError(
            "Simulate mode only supports repo-backed swe_patch traces; "
            f"got task_shape={task_shape!r}."
        )

def _detect_agent_id(trace_path: Path) -> str:
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") == "action" and record.get("agent_id"):
                return record["agent_id"]
    raise SimulateError(f"No action records with agent_id found in {trace_path}")
