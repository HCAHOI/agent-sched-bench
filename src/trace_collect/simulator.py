"""Simulate mode: replay API trace decisions with local model timing.

Uses an existing cloud-API trace as the decision blueprint (tool call
sequence), while measuring a local model's actual inference latency
(TTFT, TPOT) for each step.  Tool calls from the source trace are
executed for real so the repo ends up in the correct final state.

Usage:
    python -m trace_collect.cli simulate \
        --source-trace traces/swebench/qwen-plus/.../task.jsonl \
        --api-base http://localhost:8000/v1 \
        --model Qwen/Qwen2.5-Coder-7B-Instruct
"""

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


# ---------------------------------------------------------------------------
# Streaming LLM call with TTFT / TPOT measurement
# ---------------------------------------------------------------------------


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
        extra_body={"min_tokens": n_tokens},  # vLLM: force exactly n_tokens
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


# ---------------------------------------------------------------------------
# Tool execution (single step)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _build_simulate_run_id(model: str) -> str:
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    safe_model = model.replace("/", "-").replace(":", "-")
    return f"simulate_{safe_model}_{ts}"


def load_trace_actions(
    trace_path: Path,
    agent_id: str,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any] | None]:
    """Load v4 trace actions grouped by iteration, plus optional summary.

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
    """Look up a task by instance_id from the tasks JSON file."""
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
    """Simulate an agent run using a source API trace with local model timing.

    For each iteration in the source trace:
    1. Feed the original messages_in to the local model (streaming).
    2. Force the model to generate completion_tokens tokens, measure TTFT/TPOT.
    3. Discard the local model's output.
    4. Execute the source trace's tool call(s) for real.
    5. Record TraceAction records with API-trace decisions + local-model timing.

    Args:
        source_trace: Path to the source API trace JSONL.
        task_source: Path to tasks JSON (for prepare()).
        repos_root: Path to pre-cloned repos.
        output_dir: Output directory for the simulate trace.
        api_base: Local model API base URL (e.g. vLLM).
        api_key: API key for the local model.
        model: Local model name.
        command_timeout_s: Timeout per bash command.
        task_timeout_s: Timeout for the entire simulation.
        max_context_tokens: Token budget (unused in simulate, kept for API compat).

    Returns:
        Path to the simulate trace JSONL file.
    """
    # 1. Load source trace + detect scaffold from v5 metadata header.
    # Phase 6: refuse BFCL v4 traces with a descriptive NotImplementedError
    # BEFORE any scaffold registry lookup or agent.* import side effect.
    metadata = _load_trace_metadata(source_trace)
    _refuse_bfcl_v4_simulate(metadata)

    first_agent_id = _detect_agent_id(source_trace)
    iterations, summary = load_trace_actions(source_trace, first_agent_id)
    agent_id = first_agent_id
    scaffold = _detect_scaffold(source_trace)

    # 2. Find task
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

    # 3. Prepare environment via the scaffold registry. The lookup
    # raises NotImplementedError with a descriptive message for
    # unsupported scaffolds (e.g. openclaw before Phase 1.5.1 lands).
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

    # 3a. Construct metrics client. When metrics_url is None the client
    # returns empty PreemptionSnapshots (all-None fields) on every call
    # — this is the explicit opt-out path. Real vLLM smoke is deferred
    # to US-010 manual runbook; the absent-URL path is what runs in
    # local Ralph iterations.
    metrics_client = VLLMMetricsClient(metrics_url=metrics_url)
    logger.info(
        "vLLM metrics client: %s",
        f"enabled (url={metrics_url})" if metrics_client.is_enabled else "disabled",
    )

    # 4. Set up trace logger
    run_id = _build_simulate_run_id(model)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trace_logger = TraceLogger(output_path, run_id)

    # Write trace metadata header (scaffold from source trace)
    trace_logger.log_metadata(
        scaffold=_detect_scaffold(source_trace),
        mode="simulate",
        source_trace=str(source_trace),
        source_model=source_model,
        local_model=model,
        local_api_base=api_base,
        n_source_iterations=len(iterations),
    )

    # 5. Create streaming client
    client = AsyncOpenAI(
        base_url=api_base,
        api_key=api_key,
        timeout=180.0,
    )

    # 6. Iteration loop (action-based replay)
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

            # 6a. Streaming call to local model (measure timing, discard output)
            try:
                ttft_ms, tpot_ms, llm_latency_ms = await _call_local_model_streaming(
                    client, model, messages_in, n_tokens,
                )
            except Exception as exc:
                logger.error("Iteration %d: LLM call failed: %s", it_num, exc)
                ttft_ms, tpot_ms, llm_latency_ms = 0.0, 0.0, 0.0
                failed_iters += 1

            ts_after_llm = time.time()

            # 6a-bis. Snapshot vLLM scheduler state. Phase 2 of the
            # trace-sim-vastai-pipeline plan: store ABSOLUTE field values
            # field-for-field under sim_metrics.vllm_scheduler_snapshot.
            # Deltas are computed post-hoc in src/analysis/sim_metrics_delta.py.
            scheduler_snapshot = metrics_client.get_snapshot()

            # 6b. Emit llm_call TraceAction. The legacy top-level
            # ttft_ms / tpot_ms / llm_latency_ms fields are preserved for
            # backward compatibility with pre-Phase-2 readers; the new
            # nested sim_metrics blob is the canonical location for
            # Phase-2-aware analysis (Gantt tooltips, sim_metrics_delta).
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
                        # PreemptionSnapshot uses @dataclass(slots=True) so
                        # instances have no __dict__; use dataclasses.asdict
                        # to serialize all fields field-for-field.
                        "vllm_scheduler_snapshot": dataclasses.asdict(
                            scheduler_snapshot
                        ),
                        # Phase 1.5.1 warmup tagging: position-index-based
                        # (NOT iteration-number-based) so sparse iteration
                        # numbers in source traces don't break the cutoff.
                        "warmup": i < warmup_skip_iterations,
                    },
                },
            )
            trace_logger.log_trace_action(agent_id, llm_record)

            # 6c. Execute tool calls from source trace.
            # Phase 1.5.1: MCP tools (tool_name starts with "mcp_") are
            # REPLAYED FROM RECORDED RESULTS — never re-dispatched to a
            # live MCP server (Pre-mortem C item 2 of trace-sim-vastai-
            # pipeline plan: zero context7 egress during replay).
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
                    # Reuse the recorded MCP result instead of re-dispatching.
                    # Provenance is recorded under sim_metrics.source so
                    # downstream analysis can distinguish replayed vs live.
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

        # 7. Summary
        simulate_summary: dict[str, Any] = {
            "agent_id": agent_id,
            "task_id": agent_id,
            "n_steps": succeeded_iters,
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

        # Cleanup workdir via the scaffold's PreparedWorkspace.cleanup
        # closure. The closure captures the scaffold-specific state
        # (e.g. mini-swe agent._workdir) and owns the rmtree call.
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
    """Detect the scaffold from a trace's trace_metadata record."""
    metadata = _load_trace_metadata(trace_path)
    return metadata.get("scaffold", "unknown") if metadata else "unknown"


def _load_trace_metadata(trace_path: Path) -> dict[str, Any] | None:
    """Read the first trace_metadata record from a v5 trace JSONL.

    Returns the parsed dict, or None if no metadata record is present
    in the first few non-empty lines (the metadata is conventionally
    line 0 of a v5 trace).
    """
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
            # Only check the first few records before giving up
            if record.get("type") in ("step", "summary", "action"):
                break
    return None


def _refuse_bfcl_v4_simulate(metadata: dict[str, Any] | None) -> None:
    """Phase 6: refuse BFCL v4 traces in the simulator with a descriptive error.

    BFCL v4 has ``task_shape='function_call'`` and ``needs_prepare=False``
    — there is no repo to clone, no workspace to prepare, and the
    function-call dispatch shape is fundamentally incompatible with the
    simulator's iterate-then-replay model. Mirrors the existing refusal
    pattern at ``src/agents/benchmarks/bfcl_v4.py:294-301``.

    Triggered by ANY of:
    - ``metadata.task_shape == "function_call"``
    - ``metadata.benchmark`` is one of the BFCL slugs (``"bfcl-v4"``,
      ``"bfcl_v4"``)
    - ``metadata.needs_prepare`` is explicitly False (forward-compat
      hook for any future task type with no prepare phase)

    The check happens BEFORE scaffold registry lookup so that NO
    ``agents.*`` package gets imported as a side effect of the refusal.
    """
    if metadata is None:
        return

    is_bfcl = (
        metadata.get("task_shape") == "function_call"
        or metadata.get("benchmark") in ("bfcl-v4", "bfcl_v4")
        or metadata.get("needs_prepare") is False
    )

    if is_bfcl:
        raise NotImplementedError(
            "BFCL v4 traces have task_shape='function_call' with "
            "needs_prepare=False, which the simulator does not support. "
            "Simulate mode requires a prepare-able scaffold."
        )


def _detect_agent_id(trace_path: Path) -> str:
    """Read the first action record to detect the agent_id."""
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
