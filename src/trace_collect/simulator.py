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

from agents.base import ActionRecord, StepRecord
from harness.trace_logger import TraceLogger
from trace_collect.openclaw_tools import execute_trace_tool

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


def load_trace_steps(
    trace_path: Path,
    agent_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Load step records and optional summary for one agent from a JSONL trace.

    Returns:
        (steps sorted by step_idx, summary or None)

    Raises:
        SimulateError: if agent_id not found or no step records exist.
    """
    steps: list[dict[str, Any]] = []
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
            if record.get("type") == "step":
                steps.append(record)
            elif record.get("type") == "summary":
                summary = record

    if not steps:
        raise SimulateError(f"No step records found for agent_id={agent_id!r}")

    steps.sort(key=lambda s: s["step_idx"])

    # Validate no gaps
    indices = [s["step_idx"] for s in steps]
    expected = list(range(len(indices)))
    if indices != expected:
        raise SimulateError(
            f"Step index gaps for {agent_id}: got {indices[:5]}... expected 0..{len(indices) - 1}"
        )

    return steps, summary


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
) -> Path:
    """Simulate an agent run using a source API trace with local model timing.

    For each step in the source trace:
    1. Feed the original messages_in to the local model (streaming).
    2. Force the model to generate completion_tokens tokens, measure TTFT/TPOT.
    3. Discard the local model's output.
    4. Execute the source trace's tool call for real.
    5. Record a StepRecord with API-trace decisions + local-model timing.

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
    from agents.miniswe import MiniSWECodeAgent

    # 1. Load source trace
    # Detect agent_id: use the first step record's agent_id
    first_agent_id = _detect_agent_id(source_trace)
    steps, summary = load_trace_steps(source_trace, first_agent_id)
    agent_id = first_agent_id

    # 2. Find task
    task = _find_task(task_source, agent_id)
    source_model = (summary or {}).get("model", "unknown")
    logger.info(
        "Simulating %s: %d steps from %s, local model=%s",
        agent_id,
        len(steps),
        source_model,
        model,
    )

    # 3. Prepare environment (clone repo)
    agent = MiniSWECodeAgent(
        agent_id=agent_id,
        api_base=api_base,
        model=model,
        api_key=api_key,
        command_timeout_s=command_timeout_s,
        task_timeout_s=task_timeout_s,
        repos_root=str(repos_root),
        max_context_tokens=max_context_tokens,
    )
    await agent.prepare(task)
    repo_dir = agent._workdir / "repo"  # type: ignore[union-attr]

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
        n_source_steps=len(steps),
    )

    # 5. Create streaming client
    client = AsyncOpenAI(
        base_url=api_base,
        api_key=api_key,
        timeout=180.0,
    )

    # 6. Step loop
    wall_start = time.time()
    total_steps = len(steps)
    succeeded_steps = 0
    failed_steps = 0

    try:
        for i, step in enumerate(steps):
            step_idx = step["step_idx"]
            messages_in = step.get("messages_in")
            n_tokens = step.get("completion_tokens", 1) or 1
            tool_name = step.get("tool_name")
            tool_args = step.get("tool_args", "{}")

            if not messages_in:
                logger.warning("Step %d: no messages_in, skipping LLM call", step_idx)
                continue

            ts_start = time.time()

            # 6a. Streaming call to local model (measure timing, discard output)
            try:
                ttft_ms, tpot_ms, llm_latency_ms = await _call_local_model_streaming(
                    client,
                    model,
                    messages_in,
                    n_tokens,
                )
            except Exception as exc:
                logger.error("Step %d: LLM call failed: %s", step_idx, exc)
                ttft_ms, tpot_ms, llm_latency_ms = 0.0, 0.0, 0.0
                failed_steps += 1

            ts_after_llm = time.time()

            # 6b. Emit action record (LLM decided, before tool execution)
            action = ActionRecord(
                step_idx=step_idx,
                program_id=agent_id,
                tool_name=tool_name,
                tool_args=tool_args,
                prompt_tokens=step.get("prompt_tokens", 0),
                completion_tokens=step.get("completion_tokens", 0),
                llm_latency_ms=llm_latency_ms,
                ttft_ms=ttft_ms,
                ts=ts_after_llm,
            )
            trace_logger.log_action(agent_id, action)

            # 6c. Execute tool call from source trace
            tool_result = ""
            tool_duration_ms = 0.0
            tool_success = True
            tool_ts_start: float | None = None
            tool_ts_end: float | None = None

            if tool_name:
                tool_ts_start = time.time()
                tool_result, tool_duration_ms, tool_success = await _exec_tool(
                    agent_id,
                    repo_dir,
                    tool_name,
                    tool_args,
                    command_timeout_s,
                    step.get("raw_response"),
                )
                tool_ts_end = time.time()

            ts_end = time.time()

            # 6d. Build and emit StepRecord
            record = StepRecord(
                step_idx=step_idx,
                phase=step.get("phase", "acting"),
                program_id=agent_id,
                prompt_tokens=step.get("prompt_tokens", 0),
                completion_tokens=step.get("completion_tokens", 0),
                llm_latency_ms=llm_latency_ms,
                llm_output=step.get("llm_output", ""),
                messages_in=messages_in,
                raw_response=step.get("raw_response", {}),
                tool_name=tool_name,
                tool_args=tool_args,
                tool_result=tool_result,
                tool_duration_ms=tool_duration_ms,
                tool_success=tool_success,
                tool_timeout=False,
                tool_ts_start=tool_ts_start,
                tool_ts_end=tool_ts_end,
                ts_start=ts_start,
                ts_end=ts_end,
                ttft_ms=ttft_ms,
                tpot_ms=tpot_ms,
                extra={
                    "simulate_source": str(source_trace),
                    "simulate_step_idx": step_idx,
                    "source_llm_latency_ms": step.get("llm_latency_ms"),
                },
            )
            trace_logger.log_step(agent_id, record)
            succeeded_steps += 1

            logger.info(
                "[%d/%d] step %d: ttft=%.1fms tpot=%.2fms llm=%.0fms tool=%.0fms",
                i + 1,
                total_steps,
                step_idx,
                ttft_ms,
                tpot_ms,
                llm_latency_ms,
                tool_duration_ms,
            )

    finally:
        wall_end = time.time()

        # 7. Summary
        simulate_summary: dict[str, Any] = {
            "agent_id": agent_id,
            "task_id": agent_id,
            "n_steps": succeeded_steps,
            "elapsed_s": wall_end - wall_start,
            "source_trace": str(source_trace),
            "source_model": source_model,
            "local_model": model,
            "local_api_base": api_base,
            "succeeded_steps": succeeded_steps,
            "failed_steps": failed_steps,
        }
        trace_logger.log_summary(agent_id, simulate_summary)
        trace_logger.close()

        # Cleanup workdir
        if agent._workdir:
            import shutil

            shutil.rmtree(agent._workdir, ignore_errors=True)

    trace_file = output_path / f"{run_id}.jsonl"
    logger.info(
        "Simulate complete: %s steps=%d/%d elapsed=%.1fs -> %s",
        agent_id,
        succeeded_steps,
        total_steps,
        wall_end - wall_start,
        trace_file,
    )
    return trace_file


def _detect_scaffold(trace_path: Path) -> str:
    """Detect the scaffold from a trace's trace_metadata record."""
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
                return record.get("scaffold", "unknown")
            # Only check the first few records
            if record.get("type") in ("step", "summary"):
                break
    return "unknown"


def _detect_agent_id(trace_path: Path) -> str:
    """Read the first step record to detect the agent_id."""
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") == "step" and record.get("agent_id"):
                return record["agent_id"]
    raise SimulateError(f"No step records with agent_id found in {trace_path}")
