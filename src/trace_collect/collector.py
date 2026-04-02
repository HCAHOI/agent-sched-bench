"""SWE-Bench trace collector using an external LLM API + Docker sandbox."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.code_agent import CodeAgent
from harness.trace_logger import TraceLogger
from trace_collect.docker_sandbox import DockerSandbox

logger = logging.getLogger(__name__)


def load_tasks(task_source: str | Path) -> list[dict[str, Any]]:
    """Load tasks from a JSON file."""
    path = Path(task_source)
    return json.loads(path.read_text(encoding="utf-8"))


def load_completed_ids(output_path: Path) -> set[str]:
    """Scan an existing JSONL trace file for already-completed agent IDs."""
    completed: set[str] = set()
    if not output_path.exists():
        return completed
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("type") == "summary":
                    completed.add(entry["agent_id"])
            except json.JSONDecodeError:
                continue
    return completed


def build_run_id(model: str) -> str:
    """Build a run ID from model name and current timestamp."""
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    safe_model = model.replace("/", "-").replace(":", "-")
    return f"{safe_model}_{ts}"


async def collect_traces(
    *,
    api_base: str,
    api_key: str,
    model: str,
    task_source: str | Path,
    repos_root: str | Path,
    output_dir: str | Path,
    base_image: str = "python:3.11-slim",
    max_steps: int = 40,
    command_timeout_s: float = 120.0,
    task_timeout_s: float = 1200.0,
    sample: int | None = None,
) -> Path:
    """Collect SWE-Bench traces using an external LLM API.

    Each task is run sequentially to respect API rate limits.
    Already-completed tasks (from a prior interrupted run) are skipped
    automatically (resume support).

    Args:
        api_base: OpenAI-compatible API base URL.
        api_key: API key for authentication.
        model: Model name (e.g. "qwen-plus-latest").
        task_source: Path to tasks JSON file.
        repos_root: Path to pre-cloned repos directory.
        output_dir: Directory for output trace files.
        base_image: Docker base image for sandboxes.
        max_steps: Maximum agent steps per task.
        command_timeout_s: Timeout per bash command.
        task_timeout_s: Timeout per task overall.
        sample: If set, only run the first N tasks.

    Returns:
        Path to the output JSONL file.
    """
    tasks = load_tasks(task_source)
    if sample is not None:
        tasks = tasks[:sample]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    run_id = build_run_id(model)
    trace_file = output_path / f"{run_id}.jsonl"

    completed = load_completed_ids(trace_file)
    if completed:
        logger.info("Resuming: %d tasks already completed", len(completed))

    sandbox = DockerSandbox(
        base_image=base_image,
        repos_root=Path(repos_root).resolve() if repos_root else None,
    )
    trace_logger = TraceLogger(output_path, run_id)

    total = len(tasks)
    succeeded = 0
    failed = 0

    try:
        for i, task in enumerate(tasks):
            instance_id = task["instance_id"]
            if instance_id in completed:
                logger.info("[%d/%d] SKIP %s (already completed)", i + 1, total, instance_id)
                continue

            logger.info("[%d/%d] START %s", i + 1, total, instance_id)
            t0 = time.monotonic()

            agent = CodeAgent(
                agent_id=instance_id,
                api_base=api_base,
                model=model,
                api_key=api_key,
                max_steps=max_steps,
                command_timeout_s=command_timeout_s,
                task_timeout_s=task_timeout_s,
                repos_root=str(repos_root),
            )
            # Inject DockerSandbox instead of the default LocalSandbox
            agent._container_mgr = sandbox

            try:
                prepare_t0 = time.monotonic()
                await agent.prepare(task)
                prepare_ms = (time.monotonic() - prepare_t0) * 1000

                success = await agent.run(task)
            except Exception:
                logger.exception("FAILED %s", instance_id)
                failed += 1
                continue

            elapsed = time.monotonic() - t0

            # Log trace with extra metadata
            for record in agent.trace:
                record.extra["model"] = model
                record.extra["api_provider"] = "dashscope"
                record.extra["prepare_ms"] = prepare_ms
                trace_logger.log_step(agent.agent_id, record)

            summary = agent.summary()
            summary["elapsed_s"] = elapsed
            summary["prepare_ms"] = prepare_ms
            trace_logger.log_summary(agent.agent_id, summary)

            if success:
                succeeded += 1
            else:
                failed += 1

            steps = len(agent.trace)
            logger.info(
                "[%d/%d] DONE %s success=%s steps=%d elapsed=%.1fs",
                i + 1, total, instance_id, success, steps, elapsed,
            )
    finally:
        trace_logger.close()

    logger.info(
        "Collection complete: %d/%d succeeded, %d failed, trace -> %s",
        succeeded, total, failed, trace_file,
    )
    return trace_file
