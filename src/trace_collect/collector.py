"""SWE-Bench trace collector using an external LLM API + Docker sandbox."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.mini_swe_code_agent import MiniSWECodeAgent
from harness.trace_logger import TraceLogger

logger = logging.getLogger(__name__)


def load_tasks(task_source: str | Path) -> list[dict[str, Any]]:
    """Load tasks from a JSON file."""
    path = Path(task_source)
    return json.loads(path.read_text(encoding="utf-8"))


def load_completed_ids(run_dir: Path) -> set[str]:
    """Scan a run directory for already-completed agent IDs.

    Each task has its own JSONL file: {run_dir}/{agent_id}.jsonl.
    A task is complete if its file contains a summary record.
    """
    completed: set[str] = set()
    if not run_dir.exists():
        return completed
    for trace_file in run_dir.glob("*.jsonl"):
        agent_id = trace_file.stem
        with open(trace_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("type") == "summary":
                        completed.add(agent_id)
                        break
                except json.JSONDecodeError:
                    continue
    return completed


def build_run_id(model: str, task_source: str | Path = "") -> str:
    """Build a run ID from task source, model name and timestamp."""
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    safe_model = model.replace("/", "-").replace(":", "-")
    task_name = Path(task_source).parent.name if task_source else "unknown"
    return f"{task_name}_{safe_model}_{ts}"


async def collect_traces(
    *,
    api_base: str,
    api_key: str,
    model: str,
    task_source: str | Path,
    repos_root: str | Path,
    output_dir: str | Path,
    max_steps: int = 60,
    command_timeout_s: float = 120.0,
    task_timeout_s: float = 1200.0,
    sample: int | None = None,
    run_id: str | None = None,
    max_context_tokens: int = 256_000,
) -> Path:
    """Collect SWE-Bench traces using an external LLM API.

    Each task produces its own JSONL trace file under {output_dir}/{run_id}/.
    Already-completed tasks (from a prior interrupted run) are skipped
    automatically (resume support).

    Args:
        api_base: OpenAI-compatible API base URL.
        api_key: API key for authentication.
        model: Model name (e.g. "qwen-plus-latest").
        task_source: Path to tasks JSON file.
        repos_root: Path to pre-cloned repos directory.
        output_dir: Directory for output trace files.
        max_steps: Maximum agent steps per task.
        command_timeout_s: Timeout per bash command.
        task_timeout_s: Timeout per task overall.
        sample: If set, only run the first N tasks.
        run_id: Explicit run ID for resuming an interrupted run. If None,
            a new timestamped ID is generated.
        max_context_tokens: Token budget for sliding window context management.

    Returns:
        Path to the run directory containing per-task JSONL files.
    """
    tasks = load_tasks(task_source)
    if sample is not None:
        tasks = tasks[:sample]

    output_path = Path(output_dir)
    if run_id is None:
        run_id = build_run_id(model, task_source)
    run_dir = output_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    completed = load_completed_ids(run_dir)
    if completed:
        logger.info("Resuming: %d tasks already completed", len(completed))

    total = len(tasks)
    succeeded = 0
    failed = 0

    for i, task in enumerate(tasks):
        instance_id = task["instance_id"]
        if instance_id in completed:
            logger.info("[%d/%d] SKIP %s (already completed)", i + 1, total, instance_id)
            continue

        logger.info("[%d/%d] START %s", i + 1, total, instance_id)
        t0 = time.monotonic()

        # One TraceLogger per task → one JSONL file per task
        trace_logger = TraceLogger(run_dir, instance_id)

        agent = MiniSWECodeAgent(
            agent_id=instance_id,
            api_base=api_base,
            model=model,
            api_key=api_key,
            max_steps=max_steps,
            command_timeout_s=command_timeout_s,
            task_timeout_s=task_timeout_s,
            repos_root=str(repos_root),
            max_context_tokens=max_context_tokens,
        )
        agent._trace_logger = trace_logger
        agent.run_metadata = {"model": model, "api_provider": "dashscope"}

        prepare_ms = 0.0
        prepare_t0 = time.monotonic()
        try:
            await agent.prepare(task)
            prepare_ms = (time.monotonic() - prepare_t0) * 1000
            agent.run_metadata["prepare_ms"] = prepare_ms

            success = await agent.run(task)
        except Exception as exc:
            logger.exception("FAILED %s", instance_id)
            failed += 1
            elapsed = time.monotonic() - t0
            if prepare_ms == 0.0:
                prepare_ms = (time.monotonic() - prepare_t0) * 1000
            error_summary = agent.summary()
            error_summary["elapsed_s"] = elapsed
            error_summary["prepare_ms"] = prepare_ms
            error_summary["error"] = str(exc)
            error_summary["error_type"] = type(exc).__name__
            trace_logger.log_summary(agent.agent_id, error_summary)
            trace_logger.close()
            continue

        elapsed = time.monotonic() - t0

        summary = agent.summary()
        summary["elapsed_s"] = elapsed
        summary["prepare_ms"] = prepare_ms
        trace_logger.log_summary(agent.agent_id, summary)
        trace_logger.close()

        if success:
            succeeded += 1
        else:
            failed += 1

        steps = len(agent.trace)
        logger.info(
            "[%d/%d] DONE %s success=%s steps=%d elapsed=%.1fs",
            i + 1, total, instance_id, success, steps, elapsed,
        )

    logger.info(
        "Collection complete: %d/%d succeeded, %d failed, traces -> %s",
        succeeded, total, failed, run_dir,
    )
    return run_dir
