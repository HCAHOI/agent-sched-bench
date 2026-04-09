"""Orchestrator for one attempt_<N> collection run."""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from harness.container_image_prep import ensure_fixed_image
from harness.container_stats_sampler import (
    ContainerStatsSampler,
    summarize_samples,
)
from harness.disk_preflight import DiskSpaceError, preflight_disk
from trace_collect import attempt_layout

logger = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "")


@dataclass
class AttemptContext:
    """Shared per-task state between ``run_attempt`` and scaffold code."""

    run_dir: Path
    instance_id: str
    attempt: int
    task: dict[str, Any]
    model: str
    scaffold: str
    source_image: str
    prompt_template: str = "default"
    agent_runtime_mode: str = "host_controller"
    fixed_image: str | None = None
    container_id: str | None = None
    attempt_dir: Path = field(init=False)
    start_time: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    end_time: datetime | None = None
    container_stdout: str = ""
    permission_fix_time_s: float = 0.0

    def __post_init__(self) -> None:
        self.attempt_dir = (
            self.run_dir / self.instance_id / f"attempt_{self.attempt}"
        )

    def mark_container_ready(self, container_id: str) -> None:
        """Publish the task container id so sampling can start."""
        self.container_id = container_id

    @property
    def attempt_label(self) -> str:
        """Stable string like ``attempt_1`` (matches the manifest field)."""
        return f"attempt_{self.attempt}"

    def elapsed_seconds(self) -> float:
        """Total wall clock between ``start_time`` and ``end_time`` (or now)."""
        end = self.end_time or datetime.now(tz=timezone.utc)
        return (end - self.start_time).total_seconds()

    def start_time_iso(self) -> str:
        return self.start_time.isoformat().replace("+00:00", "")

    def end_time_iso(self) -> str:
        end = self.end_time or datetime.now(tz=timezone.utc)
        return end.isoformat().replace("+00:00", "")


def start_task_container(
    fixed_image: str,
    *,
    executable: str = "podman",
    extra_args: list[str] | None = None,
) -> str:
    """Launch the task container and return its id."""
    import os
    import subprocess

    home_dir = os.environ.get("HOME", "/root")
    cmd = [
        executable,
        "run",
        "-d",
        "--rm",
        "--userns=keep-id",
        "--network=host",
        "-v",
        f"{home_dir}:{home_dir}",
        "-w",
        "/testbed",
        "-e",
        f"HOME={home_dir}",
        "-e",
        f"PATH={home_dir}/.local/bin:/usr/local/bin:/usr/bin:/bin",
    ]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend([fixed_image, "sleep", "infinity"])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to start task container for {fixed_image}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


def stop_task_container(
    container_id: str, *, executable: str = "podman"
) -> str:
    """Capture container logs then stop and remove it. Returns log text."""
    import subprocess

    logs_text = ""
    try:
        logs = subprocess.run(
            [executable, "logs", container_id],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        logs_text = (logs.stdout or "") + (logs.stderr or "")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    try:
        subprocess.run(
            [executable, "stop", container_id],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    try:
        subprocess.run(
            [executable, "rm", "-f", container_id],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return logs_text


@dataclass
class AttemptResult:
    """Result returned by the scaffold ``inner`` coroutine."""

    success: bool
    exit_status: str | None
    trace_path: Path
    model_patch: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    n_iterations: int | None = None
    total_llm_ms: float | None = None
    total_tool_ms: float | None = None
    total_tokens: int | None = None
    runtime_proof: dict[str, Any] = field(default_factory=dict)


async def _watch_for_container_ready(
    ctx: AttemptContext,
    stop_event: threading.Event,
) -> ContainerStatsSampler | None:
    """Wait for ``ctx.container_id`` and start sampling once it appears."""
    while not stop_event.is_set():
        if ctx.container_id:
            sampler = ContainerStatsSampler(
                container_id=ctx.container_id,
                interval_s=1.0,
            )
            sampler.start()
            return sampler
        try:
            await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            return None
    return None


async def run_attempt(
    ctx: AttemptContext,
    *,
    inner: Callable[[AttemptContext], Awaitable[AttemptResult]],
    min_free_disk_gb: float = 30.0,
    executable: str = "podman",
) -> AttemptResult:
    """Execute one scaffold attempt and write its artifacts."""
    try:
        free_gb = preflight_disk(ctx.run_dir, min_free_disk_gb)
        logger.info(
            "disk preflight ok: %.2f GB free at %s", free_gb, ctx.run_dir
        )
    except DiskSpaceError as exc:
        logger.error("disk preflight failed: %s", exc)
        raise

    attempt_layout.ensure_attempt_dir(ctx.attempt_dir)

    try:
        fixed_name, fix_elapsed = ensure_fixed_image(
            ctx.source_image, executable=executable
        )
        ctx.fixed_image = fixed_name
        ctx.permission_fix_time_s = fix_elapsed
        logger.info(
            "image prep: source=%s fixed=%s elapsed=%.2fs",
            ctx.source_image,
            fixed_name,
            fix_elapsed,
        )
    except Exception as exc:
        logger.error("image prep failed: %s", exc)
        ctx.fixed_image = ctx.source_image

    stop_watcher = threading.Event()
    watcher_task = asyncio.create_task(
        _watch_for_container_ready(ctx, stop_watcher)
    )

    sampler: ContainerStatsSampler | None = None
    samples: list[dict[str, Any]] = []
    result: AttemptResult | None = None
    inner_error: BaseException | None = None

    try:
        result = await inner(ctx)
    except BaseException as exc:
        inner_error = exc
        logger.exception("scaffold inner raised: %s", exc)
    finally:
        stop_watcher.set()
        try:
            sampler = await asyncio.wait_for(watcher_task, timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            watcher_task.cancel()
            sampler = None
        except Exception:
            sampler = None
        if sampler is not None:
            samples = sampler.stop()

        ctx.end_time = datetime.now(tz=timezone.utc)

    status = "error" if inner_error is not None else "completed"
    success = bool(result.success) if result is not None else False

    manifest = {
        "status": status,
        "task": {
            "instance_id": ctx.instance_id,
            "repo": ctx.task.get("repo", ""),
            "docker_image": ctx.source_image,
        },
        "attempt": ctx.attempt_label,
        "model": {"name": ctx.model},
        "runtime": {
            "home": None,
            "wrapper_enabled": False,
            "memory_limit": None,
            "cpu_limit": None,
            "start_time": ctx.start_time_iso(),
            "end_time": ctx.end_time_iso(),
            "min_free_disk_gb": min_free_disk_gb,
            "agent_runtime_mode": ctx.agent_runtime_mode,
            "runtime_proof": result.runtime_proof if result is not None else {},
        },
        "replay": {
            "replay_ready": bool(ctx.fixed_image),
            "source_image": ctx.source_image,
            "fixed_image_name": ctx.fixed_image or "",
        },
        "result_summary": {
            "exit_code": 0 if success else 1,
            "error": str(inner_error) if inner_error is not None else (
                result.error if result is not None else None
            ),
            "total_time": ctx.elapsed_seconds(),
            "active_time": (result.total_llm_ms or 0.0) / 1000.0 if result else 0.0,
            "tool_time": (result.total_tool_ms or 0.0) / 1000.0 if result else 0.0,
        },
        "scaffold": ctx.scaffold,
        "prompt_template": ctx.prompt_template,
        "agent_runtime_mode": ctx.agent_runtime_mode,
    }

    results_payload: dict[str, Any] = {
        "image": ctx.source_image,
        "start_time": ctx.start_time_iso(),
        "end_time": ctx.end_time_iso(),
        "memory_limit": None,
        "cpu_limit": None,
        "model": ctx.model,
        "output_dir": str(ctx.attempt_dir),
        "permission_fix_time": ctx.permission_fix_time_s,
        "total_time": ctx.elapsed_seconds(),
        "active_time": manifest["result_summary"]["active_time"],
        "tool_time": manifest["result_summary"]["tool_time"],
        "replay_ready": bool(ctx.fixed_image),
        "instance_id": ctx.instance_id,
        "repo": ctx.task.get("repo", ""),
        "docker_image": ctx.source_image,
        "success": success,
        "scaffold": ctx.scaffold,
        "prompt_template": ctx.prompt_template,
        "agent_runtime_mode": ctx.agent_runtime_mode,
    }
    if result is not None and result.runtime_proof:
        results_payload["runtime_proof"] = result.runtime_proof
    if result is not None:
        results_payload["n_iterations"] = result.n_iterations
        results_payload["total_tokens"] = result.total_tokens
        if result.summary:
            results_payload["scaffold_summary"] = result.summary

    resources_summary = summarize_samples(samples)

    if result is not None and result.trace_path.exists():
        attempt_layout.copy_trace_jsonl(ctx.attempt_dir, result.trace_path)

    trace_file = ctx.attempt_dir / attempt_layout.TRACE_FILENAME
    if result is not None and result.tool_calls:
        tool_calls = result.tool_calls
    elif trace_file.exists():
        tool_calls = attempt_layout.build_tool_calls_from_trace(trace_file)
    else:
        tool_calls = []

    attempt_layout.write_run_manifest(ctx.attempt_dir, manifest)
    attempt_layout.write_results_json(ctx.attempt_dir, results_payload)
    attempt_layout.write_resources_json(
        ctx.attempt_dir, samples, summary=resources_summary
    )
    attempt_layout.write_tool_calls_json(ctx.attempt_dir, tool_calls)
    attempt_layout.write_container_stdout(ctx.attempt_dir, ctx.container_stdout)

    if inner_error is not None:
        raise inner_error
    assert result is not None
    return result


__all__ = ["AttemptContext", "AttemptResult", "run_attempt"]
