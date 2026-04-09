"""Async orchestrator for the CC-style attempt_<N>/ collection pipeline.

This module exposes ``AttemptContext`` and ``run_attempt``. Scaffolds build
an ``AttemptContext``, pass a thin ``inner`` coroutine (the actual agent run)
to ``run_attempt``, and the wrapper handles disk preflight, writable image
derivation, container resource sampling, log capture, and the six-file
attempt directory write-out.

``run_attempt`` is deliberately tolerant: if ``inner`` raises, the wrapper
still writes a manifest with ``status="error"`` so failed runs leave a
forensic record on disk.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
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
    """Shared per-task state between ``run_attempt`` and the scaffold ``inner``.

    The scaffold populates ``container_id`` via :meth:`mark_container_ready`
    the moment it has started the task container — the resource sampler uses
    that callback to know when to begin/stop polling.
    """

    run_dir: Path
    instance_id: str
    attempt: int
    task: dict[str, Any]
    model: str
    requested_model: str
    scaffold: str
    source_image: str
    prompt_template: str = "default"
    fixed_image: str | None = None
    container_id: str | None = None
    attempt_dir: Path = field(init=False)
    start_time: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    end_time: datetime | None = None
    claude_output: str = ""
    claude_stderr: str = ""
    pull_time_s: float = 0.0
    permission_fix_time_s: float = 0.0

    def __post_init__(self) -> None:
        self.attempt_dir = (
            self.run_dir / self.instance_id / f"attempt_{self.attempt}"
        )

    def mark_container_ready(self, container_id: str) -> None:
        """Called by the scaffold once ``podman run -d`` returned an id.

        The orchestrator watches ``self.container_id`` to decide when to
        start the background stats sampler. Storing the id here is the
        scaffold-to-orchestrator handshake.
        """
        self.container_id = container_id

    @property
    def attempt_label(self) -> str:
        """CC-compatible string like ``attempt_1`` (matches manifest field)."""
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
    """Launch a sleep-infinity container for the task and return its id.

    This is the shared "podman run -d" bootstrap that openclaw calls before
    starting its agent loop. mini-swe-agent does NOT use this — it lets
    ``DockerEnvironment`` own the container lifecycle. The container is
    started with ``--userns=keep-id`` + host ``$HOME`` mount so /testbed
    (chown'd by ``ensure_fixed_image``) is writable by the host uid.
    """
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
    """What the scaffold ``inner`` coroutine returns to ``run_attempt``.

    ``trace_path`` is the on-disk v5 trace.jsonl that the scaffold produced;
    ``run_attempt`` copies it into ``ctx.attempt_dir / "trace.jsonl"``. The
    other fields are scaffold-filled summary data that flows into
    ``results.json`` / ``run_manifest.json``.
    """

    success: bool
    exit_status: str | None
    trace_path: Path
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    n_steps: int | None = None
    total_llm_ms: float | None = None
    total_tool_ms: float | None = None
    total_tokens: int | None = None


async def _watch_for_container_ready(
    ctx: AttemptContext,
    stop_event: threading.Event,
) -> ContainerStatsSampler | None:
    """Poll ``ctx.container_id`` and spawn a sampler once it's populated.

    Runs as a background asyncio task alongside ``inner``. Returns the started
    sampler so the caller can stop it after inner returns.
    """
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
    """Execute one scaffold attempt end-to-end with CC-style artifacts.

    Steps:
      1. ``preflight_disk`` (raises ``DiskSpaceError`` on shortfall)
      2. ``ensure_fixed_image`` (no-op when source image is root-default)
      3. spawn background task that starts the stats sampler when the
         scaffold calls ``ctx.mark_container_ready``
      4. ``await inner(ctx)`` — the scaffold's agent loop
      5. stop sampler, collect samples
      6. write all six attempt files (trace.jsonl, run_manifest, results,
         resources, tool_calls, claude_output)

    On any exception from steps 1-4 the wrapper still writes a manifest with
    ``status="error"`` so failed runs leave a forensic record, then re-raises.
    """
    # Step 1: disk preflight
    try:
        free_gb = preflight_disk(ctx.run_dir, min_free_disk_gb)
        logger.info(
            "disk preflight ok: %.2f GB free at %s", free_gb, ctx.run_dir
        )
    except DiskSpaceError as exc:
        logger.error("disk preflight failed: %s", exc)
        raise

    attempt_layout.ensure_attempt_dir(ctx.attempt_dir)

    # Step 2: writable image derivative (probe-gated; usually a no-op for root images)
    try:
        fix_t0 = time.time()
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
        # Continue — scaffold can still try the source image.

    # Step 3: container-ready watcher (async) + sampler
    stop_watcher = threading.Event()
    watcher_task = asyncio.create_task(
        _watch_for_container_ready(ctx, stop_watcher)
    )

    sampler: ContainerStatsSampler | None = None
    samples: list[dict[str, Any]] = []
    result: AttemptResult | None = None
    inner_error: BaseException | None = None

    try:
        # Step 4: run the scaffold inner
        result = await inner(ctx)
    except BaseException as exc:
        inner_error = exc
        logger.exception("scaffold inner raised: %s", exc)
    finally:
        # Step 5: stop sampler + watcher
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

    # Step 6: write artifacts
    status = "error" if inner_error is not None else "completed"
    success = bool(result.success) if result is not None else False

    manifest = {
        "status": status,
        "characterization_only": False,
        "task": {
            "instance_id": ctx.instance_id,
            "repo": ctx.task.get("repo", ""),
            "docker_image": ctx.source_image,
        },
        "attempt": ctx.attempt_label,
        "model": {
            "requested": ctx.requested_model,
            "resolved": ctx.model,
            "claude_binary": None,
            "auth_mode": None,
            "auth_env_vars_present": [],
            "extra_env_keys": [],
        },
        "runtime": {
            "home": None,
            "wrapper_enabled": False,
            "memory_limit": None,
            "cpu_limit": None,
            "start_time": ctx.start_time_iso(),
            "end_time": ctx.end_time_iso(),
            "min_free_disk_gb": min_free_disk_gb,
        },
        "replay": {
            "replay_ready": bool(ctx.fixed_image),
            "source_image": ctx.source_image,
            "fixed_image_name": ctx.fixed_image or "",
            "tool_call_count": len(result.tool_calls) if result is not None else 0,
        },
        "result_summary": {
            "exit_code": 0 if success else 1,
            "error": str(inner_error) if inner_error is not None else (
                result.error if result is not None else None
            ),
            "total_time": ctx.elapsed_seconds(),
            "characterization_time": ctx.elapsed_seconds(),
            "active_time": (result.total_llm_ms or 0.0) / 1000.0 if result else 0.0,
            "tool_time": (result.total_tool_ms or 0.0) / 1000.0 if result else 0.0,
            "tool_ratio_active": 0.0,
        },
        "scaffold": ctx.scaffold,
        "prompt_template": ctx.prompt_template,
    }

    # results.json — more detailed timing breakdown (mirrors CC schema)
    results_payload: dict[str, Any] = {
        "image": ctx.source_image,
        "start_time": ctx.start_time_iso(),
        "end_time": ctx.end_time_iso(),
        "memory_limit": None,
        "cpu_limit": None,
        "model": ctx.model,
        "model_requested": ctx.requested_model,
        "characterization_only": False,
        "output_dir": str(ctx.attempt_dir),
        "claude_binary": None,
        "run_tests": ctx.prompt_template == "cc_aligned",
        "pull_time": ctx.pull_time_s,
        "permission_fix_time": ctx.permission_fix_time_s,
        "claude_time": ctx.elapsed_seconds(),
        "claude_output": {
            "stdout": ctx.claude_output,
            "stderr": ctx.claude_stderr,
            "exit_code": 0 if success else 1,
        },
        "resource_samples": {"samples": samples, "summary": {}},
        "total_time": ctx.elapsed_seconds(),
        "characterization_time": ctx.elapsed_seconds(),
        "active_time": manifest["result_summary"]["active_time"],
        "tool_time": manifest["result_summary"]["tool_time"],
        "replay_ready": bool(ctx.fixed_image),
        "instance_id": ctx.instance_id,
        "repo": ctx.task.get("repo", ""),
        "docker_image": ctx.source_image,
        "success": success,
        "scaffold": ctx.scaffold,
        "prompt_template": ctx.prompt_template,
    }
    if result is not None:
        results_payload["n_steps"] = result.n_steps
        results_payload["total_tokens"] = result.total_tokens
        if result.summary:
            results_payload["scaffold_summary"] = result.summary

    resources_summary = summarize_samples(samples)
    results_payload["resource_samples"] = {
        "samples": samples,
        "summary": resources_summary,
    }

    # trace.jsonl comes first — tool_calls.json is derived from it.
    if result is not None and result.trace_path.exists():
        attempt_layout.copy_trace_jsonl(ctx.attempt_dir, result.trace_path)

    trace_file = ctx.attempt_dir / attempt_layout.TRACE_FILENAME
    if result is not None and result.tool_calls:
        tool_calls = result.tool_calls
    elif trace_file.exists():
        tool_calls = attempt_layout.build_tool_calls_from_trace(trace_file)
    else:
        tool_calls = []

    # Update the manifest tool_call_count now that we know it.
    manifest["replay"]["tool_call_count"] = len(tool_calls)

    attempt_layout.write_run_manifest(ctx.attempt_dir, manifest)
    attempt_layout.write_results_json(ctx.attempt_dir, results_payload)
    attempt_layout.write_resources_json(
        ctx.attempt_dir, samples, summary=resources_summary
    )
    attempt_layout.write_tool_calls_json(ctx.attempt_dir, tool_calls)
    attempt_layout.write_claude_output(ctx.attempt_dir, ctx.claude_output)
    attempt_layout.write_claude_stderr(ctx.attempt_dir, ctx.claude_stderr)

    if inner_error is not None:
        raise inner_error
    assert result is not None
    return result


__all__ = ["AttemptContext", "AttemptResult", "run_attempt"]
