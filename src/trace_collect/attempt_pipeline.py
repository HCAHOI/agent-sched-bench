"""Orchestrator for one attempt_<N> collection run."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import subprocess
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from harness.container_image_prep import ensure_fixed_image
from harness.container_runtime import container_run_user_args
from harness.container_stats_sampler import (
    ContainerStatsSampler,
    summarize_samples,
)
from harness.process_stats_sampler import ProcessStatsSampler
from harness.disk_preflight import DiskSpaceError, preflight_disk
from trace_collect import attempt_layout

logger = logging.getLogger(__name__)

_ERROR_EXIT_STATUSES = frozenset(
    {
        "error",
        "tool_error",
        "empty_final_response",
        "timeout",
        "failed",
    }
)

_TASK_CONTAINER_ENV_PASSTHROUGH = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "PIP_INDEX_URL",
    "TASK_CONTAINER_PIP_INDEX_URL",
    "TASK_CONTAINER_PIP_EXTRA_INDEX_URL",
    "TASK_CONTAINER_PIP_TRUSTED_HOST",
    "TASK_CONTAINER_PIP_CERT",
    "TASK_CONTAINER_SSL_CERT_FILE",
    "TASK_CONTAINER_HTTP_PROXY",
    "TASK_CONTAINER_HTTPS_PROXY",
    "TASK_CONTAINER_ALL_PROXY",
    "TASK_CONTAINER_NO_PROXY",
    "TASK_CONTAINER_APT_MIRROR",
    "TASK_CONTAINER_APT_SECURITY_MIRROR",
    "NANOBOT_MAX_CONCURRENT_REQUESTS",
    # LLM client timeouts: slow provider streams trip the 90s idle/SDK default.
    "NANOBOT_STREAM_IDLE_TIMEOUT_S",
    "OPENCLAW_LLM_TIMEOUT_S",
)


def _is_missing_container_inspect_error(
    result: subprocess.CompletedProcess[str],
) -> bool:
    if result.returncode == 0:
        return False
    message = ((result.stderr or "") + "\n" + (result.stdout or "")).lower()
    return (
        "no such object" in message
        or "no such container" in message
        or "does not exist" in message
    )


def _is_container_removal_in_progress(message: str) -> bool:
    normalized = message.lower()
    return "removal of container" in normalized and "is already in progress" in normalized


def _inspect_container_exists(container_id: str, *, executable: str) -> tuple[bool, str | None]:
    try:
        inspect = subprocess.run(
            [executable, "inspect", container_id],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return True, f"{executable} inspect {container_id} timed out after 30s"
    except FileNotFoundError as exc:
        raise RuntimeError(f"container executable not found: {executable}") from exc
    if inspect.returncode == 0:
        return True, None
    if _is_missing_container_inspect_error(inspect):
        return False, None
    message = (inspect.stderr or inspect.stdout or "").strip()
    detail = (
        f"{executable} inspect {container_id} failed with exit {inspect.returncode}"
        + (f": {message}" if message else "")
    )
    return True, detail


_ATTEMPT_DIR_RE = re.compile(r"^attempt_(\d+)$")


def next_attempt_number_in(instance_dir: Path) -> int:
    """Return the next attempt_<N> index for a pre-joined instance dir."""
    if not instance_dir.exists():
        return 1
    max_attempt = 0
    for child in instance_dir.iterdir():
        if not child.is_dir():
            continue
        match = _ATTEMPT_DIR_RE.fullmatch(child.name)
        if match is None:
            continue
        max_attempt = max(max_attempt, int(match.group(1)))
    return max_attempt + 1


def sanitize_path_segment(value: str) -> str:
    """Replace path-hostile chars (/ and :) with '-' for one path segment."""
    return value.replace("/", "-").replace(":", "-")


def mcp_config_label(mcp_config: str | None) -> str | None:
    """Map ``--mcp-config`` to the value stored in trace metadata."""
    if mcp_config is None:
        return None
    if mcp_config == "none":
        return "none"
    return Path(mcp_config).name


@dataclass
class AttemptContext:
    """Shared per-task state between ``run_attempt`` and scaffold code."""

    run_dir: Path
    instance_id: str
    attempt: int
    task: dict[str, Any]
    model: str
    scaffold: str
    source_image: str | None
    prompt_template: str = "default"
    agent_runtime_mode: str = "host_controller"
    execution_environment: str = "container"
    fixed_image: str | None = None
    container_id: str | None = None
    attempt_dir: Path = field(init=False)
    start_time: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    end_time: datetime | None = None
    container_stdout: str = ""
    permission_fix_time_s: float = 0.0

    def __post_init__(self) -> None:
        self.attempt_dir = self.run_dir / self.instance_id / f"attempt_{self.attempt}"

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
    executable: str,
    extra_args: list[str] | None = None,
    network_mode: str = "host",
    run_as_host_user: bool = True,
    mount_host_home: bool = True,
    container_home: str | None = None,
) -> str:
    """Launch the task container and return its id."""
    home_dir = container_home or os.environ.get("HOME", "/root")
    cmd = [
        executable,
        "run",
        "-d",
        "--rm",
        f"--network={network_mode}",
        "-w",
        "/testbed",
    ]
    if mount_host_home:
        cmd.extend(
            [
                "-v",
                f"{home_dir}:{home_dir}",
            ]
        )
    cmd.extend(
        [
            "-e",
            f"HOME={home_dir}",
            "-e",
            f"PATH={home_dir}/.local/bin:/usr/local/bin:/usr/bin:/bin",
        ]
    )
    for env_name in _TASK_CONTAINER_ENV_PASSTHROUGH:
        value = os.environ.get(env_name)
        if value:
            cmd.extend(["-e", f"{env_name}={value}"])
    if run_as_host_user:
        cmd.extend(container_run_user_args(executable))
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


def _validate_apt_mirror_url(value: str, *, env_name: str) -> str:
    normalized = value.rstrip("/")
    parsed = urllib.parse.urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{env_name} must be an absolute http(s) URL")
    if any(ch.isspace() for ch in normalized):
        raise ValueError(f"{env_name} must not contain whitespace")
    return normalized


def configure_task_container_apt_mirror(
    container_id: str,
    *,
    executable: str,
) -> dict[str, str] | None:
    """Configure Debian/Ubuntu apt mirrors inside a running task container.

    This is opt-in via TASK_CONTAINER_APT_MIRROR. It is an infrastructure
    mirror, not benchmark-specific behavior; trace commands still execute as
    recorded, but apt resolves packages from the configured mirror.
    """
    main_mirror = os.environ.get("TASK_CONTAINER_APT_MIRROR")
    if not main_mirror:
        return None
    main_mirror = _validate_apt_mirror_url(
        main_mirror,
        env_name="TASK_CONTAINER_APT_MIRROR",
    )
    security_mirror_env = os.environ.get("TASK_CONTAINER_APT_SECURITY_MIRROR")
    if security_mirror_env:
        security_mirror_env = _validate_apt_mirror_url(
            security_mirror_env,
            env_name="TASK_CONTAINER_APT_SECURITY_MIRROR",
        )
    script = f"""
set -eu
main_mirror={shlex.quote(main_mirror)}
security_mirror_env={shlex.quote(security_mirror_env or "")}
. /etc/os-release
case "${{ID:-}}" in
  debian)
    components="main"
    signed_by="/usr/share/keyrings/debian-archive-keyring.gpg"
    ;;
  ubuntu)
    components="main restricted universe multiverse"
    signed_by="/usr/share/keyrings/ubuntu-archive-keyring.gpg"
    ;;
  *)
    echo "apt mirror skipped: unsupported distro: ${{ID:-unknown}}"
    exit 0
    ;;
esac
codename="${{VERSION_CODENAME:-}}"
if [ -z "$codename" ]; then
  echo "apt mirror unsupported ${{ID:-unknown}} image without VERSION_CODENAME" >&2
  exit 1
fi
if [ -n "$security_mirror_env" ]; then
  security_mirror="$security_mirror_env"
else
  case "${{ID:-}}" in
    debian)
      case "$main_mirror" in
        */debian) security_mirror="${{main_mirror%/debian}}/debian-security" ;;
        *) security_mirror="$main_mirror" ;;
      esac
      ;;
    ubuntu)
      security_mirror="$main_mirror"
      ;;
  esac
fi
mkdir -p /etc/apt/sources.list.d
for source_file in /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do
  [ -e "$source_file" ] || continue
  case "$source_file" in
    */agent-sched-bench-mirror.*|*.agent-sched-bench-disabled) continue ;;
  esac
  mv "$source_file" "$source_file.agent-sched-bench-disabled"
done
cat > /etc/apt/sources.list.d/agent-sched-bench-mirror.sources <<EOF
Types: deb
URIs: $main_mirror
Suites: $codename $codename-updates
Components: $components
Signed-By: $signed_by

Types: deb
URIs: $security_mirror
Suites: $codename-security
Components: $components
Signed-By: $signed_by
EOF
echo "apt mirror configured: distro=${{ID:-unknown}} main=$main_mirror security=$security_mirror"
"""
    result = subprocess.run(
        [executable, "exec", "-i", "--user", "0:0", container_id, "/bin/sh", "-s"],
        input=script,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "failed to configure task-container apt mirror: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    stdout = result.stdout.strip()
    security_match = re.search(r"\bsecurity=([^\s]+)", stdout)
    return {
        "configured": "false" if stdout.startswith("apt mirror skipped:") else "true",
        "main_mirror": main_mirror,
        "security_mirror": security_match.group(1) if security_match else "",
        "stdout": stdout,
    }


def stop_task_container(container_id: str, *, executable: str) -> str:
    """Capture container logs then stop and remove it. Returns log text."""
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

    errors: list[str] = []
    removal_in_progress = False
    for cmd, timeout in (
        ([executable, "stop", container_id], 30),
        ([executable, "rm", "-f", container_id], 60),
    ):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            errors.append(f"{' '.join(cmd)} timed out after {timeout}s")
            continue
        except FileNotFoundError as exc:
            raise RuntimeError(f"container executable not found: {executable}") from exc
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "").strip()
            if cmd[:3] == [executable, "rm", "-f"] and _is_container_removal_in_progress(
                message
            ):
                removal_in_progress = True
                continue
            errors.append(
                f"{' '.join(cmd)} failed with exit {result.returncode}"
                + (f": {message}" if message else "")
            )

    inspect_exists, inspect_error = _inspect_container_exists(
        container_id,
        executable=executable,
    )
    if inspect_error:
        errors.append(inspect_error)

    if inspect_exists and removal_in_progress:
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            time.sleep(0.5)
            inspect_exists, inspect_error = _inspect_container_exists(
                container_id,
                executable=executable,
            )
            if inspect_error:
                errors.append(inspect_error)
                break
            if not inspect_exists:
                break
        if inspect_exists:
            errors.append(
                f"{executable} rm -f {container_id} reported removal in progress, "
                "but the container still exists after 60s"
            )

    if inspect_exists:
        detail = "; ".join(errors) if errors else "container still exists after cleanup"
        raise RuntimeError(f"Failed to remove task container {container_id}: {detail}")
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
    *,
    container_executable: str,
) -> ContainerStatsSampler | None:
    """Wait for ``ctx.container_id`` and start sampling once it appears."""
    while not stop_event.is_set():
        if ctx.container_id:
            if _container_is_inspectable(
                ctx.container_id,
                container_executable=container_executable,
            ):
                sampler = ContainerStatsSampler(
                    container_id=ctx.container_id,
                    interval_s=1.0,
                    executable=container_executable,
                )
                sampler.start()
                return sampler
        try:
            await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            return None
    return None


def _container_is_inspectable(
    container_id: str,
    *,
    container_executable: str,
) -> bool:
    try:
        result = subprocess.run(
            [
                container_executable,
                "inspect",
                "--format",
                "{{.Id}}",
                container_id,
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


async def run_attempt(
    ctx: AttemptContext,
    *,
    inner: Callable[[AttemptContext], Awaitable[AttemptResult]],
    min_free_disk_gb: float = 30.0,
    container_executable: str | None,
) -> AttemptResult:
    """Execute one scaffold attempt and write its artifacts."""
    try:
        free_gb = preflight_disk(ctx.run_dir, min_free_disk_gb)
        logger.info("disk preflight ok: %.2f GB free at %s", free_gb, ctx.run_dir)
    except DiskSpaceError as exc:
        logger.error("disk preflight failed: %s", exc)
        raise

    attempt_layout.ensure_attempt_dir(ctx.attempt_dir)

    if ctx.source_image:
        if container_executable is None:
            raise ValueError("container_executable is required for container tasks")
        fixed_name, fix_elapsed = ensure_fixed_image(
            ctx.source_image,
            container_executable=container_executable,
        )
        ctx.fixed_image = fixed_name
        ctx.permission_fix_time_s = fix_elapsed
        logger.info(
            "image prep: source=%s fixed=%s elapsed=%.2fs",
            ctx.source_image,
            fixed_name,
            fix_elapsed,
        )
    else:
        ctx.fixed_image = None
        ctx.permission_fix_time_s = 0.0

    stop_watcher = threading.Event()
    watcher_task: asyncio.Task[ContainerStatsSampler | None] | None = None
    if container_executable is not None:
        watcher_task = asyncio.create_task(
            _watch_for_container_ready(
                ctx,
                stop_watcher,
                container_executable=container_executable,
            )
        )

    process_sampler: ProcessStatsSampler | None = None
    if ctx.execution_environment == "host":
        process_sampler = ProcessStatsSampler(pid=os.getpid(), interval_s=1.0)
        process_sampler.start()

    sampler: ContainerStatsSampler | ProcessStatsSampler | None = None
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
        if watcher_task is not None:
            try:
                sampler = await asyncio.wait_for(watcher_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                watcher_task.cancel()
                sampler = None
        if sampler is not None:
            samples = sampler.stop()
        if process_sampler is not None:
            process_samples = process_sampler.stop()
            if not samples:
                samples = process_samples
        ctx.end_time = datetime.now(tz=timezone.utc)

    status = "completed"
    if inner_error is not None:
        status = "error"
    elif result is not None and result.exit_status == "max_iterations":
        status = "exhausted"
    elif result is not None and (
        not result.success
        or (
            result.exit_status is not None
            and result.exit_status in _ERROR_EXIT_STATUSES
        )
    ):
        status = "error"
    success = bool(result.success) if result is not None else False
    task_payload: dict[str, Any] = {
        "instance_id": ctx.instance_id,
        "repo": ctx.task.get("repo"),
        "docker_image": ctx.source_image,
    }
    for key in ("task_source_kind", "task_source_id", "task_source_path"):
        if key in ctx.task:
            task_payload[key] = ctx.task.get(key)

    manifest = {
        "status": status,
        "task": task_payload,
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
            "fixed_image_name": ctx.fixed_image,
        },
        "result_summary": {
            "exit_code": 0 if success else 1,
            "exit_status": result.exit_status if result is not None else None,
            "error": str(inner_error)
            if inner_error is not None
            else (result.error if result is not None else None),
            "total_time": ctx.elapsed_seconds(),
            "active_time": (result.total_llm_ms or 0.0) / 1000.0 if result else 0.0,
            "tool_time": (result.total_tool_ms or 0.0) / 1000.0 if result else 0.0,
        },
        "scaffold": ctx.scaffold,
        "prompt_template": ctx.prompt_template,
        "agent_runtime_mode": ctx.agent_runtime_mode,
        "execution_environment": ctx.execution_environment,
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
        "repo": ctx.task.get("repo"),
        "docker_image": ctx.source_image,
        "success": success,
        "scaffold": ctx.scaffold,
        "prompt_template": ctx.prompt_template,
        "agent_runtime_mode": ctx.agent_runtime_mode,
    }
    for key in (
        "task_source_kind",
        "task_source_id",
        "task_source_path",
        "tb_version",
        "tb_dataset",
        "tb_registry_source",
        "adapter_kind",
        "agent_import_path",
    ):
        if key in ctx.task:
            results_payload[key] = ctx.task.get(key)
        elif result is not None and key in result.summary:
            results_payload[key] = result.summary.get(key)
    if result is not None and result.runtime_proof:
        results_payload["runtime_proof"] = result.runtime_proof
    if result is not None:
        results_payload["n_iterations"] = result.n_iterations
        results_payload["total_tokens"] = result.total_tokens
        if result.summary:
            results_payload["scaffold_summary"] = result.summary

    resources_summary = summarize_samples(samples)

    if result is not None and result.trace_path.exists():
        attempt_layout.copy_trace_jsonl(
            ctx.attempt_dir,
            result.trace_path,
        )

    trace_file = ctx.attempt_dir / attempt_layout.TRACE_FILENAME
    if result is not None and result.tool_calls:
        tool_calls = result.tool_calls
    elif trace_file.exists():
        tool_calls = attempt_layout.build_tool_calls_from_trace(trace_file)
    else:
        tool_calls = []

    openclaw_tool_results_dir = (
        ctx.attempt_dir / "openclaw-runtime" / "tool-results"
    )
    if openclaw_tool_results_dir.exists():
        manifest.setdefault("artifacts", {})["openclaw_tool_results_dir"] = str(
            openclaw_tool_results_dir.relative_to(ctx.attempt_dir)
        )

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
