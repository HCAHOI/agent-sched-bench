from __future__ import annotations

import asyncio
import importlib.metadata
import importlib.util
import json
import math
import os
import signal
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from agents.openclaw.tools.shell import MAX_EXEC_TOOL_TIMEOUT_SEC
from trace_collect.attempt_pipeline import (
    AttemptContext,
    AttemptResult,
    stop_task_container,
)

_TOOL_STALL_TIMEOUT_MULTIPLIER = 1.2
_LLM_STALL_TIMEOUT_GRACE_SEC = 60.0


def _optional_positive_float(value: Any, *, name: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive float, got {value!r}") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be a finite positive float, got {value!r}")
    return parsed


def _required_positive_float(mapping: dict[str, Any], key: str, *, name: str) -> float:
    if key not in mapping:
        raise ValueError(f"{name}.{key} is required when {name} is configured")
    parsed = _optional_positive_float(mapping[key], name=f"{name}.{key}")
    assert parsed is not None
    return parsed


def _parse_progress_watchdog(extras: dict[str, Any]) -> dict[str, float] | None:
    raw = extras.get("progress_watchdog")
    if raw is None or raw is False:
        return None
    if not isinstance(raw, dict):
        raise ValueError("benchmark_extras.progress_watchdog must be a mapping")
    if raw.get("enabled", True) is False:
        return None
    name = "benchmark_extras.progress_watchdog"
    config = {
        "poll_sec": _required_positive_float(raw, "poll_sec", name=name),
        "no_progress_timeout_sec": _required_positive_float(
            raw, "no_progress_timeout_sec", name=name
        ),
        "tool_stall_sec": _required_positive_float(
            raw, "tool_stall_sec", name=name
        ),
        "llm_stall_sec": _required_positive_float(raw, "llm_stall_sec", name=name),
    }
    min_tool_stall_sec = MAX_EXEC_TOOL_TIMEOUT_SEC * _TOOL_STALL_TIMEOUT_MULTIPLIER
    if config["tool_stall_sec"] < min_tool_stall_sec:
        raise ValueError(
            f"{name}.tool_stall_sec must be at least "
            f"{_TOOL_STALL_TIMEOUT_MULTIPLIER}x the max tool timeout "
            f"({min_tool_stall_sec:.1f}s), got {config['tool_stall_sec']!r}"
        )
    return config


def _watchdog_failure(
    *,
    exit_status: str,
    idle_seconds: float,
    last_event: str | None,
    trace_path: Path | None,
) -> dict[str, Any]:
    event_label = last_event or "unknown"
    return {
        "exit_status": exit_status,
        "error": (
            f"no Terminal-Bench trace/recording progress for "
            f"{idle_seconds:.1f}s; last_event={event_label}"
        ),
        "idle_seconds": round(idle_seconds, 1),
        "last_event": last_event,
        "trace_path": str(trace_path) if trace_path else None,
    }


def _safe_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _last_trace_event(trace_path: Path) -> str | None:
    last_event: str | None = None
    try:
        lines = trace_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        event = record.get("event")
        if record.get("type") == "event" and isinstance(event, str):
            last_event = event
    return last_event


class TerminalBenchRunner:
    AGENT_IMPORT_PATH = "agents.terminal_bench.openclaw_agent:TerminalBenchOpenClawAgent"
    TRACE_FILENAME = "openclaw-trace.jsonl"
    N_ATTEMPTS = 1

    def __init__(
        self,
        *,
        provider_name: str | None,
        env_key: str | None,
        api_base: str,
        api_key: str,
        model: str,
        workspace_base: Path,
        max_iterations: int,
        context_window_tokens: int,
        benchmark_slug: str,
        benchmark_extras: dict[str, Any],
        mcp_config: str | None = None,
    ) -> None:
        self.provider_name = provider_name or "openai"
        self.env_key = env_key or "OPENAI_API_KEY"
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.workspace_base = Path(workspace_base)
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens
        self.benchmark_slug = benchmark_slug
        self.benchmark_extras = dict(benchmark_extras)
        self.global_agent_timeout_sec = _optional_positive_float(
            self.benchmark_extras.get("global_agent_timeout_sec"),
            name="benchmark_extras.global_agent_timeout_sec",
        )
        self.llm_timeout_sec = _optional_positive_float(
            self.benchmark_extras.get("llm_timeout_sec"),
            name="benchmark_extras.llm_timeout_sec",
        )
        self.progress_watchdog = _parse_progress_watchdog(self.benchmark_extras)
        if (
            self.llm_timeout_sec is not None
            and self.progress_watchdog is not None
            and self.progress_watchdog["llm_stall_sec"]
            <= self.llm_timeout_sec + _LLM_STALL_TIMEOUT_GRACE_SEC
        ):
            raise ValueError(
                "benchmark_extras.progress_watchdog.llm_stall_sec must be at least "
                f"{_LLM_STALL_TIMEOUT_GRACE_SEC:.0f}s greater than "
                "benchmark_extras.llm_timeout_sec"
            )
        self.mcp_config = self._resolve_mcp_config(mcp_config)
        self.mcp_config_label = self._mcp_config_label(mcp_config)

    async def run_openclaw_task(
        self,
        task: dict[str, Any],
        *,
        attempt_ctx: AttemptContext,
        prompt_template: str,
    ) -> AttemptResult:
        return await asyncio.to_thread(
            self._run_openclaw_task_sync,
            task,
            attempt_ctx,
            prompt_template,
        )

    async def run_task(
        self,
        task: dict[str, Any],
        *,
        attempt_ctx: AttemptContext,
        prompt_template: str,
    ) -> AttemptResult:
        return await self.run_openclaw_task(
            task,
            attempt_ctx=attempt_ctx,
            prompt_template=prompt_template,
        )

    def _run_openclaw_task_sync(
        self,
        task: dict[str, Any],
        attempt_ctx: AttemptContext,
        prompt_template: str,
    ) -> AttemptResult:
        proof = self._preflight()
        run_root = attempt_ctx.attempt_dir / "_terminal_bench_run"
        run_id = attempt_ctx.instance_id.replace("/", "_")
        run_root.mkdir(parents=True, exist_ok=True)

        attempt_ctx.mark_container_ready(
            self._expected_client_container_name(
                task_id=str(task["task_id"]),
                run_id=run_id,
            )
        )
        command = self._build_tb_command(
            task=task,
            run_root=run_root,
            run_id=run_id,
            prompt_template=prompt_template,
        )
        env = os.environ.copy()
        repo_root = Path(__file__).resolve().parents[3]
        pythonpath = f"{repo_root / 'src'}:{repo_root}:{env.get('PYTHONPATH', '')}"
        env["PYTHONPATH"] = pythonpath.rstrip(":")
        env[self.env_key] = self.api_key
        tb_run_path = run_root / run_id
        completed = self._run_tb_process(
            command=command,
            cwd=repo_root,
            env=env,
            watchdog=self.progress_watchdog,
            tb_run_path=tb_run_path,
            recordings_dir=attempt_ctx.attempt_dir / "recordings",
            stdout_path=run_root / "tb-run-stdout.txt",
            stderr_path=run_root / "tb-run-stderr.txt",
        )
        tb_process_logs = self._write_tb_process_logs(
            run_root=run_root,
            result=completed,
        )
        if completed["stalled"] is not None:
            stalled_container_cleanup = self._cleanup_stalled_container(
                container_id=attempt_ctx.container_id,
                container_executable=proof["docker_path"],
                run_root=run_root,
            )
            return self._stalled_result(
                failure=completed["stalled"],
                attempt_ctx=attempt_ctx,
                task=task,
                prompt_template=prompt_template,
                tb_version=proof["tb_version"],
                tb_run_path=tb_run_path,
                runtime_proof=proof,
                tb_process_logs=tb_process_logs,
                stalled_container_cleanup=stalled_container_cleanup,
            )
        if completed["returncode"] != 0:
            raise RuntimeError(
                "terminal-bench run failed: "
                f"{completed['stderr'].strip() or completed['stdout'].strip()}"
            )

        success = self._extract_success(tb_run_path)
        trace_path = self._find_trace_path(tb_run_path)
        summary = self._summary(
            tb_version=proof["tb_version"],
            task=task,
            tb_run_path=tb_run_path,
            tb_process_logs=tb_process_logs,
        )
        if not trace_path.exists():
            raise RuntimeError(f"terminal-bench trace file not found under {tb_run_path}")
        normalized_trace = run_root / f"{attempt_ctx.instance_id}-terminal-bench-trace.jsonl"
        self._augment_trace_metadata(
            src=trace_path,
            dst=normalized_trace,
            task=task,
            prompt_template=prompt_template,
            tb_version=proof["tb_version"],
        )
        return AttemptResult(
            success=success,
            exit_status="completed" if success else "failed",
            trace_path=normalized_trace,
            model_patch="",
            error=None if success else "Terminal-Bench task did not resolve",
            summary=summary,
            runtime_proof=proof,
        )

    def _run_tb_process(
        self,
        *,
        command: list[str],
        cwd: Path,
        env: dict[str, str],
        watchdog: dict[str, float] | None,
        tb_run_path: Path,
        recordings_dir: Path,
        stdout_path: Path,
        stderr_path: Path,
    ) -> dict[str, Any]:
        if watchdog is None:
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            return {
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "stalled": None,
            }

        stalled: dict[str, Any] | None = None
        watch_started_at = time.monotonic()
        last_progress_mtime: float | None = None
        last_progress_seen_at = watch_started_at
        last_event: str | None = None
        with (
            stdout_path.open("w", encoding="utf-8") as stdout_handle,
            stderr_path.open("w", encoding="utf-8") as stderr_handle,
        ):
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                start_new_session=True,
            )
            while process.poll() is None:
                now = time.monotonic()
                latest_mtime, trace_path, trace_event = self._progress_snapshot(
                    tb_run_path=tb_run_path,
                    recordings_dir=recordings_dir,
                )
                stalled = self._stall_reason(
                    watchdog=watchdog,
                    now=now,
                    watch_started_at=watch_started_at,
                    latest_mtime=latest_mtime,
                    last_progress_mtime=last_progress_mtime,
                    last_progress_seen_at=last_progress_seen_at,
                    last_event=trace_event or last_event,
                    trace_path=trace_path,
                )
                if stalled is not None:
                    if process.poll() is not None:
                        stalled = None
                        break
                    self._signal_process_group(process, signal.SIGTERM)
                    break
                if (
                    latest_mtime is not None
                    and (
                        last_progress_mtime is None
                        or latest_mtime > last_progress_mtime
                    )
                ):
                    last_progress_mtime = latest_mtime
                    last_progress_seen_at = now
                    last_event = trace_event or last_event
                time.sleep(watchdog["poll_sec"])
            try:
                process.wait(timeout=30 if stalled else None)
            except subprocess.TimeoutExpired:
                self._signal_process_group(process, signal.SIGKILL)
                process.wait()
        return {
            "returncode": process.returncode if process.returncode is not None else -1,
            "stdout": stdout_path.read_text(encoding="utf-8"),
            "stderr": stderr_path.read_text(encoding="utf-8"),
            "stalled": stalled,
        }

    def _progress_snapshot(
        self,
        *,
        tb_run_path: Path,
        recordings_dir: Path,
    ) -> tuple[float | None, Path | None, str | None]:
        trace_path = self._find_existing_trace_path(tb_run_path)
        trace_mtime = _safe_mtime(trace_path) if trace_path else None
        recording_mtime = self._latest_recording_mtime(recordings_dir)
        mtimes = [mtime for mtime in (trace_mtime, recording_mtime) if mtime is not None]
        return (
            max(mtimes) if mtimes else None,
            trace_path,
            _last_trace_event(trace_path) if trace_path else None,
        )

    def _stall_reason(
        self,
        *,
        watchdog: dict[str, float],
        now: float,
        watch_started_at: float,
        latest_mtime: float | None,
        last_progress_mtime: float | None,
        last_progress_seen_at: float,
        last_event: str | None,
        trace_path: Path | None,
    ) -> dict[str, Any] | None:
        if latest_mtime is None:
            idle_seconds = now - watch_started_at
            return (
                _watchdog_failure(
                    exit_status="stalled_no_progress",
                    idle_seconds=idle_seconds,
                    last_event=None,
                    trace_path=None,
                )
                if idle_seconds >= watchdog["no_progress_timeout_sec"]
                else None
            )
        if last_progress_mtime is None or latest_mtime > last_progress_mtime:
            return None

        idle_seconds = now - last_progress_seen_at
        if last_event == "tool_exec_start":
            threshold = watchdog["tool_stall_sec"]
            exit_status = "stalled_tool_completion"
        elif last_event == "llm_call_start":
            threshold = watchdog["llm_stall_sec"]
            exit_status = "stalled_llm_generation"
        else:
            threshold = watchdog["no_progress_timeout_sec"]
            exit_status = "stalled_no_progress"
        if idle_seconds < threshold:
            return None
        return _watchdog_failure(
            exit_status=exit_status,
            idle_seconds=idle_seconds,
            last_event=last_event,
            trace_path=trace_path,
        )

    @staticmethod
    def _signal_process_group(process: subprocess.Popen[str], sig: int) -> None:
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        except PermissionError:
            if sig == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()

    @staticmethod
    def _write_tb_process_logs(
        *,
        run_root: Path,
        result: dict[str, Any],
    ) -> dict[str, str]:
        stdout_path = run_root / "tb-run-stdout.txt"
        stderr_path = run_root / "tb-run-stderr.txt"
        stdout_path.write_text(result["stdout"], encoding="utf-8")
        stderr_path.write_text(result["stderr"], encoding="utf-8")
        return {
            "tb_stdout_path": str(stdout_path),
            "tb_stderr_path": str(stderr_path),
        }

    def _cleanup_stalled_container(
        self,
        *,
        container_id: str | None,
        container_executable: str,
        run_root: Path,
    ) -> dict[str, Any]:
        if not container_id:
            return {"container_cleanup_confirmed": False, "container_id": None}
        logs_text = stop_task_container(container_id, executable=container_executable)
        logs_path = run_root / "stalled-container-logs.txt"
        logs_path.write_text(logs_text, encoding="utf-8")
        return {
            "container_cleanup_confirmed": not self._container_exists(
                container_id=container_id,
                container_executable=container_executable,
            ),
            "container_id": container_id,
            "container_logs_path": str(logs_path),
        }

    @staticmethod
    def _container_exists(
        *,
        container_id: str,
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
                timeout=10,
                check=False,
            )
        except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired):
            return True
        return result.returncode == 0

    def _stalled_result(
        self,
        *,
        failure: dict[str, Any],
        attempt_ctx: AttemptContext,
        task: dict[str, Any],
        prompt_template: str,
        tb_version: str,
        tb_run_path: Path,
        runtime_proof: dict[str, Any],
        tb_process_logs: dict[str, str],
        stalled_container_cleanup: dict[str, Any],
    ) -> AttemptResult:
        normalized_trace = (
            attempt_ctx.attempt_dir
            / "_terminal_bench_run"
            / f"{attempt_ctx.instance_id}-terminal-bench-trace.jsonl"
        )
        trace_path = (
            Path(failure["trace_path"])
            if failure.get("trace_path")
            else self._find_trace_path(tb_run_path)
        )
        if trace_path.exists():
            self._augment_trace_metadata(
                src=trace_path,
                dst=normalized_trace,
                task=task,
                prompt_template=prompt_template,
                tb_version=tb_version,
                watchdog_failure=failure,
            )
        else:
            self._write_minimal_trace_metadata(
                dst=normalized_trace,
                task=task,
                prompt_template=prompt_template,
                tb_version=tb_version,
                watchdog_failure=failure,
            )

        summary = self._summary(
            tb_version=tb_version,
            task=task,
            tb_run_path=tb_run_path,
            tb_process_logs=tb_process_logs,
        )
        summary["watchdog_failure"] = failure
        summary["stalled_container_cleanup"] = stalled_container_cleanup
        return AttemptResult(
            success=False,
            exit_status=failure["exit_status"],
            trace_path=normalized_trace,
            model_patch="",
            error=failure["error"],
            summary=summary,
            runtime_proof=runtime_proof,
        )

    def _preflight(self) -> dict[str, Any]:
        if importlib.util.find_spec("terminal_bench") is None:
            raise RuntimeError("terminal-bench Python package is not installed")
        tb_bin = shutil.which("tb")
        if not tb_bin:
            raise RuntimeError("tb CLI is not available on PATH")
        docker_bin = shutil.which("docker")
        if not docker_bin:
            raise RuntimeError("docker is not available on PATH")
        return {
            "tb_version": importlib.metadata.version("terminal-bench"),
            "tb_path": tb_bin,
            "docker_path": docker_bin,
            "agent_runtime_mode": "host_controller",
        }

    @classmethod
    def _expected_client_container_name(cls, *, task_id: str, run_id: str) -> str:
        trial_name = f"{task_id}.1-of-{cls.N_ATTEMPTS}.{run_id}"
        return trial_name.replace(".", "-")

    def _build_tb_command(
        self,
        *,
        task: dict[str, Any],
        run_root: Path,
        run_id: str,
        prompt_template: str,
    ) -> list[str]:
        dataset_root = task["dataset_root"]
        task_id = task["task_id"]
        command = [
            "tb",
            "run",
            "--dataset-path",
            str(dataset_root),
            "--task-id",
            str(task_id),
            "--output-path",
            str(run_root),
            "--run-id",
            run_id,
            "--n-concurrent",
            "1",
            "--n-attempts",
            str(self.N_ATTEMPTS),
            "--agent-import-path",
            self.AGENT_IMPORT_PATH,
            "--model",
            self.model,
            "--agent-kwarg",
            f"provider_name={self.provider_name}",
            "--agent-kwarg",
            f"api_base={self.api_base}",
            "--agent-kwarg",
            f"env_key={self.env_key}",
            "--agent-kwarg",
            f"max_iterations={self.max_iterations}",
        ]
        if self.llm_timeout_sec is not None:
            command.extend(
                [
                    "--agent-kwarg",
                    f"llm_timeout_sec={self.llm_timeout_sec}",
                ]
            )
        if self.global_agent_timeout_sec is not None:
            command.extend(
                [
                    "--global-agent-timeout-sec",
                    str(self.global_agent_timeout_sec),
                ]
            )
        prompt_template_path = self._materialize_prompt_template(
            prompt_template=prompt_template,
            run_root=run_root,
        )
        command.extend(
            [
                "--agent-kwarg",
                f"prompt_template={prompt_template_path}",
            ]
        )
        if self.mcp_config is not None:
            command.extend(
                [
                    "--agent-kwarg",
                    f"mcp_config_path={self.mcp_config}",
                ]
            )
        return command

    def _extract_success(self, tb_run_path: Path) -> bool:
        results_path = tb_run_path / "results.json"
        if not results_path.exists():
            raise RuntimeError(f"terminal-bench results.json missing at {results_path}")
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        results = payload.get("results") or []
        if not results:
            return False
        first = results[0]
        return bool(first.get("is_resolved"))

    def _find_trace_path(self, tb_run_path: Path) -> Path:
        trace_path = self._find_existing_trace_path(tb_run_path)
        if trace_path is not None:
            return trace_path
        return tb_run_path / "missing-trace.jsonl"

    def _find_existing_trace_path(self, tb_run_path: Path) -> Path | None:
        traces = sorted(tb_run_path.glob(f"**/agent-logs/{self._trace_filename()}"))
        return traces[0] if traces else None

    @staticmethod
    def _latest_recording_mtime(recordings_dir: Path) -> float | None:
        if not recordings_dir.exists():
            return None
        latest: float | None = None
        for path in recordings_dir.rglob("*"):
            if not path.is_file():
                continue
            mtime = _safe_mtime(path)
            if mtime is not None and (latest is None or mtime > latest):
                latest = mtime
        return latest

    def _augment_trace_metadata(
        self,
        *,
        src: Path,
        dst: Path,
        task: dict[str, Any],
        prompt_template: str,
        tb_version: str,
        watchdog_failure: dict[str, Any] | None = None,
    ) -> None:
        lines = src.read_text(encoding="utf-8").splitlines()
        source_metadata: dict[str, Any] | None = None
        body_start = 0
        for idx, line in enumerate(lines):
            if not line.strip():
                body_start = idx + 1
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                body_start = idx + 1
                continue
            if record.get("type") == "trace_metadata":
                source_metadata = record
                body_start = idx + 1
            else:
                body_start = idx
            break

        merged = self._trace_metadata(
            source_metadata=source_metadata,
            task=task,
            prompt_template=prompt_template,
            tb_version=tb_version,
            watchdog_failure=watchdog_failure,
        )

        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(merged, ensure_ascii=False) + "\n")
            for line in lines[body_start:]:
                if not line.strip():
                    continue
                handle.write(line + "\n")

    def _write_minimal_trace_metadata(
        self,
        *,
        dst: Path,
        task: dict[str, Any],
        prompt_template: str,
        tb_version: str,
        watchdog_failure: dict[str, Any],
    ) -> None:
        merged = self._trace_metadata(
            source_metadata=None,
            task=task,
            prompt_template=prompt_template,
            tb_version=tb_version,
            watchdog_failure=watchdog_failure,
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(merged, ensure_ascii=False) + "\n", encoding="utf-8")

    def _trace_metadata(
        self,
        *,
        source_metadata: dict[str, Any] | None,
        task: dict[str, Any],
        prompt_template: str,
        tb_version: str,
        watchdog_failure: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        merged: dict[str, Any] = dict(source_metadata or {})
        merged.update(
            {
                "type": "trace_metadata",
                "trace_format_version": 5,
                "mode": "collect",
                "scaffold": "openclaw",
                "execution_environment": "container",
                "benchmark": self.benchmark_slug,
                "model": self.model,
                "max_iterations": self.max_iterations,
                "instance_id": task["instance_id"],
                "agent_runtime_mode": "host_controller",
                "prompt_template": prompt_template,
                "task_source_kind": task.get("task_source_kind"),
                "task_source_id": task.get("task_source_id"),
                "task_source_path": task.get("task_source_path"),
                "tb_version": tb_version,
                "tb_dataset": task.get("tb_dataset"),
                "tb_registry_source": task.get("tb_registry_source"),
                "adapter_kind": "terminal_bench_openclaw",
                "agent_import_path": self.AGENT_IMPORT_PATH,
            }
        )
        run_config = dict(merged.get("run_config") or {})
        if self.mcp_config_label is not None:
            run_config["mcp_config"] = self.mcp_config_label
        if self.global_agent_timeout_sec is not None:
            run_config["global_agent_timeout_sec"] = self.global_agent_timeout_sec
        if self.llm_timeout_sec is not None:
            run_config["llm_timeout_sec"] = self.llm_timeout_sec
        if self.progress_watchdog is not None:
            run_config["progress_watchdog"] = self.progress_watchdog
        if watchdog_failure is not None:
            run_config["watchdog_exit_status"] = watchdog_failure["exit_status"]
            run_config["watchdog_error"] = watchdog_failure["error"]
        if run_config:
            merged["run_config"] = run_config
        return merged

    def _summary(
        self,
        *,
        tb_version: str,
        task: dict[str, Any],
        tb_run_path: Path,
        tb_process_logs: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "tb_version": tb_version,
            "tb_dataset": task.get("tb_dataset"),
            "tb_registry_source": task.get("tb_registry_source"),
            "adapter_kind": "terminal_bench_openclaw",
            "agent_import_path": self.AGENT_IMPORT_PATH,
            "tb_run_path": str(tb_run_path),
        }
        if self.global_agent_timeout_sec is not None:
            summary["global_agent_timeout_sec"] = self.global_agent_timeout_sec
        if self.llm_timeout_sec is not None:
            summary["llm_timeout_sec"] = self.llm_timeout_sec
        if self.progress_watchdog is not None:
            summary["progress_watchdog"] = self.progress_watchdog
        if self.mcp_config_label is not None:
            summary["mcp_config"] = self.mcp_config_label
        if tb_process_logs:
            summary.update(tb_process_logs)
        return summary

    def _trace_filename(self) -> str:
        return self.TRACE_FILENAME

    def _materialize_prompt_template(
        self,
        *,
        prompt_template: str,
        run_root: Path,
    ) -> Path:
        from trace_collect.prompt_loader import load_prompt_template

        run_root.mkdir(parents=True, exist_ok=True)
        template_text = load_prompt_template(prompt_template).replace(
            "{{task}}",
            "{{ instruction }}",
        )
        safe_name = prompt_template.replace("/", "_")
        template_path = run_root / f"prompt-template-{safe_name}.j2"
        template_path.write_text(template_text, encoding="utf-8")
        return template_path.resolve()

    @staticmethod
    def _resolve_mcp_config(mcp_config: str | None) -> str | None:
        if mcp_config in {None, "none"}:
            return None
        return str(Path(mcp_config).expanduser().resolve())

    @staticmethod
    def _mcp_config_label(mcp_config: str | None) -> str | None:
        if mcp_config is None:
            return None
        if mcp_config == "none":
            return "none"
        return Path(mcp_config).name
