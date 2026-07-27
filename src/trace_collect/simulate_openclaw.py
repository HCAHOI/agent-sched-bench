from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from harness.trace_logger import TraceLogger
from trace_collect.openclaw_host_runtime import replay_action_failure_counts
from trace_collect.simulate_outputs import _make_task_stats, _make_trace_summary
from trace_collect.simulate_types import (
    LLMTimingConfig,
    LoadedTraceSession,
    PreparedTraceSession,
    ReplayTaskStats,
    SleepDrift,
)
from trace_collect.simulate_utils import (
    _coerce_action_bounds,
    _source_model,
    _summarize_sleep_drifts,
)


def _append_replay_record(trace_logger: TraceLogger, record: dict[str, Any]) -> None:
    handle = getattr(trace_logger, "_handle")
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def _source_terminal_reason(loaded: LoadedTraceSession) -> str:
    summary = loaded.summary or {}
    if summary.get("success") is not False:
        return "completed"
    replayable = [
        action
        for action in loaded.actions
        if action.get("action_type") in {"llm_call", "tool_exec"}
    ]
    if not replayable:
        return "unsupported"
    last = replayable[-1]
    raw_response = (last.get("data") or {}).get("raw_response") or {}
    choices = raw_response.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    finish_reason = choice.get("finish_reason")
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    if last.get("action_type") == "llm_call":
        if finish_reason == "tool_calls" or message.get("tool_calls"):
            return "trace_ended_before_tools"
        if finish_reason == "error":
            return "llm_error"

    n_iterations = summary.get("n_iterations")
    max_iterations = (loaded.metadata or {}).get("max_iterations")
    if (
        isinstance(n_iterations, int)
        and isinstance(max_iterations, int)
        and n_iterations >= max_iterations
    ):
        return "max_iterations"
    if last.get("action_type") == "tool_exec":
        return "trace_ended_after_tools"
    return "completed"


def _openclaw_worker_timeout_s(
    loaded: LoadedTraceSession,
    *,
    replay_speed: float,
    command_timeout_s: float,
    setup_timeout_s: float = 0.0,
) -> float:
    action_count = max(1, len(loaded.actions))
    if not loaded.actions:
        return command_timeout_s + setup_timeout_s + 120.0
    bounds = [
        _coerce_action_bounds(action, source_trace=loaded.source_trace)
        for action in loaded.actions
    ]
    source_span_s = max(end for _, end in bounds) - min(start for start, _ in bounds)
    tool_count = sum(
        1 for action in loaded.actions if action.get("action_type") == "tool_exec"
    )
    return max(
        command_timeout_s + setup_timeout_s + 120.0,
        source_span_s / replay_speed
        + (tool_count + 2) * command_timeout_s
        + setup_timeout_s
        + 120.0,
        action_count * 5.0,
    )


async def _stream_worker_output(stream: Any, path: Path) -> None:
    with path.open("wb") as handle:
        while True:
            chunk = await stream.read(8192)
            if not chunk:
                return
            handle.write(chunk)
            handle.flush()


async def _run_openclaw_worker_process(
    *,
    request_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout_s: float,
) -> int:
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "trace_collect.openclaw_host_replay_worker",
        "--request",
        str(request_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None
    stdout_task = asyncio.create_task(_stream_worker_output(proc.stdout, stdout_path))
    stderr_task = asyncio.create_task(_stream_worker_output(proc.stderr, stderr_path))
    try:
        await asyncio.wait_for(proc.wait(), timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        with stderr_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"\nOpenClaw host replay worker timed out after {timeout_s:.1f}s\n"
            )
        return 124
    finally:
        await asyncio.gather(stdout_task, stderr_task)
    return int(proc.returncode or 0)


async def _run_openclaw_replay_session(
    prepared_session: PreparedTraceSession,
    *,
    trace_logger: TraceLogger,
    replay_speed: float,
    llm_timing: LLMTimingConfig,
    command_timeout_s: float,
    warmup_skip_iterations: int = 0,
) -> ReplayTaskStats:
    """Replay OpenClaw by launching one host worker process for this trace."""
    del warmup_skip_iterations

    loaded = prepared_session.loaded
    ctr = prepared_session.container
    if ctr is None:
        raise RuntimeError("OpenClaw host replay requires a prepared task container")
    if prepared_session.task_output_dir is None:
        raise RuntimeError("OpenClaw host replay requires task_output_dir")

    task_output_dir = prepared_session.task_output_dir
    trace_file = task_output_dir / "openclaw_host_replay.jsonl"
    runtime_dir = task_output_dir / "openclaw-runtime"
    workspace = task_output_dir / "host-workspace"
    status_path = task_output_dir / "openclaw_host_replay_status.json"
    request_path = task_output_dir / "openclaw_host_replay_request.json"
    stdout_path = task_output_dir / "openclaw_host_replay_stdout.txt"
    stderr_path = task_output_dir / "openclaw_host_replay_stderr.txt"
    resource_artifact_path = task_output_dir / "resource_observations.json"
    prompt = str(
        loaded.task.get("problem_statement") or "Replay source OpenClaw trace."
    )
    tool_resource_profile = os.environ.get("TOOL_RESOURCE_PROFILE")
    resource_enabled = tool_resource_profile is not None
    repo = loaded.task.get("repo")
    resource_scope = (
        repo.strip()
        if isinstance(repo, str) and repo.strip()
        else f"task:{loaded.task_instance_id}"
    )
    resource_run_tokens = json.loads(os.environ.get("TOOL_RESOURCE_RUN_TOKENS", "{}"))
    if not isinstance(resource_run_tokens, dict):
        raise ValueError("TOOL_RESOURCE_RUN_TOKENS must be a JSON object")
    resource_run_token = resource_run_tokens.get(resource_scope)
    if resource_run_token is not None and not isinstance(resource_run_token, str):
        raise ValueError("tool-resource run token must be a string")
    resource_setup_timeout_s = 0.0
    if resource_enabled:
        from tool_resource.resource_protocol import RESOURCE_OPERATION_TIMEOUTS_S

        resource_setup_timeout_s = RESOURCE_OPERATION_TIMEOUTS_S[
            "AwaitTraceReady"
        ]
    request = {
        "source_trace": str(loaded.source_trace),
        "source_actions": loaded.actions,
        "task_prompt": prompt,
        "prompt": prompt,
        "output_trace": str(trace_file),
        "runtime_dir": str(runtime_dir),
        "runtime_artifact_root_map": prepared_session.runtime_artifact_root_map,
        "workspace": str(workspace),
        "status_path": str(status_path),
        "resource_artifact_path": str(resource_artifact_path),
        "container_executable": ctr.container_executable,
        "container_id": ctr.container_id,
        "container_workdir": ctr.workdir,
        "container_python_runtime": ctr.python_runtime,
        "container_pythonpath": ctr.pythonpath,
        "replay_speed": replay_speed,
        "llm_timing": {
            "mode": llm_timing.mode,
            "ttft_ms": llm_timing.ttft_ms,
            "tpot_ms": llm_timing.tpot_ms,
        },
        "command_timeout_s": command_timeout_s,
        "tool_resource_profile": tool_resource_profile,
        "tool_resource_run_token": resource_run_token,
        "task_instance_id": loaded.task_instance_id,
        "repo": loaded.task.get("repo"),
        "source_action_agent_id": loaded.source_action_agent_id,
        "run_instance_id": loaded.run_instance_id,
        "manifest_index": loaded.manifest_index,
        "source_model": _source_model(loaded),
        "source_success": (loaded.summary or {}).get("success"),
        "source_terminal_reason": _source_terminal_reason(loaded),
        "expected_action_count": sum(
            1
            for action in loaded.actions
            if action.get("action_type") in {"llm_call", "tool_exec"}
        ),
    }
    task_output_dir.mkdir(parents=True, exist_ok=True)
    request_path.write_text(
        json.dumps(request, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    worker_returncode = await _run_openclaw_worker_process(
        request_path=request_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout_s=_openclaw_worker_timeout_s(
            loaded,
            replay_speed=replay_speed,
            command_timeout_s=command_timeout_s,
            setup_timeout_s=resource_setup_timeout_s,
        ),
    )
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
    else:
        status = {
            "success": False,
            "stop_reason": "error",
            "error": "OpenClaw host replay worker did not write status",
            "elapsed_s": 0.0,
            "sleep_records": [],
            "agent_execution_environment": "host",
            "tool_execution_environment": "task_container",
            "tool_container_id": ctr.container_id,
            "tool_container_user": "unknown",
            "openclaw_host_pid": None,
            "telemetry_integrity_failed": resource_enabled,
            "replay_execution": "incomplete",
            "telemetry_quality": ("unavailable" if resource_enabled else "ok"),
            "formal_completeness": (
                "unavailable" if resource_enabled else "not_requested"
            ),
            "call_coverage": None,
            "collection_validity": ("invalid" if resource_enabled else "not_requested"),
            "telemetry_errors": ["worker status unavailable"],
        }

    emitted_records: list[dict[str, Any]] = []
    replay_action_records: list[dict[str, Any]] = []
    if trace_file.exists():
        for line in trace_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            rtype = record.get("type")
            if rtype == "trace_metadata" or rtype == "event":
                continue
            if rtype == "action":
                data = dict(record.get("data") or {})
                data.setdefault("run_instance_id", loaded.run_instance_id)
                data.setdefault("task_instance_id", loaded.task_instance_id)
                data.setdefault("source_action_agent_id", loaded.source_action_agent_id)
                data.setdefault("source_agent_id", loaded.source_action_agent_id)
                data.setdefault("manifest_index", loaded.manifest_index)
                data.setdefault("simulate_source", str(loaded.source_trace))
                data.setdefault("replay_mode", "openclaw_host_worker")
                data.setdefault("replay_speed", replay_speed)
                data.setdefault("openclaw_host_pid", status.get("openclaw_host_pid"))
                record["data"] = data
                replay_action_records.append(record)
            emitted_records.append(record)

    action_counts = replay_action_failure_counts(loaded.actions, replay_action_records)
    expected_actions = int(request["expected_action_count"])
    missing_actions = max(0, expected_actions - action_counts.emitted_actions)
    failed_actions = action_counts.unexpected_replay_failed_actions + missing_actions
    if (
        not action_counts.action_sequence_matches
        or worker_returncode != 0
        or status.get("success") is not True
    ):
        failed_actions = max(1, failed_actions)
    replay_execution = status.get(
        "replay_execution",
        "completed" if status.get("success") else "failed",
    )
    if failed_actions and replay_execution == "completed":
        replay_execution = "failed"
    telemetry_integrity_failed = bool(
        status.get("telemetry_integrity_failed", False)
    )
    telemetry_quality = status.get(
        "telemetry_quality",
        "unavailable" if resource_enabled else "ok",
    )
    formal_completeness = status.get(
        "formal_completeness",
        "unavailable" if resource_enabled else "not_requested",
    )
    call_coverage = status.get("call_coverage")
    collection_validity = status.get(
        "collection_validity",
        "invalid" if resource_enabled else "not_requested",
    )
    telemetry_errors = list(status.get("telemetry_errors") or [])
    task_success = (
        failed_actions == 0 and (loaded.summary or {}).get("success") is not False
    )

    sleep_drifts = [
        SleepDrift(
            phase=str(item.get("phase", "llm_replay")),
            expected_s=float(item.get("expected_s", 0.0)),
            actual_s=float(item.get("actual_s", 0.0)),
        )
        for item in status.get("sleep_records", [])
        if isinstance(item, dict)
    ]
    summary_extra = {
        "replay_mode": "openclaw_host_worker",
        "replay_speed": replay_speed,
        "llm_timing_mode": llm_timing.mode,
        "sleep_drift": _summarize_sleep_drifts(sleep_drifts),
        "failed_actions": failed_actions,
        "emitted_actions": action_counts.emitted_actions,
        "source_failed_actions": action_counts.source_failed_actions,
        "replay_failed_actions": action_counts.replay_failed_actions,
        "unexpected_replay_failed_actions": (
            action_counts.unexpected_replay_failed_actions
        ),
        "expected_actions": expected_actions,
        "missing_source_action_count": missing_actions,
        "action_sequence_matches": action_counts.action_sequence_matches,
        "source_terminal_reason": request["source_terminal_reason"],
        "worker_returncode": worker_returncode,
        "worker_stop_reason": status.get("stop_reason"),
        "worker_error": status.get("error"),
        "worker_request_path": str(request_path),
        "worker_status_path": str(status_path),
        "worker_stdout_path": str(stdout_path),
        "worker_stderr_path": str(stderr_path),
        "worker_trace_path": str(trace_file),
        "agent_execution_environment": status.get(
            "agent_execution_environment", "host"
        ),
        "tool_execution_environment": status.get(
            "tool_execution_environment", "task_container"
        ),
        "tool_runtime": status.get("tool_runtime"),
        "tool_container_id": status.get("tool_container_id", ctr.container_id),
        "tool_container_user": status.get("tool_container_user", "unknown"),
        "tool_container_user_id": status.get("tool_container_user_id"),
        "tool_container_workdir": status.get("tool_container_workdir"),
        "openclaw_host_pid": status.get("openclaw_host_pid"),
        "tool_resource": status.get(
            "tool_resource",
            {"profile": tool_resource_profile, "service_enabled": resource_enabled},
        ),
        "resource_artifact_path": (
            str(resource_artifact_path) if resource_enabled else None
        ),
        "telemetry_integrity_failed": telemetry_integrity_failed,
        "replay_execution": replay_execution,
        "telemetry_quality": telemetry_quality,
        "formal_completeness": formal_completeness,
        "call_coverage": call_coverage,
        "collection_validity": collection_validity,
        "telemetry_errors": telemetry_errors,
    }
    summary_seen = False
    for record in emitted_records:
        if record.get("type") == "summary":
            summary_seen = True
            record.update(
                _make_trace_summary(
                    loaded=loaded,
                    success=task_success,
                    elapsed_s=float(status.get("elapsed_s") or 0.0),
                    source_model=_source_model(loaded),
                    extra=summary_extra,
                )
            )
        _append_replay_record(trace_logger, record)
    if not summary_seen:
        trace_logger.log_summary(
            loaded.agent_id,
            _make_trace_summary(
                loaded=loaded,
                success=task_success,
                elapsed_s=float(status.get("elapsed_s") or 0.0),
                source_model=_source_model(loaded),
                extra=summary_extra,
            ),
        )
    task_stats = _make_task_stats(
        loaded=loaded,
        success=task_success,
        elapsed_s=float(status.get("elapsed_s") or 0.0),
        failed_action_count=failed_actions,
    )
    return task_stats
