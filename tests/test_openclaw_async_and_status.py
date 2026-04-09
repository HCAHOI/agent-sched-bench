"""Regression tests for OpenClaw async daemon trace-path forwarding and
v4-aware ``get_session_status`` parsing.

These pin down the codex-critic findings against the prior version that:
1. silently dropped ``--trace-output`` when running in ``--async`` mode, and
2. parsed ``type=step`` records and the ``n_iterations`` summary key, both of
   which no longer exist after the v4 action/event refactor.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("agents.openclaw._cli", reason="requires openclaw")

from agents.openclaw._cli import _run_async, build_parser
from agents.openclaw._daemon import (
    get_session_status,
    pid_file_for_session,
    write_pid_file,
)


# ── _run_async: argv must include --trace-output ────────────────────


def _make_async_args(workspace: Path, **overrides) -> object:
    args = build_parser().parse_args(
        ["--prompt", "do something", "--workspace", str(workspace), "--async"]
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def test_run_async_forwards_explicit_trace_output(tmp_path: Path) -> None:
    """When ``--trace-output`` is given, the spawned daemon must receive it."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    explicit = tmp_path / "out" / "custom.jsonl"
    args = _make_async_args(workspace, trace_output=str(explicit))

    captured: dict[str, object] = {}

    def _fake_spawn(cmd, pid_file, session_id, *, extra_env=None, trace_file=None):
        captured["cmd"] = list(cmd)
        captured["trace_file"] = trace_file
        return 12345

    with patch("agents.openclaw._daemon.spawn_daemon", side_effect=_fake_spawn), \
         patch.dict("os.environ", {"OPENROUTER_API_KEY": "fake-key"}):
        rc = _run_async(args)

    assert rc == 0
    cmd = captured["cmd"]
    assert "--trace-output" in cmd, f"--trace-output missing from daemon argv: {cmd}"
    idx = cmd.index("--trace-output")
    assert cmd[idx + 1] == str(explicit.expanduser().resolve())
    # spawn_daemon also receives the absolute trace_file so it can persist
    # the path in the PID metadata for --status to recover later.
    assert captured["trace_file"] == explicit.expanduser().resolve()


def test_run_async_default_trace_output_under_repo_root(tmp_path: Path) -> None:
    """No ``--trace-output`` override → daemon argv still carries the
    resolved default path (NOT a missing flag)."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    args = _make_async_args(workspace)

    captured: dict[str, object] = {}

    def _fake_spawn(cmd, pid_file, session_id, *, extra_env=None, trace_file=None):
        captured["cmd"] = list(cmd)
        captured["trace_file"] = trace_file
        return 23456

    with patch("agents.openclaw._daemon.spawn_daemon", side_effect=_fake_spawn), \
         patch.dict("os.environ", {"OPENROUTER_API_KEY": "fake-key"}):
        _run_async(args)

    cmd = captured["cmd"]
    assert "--trace-output" in cmd
    idx = cmd.index("--trace-output")
    forwarded = Path(cmd[idx + 1])
    # Default path is under <repo>/traces/openclaw_cli/...
    assert "traces" in forwarded.parts
    assert "openclaw_cli" in forwarded.parts
    assert forwarded.suffix == ".jsonl"


# ── get_session_status: v4 parsing + PID-based trace path lookup ────


def _v4_trace(path: Path, *, n_actions: int, n_iterations: int, elapsed_s: float) -> None:
    """Write a minimal v4 trace with the requested action/summary counts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({
            "type": "trace_metadata", "scaffold": "openclaw",
            "trace_format_version": 5, "model": "test",
        })
    ]
    for i in range(n_actions):
        lines.append(json.dumps({
            "type": "action",
            "action_type": "llm_call",
            "action_id": f"llm_{i}",
            "agent_id": "a1",
            "iteration": i % n_iterations if n_iterations else i,
            "ts_start": 1000.0 + i,
            "ts_end": 1000.5 + i,
            "data": {},
        }))
    lines.append(json.dumps({
        "type": "summary",
        "agent_id": "a1",
        "n_actions": n_actions,
        "n_iterations": n_iterations,
        "elapsed_s": elapsed_s,
    }))
    path.write_text("\n".join(lines) + "\n")


def test_get_session_status_reads_v4_actions_via_pid_metadata(tmp_path: Path) -> None:
    """The PID file persists the absolute trace path; status counts v4
    actions and reports n_actions / n_iterations / elapsed_s correctly."""
    workspace = tmp_path / "ws"
    pid_file = pid_file_for_session(workspace, "oc-abc")
    trace = tmp_path / "out" / "oc-abc.jsonl"
    _v4_trace(trace, n_actions=12, n_iterations=4, elapsed_s=37.5)

    # Pretend the daemon already exited (use a non-existent PID).
    write_pid_file(pid_file, pid=999999, session_id="oc-abc", trace_file=trace)

    status = get_session_status("oc-abc", workspace)
    assert status["session_id"] == "oc-abc"
    assert status["status"] == "completed"
    assert status["trace_file"] == str(trace)
    assert status["n_actions"] == 12
    assert status["n_iterations"] == 4
    assert status["elapsed_s"] == pytest.approx(37.5)
    # No legacy v3 step keys leak into the response.
    assert "steps" not in status


def test_get_session_status_handles_v4_trace_with_no_summary(tmp_path: Path) -> None:
    """Mid-flight trace (no summary record yet) — count actions and
    distinct iterations on the fly."""
    workspace = tmp_path / "ws"
    pid_file = pid_file_for_session(workspace, "oc-mid")
    trace = tmp_path / "out" / "oc-mid.jsonl"
    trace.parent.mkdir(parents=True, exist_ok=True)
    trace.write_text(
        '\n'.join([
            json.dumps({"type": "trace_metadata", "scaffold": "openclaw",
                        "trace_format_version": 5}),
            json.dumps({"type": "action", "action_type": "llm_call",
                        "action_id": "llm_0", "agent_id": "a", "iteration": 0,
                        "ts_start": 1.0, "ts_end": 2.0, "data": {}}),
            json.dumps({"type": "action", "action_type": "tool_exec",
                        "action_id": "tool_0_bash", "agent_id": "a",
                        "iteration": 0, "ts_start": 2.0, "ts_end": 2.1,
                        "data": {"tool_name": "bash"}}),
            json.dumps({"type": "action", "action_type": "llm_call",
                        "action_id": "llm_1", "agent_id": "a", "iteration": 1,
                        "ts_start": 3.0, "ts_end": 4.0, "data": {}}),
        ]) + "\n"
    )
    write_pid_file(pid_file, pid=999998, session_id="oc-mid", trace_file=trace)

    status = get_session_status("oc-mid", workspace)
    assert status["n_actions"] == 3
    assert status["n_iterations"] == 2  # iterations 0 and 1
    assert status["trace_file"] == str(trace)


def test_get_session_status_is_idempotent_for_completed_sessions(tmp_path: Path) -> None:
    """Regression for the codex review finding: a completed v4 async
    session must remain status-queryable across multiple calls. Earlier
    the PID file was deleted on the first ``--status`` call, taking the
    canonical trace path with it and downgrading the second response to
    ``status="unknown"`` / ``trace_file=None``.
    """
    workspace = tmp_path / "ws"
    pid_file = pid_file_for_session(workspace, "oc-rerun")
    trace = tmp_path / "out" / "oc-rerun.jsonl"
    _v4_trace(trace, n_actions=4, n_iterations=2, elapsed_s=12.0)
    write_pid_file(pid_file, pid=999990, session_id="oc-rerun", trace_file=trace)

    first = get_session_status("oc-rerun", workspace)
    second = get_session_status("oc-rerun", workspace)
    third = get_session_status("oc-rerun", workspace)

    for label, snap in (("first", first), ("second", second), ("third", third)):
        assert snap["status"] == "completed", f"{label}: status drifted"
        assert snap["trace_file"] == str(trace), (
            f"{label}: lost trace_file ({snap['trace_file']})"
        )
        assert snap["n_actions"] == 4
        assert snap["n_iterations"] == 2
    # The PID file must still exist after multiple completed-status queries.
    assert pid_file.exists(), "PID file was deleted, breaking idempotency"


def test_get_session_status_legacy_workspace_fallback(tmp_path: Path) -> None:
    """Old daemons spawned before US-011 didn't persist trace_file in the
    PID metadata — fall back to the legacy hidden workspace path."""
    workspace = tmp_path / "ws"
    legacy = workspace / ".openclaw" / "traces" / "oc-old.jsonl"
    _v4_trace(legacy, n_actions=2, n_iterations=2, elapsed_s=1.0)

    # No PID file at all → should still find the trace via fallback
    status = get_session_status("oc-old", workspace)
    assert status["status"] == "completed"
    assert status["trace_file"] == str(legacy)
    assert status["n_actions"] == 2
