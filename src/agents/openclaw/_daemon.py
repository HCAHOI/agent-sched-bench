"""Background process management for OpenClaw async mode.

Handles daemon spawning via subprocess, PID file tracking, and status queries.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# PID file management
# ---------------------------------------------------------------------------


def write_pid_file(
    pid_file: Path,
    pid: int,
    session_id: str,
) -> None:
    """Write a JSON PID file for a daemon process."""
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "pid": pid,
        "session_id": session_id,
        "started_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    pid_file.write_text(json.dumps(data) + "\n", encoding="utf-8")


def read_pid_file(pid_file: Path) -> dict[str, Any] | None:
    """Read and parse a PID file. Returns None if missing or malformed."""
    if not pid_file.exists():
        return None
    try:
        return json.loads(pid_file.read_text(encoding="utf-8").strip())
    except (json.JSONDecodeError, OSError):
        return None


def cleanup_pid_file(pid_file: Path) -> None:
    """Remove a PID file if it exists."""
    try:
        pid_file.unlink(missing_ok=True)
    except OSError:
        pass


def pid_file_for_session(workspace: Path, session_id: str) -> Path:
    """Return the canonical PID file path for a session."""
    return workspace / ".openclaw" / "pids" / f"{session_id}.pid"


# ---------------------------------------------------------------------------
# Process liveness check
# ---------------------------------------------------------------------------


def _is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is alive."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


# ---------------------------------------------------------------------------
# Daemon spawning
# ---------------------------------------------------------------------------


def spawn_daemon(
    cmd: list[str],
    pid_file: Path,
    session_id: str,
    *,
    extra_env: dict[str, str] | None = None,
) -> int:
    """Spawn a detached daemon process and write a PID file.

    Args:
        cmd: Full command to execute as a subprocess.
        pid_file: Path to write the PID file.
        session_id: Session identifier for the PID file metadata.
        extra_env: Extra environment variables (e.g. API keys — avoids
            leaking secrets in argv visible to ``ps``).

    Returns:
        The child process PID.
    """
    # Ensure log directory exists
    log_dir = pid_file.parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{session_id}.log"

    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)

    with open(log_file, "w", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )

    write_pid_file(pid_file, proc.pid, session_id)
    return proc.pid


# ---------------------------------------------------------------------------
# Status query
# ---------------------------------------------------------------------------


def get_session_status(session_id: str, workspace: Path) -> dict[str, Any]:
    """Get the current status of a session.

    Returns a dict with: session_id, status, pid, trace_file, steps, etc.
    """
    pid_file = pid_file_for_session(workspace, session_id)
    pid_data = read_pid_file(pid_file)

    status: str
    pid: int | None = None

    if pid_data:
        pid = pid_data.get("pid")
        if pid and _is_pid_alive(pid):
            status = "running"
        else:
            status = "completed"
            cleanup_pid_file(pid_file)
    else:
        # No PID file — check if session/trace exists
        trace_file = workspace / ".openclaw" / "traces" / f"{session_id}.jsonl"
        if trace_file.exists():
            status = "completed"
        else:
            status = "unknown"

    # Try to read trace summary
    trace_file = workspace / ".openclaw" / "traces" / f"{session_id}.jsonl"
    steps = 0
    elapsed_s = 0.0
    if trace_file.exists():
        try:
            for line in trace_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") == "step":
                    steps += 1
                elif rec.get("type") == "summary":
                    elapsed_s = rec.get("elapsed_s", 0)
                    steps = rec.get("n_steps", steps)
        except OSError:
            pass

    return {
        "session_id": session_id,
        "status": status,
        "pid": pid,
        "workspace": str(workspace),
        "trace_file": str(trace_file),
        "steps": steps,
        "elapsed_s": round(elapsed_s, 2),
    }
