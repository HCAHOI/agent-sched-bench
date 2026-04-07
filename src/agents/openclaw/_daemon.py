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
    *,
    trace_file: Path | None = None,
) -> None:
    """Write a JSON PID file for a daemon process.

    ``trace_file`` is persisted so ``get_session_status()`` can locate the
    trace JSONL after the v4 migration moved it out of the workspace into
    ``<repo>/traces/openclaw_cli/...``. Older PID files without this field
    still load fine — the status query falls back to the legacy hidden
    location for backward compatibility on existing daemons.
    """
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "pid": pid,
        "session_id": session_id,
        "started_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    if trace_file is not None:
        data["trace_file"] = str(trace_file)
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
    trace_file: Path | None = None,
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

    write_pid_file(pid_file, proc.pid, session_id, trace_file=trace_file)
    return proc.pid


# ---------------------------------------------------------------------------
# Status query
# ---------------------------------------------------------------------------


def get_session_status(session_id: str, workspace: Path) -> dict[str, Any]:
    """Get the current status of a session.

    Resolves the trace file via the PID metadata first (since the v4
    migration moved traces out of the workspace), then falls back to the
    legacy ``<workspace>/.openclaw/traces/<sid>.jsonl`` location for any
    daemons that were spawned before the new ``trace_file`` field was
    persisted. Counts v4 ``action`` records and reads v4 summary keys.
    """
    pid_file = pid_file_for_session(workspace, session_id)
    pid_data = read_pid_file(pid_file)

    status: str
    pid: int | None = None
    trace_file: Path | None = None

    if pid_data:
        pid = pid_data.get("pid")
        # Recover the absolute trace path persisted at spawn time.
        if pid_data.get("trace_file"):
            trace_file = Path(pid_data["trace_file"])
        if pid and _is_pid_alive(pid):
            status = "running"
        else:
            # Keep the PID file in place even after the daemon exits —
            # ``_is_pid_alive`` already discriminates running vs completed
            # on every subsequent query, AND for v4 traces the PID file
            # is the canonical record of the absolute trace path. Deleting
            # it would make the second ``--status`` call lose the trace
            # path and downgrade the result to ``status="unknown"``.
            status = "completed"
    else:
        status = "unknown"

    # Fallback chain when PID metadata didn't carry the trace path
    # (e.g. legacy daemon predating the v4 migration).
    if trace_file is None:
        legacy = workspace / ".openclaw" / "traces" / f"{session_id}.jsonl"
        if legacy.exists():
            trace_file = legacy
            if status == "unknown":
                status = "completed"

    n_actions = 0
    n_iterations = 0
    elapsed_s = 0.0
    if trace_file is not None and trace_file.exists():
        try:
            distinct_iters: set[int] = set()
            for line in trace_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec_type = rec.get("type")
                if rec_type == "action":
                    n_actions += 1
                    distinct_iters.add(rec.get("iteration", 0))
                elif rec_type == "summary":
                    elapsed_s = rec.get("elapsed_s", elapsed_s)
                    # Prefer authoritative summary counts when present.
                    n_actions = rec.get("n_actions", n_actions)
                    n_iterations = rec.get("n_iterations", n_iterations)
            if not n_iterations:
                n_iterations = len(distinct_iters)
        except OSError:
            pass

    return {
        "session_id": session_id,
        "status": status,
        "pid": pid,
        "workspace": str(workspace),
        "trace_file": str(trace_file) if trace_file is not None else None,
        "n_actions": n_actions,
        "n_iterations": n_iterations,
        "elapsed_s": round(elapsed_s, 2),
    }
