"""Claude Code import cache helpers."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import threading
from pathlib import Path

from trace_collect.claude_code_import import import_claude_code_session


CACHE_ROOT = (
    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    / "agent-sched-bench"
    / "gantt-cc-import"
)
_IMPORTER_VERSION = "1"
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def cache_key(session_path: Path) -> str:
    """Return the cache key for a raw Claude Code session file."""
    resolved_path = Path(session_path).expanduser().resolve()
    stat = resolved_path.stat()
    payload = (
        f"{resolved_path}|{stat.st_mtime_ns}|{stat.st_size}|{_IMPORTER_VERSION}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_or_import(cc_path: Path) -> Path:
    """Return a cached converted v5 trace, importing on cache miss."""
    resolved_path = Path(cc_path).expanduser().resolve()
    key = cache_key(resolved_path)
    target_path = CACHE_ROOT / f"{key}.jsonl"
    if target_path.exists():
        return target_path

    lock = _get_lock(key)
    with lock:
        if target_path.exists():
            return target_path

        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="gantt-cc-import-") as tmp_dir:
            emitted_path = import_claude_code_session(
                session_path=resolved_path,
                output_dir=Path(tmp_dir),
            )
            shutil.move(str(emitted_path), str(target_path))
        return target_path


def _get_lock(key: str) -> threading.Lock:
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[key] = lock
        return lock
