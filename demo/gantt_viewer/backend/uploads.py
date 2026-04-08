"""Upload persistence helpers for ad hoc trace files."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path


UPLOAD_ROOT = (
    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    / "agent-sched-bench"
    / "gantt-uploads"
)


def persist_upload(filename: str, content: bytes) -> Path:
    """Persist an uploaded trace file and return its path."""
    suffix = Path(filename).suffix or ".jsonl"
    digest = hashlib.sha256(content).hexdigest()
    target = UPLOAD_ROOT / f"{digest}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


def build_upload_id(filename: str, content: bytes) -> str:
    """Build a stable upload descriptor id."""
    digest = hashlib.sha256(content).hexdigest()[:10]
    slug = re.sub(r"[^a-z0-9_-]+", "-", Path(filename).stem.lower()).strip("-") or "trace"
    return f"upload-{slug}-{digest}"
