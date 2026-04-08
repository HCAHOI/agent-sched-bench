"""Claude Code import cache helpers."""

from __future__ import annotations

import os
from pathlib import Path


CACHE_ROOT = (
    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    / "agent-sched-bench"
    / "gantt-cc-import"
)
