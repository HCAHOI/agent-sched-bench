"""Task-id helpers for the offline clause-telemetry evaluation lane.

The per-tool-call CPU and container-memory label extraction that used to live
here served the bash-xtrace proxy evaluation only. Collection moved entirely to
eBPF clause telemetry, whose labels come from the telemetry artifact itself, so
that extraction was removed along with the proxy lane; git history holds it.
"""

from __future__ import annotations

import re

_REPO_SUFFIX_RE = re.compile(r"-\d+$")


def repo_of(task_id: str) -> str:
    """Repository of a task id, dropping the trailing instance number."""

    return _REPO_SUFFIX_RE.sub("", task_id)


__all__ = ["repo_of"]
