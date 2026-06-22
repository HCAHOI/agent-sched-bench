"""Runtime trace ingestion helpers for the Gantt viewer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from demo.gantt_viewer.backend.discovery import sniff_format


@dataclass(frozen=True, slots=True)
class CanonicalizedTrace:
    canonical_path: Path
    source_format: Literal["trace"]


def ensure_canonical_trace_path(path: Path) -> CanonicalizedTrace:
    """Resolve a path to a canonical trace, raising ValueError if it is not one."""
    resolved_path = Path(path).expanduser().resolve()
    source_format = sniff_format(resolved_path)
    return CanonicalizedTrace(
        canonical_path=resolved_path,
        source_format=source_format,
    )
