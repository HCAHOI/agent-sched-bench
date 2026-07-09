"""Shared output writer for latency evaluation summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_summary_outputs(
    summary: dict[str, Any],
    *,
    detail_key: str,
    summary_path: Path | None = None,
    detail_path: Path | None = None,
) -> None:
    """Write a summary JSON (without per-row details) and a detail JSONL.

    ``detail_key`` names the per-row list inside ``summary`` (e.g.
    ``decisions`` or ``predictions``); it is dropped from the summary file and
    written one JSON object per line to ``detail_path``.
    """

    summary_without_details = {
        key: value for key, value in summary.items() if key != detail_key
    }
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary_without_details, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    if detail_path is not None:
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        with detail_path.open("w", encoding="utf-8") as fh:
            for row in summary[detail_key]:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                fh.write("\n")


__all__ = ["write_summary_outputs"]
