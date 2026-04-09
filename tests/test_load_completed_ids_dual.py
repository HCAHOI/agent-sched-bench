"""Tests for load_completed_ids dual-path dedup."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trace_collect.collector import load_completed_ids  # noqa: E402


def _write_trace(path: Path, *, has_summary: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {"type": "trace_metadata", "scaffold": "mini-swe-agent", "trace_format_version": 5}
        )
    ]
    if has_summary:
        lines.append(json.dumps({"type": "summary", "agent_id": "x", "success": True}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_completed_ids_legacy_flat_layout(tmp_path: Path) -> None:
    _write_trace(tmp_path / "django__django-12345.jsonl", has_summary=True)
    _write_trace(tmp_path / "sympy__sympy-67890.jsonl", has_summary=False)
    _write_trace(tmp_path / "wemake__styleguide-2343.jsonl", has_summary=True)
    # results.jsonl must NOT be treated as a completed instance id
    (tmp_path / "results.jsonl").write_text("{}\n", encoding="utf-8")

    completed = load_completed_ids(tmp_path)
    assert completed == {"django__django-12345", "wemake__styleguide-2343"}


def test_load_completed_ids_new_nested_layout(tmp_path: Path) -> None:
    _write_trace(
        tmp_path / "mozilla__bleach-259" / "attempt_1" / "trace.jsonl",
        has_summary=True,
    )
    _write_trace(
        tmp_path / "encode__httpx-2701" / "attempt_1" / "trace.jsonl",
        has_summary=False,
    )
    _write_trace(
        tmp_path / "tobymao__sqlglot-3425" / "attempt_2" / "trace.jsonl",
        has_summary=True,
    )
    completed = load_completed_ids(tmp_path)
    assert completed == {"mozilla__bleach-259", "tobymao__sqlglot-3425"}


def test_load_completed_ids_mixed_layouts(tmp_path: Path) -> None:
    # Legacy flat
    _write_trace(tmp_path / "legacy__task-1.jsonl", has_summary=True)
    # New nested
    _write_trace(
        tmp_path / "new__task-2" / "attempt_1" / "trace.jsonl",
        has_summary=True,
    )
    completed = load_completed_ids(tmp_path)
    assert completed == {"legacy__task-1", "new__task-2"}


def test_load_completed_ids_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert load_completed_ids(tmp_path / "does_not_exist") == set()
