"""Tests for the Claude Code import cache."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from demo.gantt_viewer.backend import cc_cache


REPO_ROOT = Path(__file__).resolve().parents[3]
CC_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "claude_code_minimal.jsonl"


def test_load_or_import_cache_miss_then_hit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(cc_cache, "CACHE_ROOT", cache_root)

    first = cc_cache.load_or_import(CC_FIXTURE)
    assert first.exists()
    first_stat = first.stat().st_mtime_ns

    second = cc_cache.load_or_import(CC_FIXTURE)
    assert second == first
    assert second.stat().st_mtime_ns == first_stat
    assert list(cache_root.glob("*.jsonl")) == [first]


def test_cache_key_changes_after_source_mtime_update(tmp_path: Path) -> None:
    session_copy = tmp_path / "claude_code_minimal.jsonl"
    shutil.copy2(CC_FIXTURE, session_copy)

    first_key = cc_cache.cache_key(session_copy)
    stat = session_copy.stat()
    os.utime(
        session_copy,
        ns=(stat.st_atime_ns + 1_000_000_000, stat.st_mtime_ns + 1_000_000_000),
    )
    second_key = cc_cache.cache_key(session_copy)

    assert first_key != second_key
