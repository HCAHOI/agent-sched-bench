"""Tests for the Claude Code import cache."""

from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path

import pytest

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


def test_load_or_import_propagates_importer_errors(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """If import_claude_code_session raises, load_or_import propagates — no silent swallow."""
    monkeypatch.setattr(cc_cache, "CACHE_ROOT", tmp_path / "cache")

    def boom(**kwargs):
        raise RuntimeError("simulated import failure")

    monkeypatch.setattr(cc_cache, "import_claude_code_session", boom)

    with pytest.raises(RuntimeError, match="simulated import failure"):
        cc_cache.load_or_import(CC_FIXTURE)


def test_load_or_import_dedupes_concurrent_calls(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Concurrent load_or_import calls for the same source fire the importer once."""
    monkeypatch.setattr(cc_cache, "CACHE_ROOT", tmp_path / "cache")

    call_count = {"n": 0}
    real_import = cc_cache.import_claude_code_session
    enter_gate = threading.Event()
    release_gate = threading.Event()

    def slow_counting_import(**kwargs):
        call_count["n"] += 1
        enter_gate.set()
        # Block the first import until the second thread has had a chance to
        # try to acquire the per-hash lock and observe the cache miss.
        release_gate.wait(timeout=5.0)
        return real_import(**kwargs)

    monkeypatch.setattr(cc_cache, "import_claude_code_session", slow_counting_import)

    results: list[Path] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            results.append(cc_cache.load_or_import(CC_FIXTURE))
        except BaseException as exc:  # noqa: BLE001 — propagate to main thread
            errors.append(exc)

    thread_a = threading.Thread(target=worker)
    thread_b = threading.Thread(target=worker)
    thread_a.start()
    # Ensure A has entered the importer (and is holding the lock) before B races in.
    assert enter_gate.wait(timeout=5.0), "first import never entered"
    thread_b.start()
    # Give B a moment to block on the per-hash lock.
    time.sleep(0.1)
    release_gate.set()
    thread_a.join(timeout=5.0)
    thread_b.join(timeout=5.0)

    assert not errors, f"worker raised: {errors!r}"
    assert call_count["n"] == 1, "per-hash lock should dedupe concurrent imports"
    assert len(results) == 2
    assert results[0] == results[1]
