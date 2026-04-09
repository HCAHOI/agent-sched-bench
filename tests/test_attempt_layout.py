"""Tests for src/trace_collect/attempt_layout.py and AttemptContext."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trace_collect import attempt_layout  # noqa: E402
from trace_collect.attempt_pipeline import AttemptContext  # noqa: E402


REFERENCE_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "traces"
    / "swe-rebench"
    / "claude-code-haiku"
    / "mozilla__bleach-259"
    / "attempt_1"
    / "run_manifest.json"
)


def test_write_run_manifest_matches_cc_top_level_keys(tmp_path: Path) -> None:
    assert REFERENCE_MANIFEST.exists(), (
        "CC reference manifest missing; schema test cannot run"
    )
    reference = json.loads(REFERENCE_MANIFEST.read_text(encoding="utf-8"))

    manifest_dict = {
        "status": "completed",
        "characterization_only": False,
        "task": reference["task"],
        "attempt": "attempt_1",
        "model": {"requested": "mini-swe-agent", "claude_binary": None},
        "runtime": {
            "start_time": "2026-04-09T05:00:00",
            "end_time": "2026-04-09T05:05:00",
            "min_free_disk_gb": 30.0,
        },
        "replay": {"replay_ready": False},
        "result_summary": {"exit_code": 0, "total_time": 300.0, "error": None},
    }
    attempt_dir = tmp_path / "attempt_1"
    attempt_layout.write_run_manifest(attempt_dir, manifest_dict)
    written = json.loads((attempt_dir / "run_manifest.json").read_text())

    assert written["schema_version"] == attempt_layout.SCHEMA_VERSION
    assert set(reference.keys()).issubset(set(written.keys())), (
        f"Missing CC fields: {set(reference.keys()) - set(written.keys())}"
    )


def test_write_resources_json_uses_samples_wrapper(tmp_path: Path) -> None:
    samples = [
        {
            "timestamp": "2026-04-09T05:00:01",
            "epoch": 1775700001.0,
            "mem_usage": "48MB / 52GB",
            "mem_percent": "0.09%",
            "cpu_percent": "1.5%",
        }
    ]
    attempt_layout.write_resources_json(tmp_path, samples)
    doc = json.loads((tmp_path / "resources.json").read_text())
    assert "samples" in doc
    assert isinstance(doc["samples"], list)
    assert doc["samples"] == samples
    assert doc["summary"] == {}


def test_write_tool_calls_json_is_top_level_list(tmp_path: Path) -> None:
    tool_calls = [
        {
            "timestamp": "2026-04-09T05:00:02Z",
            "tool": "Bash",
            "id": "call_0",
            "input": {"command": "ls"},
            "end_timestamp": "2026-04-09T05:00:02Z",
            "duration_ms": 17.0,
            "result_preview": "file.txt",
        }
    ]
    attempt_layout.write_tool_calls_json(tmp_path, tool_calls)
    doc = json.loads((tmp_path / "tool_calls.json").read_text())
    assert isinstance(doc, list)
    assert doc == tool_calls


def test_write_claude_output_and_stderr(tmp_path: Path) -> None:
    attempt_layout.write_claude_output(tmp_path, "hello stdout")
    attempt_layout.write_claude_stderr(tmp_path, "hello stderr")
    assert (tmp_path / "claude_output.txt").read_text() == "hello stdout"
    assert (tmp_path / "claude_stderr.txt").read_text() == "hello stderr"


def test_copy_trace_jsonl_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text('{"type":"trace_metadata"}\n', encoding="utf-8")
    attempt_layout.copy_trace_jsonl(tmp_path, source)
    target = tmp_path / "trace.jsonl"
    assert target.exists()
    assert target.read_text() == '{"type":"trace_metadata"}\n'


def test_copy_trace_jsonl_same_path_no_op(tmp_path: Path) -> None:
    existing = tmp_path / "trace.jsonl"
    existing.write_text("x", encoding="utf-8")
    attempt_layout.copy_trace_jsonl(tmp_path, existing)
    assert existing.read_text() == "x"


def test_attempt_context_computes_attempt_dir(tmp_path: Path) -> None:
    ctx = AttemptContext(
        run_dir=tmp_path,
        instance_id="mozilla__bleach-259",
        attempt=1,
        task={"instance_id": "mozilla__bleach-259"},
        model="qwen-plus-latest",
        requested_model="qwen-plus",
        scaffold="mini-swe-agent",
        source_image="swerebench/...",
    )
    assert ctx.attempt_dir == tmp_path / "mozilla__bleach-259" / "attempt_1"
    assert ctx.attempt_label == "attempt_1"
    assert ctx.container_id is None
    ctx.mark_container_ready("abc123")
    assert ctx.container_id == "abc123"


def test_attempt_context_elapsed_seconds_monotonic(tmp_path: Path) -> None:
    start = datetime(2026, 4, 9, 5, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 4, 9, 5, 0, 42, tzinfo=timezone.utc)
    ctx = AttemptContext(
        run_dir=tmp_path,
        instance_id="x",
        attempt=1,
        task={},
        model="m",
        requested_model="m",
        scaffold="mini-swe-agent",
        source_image="img",
    )
    ctx.start_time = start
    ctx.end_time = end
    assert ctx.elapsed_seconds() == 42.0
