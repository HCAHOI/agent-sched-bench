"""Tests for trace discovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from demo.gantt_viewer.backend.discovery import (
    discover_traces,
    load_discovery_config,
    sniff_format,
)
from demo.gantt_viewer.tests.helpers import (
    write_claude_code_trace,
    write_config,
    write_v5_trace,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CC_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "claude_code_minimal.jsonl"


def test_sniff_format_v5(tmp_path: Path) -> None:
    trace_path = write_v5_trace(tmp_path / "runs" / "task-1" / "trace.jsonl", [])
    assert sniff_format(trace_path) == "v5"


def test_sniff_format_claude_code(tmp_path: Path) -> None:
    trace_path = write_claude_code_trace(
        tmp_path / "cc" / "example-task" / "attempt_1" / "trace.jsonl"
    )
    assert sniff_format(trace_path) == "claude-code"


def test_sniff_format_skips_claude_code_preamble_records() -> None:
    assert sniff_format(CC_FIXTURE) == "claude-code"


def test_sniff_format_empty_file_raises(tmp_path: Path) -> None:
    trace_path = tmp_path / "empty.jsonl"
    trace_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty JSONL file"):
        sniff_format(trace_path)


def test_discover_traces_builds_expected_ids(tmp_path: Path) -> None:
    v5_path = write_v5_trace(
        tmp_path / "runs" / "repo__issue-1" / "trace.jsonl",
        [],
    )
    cc_path = write_claude_code_trace(
        tmp_path / "cc" / "encode_httpx-2701" / "attempt_1" / "trace.jsonl"
    )
    config_path = write_config(
        tmp_path / "config.yaml",
        [str(v5_path), str(cc_path)],
    )

    config = load_discovery_config(config_path)
    descriptors = discover_traces(config)

    assert [descriptor.id for descriptor in descriptors] == [
        "ac1-encode_httpx-2701",
        "ac1-repo__issue-1",
    ]
    assert descriptors[0].label == "encode_httpx-2701"
    assert descriptors[0].source_format == "claude-code"
    assert descriptors[1].source_format == "v5"


def test_sniff_format_unknown_shape_raises(tmp_path: Path) -> None:
    trace_path = tmp_path / "unknown.jsonl"
    trace_path.write_text(json.dumps({"foo": "bar"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unable to sniff trace format"):
        sniff_format(trace_path)
