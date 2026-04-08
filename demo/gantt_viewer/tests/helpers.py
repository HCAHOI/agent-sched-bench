"""Shared test helpers for the Gantt viewer backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_v5_trace(
    trace_path: Path,
    records: list[dict[str, Any]],
    *,
    scaffold: str = "synthetic",
    model: str = "test-model",
    max_iterations: int = 10,
) -> Path:
    """Write a minimal v5 trace JSONL file."""
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "type": "trace_metadata",
        "scaffold": scaffold,
        "model": model,
        "trace_format_version": 5,
        "max_iterations": max_iterations,
    }
    with trace_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header) + "\n")
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return trace_path


def write_claude_code_trace(trace_path: Path) -> Path:
    """Write the minimal raw Claude Code session shape needed for sniffing."""
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "type": "queue-operation",
            "operation": "enqueue",
            "timestamp": "2026-04-08T13:40:41.615Z",
            "sessionId": "session-1",
            "content": "hello",
        },
        {
            "type": "user",
            "sessionId": "session-1",
            "timestamp": "2026-04-08T13:40:41.644Z",
            "message": {"role": "user", "content": "hello"},
        },
    ]
    with trace_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return trace_path


def write_config(config_path: Path, trace_paths: list[str]) -> Path:
    """Write a minimal discovery config."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["groups:", '  - name: "AC1 — test group"', "    paths:"]
    for trace_path in trace_paths:
        lines.append(f"      - {trace_path}")
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path
