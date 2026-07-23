from __future__ import annotations

import json
from pathlib import Path

import pytest

from trace_collect.tool_latency_dataset import (
    load_tool_latency_corpus,
    read_tool_latency_corpus_manifest,
    require_explicit_trace_task_ids,
)


_FIXED_COSTS_MS = [float(cost) for cost in range(500, 5_001, 500)]


def test_corpus_manifest_resolves_inputs_and_validates_types(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, [f"task-{index}" for index in range(5)])

    manifest = read_tool_latency_corpus_manifest(manifest_path, repo_root=tmp_path)

    assert manifest["costs_ms"] == _FIXED_COSTS_MS
    assert manifest["expected_task_count"] == 5
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["skip_leading_cd"] = 0
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a bool"):
        read_tool_latency_corpus_manifest(manifest_path, repo_root=tmp_path)


def test_corpus_reader_uses_explicit_logical_task_metadata() -> None:
    fixture = Path(__file__).resolve().parent / "fixtures/openclaw_minimal_v5.jsonl"

    task_by_trace = require_explicit_trace_task_ids([fixture])

    assert task_by_trace[str(fixture.resolve())] == "test-openclaw-1"


def test_load_corpus_extracts_only_the_pinned_tasks(tmp_path: Path) -> None:
    task_ids = [f"task-{index}" for index in range(5)]
    trace_root = tmp_path / "traces"
    for task_id in task_ids:
        trace_dir = trace_root / task_id
        trace_dir.mkdir(parents=True)
        (trace_dir / "trace.jsonl").write_text(
            json.dumps(
                {
                    "type": "trace_metadata",
                    "trace_format_version": 5,
                    "scaffold": "test",
                    "instance_id": task_id,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "action",
                    "action_type": "tool_exec",
                    "action_id": "tool-0",
                    "agent_id": task_id,
                    "iteration": 0,
                    "ts_start": 1.0,
                    "ts_end": 1.1,
                    "data": {"tool_name": "exec", "success": True},
                }
            )
            + "\n",
            encoding="utf-8",
        )
    manifest_path = _write_manifest(tmp_path, task_ids)

    samples_by_task, loaded_task_ids, _ = load_tool_latency_corpus(manifest_path)

    assert loaded_task_ids == task_ids
    assert set(samples_by_task) == set(task_ids)
    assert all(len(samples) == 1 for samples in samples_by_task.values())


def _write_manifest(tmp_path: Path, task_ids: list[str]) -> Path:
    trace_root = tmp_path / "traces"
    trace_root.mkdir(exist_ok=True)
    payload = {
        "schema_version": 1,
        "collection_id": "corpus",
        "trace_root": str(trace_root),
        "task_ids": task_ids,
        "expected_task_count": 5,
        "fold_count": 5,
        "inner_folds": 4,
        "costs_ms": _FIXED_COSTS_MS,
        "guard_ms": 0,
        "min_tool_history": 1,
        "min_profile_tasks": 1,
        "command_field": "command",
        "max_prefix_depth": 4,
        "skip_leading_cd": False,
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest
