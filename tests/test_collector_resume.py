"""Tests for trace collection resume terminal-state handling."""

from __future__ import annotations

import json
from pathlib import Path

from trace_collect.collector import load_completed_ids


def _write_manifest(
    run_dir: Path,
    instance_id: str,
    *,
    status: str,
    exit_status: str | None = None,
    error: str | None = None,
) -> None:
    attempt_dir = run_dir / instance_id / "attempt_1"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "status": status,
        "result_summary": {
            "exit_code": 0 if status == "completed" else 1,
            "error": error,
        },
    }
    if exit_status is not None:
        result_summary = payload["result_summary"]
        assert isinstance(result_summary, dict)
        result_summary["exit_status"] = exit_status
    (attempt_dir / "run_manifest.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_load_completed_ids_treats_exhausted_as_resume_terminal(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_manifest(run_dir, "task-completed", status="completed")
    _write_manifest(
        run_dir,
        "task-exhausted",
        status="exhausted",
        exit_status="max_iterations",
        error="I reached the maximum number of tool call iterations.",
    )
    _write_manifest(
        run_dir,
        "task-error",
        status="error",
        exit_status="tool_error",
        error="tool failed",
    )

    assert load_completed_ids(run_dir) == {"task-completed", "task-exhausted"}


def test_load_completed_ids_does_not_treat_error_manifests_as_terminal(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_manifest(
        run_dir,
        "task-tool-error",
        status="error",
        exit_status="tool_error",
        error="tool failed after mentioning maximum number of tool call iterations",
    )
    _write_manifest(
        run_dir,
        "task-legacy-exhausted",
        status="error",
        error="I reached the maximum number of tool call iterations.",
    )
    _write_manifest(
        run_dir,
        "task-error-with-max-exit-status",
        status="error",
        exit_status="max_iterations",
        error="I reached the maximum number of tool call iterations.",
    )

    assert load_completed_ids(run_dir) == set()
