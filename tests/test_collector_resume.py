"""Tests for trace collection resume terminal-state handling."""

from __future__ import annotations

import json
from pathlib import Path

from trace_collect.collector import (
    CollectedTaskResult,
    load_completed_ids,
    load_terminal_results,
    write_merged_results_jsonl,
)


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
    (attempt_dir / "results.json").write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "success": status == "completed",
                "model_patch": "",
                "total_time": 1.0,
                "n_iterations": 1,
            }
        ),
        encoding="utf-8",
    )


def _write_resource_evidence(run_dir: Path, instance_id: str, *, valid: bool) -> None:
    (run_dir / instance_id / "attempt_1" / "resource_observations.json").write_text(
        json.dumps(
            {
                "telemetry_quality": "ok" if valid else "unavailable",
                "collection_validity": "valid" if valid else "invalid",
                "cleanup": "ok" if valid else "not_started",
            }
        ),
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


def test_resume_rejects_terminal_attempt_with_invalid_resource_evidence(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_manifest(run_dir, "task-invalid", status="completed")
    _write_resource_evidence(run_dir, "task-invalid", valid=False)
    _write_manifest(run_dir, "task-valid", status="completed")
    _write_resource_evidence(run_dir, "task-valid", valid=True)
    _write_manifest(run_dir, "task-malformed", status="completed")
    (
        run_dir / "task-malformed" / "attempt_1" / "resource_observations.json"
    ).write_text("null", encoding="utf-8")

    assert load_completed_ids(run_dir) == {"task-valid"}
    assert set(load_terminal_results(run_dir)) == {"task-valid"}


def test_resume_rebuilds_and_merges_complete_results_index(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_manifest(run_dir, "task-a", status="completed", exit_status="completed")
    prior = load_terminal_results(run_dir)
    current = CollectedTaskResult(
        instance_id="task-b",
        attempt_dir=run_dir / "task-b" / "attempt_1",
        success=False,
        exit_status="error",
        error="retryable infrastructure failure",
    )

    results_path = run_dir / "results.jsonl"
    write_merged_results_jsonl(
        [{"instance_id": "task-b"}, {"instance_id": "task-a"}],
        prior,
        [current],
        results_path,
    )

    rows = [json.loads(line) for line in results_path.read_text().splitlines()]
    assert [row["instance_id"] for row in rows] == ["task-b", "task-a"]
    assert rows[1]["success"] is True
