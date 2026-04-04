from __future__ import annotations

import json
from pathlib import Path

from trace_collect.cli import parse_collect_args
from trace_collect.collector import (
    CollectedTaskResult,
    _rewrite_trace_summary,
    _write_predictions,
)


def test_parse_collect_args_exposes_harness_flags() -> None:
    args = parse_collect_args(
        [
            "--evaluate",
            "--harness-dataset",
            "custom-dataset",
            "--harness-split",
            "dev",
            "--harness-workers",
            "2",
            "--harness-timeout",
            "900",
            "--harness-run-id",
            "demo-run",
            "--harness-report-dir",
            "reports/demo",
            "--harness-namespace",
            "demo-ns",
        ]
    )

    assert args.evaluate is True
    assert args.harness_dataset == "custom-dataset"
    assert args.harness_split == "dev"
    assert args.harness_workers == 2
    assert args.harness_timeout == 900
    assert args.harness_run_id == "demo-run"
    assert args.harness_report_dir == "reports/demo"
    assert args.harness_namespace == "demo-ns"


def test_write_predictions_keeps_only_non_empty_patches(tmp_path: Path) -> None:
    predictions_path = tmp_path / "preds.json"
    count = _write_predictions(
        [
            CollectedTaskResult(
                instance_id="task-1",
                trace_file=tmp_path / "task-1.jsonl",
                success=True,
                success_basis="patch_generated",
                patch_generated=True,
                model_patch="diff --git a/foo b/foo",
            ),
            CollectedTaskResult(
                instance_id="task-2",
                trace_file=tmp_path / "task-2.jsonl",
                success=False,
                success_basis="patch_generated",
                patch_generated=False,
                model_patch="",
            ),
        ],
        model_name="Qwen3.6-Plus",
        predictions_path=predictions_path,
    )

    payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    assert count == 1
    assert list(payload) == ["task-1"]
    assert payload["task-1"]["model_name_or_path"] == "Qwen3.6-Plus"


def test_rewrite_trace_summary_updates_success_basis(tmp_path: Path) -> None:
    trace_path = tmp_path / "task-1.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps({"type": "step", "agent_id": "task-1", "step_idx": 0}),
                json.dumps({"type": "summary", "agent_id": "task-1", "success": True}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = CollectedTaskResult(
        instance_id="task-1",
        trace_file=trace_path,
        success=False,
        success_basis="official_resolved",
        patch_generated=True,
        model_patch="diff --git a/foo b/foo",
        exit_status="Submitted",
        official_resolved=False,
        evaluation_run_id="run-1",
        evaluation_report_path=str(tmp_path / "report.json"),
    )

    _rewrite_trace_summary(trace_path, result)

    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = rows[-1]
    assert summary["type"] == "summary"
    assert summary["success"] is False
    assert summary["success_basis"] == "official_resolved"
    assert summary["patch_generated"] is True
    assert summary["official_resolved"] is False
    assert summary["evaluation_run_id"] == "run-1"
