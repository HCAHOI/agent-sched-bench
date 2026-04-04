from __future__ import annotations

import json
from pathlib import Path

from trace_collect.openclaw_import import import_openclaw_run


def test_import_openclaw_run_copies_trace_and_results(tmp_path: Path) -> None:
    source_trace = tmp_path / "source-trace.jsonl"
    source_trace.write_text(
        "\n".join(
            [
                json.dumps({"type": "step", "agent_id": "task-1", "step_idx": 0}),
                json.dumps(
                    {
                        "type": "event",
                        "agent_id": "task-1",
                        "event": "skill_load",
                        "category": "SCHEDULING",
                    }
                ),
                json.dumps({"type": "summary", "agent_id": "task-1", "success": True}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source_results = tmp_path / "nanobot-results.jsonl"
    source_results.write_text(
        json.dumps(
            {
                "instance_id": "task-1",
                "trace_file": str(source_trace),
                "model_patch": "diff --git a/foo b/foo",
                "patch_generated": True,
                "official_resolved": True,
                "evaluation_run_id": "eval-1",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    run_dir = import_openclaw_run(
        results_path=source_results,
        output_dir=tmp_path / "traces",
        model_name="Qwen3.6-Plus",
        run_id="demo",
    )

    imported_trace = run_dir / "task-1.jsonl"
    imported_results = run_dir / "results.jsonl"
    preds_path = run_dir / "preds.json"
    manifest_path = run_dir / "import_manifest.json"

    assert imported_trace.exists()
    assert imported_results.exists()
    assert preds_path.exists()
    assert manifest_path.exists()

    rows = [
        json.loads(line)
        for line in imported_results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["instance_id"] == "task-1"
    assert rows[0]["success_basis"] == "official_resolved"
    assert rows[0]["resolved"] is True

    preds = json.loads(preds_path.read_text(encoding="utf-8"))
    assert preds["task-1"]["model_name_or_path"] == "Qwen3.6-Plus"


def test_import_openclaw_run_normalizes_summary_success_to_benchmark_result(
    tmp_path: Path,
) -> None:
    source_trace = tmp_path / "source-trace.jsonl"
    source_trace.write_text(
        "\n".join(
            [
                json.dumps({"type": "step", "agent_id": "task-1", "step_idx": 0}),
                json.dumps({"type": "summary", "agent_id": "task-1", "success": True}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source_results = tmp_path / "nanobot-results.jsonl"
    source_results.write_text(
        json.dumps(
            {
                "instance_id": "task-1",
                "trace_file": str(source_trace),
                "model_patch": "",
                "patch_generated": False,
                "official_resolved": False,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    run_dir = import_openclaw_run(
        results_path=source_results,
        output_dir=tmp_path / "traces",
        model_name="Qwen3.6-Plus",
        run_id="normalized",
    )

    imported_trace = run_dir / "task-1.jsonl"
    rows = [
        json.loads(line)
        for line in imported_trace.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = next(row for row in rows if row["type"] == "summary")

    assert summary["source_success"] is True
    assert summary["success"] is False
    assert summary["success_basis"] == "official_resolved"
    assert summary["official_resolved"] is False
    assert summary["patch_generated"] is False


def test_import_openclaw_run_recovers_malformed_tool_args_from_raw_response(
    tmp_path: Path,
) -> None:
    source_trace = tmp_path / "source-trace.jsonl"
    malformed_tool_args = '{"edit_file": {"path": "/tmp/source/task-1/file.py", "new_text": "unterminated}'
    source_trace.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "step",
                        "agent_id": "task-1",
                        "step_idx": 0,
                        "tool_name": "edit_file",
                        "tool_args": malformed_tool_args,
                        "raw_response": {
                            "choices": [
                                {
                                    "message": {
                                        "tool_calls": [
                                            {
                                                "id": "call_0",
                                                "type": "function",
                                                "function": {
                                                    "name": "edit_file",
                                                    "arguments": json.dumps(
                                                        {
                                                            "path": "/tmp/source/task-1/file.py",
                                                            "old_text": "before\n",
                                                            "new_text": "after\n",
                                                        }
                                                    ),
                                                },
                                            }
                                        ]
                                    }
                                }
                            ]
                        },
                    }
                ),
                json.dumps({"type": "summary", "agent_id": "task-1", "success": False}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source_results = tmp_path / "nanobot-results.jsonl"
    source_results.write_text(
        json.dumps(
            {
                "instance_id": "task-1",
                "trace_file": str(source_trace),
                "model_patch": "",
                "patch_generated": False,
                "official_resolved": False,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    run_dir = import_openclaw_run(
        results_path=source_results,
        output_dir=tmp_path / "traces",
        model_name="Qwen3.6-Plus",
        run_id="recovered",
    )

    imported_trace = run_dir / "task-1.jsonl"
    rows = [
        json.loads(line)
        for line in imported_trace.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    step = next(row for row in rows if row["type"] == "step")
    parsed_args = json.loads(step["tool_args"])

    assert parsed_args["edit_file"]["path"] == "/tmp/source/task-1/file.py"
    assert parsed_args["edit_file"]["new_text"] == "after\n"
