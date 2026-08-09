import json
from pathlib import Path

from scripts.evaluation.evaluate_doc_tool_semantics import (
    build_split_manifest,
    census_attempts,
    traced_task_ids,
)


def _tasks() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for repo, prefix, count in (
        ("tobymao/sqlglot", "sql", 4),
        ("PennyLaneAI/pennylane", "pl", 4),
        ("iterative/dvc", "dvc", 5),
    ):
        for index in range(count):
            rows.append(
                {
                    "repo": repo,
                    "instance_id": f"{prefix}-{index}",
                    "created_at": f"2026-01-{count - index:02d} 00:00:00",
                    "version": str(index),
                }
            )
    return rows


def test_split_freeze_quarantines_directory_names_and_orders_metadata(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "traces"
    quarantined = trace_root / "old-run" / "pl-1"
    quarantined.mkdir(parents=True)
    (quarantined / "must-not-be-read").write_text("private outcome")

    traced = traced_task_ids(trace_root, {row["instance_id"] for row in _tasks()})
    manifest = build_split_manifest(
        _tasks(),
        traced,
        sqlglot_development_ids={"sql-0", "sql-1", "sql-2", "sql-3"},
        sizes={
            "sqlglot": (2, 2),
            "pennylane": (1, 2),
            "dvc": (2, 3),
        },
    )

    assert traced == {"pl-1"}
    assert manifest["cohorts"]["sqlglot"] == {
        "development_warmup": ["sql-3", "sql-2"],
        "development_scored": ["sql-1", "sql-0"],
    }
    assert manifest["cohorts"]["pennylane"] == {
        "warmup": ["pl-3"],
        "validation": ["pl-2", "pl-0"],
    }
    assert manifest["cohorts"]["dvc"] == {
        "warmup": ["dvc-4", "dvc-3"],
        "final": ["dvc-2", "dvc-1", "dvc-0"],
    }
    assert manifest["integrity"]["all_split_ids_disjoint"] is True
    assert "private outcome" not in json.dumps(manifest)


def test_census_projects_valid_exec_commands_to_counts_only(tmp_path: Path) -> None:
    attempt = tmp_path / "sql-1" / "attempt-1"
    attempt.mkdir(parents=True)
    (attempt / "resource_observations.json").write_text(
        json.dumps(
            {
                "collection_validity": "valid",
                "workload_execution": "completed",
                "telemetry_quality": "ok",
                "cleanup": "ok",
                "secret_resource_label": 999,
            }
        )
    )
    (attempt / "tool_calls.json").write_text(
        json.dumps(
            [
                {
                    "tool": "exec",
                    "input": {
                        "command": "python -m pip install a b && python -m pytest -q"
                    },
                    "output": "private result",
                },
                {"tool": "read", "input": {"path": "private"}},
            ]
        )
    )

    census = census_attempts([("sql-1", "v1", attempt)])

    assert census == {
        "tasks": 1,
        "repo_versions": ["v1"],
        "exec_commands": 1,
        "parsed_clauses": 2,
        "parse_failures": 0,
        "tools": {
            "git": {"invocations": 0, "tasks": 0},
            "make": {"invocations": 0, "tasks": 0},
            "pip_install": {"invocations": 1, "tasks": 1},
            "pytest": {"invocations": 1, "tasks": 1},
        },
    }
    serialized = json.dumps(census)
    assert "install a b" not in serialized
    assert "private" not in serialized
