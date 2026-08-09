import json
from pathlib import Path

from scripts.evaluation.evaluate_doc_tool_semantics import (
    build_split_manifest,
    census_attempts,
    evaluate_doc_semantics,
    traced_task_ids,
)
from scripts.evaluation.evaluate_clause_resource_classes import CommandRow, Row
from tool_resource.tool_spec import validate_tool_spec


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


def test_causal_evaluator_keeps_rows_and_falls_back_for_unsupported_tools() -> None:
    spec = validate_tool_spec(
        {
            "schema": "tool-spec-v1",
            "tool": "pytest",
            "documented_version": "8.3.5",
            "invocations": [
                {"tokens": ["pytest"], "operation": "run"},
                {"tokens": ["python", "-m", "pytest"], "operation": "run"},
            ],
            "operations": [
                {
                    "name": "run",
                    "arguments": [],
                    "positionals": {
                        "id": "targets",
                        "role": "work_item",
                        "min_items": 1,
                        "max_items": 8,
                    },
                }
            ],
            "relations": [
                {
                    "kind": "unordered_collection",
                    "operation": "run",
                    "argument": "targets",
                }
            ],
        }
    )
    assert spec is not None

    def clause(
        task: int,
        call: int,
        command: str,
        argv: tuple[str, ...],
        *,
        latency: float,
    ) -> tuple[Row, CommandRow]:
        row = Row(
            task_id=f"target__repo-{task}",
            repo="target__repo",
            manifest_index=task - 1,
            bin=argv[0],
            argv=argv,
            latency_ms=latency,
            peak_cpu_cores=3.0 if latency > 500 else 1.0,
            sampled_peak_rss_mb=600.0 if latency > 500 else 100.0,
            disk_read_write_bytes_total=2_000_000.0 if latency > 500 else 0.0,
        )
        return row, CommandRow(
            task_id=row.task_id,
            repo=row.repo,
            manifest_index=row.manifest_index,
            call_index=call,
            call_id=f"call-{task}-{call}",
            command=command,
            duration_ms=latency,
            clauses=(row,),
        )

    pairs = [
        clause(1, 0, "pytest a.py", ("pytest", "a.py"), latency=100.0),
        clause(
            1,
            1,
            "pytest -n 2 a.py",
            ("pytest", "-n", "2", "a.py"),
            latency=9_000.0,
        ),
        clause(
            2,
            0,
            "pytest b.py",
            ("pytest", "b.py"),
            latency=9_000.0,
        ),
        clause(
            2,
            1,
            "pytest b.py",
            ("pytest", "b.py"),
            latency=9_000.0,
        ),
        clause(2, 2, "echo hi", ("echo", "hi"), latency=100.0),
        clause(
            3,
            0,
            "pytest b.py",
            ("pytest", "b.py"),
            latency=9_000.0,
        ),
    ]
    public = [
        Row(
            task_id="public__repo-1",
            repo="public__repo",
            manifest_index=0,
            bin="pytest",
            argv=("pytest", "a.py"),
            latency_ms=100.0,
            peak_cpu_cores=1.0,
            sampled_peak_rss_mb=100.0,
            disk_read_write_bytes_total=0.0,
        )
    ]

    result, rows = evaluate_doc_semantics(
        public,
        ["target__repo-1", "target__repo-2", "target__repo-3"],
        [row for row, _command in pairs],
        [command for _row, command in pairs],
        {"pytest": spec},
        warmup_task_count=1,
        provenance={"fixture": True},
        allow_unverified_versions=True,
        events_by_task={
            "target__repo-1": [],
            "target__repo-2": [],
            "target__repo-3": [],
        },
    )

    assert result["row_identity"]["identical_command_ids_and_labels"] is True
    assert set(rows[0]["arms"]) == {
        "majority",
        "raw_prefix",
        "clause_kb",
        "task_aware",
        "generic_poset",
        "docs_poset",
    }
    assert rows[0]["arms"]["docs_poset"]["probability_by_bucket"] == rows[1][
        "arms"
    ]["docs_poset"]["probability_by_bucket"]
    assert rows[0]["arms"]["generic_poset"]["provenance"]["latency"][
        "clauses"
    ][0]["local_observation_count"] == rows[0]["arms"]["docs_poset"][
        "provenance"
    ]["latency"]["clauses"][0]["local_observation_count"] == 1
    assert rows[2]["arms"]["docs_poset"]["probability_by_bucket"] == rows[2][
        "arms"
    ]["clause_kb"]["probability_by_bucket"]
    assert rows[3]["arms"]["docs_poset"]["probability_by_bucket"] != rows[0][
        "arms"
    ]["docs_poset"]["probability_by_bucket"]


def test_raw_prefix_miss_uses_frozen_public_fallback() -> None:
    spec = validate_tool_spec(
        {
            "schema": "tool-spec-v1",
            "tool": "pytest",
            "documented_version": "8.3.5",
            "invocations": [{"tokens": ["pytest"], "operation": "run"}],
            "operations": [
                {
                    "name": "run",
                    "arguments": [],
                    "positionals": {
                        "id": "targets",
                        "role": "work_item",
                        "min_items": 1,
                        "max_items": 8,
                    },
                }
            ],
            "relations": [],
        }
    )
    assert spec is not None

    def observed(
        task_id: str,
        manifest_index: int,
        bin_: str,
        argv: tuple[str, ...],
        latency: float,
    ) -> Row:
        return Row(
            task_id=task_id,
            repo="target__repo" if task_id.startswith("target") else "public__repo",
            manifest_index=manifest_index,
            bin=bin_,
            argv=argv,
            latency_ms=latency,
            peak_cpu_cores=1.0,
            sampled_peak_rss_mb=100.0,
            disk_read_write_bytes_total=0.0,
        )

    warmup = observed("target__repo-1", 0, "echo", ("echo", "hi"), 9_000.0)
    cd = observed("target__repo-2", 1, "cd", ("cd", "/tmp"), 100.0)
    echo = observed("target__repo-2", 1, "echo", ("echo", "hi"), 100.0)
    commands = [
        CommandRow(
            warmup.task_id,
            warmup.repo,
            0,
            0,
            "warmup",
            "echo hi",
            9_000.0,
            (warmup,),
        ),
        CommandRow(
            cd.task_id,
            cd.repo,
            1,
            0,
            "scored",
            "cd /tmp && echo hi",
            100.0,
            (cd, echo),
        ),
    ]
    public = [
        observed("public__repo-1", 0, "cd", ("cd", "/tmp"), 100.0),
        observed("public__repo-1", 0, "echo", ("echo", "hi"), 100.0),
        observed("public__repo-1", 0, "pytest", ("pytest", "a.py"), 100.0),
    ]

    _result, rows = evaluate_doc_semantics(
        public,
        ["target__repo-1", "target__repo-2"],
        [warmup, cd, echo],
        commands,
        {"pytest": spec},
        warmup_task_count=1,
        provenance={"fixture": True},
        allow_unverified_versions=True,
        events_by_task={"target__repo-1": [], "target__repo-2": []},
    )

    assert rows[0]["arms"]["raw_prefix"]["prediction"]["latency"] == 0
    assert rows[0]["arms"]["clause_kb"]["prediction"]["latency"] == 3
