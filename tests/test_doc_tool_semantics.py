import json
from pathlib import Path
import subprocess

import pytest

import scripts.evaluation.evaluate_doc_tool_semantics as doc_tool_semantics
from scripts.evaluation.evaluate_doc_tool_semantics import (
    _generation_status,
    _generated_specs,
    _history_row,
    _raw_prefix_keys,
    _render_generation_prompt,
    _validated_response_text,
    build_split_manifest,
    census_attempts,
    evaluate_doc_semantics,
    traced_task_ids,
)
from scripts.evaluation.evaluate_clause_resource_classes import CommandRow, Row
from tool_resource.tool_spec import validate_tool_spec


def test_raw_prefix_keys_reuse_canonical_shell_tokens() -> None:
    assert _raw_prefix_keys("FOO=1 /usr/bin/python -m pytest -q > out") == (
        "exec:python",
        "exec:python -m",
        "exec:python -m pytest",
        "exec:python -m pytest -q",
    )
    assert _raw_prefix_keys("echo '") == ()


def test_history_row_fills_unavailable_resource_targets() -> None:
    clause = Row(
        task_id="owner__repo-1",
        repo="owner__repo",
        manifest_index=0,
        bin="true",
        argv=("true",),
        latency_ms=1.0,
        peak_cpu_cores=None,
        sampled_peak_rss_mb=None,
        disk_read_write_bytes_total=None,
    )
    command = CommandRow(
        task_id=clause.task_id,
        repo=clause.repo,
        manifest_index=0,
        call_index=0,
        call_id="call_1",
        command="true",
        duration_ms=1.0,
        clauses=(clause,),
    )

    row = _history_row(
        command,
        {"latency": 0},
        {"latency": (1.0, 0.0, 0.0, 0.0, 0.0)},
    )

    assert row.current == {
        "latency": 0,
        "peak_cpu_cores": None,
        "sampled_peak_rss_mb": None,
        "disk_read_write_bytes_total": None,
    }
    assert row.pmfs["latency"] == (1.0, 0.0, 0.0, 0.0, 0.0)
    assert all(row.pmfs[target] is None for target in row.pmfs if target != "latency")


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


def test_generation_prompt_contains_only_the_selected_tool_source() -> None:
    prompt = _render_generation_prompt(
        "tool={{TOOL}} version={{VERSION}}\n{{DOCUMENTATION}}",
        "pytest",
        "8.3.5",
        "official help only",
    )

    assert prompt == "tool=pytest version=8.3.5\nofficial help only"


def test_generated_specs_rejects_a_fabricated_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doc_tool_semantics, "_GENERATION_OUTPUT", tmp_path)
    (tmp_path / "generation-artifact.json").write_text(
        json.dumps(
            {
                "schema": "offline-tool-semantics-generation-v1",
                "preregistration_commit": "fabricated",
                "model": "not-the-frozen-model",
                "requested_service_tier": "fast",
                "reasoning_effort": "medium",
                "temperature": "unsupported_by_codex_provider",
                "codex_version": "codex 1.0",
                "tools": {},
            }
        )
    )

    with pytest.raises(ValueError, match="generation artifact differs"):
        _generated_specs()


def test_generation_status_keeps_an_overbudget_response_unsupported() -> None:
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

    assert (
        _generation_status("pytest", "8.3.5", spec, 100, {"input_tokens": 64_001})
        == "unsupported_structural_failure"
    )


def test_generation_response_rejects_text_different_from_agent_message() -> None:
    with pytest.raises(ValueError, match="response differs"):
        _validated_response_text(
            [
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "{}"},
                }
            ],
            '{"schema":"tool-spec-v1"}',
            "pytest",
        )


def test_generation_records_post_call_validation_failure_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_codex_call(
        prompt: str, schema: dict[str, object], directory: Path, tool: str
    ) -> tuple[dict[str, object], dict[str, object]]:
        del schema
        calls.append(tool)
        response_text = "{}"
        (directory / f"{tool}.response.json").write_text(response_text)
        (directory / f"{tool}.events.jsonl").write_text(
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "type": "agent_message",
                        "text": "mismatch" if tool == "git" else response_text,
                    },
                }
            )
            + "\n"
        )
        return {}, {
            "prompt_bytes": len(prompt.encode()),
            "wall_seconds": 0.0,
            "usage": {
                "input_tokens": 1,
                "cached_input_tokens": 0,
                "output_tokens": 1,
            },
        }

    monkeypatch.setattr(
        doc_tool_semantics, "_GENERATION_OUTPUT", tmp_path / "generation"
    )
    monkeypatch.setattr(
        doc_tool_semantics, "_generation_worktree_head", lambda: "f" * 40
    )
    monkeypatch.setattr(doc_tool_semantics, "_codex_call", fake_codex_call)
    monkeypatch.setattr(
        doc_tool_semantics.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "fixture\n"),
    )

    doc_tool_semantics._generate_specs()

    artifact = json.loads(
        (tmp_path / "generation" / "generation-artifact.json").read_text()
    )
    assert artifact["tools"]["git"]["status"] == "unsupported_generation_failure"
    assert calls == ["git", "make", "pip_install", "pytest"]


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
    assert (
        rows[0]["arms"]["docs_poset"]["probability_by_bucket"]
        == rows[1]["arms"]["docs_poset"]["probability_by_bucket"]
    )
    assert (
        rows[0]["arms"]["generic_poset"]["provenance"]["latency"]["clauses"][0][
            "local_observation_count"
        ]
        == rows[0]["arms"]["docs_poset"]["provenance"]["latency"]["clauses"][0][
            "local_observation_count"
        ]
        == 1
    )
    assert (
        rows[2]["arms"]["docs_poset"]["probability_by_bucket"]
        == rows[2]["arms"]["clause_kb"]["probability_by_bucket"]
    )
    assert (
        rows[3]["arms"]["docs_poset"]["probability_by_bucket"]
        != rows[0]["arms"]["docs_poset"]["probability_by_bucket"]
    )


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
