#!/usr/bin/env python3
"""Freeze and evaluate documentation-derived tool semantics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from tool_resource.clause_parser import parse_command_clauses
from tool_resource.pip_semantics import parse_pip_install
from tool_resource.pytest_semantics import is_pytest_invocation


SCHEMA = "offline-tool-semantics-splits-v1"
REPOSITORIES = {
    "sqlglot": "tobymao/sqlglot",
    "pennylane": "PennyLaneAI/pennylane",
    "dvc": "iterative/dvc",
}
DEFAULT_SIZES = {
    "sqlglot": (100, 100),
    "pennylane": (16, 16),
    "dvc": (32, 33),
}
_VALID_STATUS = {
    "collection_validity": "valid",
    "workload_execution": "completed",
    "telemetry_quality": "ok",
    "cleanup": "ok",
}
_TOOL_NAMES = ("git", "make", "pip_install", "pytest")


def traced_task_ids(trace_root: Path, known_ids: set[str]) -> set[str]:
    """Return task IDs appearing as directory names without opening their files."""

    found: set[str] = set()
    for root, directories, _files in os.walk(trace_root):
        del root
        retained: list[str] = []
        for name in directories:
            if name in known_ids:
                found.add(name)
            else:
                retained.append(name)
        directories[:] = retained
    return found


def _ordered_tasks(
    tasks: Sequence[Mapping[str, Any]], repo: str
) -> list[Mapping[str, Any]]:
    selected = [row for row in tasks if row.get("repo") == repo]
    if any(
        not isinstance(row.get("instance_id"), str)
        or not isinstance(row.get("created_at"), str)
        for row in selected
    ):
        raise ValueError(f"{repo}: task metadata lacks identity or created_at")
    return sorted(selected, key=lambda row: (row["created_at"], row["instance_id"]))


def _slice_exact(ids: Sequence[str], sizes: tuple[int, int], repo: str) -> tuple[list[str], list[str]]:
    expected = sum(sizes)
    if len(ids) != expected:
        raise ValueError(f"{repo}: expected {expected} eligible tasks, found {len(ids)}")
    return list(ids[: sizes[0]]), list(ids[sizes[0] :])


def build_split_manifest(
    tasks: Sequence[Mapping[str, Any]],
    traced_ids: set[str],
    *,
    sqlglot_development_ids: set[str],
    sizes: Mapping[str, tuple[int, int]] = DEFAULT_SIZES,
) -> dict[str, Any]:
    """Freeze deterministic development/validation/final task identities."""

    all_ids = [row.get("instance_id") for row in tasks]
    if any(not isinstance(task_id, str) for task_id in all_ids):
        raise ValueError("task manifest contains a non-string instance_id")
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("task manifest contains duplicate instance IDs")

    ordered = {
        name: _ordered_tasks(tasks, repo) for name, repo in REPOSITORIES.items()
    }
    sql_ids = [
        row["instance_id"]
        for row in ordered["sqlglot"]
        if row["instance_id"] in sqlglot_development_ids
    ]
    if set(sql_ids) != sqlglot_development_ids:
        raise ValueError("SQLGlot development IDs differ from task metadata")
    sql_warmup, sql_scored = _slice_exact(sql_ids, sizes["sqlglot"], "sqlglot")

    fresh: dict[str, list[str]] = {}
    for name in ("pennylane", "dvc"):
        fresh[name] = [
            row["instance_id"]
            for row in ordered[name]
            if row["instance_id"] not in traced_ids
        ]
    pennylane_warmup, pennylane_validation = _slice_exact(
        fresh["pennylane"], sizes["pennylane"], "pennylane"
    )
    dvc_warmup, dvc_final = _slice_exact(fresh["dvc"], sizes["dvc"], "dvc")

    cohorts = {
        "sqlglot": {
            "development_warmup": sql_warmup,
            "development_scored": sql_scored,
        },
        "pennylane": {
            "warmup": pennylane_warmup,
            "validation": pennylane_validation,
        },
        "dvc": {"warmup": dvc_warmup, "final": dvc_final},
    }
    split_ids = [task_id for cohort in cohorts.values() for ids in cohort.values() for task_id in ids]
    return {
        "schema": SCHEMA,
        "cohorts": cohorts,
        "quarantined_task_counts": {
            name: sum(row["instance_id"] in traced_ids for row in ordered[name])
            for name in ("pennylane", "dvc")
        },
        "integrity": {
            "all_split_ids_disjoint": len(split_ids) == len(set(split_ids)),
            "ordering": "created_at_then_instance_id",
            "quarantine": "any_existing_task_directory",
        },
    }


def _tool_name(argv: Sequence[str]) -> str | None:
    if parse_pip_install(argv) is not None:
        return "pip_install"
    if is_pytest_invocation(argv):
        return "pytest"
    if not argv:
        return None
    binary = PurePosixPath(str(argv[0])).name.lower()
    if binary == "git":
        return "git"
    if binary in {"make", "gmake"}:
        return "make"
    return None


def census_attempts(
    attempts: Iterable[tuple[str, str, Path]],
) -> dict[str, Any]:
    """Count documented-tool invocations without retaining commands or outcomes."""

    task_ids: set[str] = set()
    versions: set[str] = set()
    exec_commands = parsed_clauses = parse_failures = 0
    invocations = {name: 0 for name in _TOOL_NAMES}
    tool_tasks: dict[str, set[str]] = defaultdict(set)
    for task_id, version, attempt in attempts:
        status = json.loads((attempt / "resource_observations.json").read_text())
        mismatch = {
            key: status.get(key)
            for key, expected in _VALID_STATUS.items()
            if status.get(key) != expected
        }
        if mismatch:
            raise ValueError(f"{attempt}: invalid collection status {mismatch}")
        task_ids.add(task_id)
        versions.add(version)
        calls = json.loads((attempt / "tool_calls.json").read_text())
        if not isinstance(calls, list):
            raise ValueError(f"{attempt}: tool_calls.json is not an array")
        for call in calls:
            if not isinstance(call, Mapping) or call.get("tool") != "exec":
                continue
            input_value = call.get("input")
            command = input_value.get("command") if isinstance(input_value, Mapping) else None
            if not isinstance(command, str):
                raise ValueError(f"{attempt}: exec call lacks a command")
            exec_commands += 1
            parsed = parse_command_clauses(command)
            if parsed.get("parse_failed"):
                parse_failures += 1
                continue
            clauses = parsed.get("clauses")
            if not isinstance(clauses, list):
                raise ValueError("clause parser returned no clause list")
            parsed_clauses += len(clauses)
            for clause in clauses:
                argv_value = clause.get("argv") if isinstance(clause, Mapping) else None
                if not isinstance(argv_value, list):
                    continue
                name = _tool_name(tuple(str(value) for value in argv_value))
                if name is not None:
                    invocations[name] += 1
                    tool_tasks[name].add(task_id)
    return {
        "tasks": len(task_ids),
        "repo_versions": sorted(versions),
        "exec_commands": exec_commands,
        "parsed_clauses": parsed_clauses,
        "parse_failures": parse_failures,
        "tools": {
            name: {"invocations": invocations[name], "tasks": len(tool_tasks[name])}
            for name in _TOOL_NAMES
        },
    }


def _run_attempts(run_dir: Path, versions: Mapping[str, str]) -> list[tuple[str, str, Path]]:
    attempts: list[tuple[str, str, Path]] = []
    with (run_dir / "results.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            task_id = row.get("instance_id")
            attempt_value = row.get("attempt_dir")
            if row.get("success") is not True or not isinstance(task_id, str) or not isinstance(attempt_value, str):
                raise ValueError(f"{run_dir}: result is not a successful final attempt")
            attempt = Path(attempt_value)
            if not attempt.is_absolute():
                attempt = run_dir / attempt
            attempt = attempt.resolve()
            if not attempt.is_relative_to(run_dir.resolve()):
                raise ValueError(f"{run_dir}: attempt escapes its run directory")
            attempts.append((task_id, versions[task_id], attempt))
    return attempts


def _freeze(args: argparse.Namespace) -> None:
    tasks = json.loads(args.tasks.read_text())
    if not isinstance(tasks, list):
        raise ValueError("task manifest is not an array")
    versions = {
        row["instance_id"]: str(row["version"])
        for row in tasks
        if isinstance(row, Mapping)
        and isinstance(row.get("instance_id"), str)
        and row.get("version") is not None
    }
    attempts = [
        attempt
        for run_dir in args.sqlglot_run
        for attempt in _run_attempts(run_dir.resolve(), versions)
    ]
    development_ids = {task_id for task_id, _version, _attempt in attempts}
    known_ids = {
        row["instance_id"]
        for row in tasks
        if isinstance(row, Mapping) and isinstance(row.get("instance_id"), str)
    }
    result = build_split_manifest(
        tasks,
        traced_task_ids(args.trace_root, known_ids),
        sqlglot_development_ids=development_ids,
    )
    result["development_coverage"] = census_attempts(attempts)
    result["sources"] = {
        "tasks": str(args.tasks.resolve()),
        "trace_root": str(args.trace_root.resolve()),
        "sqlglot_runs": [str(path.resolve()) for path in args.sqlglot_run],
    }
    if args.out.exists():
        raise FileExistsError("split output already exists")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--tasks", type=Path, required=True)
    freeze.add_argument("--trace-root", type=Path, required=True)
    freeze.add_argument("--sqlglot-run", type=Path, action="append", required=True)
    freeze.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        _freeze(args)


if __name__ == "__main__":
    main()
