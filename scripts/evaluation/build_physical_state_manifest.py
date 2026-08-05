#!/usr/bin/env python3
"""Freeze label-free replay inputs for the SQLGlot physical-state experiment."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import sys
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from tool_resource.clause_parser import parse_command_clauses  # noqa: E402
from tool_resource.pytest_semantics import (  # noqa: E402
    is_pytest_invocation,
    parse_pytest,
)
from trace_collect.openclaw_tools import (  # noqa: E402
    source_runtime_artifact_path_from_tool_call,
)

VERSION = "sqlglot-physical-state-manifest-v1"
SEED = 42
EXPECTED_ELIGIBLE = 81
FROZEN_TASK_IDS = (
    "tobymao__sqlglot-4459",
    "tobymao__sqlglot-4004",
    "tobymao__sqlglot-4390",
    "tobymao__sqlglot-4430",
    "tobymao__sqlglot-4165",
    "tobymao__sqlglot-3975",
    "tobymao__sqlglot-4438",
    "tobymao__sqlglot-3891",
    "tobymao__sqlglot-4519",
    "tobymao__sqlglot-4148",
    "tobymao__sqlglot-4696",
    "tobymao__sqlglot-4393",
)
REPLAY_TOOLS = frozenset({"exec", "write_file", "edit_file"})
IGNORED_TOOLS = frozenset(
    {
        "list_dir",
        "read_file",
        "web_search",
        "web_fetch",
        "message",
        "spawn",
        "sessions_yield",
    }
)
_EXIT = re.compile(r"(?:^|\n)Exit code: (-?\d+)\s*$")


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _actions(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        if row.get("type") == "action" and row.get("action_type") == "tool_exec":
            rows.append({"line_number": line_number, **row})
    return rows


def _exec_command(action: Mapping[str, Any]) -> str | None:
    data = action.get("data")
    if not isinstance(data, Mapping) or data.get("tool_name") != "exec":
        return None
    raw = data.get("tool_args")
    if not isinstance(raw, str):
        return None
    args = json.loads(raw)
    command = args.get("command") if isinstance(args, Mapping) else None
    return command if isinstance(command, str) else None


def _exit_code(action: Mapping[str, Any]) -> int | None:
    data = action.get("data")
    if not isinstance(data, Mapping):
        return None
    match = _EXIT.search(str(data.get("tool_result") or ""))
    return int(match.group(1)) if match else None


def _is_successful_pytest(action: Mapping[str, Any]) -> bool:
    command = _exec_command(action)
    if command is None or _exit_code(action) != 0:
        return False
    parsed = parse_command_clauses(command)
    clauses = parsed.get("clauses") if isinstance(parsed, Mapping) else None
    if parsed.get("parse_failed") or not isinstance(clauses, list) or len(clauses) != 1:
        return False
    argv = tuple(str(value) for value in clauses[0].get("argv", ()))
    return is_pytest_invocation(argv) and parse_pytest(argv) is not None


def _valid_traces(trace_root: Path) -> list[Path]:
    traces = []
    for path in sorted(trace_root.glob("*/attempt_*/trace.jsonl")):
        attempt = path.parent
        observations = json.loads(
            (attempt / "resource_observations.json").read_text(encoding="utf-8")
        )
        result = json.loads((attempt / "results.json").read_text(encoding="utf-8"))
        if observations.get("collection_validity") == "valid" and result.get("success") is True:
            traces.append(path)
    task_ids = [path.parent.parent.name for path in traces]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("more than one valid trace exists for a task")
    return traces


def build_manifest(
    trace_root: Path,
    tasks_path: Path,
    *,
    seed: int = SEED,
    expected_eligible: int = EXPECTED_ELIGIBLE,
    expected_selected: Sequence[str] = FROZEN_TASK_IDS,
    repo_root: Path = _ROOT,
) -> dict[str, Any]:
    eligible: dict[str, tuple[Path, list[dict[str, Any]], int]] = {}
    for trace in _valid_traces(trace_root):
        actions = _actions(trace)
        target_index = next(
            (index for index, action in enumerate(actions) if _is_successful_pytest(action)),
            None,
        )
        if target_index is not None:
            eligible[trace.parent.parent.name] = (trace, actions, target_index)
    if len(eligible) != expected_eligible:
        raise ValueError(
            f"eligible task count changed: {len(eligible)} != {expected_eligible}"
        )

    selected = sorted(eligible)
    random.Random(seed).shuffle(selected)
    selected = selected[: len(expected_selected)]
    if tuple(selected) != tuple(expected_selected):
        raise ValueError("seeded task selection differs from the frozen protocol")

    task_rows = json.loads(tasks_path.read_text(encoding="utf-8"))
    images = {
        str(row["instance_id"]): str(row.get("image_name") or row["docker_image"])
        for row in task_rows
        if isinstance(row, Mapping) and row.get("instance_id") in selected
    }
    if set(images) != set(selected):
        raise ValueError("selected task image metadata is incomplete")

    manifest_tasks = []
    for task_id in selected:
        trace, actions, target_index = eligible[task_id]
        target = actions[target_index]
        prefix = actions[:target_index]
        counts = Counter(str(action["data"].get("tool_name")) for action in prefix)
        unknown = set(counts) - REPLAY_TOOLS - IGNORED_TOOLS
        if unknown:
            raise ValueError(f"unsupported prefix tools for {task_id}: {sorted(unknown)}")
        replay = [
            action for action in prefix if action["data"].get("tool_name") in REPLAY_TOOLS
        ]
        for action in replay:
            data = action["data"]
            artifact = source_runtime_artifact_path_from_tool_call(
                tool_name=str(data["tool_name"]),
                tool_args_json=str(data["tool_args"]),
            )
            if artifact is not None:
                raise ValueError(
                    f"prefix action {action['action_id']} needs runtime artifact {artifact}"
                )
        command = _exec_command(target)
        if command is None:
            raise AssertionError("selected target is not an exec command")
        manifest_tasks.append(
            {
                "task_id": task_id,
                "image": images[task_id],
                "source_trace": _relative(trace, repo_root),
                "target_action_id": target["action_id"],
                "target_trace_line": target["line_number"],
                "target_command": command,
                "target_source_duration_ms": target["data"].get("duration_ms"),
                "prefix_tool_actions": len(prefix),
                "prefix_replay_action_ids": [action["action_id"] for action in replay],
                "prefix_replay_source_duration_ms": sum(
                    float(action["data"].get("duration_ms") or 0.0) for action in replay
                ),
                "prefix_tool_counts": dict(sorted(counts.items())),
            }
        )
    return {
        "schema": VERSION,
        "selection": {
            "seed": seed,
            "eligible_tasks": len(eligible),
            "selected_tasks": len(selected),
            "criterion": "first successful single-clause frozen-PytestSignature command",
            "resource_labels_read": False,
        },
        "inputs": {
            "trace_root": _relative(trace_root, repo_root),
            "tasks": _relative(tasks_path, repo_root),
        },
        "prefix_replay": {
            "replayed_tools": sorted(REPLAY_TOOLS),
            "ignored_nonmutating_tools": sorted(IGNORED_TOOLS),
            "unknown_tools_fail_closed": True,
            "source_runtime_artifacts_allowed": False,
        },
        "tasks": manifest_tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("output already exists")
    manifest = build_manifest(args.trace_root, args.tasks)
    args.out.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
