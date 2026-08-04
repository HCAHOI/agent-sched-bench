#!/usr/bin/env python3
"""Score command predictions after a causal same-repository warm-up."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from scripts.evaluation.evaluate_clause_latency_buckets import (  # noqa: E402
    PipExecEvent,
    evaluate_full_test_agent_state,
    evaluate_full_test_phase,
    evaluate_interaction_commands,
    evaluate_pip_resources,
    evaluate_pip_semantics,
    evaluate_poset_resources,
    evaluate_prequential_commands,
    evaluate_pytest_semantics,
)
from scripts.evaluation.evaluate_clause_resource_classes import (  # noqa: E402
    load_rows,
    load_run_rows,
)
from tool_resource_eval.labels import repo_of  # noqa: E402


def _load_exec_events(
    run_dir: Path,
    expected_task_ids: list[str],
) -> dict[str, list[PipExecEvent]]:
    """Read raw exec commands and outputs from the accepted final attempts."""

    run_dir = run_dir.resolve()
    events: dict[str, list[PipExecEvent]] = {}
    with (run_dir / "results.jsonl").open(encoding="utf-8") as handle:
        for record in map(json.loads, handle):
            task_id = record.get("instance_id")
            attempt_value = record.get("attempt_dir")
            if not isinstance(task_id, str) or not isinstance(attempt_value, str):
                raise ValueError("results.jsonl lacks a task or attempt path")
            attempt_dir = Path(attempt_value)
            if not attempt_dir.is_absolute():
                attempt_dir = run_dir / attempt_dir
            attempt_dir = attempt_dir.resolve()
            if not attempt_dir.is_relative_to(run_dir):
                raise ValueError(f"attempt path escapes the run: {attempt_dir}")
            task_events: list[PipExecEvent] = []
            with (attempt_dir / "trace.jsonl").open(encoding="utf-8") as trace:
                for line in trace:
                    action = json.loads(line)
                    data = action.get("data")
                    if (
                        action.get("type") != "action"
                        or action.get("action_type") != "tool_exec"
                        or not isinstance(data, dict)
                        or data.get("tool_name") != "exec"
                    ):
                        continue
                    call_id = data.get("tool_call_id")
                    tool_args = data.get("tool_args")
                    tool_result = data.get("tool_result")
                    if not all(isinstance(value, str) for value in (call_id, tool_args, tool_result)):
                        raise ValueError(f"{attempt_dir}: raw exec action is incomplete")
                    arguments = json.loads(tool_args)
                    command = arguments.get("command") if isinstance(arguments, dict) else None
                    if not isinstance(command, str):
                        raise ValueError(f"{attempt_dir}: raw exec action lacks command")
                    ts_start = action.get("ts_start")
                    ts_end = action.get("ts_end")
                    if (
                        not isinstance(ts_start, (int, float))
                        or isinstance(ts_start, bool)
                        or not isinstance(ts_end, (int, float))
                        or isinstance(ts_end, bool)
                        or ts_end < ts_start
                    ):
                        raise ValueError(f"{attempt_dir}: raw exec timestamps are invalid")
                    task_events.append(
                        PipExecEvent(call_id, command, tool_result, float(ts_start), float(ts_end))
                    )
            events[task_id] = task_events
    if list(events) != expected_task_ids:
        raise ValueError("raw exec event tasks differ from accepted task order")
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--public-telemetry", type=Path, action="append", required=True)
    parser.add_argument("--exclude-public-repo", action="append", default=[])
    parser.add_argument("--warmup-tasks", type=int, default=80)
    parser.add_argument(
        "--interaction-architectures",
        action="store_true",
        help="score the full causal stream with the two frozen non-trie candidates",
    )
    parser.add_argument(
        "--poset-resources-after-latency",
        type=Path,
        help="score resources only when this latency result contains a poset GO",
    )
    parser.add_argument(
        "--pip-semantics",
        action="store_true",
        help="score the frozen pip semantic and task-state latency arms",
    )
    parser.add_argument(
        "--pip-resources-after-latency",
        type=Path,
        help="transfer pip semantics when this matching latency result passed",
    )
    parser.add_argument(
        "--pytest-semantics",
        action="store_true",
        help="score the frozen SQLGlot-local pytest work signature",
    )
    parser.add_argument(
        "--full-test-phase",
        action="store_true",
        help="score the task-local third-or-later full-test correction",
    )
    parser.add_argument(
        "--full-test-agent-states",
        type=Path,
        help="score frozen blind agent states over the full-test-phase arm",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dump-rows", type=Path, required=True)
    args = parser.parse_args()

    task_ids, clauses, commands = load_run_rows(args.run_dir)
    excluded = {repo_of(task_id) for task_id in task_ids} | set(
        args.exclude_public_repo
    )
    public = [row for path in args.public_telemetry for row in load_rows(path)]
    raw_public_count = len(public)
    public = [row for row in public if row.repo not in excluded]
    overlap = {row.task_id for row in public} & set(task_ids)
    if not public or overlap:
        raise ValueError(
            "public evidence is empty or overlaps target tasks: "
            f"{sorted(overlap)[:3]}"
        )
    provenance = {
        "target_run_dir": str(args.run_dir.resolve()),
        "public_telemetry": [str(path.resolve()) for path in args.public_telemetry],
        "public_excluded_repositories": sorted(excluded),
        "public_clause_observations_before_repo_filter": raw_public_count,
        "public_clause_observations_after_repo_filter": len(public),
        "public_structure_unknown": sum(not row.structure_known for row in public),
        "public_online_eligible_clause_observations": sum(
            row.structure_known and row.pipeline_position <= 0 for row in public
        ),
        "public_structure_recovery": (
            "current parser validated against static_word_intent; earliest/latest "
            "ordered alignment must identify one static clause"
        ),
        "target_task_order_source": "results.jsonl successful final attempts",
    }
    selected_modes = sum(
        bool(value)
        for value in (
            args.interaction_architectures,
            args.poset_resources_after_latency,
            args.pip_semantics,
            args.pip_resources_after_latency,
            args.pytest_semantics,
            args.full_test_phase,
            args.full_test_agent_states,
        )
    )
    if selected_modes > 1:
        raise ValueError("evaluation modes are mutually exclusive")
    if args.poset_resources_after_latency:
        gate_path = args.poset_resources_after_latency.resolve()
        result, rows = evaluate_poset_resources(
            public,
            task_ids,
            clauses,
            commands,
            {**provenance, "latency_gate_result": str(gate_path)},
            json.loads(gate_path.read_text(encoding="utf-8")),
        )
    elif args.interaction_architectures:
        result, rows = evaluate_interaction_commands(
            public,
            task_ids,
            clauses,
            commands,
            provenance,
        )
    elif args.pip_semantics:
        result, rows = evaluate_pip_semantics(
            public,
            task_ids,
            clauses,
            commands,
            _load_exec_events(args.run_dir, task_ids),
            provenance,
            expected_pip_baseline=(119, 97),
        )
    elif args.pip_resources_after_latency:
        gate_path = args.pip_resources_after_latency.resolve()
        result, rows = evaluate_pip_resources(
            public,
            task_ids,
            clauses,
            commands,
            _load_exec_events(args.run_dir, task_ids),
            {**provenance, "latency_gate_result": str(gate_path)},
            json.loads(gate_path.read_text(encoding="utf-8")),
            expected_pip_baseline=(119, 97),
        )
    elif args.pytest_semantics:
        result, rows = evaluate_pytest_semantics(
            public,
            task_ids,
            clauses,
            commands,
            provenance,
            expected_current=(1792, 1568),
            expected_coverage=(398, 339, 88, 125),
        )
    elif args.full_test_phase:
        result, rows = evaluate_full_test_phase(
            public,
            task_ids,
            clauses,
            commands,
            _load_exec_events(args.run_dir, task_ids),
            provenance,
            warmup_task_count=args.warmup_tasks,
        )
    elif args.full_test_agent_states:
        state_path = args.full_test_agent_states.resolve()
        result, rows = evaluate_full_test_agent_state(
            public,
            task_ids,
            clauses,
            commands,
            _load_exec_events(args.run_dir, task_ids),
            json.loads(state_path.read_text(encoding="utf-8")),
            {**provenance, "agent_state_artifact": str(state_path)},
            warmup_task_count=args.warmup_tasks,
        )
    else:
        result, rows = evaluate_prequential_commands(
            public,
            task_ids,
            clauses,
            commands,
            provenance,
            warmup_task_count=args.warmup_tasks,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.dump_rows.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.dump_rows.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


if __name__ == "__main__":
    main()
