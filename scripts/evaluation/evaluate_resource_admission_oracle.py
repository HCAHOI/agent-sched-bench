#!/usr/bin/env python3
"""Evaluate the frozen CPU/RSS command-admission action-space oracle."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from scripts.evaluation.evaluate_clause_resource_classes import (  # noqa: E402
    CommandRow,
    load_run_rows,
)
from tool_resource.runtime_kb import (  # noqa: E402
    _command_stages,
)
from tool_resource_eval.cachewise_kv_factorial import (  # noqa: E402
    LOAD,
    SEEDS,
    _bootstrap,
)
from tool_resource_eval.resource_admission import (  # noqa: E402
    AdmissionCommand,
    AdmissionProgram,
    simulate_admission,
)
from trace_collect.trace_data import TraceData  # noqa: E402


VERSION = "resource-admission-oracle-v1"
CPU_CAPACITY = 8.0
RSS_CAPACITY_MB = 16_000.0
MINIMUM_MAKESPAN_REDUCTION = 0.10
MINIMUM_OVERLAP_COMMANDS = 20
MINIMUM_OVERLAP_TASKS = 10
RUN = (
    _ROOT
    / "traces/swe-rebench/gpt-5.6-sol"
    / "sqlglot-100-c2-fast-requested-ebpf-a0419d9-20260803"
)


def _reservation(row: CommandRow) -> tuple[float, float, str]:
    if any(not clause.structure_known for clause in row.clauses):
        return CPU_CAPACITY, RSS_CAPACITY_MB, "structure_full_fallback"
    stages = _command_stages(
        [
            {
                "in_pipe": clause.in_pipe,
                "in_subst": clause.in_subst,
                "pipeline_position": clause.pipeline_position,
            }
            for clause in row.clauses
        ]
    )
    if stages is None:
        return CPU_CAPACITY, RSS_CAPACITY_MB, "structure_full_fallback"

    cpu_full = any(clause.peak_cpu_cores is None for clause in row.clauses)
    rss_full = any(clause.sampled_peak_rss_mb is None for clause in row.clauses)
    cpu = [float(clause.peak_cpu_cores) for clause in row.clauses if clause.peak_cpu_cores is not None]
    rss = [float(clause.sampled_peak_rss_mb) for clause in row.clauses if clause.sampled_peak_rss_mb is not None]

    cpu_reservation = (
        CPU_CAPACITY
        if cpu_full
        else min(
            CPU_CAPACITY,
            max(sum(cpu[index] for index in stage) for stage in stages),
        )
    )
    rss_reservation = (
        RSS_CAPACITY_MB
        if rss_full
        else min(
            RSS_CAPACITY_MB,
            max(sum(rss[index] for index in stage) for stage in stages),
        )
    )
    if cpu_full and rss_full:
        source = "cpu_rss_null_target_fallback"
    elif cpu_full:
        source = "cpu_null_target_fallback"
    elif rss_full:
        source = "rss_null_target_fallback"
    else:
        source = "observed_upper_bound"
    return cpu_reservation, rss_reservation, source


def _run_records() -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in (RUN / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if len(records) != 100 or any(record.get("success") is not True for record in records):
        raise ValueError("admission oracle requires the frozen successful SQLGlot100 run")
    return records


def _artifact_paths(
    records: list[dict[str, Any]],
) -> tuple[dict[str, Path], dict[tuple[str, str], int]]:
    traces: dict[str, Path] = {}
    raw_clause_counts: dict[tuple[str, str], int] = {}
    for record in records:
        task_id = record.get("instance_id")
        attempt_value = record.get("attempt_dir")
        if not isinstance(task_id, str) or not isinstance(attempt_value, str):
            raise ValueError("collector result lacks task or attempt identity")
        attempt = Path(attempt_value)
        if not attempt.is_absolute():
            attempt = RUN / attempt
        attempt = attempt.resolve()
        if not attempt.is_relative_to(RUN.resolve()) or attempt.parent.name != task_id:
            raise ValueError(f"attempt path does not belong to {task_id}")
        trace = attempt / "trace.jsonl"
        if task_id in traces or not trace.is_file():
            raise ValueError(f"duplicate task or missing trace: {task_id}")
        traces[task_id] = trace
        artifact = json.loads(
            (attempt / "resource_observations.json").read_text(encoding="utf-8")
        )
        for call in artifact.get("calls", []):
            if not isinstance(call, Mapping) or call.get("eligible_for_kb") is not True:
                continue
            call_id = call.get("tool_call_id")
            clauses = call.get("clauses")
            if isinstance(call_id, str) and isinstance(clauses, list):
                raw_clause_counts[(task_id, call_id)] = len(clauses)
    return traces, raw_clause_counts


def _program(
    task_id: str,
    trace_path: Path,
    command_rows: Mapping[tuple[str, str], CommandRow],
    raw_clause_counts: Mapping[tuple[str, str], int],
    source_counts: Counter[str],
) -> AdmissionProgram:
    trace = TraceData.load(trace_path)
    actions = sorted(trace.actions, key=lambda action: float(action["ts_start"]))
    if not actions:
        raise ValueError(f"{trace_path}: trace has no actions")
    exec_actions = [
        action
        for action in actions
        if action.get("action_type") == "tool_exec"
        and isinstance(action.get("data"), Mapping)
        and action["data"].get("tool_name") == "exec"
    ]
    if not exec_actions:
        raise ValueError(f"{trace_path}: task has no exec commands")
    task_start = min(float(action["ts_start"]) for action in actions)
    task_end = max(float(action["ts_end"]) for action in actions)
    commands: list[AdmissionCommand] = []
    for index, action in enumerate(exec_actions):
        data = action["data"]
        call_id = data.get("tool_call_id")
        start = float(action["ts_start"])
        end = float(action["ts_end"])
        if not isinstance(call_id, str) or not math.isfinite(start + end) or end <= start:
            raise ValueError(f"{trace_path}: invalid exec action")
        next_start = (
            float(exec_actions[index + 1]["ts_start"])
            if index + 1 < len(exec_actions)
            else end
        )
        if next_start < end - 1e-6:
            raise ValueError(f"{trace_path}: exec actions overlap within one task")
        key = (task_id, call_id)
        row = command_rows.get(key)
        if row is None or raw_clause_counts.get(key) != len(row.clauses):
            cpu, rss, source = CPU_CAPACITY, RSS_CAPACITY_MB, "unmatched_full_fallback"
        else:
            cpu, rss, source = _reservation(row)
        source_counts[source] += 1
        commands.append(
            AdmissionCommand(
                command_id=f"{task_id}:{call_id}",
                duration_s=end - start,
                cpu_cores=cpu,
                rss_mb=rss,
                delay_after_s=max(0.0, next_start - end),
            )
        )
    first_start = float(exec_actions[0]["ts_start"])
    last_end = float(exec_actions[-1]["ts_end"])
    return AdmissionProgram(
        task_id=task_id,
        initial_delay_s=max(0.0, first_start - task_start),
        commands=tuple(commands),
        tail_s=max(0.0, task_end - last_end),
    )


def run() -> dict[str, Any]:
    records = _run_records()
    task_ids, _clauses, rows = load_run_rows(RUN)
    if len(task_ids) != 100:
        raise ValueError("aggregate loader and frozen task pool differ")
    command_rows = {(row.task_id, row.call_id): row for row in rows}
    if len(command_rows) != len(rows):
        raise ValueError("eligible command rows contain duplicate call IDs")
    traces, raw_clause_counts = _artifact_paths(records)
    if set(task_ids) != set(traces):
        raise ValueError("trace and aggregate task pools differ")

    source_counts: Counter[str] = Counter()
    programs = {
        task_id: _program(
            task_id,
            traces[task_id],
            command_rows,
            raw_clause_counts,
            source_counts,
        )
        for task_id in task_ids
    }
    schedule_results = []
    overlap_ids: set[str] = set()
    for seed in SEEDS:
        selected = sorted(programs)
        np.random.default_rng(seed).shuffle(selected)
        selected = selected[:LOAD]
        chosen = [programs[task_id] for task_id in selected]
        control = simulate_admission(
            chosen,
            cpu_capacity=CPU_CAPACITY,
            rss_capacity_mb=RSS_CAPACITY_MB,
            fixed_high=True,
        )
        oracle = simulate_admission(
            chosen,
            cpu_capacity=CPU_CAPACITY,
            rss_capacity_mb=RSS_CAPACITY_MB,
            fixed_high=False,
        )
        if (
            control["command_count"] != oracle["command_count"]
            or control["total_command_service_s"] != oracle["total_command_service_s"]
        ):
            raise ValueError("admission arms evaluated different commands or durations")
        overlap_ids.update(str(value) for value in oracle.pop("overlapped_command_ids"))
        control.pop("overlapped_command_ids")
        schedule_results.append(
            {"seed": seed, "task_ids": selected, "arms": {"fixed_high": control, "oracle": oracle}}
        )

    metrics = (
        "makespan_s",
        "mean_task_completion_s",
        "total_command_queue_s",
        "reserved_cpu_core_s",
        "reserved_rss_mb_s",
        "max_concurrent_commands",
    )
    means = {
        arm: {
            metric: float(
                np.mean([row["arms"][arm][metric] for row in schedule_results])
            )
            for metric in metrics
        }
        for arm in ("fixed_high", "oracle")
    }
    deltas = [
        float(row["arms"]["oracle"]["makespan_s"])
        - float(row["arms"]["fixed_high"]["makespan_s"])
        for row in schedule_results
    ]
    reduction = (
        means["fixed_high"]["makespan_s"] - means["oracle"]["makespan_s"]
    ) / means["fixed_high"]["makespan_s"]
    comparison = {
        "candidate": "oracle",
        "baseline": "fixed_high",
        "metric": "mean batch makespan_s; lower is better",
        "relative_reduction_of_means": reduction,
        **_bootstrap(deltas),
    }
    overlap_tasks = {command_id.split(":", 1)[0] for command_id in overlap_ids}
    identical = all(
        row["arms"]["fixed_high"]["command_count"]
        == row["arms"]["oracle"]["command_count"]
        and row["arms"]["fixed_high"]["total_command_service_s"]
        == row["arms"]["oracle"]["total_command_service_s"]
        for row in schedule_results
    )
    no_violation = not any(
        bool(row["arms"][arm]["capacity_violation"])
        for row in schedule_results
        for arm in ("fixed_high", "oracle")
    )
    gate = {
        "makespan_reduction_at_least_10_percent": reduction
        >= MINIMUM_MAKESPAN_REDUCTION,
        "paired_ci_below_zero": comparison["ci95_paired_seed_bootstrap"][1] < 0.0,
        "at_least_20_overlapped_commands": len(overlap_ids)
        >= MINIMUM_OVERLAP_COMMANDS,
        "at_least_10_overlap_tasks": len(overlap_tasks) >= MINIMUM_OVERLAP_TASKS,
        "identical_commands_and_durations": identical,
        "no_capacity_violation": no_violation,
    }
    gate["go"] = all(gate.values())
    return {
        "schema": VERSION,
        "status": (
            "development_go_to_predictor_admission"
            if gate["go"]
            else "development_no_go_close_command_admission"
        ),
        "claim_bearing": False,
        "protocol": {
            "task_pool": "exposed SQLGlot100",
            "scheduler": "FCFS-ready with work-conserving backfill",
            "load": LOAD,
            "seeds": list(SEEDS),
            "cpu_capacity": CPU_CAPACITY,
            "rss_capacity_mb": RSS_CAPACITY_MB,
            "control": "full CPU and RSS host reservation per exec",
            "oracle": "canonical clause-composed telemetry upper bound with conservative null fallback",
            "primary": "mean batch makespan_s; lower is better",
            "minimum_relative_reduction": MINIMUM_MAKESPAN_REDUCTION,
        },
        "coverage": {
            "tasks": len(programs),
            "commands": sum(len(program.commands) for program in programs.values()),
            "reservation_sources": dict(sorted(source_counts.items())),
            "distinct_overlapped_commands": len(overlap_ids),
            "distinct_overlap_tasks": len(overlap_tasks),
        },
        "mean_metrics": means,
        "primary_comparison": comparison,
        "gate": gate,
        "action_changes": {"overlapped_command_ids": sorted(overlap_ids)},
        "schedule_results": schedule_results,
        "limitations": [
            "The oracle uses completed command telemetry unavailable at BeginCall.",
            "Recorded command durations do not change under modeled co-admission.",
            "Idle container memory and Disk, network, and cache contention are not modeled.",
            "The paired seed bootstrap measures schedule sensitivity, not independent-task uncertainty.",
        ],
    }


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("output directory already exists")
    result = run()
    result["inputs"] = {"run": str(RUN.resolve()), "git_sha": _git_sha()}
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
