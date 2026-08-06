#!/usr/bin/env python3
"""Replay admission with the physical CPU-work/quota duration lower bound."""

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

from scripts.evaluation.evaluate_kv_prediction_actionability import (  # noqa: E402
    SPLIT,
    VALIDATION_RUN,
)
from scripts.evaluation.evaluate_resource_admission_oracle import (  # noqa: E402
    CPU_CAPACITY,
    RSS_CAPACITY_MB,
    _program,
)
from scripts.evaluation.evaluate_resource_admission_predictors import (  # noqa: E402
    _full_reservation_map,
    _static_predictions,
    _validation_commands,
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


VERSION = "cpu-work-admission-v1"
MINIMUM_COVERAGE = 0.80
MAX_VALID_CPU_RATE = 8.1
MINIMUM_REDUCTION = 0.05
MINIMUM_CHANGED_COMMANDS = 20
MINIMUM_CHANGED_TASKS = 10
EXPECTED_CHANGED_REQUESTS = 55
PRIOR_RESULT = (
    _ROOT
    / "analysis/results/tool-resource-5-3-3-3-20260804"
    / "sqlglot50-resource-admission-predictors-v1/result.json"
)


def _clause_cpu_work(clauses: list[Any]) -> float | None:
    seen: set[tuple[Any, ...]] = set()
    values: list[int] = []
    for clause in clauses:
        if not isinstance(clause, Mapping):
            return None
        argv = clause.get("argv")
        if not isinstance(argv, list):
            return None
        identity = (
            clause.get("bin"),
            tuple(argv),
            clause.get("ts_start"),
            clause.get("ts_end"),
            clause.get("pipeline_position"),
            clause.get("in_loop"),
            clause.get("in_pipe"),
            clause.get("in_subst"),
        )
        if identity in seen:
            raise ValueError(f"duplicate clause CPU-work identity: {identity!r}")
        seen.add(identity)
        value = clause.get("cpu_ns_cumulative")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return None
        values.append(value)
    return sum(values) / 1e9 if values else None


def _cpu_work(task_ids: list[str]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    work: dict[str, float] = {}
    metadata: list[dict[str, Any]] = []
    for task_id in task_ids:
        artifact_path = (
            VALIDATION_RUN / task_id / "attempt_1/resource_observations.json"
        )
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        task_work_before = len(work)
        for call in artifact.get("calls", []):
            if not isinstance(call, Mapping) or call.get("eligible_for_kb") is not True:
                continue
            call_id = call.get("tool_call_id")
            clauses = call.get("clauses")
            if (
                not isinstance(call_id, str)
                or not isinstance(clauses, list)
                or not clauses
            ):
                continue
            value = _clause_cpu_work(clauses)
            if value is None:
                continue
            command_id = f"{task_id}:{call_id}"
            if command_id in work:
                raise ValueError(f"duplicate CPU work: {command_id}")
            work[command_id] = value
        metadata.append(
            {
                "task_id": task_id,
                "path": str(artifact_path.resolve()),
                "run_id": artifact.get("run_id"),
                "trace_id": artifact.get("trace_id"),
                "pinned_snapshot_id": artifact.get("pinned_snapshot_id"),
                "version": artifact.get("version"),
                "canonicalizer_version": artifact.get("canonicalizer_version"),
                "collection_validity": artifact.get("collection_validity"),
                "workload_execution": artifact.get("workload_execution"),
                "telemetry_quality": artifact.get("telemetry_quality"),
                "cleanup": artifact.get("cleanup"),
                "cpu_work_commands": len(work) - task_work_before,
            }
        )
    return work, metadata


def _cpu_floor_programs(
    programs: Mapping[str, AdmissionProgram],
    requests: Mapping[str, tuple[float, float]],
    cpu_work_s: Mapping[str, float],
) -> tuple[dict[str, AdmissionProgram], dict[str, Any]]:
    adjusted: dict[str, AdmissionProgram] = {}
    changed: set[str] = set()
    added_service_s = 0.0
    for task_id, program in programs.items():
        commands = []
        for command in program.commands:
            cpu_request = requests[command.command_id][0]
            work = cpu_work_s.get(command.command_id)
            duration = command.duration_s
            if work is not None:
                duration = max(duration, work / cpu_request)
            if duration > command.duration_s + 1e-12:
                changed.add(command.command_id)
                added_service_s += duration - command.duration_s
            commands.append(
                AdmissionCommand(
                    command.command_id,
                    duration,
                    command.cpu_cores,
                    command.rss_mb,
                    command.delay_after_s,
                )
            )
        adjusted[task_id] = AdmissionProgram(
            task_id,
            program.initial_delay_s,
            tuple(commands),
            program.tail_s,
        )
    return adjusted, {
        "dilated_commands": len(changed),
        "dilated_tasks": len({value.split(":", 1)[0] for value in changed}),
        "added_service_s": added_service_s,
        "dilated_command_ids": sorted(changed),
    }


def _shared_rss_requests(
    programs: Mapping[str, AdmissionProgram],
    predicted: Mapping[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    return {
        command.command_id: (predicted[command.command_id][0], command.rss_mb)
        for program in programs.values()
        for command in program.commands
    }


def _simulate(
    programs: Mapping[str, AdmissionProgram],
    selected: list[str],
    requests: Mapping[str, tuple[float, float]],
) -> dict[str, object]:
    result = simulate_admission(
        [programs[task_id] for task_id in selected],
        cpu_capacity=CPU_CAPACITY,
        rss_capacity_mb=RSS_CAPACITY_MB,
        fixed_high=False,
        requested_reservations=requests,
    )
    result.pop("overlapped_command_ids")
    result.pop("modeled_capacity_exposure_command_ids")
    return result


def _require_committed_inputs() -> None:
    paths = (
        Path(__file__).resolve(),
        SPLIT.resolve(),
        PRIOR_RESULT.resolve(),
    )
    for path in paths:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            cwd=_ROOT,
            check=True,
            capture_output=True,
        )
    subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *(str(path) for path in paths)],
        cwd=_ROOT,
        check=True,
    )


def run() -> dict[str, Any]:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    task_ids = list(split["validation"])
    current_predictions, sota_predictions, prediction_coverage = _static_predictions(
        task_ids
    )
    command_rows, raw_counts, traces = _validation_commands(task_ids)
    source_counts: Counter[str] = Counter()
    programs = {
        task_id: _program(
            task_id,
            traces[task_id],
            command_rows,
            raw_counts,
            source_counts,
        )
        for task_id in task_ids
    }
    current_requests = _full_reservation_map(programs, current_predictions)
    sota_requests = _full_reservation_map(programs, sota_predictions)
    changed_requests = {
        command_id
        for command_id in current_requests
        if current_requests[command_id] != sota_requests[command_id]
    }
    prior = json.loads(PRIOR_RESULT.read_text(encoding="utf-8"))
    prior_changed = set(prior["changed_request_ids"])
    if (
        len(changed_requests) != EXPECTED_CHANGED_REQUESTS
        or len(prior_changed) != EXPECTED_CHANGED_REQUESTS
        or changed_requests != prior_changed
    ):
        raise ValueError("changed-request cohort differs from the frozen 55 commands")
    work, validation_metadata = _cpu_work(task_ids)
    all_commands = {
        command.command_id
        for program in programs.values()
        for command in program.commands
    }
    mapped_work = all_commands & work.keys()
    changed_mapped = changed_requests & work.keys()
    rates = {
        command.command_id: work[command.command_id] / command.duration_s
        for program in programs.values()
        for command in program.commands
        if command.command_id in work
    }
    invalid_rates = {
        command_id: rate
        for command_id, rate in rates.items()
        if rate > MAX_VALID_CPU_RATE
    }
    coverage_gate = {
        "all_command_fraction": len(mapped_work) / len(all_commands),
        "changed_command_fraction": len(changed_mapped) / len(changed_requests),
        "at_least_80_percent_all_commands": len(mapped_work) / len(all_commands)
        >= MINIMUM_COVERAGE,
        "at_least_80_percent_changed_commands": len(changed_mapped)
        / len(changed_requests)
        >= MINIMUM_COVERAGE,
        "zero_impossible_cpu_rates": not invalid_rates,
        "maximum_cpu_work_over_wall": max(rates.values()),
    }
    if not all(
        coverage_gate[key]
        for key in (
            "at_least_80_percent_all_commands",
            "at_least_80_percent_changed_commands",
            "zero_impossible_cpu_rates",
        )
    ):
        raise ValueError("frozen CPU-work evidence gate failed")

    current_programs, current_dilation = _cpu_floor_programs(
        programs, current_requests, work
    )
    sota_programs, sota_dilation = _cpu_floor_programs(programs, sota_requests, work)
    current_shared_rss = _shared_rss_requests(programs, current_requests)
    sota_shared_rss = _shared_rss_requests(programs, sota_requests)
    current_isolated_programs, current_isolated_dilation = _cpu_floor_programs(
        programs, current_shared_rss, work
    )
    sota_isolated_programs, sota_isolated_dilation = _cpu_floor_programs(
        programs, sota_shared_rss, work
    )

    service_changed = {
        current.command_id
        for task_id in programs
        for current, sota in zip(
            current_programs[task_id].commands,
            sota_programs[task_id].commands,
            strict=True,
        )
        if not math.isclose(current.duration_s, sota.duration_s, abs_tol=1e-12)
    }
    selections = []
    recorded_results = []
    for seed in SEEDS:
        selected = sorted(programs)
        np.random.default_rng(seed).shuffle(selected)
        selected = selected[:LOAD]
        selections.append((seed, selected))
        recorded_results.append(
            {
                "seed": seed,
                "task_ids": selected,
                "arms": {
                    "recorded_current": _simulate(programs, selected, current_requests),
                    "recorded_sota": _simulate(programs, selected, sota_requests),
                },
            }
        )
    recorded_means = {
        arm: float(
            np.mean(
                [
                    row["arms"][f"recorded_{arm}"]["makespan_s"]
                    for row in recorded_results
                ]
            )
        )
        for arm in ("current", "sota")
    }
    reproduction = {
        arm: math.isclose(
            recorded_means[arm],
            float(prior["mean_metrics"][arm]["makespan_s"]),
            abs_tol=1e-9,
        )
        for arm in ("current", "sota")
    }
    if not all(reproduction.values()):
        raise ValueError("recorded-duration control does not reproduce prior means")

    schedule_results = []
    for recorded, (seed, selected) in zip(recorded_results, selections, strict=True):
        arms = {
            **recorded["arms"],
            "cpu_floor_current": _simulate(
                current_programs, selected, current_requests
            ),
            "cpu_floor_sota": _simulate(sota_programs, selected, sota_requests),
            "shared_rss_current": _simulate(
                current_isolated_programs, selected, current_shared_rss
            ),
            "shared_rss_sota": _simulate(
                sota_isolated_programs, selected, sota_shared_rss
            ),
        }
        if len({value["command_count"] for value in arms.values()}) != 1:
            raise ValueError("admission arms evaluated different commands")
        schedule_results.append({"seed": seed, "task_ids": selected, "arms": arms})

    metrics = (
        "makespan_s",
        "mean_task_completion_s",
        "total_command_queue_s",
        "total_command_service_s",
        "reserved_cpu_core_s",
        "reserved_rss_mb_s",
        "max_concurrent_commands",
        "modeled_capacity_exposure_events",
    )
    arm_names = tuple(schedule_results[0]["arms"])
    means = {
        arm: {
            metric: float(
                np.mean([row["arms"][arm][metric] for row in schedule_results])
            )
            for metric in metrics
        }
        for arm in arm_names
    }
    deltas = [
        float(row["arms"]["cpu_floor_sota"]["makespan_s"])
        - float(row["arms"]["cpu_floor_current"]["makespan_s"])
        for row in schedule_results
    ]
    current_makespan = means["cpu_floor_current"]["makespan_s"]
    sota_makespan = means["cpu_floor_sota"]["makespan_s"]
    reduction = (current_makespan - sota_makespan) / current_makespan
    comparison = {
        "candidate": "cpu_floor_sota",
        "baseline": "cpu_floor_current",
        "metric": "mean batch makespan_s; lower is better",
        "relative_reduction_of_means": reduction,
        **_bootstrap(deltas),
    }
    gate = {
        "coverage_gate_passed": all(
            coverage_gate[key]
            for key in (
                "at_least_80_percent_all_commands",
                "at_least_80_percent_changed_commands",
                "zero_impossible_cpu_rates",
            )
        ),
        "recorded_control_reproduced": all(reproduction.values()),
        "sota_reduction_at_least_5_percent": reduction >= MINIMUM_REDUCTION,
        "paired_ci_below_zero": comparison["ci95_paired_seed_bootstrap"][1] < 0.0,
        "at_least_20_service_changed_commands": len(service_changed)
        >= MINIMUM_CHANGED_COMMANDS,
        "at_least_10_service_changed_tasks": len(
            {value.split(":", 1)[0] for value in service_changed}
        )
        >= MINIMUM_CHANGED_TASKS,
    }
    gate["promising"] = all(gate.values())
    return {
        "schema": VERSION,
        "status": (
            "development_promising_cpu_cost_mechanism"
            if gate["promising"]
            else "development_no_go_cpu_work_floor"
        ),
        "claim_bearing": False,
        "protocol": {
            "task_pool": "exposed SQLGlot validation50",
            "validation_task_ids": task_ids,
            "scheduler": "unchanged FCFS-ready work-conserving backfill",
            "load": LOAD,
            "seeds": list(SEEDS),
            "duration": "max(recorded_s, sum(clause cpu_ns) / requested_cpu_cores)",
            "minimum_coverage": MINIMUM_COVERAGE,
            "maximum_valid_cpu_rate": MAX_VALID_CPU_RATE,
            "minimum_relative_reduction": MINIMUM_REDUCTION,
        },
        "coverage": {
            "tasks": len(programs),
            "commands": len(all_commands),
            "cpu_work_commands": len(mapped_work),
            "missing_cpu_work_command_ids": sorted(all_commands - mapped_work),
            "changed_request_commands": len(changed_requests),
            "changed_request_ids": sorted(changed_requests),
            "changed_request_cpu_work_commands": len(changed_mapped),
            "service_changed_commands": len(service_changed),
            "service_changed_tasks": len(
                {value.split(":", 1)[0] for value in service_changed}
            ),
            "invalid_cpu_rates": invalid_rates,
            "cpu_work_core_s_by_command": dict(sorted(work.items())),
            "validation_artifacts": validation_metadata,
            "telemetry_reservation_sources": dict(sorted(source_counts.items())),
            "prediction": prediction_coverage,
            "gate": coverage_gate,
        },
        "recorded_control_reproduction": reproduction,
        "duration_adjustment": {
            "current": current_dilation,
            "sota": sota_dilation,
            "shared_rss_current": current_isolated_dilation,
            "shared_rss_sota": sota_isolated_dilation,
            "service_changed_command_ids": sorted(service_changed),
        },
        "mean_metrics": means,
        "primary_comparison": comparison,
        "gate": gate,
        "schedule_results": schedule_results,
        "limitations": [
            "The SQLGlot tasks and predictor outputs are development-exposed.",
            "The work-conservation formula is a lower bound under an enforced per-command CPU quota, not a complete contention model.",
            "Commands without complete CPU-work evidence retain recorded duration.",
            "Memory under-reservation remains unpriced in the primary replay.",
            "The shared-RSS arms use hindsight telemetry only for attribution.",
        ],
    }


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("output directory already exists")
    _require_committed_inputs()
    result = run()
    result["inputs"] = {
        "validation_run": str(VALIDATION_RUN.resolve()),
        "split": str(SPLIT.resolve()),
        "prior_result": str(PRIOR_RESULT.resolve()),
        "git_sha": _git_sha(),
    }
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
