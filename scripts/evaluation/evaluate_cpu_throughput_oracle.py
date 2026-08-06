#!/usr/bin/env python3
"""Compare peak-CPU and CPU-throughput reservation targets."""

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

from scripts.evaluation.evaluate_cpu_work_admission import (  # noqa: E402
    MAX_VALID_CPU_RATE,
    _cpu_floor_programs,
)
from scripts.evaluation.evaluate_kv_prediction_actionability import (  # noqa: E402
    SPLIT,
)
from scripts.evaluation.evaluate_resource_admission_oracle import (  # noqa: E402
    CPU_CAPACITY,
    RSS_CAPACITY_MB,
    _program,
)
from scripts.evaluation.evaluate_resource_admission_predictors import (  # noqa: E402
    _validation_commands,
)
from tool_resource_eval.cachewise_kv_factorial import (  # noqa: E402
    LOAD,
    SEEDS,
    _bootstrap,
)
from tool_resource_eval.resource_admission import (  # noqa: E402
    AdmissionProgram,
    simulate_admission,
)


VERSION = "cpu-throughput-oracle-v1"
MINIMUM_REDUCTION = 0.10
MINIMUM_CHANGED_COMMANDS = 20
MINIMUM_CHANGED_TASKS = 10
CPU_WORK_RESULT = (
    _ROOT
    / "analysis/results/tool-resource-5-3-3-3-20260804"
    / "sqlglot50-cpu-work-admission-v1/result.json"
)
PRIOR_ADMISSION_RESULT = (
    _ROOT
    / "analysis/results/tool-resource-5-3-3-3-20260804"
    / "sqlglot50-resource-admission-predictors-v1/result.json"
)
THROUGHPUT_RESULT = (
    _ROOT
    / "analysis/results/tool-resource-5-3-3-3-20260804"
    / "sqlglot50-cpu-throughput-oracle-v1/result.json"
)


def _bucket_cpu(value: float) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("CPU target must be finite and non-negative")
    if value <= 2.0:
        return 2.0
    if value <= 4.0:
        return 4.0
    return 8.0


def _requests(
    programs: Mapping[str, AdmissionProgram],
    cpu_work_s: Mapping[str, float],
) -> tuple[
    dict[str, tuple[float, float]],
    dict[str, tuple[float, float]],
    set[str],
]:
    peak: dict[str, tuple[float, float]] = {}
    throughput: dict[str, tuple[float, float]] = {}
    missing: set[str] = set()
    for program in programs.values():
        for command in program.commands:
            peak[command.command_id] = (
                _bucket_cpu(command.cpu_cores),
                command.rss_mb,
            )
            work = cpu_work_s.get(command.command_id)
            if work is None:
                cpu = CPU_CAPACITY
                missing.add(command.command_id)
            else:
                cpu = _bucket_cpu(work / command.duration_s)
            throughput[command.command_id] = (cpu, command.rss_mb)
    return peak, throughput, missing


def _simulate(
    programs: Mapping[str, AdmissionProgram],
    selected: list[str],
    *,
    requests: Mapping[str, tuple[float, float]] | None = None,
    fixed_high: bool = False,
) -> tuple[dict[str, object], set[str]]:
    result = simulate_admission(
        [programs[task_id] for task_id in selected],
        cpu_capacity=CPU_CAPACITY,
        rss_capacity_mb=RSS_CAPACITY_MB,
        fixed_high=fixed_high,
        requested_reservations=requests,
    )
    overlaps = {str(value) for value in result.pop("overlapped_command_ids")}
    result.pop("modeled_capacity_exposure_command_ids")
    return result, overlaps


def _require_committed_inputs(extra: tuple[Path, ...] = ()) -> None:
    paths = (
        Path(__file__).resolve(),
        SPLIT.resolve(),
        CPU_WORK_RESULT.resolve(),
        PRIOR_ADMISSION_RESULT.resolve(),
        *extra,
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


def _program_snapshot(
    programs: Mapping[str, AdmissionProgram],
    cpu_work: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    return {
        command.command_id: {
            "task_id": task_id,
            "recorded_duration_s": command.duration_s,
            "peak_cpu_cores": command.cpu_cores,
            "rss_mb": command.rss_mb,
            "delay_after_s": command.delay_after_s,
            "cpu_work_core_s": cpu_work.get(command.command_id),
        }
        for task_id, program in sorted(programs.items())
        for command in program.commands
    }


def run() -> dict[str, Any]:
    prior = json.loads(CPU_WORK_RESULT.read_text(encoding="utf-8"))
    prior_admission = json.loads(PRIOR_ADMISSION_RESULT.read_text(encoding="utf-8"))
    task_ids = list(json.loads(SPLIT.read_text(encoding="utf-8"))["validation"])
    if task_ids != prior["protocol"]["validation_task_ids"]:
        raise ValueError("validation cohort differs from reviewed CPU-work result")
    cpu_work = {
        command_id: float(value)
        for command_id, value in prior["coverage"]["cpu_work_core_s_by_command"].items()
    }
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
    all_commands = {
        command.command_id
        for program in programs.values()
        for command in program.commands
    }
    reviewed_commands = set(cpu_work) | set(
        prior["coverage"]["missing_cpu_work_command_ids"]
    )
    if all_commands != reviewed_commands:
        raise ValueError(
            "replay command identity differs from reviewed CPU-work result"
        )
    rates = {
        command.command_id: cpu_work[command.command_id] / command.duration_s
        for program in programs.values()
        for command in program.commands
        if command.command_id in cpu_work
    }
    if not rates or max(rates.values()) > MAX_VALID_CPU_RATE:
        raise ValueError("reviewed CPU-work validity no longer holds")

    peak_requests, throughput_requests, missing = _requests(programs, cpu_work)
    changed = {
        command_id
        for command_id in peak_requests
        if peak_requests[command_id][0] != throughput_requests[command_id][0]
    }
    peak_programs, peak_dilation = _cpu_floor_programs(
        programs, peak_requests, cpu_work
    )
    throughput_programs, throughput_dilation = _cpu_floor_programs(
        programs, throughput_requests, cpu_work
    )

    selections = []
    control_results = []
    prior_by_seed = {
        int(row["seed"]): row for row in prior_admission["schedule_results"]
    }
    for seed in SEEDS:
        selected = sorted(programs)
        np.random.default_rng(seed).shuffle(selected)
        selected = selected[:LOAD]
        selections.append((seed, selected))
        fixed, _ = _simulate(programs, selected, fixed_high=True)
        continuous, _ = _simulate(programs, selected)
        controls = {"fixed_high": fixed, "continuous_peak": continuous}
        expected = prior_by_seed[seed]["arms"]
        if (
            selected != prior_by_seed[seed]["task_ids"]
            or fixed != expected["fixed_high"]
            or continuous != expected["oracle"]
        ):
            raise ValueError("fixed-high or continuous-peak control drifted")
        control_results.append({"seed": seed, "task_ids": selected, "arms": controls})

    schedule_results = []
    overlaps: dict[str, set[str]] = {
        "peak_bucket": set(),
        "throughput_bucket": set(),
    }
    for controls, (seed, selected) in zip(control_results, selections, strict=True):
        peak, peak_overlap = _simulate(
            peak_programs,
            selected,
            requests=peak_requests,
        )
        throughput, throughput_overlap = _simulate(
            throughput_programs,
            selected,
            requests=throughput_requests,
        )
        overlaps["peak_bucket"].update(peak_overlap)
        overlaps["throughput_bucket"].update(throughput_overlap)
        arms = {
            **controls["arms"],
            "peak_bucket": peak,
            "throughput_bucket": throughput,
        }
        if len({value["command_count"] for value in arms.values()}) != 1:
            raise ValueError("oracle arms evaluated different commands")
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
        float(row["arms"]["throughput_bucket"]["makespan_s"])
        - float(row["arms"]["peak_bucket"]["makespan_s"])
        for row in schedule_results
    ]
    peak_makespan = means["peak_bucket"]["makespan_s"]
    throughput_makespan = means["throughput_bucket"]["makespan_s"]
    reduction = (peak_makespan - throughput_makespan) / peak_makespan
    comparison = {
        "candidate": "throughput_bucket",
        "baseline": "peak_bucket",
        "metric": "mean batch makespan_s; lower is better",
        "relative_reduction_of_means": reduction,
        **_bootstrap(deltas),
    }
    no_dilation = (
        peak_dilation["dilated_commands"] == 0
        and throughput_dilation["dilated_commands"] == 0
    )
    missing_full = all(
        throughput_requests[command_id][0] == CPU_CAPACITY for command_id in missing
    )
    no_capacity_violation = not any(
        bool(row["arms"][arm]["capacity_violation"])
        for row in schedule_results
        for arm in ("peak_bucket", "throughput_bucket")
    )
    gate = {
        "makespan_reduction_at_least_10_percent": reduction >= MINIMUM_REDUCTION,
        "paired_ci_below_zero": comparison["ci95_paired_seed_bootstrap"][1] < 0.0,
        "zero_duration_dilation": no_dilation,
        "at_least_20_changed_commands": len(changed) >= MINIMUM_CHANGED_COMMANDS,
        "at_least_10_changed_tasks": len({value.split(":", 1)[0] for value in changed})
        >= MINIMUM_CHANGED_TASKS,
        "missing_commands_use_full_cpu": missing_full,
        "identical_commands_and_cpu_work": all(
            row["arms"]["peak_bucket"]["command_count"]
            == row["arms"]["throughput_bucket"]["command_count"]
            for row in schedule_results
        ),
        "no_requested_capacity_violation": no_capacity_violation,
    }
    gate["go"] = all(gate.values())
    return {
        "schema": VERSION,
        "status": (
            "development_go_to_throughput_prediction"
            if gate["go"]
            else "development_no_go_cpu_throughput_target"
        ),
        "claim_bearing": False,
        "protocol": {
            "task_pool": "exposed SQLGlot validation50",
            "validation_task_ids": task_ids,
            "load": LOAD,
            "seeds": list(SEEDS),
            "cpu_classes": [2.0, 4.0, 8.0],
            "shared_rss": "hindsight clause-composed telemetry bound",
            "primary": "throughput-target minus peak-target mean makespan_s",
            "minimum_relative_reduction": MINIMUM_REDUCTION,
        },
        "coverage": {
            "tasks": len(programs),
            "commands": len(all_commands),
            "cpu_work_commands": len(cpu_work),
            "missing_cpu_work_command_ids": sorted(missing),
            "changed_cpu_request_commands": len(changed),
            "changed_cpu_request_tasks": len(
                {value.split(":", 1)[0] for value in changed}
            ),
            "changed_cpu_request_ids": sorted(changed),
            "maximum_cpu_work_over_wall": max(rates.values()),
            "telemetry_reservation_sources": dict(sorted(source_counts.items())),
        },
        "requests": {
            "peak_bucket": {
                key: list(value) for key, value in sorted(peak_requests.items())
            },
            "throughput_bucket": {
                key: list(value) for key, value in sorted(throughput_requests.items())
            },
        },
        "program_inputs_by_command": _program_snapshot(programs, cpu_work),
        "control_reproduction": {
            "prior_admission_result": str(PRIOR_ADMISSION_RESULT.resolve()),
            "all_seed_task_ids_and_metrics_exact": True,
        },
        "duration_adjustment": {
            "peak_bucket": peak_dilation,
            "throughput_bucket": throughput_dilation,
        },
        "mean_metrics": means,
        "primary_comparison": comparison,
        "overlap": {
            arm: {
                "distinct_commands": len(ids),
                "distinct_tasks": len({value.split(":", 1)[0] for value in ids}),
            }
            for arm, ids in overlaps.items()
        },
        "gate": gate,
        "schedule_results": schedule_results,
        "limitations": [
            "This is a hindsight action-space oracle on development-exposed tasks.",
            "Average CPU work does not reveal critical-path parallelism or short-timescale contention.",
            "The shared RSS reservation uses completed command telemetry.",
            "Recorded duration is preserved whenever requested average throughput is sufficient; real quota scaling may be worse.",
        ],
    }


def run_two_core_baseline() -> dict[str, Any]:
    prior = json.loads(THROUGHPUT_RESULT.read_text(encoding="utf-8"))
    task_ids = list(json.loads(SPLIT.read_text(encoding="utf-8"))["validation"])
    if task_ids != prior["protocol"]["validation_task_ids"]:
        raise ValueError("validation cohort differs from throughput oracle")
    cpu_work = {
        command_id: float(row["cpu_work_core_s"])
        for command_id, row in prior["program_inputs_by_command"].items()
        if row["cpu_work_core_s"] is not None
    }
    missing = set(prior["coverage"]["missing_cpu_work_command_ids"])
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
    if _program_snapshot(programs, cpu_work) != prior["program_inputs_by_command"]:
        raise ValueError("command program differs from throughput oracle")

    throughput_requests = {
        command_id: tuple(value)
        for command_id, value in prior["requests"]["throughput_bucket"].items()
    }
    two_core_requests = {
        command.command_id: (
            CPU_CAPACITY if command.command_id in missing else 2.0,
            command.rss_mb,
        )
        for program in programs.values()
        for command in program.commands
    }
    if not all(
        two_core_requests[command_id][0] == CPU_CAPACITY for command_id in missing
    ):
        raise ValueError("missing CPU-work commands do not use full-host fallback")
    throughput_programs, throughput_dilation = _cpu_floor_programs(
        programs, throughput_requests, cpu_work
    )
    two_core_programs, two_core_dilation = _cpu_floor_programs(
        programs, two_core_requests, cpu_work
    )

    prior_by_seed = {int(row["seed"]): row for row in prior["schedule_results"]}
    schedule_results = []
    for seed in SEEDS:
        selected = sorted(programs)
        np.random.default_rng(seed).shuffle(selected)
        selected = selected[:LOAD]
        oracle, _ = _simulate(
            throughput_programs,
            selected,
            requests=throughput_requests,
        )
        if (
            selected != prior_by_seed[seed]["task_ids"]
            or oracle != prior_by_seed[seed]["arms"]["throughput_bucket"]
        ):
            raise ValueError("throughput-oracle schedule control drifted")
        two_core, _ = _simulate(
            two_core_programs,
            selected,
            requests=two_core_requests,
        )
        schedule_results.append(
            {
                "seed": seed,
                "task_ids": selected,
                "arms": {"throughput_oracle": oracle, "two_core_default": two_core},
            }
        )

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
    means = {
        arm: {
            metric: float(
                np.mean([row["arms"][arm][metric] for row in schedule_results])
            )
            for metric in metrics
        }
        for arm in ("throughput_oracle", "two_core_default")
    }
    regrets = [
        (
            float(row["arms"]["two_core_default"]["makespan_s"])
            - float(row["arms"]["throughput_oracle"]["makespan_s"])
        )
        / float(row["arms"]["throughput_oracle"]["makespan_s"])
        for row in schedule_results
    ]
    comparison = {
        "candidate": "two_core_default",
        "baseline": "throughput_oracle",
        "metric": "paired relative makespan regret; lower is better",
        **_bootstrap(regrets),
    }
    upper = comparison["ci95_paired_seed_bootstrap"][1]
    lower = comparison["ci95_paired_seed_bootstrap"][0]
    mean_regret = comparison["mean"]
    adequate = mean_regret <= 0.05 and upper <= 0.05
    prediction_headroom = mean_regret > 0.05 and lower > 0.0
    no_violation = not any(
        bool(row["arms"][arm]["capacity_violation"])
        for row in schedule_results
        for arm in ("throughput_oracle", "two_core_default")
    )
    validity = {
        "prior_oracle_reproduced": True,
        "missing_commands_use_full_cpu": all(
            two_core_requests[command_id][0] == CPU_CAPACITY for command_id in missing
        ),
        "identical_commands_and_work": all(
            row["arms"]["throughput_oracle"]["command_count"]
            == row["arms"]["two_core_default"]["command_count"]
            for row in schedule_results
        ),
        "no_requested_capacity_violation": no_violation,
    }
    if not all(validity.values()):
        raise ValueError("two-core baseline validity gate failed")
    if adequate:
        status = "development_stop_predictor_two_core_adequate"
    elif prediction_headroom:
        status = "development_go_to_throughput_predictor"
    else:
        status = "development_inconclusive_two_core_regret"
    return {
        "schema": "two-core-throughput-baseline-v1",
        "status": status,
        "claim_bearing": False,
        "protocol": {
            "task_pool": "exposed SQLGlot validation50",
            "validation_task_ids": task_ids,
            "load": LOAD,
            "seeds": list(SEEDS),
            "maximum_adequate_regret": 0.05,
            "shared_rss": "reviewed hindsight throughput-oracle RSS",
        },
        "coverage": {
            "tasks": len(programs),
            "commands": len(prior["program_inputs_by_command"]),
            "cpu_work_commands": len(cpu_work),
            "missing_cpu_work_command_ids": sorted(missing),
        },
        "requests": {
            key: list(value) for key, value in sorted(two_core_requests.items())
        },
        "duration_adjustment": {
            "throughput_oracle": throughput_dilation,
            "two_core_default": two_core_dilation,
        },
        "mean_metrics": means,
        "primary_comparison": comparison,
        "decision": {
            "two_core_adequate": adequate,
            "prediction_has_actionable_headroom": prediction_headroom,
        },
        "validity": validity,
        "schedule_results": schedule_results,
        "limitations": [
            "This baseline uses hindsight RSS and development-exposed tasks.",
            "The CPU-work duration floor is optimistic about critical-path scaling.",
            "Missing CPU-work commands conservatively reserve the full host.",
        ],
    }


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--two-core-baseline", action="store_true")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("output directory already exists")
    _require_committed_inputs(
        (THROUGHPUT_RESULT.resolve(),) if args.two_core_baseline else ()
    )
    result = run_two_core_baseline() if args.two_core_baseline else run()
    result["inputs"] = {
        "split": str(SPLIT.resolve()),
        "cpu_work_result": str(CPU_WORK_RESULT.resolve()),
        "prior_admission_result": str(PRIOR_ADMISSION_RESULT.resolve()),
        "throughput_result": (
            str(THROUGHPUT_RESULT.resolve()) if args.two_core_baseline else None
        ),
        "git_sha": _git_sha(),
    }
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
