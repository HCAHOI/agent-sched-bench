#!/usr/bin/env python3
"""Evaluate frozen RSS predictors as CPU-idle backfill safety gates."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from scripts.evaluation.evaluate_cpu_feedback_admission import (  # noqa: E402
    _file_identities,
    _profiles,
    _service_inflation,
)
from scripts.evaluation.evaluate_cpu_idle_backfill_oracle import (  # noqa: E402
    _comparison,
    _source_identities,
)
from scripts.evaluation.evaluate_kv_prediction_actionability import (  # noqa: E402
    FIT_ROWS,
    PHASE_ARTIFACT,
    PHASE_RESULT,
    PHASE_ROWS,
    SPLIT,
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
)
from tool_resource_eval.resource_admission import (  # noqa: E402
    AdmissionProgram,
    simulate_idle_backfill,
)


VERSION = "cpu-idle-rss-safety-v1"
SHORT_NULL_VERSION = "cpu-idle-short-null-v1"
ARMS = (
    "serial8",
    "oracle_rss_fcfs",
    "clause_kb_rss_fcfs",
    "task_aware_rss_fcfs",
)
MINIMUM_CANDIDATE_REDUCTION = 0.05
MINIMUM_ORACLE_CAPTURE = 0.50
MINIMUM_GAIN_OVER_CLAUSE = 0.01
MAXIMUM_SERVICE_INFLATION = 0.05
MAXIMUM_MAKESPAN_REGRESSION = 0.01
PROTOCOL = _ROOT / "analysis/development/cpu-idle-rss-safety-protocol.md"
SHORT_NULL_PROTOCOL = (
    _ROOT / "analysis/development/cpu-idle-short-null-amendment.md"
)
SOURCE_POLICIES = ("conservative", "short-null-low")
SHORT_NULL_RSS_MB = 500.0
SHORT_NULL_MAX_LATENCY_MS = 500.0
SHORT_NULL_REASON = "unknown:insufficient_rss_samples"


def _protocol_for(source_policy: str) -> Path:
    if source_policy not in SOURCE_POLICIES:
        raise ValueError(f"unknown RSS source policy: {source_policy}")
    return PROTOCOL if source_policy == "conservative" else SHORT_NULL_PROTOCOL


def _gate(
    *,
    candidate_reduction: float,
    oracle_reduction: float,
    clause_reduction: float,
    bootstrap_high: float,
    service_inflation: float,
    makespan_regression: float,
    exposure_events: int,
    violation: bool,
) -> dict[str, bool]:
    gate = {
        "candidate_reduction_at_least_5_percent": (
            candidate_reduction >= MINIMUM_CANDIDATE_REDUCTION
        ),
        "captures_at_least_half_oracle_reduction": (
            oracle_reduction > 0.0
            and candidate_reduction >= MINIMUM_ORACLE_CAPTURE * oracle_reduction
        ),
        "gain_over_clause_at_least_1_percentage_point": (
            candidate_reduction - clause_reduction >= MINIMUM_GAIN_OVER_CLAUSE
        ),
        "paired_order_bootstrap_upper_below_zero": bootstrap_high < 0.0,
        "service_inflation_at_most_5_percent": (
            service_inflation <= MAXIMUM_SERVICE_INFLATION
        ),
        "makespan_regression_at_most_1_percent": (
            makespan_regression <= MAXIMUM_MAKESPAN_REGRESSION
        ),
        "zero_source_rss_exposures": exposure_events == 0,
        "zero_capacity_or_work_violations": not violation,
    }
    gate["go"] = all(gate.values())
    return gate


def _short_null_gate(
    *,
    candidate_reduction: float,
    oracle_reduction: float,
    bootstrap_high: float,
    service_inflation: float,
    makespan_regression: float,
    exposure_events: int,
    violation: bool,
) -> dict[str, bool]:
    gate = {
        "candidate_reduction_at_least_5_percent": (
            candidate_reduction >= MINIMUM_CANDIDATE_REDUCTION
        ),
        "captures_at_least_half_oracle_reduction": (
            oracle_reduction > 0.0
            and candidate_reduction >= MINIMUM_ORACLE_CAPTURE * oracle_reduction
        ),
        "paired_order_bootstrap_upper_below_zero": bootstrap_high < 0.0,
        "service_inflation_at_most_5_percent": (
            service_inflation <= MAXIMUM_SERVICE_INFLATION
        ),
        "makespan_regression_at_most_1_percent": (
            makespan_regression <= MAXIMUM_MAKESPAN_REGRESSION
        ),
        "zero_confirmed_source_rss_exposures": exposure_events == 0,
        "zero_capacity_or_work_violations": not violation,
    }
    gate["go"] = all(gate.values())
    return gate


def _mean_metrics(schedule_results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "command_count",
        "recorded_command_service_s",
        "total_command_service_s",
        "added_service_s",
        "service_inflation",
        "makespan_s",
        "mean_task_completion_s",
        "total_command_queue_s",
        "reserved_rss_mb_s",
        "max_modeled_rss_demand_mb",
        "modeled_capacity_exposure_events",
        "modeled_capacity_exposure_commands",
        "rss_unverified_overlap_events",
        "rss_unverified_overlap_commands",
        "max_concurrent_commands",
        "normal_starts",
        "speculative_starts",
        "speculative_completions",
        "promotions",
        "speculative_cpu_work_core_s",
        "total_cpu_work_core_s",
        "served_cpu_work_core_s",
    )
    return {
        arm: {
            metric: float(
                np.mean([row["arms"][arm][metric] for row in schedule_results])
            )
            for metric in metrics
        }
        for arm in ARMS
    }


def _arm_specs(
    command_ids: set[str],
    oracle_rss: dict[str, float],
    clause_rss: dict[str, float],
    task_rss: dict[str, float],
) -> dict[str, tuple[str, set[str], dict[str, float]]]:
    return {
        "serial8": ("serial", command_ids, oracle_rss),
        "oracle_rss_fcfs": ("fcfs", command_ids, oracle_rss),
        "clause_kb_rss_fcfs": ("fcfs", command_ids, clause_rss),
        "task_aware_rss_fcfs": ("fcfs", command_ids, task_rss),
    }


def _short_null_command_rows(
    command_rows: dict[tuple[str, str], Any],
    traces: dict[str, Path],
) -> tuple[dict[tuple[str, str], Any], set[str], dict[str, int]]:
    adjusted = dict(command_rows)
    visited: set[tuple[str, str]] = set()
    imputed_ids: set[str] = set()
    imputed_clauses = 0
    for task_id, trace in traces.items():
        artifact = json.loads(
            (trace.parent / "resource_observations.json").read_text(encoding="utf-8")
        )
        for call in artifact.get("calls", []):
            call_id = call.get("tool_call_id")
            key = (task_id, call_id)
            if key not in command_rows:
                continue
            if key in visited:
                raise ValueError(f"duplicate source call: {key}")
            raw_clauses = call.get("clauses")
            row = command_rows[key]
            if not isinstance(raw_clauses, list) or len(raw_clauses) != len(row.clauses):
                raise ValueError(f"short-null source clauses differ: {key}")
            clauses = []
            for clause, raw_clause in zip(row.clauses, raw_clauses, strict=True):
                if (
                    not isinstance(raw_clause, dict)
                    or raw_clause.get("bin") != clause.bin
                    or tuple(raw_clause.get("argv", ())) != clause.argv
                ):
                    raise ValueError(f"short-null clause identity differs: {key}")
                availability = raw_clause.get("availability", {})
                if (
                    clause.sampled_peak_rss_mb is None
                    and clause.latency_ms < SHORT_NULL_MAX_LATENCY_MS
                    and availability.get("memory") == SHORT_NULL_REASON
                ):
                    clause = replace(clause, sampled_peak_rss_mb=SHORT_NULL_RSS_MB)
                    imputed_ids.add(f"{task_id}:{call_id}")
                    imputed_clauses += 1
                clauses.append(clause)
            adjusted[key] = replace(row, clauses=tuple(clauses))
            visited.add(key)
    if visited != set(command_rows):
        raise ValueError("short-null source rows differ from validation commands")
    return adjusted, imputed_ids, {
        "imputed_clauses": imputed_clauses,
        "imputed_commands": len(imputed_ids),
    }


def run(
    *,
    seeds: tuple[int, ...] = SEEDS,
    source_policy: str = "conservative",
) -> dict[str, Any]:
    _protocol_for(source_policy)
    if not seeds or not set(seeds) <= set(SEEDS):
        raise ValueError("RSS-safety smoke seeds must belong to frozen orders")
    validation_ids = list(json.loads(SPLIT.read_text(encoding="utf-8"))["validation"])
    if len(validation_ids) != 50 or LOAD != 40 or tuple(SEEDS) != tuple(range(32)):
        raise ValueError("RSS-safety frozen population or orders changed")
    clause_predictions, task_predictions, prediction_coverage = _static_predictions(
        validation_ids
    )
    command_rows, raw_counts, traces = _validation_commands(validation_ids)
    imputed_ids: set[str] = set()
    imputation = {"imputed_clauses": 0, "imputed_commands": 0}
    source_rows = command_rows
    if source_policy == "short-null-low":
        source_rows, imputed_ids, imputation = _short_null_command_rows(
            command_rows, traces
        )
    source_counts: Counter[str] = Counter()
    programs: dict[str, AdmissionProgram] = {
        task_id: _program(
            task_id,
            traces[task_id],
            source_rows,
            raw_counts,
            source_counts,
        )
        for task_id in validation_ids
    }
    profiles, _trace_identities = _profiles(programs, traces)
    command_ids = {
        command.command_id
        for program in programs.values()
        for command in program.commands
    }
    if set(profiles) != command_ids:
        raise ValueError("RSS-safety evaluation requires complete CPU profiles")
    oracle_rss = {
        command.command_id: command.rss_mb
        for program in programs.values()
        for command in program.commands
    }
    clause_rss = {
        command_id: reservation[1]
        for command_id, reservation in _full_reservation_map(
            programs, clause_predictions
        ).items()
    }
    task_rss = {
        command_id: reservation[1]
        for command_id, reservation in _full_reservation_map(
            programs, task_predictions
        ).items()
    }
    if set(oracle_rss) != command_ids or set(clause_rss) != command_ids or set(task_rss) != command_ids:
        raise ValueError("RSS-safety reservation maps differ from source commands")

    arm_specs = _arm_specs(command_ids, oracle_rss, clause_rss, task_rss)
    schedule_results: list[dict[str, Any]] = []
    violation = False
    exposure_totals = {arm: 0 for arm in ARMS}
    unverified_overlap_totals = {arm: 0 for arm in ARMS}
    for seed in seeds:
        selected = sorted(programs)
        np.random.default_rng(seed).shuffle(selected)
        selected = selected[:LOAD]
        chosen = [programs[task_id] for task_id in selected]
        selected_ids = {
            command.command_id
            for program in chosen
            for command in program.commands
        }
        selected_profiles = {
            command_id: profiles[command_id] for command_id in selected_ids
        }
        arms = {
            arm: simulate_idle_backfill(
                chosen,
                cpu_capacity=CPU_CAPACITY,
                rss_capacity_mb=RSS_CAPACITY_MB,
                cpu_work_profiles=selected_profiles,
                speculative_eligible_command_ids=eligible & selected_ids,
                rss_reservations={
                    command_id: rss[command_id] for command_id in selected_ids
                },
                rss_unverified_command_ids=imputed_ids & selected_ids,
                selection=selection,
            )
            for arm, (selection, eligible, rss) in arm_specs.items()
        }
        identity = {
            (
                arm["command_count"],
                arm["recorded_command_service_s"],
                arm["total_cpu_work_core_s"],
            )
            for arm in arms.values()
        }
        if len(identity) != 1:
            raise ValueError("RSS-safety arms used different source work")
        for arm_name, arm in arms.items():
            arm["service_inflation"] = _service_inflation(arm)
            arm["modeled_capacity_exposure_commands"] = len(
                arm["modeled_capacity_exposure_command_ids"]
            )
            arm["rss_unverified_overlap_commands"] = len(
                arm["rss_unverified_overlap_command_ids"]
            )
            exposure_totals[arm_name] += int(
                arm["modeled_capacity_exposure_events"]
            )
            unverified_overlap_totals[arm_name] += int(
                arm["rss_unverified_overlap_events"]
            )
            violation |= bool(
                arm["capacity_violation"]
                or arm["physical_capacity_violation"]
                or not math.isclose(
                    arm["total_cpu_work_core_s"],
                    arm["served_cpu_work_core_s"],
                    rel_tol=1e-12,
                    abs_tol=1e-7,
                )
            )
            arm.pop("modeled_capacity_exposure_command_ids")
            arm.pop("rss_unverified_overlap_command_ids")
            arm.pop("speculative_start_ids")
            arm.pop("service_s_by_command")
            arm.pop("start_s_by_command")
        schedule_results.append({"seed": seed, "task_ids": selected, "arms": arms})

    means = _mean_metrics(schedule_results)
    comparisons = {
        name: _comparison(
            schedule_results,
            means,
            baseline="serial8",
            candidate=arm,
        )
        for name, arm in (
            ("oracle", "oracle_rss_fcfs"),
            ("clause_kb", "clause_kb_rss_fcfs"),
            ("task_aware", "task_aware_rss_fcfs"),
        )
    }
    task = comparisons["task_aware"]
    if source_policy == "conservative":
        gate: dict[str, Any] = _gate(
            candidate_reduction=task["mean_per_order_relative_reduction"],
            oracle_reduction=comparisons["oracle"][
                "mean_per_order_relative_reduction"
            ],
            clause_reduction=comparisons["clause_kb"][
                "mean_per_order_relative_reduction"
            ],
            bootstrap_high=task["ci95_paired_seed_bootstrap"][1],
            service_inflation=task["candidate_service_inflation"],
            makespan_regression=task["makespan_regression"],
            exposure_events=exposure_totals["task_aware_rss_fcfs"],
            violation=violation,
        )
        selected_arm = None
    else:
        amended_gates = {
            name: _short_null_gate(
                candidate_reduction=comparisons[name][
                    "mean_per_order_relative_reduction"
                ],
                oracle_reduction=comparisons["oracle"][
                    "mean_per_order_relative_reduction"
                ],
                bootstrap_high=comparisons[name][
                    "ci95_paired_seed_bootstrap"
                ][1],
                service_inflation=comparisons[name]["candidate_service_inflation"],
                makespan_regression=comparisons[name]["makespan_regression"],
                exposure_events=exposure_totals[arm],
                violation=violation,
            )
            for name, arm in (
                ("clause_kb", "clause_kb_rss_fcfs"),
                ("task_aware", "task_aware_rss_fcfs"),
            )
        }
        passing = [name for name, values in amended_gates.items() if values["go"]]
        selected_arm = (
            max(
                passing,
                key=lambda name: comparisons[name][
                    "mean_per_order_relative_reduction"
                ],
            )
            if passing
            else None
        )
        gate = {
            "go": selected_arm is not None,
            "selected_arm": selected_arm,
            "arms": amended_gates,
        }
    integrity = {
        "all_arms_use_identical_commands_and_source_work": all(
            len(
                {
                    (
                        arm["command_count"],
                        arm["recorded_command_service_s"],
                        arm["total_cpu_work_core_s"],
                    )
                    for arm in row["arms"].values()
                }
            )
            == 1
            for row in schedule_results
        ),
        "all_commands_have_cpu_profiles": set(profiles) == command_ids,
        "zero_predicted_capacity_physical_cpu_or_work_violations": not violation,
        "serial_reconstructs_recorded_service": all(
            math.isclose(
                row["arms"]["serial8"]["total_command_service_s"],
                row["arms"]["serial8"]["recorded_command_service_s"],
                rel_tol=1e-9,
                abs_tol=1e-6,
            )
            for row in schedule_results
        ),
        "oracle_has_zero_source_rss_exposures": (
            exposure_totals["oracle_rss_fcfs"] == 0
        ),
    }
    if not all(integrity.values()):
        raise ValueError(f"RSS-safety integrity failure: {integrity}")

    return {
        "schema": VERSION if source_policy == "conservative" else SHORT_NULL_VERSION,
        "status": (
            (
                "development_go_to_short_null_memory_calibration"
                if gate["go"]
                else "development_stop_short_null_cpu_idle"
            )
            if source_policy == "short-null-low"
            else (
                "development_go_to_physical_cpu_idle_calibration"
                if gate["go"]
                else "development_stop_predictor_backed_cpu_idle"
            )
        ),
        "claim_bearing": False,
        "protocol": {
            "task_pool": "development-exposed SQLGlot validation50",
            "source_policy": source_policy,
            "load": LOAD,
            "seeds": list(seeds),
            "cpu_capacity": CPU_CAPACITY,
            "rss_capacity_mb": RSS_CAPACITY_MB,
            "rss_class_reservations_mb": [500.0, 2_000.0, 16_000.0],
            "selection": "FCFS in every backfill arm",
            "cpu_execution": "strict normal priority; speculation uses residual CPU",
            "minimum_candidate_reduction": MINIMUM_CANDIDATE_REDUCTION,
            "minimum_oracle_capture": MINIMUM_ORACLE_CAPTURE,
            "maximum_service_inflation": MAXIMUM_SERVICE_INFLATION,
            "maximum_makespan_regression": MAXIMUM_MAKESPAN_REGRESSION,
            **(
                {"minimum_gain_over_clause": MINIMUM_GAIN_OVER_CLAUSE}
                if source_policy == "conservative"
                else {
                    "short_null_max_latency_ms": SHORT_NULL_MAX_LATENCY_MS,
                    "short_null_rss_mb": SHORT_NULL_RSS_MB,
                    "short_null_reason": SHORT_NULL_REASON,
                }
            ),
        },
        "coverage": {
            "tasks": len(programs),
            "commands": len(command_ids),
            "commands_with_valid_cpu_profile": len(profiles),
            "commands_with_valid_source_rss": sum(
                count
                for source, count in source_counts.items()
                if source in {"observed_upper_bound", "cpu_null_target_fallback"}
            ),
            "prediction": prediction_coverage,
            "telemetry_reservation_sources": dict(sorted(source_counts.items())),
            "short_null_imputation": imputation,
        },
        "source_files": _source_identities(traces),
        "prediction_input_files": _file_identities(
            (FIT_ROWS, PHASE_ROWS, PHASE_RESULT, PHASE_ARTIFACT)
        ),
        "mean_metrics": means,
        "comparisons": comparisons,
        "source_rss_exposure_events": exposure_totals,
        "short_null_unverified_overlap_events": unverified_overlap_totals,
        "gate": gate,
        "integrity": integrity,
        "schedule_results": schedule_results,
        "limitations": [
            "All tasks, predictions, and outcomes are development-exposed.",
            "Strict CPU priority is an optimistic ceiling for cgroup cpu.idle.",
            (
                "Short insufficient-sample nulls are imputed to 500 MB but remain physically unverified."
                if source_policy == "short-null-low"
                else "Unavailable source RSS is conservatively modeled as 16,000 MB."
            ),
            "Disk and network contention receive no modeled performance benefit.",
            "CPU work is uniform within each source telemetry interval.",
        ],
    }


def _require_clean_committed_inputs(source_policy: str) -> None:
    protocol = _protocol_for(source_policy)
    paths = (
        Path(__file__).resolve(),
        protocol.resolve(),
        SPLIT.resolve(),
        FIT_ROWS.resolve(),
        PHASE_ROWS.resolve(),
        PHASE_RESULT.resolve(),
        PHASE_ARTIFACT.resolve(),
    )
    for path in paths:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            cwd=_ROOT,
            check=True,
            capture_output=True,
        )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("formal RSS-safety evaluation requires a clean worktree")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--source-policy",
        choices=SOURCE_POLICIES,
        default="conservative",
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("output directory already exists")
    _require_clean_committed_inputs(args.source_policy)
    result = run(source_policy=args.source_policy)
    protocol = _protocol_for(args.source_policy)
    result["inputs"] = {
        "split": str(SPLIT.resolve()),
        "protocol": str(protocol.resolve()),
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    }
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
