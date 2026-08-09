#!/usr/bin/env python3
"""Evaluate strict-idle CPU backfill and its selection headroom."""

from __future__ import annotations

import argparse
from collections import Counter
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
    _file_identity,
    _profiles,
    _service_inflation,
)
from scripts.evaluation.evaluate_kv_prediction_actionability import (  # noqa: E402
    SPLIT,
)
from scripts.evaluation.evaluate_resource_admission_oracle import (  # noqa: E402
    CPU_CAPACITY,
    RSS_CAPACITY_MB,
    _program,
    _reservation,
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
    simulate_idle_backfill,
)


VERSION = "cpu-idle-backfill-oracle-v1"
ARMS = ("serial8", "fcfs_idle", "oracle_idle")
ACTION_MINIMUM_REDUCTION = 0.05
SELECTION_MINIMUM_REDUCTION = 0.10
MAXIMUM_SERVICE_INFLATION = 0.05
MAXIMUM_MAKESPAN_REGRESSION = 0.01
PROTOCOL = (
    _ROOT / "analysis/development/cpu-idle-speculative-backfill-protocol.md"
)


def _gate(
    *,
    reduction: float,
    minimum_reduction: float,
    bootstrap_high: float,
    service_inflation: float,
    makespan_regression: float,
    violation: bool,
) -> dict[str, bool]:
    gate = {
        "mean_task_completion_reduction_meets_minimum": (
            reduction >= minimum_reduction
        ),
        "paired_order_bootstrap_upper_below_zero": bootstrap_high < 0.0,
        "service_inflation_at_most_5_percent": (
            service_inflation <= MAXIMUM_SERVICE_INFLATION
        ),
        "makespan_regression_at_most_1_percent": (
            makespan_regression <= MAXIMUM_MAKESPAN_REGRESSION
        ),
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


def _comparison(
    schedule_results: list[dict[str, Any]],
    means: dict[str, Any],
    *,
    baseline: str,
    candidate: str,
) -> dict[str, Any]:
    relative_reductions = []
    deltas = []
    for row in schedule_results:
        baseline_value = float(row["arms"][baseline]["mean_task_completion_s"])
        candidate_value = float(row["arms"][candidate]["mean_task_completion_s"])
        relative_reductions.append(
            (baseline_value - candidate_value) / baseline_value
        )
        deltas.append(candidate_value - baseline_value)
    return {
        "baseline": baseline,
        "candidate": candidate,
        "metric": "mean task completion seconds; lower is better",
        "mean_per_order_relative_reduction": float(
            np.mean(relative_reductions)
        ),
        "orders_improved": sum(delta < 0.0 for delta in deltas),
        "candidate_service_inflation": _service_inflation(means[candidate]),
        "makespan_regression": (
            means[candidate]["makespan_s"] / means[baseline]["makespan_s"] - 1.0
        ),
        **_bootstrap(deltas),
    }


def _source_identities(
    traces: dict[str, Path],
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        task_id: {
            "trace": _file_identity(trace_path),
            "resource_observations": _file_identity(
                trace_path.parent / "resource_observations.json"
            ),
        }
        for task_id, trace_path in sorted(traces.items())
    }


def _rss_eligible_command_ids(
    programs: dict[str, AdmissionProgram],
    command_rows: dict[tuple[str, str], Any],
    raw_counts: dict[tuple[str, str], int],
) -> set[str]:
    eligible = set()
    for task_id, program in programs.items():
        prefix = f"{task_id}:"
        for command in program.commands:
            call_id = command.command_id.removeprefix(prefix)
            key = (task_id, call_id)
            row = command_rows.get(key)
            if row is None or raw_counts.get(key) != len(row.clauses):
                continue
            source = _reservation(row)[2]
            if source in {"observed_upper_bound", "cpu_null_target_fallback"}:
                eligible.add(command.command_id)
    return eligible


def run(*, seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    if not seeds or not set(seeds) <= set(SEEDS):
        raise ValueError("idle-backfill smoke seeds must belong to the frozen orders")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    validation_ids = list(split["validation"])
    if len(validation_ids) != 50 or LOAD != 40 or tuple(SEEDS) != tuple(range(32)):
        raise ValueError("idle-backfill frozen population or orders changed")
    command_rows, raw_counts, traces = _validation_commands(validation_ids)
    source_counts: Counter[str] = Counter()
    programs: dict[str, AdmissionProgram] = {
        task_id: _program(
            task_id,
            traces[task_id],
            command_rows,
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
        raise ValueError("idle-backfill evaluation requires complete CPU profiles")
    rss_eligible_ids = _rss_eligible_command_ids(programs, command_rows, raw_counts)

    selections = {
        "serial8": "serial",
        "fcfs_idle": "fcfs",
        "oracle_idle": "shortest",
    }
    schedule_results: list[dict[str, Any]] = []
    violation = False
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
                speculative_eligible_command_ids=rss_eligible_ids & selected_ids,
                selection=selection,
            )
            for arm, selection in selections.items()
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
            raise ValueError("idle-backfill arms used different source work")
        for arm in arms.values():
            arm["service_inflation"] = _service_inflation(arm)
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
            arm.pop("speculative_start_ids")
            arm.pop("service_s_by_command")
            arm.pop("start_s_by_command")
        schedule_results.append({"seed": seed, "task_ids": selected, "arms": arms})

    means = _mean_metrics(schedule_results)
    action = _comparison(
        schedule_results,
        means,
        baseline="serial8",
        candidate="fcfs_idle",
    )
    selection = _comparison(
        schedule_results,
        means,
        baseline="fcfs_idle",
        candidate="oracle_idle",
    )
    action_gate = _gate(
        reduction=action["mean_per_order_relative_reduction"],
        minimum_reduction=ACTION_MINIMUM_REDUCTION,
        bootstrap_high=action["ci95_paired_seed_bootstrap"][1],
        service_inflation=action["candidate_service_inflation"],
        makespan_regression=action["makespan_regression"],
        violation=violation,
    )
    selection_gate = _gate(
        reduction=selection["mean_per_order_relative_reduction"],
        minimum_reduction=SELECTION_MINIMUM_REDUCTION,
        bootstrap_high=selection["ci95_paired_seed_bootstrap"][1],
        service_inflation=selection["candidate_service_inflation"],
        makespan_regression=selection["makespan_regression"],
        violation=violation,
    )
    advance = action_gate["go"] and selection_gate["go"]
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
        "zero_capacity_or_work_violations": not violation,
        "serial_reconstructs_recorded_service": all(
            math.isclose(
                row["arms"]["serial8"]["total_command_service_s"],
                row["arms"]["serial8"]["recorded_command_service_s"],
                rel_tol=1e-9,
                abs_tol=1e-6,
            )
            for row in schedule_results
        ),
    }
    if not all(integrity.values()):
        raise ValueError(f"idle-backfill integrity failure: {integrity}")

    return {
        "schema": VERSION,
        "status": (
            "development_go_to_predictor_selector_protocol"
            if advance
            else (
                "development_idle_backfill_without_prediction_headroom"
                if action_gate["go"]
                else "development_stop_idle_backfill_action"
            )
        ),
        "claim_bearing": False,
        "protocol": {
            "task_pool": "development-exposed SQLGlot validation50",
            "load": LOAD,
            "seeds": list(seeds),
            "cpu_capacity": CPU_CAPACITY,
            "rss_capacity_mb": RSS_CAPACITY_MB,
            "normal_slots": 1,
            "speculative_slots": 1,
            "cpu_execution": "strict normal priority; speculation uses residual CPU",
            "action_minimum_reduction": ACTION_MINIMUM_REDUCTION,
            "selection_minimum_reduction": SELECTION_MINIMUM_REDUCTION,
            "maximum_service_inflation": MAXIMUM_SERVICE_INFLATION,
            "maximum_makespan_regression": MAXIMUM_MAKESPAN_REGRESSION,
        },
        "coverage": {
            "tasks": len(programs),
            "commands": len(command_ids),
            "commands_with_valid_cpu_profile": len(profiles),
            "commands_with_valid_speculative_rss": len(rss_eligible_ids),
            "telemetry_reservation_sources": dict(sorted(source_counts.items())),
        },
        "source_files": _source_identities(traces),
        "mean_metrics": means,
        "comparisons": {"action": action, "selection": selection},
        "gate": {
            "action": action_gate,
            "selection": selection_gate,
            "advance_to_predictor_selector": advance,
        },
        "integrity": integrity,
        "schedule_results": schedule_results,
        "limitations": [
            "All tasks and outcomes are development-exposed.",
            "Strict CPU priority is an optimistic ceiling for cgroup cpu.idle.",
            "Both backfill arms use hindsight RSS fit for a shared safety control.",
            "Disk and network contention receive no modeled performance benefit.",
            "CPU work is uniform within each source telemetry interval.",
        ],
    }


def _committed_input_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        PROTOCOL.resolve(),
        SPLIT.resolve(),
        (_ROOT / "src/tool_resource_eval/resource_admission.py").resolve(),
        (_ROOT / "src/tool_resource_eval/early_cpu_reservation.py").resolve(),
        (_ROOT / "src/tool_resource/runtime_kb.py").resolve(),
        (_ROOT / "src/trace_collect/trace_data.py").resolve(),
        (_ROOT / "scripts/evaluation/evaluate_cpu_feedback_admission.py").resolve(),
        (_ROOT / "scripts/evaluation/evaluate_clause_resource_classes.py").resolve(),
        (_ROOT / "scripts/evaluation/evaluate_kv_prediction_actionability.py").resolve(),
        (_ROOT / "scripts/evaluation/evaluate_resource_admission_oracle.py").resolve(),
        (_ROOT / "scripts/evaluation/evaluate_resource_admission_predictors.py").resolve(),
        (_ROOT / "src/tool_resource_eval/cachewise_kv_factorial.py").resolve(),
    )


def _require_committed_inputs() -> None:
    paths = _committed_input_paths()
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("output directory already exists")
    _require_committed_inputs()
    result = run()
    result["inputs"] = {
        "split": str(SPLIT.resolve()),
        "protocol": str(PROTOCOL.resolve()),
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
