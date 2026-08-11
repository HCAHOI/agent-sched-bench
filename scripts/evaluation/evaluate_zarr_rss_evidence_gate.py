#!/usr/bin/env python3
"""Evaluate the frozen post-hoc exact-supported Zarr RSS demotion policy."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from scripts.evaluation.evaluate_zarr_rss_backfill import (  # noqa: E402
    CPU_CAPACITY,
    RSS_CAPACITY_MB,
    RSS_REQUESTS,
    RSS_TARGET,
    Dataset,
    _compact,
    _comparison,
    _development_traces,
    _labels,
    _load_dataset,
    _load_split,
    _predictions,
    _run_traces,
)
from tool_resource_eval.resource_admission import simulate_idle_backfill  # noqa: E402

VERSION = "zarr-rss-exact-demotion-v1"
MIN_EXACT_SUPPORT_TASKS = 2
VALIDATION_RUN = _ROOT / (
    "traces/swe-rebench/gpt-5.6-sol/"
    "zarr-phase-validation10-c1-fast-ebpf-20260811"
)


def _guarded_reservation(
    proposal_mb: float, labels_by_task: Mapping[str, int]
) -> float:
    """Return a sub-capacity proposal only with repeated conservative support."""

    if proposal_mb >= RSS_CAPACITY_MB:
        return RSS_CAPACITY_MB
    if len(labels_by_task) < MIN_EXACT_SUPPORT_TASKS:
        return RSS_CAPACITY_MB
    historical_upper_mb = RSS_REQUESTS[max(labels_by_task.values())]
    return proposal_mb if historical_upper_mb <= proposal_mb else RSS_CAPACITY_MB


def _exact_support(
    fit: Dataset, fit_task_ids: Sequence[str]
) -> dict[str, dict[str, int]]:
    labels: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for task_id in fit_task_ids:
        for row in fit.commands_by_task[task_id]:
            label = _labels(row)[RSS_TARGET]
            if label is not None:
                labels[row.command][task_id].append(label)
    return {
        command: {
            task_id: max(task_labels)
            for task_id, task_labels in by_task.items()
        }
        for command, by_task in labels.items()
    }


def _candidate_reservations(
    fit: Dataset,
    target: Dataset,
    reservations: Mapping[str, Mapping[str, float]],
    *,
    leave_one_out: bool,
) -> tuple[dict[str, float], dict[str, Any]]:
    output = {command_id: RSS_CAPACITY_MB for command_id in target.profiles}
    proposed_low: list[str] = []
    authorized_low: list[str] = []
    blocked_low: list[str] = []
    proposed_demotions: list[str] = []
    authorized_demotions: list[str] = []
    for target_task in target.task_ids:
        fit_task_ids = tuple(
            task_id
            for task_id in fit.task_ids
            if not leave_one_out or task_id != target_task
        )
        support = _exact_support(fit, fit_task_ids)
        for row in target.commands_by_task[target_task]:
            command_id = f"{row.task_id}:{row.call_id}"
            clause = reservations["clause_kb"][command_id]
            task_upper = reservations["task_aware_upper"][command_id]
            proposal = min(clause, task_upper)
            if task_upper < clause:
                proposed_demotions.append(command_id)
            if proposal < RSS_CAPACITY_MB:
                proposed_low.append(command_id)
            guarded = _guarded_reservation(proposal, support.get(row.command, {}))
            output[command_id] = guarded
            if guarded < RSS_CAPACITY_MB:
                authorized_low.append(command_id)
            elif proposal < RSS_CAPACITY_MB:
                blocked_low.append(command_id)
            if guarded < clause:
                authorized_demotions.append(command_id)
    return output, {
        "proposed_low": len(proposed_low),
        "authorized_low": len(authorized_low),
        "blocked_low": len(blocked_low),
        "proposed_demotions": len(proposed_demotions),
        "authorized_demotions": len(authorized_demotions),
        "authorized_task_count": len(
            {command_id.split(":", 1)[0] for command_id in authorized_low}
        ),
        "proposed_demotion_ids": proposed_demotions,
        "authorized_demotion_ids": authorized_demotions,
        "authorized_demotion_command_counts": dict(
            Counter(
                row.command
                for task_id in target.task_ids
                for row in target.commands_by_task[task_id]
                if f"{row.task_id}:{row.call_id}" in authorized_demotions
            )
        ),
        "blocked_low_ids": blocked_low,
    }


def _reservation_accuracy(
    target: Dataset, reservations: Mapping[str, float]
) -> dict[str, Any]:
    pairs: list[tuple[int, int]] = []
    for task_id in target.task_ids:
        for row in target.commands_by_task[task_id]:
            truth = _labels(row)[RSS_TARGET]
            if truth is not None:
                pairs.append(
                    (
                        truth,
                        RSS_REQUESTS.index(
                            reservations[f"{row.task_id}:{row.call_id}"]
                        ),
                    )
                )
    return {
        "eligible": len(pairs),
        "accuracy": sum(truth == predicted for truth, predicted in pairs)
        / len(pairs),
        "underpredictions": sum(predicted < truth for truth, predicted in pairs),
    }


def _evaluate_cohort(
    fit: Dataset, target: Dataset, *, leave_one_out: bool
) -> dict[str, Any]:
    reservations, prediction = _predictions(
        fit, target, leave_one_out=leave_one_out
    )
    ungated = {
        command_id: min(
            reservations["clause_kb"][command_id],
            reservations["task_aware_upper"][command_id],
        )
        for command_id in target.profiles
    }
    candidate, evidence = _candidate_reservations(
        fit, target, reservations, leave_one_out=leave_one_out
    )
    programs = [target.programs[task_id] for task_id in sorted(target.programs)]
    command_ids = set(target.profiles)
    known_source = {
        command_id
        for command_id, source in target.rss_source_by_command.items()
        if source != "full_fallback"
    }
    common = {
        "cpu_capacity": CPU_CAPACITY,
        "rss_capacity_mb": RSS_CAPACITY_MB,
        "cpu_work_profiles": target.profiles,
        "rss_unverified_command_ids": set(target.unverified_command_ids),
        "selection": "fcfs",
    }
    raw = {
        "serial8": simulate_idle_backfill(
            programs,
            cpu_capacity=CPU_CAPACITY,
            rss_capacity_mb=RSS_CAPACITY_MB,
            cpu_work_profiles=target.profiles,
            speculative_eligible_command_ids=set(),
            selection="serial",
        ),
        "source_bound_oracle": simulate_idle_backfill(
            programs,
            speculative_eligible_command_ids=known_source,
            **common,
        ),
        "clause_kb": simulate_idle_backfill(
            programs,
            speculative_eligible_command_ids=command_ids,
            rss_reservations=reservations["clause_kb"],
            **common,
        ),
        "ungated_demotion": simulate_idle_backfill(
            programs,
            speculative_eligible_command_ids=command_ids,
            rss_reservations=ungated,
            **common,
        ),
        "exact_supported_demotion": simulate_idle_backfill(
            programs,
            speculative_eligible_command_ids=command_ids,
            rss_reservations=candidate,
            **common,
        ),
    }
    arms = {name: _compact(metrics) for name, metrics in raw.items()}
    serial = arms["serial8"]
    comparisons = {
        name: _comparison(serial, metrics)
        for name, metrics in arms.items()
        if name != "serial8"
    }
    primary = arms["exact_supported_demotion"]
    primary_comparison = comparisons["exact_supported_demotion"]
    clause_comparison = comparisons["clause_kb"]
    work_ok = math.isclose(
        primary["total_cpu_work_core_s"],
        primary["served_cpu_work_core_s"],
        rel_tol=1e-12,
        abs_tol=1e-7,
    )
    gate = {
        "mean_completion_reduction_at_least_5_percent": primary_comparison[
            "mean_task_completion_reduction"
        ]
        >= 0.05,
        "beats_clause_kb_by_1_percentage_point": primary_comparison[
            "mean_task_completion_reduction"
        ]
        - clause_comparison["mean_task_completion_reduction"]
        >= 0.01,
        "makespan_lower_than_serial": primary_comparison["makespan_reduction"]
        > 0.0,
        "service_inflation_at_most_5_percent": primary["service_inflation"]
        <= 0.05,
        "zero_source_bound_or_capacity_violations": not (
            primary["modeled_capacity_exposure_events"]
            or primary["capacity_violation"]
            or primary["physical_capacity_violation"]
        ),
        "cpu_work_conserved": work_ok,
        "at_least_20_speculative_starts": primary["speculative_starts"] >= 20,
        "at_least_5_speculative_tasks": primary["speculative_task_count"] >= 5,
    }
    gate["go"] = all(gate.values())
    return {
        "tasks": list(target.task_ids),
        "leave_one_out": leave_one_out,
        "prediction_rows": prediction["rows"],
        "evidence_gate": evidence,
        "reservation_accuracy": {
            "clause_kb": _reservation_accuracy(
                target, reservations["clause_kb"]
            ),
            "ungated_demotion": _reservation_accuracy(target, ungated),
            "exact_supported_demotion": _reservation_accuracy(target, candidate),
        },
        "arms": arms,
        "comparisons_vs_serial8": comparisons,
        "candidate_minus_clause_kb_mean_reduction_percentage_points": 100.0
        * (
            primary_comparison["mean_task_completion_reduction"]
            - clause_comparison["mean_task_completion_reduction"]
        ),
        "gate": gate,
    }


def evaluate() -> dict[str, Any]:
    started = time.monotonic()
    split = _load_split()
    fit_ids = tuple(
        task_id
        for task_id in split["development"]
        if task_id not in split["model_fit_exclusions"]
    )
    fit = _load_dataset(fit_ids, _development_traces(fit_ids))
    validation_ids = tuple(split["validation"])
    validation = _load_dataset(
        validation_ids,
        _run_traces(VALIDATION_RUN.resolve(), validation_ids),
    )
    cohorts = {
        "development_loto": _evaluate_cohort(fit, fit, leave_one_out=True),
        "exposed_validation": _evaluate_cohort(
            fit, validation, leave_one_out=False
        ),
    }
    go = all(cohort["gate"]["go"] for cohort in cohorts.values())
    return {
        "schema": VERSION,
        "status": "posthoc_development_go" if go else "posthoc_development_no_go",
        "claim_bearing": False,
        "protocol": {
            "candidate": "min Clause-KB and Task-Aware upper, then exact support gate",
            "minimum_independent_exact_support_tasks": MIN_EXACT_SUPPORT_TASKS,
            "historical_upper_must_not_exceed_proposal": True,
            "target_updates": "none",
            "final_read": False,
        },
        "cohorts": cohorts,
        "gate": {"both_cohorts_go": go},
        "cost": {
            "prediction_time_agent_calls": 0,
            "gpu_runtime_s": 0.0,
            "evaluator_wall_s": time.monotonic() - started,
        },
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "inputs": {
            "validation_run": str(VALIDATION_RUN.resolve()),
            "final_read": False,
        },
        "limitations": [
            "Both cohorts are development-exposed; this result cannot confirm the policy.",
            "The source bound includes the frozen short-null upper policy.",
            "The scheduler is replayed from isolated profiles, not concurrent containers.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("output already exists")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("formal evaluation requires a clean committed checkout")
    result = evaluate()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.out), "status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
