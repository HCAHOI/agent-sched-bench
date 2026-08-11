#!/usr/bin/env python3
"""Evaluate the frozen Zarr weak-compound-evidence RSS safety gate."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
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
    _load_dataset,
    _load_split,
    _predictions,
    _run_traces,
)
from scripts.evaluation.evaluate_zarr_rss_evidence_gate import (  # noqa: E402
    VALIDATION_RUN,
    _reservation_accuracy,
)
from tool_resource.runtime_kb import ClauseResourceKB  # noqa: E402
from tool_resource_eval.resource_admission import simulate_idle_backfill  # noqa: E402

VERSION = "zarr-rss-compound-evidence-gate-v1"


def _guarded_reservations(
    clause: Mapping[str, float],
    provenance: Mapping[str, Mapping[str, Any] | None],
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    candidate = dict(clause)
    all_compound = dict(clause)
    weak_ids: list[str] = []
    compound_ids: list[str] = []
    for command_id, item in provenance.items():
        if item is None or item["key_kind"] != "shell_execution_graph":
            continue
        compound_ids.append(command_id)
        all_compound[command_id] = RSS_CAPACITY_MB
        if any(
            not str(path).endswith(":exact_clause")
            for path in item["fallback_path"]
        ):
            weak_ids.append(command_id)
            candidate[command_id] = RSS_CAPACITY_MB

    changed = [command_id for command_id in weak_ids if clause[command_id] < RSS_CAPACITY_MB]
    all_changed = [
        command_id
        for command_id in compound_ids
        if clause[command_id] < RSS_CAPACITY_MB
    ]
    return candidate, all_compound, {
        "weak_compound_predictions": len(weak_ids),
        "changed_reservations": len(changed),
        "changed_tasks": len({value.split(":", 1)[0] for value in changed}),
        "changed_command_counts": dict(
            Counter(str(provenance[value]["command"]) for value in changed)  # type: ignore[index]
        ),
        "changed_ids": changed,
        "all_compound_predictions": len(compound_ids),
        "all_compound_changed_reservations": len(all_changed),
    }


def _clause_provenance(
    fit: Dataset,
    target: Dataset,
    clause: Mapping[str, float],
    *,
    leave_one_out: bool,
) -> dict[str, Mapping[str, Any] | None]:
    output: dict[str, Mapping[str, Any] | None] = {}
    for target_task in target.task_ids:
        fit_task_ids = tuple(
            task_id
            for task_id in fit.task_ids
            if not leave_one_out or task_id != target_task
        )
        kb = ClauseResourceKB()
        for task_id in fit_task_ids:
            for row in fit.clauses_by_task[task_id]:
                kb.observe_completed_clause(row.observation(0.0, 1.0))
        for row in target.commands_by_task[target_task]:
            command_id = f"{row.task_id}:{row.call_id}"
            prediction = kb.predict_command_resource_buckets(
                row.repo, row.command, 3.0
            ).classifications[RSS_TARGET]
            expected = (
                RSS_CAPACITY_MB
                if prediction is None
                else RSS_REQUESTS[prediction.bucket_id]
            )
            if clause[command_id] != expected:
                raise ValueError("Clause-KB reservation and provenance disagree")
            output[command_id] = (
                None
                if prediction is None
                else {
                    "command": row.command,
                    "key_kind": prediction.key_kind,
                    "fallback_path": list(prediction.fallback_path),
                }
            )
    return output


def _evaluate_cohort(
    fit: Dataset, target: Dataset, *, leave_one_out: bool
) -> dict[str, Any]:
    reservations, prediction = _predictions(
        fit, target, leave_one_out=leave_one_out
    )
    clause = reservations["clause_kb"]
    provenance = _clause_provenance(
        fit, target, clause, leave_one_out=leave_one_out
    )
    candidate, all_compound, guard = _guarded_reservations(clause, provenance)
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
            rss_reservations=clause,
            **common,
        ),
        "all_compound_guard": simulate_idle_backfill(
            programs,
            speculative_eligible_command_ids=command_ids,
            rss_reservations=all_compound,
            **common,
        ),
        "compound_evidence_guard": simulate_idle_backfill(
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
    primary = arms["compound_evidence_guard"]
    primary_comparison = comparisons["compound_evidence_guard"]
    clause_metrics = arms["clause_kb"]
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
        "retains_90_percent_of_clause_kb_reduction": clause_comparison[
            "mean_task_completion_reduction"
        ]
        > 0.0
        and primary_comparison["mean_task_completion_reduction"]
        >= 0.9 * clause_comparison["mean_task_completion_reduction"],
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
        "retains_80_percent_of_clause_kb_starts": primary[
            "speculative_starts"
        ]
        >= 0.8 * clause_metrics["speculative_starts"],
        "at_least_20_speculative_starts": primary["speculative_starts"] >= 20,
        "at_least_5_speculative_tasks": primary["speculative_task_count"] >= 5,
    }
    gate["go"] = all(gate.values())
    return {
        "tasks": list(target.task_ids),
        "leave_one_out": leave_one_out,
        "prediction_rows": prediction["rows"],
        "guard": guard,
        "reservation_accuracy": {
            "clause_kb": _reservation_accuracy(target, clause),
            "all_compound_guard": _reservation_accuracy(target, all_compound),
            "compound_evidence_guard": _reservation_accuracy(target, candidate),
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
            "candidate": "Clause-KB with non-exact compound predictions reserved at 16000 MB",
            "mechanism_control": "all compound predictions reserved at 16000 MB",
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
            "Both cohorts and the exposure cases used to design this guard are development-exposed.",
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
