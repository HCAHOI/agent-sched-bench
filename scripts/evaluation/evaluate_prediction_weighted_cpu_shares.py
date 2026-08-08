#!/usr/bin/env python3
"""Test latency predictions as work-conserving CPU-share priorities."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from scripts.evaluation.evaluate_cpu_throughput_oracle import (  # noqa: E402
    THROUGHPUT_RESULT,
    _program_snapshot,
)
from scripts.evaluation.evaluate_kv_prediction_actionability import (  # noqa: E402
    FIT_ROWS,
    PHASE_ARTIFACT,
    PHASE_RESULT,
    PHASE_ROWS,
    SPLIT,
    VALIDATION_RUN,
    _static_predictions,
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
    simulate_burstable_admission,
)


VERSION = "prediction-weighted-cpu-shares-v1"
LATENCY_BUCKET_COUNT = 5
MINIMUM_COMPLETION_REDUCTION = 0.02
MAXIMUM_MAKESPAN_REGRESSION = 0.01
MINIMUM_ORACLE_REDUCTION = 0.05
MINIMUM_CHANGED_COMMANDS = 20
MINIMUM_CHANGED_TASKS = 10


def _priority_weight(pmf: Sequence[float] | None) -> float:
    if pmf is None:
        return 1.0
    if len(pmf) != LATENCY_BUCKET_COUNT or any(
        not math.isfinite(float(value)) or float(value) < 0.0 for value in pmf
    ):
        raise ValueError("latency priority requires a valid five-bucket PMF")
    hard = max(range(len(pmf)), key=pmf.__getitem__)
    return float(LATENCY_BUCKET_COUNT - hard)


def _gate(
    *,
    predicted_reduction: float,
    predicted_ci: tuple[float, float],
    makespan_regression: float,
    oracle_reduction: float,
    changed_commands: int,
    changed_tasks: int,
) -> dict[str, bool]:
    gate = {
        "predicted_mean_completion_reduction_at_least_2_percent": predicted_reduction
        >= MINIMUM_COMPLETION_REDUCTION,
        "paired_completion_ci_below_zero": predicted_ci[1] < 0.0,
        "makespan_regression_at_most_1_percent": makespan_regression
        <= MAXIMUM_MAKESPAN_REGRESSION,
        "oracle_mean_completion_reduction_at_least_5_percent": oracle_reduction
        >= MINIMUM_ORACLE_REDUCTION,
        "at_least_20_changed_commands": changed_commands
        >= MINIMUM_CHANGED_COMMANDS,
        "at_least_10_changed_tasks": changed_tasks >= MINIMUM_CHANGED_TASKS,
    }
    gate["go"] = all(gate.values())
    return gate


def _weights(
    commands: set[str],
    pmfs: Mapping[tuple[str, str], Sequence[float]],
) -> dict[str, float]:
    by_id = {f"{task_id}:{call_id}": pmf for (task_id, call_id), pmf in pmfs.items()}
    if not set(by_id) <= commands:
        raise ValueError("latency predictions contain commands outside the scheduler")
    return {
        command_id: _priority_weight(by_id.get(command_id))
        for command_id in commands
    }


def _source_artifacts(task_ids: Sequence[str]) -> dict[str, dict[str, dict[str, str]]]:
    artifacts: dict[str, dict[str, dict[str, str]]] = {}
    for task_id in task_ids:
        attempt = VALIDATION_RUN / task_id / "attempt_1"
        artifacts[task_id] = {}
        for name in ("trace.jsonl", "resource_observations.json"):
            path = attempt / name
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            artifacts[task_id][name] = {
                "path": str(path.resolve()),
                "sha256": digest,
            }
    return artifacts


def _complete_program_snapshot(
    programs: Mapping[str, AdmissionProgram],
    cpu_work: Mapping[str, float],
    max_cpu: Mapping[str, float],
) -> dict[str, Any]:
    return {
        task_id: {
            "initial_delay_s": program.initial_delay_s,
            "tail_s": program.tail_s,
            "commands": [
                {
                    "command_id": command.command_id,
                    "recorded_duration_s": command.duration_s,
                    "peak_cpu_cores": command.cpu_cores,
                    "rss_mb": command.rss_mb,
                    "delay_after_s": command.delay_after_s,
                    "cpu_work_core_s": cpu_work.get(command.command_id),
                    "max_cpu_cores": max_cpu[command.command_id],
                }
                for command in program.commands
            ],
        }
        for task_id, program in sorted(programs.items())
    }


def _service_changes(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> set[str]:
    if set(candidate) != set(baseline):
        raise ValueError("CPU-share arms contain different command service rows")
    return {
        command_id
        for command_id in candidate
        if not math.isclose(
            float(candidate[command_id]),
            float(baseline[command_id]),
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
    }


def _comparison(
    schedule_results: Sequence[Mapping[str, Any]],
    means: Mapping[str, Mapping[str, float]],
    candidate: str,
) -> dict[str, Any]:
    deltas = [
        float(row["arms"][candidate]["mean_task_completion_s"])
        - float(row["arms"]["equal_share"]["mean_task_completion_s"])
        for row in schedule_results
    ]
    baseline = means["equal_share"]["mean_task_completion_s"]
    candidate_mean = means[candidate]["mean_task_completion_s"]
    return {
        "candidate": candidate,
        "baseline": "equal_share",
        "metric": "mean task completion time; lower is better",
        "relative_reduction_of_means": (baseline - candidate_mean) / baseline,
        **_bootstrap(deltas),
    }


def run() -> dict[str, Any]:
    prior = json.loads(THROUGHPUT_RESULT.read_text(encoding="utf-8"))
    task_ids = list(json.loads(SPLIT.read_text(encoding="utf-8"))["validation"])
    if task_ids != prior["protocol"]["validation_task_ids"]:
        raise ValueError("validation cohort differs from the reviewed CPU-work result")
    cpu_work = {
        command_id: float(row["cpu_work_core_s"])
        for command_id, row in prior["program_inputs_by_command"].items()
        if row["cpu_work_core_s"] is not None
    }
    source_artifacts = _source_artifacts(task_ids)
    command_rows, raw_counts, traces = _validation_commands(task_ids)
    source_counts: Counter[str] = Counter()
    programs: dict[str, AdmissionProgram] = {
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
        raise ValueError("command programs differ from the reviewed throughput result")
    commands = {
        command.command_id
        for program in programs.values()
        for command in program.commands
    }
    max_cpu = {
        command_id: float(request[0])
        for command_id, request in prior["requests"]["throughput_bucket"].items()
    }
    if set(max_cpu) != commands or set(cpu_work) > commands:
        raise ValueError("reviewed CPU demand differs from scheduler commands")
    if _source_artifacts(task_ids) != source_artifacts:
        raise ValueError("source artifacts changed while constructing programs")
    program_snapshot = _complete_program_snapshot(programs, cpu_work, max_cpu)

    clause_pmfs, task_pmfs, prediction_source = _static_predictions(task_ids)
    labels = {
        f"{row['task_id']}:{row['call_id']}": int(row["latency_label"])
        for row in map(json.loads, PHASE_ROWS.read_text(encoding="utf-8").splitlines())
    }
    if not set(labels) <= commands or any(
        label not in range(LATENCY_BUCKET_COUNT) for label in labels.values()
    ):
        raise ValueError("latency labels differ from scheduler commands")
    weights = {
        "equal_share": {command_id: 1.0 for command_id in commands},
        "clause_kb_priority": _weights(commands, clause_pmfs),
        "task_aware_priority": _weights(commands, task_pmfs),
        "latency_oracle_priority": {
            command_id: float(
                LATENCY_BUCKET_COUNT - labels.get(command_id, LATENCY_BUCKET_COUNT - 1)
            )
            for command_id in commands
        },
    }
    configured_non_unit = {
        command_id
        for command_id in commands
        if weights["task_aware_priority"][command_id] != 1.0
    }
    reservations = {
        command.command_id: (2.0, command.rss_mb)
        for program in programs.values()
        for command in program.commands
    }

    prior_by_seed = {int(row["seed"]): row for row in prior["schedule_results"]}
    schedule_results = []
    effective_changed: set[str] = set()
    for seed in SEEDS:
        selected = sorted(programs)
        np.random.default_rng(seed).shuffle(selected)
        selected = selected[:LOAD]
        if selected != prior_by_seed[seed]["task_ids"]:
            raise ValueError("paired scheduler task selection drifted")
        selected_programs = [programs[task_id] for task_id in selected]
        selected_commands = {
            command.command_id
            for program in selected_programs
            for command in program.commands
        }
        arm_results = {
            arm: simulate_burstable_admission(
                selected_programs,
                cpu_capacity=CPU_CAPACITY,
                rss_capacity_mb=RSS_CAPACITY_MB,
                requested_reservations={
                    command_id: reservations[command_id]
                    for command_id in selected_commands
                },
                cpu_work_core_s={
                    command_id: value
                    for command_id, value in cpu_work.items()
                    if command_id in selected_commands
                },
                max_cpu_cores={
                    command_id: max_cpu[command_id]
                    for command_id in selected_commands
                },
                cpu_share_weights={
                    command_id: arm_weights[command_id]
                    for command_id in selected_commands
                },
            )
            for arm, arm_weights in weights.items()
        }
        identities = {
            (
                int(metrics["command_count"]),
                float(metrics["total_cpu_work_core_s"]),
                float(metrics["served_cpu_work_core_s"]),
            )
            for metrics in arm_results.values()
        }
        if len(identities) != 1 or any(
            bool(metrics["capacity_violation"]) for metrics in arm_results.values()
        ):
            raise ValueError("CPU-share arms differ in commands, work, or capacity")
        equal_service = arm_results["equal_share"]["service_s_by_command"]
        predicted_service = arm_results["task_aware_priority"][
            "service_s_by_command"
        ]
        effective_changed.update(_service_changes(predicted_service, equal_service))
        arms = {
            arm: {
                key: value
                for key, value in metrics.items()
                if key != "service_s_by_command"
            }
            for arm, metrics in arm_results.items()
        }
        schedule_results.append({"seed": seed, "task_ids": selected, "arms": arms})

    effective_changed_tasks = {
        command_id.split(":", 1)[0] for command_id in effective_changed
    }

    metrics = (
        "makespan_s",
        "mean_task_completion_s",
        "total_command_queue_s",
        "total_command_service_s",
        "reserved_cpu_core_s",
        "max_concurrent_commands",
    )
    means = {
        arm: {
            metric: float(
                np.mean([row["arms"][arm][metric] for row in schedule_results])
            )
            for metric in metrics
        }
        for arm in weights
    }
    comparisons = {
        arm: _comparison(schedule_results, means, arm)
        for arm in (
            "clause_kb_priority",
            "task_aware_priority",
            "latency_oracle_priority",
        )
    }
    primary = comparisons["task_aware_priority"]
    equal_makespan = means["equal_share"]["makespan_s"]
    makespan_regression = (
        means["task_aware_priority"]["makespan_s"] - equal_makespan
    ) / equal_makespan
    gate = _gate(
        predicted_reduction=primary["relative_reduction_of_means"],
        predicted_ci=tuple(primary["ci95_paired_seed_bootstrap"]),
        makespan_regression=makespan_regression,
        oracle_reduction=comparisons["latency_oracle_priority"][
            "relative_reduction_of_means"
        ],
        changed_commands=len(effective_changed),
        changed_tasks=len(effective_changed_tasks),
    )
    return {
        "schema": VERSION,
        "status": (
            "development_promising_prediction_weighted_shares"
            if gate["go"]
            else "development_no_go_prediction_weighted_shares"
        ),
        "claim_bearing": False,
        "protocol": {
            "task_pool": "development-exposed SQLGlot validation50",
            "load": LOAD,
            "seeds": list(SEEDS),
            "cpu_capacity": CPU_CAPACITY,
            "admission_charge_cores": 2.0,
            "priority": "five minus predicted latency-bucket index; unavailable maps to one",
            "primary": "Task-Aware priority versus equal shares mean task completion",
            "minimum_completion_reduction": MINIMUM_COMPLETION_REDUCTION,
            "maximum_makespan_regression": MAXIMUM_MAKESPAN_REGRESSION,
        },
        "coverage": {
            "tasks": len(programs),
            "commands": len(commands),
            "prediction_source": prediction_source,
            "configured_non_unit_weight_commands": len(configured_non_unit),
            "effective_service_changed_commands": len(effective_changed),
            "effective_service_changed_tasks": len(effective_changed_tasks),
            "telemetry_sources": dict(sorted(source_counts.items())),
        },
        "source_artifacts": source_artifacts,
        "program_snapshot": program_snapshot,
        "mean_metrics": means,
        "comparisons": comparisons,
        "primary_makespan_regression": makespan_regression,
        "gate": gate,
        "schedule_results": schedule_results,
        "limitations": [
            "This is a fluid CPU-work simulation, not physical CFS scheduling.",
            "CPU work is divisible and RSS/demand classes use hindsight telemetry.",
            "Predictions and all SQLGlot tasks are development-exposed.",
            "The policy optimizes mean completion and only gates makespan non-regression.",
        ],
    }


def _require_committed_inputs() -> None:
    paths = (
        Path(__file__).resolve(),
        (_ROOT / "src/tool_resource_eval/resource_admission.py").resolve(),
        (_ROOT / "src/tool_resource_eval/cachewise_kv_factorial.py").resolve(),
        (_ROOT / "scripts/evaluation/evaluate_cpu_throughput_oracle.py").resolve(),
        (
            _ROOT / "scripts/evaluation/evaluate_kv_prediction_actionability.py"
        ).resolve(),
        (_ROOT / "scripts/evaluation/evaluate_resource_admission_oracle.py").resolve(),
        (
            _ROOT / "scripts/evaluation/evaluate_resource_admission_predictors.py"
        ).resolve(),
        (_ROOT / "scripts/evaluation/evaluate_semantic_work_units.py").resolve(),
        (_ROOT / "scripts/evaluation/evaluate_command_history_residual.py").resolve(),
        THROUGHPUT_RESULT.resolve(),
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
    subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *(str(path) for path in paths)],
        cwd=_ROOT,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "scripts/evaluation",
            "src/tool_resource",
            "src/tool_resource_eval",
            "src/trace_collect/trace_data.py",
        ],
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
        "throughput_result": str(THROUGHPUT_RESULT.resolve()),
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
