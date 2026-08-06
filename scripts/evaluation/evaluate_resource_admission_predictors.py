#!/usr/bin/env python3
"""Evaluate frozen Current and SOTA CPU/RSS admission requests."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
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
    _row_from_clause,
)
from scripts.evaluation.evaluate_command_history_residual import (  # noqa: E402
    Row,
    _load_rows,
)
from scripts.evaluation.evaluate_kv_prediction_actionability import (  # noqa: E402
    DEVELOPMENT_RUN,
    FIT_ROWS,
    PHASE_ARTIFACT,
    PHASE_RESULT,
    PHASE_ROWS,
    SPLIT,
    VALIDATION_RUN,
    _dev100_static_rows,
)
from scripts.evaluation.evaluate_resource_admission_oracle import (  # noqa: E402
    CPU_CAPACITY,
    MINIMUM_MAKESPAN_REDUCTION,
    RSS_CAPACITY_MB,
    _program,
)
from scripts.evaluation.evaluate_semantic_work_units import (  # noqa: E402
    run as run_semantic_work_units,
)
from tool_resource.runtime_kb import RESOURCE_BUCKET_LABELS  # noqa: E402
from tool_resource_eval.cachewise_kv_factorial import (  # noqa: E402
    LOAD,
    SEEDS,
    _bootstrap,
)
from tool_resource_eval.resource_admission import simulate_admission  # noqa: E402


VERSION = "resource-admission-predictors-v1"
MINIMUM_SOTA_REDUCTION = 0.05
MINIMUM_ORACLE_HEADROOM_CAPTURE = 0.50
MINIMUM_CHANGED_COMMANDS = 20
MINIMUM_CHANGED_TASKS = 10
CPU_REQUESTS = (2.0, 4.0, CPU_CAPACITY)
RSS_REQUESTS = (500.0, 2_000.0, RSS_CAPACITY_MB)
TARGETS = ("peak_cpu_cores", "sampled_peak_rss_mb")


def _index(value: Any) -> int | None:
    if value is None:
        return None
    return RESOURCE_BUCKET_LABELS.index(str(value))


def _request(cpu: Any, rss: Any) -> tuple[float, float]:
    cpu_index = _index(cpu)
    rss_index = _index(rss)
    return (
        CPU_CAPACITY if cpu_index is None else CPU_REQUESTS[cpu_index],
        RSS_CAPACITY_MB if rss_index is None else RSS_REQUESTS[rss_index],
    )


def _static_predictions(
    validation_ids: list[str],
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]], dict[str, Any]]:
    values = [json.loads(line) for line in PHASE_ROWS.read_text().splitlines()]
    task_order = list(dict.fromkeys(str(value["task_id"]) for value in values))
    call_keys = [(str(value["task_id"]), str(value["call_id"])) for value in values]
    phase_result = json.loads(PHASE_RESULT.read_text())
    if (
        task_order != validation_ids
        or len(values) != 1044
        or len(set(call_keys)) != len(call_keys)
        or phase_result.get("schema") != "full-test-phase-fresh-evaluation-v1"
        or phase_result.get("role") != "validation"
        or phase_result.get("development_run") != str(DEVELOPMENT_RUN.resolve())
    ):
        raise ValueError("phase rows differ from the exposed validation population")

    fit = _load_rows(FIT_ROWS)
    base = _dev100_static_rows(values)
    base_by_task: dict[str, list[Row]] = defaultdict(list)
    for row in base:
        base_by_task[row.task_id].append(row)
    semantic_by_sample: dict[str, dict[str, Any]] = {}
    for task_id in validation_ids:
        _result, task_rows = run_semantic_work_units(fit, base_by_task[task_id])
        semantic_by_sample.update((row["sample_id"], row) for row in task_rows)
    if set(semantic_by_sample) != {row.sample_id for row in base}:
        raise ValueError("static semantic rows differ from validation commands")

    phase_pmfs = json.loads(PHASE_ARTIFACT.read_text())["pmfs"]
    phase_hard = {
        target: max(range(3), key=phase_pmfs[target].__getitem__)
        for target in TARGETS
    }
    current: dict[str, tuple[float, float]] = {}
    sota: dict[str, tuple[float, float]] = {}
    phase_raised = Counter[str]()
    unavailable = Counter[str]()
    for value, base_row in zip(values, base, strict=True):
        semantic = semantic_by_sample[base_row.sample_id]["arms"]["semantic_work_units"]
        current_values = dict(base_row.current)
        sota_values = {
            target: semantic["candidate"].get(target) for target in TARGETS
        }
        for target in TARGETS:
            current_index = _index(current_values.get(target))
            semantic_index = _index(sota_values[target])
            if semantic_index is None:
                unavailable[target] += 1
                continue
            if (
                value["full_test_phase"] == 2
                and current_index is not None
                and phase_hard[target] > current_index
                and phase_hard[target] > semantic_index
            ):
                sota_values[target] = RESOURCE_BUCKET_LABELS[phase_hard[target]]
                phase_raised[target] += 1
        command_id = f"{base_row.task_id}:{value['call_id']}"
        current[command_id] = _request(
            current_values.get(TARGETS[0]), current_values.get(TARGETS[1])
        )
        sota[command_id] = _request(sota_values[TARGETS[0]], sota_values[TARGETS[1]])
    return current, sota, {
        "rows": len(values),
        "phase_raised": dict(sorted(phase_raised.items())),
        "semantic_unavailable": dict(sorted(unavailable.items())),
    }


def _validation_commands(
    validation_ids: list[str],
) -> tuple[dict[tuple[str, str], CommandRow], dict[tuple[str, str], int], dict[str, Path]]:
    rows: dict[tuple[str, str], CommandRow] = {}
    raw_counts: dict[tuple[str, str], int] = {}
    traces: dict[str, Path] = {}
    required_status = {
        "collection_validity": "valid",
        "workload_execution": "completed",
        "telemetry_quality": "ok",
        "cleanup": "ok",
    }
    for manifest_index, task_id in enumerate(validation_ids):
        attempt = VALIDATION_RUN / task_id / "attempt_1"
        artifact_path = attempt / "resource_observations.json"
        trace_path = attempt / "trace.jsonl"
        if not artifact_path.is_file() or not trace_path.is_file():
            raise FileNotFoundError(f"validation task artifact missing: {task_id}")
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if any(artifact.get(key) != expected for key, expected in required_status.items()):
            raise ValueError(f"{artifact_path}: telemetry is not evidence-valid")
        traces[task_id] = trace_path
        for call_index, call in enumerate(artifact.get("calls", [])):
            if not isinstance(call, Mapping) or call.get("eligible_for_kb") is not True:
                continue
            call_id = call.get("tool_call_id")
            command = call.get("command")
            clauses = call.get("clauses")
            if not isinstance(call_id, str) or not isinstance(command, str) or not isinstance(clauses, list):
                raise ValueError(f"{artifact_path}: eligible call lacks identity")
            key = (task_id, call_id)
            if key in rows:
                raise ValueError(f"{artifact_path}: duplicate eligible call")
            raw_counts[key] = len(clauses)
            retained = tuple(
                row
                for clause in clauses
                if isinstance(clause, Mapping)
                and (row := _row_from_clause(task_id, manifest_index, clause)) is not None
            )
            if retained:
                rows[key] = CommandRow(
                    task_id,
                    "tobymao/sqlglot",
                    manifest_index,
                    call_index,
                    call_id,
                    command,
                    0.0,
                    retained,
                )
    return rows, raw_counts, traces


def _full_reservation_map(
    programs: Mapping[str, Any],
    predictions: Mapping[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    return {
        command.command_id: predictions.get(
            command.command_id, (CPU_CAPACITY, RSS_CAPACITY_MB)
        )
        for program in programs.values()
        for command in program.commands
    }


def run() -> dict[str, Any]:
    split = json.loads(SPLIT.read_text())
    validation_ids = list(split["validation"])
    if len(validation_ids) != 50:
        raise ValueError("validation split differs from the frozen protocol")
    current_predictions, sota_predictions, prediction_coverage = _static_predictions(
        validation_ids
    )
    command_rows, raw_counts, traces = _validation_commands(validation_ids)
    source_counts: Counter[str] = Counter()
    programs = {
        task_id: _program(
            task_id,
            traces[task_id],
            command_rows,
            raw_counts,
            source_counts,
        )
        for task_id in validation_ids
    }
    current_requests = _full_reservation_map(programs, current_predictions)
    sota_requests = _full_reservation_map(programs, sota_predictions)
    changed_ids = {
        command_id
        for command_id in current_requests
        if current_requests[command_id] != sota_requests[command_id]
    }
    changed_tasks = {command_id.split(":", 1)[0] for command_id in changed_ids}

    schedule_results = []
    exposure_ids: dict[str, set[str]] = {"current": set(), "sota": set()}
    for seed in SEEDS:
        selected = sorted(programs)
        np.random.default_rng(seed).shuffle(selected)
        selected = selected[:LOAD]
        chosen = [programs[task_id] for task_id in selected]
        arms = {
            "fixed_high": simulate_admission(
                chosen,
                cpu_capacity=CPU_CAPACITY,
                rss_capacity_mb=RSS_CAPACITY_MB,
                fixed_high=True,
            ),
            "oracle": simulate_admission(
                chosen,
                cpu_capacity=CPU_CAPACITY,
                rss_capacity_mb=RSS_CAPACITY_MB,
                fixed_high=False,
            ),
            "current": simulate_admission(
                chosen,
                cpu_capacity=CPU_CAPACITY,
                rss_capacity_mb=RSS_CAPACITY_MB,
                fixed_high=False,
                requested_reservations=current_requests,
            ),
            "sota": simulate_admission(
                chosen,
                cpu_capacity=CPU_CAPACITY,
                rss_capacity_mb=RSS_CAPACITY_MB,
                fixed_high=False,
                requested_reservations=sota_requests,
            ),
        }
        for arm, metrics in arms.items():
            metrics.pop("overlapped_command_ids")
            ids = metrics.pop("modeled_capacity_exposure_command_ids")
            if arm in exposure_ids:
                exposure_ids[arm].update(str(value) for value in ids)
        identities = {
            (metrics["command_count"], metrics["total_command_service_s"])
            for metrics in arms.values()
        }
        if len(identities) != 1:
            raise ValueError("admission arms evaluated different commands or durations")
        schedule_results.append({"seed": seed, "task_ids": selected, "arms": arms})

    metrics = (
        "makespan_s",
        "mean_task_completion_s",
        "total_command_queue_s",
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
        for arm in ("fixed_high", "oracle", "current", "sota")
    }
    deltas = [
        float(row["arms"]["sota"]["makespan_s"])
        - float(row["arms"]["current"]["makespan_s"])
        for row in schedule_results
    ]
    current_makespan = means["current"]["makespan_s"]
    sota_makespan = means["sota"]["makespan_s"]
    fixed_makespan = means["fixed_high"]["makespan_s"]
    oracle_makespan = means["oracle"]["makespan_s"]
    sota_reduction = (current_makespan - sota_makespan) / current_makespan
    oracle_reduction = (fixed_makespan - oracle_makespan) / fixed_makespan
    oracle_headroom = fixed_makespan - oracle_makespan
    captured = (fixed_makespan - sota_makespan) / oracle_headroom
    comparison = {
        "candidate": "sota",
        "baseline": "current",
        "metric": "mean batch makespan_s; lower is better",
        "relative_reduction_of_means": sota_reduction,
        "oracle_headroom_capture": captured,
        **_bootstrap(deltas),
    }
    identical = all(
        len(
            {
                (metrics["command_count"], metrics["total_command_service_s"])
                for metrics in row["arms"].values()
            }
        )
        == 1
        for row in schedule_results
    )
    gate = {
        "validation_oracle_reduction_at_least_10_percent": oracle_reduction
        >= MINIMUM_MAKESPAN_REDUCTION,
        "sota_reduction_over_current_at_least_5_percent": sota_reduction
        >= MINIMUM_SOTA_REDUCTION,
        "paired_ci_below_zero": comparison["ci95_paired_seed_bootstrap"][1] < 0.0,
        "captures_half_oracle_headroom": captured >= MINIMUM_ORACLE_HEADROOM_CAPTURE,
        "at_least_20_changed_commands": len(changed_ids) >= MINIMUM_CHANGED_COMMANDS,
        "at_least_10_changed_tasks": len(changed_tasks) >= MINIMUM_CHANGED_TASKS,
        "zero_sota_conservative_exposures": not exposure_ids["sota"],
        "identical_commands_and_durations": identical,
    }
    gate["go"] = all(gate.values())
    return {
        "schema": VERSION,
        "status": (
            "development_go_to_real_admission_experiment"
            if gate["go"]
            else "development_no_go_hard_class_admission"
        ),
        "claim_bearing": False,
        "protocol": {
            "task_pool": "exposed SQLGlot validation50",
            "scheduler": "FCFS-ready with work-conserving backfill",
            "load": LOAD,
            "seeds": list(SEEDS),
            "cpu_capacity": CPU_CAPACITY,
            "rss_capacity_mb": RSS_CAPACITY_MB,
            "cpu_class_requests": list(CPU_REQUESTS),
            "rss_class_requests_mb": list(RSS_REQUESTS),
            "cross_validation_task_updates": False,
            "primary": "SOTA-minus-Current mean batch makespan_s",
        },
        "coverage": {
            "tasks": len(programs),
            "commands": sum(len(program.commands) for program in programs.values()),
            "telemetry_reservation_sources": dict(sorted(source_counts.items())),
            "prediction": prediction_coverage,
            "changed_request_commands": len(changed_ids),
            "changed_request_tasks": len(changed_tasks),
            "current_exposure_commands": len(exposure_ids["current"]),
            "sota_exposure_commands": len(exposure_ids["sota"]),
        },
        "mean_metrics": means,
        "validation_oracle_reduction": oracle_reduction,
        "primary_comparison": comparison,
        "gate": gate,
        "changed_request_ids": sorted(changed_ids),
        "exposure_command_ids": {
            arm: sorted(ids) for arm, ids in exposure_ids.items()
        },
        "schedule_results": schedule_results,
        "limitations": [
            "Validation tasks and predictor components are development-exposed.",
            "Conservative exposure is lack of safety evidence, not an observed overload.",
            "Recorded durations do not model throttling, OOM, or performance interference.",
            "The final SQLGlot partition remains unopened for resource outcomes.",
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
    result["inputs"] = {
        "split": str(SPLIT.resolve()),
        "validation_run": str(VALIDATION_RUN.resolve()),
        "phase_rows": str(PHASE_ROWS.resolve()),
        "phase_artifact": str(PHASE_ARTIFACT.resolve()),
        "git_sha": _git_sha(),
    }
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
