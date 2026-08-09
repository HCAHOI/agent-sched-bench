#!/usr/bin/env python3
"""Evaluate causal CPU feedback as a command-admission action."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
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
    _bootstrap,
)
from tool_resource_eval.early_cpu_reservation import (  # noqa: E402
    CPU_UPDATE_DELAY_S,
    SAMPLE_INTERVAL_S,
    THROUGHPUT_CPU_PAGES,
    cpu_work_profile,
)
from tool_resource_eval.resource_admission import (  # noqa: E402
    AdmissionProgram,
    simulate_feedback_admission,
)
from trace_collect.trace_data import TraceData  # noqa: E402


VERSION = "cpu-feedback-admission-v1"
ARMS = ("fixed8", "task_aware_static", "task_aware_feedback")
MINIMUM_MEAN_COMPLETION_REDUCTION = 0.05
MAXIMUM_SERVICE_INFLATION = 0.05
PROTOCOL = _ROOT / "analysis/development/cpu-feedback-admission-protocol.md"


def _assert_frozen_protocol(
    load: int,
    seeds: tuple[int, ...],
    cpu_capacity: float,
    rss_capacity_mb: float,
) -> None:
    if (
        load != 40
        or seeds != tuple(range(32))
        or cpu_capacity != 8.0
        or rss_capacity_mb != 16_000.0
        or tuple(THROUGHPUT_CPU_PAGES) != (2, 4, 8)
        or SAMPLE_INTERVAL_S != 0.5
        or not math.isclose(CPU_UPDATE_DELAY_S, 0.14132007875, abs_tol=1e-15)
    ):
        raise ValueError("frozen protocol constants changed")


def _gate(
    relative_mean_completion_reduction: float,
    paired_bootstrap_high: float,
    service_inflation: float,
    capacity_violation: bool,
) -> dict[str, bool]:
    gate = {
        "mean_task_completion_reduction_at_least_5_percent": (
            relative_mean_completion_reduction
            >= MINIMUM_MEAN_COMPLETION_REDUCTION
        ),
        "paired_order_bootstrap_upper_below_zero": paired_bootstrap_high < 0.0,
        "service_inflation_at_most_5_percent": (
            service_inflation <= MAXIMUM_SERVICE_INFLATION
        ),
        "zero_capacity_violations": not capacity_violation,
    }
    gate["go"] = all(gate.values())
    return gate


def _mean_relative_reduction(schedule_results: list[dict[str, Any]]) -> float:
    reductions = []
    for row in schedule_results:
        baseline = float(
            row["arms"]["task_aware_static"]["mean_task_completion_s"]
        )
        candidate = float(
            row["arms"]["task_aware_feedback"]["mean_task_completion_s"]
        )
        if baseline <= 0.0:
            raise ValueError("mean task completion must be positive")
        reductions.append((baseline - candidate) / baseline)
    if not reductions:
        raise ValueError("relative reduction requires schedule results")
    return float(np.mean(reductions))


def _file_identity(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest,
    }


def _file_identities(paths: tuple[Path, ...]) -> dict[str, dict[str, Any]]:
    return {str(path.resolve()): _file_identity(path) for path in paths}


def _profiles(
    programs: Mapping[str, AdmissionProgram], traces: Mapping[str, Path]
) -> tuple[
    dict[str, tuple[tuple[float, float], ...]],
    dict[str, dict[str, Any]],
]:
    command_ids = {
        command.command_id
        for program in programs.values()
        for command in program.commands
    }
    seen: set[str] = set()
    profiles: dict[str, tuple[tuple[float, float], ...]] = {}
    identities: dict[str, dict[str, Any]] = {}
    for task_id in sorted(programs):
        trace_path = traces[task_id]
        identities[task_id] = _file_identity(trace_path)
        for action in TraceData.load(trace_path).actions:
            data = action.get("data")
            if (
                action.get("action_type") != "tool_exec"
                or not isinstance(data, Mapping)
                or data.get("tool_name") != "exec"
            ):
                continue
            call_id = data.get("tool_call_id")
            if not isinstance(call_id, str):
                raise ValueError(f"{trace_path}: exec action lacks tool_call_id")
            command_id = f"{task_id}:{call_id}"
            if command_id not in command_ids or command_id in seen:
                raise ValueError(f"{trace_path}: exec command identity differs")
            seen.add(command_id)
            profile = cpu_work_profile(action)
            if profile is not None:
                profiles[command_id] = profile
    if seen != command_ids:
        raise ValueError("trace exec commands differ from admission programs")
    return profiles, identities


def _mean_metrics(schedule_results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "command_count",
        "recorded_command_service_s",
        "total_command_service_s",
        "added_service_s",
        "makespan_s",
        "mean_task_completion_s",
        "total_command_queue_s",
        "reserved_cpu_core_s",
        "reserved_rss_mb_s",
        "max_concurrent_commands",
        "feedback_updates",
        "reservation_shrinks",
        "reservation_expansions",
        "denied_expansions",
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


def run() -> dict[str, Any]:
    _assert_frozen_protocol(LOAD, SEEDS, CPU_CAPACITY, RSS_CAPACITY_MB)
    validation_ids = list(json.loads(SPLIT.read_text(encoding="utf-8"))["validation"])
    if len(validation_ids) != 50:
        raise ValueError("validation split differs from the frozen protocol")
    _clause_predictions, task_predictions, prediction_coverage = _static_predictions(
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
    profiles, source_files = _profiles(programs, traces)
    task_requests = _full_reservation_map(programs, task_predictions)
    fixed_requests = {
        command.command_id: (CPU_CAPACITY, RSS_CAPACITY_MB)
        for program in programs.values()
        for command in program.commands
    }
    all_command_ids = set(fixed_requests)
    if set(task_requests) != all_command_ids or not set(profiles) <= all_command_ids:
        raise ValueError("feedback inputs differ from admission commands")

    schedule_results: list[dict[str, Any]] = []
    capacity_violation = False
    for seed in SEEDS:
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
            command_id: profile
            for command_id, profile in profiles.items()
            if command_id in selected_ids
        }
        common = {
            "cpu_capacity": CPU_CAPACITY,
            "rss_capacity_mb": RSS_CAPACITY_MB,
            "cpu_work_profiles": selected_profiles,
            "sample_interval_s": SAMPLE_INTERVAL_S,
            "update_delay_s": CPU_UPDATE_DELAY_S,
            "cpu_pages": tuple(float(page) for page in THROUGHPUT_CPU_PAGES),
        }
        arms = {
            "fixed8": simulate_feedback_admission(
                chosen,
                **common,
                requested_reservations={
                    command_id: fixed_requests[command_id]
                    for command_id in selected_ids
                },
                feedback=False,
            ),
            "task_aware_static": simulate_feedback_admission(
                chosen,
                **common,
                requested_reservations={
                    command_id: task_requests[command_id]
                    for command_id in selected_ids
                },
                feedback=False,
            ),
            "task_aware_feedback": simulate_feedback_admission(
                chosen,
                **common,
                requested_reservations={
                    command_id: task_requests[command_id]
                    for command_id in selected_ids
                },
                feedback=True,
            ),
        }
        identity = {
            (
                arm["command_count"],
                arm["recorded_command_service_s"],
            )
            for arm in arms.values()
        }
        if len(identity) != 1:
            raise ValueError("admission arms used different commands or source service")
        for arm in arms.values():
            capacity_violation |= bool(arm["capacity_violation"])
            arm.pop("overlapped_command_ids")
            arm.pop("service_s_by_command")
            arm.pop("start_s_by_command")
        schedule_results.append({"seed": seed, "task_ids": selected, "arms": arms})

    means = _mean_metrics(schedule_results)
    relative_reduction = _mean_relative_reduction(schedule_results)
    deltas = [
        row["arms"]["task_aware_feedback"]["mean_task_completion_s"]
        - row["arms"]["task_aware_static"]["mean_task_completion_s"]
        for row in schedule_results
    ]
    comparison = {
        "candidate": "task_aware_feedback",
        "baseline": "task_aware_static",
        "metric": "mean task completion seconds; lower is better",
        "mean_per_order_relative_reduction": relative_reduction,
        **_bootstrap(deltas),
    }
    recorded = means["fixed8"]["recorded_command_service_s"]
    service_inflation = (
        means["task_aware_feedback"]["total_command_service_s"] / recorded - 1.0
    )
    bootstrap_high = comparison["ci95_paired_seed_bootstrap"][1]
    gate = _gate(
        relative_reduction,
        float(bootstrap_high),
        service_inflation,
        capacity_violation,
    )
    fixed_reconstruction = all(
        math.isclose(
            row["arms"]["fixed8"]["total_command_service_s"],
            row["arms"]["fixed8"]["recorded_command_service_s"],
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
        for row in schedule_results
    )
    integrity = {
        "fixed8_reconstructs_recorded_service": fixed_reconstruction,
        "all_arms_use_identical_commands_and_source_service": all(
            len(
                {
                    (
                        arm["command_count"],
                        arm["recorded_command_service_s"],
                    )
                    for arm in row["arms"].values()
                }
            )
            == 1
            for row in schedule_results
        ),
        "all_profiles_belong_to_commands": set(profiles) <= all_command_ids,
        "zero_capacity_violations": not capacity_violation,
    }
    if not all(integrity.values()):
        raise ValueError(f"feedback admission integrity failure: {integrity}")

    initial_pages = Counter(int(request[0]) for request in task_requests.values())
    profile_service = sum(
        next(
            command.duration_s
            for program in programs.values()
            for command in program.commands
            if command.command_id == command_id
        )
        for command_id in profiles
    )
    total_service = sum(
        command.duration_s
        for program in programs.values()
        for command in program.commands
    )
    return {
        "schema": VERSION,
        "status": (
            "development_go_to_physical_feedback_admission"
            if gate["go"]
            else "development_stop_before_physical_feedback_admission"
        ),
        "claim_bearing": False,
        "protocol": {
            "task_pool": "development-exposed SQLGlot validation50",
            "load": LOAD,
            "seeds": list(SEEDS),
            "cpu_capacity": CPU_CAPACITY,
            "rss_capacity_mb": RSS_CAPACITY_MB,
            "cpu_pages": list(THROUGHPUT_CPU_PAGES),
            "sample_interval_s": SAMPLE_INTERVAL_S,
            "update_delay_s": CPU_UPDATE_DELAY_S,
            "primary": "Task-Aware feedback minus static mean task completion",
            "minimum_relative_reduction": MINIMUM_MEAN_COMPLETION_REDUCTION,
            "maximum_service_inflation": MAXIMUM_SERVICE_INFLATION,
        },
        "coverage": {
            "tasks": len(programs),
            "commands": len(all_command_ids),
            "commands_with_valid_cpu_profile": len(profiles),
            "profile_command_fraction": len(profiles) / len(all_command_ids),
            "profile_service_fraction": profile_service / total_service,
            "initial_cpu_pages": {
                str(page): initial_pages[page] for page in THROUGHPUT_CPU_PAGES
            },
            "prediction": prediction_coverage,
            "telemetry_reservation_sources": dict(sorted(source_counts.items())),
        },
        "source_files": source_files,
        "prediction_input_files": _file_identities(
            (FIT_ROWS, PHASE_ROWS, PHASE_RESULT, PHASE_ARTIFACT)
        ),
        "mean_metrics": means,
        "primary_comparison": comparison,
        "candidate_service_inflation": service_inflation,
        "orders_improved": sum(delta < 0.0 for delta in deltas),
        "gate": gate,
        "integrity": integrity,
        "schedule_results": schedule_results,
        "context": {
            "equal_share_physical_control": (
                "analysis/results/tool-resource-5-3-3-3-20260804/"
                "sqlglot-final48-counterbalanced-rolling-exact-v1/result.json"
            )
        },
        "limitations": [
            "All tasks and predictor outputs are development-exposed.",
            "CPU work is uniform within each source telemetry interval.",
            "Denied expansions retain the current hard page until the next observation.",
            "The paired bootstrap measures workload-order sensitivity, not new-task uncertainty.",
            "This is an event replay, not a physical runtime result.",
        ],
    }


def _require_committed_inputs() -> None:
    paths = (
        Path(__file__).resolve(),
        PROTOCOL.resolve(),
        SPLIT.resolve(),
        FIT_ROWS.resolve(),
        PHASE_ROWS.resolve(),
        PHASE_RESULT.resolve(),
        PHASE_ARTIFACT.resolve(),
        (_ROOT / "src/tool_resource_eval/early_cpu_reservation.py").resolve(),
        (_ROOT / "src/tool_resource_eval/resource_admission.py").resolve(),
        (_ROOT / "src/tool_resource_eval/cachewise_kv_factorial.py").resolve(),
        (_ROOT / "scripts/evaluation/evaluate_kv_prediction_actionability.py").resolve(),
        (_ROOT / "scripts/evaluation/evaluate_resource_admission_oracle.py").resolve(),
        (_ROOT / "scripts/evaluation/evaluate_resource_admission_predictors.py").resolve(),
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
