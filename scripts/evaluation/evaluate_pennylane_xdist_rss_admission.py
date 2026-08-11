#!/usr/bin/env python3
"""Evaluate scope-conditioned pytest-xdist RSS as an admission action."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from scripts.evaluation.evaluate_pennylane_temporal_rss_ceiling import (  # noqa: E402
    _EXPECTED_TASK_IDS,
)
from scripts.evaluation.evaluate_zarr_rss_backfill import _load_dataset  # noqa: E402
from tool_resource_eval.resource_admission import simulate_idle_backfill  # noqa: E402


_PROTOCOL_GIT_SHA = "8cfbb463aecd6efa36d43feb6476225d866e5a31"
_SPLIT = _ROOT / "analysis/development/pennylane-survival-action-split.json"
_PREDICTIONS = _ROOT / "analysis/results/pennylane-xdist-rss-positive-control-v2"
_OUTPUT = _ROOT / "analysis/results/pennylane-xdist-rss-admission-v1"
_CPU_CAPACITY = 8.0
_RSS_CAPACITY_MB = 16_000.0
_RSS_REQUESTS = (500.0, 2_000.0, _RSS_CAPACITY_MB)
_PREDICTION_ARMS = ("clause_kb", "count_unconditioned", "scope_conditioned")
_UPSTREAM_IDENTITY = (
    "pennylane-xdist-rss-positive-control-v2",
    "development_go",
    "29744ed1fbbb461c39344e1b0d5636ba13aab3aa",
    "3eda2581a75e1d16655b53406520cae8d04d1bd9",
)


def _require_clean_checkout() -> str:
    if subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout:
        raise ValueError("formal evaluation requires a clean committed checkout")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", _PROTOCOL_GIT_SHA, "HEAD"],
        cwd=_ROOT,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _frozen_bytes(path: Path) -> bytes:
    relative = path.relative_to(_ROOT).as_posix()
    frozen = subprocess.run(
        ["git", "show", f"{_PROTOCOL_GIT_SHA}:{relative}"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if path.read_bytes() != frozen:
        raise ValueError(f"frozen input changed: {relative}")
    return frozen


def _hard_bucket(pmf: Sequence[float] | None) -> int | None:
    if pmf is None:
        return None
    if (
        len(pmf) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0.0
            for value in pmf
        )
        or not math.isclose(sum(pmf), 1.0, rel_tol=1e-9, abs_tol=1e-9)
    ):
        raise ValueError("invalid frozen RSS PMF")
    return max(range(3), key=pmf.__getitem__)


def prediction_reservations(
    command_ids: set[str], rows: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, float]]:
    reservations = {
        arm: {command_id: _RSS_CAPACITY_MB for command_id in command_ids}
        for arm in _PREDICTION_ARMS
    }
    seen: set[str] = set()
    for row in rows:
        command_id = f"{row['task_id']}:{row['call_id']}"
        if command_id in seen or command_id not in command_ids:
            raise ValueError(f"prediction row identity mismatch: {command_id}")
        seen.add(command_id)
        for arm in _PREDICTION_ARMS:
            bucket = _hard_bucket(row.get(arm))
            if bucket is not None:
                reservations[arm][command_id] = _RSS_REQUESTS[bucket]
    return reservations


def decision_changes(
    baseline_ids: Sequence[str], candidate_ids: Sequence[str]
) -> dict[str, Any]:
    changed = sorted(set(baseline_ids) ^ set(candidate_ids))
    return {
        "commands": len(changed),
        "tasks": len({command_id.split(":", 1)[0] for command_id in changed}),
        "command_ids": changed,
    }


def _comparison(serial: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float]:
    return {
        "mean_task_completion_reduction": 1.0
        - candidate["mean_task_completion_s"] / serial["mean_task_completion_s"],
        "makespan_reduction": 1.0 - candidate["makespan_s"] / serial["makespan_s"],
        "service_inflation": candidate["total_command_service_s"]
        / candidate["recorded_command_service_s"]
        - 1.0,
    }


def _gate(
    arms: Mapping[str, Mapping[str, Any]], changes: Mapping[str, Any]
) -> dict[str, bool]:
    serial = arms["serial8"]
    candidate = arms["scope_conditioned"]
    comparison = _comparison(serial, candidate)
    work_conserved = math.isclose(
        candidate["total_cpu_work_core_s"],
        candidate["served_cpu_work_core_s"],
        rel_tol=1e-12,
        abs_tol=1e-7,
    )
    checks = {
        "zero_static_rss_exposure": candidate["modeled_capacity_exposure_events"] == 0,
        "zero_reservation_capacity_violation": not candidate["capacity_violation"],
        "zero_physical_cpu_capacity_violation": not candidate["physical_capacity_violation"],
        "cpu_work_conserved": work_conserved,
        "mean_completion_reduction_at_least_5_percent": comparison[
            "mean_task_completion_reduction"
        ]
        >= 0.05,
        "makespan_lower_than_serial": comparison["makespan_reduction"] > 0.0,
        "service_inflation_at_most_5_percent": comparison["service_inflation"] <= 0.05,
        "exposure_strictly_below_clause_kb": candidate[
            "modeled_capacity_exposure_events"
        ]
        < arms["clause_kb"]["modeled_capacity_exposure_events"],
        "exposure_strictly_below_count_unconditioned": candidate[
            "modeled_capacity_exposure_events"
        ]
        < arms["count_unconditioned"]["modeled_capacity_exposure_events"],
        "at_least_20_speculative_starts": candidate["speculative_starts"] >= 20,
        "at_least_5_speculative_tasks": candidate["speculative_task_count"] >= 5,
        "at_least_10_changed_start_decisions": changes["commands"] >= 10,
        "changed_start_decisions_span_3_tasks": changes["tasks"] >= 3,
    }
    return checks | {"go": all(checks.values())}


def _compact(metrics: Mapping[str, Any]) -> dict[str, Any]:
    starts = list(metrics["speculative_start_ids"])
    keys = (
        "command_count",
        "recorded_command_service_s",
        "total_command_service_s",
        "makespan_s",
        "mean_task_completion_s",
        "total_command_queue_s",
        "max_concurrent_commands",
        "speculative_starts",
        "modeled_capacity_exposure_events",
        "max_modeled_rss_demand_mb",
        "capacity_violation",
        "physical_capacity_violation",
        "total_cpu_work_core_s",
        "served_cpu_work_core_s",
    )
    return {key: metrics[key] for key in keys} | {
        "speculative_task_count": len(
            {command_id.split(":", 1)[0] for command_id in starts}
        ),
        "speculative_start_ids": starts,
        "modeled_capacity_exposure_command_ids": list(
            metrics["modeled_capacity_exposure_command_ids"]
        ),
    }


def evaluate(evaluation_git_sha: str) -> dict[str, Any]:
    started = time.monotonic()
    upstream = json.loads(_frozen_bytes(_PREDICTIONS / "result.json"))
    if (
        (
            upstream.get("schema"),
            upstream.get("status"),
            upstream.get("protocol_git_sha"),
            upstream.get("evaluation_git_sha"),
        )
        != _UPSTREAM_IDENTITY
        or upstream.get("evidence", {}).get("replay_tasks")
        != list(_EXPECTED_TASK_IDS)
        or upstream.get("evidence", {}).get("carrier_commands") != 28
    ):
        raise ValueError("upstream prediction result differs from the freeze")
    split = json.loads(_frozen_bytes(_SPLIT))
    replay = {str(item["task_id"]): item for item in split["replay"]}
    traces = {
        task_id: _ROOT / str(replay[task_id]["trace"])
        for task_id in _EXPECTED_TASK_IDS
    }
    dataset = _load_dataset(_EXPECTED_TASK_IDS, traces)
    rows = [
        json.loads(line)
        for line in _frozen_bytes(_PREDICTIONS / "rows.jsonl").splitlines()
    ]
    command_ids = set(dataset.profiles)
    expected_rows = {
        f"{task_id}:{row.call_id}": row.command
        for task_id in _EXPECTED_TASK_IDS
        for row in dataset.commands_by_task[task_id]
    }
    actual_rows = {f"{row['task_id']}:{row['call_id']}": row["command"] for row in rows}
    if len(command_ids) != 570 or len(rows) != 528 or actual_rows != expected_rows:
        raise ValueError("prediction rows differ from the frozen command population")
    reservations = prediction_reservations(command_ids, rows)
    programs = [dataset.programs[task_id] for task_id in _EXPECTED_TASK_IDS]
    common = {
        "cpu_capacity": _CPU_CAPACITY,
        "rss_capacity_mb": _RSS_CAPACITY_MB,
        "cpu_work_profiles": dataset.profiles,
        "speculative_eligible_command_ids": command_ids,
        "selection": "fcfs",
    }
    raw_arms = {
        "serial8": simulate_idle_backfill(
            programs,
            cpu_capacity=_CPU_CAPACITY,
            rss_capacity_mb=_RSS_CAPACITY_MB,
            cpu_work_profiles=dataset.profiles,
            speculative_eligible_command_ids=set(),
            selection="serial",
        ),
        **{
            arm: simulate_idle_backfill(
                programs, rss_reservations=reservations[arm], **common
            )
            for arm in _PREDICTION_ARMS
        },
        "exact_rss_reference": simulate_idle_backfill(
            programs,
            rss_reservations={
                command.command_id: command.rss_mb
                for program in programs
                for command in program.commands
            },
            **common,
        ),
    }
    arms = {name: _compact(metrics) for name, metrics in raw_arms.items()}
    changes = decision_changes(
        arms["clause_kb"]["speculative_start_ids"],
        arms["scope_conditioned"]["speculative_start_ids"],
    )
    gate = _gate(arms, changes)
    serial = arms["serial8"]
    return {
        "schema": "pennylane-xdist-rss-admission-v1",
        "status": "development_go" if gate["go"] else "development_no_go",
        "claim_bearing": False,
        "protocol_git_sha": _PROTOCOL_GIT_SHA,
        "evaluation_git_sha": evaluation_git_sha,
        "protocol": {
            "task_order": list(_EXPECTED_TASK_IDS),
            "arrival": "one concurrent wave; frozen task order breaks ties",
            "cpu_capacity": _CPU_CAPACITY,
            "rss_capacity_mb": _RSS_CAPACITY_MB,
            "rss_requests_mb": list(_RSS_REQUESTS),
            "selection": "strict-priority FCFS with at most one backfill command",
            "unknown_prediction": "16000 MB",
            "physical_rss": "canonical static command peak composition",
        },
        "evidence": {
            "tasks": len(programs),
            "commands": len(command_ids),
            "prediction_rows": len(rows),
            "unverified_physical_rss_commands": len(dataset.unverified_command_ids),
            "untouched_pennylane_tasks_read": 0,
        },
        "arms": arms,
        "comparisons_vs_serial8": {
            name: _comparison(serial, metrics)
            for name, metrics in arms.items()
            if name != "serial8"
        },
        "scope_conditioned_vs_clause_kb_decisions": changes,
        "gate": gate,
        "cost": {
            "prediction_time_agent_calls": 0,
            "gpu_runtime_s": 0.0,
            "evaluation_wall_s": time.monotonic() - started,
        },
        "inputs": {
            "prediction_result": str((_PREDICTIONS / "result.json").resolve()),
            "prediction_rows": str((_PREDICTIONS / "rows.jsonl").resolve()),
            "split": str(_SPLIT.resolve()),
            "traces": {task_id: str(path.resolve()) for task_id, path in traces.items()},
        },
        "limitations": [
            "Development-exposed tasks and predictions; not confirmatory evidence.",
            "Static command peak sums are conservative and do not model peak timing.",
            "Recorded isolated CPU profiles drive a simulator, not concurrent containers.",
            "The exact-RSS arm is a reference, not a performance ceiling.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_OUTPUT / "result.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = evaluate(_require_clean_checkout())
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
