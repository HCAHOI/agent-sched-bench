#!/usr/bin/env python3
"""Evaluate fit-only continuous RSS envelopes for pytest-xdist admission."""

from __future__ import annotations

import argparse
from collections import defaultdict
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
from scripts.evaluation.evaluate_pennylane_xdist_rss_admission import (  # noqa: E402
    _RSS_CAPACITY_MB,
    _compact,
    _hard_bucket,
    decision_changes,
    prediction_reservations,
)
from scripts.evaluation.evaluate_pennylane_xdist_rss_positive_control import (  # noqa: E402
    parse_pytest_workers,
    pytest_scope,
)
from scripts.evaluation.evaluate_zarr_rss_backfill import _load_dataset  # noqa: E402
from tool_resource_eval.resource_admission import simulate_idle_backfill  # noqa: E402


_PROTOCOL_GIT_SHA = "330d928d4947f5eeca359fc31b58946ff0babb60"
_SPLIT = _ROOT / "analysis/development/pennylane-survival-action-split.json"
_PREDICTIONS = _ROOT / "analysis/results/pennylane-xdist-rss-positive-control-v2"
_BASELINE = _ROOT / "analysis/results/pennylane-xdist-rss-admission-v1/result.json"
_OUTPUT = _ROOT / "analysis/results/pennylane-xdist-rss-fit-envelope-admission-v1"
_CPU_CAPACITY = 8.0


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


def continuous_reservations(
    command_ids: set[str],
    rows: Sequence[Mapping[str, Any]],
    finite_envelopes: Mapping[tuple[int, str], float],
) -> tuple[dict[str, float], dict[str, Any]]:
    reservations = prediction_reservations(command_ids, rows)["scope_conditioned"]
    changed: list[str] = []
    for row in rows:
        if not row.get("carrier") or _hard_bucket(row.get("scope_conditioned")) != 2:
            continue
        key = row.get("workers"), row.get("scope")
        if key not in finite_envelopes:
            continue
        command_id = f"{row['task_id']}:{row['call_id']}"
        reservations[command_id] = finite_envelopes[key]  # type: ignore[index]
        changed.append(command_id)
    return reservations, {
        "commands": len(changed),
        "tasks": len({command_id.split(":", 1)[0] for command_id in changed}),
        "command_ids": sorted(changed),
    }


def _comparison(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float]:
    return {
        "mean_task_completion_reduction": 1.0
        - candidate["mean_task_completion_s"] / baseline["mean_task_completion_s"],
        "makespan_reduction": 1.0 - candidate["makespan_s"] / baseline["makespan_s"],
        "service_inflation": candidate["total_command_service_s"]
        / candidate["recorded_command_service_s"]
        - 1.0,
    }


def _gate(
    serial: Mapping[str, Any],
    scope: Mapping[str, Any],
    candidate: Mapping[str, Any],
    changes: Mapping[str, Any],
) -> dict[str, bool]:
    vs_serial = _comparison(serial, candidate)
    checks = {
        "zero_modeled_static_rss_exposure": candidate[
            "modeled_capacity_exposure_events"
        ]
        == 0,
        "zero_reservation_capacity_violation": not candidate["capacity_violation"],
        "zero_physical_cpu_capacity_violation": not candidate[
            "physical_capacity_violation"
        ],
        "cpu_work_conserved": math.isclose(
            candidate["total_cpu_work_core_s"],
            candidate["served_cpu_work_core_s"],
            rel_tol=1e-12,
            abs_tol=1e-7,
        ),
        "mean_completion_reduction_at_least_5_percent": vs_serial[
            "mean_task_completion_reduction"
        ]
        >= 0.05,
        "makespan_lower_than_serial": vs_serial["makespan_reduction"] > 0.0,
        "service_inflation_at_most_5_percent": vs_serial["service_inflation"]
        <= 0.05,
        "mean_completion_at_least_1_percent_below_scope": 1.0
        - candidate["mean_task_completion_s"] / scope["mean_task_completion_s"]
        >= 0.01,
        "at_least_10_changed_start_decisions": changes["commands"] >= 10,
        "changed_start_decisions_span_3_tasks": changes["tasks"] >= 3,
    }
    return checks | {"go": all(checks.values())}


def evaluate(evaluation_git_sha: str) -> dict[str, Any]:
    started = time.monotonic()
    split = json.loads(_frozen_bytes(_SPLIT))
    prediction_result = json.loads(_frozen_bytes(_PREDICTIONS / "result.json"))
    baseline = json.loads(_frozen_bytes(_BASELINE))
    rows = [
        json.loads(line)
        for line in _frozen_bytes(_PREDICTIONS / "rows.jsonl").splitlines()
    ]
    if (
        prediction_result.get("status") != "development_go"
        or baseline.get("schema") != "pennylane-xdist-rss-admission-v1"
        or baseline.get("evaluation_git_sha")
        != "575fd15d052f9c69a8920b8150e1584663c5d6f9"
        or baseline.get("protocol", {}).get("task_order") != list(_EXPECTED_TASK_IDS)
    ):
        raise ValueError("committed predecessor differs from the freeze")

    fit_items = split["fit"]
    fit_ids = tuple(str(item["task_id"]) for item in fit_items)
    fit = _load_dataset(
        fit_ids,
        {str(item["task_id"]): _ROOT / str(item["trace"]) for item in fit_items},
    )
    calibration: dict[str, list[Any]] = defaultdict(list)
    for task_id in fit_ids:
        for row in fit.clauses_by_task[task_id]:
            scope = pytest_scope(row.argv)
            if (
                scope is not None
                and parse_pytest_workers(row.argv) == "serial"
                and row.sampled_peak_rss_mb is not None
            ):
                calibration[scope].append(row)

    cells = {
        (int(row["workers"]), str(row["scope"]))
        for row in rows
        if row.get("carrier") and isinstance(row.get("workers"), int)
    }
    cell_evidence: dict[str, Any] = {}
    finite_envelopes: dict[tuple[int, str], float] = {}
    for workers, scope in sorted(cells):
        values = calibration[scope]
        bound = (workers + 1) * max(float(row.sampled_peak_rss_mb) for row in values)
        tasks = {row.task_id for row in values}
        finite = len(values) >= 10 and len(tasks) >= 3 and bound < _RSS_CAPACITY_MB
        cell_evidence[f"{workers}/{scope}"] = {
            "fit_clauses": len(values),
            "fit_tasks": len(tasks),
            "scaled_max_rss_mb": bound,
            "finite": finite,
        }
        if finite:
            finite_envelopes[workers, scope] = bound

    replay = {str(item["task_id"]): item for item in split["replay"]}
    target = _load_dataset(
        _EXPECTED_TASK_IDS,
        {
            task_id: _ROOT / str(replay[task_id]["trace"])
            for task_id in _EXPECTED_TASK_IDS
        },
    )
    command_ids = set(target.profiles)
    expected_rows = {
        f"{task_id}:{row.call_id}": row.command
        for task_id in _EXPECTED_TASK_IDS
        for row in target.commands_by_task[task_id]
    }
    actual_rows = {f"{row['task_id']}:{row['call_id']}": row["command"] for row in rows}
    if len(command_ids) != 570 or len(rows) != 528 or actual_rows != expected_rows:
        raise ValueError("prediction rows differ from the frozen command population")
    reservations, activated = continuous_reservations(
        command_ids, rows, finite_envelopes
    )
    if activated["commands"] < 10 or activated["tasks"] < 3:
        raise ValueError("fit-envelope activation misses the frozen pre-outcome gate")

    programs = [target.programs[task_id] for task_id in _EXPECTED_TASK_IDS]
    candidate = _compact(
        simulate_idle_backfill(
            programs,
            cpu_capacity=_CPU_CAPACITY,
            rss_capacity_mb=_RSS_CAPACITY_MB,
            cpu_work_profiles=target.profiles,
            speculative_eligible_command_ids=command_ids,
            rss_reservations=reservations,
            selection="fcfs",
        )
    )
    predecessor_arms = baseline["arms"]
    changes = decision_changes(
        predecessor_arms["scope_conditioned"]["speculative_start_ids"],
        candidate["speculative_start_ids"],
    )
    gate = _gate(
        predecessor_arms["serial8"],
        predecessor_arms["scope_conditioned"],
        candidate,
        changes,
    )
    return {
        "schema": "pennylane-xdist-rss-fit-envelope-admission-v1",
        "status": "development_go" if gate["go"] else "development_no_go",
        "claim_bearing": False,
        "protocol_git_sha": _PROTOCOL_GIT_SHA,
        "evaluation_git_sha": evaluation_git_sha,
        "evidence": {
            "fit_tasks": len(fit_ids),
            "replay_tasks": len(_EXPECTED_TASK_IDS),
            "commands": len(command_ids),
            "cells": cell_evidence,
            "activated_reservations": activated,
            "untouched_pennylane_tasks_read": 0,
        },
        "arm": candidate,
        "committed_predecessor_arms": {
            name: predecessor_arms[name]
            for name in (
                "serial8",
                "clause_kb",
                "scope_conditioned",
                "exact_rss_reference",
            )
        },
        "comparisons": {
            "vs_serial8": _comparison(predecessor_arms["serial8"], candidate),
            "vs_scope_conditioned": _comparison(
                predecessor_arms["scope_conditioned"], candidate
            ),
        },
        "start_decisions_vs_scope_conditioned": changes,
        "gate": gate,
        "cost": {
            "prediction_time_agent_calls": 0,
            "gpu_runtime_s": 0.0,
            "evaluation_wall_s": time.monotonic() - started,
        },
        "limitations": [
            "Development-exposed replay tasks; the gate was frozen after predecessor outcomes.",
            "A fit maximum is an empirical envelope, not a distribution-free safety bound.",
            "Modeled RSS uses conservative static command peaks, not simultaneous samples.",
            "Only 4-worker broad High predictions activate finite reservations.",
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
