#!/usr/bin/env python3
"""Evaluate the frozen zero-cost PennyLane container-parking ceiling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from scripts.evaluation.evaluate_pennylane_joint_phase_packing import (  # noqa: E402
    Capacities,
    TaskProfile,
    _BIN_S,
    _load_profiles,
    simulate,
)

_PROTOCOL = (
    _ROOT / "analysis/development/pennylane-perfect-container-parking-protocol.md"
)
_PROTOCOL_GIT_SHA = "0daae2928c6e60b402393f76cfa89e2c27549d40"
_OUTPUT = (
    _ROOT
    / "analysis/results/pennylane-perfect-container-parking-development-v1/result.json"
)


def _require_clean_checkout() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("formal evaluation requires a clean committed checkout")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", _PROTOCOL_GIT_SHA, "HEAD"],
        cwd=_ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("frozen perfect-parking protocol is not an ancestor of HEAD")
    relative = _PROTOCOL.relative_to(_ROOT).as_posix()
    committed = subprocess.run(
        ["git", "show", f"{_PROTOCOL_GIT_SHA}:{relative}"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if _PROTOCOL.read_bytes() != committed:
        raise ValueError("perfect-parking protocol differs from its frozen commit")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _park(profile: TaskProfile) -> TaskProfile:
    llm = profile.gpu > 0.0
    return TaskProfile(
        task_id=profile.task_id,
        gpu=profile.gpu.copy(),
        cpu=np.where(llm, 0.0, profile.cpu),
        rss=np.where(llm, 0.0, profile.rss),
    )


def _gate(
    always: dict[str, Any], perfect: dict[str, Any], earlier_start_count: int
) -> dict[str, Any]:
    mean_reduction = (
        1.0 - perfect["mean_task_completion_s"] / always["mean_task_completion_s"]
    )
    checks = {
        "perfect_parking_zero_capacity_violations": perfect["feasible"],
        "at_least_one_task_starts_earlier": earlier_start_count > 0,
        "mean_task_completion_improves_at_least_5pct": mean_reduction >= 0.05,
    }
    return {
        "status": "go" if all(checks.values()) else "no_go",
        "mean_task_completion_reduction": mean_reduction,
        "checks": checks,
    }


def _evaluate_profiles(
    profiles: list[TaskProfile], capacities: Capacities
) -> dict[str, Any]:
    parked_profiles = [_park(profile) for profile in profiles]
    always = simulate(profiles, "joint", active_cap=None, capacities=capacities)
    perfect = simulate(parked_profiles, "joint", active_cap=None, capacities=capacities)
    task_deltas = []
    releases = []
    for profile, parked in zip(profiles, parked_profiles, strict=True):
        start_advance = (
            always["start_s_by_task"][profile.task_id]
            - perfect["start_s_by_task"][profile.task_id]
        )
        completion_advance = (
            always["completion_s_by_task"][profile.task_id]
            - perfect["completion_s_by_task"][profile.task_id]
        )
        task_deltas.append(
            {
                "task_id": profile.task_id,
                "start_advance_s": start_advance,
                "completion_advance_s": completion_advance,
            }
        )
        releases.append(
            {
                "task_id": profile.task_id,
                "llm_bins": int(np.count_nonzero(profile.gpu)),
                "cpu_core_s": float((profile.cpu - parked.cpu).sum()) * _BIN_S,
                "rss_mib_s": float((profile.rss - parked.rss).sum()) * _BIN_S,
            }
        )
    earlier = [row for row in task_deltas if row["start_advance_s"] > 0.0]
    gate = _gate(always, perfect, len(earlier))
    return {
        "status": gate["status"],
        "arms": {"always_resident": always, "perfect_parking": perfect},
        "comparison": {
            "mean_task_completion_reduction": gate["mean_task_completion_reduction"],
            "makespan_reduction": 1.0 - perfect["makespan_s"] / always["makespan_s"],
            "earlier_start_task_count": len(earlier),
            "total_start_advance_s": sum(row["start_advance_s"] for row in earlier),
            "total_completion_advance_s": sum(
                max(0.0, row["completion_advance_s"]) for row in task_deltas
            ),
            "removed_cpu_core_s": sum(row["cpu_core_s"] for row in releases),
            "removed_rss_mib_s": sum(row["rss_mib_s"] for row in releases),
        },
        "gate": {"status": gate["status"], "checks": gate["checks"]},
        "task_deltas": task_deltas,
        "released_resource_time_by_task": releases,
    }


def evaluate(git_sha: str) -> dict[str, Any]:
    profiles, evidence = _load_profiles()
    result = _evaluate_profiles(profiles, Capacities())
    return {
        "schema": "pennylane-perfect-container-parking-development-v1",
        "status": result["status"],
        "corpus_role": "development_exposed",
        "git_sha": git_sha,
        "protocol_git_sha": _PROTOCOL_GIT_SHA,
        "protocol": _PROTOCOL.relative_to(_ROOT).as_posix(),
        "evidence": evidence,
        **result,
        "interpretation_boundary": (
            "Zero-cost hindsight ceiling on fixed two-second profiles. It does "
            "not measure checkpoint, restore, transfer, or physical scheduling cost."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=_OUTPUT)
    args = parser.parse_args()
    result = evaluate(_require_clean_checkout())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "gate": result["gate"]}, indent=2))


if __name__ == "__main__":
    main()
