#!/usr/bin/env python3
"""Evaluate the frozen PennyLane joint task-admission ceiling."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Literal

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from harness.container_stats_sampler import (  # noqa: E402
    _parse_memory_mb,
    _parse_percent,
)


_RUN_DIR = (
    _ROOT
    / "traces/swe-rebench/gpt-5.6-sol/"
    "pennylane-all76-clean-ebpf-20260816"
)
_PROTOCOL = (
    _ROOT / "analysis/development/pennylane-joint-phase-packing-protocol.md"
)
_OUTPUT = _ROOT / "analysis/results/pennylane-joint-phase-packing-v1/result.json"
_BIN_S = 2.0
_EXPECTED_TASKS = 70
_INPUT_TREE_SHA256 = "53d8faa0a6bd98dcfb38bfdf0916c196ff8d54ae6a305828a7b6f8509f3f7f16"
_EXCLUDED = {
    "PennyLaneAI__pennylane-5538",
    "PennyLaneAI__pennylane-5582",
    "PennyLaneAI__pennylane-5761",
    "PennyLaneAI__pennylane-5846",
    "PennyLaneAI__pennylane-5851",
    "PennyLaneAI__pennylane-5866",
}
Gate = Literal["none", "gpu", "tool", "joint"]


@dataclass(frozen=True)
class Capacities:
    gpu_slots: float = 4.0
    cpu_cores: float = 28.0
    rss_mb: float = 80_000.0


@dataclass(frozen=True)
class TaskProfile:
    task_id: str
    gpu: np.ndarray
    cpu: np.ndarray
    rss: np.ndarray

    @property
    def bins(self) -> int:
        return len(self.gpu)


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
    relative = _PROTOCOL.relative_to(_ROOT).as_posix()
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if _PROTOCOL.read_bytes() != committed:
        raise ValueError("joint phase-packing protocol differs from HEAD")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _trace_times(path: Path) -> tuple[dict[str, Any], list[tuple[float, float]], float, float]:
    metadata: dict[str, Any] | None = None
    llm: list[tuple[float, float]] = []
    first = math.inf
    last = -math.inf
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("type") == "trace_metadata":
                metadata = row
            if row.get("type") != "action":
                continue
            start = float(row["ts_start"])
            end = float(row["ts_end"])
            if not math.isfinite(start) or not math.isfinite(end) or end < start:
                raise ValueError(f"{path}: invalid action interval")
            first = min(first, start)
            last = max(last, end)
            if row.get("action_type") == "llm_call":
                if end == start:
                    raise ValueError(f"{path}: zero-duration LLM call")
                llm.append((start, end))
    if metadata is None or not llm or not math.isfinite(first) or not math.isfinite(last):
        raise ValueError(f"{path}: missing metadata, actions, or LLM calls")
    return metadata, llm, first, last


def _held_samples(
    samples: list[dict[str, Any]], origin: float, bins: int
) -> tuple[np.ndarray, np.ndarray]:
    cpu_updates: dict[int, list[float]] = {}
    rss_updates: dict[int, list[float]] = {}
    for row in samples:
        epoch = float(row["epoch"])
        cpu = _parse_percent(str(row.get("cpu_percent", "")))
        rss = _parse_memory_mb(str(row.get("mem_usage", "")))
        if (
            not math.isfinite(epoch)
            or cpu is None
            or rss is None
            or not math.isfinite(cpu)
            or not math.isfinite(rss)
            or cpu < 0.0
            or rss < 0.0
        ):
            raise ValueError("invalid task-resource sample")
        index = min(max(0, int((epoch - origin) // _BIN_S)), bins - 1)
        cpu_updates.setdefault(index, []).append(cpu / 100.0)
        rss_updates.setdefault(index, []).append(rss)
    cpu_profile = np.zeros(bins, dtype=np.float64)
    rss_profile = np.zeros(bins, dtype=np.float64)
    cpu = rss = 0.0
    for index in range(bins):
        cpu_values = cpu_updates.get(index, ())
        rss_values = rss_updates.get(index, ())
        cpu_profile[index] = max((cpu, *cpu_values))
        rss_profile[index] = max((rss, *rss_values))
        if cpu_values:
            cpu = cpu_values[-1]
            rss = rss_values[-1]
    return cpu_profile, rss_profile


def _input_tree_sha256() -> str:
    digest = hashlib.sha256()
    count = 0
    for task_dir in sorted(path for path in _RUN_DIR.iterdir() if path.is_dir()):
        for name in ("trace.jsonl", "resources.json"):
            path = task_dir / "attempt_1" / name
            relative = path.relative_to(_RUN_DIR).as_posix().encode()
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            count += 1
    if count != 152:
        raise ValueError(f"expected 152 formal input files, found {count}")
    return digest.hexdigest()


def _load_profiles() -> tuple[list[TaskProfile], dict[str, Any]]:
    input_sha256 = _input_tree_sha256()
    if input_sha256 != _INPUT_TREE_SHA256:
        raise ValueError(f"PennyLane input tree changed: {input_sha256}")
    profiles: list[TaskProfile] = []
    excluded: dict[str, str] = {}
    sample_gaps: list[float] = []
    for task_dir in sorted(_RUN_DIR.iterdir()):
        attempt = task_dir / "attempt_1"
        if not attempt.is_dir():
            continue
        trace_path = attempt / "trace.jsonl"
        resources_path = attempt / "resources.json"
        metadata, llm, action_first, action_last = _trace_times(trace_path)
        resources = json.loads(resources_path.read_text(encoding="utf-8"))
        samples = sorted(resources.get("samples", []), key=lambda row: float(row["epoch"]))
        if task_dir.name in _EXCLUDED:
            if not (
                metadata.get("mode") == "simulate"
                and metadata.get("llm_timing_mode") == "source_scaled"
                and metadata.get("replay_speed") == 20.0
                and not samples
            ):
                raise ValueError(f"{task_dir.name}: frozen exclusion no longer matches")
            excluded[task_dir.name] = "source_scaled_llm_and_empty_task_resources"
            continue
        if not (
            metadata.get("mode") == "collect"
            and metadata.get("simulate_mode") is None
            and metadata.get("llm_timing_mode") is None
            and metadata.get("replay_speed") is None
            and samples
        ):
            raise ValueError(f"{task_dir.name}: task is not an ordinary collection")
        epochs = [float(row["epoch"]) for row in samples]
        if any(right <= left for left, right in zip(epochs, epochs[1:])):
            raise ValueError(f"{task_dir.name}: non-increasing resource samples")
        sample_gaps.extend(right - left for left, right in zip(epochs, epochs[1:]))
        origin = min(action_first, epochs[0])
        end = max(action_last, epochs[-1])
        bins = max(1, math.ceil((end - origin) / _BIN_S))
        cpu, rss = _held_samples(samples, origin, bins)
        gpu = np.zeros(bins, dtype=np.float64)
        for start, finish in llm:
            first_bin = max(0, int((start - origin) // _BIN_S))
            end_bin = min(bins, math.ceil((finish - origin) / _BIN_S))
            gpu[first_bin:end_bin] += 1.0
        if gpu.max(initial=0.0) > 1.0:
            raise ValueError(f"{task_dir.name}: overlapping LLM calls within one task")
        profiles.append(TaskProfile(task_dir.name, gpu, cpu, rss))
    if len(profiles) != _EXPECTED_TASKS or set(excluded) != _EXCLUDED:
        raise ValueError(
            f"PennyLane cohort changed: {len(profiles)} profiles, excluded={sorted(excluded)}"
        )
    return profiles, {
        "task_count": len(profiles),
        "input_tree_sha256": input_sha256,
        "excluded": excluded,
        "resource_profile_bins": sum(profile.bins for profile in profiles),
        "median_raw_sample_gap_s": statistics.median(sample_gaps),
        "maximum_raw_sample_gap_s": max(sample_gaps),
        "llm_slot_bins": int(sum(profile.gpu.sum() for profile in profiles)),
    }


def _fits(
    profile: TaskProfile,
    now: int,
    gpu: np.ndarray,
    cpu: np.ndarray,
    rss: np.ndarray,
    gate: Gate,
    capacities: Capacities,
) -> bool:
    end = now + profile.bins
    if gate in {"gpu", "joint"} and np.any(
        gpu[now:end] + profile.gpu > capacities.gpu_slots + 1e-9
    ):
        return False
    return not (
        gate in {"tool", "joint"}
        and (
            np.any(cpu[now:end] + profile.cpu > capacities.cpu_cores + 1e-9)
            or np.any(rss[now:end] + profile.rss > capacities.rss_mb + 1e-9)
        )
    )


def simulate(
    profiles: list[TaskProfile],
    gate: Gate,
    *,
    active_cap: int | None,
    capacities: Capacities = Capacities(),
) -> dict[str, Any]:
    if not profiles or gate not in {"none", "gpu", "tool", "joint"}:
        raise ValueError("phase packing requires profiles and a known gate")
    if active_cap is not None and active_cap <= 0:
        raise ValueError("active_cap must be positive")
    if any(
        profile.gpu.max(initial=0.0) > capacities.gpu_slots
        or profile.cpu.max(initial=0.0) > capacities.cpu_cores
        or profile.rss.max(initial=0.0) > capacities.rss_mb
        for profile in profiles
    ):
        raise ValueError("an individual task profile exceeds capacity")
    horizon = sum(profile.bins for profile in profiles) + 1
    gpu = np.zeros(horizon, dtype=np.float64)
    cpu = np.zeros(horizon, dtype=np.float64)
    rss = np.zeros(horizon, dtype=np.float64)
    pending = list(profiles)
    active: list[tuple[int, TaskProfile]] = []
    starts: dict[str, int] = {}
    completions: dict[str, int] = {}
    now = 0

    while pending:
        for end, profile in list(active):
            if end <= now:
                active.remove((end, profile))
                completions[profile.task_id] = end
        admitted: list[TaskProfile] = []
        for profile in pending:
            if active_cap is not None and len(active) >= active_cap:
                break
            if not _fits(profile, now, gpu, cpu, rss, gate, capacities):
                continue
            end = now + profile.bins
            starts[profile.task_id] = now
            active.append((end, profile))
            gpu[now:end] += profile.gpu
            cpu[now:end] += profile.cpu
            rss[now:end] += profile.rss
            admitted.append(profile)
        pending = [profile for profile in pending if profile not in admitted]
        if not pending:
            break
        if not active:
            raise ValueError(f"{pending[0].task_id}: no admissible task")
        if active_cap is not None and len(active) >= active_cap:
            now = min(end for end, _profile in active)
        else:
            now += 1

    completions.update((profile.task_id, end) for end, profile in active)
    if set(starts) != {profile.task_id for profile in profiles} or set(completions) != set(starts):
        raise ValueError("phase packing did not schedule every task exactly once")
    makespan = max(completions.values())
    completion_values = list(completions.values())
    events = sorted(
        [(start, 1) for start in starts.values()]
        + [(end, -1) for end in completions.values()],
        key=lambda event: (event[0], event[1]),
    )
    maximum_active = current = 0
    for _time, delta in events:
        current += delta
        maximum_active = max(maximum_active, current)
    first_completion = min(completion_values)
    view = slice(0, makespan)
    violation_bins = {
        "gpu": int(np.count_nonzero(gpu[view] > capacities.gpu_slots + 1e-9)),
        "cpu": int(np.count_nonzero(cpu[view] > capacities.cpu_cores + 1e-9)),
        "rss": int(np.count_nonzero(rss[view] > capacities.rss_mb + 1e-9)),
    }
    return {
        "gate": gate,
        "active_cap": active_cap,
        "mean_task_completion_s": statistics.fmean(completion_values) * _BIN_S,
        "makespan_s": makespan * _BIN_S,
        "starts_before_first_completion": sum(
            start < first_completion for start in starts.values()
        ),
        "maximum_active_tasks": maximum_active,
        "llm_slot_utilization": float(gpu[view].sum())
        / (capacities.gpu_slots * makespan),
        "cpu_utilization": float(cpu[view].sum())
        / (capacities.cpu_cores * makespan),
        "rss_utilization": float(rss[view].sum())
        / (capacities.rss_mb * makespan),
        "peak_llm_slots": float(gpu[view].max(initial=0.0)),
        "peak_cpu_cores": float(cpu[view].max(initial=0.0)),
        "peak_rss_mb": float(rss[view].max(initial=0.0)),
        "violation_bins": violation_bins,
        "feasible": not any(violation_bins.values()),
        "start_s_by_task": {
            task_id: start * _BIN_S for task_id, start in sorted(starts.items())
        },
        "completion_s_by_task": {
            task_id: end * _BIN_S for task_id, end in sorted(completions.items())
        },
    }


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"start_s_by_task", "completion_s_by_task"}
    }


def best_feasible(
    profiles: list[TaskProfile], gate: Gate, capacities: Capacities
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    uncapped = simulate(profiles, gate, active_cap=None, capacities=capacities)
    search: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for cap in range(1, int(uncapped["maximum_active_tasks"]) + 1):
        result = simulate(profiles, gate, active_cap=cap, capacities=capacities)
        search.append(_compact(result))
        if result["feasible"]:
            candidates.append(result)
    if not candidates:
        raise ValueError(f"{gate}: no feasible global cap")
    return min(
        candidates,
        key=lambda result: (result["mean_task_completion_s"], result["active_cap"]),
    ), search


def _gate(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    joint = arms["joint"]
    reductions = {
        name: 1.0
        - joint["mean_task_completion_s"] / arms[name]["mean_task_completion_s"]
        for name in ("gpu_only", "tool_only")
    }
    checks = {
        "joint_zero_capacity_violations": joint["feasible"],
        "joint_mean_completion_at_least_5pct_below_gpu_only": reductions["gpu_only"]
        >= 0.05,
        "joint_mean_completion_at_least_5pct_below_tool_only": reductions["tool_only"]
        >= 0.05,
        "joint_makespan_no_higher_than_gpu_only": joint["makespan_s"]
        <= arms["gpu_only"]["makespan_s"],
        "joint_makespan_no_higher_than_tool_only": joint["makespan_s"]
        <= arms["tool_only"]["makespan_s"],
    }
    return {
        "status": "go" if all(checks.values()) else "no_go",
        "mean_completion_reduction": reductions,
        "checks": checks,
    }


def evaluate(git_sha: str) -> dict[str, Any]:
    profiles, evidence = _load_profiles()
    capacities = Capacities()
    static, static_search = best_feasible(profiles, "none", capacities)
    gpu, gpu_search = best_feasible(profiles, "gpu", capacities)
    tool, tool_search = best_feasible(profiles, "tool", capacities)
    joint = simulate(profiles, "joint", active_cap=None, capacities=capacities)
    arms = {"static": static, "gpu_only": gpu, "tool_only": tool, "joint": joint}
    gate = _gate(arms)
    return {
        "schema": "pennylane-joint-phase-packing-v1",
        "status": gate["status"],
        "corpus_role": "development_exposed",
        "git_sha": git_sha,
        "protocol": _PROTOCOL.relative_to(_ROOT).as_posix(),
        "run_dir": _RUN_DIR.relative_to(_ROOT).as_posix(),
        "bin_s": _BIN_S,
        "capacities": {
            "llm_request_slots": capacities.gpu_slots,
            "cpu_cores": capacities.cpu_cores,
            "rss_mb": capacities.rss_mb,
        },
        "evidence": evidence,
        "task_ids": [profile.task_id for profile in profiles],
        "arms": arms,
        "cap_search": {
            "static": static_search,
            "gpu_only": gpu_search,
            "tool_only": tool_search,
        },
        "gate": gate,
        "interpretation_boundary": (
            "Hindsight task-admission ceiling with fixed two-second profiles and "
            "recorded Codex LLM occupancy. It is not A100 service simulation or a "
            "causal scheduler, and modeled service inflation is zero by construction."
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
