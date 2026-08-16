#!/usr/bin/env python3
"""Evaluate frozen causal tool admission on PennyLane phase trajectories."""

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
import tempfile
from typing import Any, Literal, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from scripts.evaluation.evaluate_clause_latency_buckets import (  # noqa: E402
    evaluate_prequential_commands,
)
from scripts.evaluation.evaluate_clause_resource_classes import (  # noqa: E402
    CommandRow,
    load_rows,
    load_run_rows,
)
from scripts.evaluation.evaluate_command_prequential import (  # noqa: E402
    _load_exec_events,
)
from scripts.evaluation.evaluate_doc_tool_semantics import _task_aware  # noqa: E402
from scripts.evaluation.evaluate_pennylane_joint_phase_packing import (  # noqa: E402
    Capacities,
    TaskProfile,
    _INPUT_TREE_SHA256,
    _load_profiles,
    simulate as simulate_oracle,
)
from scripts.evaluation.evaluate_pennylane_multitarget import (  # noqa: E402
    PUBLIC_TELEMETRY,
    RUN_DIR,
    _write_results_view,
)

_PROTOCOL = (
    _ROOT / "analysis/development/pennylane-causal-joint-tool-admission-protocol.md"
)
_OUTPUT = (
    _ROOT / "analysis/results/pennylane-causal-joint-tool-admission-v1/result.json"
)
_BIN_S = 2.0
_ACTIVE_CAP = 8
_CPU_PAGES = (2.0, 4.0, 43.0)
_RSS_PAGES = (500.0, 2_000.0, 80_000.0)
_PUBLIC_SHA256 = {
    "simulate_cloud_model_c2_20260726T005356962.jsonl": (
        "1441c0113752b1508341a83b303cc0f059d3455cd78debd0a584fa051e93318e"
    ),
    "simulate_cloud_model_c2_20260726T024552768.jsonl": (
        "f4dc3e360d6f24c0c2436da956a9d1adf448defaa0a41034bb9373d2f8e6ee7c"
    ),
}
_EPS = 1e-8
Arm = Literal[
    "fixed4",
    "fixed8",
    "serial_tool",
    "task_aware_static",
    "task_aware_feedback",
]


@dataclass(frozen=True)
class Segment:
    duration_s: float
    kind: Literal["llm", "exec", "delay"]
    action_id: str | None
    cpu_cores: float
    rss_mb: float


@dataclass(frozen=True)
class Program:
    task_id: str
    segments: tuple[Segment, ...]


@dataclass
class _State:
    program: Program
    rank: int
    index: int = 0
    finish_s: float | None = None
    ready_s: float = 0.0
    held_rss_mb: float = 0.0
    cpu_request: float = 0.0
    rss_request: float = 0.0

    @property
    def segment(self) -> Segment:
        return self.program.segments[self.index]


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
        raise ValueError("causal joint-admission protocol differs from HEAD")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _actions_and_origin(task_id: str, bins: int) -> tuple[list[dict[str, Any]], float]:
    attempt = RUN_DIR / task_id / "attempt_1"
    actions: list[dict[str, Any]] = []
    with (attempt / "trace.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("type") != "action":
                continue
            start = float(row["ts_start"])
            end = float(row["ts_end"])
            if not math.isfinite(start) or not math.isfinite(end) or end < start:
                raise ValueError(f"{task_id}: invalid action interval")
            actions.append(row)
    actions.sort(key=lambda row: (float(row["ts_start"]), float(row["ts_end"])))
    if not actions or any(
        float(right["ts_start"]) < float(left["ts_end"]) - 1e-6
        for left, right in zip(actions, actions[1:])
    ):
        raise ValueError(f"{task_id}: missing or overlapping actions")
    resources = json.loads((attempt / "resources.json").read_text(encoding="utf-8"))
    samples = resources.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{task_id}: task samples unavailable")
    origin = min(float(actions[0]["ts_start"]), float(samples[0]["epoch"]))
    if origin + bins * _BIN_S + _EPS < float(actions[-1]["ts_end"]):
        raise ValueError(f"{task_id}: profile ends before its actions")
    return actions, origin


def _action_kind(row: Mapping[str, Any]) -> tuple[str, str | None]:
    if row.get("action_type") == "llm_call":
        return "llm", str(row["action_id"])
    data = row.get("data")
    if (
        row.get("action_type") == "tool_exec"
        and isinstance(data, Mapping)
        and data.get("tool_name") == "exec"
    ):
        call_id = data.get("tool_call_id")
        timeline = data.get("resource_timeline")
        if not isinstance(call_id, str) or not (
            isinstance(timeline, Mapping)
            and isinstance(timeline.get("samples"), list)
            and timeline["samples"]
        ):
            raise ValueError("exec action lacks identity or CPU timeline")
        return "exec", call_id
    return "delay", None


def _segments(profile: TaskProfile) -> Program:
    actions, origin = _actions_and_origin(profile.task_id, profile.bins)
    end = origin + profile.bins * _BIN_S
    boundaries = {origin + index * _BIN_S for index in range(profile.bins + 1)}
    for row in actions:
        boundaries.add(max(origin, float(row["ts_start"])))
        boundaries.add(min(end, float(row["ts_end"])))
    ordered = sorted(value for value in boundaries if origin <= value <= end)
    segments: list[Segment] = []
    action_index = 0
    for left, right in zip(ordered, ordered[1:]):
        if right - left <= _EPS:
            continue
        midpoint = (left + right) / 2.0
        while (
            action_index < len(actions)
            and float(actions[action_index]["ts_end"]) <= midpoint
        ):
            action_index += 1
        action = (
            actions[action_index]
            if action_index < len(actions)
            and float(actions[action_index]["ts_start"])
            <= midpoint
            < float(actions[action_index]["ts_end"])
            else None
        )
        kind, action_id = ("delay", None) if action is None else _action_kind(action)
        bin_index = min(int((midpoint - origin) // _BIN_S), profile.bins - 1)
        segment = Segment(
            right - left,
            kind,  # type: ignore[arg-type]
            action_id,
            float(profile.cpu[bin_index]),
            float(profile.rss[bin_index]),
        )
        segments.append(segment)
    if not segments or not math.isclose(
        sum(segment.duration_s for segment in segments),
        profile.bins * _BIN_S,
        rel_tol=1e-12,
        abs_tol=1e-5,
    ):
        raise ValueError(f"{profile.task_id}: segments do not preserve duration")
    return Program(profile.task_id, tuple(segments))


def _reservations(
    fit_ids: Sequence[str], replay_ids: Sequence[str]
) -> tuple[dict[tuple[str, str], tuple[float, float]], dict[str, Any]]:
    task_ids = [*fit_ids, *replay_ids]
    with tempfile.TemporaryDirectory() as temporary:
        view = Path(temporary) / "results.jsonl"
        _write_results_view(view, task_ids)
        loaded, clauses, commands = load_run_rows(RUN_DIR, results_path=view)
        events = _load_exec_events(RUN_DIR, task_ids, results_path=view)
    if loaded != task_ids:
        raise ValueError("predictor tasks differ from frozen causal split")
    public_inputs = []
    public = []
    for path in PUBLIC_TELEMETRY:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = _PUBLIC_SHA256.get(path.name)
        if digest != expected:
            raise ValueError(f"public predictor input changed: {path}")
        rows = load_rows(path)
        public.extend(rows)
        public_inputs.append(
            {
                "path": path.relative_to(_ROOT).as_posix(),
                "sha256": digest,
                "rows": len(rows),
            }
        )
    public = [row for row in public if row.repo != "PennyLaneAI__pennylane"]
    provenance = {
        "role": "development_exposed_causal_action",
        "input_tree_sha256": _INPUT_TREE_SHA256,
    }
    _baseline, baseline_rows = evaluate_prequential_commands(
        public,
        task_ids,
        clauses,
        commands,
        provenance,
        warmup_task_count=len(fit_ids),
    )
    by_task: dict[str, list[CommandRow]] = {task_id: [] for task_id in task_ids}
    for row in commands:
        by_task[row.task_id].append(row)
    fit = [row for task_id in fit_ids for row in by_task[task_id]]
    replay = [row for task_id in replay_ids for row in by_task[task_id]]
    baseline_by_sample = {str(row["sample_id"]): row for row in baseline_rows}
    predictions = _task_aware(fit, replay, baseline_by_sample, events_by_task=events)
    requested: dict[tuple[str, str], tuple[float, float]] = {}
    available_cpu = available_rss = 0
    for row in replay:
        sample_id = f"{row.task_id}:{row.call_index}"
        pmfs = predictions[sample_id]["candidate_probability_by_bucket"]
        cpu_pmf = pmfs.get("peak_cpu_cores")
        rss_pmf = pmfs.get("sampled_peak_rss_mb")
        cpu_index = (
            2 if cpu_pmf is None else max(range(3), key=lambda index: cpu_pmf[index])
        )
        rss_index = (
            2 if rss_pmf is None else max(range(3), key=lambda index: rss_pmf[index])
        )
        available_cpu += cpu_pmf is not None
        available_rss += rss_pmf is not None
        requested[(row.task_id, row.call_id)] = (
            _CPU_PAGES[cpu_index],
            _RSS_PAGES[rss_index],
        )
    return requested, {
        "fit_tasks": len(fit_ids),
        "replay_tasks": len(replay_ids),
        "fit_commands": len(fit),
        "replay_commands": len(replay),
        "cpu_predictions_available": available_cpu,
        "rss_predictions_available": available_rss,
        "public_inputs": public_inputs,
    }


def _rss_request(states: Sequence[_State]) -> float:
    return sum(
        max(state.held_rss_mb, state.rss_request)
        if state.finish_s is not None and state.segment.kind == "exec"
        else state.held_rss_mb
        for state in states
    )


def simulate(
    programs: Sequence[Program],
    arm: Arm,
    reservations: Mapping[tuple[str, str], tuple[float, float]],
    *,
    capacities: Capacities = Capacities(),
) -> dict[str, Any]:
    if not programs or arm not in {
        "fixed4",
        "fixed8",
        "serial_tool",
        "task_aware_static",
        "task_aware_feedback",
    }:
        raise ValueError("causal joint admission requires programs and a known arm")
    active_cap = 4 if arm == "fixed4" else _ACTIVE_CAP
    pending = list(enumerate(programs))
    active: list[_State] = []
    completions: dict[str, float] = {}
    admissions: dict[str, float] = {}
    now = 0.0
    llm_queue_s = tool_queue_s = 0.0
    cpu_work = rss_mb_s = llm_slot_s = 0.0
    violation_s = {"gpu": 0.0, "cpu": 0.0, "rss": 0.0}
    peak_gpu = peak_cpu = peak_rss = 0.0
    reservation_under_s = {"cpu": 0.0, "rss": 0.0}
    overlapping_exec_tasks: set[str] = set()
    max_active = 0
    deadlock: dict[str, Any] | None = None

    def admit() -> None:
        while pending and len(active) < active_cap:
            rank, program = pending.pop(0)
            state = _State(
                program,
                rank,
                held_rss_mb=0.0,
            )
            active.append(state)
            admissions[program.task_id] = now

    while active or pending:
        admit()
        max_active = max(max_active, len(active))

        completed_segments = [
            state
            for state in active
            if state.finish_s is not None and state.finish_s <= now + _EPS
        ]
        for state in completed_segments:
            previous = state.segment
            state.held_rss_mb = previous.rss_mb
            state.index += 1
            state.finish_s = None
            if state.index == len(state.program.segments):
                completions[state.program.task_id] = now
                active.remove(state)
                continue
            current = state.segment
            continuation = (
                previous.kind in {"llm", "exec"}
                and current.kind == previous.kind
                and current.action_id == previous.action_id
            )
            if continuation:
                if arm == "task_aware_feedback" and current.kind == "exec":
                    state.cpu_request = previous.cpu_cores
                    state.rss_request = previous.rss_mb
                state.finish_s = now + current.duration_s
            else:
                state.cpu_request = state.rss_request = 0.0
                state.ready_s = now

        admit()
        max_active = max(max_active, len(active))
        if not active and not pending:
            break
        running_llm = sum(
            state.finish_s is not None and state.segment.kind == "llm"
            for state in active
        )
        running_exec = sum(
            state.finish_s is not None and state.segment.kind == "exec"
            for state in active
        )
        reserved_cpu = sum(
            state.cpu_request
            for state in active
            if state.finish_s is not None and state.segment.kind == "exec"
        )
        reserved_rss = _rss_request(active)

        ready = sorted(
            (state for state in active if state.finish_s is None),
            key=lambda state: (state.ready_s, state.rank),
        )
        for state in ready:
            segment = state.segment
            cpu_request = rss_request = 0.0
            allowed = True
            if segment.kind == "llm":
                allowed = running_llm < capacities.gpu_slots
            elif segment.kind == "exec":
                if arm == "serial_tool":
                    allowed = running_exec == 0
                elif arm in {"task_aware_static", "task_aware_feedback"}:
                    cpu_request, rss_request = reservations.get(
                        (state.program.task_id, str(segment.action_id)),
                        (_CPU_PAGES[-1], _RSS_PAGES[-1]),
                    )
                    candidate_rss = (
                        reserved_rss
                        - state.held_rss_mb
                        + max(state.held_rss_mb, rss_request)
                    )
                    allowed = (
                        reserved_cpu + cpu_request <= capacities.cpu_cores + _EPS
                        and candidate_rss <= capacities.rss_mb + _EPS
                    )
            if not allowed:
                continue
            state.cpu_request = cpu_request
            state.rss_request = rss_request
            state.finish_s = now + segment.duration_s
            if segment.kind == "llm":
                running_llm += 1
            elif segment.kind == "exec":
                running_exec += 1
                reserved_cpu += cpu_request
                reserved_rss = _rss_request(active)

        running = [state for state in active if state.finish_s is not None]
        if not running:
            deadlock = {
                "time_s": now,
                "ready_tasks": [state.program.task_id for state in ready],
                "ready_kinds": [state.segment.kind for state in ready],
            }
            break
        next_time = min(float(state.finish_s) for state in running)
        duration = next_time - now
        if duration <= 0.0 or not math.isfinite(duration):
            raise ValueError("causal joint admission made no temporal progress")
        waiting = [state for state in active if state.finish_s is None]
        llm_queue_s += duration * sum(state.segment.kind == "llm" for state in waiting)
        tool_queue_s += duration * sum(
            state.segment.kind == "exec" for state in waiting
        )

        gpu = float(sum(state.segment.kind == "llm" for state in running))
        cpu = sum(state.segment.cpu_cores for state in running)
        rss = sum(
            state.segment.rss_mb if state.finish_s is not None else state.held_rss_mb
            for state in active
        )
        peak_gpu = max(peak_gpu, gpu)
        peak_cpu = max(peak_cpu, cpu)
        peak_rss = max(peak_rss, rss)
        llm_slot_s += gpu * duration
        cpu_work += cpu * duration
        rss_mb_s += rss * duration
        violation_s["gpu"] += duration if gpu > capacities.gpu_slots + _EPS else 0.0
        violation_s["cpu"] += duration if cpu > capacities.cpu_cores + _EPS else 0.0
        violation_s["rss"] += duration if rss > capacities.rss_mb + _EPS else 0.0
        if arm in {"task_aware_static", "task_aware_feedback"}:
            actual_exec_cpu = sum(
                state.segment.cpu_cores
                for state in running
                if state.segment.kind == "exec"
            )
            actual_rss = sum(
                state.segment.rss_mb
                if state.finish_s is not None
                else state.held_rss_mb
                for state in active
            )
            reservation_under_s["cpu"] += (
                duration if actual_exec_cpu > reserved_cpu + _EPS else 0.0
            )
            reservation_under_s["rss"] += (
                duration if actual_rss > _rss_request(active) + _EPS else 0.0
            )
        exec_tasks = {
            state.program.task_id for state in running if state.segment.kind == "exec"
        }
        if len(exec_tasks) >= 2:
            overlapping_exec_tasks.update(exec_tasks)
        now = next_time

    complete = len(completions) == len(programs)
    completion_values = list(completions.values())
    makespan = max(completion_values, default=now)
    first_completion = min(completion_values, default=math.inf)
    return {
        "completed": complete,
        "completed_tasks": len(completions),
        "active_cap": active_cap,
        "mean_task_completion_s": (
            statistics.fmean(completion_values) if complete else None
        ),
        "makespan_s": makespan if complete else None,
        "llm_queue_s": llm_queue_s,
        "tool_queue_s": tool_queue_s,
        "starts_before_first_completion": sum(
            value < first_completion for value in admissions.values()
        ),
        "maximum_active_tasks": max_active,
        "llm_slot_utilization": (
            llm_slot_s / (capacities.gpu_slots * makespan) if complete else None
        ),
        "cpu_utilization": (
            cpu_work / (capacities.cpu_cores * makespan) if complete else None
        ),
        "rss_utilization": (
            rss_mb_s / (capacities.rss_mb * makespan) if complete else None
        ),
        "peak_llm_slots": peak_gpu,
        "peak_cpu_cores": peak_cpu,
        "peak_rss_mb": peak_rss,
        "capacity_violation_s": violation_s,
        "reservation_underprediction_s": reservation_under_s,
        "tasks_with_overlapped_exec": len(overlapping_exec_tasks),
        "deadlock": deadlock,
    }


def _gate(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    candidate = arms["task_aware_feedback"]
    baseline = arms["serial_tool"]
    candidate_mean = candidate["mean_task_completion_s"]
    baseline_mean = baseline["mean_task_completion_s"]
    candidate_safe = candidate["completed"] and not any(
        float(value) > _EPS for value in candidate["capacity_violation_s"].values()
    )
    baseline_safe = baseline["completed"] and not any(
        float(value) > _EPS for value in baseline["capacity_violation_s"].values()
    )
    reduction = (
        None
        if not candidate_safe
        or not baseline_safe
        or candidate_mean is None
        or baseline_mean is None
        else 1.0 - float(candidate_mean) / float(baseline_mean)
    )
    checks = {
        "candidate_completed_and_safe": candidate_safe,
        "serial_tool_completed_and_safe": baseline_safe,
        "mean_completion_at_least_5pct_below_serial_tool": (
            reduction is not None and reduction >= 0.05
        ),
        "makespan_no_higher_than_serial_tool": (
            candidate["makespan_s"] is not None
            and baseline["makespan_s"] is not None
            and candidate["makespan_s"] <= baseline["makespan_s"]
        ),
        "overlap_spans_at_least_20_tasks": candidate["tasks_with_overlapped_exec"]
        >= 20,
    }
    return {
        "status": "go" if all(checks.values()) else "no_go",
        "mean_completion_reduction_vs_serial_tool": reduction,
        "checks": checks,
    }


def evaluate(git_sha: str) -> dict[str, Any]:
    profiles, source_evidence = _load_profiles()
    ordered = sorted(profiles, key=lambda profile: profile.task_id)
    fit_profiles = ordered[:35]
    replay_profiles = ordered[35:]
    fit_ids = [profile.task_id for profile in fit_profiles]
    replay_ids = [profile.task_id for profile in replay_profiles]
    programs = [_segments(profile) for profile in replay_profiles]
    reservations, prediction_evidence = _reservations(fit_ids, replay_ids)
    arms = {
        arm: simulate(programs, arm, reservations)
        for arm in (
            "fixed4",
            "fixed8",
            "serial_tool",
            "task_aware_static",
            "task_aware_feedback",
        )
    }
    oracle = simulate_oracle(
        replay_profiles,
        "joint",
        active_cap=_ACTIVE_CAP,
        capacities=Capacities(),
    )
    arms["joint_oracle_cap8"] = oracle
    gate = _gate(arms)
    return {
        "schema": "pennylane-causal-joint-tool-admission-v1",
        "status": gate["status"],
        "corpus_role": "development_exposed",
        "git_sha": git_sha,
        "protocol": _PROTOCOL.relative_to(_ROOT).as_posix(),
        "run_dir": RUN_DIR.relative_to(_ROOT).as_posix(),
        "input_tree_sha256": _INPUT_TREE_SHA256,
        "capacities": {
            "llm_request_slots": 4,
            "cpu_cores": 43,
            "rss_mb": 80_000,
            "active_tasks": _ACTIVE_CAP,
        },
        "fit_task_ids": fit_ids,
        "replay_task_ids": replay_ids,
        "evidence": {
            "source": source_evidence,
            "prediction": prediction_evidence,
            "segment_count": sum(len(program.segments) for program in programs),
            "prediction_time_agent_calls": 0,
        },
        "arms": arms,
        "gate": gate,
        "interpretation_boundary": (
            "Causal action-visibility replay with shifted fixed-service trajectories. "
            "A zero-violation result is required before completion-time credit; this "
            "is not physical concurrency, GPU service, or fresh-data confirmation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_OUTPUT)
    args = parser.parse_args()
    result = evaluate(_require_clean_checkout())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "gate": result["gate"]}, indent=2))


if __name__ == "__main__":
    main()
