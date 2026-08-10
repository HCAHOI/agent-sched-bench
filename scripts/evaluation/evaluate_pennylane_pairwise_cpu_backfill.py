#!/usr/bin/env python3
"""Evaluate frozen PennyLane pairwise CPU backfill ceilings."""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from tool_resource.runtime_kb import (  # noqa: E402
    _command_stages,
    parse_command_clauses,
)
from tool_resource_eval.early_cpu_reservation import cpu_work_profile  # noqa: E402
from tool_resource_eval.resource_admission import (  # noqa: E402
    AdmissionCommand,
    AdmissionProgram,
    simulate_idle_backfill,
)
from trace_collect.trace_data import TraceData  # noqa: E402


_PROTOCOL_GIT_SHA = "c9291b04ff0adc5da4c24a34b82c00a2a9cc6589"
_SPLIT = _ROOT / "analysis/development/pennylane-survival-action-split.json"
_CPU_CAPACITY = 8.0
_RSS_CAPACITY_MB = 16_000.0
_MINIMUM_REDUCTION = 0.05
_MAXIMUM_SERVICE_INFLATION = 0.05
_ARMS = (
    "serial8",
    "rss_safe_fcfs",
    "pairwise_mean_fcfs",
    "pairwise_exact_peak_fcfs",
    "pairwise_canonical_bucket_fcfs",
    "pairwise_phase_shape_fcfs",
    "clause_kb_phase_envelope_fcfs",
)


def _require_clean_checkout() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("formal evaluation requires a clean committed checkout")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", _PROTOCOL_GIT_SHA, "HEAD"],
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_frozen_split() -> dict[str, Any]:
    repo_path = _SPLIT.relative_to(_ROOT).as_posix()
    expected = subprocess.run(
        ["git", "show", f"{_PROTOCOL_GIT_SHA}:{repo_path}"],
        check=True,
        capture_output=True,
    ).stdout
    if _SPLIT.read_bytes() != expected:
        raise ValueError("frozen PennyLane split changed")
    return json.loads(expected)


def _valid_artifact(artifact: Mapping[str, Any]) -> bool:
    return (
        artifact.get("collection_validity"),
        artifact.get("workload_execution"),
        artifact.get("telemetry_quality"),
        artifact.get("cleanup"),
    ) == ("valid", "completed", "ok", "ok")


def _observed_rss(call: Mapping[str, Any] | None) -> tuple[float, bool]:
    if (
        call is None
        or call.get("eligible_for_kb") is not True
        or call.get("invalid_reasons")
    ):
        return _RSS_CAPACITY_MB, False
    clauses = call.get("clauses")
    if not isinstance(clauses, list) or not clauses:
        return _RSS_CAPACITY_MB, False
    stages = _command_stages(
        [
            {
                "in_pipe": clause.get("in_pipe"),
                "in_subst": clause.get("in_subst"),
                "pipeline_position": clause.get("pipeline_position"),
            }
            for clause in clauses
            if isinstance(clause, Mapping)
        ]
    )
    values = [
        clause.get("sampled_peak_rss_mb")
        for clause in clauses
        if isinstance(clause, Mapping)
    ]
    if (
        stages is None
        or len(values) != len(clauses)
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0.0
            for value in values
        )
    ):
        return _RSS_CAPACITY_MB, False
    rss_mb = max(sum(float(values[index]) for index in stage) for stage in stages)
    return min(_RSS_CAPACITY_MB, rss_mb), True


def _load_programs(
    split: Mapping[str, Any],
) -> tuple[
    dict[str, AdmissionProgram],
    dict[str, tuple[tuple[float, float], ...]],
    dict[str, str],
    dict[str, float],
    set[str],
    list[dict[str, Any]],
    Counter[str],
]:
    programs: dict[str, AdmissionProgram] = {}
    profiles: dict[str, tuple[tuple[float, float], ...]] = {}
    command_text: dict[str, str] = {}
    mean_cpu: dict[str, float] = {}
    rss_safe: set[str] = set()
    excluded: list[dict[str, Any]] = []
    rss_sources: Counter[str] = Counter()
    for item in split["replay"]:
        task_id = str(item["task_id"])
        trace_path = _ROOT / str(item["trace"])
        artifact = json.loads(
            (trace_path.parent / "resource_observations.json").read_text(
                encoding="utf-8"
            )
        )
        if not _valid_artifact(artifact):
            excluded.append(
                {
                    "task_id": task_id,
                    "collection_validity": artifact.get("collection_validity"),
                    "telemetry_quality": artifact.get("telemetry_quality"),
                    "cleanup": artifact.get("cleanup"),
                }
            )
            continue
        calls = {
            call.get("tool_call_id"): call
            for call in artifact.get("calls", [])
            if isinstance(call, Mapping)
            and isinstance(call.get("tool_call_id"), str)
        }
        actions = sorted(
            TraceData.load(trace_path).actions,
            key=lambda action: float(action["ts_start"]),
        )
        exec_actions = [
            action
            for action in actions
            if action.get("action_type") == "tool_exec"
            and isinstance(action.get("data"), Mapping)
            and action["data"].get("tool_name") == "exec"
        ]
        if not exec_actions:
            raise ValueError(f"{task_id}: no exec commands")
        commands: list[AdmissionCommand] = []
        for index, action in enumerate(exec_actions):
            data = action["data"]
            call_id = data.get("tool_call_id")
            if not isinstance(call_id, str):
                raise ValueError(f"{task_id}: exec command lacks tool_call_id")
            command_id = f"{task_id}:{call_id}"
            start_s = float(action["ts_start"])
            end_s = float(action["ts_end"])
            profile = cpu_work_profile(action)
            if profile is None:
                raise ValueError(f"{command_id}: CPU profile is unavailable")
            profiles[command_id] = profile
            tool_args = data.get("tool_args")
            parsed_args = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
            if not isinstance(parsed_args, Mapping) or not isinstance(
                parsed_args.get("command"), str
            ):
                raise ValueError(f"{command_id}: command text is unavailable")
            command_text[command_id] = parsed_args["command"]
            mean_cpu[command_id] = sum(cpu for _span, cpu in profile) / sum(
                span for span, _cpu in profile
            )
            rss_mb, safe = _observed_rss(calls.get(call_id))
            rss_sources["observed_upper_bound" if safe else "full_fallback"] += 1
            if safe:
                rss_safe.add(command_id)
            next_start_s = (
                float(exec_actions[index + 1]["ts_start"])
                if index + 1 < len(exec_actions)
                else end_s
            )
            commands.append(
                AdmissionCommand(
                    command_id,
                    end_s - start_s,
                    _CPU_CAPACITY,
                    rss_mb,
                    max(0.0, next_start_s - end_s),
                )
            )
        task_start_s = min(float(action["ts_start"]) for action in actions)
        task_end_s = max(float(action["ts_end"]) for action in actions)
        programs[task_id] = AdmissionProgram(
            task_id,
            max(0.0, float(exec_actions[0]["ts_start"]) - task_start_s),
            tuple(commands),
            max(0.0, task_end_s - float(exec_actions[-1]["ts_end"])),
        )
    return (
        programs,
        profiles,
        command_text,
        mean_cpu,
        rss_safe,
        excluded,
        rss_sources,
    )


def _comparison(
    arms: Mapping[str, Mapping[str, object]], arm: str
) -> dict[str, Any]:
    baseline = arms["serial8"]
    candidate = arms[arm]
    return {
        "baseline": "serial8",
        "candidate": arm,
        "mean_task_completion_reduction": 1.0
        - float(candidate["mean_task_completion_s"])
        / float(baseline["mean_task_completion_s"]),
        "service_inflation": float(candidate["total_command_service_s"])
        / float(candidate["recorded_command_service_s"])
        - 1.0,
        "makespan_reduction": 1.0
        - float(candidate["makespan_s"]) / float(baseline["makespan_s"]),
    }


def _bucket_upper(peak_cpu: float) -> float:
    if peak_cpu <= 2.0:
        return 2.0
    if peak_cpu <= 4.0:
        return 4.0
    return _CPU_CAPACITY


def _command_signatures(command: str) -> tuple[tuple[Any, ...], ...]:
    parsed = parse_command_clauses(command)
    if parsed["parse_failed"] or not parsed["clauses"]:
        return ()
    operators = tuple(edge["operator"] for edge in parsed["control_edges"])

    def signature(label: str, depth: int | None) -> tuple[Any, ...]:
        clauses = []
        for clause in parsed["clauses"]:
            tokens = (clause["bin"], *clause["argv"][1:])
            clauses.append(
                (
                    clause["in_loop"],
                    clause["in_pipe"],
                    clause["in_subst"],
                    clause["pipeline_position"],
                    tuple(clause["structural_context"]),
                    tokens if depth is None else tokens[:depth],
                )
            )
        return label, operators, tuple(clauses)

    return (
        signature("exact", None),
        *(signature(f"depth_{depth}", depth) for depth in (4, 3, 2, 1)),
    )


def _profile_envelope(
    profiles: tuple[tuple[tuple[float, float], ...], ...],
) -> tuple[tuple[float, float], ...]:
    boundaries = {0.0}
    for profile in profiles:
        elapsed = 0.0
        for span, _work in profile:
            elapsed += span
            boundaries.add(elapsed)
    ordered = sorted(boundaries)
    envelope = []
    for start, end in zip(ordered, ordered[1:]):
        midpoint = (start + end) / 2.0
        rates = []
        for profile in profiles:
            elapsed = 0.0
            for span, work in profile:
                if midpoint < elapsed + span:
                    rates.append(work / span)
                    break
                elapsed += span
        span = end - start
        envelope.append((span, max(rates, default=0.0) * span))
    return tuple(envelope)


def _fit_candidate_envelopes(
    fit_commands: Mapping[str, str],
    fit_profiles: Mapping[str, tuple[tuple[float, float], ...]],
    replay_commands: Mapping[str, str],
    replay_profiles: Mapping[str, tuple[tuple[float, float], ...]],
    eligible_ids: set[str],
) -> tuple[dict[str, tuple[tuple[float, float], ...]], dict[str, Any]]:
    fit_by_signature: dict[
        tuple[Any, ...], list[tuple[tuple[float, float], ...]]
    ] = {}
    for command_id, command in fit_commands.items():
        for signature in _command_signatures(command):
            fit_by_signature.setdefault(signature, []).append(fit_profiles[command_id])

    cache: dict[tuple[Any, ...], tuple[tuple[float, float], ...]] = {}
    predicted: dict[str, tuple[tuple[float, float], ...]] = {}
    levels: Counter[str] = Counter()
    supports: Counter[int] = Counter()
    outlasted = 0
    maximum_unmodeled_tail_s = 0.0
    for command_id in sorted(eligible_ids):
        match = next(
            (
                signature
                for signature in _command_signatures(replay_commands[command_id])
                if signature in fit_by_signature
            ),
            None,
        )
        if match is None:
            continue
        if match not in cache:
            cache[match] = _profile_envelope(tuple(fit_by_signature[match]))
        envelope = cache[match]
        predicted[command_id] = envelope
        levels[str(match[0])] += 1
        supports[len(fit_by_signature[match])] += 1
        actual_duration = sum(span for span, _work in replay_profiles[command_id])
        predicted_duration = sum(span for span, _work in envelope)
        if actual_duration > predicted_duration + 1e-9:
            outlasted += 1
            maximum_unmodeled_tail_s = max(
                maximum_unmodeled_tail_s,
                actual_duration - predicted_duration,
            )
    return predicted, {
        "fit_signatures": len(fit_by_signature),
        "matched_commands": len(predicted),
        "match_levels": dict(sorted(levels.items())),
        "fit_support_counts": {
            str(support): count for support, count in sorted(supports.items())
        },
        "commands_outlasting_envelope": outlasted,
        "maximum_unmodeled_tail_s": maximum_unmodeled_tail_s,
    }


def _evaluate() -> dict[str, Any]:
    started = time.monotonic()
    git_sha = _require_clean_checkout()
    split = _require_frozen_split()
    (
        programs,
        profiles,
        commands,
        mean_cpu,
        rss_safe,
        excluded,
        rss_sources,
    ) = _load_programs(split)
    (
        fit_programs,
        fit_profiles,
        fit_commands,
        _fit_mean_cpu,
        _fit_rss_safe,
        fit_excluded,
        _fit_rss_sources,
    ) = _load_programs({"replay": split["fit"]})
    if len(programs) != 15 or len(profiles) != 570 or len(excluded) != 11:
        raise ValueError("PennyLane evidence-valid population changed")
    if len(fit_programs) != 15 or len(fit_profiles) != 637 or fit_excluded:
        raise ValueError("PennyLane fit population changed")
    candidate_profiles, envelope_evidence = _fit_candidate_envelopes(
        fit_commands,
        fit_profiles,
        commands,
        profiles,
        rss_safe,
    )
    chosen = [programs[task_id] for task_id in sorted(programs)]
    peak_cpu = {
        command_id: max(cpu / span for span, cpu in profile)
        for command_id, profile in profiles.items()
    }
    bucket_cpu = {
        command_id: _bucket_upper(value) for command_id, value in peak_cpu.items()
    }
    arms = {
        "serial8": simulate_idle_backfill(
            chosen,
            cpu_capacity=_CPU_CAPACITY,
            rss_capacity_mb=_RSS_CAPACITY_MB,
            cpu_work_profiles=profiles,
            speculative_eligible_command_ids=rss_safe,
            selection="serial",
        ),
        "rss_safe_fcfs": simulate_idle_backfill(
            chosen,
            cpu_capacity=_CPU_CAPACITY,
            rss_capacity_mb=_RSS_CAPACITY_MB,
            cpu_work_profiles=profiles,
            speculative_eligible_command_ids=rss_safe,
            selection="fcfs",
        ),
        "pairwise_mean_fcfs": simulate_idle_backfill(
            chosen,
            cpu_capacity=_CPU_CAPACITY,
            rss_capacity_mb=_RSS_CAPACITY_MB,
            cpu_work_profiles=profiles,
            speculative_eligible_command_ids=rss_safe,
            pairwise_cpu_demands=mean_cpu,
            selection="fcfs",
        ),
        "pairwise_exact_peak_fcfs": simulate_idle_backfill(
            chosen,
            cpu_capacity=_CPU_CAPACITY,
            rss_capacity_mb=_RSS_CAPACITY_MB,
            cpu_work_profiles=profiles,
            speculative_eligible_command_ids=rss_safe,
            pairwise_cpu_demands=peak_cpu,
            selection="fcfs",
        ),
        "pairwise_canonical_bucket_fcfs": simulate_idle_backfill(
            chosen,
            cpu_capacity=_CPU_CAPACITY,
            rss_capacity_mb=_RSS_CAPACITY_MB,
            cpu_work_profiles=profiles,
            speculative_eligible_command_ids=rss_safe,
            pairwise_cpu_demands=bucket_cpu,
            selection="fcfs",
        ),
        "pairwise_phase_shape_fcfs": simulate_idle_backfill(
            chosen,
            cpu_capacity=_CPU_CAPACITY,
            rss_capacity_mb=_RSS_CAPACITY_MB,
            cpu_work_profiles=profiles,
            speculative_eligible_command_ids=rss_safe,
            require_pairwise_profile_compatibility=True,
            selection="fcfs",
        ),
        "clause_kb_phase_envelope_fcfs": simulate_idle_backfill(
            chosen,
            cpu_capacity=_CPU_CAPACITY,
            rss_capacity_mb=_RSS_CAPACITY_MB,
            cpu_work_profiles=profiles,
            speculative_eligible_command_ids=set(candidate_profiles),
            pairwise_candidate_profiles=candidate_profiles,
            require_pairwise_profile_compatibility=True,
            selection="fcfs",
        ),
    }
    violation_by_arm = {}
    for arm, metrics in arms.items():
        violation_by_arm[arm] = bool(
            metrics["capacity_violation"]
            or metrics["physical_capacity_violation"]
            or metrics["modeled_capacity_exposure_events"]
            or not math.isclose(
                float(metrics["total_cpu_work_core_s"]),
                float(metrics["served_cpu_work_core_s"]),
                rel_tol=1e-12,
                abs_tol=1e-7,
            )
        )
        for key in (
            "speculative_start_ids",
            "service_s_by_command",
            "start_s_by_command",
            "modeled_capacity_exposure_command_ids",
            "rss_unverified_overlap_command_ids",
        ):
            metrics.pop(key)
    comparisons = {
        arm: _comparison(arms, arm) for arm in _ARMS if arm != "serial8"
    }
    gates = {
        arm: {
            "mean_task_completion_reduction_at_least_5_percent": comparisons[arm][
                "mean_task_completion_reduction"
            ]
            >= _MINIMUM_REDUCTION,
            "service_inflation_at_most_5_percent": comparisons[arm][
                "service_inflation"
            ]
            <= _MAXIMUM_SERVICE_INFLATION,
            "zero_capacity_or_work_violations": not violation_by_arm[arm],
        }
        for arm in ("clause_kb_phase_envelope_fcfs",)
    }
    for gate in gates.values():
        gate["go"] = all(gate.values())
    phase_go = gates["clause_kb_phase_envelope_fcfs"]["go"]
    return {
        "schema_version": 1,
        "status": "development_go_to_foreground_feedback_feasibility"
        if phase_go
        else "development_no_go",
        "generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_sha": git_sha,
        "protocol_git_sha": _PROTOCOL_GIT_SHA,
        "protocol": "tool-resource-canonical-objective.md Section 5.16",
        "config": {
            "split": str(_SPLIT.resolve()),
            "cpu_capacity": _CPU_CAPACITY,
            "rss_capacity_mb": _RSS_CAPACITY_MB,
            "task_selection": "all evidence-valid replay tasks; no subsampling",
            "arrival_model": "one concurrent task-start wave with recorded first-command delays",
            "pairwise_rules": {
                "mean": "foreground mean CPU + candidate mean CPU <= 8",
                "exact_peak": "foreground exact peak CPU + candidate exact peak CPU <= 8",
                "canonical_bucket": "foreground and candidate peak labels map to upper bounds 2/4/8 whose sum <= 8",
                "phase_shape": "aligned remaining foreground and candidate sampled CPU demand never exceeds 8",
                "clause_kb_phase_envelope": "fit-only pointwise-max candidate envelope plus actual remaining foreground profile",
            },
            "candidate_signature_hierarchy": [
                "exact",
                "depth_4",
                "depth_3",
                "depth_2",
                "depth_1_binary",
            ],
            "rss_rule": "observed composed upper bound only",
        },
        "evidence": {
            "fit_tasks_read": len(fit_programs),
            "fit_exec_commands": len(fit_profiles),
            "replay_tasks": len(programs),
            "excluded_replay_tasks": excluded,
            "exec_commands": len(profiles),
            "rss_safe_commands": len(rss_safe),
            "rss_source_counts": dict(sorted(rss_sources.items())),
            "reserved_tasks_read": 0,
            "candidate_envelopes": envelope_evidence,
        },
        "arms": arms,
        "comparisons_vs_serial8": comparisons,
        "gates": gates,
        "go_to_foreground_feedback_feasibility": phase_go,
        "cost": {
            "prediction_time_agent_calls": 0,
            "gpu_runtime_s": 0.0,
            "evaluator_wall_s": time.monotonic() - started,
        },
        "limitations": [
            "All 15 scored PennyLane tasks are development-exposed.",
            "Candidate envelopes use fit tasks only, but foreground future shape and RSS safety remain hindsight oracles.",
            "The action-space ceiling contains one physically defined arrival wave, not workload-order uncertainty.",
            "The deterministic replay preserves recorded commands but models CPU sharing.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = _evaluate()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"output": str(args.out), "status": result["status"], "gates": result["gates"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
