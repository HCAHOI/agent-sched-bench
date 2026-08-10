#!/usr/bin/env python3
"""Evaluate static command-survival KV actions against five-second feedback."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Sequence
import datetime as dt
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.evaluation.evaluate_clause_resource_classes import load_rows  # noqa: E402
from scripts.evaluation.evaluate_gpu_tool_gap_actions import (  # noqa: E402
    _comparison,
    _summarize,
    score_gap_action,
)
from spike.multitenant import load_trace_programs  # noqa: E402
from tool_resource.clause_parser import parse_command_clauses  # noqa: E402
from tool_resource.runtime_kb import (  # noqa: E402
    CANONICAL_LATENCY_BUCKETS,
    ClauseResourceKB,
)
from tool_resource_eval.labels import repo_of  # noqa: E402


_DEFAULT_TASK_ROWS = Path(
    "analysis/results/tool-resource-5-3-3-3-20260804/"
    "swe100-277-cpu-feedback-generality-v1/task_rows.jsonl"
)
_DEFAULT_FIT = Path(
    "traces/swe-rebench/qwen3.7-max/swe100-full-5be74da-20260726/"
    "simulate_cloud_model_c2_20260726T005356962.jsonl"
)
_DEFAULT_EVAL = Path(
    "traces/swe-rebench/qwen3.7-max/swe277-full-5be74da-20260726/"
    "simulate_cloud_model_c2_20260726T024552768.jsonl"
)
_DEFAULT_SOURCE = Path(
    "traces/swe-rebench/qwen3.7-max/fresh-seed42-skip150-n200"
)
_DEFAULT_IDS = Path(
    "analysis/serving/w5-multitenant/inputs/"
    "swe-rebench-fresh277-task-ids.txt"
)
_DEFAULT_GPU_GAPS = Path(
    "analysis/results/gpu-tool-gap-actions-a100-instruct-20260809/result.json"
)
_ARMS = (
    "deadline_feedback",
    "hard_b4",
    "survival_majority",
    "expected_pareto",
    "exec_survival_oracle",
)


def command_latency_evidence(
    kb: ClauseResourceKB,
    repo: str,
    command: str,
    *,
    query_ts: float,
) -> dict[str, Any] | None:
    """Return the selected causal duration samples for one command."""

    result = kb.predict_command_latency_bucket(
        repo,
        command,
        query_ts,
        CANONICAL_LATENCY_BUCKETS,
    )
    prediction = result.prediction
    if prediction is None:
        return None
    parsed = parse_command_clauses(command)
    clauses = parsed["clauses"]
    if parsed["parse_failed"] or not clauses:
        return None
    if len(clauses) == 1:
        clause = clauses[0]
        selected = kb._select(  # noqa: SLF001 - evaluator needs raw causal evidence
            repo,
            "latency_ms",
            str(clause["bin"]),
            tuple(str(value) for value in clause["argv"]),
        )
        values = None if selected is None else selected[0]
    else:
        composed = kb._composed_command_values(  # noqa: SLF001
            repo,
            command,
            clauses,
            "latency_ms",
        )
        values = None if composed is None else composed[0]
    if values is None:
        return None
    return {
        "duration_samples_ms": tuple(float(value) for value in values),
        "scope": prediction.scope,
        "key_kind": prediction.key_kind,
        "evidence_count": prediction.evidence_count,
        "hard_bucket": max(
            range(len(prediction.probability_by_bucket)),
            key=prediction.probability_by_bucket.__getitem__,
        ),
    }


def expected_early_action(
    duration_samples_ms: Sequence[float],
    *,
    deadline_ms: float,
    size_gib: float,
    swap_out_ms: float,
    swap_in_ms: float,
) -> dict[str, Any]:
    """Return the history-only Pareto decision for an immediate KV offload."""

    values = tuple(float(value) for value in duration_samples_ms)
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("duration samples must be finite and non-negative")
    physical = (deadline_ms, size_gib, swap_out_ms, swap_in_ms)
    if any(not math.isfinite(value) or value <= 0.0 for value in physical):
        raise ValueError("deadline and transfer costs must be finite and positive")

    def score(duration_ms: float, trigger_ms: float) -> tuple[float, float]:
        if trigger_ms >= duration_ms:
            return 0.0, 0.0
        complete_ms = trigger_ms + swap_out_ms
        released_gib_s = size_gib * max(0.0, duration_ms - complete_ms) / 1000.0
        stall_ms = max(0.0, complete_ms - duration_ms) + swap_in_ms
        return released_gib_s, stall_ms

    deltas = [
        tuple(early - feedback for early, feedback in zip(score(value, 0.0), score(value, deadline_ms), strict=True))
        for value in values
    ]
    released_delta = math.fsum(row[0] for row in deltas) / len(deltas)
    stall_delta = math.fsum(row[1] for row in deltas) / len(deltas)
    return {
        "probability_survives_deadline": sum(value > deadline_ms for value in values)
        / len(values),
        "expected_released_delta_gib_s": released_delta,
        "expected_stall_delta_ms": stall_delta,
        "evidence_count": len(values),
        "act_early": released_delta > 0.0 and stall_delta <= 0.0,
    }


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _valid_task_ids(path: Path) -> dict[str, set[str]]:
    valid: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            valid[str(row["corpus"])].add(str(row["task_id"]))
    if {name: len(ids) for name, ids in valid.items()} != {
        "swe100": 82,
        "swe277": 177,
    }:
        raise ValueError("the frozen telemetry-valid task population changed")
    return valid


def _ordered_ids(path: Path, selected: set[str]) -> list[str]:
    ordered = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(ordered) != len(set(ordered)) or set(ordered) & selected != selected:
        raise ValueError("task ID file is duplicate or incomplete")
    selected_order = [task_id for task_id in ordered if task_id in selected]
    frozen = [
        line
        for line in _DEFAULT_IDS.read_text(encoding="utf-8").splitlines()
        if line in selected
    ]
    if selected_order != frozen:
        raise ValueError("task IDs differ from the frozen replay order")
    return selected_order


def _physical(gap: dict[str, Any], *, bytes_per_token: int) -> dict[str, float]:
    point = gap["transfer_point"]
    retained_tokens = int(gap["retained_tokens"])
    return {
        "size_gib": retained_tokens * bytes_per_token / 2**30,
        "swap_out_ms": float(point["swap_out_ms"]),
        "swap_in_ms": float(point["swap_in_ms"]),
    }


def _same_score(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(
        left[key] == right[key]
        if isinstance(left[key], (bool, int)) or left[key] is None
        else math.isclose(float(left[key]), float(right[key]), abs_tol=1e-9)
        for key in left
    )


def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    valid = _valid_task_ids(args.task_rows)
    ordered_ids = _ordered_ids(args.task_ids, valid["swe277"])
    fit_rows = [
        row
        for row in load_rows(args.fit_telemetry)
        if row.task_id in valid["swe100"]
        and row.structure_known
        and row.pipeline_position <= 0
    ]
    eval_rows = [
        row
        for row in load_rows(args.eval_telemetry)
        if row.task_id in valid["swe277"]
        and row.structure_known
        and row.pipeline_position <= 0
    ]
    if len(fit_rows) != 2605 or len(eval_rows) != 6260:
        raise ValueError("the frozen causal clause population changed")
    fit_evidence_tasks = {row.task_id for row in fit_rows}
    if not fit_evidence_tasks <= valid["swe100"] or len(fit_evidence_tasks) != 81:
        raise ValueError("the frozen fit evidence task coverage changed")
    if {row.task_id for row in fit_rows} & {row.task_id for row in eval_rows}:
        raise ValueError("fit and replay tasks overlap")

    eval_by_task: dict[str, list[Any]] = defaultdict(list)
    for row in eval_rows:
        eval_by_task[row.task_id].append(row)
    missing_local = set(ordered_ids) - set(eval_by_task)
    if missing_local:
        raise ValueError(f"replay tasks lack clause telemetry: {sorted(missing_local)[:3]}")

    programs = load_trace_programs(args.source_root, task_ids=ordered_ids, seed=42)
    source_by_task = {program.task_id: program for program in programs}
    gpu_payload = json.loads(args.gpu_gaps.read_text(encoding="utf-8"))
    deadline_ms = float(gpu_payload["config"]["deadline_ms"])
    if deadline_ms != 5000.0:
        raise ValueError("the frozen feedback deadline changed")
    profile_payload = json.loads(args.kv_profile.read_text(encoding="utf-8"))
    bytes_per_token = int(profile_payload["bytes_per_token"])
    gap_by_key = {
        (str(gap["task_id"]), int(gap["turn_index"])): gap
        for gap in gpu_payload["gaps"]
        if gap["task_id"] in valid["swe277"]
    }
    if len(gap_by_key) != 8273:
        raise ValueError("the frozen tool-gap population changed")

    repos = sorted({repo_of(task_id) for task_id in ordered_ids})
    kbs = {
        repo: ClauseResourceKB.fit_public(
            row.observation(0.0, 1.0) for row in fit_rows if row.repo != repo
        )
        for repo in repos
    }
    counters: Counter[str] = Counter()
    provenance: Counter[str] = Counter()
    gap_rows: list[dict[str, Any]] = []
    decision_examples: list[dict[str, Any]] = []

    for ordinal, task_id in enumerate(ordered_ids):
        program = source_by_task[task_id]
        repo = repo_of(task_id)
        kb = kbs[repo]
        query_ts = float(ordinal * 2 + 1)
        gap_turn_count = len(program.turns) - 1 + bool(program.omitted_terminal_llm_calls)
        for turn_index, turn in enumerate(program.turns[:gap_turn_count]):
            gap = gap_by_key.pop((task_id, turn_index))
            if len(turn.tools) != int(gap["tool_count"]) or not math.isclose(
                turn.gap_ms, float(gap["gap_ms"]), abs_tol=1e-9
            ):
                raise ValueError(f"source/GPU gap mismatch at {task_id}:{turn_index}")
            physical = _physical(gap, bytes_per_token=bytes_per_token)
            triggers = {arm: [] for arm in _ARMS}
            decisions: list[dict[str, Any]] = []
            for tool in turn.tools:
                for arm in _ARMS:
                    triggers[arm].append(deadline_ms)
                if tool.tool_name != "exec":
                    counters["non_exec_gap_tools"] += 1
                    continue
                counters["exec_gap_tools"] += 1
                evidence = command_latency_evidence(
                    kb,
                    repo,
                    tool.command,
                    query_ts=query_ts,
                )
                duration_ms = tool.end_offset_ms - tool.start_offset_ms
                if duration_ms > deadline_ms:
                    triggers["exec_survival_oracle"][-1] = 0.0
                    counters["exec_survival_oracle_early"] += 1
                if evidence is None:
                    counters["exec_without_evidence"] += 1
                    continue
                counters["exec_with_evidence"] += 1
                provenance[f"{evidence['scope']}:{evidence['key_kind']}"] += 1
                action = expected_early_action(
                    evidence["duration_samples_ms"],
                    deadline_ms=deadline_ms,
                    **physical,
                )
                if evidence["hard_bucket"] == 4:
                    triggers["hard_b4"][-1] = 0.0
                    counters["hard_b4_early"] += 1
                if action["probability_survives_deadline"] > 0.5:
                    triggers["survival_majority"][-1] = 0.0
                    counters["survival_majority_early"] += 1
                if action["act_early"]:
                    triggers["expected_pareto"][-1] = 0.0
                    counters["expected_pareto_early"] += 1
                if len(decision_examples) < 50 and (
                    action["act_early"] or evidence["hard_bucket"] == 4
                ):
                    decisions.append(
                        {
                            "command": tool.command,
                            "duration_ms": duration_ms,
                            "scope": evidence["scope"],
                            "key_kind": evidence["key_kind"],
                            "hard_bucket": evidence["hard_bucket"],
                            **action,
                        }
                    )
            no_prerestore = (None,) * len(turn.tools)
            arms = {
                arm: score_gap_action(
                    gap_ms=turn.gap_ms,
                    tools=turn.tools,
                    triggers_ms=arm_triggers,
                    prerestore_starts_ms=no_prerestore,
                    **physical,
                )
                for arm, arm_triggers in triggers.items()
            }
            if not _same_score(arms["deadline_feedback"], gap["arms"]["deadline_reactive"]):
                raise ValueError(f"feedback score mismatch at {task_id}:{turn_index}")
            gap_rows.append(
                {
                    "task_id": task_id,
                    "turn_index": turn_index,
                    "tool_families": gap["tool_families"],
                    "arms": arms,
                }
            )
            for decision in decisions:
                if len(decision_examples) < 50:
                    decision_examples.append(
                        {"task_id": task_id, "turn_index": turn_index, **decision}
                    )
        settle_ts = query_ts + 1.0
        for row in eval_by_task[task_id]:
            kb.observe_completed_clause(row.observation(query_ts, settle_ts))
    if gap_by_key:
        raise ValueError(f"unscored GPU gaps remain: {len(gap_by_key)}")
    if counters["exec_gap_tools"] != 5799:
        raise ValueError("the frozen exec action population changed")

    summaries = {arm: _summarize(gap_rows, arm) for arm in _ARMS}
    comparisons = {
        arm: _comparison(gap_rows, arm, "deadline_feedback")
        for arm in _ARMS
        if arm != "deadline_feedback"
    }
    primary = comparisons["expected_pareto"]
    checks = {
        "released_gib_s_strictly_higher": primary["released_gib_s_delta"] > 0.0,
        "critical_path_stall_not_higher": primary["critical_path_stall_ms_delta"] <= 0.0,
        "changed_at_least_20_tasks": primary["changed_task_count"] >= 20,
    }
    go = all(checks.values())
    return {
        "schema_version": 1,
        "status": "development_go" if go else "development_no_go",
        "generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "protocol": (
            "Frozen CPU-only action-specific survival evaluation on exposed SWE "
            "traces; no canonical predictor change and no runtime agent"
        ),
        "config": {
            "task_rows": str(args.task_rows.resolve()),
            "fit_telemetry": str(args.fit_telemetry.resolve()),
            "eval_telemetry": str(args.eval_telemetry.resolve()),
            "source_root": str(args.source_root.resolve()),
            "task_ids": str(args.task_ids.resolve()),
            "gpu_gaps": str(args.gpu_gaps.resolve()),
            "kv_profile": str(args.kv_profile.resolve()),
            "deadline_ms": deadline_ms,
            "bytes_per_token": bytes_per_token,
            "task_update": "whole-task-final only; strictly before next task",
            "primary_arm": "expected_pareto",
            "early_decision": (
                "expected released GiB*s delta > 0 and expected critical-path "
                "stall delta <= 0"
            ),
        },
        "evidence": {
            "fit_population_task_count": len(valid["swe100"]),
            "fit_tasks_with_eligible_clauses": len(fit_evidence_tasks),
            "fit_tasks_without_eligible_clauses": sorted(
                valid["swe100"] - fit_evidence_tasks
            ),
            "fit_clause_count": len(fit_rows),
            "replay_task_count": len(ordered_ids),
            "replay_clause_count": len(eval_rows),
            "gap_count": len(gap_rows),
            **dict(sorted(counters.items())),
            "selected_evidence_provenance": dict(sorted(provenance.items())),
        },
        "arms": summaries,
        "comparisons_vs_deadline_feedback": comparisons,
        "gate": {"checks": checks, "passed": go},
        "decision_examples": decision_examples,
        "cost": {
            "prediction_time_agent_calls": 0,
            "gpu_runtime_s": 0.0,
            "evaluator_wall_s": time.monotonic() - started,
        },
        "limitations": [
            "All SWE100/277 evidence is development-exposed; this is not confirmation.",
            "The action model reuses measured A100 transfer costs and recorded gaps; it is not a live serving run.",
            "Timeout is retained in the source trace but is not used as remaining-work evidence.",
            "The survival oracle uses realized exec duration and is an upper bound only.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-rows", type=Path, default=_DEFAULT_TASK_ROWS)
    parser.add_argument("--fit-telemetry", type=Path, default=_DEFAULT_FIT)
    parser.add_argument("--eval-telemetry", type=Path, default=_DEFAULT_EVAL)
    parser.add_argument("--source-root", type=Path, default=_DEFAULT_SOURCE)
    parser.add_argument("--task-ids", type=Path, default=_DEFAULT_IDS)
    parser.add_argument("--gpu-gaps", type=Path, default=_DEFAULT_GPU_GAPS)
    parser.add_argument(
        "--kv-profile",
        type=Path,
        default=Path(
            "analysis/serving/tool-time-rho-measurement-a100-instruct-20260809/"
            "kv_swap.json"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = _evaluate(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
