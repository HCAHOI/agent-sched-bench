#!/usr/bin/env python3
"""Test whether exact recurrence advances the first physical KV loan."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import datetime as dt
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from scripts.evaluation.evaluate_gpu_tool_gap_actions import (  # noqa: E402
    _load_transfer_points,
    choose_transfer_point,
)
from scripts.evaluation.evaluate_static_survival_gap_action import (  # noqa: E402
    expected_early_action,
)
from scripts.evaluation.evaluate_survival_work_state_action import (  # noqa: E402
    CausalGroupedDurationMemory,
    _program_context,
    command_work_key,
)
from scripts.evaluation.evaluate_task_pareto_survival_action import (  # noqa: E402
    _DEFAULT_KV_PROFILE,
    _DEFAULT_SPLIT,
    _load_split,
    _load_tokenizer,
    _retained_tokens,
)
from spike.multitenant import TraceProgram  # noqa: E402
from tool_resource_eval.labels import repo_of  # noqa: E402
from trace_collect.trace_data import TraceData  # noqa: E402


_PROTOCOL_GIT_SHA = "a8bfc62be741d2a2a5ab4e8ac65d5ca0d825ba63"
_DEADLINE_MS = 5_000.0
_MINIMUM_CHANGED_TASKS = 8
_BYTES_PER_GIB = 2**30


@dataclass(frozen=True)
class ToolTimeline:
    turn_index: int
    tool_index: int
    command: str
    start_ms: float
    duration_ms: float
    gap_end_ms: float
    swap_out_ms: float
    swap_in_ms: float
    size_gib: float
    retained_tokens: int


def first_emitted_loan(
    tools: Sequence[ToolTimeline], triggers_ms: Sequence[float]
) -> dict[str, Any] | None:
    """Return the first emitted loan and when its memory becomes usable."""
    if len(tools) != len(triggers_ms):
        raise ValueError("tools and triggers must align")
    candidates: list[tuple[float, int, int, ToolTimeline, float]] = []
    for tool, trigger_ms in zip(tools, triggers_ms, strict=True):
        if not math.isfinite(trigger_ms) or trigger_ms < 0.0:
            raise ValueError("loan triggers must be finite and non-negative")
        if trigger_ms >= tool.duration_ms:
            continue
        decision_ms = tool.start_ms + trigger_ms
        candidates.append(
            (decision_ms, tool.turn_index, tool.tool_index, tool, trigger_ms)
        )
    if not candidates:
        return None
    decision_ms, _turn, _index, tool, trigger_ms = min(candidates)
    usable_ms = decision_ms + tool.swap_out_ms
    return {
        **asdict(tool),
        "trigger_ms": trigger_ms,
        "decision_ms": decision_ms,
        "usable_ms": usable_ms,
        "critical_path_stall_ms": (
            max(0.0, usable_ms - tool.gap_end_ms) + tool.swap_in_ms
        ),
    }


def admission_opportunity_ms(
    loan: dict[str, Any] | None, task_terminal_ms: float
) -> float:
    """Return whichever releases capacity first: the loan or task completion."""
    return task_terminal_ms if loan is None else min(float(loan["usable_ms"]), task_terminal_ms)


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


def _require_frozen(path: Path) -> None:
    repo_path = path.resolve().relative_to(_ROOT).as_posix()
    expected = subprocess.run(
        ["git", "show", f"{_PROTOCOL_GIT_SHA}:{repo_path}"],
        check=True,
        capture_output=True,
    ).stdout
    if path.read_bytes() != expected:
        raise ValueError(f"frozen input changed: {repo_path}")


def _timeline_origin_and_ends(program: TraceProgram) -> tuple[float, tuple[float, ...], float]:
    trace = TraceData.load(Path(program.trace_path))
    calls = sorted(
        (row for row in trace.actions if row.get("action_type") == "llm_call"),
        key=lambda row: float(row["ts_start"]),
    )
    if len(calls) < len(program.turns):
        raise ValueError(f"{program.task_id}: source LLM calls disappeared")
    origin = float(calls[0]["ts_start"])
    ends = tuple((float(call["ts_end"]) - origin) * 1_000.0 for call in calls[: len(program.turns)])
    terminal_ms = max(float(row["ts_end"]) for row in trace.actions) - origin
    return origin, ends, terminal_ms * 1_000.0


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _evaluate() -> dict[str, Any]:
    started = time.monotonic()
    git_sha = _require_clean_checkout()
    _require_frozen(_DEFAULT_SPLIT)
    _require_frozen(_DEFAULT_KV_PROFILE)
    fit, replay = _load_split(_DEFAULT_SPLIT)
    tokenizer = _load_tokenizer()
    profile = json.loads(_DEFAULT_KV_PROFILE.read_text(encoding="utf-8"))
    bytes_per_token = int(profile["bytes_per_token"])
    points = _load_transfer_points(_DEFAULT_KV_PROFILE)

    memory = CausalGroupedDurationMemory(())
    for program in fit:
        memory.observe_task(program.task_id, _program_context(program).observations)

    first_response_tokens = {
        program.task_id: _retained_tokens(program.turns[0], tokenizer)
        for program in replay
    }
    task_rows: list[dict[str, Any]] = []
    coverage: dict[str, list[str]] = {}
    for program in replay:
        _origin, llm_end_ms, terminal_ms = _timeline_origin_and_ends(program)
        tools: list[ToolTimeline] = []
        candidate_triggers: list[float] = []
        exact_metadata: dict[tuple[int, int], dict[str, int]] = {}
        for turn_index, turn in enumerate(program.turns):
            retained_tokens = _retained_tokens(turn, tokenizer)
            if retained_tokens <= 0:
                continue
            point = choose_transfer_point(points, retained_tokens)
            size_gib = retained_tokens * bytes_per_token / _BYTES_PER_GIB
            gap_end_ms = llm_end_ms[turn_index] + turn.gap_ms
            for tool_index, tool in enumerate(turn.tools):
                if tool.tool_name != "exec":
                    continue
                timeline = ToolTimeline(
                    turn_index,
                    tool_index,
                    tool.command,
                    llm_end_ms[turn_index] + tool.start_offset_ms,
                    tool.end_offset_ms - tool.start_offset_ms,
                    gap_end_ms,
                    point.swap_out_ms,
                    point.swap_in_ms,
                    size_gib,
                    retained_tokens,
                )
                grouped = memory.query(
                    repo_of(program.task_id),
                    tool.command,
                    command_work_key(tool.command),
                )["exact_command"]
                values = tuple(
                    value for task_values in grouped.values() for value in task_values
                )
                trigger_ms = _DEADLINE_MS
                if values and expected_early_action(
                    values,
                    deadline_ms=_DEADLINE_MS,
                    size_gib=size_gib,
                    swap_out_ms=point.swap_out_ms,
                    swap_in_ms=point.swap_in_ms,
                )["act_early"]:
                    trigger_ms = 0.0
                tools.append(timeline)
                candidate_triggers.append(trigger_ms)
                exact_metadata[(turn_index, tool_index)] = {
                    "history_task_count": len(grouped),
                    "history_count": len(values),
                }

        feedback = first_emitted_loan(tools, (_DEADLINE_MS,) * len(tools))
        candidate = first_emitted_loan(tools, candidate_triggers)
        feedback_admission_ms = admission_opportunity_ms(feedback, terminal_ms)
        candidate_admission_ms = admission_opportunity_ms(candidate, terminal_ms)
        advance_ms = feedback_admission_ms - candidate_admission_ms
        changed = advance_ms > 1e-9
        if candidate is not None:
            candidate.update(
                exact_metadata[(candidate["turn_index"], candidate["tool_index"])]
            )
        row = {
            "task_id": program.task_id,
            "task_terminal_ms": terminal_ms,
            "feedback_first_loan": feedback,
            "exact_immediate_feedback_first_loan": candidate,
            "feedback_admission_opportunity_ms": feedback_admission_ms,
            "candidate_admission_opportunity_ms": candidate_admission_ms,
            "admission_advance_ms": advance_ms,
            "advanced": changed,
        }
        task_rows.append(row)
        if changed and candidate is not None:
            coverage[program.task_id] = [
                borrower.task_id
                for borrower in replay
                if borrower.task_id != program.task_id
                and first_response_tokens[borrower.task_id]
                <= int(candidate["retained_tokens"])
            ]

    changed_rows = [row for row in task_rows if row["advanced"]]
    advances = [float(row["admission_advance_ms"]) for row in changed_rows]
    feedback_stall = math.fsum(
        float(row["feedback_first_loan"]["critical_path_stall_ms"])
        for row in task_rows
        if row["feedback_first_loan"] is not None
    )
    candidate_stall = math.fsum(
        float(
            row["exact_immediate_feedback_first_loan"]["critical_path_stall_ms"]
        )
        for row in task_rows
        if row["exact_immediate_feedback_first_loan"] is not None
    )
    checks = {
        "advanced_at_least_8_tasks": len(changed_rows) >= _MINIMUM_CHANGED_TASKS,
        "aggregate_stall_not_higher": candidate_stall <= feedback_stall + 1e-9,
    }
    passed = all(checks.values())
    pair_count = len(changed_rows) * (len(replay) - 1)
    coverage_edges = sum(len(ids) for ids in coverage.values())
    return {
        "schema_version": 1,
        "status": (
            "development_go_to_fresh_live"
            if passed
            else "development_no_go_first_loan"
        ),
        "generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_sha": git_sha,
        "protocol_git_sha": _PROTOCOL_GIT_SHA,
        "protocol": "tool-resource-canonical-objective.md Section 5.12",
        "config": {
            "fit_task_count": len(fit),
            "replay_task_count": len(replay),
            "history_update": "fixed fit only; no replay-task updates",
            "matching": "complete raw exact command; no fallback",
            "deadline_ms": _DEADLINE_MS,
            "one_loan_per_task": True,
            "loan_available_after": "measured A100 swap-out completion",
        },
        "comparison": {
            "changed_task_count": len(changed_rows),
            "changed_task_ids": [row["task_id"] for row in changed_rows],
            "total_admission_advance_ms_upper_bound": math.fsum(advances),
            "median_admission_advance_ms": (
                statistics.median(advances) if advances else 0.0
            ),
            "p95_admission_advance_ms": _percentile(advances, 95.0),
            "feedback_critical_path_stall_ms": feedback_stall,
            "candidate_critical_path_stall_ms": candidate_stall,
            "critical_path_stall_delta_ms": candidate_stall - feedback_stall,
        },
        "borrower_capacity": {
            "first_response_retained_tokens": first_response_tokens,
            "fit_task_ids_by_changed_lender": coverage,
            "pairwise_fit_edge_count": coverage_edges,
            "pairwise_possible_edge_count": pair_count,
            "pairwise_fit_fraction": (
                coverage_edges / pair_count if pair_count else 0.0
            ),
        },
        "gate": {"checks": checks, "passed": passed},
        "task_rows": task_rows,
        "cost": {
            "prediction_time_agent_calls": 0,
            "gpu_runtime_s": 0.0,
            "evaluator_wall_s": time.monotonic() - started,
        },
        "limitations": [
            "All 41 PennyLane inputs are development-exposed.",
            "Admission advance assumes an always-nonempty queue and is not a completion-time measurement.",
            "Pairwise KV fit does not model concurrent decode growth or GPU compute contention.",
            "Recorded task terminal and gap trajectories are replayed rather than live serving.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = _evaluate()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), "status": result["status"], "gate": result["gate"]}, indent=2))


if __name__ == "__main__":
    main()
