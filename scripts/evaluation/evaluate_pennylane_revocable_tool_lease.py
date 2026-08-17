#!/usr/bin/env python3
"""Screen revocable tool-phase leases on the exposed PennyLane trajectories."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import subprocess
from typing import Any, Literal, Sequence


_ROOT = Path(__file__).resolve().parents[2]
_RUN_DIR = (
    _ROOT
    / "traces/swe-rebench/gpt-5.6-sol/"
    "pennylane-all76-clean-ebpf-20260816"
)
_OUTPUT = (
    _ROOT
    / "analysis/results/pennylane-revocable-tool-lease-development-v1/result.json"
)
_TASK_IDS = (
    "PennyLaneAI__pennylane-2603",
    "PennyLaneAI__pennylane-2654",
    "PennyLaneAI__pennylane-2668",
    "PennyLaneAI__pennylane-2834",
    "PennyLaneAI__pennylane-2947",
    "PennyLaneAI__pennylane-2964",
    "PennyLaneAI__pennylane-3024",
    "PennyLaneAI__pennylane-3033",
)
_BASE_CONCURRENCY = 4
_FEEDBACK_BUDGET_S = 34.790656
_EPS = 1e-8
Arm = Literal["fixed4", "permanent_loan", "revocable_lease"]


@dataclass(frozen=True)
class Action:
    duration_s: float
    kind: Literal["llm", "exec", "other", "delay"]
    prompt_tokens: int = 0


@dataclass(frozen=True)
class Program:
    task_id: str
    actions: tuple[Action, ...]


@dataclass
class _State:
    program: Program
    rank: int
    base: bool
    lender: str | None = None
    index: int = 0
    finish_s: float | None = None
    trigger_s: float | None = None

    @property
    def action(self) -> Action:
        return self.program.actions[self.index]


def _git_sha() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("evaluation requires a clean committed checkout")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_program(task_id: str) -> Program:
    path = _RUN_DIR / task_id / "attempt_1/trace.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    raw_actions = [row for row in rows if row.get("type") == "action"]
    raw_actions.sort(key=lambda row: (float(row["ts_start"]), float(row["ts_end"])))
    if not raw_actions:
        raise ValueError(f"{task_id}: no actions")

    actions: list[Action] = []
    previous_end = float(raw_actions[0]["ts_start"])
    for row in raw_actions:
        start = float(row["ts_start"])
        end = float(row["ts_end"])
        if not all(math.isfinite(value) for value in (start, end)) or end < start:
            raise ValueError(f"{task_id}: invalid action interval")
        if start < previous_end - 1e-6:
            raise ValueError(f"{task_id}: overlapping actions")
        if start > previous_end + _EPS:
            actions.append(Action(start - previous_end, "delay"))
        data = row.get("data") if isinstance(row.get("data"), dict) else {}
        if row.get("action_type") == "llm_call":
            prompt_tokens = data.get("prompt_tokens")
            if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool):
                raise ValueError(f"{task_id}: LLM action lacks prompt tokens")
            kind = "llm"
        elif row.get("action_type") == "tool_exec" and data.get("tool_name") == "exec":
            kind = "exec"
            prompt_tokens = 0
        else:
            kind = "other"
            prompt_tokens = 0
        actions.append(Action(end - start, kind, prompt_tokens))
        previous_end = end
    return Program(task_id, tuple(actions))


def simulate(
    programs: Sequence[Program],
    arm: Arm,
    *,
    base_concurrency: int = _BASE_CONCURRENCY,
    feedback_budget_s: float = _FEEDBACK_BUDGET_S,
) -> dict[str, Any]:
    if arm not in {"fixed4", "permanent_loan", "revocable_lease"}:
        raise ValueError(f"unknown arm: {arm}")
    if len(programs) <= base_concurrency or feedback_budget_s <= 0:
        raise ValueError("screen requires waiting tasks and a positive feedback budget")

    pending = list(enumerate(programs))
    states: list[_State] = []
    foreground_ids = {program.task_id for program in programs[:base_concurrency]}
    loaned: dict[str, _State] = {}
    active_leases: set[str] = set()
    completions: dict[str, float] = {}
    now = 0.0
    admissions = 0
    feedback_triggers = 0
    promotions = 0
    paused_borrower_s = 0.0
    request_overlap_s = 0.0
    prompt_overlap_token_s = 0.0
    peak_llm_requests = 0
    peak_prompt_tokens = 0
    deadlock: dict[str, Any] | None = None

    def admit(*, base: bool, lender: str | None = None) -> _State | None:
        nonlocal admissions
        if not pending:
            return None
        rank, program = pending.pop(0)
        state = _State(program, rank, base=base, lender=lender)
        states.append(state)
        admissions += 1
        return state

    for _ in range(base_concurrency):
        admit(base=True)

    def allowed(state: _State) -> bool:
        if arm == "permanent_loan":
            return True
        return state.base or (
            state.lender is not None and state.lender in active_leases
        )

    def restore_base_capacity() -> None:
        nonlocal promotions
        while sum(state.base for state in states) < base_concurrency:
            if arm == "revocable_lease":
                borrower = min(
                    (state for state in states if not state.base),
                    key=lambda state: state.rank,
                    default=None,
                )
                if borrower is not None:
                    borrower.base = True
                    promotions += 1
                    continue
            if admit(base=True) is None:
                break

    def start_ready() -> None:
        for state in sorted(states, key=lambda item: item.rank):
            if state.finish_s is not None or not allowed(state):
                continue
            action = state.action
            state.finish_s = now + action.duration_s
            state.trigger_s = None
            if (
                action.kind == "exec"
                and state.program.task_id in foreground_ids
                and action.duration_s > feedback_budget_s
                and (arm == "revocable_lease" or state.program.task_id not in loaned)
            ):
                state.trigger_s = now + feedback_budget_s

    while states or pending:
        restore_base_capacity()
        start_ready()
        event_times = [
            value
            for state in states
            for value in (state.finish_s, state.trigger_s)
            if value is not None
        ]
        if not event_times:
            deadlock = {
                "time_s": now,
                "pending": len(pending),
                "paused_tasks": [state.program.task_id for state in states],
            }
            break
        next_time = min(event_times)
        duration = next_time - now
        if duration < -_EPS or not math.isfinite(duration):
            raise ValueError("simulation made invalid temporal progress")

        running_llm = [
            state for state in states if state.finish_s is not None and state.action.kind == "llm"
        ]
        prompts = [state.action.prompt_tokens for state in running_llm]
        peak_llm_requests = max(peak_llm_requests, len(running_llm))
        peak_prompt_tokens = max(peak_prompt_tokens, sum(prompts))
        if len(running_llm) > 1:
            request_overlap_s += duration * (len(running_llm) - 1)
            prompt_overlap_token_s += duration * (sum(prompts) - max(prompts))
        if arm == "revocable_lease":
            paused_borrower_s += duration * sum(
                not state.base and state.finish_s is None and not allowed(state)
                for state in states
            )
        now = next_time

        for state in list(states):
            if state.trigger_s is None or state.trigger_s > now + _EPS:
                continue
            state.trigger_s = None
            lender = state.program.task_id
            borrower = loaned.get(lender)
            if borrower is None:
                borrower = admit(base=False, lender=lender)
                if borrower is not None:
                    loaned[lender] = borrower
            if borrower is not None:
                feedback_triggers += 1
                if arm == "revocable_lease":
                    active_leases.add(lender)

        base_finished = False
        for state in list(states):
            if state.finish_s is None or state.finish_s > now + _EPS:
                continue
            completed_action = state.action
            state.finish_s = None
            state.trigger_s = None
            if completed_action.kind == "exec":
                active_leases.discard(state.program.task_id)
            state.index += 1
            if state.index == len(state.program.actions):
                completions[state.program.task_id] = now
                base_finished |= state.base
                states.remove(state)
        if base_finished:
            restore_base_capacity()

    complete = len(completions) == len(programs)
    completion_values = list(completions.values())
    return {
        "completed": complete,
        "completed_tasks": len(completions),
        "mean_task_completion_s": statistics.fmean(completion_values) if complete else None,
        "makespan_s": max(completion_values) if complete else None,
        "feedback_trigger_count": feedback_triggers,
        "admitted_tasks": admissions,
        "borrower_promotions": promotions,
        "paused_borrower_s": paused_borrower_s,
        "llm_request_overlap_s": request_overlap_s,
        "prompt_overlap_token_s": prompt_overlap_token_s,
        "peak_simultaneous_llm_requests": peak_llm_requests,
        "peak_simultaneous_prompt_tokens": peak_prompt_tokens,
        "deadlock": deadlock,
        "task_completion_s": dict(sorted(completions.items())),
    }


def _reduction(candidate: float, baseline: float) -> float:
    return 1.0 - candidate / baseline


def evaluate(git_sha: str) -> dict[str, Any]:
    programs = [_load_program(task_id) for task_id in _TASK_IDS]
    arms = {
        arm: simulate(programs, arm)
        for arm in ("fixed4", "permanent_loan", "revocable_lease")
    }
    fixed = arms["fixed4"]
    permanent = arms["permanent_loan"]
    lease = arms["revocable_lease"]
    fixed_mean = float(fixed["mean_task_completion_s"])
    permanent_mean = float(permanent["mean_task_completion_s"])
    lease_mean = float(lease["mean_task_completion_s"])
    permanent_gain = fixed_mean - permanent_mean
    retained_gain = (
        (fixed_mean - lease_mean) / permanent_gain if permanent_gain > 0 else -math.inf
    )
    permanent_overlap = float(permanent["llm_request_overlap_s"])
    overlap_reduction = (
        _reduction(float(lease["llm_request_overlap_s"]), permanent_overlap)
        if permanent_overlap > 0
        else -math.inf
    )
    checks = {
        "lease_completed_without_deadlock": lease["completed"] and lease["deadlock"] is None,
        "lease_mean_completion_at_least_5pct_below_fixed4": _reduction(lease_mean, fixed_mean) >= 0.05,
        "lease_makespan_strictly_below_fixed4": float(lease["makespan_s"]) < float(fixed["makespan_s"]),
        "lease_retains_at_least_80pct_of_permanent_gain": retained_gain >= 0.80,
        "lease_reduces_request_overlap_at_least_25pct": overlap_reduction >= 0.25,
        "lease_peak_prompt_tokens_no_higher_than_permanent": int(lease["peak_simultaneous_prompt_tokens"]) <= int(permanent["peak_simultaneous_prompt_tokens"]),
    }
    status = "go" if all(checks.values()) else "no_go"
    return {
        "schema": "pennylane-revocable-tool-lease-development-v1",
        "status": status,
        "corpus_role": "development_exposed_actionability_screen",
        "git_sha": git_sha,
        "input": {
            "run_dir": _RUN_DIR.relative_to(_ROOT).as_posix(),
            "task_ids": list(_TASK_IDS),
            "task_count": len(programs),
            "action_count": sum(
                action.kind != "delay" for program in programs for action in program.actions
            ),
            "llm_action_count": sum(
                action.kind == "llm" for program in programs for action in program.actions
            ),
            "exec_action_count": sum(
                action.kind == "exec" for program in programs for action in program.actions
            ),
        },
        "configuration": {
            "base_task_concurrency": _BASE_CONCURRENCY,
            "feedback_budget_s": _FEEDBACK_BUDGET_S,
            "lease_boundary": "finish_inflight_action_then_pause",
            "prediction_time_agent_calls": 0,
        },
        "arms": arms,
        "comparisons": {
            "permanent_mean_completion_reduction_vs_fixed4": _reduction(permanent_mean, fixed_mean),
            "lease_mean_completion_reduction_vs_fixed4": _reduction(lease_mean, fixed_mean),
            "lease_fraction_of_permanent_mean_completion_gain": retained_gain,
            "lease_request_overlap_reduction_vs_permanent": overlap_reduction,
        },
        "gate": {"status": status, "checks": checks},
        "interpretation_boundary": (
            "Recorded action durations and prompt-token counts are replayed without a GPU service model. "
            "This screen can authorize a physical tail test but cannot establish TTFT safety."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_OUTPUT)
    args = parser.parse_args()
    result = evaluate(_git_sha())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "gate": result["gate"]}, indent=2))


if __name__ == "__main__":
    main()
