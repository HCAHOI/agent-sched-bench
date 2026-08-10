#!/usr/bin/env python3
"""Evaluate complete-work and causal-state KV survival actions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
import datetime as dt
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.evaluation.evaluate_gpu_tool_gap_actions import (  # noqa: E402
    _comparison,
    _summarize,
    score_gap_action,
)
from scripts.evaluation.evaluate_static_survival_gap_action import (  # noqa: E402
    _DEFAULT_GPU_GAPS,
    _DEFAULT_IDS,
    _DEFAULT_SOURCE,
    _DEFAULT_TASK_ROWS,
    _ordered_ids,
    _physical,
    _same_score,
    _valid_task_ids,
    expected_early_action,
)
from spike.multitenant import TraceProgram, load_trace_programs  # noqa: E402
from tool_resource.clause_parser import parse_command_clauses  # noqa: E402
from tool_resource.pip_semantics import (  # noqa: E402
    PipInstallSignature,
    PipTaskState,
    parse_pip_install,
)
from tool_resource.pytest_semantics import (  # noqa: E402
    PytestSignature,
    parse_pytest,
)
from tool_resource_eval.labels import repo_of  # noqa: E402


_APT_IGNORED_FLAGS = {"-y", "--yes", "--no-install-recommends"}
_APT_QUIET = re.compile(r"-q+")
_EXIT_CODE = re.compile(r"(?:^|\n)Exit code: (-?\d+)\s*$")
_DEFAULT_FIT_SOURCE = Path(
    "traces/swe-rebench/qwen3.7-max/swe100-full-5be74da-20260726"
)
_DEFAULT_KV_PROFILE = Path(
    "analysis/serving/tool-time-rho-measurement-a100-instruct-20260809/"
    "kv_swap.json"
)
_ARMS = (
    "deadline_feedback",
    "exact_command",
    "work_signature",
    "work_plus_state",
    "exec_survival_oracle",
)


@dataclass(frozen=True)
class AptWork:
    operation: str
    packages: tuple[str, ...]


WorkKey = tuple[tuple[str, Hashable], ...]
StateKey = tuple[tuple[Any, ...], ...]


def parse_apt_work(argv: Sequence[str]) -> AptWork | None:
    words = tuple(str(word) for word in argv)
    if len(words) < 2 or PurePosixPath(words[0]).name.lower() not in {
        "apt",
        "apt-get",
    }:
        return None
    operation = words[1].lower()
    if operation not in {"update", "install"}:
        return None
    packages: list[str] = []
    for word in words[2:]:
        if word in _APT_IGNORED_FLAGS or _APT_QUIET.fullmatch(word):
            continue
        if word.startswith("-"):
            return None
        packages.append(word.lower())
    if operation == "update":
        return AptWork(operation, ()) if not packages else None
    if not packages:
        return None
    return AptWork(operation, tuple(sorted(packages)))


def command_work_key(command: str) -> WorkKey | None:
    parsed = parse_command_clauses(command)
    if parsed["parse_failed"]:
        return None
    work: list[tuple[str, Hashable]] = []
    for clause in parsed["clauses"]:
        argv = tuple(str(value) for value in clause["argv"])
        if (signature := parse_pip_install(argv)) is not None:
            work.append(("pip", signature))
        elif (signature := parse_pytest(argv)) is not None:
            work.append(("pytest", signature))
        elif (signature := parse_apt_work(argv)) is not None:
            work.append(("apt", signature))
    return tuple(work) or None


class CausalWorkState:
    """Direct facts from earlier completed commands in one task."""

    def __init__(self) -> None:
        self._pip = PipTaskState()
        self._prior_invocation = {"apt": False, "pip": False, "pytest": False}
        self._apt_installed: set[str] = set()

    def query(self, work: WorkKey) -> StateKey:
        state: list[tuple[Any, ...]] = []
        for kind, signature in work:
            if kind == "apt":
                assert isinstance(signature, AptWork)
                remaining = (
                    tuple(
                        package
                        for package in signature.packages
                        if package not in self._apt_installed
                    )
                    if signature.operation == "install"
                    else ()
                )
                state.append((kind, self._prior_invocation[kind], remaining))
            elif kind == "pip":
                assert isinstance(signature, PipInstallSignature)
                query = self._pip.query(signature)
                state.append(
                    (
                        kind,
                        self._prior_invocation[kind],
                        query.availability,
                        query.remaining_packages,
                    )
                )
            else:
                assert kind == "pytest" and isinstance(signature, PytestSignature)
                state.append((kind, self._prior_invocation[kind]))
        return tuple(state)

    def observe(self, command: str, tool_result: str) -> None:
        parsed = parse_command_clauses(command)
        if parsed["parse_failed"] or len(parsed["clauses"]) != 1:
            return
        work = command_work_key(command)
        if work is None:
            return
        self._pip.observe(command, tool_result)
        exit_match = _EXIT_CODE.search(tool_result)
        exit_code = None if exit_match is None else int(exit_match.group(1))
        for kind, signature in work:
            self._prior_invocation[kind] = True
            if kind == "apt" and exit_code == 0:
                assert isinstance(signature, AptWork)
                if signature.operation == "install":
                    self._apt_installed.update(signature.packages)


@dataclass(frozen=True)
class DurationObservation:
    repo: str
    command: str
    work: WorkKey | None
    state: StateKey | None
    duration_ms: float


@dataclass(frozen=True)
class ReplayExec:
    command: str
    duration_ms: float
    tool_result: str


def duration_observations(
    repo: str,
    events: Sequence[ReplayExec],
) -> tuple[DurationObservation, ...]:
    state = CausalWorkState()
    rows: list[DurationObservation] = []
    for event in events:
        work = command_work_key(event.command)
        rows.append(
            DurationObservation(
                repo,
                event.command,
                work,
                None if work is None else state.query(work),
                event.duration_ms,
            )
        )
        state.observe(event.command, event.tool_result)
    return tuple(rows)


class CausalDurationMemory:
    """Public leave-repo-out durations plus explicitly settled local tasks."""

    def __init__(self, public: Sequence[DurationObservation]) -> None:
        self._public = tuple(public)
        self._local: dict[str, list[DurationObservation]] = defaultdict(list)

    def observe_task(self, observations: Sequence[DurationObservation]) -> None:
        for observation in observations:
            self._local[observation.repo].append(observation)

    @staticmethod
    def _values(
        rows: Sequence[DurationObservation],
        field: str,
        key: Hashable,
    ) -> tuple[float, ...]:
        return tuple(
            row.duration_ms for row in rows if getattr(row, field) == key
        )

    def query(
        self,
        repo: str,
        command: str,
        work: WorkKey | None,
        state: StateKey | None,
    ) -> dict[str, tuple[float, ...]]:
        local = self._local.get(repo, ())
        public = tuple(row for row in self._public if row.repo != repo)

        def select(field: str, key: Hashable | None) -> tuple[float, ...]:
            if key is None:
                return ()
            return self._values(local, field, key) or self._values(public, field, key)

        def select_work_state() -> tuple[float, ...]:
            if work is None or state is None:
                return ()
            local_values = tuple(
                row.duration_ms
                for row in local
                if row.work == work and row.state == state
            )
            return local_values or tuple(
                row.duration_ms
                for row in public
                if row.work == work and row.state == state
            )

        return {
            "exact_command": select("command", command),
            "work_signature": select("work", work),
            "work_plus_state": select_work_state(),
        }


@dataclass(frozen=True)
class ProgramContext:
    results: dict[tuple[int, int], str]
    states: dict[tuple[int, int], StateKey | None]
    observations: tuple[DurationObservation, ...]


def _raw_tool_results(program: TraceProgram) -> list[tuple[str, str, str]]:
    raw: list[tuple[str, str, str]] = []
    with Path(program.trace_path).open(encoding="utf-8") as handle:
        for line in handle:
            action = json.loads(line)
            data = action.get("data")
            if (
                action.get("type") != "action"
                or action.get("action_type") != "tool_exec"
                or not isinstance(data, dict)
            ):
                continue
            tool_name = data.get("tool_name")
            tool_result = data.get("tool_result")
            tool_args = data.get("tool_args")
            if not isinstance(tool_name, str) or not isinstance(tool_result, str):
                raise ValueError(f"{program.trace_path}: incomplete tool action")
            arguments = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
            command = (
                arguments.get("command", "") if isinstance(arguments, dict) else ""
            )
            if not isinstance(command, str):
                raise ValueError(f"{program.trace_path}: non-string command")
            raw.append((tool_name, command, tool_result))
    return raw


def _program_context(program: TraceProgram) -> ProgramContext:
    indexed_tools = [
        (turn_index, tool_index, tool)
        for turn_index, turn in enumerate(program.turns)
        for tool_index, tool in enumerate(turn.tools)
    ]
    raw = _raw_tool_results(program)
    if len(raw) != len(indexed_tools):
        raise ValueError(f"{program.task_id}: source/raw tool counts differ")
    results: dict[tuple[int, int], str] = {}
    for (turn_index, tool_index, tool), (name, command, result) in zip(
        indexed_tools, raw, strict=True
    ):
        if name != tool.tool_name or (name == "exec" and command != tool.command):
            raise ValueError(f"{program.task_id}: source/raw tool identity differs")
        results[(turn_index, tool_index)] = result

    state = CausalWorkState()
    states: dict[tuple[int, int], StateKey | None] = {}
    observations: list[DurationObservation] = []
    repo = repo_of(program.task_id)
    for turn_index, turn in enumerate(program.turns):
        pending: list[tuple[float, int, str, str]] = []
        order = sorted(
            enumerate(turn.tools),
            key=lambda item: (item[1].start_offset_ms, item[0]),
        )
        for tool_index, tool in order:
            completed = sorted(
                (item for item in pending if item[0] < tool.start_offset_ms),
                key=lambda item: (item[0], item[1]),
            )
            for item in completed:
                state.observe(item[2], item[3])
                pending.remove(item)
            if tool.tool_name != "exec":
                continue
            work = command_work_key(tool.command)
            state_key = None if work is None else state.query(work)
            states[(turn_index, tool_index)] = state_key
            observations.append(
                DurationObservation(
                    repo,
                    tool.command,
                    work,
                    state_key,
                    tool.end_offset_ms - tool.start_offset_ms,
                )
            )
            pending.append(
                (
                    tool.end_offset_ms,
                    tool_index,
                    tool.command,
                    results[(turn_index, tool_index)],
                )
            )
        for _end, _index, command, result in sorted(pending):
            state.observe(command, result)
    return ProgramContext(results, states, tuple(observations))


def _git_sha() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("formal evaluation requires a clean worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _assert_program_order(
    programs: Sequence[Any],
    expected_ids: Sequence[str],
    *,
    expected_count: int,
) -> None:
    observed = [str(program.task_id) for program in programs]
    if len(observed) != expected_count or observed != list(expected_ids):
        raise ValueError("trace programs differ from the frozen task order")


def _arm_gate(comparison: dict[str, Any]) -> dict[str, bool]:
    return {
        "released_gib_s_strictly_higher": comparison["released_gib_s_delta"]
        > 0.0,
        "critical_path_stall_not_higher": comparison[
            "critical_path_stall_ms_delta"
        ]
        <= 0.0,
        "changed_at_least_20_tasks": comparison["changed_task_count"] >= 20,
    }


def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    git_sha = _git_sha()
    valid = _valid_task_ids(args.task_rows)
    if valid != _valid_task_ids(_DEFAULT_TASK_ROWS):
        raise ValueError("task rows differ from the frozen 82/177 populations")
    replay_ids = _ordered_ids(args.task_ids, valid["swe277"])
    fit_ids = sorted(valid["swe100"])
    replay_programs = load_trace_programs(
        args.replay_source, task_ids=replay_ids, seed=42
    )
    fit_programs = load_trace_programs(args.fit_source, task_ids=fit_ids, seed=42)
    _assert_program_order(fit_programs, fit_ids, expected_count=82)
    _assert_program_order(replay_programs, replay_ids, expected_count=177)
    fit_contexts = {program.task_id: _program_context(program) for program in fit_programs}
    replay_contexts = {
        program.task_id: _program_context(program) for program in replay_programs
    }
    public = tuple(
        observation
        for task_id in fit_ids
        for observation in fit_contexts[task_id].observations
    )
    memory = CausalDurationMemory(public)

    gpu_payload = json.loads(args.gpu_gaps.read_text(encoding="utf-8"))
    deadline_ms = float(gpu_payload["config"]["deadline_ms"])
    if deadline_ms != 5000.0:
        raise ValueError("the frozen feedback deadline changed")
    profile = json.loads(args.kv_profile.read_text(encoding="utf-8"))
    bytes_per_token = int(profile["bytes_per_token"])
    gap_by_key = {
        (str(gap["task_id"]), int(gap["turn_index"])): gap
        for gap in gpu_payload["gaps"]
        if gap["task_id"] in valid["swe277"]
    }
    if len(gap_by_key) != 8273:
        raise ValueError("the frozen tool-gap population changed")

    counters: Counter[str] = Counter()
    gap_rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for program in replay_programs:
        context = replay_contexts[program.task_id]
        repo = repo_of(program.task_id)
        gap_turn_count = len(program.turns) - 1 + bool(program.omitted_terminal_llm_calls)
        for turn_index, turn in enumerate(program.turns[:gap_turn_count]):
            gap = gap_by_key.pop((program.task_id, turn_index))
            if len(turn.tools) != int(gap["tool_count"]) or not math.isclose(
                turn.gap_ms, float(gap["gap_ms"]), abs_tol=1e-9
            ):
                raise ValueError(
                    f"source/GPU gap mismatch at {program.task_id}:{turn_index}"
                )
            physical = _physical(gap, bytes_per_token=bytes_per_token)
            triggers = {arm: [deadline_ms] * len(turn.tools) for arm in _ARMS}
            for tool_index, tool in enumerate(turn.tools):
                if tool.tool_name != "exec":
                    counters["non_exec_gap_tools"] += 1
                    continue
                counters["exec_gap_tools"] += 1
                duration_ms = tool.end_offset_ms - tool.start_offset_ms
                if duration_ms > deadline_ms:
                    triggers["exec_survival_oracle"][tool_index] = 0.0
                    counters["exec_survival_oracle_early"] += 1
                work = command_work_key(tool.command)
                state = context.states[(turn_index, tool_index)]
                histories = memory.query(repo, tool.command, work, state)
                for arm in ("exact_command", "work_signature", "work_plus_state"):
                    values = histories[arm]
                    if not values:
                        counters[f"{arm}_unavailable"] += 1
                        continue
                    counters[f"{arm}_available"] += 1
                    action = expected_early_action(
                        values,
                        deadline_ms=deadline_ms,
                        **physical,
                    )
                    if action["act_early"]:
                        triggers[arm][tool_index] = 0.0
                        counters[f"{arm}_early"] += 1
                        if len(examples) < 100:
                            examples.append(
                                {
                                    "task_id": program.task_id,
                                    "turn_index": turn_index,
                                    "tool_index": tool_index,
                                    "arm": arm,
                                    "command": tool.command,
                                    "work": repr(work),
                                    "state": repr(state),
                                    "history_count": len(values),
                                    "observed_duration_ms": duration_ms,
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
            if not _same_score(
                arms["deadline_feedback"], gap["arms"]["deadline_reactive"]
            ):
                raise ValueError(
                    f"feedback score mismatch at {program.task_id}:{turn_index}"
                )
            gap_rows.append(
                {
                    "task_id": program.task_id,
                    "turn_index": turn_index,
                    "tool_families": gap["tool_families"],
                    "arms": arms,
                }
            )
        memory.observe_task(context.observations)
    if gap_by_key or counters["exec_gap_tools"] != 5799:
        raise ValueError("the frozen action population was not scored exactly once")

    summaries = {arm: _summarize(gap_rows, arm) for arm in _ARMS}
    comparisons = {
        arm: _comparison(gap_rows, arm, "deadline_feedback")
        for arm in _ARMS
        if arm != "deadline_feedback"
    }
    gates = {
        arm: _arm_gate(comparisons[arm])
        for arm in ("exact_command", "work_signature", "work_plus_state")
    }
    passed = {arm: all(checks.values()) for arm, checks in gates.items()}
    work = comparisons["work_signature"]
    state = comparisons["work_plus_state"]
    state_pareto_dominates = (
        state["released_gib_s_delta"] >= work["released_gib_s_delta"]
        and state["critical_path_stall_ms_delta"]
        <= work["critical_path_stall_ms_delta"]
        and (
            state["released_gib_s_delta"] > work["released_gib_s_delta"]
            or state["critical_path_stall_ms_delta"]
            < work["critical_path_stall_ms_delta"]
        )
    )
    state_justified = passed["work_plus_state"] and (
        not passed["work_signature"] or state_pareto_dominates
    )
    any_go = any(passed.values())
    return {
        "schema_version": 1,
        "status": "development_go" if any_go else "development_no_go",
        "generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_sha": git_sha,
        "protocol": (
            "One-shot exposed-data attribution of exact complete-command, exact "
            "tool-work, and causal work-state survival evidence"
        ),
        "config": {
            "task_rows": str(args.task_rows.resolve()),
            "fit_source": str(args.fit_source.resolve()),
            "replay_source": str(args.replay_source.resolve()),
            "task_ids": str(args.task_ids.resolve()),
            "gpu_gaps": str(args.gpu_gaps.resolve()),
            "kv_profile": str(args.kv_profile.resolve()),
            "deadline_ms": deadline_ms,
            "bytes_per_token": bytes_per_token,
            "history_update": "whole-task-final only",
            "state_visibility": "causally completed prior commands in current task",
            "matching": "exact only; no similarity or minimum support",
            "action": (
                "early iff expected released GiB*s delta > 0 and expected "
                "critical-path stall delta <= 0; otherwise five-second feedback"
            ),
        },
        "evidence": {
            "fit_population_task_count": len(fit_programs),
            "fit_exec_observations": len(public),
            "replay_task_count": len(replay_programs),
            "gap_count": len(gap_rows),
            **dict(sorted(counters.items())),
        },
        "arms": summaries,
        "comparisons_vs_deadline_feedback": comparisons,
        "gates": {
            "by_arm": {
                arm: {"checks": gates[arm], "passed": passed[arm]}
                for arm in gates
            },
            "any_arm_passed": any_go,
            "state_pareto_dominates_work_signature": state_pareto_dominates,
            "state_justified": state_justified,
            "selection_rule": (
                "Report all passing arms; retain state only if it passes and the "
                "simpler work arm fails or is Pareto-dominated"
            ),
        },
        "decision_examples": examples,
        "cost": {
            "prediction_time_agent_calls": 0,
            "gpu_runtime_s": 0.0,
            "evaluator_wall_s": time.monotonic() - started,
        },
        "limitations": [
            "All SWE evidence is development-exposed; this is not confirmation.",
            "The deterministic apt/pip/pytest model is a manual positive control, not a general shipped mechanism.",
            "State is limited to prior invocation and explicit successful install facts; initial image and cache contents remain unknown.",
            "Recorded gaps and measured A100 transfer costs are replayed rather than live serving.",
            "The realized survival oracle is hindsight-only.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-rows", type=Path, default=_DEFAULT_TASK_ROWS)
    parser.add_argument("--fit-source", type=Path, default=_DEFAULT_FIT_SOURCE)
    parser.add_argument("--replay-source", type=Path, default=_DEFAULT_SOURCE)
    parser.add_argument("--task-ids", type=Path, default=_DEFAULT_IDS)
    parser.add_argument("--gpu-gaps", type=Path, default=_DEFAULT_GPU_GAPS)
    parser.add_argument("--kv-profile", type=Path, default=_DEFAULT_KV_PROFILE)
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
