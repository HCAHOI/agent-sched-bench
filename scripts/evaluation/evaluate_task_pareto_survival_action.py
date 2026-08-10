#!/usr/bin/env python3
"""Evaluate task-unanimous memory-time triggers on a frozen same-repo split."""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib.request import urlopen

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.evaluation.evaluate_gpu_tool_gap_actions import (  # noqa: E402
    _comparison,
    _load_transfer_points,
    _summarize,
    choose_transfer_point,
    score_gap_action,
)
from scripts.evaluation.evaluate_static_survival_gap_action import (  # noqa: E402
    expected_early_action,
)
from scripts.evaluation.evaluate_survival_work_state_action import (  # noqa: E402
    CausalGroupedDurationMemory,
    _expected_trigger_delta,
    _program_context,
    command_work_key,
)
from spike.multitenant import (  # noqa: E402
    TraceProgram,
    TraceTurn,
    load_trace_programs,
    prepare_llama_chat_messages,
)
from tool_resource_eval.labels import repo_of  # noqa: E402
from tool_time.command import shell_command_heads  # noqa: E402


_DEFAULT_SPLIT = Path(
    "analysis/development/pennylane-survival-action-split.json"
)
_DEFAULT_KV_PROFILE = Path(
    "analysis/serving/tool-time-rho-measurement-a100-instruct-20260809/"
    "kv_swap.json"
)
_DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
_PROTOCOL_GIT_SHA = "5cc9ff96aa2274707772ab2c3545a75b0806adaf"
_SPLIT_REPO_PATH = "analysis/development/pennylane-survival-action-split.json"
_KV_PROFILE_REPO_PATH = (
    "analysis/serving/tool-time-rho-measurement-a100-instruct-20260809/"
    "kv_swap.json"
)
_BLOCK_SIZE = 16
_TOKENIZER_REPO = "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"
_TOKENIZER_REVISION = "6a426ef8adc0b4b96408001a9628d71f01c9ceca"
_TOKENIZER_FILES = {
    "tokenizer.json": "5cc5f00a5b203e90a27a3bd60d1ec393b07971e8",
    "tokenizer_config.json": "db88166e2bc4c799fd5d1ae643b75e84d03ee70e",
    "special_tokens_map.json": "02ee80b6196926a5ad790a004d9efd6ab1ba6542",
}
_TRANSFORMERS_VERSION = "4.57.6"
_DEADLINE_MS = 5000.0
_MIN_CHANGED_TASKS = 8
_ARMS = (
    "deadline_feedback",
    "exact_immediate",
    "work_immediate",
    "exact_task_pareto",
    "work_task_pareto",
    "exec_survival_oracle",
)


def task_unanimous_pareto_trigger_ms(
    values_by_task: dict[str, tuple[float, ...]],
    *,
    deadline_ms: float,
    size_gib: float,
    swap_out_ms: float,
    swap_in_ms: float,
) -> float:
    """Return the earliest empirically safe trigger shared by every task."""
    grouped = {
        task_id: tuple(float(value) for value in values)
        for task_id, values in values_by_task.items()
        if values
    }
    if len(grouped) < 2:
        return deadline_ms
    short_values = [
        value
        for values in grouped.values()
        for value in values
        if value <= deadline_ms
    ]
    trigger_ms = max(short_values, default=0.0)
    if trigger_ms >= deadline_ms:
        return deadline_ms
    for values in grouped.values():
        released_delta, stall_delta = _expected_trigger_delta(
            values,
            trigger_ms=trigger_ms,
            deadline_ms=deadline_ms,
            size_gib=size_gib,
            swap_out_ms=swap_out_ms,
            swap_in_ms=swap_in_ms,
        )
        if released_delta <= 0.0 or stall_delta > 0.0:
            return deadline_ms
    return trigger_ms


def _load_split(path: Path) -> tuple[list[TraceProgram], list[TraceProgram]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("corpus_role") != "development_exposed":
        raise ValueError("this evaluator accepts development-exposed data only")
    partitions = {}
    for name, expected_count in (("fit", 15), ("replay", 26)):
        rows = payload.get(name)
        if not isinstance(rows, list) or len(rows) != expected_count:
            raise ValueError(f"{name} must contain {expected_count} rows")
        ids = [str(row["task_id"]) for row in rows]
        traces = [
            trace if trace.is_absolute() else _REPO_ROOT / trace
            for row in rows
            for trace in (Path(str(row["trace"])),)
        ]
        programs = load_trace_programs(traces, task_ids=ids)
        if [program.task_id for program in programs] != ids:
            raise ValueError(f"{name} trace order differs from the frozen split")
        partitions[name] = programs
    fit = partitions["fit"]
    replay = partitions["replay"]
    all_ids = [program.task_id for program in (*fit, *replay)]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("fit and replay task IDs overlap")
    repos = {repo_of(task_id) for task_id in all_ids}
    if len(repos) != 1:
        raise ValueError("the frozen split must contain exactly one repository")
    numeric_ids = [int(task_id.rsplit("-", 1)[1]) for task_id in all_ids]
    if numeric_ids != sorted(numeric_ids):
        raise ValueError("the frozen split is not in numeric PR order")
    return fit, replay


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


def _require_frozen_file(path: Path, repo_path: str) -> None:
    resolved = path.resolve()
    expected_path = (_REPO_ROOT / repo_path).resolve()
    if resolved != expected_path:
        raise ValueError(f"frozen input path changed: {resolved}")
    expected = subprocess.run(
        ["git", "show", f"{_PROTOCOL_GIT_SHA}:{repo_path}"],
        check=True,
        capture_output=True,
    ).stdout
    if path.read_bytes() != expected:
        raise ValueError(f"frozen input content changed: {repo_path}")


def _git_blob_oid(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def _require_closed_tokenizer_cache(cache: Path) -> None:
    unexpected = sorted(
        item.name for item in cache.iterdir() if item.name not in _TOKENIZER_FILES
    )
    if unexpected:
        raise ValueError(f"unexpected tokenizer cache entries: {unexpected}")


def _load_tokenizer() -> Any:
    if version("transformers") != _TRANSFORMERS_VERSION:
        raise ValueError(f"transformers must equal {_TRANSFORMERS_VERSION}")
    cache = (
        Path(tempfile.gettempdir())
        / "agent-sched-bench-tokenizer"
        / _TOKENIZER_REVISION
    )
    cache.mkdir(parents=True, exist_ok=True)
    base_url = f"https://huggingface.co/{_TOKENIZER_REPO}/resolve/{_TOKENIZER_REVISION}"
    for file_name, expected_oid in _TOKENIZER_FILES.items():
        target = cache / file_name
        data = target.read_bytes() if target.exists() else b""
        if _git_blob_oid(data) != expected_oid:
            with urlopen(f"{base_url}/{file_name}", timeout=60) as response:
                data = response.read()
            if _git_blob_oid(data) != expected_oid:
                raise ValueError(f"tokenizer blob identity changed: {file_name}")
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_bytes(data)
            temporary.replace(target)
    _require_closed_tokenizer_cache(cache)

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(cache, local_files_only=True)


def _retained_tokens(turn: TraceTurn, tokenizer: Any) -> int:
    prompt_tokens = len(
        tokenizer.apply_chat_template(
            prepare_llama_chat_messages(turn.messages),
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    sequence_tokens = prompt_tokens + turn.completion_tokens
    return max(0, (sequence_tokens - 1) // _BLOCK_SIZE) * _BLOCK_SIZE


def _physical(
    *, retained_tokens: int, bytes_per_token: int, point: Any
) -> dict[str, float]:
    return {
        "size_gib": retained_tokens * bytes_per_token / 2**30,
        "swap_out_ms": point.swap_out_ms,
        "swap_in_ms": point.swap_in_ms,
    }


def _gate(comparison: dict[str, Any]) -> dict[str, bool]:
    return {
        "released_gib_s_strictly_higher": comparison["released_gib_s_delta"]
        > 0.0,
        "critical_path_stall_not_higher": comparison[
            "critical_path_stall_ms_delta"
        ]
        <= 0.0,
        "changed_at_least_8_tasks": comparison["changed_task_count"]
        >= _MIN_CHANGED_TASKS,
    }


def _select(
    passed: dict[str, bool], comparisons: dict[str, dict[str, Any]]
) -> tuple[bool, bool, str | None]:
    exact = comparisons["exact_task_pareto"]
    work = comparisons["work_task_pareto"]
    work_dominates = (
        work["released_gib_s_delta"] >= exact["released_gib_s_delta"]
        and work["critical_path_stall_ms_delta"]
        <= exact["critical_path_stall_ms_delta"]
        and (
            work["released_gib_s_delta"] > exact["released_gib_s_delta"]
            or work["critical_path_stall_ms_delta"]
            < exact["critical_path_stall_ms_delta"]
        )
    )
    primary_go = passed["work_task_pareto"]
    if not primary_go:
        return False, work_dominates, None
    selected = (
        "work_task_pareto"
        if not passed["exact_task_pareto"] or work_dominates
        else "exact_task_pareto"
    )
    return True, work_dominates, selected


def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    git_sha = _require_clean_checkout()
    _require_frozen_file(_DEFAULT_SPLIT, _SPLIT_REPO_PATH)
    _require_frozen_file(_DEFAULT_KV_PROFILE, _KV_PROFILE_REPO_PATH)
    fit_programs, replay_programs = _load_split(_DEFAULT_SPLIT)
    fit_contexts = {program.task_id: _program_context(program) for program in fit_programs}
    replay_contexts = {
        program.task_id: _program_context(program) for program in replay_programs
    }
    memory = CausalGroupedDurationMemory(())
    for program in fit_programs:
        memory.observe_task(program.task_id, fit_contexts[program.task_id].observations)

    profile = json.loads(_DEFAULT_KV_PROFILE.read_text(encoding="utf-8"))
    if profile.get("model") != _DEFAULT_MODEL:
        raise ValueError("KV profile model differs from the frozen tokenizer model")
    bytes_per_token = int(profile["bytes_per_token"])
    points = _load_transfer_points(_DEFAULT_KV_PROFILE)
    if any(point.swap_out_ms >= _DEADLINE_MS for point in points):
        raise ValueError("every measured swap-out must fit within the deadline")

    tokenizer = _load_tokenizer()
    counters: Counter[str] = Counter()
    gap_rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for program in replay_programs:
        context = replay_contexts[program.task_id]
        repo = repo_of(program.task_id)
        gap_turn_count = len(program.turns) - 1 + bool(program.omitted_terminal_llm_calls)
        for turn_index, turn in enumerate(program.turns[:gap_turn_count]):
            retained_tokens = _retained_tokens(turn, tokenizer)
            point = choose_transfer_point(points, max(1, retained_tokens))
            physical = _physical(
                retained_tokens=retained_tokens,
                bytes_per_token=bytes_per_token,
                point=point,
            )
            triggers = {arm: [_DEADLINE_MS] * len(turn.tools) for arm in _ARMS}
            for tool_index, tool in enumerate(turn.tools):
                if tool.tool_name != "exec":
                    counters["non_exec_gap_tools"] += 1
                    continue
                counters["exec_gap_tools"] += 1
                duration_ms = tool.end_offset_ms - tool.start_offset_ms
                if duration_ms > _DEADLINE_MS:
                    triggers["exec_survival_oracle"][tool_index] = 0.0
                    counters["exec_survival_oracle_early"] += 1
                work = command_work_key(tool.command)
                histories = memory.query(repo, tool.command, work)
                for history_arm, immediate_arm, pareto_arm in (
                    ("exact_command", "exact_immediate", "exact_task_pareto"),
                    ("work_signature", "work_immediate", "work_task_pareto"),
                ):
                    grouped = histories[history_arm]
                    if not grouped:
                        counters[f"{history_arm}_unavailable"] += 1
                        continue
                    counters[f"{history_arm}_available"] += 1
                    values = tuple(
                        value for task_values in grouped.values() for value in task_values
                    )
                    if expected_early_action(
                        values, deadline_ms=_DEADLINE_MS, **physical
                    )["act_early"]:
                        triggers[immediate_arm][tool_index] = 0.0
                        counters[f"{immediate_arm}_early"] += 1
                    trigger_ms = task_unanimous_pareto_trigger_ms(
                        grouped, deadline_ms=_DEADLINE_MS, **physical
                    )
                    if trigger_ms >= _DEADLINE_MS:
                        counters[f"{pareto_arm}_feedback"] += 1
                        continue
                    triggers[pareto_arm][tool_index] = trigger_ms
                    counters[f"{pareto_arm}_early"] += 1
                    counters[f"{pareto_arm}_actual_long"] += int(
                        duration_ms > _DEADLINE_MS
                    )
                    counters[f"{pareto_arm}_actual_short"] += int(
                        duration_ms <= _DEADLINE_MS
                    )
                    if len(examples) < 100:
                        examples.append(
                            {
                                "task_id": program.task_id,
                                "turn_index": turn_index,
                                "tool_index": tool_index,
                                "arm": pareto_arm,
                                "command": tool.command,
                                "history_task_count": len(grouped),
                                "history_count": len(values),
                                "trigger_ms": trigger_ms,
                                "observed_duration_ms": duration_ms,
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
            gap_rows.append(
                {
                    "task_id": program.task_id,
                    "turn_index": turn_index,
                    "tool_families": [
                        "+".join(sorted(set(heads)))
                        if (heads := shell_command_heads(tool.command))
                        else tool.tool_name
                        for tool in turn.tools
                    ],
                    "arms": arms,
                }
            )
        memory.observe_task(program.task_id, context.observations)

    summaries = {arm: _summarize(gap_rows, arm) for arm in _ARMS}
    comparisons = {
        arm: _comparison(gap_rows, arm, "deadline_feedback")
        for arm in _ARMS
        if arm != "deadline_feedback"
    }
    gate_checks = {
        arm: _gate(comparisons[arm])
        for arm in ("exact_task_pareto", "work_task_pareto")
    }
    passed = {arm: all(checks.values()) for arm, checks in gate_checks.items()}
    primary_go, work_dominates, selected = _select(passed, comparisons)
    return {
        "schema_version": 1,
        "status": "development_go" if primary_go else "development_no_go",
        "generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_sha": git_sha,
        "protocol_git_sha": _PROTOCOL_GIT_SHA,
        "protocol": "tool-resource-canonical-objective.md Section 5.11",
        "config": {
            "split": str(_DEFAULT_SPLIT.resolve()),
            "kv_profile": str(_DEFAULT_KV_PROFILE.resolve()),
            "model": _DEFAULT_MODEL,
            "block_size": _BLOCK_SIZE,
            "tokenizer_repo": _TOKENIZER_REPO,
            "tokenizer_revision": _TOKENIZER_REVISION,
            "tokenizer_git_blob_oids": _TOKENIZER_FILES,
            "transformers_version": _TRANSFORMERS_VERSION,
            "deadline_ms": _DEADLINE_MS,
            "minimum_changed_tasks": _MIN_CHANGED_TASKS,
            "history_update": "whole-task-final in numeric PR order",
            "matching": "exact command and exact work signature; no fallback",
        },
        "evidence": {
            "fit_task_count": len(fit_programs),
            "fit_exec_observations": sum(
                len(context.observations) for context in fit_contexts.values()
            ),
            "fit_omitted_terminal_tool_count": sum(
                context.omitted_terminal_tool_count
                for context in fit_contexts.values()
            ),
            "replay_task_count": len(replay_programs),
            "replay_omitted_terminal_tool_count": sum(
                context.omitted_terminal_tool_count
                for context in replay_contexts.values()
            ),
            "gap_count": len(gap_rows),
            **dict(sorted(counters.items())),
        },
        "arms": summaries,
        "comparisons_vs_deadline_feedback": comparisons,
        "gates": {
            "by_arm": {
                arm: {"checks": gate_checks[arm], "passed": passed[arm]}
                for arm in gate_checks
            },
            "primary_work_task_pareto_passed": primary_go,
            "work_pareto_dominates_exact": work_dominates,
            "selected_arm": selected,
        },
        "decision_examples": examples,
        "cost": {
            "prediction_time_agent_calls": 0,
            "gpu_runtime_s": 0.0,
            "evaluator_wall_s": time.monotonic() - started,
        },
        "limitations": [
            "All 41 collected PennyLane tasks are development-exposed.",
            "The work parser is a manual positive control, not a shipped general mechanism.",
            "Recorded gaps and measured A100 transfer costs are replayed rather than live serving.",
            "The realized survival oracle is hindsight-only.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = _evaluate(args)
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
