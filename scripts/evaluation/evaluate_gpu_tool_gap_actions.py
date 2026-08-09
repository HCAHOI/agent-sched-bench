#!/usr/bin/env python3
"""Evaluate causal tool-gap KV actions with measured GPU transfer costs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import datetime as dt
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from spike.multitenant import (  # noqa: E402
    ToolSpan,
    TraceProgram,
    load_trace_programs,
    prepare_llama_chat_messages,
)
from tool_time.command import (  # noqa: E402
    make_row_command_prefix_keys,
    shell_command_heads,
)
from tool_time.offline_evaluation import evaluate_offline_probe_clock  # noqa: E402
from tool_time.prerestore import prerestore_start_ms  # noqa: E402
from tool_time.prior import (  # noqa: E402
    build_latency_prior,
    latency_prior_hierarchy,
)
from tool_time.statistics import resample_task_totals  # noqa: E402


@dataclass(frozen=True)
class TransferPoint:
    """One directly measured, conservative KV transfer operating point."""

    tokens: int
    size_gib: float
    swap_out_ms: float
    swap_in_ms: float


def choose_transfer_point(
    points: Sequence[TransferPoint], context_tokens: int
) -> TransferPoint:
    """Return the smallest measured point covering ``context_tokens``."""
    if context_tokens <= 0:
        raise ValueError("context_tokens must be positive")
    for point in sorted(points, key=lambda row: row.tokens):
        if context_tokens <= point.tokens:
            return point
    raise ValueError(
        f"context {context_tokens} exceeds measured maximum "
        f"{max(point.tokens for point in points)}"
    )


def score_gap_action(
    *,
    gap_ms: float,
    tools: Sequence[ToolSpan],
    triggers_ms: Sequence[float],
    prerestore_starts_ms: Sequence[float | None],
    size_gib: float,
    swap_out_ms: float,
    swap_in_ms: float,
) -> dict[str, Any]:
    """Score one causal gap; realized spans are scorer-only."""
    if len(tools) != len(triggers_ms) or len(tools) != len(prerestore_starts_ms):
        raise ValueError("tools, triggers, and pre-restore starts must align")
    if min(gap_ms, size_gib, swap_out_ms, swap_in_ms) < 0.0:
        raise ValueError("gap and physical costs must be non-negative")
    candidates = []
    for index, (tool, trigger_ms) in enumerate(zip(tools, triggers_ms, strict=True)):
        if trigger_ms < 0.0:
            raise ValueError("trigger times must be non-negative")
        start_ms = tool.start_offset_ms + trigger_ms
        if start_ms < gap_ms and tool.end_offset_ms > start_ms:
            candidates.append((start_ms, index))
    if not candidates:
        return {
            "offloaded": False,
            "offload_tool_index": None,
            "offload_start_ms": None,
            "offload_complete_ms": None,
            "prerestore_start_ms": None,
            "released_gib_s": 0.0,
            "critical_path_stall_ms": 0.0,
            "reactive_stall_ms": 0.0,
            "hidden_reload_ms": 0.0,
            "wasted_reload_ms": 0.0,
            "early_residency_gib_s": 0.0,
        }

    offload_start_ms, tool_index = min(candidates)
    offload_complete_ms = offload_start_ms + swap_out_ms
    reactive_stall_ms = max(0.0, offload_complete_ms - gap_ms) + swap_in_ms
    reactive_released_ms = max(0.0, gap_ms - offload_complete_ms)
    released_gib_s = size_gib * reactive_released_ms / 1000.0

    planned = prerestore_starts_ms[tool_index]
    if planned is None:
        prerestore_absolute_ms = None
    else:
        prerestore_absolute_ms = max(
            offload_complete_ms,
            tools[tool_index].start_offset_ms + planned,
        )
    if prerestore_absolute_ms is None or prerestore_absolute_ms >= gap_ms:
        hidden_ms = wasted_ms = early_residency_gib_s = 0.0
        critical_path_stall_ms = reactive_stall_ms
    else:
        lead_ms = gap_ms - prerestore_absolute_ms
        if lead_ms <= swap_in_ms:
            hidden_ms, wasted_ms = lead_ms, 0.0
        else:
            hidden_ms, wasted_ms = 0.0, swap_in_ms
        early_residency_gib_s = size_gib * min(lead_ms, swap_in_ms) / 1000.0
        released_gib_s = max(0.0, released_gib_s - early_residency_gib_s)
        critical_path_stall_ms = reactive_stall_ms - hidden_ms

    return {
        "offloaded": True,
        "offload_tool_index": tool_index,
        "offload_start_ms": offload_start_ms,
        "offload_complete_ms": offload_complete_ms,
        "prerestore_start_ms": prerestore_absolute_ms,
        "released_gib_s": released_gib_s,
        "critical_path_stall_ms": critical_path_stall_ms,
        "reactive_stall_ms": reactive_stall_ms,
        "hidden_reload_ms": hidden_ms,
        "wasted_reload_ms": wasted_ms,
        "early_residency_gib_s": early_residency_gib_s,
    }


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _load_transfer_points(path: Path) -> tuple[TransferPoint, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    measurements = payload.get("measurements")
    if not isinstance(measurements, list) or not measurements:
        raise ValueError(f"{path}: measurements must be non-empty")
    points = tuple(
        TransferPoint(
            tokens=int(row["tokens"]),
            size_gib=float(row["bytes"]) / 2**30,
            swap_out_ms=float(row["swap_out_ms"]),
            swap_in_ms=float(row["swap_in_ms"]),
        )
        for row in measurements
    )
    if any(
        point.tokens <= 0
        or point.size_gib <= 0.0
        or point.swap_out_ms <= 0.0
        or point.swap_in_ms <= 0.0
        or not all(
            math.isfinite(value)
            for value in (point.size_gib, point.swap_out_ms, point.swap_in_ms)
        )
        for point in points
    ):
        raise ValueError(f"{path}: invalid transfer measurement")
    if [point.tokens for point in points] != sorted({point.tokens for point in points}):
        raise ValueError(f"{path}: token points must be unique and sorted")
    return points


def _task_ids(value: str | list[str]) -> list[str]:
    paths = [value] if isinstance(value, str) else value
    ids = [
        task_id
        for path in paths
        for task_id in Path(path).read_text(encoding="utf-8").splitlines()
        if task_id
    ]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("task IDs must be non-empty and unique")
    return ids


def _validate_task_counts(
    workload: dict[str, Any], replay_task_ids: set[str], profile_task_ids: set[str]
) -> None:
    expected_replay = int(workload["expected_task_count"])
    expected_profile = int(workload["expected_profile_task_count"])
    if len(replay_task_ids) != expected_replay:
        raise ValueError(
            f"expected {expected_replay} replay tasks, got {len(replay_task_ids)}"
        )
    if len(profile_task_ids) != expected_profile:
        raise ValueError(
            f"expected {expected_profile} profile tasks, got {len(profile_task_ids)}"
        )


def _sample_id(program: TraceProgram, turn_index: int, tool_index: int) -> str:
    return f"{program.task_id}::turn={turn_index}::tool={tool_index}"


def _latency_rows(programs: Sequence[TraceProgram]) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": _sample_id(program, turn_index, tool_index),
            "task_id": program.task_id,
            "source_trace": program.trace_path,
            "tool_name": tool.tool_name,
            "latency_ms": tool.end_offset_ms - tool.start_offset_ms,
            "tool_args": {"command": tool.command},
        }
        for program in programs
        for turn_index, turn in enumerate(program.turns)
        for tool_index, tool in enumerate(turn.tools)
    ]


def _fit_one_point(
    point: TransferPoint,
    profile_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    deadline_ms: float,
    inner_folds: int,
    max_prefix_depth: int,
    min_tool_history: int,
    min_profile_tasks: int,
) -> tuple[int, dict[str, Any], dict[str, dict[str, Any]]]:
    result = evaluate_offline_probe_clock(
        eval_rows,
        profile_rows=profile_rows,
        kv_costs_ms=(point.swap_out_ms,),
        guard_ms=deadline_ms - point.swap_out_ms,
        inner_folds=inner_folds,
        min_tool_history=min_tool_history,
        min_profile_tasks=min_profile_tasks,
        command_field="command",
        max_prefix_depth=max_prefix_depth,
        skip_leading_cd=False,
        restore_cost_fraction=point.swap_in_ms / point.swap_out_ms,
    )
    decisions = {
        str(row["sample_id"]): {
            "trigger_ms": float(row["offline_gated_robust_trigger_ms"]),
            "candidate_trigger_ms": float(row["probe_robust_candidate_trigger_ms"]),
            "margin_normalized": float(row["probe_robust_margin_normalized"]),
            "prior_source": row["probe_robust_source"],
            "prior_group_key": row["probe_robust_group_key"],
            "prior_task_count": int(row["probe_robust_task_count"]),
        }
        for row in result["decisions"]
    }
    return point.tokens, result["robust_calibration"], decisions


def _fit_points(
    points: Sequence[TransferPoint],
    *,
    profile_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    deadline_ms: float,
    inner_folds: int,
    max_prefix_depth: int,
    min_tool_history: int,
    min_profile_tasks: int,
    workers: int,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, dict[str, Any]]]]:
    calibrations: dict[int, dict[str, Any]] = {}
    decisions: dict[int, dict[str, dict[str, Any]]] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _fit_one_point,
                point,
                profile_rows,
                eval_rows,
                deadline_ms,
                inner_folds,
                max_prefix_depth,
                min_tool_history,
                min_profile_tasks,
            ): point
            for point in points
        }
        for future in as_completed(futures):
            tokens, calibration, point_decisions = future.result()
            calibrations[tokens] = calibration
            decisions[tokens] = point_decisions
            print(f"fitted robust clock at {tokens} tokens", flush=True)
    return calibrations, decisions


def _prerestore_starts(
    *,
    prior: Any,
    keyer: Any,
    program: TraceProgram,
    turn_index: int,
    triggers: Sequence[float],
    swap_in_ms: float,
    min_tool_history: int,
    min_profile_tasks: int,
    cache: dict[tuple[int, float, float], float | None],
) -> tuple[float | None, ...]:
    output: list[float | None] = []
    for tool_index, (tool, trigger) in enumerate(
        zip(program.turns[turn_index].tools, triggers, strict=True)
    ):
        row = {
            "sample_id": _sample_id(program, turn_index, tool_index),
            "task_id": program.task_id,
            "source_trace": program.trace_path,
            "tool_name": tool.tool_name,
            "latency_ms": tool.end_offset_ms - tool.start_offset_ms,
            "tool_args": {"command": tool.command},
        }
        node = latency_prior_hierarchy(
            prior,
            tool.tool_name,
            keyer(row),
            min_tool_history=min_tool_history,
            min_profile_tasks=min_profile_tasks,
        )[-1]
        key = (id(node.values), float(trigger), swap_in_ms)
        if key not in cache:
            cache[key] = prerestore_start_ms(
                node,
                swap_trigger_ms=float(trigger),
                restore_cost_ms=swap_in_ms,
            )
        output.append(cache[key])
    return tuple(output)


def _summarize(rows: Sequence[dict[str, Any]], arm: str) -> dict[str, Any]:
    values = [row["arms"][arm] for row in rows]
    return {
        "gap_count": len(values),
        "offload_count": sum(bool(row["offloaded"]) for row in values),
        "prerestore_plan_count": sum(
            row["prerestore_start_ms"] is not None for row in values
        ),
        "prerestore_fired_count": sum(
            float(row["hidden_reload_ms"]) > 0.0
            or float(row["wasted_reload_ms"]) > 0.0
            for row in values
        ),
        "released_gib_s": math.fsum(float(row["released_gib_s"]) for row in values),
        "critical_path_stall_ms": math.fsum(
            float(row["critical_path_stall_ms"]) for row in values
        ),
        "hidden_reload_ms": math.fsum(float(row["hidden_reload_ms"]) for row in values),
        "wasted_reload_ms": math.fsum(float(row["wasted_reload_ms"]) for row in values),
        "early_residency_gib_s": math.fsum(
            float(row["early_residency_gib_s"]) for row in values
        ),
    }


def _comparison(
    rows: Sequence[dict[str, Any]], candidate: str, baseline: str
) -> dict[str, Any]:
    changed_tasks: set[str] = set()
    changed_gaps = 0
    by_task: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    by_signature: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for row in rows:
        cand = row["arms"][candidate]
        base = row["arms"][baseline]
        released_delta = float(cand["released_gib_s"]) - float(base["released_gib_s"])
        stall_delta = float(cand["critical_path_stall_ms"]) - float(
            base["critical_path_stall_ms"]
        )
        if not math.isclose(released_delta, 0.0, abs_tol=1e-12) or not math.isclose(
            stall_delta, 0.0, abs_tol=1e-9
        ):
            changed_gaps += 1
            changed_tasks.add(str(row["task_id"]))
        by_task[str(row["task_id"])][0] += released_delta
        by_task[str(row["task_id"])][1] += stall_delta
        candidate_index = cand["offload_tool_index"]
        baseline_index = base["offload_tool_index"]
        selected_index = (
            candidate_index if candidate_index is not None else baseline_index
        )
        family = (
            "__no_action__"
            if selected_index is None
            else row["tool_families"][selected_index]
        )
        by_signature[str(family)][0] += released_delta
        by_signature[str(family)][1] += stall_delta
    task_ids = sorted(by_task)
    matrix = np.asarray([by_task[task_id] for task_id in task_ids], dtype=float)
    boot = resample_task_totals(matrix, replicates=10_000, seed=0)
    released_delta, stall_delta = np.sum(matrix, axis=0)
    baseline_stall = math.fsum(
        float(row["arms"][baseline]["critical_path_stall_ms"]) for row in rows
    )
    return {
        "candidate": candidate,
        "baseline": baseline,
        "released_gib_s_delta": float(released_delta),
        "critical_path_stall_ms_delta": float(stall_delta),
        "critical_path_stall_reduction_fraction": (
            -float(stall_delta) / baseline_stall if baseline_stall > 0.0 else None
        ),
        "changed_gap_count": changed_gaps,
        "changed_task_count": len(changed_tasks),
        "task_bootstrap_95pct": {
            "released_gib_s_delta": [
                float(np.percentile(boot[:, 0], 2.5)),
                float(np.percentile(boot[:, 0], 97.5)),
            ],
            "critical_path_stall_ms_delta": [
                float(np.percentile(boot[:, 1], 2.5)),
                float(np.percentile(boot[:, 1], 97.5)),
            ],
        },
        "largest_signature_deltas": [
            {
                "command_family": signature,
                "released_gib_s_delta": values[0],
                "critical_path_stall_ms_delta": values[1],
            }
            for signature, values in sorted(
                by_signature.items(),
                key=lambda item: abs(item[1][0]),
                reverse=True,
            )[:10]
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/serving/w5_multitenant.yaml")
    )
    parser.add_argument("--workload", default="swe-rebench-277-development-exposed")
    parser.add_argument(
        "--kv-profile",
        type=Path,
        default=Path(
            "analysis/serving/tool-time-rho-measurement-a100-instruct-20260809/kv_swap.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--max-prefix-depth", type=int, default=4)
    parser.add_argument("--min-tool-history", type=int, default=1)
    parser.add_argument("--min-profile-tasks", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    workloads = [row for row in config["workloads"] if row["name"] == args.workload]
    if len(workloads) != 1:
        raise ValueError(f"expected one workload named {args.workload!r}")
    workload = workloads[0]
    if workload["corpus_role"] != "development_exposed":
        raise ValueError("this evaluator is restricted to development_exposed data")
    deadline_ms = float(workload["deadline_ms"])
    points = _load_transfer_points(args.kv_profile)
    if any(point.swap_out_ms >= deadline_ms for point in points):
        raise ValueError("every measured swap-out cost must be below the deadline")

    replay_ids = _task_ids(workload["task_ids_file"])
    replay_programs = load_trace_programs(
        workload["replay_trace_root"], task_ids=replay_ids, seed=config["seed"]
    )
    profile_programs = load_trace_programs(
        workload["profile_trace_root"], seed=config["seed"]
    )
    replay_task_ids = {program.task_id for program in replay_programs}
    profile_task_ids = {program.task_id for program in profile_programs}
    _validate_task_counts(workload, replay_task_ids, profile_task_ids)
    if replay_task_ids & profile_task_ids:
        raise ValueError("profile and replay tasks must be disjoint")
    profile_rows = _latency_rows(profile_programs)
    eval_rows = _latency_rows(replay_programs)

    print(
        f"fitting {len(points)} physical points on {len(profile_rows)} profile "
        f"and {len(eval_rows)} replay tool calls",
        flush=True,
    )
    calibrations, decisions = _fit_points(
        points,
        profile_rows=profile_rows,
        eval_rows=eval_rows,
        deadline_ms=deadline_ms,
        inner_folds=args.inner_folds,
        max_prefix_depth=args.max_prefix_depth,
        min_tool_history=args.min_tool_history,
        min_profile_tasks=args.min_profile_tasks,
        workers=min(args.workers, len(points)),
    )

    from transformers import AutoTokenizer

    model = str(config["serving"]["model"])
    tokenizer = AutoTokenizer.from_pretrained(model)
    block_size = int(config["serving"]["block_size"])
    bytes_per_token = int(
        json.loads(args.kv_profile.read_text(encoding="utf-8"))["bytes_per_token"]
    )
    keyer = make_row_command_prefix_keys(
        "command", max_depth=args.max_prefix_depth, skip_leading_cd=False
    )
    prior = build_latency_prior(profile_rows, row_group_keys=keyer)
    pre_cache: dict[tuple[int, float, float], float | None] = {}
    gap_rows: list[dict[str, Any]] = []

    for program in replay_programs:
        gap_turn_count = (
            len(program.turns) - 1 + bool(program.omitted_terminal_llm_calls)
        )
        for turn_index, turn in enumerate(program.turns[:gap_turn_count]):
            prompt_tokens = len(
                tokenizer.apply_chat_template(
                    prepare_llama_chat_messages(turn.messages),
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
            retained_tokens = max(0, (prompt_tokens - 1) // block_size) * block_size
            point = choose_transfer_point(points, max(1, retained_tokens))
            size_gib = retained_tokens * bytes_per_token / 2**30
            point_decisions = decisions[point.tokens]
            robust_triggers = tuple(
                float(
                    point_decisions[_sample_id(program, turn_index, index)][
                        "trigger_ms"
                    ]
                )
                for index, _ in enumerate(turn.tools)
            )
            deadline_triggers = (deadline_ms,) * len(turn.tools)
            deadline_pre = _prerestore_starts(
                prior=prior,
                keyer=keyer,
                program=program,
                turn_index=turn_index,
                triggers=deadline_triggers,
                swap_in_ms=point.swap_in_ms,
                min_tool_history=args.min_tool_history,
                min_profile_tasks=args.min_profile_tasks,
                cache=pre_cache,
            )
            robust_pre = _prerestore_starts(
                prior=prior,
                keyer=keyer,
                program=program,
                turn_index=turn_index,
                triggers=robust_triggers,
                swap_in_ms=point.swap_in_ms,
                min_tool_history=args.min_tool_history,
                min_profile_tasks=args.min_profile_tasks,
                cache=pre_cache,
            )
            no_pre = (None,) * len(turn.tools)
            arms = {
                "deadline_reactive": score_gap_action(
                    gap_ms=turn.gap_ms,
                    tools=turn.tools,
                    triggers_ms=deadline_triggers,
                    prerestore_starts_ms=no_pre,
                    size_gib=size_gib,
                    swap_out_ms=point.swap_out_ms,
                    swap_in_ms=point.swap_in_ms,
                ),
                "robust_reactive": score_gap_action(
                    gap_ms=turn.gap_ms,
                    tools=turn.tools,
                    triggers_ms=robust_triggers,
                    prerestore_starts_ms=no_pre,
                    size_gib=size_gib,
                    swap_out_ms=point.swap_out_ms,
                    swap_in_ms=point.swap_in_ms,
                ),
                "deadline_prerestore": score_gap_action(
                    gap_ms=turn.gap_ms,
                    tools=turn.tools,
                    triggers_ms=deadline_triggers,
                    prerestore_starts_ms=deadline_pre,
                    size_gib=size_gib,
                    swap_out_ms=point.swap_out_ms,
                    swap_in_ms=point.swap_in_ms,
                ),
                "robust_prerestore": score_gap_action(
                    gap_ms=turn.gap_ms,
                    tools=turn.tools,
                    triggers_ms=robust_triggers,
                    prerestore_starts_ms=robust_pre,
                    size_gib=size_gib,
                    swap_out_ms=point.swap_out_ms,
                    swap_in_ms=point.swap_in_ms,
                ),
            }
            gap_rows.append(
                {
                    "task_id": program.task_id,
                    "turn_index": turn_index,
                    "tool_signature": turn.tool_signature,
                    "tool_families": [
                        "+".join(sorted(set(heads)))
                        if (heads := shell_command_heads(tool.command))
                        else tool.tool_name
                        for tool in turn.tools
                    ],
                    "tool_count": len(turn.tools),
                    "gap_ms": turn.gap_ms,
                    "actual_prompt_tokens": prompt_tokens,
                    "retained_tokens": retained_tokens,
                    "transfer_point_tokens": point.tokens,
                    "transfer_point": asdict(point),
                    "arms": arms,
                }
            )
        print(f"scored {program.task_id}", flush=True)

    arm_names = (
        "deadline_reactive",
        "robust_reactive",
        "deadline_prerestore",
        "robust_prerestore",
    )
    summaries = {arm: _summarize(gap_rows, arm) for arm in arm_names}
    robust = _comparison(gap_rows, "robust_reactive", "deadline_reactive")
    deadline_pre = _comparison(gap_rows, "deadline_prerestore", "deadline_reactive")
    robust_pre = _comparison(gap_rows, "robust_prerestore", "robust_reactive")
    robust_go = (
        robust["released_gib_s_delta"] > 0.0
        and robust["critical_path_stall_ms_delta"] <= 0.0
        and robust["changed_task_count"] >= 20
    )
    deadline_pre_go = (
        deadline_pre["changed_task_count"] >= 20
        and (deadline_pre["critical_path_stall_reduction_fraction"] or 0.0) >= 0.05
    )
    robust_pre_go = (
        robust_pre["changed_task_count"] >= 20
        and (robust_pre["critical_path_stall_reduction_fraction"] or 0.0) >= 0.05
    )
    live_2x2_go = robust_go and (deadline_pre_go or robust_pre_go)
    result = {
        "schema_version": 1,
        "status": "go" if live_2x2_go else "no_go",
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "protocol": (
            "tool-resource-canonical-objective.md Section 5.1; exposed "
            "development evidence only"
        ),
        "config": {
            "source": str(args.config),
            "workload": args.workload,
            "deadline_ms": deadline_ms,
            "kv_profile": str(args.kv_profile),
            "inner_folds": args.inner_folds,
            "max_prefix_depth": args.max_prefix_depth,
            "min_tool_history": args.min_tool_history,
            "min_profile_tasks": args.min_profile_tasks,
            "context_cost_rule": "smallest measured token point >= retained tokens",
            "tokenizer": model,
            "block_size": block_size,
        },
        "evidence": {
            "profile_task_count": len(profile_task_ids),
            "profile_tool_call_count": len(profile_rows),
            "replay_task_count": len(replay_task_ids),
            "replay_tool_call_count": len(eval_rows),
            "profile_replay_overlap": 0,
            "gap_count": len(gap_rows),
        },
        "transfer_points": [asdict(point) for point in points],
        "robust_calibrations": {
            str(tokens): calibration
            for tokens, calibration in sorted(calibrations.items())
        },
        "arms": summaries,
        "comparisons": {
            "robust_timing": robust,
            "deadline_prerestore": deadline_pre,
            "robust_prerestore": robust_pre,
        },
        "gates": {
            "robust_timing_go": robust_go,
            "deadline_prerestore_go": deadline_pre_go,
            "robust_prerestore_go": robust_pre_go,
            "live_2x2_go": live_2x2_go,
        },
        "limitations": [
            "offline scorer values released capacity and transfer stall separately",
            "real co-tenant admission and scheduling utility require live GPU replay",
            "pre-restore uses the existing conservative re-eviction accounting",
            "all replay outcomes are development-exposed",
        ],
        "gaps": gap_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "gates": result["gates"]}, indent=2))


if __name__ == "__main__":
    main()
