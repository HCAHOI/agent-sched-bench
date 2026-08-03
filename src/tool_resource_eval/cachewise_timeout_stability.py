"""Test exact tool timeouts with exhaustive task-deletion sensitivity."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tool_resource_eval.cachewise_reproduction import Gap, _histories, _remaining_mean
from tool_resource_eval.cachewise_synthetic_concurrency import (
    SCHEDULE_SEEDS,
    _load_task_subset,
    _partition_task_ids,
    _schedule_bootstrap,
    synthetic_schedule,
)


ARMS = ("tool", "timeout")
TimeoutKey = tuple[str, tuple[float | None, ...]]


def _timeout_signature(args_text: str) -> tuple[float | None, ...] | None:
    batch = json.loads(args_text)
    signature: list[float | None] = []
    found = False
    for args in batch:
        value = args.get("timeout") if isinstance(args, dict) else None
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value > 0
        ):
            signature.append(float(value))
            found = True
        else:
            signature.append(None)
    return tuple(signature) if found else None


def _timeout_histories(gaps: Iterable[Gap]) -> dict[TimeoutKey, np.ndarray]:
    grouped: dict[TimeoutKey, list[float]] = defaultdict(list)
    for gap in gaps:
        signature = _timeout_signature(gap.args_text)
        if signature is not None:
            grouped[(gap.tool_key, signature)].append(gap.end - gap.start)
    return {key: np.asarray(values) for key, values in grouped.items()}


def _predict(
    gap: Gap,
    elapsed: float,
    arm: str,
    global_history: np.ndarray,
    tool_history: dict[str, np.ndarray],
    timeout_history: dict[TimeoutKey, np.ndarray],
    signature_cache: dict[str, tuple[float | None, ...] | None],
) -> float:
    if arm == "timeout":
        if gap.args_text not in signature_cache:
            signature_cache[gap.args_text] = _timeout_signature(gap.args_text)
        signature = signature_cache[gap.args_text]
        if signature is not None:
            prediction = _remaining_mean(
                timeout_history.get((gap.tool_key, signature), np.asarray([])),
                elapsed,
            )
            if prediction is not None:
                return prediction
    prediction = _remaining_mean(
        tool_history.get(gap.tool_key, np.asarray([])), elapsed
    )
    if prediction is not None:
        return prediction
    return _remaining_mean(global_history, elapsed) or 0.0


def _choice(
    candidates: list[Gap],
    predictions: dict[Gap, float],
    now: float,
    excluded: str | None = None,
) -> tuple[float, Gap] | None:
    eligible = [gap for gap in candidates if gap.session_id != excluded]
    if len(eligible) < 2:
        return None
    oracle_remaining = max(gap.end - now for gap in eligible)
    chosen = max(
        eligible,
        key=lambda gap: (predictions[gap], gap.session_id),
    )
    return oracle_remaining - (chosen.end - now), chosen


def _evaluate_schedule(
    gaps: list[Gap],
    global_history: np.ndarray,
    tool_history: dict[str, np.ndarray],
    timeout_history: dict[TimeoutKey, np.ndarray],
    task_ids: Iterable[str],
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    tasks = tuple(task_ids)
    sums = dict.fromkeys(ARMS, 0.0)
    correct = dict.fromkeys(ARMS, 0)
    adjustments = {task: dict.fromkeys(ARMS, 0.0) for task in tasks}
    removed = dict.fromkeys(tasks, 0)
    signature_cache: dict[str, tuple[float | None, ...] | None] = {}
    active: list[Gap] = []
    ranking_events = choice_changes = 0

    for gap in gaps:
        active = [
            other
            for other in active
            if other.end > gap.start and other.session_id != gap.session_id
        ]
        candidates = [*active, gap]
        if len(candidates) >= 2:
            predictions = {
                arm: {
                    candidate: _predict(
                        candidate,
                        gap.start - candidate.start,
                        arm,
                        global_history,
                        tool_history,
                        timeout_history,
                        signature_cache,
                    )
                    for candidate in candidates
                }
                for arm in ARMS
            }
            choices = {
                arm: _choice(candidates, predictions[arm], gap.start) for arm in ARMS
            }
            ranking_events += 1
            for arm in ARMS:
                regret, _ = choices[arm]  # type: ignore[misc]
                sums[arm] += regret
                correct[arm] += regret <= 1e-12
            choice_changes += choices["tool"][1] != choices["timeout"][1]  # type: ignore[index]

            for excluded in {candidate.session_id for candidate in candidates}:
                if excluded not in adjustments:
                    continue
                if excluded == gap.session_id:
                    removed[excluded] += 1
                    for arm in ARMS:
                        adjustments[excluded][arm] -= choices[arm][0]  # type: ignore[index]
                    continue
                excluded_choices = {
                    arm: _choice(
                        candidates,
                        predictions[arm],
                        gap.start,
                        excluded,
                    )
                    for arm in ARMS
                }
                if excluded_choices["tool"] is None:
                    removed[excluded] += 1
                    for arm in ARMS:
                        adjustments[excluded][arm] -= choices[arm][0]  # type: ignore[index]
                    continue
                for arm in ARMS:
                    adjustments[excluded][arm] += (
                        excluded_choices[arm][0] - choices[arm][0]  # type: ignore[index]
                    )
        active.append(gap)

    if not ranking_events:
        raise ValueError("schedule has no overlapping paused sessions")
    metrics = {
        "ranking_events": ranking_events,
        "choice_changes": choice_changes,
        "arms": {
            arm: {
                "mean_regret_s": sums[arm] / ranking_events,
                "top1_oracle_agreement": correct[arm] / ranking_events,
            }
            for arm in ARMS
        },
    }
    leave_one_out: dict[str, dict[str, float]] = {}
    for task in tasks:
        remaining_events = ranking_events - removed[task]
        if not remaining_events:
            raise ValueError(f"deleting {task} leaves no ranking events")
        leave_one_out[task] = {
            arm: (sums[arm] + adjustments[task][arm]) / remaining_events for arm in ARMS
        }
    return metrics, leave_one_out


def run(root: Path, split_task_ids: Path) -> dict[str, Any]:
    canonical_ids = {path.name for path in root.iterdir() if path.is_dir()}
    selected_ids = [
        line.strip()
        for line in split_task_ids.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fit_ids, eval_ids = _partition_task_ids(canonical_ids, selected_ids)
    fit, fit_tasks = _load_task_subset(root, fit_ids)
    evaluation, eval_tasks = _load_task_subset(root, eval_ids)
    eval_ids_with_gaps = sorted({gap.session_id for gap in evaluation})
    global_history, tool_history = _histories(fit)
    timeout_history = _timeout_histories(fit)

    schedule_rows: list[dict[str, Any]] = []
    deletion_deltas: dict[str, list[float]] = defaultdict(list)
    for seed in SCHEDULE_SEEDS:
        scheduled, schedule = synthetic_schedule(fit, evaluation, seed=seed)
        metrics, leave_one_out = _evaluate_schedule(
            scheduled,
            global_history,
            tool_history,
            timeout_history,
            eval_ids_with_gaps,
        )
        delta = (
            metrics["arms"]["timeout"]["mean_regret_s"]
            - metrics["arms"]["tool"]["mean_regret_s"]
        )
        schedule_rows.append(
            {
                "seed": seed,
                "ranking_events": metrics["ranking_events"],
                "choice_changes": metrics["choice_changes"],
                "tool_mean_regret_s": metrics["arms"]["tool"]["mean_regret_s"],
                "timeout_mean_regret_s": metrics["arms"]["timeout"]["mean_regret_s"],
                "delta_mean_regret_s": delta,
                "tool_top1_oracle_agreement": metrics["arms"]["tool"][
                    "top1_oracle_agreement"
                ],
                "timeout_top1_oracle_agreement": metrics["arms"]["timeout"][
                    "top1_oracle_agreement"
                ],
                "realized_mean_live_sessions": schedule["realized_mean_live_sessions"],
                "realized_max_live_sessions": schedule["realized_max_live_sessions"],
            }
        )
        for task, arm_means in leave_one_out.items():
            deletion_deltas[task].append(arm_means["timeout"] - arm_means["tool"])

    primary = _schedule_bootstrap([row["delta_mean_regret_s"] for row in schedule_rows])
    task_deletions = [
        {
            "excluded_task": task,
            "delta_mean_regret_s": float(np.mean(deltas)),
            "helpful_schedules": sum(delta < 0 for delta in deltas),
            "harmful_schedules": sum(delta > 0 for delta in deltas),
        }
        for task, deltas in deletion_deltas.items()
    ]
    task_deletions.sort(key=lambda row: row["delta_mean_regret_s"], reverse=True)
    worst_deletion = task_deletions[0]
    stability_go = worst_deletion["delta_mean_regret_s"] < 0.0

    return {
        "status": "development-exposed timeout-only task-stability diagnostic",
        "question": "Does exact inference-time timeout conditioning improve TB victim ordering without depending on any one evaluation task?",
        "protocol": {
            "root": str(root.resolve()),
            "split_task_ids": str(split_task_ids.resolve()),
            "timeout_key": "(tool batch, exact tuple of positive finite numeric timeout values); missing/unusable values fall back to tool-name",
            "schedule_seeds": list(SCHEDULE_SEEDS),
            "primary": "mean_regret(timeout) - mean_regret(tool)",
            "aggregate_go": "upper endpoint of 95% paired schedule bootstrap CI < 0",
            "task_stability_go": "maximum mean delta over exhaustive delete-one-evaluation-task schedules < 0",
        },
        "data": {
            "fit_tasks_declared": len(fit_ids),
            "fit_tasks_with_gaps": fit_tasks,
            "fit_gaps": len(fit),
            "eval_tasks_declared": len(eval_ids),
            "eval_tasks_with_gaps": eval_tasks,
            "eval_gaps": len(evaluation),
            "timeout_history_keys": len(timeout_history),
        },
        "schedule_results": schedule_rows,
        "primary_comparison": primary,
        "task_deletion_results": task_deletions,
        "task_stability": {
            "evaluated_tasks": len(task_deletions),
            "worst_deletion": worst_deletion,
            "go": stability_go,
        },
        "decision": "PASSES timeout-only task-stability gate"
        if primary["go"] and stability_go
        else "STOP timeout-only direction",
        "review": "Independent bounded review found no critical or major issue; its zero-event deletion minor was fixed before the formal run.",
        "limitations": [
            "The TB traces, split, and schedules are development-exposed.",
            "Delete-one-task sensitivity is not independent task-population uncertainty.",
            "The simulation holds isolated durations fixed and models no contention, KV pressure, eviction feedback, or JCT.",
        ],
    }


def self_check() -> None:
    assert _timeout_signature('[{"timeout":30},{"x":1}]') == (30.0, None)
    assert _timeout_signature('[{"timeout":false}]') is None
    fit = [
        Gap("f1", 0.0, 10.0, '["exec"]', '[{"timeout":10}]'),
        Gap("f2", 0.0, 100.0, '["exec"]', '[{"timeout":100}]'),
    ]
    scheduled = [
        Gap("a", 0.0, 10.0, '["exec"]', '[{"timeout":100}]'),
        Gap("b", 1.0, 5.0, '["exec"]', '[{"timeout":10}]'),
        Gap("c", 2.0, 3.0, '["exec"]', '[{"timeout":10}]'),
    ]
    global_history, tool_history = _histories(fit)
    timeout_history = _timeout_histories(fit)
    _, leave_one_out = _evaluate_schedule(
        scheduled,
        global_history,
        tool_history,
        timeout_history,
        ("a", "b", "c"),
    )
    for excluded in leave_one_out:
        filtered = [gap for gap in scheduled if gap.session_id != excluded]
        brute, _ = _evaluate_schedule(
            filtered,
            global_history,
            tool_history,
            timeout_history,
            (),
        )
        for arm in ARMS:
            assert np.isclose(
                leave_one_out[excluded][arm],
                brute["arms"][arm]["mean_regret_s"],
            )
    try:
        _evaluate_schedule(
            scheduled[:2],
            global_history,
            tool_history,
            timeout_history,
            ("a",),
        )
    except ValueError as error:
        assert str(error) == "deleting a leaves no ranking events"
    else:
        raise AssertionError("zero-event task deletion must fail")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("traces/terminal-bench/tb-all/canonical"),
    )
    parser.add_argument(
        "--split-task-ids",
        type=Path,
        default=Path(
            "traces/terminal-bench/tb-all/runs/"
            "codex-glm100-unified-20260721/task_ids.txt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "analysis/results/cachewise-tb-timeout-stability-20260731/result.json"
        ),
    )
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("self-check passed")
        return
    result = run(args.root, args.split_task_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
