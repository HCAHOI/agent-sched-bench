"""Test causal exec progress counters against tool-name reuse ordering."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from tool_resource_eval.cachewise_reproduction import (
    _action_index,
    _cluster_bootstrap_delta,
)
from trace_collect.resource_timeline import valid_resource_timeline
from trace_collect.tool_gap_extractor import (
    discover_trace_files,
    extract_tool_gap_windows,
)


FEATURE_NAMES = (
    "gap_elapsed_s",
    "sample_count",
    "cumulative_cpu_core_s",
    "mean_cpu_cores",
    "recent_cpu_cores",
    "cpu_fraction_of_quota",
    "cpu_quota_cores",
    "log1p_cumulative_net_rx_bytes",
    "log1p_cumulative_net_tx_bytes",
    "log1p_recent_net_rx_bytes_per_s",
    "log1p_recent_net_tx_bytes_per_s",
)
SAMPLE_AVAILABILITY_PAD_S = 0.050


@dataclass(frozen=True)
class ProgressPoint:
    available_at: float
    features: tuple[float, ...]


@dataclass(frozen=True)
class ProgressGap:
    session_id: str
    start: float
    end: float
    tool_key: str
    points: tuple[ProgressPoint, ...] = ()


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, number) if math.isfinite(number) else 0.0


def _progress_points(
    *,
    gap_start: float,
    gap_end: float,
    action: dict[str, Any],
) -> tuple[ProgressPoint, ...]:
    data = action.get("data") or {}
    timeline = valid_resource_timeline(data.get("resource_timeline"))
    if timeline is None:
        return ()
    samples = [sample for sample in timeline["samples"] if isinstance(sample, dict)]
    if not samples:
        return ()

    action_start = float(action["ts_start"])
    action_end = float(action["ts_end"])
    last_offset = _number(samples[-1].get("offset_s"))
    total_overhead = action_end - action_start - last_offset
    if total_overhead > SAMPLE_AVAILABILITY_PAD_S:
        raise ValueError(
            f"resource timeline overhead {total_overhead:.6f}s exceeds "
            f"the frozen {SAMPLE_AVAILABILITY_PAD_S:.3f}s availability pad"
        )

    total_dt = total_cpu = total_opportunity = 0.0
    total_rx = total_tx = 0.0
    points: list[ProgressPoint] = []
    for index, sample in enumerate(samples, start=1):
        offset = _number(sample.get("offset_s"))
        dt = _number(sample.get("dt_s"))
        cpu = _number(sample.get("cpu_core_s"))
        opportunity = _number(sample.get("cpu_opportunity_core_s"))
        rx = _number(sample.get("net_rx_bytes"))
        tx = _number(sample.get("net_tx_bytes"))
        available_at = action_start + offset + SAMPLE_AVAILABILITY_PAD_S
        if dt <= 0 or available_at > gap_end:
            continue
        total_dt += dt
        total_cpu += cpu
        total_opportunity += opportunity
        total_rx += rx
        total_tx += tx
        quota = _number(sample.get("cpu_quota_cores"))
        features = (
            max(0.0, available_at - gap_start),
            float(index),
            total_cpu,
            total_cpu / total_dt,
            cpu / dt,
            total_cpu / total_opportunity if total_opportunity > 0 else 0.0,
            quota,
            math.log1p(total_rx),
            math.log1p(total_tx),
            math.log1p(rx / dt),
            math.log1p(tx / dt),
        )
        points.append(ProgressPoint(available_at=available_at, features=features))
    return tuple(points)


def load_gaps(root: Path) -> list[ProgressGap]:
    gaps: list[ProgressGap] = []
    for trace_path in discover_trace_files([root]):
        windows = extract_tool_gap_windows(trace_path)
        if not windows:
            continue
        llm, tools = _action_index(trace_path)
        for window in windows:
            start = float(llm[window.llm_action_id]["ts_end"])
            end = float(llm[window.next_llm_action_id]["ts_start"])
            points: tuple[ProgressPoint, ...] = ()
            if window.tool_names == ("exec",) and len(window.tool_call_ids) == 1:
                points = _progress_points(
                    gap_start=start,
                    gap_end=end,
                    action=tools[window.tool_call_ids[0]],
                )
            gaps.append(
                ProgressGap(
                    session_id=window.instance_id or window.source_trace,
                    start=start,
                    end=end,
                    tool_key=json.dumps(window.tool_names, separators=(",", ":")),
                    points=points,
                )
            )
    gaps.sort(key=lambda gap: (gap.start, gap.end, gap.session_id))
    if not gaps:
        raise ValueError(f"no tool gaps found under {root}")
    return gaps


def _histories(
    gaps: Iterable[ProgressGap],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    global_values: list[float] = []
    by_tool: dict[str, list[float]] = defaultdict(list)
    for gap in gaps:
        duration = gap.end - gap.start
        global_values.append(duration)
        by_tool[gap.tool_key].append(duration)
    return np.asarray(global_values), {
        tool: np.asarray(values) for tool, values in by_tool.items()
    }


def _remaining_mean(durations: np.ndarray, elapsed: float) -> float | None:
    survivors = durations[durations > elapsed]
    if not len(survivors):
        return None
    return float(np.mean(survivors - elapsed))


def _baseline_prediction(
    gap: ProgressGap,
    query_at: float,
    global_history: np.ndarray,
    tool_history: dict[str, np.ndarray],
) -> float:
    elapsed = query_at - gap.start
    prediction = _remaining_mean(
        tool_history.get(gap.tool_key, np.asarray([])), elapsed
    )
    if prediction is not None:
        return prediction
    return _remaining_mean(global_history, elapsed) or 0.0


def _latest_point(gap: ProgressGap, query_at: float) -> ProgressPoint | None:
    index = bisect_right([point.available_at for point in gap.points], query_at)
    return gap.points[index - 1] if index else None


def _training_rows(gaps: Iterable[ProgressGap]) -> tuple[np.ndarray, np.ndarray]:
    features: list[tuple[float, ...]] = []
    labels: list[float] = []
    for gap in gaps:
        for point in gap.points:
            remaining = gap.end - point.available_at
            if remaining < 0:
                continue
            features.append(point.features)
            labels.append(remaining)
    if not features:
        raise ValueError("fit corpus has no causal progress rows")
    return np.asarray(features), np.asarray(labels)


def evaluate(
    gaps: list[ProgressGap],
    *,
    model: HistGradientBoostingRegressor,
    global_history: np.ndarray,
    tool_history: dict[str, np.ndarray],
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, list[tuple[tuple[str, ...], float]]],
    dict[str, int],
    dict[str, float | int],
]:
    arms = ("tool", "progress")
    rows: dict[str, list[tuple[tuple[str, ...], float]]] = {arm: [] for arm in arms}
    correct = {arm: 0 for arm in arms}
    active: list[ProgressGap] = []
    ranking_events_with_progress = changed_choices = 0
    changed_better = changed_worse = 0
    help_s = harm_s = 0.0
    point_baseline_errors: list[float] = []
    point_progress_errors: list[float] = []
    point_baseline_biases: list[float] = []
    point_progress_biases: list[float] = []
    for gap in gaps:
        active = [
            other
            for other in active
            if other.end > gap.start and other.session_id != gap.session_id
        ]
        candidates = [*active, gap]
        if len(candidates) >= 2:
            pair = tuple(sorted(candidate.session_id for candidate in candidates))
            oracle_remaining = max(
                candidate.end - gap.start for candidate in candidates
            )
            points = {
                candidate.session_id: _latest_point(candidate, gap.start)
                for candidate in candidates
            }
            ranking_events_with_progress += any(points.values())
            scores: dict[str, tuple[float, float]] = {}
            for candidate in candidates:
                baseline = _baseline_prediction(
                    candidate, gap.start, global_history, tool_history
                )
                point = points[candidate.session_id]
                progress = (
                    baseline
                    if point is None
                    else max(0.0, float(model.predict([point.features])[0]))
                )
                scores[candidate.session_id] = (baseline, progress)
                if point is not None:
                    actual = candidate.end - gap.start
                    point_baseline_errors.append(abs(baseline - actual))
                    point_progress_errors.append(abs(progress - actual))
                    point_baseline_biases.append(baseline - actual)
                    point_progress_biases.append(progress - actual)

            chosen: dict[str, ProgressGap] = {}
            regrets: dict[str, float] = {}
            for arm_index, arm in enumerate(arms):
                chosen[arm] = max(
                    candidates,
                    key=lambda candidate: (
                        scores[candidate.session_id][arm_index],
                        candidate.session_id,
                    ),
                )
                regret = oracle_remaining - (chosen[arm].end - gap.start)
                regrets[arm] = regret
                rows[arm].append((pair, regret))
                correct[arm] += regret <= 1e-12
            if chosen["tool"] != chosen["progress"]:
                changed_choices += 1
                delta = regrets["progress"] - regrets["tool"]
                if delta < 0:
                    changed_better += 1
                    help_s -= delta
                elif delta > 0:
                    changed_worse += 1
                    harm_s += delta
        active.append(gap)

    if not rows["tool"]:
        raise ValueError("evaluation corpus has no overlapping paused sessions")
    metrics: dict[str, dict[str, float]] = {}
    for arm, arm_rows in rows.items():
        regrets = np.asarray([regret for _, regret in arm_rows])
        metrics[arm] = {
            "mean_regret_s": float(np.mean(regrets)),
            "top1_oracle_agreement": correct[arm] / len(arm_rows),
            "p95_regret_s": float(np.quantile(regrets, 0.95)),
            "p99_regret_s": float(np.quantile(regrets, 0.99)),
            "max_regret_s": float(np.max(regrets)),
        }
    coverage = {
        "ranking_events": len(rows["tool"]),
        "ranking_events_with_causal_progress": ranking_events_with_progress,
        "changed_choices": changed_choices,
    }
    diagnostics: dict[str, float | int] = {
        "changed_better": changed_better,
        "changed_worse": changed_worse,
        "cumulative_help_s": help_s,
        "cumulative_harm_s": harm_s,
        "point_candidate_rows": len(point_baseline_errors),
        "tool_point_mae_s": float(np.mean(point_baseline_errors)),
        "progress_point_mae_s": float(np.mean(point_progress_errors)),
        "tool_point_bias_s": float(np.mean(point_baseline_biases)),
        "progress_point_bias_s": float(np.mean(point_progress_biases)),
    }
    return metrics, rows, coverage, diagnostics


def run(fit_root: Path, eval_root: Path) -> dict[str, Any]:
    fit = load_gaps(fit_root)
    evaluation = load_gaps(eval_root)
    features, labels = _training_rows(fit)
    model = HistGradientBoostingRegressor(random_state=0).fit(features, labels)
    global_history, tool_history = _histories(fit)
    metrics, rows, coverage, diagnostics = evaluate(
        evaluation,
        model=model,
        global_history=global_history,
        tool_history=tool_history,
    )
    primary = _cluster_bootstrap_delta(rows["progress"], rows["tool"])
    return {
        "status": "development-exposed diagnostic",
        "question": "Do causal in-exec CPU/network counters improve observed-overlap remaining-time ordering over tool-name survival?",
        "protocol": {
            "fit_root": str(fit_root.resolve()),
            "eval_root": str(eval_root.resolve()),
            "fit_unit": "one row per observed resource-timeline sample endpoint in a single-exec gap",
            "eval_unit": "all observed-overlap gap-start ranking events; unsupported gaps fall back to tool-name survival",
            "features": list(FEATURE_NAMES),
            "model": "sklearn HistGradientBoostingRegressor defaults, random_state=0",
            "primary": "mean_regret(progress) - mean_regret(tool)",
            "go_criterion": "upper endpoint of 95% unordered-task-pair cluster-bootstrap CI < 0",
            "sample_availability": "tool action start + recorder offset + frozen 0.050 s pad; action end is used only for a fail-closed bound audit",
        },
        "data": {
            "fit_gaps": len(fit),
            "fit_single_exec_gaps_with_progress": sum(bool(gap.points) for gap in fit),
            "fit_progress_rows": len(labels),
            "eval_gaps": len(evaluation),
            **coverage,
            "task_pair_clusters": len({pair for pair, _ in rows["tool"]}),
            "max_concurrency": max(len(pair) for pair, _ in rows["tool"]),
        },
        "metrics": metrics,
        "mechanism_diagnostics": diagnostics,
        "primary_comparison": primary,
        "decision": "JUSTIFIES independent validation"
        if primary["go"]
        else "STOP existing-counter direction",
        "review": "Independent review found no critical or major issue after a causal timestamp fix and focused re-review.",
        "limitations": [
            "Both SWE corpora and fresh-277 outcomes are development-exposed.",
            "Counters are whole task-container cgroup deltas during a single exec, not clause-exclusive attribution.",
            "Observed ranking concurrency is two and no actual KV memory pressure or eviction is present.",
            "CPU/network progress cannot establish the value of uncollected time-resolved disk, RSS, or process-phase signals.",
        ],
    }


def self_check() -> None:
    action = {
        "ts_start": 10.0,
        "ts_end": 12.003,
        "data": {
            "resource_timeline": {
                "version": 1,
                "samples": [
                    {
                        "offset_s": 1.0,
                        "dt_s": 1.0,
                        "cpu_core_s": 0.5,
                        "cpu_opportunity_core_s": 2.0,
                        "cpu_quota_cores": 2.0,
                        "net_rx_bytes": 9,
                        "net_tx_bytes": 3,
                    },
                    {
                        "offset_s": 2.0,
                        "dt_s": 1.0,
                        "cpu_core_s": 1.0,
                        "cpu_opportunity_core_s": 2.0,
                        "cpu_quota_cores": 2.0,
                        "net_rx_bytes": 0,
                        "net_tx_bytes": 7,
                    },
                ],
            }
        },
    }
    points = _progress_points(gap_start=9.8, gap_end=12.3, action=action)
    assert [point.available_at for point in points] == [11.05, 12.05]
    assert points[-1].features[2] == 1.5
    assert points[-1].features[5] == 0.375
    gap = ProgressGap("s", 9.8, 12.3, '["exec"]', points)
    assert _latest_point(gap, 11.049) is None
    assert _latest_point(gap, 11.05) == points[0]
    assert _latest_point(gap, 12.05) == points[1]
    try:
        _progress_points(
            gap_start=9.8,
            gap_end=12.3,
            action={**action, "ts_end": 12.2},
        )
    except ValueError:
        pass
    else:
        raise AssertionError("availability audit did not fail closed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument(
        "--fit-root",
        type=Path,
        default=Path("traces/swe-rebench/qwen3.7-max/offline-gated-confirm-100-v2"),
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("traces/swe-rebench/qwen3.7-max/fresh-seed42-skip150-n200"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "analysis/results/cachewise-progress-counter-20260731/result.json"
        ),
    )
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("self-check passed")
        return
    result = run(args.fit_root, args.eval_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
