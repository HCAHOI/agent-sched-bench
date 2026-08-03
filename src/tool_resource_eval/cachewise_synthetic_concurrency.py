"""Evaluate CacheWise ordering over one fixed family of synthetic arrivals."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tool_resource_eval.cachewise_reproduction import (
    CLUSTER_COUNTS,
    Gap,
    _histories,
    evaluate,
    fit_clusters,
    load_gaps,
)
from tool_resource_eval.cachewise_repo_locality import (
    RankingRow,
    _arm_metrics,
    _configured_task_ids,
    _fit_history,
    _partition_gaps,
    evaluate as evaluate_repo,
    repo_key,
)


TARGET_CONCURRENCY = 32
ARRIVAL_SEED = 0
SCHEDULE_SEEDS = tuple(range(32))
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 0


def _by_session(gaps: Iterable[Gap]) -> dict[str, list[Gap]]:
    grouped: dict[str, list[Gap]] = defaultdict(list)
    for gap in gaps:
        grouped[gap.session_id].append(gap)
    return grouped


def _partition_task_ids(
    canonical_ids: set[str], selected_ids: list[str]
) -> tuple[list[str], list[str]]:
    selected = set(selected_ids)
    if len(selected) != len(selected_ids):
        raise ValueError("split task manifest contains duplicate IDs")
    unknown = selected - canonical_ids
    if unknown:
        raise ValueError(f"split task manifest contains unknown IDs: {sorted(unknown)}")
    fit = canonical_ids - selected
    if not selected or not fit:
        raise ValueError(
            "split task manifest must leave non-empty fit and evaluation sets"
        )
    return sorted(fit), sorted(selected)


def _load_task_subset(root: Path, task_ids: list[str]) -> tuple[list[Gap], int]:
    gaps: list[Gap] = []
    observed_tasks = 0
    for task_id in task_ids:
        task_root = root / task_id
        try:
            task_gaps = load_gaps(task_root)
        except ValueError as error:
            if str(error) != f"no tool gaps found under {task_root}":
                raise
            continue
        observed_tasks += 1
        gaps.extend(replace(gap, session_id=task_id) for gap in task_gaps)
    gaps.sort(key=lambda gap: (gap.start, gap.end, gap.session_id))
    return gaps, observed_tasks


def synthetic_schedule(
    fit: list[Gap],
    evaluation: list[Gap],
    *,
    target_concurrency: int = TARGET_CONCURRENCY,
    seed: int = ARRIVAL_SEED,
) -> tuple[list[Gap], dict[str, float | int]]:
    if target_concurrency < 1:
        raise ValueError("target_concurrency must be positive")
    fit_sessions = _by_session(fit)
    eval_sessions = _by_session(evaluation)
    fit_spans = [
        max(gap.end for gap in rows) - min(gap.start for gap in rows)
        for rows in fit_sessions.values()
    ]
    if not fit_spans or not eval_sessions:
        raise ValueError("fit and evaluation must contain session gaps")
    interarrival_s = float(np.median(fit_spans)) / target_concurrency

    order = sorted(eval_sessions)
    np.random.default_rng(seed).shuffle(order)
    shifted: list[Gap] = []
    windows: list[tuple[float, float]] = []
    for index, session_id in enumerate(order):
        rows = eval_sessions[session_id]
        origin = min(gap.start for gap in rows)
        arrival = index * interarrival_s
        session_end = arrival + max(gap.end for gap in rows) - origin
        windows.append((arrival, session_end))
        shifted.extend(
            replace(
                gap,
                start=arrival + gap.start - origin,
                end=arrival + gap.end - origin,
            )
            for gap in rows
        )
    shifted.sort(key=lambda gap: (gap.start, gap.end, gap.session_id))

    events = sorted(
        [(start, 1) for start, _ in windows] + [(end, -1) for _, end in windows]
    )
    live = max_live = 0
    for _, delta in events:
        live += delta
        max_live = max(max_live, live)
    horizon = max(end for _, end in windows) - min(start for start, _ in windows)
    return shifted, {
        "target_concurrency": target_concurrency,
        "arrival_seed": seed,
        "fit_median_gap_span_s": float(np.median(fit_spans)),
        "interarrival_s": interarrival_s,
        "realized_max_live_sessions": max_live,
        "realized_mean_live_sessions": sum(end - start for start, end in windows)
        / horizon,
    }


def _schedule_bootstrap(deltas: list[float]) -> dict[str, Any]:
    values = np.asarray(deltas)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.choice(values, size=(BOOTSTRAP_DRAWS, len(values)), replace=True)
    low, high = np.quantile(np.mean(sampled, axis=1), [0.025, 0.975])
    return {
        "delta_mean_regret_s": float(np.mean(values)),
        "ci95_schedule_bootstrap_s": [float(low), float(high)],
        "schedule_repetitions": len(values),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "go": bool(high < 0.0),
    }


def _repo_delta(rows: Iterable[RankingRow]) -> float:
    return float(
        np.mean([row.regrets["repo_c100"] - row.regrets["repo_tool"] for row in rows])
    )


def run_same_repo_cohort(trace_root: Path, corpus_config: Path) -> dict[str, Any]:
    config = json.loads(corpus_config.read_text(encoding="utf-8"))
    fit_ids = _configured_task_ids(config, "fit")
    eval_ids = _configured_task_ids(config, "eval")
    overlap = set(fit_ids) & set(eval_ids)
    if overlap:
        raise ValueError(f"fit/evaluation task IDs overlap: {sorted(overlap)}")
    repositories = {repo_key(task_id) for task_id in (*fit_ids, *eval_ids)}
    if len(repositories) != 1:
        raise ValueError(
            f"configured cohort must contain one repository: {sorted(repositories)}"
        )
    repository = next(iter(repositories))

    followups = config.get("followup_evaluations")
    protocol = followups.get("synthetic_c32") if isinstance(followups, dict) else None
    if not isinstance(protocol, dict):
        raise ValueError("corpus config lacks followup_evaluations.synthetic_c32")
    if protocol.get("arms") != ["tool_name", "c100"]:
        raise ValueError("synthetic_c32 arms must be tool_name and c100")
    target = protocol.get("target_concurrency")
    seeds = protocol.get("schedule_seeds")
    if not isinstance(target, int) or target < 1:
        raise ValueError("synthetic_c32 target_concurrency must be positive")
    if (
        not isinstance(seeds, list)
        or not seeds
        or not all(isinstance(seed, int) for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise ValueError("synthetic_c32 schedule_seeds must be unique integers")

    fit, evaluation = _partition_gaps(load_gaps(trace_root), fit_ids, eval_ids)
    fit_observed = {gap.session_id for gap in fit}
    eval_observed = {gap.session_id for gap in evaluation}
    if fit_observed != set(fit_ids) or eval_observed != set(eval_ids):
        raise ValueError(
            "configured tasks without gaps: "
            f"fit={sorted(set(fit_ids) - fit_observed)}, "
            f"eval={sorted(set(eval_ids) - eval_observed)}"
        )

    history = _fit_history(fit)
    pooled_cache: dict[tuple[int, str, str], int] = {}
    local_caches: dict[str, dict[tuple[int, str, str], int]] = {repository: {}}
    schedule_rows: list[dict[str, Any]] = []
    deletion_deltas: dict[str, list[float]] = defaultdict(list)
    for seed in seeds:
        scheduled, schedule = synthetic_schedule(
            fit, evaluation, target_concurrency=target, seed=seed
        )
        rows = evaluate_repo(
            scheduled,
            history,
            {repository: history},
            pooled_cache=pooled_cache,
            local_caches=local_caches,
        )
        metrics = _arm_metrics(rows)
        delta = _repo_delta(rows)
        schedule_rows.append(
            {
                "seed": seed,
                "ranking_events": len(rows),
                "choice_changes": sum(
                    row.choices["repo_c100"] != row.choices["repo_tool"] for row in rows
                ),
                "tool_name_mean_regret_s": metrics["repo_tool"]["mean_regret_s"],
                "c100_mean_regret_s": metrics["repo_c100"]["mean_regret_s"],
                "delta_mean_regret_s": delta,
                "tool_name_top1_oracle_agreement": metrics["repo_tool"][
                    "top1_oracle_agreement"
                ],
                "c100_top1_oracle_agreement": metrics["repo_c100"][
                    "top1_oracle_agreement"
                ],
                "realized_mean_live_sessions": schedule["realized_mean_live_sessions"],
                "realized_max_live_sessions": schedule["realized_max_live_sessions"],
                "max_simultaneous_active_gaps": max(len(row.task_pair) for row in rows),
            }
        )
        for task in eval_ids:
            retained = [gap for gap in scheduled if gap.session_id != task]
            deletion_rows = evaluate_repo(
                retained,
                history,
                {repository: history},
                pooled_cache=pooled_cache,
                local_caches=local_caches,
            )
            deletion_deltas[task].append(_repo_delta(deletion_rows))

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
    arm_metrics = {
        "tool_name": {
            "mean_regret_s": float(
                np.mean([row["tool_name_mean_regret_s"] for row in schedule_rows])
            ),
            "top1_oracle_agreement": float(
                np.mean(
                    [row["tool_name_top1_oracle_agreement"] for row in schedule_rows]
                )
            ),
        },
        "c100": {
            "mean_regret_s": float(
                np.mean([row["c100_mean_regret_s"] for row in schedule_rows])
            ),
            "top1_oracle_agreement": float(
                np.mean([row["c100_top1_oracle_agreement"] for row in schedule_rows])
            ),
        },
    }

    return {
        "status": "development-exposed synthetic same-repository mechanism test",
        "question": protocol.get("question"),
        "protocol": {
            "trace_root": str(trace_root.resolve()),
            "corpus_config": str(corpus_config.resolve()),
            "collection_id": config.get("collection_id"),
            "fit_selection": "configured fit.task_ids only; fit once before all schedules",
            "eval_selection": "configured eval.task_ids only",
            "schedule": protocol.get("arrival_model"),
            "arms": protocol.get("arms"),
            "primary": protocol.get("primary_metric"),
            "primary_go_criterion": protocol.get("primary_go_criterion"),
            "task_stability_go_criterion": protocol.get("task_stability_go_criterion"),
            "task_deletion": "remove one task after assigning arrivals; preserve all other task timestamps and recompute every remaining ranking event without refitting",
        },
        "data": {
            "repository": repository,
            "fit_tasks": len(fit_ids),
            "fit_gaps": len(fit),
            "eval_tasks": len(eval_ids),
            "eval_gaps": len(evaluation),
            "schedule_repetitions": len(seeds),
            "schedule_seeds": seeds,
            "target_arrival_rate_concurrency": target,
            "fit_median_session_span_s": schedule["fit_median_gap_span_s"],
            "interarrival_s": schedule["interarrival_s"],
            "realized_mean_live_sessions_mean": float(
                np.mean([row["realized_mean_live_sessions"] for row in schedule_rows])
            ),
            "realized_max_live_sessions_range": [
                min(row["realized_max_live_sessions"] for row in schedule_rows),
                max(row["realized_max_live_sessions"] for row in schedule_rows),
            ],
            "ranking_events_mean": float(
                np.mean([row["ranking_events"] for row in schedule_rows])
            ),
            "ranking_events_range": [
                min(row["ranking_events"] for row in schedule_rows),
                max(row["ranking_events"] for row in schedule_rows),
            ],
            "max_simultaneous_active_gaps": max(
                row["max_simultaneous_active_gaps"] for row in schedule_rows
            ),
        },
        "schedule_mean_metrics": arm_metrics,
        "schedule_results": schedule_rows,
        "primary_comparison": primary,
        "task_deletion_results": task_deletions,
        "task_stability": {
            "evaluated_tasks": len(task_deletions),
            "worst_deletion": worst_deletion,
            "go": stability_go,
        },
        "decision": (
            "GO to one live pressure experiment"
            if primary["go"] and stability_go
            else "STOP before live pressure or scheduler experiments"
        ),
        "review": "Independent bounded review found no critical or major issue; its bootstrap-seed provenance minor was fixed before the formal run.",
        "limitations": [
            protocol.get("evidence_boundary"),
            "The target-32 value controls arrival rate; with 24 evaluation tasks, realized peak live sessions is at most 24.",
            "The schedules and evaluation cohort are development-exposed.",
            "The schedule bootstrap measures arrival-permutation sensitivity conditional on fixed tasks, not task-population uncertainty.",
            "Delete-one-task sensitivity is not an independent confirmation set.",
        ],
    }


def run(
    fit_root: Path,
    eval_root: Path,
    *,
    split_task_ids: Path | None = None,
) -> dict[str, Any]:
    selection: dict[str, Any] = {}
    if split_task_ids is None:
        fit = load_gaps(fit_root)
        evaluation = load_gaps(eval_root)
    else:
        if fit_root.resolve() != eval_root.resolve():
            raise ValueError(
                "manifest split requires identical fit and evaluation roots"
            )
        canonical_ids = {path.name for path in fit_root.iterdir() if path.is_dir()}
        selected_ids = [
            line.strip()
            for line in split_task_ids.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        fit_ids, eval_ids = _partition_task_ids(canonical_ids, selected_ids)
        fit, fit_observed = _load_task_subset(fit_root, fit_ids)
        evaluation, eval_observed = _load_task_subset(eval_root, eval_ids)
        selection = {
            "split_task_ids": str(split_task_ids.resolve()),
            "fit_tasks_declared": len(fit_ids),
            "fit_tasks_with_gaps": fit_observed,
            "eval_tasks_declared": len(eval_ids),
            "eval_tasks_with_gaps": eval_observed,
        }
    global_history, tool_history = _histories(fit)
    clusters, occupied = fit_clusters(fit)
    label_cache: dict[tuple[int, str, str], int] = {}
    schedule_rows: list[dict[str, float | int]] = []
    metric_rows: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    deltas: list[float] = []
    for seed in SCHEDULE_SEEDS:
        scheduled, schedule = synthetic_schedule(fit, evaluation, seed=seed)
        metrics, rows = evaluate(
            scheduled,
            global_history,
            tool_history,
            clusters,
            label_cache=label_cache,
        )
        for arm, arm_rows in rows.items():
            regrets = np.asarray([regret for _, regret in arm_rows])
            metrics[arm].update(
                p95_regret_s=float(np.quantile(regrets, 0.95)),
                p99_regret_s=float(np.quantile(regrets, 0.99)),
                max_regret_s=float(np.max(regrets)),
            )
            for metric, value in metrics[arm].items():
                metric_rows[arm][metric].append(value)
        delta = metrics["c100"]["mean_regret_s"] - metrics["tool"]["mean_regret_s"]
        deltas.append(delta)
        schedule_rows.append(
            {
                "seed": seed,
                "ranking_events": len(rows["tool"]),
                "delta_mean_regret_s": delta,
                "realized_mean_live_sessions": schedule["realized_mean_live_sessions"],
                "realized_max_live_sessions": schedule["realized_max_live_sessions"],
                "max_simultaneous_active_gaps": max(
                    len(candidates) for candidates, _ in rows["tool"]
                ),
            }
        )
    primary = _schedule_bootstrap(deltas)
    metrics = {
        arm: {metric: float(np.mean(values)) for metric, values in arm_metrics.items()}
        for arm, arm_metrics in metric_rows.items()
    }
    first_schedule = synthetic_schedule(fit, evaluation, seed=SCHEDULE_SEEDS[0])[1]

    return {
        "status": "development-exposed synthetic-concurrency diagnostic",
        "question": "Does CacheWise C100 improve victim ordering in expectation over 32 fixed c32 trace-driven arrival streams?",
        "protocol": {
            "fit_root": str(fit_root.resolve()),
            "eval_root": str(eval_root.resolve()),
            **({"split_task_ids": selection["split_task_ids"]} if selection else {}),
            "schedule": "for seeds 0-31, shuffle sorted session IDs with NumPy default_rng(seed); open-loop arrivals at fit-median-gap-span / 32; preserve within-session gap offsets and durations",
            "arms": ["global", "tool", "c20", "c50", "c100"],
            "tfidf_max_features": 5000,
            "kmeans_random_state": 0,
            "primary": "mean_regret(c100) - mean_regret(tool)",
            "go_criterion": "upper endpoint of 95% paired schedule-level bootstrap CI < 0",
        },
        "data": {
            **selection,
            "fit_gaps": len(fit),
            "fit_sessions": len(_by_session(fit)),
            "eval_gaps": len(evaluation),
            "eval_sessions": len(_by_session(evaluation)),
            "schedule_repetitions": len(SCHEDULE_SEEDS),
            "schedule_seeds": list(SCHEDULE_SEEDS),
            "target_concurrency": TARGET_CONCURRENCY,
            "fit_median_gap_span_s": first_schedule["fit_median_gap_span_s"],
            "interarrival_s": first_schedule["interarrival_s"],
            "ranking_events_mean": float(
                np.mean([row["ranking_events"] for row in schedule_rows])
            ),
            "ranking_events_range": [
                min(row["ranking_events"] for row in schedule_rows),
                max(row["ranking_events"] for row in schedule_rows),
            ],
            "realized_mean_live_sessions_mean": float(
                np.mean([row["realized_mean_live_sessions"] for row in schedule_rows])
            ),
            "realized_max_live_sessions_range": [
                min(row["realized_max_live_sessions"] for row in schedule_rows),
                max(row["realized_max_live_sessions"] for row in schedule_rows),
            ],
            "max_simultaneous_active_gaps": max(
                row["max_simultaneous_active_gaps"] for row in schedule_rows
            ),
        },
        "cluster_models": {
            f"c{count}": {
                "tool_batch_keys": len(clusters[count]),
                "occupied_clusters": sum(occupied[count].values()),
            }
            for count in CLUSTER_COUNTS
        },
        "schedule_mean_metrics": metrics,
        "schedule_results": schedule_rows,
        "primary_comparison": primary,
        "decision": "JUSTIFIES independent synthetic validation"
        if primary["go"]
        else "STOP synthetic-concurrency direction",
        "review": "Independent bounded reviews found no critical or major issue in schedule uncertainty or manifest-split provenance.",
        "limitations": [
            "The fit/evaluation traces and these schedules are development-exposed.",
            "The interval measures arrival-permutation sensitivity conditional on these fixed tasks, not task-population uncertainty.",
            "The simulation holds isolated-trace durations fixed and does not model resource contention.",
            "No KV capacity, eviction/recomputation feedback, eviction count, or end-to-end JCT is modeled.",
            "A positive result cannot reverse the negative observed-overlap result.",
        ],
    }


def self_check() -> None:
    assert _partition_task_ids({"a", "b", "c"}, ["b"]) == (["a", "c"], ["b"])
    fit = [
        Gap("fit-a", 10.0, 18.0, "exec", "a"),
        Gap("fit-b", 20.0, 32.0, "exec", "b"),
    ]
    evaluation = [
        Gap("x", 100.0, 102.0, "exec", "x1"),
        Gap("x", 104.0, 105.0, "exec", "x2"),
        Gap("y", 200.0, 203.0, "exec", "y"),
    ]
    scheduled, metadata = synthetic_schedule(
        fit, evaluation, target_concurrency=2, seed=0
    )
    assert metadata["interarrival_s"] == 5.0
    starts = {
        gap.session_id: min(
            row.start for row in scheduled if row.session_id == gap.session_id
        )
        for gap in scheduled
    }
    assert sorted(starts.values()) == [0.0, 5.0]
    assert sorted(gap.end - gap.start for gap in scheduled) == [1.0, 2.0, 3.0]
    assert _schedule_bootstrap([-1.0] * 32)["go"] is True


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
    )
    parser.add_argument(
        "--split-task-ids",
        type=Path,
        help="evaluate listed task directories and fit on their complement",
    )
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--corpus-config", type=Path)
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("self-check passed")
        return
    if (args.trace_root is None) != (args.corpus_config is None):
        parser.error("--trace-root and --corpus-config must be provided together")
    if args.trace_root is not None:
        if args.output is None:
            parser.error("--output is required for a configured cohort run")
        result = run_same_repo_cohort(args.trace_root, args.corpus_config)
    else:
        result = run(
            args.fit_root,
            args.eval_root,
            split_task_ids=args.split_task_ids,
        )
    output = args.output or Path(
        "analysis/results/cachewise-swe-synthetic-c32-20260731/result.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
