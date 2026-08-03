"""Reproduce CacheWise's argument-cluster reuse ordering on SWE traces."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable
import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer

from trace_collect.tool_gap_extractor import (
    discover_trace_files,
    extract_tool_gap_windows,
)


CLUSTER_COUNTS = (20, 50, 100)


@dataclass(frozen=True)
class Gap:
    session_id: str
    start: float
    end: float
    tool_key: str
    args_text: str


@dataclass(frozen=True)
class ClusterModel:
    vectorizer: TfidfVectorizer
    kmeans: KMeans
    durations: tuple[np.ndarray, ...]


def _parse_args(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _action_index(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    llm: dict[str, dict[str, Any]] = {}
    tools: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("type") != "action":
                continue
            action_id = str(row.get("action_id") or "")
            if row.get("action_type") == "llm_call":
                llm[action_id] = row
            elif row.get("action_type") == "tool_exec":
                data = row.get("data") or {}
                call_id = str(data.get("tool_call_id") or action_id)
                tools[call_id] = row
    return llm, tools


def load_gaps(root: Path) -> list[Gap]:
    gaps: list[Gap] = []
    for trace_path in discover_trace_files([root]):
        windows = extract_tool_gap_windows(trace_path)
        if not windows:
            continue
        llm, tools = _action_index(trace_path)
        for window in windows:
            start = float(llm[window.llm_action_id]["ts_end"])
            end = float(llm[window.next_llm_action_id]["ts_start"])
            batch_args: list[Any] = []
            for call_id in window.tool_call_ids:
                data = tools[call_id].get("data") or {}
                batch_args.append(_parse_args(data.get("tool_args")))
            gaps.append(
                Gap(
                    session_id=window.instance_id or window.source_trace,
                    start=start,
                    end=end,
                    tool_key=json.dumps(window.tool_names, separators=(",", ":")),
                    args_text=json.dumps(
                        batch_args,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
            )
    gaps.sort(key=lambda gap: (gap.start, gap.end, gap.session_id))
    if not gaps:
        raise ValueError(f"no tool gaps found under {root}")
    return gaps


def _histories(gaps: Iterable[Gap]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    by_tool: dict[str, list[float]] = defaultdict(list)
    durations: list[float] = []
    for gap in gaps:
        duration = gap.end - gap.start
        durations.append(duration)
        by_tool[gap.tool_key].append(duration)
    return np.asarray(durations), {
        tool: np.asarray(values) for tool, values in by_tool.items()
    }


def fit_clusters(
    gaps: list[Gap],
    *,
    cluster_counts: Iterable[int] = CLUSTER_COUNTS,
) -> tuple[dict[int, dict[str, ClusterModel]], dict[int, dict[str, int]]]:
    counts = tuple(cluster_counts)
    if not counts or any(count < 1 for count in counts):
        raise ValueError("cluster_counts must contain positive integers")
    grouped: dict[str, list[Gap]] = defaultdict(list)
    for gap in gaps:
        grouped[gap.tool_key].append(gap)

    models = {count: {} for count in counts}
    occupied = {count: {} for count in counts}
    for tool_key, rows in grouped.items():
        texts = [row.args_text for row in rows]
        vectorizer = TfidfVectorizer(max_features=5000)
        try:
            matrix = vectorizer.fit_transform(texts)
        except ValueError:
            continue
        for count in counts:
            n_clusters = min(count, len(rows), len(set(texts)))
            if n_clusters < 1:
                continue
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                labels = kmeans.fit_predict(matrix)
            cluster_durations = tuple(
                np.asarray(
                    [
                        row.end - row.start
                        for row, label in zip(rows, labels, strict=True)
                        if label == cluster
                    ]
                )
                for cluster in range(n_clusters)
            )
            models[count][tool_key] = ClusterModel(
                vectorizer=vectorizer,
                kmeans=kmeans,
                durations=cluster_durations,
            )
            occupied[count][tool_key] = sum(
                len(values) > 0 for values in cluster_durations
            )
    return models, occupied


def _remaining_mean(durations: np.ndarray, elapsed: float) -> float | None:
    survivors = durations[durations > elapsed]
    if not len(survivors):
        return None
    return float(np.mean(survivors - elapsed))


def _predict(
    gap: Gap,
    elapsed: float,
    arm: str,
    global_history: np.ndarray,
    tool_history: dict[str, np.ndarray],
    clusters: dict[int, dict[str, ClusterModel]],
    label_cache: dict[tuple[int, str, str], int] | None = None,
) -> float:
    if arm.startswith("c"):
        count = int(arm[1:])
        model = clusters[count].get(gap.tool_key)
        if model is not None:
            cache_key = (count, gap.tool_key, gap.args_text)
            label = label_cache.get(cache_key) if label_cache is not None else None
            if label is None:
                label = int(
                    model.kmeans.predict(model.vectorizer.transform([gap.args_text]))[0]
                )
                if label_cache is not None:
                    label_cache[cache_key] = label
            prediction = _remaining_mean(model.durations[label], elapsed)
            if prediction is not None:
                return prediction
    if arm != "global":
        prediction = _remaining_mean(
            tool_history.get(gap.tool_key, np.asarray([])), elapsed
        )
        if prediction is not None:
            return prediction
    return _remaining_mean(global_history, elapsed) or 0.0


def evaluate(
    gaps: list[Gap],
    global_history: np.ndarray,
    tool_history: dict[str, np.ndarray],
    clusters: dict[int, dict[str, ClusterModel]],
    *,
    label_cache: dict[tuple[int, str, str], int] | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, list[tuple[tuple[str, ...], float]]]]:
    arms = ("global", "tool", "c20", "c50", "c100")
    active: list[Gap] = []
    rows: dict[str, list[tuple[tuple[str, ...], float]]] = {arm: [] for arm in arms}
    correct = {arm: 0 for arm in arms}
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
            for arm in arms:
                chosen = max(
                    candidates,
                    key=lambda candidate: (
                        _predict(
                            candidate,
                            gap.start - candidate.start,
                            arm,
                            global_history,
                            tool_history,
                            clusters,
                            label_cache,
                        ),
                        candidate.session_id,
                    ),
                )
                regret = oracle_remaining - (chosen.end - gap.start)
                rows[arm].append((pair, regret))
                correct[arm] += regret <= 1e-12
        active.append(gap)

    if not rows["tool"]:
        raise ValueError("evaluation corpus has no overlapping paused sessions")
    metrics = {
        arm: {
            "mean_regret_s": float(np.mean([regret for _, regret in arm_rows])),
            "top1_oracle_agreement": correct[arm] / len(arm_rows),
        }
        for arm, arm_rows in rows.items()
    }
    return metrics, rows


def _cluster_bootstrap_delta(
    candidate: list[tuple[tuple[str, ...], float]],
    baseline: list[tuple[tuple[str, ...], float]],
    *,
    draws: int = 10_000,
) -> dict[str, Any]:
    if [pair for pair, _ in candidate] != [pair for pair, _ in baseline]:
        raise ValueError("candidate and baseline decisions do not align")
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for (pair, candidate_regret), (_, baseline_regret) in zip(
        candidate, baseline, strict=True
    ):
        grouped[pair].append(candidate_regret - baseline_regret)
    clusters = list(grouped.values())
    rng = np.random.default_rng(0)
    boot = np.empty(draws)
    for draw in range(draws):
        sampled = rng.integers(0, len(clusters), len(clusters))
        total = sum(sum(clusters[index]) for index in sampled)
        count = sum(len(clusters[index]) for index in sampled)
        boot[draw] = total / count
    point = float(np.mean([value for values in clusters for value in values]))
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "delta_mean_regret_s": point,
        "ci95_task_pair_cluster_bootstrap_s": [float(low), float(high)],
        "task_pair_clusters": len(clusters),
        "bootstrap_draws": draws,
        "go": bool(high < 0.0),
    }


def run(fit_root: Path, eval_root: Path) -> dict[str, Any]:
    fit = load_gaps(fit_root)
    evaluation = load_gaps(eval_root)
    global_history, tool_history = _histories(fit)
    clusters, occupied = fit_clusters(fit)
    metrics, rows = evaluate(evaluation, global_history, tool_history, clusters)
    primary = _cluster_bootstrap_delta(rows["c100"], rows["tool"])
    return {
        "status": "development-exposed diagnostic",
        "question": "Does CacheWise C100 whole-argument clustering improve observed-overlap pairwise reuse ordering over tool-name history?",
        "protocol": {
            "fit_root": str(fit_root.resolve()),
            "eval_root": str(eval_root.resolve()),
            "unit": "LLM-end to next-LLM-start tool gap; parallel outer calls form one ordered batch",
            "arms": ["global", "tool", "c20", "c50", "c100"],
            "tfidf_max_features": 5000,
            "kmeans_random_state": 0,
            "primary": "mean_regret(c100) - mean_regret(tool)",
            "go_criterion": "upper endpoint of 95% unordered-task-pair cluster-bootstrap CI < 0",
        },
        "data": {
            "fit_gaps": len(fit),
            "eval_gaps": len(evaluation),
            "ranking_events": len(rows["tool"]),
            "task_pair_clusters": len({pair for pair, _ in rows["tool"]}),
            "max_concurrency": max(len(pair) for pair, _ in rows["tool"]),
        },
        "cluster_models": {
            f"c{count}": {
                "tool_batch_keys": len(clusters[count]),
                "occupied_clusters": sum(occupied[count].values()),
            }
            for count in CLUSTER_COUNTS
        },
        "metrics": metrics,
        "primary_comparison": primary,
        "decision": "GO to one live pressure experiment"
        if primary["go"]
        else "STOP; do not descend to clause features to rescue the result",
        "review": "Independent bounded review found no critical or major issue; its terminology-only minor was fixed before this run.",
        "limitations": [
            "Both SWE corpora are development-exposed.",
            "Observed concurrency is two, not CacheWise's 30-50 sessions.",
            "This tests a hypothetical victim ranking on observed overlapping gaps; no memory pressure or actual eviction was observed.",
            "This does not test eviction count or end-to-end completion time.",
            "Some repositories overlap between fit and evaluation.",
        ],
    }


def self_check() -> None:
    history = np.asarray([1.0, 3.0, 5.0])
    assert _remaining_mean(history, 2.0) == 2.0
    assert _remaining_mean(history, 5.0) is None

    fit = [
        Gap("a", 0.0, duration, "exec", text)
        for text, duration in [
            ("quick status", 1.0),
            ("quick status", 1.2),
            ("long test suite", 9.0),
            ("long test suite", 10.0),
        ]
    ]
    global_history, tool_history = _histories(fit)
    clusters, _ = fit_clusters(fit)
    quick = Gap("q", 20.0, 21.0, "exec", "quick status")
    slow = Gap("s", 20.0, 30.0, "exec", "long test suite")
    assert _predict(
        slow, 0.0, "c20", global_history, tool_history, clusters
    ) > _predict(quick, 0.0, "c20", global_history, tool_history, clusters)


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
            "analysis/results/cachewise-swe-reproduction-20260731/result.json"
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
