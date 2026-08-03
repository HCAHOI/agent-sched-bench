"""Test whether task-disjoint same-repository history rescues CacheWise C100."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np

from tool_resource_eval.cachewise_reproduction import (
    ClusterModel,
    Gap,
    _cluster_bootstrap_delta,
    _histories,
    _predict,
    fit_clusters,
    load_gaps,
)


ARMS = ("pooled_tool", "pooled_c100", "repo_tool", "repo_c100")
C100 = (100,)


@dataclass(frozen=True)
class History:
    global_durations: np.ndarray
    tool_durations: dict[str, np.ndarray]
    clusters: dict[int, dict[str, ClusterModel]]


@dataclass(frozen=True)
class RankingRow:
    task_pair: tuple[str, ...]
    repos: frozenset[str]
    regrets: dict[str, float]
    choices: dict[str, str]


def repo_key(session_id: str) -> str:
    match = re.fullmatch(r"(.+)-[0-9]+", session_id)
    if match is None:
        raise ValueError(f"SWE task ID lacks trailing issue number: {session_id}")
    return match.group(1)


def _fit_history(gaps: list[Gap]) -> History:
    global_durations, tool_durations = _histories(gaps)
    clusters, _ = fit_clusters(gaps, cluster_counts=C100)
    return History(global_durations, tool_durations, clusters)


def _fit_histories(
    fit: list[Gap], evaluation: list[Gap]
) -> tuple[History, dict[str, History]]:
    by_repo: dict[str, list[Gap]] = defaultdict(list)
    for gap in fit:
        by_repo[repo_key(gap.session_id)].append(gap)
    eval_repos = {repo_key(gap.session_id) for gap in evaluation}
    local = {
        repo: _fit_history(gaps)
        for repo, gaps in sorted(by_repo.items())
        if repo in eval_repos
    }
    return _fit_history(fit), local


def _prediction(
    gap: Gap,
    elapsed: float,
    arm: str,
    pooled: History,
    local: dict[str, History],
    pooled_cache: dict[tuple[int, str, str], int],
    local_caches: dict[str, dict[tuple[int, str, str], int]],
) -> float:
    is_local = arm.startswith("repo_")
    history = local[repo_key(gap.session_id)] if is_local else pooled
    model_arm = "c100" if arm.endswith("c100") else "tool"
    cache = local_caches[repo_key(gap.session_id)] if is_local else pooled_cache
    return _predict(
        gap,
        elapsed,
        model_arm,
        history.global_durations,
        history.tool_durations,
        history.clusters,
        cache,
    )


def evaluate(
    gaps: list[Gap], pooled: History, local: dict[str, History]
) -> list[RankingRow]:
    active: list[Gap] = []
    rows: list[RankingRow] = []
    pooled_cache: dict[tuple[int, str, str], int] = {}
    local_caches = {repo: {} for repo in local}
    for gap in gaps:
        active = [
            other
            for other in active
            if other.end > gap.start and other.session_id != gap.session_id
        ]
        candidates = [*active, gap]
        repos = frozenset(repo_key(candidate.session_id) for candidate in candidates)
        if len(candidates) >= 2 and repos <= local.keys():
            oracle_remaining = max(
                candidate.end - gap.start for candidate in candidates
            )
            regrets: dict[str, float] = {}
            choices: dict[str, str] = {}
            for arm in ARMS:
                chosen = max(
                    candidates,
                    key=lambda candidate: (
                        _prediction(
                            candidate,
                            gap.start - candidate.start,
                            arm,
                            pooled,
                            local,
                            pooled_cache,
                            local_caches,
                        ),
                        candidate.session_id,
                    ),
                )
                regrets[arm] = oracle_remaining - (chosen.end - gap.start)
                choices[arm] = chosen.session_id
            rows.append(
                RankingRow(
                    task_pair=tuple(
                        sorted(candidate.session_id for candidate in candidates)
                    ),
                    repos=repos,
                    regrets=regrets,
                    choices=choices,
                )
            )
        active.append(gap)
    if not rows:
        raise ValueError("no ranking events have same-repository fit history")
    return rows


def _comparison(
    rows: Iterable[RankingRow], candidate: str, baseline: str
) -> dict[str, Any]:
    selected = list(rows)
    candidate_rows = [(row.task_pair, row.regrets[candidate]) for row in selected]
    baseline_rows = [(row.task_pair, row.regrets[baseline]) for row in selected]
    result = _cluster_bootstrap_delta(candidate_rows, baseline_rows)
    changes = [
        row.regrets[candidate] - row.regrets[baseline]
        for row in selected
        if row.choices[candidate] != row.choices[baseline]
    ]
    result.update(
        choice_changes=len(changes),
        helpful_changes=sum(delta < 0 for delta in changes),
        harmful_changes=sum(delta > 0 for delta in changes),
        helpful_seconds=float(-sum(delta for delta in changes if delta < 0)),
        harmful_seconds=float(sum(delta for delta in changes if delta > 0)),
    )
    return result


def _repo_deletions(
    rows: list[RankingRow], evaluation_repos: Iterable[str]
) -> list[dict[str, Any]]:
    results = []
    for repo in sorted(set(evaluation_repos)):
        retained = [row for row in rows if repo not in row.repos]
        if not retained:
            raise ValueError(f"deleting {repo} leaves no ranking events")
        deltas = [
            row.regrets["repo_c100"] - row.regrets["repo_tool"] for row in retained
        ]
        results.append(
            {
                "excluded_repo": repo,
                "removed_events": len(rows) - len(retained),
                "ranking_events": len(retained),
                "delta_mean_regret_s": float(np.mean(deltas)),
            }
        )
    results.sort(key=lambda row: row["delta_mean_regret_s"], reverse=True)
    return results


def _task_deletions(
    rows: list[RankingRow], evaluation_tasks: Iterable[str], primary_delta: float
) -> list[dict[str, Any]]:
    results = []
    primary_sign = np.sign(primary_delta)
    for task in sorted(set(evaluation_tasks)):
        retained = [row for row in rows if task not in row.task_pair]
        if not retained:
            raise ValueError(f"deleting {task} leaves no ranking events")
        deltas = [
            row.regrets["repo_c100"] - row.regrets["repo_tool"] for row in retained
        ]
        delta = float(np.mean(deltas))
        results.append(
            {
                "excluded_task": task,
                "removed_events": len(rows) - len(retained),
                "ranking_events": len(retained),
                "delta_mean_regret_s": delta,
                "changes_primary_sign": bool(np.sign(delta) != primary_sign),
            }
        )
    results.sort(key=lambda row: row["delta_mean_regret_s"], reverse=True)
    return results


def _arm_metrics(rows: list[RankingRow]) -> dict[str, dict[str, float]]:
    return {
        arm: {
            "mean_regret_s": float(np.mean([row.regrets[arm] for row in rows])),
            "top1_oracle_agreement": float(
                np.mean([row.regrets[arm] <= 1e-12 for row in rows])
            ),
        }
        for arm in ARMS
    }


def run(fit_root: Path, eval_root: Path) -> dict[str, Any]:
    fit = load_gaps(fit_root)
    evaluation = load_gaps(eval_root)
    fit_tasks = {gap.session_id for gap in fit}
    eval_tasks = {gap.session_id for gap in evaluation}
    overlap = fit_tasks & eval_tasks
    if overlap:
        raise ValueError(f"fit/evaluation task IDs overlap: {sorted(overlap)}")
    pooled, local = _fit_histories(fit, evaluation)
    rows = evaluate(evaluation, pooled, local)
    primary = _comparison(rows, "repo_c100", "repo_tool")
    comparisons = {
        "pooled_c100_minus_pooled_tool": _comparison(
            rows, "pooled_c100", "pooled_tool"
        ),
        "repo_c100_minus_repo_tool": primary,
        "repo_tool_minus_pooled_tool": _comparison(rows, "repo_tool", "pooled_tool"),
        "repo_c100_minus_pooled_c100": _comparison(rows, "repo_c100", "pooled_c100"),
    }
    eval_repos = {repo_key(task) for task in eval_tasks}
    deletions = _repo_deletions(rows, eval_repos)
    stability_go = deletions[0]["delta_mean_regret_s"] < 0.0
    same_repo_rows = [row for row in rows if len(row.repos) == 1]
    return {
        "status": "development-exposed same-repository diagnostic",
        "question": "Does task-disjoint same-repository history make CacheWise C100 outperform a same-repository tool-name baseline?",
        "protocol": {
            "fit_root": str(fit_root.resolve()),
            "eval_root": str(eval_root.resolve()),
            "repo_key": "strip only the trailing -<issue number> from SWE task ID",
            "eligibility": "natural ranking event where every candidate repository has at least one task in fit",
            "repo_history": "fit tasks from only that candidate repository; exhausted cluster falls back to repo tool-name then repo global",
            "primary": "mean_regret(repo_c100) - mean_regret(repo_tool)",
            "go_criterion": "upper endpoint of 95% unordered-task-pair cluster-bootstrap CI < 0",
            "task_stability": "maximum mean primary delta after deleting every event involving one evaluation repository < 0",
        },
        "data": {
            "fit_tasks": len(fit_tasks),
            "fit_gaps": len(fit),
            "eval_tasks": len(eval_tasks),
            "eval_repositories": len(eval_repos),
            "eval_gaps": len(evaluation),
            "overlapping_repositories": len(local),
            "eligible_ranking_events": len(rows),
            "eligible_task_pairs": len({row.task_pair for row in rows}),
            "same_repo_ranking_events": len(same_repo_rows),
            "same_repo_task_pairs": len({row.task_pair for row in same_repo_rows}),
        },
        "metrics": _arm_metrics(rows),
        "comparisons": comparisons,
        "primary_comparison": primary,
        "repo_deletion_results": deletions,
        "repo_stability": {
            "evaluated_repositories": len(deletions),
            "worst_deletion": deletions[0],
            "go": stability_go,
        },
        "decision": "JUSTIFIES targeted multi-repository collection"
        if primary["go"] and stability_go
        else "STOP; do not collect new traces to rescue repository locality",
        "review": "Independent bounded review found no critical or major issue; its repo-deletion completeness and self-check minors were fixed before the formal run.",
        "limitations": [
            "Both SWE corpora and this repository-locality hypothesis are development-exposed.",
            "Most repository-local histories contain only one or two fit tasks.",
            "Natural overlap reaches concurrency two and contains almost no same-repository candidate pairs.",
            "This measures hypothetical victim regret, not KV pressure, eviction count, or JCT.",
        ],
    }


def _configured_task_ids(config: dict[str, Any], section: str) -> tuple[str, ...]:
    value = config.get(section)
    task_ids = value.get("task_ids") if isinstance(value, dict) else None
    if (
        not isinstance(task_ids, list)
        or not task_ids
        or not all(isinstance(task_id, str) and task_id for task_id in task_ids)
    ):
        raise ValueError(f"corpus config {section}.task_ids must be non-empty strings")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"corpus config {section}.task_ids contains duplicates")
    return tuple(task_ids)


def _partition_gaps(
    gaps: Iterable[Gap], fit_ids: Iterable[str], eval_ids: Iterable[str]
) -> tuple[list[Gap], list[Gap]]:
    fit_set = set(fit_ids)
    eval_set = set(eval_ids)
    return (
        [gap for gap in gaps if gap.session_id in fit_set],
        [gap for gap in gaps if gap.session_id in eval_set],
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
            f"same-repository cohort must contain exactly one repository: {sorted(repositories)}"
        )
    repository = next(iter(repositories))

    gaps = load_gaps(trace_root)
    fit_set = set(fit_ids)
    eval_set = set(eval_ids)
    fit, evaluation = _partition_gaps(gaps, fit_ids, eval_ids)
    if not fit or not evaluation:
        raise ValueError("configured fit and evaluation tasks must both contain gaps")

    history = _fit_history(fit)
    rows = evaluate(evaluation, history, {repository: history})
    primary = _comparison(rows, "repo_c100", "repo_tool")
    deletions = _task_deletions(rows, eval_ids, primary["delta_mean_regret_s"])
    sign_changes = [
        row["excluded_task"] for row in deletions if row["changes_primary_sign"]
    ]
    stability_go = deletions[0]["delta_mean_regret_s"] < 0.0

    for row in rows:
        if (
            row.regrets["pooled_tool"] != row.regrets["repo_tool"]
            or row.regrets["pooled_c100"] != row.regrets["repo_c100"]
            or row.choices["pooled_tool"] != row.choices["repo_tool"]
            or row.choices["pooled_c100"] != row.choices["repo_c100"]
        ):
            raise AssertionError("single-repository pooled and repo controls diverged")

    evaluation_config = config.get("evaluation")
    evaluation_config = evaluation_config if isinstance(evaluation_config, dict) else {}
    return {
        "status": "development-exposed dense same-repository mechanism test",
        "question": evaluation_config.get(
            "question",
            "Does dense same-repository history make CacheWise C100 outperform tool-name history?",
        ),
        "protocol": {
            "trace_root": str(trace_root.resolve()),
            "corpus_config": str(corpus_config.resolve()),
            "collection_id": config.get("collection_id"),
            "fit_selection": "configured fit.task_ids only",
            "eval_selection": "configured eval.task_ids only",
            "history": "all configured fit-task gaps from the one repository; evaluation gaps are never observed",
            "unit": "LLM-end to next-LLM-start tool gap; parallel outer calls form one ordered batch",
            "candidate": "C100 TF-IDF/KMeans whole-argument history",
            "baseline": "tool-name history on the identical fit gaps",
            "primary": "mean_regret(repo_c100) - mean_regret(repo_tool)",
            "go_criterion": evaluation_config.get("go_criterion"),
            "task_stability": evaluation_config.get("task_stability"),
            "comparison_scope": evaluation_config.get("comparison_scope"),
            "bootstrap": "10,000 draws clustered by unordered active-task tuple",
        },
        "data": {
            "repository": repository,
            "configured_fit_tasks": len(fit_ids),
            "fit_tasks_with_gaps": len({gap.session_id for gap in fit}),
            "fit_tasks_without_gaps": sorted(fit_set - {gap.session_id for gap in fit}),
            "fit_gaps": len(fit),
            "configured_eval_tasks": len(eval_ids),
            "eval_tasks_with_gaps": len({gap.session_id for gap in evaluation}),
            "eval_tasks_without_gaps": sorted(
                eval_set - {gap.session_id for gap in evaluation}
            ),
            "eval_gaps": len(evaluation),
            "eligible_ranking_events": len(rows),
            "eligible_task_pairs": len({row.task_pair for row in rows}),
            "max_concurrency": max(len(row.task_pair) for row in rows),
        },
        "metrics": _arm_metrics(rows),
        "primary_comparison": primary,
        "task_deletion_results": deletions,
        "task_stability": {
            "evaluated_tasks": len(deletions),
            "worst_deletion": deletions[0],
            "sign_change_tasks": sign_changes,
            "go": stability_go,
        },
        "decision": (
            "GO; dense same-repository history passes the primary and task-stability gates"
            if primary["go"] and stability_go
            else "STOP; dense same-repository history does not robustly rescue CacheWise C100"
        ),
        "limitations": [
            "This same-repository hypothesis and cohort are development-exposed.",
            "Natural overlap reaches concurrency two, not CacheWise's 30-50 sessions.",
            "This measures hypothetical victim regret, not KV pressure, eviction count, or JCT.",
            "The eBPF counters are excluded from model selection and reserved for mechanism analysis.",
        ],
    }


def self_check() -> None:
    assert repo_key("owner__repo-123") == "owner__repo"
    assert _configured_task_ids({"fit": {"task_ids": ["a", "b"]}}, "fit") == (
        "a",
        "b",
    )
    try:
        repo_key("owner__repo")
    except ValueError:
        pass
    else:
        raise AssertionError("repo_key must reject a missing issue number")
    selected_fit, selected_eval = _partition_gaps(
        [
            Gap("a__repo-1", 0.0, 1.0, "exec", "fit"),
            Gap("a__repo-2", 0.0, 1.0, "exec", "eval"),
            Gap("a__repo-3", 0.0, 1.0, "exec", "excluded"),
        ],
        ["a__repo-1"],
        ["a__repo-2"],
    )
    assert [gap.session_id for gap in selected_fit] == ["a__repo-1"]
    assert [gap.session_id for gap in selected_eval] == ["a__repo-2"]
    fit = [
        Gap("a__repo-1", 0.0, 1.0, '["exec"]', "short"),
        Gap("a__repo-1", 0.0, 10.0, '["exec"]', "long"),
        Gap("b__repo-1", 0.0, 2.0, '["exec"]', "short"),
        Gap("b__repo-1", 0.0, 20.0, '["exec"]', "long"),
    ]
    evaluation = [
        Gap("a__repo-2", 0.0, 10.0, '["exec"]', "long"),
        Gap("b__repo-2", 1.0, 3.0, '["exec"]', "short"),
    ]
    pooled, local = _fit_histories(fit, evaluation)
    assert sorted(local["a__repo"].global_durations.tolist()) == [1.0, 10.0]
    assert sorted(local["b__repo"].global_durations.tolist()) == [2.0, 20.0]
    pooled_prediction = _prediction(
        evaluation[0],
        0.0,
        "pooled_c100",
        pooled,
        local,
        {},
        {repo: {} for repo in local},
    )
    local_prediction = _prediction(
        evaluation[0], 0.0, "repo_c100", pooled, local, {}, {repo: {} for repo in local}
    )
    assert pooled_prediction == 15.0
    assert local_prediction == 10.0
    rows = evaluate(evaluation, pooled, local)
    assert len(rows) == 1
    assert rows[0].regrets["repo_c100"] == 0.0
    assert rows[0].regrets["repo_tool"] == 7.0

    def deletion_row(pair: tuple[str, ...], delta: float) -> RankingRow:
        regrets = {arm: 0.0 for arm in ARMS}
        regrets["repo_c100"] = delta
        return RankingRow(
            task_pair=pair,
            repos=frozenset({"a__repo"}),
            regrets=regrets,
            choices={arm: pair[0] for arm in ARMS},
        )

    deletion_rows = [
        deletion_row(("a__repo-2", "a__repo-3"), -2.0),
        deletion_row(("a__repo-3", "a__repo-4"), 1.0),
        deletion_row(("a__repo-2", "a__repo-4"), -2.0),
    ]
    deletions = _task_deletions(
        deletion_rows,
        ["a__repo-2", "a__repo-3", "a__repo-4", "a__repo-5"],
        -1.0,
    )
    by_task = {row["excluded_task"]: row for row in deletions}
    assert by_task["a__repo-2"]["changes_primary_sign"]
    assert by_task["a__repo-5"]["removed_events"] == 0


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
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--corpus-config", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
    )
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
        result = run(args.fit_root, args.eval_root)
    output = args.output or Path(
        "analysis/results/cachewise-swe-repo-locality-20260803/result.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
