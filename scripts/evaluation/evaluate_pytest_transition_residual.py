#!/usr/bin/env python3
"""Test a causal pytest occurrence-by-argument-transition correction."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from scripts.evaluation.evaluate_clause_latency_buckets import (  # noqa: E402
    _accuracy_delta,
    _phase_changes,
)
from scripts.evaluation.evaluate_command_history_residual import (  # noqa: E402
    BUCKETS,
    SPLIT_MANIFEST,
    TARGETS,
    _fail_closed_metrics,
)
from scripts.evaluation.evaluate_command_outcome_memory import (  # noqa: E402
    ALL_TARGETS,
    DISK,
    FROZEN_FIT_ROWS,
    _hard,
)
from scripts.evaluation.evaluate_multitarget_sota import (  # noqa: E402
    FROZEN_ROW_COUNT,
)
from scripts.evaluation.evaluate_pytest_target_overlap import (  # noqa: E402
    _partition,
    _pytest_query,
)

VERSION = "pytest-transition-residual-v1"
FROZEN_FIT_ROW_COUNT = 1420
ACTIVE_OCCURRENCE_BINS = ("second", "third_plus")
ARMS = (
    "pytest_semantic_base",
    "smoothed_base",
    "occurrence_only",
    "transition_only",
    "phase_transition",
)
NO_PREVIOUS = "no_comparable_previous"
TRANSITIONS = (
    NO_PREVIOUS,
    "exact_repeat",
    "expand",
    "contract",
    "reshape",
    "disjoint",
    "targeted_to_full",
    "full_to_targeted",
)
RESULTS = _ROOT / "analysis/results/tool-resource-5-3-3-3-20260804"
FROZEN_SEMANTIC_ROWS = RESULTS / "sqlglot50-semantic-work-units-v1/rows.jsonl"
FROZEN_SOTA_ROWS = RESULTS / "sqlglot50-multitarget-sota-v1/rows.jsonl"
FROZEN_FIT_TASKS = tuple(
    json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))["development"][-80:]
)


def _transition(previous: frozenset[str], current: frozenset[str]) -> str:
    if previous == current:
        return "exact_repeat"
    if not previous:
        return "full_to_targeted"
    if not current:
        return "targeted_to_full"
    overlap = previous & current
    if not overlap:
        return "disjoint"
    if previous < current:
        return "expand"
    if current < previous:
        return "contract"
    return "reshape"


def _states(rows: Sequence[Mapping[str, Any]]) -> list[tuple[str, str] | None]:
    states: list[tuple[str, str] | None] = []
    task_id: str | None = None
    occurrence = 0
    previous: dict[Any, frozenset[str]] = {}
    for row in rows:
        if row["task_id"] != task_id:
            task_id = row["task_id"]
            occurrence = 0
            previous = {}
        query = _pytest_query(row["command"])
        if query is None:
            states.append(None)
            continue
        signature, units = query
        occurrence += 1
        phase = "first" if occurrence == 1 else "second" if occurrence == 2 else "third_plus"
        key = _partition(signature)
        relation = NO_PREVIOUS if key not in previous else _transition(previous[key], units)
        previous[key] = units
        states.append((phase, relation))
    return states


def _labels(row: Mapping[str, Any]) -> Mapping[str, int | None]:
    labels = row.get("labels")
    if isinstance(labels, dict):
        return labels
    return {"latency": row.get("latency_label"), **dict(row["resource_labels"])}


def _new_counts() -> dict[str, dict[str, Any]]:
    return {
        target: {
            "base": Counter(),
            "phase": defaultdict(Counter),
            "relation": defaultdict(Counter),
            "joint": defaultdict(Counter),
        }
        for target in TARGETS
    }


def _absorb(
    rows: Sequence[Mapping[str, Any]],
    states: Sequence[tuple[str, str] | None],
    counts: dict[str, dict[str, Any]],
) -> None:
    for row, state in zip(rows, states, strict=True):
        if state is None:
            continue
        phase, relation = state
        labels = _labels(row)
        for target in TARGETS:
            label = labels[target]
            if label is None:
                continue
            counts[target]["base"][label] += 1
            counts[target]["phase"][phase][label] += 1
            counts[target]["relation"][relation][label] += 1
            counts[target]["joint"][state][label] += 1


def _posterior(values: Counter[int], parent: Sequence[float]) -> list[float]:
    prior_mass = len(parent)
    total = sum(values.values()) + prior_mass
    return [(values[index] + prior_mass * parent[index]) / total for index in range(len(parent))]


def _normalize(values: Sequence[float]) -> list[float]:
    total = sum(values)
    return [value / total for value in values]


def _state_distribution(
    counts: Mapping[str, Any], state: tuple[str, str], arm: str, buckets: int
) -> tuple[list[float], list[float]]:
    uniform = [1.0 / buckets] * buckets
    base = _posterior(counts["base"], uniform)
    phase, relation = state
    phase_pmf = _posterior(counts["phase"][phase], base)
    if relation == NO_PREVIOUS:
        return base, phase_pmf if arm != "transition_only" else base
    relation_pmf = _posterior(counts["relation"][relation], base)
    if arm == "occurrence_only":
        return base, phase_pmf
    if arm == "transition_only":
        return base, relation_pmf
    parent = _normalize(
        [phase_pmf[index] * relation_pmf[index] / base[index] for index in range(buckets)]
    )
    return base, _posterior(counts["joint"][state], parent)


def _correct(
    base_pmf: Sequence[float],
    support: float,
    population: Sequence[float],
    state: Sequence[float],
) -> list[float]:
    prior_mass = len(base_pmf)
    smoothed = [
        (support * base_pmf[index] + prior_mass * population[index])
        / (support + prior_mass)
        for index in range(len(base_pmf))
    ]
    return _normalize(
        [smoothed[index] * state[index] / population[index] for index in range(len(base_pmf))]
    )


def _groups(rows: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start = 0
    while start < len(rows):
        end = start + 1
        while end < len(rows) and rows[end]["task_id"] == rows[start]["task_id"]:
            end += 1
        groups.append((start, end))
        start = end
    return groups


def _validate_population(
    semantic_rows: Sequence[Mapping[str, Any]], sota_rows: Sequence[Mapping[str, Any]]
) -> None:
    if len(semantic_rows) != FROZEN_ROW_COUNT or len(sota_rows) != FROZEN_ROW_COUNT:
        raise ValueError("validation rows differ from the frozen population")
    sample_ids = [row["sample_id"] for row in semantic_rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("validation rows contain duplicate sample IDs")
    for semantic, sota in zip(semantic_rows, sota_rows, strict=True):
        for key in (
            "sample_id",
            "task_id",
            "command",
            "labels",
            "current_dynamic",
            "current_probability_by_bucket",
        ):
            if semantic[key] != sota[key]:
                raise ValueError(f"validation rows disagree on {key}")


def _validate_fit(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != FROZEN_FIT_ROW_COUNT:
        raise ValueError("fit rows differ from the frozen population")
    sample_ids = [row["sample_id"] for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("fit rows contain duplicate sample IDs")
    task_ids = tuple(rows[start]["task_id"] for start, _end in _groups(rows))
    if task_ids != FROZEN_FIT_TASKS:
        raise ValueError("fit task order or contiguity differs from the frozen population")


def run(
    fit_rows: Sequence[Mapping[str, Any]],
    semantic_rows: Sequence[Mapping[str, Any]],
    sota_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _validate_fit(fit_rows)
    _validate_population(semantic_rows, sota_rows)
    fit_states = _states(fit_rows)
    validation_states = _states(semantic_rows)
    counts = _new_counts()
    _absorb(fit_rows, fit_states, counts)

    rows: list[dict[str, Any]] = []
    for start, end in _groups(semantic_rows):
        for semantic, sota, state in zip(
            semantic_rows[start:end],
            sota_rows[start:end],
            validation_states[start:end],
            strict=True,
        ):
            sota_arm = {
                "candidate": deepcopy(sota["candidate"]),
                "candidate_probability_by_bucket": deepcopy(
                    sota["candidate_probability_by_bucket"]
                ),
                "provenance": deepcopy(sota["provenance"]),
            }
            semantic_arm = deepcopy(sota_arm)
            if state is not None:
                source = semantic["arms"]["semantic_work_units"]
                for target in TARGETS:
                    semantic_arm["candidate"][target] = source["candidate"][target]
                    semantic_arm["candidate_probability_by_bucket"][target] = deepcopy(
                        source["candidate_probability_by_bucket"][target]
                    )
                    semantic_arm["provenance"][target] = deepcopy(
                        source["provenance"][target]
                    )
            arms = {arm: deepcopy(semantic_arm) for arm in ARMS}
            if state is not None and state[0] in ACTIVE_OCCURRENCE_BINS:
                for target in TARGETS:
                    provenance = semantic_arm["provenance"][target]
                    support = float(
                        provenance.get("weight_sum", provenance.get("support", 0))
                    )
                    if support <= 0.0:
                        continue
                    for arm in ARMS[1:]:
                        base_pmf = semantic_arm["candidate_probability_by_bucket"][target]
                        if base_pmf is None:
                            continue
                        if arm == "smoothed_base":
                            population, state_pmf = _state_distribution(
                                counts[target], state, "transition_only", BUCKETS[target]
                            )
                            state_pmf = population
                        else:
                            population, state_pmf = _state_distribution(
                                counts[target], state, arm, BUCKETS[target]
                            )
                        pmf = _correct(base_pmf, support, population, state_pmf)
                        arms[arm]["candidate"][target] = _hard(target, pmf)
                        arms[arm]["candidate_probability_by_bucket"][target] = pmf
                        arms[arm]["provenance"][target] = {
                            "source": arm,
                            "phase": state[0],
                            "relation": state[1],
                            "semantic_support": support,
                            "phase_support": sum(counts[target]["phase"][state[0]].values()),
                            "relation_support": sum(counts[target]["relation"][state[1]].values()),
                            "joint_support": sum(counts[target]["joint"][state].values()),
                        }
            rows.append(
                {
                    "sample_id": semantic["sample_id"],
                    "task_id": semantic["task_id"],
                    "command": semantic["command"],
                    "labels": deepcopy(semantic["labels"]),
                    "current_dynamic": deepcopy(semantic["current_dynamic"]),
                    "current_probability_by_bucket": deepcopy(
                        semantic["current_probability_by_bucket"]
                    ),
                    "sota": sota_arm["candidate"],
                    "sota_probability_by_bucket": sota_arm[
                        "candidate_probability_by_bucket"
                    ],
                    "pytest_state": state,
                    "arms": arms,
                }
            )
        _absorb(
            semantic_rows[start:end], validation_states[start:end], counts
        )

    sota_metric_rows = [
        {
            **{key: value for key, value in row.items() if key != "arms"},
            "candidate": row["sota"],
            "candidate_probability_by_bucket": row["sota_probability_by_bucket"],
        }
        for row in rows
    ]
    arm_rows = {
        arm: [
            {
                **{key: value for key, value in row.items() if key != "arms"},
                "candidate": row["arms"][arm]["candidate"],
                "candidate_probability_by_bucket": row["arms"][arm][
                    "candidate_probability_by_bucket"
                ],
            }
            for row in rows
        ]
        for arm in ARMS
    }
    sota_metrics = {
        target: _fail_closed_metrics(sota_metric_rows, target) for target in ALL_TARGETS
    }
    metrics: dict[str, Any] = {}
    macro: dict[str, float] = {}
    for arm in ARMS:
        metrics[arm] = {}
        accuracies: list[float] = []
        for target in ALL_TARGETS:
            candidate = _fail_closed_metrics(arm_rows[arm], target)
            key = "exact_class_accuracy" if target == "latency" else "accuracy"
            accuracies.append(candidate[key])
            metrics[arm][target] = {
                "sota": sota_metrics[target],
                "candidate": candidate,
                "delta_vs_sota_percentage_points": _accuracy_delta(
                    candidate[key], sota_metrics[target][key]
                ),
                "changes_vs_sota": _phase_changes(
                    arm_rows[arm], target, reference="sota"
                ),
            }
        macro[arm] = sum(accuracies[:3]) / len(TARGETS)

    primary = metrics["phase_transition"]
    no_regression = all(
        primary[target]["delta_vs_sota_percentage_points"] >= 0.0 for target in TARGETS
    )
    sota_macro = sum(
        sota_metrics[target]["exact_class_accuracy" if target == "latency" else "accuracy"]
        for target in TARGETS
    ) / len(TARGETS)
    helpful = sum(primary[target]["changes_vs_sota"]["helpful"] for target in TARGETS)
    harmful = sum(primary[target]["changes_vs_sota"]["harmful"] for target in TARGETS)
    changed_tasks = {
        row["task_id"]
        for row in arm_rows["phase_transition"]
        if any(row["candidate"][target] != row["sota"][target] for target in TARGETS)
    }
    disk_identity = all(
        row["candidate"][DISK] == row["sota"][DISK]
        and row["candidate_probability_by_bucket"][DISK]
        == row["sota_probability_by_bucket"][DISK]
        for arm in ARMS
        for row in arm_rows[arm]
    )
    non_pytest_identity = all(
        row["pytest_state"] is not None
        or (
            row["candidate"] == row["sota"]
            and row["candidate_probability_by_bucket"]
            == row["sota_probability_by_bucket"]
        )
        for arm in ARMS
        for row in arm_rows[arm]
    )
    interaction_wins = (
        macro["phase_transition"] > macro["smoothed_base"]
        and macro["phase_transition"] > macro["occurrence_only"]
        and macro["phase_transition"] > macro["transition_only"]
    )
    go = (
        no_regression
        and macro["phase_transition"] > sota_macro
        and interaction_wins
        and helpful > harmful
        and len(changed_tasks) >= 5
        and disk_identity
        and non_pytest_identity
    )
    state_counts = Counter(state for state in validation_states if state is not None)
    return {
        "schema": VERSION,
        "status": "development_lead_go" if go else "development_lead_no_go",
        "claim_bearing": False,
        "protocol": {
            "decision_time": "BeginCall",
            "tool": "pytest",
            "active_occurrence_bins": list(ACTIVE_OCCURRENCE_BINS),
            "first_occurrence": "unchanged_semantic_base",
            "transition_categories": list(TRANSITIONS),
            "prior_mass": "number_of_target_buckets",
            "parent": "laplace_binary_then_occurrence_and_transition_then_joint",
            "correction": "support_smoothed_semantic_pmf_times_state_over_pytest_population",
            "cross_task_update": "whole_task_settlement",
            "current_task_resource_outcomes_visible": False,
        },
        "coverage": {
            "fit_pytest_rows": sum(state is not None for state in fit_states),
            "validation_pytest_rows": sum(state is not None for state in validation_states),
            "validation_repeated_pytest_rows": sum(
                state is not None and state[0] in ACTIVE_OCCURRENCE_BINS
                for state in validation_states
            ),
            "validation_state_counts": {
                f"{phase}:{relation}": count
                for (phase, relation), count in sorted(state_counts.items())
            },
        },
        "sota_metrics": sota_metrics,
        "arms": metrics,
        "macro_latency_cpu_rss_accuracy": {"sota": sota_macro, **macro},
        "gate": {
            "go": go,
            "no_latency_cpu_rss_regression": no_regression,
            "macro_strictly_above_sota": macro["phase_transition"] > sota_macro,
            "joint_strictly_above_both_ablations": interaction_wins,
            "helpful": helpful,
            "harmful": harmful,
            "changed_tasks": len(changed_tasks),
            "minimum_changed_tasks": 5,
            "disk_bit_identical": disk_identity,
            "non_pytest_bit_identical": non_pytest_identity,
            "final_partition_authorized": False,
        },
    }, rows


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("output directory already exists")
    result, rows = run(
        _load(FROZEN_FIT_ROWS),
        _load(FROZEN_SEMANTIC_ROWS),
        _load(FROZEN_SOTA_ROWS),
    )
    result["inputs"] = {
        "fit_rows": str(FROZEN_FIT_ROWS),
        "semantic_rows": str(FROZEN_SEMANTIC_ROWS),
        "sota_rows": str(FROZEN_SOTA_ROWS),
        "git_sha": _git_sha(),
    }
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
