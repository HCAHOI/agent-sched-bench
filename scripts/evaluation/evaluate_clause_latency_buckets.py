#!/usr/bin/env python3
"""Evaluate clause latency buckets on canonical eBPF clause telemetry.

The former bash-xtrace proxy lane was removed once collection moved entirely to
eBPF clause telemetry; git history holds it if it is ever needed again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

# Canonical clause telemetry is already loaded by the resource-class evaluator;
# reuse that loader rather than re-deriving the artifact shape here.
from scripts.evaluation.evaluate_clause_resource_classes import (  # noqa: E402
    CommandRow,
    Row,
    _empty_confusion,
    _finalize_resource_metric,
    command_resource_label,
    load_rows,
)
from tool_resource_eval.labels import repo_of  # noqa: E402
from tool_resource.runtime_kb import (  # noqa: E402
    CANONICAL_LATENCY_BUCKETS,
    CANONICAL_RESOURCE_HEAVY_THRESHOLDS,
    SHRINKAGE_ALPHA_GRID,
    SHORT_NULL_LIGHT_MAX_LATENCY_MS,
    STRUCTURED_ARGV_REPRESENTATION,
    ClauseHeavyLightPrediction,
    ClauseLatencyBucketPrediction,
    ClauseResourceKB,
    _canonical_dynamic_value,
    _structured_argv_parts,
)

INTERACTION_ALPHA = 16.0
INTERACTION_FEATURE_VERSION = "generic-argv-v3-role-interaction-set-v1"


@dataclass(frozen=True)
class ScoredRow:
    sample_id: str
    task_id: str
    repo: str
    command: str
    label_bucket: int
    probability_by_bucket: tuple[float, ...] | None
    layer: str | None
    key_kind: str | None
    evidence_count: int
    fallback_path: tuple[str, ...] | None
    canonicalizer_version: str | None
    arbitration: str | None
    local_key_kind: str | None
    local_evidence_count: int
    public_key_kind: str | None
    public_evidence_count: int
    shrinkage_alpha: float | None
    unavailable_reason: str | None
    mapping_evidence: str


@dataclass(frozen=True)
class InteractionFeatureSet:
    """Typed non-binary clause features partitioned by normalized binary."""

    bin: str
    features: frozenset[str]


@dataclass(frozen=True)
class _InteractionObservation:
    observation_id: int
    task_id: str
    row: Row
    feature_set: InteractionFeatureSet


@dataclass(frozen=True)
class _PosetMatch:
    exact: bool
    observations: tuple[_InteractionObservation, ...]
    frontier: frozenset[frozenset[str]]


@dataclass(frozen=True)
class _KernelContribution:
    observation: _InteractionObservation
    shared_feature_count: int
    weight: float


@dataclass(frozen=True)
class _KernelMatch:
    exact: bool
    contributions: tuple[_KernelContribution, ...]


def _interaction_feature_set(
    bin_: str,
    argv: Sequence[str],
    stable_subcommands: frozenset[tuple[str, str]],
) -> InteractionFeatureSet:
    """Build the frozen typed set without raw dynamic argument values."""

    subcommand, options, positionals = _structured_argv_parts(argv)
    if subcommand is None:
        subcommand_feature = "subcommand:<NONE>"
    elif (bin_, subcommand) in stable_subcommands:
        subcommand_feature = f"subcommand:{subcommand}"
    else:
        subcommand_feature = f"subcommand:{_canonical_dynamic_value(subcommand)}"
    features = {subcommand_feature}
    for option, count in Counter(options).items():
        features.update(
            f"option:{option}:occurrence:{occurrence}"
            for occurrence in range(1, count + 1)
        )
    for slot, positional in enumerate(positionals):
        features.add(
            f"boundary:{slot}:--"
            if positional == "boundary:--"
            else f"positional:{slot}:{positional}"
        )
    return InteractionFeatureSet(bin=bin_, features=frozenset(features))


def _maximal_intersections(
    query: frozenset[str],
    nodes: Sequence[frozenset[str]],
) -> frozenset[frozenset[str]]:
    """Return the query-induced maximal non-empty feature intersections."""

    intersections = {query & node for node in nodes}
    intersections.discard(frozenset())
    return frozenset(
        candidate
        for candidate in intersections
        if not any(candidate < other for other in intersections)
    )


class _InteractionPosetKB:
    """Slow exact interaction-poset evaluator; it contains no trie index."""

    def __init__(
        self,
        stable_subcommands: frozenset[tuple[str, str]] = frozenset(),
    ) -> None:
        self.stable_subcommands = stable_subcommands
        self._next_observation_id = 0
        self._exact: dict[
            tuple[str, tuple[str, ...]], list[_InteractionObservation]
        ] = defaultdict(list)
        self._nodes: dict[
            str, dict[frozenset[str], list[_InteractionObservation]]
        ] = defaultdict(lambda: defaultdict(list))

    def observe(self, rows: Sequence[Row]) -> None:
        for row in rows:
            observation = _InteractionObservation(
                observation_id=self._next_observation_id,
                task_id=row.task_id,
                row=row,
                feature_set=_interaction_feature_set(
                    row.bin,
                    row.argv,
                    self.stable_subcommands,
                ),
            )
            self._next_observation_id += 1
            self._exact[(row.bin, row.argv[1:])].append(observation)
            self._nodes[row.bin][observation.feature_set.features].append(observation)

    def query(self, row: Row) -> _PosetMatch:
        exact = tuple(self._exact.get((row.bin, row.argv[1:]), ()))
        if exact:
            return _PosetMatch(
                exact=True,
                observations=exact,
                frontier=frozenset(),
            )
        query = _interaction_feature_set(
            row.bin,
            row.argv,
            self.stable_subcommands,
        ).features
        nodes = self._nodes.get(row.bin, {})
        frontier = _maximal_intersections(query, tuple(nodes))
        selected = tuple(
            observation
            for node, observations in nodes.items()
            if query & node in frontier
            for observation in observations
        )
        if len({item.observation_id for item in selected}) != len(selected):
            raise AssertionError("poset matched one observation more than once")
        return _PosetMatch(
            exact=False,
            observations=selected,
            frontier=frontier,
        )


def _subset_count(size: int, order: int | None = None) -> int:
    """Count non-empty subsets, optionally limited to a maximum order."""

    if size <= 0:
        return 0
    if order is None:
        return 2**size - 1
    if order <= 0:
        raise ValueError("subset order must be positive")
    return sum(math.comb(size, degree) for degree in range(1, min(order, size) + 1))


def _subset_kernel(
    query: frozenset[str],
    history: frozenset[str],
    order: int | None = None,
) -> float:
    """Length-normalized overlap in the non-empty subset feature space."""

    shared = len(query & history)
    numerator = _subset_count(shared, order)
    denominator = math.sqrt(
        _subset_count(len(query), order) * _subset_count(len(history), order)
    )
    return numerator / denominator if denominator else 0.0


class _EpisodicSubsetKB:
    """Observation memory weighted by the all-subset kernel; no trie index."""

    def __init__(
        self,
        stable_subcommands: frozenset[tuple[str, str]] = frozenset(),
        *,
        order: int | None = None,
    ) -> None:
        if order is not None and order <= 0:
            raise ValueError("subset order must be positive")
        self.stable_subcommands = stable_subcommands
        self.order = order
        self._next_observation_id = 0
        self._exact: dict[
            tuple[str, tuple[str, ...]], list[_InteractionObservation]
        ] = defaultdict(list)
        self._by_bin: dict[str, list[_InteractionObservation]] = defaultdict(list)

    def observe(self, rows: Sequence[Row]) -> None:
        for row in rows:
            observation = _InteractionObservation(
                observation_id=self._next_observation_id,
                task_id=row.task_id,
                row=row,
                feature_set=_interaction_feature_set(
                    row.bin,
                    row.argv,
                    self.stable_subcommands,
                ),
            )
            self._next_observation_id += 1
            self._exact[(row.bin, row.argv[1:])].append(observation)
            self._by_bin[row.bin].append(observation)

    def query(self, row: Row) -> _KernelMatch:
        exact = tuple(self._exact.get((row.bin, row.argv[1:]), ()))
        if exact:
            return _KernelMatch(
                exact=True,
                contributions=tuple(
                    _KernelContribution(
                        observation=observation,
                        shared_feature_count=len(observation.feature_set.features),
                        weight=1.0,
                    )
                    for observation in exact
                ),
            )
        query = _interaction_feature_set(
            row.bin,
            row.argv,
            self.stable_subcommands,
        ).features
        contributions = tuple(
            _KernelContribution(
                observation=observation,
                shared_feature_count=len(query & observation.feature_set.features),
                weight=weight,
            )
            for observation in self._by_bin.get(row.bin, ())
            if (
                weight := _subset_kernel(
                    query,
                    observation.feature_set.features,
                    self.order,
                )
            )
            > 0.0
        )
        if len({item.observation.observation_id for item in contributions}) != len(
            contributions
        ):
            raise AssertionError("subset kernel matched one observation more than once")
        return _KernelMatch(exact=False, contributions=contributions)


def _validate_partition(
    fit_tasks: Sequence[str], eval_tasks: Sequence[str]
) -> dict[str, int]:
    fit = set(fit_tasks)
    evaluate = set(eval_tasks)
    overlap = fit & evaluate
    if overlap:
        raise ValueError(f"fit/eval task overlap: {sorted(overlap)[:3]}")
    fit_repos = {repo_of(task) for task in fit}
    eval_repos = {repo_of(task) for task in evaluate}
    return {
        "fit_task_count": len(fit),
        "eval_task_count": len(evaluate),
        "task_overlap_count": 0,
        "fit_repo_count": len(fit_repos),
        "eval_repo_count": len(eval_repos),
        "repo_overlap_count": len(fit_repos & eval_repos),
    }


def _exact_bucket_metrics(rows: Sequence[ScoredRow]) -> dict[str, Any]:
    correct = sum(_exact_bucket_correct(row) for row in rows)
    return {
        "three_class_accuracy": correct / len(rows) if rows else None,
        "eligible_examples": len(rows),
    }


def _argmax_bucket(row: ScoredRow) -> int:
    """Predicted bucket: highest probability, lowest bucket id on ties."""

    assert row.probability_by_bucket is not None
    return _argmax_probabilities(row.probability_by_bucket)


def _argmax_probabilities(probabilities: Sequence[float]) -> int:
    """Highest-probability bucket, with the shortest bucket winning ties."""

    return max(
        range(CANONICAL_LATENCY_BUCKETS.bucket_count),
        key=probabilities.__getitem__,
    )


def _exact_bucket_correct(row: ScoredRow) -> bool:
    return _argmax_bucket(row) == row.label_bucket


def _bucket_intervals() -> list[dict[str, Any]]:
    """Report the [0, b0], (b_{i-1}, b_i], tail semantics the KB implements."""

    edges = CANONICAL_LATENCY_BUCKETS.edges_ms
    return [
        {
            "bucket_id": bucket,
            "lower_ms": 0.0 if bucket == 0 else edges[bucket - 1],
            "lower_inclusive": bucket == 0,
            "upper_ms": edges[bucket] if bucket < len(edges) else None,
            "upper_inclusive": bucket < len(edges),
        }
        for bucket in range(CANONICAL_LATENCY_BUCKETS.bucket_count)
    ]


def _scored_row(
    row: Row,
    clause_index: int,
    prediction: ClauseLatencyBucketPrediction | None,
    unavailable_reason: str | None = None,
) -> ScoredRow:
    return ScoredRow(
        sample_id=f"{row.task_id}:{row.manifest_index}:{clause_index}",
        task_id=row.task_id,
        repo=row.repo,
        command=" ".join(row.argv),
        label_bucket=CANONICAL_LATENCY_BUCKETS.bucket_id(row.latency_ms),
        probability_by_bucket=(
            None if prediction is None else prediction.probability_by_bucket
        ),
        layer=None if prediction is None else prediction.scope,
        key_kind=None if prediction is None else prediction.key_kind,
        evidence_count=0 if prediction is None else prediction.evidence_count,
        fallback_path=None if prediction is None else prediction.fallback_path,
        canonicalizer_version=(
            None if prediction is None else prediction.canonicalizer_version
        ),
        arbitration=None if prediction is None else prediction.arbitration,
        local_key_kind=None if prediction is None else prediction.local_key_kind,
        local_evidence_count=(
            0 if prediction is None else prediction.local_evidence_count
        ),
        public_key_kind=None if prediction is None else prediction.public_key_kind,
        public_evidence_count=(
            0 if prediction is None else prediction.public_evidence_count
        ),
        shrinkage_alpha=None if prediction is None else prediction.shrinkage_alpha,
        unavailable_reason=unavailable_reason,
        mapping_evidence="canonical_clause_telemetry",
    )


def _command_scored_row(
    row: CommandRow,
    prediction: ClauseLatencyBucketPrediction | None,
    unavailable_reason: str | None,
) -> ScoredRow:
    return replace(
        _scored_row(row.clauses[0], row.call_index, prediction, unavailable_reason),
        sample_id=f"{row.task_id}:{row.manifest_index}:call:{row.call_index}",
        command=row.command,
        label_bucket=CANONICAL_LATENCY_BUCKETS.bucket_id(row.duration_ms),
        mapping_evidence="command_predictor",
    )


def _oracle_prediction(
    candidates: Sequence[ClauseLatencyBucketPrediction],
    label_bucket: int,
    fallback: ClauseLatencyBucketPrediction,
) -> ClauseLatencyBucketPrediction:
    """Analysis-only hindsight choice; never a deployable selector."""

    return next(
        (
            candidate
            for candidate in candidates
            if _argmax_probabilities(candidate.probability_by_bucket) == label_bucket
        ),
        fallback,
    )


def _bounded_node_oracle_candidates(
    *groups: Sequence[ClauseLatencyBucketPrediction],
) -> tuple[ClauseLatencyBucketPrediction, ...]:
    """Keep only the exact/structured/bin/global nodes fixed by the contract."""

    allowed = {"exact_clause", "structured_argv", "bin", "global"}
    return tuple(
        candidate
        for group in groups
        for candidate in group
        if candidate.key_kind in allowed
    )


def _inner_repo_folds(repositories: Sequence[str]) -> dict[str, int]:
    if len(repositories) < 2:
        raise ValueError("Candidate S alpha selection requires at least two fit repos")
    fold_count = min(5, len(repositories))
    ordered = sorted(
        repositories,
        key=lambda repo: (hashlib.sha256(repo.encode()).digest(), repo),
    )
    return {repo: index % fold_count for index, repo in enumerate(ordered)}


def _score_shrinkage_alpha_fold(
    train_rows: Sequence[Row],
    validation_rows: Sequence[Row],
) -> dict[float, dict[str, int]]:
    validation_repos = sorted({row.repo for row in validation_rows})
    kbs = {
        repo: ClauseResourceKB.fit_public(
            (row.observation(0.0, 1.0) for row in train_rows),
            representation=STRUCTURED_ARGV_REPRESENTATION,
        )
        for repo in validation_repos
    }
    by_task: dict[tuple[int, str], list[Row]] = defaultdict(list)
    for row in validation_rows:
        by_task[(row.manifest_index, row.task_id)].append(row)
    scores = {
        alpha: {"correct": 0, "eligible_examples": 0}
        for alpha in SHRINKAGE_ALPHA_GRID
    }
    for task_ordinal, task_key in enumerate(sorted(by_task)):
        rows = by_task[task_key]
        query_ts = float(task_ordinal * 2 + 1)
        settle_ts = query_ts + 1.0
        for row in rows:
            predictions = kbs[
                row.repo
            ].diagnostic_clause_latency_shrinkage_predictions(
                row.repo,
                row.bin,
                row.argv,
                CANONICAL_LATENCY_BUCKETS,
                SHRINKAGE_ALPHA_GRID,
                ts_start=query_ts,
            )
            label = CANONICAL_LATENCY_BUCKETS.bucket_id(row.latency_ms)
            for alpha, prediction in zip(
                SHRINKAGE_ALPHA_GRID,
                predictions,
                strict=True,
            ):
                scores[alpha]["eligible_examples"] += 1
                scores[alpha]["correct"] += (
                    _argmax_probabilities(prediction.probability_by_bucket) == label
                )
        for row in rows:
            kbs[row.repo].observe_completed_clause(
                row.observation(query_ts, settle_ts)
            )
    return scores


def _select_shrinkage_alpha(fit_rows: Sequence[Row]) -> dict[str, Any]:
    """Select one Candidate S alpha using latency-only fit-repository folds."""

    repositories = sorted({row.repo for row in fit_rows})
    fold_by_repo = _inner_repo_folds(repositories)
    fold_count = max(fold_by_repo.values()) + 1
    totals = {
        alpha: {"correct": 0, "eligible_examples": 0}
        for alpha in SHRINKAGE_ALPHA_GRID
    }
    folds = []
    for fold in range(fold_count):
        validation_repos = {
            repo for repo, assigned in fold_by_repo.items() if assigned == fold
        }
        train_rows = [row for row in fit_rows if row.repo not in validation_repos]
        validation_rows = [
            row for row in fit_rows if row.repo in validation_repos
        ]
        scores = _score_shrinkage_alpha_fold(train_rows, validation_rows)
        for alpha, score in scores.items():
            totals[alpha]["correct"] += score["correct"]
            totals[alpha]["eligible_examples"] += score["eligible_examples"]
        folds.append(
            {
                "fold": fold,
                "train_repo_count": len(repositories) - len(validation_repos),
                "validation_repo_count": len(validation_repos),
                "validation_row_count": len(validation_rows),
                "accuracy_by_alpha": {
                    str(int(alpha)): (
                        score["correct"] / score["eligible_examples"]
                    )
                    for alpha, score in scores.items()
                },
            }
        )
    accuracy_by_alpha = {
        alpha: score["correct"] / score["eligible_examples"]
        for alpha, score in totals.items()
    }
    selected = max(
        SHRINKAGE_ALPHA_GRID,
        key=lambda alpha: (accuracy_by_alpha[alpha], alpha),
    )
    return {
        "method": "repository_grouped_inner_fit_folds",
        "fold_assignment": "sha256(repository) order, round-robin over five folds",
        "fold_count": fold_count,
        "alpha_grid": list(SHRINKAGE_ALPHA_GRID),
        "tie_break": "larger_alpha",
        "selection_target": "three_class_latency_accuracy",
        "fit_repo_count": len(repositories),
        "fit_row_count": len(fit_rows),
        "accuracy_by_alpha": {
            str(int(alpha)): accuracy for alpha, accuracy in accuracy_by_alpha.items()
        },
        "selected_alpha": selected,
        "folds": folds,
        "outer_labels_used": False,
    }


def _telemetry_scored_arms(
    fit_rows: Sequence[Row],
    eval_rows: Sequence[Row],
    *,
    shrinkage_alpha: float | None = None,
) -> dict[str, list[ScoredRow]]:
    """Score canonical clause telemetry with the runtime KB in causal task order.

    Public priors are fit leave-one-repo-out so the fit corpus never carries the
    evaluated repository. Within a task every clause is predicted before any of
    that task's observations settle, so there is no intra-task leakage; settled
    observations become repo-local evidence for later tasks only.
    """

    eval_repos = {row.repo for row in eval_rows}
    kb_by_repo = {
        repo: ClauseResourceKB.fit_public(
            row.observation(0.0, 1.0) for row in fit_rows if row.repo != repo
        )
        for repo in sorted(eval_repos)
    }
    candidate_kb_by_repo = {
        repo: ClauseResourceKB.fit_public(
            (row.observation(0.0, 1.0) for row in fit_rows if row.repo != repo),
            representation=STRUCTURED_ARGV_REPRESENTATION,
        )
        for repo in sorted(eval_repos)
    }
    shrinkage_kb_by_repo = (
        {
            repo: ClauseResourceKB.fit_public(
                (row.observation(0.0, 1.0) for row in fit_rows if row.repo != repo),
                representation=STRUCTURED_ARGV_REPRESENTATION,
                shrinkage_alpha=shrinkage_alpha,
            )
            for repo in sorted(eval_repos)
        }
        if shrinkage_alpha is not None
        else {}
    )
    by_task: dict[tuple[int, str], list[Row]] = defaultdict(list)
    for row in eval_rows:
        by_task[(row.manifest_index, row.task_id)].append(row)
    scored: dict[str, list[ScoredRow]] = {
        name: []
        for name in (
            "current",
            "public_only",
            "local_only",
            "candidate_r_public_only",
            "candidate_r",
            "current_public_oracle",
            "node_oracle",
        )
    }
    if shrinkage_alpha is not None:
        scored["candidate_s"] = []
    for task_ordinal, task_key in enumerate(sorted(by_task)):
        rows = by_task[task_key]
        # Query strictly before this task's observations settle; settle strictly
        # before the next task queries.
        query_ts = float(task_ordinal * 2 + 1)
        settle_ts = query_ts + 1.0
        for clause_index, row in enumerate(rows):
            current_kb = kb_by_repo[row.repo]
            candidate_kb = candidate_kb_by_repo[row.repo]
            shrinkage_kb = shrinkage_kb_by_repo.get(row.repo)
            # Raises when no evidence node exists; never falls back to synthetic.
            prediction = current_kb.predict_clause_latency_bucket(
                row.repo,
                row.bin,
                row.argv,
                CANONICAL_LATENCY_BUCKETS,
                ts_start=query_ts,
            )
            candidates = current_kb.diagnostic_clause_latency_candidates(
                row.repo,
                row.bin,
                row.argv,
                CANONICAL_LATENCY_BUCKETS,
                ts_start=query_ts,
            )
            if not candidates or candidates[0] != prediction:
                raise AssertionError(
                    "diagnostic candidates differ from runtime prediction"
                )
            public = next(
                (item for item in candidates if item.scope == "public"),
                None,
            )
            local = next(
                (item for item in candidates if item.scope == "repo"),
                None,
            )
            candidate_prediction = candidate_kb.predict_clause_latency_bucket(
                row.repo,
                row.bin,
                row.argv,
                CANONICAL_LATENCY_BUCKETS,
                ts_start=query_ts,
            )
            candidate_nodes = candidate_kb.diagnostic_clause_latency_candidates(
                row.repo,
                row.bin,
                row.argv,
                CANONICAL_LATENCY_BUCKETS,
                ts_start=query_ts,
            )
            if not candidate_nodes or candidate_nodes[0] != candidate_prediction:
                raise AssertionError(
                    "Candidate R diagnostic nodes differ from runtime prediction"
                )
            candidate_public = next(
                (item for item in candidate_nodes if item.scope == "public"),
                None,
            )
            label = CANONICAL_LATENCY_BUCKETS.bucket_id(row.latency_ms)
            scored["current"].append(_scored_row(row, clause_index, prediction))
            scored["public_only"].append(
                _scored_row(
                    row,
                    clause_index,
                    public,
                    None if public is not None else "no_public_evidence",
                )
            )
            scored["local_only"].append(
                _scored_row(
                    row,
                    clause_index,
                    local,
                    None if local is not None else "no_local_evidence",
                )
            )
            scored["candidate_r_public_only"].append(
                _scored_row(
                    row,
                    clause_index,
                    candidate_public,
                    (
                        None
                        if candidate_public is not None
                        else "no_structured_public_evidence"
                    ),
                )
            )
            scored["candidate_r"].append(
                _scored_row(row, clause_index, candidate_prediction)
            )
            if shrinkage_kb is not None:
                shrinkage_prediction = (
                    shrinkage_kb.predict_clause_latency_bucket(
                        row.repo,
                        row.bin,
                        row.argv,
                        CANONICAL_LATENCY_BUCKETS,
                        ts_start=query_ts,
                    )
                )
                scored["candidate_s"].append(
                    _scored_row(row, clause_index, shrinkage_prediction)
                )
            current_public = [prediction]
            if public is not None and public != prediction:
                current_public.append(public)
            scored["current_public_oracle"].append(
                _scored_row(
                    row,
                    clause_index,
                    _oracle_prediction(current_public, label, prediction),
                )
            )
            scored["node_oracle"].append(
                _scored_row(
                    row,
                    clause_index,
                    _oracle_prediction(
                        _bounded_node_oracle_candidates(
                            candidates,
                            candidate_nodes,
                        ),
                        label,
                        prediction,
                    ),
                )
            )
        for row in rows:
            observation = row.observation(query_ts, settle_ts)
            kb_by_repo[row.repo].observe_completed_clause(observation)
            candidate_kb_by_repo[row.repo].observe_completed_clause(observation)
            if row.repo in shrinkage_kb_by_repo:
                shrinkage_kb_by_repo[row.repo].observe_completed_clause(observation)
    if not scored["current"]:
        raise ValueError("no eligible clause telemetry observations to score")
    return scored


def _telemetry_scored_rows(
    fit_rows: Sequence[Row], eval_rows: Sequence[Row]
) -> list[ScoredRow]:
    """Compatibility helper returning the deployable current arm only."""

    return _telemetry_scored_arms(fit_rows, eval_rows)["current"]


def _support_band(evidence_count: int) -> str:
    if evidence_count == 1:
        return "1"
    if evidence_count <= 4:
        return "2-4"
    return "5+"


def _telemetry_metrics(
    rows: Sequence[ScoredRow],
    *,
    current_accuracy: float | None = None,
) -> dict[str, Any]:
    bucket_count = CANONICAL_LATENCY_BUCKETS.bucket_count
    if bucket_count != 3:
        raise AssertionError(f"canonical latency objective has {bucket_count} classes")
    if not rows:
        raise ValueError("no rows to score")
    label_counts: Counter[int] = Counter(row.label_bucket for row in rows)
    majority_bucket, majority_count = min(
        label_counts.most_common(),
        key=lambda item: (-item[1], item[0]),
    )
    known = [row for row in rows if row.probability_by_bucket is not None]
    confusion = [[0] * bucket_count for _ in range(bucket_count)]
    predicted_counts: Counter[int] = Counter()
    for row in known:
        predicted = _argmax_bucket(row)
        confusion[row.label_bucket][predicted] += 1
        predicted_counts[predicted] += 1
    class_names = ("short", "middle", "long")
    per_class = []
    for bucket in range(bucket_count):
        support = label_counts.get(bucket, 0)
        predicted_total = predicted_counts.get(bucket, 0)
        per_class.append(
            {
                "class": class_names[bucket],
                "class_id": bucket,
                "label_count": support,
                "label_share": support / len(rows),
                "predicted_count": predicted_total,
                "predicted_share": predicted_total / len(rows),
            }
        )
    available_accuracy = _exact_bucket_metrics(known)["three_class_accuracy"]
    complete_accuracy = available_accuracy if len(known) == len(rows) else None
    majority_accuracy = majority_count / len(rows)
    if current_accuracy is None and complete_accuracy is not None:
        current_accuracy = complete_accuracy
    support_counts = Counter(_support_band(row.evidence_count) for row in known)
    evidence_count_counts = Counter(row.evidence_count for row in known)
    return {
        "eligible_examples": len(rows),
        "three_class_accuracy": complete_accuracy,
        "available_only_accuracy": available_accuracy,
        "prediction_available": len(known),
        "prediction_coverage": len(known) / len(rows),
        "majority_class": class_names[majority_bucket],
        "majority_class_id": majority_bucket,
        "majority_class_accuracy": majority_accuracy,
        "current_accuracy": current_accuracy,
        "accuracy_minus_majority_percentage_points": (
            None
            if complete_accuracy is None
            else 100.0 * (complete_accuracy - majority_accuracy)
        ),
        "accuracy_minus_current_percentage_points": (
            None
            if complete_accuracy is None or current_accuracy is None
            else 100.0 * (complete_accuracy - current_accuracy)
        ),
        "prediction_unavailable": len(rows) - len(known),
        "confusion_label_by_prediction": confusion,
        "per_class": per_class,
        "scope_counts": dict(sorted(Counter(row.layer for row in known).items())),
        "key_kind_counts": dict(sorted(Counter(row.key_kind for row in known).items())),
        "support_band_counts": {
            band: support_counts[band] for band in ("1", "2-4", "5+")
        },
        "evidence_count_counts": {
            str(count): frequency
            for count, frequency in sorted(evidence_count_counts.items())
        },
        "fallback_path_counts": dict(
            sorted(Counter(":".join(row.fallback_path or ()) for row in known).items())
        ),
        "canonicalizer_version_counts": dict(
            sorted(Counter(row.canonicalizer_version for row in known).items())
        ),
        "arbitration_counts": dict(
            sorted(Counter(row.arbitration for row in known).items())
        ),
        "local_key_kind_counts": dict(
            sorted(
                Counter(
                    row.local_key_kind
                    for row in known
                    if row.local_key_kind is not None
                ).items()
            )
        ),
        "public_key_kind_counts": dict(
            sorted(
                Counter(
                    row.public_key_kind
                    for row in known
                    if row.public_key_kind is not None
                ).items()
            )
        ),
        "local_support_band_counts": dict(
            sorted(
                Counter(
                    _support_band(row.local_evidence_count)
                    for row in known
                    if row.local_evidence_count > 0
                ).items()
            )
        ),
        "public_support_band_counts": dict(
            sorted(
                Counter(
                    _support_band(row.public_evidence_count)
                    for row in known
                    if row.public_evidence_count > 0
                ).items()
            )
        ),
        "shrinkage_alpha_counts": {
            str(alpha): count
            for alpha, count in sorted(
                Counter(
                    row.shrinkage_alpha
                    for row in known
                    if row.shrinkage_alpha is not None
                ).items()
            )
        },
    }


def _record_resource_prediction(
    raw: dict[str, Any],
    label: bool,
    prediction: ClauseHeavyLightPrediction | None,
) -> None:
    if prediction is None:
        raw["prediction_unavailable"] += 1
        return
    raw["provenance_counts"][
        f"{prediction.scope}:{prediction.key_kind}:"
        f"{prediction.canonicalizer_version}:{prediction.arbitration}"
    ] += 1
    predicted = prediction.label == "heavy"
    outcome = "tp" if label and predicted else "fn" if label else "fp" if predicted else "tn"
    raw[outcome] += 1


def _accuracy_delta(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else 100.0 * (left - right)


def evaluate_prequential_commands(
    public_rows: Sequence[Row],
    task_ids: Sequence[str],
    clause_rows: Sequence[Row],
    command_rows: Sequence[CommandRow],
    provenance: Mapping[str, Any],
    *,
    warmup_task_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate the command predictor after a causal same-repo warm-up."""

    if not 0 < warmup_task_count < len(task_ids):
        raise ValueError("warmup task count must leave at least one test task")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("target task order contains duplicates")
    target_repos = {repo_of(task_id) for task_id in task_ids}
    if len(target_repos) != 1:
        raise ValueError("same-repo prequential evaluation requires one target repo")
    target_repo = next(iter(target_repos))
    if any(row.repo == target_repo for row in public_rows):
        raise ValueError("public evidence contains the target repository")
    task_index = {task_id: index for index, task_id in enumerate(task_ids)}
    if any(
        row.task_id not in task_index
        or row.manifest_index != task_index[row.task_id]
        for row in (*clause_rows, *command_rows)
    ):
        raise ValueError("target rows differ from results.jsonl task order")

    clauses_by_task: dict[str, list[Row]] = defaultdict(list)
    commands_by_task: dict[str, list[CommandRow]] = defaultdict(list)
    for row in clause_rows:
        clauses_by_task[row.task_id].append(row)
    for row in command_rows:
        commands_by_task[row.task_id].append(row)
    public_evidence = [
        row
        for row in public_rows
        if row.structure_known and row.pipeline_position <= 0
    ]
    kb = ClauseResourceKB.fit_public(
        row.observation(0.0, 1.0) for row in public_evidence
    )
    for ordinal, task_id in enumerate(task_ids[:warmup_task_count]):
        query_ts = float(ordinal * 2 + 3)
        settle_ts = query_ts + 0.5
        for row in clauses_by_task[task_id]:
            kb.observe_completed_clause(row.observation(query_ts, settle_ts))

    snapshot = kb.to_json_obj()
    kbs = {
        "current_dynamic": ClauseResourceKB.from_json_obj(snapshot),
        "frozen_at_80": ClauseResourceKB.from_json_obj(snapshot),
    }
    latency_arms: dict[str, list[ScoredRow]] = {
        arm: [] for arm in kbs
    }
    resource_raw = {
        resource: {
            "label_source_counts": Counter(),
            "arms": {arm: _empty_confusion() for arm in kbs},
        }
        for resource in CANONICAL_RESOURCE_HEAVY_THRESHOLDS
    }
    sidecar: list[dict[str, Any]] = []
    test_ids = list(task_ids[warmup_task_count:])
    for ordinal, task_id in enumerate(test_ids, start=warmup_task_count):
        query_ts = float(ordinal * 2 + 3)
        for row in commands_by_task[task_id]:
            predictions: dict[str, dict[str, Any]] = {}
            resource_predictions: dict[
                str, Mapping[str, ClauseHeavyLightPrediction | None]
            ] = {}
            for arm, arm_kb in kbs.items():
                latency = arm_kb.predict_command_latency_bucket(
                    row.repo,
                    row.command,
                    query_ts,
                    CANONICAL_LATENCY_BUCKETS,
                )
                resources = arm_kb.predict_command_resource_classes(
                    row.repo,
                    row.command,
                    query_ts,
                )
                resource_predictions[arm] = resources.classifications
                latency_arms[arm].append(
                    _command_scored_row(
                        row,
                        latency.prediction,
                        latency.unavailable_reason,
                    )
                )
                predictions[arm] = {
                    "latency": (
                        None
                        if latency.prediction is None
                        else _argmax_probabilities(
                            latency.prediction.probability_by_bucket
                        )
                    ),
                    **{
                        resource: None if prediction is None else prediction.label
                        for resource, prediction in resources.classifications.items()
                    },
                }
            resource_labels: dict[str, bool | None] = {}
            resource_sources: dict[str, str] = {}
            for resource in CANONICAL_RESOURCE_HEAVY_THRESHOLDS:
                label, source = command_resource_label(row, resource)
                resource_labels[resource] = label
                resource_sources[resource] = source
                resource_raw[resource]["label_source_counts"][source] += 1
                if label is None:
                    continue
                for arm in kbs:
                    _record_resource_prediction(
                        resource_raw[resource]["arms"][arm],
                        label,
                        resource_predictions[arm][resource],
                    )
            sidecar.append(
                {
                    "sample_id": f"{row.task_id}:{row.call_index}",
                    "task_id": row.task_id,
                    "command": row.command,
                    "clause_count": len(row.clauses),
                    "latency_label": CANONICAL_LATENCY_BUCKETS.bucket_id(
                        row.duration_ms
                    ),
                    "resource_labels": resource_labels,
                    "resource_label_sources": resource_sources,
                    **predictions,
                }
            )
        settle_ts = query_ts + 0.5
        for row in clauses_by_task[task_id]:
            kbs["current_dynamic"].observe_completed_clause(
                row.observation(query_ts, settle_ts)
            )

    latency_identity = [
        (row.sample_id, row.label_bucket) for row in latency_arms["current_dynamic"]
    ]
    if latency_identity != [
        (row.sample_id, row.label_bucket) for row in latency_arms["frozen_at_80"]
    ]:
        raise AssertionError("command latency arms differ in rows or labels")
    dynamic_latency = _telemetry_metrics(latency_arms["current_dynamic"])
    frozen_latency = _telemetry_metrics(latency_arms["frozen_at_80"])
    resources: dict[str, Any] = {}
    for resource, raw in resource_raw.items():
        dynamic = _finalize_resource_metric(
            raw["arms"]["current_dynamic"],
            raw["label_source_counts"],
        )
        frozen = _finalize_resource_metric(
            raw["arms"]["frozen_at_80"],
            raw["label_source_counts"],
        )
        resources[resource] = {
            "threshold": CANONICAL_RESOURCE_HEAVY_THRESHOLDS[resource],
            "majority": {
                "class": dynamic["majority_class"],
                "accuracy": dynamic["majority_class_accuracy"],
            },
            "constant_light_accuracy": dynamic["majority_light_accuracy"],
            "current_dynamic": dynamic,
            "frozen_at_80": frozen,
            "dynamic_minus_frozen_percentage_points": _accuracy_delta(
                dynamic["accuracy"], frozen["accuracy"]
            ),
            "dynamic_minus_majority_percentage_points": _accuracy_delta(
                dynamic["accuracy"], dynamic["majority_class_accuracy"]
            ),
        }
    result = {
        "status": "development_exposed_same_repo_command_prequential_baseline",
        "claim_bearing": False,
        "objective": "command_latency_and_resource_prediction",
        "inputs": dict(provenance),
        "protocol": {
            "evaluation_unit": "eligible_exec_command",
            "warmup_task_count": warmup_task_count,
            "test_task_count": len(test_ids),
            "public_layer": "frozen cross-repository bin/global",
            "local_layer": "causal repository exact/argv-prefix/bin",
            "arbitration": "hard repository-first first-nonempty",
            "dynamic_update": "after successful whole-task finalization",
            "majority": "constant class from the identical eligible test labels",
            "compound_composition": "empirical-shell-graph-v1",
        },
        "tasks": {
            "ordered": list(task_ids),
            "warmup": list(task_ids[:warmup_task_count]),
            "test": test_ids,
        },
        "counts": {
            "public_clause_observations": len(public_evidence),
            "target_clause_observations": len(clause_rows),
            "target_commands": len(command_rows),
            "test_commands": len(sidecar),
        },
        "row_identity": {
            "identical_command_rows_and_labels_across_arms": True,
            "test_sample_ids_unique": len({row["sample_id"] for row in sidecar})
            == len(sidecar),
        },
        "latency": {
            "bucket_edges_ms": list(CANONICAL_LATENCY_BUCKETS.edges_ms),
            "bucket_intervals": _bucket_intervals(),
            "majority": {
                "class": dynamic_latency["majority_class"],
                "class_id": dynamic_latency["majority_class_id"],
                "accuracy": dynamic_latency["majority_class_accuracy"],
            },
            "current_dynamic": dynamic_latency,
            "frozen_at_80": frozen_latency,
            "dynamic_minus_frozen_percentage_points": _accuracy_delta(
                dynamic_latency["three_class_accuracy"],
                frozen_latency["three_class_accuracy"],
            ),
        },
        "resources": resources,
        "label_policy": {
            "latency_truth": "tool_calls.json command duration_ms",
            "resource_truth": (
                "strict bounds from retained clause aggregates; ambiguous "
                "pipeline peaks are unavailable"
            ),
            "short_null_light_max_latency_ms_exclusive": (
                SHORT_NULL_LIGHT_MAX_LATENCY_MS
            ),
        },
    }
    return result, sidecar


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _paired_repo_cluster_bootstrap(
    current: Sequence[ScoredRow],
    candidate: Sequence[ScoredRow],
    *,
    seed: int = 0,
    draws: int = 2000,
    candidate_name: str = "candidate_r",
) -> dict[str, Any]:
    """Paired repository-cluster uncertainty for Candidate R's exact score."""

    identity = [(row.sample_id, row.label_bucket, row.repo) for row in current]
    if identity != [
        (row.sample_id, row.label_bucket, row.repo) for row in candidate
    ]:
        raise AssertionError("bootstrap arms have different rows or labels")
    if any(
        row.probability_by_bucket is None for row in (*current, *candidate)
    ):
        raise ValueError("bootstrap refuses unavailable predictions")
    indices_by_repo: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(current):
        indices_by_repo[row.repo].append(index)
    repositories = sorted(indices_by_repo)
    rng = random.Random(seed)

    def statistic(indices: Sequence[int]) -> float:
        current_correct = sum(
            _argmax_bucket(current[index]) == current[index].label_bucket
            for index in indices
        )
        candidate_correct = sum(
            _argmax_bucket(candidate[index]) == candidate[index].label_bucket
            for index in indices
        )
        labels = Counter(current[index].label_bucket for index in indices)
        majority_count = max(labels.values())
        denominator = len(indices)
        return candidate_correct / denominator - max(
            current_correct / denominator,
            majority_count / denominator,
        )

    point = statistic(range(len(current)))
    samples = []
    for _ in range(draws):
        sampled_repositories = [
            repositories[rng.randrange(len(repositories))]
            for _ in range(len(repositories))
        ]
        sampled_indices = [
            index
            for repo in sampled_repositories
            for index in indices_by_repo[repo]
        ]
        samples.append(statistic(sampled_indices))
    lower = _percentile(samples, 0.025)
    median = _percentile(samples, 0.5)
    upper = _percentile(samples, 0.975)
    return {
        "method": "paired_repository_cluster_percentile_bootstrap",
        "cluster_unit": "repository",
        "seed": seed,
        "draws": draws,
        "statistic": (
            f"{candidate_name}_accuracy_minus_max_current_same_resample_majority"
        ),
        "point_estimate": point,
        "median": median,
        "interval_95": [lower, upper],
        "positive_draw_fraction": sum(value > 0.0 for value in samples) / draws,
        "sign_positive": median > 0.0,
        "confirmatory": False,
    }


def evaluate_clause_telemetry(
    fit_rows: Sequence[Row],
    eval_rows: Sequence[Row],
    provenance: Mapping[str, Any],
    *,
    include_candidate_s: bool = False,
) -> tuple[dict[str, Any], list[ScoredRow]]:
    """Evaluate latency buckets on canonical eBPF clause telemetry."""

    selection = _select_shrinkage_alpha(fit_rows) if include_candidate_s else None
    arms = _telemetry_scored_arms(
        fit_rows,
        eval_rows,
        shrinkage_alpha=(
            None if selection is None else float(selection["selected_alpha"])
        ),
    )
    rows = arms["current"]
    identity = [(row.sample_id, row.label_bucket) for row in rows]
    if any(
        identity != [(row.sample_id, row.label_bucket) for row in arm_rows]
        for arm_rows in arms.values()
    ):
        raise AssertionError("baseline/oracle row identity or labels differ")
    current_metrics = _telemetry_metrics(rows)
    current_accuracy = current_metrics["three_class_accuracy"]
    assert current_accuracy is not None
    candidate_metrics = _telemetry_metrics(
        arms["candidate_r"],
        current_accuracy=current_accuracy,
    )
    majority = {
        "class": current_metrics["majority_class"],
        "class_id": current_metrics["majority_class_id"],
        "accuracy": current_metrics["majority_class_accuracy"],
        "eligible_examples": len(rows),
    }
    result = {
            "status": "development_exposed_canonical_clause_telemetry",
            "claim_bearing": False,
            "objective": "clause_latency_bucket_prediction",
            "bucket_edges_ms": list(CANONICAL_LATENCY_BUCKETS.edges_ms),
            "bucket_intervals": _bucket_intervals(),
            "fit_clause_observation_count": len(fit_rows),
            "eval_clause_observation_count": len(eval_rows),
            "row_identity": {
                "identical_row_ids_and_labels": True,
                "eligible_row_count": len(rows),
            },
            "baselines": {
                "majority": majority,
                "current": current_metrics,
                "public_only": _telemetry_metrics(
                    arms["public_only"],
                    current_accuracy=current_accuracy,
                ),
                "local_only_diagnostic": {
                    **_telemetry_metrics(
                        arms["local_only"],
                        current_accuracy=current_accuracy,
                    ),
                    "selection_forbidden": True,
                },
            },
            "candidates": {
                "candidate_r_public_only": {
                    **_telemetry_metrics(
                        arms["candidate_r_public_only"],
                        current_accuracy=current_accuracy,
                    ),
                    "representation_only_diagnostic": True,
                },
                "candidate_r": candidate_metrics,
            },
            "oracles": {
                "current_public": {
                    **_telemetry_metrics(
                        arms["current_public_oracle"],
                        current_accuracy=current_accuracy,
                    ),
                    "oracle": True,
                    "deployable": False,
                },
                "current_and_candidate_nodes": {
                    **_telemetry_metrics(
                        arms["node_oracle"],
                        current_accuracy=current_accuracy,
                    ),
                    "oracle": True,
                    "deployable": False,
                    "candidate_nodes": (
                        "deployable current prediction as the non-oracle fallback; "
                        "hindsight choice is limited to repo exact/structured/bin "
                        "and public structured/bin/global"
                    ),
                },
            },
            "uncertainty": {
                "candidate_r_vs_best_baseline": _paired_repo_cluster_bootstrap(
                    arms["current"],
                    arms["candidate_r"],
                )
            },
            "metrics": current_metrics,
            "provenance": dict(provenance),
        }
    if selection is not None:
        shrinkage_metrics = _telemetry_metrics(
            arms["candidate_s"],
            current_accuracy=current_accuracy,
        )
        result["selection"] = {"candidate_s_alpha": selection}
        result["candidates"]["candidate_s"] = shrinkage_metrics
        result["uncertainty"]["candidate_s_vs_best_baseline"] = (
            _paired_repo_cluster_bootstrap(
                arms["current"],
                arms["candidate_s"],
                candidate_name="candidate_s",
            )
        )
    return result, rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # Canonical eBPF clause telemetry: one aggregated replay JSONL per corpus,
    # the same artifact evaluate_clause_resource_classes.py consumes.
    parser.add_argument("--telemetry-fit", type=Path, required=True)
    parser.add_argument("--telemetry-eval", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dump-rows", type=Path)
    parser.add_argument(
        "--candidate-s",
        action="store_true",
        help="select Candidate S alpha on fit-repository folds and score it",
    )
    return parser


def _write(args: argparse.Namespace, result: Mapping[str, Any], rows: Sequence[Any]) -> None:
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        sys.stdout.write(payload)
    else:
        args.out.write_text(payload, encoding="utf-8")
    if args.dump_rows is not None:
        args.dump_rows.write_text(
            "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )


def _run_clause_telemetry(args: argparse.Namespace) -> None:
    fit_rows = load_rows(args.telemetry_fit)
    eval_rows = load_rows(args.telemetry_eval)
    fit_tasks = sorted({row.task_id for row in fit_rows})
    eval_tasks = sorted({row.task_id for row in eval_rows})
    provenance = {
        **_validate_partition(fit_tasks, eval_tasks),
        "fit_telemetry": str(args.telemetry_fit.resolve()),
        "eval_telemetry": str(args.telemetry_eval.resolve()),
        "evidence_source": (
            "canonical eBPF clause telemetry; eligible_for_kb call and clause gates"
        ),
        "scoring_unit": "kb_eligible_clause",
        "public_prior": "leave_one_repo_out over the fit corpus",
        "candidate_r": {
            "representation": STRUCTURED_ARGV_REPRESENTATION,
            "stable_subcommand_min_distinct_fit_repositories": 3,
            "stable_subcommand_uses_labels": False,
            "repo_hierarchy": "raw exact, structured argv, bin",
            "public_hierarchy": "structured argv, bin, global",
            "arbitration": "same hard first-nonempty selection as current",
        },
        "candidate_s": (
            {
                "enabled": True,
                "representation": STRUCTURED_ARGV_REPRESENTATION,
                "arbitration": "deepest local plus deepest public posterior",
                "alpha_grid": list(SHRINKAGE_ALPHA_GRID),
                "alpha_selection": (
                    "repository-grouped inner fit folds; latency accuracy only; "
                    "larger alpha on exact tie"
                ),
                "same_alpha_for_all_targets": True,
            }
            if args.candidate_s
            else {"enabled": False}
        ),
        "bootstrap": {
            "seed": 0,
            "draws": 2000,
            "cluster_unit": "repository",
            "interval": "95% percentile",
            "positive_sign_rule": "bootstrap median > 0",
        },
        "task_order": "(manifest_index, task_id) ascending",
        "trace_update": (
            "predict every clause of a task before any of that task's observations "
            "settle; settled clauses are repo-local evidence for later tasks only"
        ),
        "unavailable_policy": (
            "missing evidence raises; no synthetic, imputed, or majority fallback"
        ),
        "edge_source": "canonical_objective",
    }
    result, rows = evaluate_clause_telemetry(
        fit_rows,
        eval_rows,
        provenance,
        include_candidate_s=args.candidate_s,
    )
    _write(args, result, rows)


def main() -> None:
    _run_clause_telemetry(_parser().parse_args())


if __name__ == "__main__":
    main()
