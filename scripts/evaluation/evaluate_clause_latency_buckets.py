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
import re
import sys
import time
import tracemalloc
from bisect import bisect_right
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
from tool_resource.clause_parser import parse_command_clauses  # noqa: E402
from tool_resource.pip_semantics import (  # noqa: E402
    PipInstallSignature,
    PipQueryState,
    PipTaskState,
    apt_installs_system_pip,
    parse_pip_install,
)
from tool_resource.runtime_kb import (  # noqa: E402
    CANONICAL_LATENCY_BUCKETS,
    CANONICAL_RESOURCE_HEAVY_THRESHOLDS,
    COMMAND_COMPOSITION_DRAWS,
    DEFAULT_HEAVY_DECISION_THRESHOLD,
    SHRINKAGE_ALPHA_GRID,
    SHORT_NULL_LIGHT_MAX_LATENCY_MS,
    STRUCTURED_ARGV_REPRESENTATION,
    ClauseHeavyLightPrediction,
    ClauseLatencyBucketPrediction,
    ClauseResourceKB,
    _canonical_dynamic_value,
    _command_stages,
    _fit_stable_subcommands,
    _structured_argv_parts,
)

INTERACTION_ALPHA = 16.0
INTERACTION_FEATURE_VERSION = "generic-argv-v3-role-interaction-set-v1"
PIP_SEMANTIC_FEATURE_VERSION = "pip-install-semantic-jaccard-v1"
PIP_CANONICAL_FEATURE_VERSION = "pip-install-canonical-exact-v1"


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
class _InteractionQueryClause:
    bin: str
    argv: tuple[str, ...]
    in_pipe: bool
    in_subst: bool
    pipeline_position: int


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


@dataclass(frozen=True)
class _WeightedClauseEvidence:
    values: tuple[float, ...]
    weights: tuple[float, ...]
    exact: bool
    nonexact_local: bool
    local_observation_count: int
    effective_sample_size: float
    contributing_task_ids: frozenset[str]
    max_shared_feature_count: int
    key_kind: str
    public_key_kind: str | None


@dataclass(frozen=True)
class PipExecEvent:
    call_id: str
    command: str
    tool_result: str


@dataclass(frozen=True)
class _PipObservation:
    observation_id: int
    task_id: str
    row: Row
    signature: PipInstallSignature
    state: PipQueryState


@dataclass(frozen=True)
class _PipMatch:
    evidence: _WeightedClauseEvidence
    state_fallback: bool
    contributors: tuple[Mapping[str, Any], ...] = ()


def _interaction_feature_set(
    bin_: str,
    argv: Sequence[str],
    stable_subcommands: frozenset[tuple[str, str]],
) -> InteractionFeatureSet:
    """Build the frozen typed set without raw dynamic argument values."""

    subcommand, options, positionals = _structured_argv_parts(argv)
    features = set()
    if subcommand is not None:
        features.add(
            f"subcommand:{subcommand}"
            if (bin_, subcommand) in stable_subcommands
            else f"subcommand:{_canonical_dynamic_value(subcommand)}"
        )
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

    def query(self, row: Row | _InteractionQueryClause) -> _PosetMatch:
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

    def query(self, row: Row | _InteractionQueryClause) -> _KernelMatch:
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


def _effective_sample_size(weights: Sequence[float]) -> float:
    total = sum(weights)
    squared = sum(weight * weight for weight in weights)
    return total * total / squared if squared else 0.0


def _pip_partition(signature: PipInstallSignature) -> tuple[str, str, tuple[str, ...]]:
    return signature.interpreter, signature.invocation, signature.flags


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = frozenset(left)
    right_set = frozenset(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


class _PipSemanticKB:
    """Slow episodic pip memory for the frozen development experiment."""

    def __init__(self, public_rows: Sequence[Row]) -> None:
        self._public_rows = tuple(public_rows)
        self._next_observation_id = len(self._public_rows)
        self._exact: dict[
            tuple[str, tuple[str, ...]], list[_PipObservation]
        ] = defaultdict(list)
        self._local: dict[
            tuple[str, str, tuple[str, ...]], list[_PipObservation]
        ] = defaultdict(list)
        self._public: dict[
            tuple[str, str, tuple[str, ...]], list[_PipObservation]
        ] = defaultdict(list)
        for observation_id, row in enumerate(self._public_rows):
            signature = parse_pip_install(row.argv)
            if signature is None:
                continue
            self._public[_pip_partition(signature)].append(
                _PipObservation(
                    observation_id=observation_id,
                    task_id=row.task_id,
                    row=row,
                    signature=signature,
                    state=PipQueryState("unknown", signature.package_names),
                )
            )

    def observe(
        self,
        task_id: str,
        row: Row,
        signature: PipInstallSignature,
        state: PipQueryState,
    ) -> None:
        observation = _PipObservation(
            observation_id=self._next_observation_id,
            task_id=task_id,
            row=row,
            signature=signature,
            state=state,
        )
        self._next_observation_id += 1
        self._exact[(row.bin, row.argv[1:])].append(observation)
        self._local[_pip_partition(signature)].append(observation)

    @property
    def observation_count(self) -> int:
        return self._next_observation_id - len(self._public_rows)

    def query(
        self,
        row: Row | _InteractionQueryClause,
        signature: PipInstallSignature,
        state: PipQueryState,
        *,
        use_state: bool,
        canonical_exact: bool = False,
        current_selected: tuple[
            Sequence[float], str, str, tuple[str, ...]
        ],
        current_public_selected: tuple[
            Sequence[float], str, str, tuple[str, ...]
        ],
    ) -> _PipMatch:
        if use_state and canonical_exact:
            raise ValueError("canonical-exact pip matching does not use task state")
        exact = tuple(self._exact.get((row.bin, row.argv[1:]), ()))
        if exact:
            return _PipMatch(
                evidence=_WeightedClauseEvidence(
                    values=tuple(item.row.latency_ms for item in exact),
                    weights=(1.0,) * len(exact),
                    exact=True,
                    nonexact_local=False,
                    local_observation_count=len(exact),
                    effective_sample_size=float(len(exact)),
                    contributing_task_ids=frozenset(item.task_id for item in exact),
                    max_shared_feature_count=len(signature.package_names),
                    key_kind="exact_clause",
                    public_key_kind=None,
                ),
                state_fallback=False,
            )

        local = tuple(self._local.get(_pip_partition(signature), ()))
        state_fallback = False
        query_packages: Sequence[str] = signature.package_names
        if use_state:
            state_local = tuple(
                item
                for item in local
                if item.state.availability == state.availability
            )
            if state_local:
                local = state_local
                query_packages = state.remaining_packages
            elif local:
                state_fallback = True
        if canonical_exact:
            local_pairs = tuple(
                (item, 1.0) for item in local if item.signature == signature
            )
            public_pairs = tuple(
                (item, 1.0)
                for item in self._public.get(_pip_partition(signature), ())
                if item.signature == signature
            )
        else:
            local_pairs = tuple(
                (item, weight)
                for item in local
                if (
                    weight := _jaccard(
                        query_packages,
                        (
                            item.state.remaining_packages
                            if use_state and not state_fallback
                            else item.signature.package_names
                        ),
                    )
                )
                > 0.0
            )
            public_pairs = tuple(
                (item, weight)
                for item in self._public.get(_pip_partition(signature), ())
                if (
                    weight := _jaccard(
                        signature.package_names, item.signature.package_names
                    )
                )
                > 0.0
            )
        if not public_pairs and (not local_pairs or not canonical_exact):
            values, scope, kind, _path = current_selected
            return _PipMatch(
                evidence=_WeightedClauseEvidence(
                    values=tuple(values),
                    weights=(1.0,) * len(values),
                    exact=scope == "repo" and kind == "exact_clause",
                    nonexact_local=False,
                    local_observation_count=len(values) if scope == "repo" else 0,
                    effective_sample_size=float(len(values)),
                    contributing_task_ids=frozenset(),
                    max_shared_feature_count=0,
                    key_kind=f"current_{scope}_{kind}",
                    public_key_kind=kind if scope == "public" else None,
                ),
                state_fallback=state_fallback,
            )

        local_weights = tuple(weight for _item, weight in local_pairs)
        effective_n = _effective_sample_size(local_weights)
        local_total = sum(local_weights)
        normalized_local = tuple(
            weight * effective_n / local_total for weight in local_weights
        )
        if public_pairs:
            public_values = tuple(
                item.row.latency_ms for item, _weight in public_pairs
            )
            public_total = sum(weight for _item, weight in public_pairs)
            normalized_public = tuple(
                weight * INTERACTION_ALPHA / public_total
                for _item, weight in public_pairs
            )
            public_key_kind = (
                "pip_canonical_exact" if canonical_exact else "pip_semantic"
            )
        else:
            _selected_values, selected_scope, selected_kind, _path = (
                current_public_selected
            )
            if selected_scope != "public" or selected_kind not in {"bin", "global"}:
                raise AssertionError("canonical prior is not a Current public node")
            prior_rows = tuple(
                (observation_id, item)
                for observation_id, item in enumerate(self._public_rows)
                if selected_kind == "global" or item.bin == row.bin
            )
            public_values = tuple(item.latency_ms for _id, item in prior_rows)
            if sorted(public_values) != list(_selected_values):
                raise AssertionError("canonical prior differs from Current public node")
            normalized_public = (INTERACTION_ALPHA / len(public_values),) * len(
                public_values
            )
            public_key_kind = f"current_public_{current_public_selected[2]}"
        shared = tuple(
            len(set(query_packages) & set(item.signature.package_names))
            for item, _weight in local_pairs
        )
        return _PipMatch(
            evidence=_WeightedClauseEvidence(
                values=(
                    *(item.row.latency_ms for item, _weight in local_pairs),
                    *public_values,
                ),
                weights=(*normalized_local, *normalized_public),
                exact=False,
                nonexact_local=bool(local_pairs),
                local_observation_count=len(local_pairs),
                effective_sample_size=effective_n,
                contributing_task_ids=frozenset(
                    item.task_id for item, _weight in local_pairs
                ),
                max_shared_feature_count=max(shared, default=0),
                key_kind=(
                    "pip_canonical_exact"
                    if canonical_exact and local_pairs
                    else "pip_canonical_exact_public"
                    if canonical_exact
                    else "pip_semantic_state"
                    if use_state and local_pairs and not state_fallback
                    else "pip_semantic"
                    if local_pairs
                    else "pip_semantic_public"
                ),
                public_key_kind=public_key_kind,
            ),
            state_fallback=state_fallback,
            contributors=(
                *(
                    {
                        "scope": "repo",
                        "observation_id": item.observation_id,
                        "task_id": item.task_id,
                        "manifest_index": item.row.manifest_index,
                        "signature": asdict(item.signature),
                        "value": item.row.latency_ms,
                        "similarity_weight": raw_weight,
                        "pooled_weight": pooled_weight,
                    }
                    for (item, raw_weight), pooled_weight in zip(
                        local_pairs, normalized_local, strict=True
                    )
                ),
                *(
                    {
                        "scope": "public",
                        "observation_id": item.observation_id,
                        "task_id": item.task_id,
                        "manifest_index": item.row.manifest_index,
                        "signature": asdict(item.signature),
                        "value": item.row.latency_ms,
                        "similarity_weight": raw_weight,
                        "pooled_weight": pooled_weight,
                    }
                    for (item, raw_weight), pooled_weight in (
                        zip(public_pairs, normalized_public, strict=True)
                        if public_pairs
                        else ()
                    )
                ),
                *(
                    {
                        "scope": "public",
                        "observation_id": observation_id,
                        "task_id": item.task_id,
                        "manifest_index": item.manifest_index,
                        "signature": (
                            None
                            if (prior_signature := parse_pip_install(item.argv)) is None
                            else asdict(prior_signature)
                        ),
                        "value": item.latency_ms,
                        "similarity_weight": None,
                        "pooled_weight": pooled_weight,
                        "prior_key_kind": public_key_kind,
                    }
                    for (observation_id, item), pooled_weight in (
                        zip(prior_rows, normalized_public, strict=True)
                        if not public_pairs
                        else ()
                    )
                ),
            ),
        )


def _pip_query_contexts(
    command: str,
    state: PipTaskState,
) -> tuple[Mapping[str, Any], tuple[PipQueryState | None, ...]]:
    parsed = parse_command_clauses(command)
    clauses = parsed.get("clauses", ())
    signatures = tuple(
        parse_pip_install(clause.get("argv", ()))
        for clause in clauses
    )
    conditional_present: set[int] = set()
    for edge in parsed.get("control_edges", ()):
        if edge.get("operator") != "&&":
            continue
        lhs = edge.get("lhs", {}).get("clause_indices", ())
        rhs = edge.get("rhs", {}).get("clause_indices", ())
        if any(
            apt_installs_system_pip(clauses[index].get("argv", ()))
            for index in lhs
        ):
            conditional_present.update(
                index
                for index in rhs
                if signatures[index] is not None
                and signatures[index].interpreter == "python3"
            )
    return parsed, tuple(
        None
        if signature is None
        else state.query(
            signature,
            conditional_present=index in conditional_present,
        )
        for index, signature in enumerate(signatures)
    )


def _pip_contexts_by_call(
    events: Sequence[PipExecEvent],
) -> dict[str, tuple[PipQueryState | None, ...]]:
    state = PipTaskState()
    contexts: dict[str, tuple[PipQueryState | None, ...]] = {}
    for event in events:
        if event.call_id in contexts:
            raise ValueError(f"duplicate tool event {event.call_id}")
        _parsed, before = _pip_query_contexts(event.command, state)
        contexts[event.call_id] = before
        state.observe(event.command, event.tool_result)
    return contexts


def _current_weighted_evidence(
    selected: tuple[Sequence[float], str, str, tuple[str, ...]],
) -> _WeightedClauseEvidence:
    values, scope, kind, _path = selected
    return _WeightedClauseEvidence(
        values=tuple(values),
        weights=(1.0,) * len(values),
        exact=scope == "repo" and kind == "exact_clause",
        nonexact_local=False,
        local_observation_count=len(values) if scope == "repo" else 0,
        effective_sample_size=float(len(values)),
        contributing_task_ids=frozenset(),
        max_shared_feature_count=0,
        key_kind=f"current_{scope}_{kind}",
        public_key_kind=kind if scope == "public" else None,
    )


def _pip_composition_seed(
    command: str,
    source: str,
    index: int,
    evidence: _WeightedClauseEvidence,
) -> str:
    if evidence.exact:
        scope, key_kind = "repo", "exact_clause"
    elif evidence.key_kind.startswith("current_"):
        scope, _, key_kind = evidence.key_kind.removeprefix("current_").partition("_")
    else:
        scope = "repo+public" if evidence.nonexact_local else "public"
        key_kind = "pip_semantic_weighted_pool"
    return f"{command}\0{source}\0{index}\0{scope}\0{key_kind}"


def _predict_pip_latency_command(
    memory: _PipSemanticKB,
    current: ClauseResourceKB,
    row: CommandRow,
    parsed: Mapping[str, Any],
    states: Sequence[PipQueryState | None],
    current_prediction: ClauseLatencyBucketPrediction,
    *,
    use_state: bool,
    canonical_exact: bool = False,
) -> tuple[ClauseLatencyBucketPrediction, dict[str, Any]]:
    parsed_clauses = parsed.get("clauses", ())
    query_clauses = _interaction_query_clauses(parsed_clauses)
    signatures = tuple(
        parse_pip_install(clause.argv) for clause in query_clauses
    )
    if not any(signature is not None for signature in signatures):
        return current_prediction, {
            "carrier": False,
            "pip_clause_count": 0,
            "state_fallback": False,
            "clauses": [],
        }
    stages = _command_stages(parsed_clauses)
    if stages is None or len(states) != len(query_clauses):
        raise ValueError(f"{row.call_id}: pip query structure is unavailable")
    evidence: list[_WeightedClauseEvidence] = []
    state_fallback = False
    semantic_used = False
    clause_diagnostics: list[dict[str, Any]] = []
    for clause, signature, query_state in zip(
        query_clauses, signatures, states, strict=True
    ):
        selected = current._select(row.repo, "latency_ms", clause.bin, clause.argv)
        if selected is None:
            raise ValueError("Current has no clause evidence")
        _local_selected, public_selected = current._select_independent_scopes(
            row.repo, "latency_ms", clause.bin, clause.argv
        )
        if public_selected is None:
            raise ValueError("Current has no frozen public clause evidence")
        if signature is None:
            item = _current_weighted_evidence(selected)
            fallback = False
        else:
            if query_state is None:
                raise AssertionError("pip clause has no causal query state")
            match = memory.query(
                clause,
                signature,
                query_state,
                use_state=use_state,
                canonical_exact=canonical_exact,
                current_selected=selected,
                current_public_selected=public_selected,
            )
            item = match.evidence
            fallback = match.state_fallback
            state_fallback |= fallback
            semantic_used |= not item.exact and not item.key_kind.startswith(
                "current_"
            )
        evidence.append(item)
        clause_diagnostics.append(
            {
                "pip": signature is not None,
                "exact": item.exact,
                "key_kind": item.key_kind,
                "local_observation_count": item.local_observation_count,
                "effective_sample_size": item.effective_sample_size,
                "public_observation_count": (
                    len(item.values) - item.local_observation_count
                ),
                "public_key_kind": item.public_key_kind,
                "contributing_task_ids": sorted(item.contributing_task_ids),
                "contributors": list(match.contributors) if signature is not None else [],
                "signature": None if signature is None else asdict(signature),
                "state": None if query_state is None else query_state.availability,
                "remaining_packages": (
                    None if query_state is None else list(query_state.remaining_packages)
                ),
                "state_fallback": fallback,
            }
        )
    if not semantic_used:
        return current_prediction, {
            "carrier": False,
            "pip_clause_count": sum(signature is not None for signature in signatures),
            "state_fallback": state_fallback,
            "clauses": clause_diagnostics,
        }
    if len(evidence) == 1:
        probabilities = _weighted_bucket_probabilities(
            evidence[0].values,
            evidence[0].weights,
        )
    else:
        draws = tuple(
            _weighted_stratified_draws(
                item.values,
                item.weights,
                _pip_composition_seed(row.command, "latency_ms", index, item),
            )
            for index, item in enumerate(evidence)
        )
        composed = tuple(
            sum(max(draws[index][draw] for index in stage) for stage in stages)
            for draw in range(COMMAND_COMPOSITION_DRAWS)
        )
        probabilities = _weighted_bucket_probabilities(
            composed,
            (1.0,) * len(composed),
        )
    local_counts = [item.local_observation_count for item in evidence]
    public_counts = [len(item.values) - item.local_observation_count for item in evidence]
    prediction = ClauseLatencyBucketPrediction(
        probability_by_bucket=probabilities,
        scope="repo+public" if any(local_counts) else "public",
        key_kind=(
            evidence[0].key_kind if len(evidence) == 1 else "shell_execution_graph"
        ),
        evidence_count=min(len(item.values) for item in evidence),
        fallback_path=tuple(
            f"clause[{index}]:{item.key_kind}" for index, item in enumerate(evidence)
        ),
        canonicalizer_version=(
            PIP_CANONICAL_FEATURE_VERSION
            if canonical_exact
            else PIP_SEMANTIC_FEATURE_VERSION
        ),
        arbitration=(
            "current-or-pip-canonical-exact-v1"
            if canonical_exact
            else "current-or-pip-semantic-jaccard-v1"
        ),
        local_key_kind=(
            ("pip_canonical_exact" if canonical_exact else "pip_semantic")
            if any(local_counts)
            else None
        ),
        local_evidence_count=min(local_counts),
        public_key_kind="+".join(
            sorted(
                {
                    item.public_key_kind
                    for item in evidence
                    if item.public_key_kind is not None
                }
            )
        ),
        public_evidence_count=min(public_counts),
        shrinkage_alpha=INTERACTION_ALPHA,
    )
    return prediction, {
        "carrier": True,
        "pip_clause_count": sum(signature is not None for signature in signatures),
        "state_fallback": state_fallback,
        "clauses": clause_diagnostics,
    }


def _predict_pip_resources(
    memories: Mapping[str, _PipSemanticKB],
    current: ClauseResourceKB,
    row: CommandRow,
    parsed: Mapping[str, Any],
    states: Sequence[PipQueryState | None],
    current_predictions: Mapping[str, ClauseHeavyLightPrediction | None],
    *,
    use_state: bool,
) -> tuple[dict[str, ClauseHeavyLightPrediction | None], dict[str, Any]]:
    parsed_clauses = parsed.get("clauses", ())
    query_clauses = _interaction_query_clauses(parsed_clauses)
    signatures = tuple(parse_pip_install(clause.argv) for clause in query_clauses)
    if not any(signature is not None for signature in signatures):
        return dict(current_predictions), {
            resource: {"carrier": False, "clauses": []}
            for resource in CANONICAL_RESOURCE_HEAVY_THRESHOLDS
        }
    stages = _command_stages(parsed_clauses)
    if stages is None or len(states) != len(query_clauses):
        raise ValueError(f"{row.call_id}: pip resource structure is unavailable")

    predictions: dict[str, ClauseHeavyLightPrediction | None] = {}
    diagnostics: dict[str, Any] = {}
    for resource, threshold in CANONICAL_RESOURCE_HEAVY_THRESHOLDS.items():
        current_prediction = current_predictions.get(resource)
        if current_prediction is None:
            predictions[resource] = None
            diagnostics[resource] = {"carrier": False, "clauses": []}
            continue
        evidence: list[_WeightedClauseEvidence] = []
        semantic_used = False
        clause_diagnostics: list[dict[str, Any]] = []
        for clause, signature, state in zip(
            query_clauses, signatures, states, strict=True
        ):
            selected = current._select(row.repo, resource, clause.bin, clause.argv)
            if selected is None:
                raise ValueError(f"Current has no {resource} clause evidence")
            _local_selected, public_selected = current._select_independent_scopes(
                row.repo, resource, clause.bin, clause.argv
            )
            if public_selected is None:
                raise ValueError(f"Current has no frozen public {resource} evidence")
            if signature is None:
                item = _current_weighted_evidence(selected)
            else:
                if state is None:
                    raise AssertionError("pip clause has no causal query state")
                item = memories[resource].query(
                    clause,
                    signature,
                    state,
                    use_state=use_state,
                    current_selected=selected,
                    current_public_selected=public_selected,
                ).evidence
                semantic_used |= item.key_kind.startswith("pip_semantic")
            evidence.append(item)
            clause_diagnostics.append(
                {
                    "pip": signature is not None,
                    "exact": item.exact,
                    "key_kind": item.key_kind,
                    "local_observation_count": item.local_observation_count,
                }
            )
        if not semantic_used:
            predictions[resource] = current_prediction
            diagnostics[resource] = {
                "carrier": False,
                "clauses": clause_diagnostics,
            }
            continue
        if len(evidence) == 1:
            total = sum(evidence[0].weights)
            probability_heavy = sum(
                weight
                for value, weight in zip(
                    evidence[0].values,
                    evidence[0].weights,
                    strict=True,
                )
                if value > threshold
            ) / total
        else:
            draws = tuple(
                _weighted_stratified_draws(
                    item.values,
                    item.weights,
                    _pip_composition_seed(row.command, resource, index, item),
                )
                for index, item in enumerate(evidence)
            )
            composed = tuple(
                (
                    sum(
                        sum(draws[index][draw] for index in stage)
                        for stage in stages
                    )
                    if resource == "disk_read_write_bytes_total"
                    else max(
                        sum(draws[index][draw] for index in stage)
                        for stage in stages
                    )
                )
                for draw in range(COMMAND_COMPOSITION_DRAWS)
            )
            probability_heavy = sum(value > threshold for value in composed) / len(
                composed
            )
        local_counts = [item.local_observation_count for item in evidence]
        public_counts = [
            len(item.values) - item.local_observation_count for item in evidence
        ]
        predictions[resource] = ClauseHeavyLightPrediction(
            resource=resource,
            threshold=threshold,
            probability_heavy=probability_heavy,
            heavy_decision_threshold=DEFAULT_HEAVY_DECISION_THRESHOLD,
            label=(
                "heavy"
                if probability_heavy > DEFAULT_HEAVY_DECISION_THRESHOLD
                else "light"
            ),
            scope="repo+public" if any(local_counts) else "public",
            key_kind=(
                evidence[0].key_kind
                if len(evidence) == 1
                else "shell_execution_graph"
            ),
            evidence_count=min(len(item.values) for item in evidence),
            fallback_path=tuple(
                f"clause[{index}]:{item.key_kind}"
                for index, item in enumerate(evidence)
            ),
            canonicalizer_version=PIP_SEMANTIC_FEATURE_VERSION,
            arbitration="current-or-pip-semantic-jaccard-v1",
            local_key_kind="pip_semantic" if any(local_counts) else None,
            local_evidence_count=min(local_counts),
            public_key_kind="pip_semantic",
            public_evidence_count=min(public_counts),
            shrinkage_alpha=INTERACTION_ALPHA,
        )
        diagnostics[resource] = {
            "carrier": True,
            "clauses": clause_diagnostics,
        }
    return predictions, diagnostics


def _candidate_clause_evidence(
    kb: _InteractionPosetKB | _EpisodicSubsetKB,
    row: Row | _InteractionQueryClause,
    public_by_bin: Mapping[str, Sequence[Row]],
    public_global: Sequence[Row],
) -> _WeightedClauseEvidence:
    """Select weighted clause values for one non-trie candidate."""

    if isinstance(kb, _InteractionPosetKB):
        match = kb.query(row)
        observations = match.observations
        raw_weights = tuple(1.0 for _ in observations)
        key_kind = "exact_clause" if match.exact else "interaction_poset"
        exact = match.exact
    else:
        match = kb.query(row)
        observations = tuple(item.observation for item in match.contributions)
        raw_weights = tuple(item.weight for item in match.contributions)
        key_kind = "exact_clause" if match.exact else (
            "subset_kernel"
            if kb.order is None
            else f"subset_kernel_k{kb.order}"
        )
        exact = match.exact
    query_features = _interaction_feature_set(
        row.bin,
        row.argv,
        kb.stable_subcommands,
    ).features
    shared_counts = tuple(
        len(query_features & observation.feature_set.features)
        for observation in observations
    )
    local_values = tuple(observation.row.latency_ms for observation in observations)
    task_ids = frozenset(observation.task_id for observation in observations)
    if exact:
        return _WeightedClauseEvidence(
            values=local_values,
            weights=raw_weights,
            exact=True,
            nonexact_local=False,
            local_observation_count=len(observations),
            effective_sample_size=float(len(observations)),
            contributing_task_ids=task_ids,
            max_shared_feature_count=max(shared_counts, default=0),
            key_kind=key_kind,
            public_key_kind=None,
        )

    public = tuple(public_by_bin.get(row.bin, ()))
    public_key_kind = "bin"
    if not public:
        public = tuple(public_global)
        public_key_kind = "global"
    if not public:
        raise ValueError("non-trie candidate has no frozen public evidence")
    effective_n = _effective_sample_size(raw_weights)
    local_total = sum(raw_weights)
    local_weights = tuple(
        weight * effective_n / local_total for weight in raw_weights
    )
    public_weight = INTERACTION_ALPHA / len(public)
    return _WeightedClauseEvidence(
        values=(*local_values, *(item.latency_ms for item in public)),
        weights=(*local_weights, *(public_weight for _ in public)),
        exact=False,
        nonexact_local=bool(observations),
        local_observation_count=len(observations),
        effective_sample_size=effective_n,
        contributing_task_ids=task_ids,
        max_shared_feature_count=max(shared_counts, default=0),
        key_kind=key_kind if observations else "public_only",
        public_key_kind=public_key_kind,
    )


def _weighted_bucket_probabilities(
    values: Sequence[float],
    weights: Sequence[float],
) -> tuple[float, ...]:
    totals = [0.0] * CANONICAL_LATENCY_BUCKETS.bucket_count
    for value, weight in zip(values, weights, strict=True):
        totals[CANONICAL_LATENCY_BUCKETS.bucket_id(value)] += weight
    total = sum(totals)
    if total <= 0.0:
        raise ValueError("weighted evidence has no positive mass")
    return tuple(value / total for value in totals)


def _weighted_stratified_draws(
    values: Sequence[float],
    weights: Sequence[float],
    seed_material: str,
) -> tuple[float, ...]:
    """Canonical deterministic draws from a weighted empirical CDF."""

    ordered = sorted(zip(values, weights, strict=True))
    cumulative: list[float] = []
    total = 0.0
    for _, weight in ordered:
        if weight < 0.0:
            raise ValueError("empirical weight must be non-negative")
        total += weight
        cumulative.append(total)
    if total <= 0.0:
        raise ValueError("weighted evidence has no positive mass")
    digest = hashlib.blake2s(seed_material.encode(), digest_size=4).digest()
    offset = int.from_bytes(digest, "big") % COMMAND_COMPOSITION_DRAWS
    return tuple(
        ordered[
            min(
                bisect_right(
                    cumulative,
                    ((draw + offset) % COMMAND_COMPOSITION_DRAWS)
                    * total
                    / COMMAND_COMPOSITION_DRAWS,
                ),
                len(ordered) - 1,
            )
        ][0]
        for draw in range(COMMAND_COMPOSITION_DRAWS)
    )


def _interaction_query_clauses(
    clauses: Sequence[Mapping[str, Any]],
) -> tuple[_InteractionQueryClause, ...]:
    return tuple(
        _InteractionQueryClause(
            bin=str(clause["bin"]),
            argv=tuple(str(value) for value in clause["argv"]),
            in_pipe=bool(clause.get("in_pipe")),
            in_subst=bool(clause.get("in_subst")),
            pipeline_position=int(clause.get("pipeline_position", -1)),
        )
        for clause in clauses
    )


def _predict_interaction_command(
    kb: _InteractionPosetKB | _EpisodicSubsetKB,
    row: CommandRow,
    parsed_clauses: Sequence[Mapping[str, Any]],
    *,
    parse_failed: bool,
    public_by_bin: Mapping[str, Sequence[Row]],
    public_global: Sequence[Row],
) -> tuple[ClauseLatencyBucketPrediction | None, str | None, dict[str, Any]]:
    if parse_failed:
        return None, "parse_failed", {"clauses": []}
    query_clauses = _interaction_query_clauses(parsed_clauses)
    if not query_clauses:
        return None, "no_executable_clause", {"clauses": []}
    stages = _command_stages(parsed_clauses)
    if stages is None:
        return None, "compound_composition_unavailable", {"clauses": []}
    evidence = tuple(
        _candidate_clause_evidence(kb, clause, public_by_bin, public_global)
        for clause in query_clauses
    )
    if len(evidence) == 1:
        probabilities = _weighted_bucket_probabilities(
            evidence[0].values,
            evidence[0].weights,
        )
    else:
        draws = tuple(
            _weighted_stratified_draws(
                item.values,
                item.weights,
                (
                    f"{row.command}\0latency_ms\0{index}\0"
                    f"{'repo' if item.exact else 'repo+public' if item.nonexact_local else 'public'}\0"
                    f"{'exact_clause' if item.exact else 'weighted_pool' if item.nonexact_local else item.public_key_kind}"
                ),
            )
            for index, item in enumerate(evidence)
        )
        composed = tuple(
            sum(max(draws[index][draw] for index in stage) for stage in stages)
            for draw in range(COMMAND_COMPOSITION_DRAWS)
        )
        probabilities = _weighted_bucket_probabilities(
            composed,
            (1.0,) * len(composed),
        )
    local_counts = [item.local_observation_count for item in evidence]
    public_counts = [
        len(item.values) - item.local_observation_count for item in evidence
    ]
    task_ids = frozenset().union(
        *(item.contributing_task_ids for item in evidence)
    )
    nonexact_local = any(item.nonexact_local for item in evidence)
    exact = all(item.exact for item in evidence)
    prediction = ClauseLatencyBucketPrediction(
        probability_by_bucket=probabilities,
        scope=(
            "repo"
            if exact
            else "repo+public"
            if any(local_counts)
            else "public"
        ),
        key_kind=evidence[0].key_kind if len(evidence) == 1 else "shell_execution_graph",
        evidence_count=min(len(item.values) for item in evidence),
        fallback_path=tuple(
            f"clause[{index}]:{item.key_kind}"
            for index, item in enumerate(evidence)
        ),
        canonicalizer_version=INTERACTION_FEATURE_VERSION,
        arbitration="exact-or-weighted-public-pooling-v1",
        local_key_kind=(
            None
            if not any(local_counts)
            else "interaction_poset"
            if isinstance(kb, _InteractionPosetKB)
            else "subset_kernel"
        ),
        local_evidence_count=min(local_counts),
        public_key_kind=(
            None
            if exact
            else "+".join(
                sorted(
                    {
                        item.public_key_kind
                        for item in evidence
                        if item.public_key_kind is not None
                    }
                )
            )
        ),
        public_evidence_count=min(public_counts),
        shrinkage_alpha=None if exact else INTERACTION_ALPHA,
    )
    diagnostics = {
        "carrier": nonexact_local,
        "all_clauses_exact": exact,
        "distinct_contributing_tasks": len(task_ids),
        "clauses": [
            {
                "exact": item.exact,
                "nonexact_local": item.nonexact_local,
                "local_observation_count": item.local_observation_count,
                "effective_sample_size": item.effective_sample_size,
                "distinct_contributing_tasks": len(item.contributing_task_ids),
                "max_shared_feature_count": item.max_shared_feature_count,
                "key_kind": item.key_kind,
                "public_key_kind": item.public_key_kind,
            }
            for item in evidence
        ],
    }
    return prediction, None, diagnostics


def _row_resource_value(row: Row, resource: str) -> float | None:
    value = getattr(row, resource)
    if value is None and row.latency_ms < SHORT_NULL_LIGHT_MAX_LATENCY_MS:
        return 0.0
    return value


def _poset_resource_evidence(
    kb: _InteractionPosetKB,
    row: _InteractionQueryClause,
    public_by_bin: Mapping[str, Sequence[Row]],
    public_global: Sequence[Row],
    resource: str,
) -> _WeightedClauseEvidence:
    match = kb.query(row)
    observations = match.observations
    local_values = tuple(
        value
        for observation in observations
        if (value := _row_resource_value(observation.row, resource)) is not None
    )
    if len(local_values) != len(observations):
        raise AssertionError("resource-specific poset contains unavailable evidence")
    task_ids = frozenset(observation.task_id for observation in observations)
    query_features = _interaction_feature_set(
        row.bin,
        row.argv,
        kb.stable_subcommands,
    ).features
    shared_counts = tuple(
        len(query_features & observation.feature_set.features)
        for observation in observations
    )
    if match.exact:
        return _WeightedClauseEvidence(
            values=local_values,
            weights=(1.0,) * len(local_values),
            exact=True,
            nonexact_local=False,
            local_observation_count=len(observations),
            effective_sample_size=float(len(observations)),
            contributing_task_ids=task_ids,
            max_shared_feature_count=max(shared_counts, default=0),
            key_kind="exact_clause",
            public_key_kind=None,
        )
    public = tuple(public_by_bin.get(row.bin, ()))
    public_key_kind = "bin"
    if not public:
        public = tuple(public_global)
        public_key_kind = "global"
    public_values = tuple(
        value
        for item in public
        if (value := _row_resource_value(item, resource)) is not None
    )
    if len(public_values) != len(public) or not public_values:
        raise AssertionError("resource public prior contains unavailable evidence")
    effective_n = float(len(observations))
    public_weight = INTERACTION_ALPHA / len(public_values)
    return _WeightedClauseEvidence(
        values=(*local_values, *public_values),
        weights=(
            *(1.0 for _ in local_values),
            *(public_weight for _ in public_values),
        ),
        exact=False,
        nonexact_local=bool(observations),
        local_observation_count=len(observations),
        effective_sample_size=effective_n,
        contributing_task_ids=task_ids,
        max_shared_feature_count=max(shared_counts, default=0),
        key_kind="interaction_poset" if observations else "public_only",
        public_key_kind=public_key_kind,
    )


def _predict_poset_resources(
    kbs: Mapping[str, _InteractionPosetKB],
    row: CommandRow,
    parsed_clauses: Sequence[Mapping[str, Any]],
    *,
    parse_failed: bool,
    public_by_resource_bin: Mapping[str, Mapping[str, Sequence[Row]]],
    public_by_resource: Mapping[str, Sequence[Row]],
) -> tuple[dict[str, ClauseHeavyLightPrediction | None], str | None, dict[str, Any]]:
    if parse_failed:
        return {}, "parse_failed", {}
    query_clauses = _interaction_query_clauses(parsed_clauses)
    if not query_clauses:
        return {}, "no_executable_clause", {}
    stages = _command_stages(parsed_clauses)
    if stages is None:
        return {}, "compound_composition_unavailable", {}
    predictions: dict[str, ClauseHeavyLightPrediction | None] = {}
    diagnostics: dict[str, Any] = {}
    for resource, threshold in CANONICAL_RESOURCE_HEAVY_THRESHOLDS.items():
        evidence = tuple(
            _poset_resource_evidence(
                kbs[resource],
                clause,
                public_by_resource_bin[resource],
                public_by_resource[resource],
                resource,
            )
            for clause in query_clauses
        )
        if len(evidence) == 1:
            total = sum(evidence[0].weights)
            probability_heavy = sum(
                weight
                for value, weight in zip(
                    evidence[0].values,
                    evidence[0].weights,
                    strict=True,
                )
                if value > threshold
            ) / total
        else:
            draws = tuple(
                _weighted_stratified_draws(
                    item.values,
                    item.weights,
                    (
                        f"{row.command}\0{resource}\0{index}\0"
                        f"{'repo' if item.exact else 'repo+public' if item.nonexact_local else 'public'}\0"
                        f"{'exact_clause' if item.exact else 'weighted_pool' if item.nonexact_local else item.public_key_kind}"
                    ),
                )
                for index, item in enumerate(evidence)
            )
            composed = tuple(
                (
                    sum(
                        sum(draws[index][draw] for index in stage)
                        for stage in stages
                    )
                    if resource == "disk_read_write_bytes_total"
                    else max(
                        sum(draws[index][draw] for index in stage)
                        for stage in stages
                    )
                )
                for draw in range(COMMAND_COMPOSITION_DRAWS)
            )
            probability_heavy = sum(value > threshold for value in composed) / len(
                composed
            )
        local_counts = [item.local_observation_count for item in evidence]
        public_counts = [
            len(item.values) - item.local_observation_count for item in evidence
        ]
        all_exact = all(item.exact for item in evidence)
        carrier = any(item.nonexact_local for item in evidence)
        predictions[resource] = ClauseHeavyLightPrediction(
            resource=resource,
            threshold=threshold,
            probability_heavy=probability_heavy,
            heavy_decision_threshold=DEFAULT_HEAVY_DECISION_THRESHOLD,
            label=(
                "heavy"
                if probability_heavy > DEFAULT_HEAVY_DECISION_THRESHOLD
                else "light"
            ),
            scope=(
                "repo"
                if all_exact
                else "repo+public"
                if any(local_counts)
                else "public"
            ),
            key_kind=(
                evidence[0].key_kind
                if len(evidence) == 1
                else "shell_execution_graph"
            ),
            evidence_count=min(len(item.values) for item in evidence),
            fallback_path=tuple(
                f"clause[{index}]:{item.key_kind}"
                for index, item in enumerate(evidence)
            ),
            canonicalizer_version=INTERACTION_FEATURE_VERSION,
            arbitration="exact-or-weighted-public-pooling-v1",
            local_key_kind="interaction_poset" if any(local_counts) else None,
            local_evidence_count=min(local_counts),
            public_key_kind=(
                None
                if all_exact
                else "+".join(
                    sorted(
                        {
                            item.public_key_kind
                            for item in evidence
                            if item.public_key_kind is not None
                        }
                    )
                )
            ),
            public_evidence_count=min(public_counts),
            shrinkage_alpha=None if all_exact else INTERACTION_ALPHA,
        )
        diagnostics[resource] = {
            "carrier": carrier,
            "all_clauses_exact": all_exact,
            "distinct_contributing_tasks": len(
                frozenset().union(
                    *(item.contributing_task_ids for item in evidence)
                )
            ),
            "minimum_effective_sample_size": min(
                item.effective_sample_size for item in evidence
            ),
        }
    return predictions, None, diagnostics


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


def _quartile_metrics(
    rows_by_arm: Mapping[str, Sequence[ScoredRow]],
    task_index: Mapping[str, int],
    task_count: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for quartile in range(4):
        selected = {
            arm: [
                row
                for row in rows
                if min(3, task_index[row.task_id] * 4 // task_count) == quartile
            ]
            for arm, rows in rows_by_arm.items()
        }
        if not selected["current"]:
            continue
        current = _telemetry_metrics(selected["current"])
        result[f"q{quartile + 1}"] = {
            arm: (
                current
                if arm == "current"
                else _telemetry_metrics(
                    rows,
                    current_accuracy=current["three_class_accuracy"],
                )
            )
            for arm, rows in selected.items()
        }
    return result


def _prediction_changes(
    current: Sequence[ScoredRow],
    candidate: Sequence[ScoredRow],
    diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if [
        (row.sample_id, row.label_bucket) for row in current
    ] != [
        (row.sample_id, row.label_bucket) for row in candidate
    ] or len(candidate) != len(diagnostics):
        raise AssertionError("prediction-change rows are not aligned")
    counts = Counter()
    carrier_counts = Counter()
    for current_row, candidate_row, diagnostic in zip(
        current,
        candidate,
        diagnostics,
        strict=True,
    ):
        current_prediction = _argmax_bucket(current_row)
        candidate_prediction = _argmax_bucket(candidate_row)
        if current_prediction == candidate_prediction:
            continue
        outcome = _prediction_change_outcome(current_row, candidate_row)
        assert outcome is not None
        counts["changed"] += 1
        counts[outcome] += 1
        if diagnostic["carrier"]:
            carrier_counts["changed"] += 1
            carrier_counts[outcome] += 1
    return {
        "all_changed_commands": {
            key: counts[key] for key in ("changed", "helpful", "harmful", "neutral")
        },
        "nonexact_carrier_changed_commands": {
            **{
                key: carrier_counts[key]
                for key in ("changed", "helpful", "harmful", "neutral")
            },
            "net_helpful_minus_harmful": (
                carrier_counts["helpful"] - carrier_counts["harmful"]
            ),
        },
    }


def _prediction_change_outcome(
    reference: ScoredRow,
    candidate: ScoredRow,
) -> str | None:
    reference_prediction = _argmax_bucket(reference)
    candidate_prediction = _argmax_bucket(candidate)
    if reference_prediction == candidate_prediction:
        return None
    if candidate_prediction == candidate.label_bucket:
        return "helpful"
    if reference_prediction == reference.label_bucket:
        return "harmful"
    return "neutral"


def _carrier_net_spread(
    reference: Sequence[ScoredRow],
    candidate: Sequence[ScoredRow],
    diagnostics: Sequence[Mapping[str, Any]],
    reference_diagnostics: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Count independent task/signature groups with positive carrier net gain."""

    task_net: Counter[str] = Counter()
    signature_net: Counter[str] = Counter()
    unattributed_multi_clause_commands = 0
    if reference_diagnostics is not None and len(reference_diagnostics) != len(
        diagnostics
    ):
        raise AssertionError("carrier diagnostics are not aligned")
    for index, (reference_row, candidate_row, diagnostic) in enumerate(
        zip(reference, candidate, diagnostics, strict=True)
    ):
        outcome = _prediction_change_outcome(reference_row, candidate_row)
        if outcome is None or not diagnostic["carrier"]:
            continue
        delta = 1 if outcome == "helpful" else -1 if outcome == "harmful" else 0
        task_net[candidate_row.task_id] += delta
        reference_clauses = (
            None
            if reference_diagnostics is None
            else reference_diagnostics[index]["clauses"]
        )
        differing_signatures = [
            json.dumps(clause["signature"], sort_keys=True, separators=(",", ":"))
            for clause_index, clause in enumerate(diagnostic["clauses"])
            if clause["pip"]
            and not clause["exact"]
            and not clause["key_kind"].startswith("current_")
            and (
                reference_clauses is None
                or clause["contributors"]
                != reference_clauses[clause_index]["contributors"]
            )
        ]
        if len(differing_signatures) == 1:
            signature_net[differing_signatures[0]] += delta
        elif len(differing_signatures) > 1:
            unattributed_multi_clause_commands += 1
    return {
        "positive_net_tasks": sorted(
            task for task, net in task_net.items() if net > 0
        ),
        "positive_net_task_count": sum(net > 0 for net in task_net.values()),
        "positive_net_signatures": sorted(
            signature for signature, net in signature_net.items() if net > 0
        ),
        "positive_net_signature_count": sum(
            net > 0 for net in signature_net.values()
        ),
        "net_by_task": dict(sorted(task_net.items())),
        "net_by_signature": dict(sorted(signature_net.items())),
        "unattributed_multi_clause_commands": unattributed_multi_clause_commands,
    }


def _candidate_signal_diagnostics(
    diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    clauses = [
        clause
        for command in diagnostics
        for clause in command["clauses"]
        if not clause["exact"]
    ]
    shared = Counter(int(clause["max_shared_feature_count"]) for clause in clauses)
    contributing_tasks = Counter(
        int(command["distinct_contributing_tasks"]) for command in diagnostics
    )
    effective = [float(clause["effective_sample_size"]) for clause in clauses]
    return {
        "exact_miss_clause_queries": len(clauses),
        "max_shared_feature_count_distribution": {
            str(value): count for value, count in sorted(shared.items())
        },
        "queries_with_m_at_least": {
            str(threshold): sum(
                count for value, count in shared.items() if value >= threshold
            )
            for threshold in (1, 2, 3)
        },
        "distinct_prior_tasks_per_command_distribution": {
            str(value): count for value, count in sorted(contributing_tasks.items())
        },
        "effective_sample_size_on_exact_misses": {
            "p50": _percentile(effective, 0.5) if effective else None,
            "p95": _percentile(effective, 0.95) if effective else None,
            "max": max(effective, default=None),
        },
        "nonexact_carrier_commands": sum(
            bool(command["carrier"]) for command in diagnostics
        ),
    }


def evaluate_interaction_commands(
    public_rows: Sequence[Row],
    task_ids: Sequence[str],
    clause_rows: Sequence[Row],
    command_rows: Sequence[CommandRow],
    provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare current and both non-trie KBs on one causal task stream."""

    if len(set(task_ids)) != len(task_ids):
        raise ValueError("target task order contains duplicates")
    if len({repo_of(task_id) for task_id in task_ids}) != 1:
        raise ValueError("interaction evaluation requires one target repository")
    target_repo = repo_of(task_ids[0])
    if any(row.repo == target_repo for row in public_rows):
        raise ValueError("public evidence contains the target repository")
    task_index = {task_id: index for index, task_id in enumerate(task_ids)}
    if any(
        row.task_id not in task_index
        or row.manifest_index != task_index[row.task_id]
        for row in (*clause_rows, *command_rows)
    ):
        raise ValueError("target rows differ from results.jsonl task order")

    public = tuple(
        row
        for row in public_rows
        if row.structure_known and row.pipeline_position <= 0
    )
    if not public:
        raise ValueError("no frozen public evidence remains after eligibility filters")
    public_by_bin: dict[str, list[Row]] = defaultdict(list)
    for row in public:
        public_by_bin[row.bin].append(row)
    stable_subcommands = _fit_stable_subcommands(
        row.observation(0.0, 1.0) for row in public
    )
    current = ClauseResourceKB.fit_public(
        row.observation(0.0, 1.0) for row in public
    )
    candidates: dict[str, _InteractionPosetKB | _EpisodicSubsetKB] = {
        "interaction_poset": _InteractionPosetKB(stable_subcommands),
        "subset_kernel": _EpisodicSubsetKB(stable_subcommands),
        **{
            f"subset_k{order}": _EpisodicSubsetKB(
                stable_subcommands,
                order=order,
            )
            for order in (1, 2, 3)
        },
    }
    clauses_by_task: dict[str, list[Row]] = defaultdict(list)
    commands_by_task: dict[str, list[CommandRow]] = defaultdict(list)
    for row in clause_rows:
        clauses_by_task[row.task_id].append(row)
    for row in command_rows:
        commands_by_task[row.task_id].append(row)

    rows_by_arm: dict[str, list[ScoredRow]] = {
        arm: [] for arm in ("current", *candidates)
    }
    diagnostics_by_arm: dict[str, list[dict[str, Any]]] = {
        arm: [] for arm in candidates
    }
    timings_ns: dict[str, list[int]] = defaultdict(list)
    sidecar: list[dict[str, Any]] = []
    started_tracing = not tracemalloc.is_tracing()
    if started_tracing:
        tracemalloc.start()
    try:
        for ordinal, task_id in enumerate(task_ids):
            query_ts = float(ordinal * 2 + 3)
            for row in commands_by_task[task_id]:
                parsed = parse_command_clauses(row.command)
                parsed_clauses = parsed["clauses"]
                parse_failed = bool(parsed["parse_failed"])
                started = time.perf_counter_ns()
                current_result = current.predict_command_latency_bucket_from_clauses(
                    row.repo,
                    parsed_clauses,
                    query_ts,
                    CANONICAL_LATENCY_BUCKETS,
                    command=row.command,
                    parse_failed=parse_failed,
                )
                timings_ns["current"].append(time.perf_counter_ns() - started)
                rows_by_arm["current"].append(
                    _command_scored_row(
                        row,
                        current_result.prediction,
                        current_result.unavailable_reason,
                    )
                )
                arm_sidecar: dict[str, Any] = {
                    "current": {
                        "probability_by_bucket": (
                            None
                            if current_result.prediction is None
                            else current_result.prediction.probability_by_bucket
                        ),
                        "prediction": (
                            None
                            if current_result.prediction is None
                            else _argmax_probabilities(
                                current_result.prediction.probability_by_bucket
                            )
                        ),
                        "unavailable_reason": current_result.unavailable_reason,
                    }
                }
                for arm, kb in candidates.items():
                    started = time.perf_counter_ns()
                    prediction, unavailable, diagnostic = _predict_interaction_command(
                        kb,
                        row,
                        parsed_clauses,
                        parse_failed=parse_failed,
                        public_by_bin=public_by_bin,
                        public_global=public,
                    )
                    timings_ns[arm].append(time.perf_counter_ns() - started)
                    scored = _command_scored_row(row, prediction, unavailable)
                    rows_by_arm[arm].append(scored)
                    diagnostics_by_arm[arm].append(diagnostic)
                    arm_sidecar[arm] = {
                        "probability_by_bucket": (
                            None if prediction is None else prediction.probability_by_bucket
                        ),
                        "prediction": (
                            None
                            if prediction is None
                            else _argmax_probabilities(prediction.probability_by_bucket)
                        ),
                        "unavailable_reason": unavailable,
                        **diagnostic,
                    }
                sidecar.append(
                    {
                        "sample_id": (
                            f"{row.task_id}:{row.manifest_index}:call:{row.call_index}"
                        ),
                        "task_id": row.task_id,
                        "task_ordinal": ordinal,
                        "task_quartile": min(3, ordinal * 4 // len(task_ids)) + 1,
                        "call_id": row.call_id,
                        "command": row.command,
                        "clause_count": len(row.clauses),
                        "latency_label": CANONICAL_LATENCY_BUCKETS.bucket_id(
                            row.duration_ms
                        ),
                        "arms": arm_sidecar,
                    }
                )
            settle_ts = query_ts + 0.5
            settled = clauses_by_task[task_id]
            for row in settled:
                current.observe_completed_clause(row.observation(query_ts, settle_ts))
            for kb in candidates.values():
                kb.observe(settled)
        peak_traced_bytes = tracemalloc.get_traced_memory()[1]
    finally:
        if started_tracing:
            tracemalloc.stop()

    identity = [
        (row.sample_id, row.label_bucket, row.probability_by_bucket is not None)
        for row in rows_by_arm["current"]
    ]
    for arm, rows in rows_by_arm.items():
        if identity != [
            (row.sample_id, row.label_bucket, row.probability_by_bucket is not None)
            for row in rows
        ]:
            raise AssertionError(f"{arm} differs in command rows, labels, or availability")
    current_metrics = _telemetry_metrics(rows_by_arm["current"])
    current_accuracy = current_metrics["three_class_accuracy"]
    if current_accuracy is None:
        raise ValueError("interaction evaluation requires complete command predictions")
    metrics = {
        arm: (
            current_metrics
            if arm == "current"
            else _telemetry_metrics(rows, current_accuracy=current_accuracy)
        )
        for arm, rows in rows_by_arm.items()
    }
    changes = {
        arm: _prediction_changes(
            rows_by_arm["current"],
            rows_by_arm[arm],
            diagnostics_by_arm[arm],
        )
        for arm in candidates
    }
    hard_predictions = {
        arm: [_argmax_bucket(row) for row in rows]
        for arm, rows in rows_by_arm.items()
    }
    labels = [row.label_bucket for row in rows_by_arm["current"]]
    oracle_correct = sum(
        label
        in {
            hard_predictions["current"][index],
            hard_predictions["interaction_poset"][index],
            hard_predictions["subset_kernel"][index],
        }
        for index, label in enumerate(labels)
    )
    oracle_accuracy = oracle_correct / len(labels)
    majority_accuracy = current_metrics["majority_class_accuracy"]
    stage0_pass = (
        any(
            changes[arm]["nonexact_carrier_changed_commands"]["changed"] > 0
            for arm in ("interaction_poset", "subset_kernel")
        )
        and oracle_accuracy > current_accuracy
        and oracle_accuracy > majority_accuracy
    )
    stage1_go = {
        arm: (
            stage0_pass
            and metrics[arm]["three_class_accuracy"] > current_accuracy
            and metrics[arm]["three_class_accuracy"] > majority_accuracy
            and changes[arm]["nonexact_carrier_changed_commands"][
                "net_helpful_minus_harmful"
            ]
            > 0
        )
        for arm in ("interaction_poset", "subset_kernel")
    }
    lookup_cost = {
        arm: {
            "calls": len(values),
            "p50_ms": _percentile(values, 0.5) / 1_000_000,
            "p95_ms": _percentile(values, 0.95) / 1_000_000,
        }
        for arm, values in timings_ns.items()
    }
    uncertainty = {
        f"{arm}_minus_current": _paired_task_cluster_bootstrap(
            rows_by_arm[arm],
            rows_by_arm["current"],
            left_name=arm,
            right_name="current",
        )
        for arm in ("interaction_poset", "subset_kernel")
    }
    uncertainty["interaction_poset_minus_subset_kernel"] = (
        _paired_task_cluster_bootstrap(
            rows_by_arm["interaction_poset"],
            rows_by_arm["subset_kernel"],
            left_name="interaction_poset",
            right_name="subset_kernel",
        )
    )
    result = {
        "status": (
            "development_exposed_latency_go"
            if any(stage1_go.values())
            else "development_exposed_latency_no_go"
            if stage0_pass
            else "development_exposed_stage0_no_signal"
        ),
        "claim_bearing": False,
        "objective": "command_latency_interaction_kb_comparison",
        "inputs": dict(provenance),
        "protocol": {
            "evaluation_unit": "eligible_exec_command",
            "task_order": "results.jsonl successful final attempts",
            "causal_update": "all task commands predict before task clauses settle",
            "public_layer": "identical frozen cross-repository bin/global values",
            "current": "raw exact/prefix/bin hard first-hit backoff",
            "interaction_poset": "maximal query-induced feature intersections",
            "subset_kernel": "length-normalized all-nonempty-subset overlap",
            "subset_kernel_ablations": [1, 2, 3],
            "exact_shortcut": "repo-local exact hash; no public pooling",
            "pooling_alpha": INTERACTION_ALPHA,
            "compound_composition": "weighted-empirical-shell-graph-v1",
            "weighted_composition_draws": COMMAND_COMPOSITION_DRAWS,
            "bootstrap": {
                "cluster_unit": "task",
                "seed": 0,
                "draws": 2000,
            },
        },
        "counts": {
            "tasks": len(task_ids),
            "commands": len(command_rows),
            "target_online_clause_observations": len(clause_rows),
            "public_online_clause_observations": len(public),
            "stored_target_observations_by_candidate": {
                arm: kb._next_observation_id for arm, kb in candidates.items()
            },
        },
        "row_identity": {
            "identical_command_ids_labels_and_availability": True,
            "unique_sample_ids": len({row[0] for row in identity}) == len(identity),
        },
        "latency": {
            "bucket_edges_ms": list(CANONICAL_LATENCY_BUCKETS.edges_ms),
            "direction": "higher accuracy is better",
            "majority": {
                "class": current_metrics["majority_class"],
                "class_id": current_metrics["majority_class_id"],
                "accuracy": majority_accuracy,
            },
            "arms": metrics,
            "by_task_order_quartile": _quartile_metrics(
                rows_by_arm,
                task_index,
                len(task_ids),
            ),
            "prediction_changes": changes,
            "uncertainty": uncertainty,
            "three_way_hindsight_oracle": {
                "arms": ["current", "interaction_poset", "subset_kernel"],
                "accuracy": oracle_accuracy,
                "deployable": False,
            },
        },
        "signal_diagnostics": {
            arm: _candidate_signal_diagnostics(diagnostics_by_arm[arm])
            for arm in candidates
        },
        "cost": {
            "lookup": lookup_cost,
            "peak_traced_evaluator_memory_bytes": peak_traced_bytes,
            "memory_measurement": "Python tracemalloc peak over causal evaluation",
        },
        "gates": {
            "stage0": {
                "pass": stage0_pass,
                "requires_nonexact_prediction_change": True,
                "requires_three_way_oracle_above_current_and_majority": True,
            },
            "stage1_resource_evaluation": {
                arm: {
                    "go": go,
                    "requires_accuracy_above_current_and_majority": True,
                    "requires_positive_nonexact_carrier_net_gain": True,
                }
                for arm, go in stage1_go.items()
            },
        },
    }
    return result, sidecar


def _settled_pip_observations(
    command: CommandRow,
    states: Sequence[PipQueryState | None],
) -> tuple[tuple[Row, PipInstallSignature, PipQueryState], ...]:
    parsed = parse_command_clauses(command.command)
    parsed_signatures = tuple(
        parse_pip_install(clause.get("argv", ()))
        for clause in parsed.get("clauses", ())
    )
    if len(parsed_signatures) != len(states):
        raise ValueError(f"{command.call_id}: parsed clauses and state differ")
    unused = set(range(len(parsed_signatures)))
    observations: list[tuple[Row, PipInstallSignature, PipQueryState]] = []
    for row in command.clauses:
        if row.pipeline_position > 0 or (signature := parse_pip_install(row.argv)) is None:
            continue
        index = next(
            (
                candidate
                for candidate in sorted(unused)
                if parsed_signatures[candidate] == signature
            ),
            None,
        )
        if index is None or states[index] is None:
            raise ValueError(f"{command.call_id}: pip telemetry cannot be aligned")
        unused.remove(index)
        observations.append((row, signature, states[index]))
    return tuple(observations)


def _pip_execution_mode(tool_result: str) -> str:
    lower = tool_result.lower()
    lines = [line for line in tool_result.splitlines() if line]
    if not lines or not lines[-1].startswith("Exit code: "):
        return "unknown_output"
    try:
        exit_code = int(lines[-1].removeprefix("Exit code: "))
    except ValueError:
        return "unknown_output"
    if exit_code != 0:
        if "no module named pip" in lower or re.search(
            r"(?:^|\n).*\bpip3?\b.*(?:not found|no such file)", lower
        ):
            return "failure_missing_pip"
        return "failure_other"
    if "downloading " in lower or "building wheel" in lower:
        return "success_download_or_build"
    if "using cached" in lower or "requirement already satisfied" in lower:
        return "success_cache_only"
    return "success_other"


def _observed_composition_summary(
    commands: Sequence[CommandRow],
    current_rows: Sequence[ScoredRow],
) -> dict[str, Any]:
    current = {row.sample_id: row for row in current_rows}
    groups: dict[str, list[tuple[CommandRow, float, int]]] = defaultdict(list)
    for row in commands:
        structures = [
            {
                "in_pipe": clause.in_pipe,
                "in_subst": clause.in_subst,
                "pipeline_position": clause.pipeline_position,
            }
            for clause in row.clauses
        ]
        stages = _command_stages(structures)
        if stages is None:
            continue
        composed_ms = sum(
            max(row.clauses[index].latency_ms for index in stage)
            for stage in stages
        )
        pip = any(parse_pip_install(clause.argv) is not None for clause in row.clauses)
        item = (row, composed_ms, len(stages))
        groups["all"].append(item)
        if pip:
            groups["pip"].append(item)
            groups["pip_compound" if len(row.clauses) > 1 else "pip_single"].append(item)

    def summarize(items: Sequence[tuple[CommandRow, float, int]]) -> dict[str, Any]:
        if not items:
            return {"commands": 0}
        oracle_correct = 0
        current_correct = 0
        residuals = []
        for row, composed_ms, _stage_count in items:
            sample_id = f"{row.task_id}:{row.manifest_index}:call:{row.call_index}"
            label = CANONICAL_LATENCY_BUCKETS.bucket_id(row.duration_ms)
            oracle_correct += CANONICAL_LATENCY_BUCKETS.bucket_id(composed_ms) == label
            current_correct += _argmax_bucket(current[sample_id]) == label
            residuals.append(row.duration_ms - composed_ms)
        return {
            "commands": len(items),
            "oracle_accuracy": oracle_correct / len(items),
            "current_accuracy": current_correct / len(items),
            "command_minus_observed_composition_ms": {
                "p50": _percentile(residuals, 0.5),
                "p90": _percentile(residuals, 0.9),
            },
        }

    return {
        "analysis_only": True,
        "definition": "sum of per-stage maximum observed clause latency",
        "groups": {
            name: summarize(groups[name])
            for name in ("all", "pip", "pip_single", "pip_compound")
        },
    }


def evaluate_pip_semantics(
    public_rows: Sequence[Row],
    task_ids: Sequence[str],
    clause_rows: Sequence[Row],
    command_rows: Sequence[CommandRow],
    events_by_task: Mapping[str, Sequence[PipExecEvent]],
    provenance: Mapping[str, Any],
    *,
    expected_pip_baseline: tuple[int, int] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Audit canonical-exact versus Jaccard pip work signatures."""

    if len(set(task_ids)) != len(task_ids):
        raise ValueError("target task order contains duplicates")
    if len({repo_of(task_id) for task_id in task_ids}) != 1:
        raise ValueError("pip evaluation requires one target repository")
    target_repo = repo_of(task_ids[0])
    if any(row.repo == target_repo for row in public_rows):
        raise ValueError("public evidence contains the target repository")
    task_index = {task_id: index for index, task_id in enumerate(task_ids)}
    if list(events_by_task) != list(task_ids) or any(
        row.task_id not in task_index
        or row.manifest_index != task_index[row.task_id]
        for row in (*clause_rows, *command_rows)
    ):
        raise ValueError("target rows or raw events differ from accepted task order")

    public = tuple(
        row
        for row in public_rows
        if row.structure_known and row.pipeline_position <= 0
    )
    if not public:
        raise ValueError("no frozen public evidence remains after eligibility filters")
    current = ClauseResourceKB.fit_public(
        row.observation(0.0, 1.0) for row in public
    )
    semantic = _PipSemanticKB(public)
    clauses_by_task: dict[str, list[Row]] = defaultdict(list)
    commands_by_task: dict[str, list[CommandRow]] = defaultdict(list)
    for row in clause_rows:
        clauses_by_task[row.task_id].append(row)
    for row in command_rows:
        commands_by_task[row.task_id].append(row)

    arms = (
        "current",
        "pip_canonical_exact",
        "pip_semantic",
        "pip_semantic_state",
    )
    scored: dict[str, list[ScoredRow]] = {arm: [] for arm in arms}
    diagnostics: dict[str, list[dict[str, Any]]] = {
        arm: [] for arm in arms[1:]
    }
    sidecar: list[dict[str, Any]] = []
    modes: dict[str, Counter[str]] = defaultdict(Counter)
    for ordinal, task_id in enumerate(task_ids):
        query_ts = float(ordinal * 2 + 3)
        events = tuple(events_by_task[task_id])
        contexts = _pip_contexts_by_call(events)
        event_by_call = {event.call_id: event for event in events}
        if len(event_by_call) != len(events):
            raise ValueError(f"{task_id}: duplicate raw exec call id")
        settled_semantic: list[
            tuple[Row, PipInstallSignature, PipQueryState]
        ] = []
        for row in commands_by_task[task_id]:
            event = event_by_call.get(row.call_id)
            if event is None or event.command != row.command:
                raise ValueError(f"{row.call_id}: eligible command lacks raw output")
            parsed = parse_command_clauses(row.command)
            current_result = current.predict_command_latency_bucket_from_clauses(
                row.repo,
                parsed["clauses"],
                query_ts,
                CANONICAL_LATENCY_BUCKETS,
                command=row.command,
                parse_failed=bool(parsed["parse_failed"]),
            )
            if current_result.prediction is None:
                raise ValueError(f"{row.call_id}: Current prediction unavailable")
            scored["current"].append(
                _command_scored_row(row, current_result.prediction, None)
            )
            states = contexts[row.call_id]
            arm_values: dict[str, Any] = {
                "current": {
                    "prediction": _argmax_probabilities(
                        current_result.prediction.probability_by_bucket
                    ),
                    "probability_by_bucket": current_result.prediction.probability_by_bucket,
                }
            }
            for arm, use_state, canonical_exact in (
                ("pip_canonical_exact", False, True),
                ("pip_semantic", False, False),
                ("pip_semantic_state", True, False),
            ):
                prediction, diagnostic = _predict_pip_latency_command(
                    semantic,
                    current,
                    row,
                    parsed,
                    states,
                    current_result.prediction,
                    use_state=use_state,
                    canonical_exact=canonical_exact,
                )
                scored[arm].append(_command_scored_row(row, prediction, None))
                diagnostics[arm].append(diagnostic)
                arm_values[arm] = {
                    "prediction": _argmax_probabilities(
                        prediction.probability_by_bucket
                    ),
                    "probability_by_bucket": prediction.probability_by_bucket,
                    **diagnostic,
                }
            pip_command = any(
                parse_pip_install(clause.get("argv", ())) is not None
                for clause in parsed["clauses"]
            )
            mode = _pip_execution_mode(event.tool_result) if pip_command else None
            if mode is not None:
                label = CANONICAL_LATENCY_BUCKETS.bucket_id(row.duration_ms)
                modes[mode]["commands"] += 1
                modes[mode][f"latency_bucket_{label}"] += 1
                for arm in arms:
                    modes[mode][f"{arm}_correct"] += (
                        arm_values[arm]["prediction"] == label
                    )
                for resource in CANONICAL_RESOURCE_HEAVY_THRESHOLDS:
                    resource_label, _source = command_resource_label(row, resource)
                    modes[mode][
                        f"{resource}_{'unavailable' if resource_label is None else 'heavy' if resource_label else 'light'}"
                    ] += 1
            sample_id = f"{row.task_id}:{row.manifest_index}:call:{row.call_index}"
            sidecar.append(
                {
                    "sample_id": sample_id,
                    "task_id": row.task_id,
                    "task_ordinal": ordinal,
                    "call_id": row.call_id,
                    "command": row.command,
                    "pip_command": pip_command,
                    "pip_execution_mode": mode,
                    "latency_label": CANONICAL_LATENCY_BUCKETS.bucket_id(
                        row.duration_ms
                    ),
                    "arms": arm_values,
                }
            )
            settled_semantic.extend(
                _settled_pip_observations(row, states)
            )
        settle_ts = query_ts + 0.5
        for row in clauses_by_task[task_id]:
            current.observe_completed_clause(row.observation(query_ts, settle_ts))
        for row, signature, state in settled_semantic:
            semantic.observe(task_id, row, signature, state)

    identity = [
        (row.sample_id, row.label_bucket, row.probability_by_bucket is not None)
        for row in scored["current"]
    ]
    for arm in arms[1:]:
        if identity != [
            (row.sample_id, row.label_bucket, row.probability_by_bucket is not None)
            for row in scored[arm]
        ]:
            raise AssertionError(f"{arm} differs in rows, labels, or availability")
    current_metrics = _telemetry_metrics(scored["current"])
    current_accuracy = current_metrics["three_class_accuracy"]
    assert current_accuracy is not None
    metrics = {
        arm: (
            current_metrics
            if arm == "current"
            else _telemetry_metrics(scored[arm], current_accuracy=current_accuracy)
        )
        for arm in arms
    }
    pip_indices = [
        index for index, row in enumerate(sidecar) if row["pip_command"]
    ]
    nonpip_indices = [
        index for index, row in enumerate(sidecar) if not row["pip_command"]
    ]
    if not pip_indices:
        raise ValueError("target run contains no eligible pip command")
    pip_rows = {
        arm: [scored[arm][index] for index in pip_indices]
        for arm in arms
    }
    pip_current_metrics = _telemetry_metrics(pip_rows["current"])
    pip_metrics = {
        arm: (
            pip_current_metrics
            if arm == "current"
            else _telemetry_metrics(
                pip_rows[arm],
                current_accuracy=pip_current_metrics["three_class_accuracy"],
            )
        )
        for arm in arms
    }
    nonpip_identical = all(
        scored[arm][index].probability_by_bucket
        == scored["current"][index].probability_by_bucket
        for arm in arms[1:]
        for index in nonpip_indices
    )
    changes = {
        arm: _prediction_changes(
            scored["current"], scored[arm], diagnostics[arm]
        )
        for arm in arms[1:]
    }
    pairwise_changes = {
        "jaccard_minus_canonical_exact": _prediction_changes(
            scored["pip_canonical_exact"],
            scored["pip_semantic"],
            diagnostics["pip_semantic"],
        ),
        "canonical_exact_minus_current": changes["pip_canonical_exact"],
    }
    carrier_spread = {
        "jaccard_minus_canonical_exact": _carrier_net_spread(
            scored["pip_canonical_exact"],
            scored["pip_semantic"],
            diagnostics["pip_semantic"],
            diagnostics["pip_canonical_exact"],
        ),
        "canonical_exact_minus_current": _carrier_net_spread(
            scored["current"],
            scored["pip_canonical_exact"],
            diagnostics["pip_canonical_exact"],
        ),
    }
    current_pip_correct = sum(
        _exact_bucket_correct(row) for row in pip_rows["current"]
    )
    baseline_reconciled = expected_pip_baseline is None or (
        len(pip_indices), current_pip_correct
    ) == expected_pip_baseline
    jaccard_change = pairwise_changes["jaccard_minus_canonical_exact"][
        "nonexact_carrier_changed_commands"
    ]
    canonical_change = pairwise_changes["canonical_exact_minus_current"][
        "nonexact_carrier_changed_commands"
    ]
    jaccard_go = (
        baseline_reconciled
        and nonpip_identical
        and metrics["pip_semantic"]["three_class_accuracy"]
        > metrics["pip_canonical_exact"]["three_class_accuracy"]
        and pip_metrics["pip_semantic"]["three_class_accuracy"]
        > pip_metrics["pip_canonical_exact"]["three_class_accuracy"]
        and jaccard_change["helpful"] > jaccard_change["harmful"]
        and carrier_spread["jaccard_minus_canonical_exact"][
            "positive_net_task_count"
        ]
        >= 2
        and carrier_spread["jaccard_minus_canonical_exact"][
            "positive_net_signature_count"
        ]
        >= 2
    )
    canonical_go = (
        baseline_reconciled
        and nonpip_identical
        and metrics["pip_canonical_exact"]["three_class_accuracy"]
        > current_accuracy
        and pip_metrics["pip_canonical_exact"]["three_class_accuracy"]
        > pip_current_metrics["three_class_accuracy"]
        and canonical_change["helpful"] > canonical_change["harmful"]
        and carrier_spread["canonical_exact_minus_current"][
            "positive_net_task_count"
        ]
        >= 2
        and carrier_spread["canonical_exact_minus_current"][
            "positive_net_signature_count"
        ]
        >= 2
    )
    selection = (
        "jaccard_semantic"
        if jaccard_go
        else "canonical_exact"
        if canonical_go
        else "stop_semantic_method"
    )
    for index, row in enumerate(sidecar):
        if not row["pip_command"]:
            continue
        row["carrier_audit"] = {
            "normalized_signatures": [
                clause["signature"]
                for clause in diagnostics["pip_semantic"][index]["clauses"]
                if clause["pip"]
            ],
            "canonical_exact_nonexact_carrier": diagnostics[
                "pip_canonical_exact"
            ][index]["carrier"],
            "jaccard_nonexact_carrier": diagnostics["pip_semantic"][index][
                "carrier"
            ],
            "canonical_exact_vs_current": (
                _prediction_change_outcome(
                    scored["current"][index], scored["pip_canonical_exact"][index]
                )
                or "unchanged"
            ),
            "jaccard_vs_canonical_exact": (
                _prediction_change_outcome(
                    scored["pip_canonical_exact"][index],
                    scored["pip_semantic"][index],
                )
                or "unchanged"
            ),
        }
    mode_summary = {
        mode: {
            **dict(sorted(counts.items())),
            **{
                f"{arm}_accuracy": counts[f"{arm}_correct"] / counts["commands"]
                for arm in arms
            },
        }
        for mode, counts in sorted(modes.items())
    }
    result = {
        "status": (
            "development_exposed_pip_carrier_selected"
            if selection != "stop_semantic_method"
            else "development_exposed_pip_carrier_no_go"
            if baseline_reconciled
            else "invalid_pip_baseline_reconciliation"
        ),
        "claim_bearing": False,
        "objective": "command_latency_pip_carrier_audit",
        "inputs": dict(provenance),
        "protocol": {
            "evaluation_unit": "eligible_exec_command",
            "causal_update": "KB observations settle only after whole task",
            "state_information": "raw outputs of earlier exec calls in the same task",
            "pip_partition": "interpreter, invocation, normalized flags",
            "pip_similarity": "Jaccard over normalized package names",
            "pip_canonical_exact": (
                "exact normalized interpreter, invocation, flags, and complete requirements"
            ),
            "pooling_alpha": INTERACTION_ALPHA,
            "exact_shortcut": "repository-local exact argv",
            "semantic_ablation": "same matching without availability or remaining-package state",
            "compound_composition": "weighted-empirical-shell-graph-v1",
            "weighted_composition_draws": COMMAND_COMPOSITION_DRAWS,
        },
        "counts": {
            "tasks": len(task_ids),
            "commands": len(command_rows),
            "pip_commands": len(pip_indices),
            "nonpip_commands": len(nonpip_indices),
            "target_online_clause_observations": len(clause_rows),
            "public_online_clause_observations": len(public),
            "stored_target_pip_observations": semantic.observation_count,
        },
        "row_identity": {
            "identical_command_ids_labels_and_availability": True,
            "nonpip_probability_vectors_bit_identical": nonpip_identical,
            "unique_sample_ids": len({row[0] for row in identity}) == len(identity),
        },
        "latency": {
            "bucket_edges_ms": list(CANONICAL_LATENCY_BUCKETS.edges_ms),
            "direction": "higher accuracy is better",
            "overall": {
                "majority": {
                    "class": current_metrics["majority_class"],
                    "accuracy": current_metrics["majority_class_accuracy"],
                },
                "arms": metrics,
            },
            "pip_commands": {
                "majority": {
                    "class": pip_current_metrics["majority_class"],
                    "accuracy": pip_current_metrics["majority_class_accuracy"],
                },
                "arms": pip_metrics,
            },
            "prediction_changes": changes,
            "pairwise_selection_changes": pairwise_changes,
            "carrier_net_spread": carrier_spread,
            "uncertainty": {
                f"{arm}_minus_current": _paired_task_cluster_bootstrap(
                    scored[arm],
                    scored["current"],
                    left_name=arm,
                    right_name="current",
                )
                for arm in arms[1:]
            },
        },
        "case_study": {
            "pip_execution_modes": mode_summary,
            "observed_clause_composition_oracle": _observed_composition_summary(
                command_rows, scored["current"]
            ),
        },
        "gates": {
            "baseline_reconciliation": {
                "pass": baseline_reconciled,
                "expected": (
                    None
                    if expected_pip_baseline is None
                    else {
                        "pip_commands": expected_pip_baseline[0],
                        "current_correct": expected_pip_baseline[1],
                    }
                ),
                "observed": {
                    "pip_commands": len(pip_indices),
                    "current_correct": current_pip_correct,
                },
            },
            "phase_a_selection": {
                "selected": selection,
                "jaccard_go": jaccard_go,
                "canonical_exact_go": canonical_go,
                "requires_strict_overall_and_pip_accuracy_gain": True,
                "requires_helpful_above_harmful": True,
                "requires_two_positive_net_tasks": True,
                "requires_two_positive_net_signatures": True,
                "requires_nonpip_bit_identity": True,
            },
            "resource_evaluation": {
                "go": False,
                "reason": "Phase A selects a representation; prior resource transfer is report-only",
            },
        },
    }
    return result, sidecar


def evaluate_pip_resources(
    public_rows: Sequence[Row],
    task_ids: Sequence[str],
    clause_rows: Sequence[Row],
    command_rows: Sequence[CommandRow],
    events_by_task: Mapping[str, Sequence[PipExecEvent]],
    provenance: Mapping[str, Any],
    latency_gate: Mapping[str, Any],
    *,
    expected_pip_baseline: tuple[int, int] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Transfer the latency-gated pip representation to the resource targets."""

    if len(set(task_ids)) != len(task_ids) or len(
        {repo_of(task_id) for task_id in task_ids}
    ) != 1:
        raise ValueError("pip resource evaluation requires unique tasks from one repo")
    target_repo = repo_of(task_ids[0])
    if any(row.repo == target_repo for row in public_rows):
        raise ValueError("public evidence contains the target repository")
    task_index = {task_id: index for index, task_id in enumerate(task_ids)}
    if list(events_by_task) != list(task_ids) or any(
        row.task_id not in task_index
        or row.manifest_index != task_index[row.task_id]
        for row in (*clause_rows, *command_rows)
    ):
        raise ValueError("target rows or raw events differ from accepted task order")
    public = tuple(
        row
        for row in public_rows
        if row.structure_known and row.pipeline_position <= 0
    )
    gate_inputs = latency_gate.get("inputs")
    gate_counts = latency_gate.get("counts")
    gate_protocol = latency_gate.get("protocol")
    gate_identity = latency_gate.get("row_identity")
    required_input_keys = (
        "target_run_dir",
        "public_telemetry",
        "public_excluded_repositories",
        "public_clause_observations_before_repo_filter",
        "public_clause_observations_after_repo_filter",
        "public_online_eligible_clause_observations",
    )
    target_pip_commands = sum(
        any(
            parse_pip_install(clause.get("argv", ())) is not None
            for clause in parse_command_clauses(row.command)["clauses"]
        )
        for row in command_rows
    )
    baseline_gate = latency_gate.get("gates", {}).get(
        "baseline_reconciliation", {}
    )
    gate_matches = (
        latency_gate.get("status") == "development_exposed_pip_latency_go"
        and latency_gate.get("objective")
        == "command_latency_pip_semantic_state_comparison"
        and isinstance(gate_inputs, Mapping)
        and all(gate_inputs.get(key) == provenance.get(key) for key in required_input_keys)
        and isinstance(gate_counts, Mapping)
        and gate_counts.get("tasks") == len(task_ids)
        and gate_counts.get("commands") == len(command_rows)
        and gate_counts.get("target_online_clause_observations") == len(clause_rows)
        and gate_counts.get("public_online_clause_observations") == len(public)
        and gate_counts.get("pip_commands") == target_pip_commands
        and isinstance(gate_protocol, Mapping)
        and all(
            gate_protocol.get(key) == value
            for key, value in {
                "evaluation_unit": "eligible_exec_command",
                "causal_update": "KB observations settle only after whole task",
                "state_information": "raw outputs of earlier exec calls in the same task",
                "pip_partition": "interpreter, invocation, normalized flags",
                "pip_similarity": "Jaccard over normalized package names",
                "pooling_alpha": INTERACTION_ALPHA,
                "exact_shortcut": "repository-local exact argv",
                "semantic_ablation": "same matching without availability or remaining-package state",
                "compound_composition": "weighted-empirical-shell-graph-v1",
                "weighted_composition_draws": COMMAND_COMPOSITION_DRAWS,
            }.items()
        )
        and isinstance(gate_identity, Mapping)
        and gate_identity.get("identical_command_ids_labels_and_availability") is True
        and gate_identity.get("nonpip_probability_vectors_bit_identical") is True
        and baseline_gate.get("pass") is True
        and baseline_gate.get("observed", {}).get("pip_commands")
        == target_pip_commands
        and (
            expected_pip_baseline is None
            or baseline_gate.get("expected")
            == {
                "pip_commands": expected_pip_baseline[0],
                "current_correct": expected_pip_baseline[1],
            }
            and baseline_gate.get("observed")
            == {
                "pip_commands": expected_pip_baseline[0],
                "current_correct": expected_pip_baseline[1],
            }
        )
        and latency_gate.get("gates", {}).get("latency", {}).get("go") is True
        and latency_gate.get("gates", {}).get("resource_evaluation", {}).get("go")
        is True
    )
    if not gate_matches:
        raise ValueError("pip latency GO does not match this resource replay")

    current = ClauseResourceKB.fit_public(
        row.observation(0.0, 1.0) for row in public
    )
    memories = {
        resource: _PipSemanticKB(
            [
                replace(row, latency_ms=value)
                for row in public
                if (value := _row_resource_value(row, resource)) is not None
            ]
        )
        for resource in CANONICAL_RESOURCE_HEAVY_THRESHOLDS
    }
    clauses_by_task: dict[str, list[Row]] = defaultdict(list)
    commands_by_task: dict[str, list[CommandRow]] = defaultdict(list)
    for row in clause_rows:
        clauses_by_task[row.task_id].append(row)
    for row in command_rows:
        commands_by_task[row.task_id].append(row)

    arms = ("current", "pip_semantic", "pip_semantic_state")
    raw = {
        resource: {
            "all": {
                "label_sources": Counter(),
                "arms": {arm: _empty_confusion() for arm in arms},
            },
            "pip": {
                "label_sources": Counter(),
                "arms": {arm: _empty_confusion() for arm in arms},
            },
        }
        for resource in CANONICAL_RESOURCE_HEAVY_THRESHOLDS
    }
    sidecar: list[dict[str, Any]] = []
    for ordinal, task_id in enumerate(task_ids):
        query_ts = float(ordinal * 2 + 3)
        events = tuple(events_by_task[task_id])
        contexts = _pip_contexts_by_call(events)
        event_by_call = {event.call_id: event for event in events}
        if len(event_by_call) != len(events):
            raise ValueError(f"{task_id}: duplicate raw exec call id")
        settled_semantic: list[
            tuple[Row, PipInstallSignature, PipQueryState]
        ] = []
        for row in commands_by_task[task_id]:
            event = event_by_call.get(row.call_id)
            if event is None or event.command != row.command:
                raise ValueError(f"{row.call_id}: eligible command lacks raw output")
            parsed = parse_command_clauses(row.command)
            current_result = current.predict_command_resource_classes_from_clauses(
                row.repo,
                parsed["clauses"],
                query_ts,
                command=row.command,
                parse_failed=bool(parsed["parse_failed"]),
            )
            states = contexts[row.call_id]
            predictions: dict[
                str, Mapping[str, ClauseHeavyLightPrediction | None]
            ] = {"current": current_result.classifications}
            arm_diagnostics: dict[str, Mapping[str, Any]] = {}
            for arm, use_state in (
                ("pip_semantic", False),
                ("pip_semantic_state", True),
            ):
                predictions[arm], arm_diagnostics[arm] = _predict_pip_resources(
                    memories,
                    current,
                    row,
                    parsed,
                    states,
                    current_result.classifications,
                    use_state=use_state,
                )
            pip_command = any(
                parse_pip_install(clause.get("argv", ())) is not None
                for clause in parsed["clauses"]
            )
            labels: dict[str, bool | None] = {}
            sources: dict[str, str] = {}
            arm_values: dict[str, dict[str, Any]] = {arm: {} for arm in arms}
            for resource in CANONICAL_RESOURCE_HEAVY_THRESHOLDS:
                label, source = command_resource_label(row, resource)
                labels[resource] = label
                sources[resource] = source
                groups = ("all", "pip") if pip_command else ("all",)
                for group in groups:
                    raw[resource][group]["label_sources"][source] += 1
                current_prediction = predictions["current"].get(resource)
                for arm in arms:
                    prediction = predictions[arm].get(resource)
                    if (prediction is None) != (current_prediction is None):
                        raise AssertionError(
                            f"{arm} changed {resource} prediction availability"
                        )
                    if label is not None:
                        for group in groups:
                            _record_resource_prediction(
                                raw[resource][group]["arms"][arm],
                                label,
                                prediction,
                            )
                    arm_values[arm][resource] = {
                        "prediction": None if prediction is None else prediction.label,
                        "probability_heavy": (
                            None if prediction is None else prediction.probability_heavy
                        ),
                        "carrier": (
                            False
                            if arm == "current"
                            else arm_diagnostics[arm][resource]["carrier"]
                        ),
                    }
            sidecar.append(
                {
                    "sample_id": (
                        f"{row.task_id}:{row.manifest_index}:call:{row.call_index}"
                    ),
                    "task_id": row.task_id,
                    "task_ordinal": ordinal,
                    "call_id": row.call_id,
                    "command": row.command,
                    "pip_command": pip_command,
                    "resource_labels": labels,
                    "resource_label_sources": sources,
                    "arms": arm_values,
                }
            )
            settled_semantic.extend(_settled_pip_observations(row, states))
        settle_ts = query_ts + 0.5
        for row in clauses_by_task[task_id]:
            current.observe_completed_clause(row.observation(query_ts, settle_ts))
        for row, signature, state in settled_semantic:
            for resource, memory in memories.items():
                value = _row_resource_value(row, resource)
                if value is not None:
                    memory.observe(
                        task_id,
                        replace(row, latency_ms=value),
                        signature,
                        state,
                    )

    resources: dict[str, Any] = {}
    nonpip_identical = True
    for resource in CANONICAL_RESOURCE_HEAVY_THRESHOLDS:
        group_metrics: dict[str, Any] = {}
        for group in ("all", "pip"):
            current_metric = _finalize_resource_metric(
                raw[resource][group]["arms"]["current"],
                raw[resource][group]["label_sources"],
            )
            group_metrics[group] = {
                "majority": {
                    "class": current_metric["majority_class"],
                    "accuracy": (
                        None
                        if current_metric["eligible_n"] == 0
                        else max(
                            current_metric["heavy_count"],
                            current_metric["eligible_n"]
                            - current_metric["heavy_count"],
                        )
                        / current_metric["eligible_n"]
                    ),
                },
                "arms": {
                    arm: _finalize_resource_metric(
                        raw[resource][group]["arms"][arm],
                        raw[resource][group]["label_sources"],
                    )
                    for arm in arms
                },
            }
        changes: dict[str, Any] = {}
        for arm in arms[1:]:
            counts = Counter()
            for row in sidecar:
                label = row["resource_labels"][resource]
                if label is None:
                    continue
                current_label = row["arms"]["current"][resource]["prediction"]
                candidate_label = row["arms"][arm][resource]["prediction"]
                if current_label == candidate_label:
                    continue
                outcome = (
                    "helpful"
                    if (candidate_label == "heavy") == label
                    else "harmful"
                    if (current_label == "heavy") == label
                    else "neutral"
                )
                counts["changed"] += 1
                counts[outcome] += 1
                if row["arms"][arm][resource]["carrier"]:
                    counts["carrier_changed"] += 1
                    counts[f"carrier_{outcome}"] += 1
            changes[arm] = {
                key: counts[key]
                for key in (
                    "changed",
                    "helpful",
                    "harmful",
                    "neutral",
                    "carrier_changed",
                    "carrier_helpful",
                    "carrier_harmful",
                    "carrier_neutral",
                )
            }
        resources[resource] = {
            "threshold": CANONICAL_RESOURCE_HEAVY_THRESHOLDS[resource],
            "direction": "higher accuracy is better",
            **group_metrics,
            "prediction_changes": changes,
        }
        nonpip_identical &= all(
            row["arms"][arm][resource]["probability_heavy"]
            == row["arms"]["current"][resource]["probability_heavy"]
            for row in sidecar
            if not row["pip_command"]
            for arm in arms[1:]
        )

    result = {
        "status": "development_exposed_pip_resource_transfer",
        "claim_bearing": False,
        "objective": "command_resource_pip_semantic_state_comparison",
        "inputs": {
            **dict(provenance),
            "latency_gate_status": latency_gate.get("status"),
        },
        "protocol": {
            "evaluation_unit": "eligible_exec_command",
            "latency_gate_required": True,
            "representation": "unchanged pip semantic Jaccard with alpha 16",
            "causal_update": "KB observations settle only after whole task",
            "resource_truth": "strict command bounds from retained clause aggregates",
            "compound_composition": "pipeline sum then sequential max for CPU/RSS; additive Disk",
        },
        "counts": {
            "tasks": len(task_ids),
            "commands": len(command_rows),
            "pip_commands": sum(row["pip_command"] for row in sidecar),
            "target_online_clause_observations": len(clause_rows),
            "public_online_clause_observations": len(public),
            "stored_target_pip_observations_by_resource": {
                resource: memory.observation_count
                for resource, memory in memories.items()
            },
        },
        "row_identity": {
            "identical_command_ids_labels_and_availability": True,
            "nonpip_probability_vectors_bit_identical": nonpip_identical,
            "unique_sample_ids": len({row["sample_id"] for row in sidecar})
            == len(sidecar),
        },
        "resources": resources,
    }
    return result, sidecar


def evaluate_poset_resources(
    public_rows: Sequence[Row],
    task_ids: Sequence[str],
    clause_rows: Sequence[Row],
    command_rows: Sequence[CommandRow],
    provenance: Mapping[str, Any],
    latency_gate: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply the frozen poset arm to command CPU, RSS, and Disk targets."""

    if len(set(task_ids)) != len(task_ids):
        raise ValueError("target task order contains duplicates")
    if len({repo_of(task_id) for task_id in task_ids}) != 1:
        raise ValueError("poset resource evaluation requires one target repository")
    target_repo = repo_of(task_ids[0])
    if any(row.repo == target_repo for row in public_rows):
        raise ValueError("public evidence contains the target repository")
    task_index = {task_id: index for index, task_id in enumerate(task_ids)}
    if any(
        row.task_id not in task_index
        or row.manifest_index != task_index[row.task_id]
        for row in (*clause_rows, *command_rows)
    ):
        raise ValueError("target rows differ from results.jsonl task order")
    public = tuple(
        row
        for row in public_rows
        if row.structure_known and row.pipeline_position <= 0
    )
    gate_inputs = latency_gate.get("inputs")
    gate_counts = latency_gate.get("counts")
    gate_protocol = latency_gate.get("protocol")
    gate_identity = latency_gate.get("row_identity")
    required_input_keys = (
        "target_run_dir",
        "public_telemetry",
        "public_excluded_repositories",
        "public_clause_observations_before_repo_filter",
        "public_clause_observations_after_repo_filter",
        "public_online_eligible_clause_observations",
    )
    gate_matches = (
        latency_gate.get("status") == "development_exposed_latency_go"
        and latency_gate.get("objective")
        == "command_latency_interaction_kb_comparison"
        and isinstance(gate_inputs, Mapping)
        and all(gate_inputs.get(key) == provenance.get(key) for key in required_input_keys)
        and isinstance(gate_counts, Mapping)
        and gate_counts.get("tasks") == len(task_ids)
        and gate_counts.get("commands") == len(command_rows)
        and gate_counts.get("target_online_clause_observations") == len(clause_rows)
        and gate_counts.get("public_online_clause_observations") == len(public)
        and isinstance(gate_protocol, Mapping)
        and gate_protocol.get("evaluation_unit") == "eligible_exec_command"
        and gate_protocol.get("pooling_alpha") == INTERACTION_ALPHA
        and isinstance(gate_identity, Mapping)
        and gate_identity.get("identical_command_ids_labels_and_availability") is True
        and latency_gate.get("gates", {}).get("stage0", {}).get("pass") is True
        and latency_gate.get("gates", {})
        .get("stage1_resource_evaluation", {})
        .get("interaction_poset", {})
        .get("go")
        is True
    )
    if not gate_matches:
        raise ValueError("latency GO does not match this resource replay")
    current = ClauseResourceKB.fit_public(
        row.observation(0.0, 1.0) for row in public
    )
    stable_subcommands = _fit_stable_subcommands(
        row.observation(0.0, 1.0) for row in public
    )
    posets = {
        resource: _InteractionPosetKB(stable_subcommands)
        for resource in CANONICAL_RESOURCE_HEAVY_THRESHOLDS
    }
    public_by_resource = {
        resource: tuple(
            row for row in public if _row_resource_value(row, resource) is not None
        )
        for resource in CANONICAL_RESOURCE_HEAVY_THRESHOLDS
    }
    if any(not rows for rows in public_by_resource.values()):
        raise ValueError("a resource has no usable frozen public evidence")
    public_by_resource_bin: dict[str, dict[str, list[Row]]] = {
        resource: defaultdict(list)
        for resource in CANONICAL_RESOURCE_HEAVY_THRESHOLDS
    }
    for resource, rows in public_by_resource.items():
        for row in rows:
            public_by_resource_bin[resource][row.bin].append(row)
    clauses_by_task: dict[str, list[Row]] = defaultdict(list)
    commands_by_task: dict[str, list[CommandRow]] = defaultdict(list)
    for row in clause_rows:
        clauses_by_task[row.task_id].append(row)
    for row in command_rows:
        commands_by_task[row.task_id].append(row)
    raw = {
        resource: {
            "label_source_counts": Counter(),
            "arms": {
                arm: _empty_confusion()
                for arm in ("current", "interaction_poset")
            },
        }
        for resource in CANONICAL_RESOURCE_HEAVY_THRESHOLDS
    }
    sidecar: list[dict[str, Any]] = []
    for ordinal, task_id in enumerate(task_ids):
        query_ts = float(ordinal * 2 + 3)
        for row in commands_by_task[task_id]:
            parsed = parse_command_clauses(row.command)
            parsed_clauses = parsed["clauses"]
            parse_failed = bool(parsed["parse_failed"])
            current_result = current.predict_command_resource_classes_from_clauses(
                row.repo,
                parsed_clauses,
                query_ts,
                command=row.command,
                parse_failed=parse_failed,
            )
            candidate, unavailable, diagnostics = _predict_poset_resources(
                posets,
                row,
                parsed_clauses,
                parse_failed=parse_failed,
                public_by_resource_bin=public_by_resource_bin,
                public_by_resource=public_by_resource,
            )
            labels: dict[str, bool | None] = {}
            sources: dict[str, str] = {}
            arm_values: dict[str, Any] = {
                "current": {},
                "interaction_poset": {},
            }
            for resource in CANONICAL_RESOURCE_HEAVY_THRESHOLDS:
                label, source = command_resource_label(row, resource)
                labels[resource] = label
                sources[resource] = source
                raw[resource]["label_source_counts"][source] += 1
                current_prediction = current_result.classifications.get(resource)
                candidate_prediction = candidate.get(resource)
                if label is not None:
                    if (current_prediction is None) != (candidate_prediction is None):
                        raise AssertionError(
                            f"{resource} prediction availability differs across arms"
                        )
                    _record_resource_prediction(
                        raw[resource]["arms"]["current"],
                        label,
                        current_prediction,
                    )
                    _record_resource_prediction(
                        raw[resource]["arms"]["interaction_poset"],
                        label,
                        candidate_prediction,
                    )
                arm_values["current"][resource] = {
                    "prediction": (
                        None if current_prediction is None else current_prediction.label
                    ),
                    "probability_heavy": (
                        None
                        if current_prediction is None
                        else current_prediction.probability_heavy
                    ),
                    "unavailable_reason": current_result.unavailable_reason,
                }
                arm_values["interaction_poset"][resource] = {
                    "prediction": (
                        None
                        if candidate_prediction is None
                        else candidate_prediction.label
                    ),
                    "probability_heavy": (
                        None
                        if candidate_prediction is None
                        else candidate_prediction.probability_heavy
                    ),
                    "unavailable_reason": unavailable,
                    **diagnostics.get(resource, {}),
                }
            sidecar.append(
                {
                    "sample_id": (
                        f"{row.task_id}:{row.manifest_index}:call:{row.call_index}"
                    ),
                    "task_id": row.task_id,
                    "task_ordinal": ordinal,
                    "call_id": row.call_id,
                    "command": row.command,
                    "resource_labels": labels,
                    "resource_label_sources": sources,
                    "arms": arm_values,
                }
            )
        settle_ts = query_ts + 0.5
        settled = clauses_by_task[task_id]
        for row in settled:
            current.observe_completed_clause(row.observation(query_ts, settle_ts))
        for resource, kb in posets.items():
            kb.observe(
                [
                    row
                    for row in settled
                    if _row_resource_value(row, resource) is not None
                ]
            )
    resources: dict[str, Any] = {}
    for resource, resource_raw in raw.items():
        current_metric = _finalize_resource_metric(
            resource_raw["arms"]["current"],
            resource_raw["label_source_counts"],
        )
        candidate_metric = _finalize_resource_metric(
            resource_raw["arms"]["interaction_poset"],
            resource_raw["label_source_counts"],
        )
        changes = Counter()
        for row in sidecar:
            label = row["resource_labels"][resource]
            if label is None:
                continue
            current_label = row["arms"]["current"][resource]["prediction"]
            candidate_label = row["arms"]["interaction_poset"][resource][
                "prediction"
            ]
            if current_label == candidate_label:
                continue
            outcome = (
                "helpful"
                if (candidate_label == "heavy") == label
                else "harmful"
                if (current_label == "heavy") == label
                else "neutral"
            )
            changes["changed"] += 1
            changes[outcome] += 1
            if row["arms"]["interaction_poset"][resource]["carrier"]:
                changes["carrier_changed"] += 1
                changes[f"carrier_{outcome}"] += 1
        resources[resource] = {
            "threshold": CANONICAL_RESOURCE_HEAVY_THRESHOLDS[resource],
            "direction": "higher accuracy is better",
            "majority": {
                "class": current_metric["majority_class"],
                "accuracy": current_metric["majority_class_accuracy"],
            },
            "constant_light_accuracy": current_metric["majority_light_accuracy"],
            "current": current_metric,
            "interaction_poset": candidate_metric,
            "candidate_minus_current_percentage_points": _accuracy_delta(
                candidate_metric["accuracy"],
                current_metric["accuracy"],
            ),
            "prediction_changes": {
                key: changes[key]
                for key in (
                    "changed",
                    "helpful",
                    "harmful",
                    "neutral",
                    "carrier_changed",
                    "carrier_helpful",
                    "carrier_harmful",
                    "carrier_neutral",
                )
            },
        }
    result = {
        "status": "development_exposed_poset_resource_evaluation",
        "claim_bearing": False,
        "objective": "command_resource_interaction_poset_comparison",
        "inputs": {
            **dict(provenance),
            "latency_gate_status": latency_gate.get("status"),
        },
        "protocol": {
            "evaluation_unit": "eligible_exec_command",
            "candidate": "latency-gated interaction_poset only",
            "causal_update": "all task commands predict before task clauses settle",
            "public_layer": "identical frozen cross-repository bin/global values",
            "exact_shortcut": "resource-usable repo-local exact hash; no pooling",
            "pooling_alpha": INTERACTION_ALPHA,
            "compound_composition": {
                "cpu_rss": "pipeline sum then stage max",
                "disk": "sum all clauses",
            },
            "short_null_light_max_latency_ms_exclusive": (
                SHORT_NULL_LIGHT_MAX_LATENCY_MS
            ),
        },
        "counts": {
            "tasks": len(task_ids),
            "commands": len(command_rows),
            "target_online_clause_observations": len(clause_rows),
            "public_online_clause_observations": len(public),
            "stored_usable_target_observations_by_resource": {
                resource: kb._next_observation_id for resource, kb in posets.items()
            },
        },
        "row_identity": {
            "identical_command_ids_labels_and_availability": True,
            "unique_sample_ids": len({row["sample_id"] for row in sidecar})
            == len(sidecar),
        },
        "resources": resources,
    }
    return result, sidecar


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _paired_task_cluster_bootstrap(
    left: Sequence[ScoredRow],
    right: Sequence[ScoredRow],
    *,
    left_name: str,
    right_name: str,
    seed: int = 0,
    draws: int = 2000,
) -> dict[str, Any]:
    identity = [
        (row.sample_id, row.task_id, row.label_bucket) for row in left
    ]
    if identity != [
        (row.sample_id, row.task_id, row.label_bucket) for row in right
    ]:
        raise AssertionError("task bootstrap arms have different rows or labels")
    if any(row.probability_by_bucket is None for row in (*left, *right)):
        raise ValueError("task bootstrap refuses unavailable predictions")
    indices_by_task: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(left):
        indices_by_task[row.task_id].append(index)
    tasks = sorted(indices_by_task)
    rng = random.Random(seed)

    def statistic(indices: Sequence[int]) -> float:
        return sum(
            (_argmax_bucket(left[index]) == left[index].label_bucket)
            - (_argmax_bucket(right[index]) == right[index].label_bucket)
            for index in indices
        ) / len(indices)

    point = statistic(range(len(left)))
    samples = []
    for _ in range(draws):
        sampled_tasks = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        samples.append(
            statistic(
                [
                    index
                    for task_id in sampled_tasks
                    for index in indices_by_task[task_id]
                ]
            )
        )
    return {
        "method": "paired_task_cluster_percentile_bootstrap",
        "cluster_unit": "task",
        "seed": seed,
        "draws": draws,
        "statistic": f"{left_name}_accuracy_minus_{right_name}_accuracy",
        "point_estimate": point,
        "median": _percentile(samples, 0.5),
        "interval_95": [
            _percentile(samples, 0.025),
            _percentile(samples, 0.975),
        ],
        "positive_draw_fraction": sum(value > 0.0 for value in samples) / draws,
        "confirmatory": False,
    }


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
