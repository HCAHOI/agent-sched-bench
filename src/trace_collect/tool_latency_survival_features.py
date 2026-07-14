"""Causal feature extraction for the discrete-time hazard latency model.

The hazard model predicts a call's latency distribution from the trace
prefix that precedes it. This module turns raw trace rows into per-call
feature records that are, by construction, functions of *only* the calls
completed (or started) before the scored call — never the call's own
latency and never any whole-trace statistic. Two stages:

``iter_causal_row_features`` walks each task in causal order with the same
completion-gated heap as :mod:`trace_collect.tool_latency_within_task`: a
call enters history only once its ``tool_ts_end`` is at or before the scored
call's ``tool_ts_start`` (a latency is usable only once observed; start
order alone leaks overlapping still-running calls). It emits a
``CausalRowFeatures`` per call carrying the within-task history summaries,
the running task mean, and the call's position — plus ``latency_ms`` as the
label, which is *never* fed back as a feature.

``fit_feature_encoder`` builds the tool and command-prefix vocabularies from
the TRAIN split only (sorted for determinism) and returns a
``FittedFeatureEncoder`` whose ``transform`` maps one ``CausalRowFeatures``
to a fixed-length numeric vector. Out-of-vocabulary tools/prefixes at
transform time contribute an all-zero block (never an error). Latency-derived
numeric features are heavy-tailed, so they enter through ``log1p``; a missing
(``None``) history value becomes ``0.0`` paired with an explicit is-present
indicator column, so absence is represented distinctly from a genuine zero.

The four ``SurvivalFeatureSpec`` toggles gate which encoder blocks are
emitted (feature ablations); the causal walk always populates every
``CausalRowFeatures`` field, so features stay a pure function of the rows,
``command_field``, ``max_prefix_depth`` and ``skip_leading_cd`` regardless of
the toggles.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import heapq
from typing import Any, Callable, Iterable, Iterator

import numpy as np

from trace_collect.command_features import make_row_command_prefix_keys
from trace_collect.latency_validation import (
    required_nonnegative_float,
    required_text,
)


@dataclass(frozen=True)
class SurvivalFeatureSpec:
    """Configuration for causal feature extraction and encoding.

    ``command_field`` names the key inside ``tool_args`` holding the shell
    command; ``None`` disables command-prefix grouping (tool-level only).
    ``max_prefix_depth`` and ``skip_leading_cd`` match the frozen empirical
    trie configuration. The four ``use_*`` flags are feature ablations that
    gate encoder blocks (tool one-hot, command-prefix multi-hot, within-task
    history summaries, running task aggregates).
    """

    command_field: str | None = "command"
    max_prefix_depth: int = 4
    skip_leading_cd: bool = False
    use_tool_identity: bool = True
    use_command_prefix: bool = True
    use_within_task_history: bool = True
    use_task_aggregates: bool = True


@dataclass(frozen=True)
class CausalRowFeatures:
    """Per-call features; every field is a function of the trace prefix
    BEFORE this call. ``latency_ms`` is the LABEL and is never a feature.

    ``group_keys`` is the call's own command-prefix key chain (general ->
    specific). The ``within_task_prefix_*`` fields summarise the deepest of
    those keys that has any completed same-task history; ``None``/``0`` mean
    no such history. ``within_task_tool_last_ms`` is the most recently
    completed same-task call of the same tool. ``call_index_in_task`` counts
    calls started before this one (0-based). ``task_running_mean_ms`` is the
    mean latency over all same-task calls completed so far (``None`` if none).
    """

    sample_id: str
    tool_name: str
    group_keys: tuple[str, ...]
    within_task_prefix_last_ms: float | None
    within_task_prefix_median_ms: float | None
    within_task_prefix_count: int
    within_task_tool_last_ms: float | None
    call_index_in_task: int
    task_running_mean_ms: float | None
    latency_ms: float


def iter_causal_row_features(
    rows: Iterable[dict[str, Any]],
    *,
    spec: SurvivalFeatureSpec,
) -> Iterator[CausalRowFeatures]:
    """Yield ``CausalRowFeatures`` in causal order, grouped by task.

    Rows are validated (required fields, ``tool_ts_end >= tool_ts_start``,
    unique ``sample_id``, one ``source_trace`` per ``task_id``), grouped by
    ``task_id``, then walked in ``(tool_ts_start, sample_id)`` order. A
    completion heap admits a prior call into history only once its
    ``tool_ts_end`` is at or before the scored call's ``tool_ts_start``, so
    overlapping still-running calls (latency not yet observable) never leak
    and a call never sees itself. No whole-trace statistics are used.
    """

    row_group_keys = (
        make_row_command_prefix_keys(
            spec.command_field,
            max_depth=spec.max_prefix_depth,
            skip_leading_cd=spec.skip_leading_cd,
        )
        if spec.command_field is not None
        else None
    )

    rows_by_task = _group_rows_by_task(rows)
    for task_id in sorted(rows_by_task):
        yield from _iter_task_features(
            rows_by_task[task_id],
            row_group_keys=row_group_keys,
        )


def _group_rows_by_task(
    rows: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Validate every row and group by ``task_id``.

    Mirrors ``within_task_trigger_rows``' validation: required text/float
    fields, ``tool_ts_end >= tool_ts_start``, globally unique ``sample_id``,
    and a single ``source_trace`` per task (timestamps are comparable only
    within one trace, so a task collected twice would mix attempts).
    """

    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_samples: set[str] = set()
    trace_by_task: dict[str, str] = {}
    for index, row in enumerate(rows):
        source = f"survival row {index}"
        sample_id = required_text(row, "sample_id", source=source)
        if sample_id in seen_samples:
            raise ValueError(f"duplicate survival sample_id: {sample_id!r}")
        seen_samples.add(sample_id)
        required_text(row, "tool_name", source=source)
        required_nonnegative_float(row, "latency_ms", source=source)
        ts_start = required_nonnegative_float(row, "tool_ts_start", source=source)
        ts_end = required_nonnegative_float(row, "tool_ts_end", source=source)
        if ts_end < ts_start:
            raise ValueError(f"{source} ends before it starts")
        task_id = required_text(row, "task_id", source=source)
        trace = required_text(row, "source_trace", source=source)
        existing_trace = trace_by_task.setdefault(task_id, trace)
        if existing_trace != trace:
            raise ValueError(
                f"task {task_id!r} spans multiple source traces: "
                f"{existing_trace!r} and {trace!r}"
            )
        rows_by_task[task_id].append(row)
    return rows_by_task


def _iter_task_features(
    task_rows: list[dict[str, Any]],
    *,
    row_group_keys: Callable[[dict[str, Any]], tuple[str, ...]] | None,
) -> Iterator[CausalRowFeatures]:
    """Walk one task's calls in causal order, yielding causal features."""

    ordered = sorted(
        task_rows,
        key=lambda row: (float(row["tool_ts_start"]), str(row["sample_id"])),
    )
    history_by_group: dict[str, list[float]] = defaultdict(list)
    history_by_tool: dict[str, list[float]] = defaultdict(list)
    pending: list[tuple[float, int, float, tuple[str, ...], str]] = []
    completed_sum = 0.0
    completed_count = 0
    for order, row in enumerate(ordered):
        ts_start = float(row["tool_ts_start"])
        while pending and pending[0][0] <= ts_start:
            _, _, done_latency, done_keys, done_tool = heapq.heappop(pending)
            for group_key in done_keys:
                history_by_group[group_key].append(done_latency)
            history_by_tool[done_tool].append(done_latency)
            completed_sum += done_latency
            completed_count += 1
        tool_name = str(row["tool_name"])
        group_keys = row_group_keys(row) if row_group_keys is not None else ()
        prefix_history = _deepest_prefix_history(group_keys, history_by_group)
        tool_history = history_by_tool.get(tool_name, [])
        yield CausalRowFeatures(
            sample_id=str(row["sample_id"]),
            tool_name=tool_name,
            group_keys=group_keys,
            within_task_prefix_last_ms=(
                prefix_history[-1] if prefix_history else None
            ),
            within_task_prefix_median_ms=(
                float(np.median(prefix_history)) if prefix_history else None
            ),
            within_task_prefix_count=len(prefix_history),
            within_task_tool_last_ms=tool_history[-1] if tool_history else None,
            call_index_in_task=order,
            task_running_mean_ms=(
                completed_sum / completed_count if completed_count else None
            ),
            latency_ms=float(row["latency_ms"]),
        )
        heapq.heappush(
            pending,
            (
                float(row["tool_ts_end"]),
                order,
                float(row["latency_ms"]),
                group_keys,
                tool_name,
            ),
        )


def _deepest_prefix_history(
    group_keys: tuple[str, ...],
    history_by_group: dict[str, list[float]],
) -> list[float]:
    """Completed-history list of the deepest prefix key with any samples.

    ``group_keys`` is ordered general -> specific, so we scan from the
    deepest key inward, mirroring ``tool_latency_within_task._select_history``.
    Returns an empty list when no prefix key has completed history.
    """

    for group_key in reversed(group_keys):
        history = history_by_group.get(group_key)
        if history:
            return history
    return []


# Numeric feature layout, applied in this fixed order after the categorical
# blocks. Latency-derived values are heavy-tailed, so they enter through
# ``log1p`` and, being optional, carry a paired is-present indicator so a
# missing value (0.0) is distinct from an observed zero. Counts are always
# present (0 is meaningful) and need no indicator.
_WITHIN_TASK_BLOCK_WIDTH = 8  # last(2) + median(2) + count(1) + tool_last(2) + idx(1)
_TASK_AGGREGATE_BLOCK_WIDTH = 2  # running_mean(2)


@dataclass(frozen=True)
class FittedFeatureEncoder:
    """Fixed-width numeric encoder fitted on a TRAIN split.

    ``tool_vocab`` and ``prefix_vocab`` map each seen tool / command-prefix
    key to a stable column index (built from train rows only, sorted for
    determinism). ``transform`` emits, in order: the tool one-hot block, the
    command-prefix multi-hot block, the within-task history numeric block,
    and the running task-aggregate block — each gated by the matching
    ``spec`` toggle. Unseen tools/prefixes contribute an all-zero block.
    """

    tool_vocab: dict[str, int]
    prefix_vocab: dict[str, int]
    spec: SurvivalFeatureSpec
    n_columns: int

    def transform(self, feats: CausalRowFeatures) -> np.ndarray:
        """Encode one ``CausalRowFeatures`` as a length-``n_columns`` vector."""

        blocks: list[np.ndarray] = []
        if self.spec.use_tool_identity:
            tool_block = np.zeros(len(self.tool_vocab), dtype=np.float64)
            index = self.tool_vocab.get(feats.tool_name)
            if index is not None:
                tool_block[index] = 1.0
            blocks.append(tool_block)
        if self.spec.use_command_prefix:
            prefix_block = np.zeros(len(self.prefix_vocab), dtype=np.float64)
            for group_key in feats.group_keys:
                index = self.prefix_vocab.get(group_key)
                if index is not None:
                    prefix_block[index] = 1.0
            blocks.append(prefix_block)
        if self.spec.use_within_task_history:
            blocks.append(_optional_latency(feats.within_task_prefix_last_ms))
            blocks.append(_optional_latency(feats.within_task_prefix_median_ms))
            blocks.append(_count_feature(feats.within_task_prefix_count))
            blocks.append(_optional_latency(feats.within_task_tool_last_ms))
            blocks.append(_count_feature(feats.call_index_in_task))
        if self.spec.use_task_aggregates:
            blocks.append(_optional_latency(feats.task_running_mean_ms))
        vector = (
            np.concatenate(blocks)
            if blocks
            else np.zeros(0, dtype=np.float64)
        )
        if vector.shape[0] != self.n_columns:
            raise AssertionError(
                f"encoded width {vector.shape[0]} != n_columns {self.n_columns}"
            )
        return vector


def _optional_latency(value: float | None) -> np.ndarray:
    """``[log1p(value), 1.0]`` when present, ``[0.0, 0.0]`` when missing."""

    if value is None:
        return np.zeros(2, dtype=np.float64)
    return np.array([float(np.log1p(value)), 1.0], dtype=np.float64)


def _count_feature(value: int) -> np.ndarray:
    """``[log1p(count)]``; counts are always present, so no indicator."""

    return np.array([float(np.log1p(value))], dtype=np.float64)


def fit_feature_encoder(
    train_rows: Iterable[dict[str, Any]],
    *,
    spec: SurvivalFeatureSpec,
) -> FittedFeatureEncoder:
    """Fit tool/prefix vocabularies from TRAIN rows only.

    Vocabularies collect every tool name and command-prefix key that appears
    in the causal walk of ``train_rows``, sorted for deterministic column
    indices. ``n_columns`` is fixed by the enabled ``spec`` blocks.
    """

    tools: set[str] = set()
    prefixes: set[str] = set()
    for feats in iter_causal_row_features(train_rows, spec=spec):
        tools.add(feats.tool_name)
        prefixes.update(feats.group_keys)
    tool_vocab = {name: index for index, name in enumerate(sorted(tools))}
    prefix_vocab = {key: index for index, key in enumerate(sorted(prefixes))}

    n_columns = 0
    if spec.use_tool_identity:
        n_columns += len(tool_vocab)
    if spec.use_command_prefix:
        n_columns += len(prefix_vocab)
    if spec.use_within_task_history:
        n_columns += _WITHIN_TASK_BLOCK_WIDTH
    if spec.use_task_aggregates:
        n_columns += _TASK_AGGREGATE_BLOCK_WIDTH

    return FittedFeatureEncoder(
        tool_vocab=tool_vocab,
        prefix_vocab=prefix_vocab,
        spec=spec,
        n_columns=n_columns,
    )


__all__ = [
    "CausalRowFeatures",
    "FittedFeatureEncoder",
    "SurvivalFeatureSpec",
    "fit_feature_encoder",
    "iter_causal_row_features",
]
