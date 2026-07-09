"""Shared causal history walk for tool latency evaluation.

Rows are scored in causal order (``source_trace``, then ``tool_ts_start``).
A completed observation enters history only once its ``tool_ts_end`` is at or
before the next scored start time, so overlapping executions never leak their
own outcome into their prediction. Rows sharing the same trace and start time
are scored against the same history snapshot and cannot see each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator

from trace_collect.latency_validation import (
    required_nonnegative_float,
    required_text,
    row_order_key,
)


@dataclass(frozen=True)
class CausalLatencyObservation:
    """One latency row paired with the history available at its start time.

    ``history`` is the live backing list for ``prediction_source`` and is only
    valid until the iterator advances; consume it immediately and never mutate
    or store it.
    """

    sample_id: str
    tool_name: str
    tool_ts_start: float
    tool_ts_end: float
    latency_ms: float
    prediction_source: str
    history: list[float]
    group_key: str | None = None


def iter_causal_latency_observations(
    rows: Iterable[dict[str, Any]],
    *,
    min_tool_history: int = 1,
    row_group_key: Callable[[dict[str, Any]], str | None] | None = None,
) -> Iterator[CausalLatencyObservation]:
    """Yield rows in causal order with their group, per-tool, or global history.

    ``prediction_source`` is ``group_history`` when ``row_group_key`` assigns
    the row a group with at least ``min_tool_history`` completed observations,
    ``tool_history`` when that many completed same-tool observations exist,
    ``global_history`` when any completed observation exists, and
    ``cold_start`` (empty history) otherwise. No tool or command classes are
    hardcoded; grouping is purely by observed ``tool_name`` and the
    data-derived group key.
    """

    if min_tool_history < 1:
        raise ValueError(f"min_tool_history must be >= 1, got {min_tool_history}")

    ordered_rows = sorted(list(rows), key=row_order_key)
    global_history: list[float] = []
    history_by_tool: dict[str, list[float]] = {}
    history_by_group: dict[str, list[float]] = {}
    pending_updates: list[tuple[float, str, str | None, float]] = []
    current_source: str | None = None

    def _apply_update(tool_name: str, group_key: str | None, latency_ms: float) -> None:
        global_history.append(latency_ms)
        history_by_tool.setdefault(tool_name, []).append(latency_ms)
        if group_key is not None:
            history_by_group.setdefault(group_key, []).append(latency_ms)

    row_index = 0
    while row_index < len(ordered_rows):
        bucket_start = row_index
        bucket_source = required_text(
            ordered_rows[bucket_start],
            "source_trace",
            source=f"row {bucket_start}",
        )
        if current_source is not None and bucket_source != current_source:
            for _, tool_name, group_key, latency_ms in pending_updates:
                _apply_update(tool_name, group_key, latency_ms)
            pending_updates = []
        current_source = bucket_source
        bucket_ts_start = required_nonnegative_float(
            ordered_rows[bucket_start],
            "tool_ts_start",
            source=f"row {bucket_start}",
        )
        ready_updates = [
            update for update in pending_updates if update[0] <= bucket_ts_start
        ]
        pending_updates = [
            update for update in pending_updates if update[0] > bucket_ts_start
        ]
        for _, tool_name, group_key, latency_ms in ready_updates:
            _apply_update(tool_name, group_key, latency_ms)

        while row_index < len(ordered_rows):
            row = ordered_rows[row_index]
            source_trace = required_text(
                row,
                "source_trace",
                source=f"row {row_index}",
            )
            tool_ts_start = required_nonnegative_float(
                row,
                "tool_ts_start",
                source=f"row {row_index}",
            )
            if source_trace != bucket_source or tool_ts_start != bucket_ts_start:
                break
            row_index += 1

        bucket_rows = ordered_rows[bucket_start:row_index]
        bucket_updates: list[tuple[float, str, str | None, float]] = []
        for scored_index, row in enumerate(bucket_rows, start=bucket_start):
            sample_id = required_text(row, "sample_id", source=f"row {scored_index}")
            tool_name = required_text(row, "tool_name", source=f"row {scored_index}")
            latency_ms = required_nonnegative_float(
                row,
                "latency_ms",
                source=f"row {scored_index}",
            )
            tool_ts_start = required_nonnegative_float(
                row,
                "tool_ts_start",
                source=f"row {scored_index}",
            )
            tool_ts_end = required_nonnegative_float(
                row,
                "tool_ts_end",
                source=f"row {scored_index}",
            )
            if tool_ts_end < tool_ts_start:
                raise ValueError(f"row {scored_index}: tool_ts_end < tool_ts_start")

            group_key = row_group_key(row) if row_group_key is not None else None
            group_history = (
                history_by_group.get(group_key, []) if group_key is not None else []
            )
            tool_history = history_by_tool.get(tool_name, [])
            if group_key is not None and len(group_history) >= min_tool_history:
                prediction_source = "group_history"
                history = group_history
            elif len(tool_history) >= min_tool_history:
                prediction_source = "tool_history"
                history = tool_history
            elif global_history:
                prediction_source = "global_history"
                history = global_history
            else:
                prediction_source = "cold_start"
                history = []

            yield CausalLatencyObservation(
                sample_id=sample_id,
                tool_name=tool_name,
                tool_ts_start=tool_ts_start,
                tool_ts_end=tool_ts_end,
                latency_ms=latency_ms,
                prediction_source=prediction_source,
                history=history,
                group_key=group_key,
            )
            bucket_updates.append((tool_ts_end, tool_name, group_key, latency_ms))

        pending_updates.extend(bucket_updates)


__all__ = ["CausalLatencyObservation", "iter_causal_latency_observations"]
