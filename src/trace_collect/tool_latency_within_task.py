"""Within-task history baseline for tool-latency action triggers.

The cross-task pipeline predicts from other tasks' calls. This baseline uses
only the current task's own earlier calls: at each tool call, the same-task
calls already **completed** at its start, in the deepest available context
(command-prefix depth back-off, then tool level), form an empirical latency
sample, and the trigger is the same expected-utility-optimal re-check time
the cross-task policies use at the same restore cost. Each row also carries
``within_task_margin_normalized`` — the history's expected utility advantage
over the deadline per kv cost, from the same ``mean_clock_region_stats``
projection the cross-task probe scores — so a cross-fitted margin guard can
gate these triggers exactly as it gates the cross-task ones. A call whose
context has no completed same-task sample falls back to the fixed deadline
with a zero margin.

With a single prior sample this reduces to the last-value rule: an earlier
long call triggers immediately, an earlier short call waits for the
deadline. History admits a call only when ``tool_ts_end <= tool_ts_start``
of the current call — a latency is usable only once observed. Start-order
alone would leak: traces contain overlapping tool intervals whose earlier
call is still running (latency unknown) when the next one starts.
"""

from __future__ import annotations

from collections import defaultdict
import heapq
import math
from typing import Any, Callable, Iterable

from trace_collect.command_features import make_row_command_prefix_keys
from trace_collect.latency_validation import (
    normalized_positive_floats,
    required_nonnegative_float,
    required_text,
)
from trace_collect.tool_latency_offline_probe import mean_clock_region_stats
from trace_collect.tool_latency_utility_clock import validate_restore_cost


def within_task_trigger_rows(
    rows: Iterable[dict[str, Any]],
    *,
    kv_costs_ms: Iterable[float],
    guard_ms: float,
    command_field: str | None = None,
    max_prefix_depth: int = 4,
    skip_leading_cd: bool = False,
    restore_cost_fraction: float = 0.0,
) -> list[dict[str, Any]]:
    """Score every call with its within-task history trigger per kv cost."""

    kv_costs = normalized_positive_floats(kv_costs_ms, label="kv cost")
    if not math.isfinite(guard_ms) or guard_ms < 0.0:
        raise ValueError(f"guard_ms must be finite and non-negative, got {guard_ms}")
    validate_restore_cost(restore_cost_fraction, label="restore_cost_fraction")
    row_group_keys = (
        make_row_command_prefix_keys(
            command_field,
            max_depth=max_prefix_depth,
            skip_leading_cd=skip_leading_cd,
        )
        if command_field is not None
        else None
    )

    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_samples: set[str] = set()
    trace_by_task: dict[str, str] = {}
    for index, row in enumerate(rows):
        source = f"within-task row {index}"
        sample_id = required_text(row, "sample_id", source=source)
        if sample_id in seen_samples:
            raise ValueError(f"duplicate within-task sample_id: {sample_id!r}")
        seen_samples.add(sample_id)
        required_text(row, "tool_name", source=source)
        required_nonnegative_float(row, "latency_ms", source=source)
        ts_start = required_nonnegative_float(row, "tool_ts_start", source=source)
        ts_end = required_nonnegative_float(row, "tool_ts_end", source=source)
        if ts_end < ts_start:
            raise ValueError(f"{source} ends before it starts")
        task_id = required_text(row, "task_id", source=source)
        # Timestamps are only comparable within one trace; a task collected
        # more than once would mix attempts into one causal history.
        trace = required_text(row, "source_trace", source=source)
        existing_trace = trace_by_task.setdefault(task_id, trace)
        if existing_trace != trace:
            raise ValueError(
                f"task {task_id!r} spans multiple source traces: "
                f"{existing_trace!r} and {trace!r}"
            )
        rows_by_task[task_id].append(row)

    output: list[dict[str, Any]] = []
    for task_id in sorted(rows_by_task):
        output.extend(
            _score_task_rows(
                rows_by_task[task_id],
                kv_costs=kv_costs,
                guard_ms=guard_ms,
                row_group_keys=row_group_keys,
                restore_cost_fraction=restore_cost_fraction,
            )
        )
    return output


def _score_task_rows(
    task_rows: list[dict[str, Any]],
    *,
    kv_costs: list[float],
    guard_ms: float,
    row_group_keys: Callable[[dict[str, Any]], tuple[str, ...]] | None,
    restore_cost_fraction: float,
) -> list[dict[str, Any]]:
    """Score one task's calls against completed same-task history.

    A call enters the history only once its ``tool_ts_end`` is at or before
    the scored call's start: overlapping calls (still running, latency not
    yet observable) never contaminate a trigger, and a call never sees
    itself.
    """

    ordered = sorted(
        task_rows,
        key=lambda row: (float(row["tool_ts_start"]), str(row["sample_id"])),
    )
    history_by_group: dict[str, list[float]] = defaultdict(list)
    history_by_tool: dict[str, list[float]] = defaultdict(list)
    pending: list[tuple[float, int, float, tuple[str, ...], str]] = []
    output: list[dict[str, Any]] = []
    for order, row in enumerate(ordered):
        ts_start = float(row["tool_ts_start"])
        while pending and pending[0][0] <= ts_start:
            _, _, done_latency, done_keys, done_tool = heapq.heappop(pending)
            for group_key in done_keys:
                history_by_group[group_key].append(done_latency)
            history_by_tool[done_tool].append(done_latency)
        group_keys = row_group_keys(row) if row_group_keys is not None else ()
        history, history_source = _select_history(
            row,
            group_keys=group_keys,
            history_by_group=history_by_group,
            history_by_tool=history_by_tool,
        )
        for kv_cost_ms in kv_costs:
            threshold_ms = kv_cost_ms + guard_ms
            if history:
                # The same estimator the cross-task probe uses: its trigger is
                # hazard_recheck_ms by construction, and normalized_margin is
                # the history's expected utility advantage over the deadline
                # (per kv cost), the score the margin guard gates on.
                stats = mean_clock_region_stats(
                    history,
                    threshold_ms=threshold_ms,
                    kv_cost_ms=kv_cost_ms,
                    restore_cost_ms=restore_cost_fraction * kv_cost_ms,
                )
                trigger_ms = stats.trigger_ms
                margin_normalized = stats.normalized_margin
            else:
                trigger_ms = threshold_ms
                margin_normalized = 0.0
            output.append(
                {
                    "sample_id": row["sample_id"],
                    "task_id": row["task_id"],
                    "tool_name": row["tool_name"],
                    "latency_ms": float(row["latency_ms"]),
                    "kv_cost_ms": kv_cost_ms,
                    "threshold_ms": threshold_ms,
                    "within_task_trigger_ms": trigger_ms,
                    "within_task_margin_normalized": margin_normalized,
                    "within_task_source": history_source,
                    "within_task_history_count": len(history),
                }
            )
        heapq.heappush(
            pending,
            (
                float(row["tool_ts_end"]),
                order,
                float(row["latency_ms"]),
                group_keys,
                str(row["tool_name"]),
            ),
        )
    return output


def _select_history(
    row: dict[str, Any],
    *,
    group_keys: tuple[str, ...],
    history_by_group: dict[str, list[float]],
    history_by_tool: dict[str, list[float]],
) -> tuple[list[float], str]:
    """Deepest context with any strictly earlier sample; mirrors the
    cross-task hierarchy order (deepest prefix first, then tool level)."""

    for group_key in reversed(group_keys):
        history = history_by_group.get(group_key)
        if history:
            return history, f"within_group:{group_key}"
    history = history_by_tool.get(str(row["tool_name"]))
    if history:
        return history, "within_tool"
    return [], "none"


__all__ = ["within_task_trigger_rows"]
