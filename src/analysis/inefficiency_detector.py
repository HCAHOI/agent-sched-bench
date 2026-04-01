from __future__ import annotations

from typing import Any

import pandas as pd


def detect_inefficiencies(
    frame: pd.DataFrame,
    *,
    metrics_frame: pd.DataFrame | None = None,
    preemption_report: dict[str, Any] | None = None,
    long_tool_wait_threshold_ms: float = 1000.0,
    idle_cache_threshold_perc: float = 50.0,
) -> dict[str, Any]:
    """Flag initial heuristic inefficiency patterns from step-level trace data."""
    step_rows = frame[frame["type"] == "step"].copy()
    acting_rows = step_rows[step_rows["phase"] == "acting"]
    long_tool_waits = acting_rows[
        acting_rows["tool_duration_ms"].fillna(0.0) > long_tool_wait_threshold_ms
    ]
    failed_tools = acting_rows[acting_rows["tool_success"] == False]  # noqa: E712
    thrashing_events = 0
    bubble_count = 0
    idle_memory_seconds = 0.0

    if preemption_report:
        thrashing_events += len(preemption_report.get("eviction_events", []))
        delta = preemption_report.get("preemption_counter_delta")
        if delta:
            thrashing_events += int(delta)

    if metrics_frame is not None and not metrics_frame.empty:
        ordered = metrics_frame.sort_values("timestamp").reset_index(drop=True)
        for index in range(len(ordered) - 1):
            row = ordered.iloc[index]
            next_row = ordered.iloc[index + 1]
            interval_s = float(next_row["timestamp"] - row["timestamp"])
            if (
                float(row.get("vllm:num_requests_waiting", 0.0)) > 0.0
                and float(row.get("vllm:num_requests_running", 0.0)) == 0.0
            ):
                bubble_count += 1
            if (
                float(row.get("vllm:gpu_cache_usage_perc", 0.0)) >= idle_cache_threshold_perc
                and float(row.get("vllm:num_requests_running", 0.0)) == 0.0
            ):
                idle_memory_seconds += interval_s
    return {
        "heuristic_long_tool_wait_count": int(len(long_tool_waits)),
        "heuristic_failed_tool_count": int(len(failed_tools)),
        "heuristic_thrashing_event_count": int(thrashing_events),
        "heuristic_bubble_count": int(bubble_count),
        "heuristic_idle_memory_seconds": float(idle_memory_seconds),
        "long_tool_wait_threshold_ms": long_tool_wait_threshold_ms,
        "idle_cache_threshold_perc": idle_cache_threshold_perc,
    }
