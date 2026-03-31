from __future__ import annotations

from typing import Any

import pandas as pd


def detect_inefficiencies(frame: pd.DataFrame, *, long_tool_wait_threshold_ms: float = 1000.0) -> dict[str, Any]:
    """Flag initial heuristic inefficiency patterns from step-level trace data."""
    step_rows = frame[frame["type"] == "step"].copy()
    acting_rows = step_rows[step_rows["phase"] == "acting"]
    long_tool_waits = acting_rows[
        acting_rows["tool_duration_ms"].fillna(0.0) > long_tool_wait_threshold_ms
    ]
    failed_tools = acting_rows[acting_rows["tool_success"] == False]  # noqa: E712
    return {
        "heuristic_long_tool_wait_count": int(len(long_tool_waits)),
        "heuristic_failed_tool_count": int(len(failed_tools)),
        "long_tool_wait_threshold_ms": long_tool_wait_threshold_ms,
    }
