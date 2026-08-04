from pathlib import Path

import pytest

from tool_resource_eval.early_cpu_reservation import (
    CPU_UPDATE_P95_S,
    SAMPLE_AVAILABILITY_PAD_S,
    action_row,
)


def _action(*, duration: float | None = None, sample_dt: float = 0.5) -> dict:
    if duration is None:
        duration = 4 * sample_dt + 0.02
    samples = [
        (sample_dt, 0.5 * sample_dt),
        (2 * sample_dt, 1.5 * sample_dt),
        (3 * sample_dt, 3.0 * sample_dt),
        (4 * sample_dt, 0.4 * sample_dt),
    ]
    return {
        "type": "action",
        "action_id": "action-1",
        "ts_start": 10.0,
        "ts_end": 10.0 + duration,
        "data": {
            "tool_name": "exec",
            "tool_call_id": "call-1",
            "resource_timeline": {
                "version": 1,
                "sample_interval_s": 0.5,
                "samples": [
                    {
                        "offset_s": offset,
                        "dt_s": sample_dt,
                        "cpu_core_s": cpu,
                        "cpu_quota_cores": 8,
                    }
                    for offset, cpu in samples
                    if offset <= duration
                ],
            },
        },
    }


def test_oracle_uses_smallest_page_covering_future_demand() -> None:
    row, reason = action_row(
        _action(), task_id="task-1", trace_path=Path("trace.jsonl")
    )

    assert reason == "eligible"
    assert row is not None
    assert row["effective_offset_s"] == pytest.approx(
        0.5 + SAMPLE_AVAILABILITY_PAD_S + CPU_UPDATE_P95_S
    )
    assert row["prefix_cpu_rate_cores"] == pytest.approx(0.5)
    assert row["future_peak_cpu_rate_cores"] == pytest.approx(3.0)
    assert row["oracle_page_cores"] == 4
    assert row["false_shrink"] is False
    assert row["saved_reserved_core_s"] == pytest.approx(4 * row["remaining_s"])


def test_oracle_requires_a_full_post_actuation_interval() -> None:
    row, reason = action_row(
        _action(duration=1.04), task_id="task-1", trace_path=Path("trace.jsonl")
    )

    assert row is None
    assert reason == "no_post_actuation_interval"


def test_oracle_rejects_short_or_cpu_missing_decision_sample() -> None:
    short_row, short_reason = action_row(
        _action(sample_dt=0.45),
        task_id="task-1",
        trace_path=Path("trace.jsonl"),
    )
    missing = _action()
    first = missing["data"]["resource_timeline"]["samples"][0]
    first.pop("cpu_core_s")
    first["net_rx_bytes"] = 1
    missing_row, missing_reason = action_row(
        missing,
        task_id="task-1",
        trace_path=Path("trace.jsonl"),
    )

    assert short_row is None
    assert short_reason == "no_full_decision_sample"
    assert missing_row is None
    assert missing_reason == "missing_decision_cpu"


def test_oracle_does_not_treat_short_tail_as_future_interval() -> None:
    action = _action(duration=1.24)
    action["data"]["resource_timeline"]["samples"].append(
        {
            "offset_s": 1.2,
            "dt_s": 0.2,
            "cpu_core_s": 1.6,
            "cpu_quota_cores": 8,
        }
    )

    row, reason = action_row(
        action, task_id="task-1", trace_path=Path("trace.jsonl")
    )

    assert row is None
    assert reason == "no_future_cpu"
