from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.evaluate_swap_deadline_policy import main as deadline_policy_main
from trace_collect.swap_deadline_policy import evaluate_deadline_policy


def _profile_rows() -> list[dict[str, object]]:
    # probe survival at T=100 is 1.0 (always swap); quick is 0.0 (never swap).
    return [
        _latency_row("p1", "probe", 900.0, tool_ts_start=0.0, source_trace="trace-p"),
        _latency_row("p2", "probe", 900.0, tool_ts_start=1.0, source_trace="trace-p"),
        _latency_row("p3", "quick", 50.0, tool_ts_start=2.0, source_trace="trace-p"),
        _latency_row("p4", "quick", 50.0, tool_ts_start=3.0, source_trace="trace-p"),
    ]


def test_deadline_recheck_recovers_missed_long_calls() -> None:
    eval_rows = [
        _latency_row("swap-long", "probe", 900.0, tool_ts_start=0.0),
        _latency_row("missed-long", "quick", 900.0, tool_ts_start=1.0),
        _latency_row("true-short", "quick", 60.0, tool_ts_start=2.0),
        _latency_row("swap-short", "probe", 50.0, tool_ts_start=3.0),
    ]

    summary = evaluate_deadline_policy(
        eval_rows,
        profile_rows=_profile_rows(),
        kv_costs_ms=[100.0],
        guard_ms=0.0,
        predictor="prior_only",
    )

    assert summary["recheck_at"] == "threshold"
    (point,) = summary["points"]
    assert point["kv_cost_ms"] == 100.0
    assert point["threshold_ms"] == 100.0
    assert point["positive_count"] == 2
    assert point["absorbed_if_oracle_ms"] == 200.0

    t0 = point["policies"]["t0_only"]
    # t=0 swaps probe rows only: hides swap-long fully, exposes 50ms on
    # swap-short, misses missed-long entirely.
    assert t0["swap_count"] == 2
    assert t0["deadline_swap_count"] == 0
    assert t0["absorbed_ms_total"] == 150.0
    assert t0["absorbed_on_long_ms_total"] == 100.0
    assert t0["exposed_ms_total"] == 50.0
    assert t0["missed_ms_total"] == 100.0
    assert t0["hidden_fraction_of_oracle"] == 0.5

    deadline = point["policies"]["deadline_recheck"]
    # missed-long outlives the 100ms deadline; the late swap hides fully
    # (remaining window 800ms >= kv cost). true-short never triggers.
    assert deadline["swap_count"] == 2
    assert deadline["deadline_swap_count"] == 1
    assert deadline["absorbed_ms_total"] == 250.0
    assert deadline["absorbed_on_long_ms_total"] == 200.0
    assert deadline["exposed_ms_total"] == 50.0
    assert deadline["missed_ms_total"] == 0.0
    assert deadline["hidden_fraction_of_oracle"] == 1.0


def test_late_swap_near_deadline_hides_partially() -> None:
    profile_rows = [
        _latency_row("p1", "quick", 50.0, tool_ts_start=0.0, source_trace="trace-p"),
        _latency_row("p2", "quick", 50.0, tool_ts_start=1.0, source_trace="trace-p"),
    ]
    eval_rows = [_latency_row("barely-long", "quick", 160.0, tool_ts_start=0.0)]

    summary = evaluate_deadline_policy(
        eval_rows,
        profile_rows=profile_rows,
        kv_costs_ms=[100.0],
        guard_ms=50.0,
        predictor="prior_only",
    )

    (point,) = summary["points"]
    assert point["threshold_ms"] == 150.0
    t0 = point["policies"]["t0_only"]
    assert t0["missed_ms_total"] == 100.0
    assert t0["hidden_fraction_of_oracle"] == 0.0
    deadline = point["policies"]["deadline_recheck"]
    # Swap starts at the 150ms deadline; only 10ms of window remains.
    assert deadline["deadline_swap_count"] == 1
    assert deadline["absorbed_on_long_ms_total"] == 10.0
    assert deadline["exposed_ms_total"] == 90.0
    assert deadline["missed_ms_total"] == 0.0
    assert deadline["hidden_fraction_of_oracle"] == pytest.approx(0.1)


def test_cold_start_rows_are_deadline_eligible() -> None:
    # online_only with no history: the t=0 decision is None (no swap), but a
    # long call still gets the proven-label deadline swap.
    eval_rows = [_latency_row("cold-long", "quick", 900.0, tool_ts_start=0.0)]

    summary = evaluate_deadline_policy(
        eval_rows,
        profile_rows=_profile_rows(),
        kv_costs_ms=[100.0],
        guard_ms=0.0,
        predictor="online_only",
    )

    (point,) = summary["points"]
    assert point["cold_start_count"] == 1
    assert point["policies"]["t0_only"]["missed_ms_total"] == 100.0
    deadline = point["policies"]["deadline_recheck"]
    assert deadline["deadline_swap_count"] == 1
    assert deadline["missed_ms_total"] == 0.0


def test_latency_exactly_at_threshold_is_short_and_never_deadline_swapped() -> None:
    eval_rows = [_latency_row("boundary", "quick", 100.0, tool_ts_start=0.0)]

    summary = evaluate_deadline_policy(
        eval_rows,
        profile_rows=_profile_rows(),
        kv_costs_ms=[100.0],
        guard_ms=0.0,
        predictor="prior_only",
    )

    (point,) = summary["points"]
    assert point["positive_count"] == 0
    deadline = point["policies"]["deadline_recheck"]
    assert deadline["swap_count"] == 0
    assert deadline["deadline_swap_count"] == 0
    assert deadline["missed_ms_total"] == 0.0


def test_multiple_kv_costs_produce_independent_points() -> None:
    eval_rows = [_latency_row("mid-long", "quick", 900.0, tool_ts_start=0.0)]

    summary = evaluate_deadline_policy(
        eval_rows,
        profile_rows=_profile_rows(),
        kv_costs_ms=[100.0, 2000.0],
        guard_ms=0.0,
        predictor="prior_only",
    )

    points = {point["kv_cost_ms"]: point for point in summary["points"]}
    assert set(points) == {100.0, 2000.0}
    assert points[100.0]["threshold_ms"] == 100.0
    assert points[2000.0]["threshold_ms"] == 2000.0
    # 900ms exceeds the 100ms threshold (deadline swap, full hide) but not
    # the 2000ms one (correctly no swap at all).
    small = points[100.0]["policies"]["deadline_recheck"]
    assert small["deadline_swap_count"] == 1
    assert small["hidden_fraction_of_oracle"] == 1.0
    large = points[2000.0]["policies"]["deadline_recheck"]
    assert points[2000.0]["positive_count"] == 0
    assert large["swap_count"] == 0
    assert large["deadline_swap_count"] == 0


def test_deadline_invariants_hold_under_segment_costs() -> None:
    # Deductions may move a swap between t0 and the deadline, but can never
    # cause a deadline swap on a short call or miss a long one.
    profile_rows = [
        _latency_row("p1", "exec", 200.0, tool_ts_start=0.0, source_trace="trace-p",
                     tool_args={"command": "prep"}),
        _latency_row("p2", "exec", 200.0, tool_ts_start=1.0, source_trace="trace-p",
                     tool_args={"command": "prep"}),
        _latency_row("p3", "exec", 500.0, tool_ts_start=2.0, source_trace="trace-p",
                     tool_args={"command": "prep && work"}),
        _latency_row("p4", "exec", 500.0, tool_ts_start=3.0, source_trace="trace-p",
                     tool_args={"command": "prep && work"}),
    ]
    eval_rows = [
        # Deducted query 50ms on exec:work [300, 300] -> t0 swap.
        _latency_row("long-compound", "exec", 900.0, tool_ts_start=0.0,
                     tool_args={"command": "prep && work"}),
        # Same aggressive t0 swap, but the call is short: immediate FP with
        # 10ms exposure - never a deadline swap.
        _latency_row("short-compound", "exec", 240.0, tool_ts_start=1.0,
                     tool_args={"command": "prep && work"}),
        # exec:prep [200, 200] says no at 250ms; the call outlives the
        # deadline and is recovered by the proven-label late swap.
        _latency_row("recovered-prep", "exec", 900.0, tool_ts_start=2.0,
                     tool_args={"command": "prep"}),
        # Short call with a no-swap t0 decision: nothing ever fires.
        _latency_row("short-prep", "exec", 200.0, tool_ts_start=3.0,
                     tool_args={"command": "prep"}),
    ]

    summary = evaluate_deadline_policy(
        eval_rows,
        profile_rows=profile_rows,
        kv_costs_ms=[250.0],
        guard_ms=0.0,
        predictor="prior_only",
        command_field="command",
        segment_costs=True,
    )

    assert summary["segment_costs"] is True
    (point,) = summary["points"]
    assert point["positive_count"] == 2
    deadline = point["policies"]["deadline_recheck"]
    assert deadline["swap_count"] == 2
    assert deadline["deadline_swap_count"] == 1
    assert deadline["missed_ms_total"] == 0.0
    assert deadline["hidden_fraction_of_oracle"] == 1.0
    assert deadline["exposed_ms_total"] == 10.0


def test_deadline_policy_rejects_invalid_costs_and_guard() -> None:
    eval_rows = [_latency_row("good", "probe", 100.0, tool_ts_start=0.0)]

    with pytest.raises(ValueError, match="kv costs must be finite and positive"):
        evaluate_deadline_policy(
            eval_rows,
            profile_rows=_profile_rows(),
            kv_costs_ms=[0.0],
            guard_ms=0.0,
            predictor="prior_only",
        )
    with pytest.raises(ValueError, match="guard_ms must be finite and non-negative"):
        evaluate_deadline_policy(
            eval_rows,
            profile_rows=_profile_rows(),
            kv_costs_ms=[100.0],
            guard_ms=-1.0,
            predictor="prior_only",
        )


def test_deadline_policy_cli_writes_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path = tmp_path / "profile.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    output_path = tmp_path / "policy.json"
    _write_jsonl(profile_path, _profile_rows())
    _write_jsonl(
        eval_path,
        [
            _latency_row("long", "quick", 900.0, tool_ts_start=0.0),
            _latency_row("short", "quick", 60.0, tool_ts_start=1.0),
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_swap_deadline_policy.py",
            "--profile-latencies",
            str(profile_path),
            "--eval-latencies",
            str(eval_path),
            "--kv-costs-ms",
            "100",
            "--guard-ms",
            "0",
            "--output",
            str(output_path),
        ],
    )

    deadline_policy_main()

    assert f"Evaluated 1 kv points over 2 rows -> {output_path}" in capsys.readouterr().out
    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["predictor"] == "prior_only"
    (point,) = summary["points"]
    assert point["policies"]["t0_only"]["missed_ms_total"] == 100.0
    assert point["policies"]["deadline_recheck"]["missed_ms_total"] == 0.0
    assert point["policies"]["deadline_recheck"]["deadline_swap_count"] == 1


def _latency_row(
    sample_id: str,
    tool_name: str,
    latency_ms: float,
    *,
    tool_ts_start: float,
    source_trace: str = "trace-e",
    tool_args: dict[str, object] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_id": sample_id,
        "source_trace": source_trace,
        "tool_name": tool_name,
        "latency_ms": latency_ms,
        "tool_ts_start": tool_ts_start,
        "tool_ts_end": tool_ts_start + latency_ms / 1000.0,
    }
    if tool_args is not None:
        row["tool_args"] = tool_args
    return row


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
