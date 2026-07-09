from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.sweep_swap_cutoffs import main as sweep_cutoffs_main
from trace_collect.swap_cutoff_sweep import evaluate_swap_cutoff_sweep


def _fixture_rows() -> list[dict[str, object]]:
    # Single tool; causal history at each row: cold, [50], [50, 900], [50, 900, 900].
    # Survival P(latency > 100): None, 0.0, 0.5, 2/3. Labels: F, T, T, F.
    return [
        _latency_row("cold", "probe", 50.0, tool_ts_start=0.0),
        _latency_row("first", "probe", 900.0, tool_ts_start=1.0),
        _latency_row("second", "probe", 900.0, tool_ts_start=2.0),
        _latency_row("third", "probe", 60.0, tool_ts_start=3.0),
    ]


def test_cutoff_zero_swaps_all_evaluated_rows_and_accounts_costs() -> None:
    summary = evaluate_swap_cutoff_sweep(
        _fixture_rows(),
        kv_costs_ms=[100.0],
        guard_ms=0.0,
        probability_cutoffs=[0.0],
    )

    assert summary["cold_start_policy"] == "no_swap"
    assert summary["row_count"] == 4
    (point,) = summary["sweep"]
    assert point["kv_cost_ms"] == 100.0
    assert point["threshold_ms"] == 100.0
    assert point["evaluated_count"] == 3
    assert point["cold_start_count"] == 1
    assert point["swap_count"] == 3
    assert point["coverage"] == 1.0
    assert point["true_positive_count"] == 2
    assert point["false_positive_count"] == 1
    assert point["false_negative_count"] == 0
    assert point["stall_rate"] == pytest.approx(1 / 3)
    # Exposed only on the early-returning swap: 100 - 60.
    assert point["exposed_ms_total"] == 40.0
    # Hidden swap time: min(100, 900) + min(100, 900) + min(100, 60).
    assert point["absorbed_ms_total"] == 260.0
    assert point["missed_ms_total"] == 0.0
    assert point["absorbed_if_oracle_ms"] == 200.0


def test_higher_cutoffs_trade_coverage_for_missed_opportunity() -> None:
    summary = evaluate_swap_cutoff_sweep(
        _fixture_rows(),
        kv_costs_ms=[100.0],
        guard_ms=0.0,
        probability_cutoffs=[0.6, 1.0],
    )

    points = {point["probability_cutoff"]: point for point in summary["sweep"]}

    # Only the third row's survival (2/3) clears 0.6, and its label is False.
    mid = points[0.6]
    assert mid["swap_count"] == 1
    assert mid["coverage"] == pytest.approx(1 / 3)
    assert mid["false_positive_count"] == 1
    assert mid["false_negative_count"] == 2
    assert mid["stall_rate"] == 1.0
    assert mid["exposed_ms_total"] == 40.0
    assert mid["absorbed_ms_total"] == 60.0
    assert mid["missed_ms_total"] == 200.0

    top = points[1.0]
    assert top["swap_count"] == 0
    assert top["coverage"] == 0.0
    assert top["stall_rate"] is None
    assert top["exposed_ms_total"] == 0.0
    assert top["absorbed_ms_total"] == 0.0
    assert top["missed_ms_total"] == 200.0


def test_cold_start_positive_rows_count_as_missed_opportunity() -> None:
    rows = [_latency_row("only", "probe", 900.0, tool_ts_start=0.0)]

    summary = evaluate_swap_cutoff_sweep(
        rows,
        kv_costs_ms=[100.0],
        guard_ms=0.0,
        probability_cutoffs=[0.5],
    )

    (point,) = summary["sweep"]
    assert point["evaluated_count"] == 0
    assert point["cold_start_count"] == 1
    assert point["coverage"] is None
    assert point["positive_count"] == 1
    assert point["missed_ms_total"] == 100.0
    assert point["absorbed_if_oracle_ms"] == 100.0


def test_guard_widens_threshold_but_exposure_uses_kv_cost() -> None:
    # Latency 120 with kv cost 100 and guard 50: label is False (120 <= 150)
    # but a swap exposes nothing because the guard region absorbed it.
    rows = [
        _latency_row("seed", "probe", 900.0, tool_ts_start=0.0),
        _latency_row("guarded", "probe", 120.0, tool_ts_start=1.0),
    ]

    summary = evaluate_swap_cutoff_sweep(
        rows,
        kv_costs_ms=[100.0],
        guard_ms=50.0,
        probability_cutoffs=[0.5],
    )

    (point,) = summary["sweep"]
    assert point["threshold_ms"] == 150.0
    assert point["swap_count"] == 1
    assert point["false_positive_count"] == 1
    assert point["exposed_ms_total"] == 0.0
    assert point["absorbed_ms_total"] == 100.0


def test_sweep_rejects_empty_rows() -> None:
    with pytest.raises(ValueError, match="no latency rows supplied"):
        evaluate_swap_cutoff_sweep(
            [],
            kv_costs_ms=[100.0],
            guard_ms=0.0,
            probability_cutoffs=[0.5],
        )


@pytest.mark.parametrize("cutoff", [float("nan"), float("inf"), -0.1, 1.5])
def test_sweep_rejects_out_of_range_cutoffs(cutoff: float) -> None:
    with pytest.raises(ValueError, match="probability cutoffs must be finite"):
        evaluate_swap_cutoff_sweep(
            _fixture_rows(),
            kv_costs_ms=[100.0],
            guard_ms=0.0,
            probability_cutoffs=[cutoff],
        )


def test_sweep_rejects_empty_cutoffs_and_invalid_costs_and_guard() -> None:
    with pytest.raises(ValueError, match="at least one probability cutoff"):
        evaluate_swap_cutoff_sweep(
            _fixture_rows(),
            kv_costs_ms=[100.0],
            guard_ms=0.0,
            probability_cutoffs=[],
        )
    with pytest.raises(ValueError, match="kv costs must be finite and positive"):
        evaluate_swap_cutoff_sweep(
            _fixture_rows(),
            kv_costs_ms=[0.0],
            guard_ms=0.0,
            probability_cutoffs=[0.5],
        )
    with pytest.raises(ValueError, match="guard_ms must be finite and non-negative"):
        evaluate_swap_cutoff_sweep(
            _fixture_rows(),
            kv_costs_ms=[100.0],
            guard_ms=-1.0,
            probability_cutoffs=[0.5],
        )


def test_sweep_swap_cutoffs_cli_derives_costs_from_kv_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    latencies_path = tmp_path / "latencies.jsonl"
    profile_path = tmp_path / "kv_profile.jsonl"
    output_path = tmp_path / "sweep.json"
    _write_jsonl(
        latencies_path,
        [
            _latency_row("first", "fixture-tool", 100.0, tool_ts_start=1.0),
            _latency_row("second", "fixture-tool", 300.0, tool_ts_start=2.0),
        ],
    )
    _write_jsonl(
        profile_path,
        [
            _profile_row(kv_size_mb=256.0, p95_ms=150.0),
            _profile_row(kv_size_mb=512.0, p95_ms=250.0),
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sweep_swap_cutoffs.py",
            "--latencies",
            str(latencies_path),
            "--kv-profile",
            str(profile_path),
            "--quantile",
            "p95",
            "--guard-ms",
            "50",
            "--cutoffs",
            "0.5",
            "--output",
            str(output_path),
        ],
    )

    sweep_cutoffs_main()

    assert (
        f"Swept 2 (kv cost, cutoff) points over 2 rows -> {output_path}"
        in capsys.readouterr().out
    )
    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["kv_costs_ms"] == [150.0, 250.0]
    assert summary["guard_ms"] == 50.0
    points = {point["kv_cost_ms"]: point for point in summary["sweep"]}
    # kv 150 -> threshold 200: second row (300ms) exceeds but history says decline.
    assert points[150.0]["threshold_ms"] == 200.0
    assert points[150.0]["false_negative_count"] == 1
    assert points[150.0]["missed_ms_total"] == 150.0
    # kv 250 -> threshold 300: 300ms does not strictly exceed, decline is correct.
    assert points[250.0]["threshold_ms"] == 300.0
    assert points[250.0]["true_negative_count"] == 1
    assert points[250.0]["missed_ms_total"] == 0.0


def _latency_row(
    sample_id: str,
    tool_name: str,
    latency_ms: float,
    *,
    tool_ts_start: float,
    source_trace: str = "trace-a",
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "source_trace": source_trace,
        "tool_name": tool_name,
        "latency_ms": latency_ms,
        "tool_ts_start": tool_ts_start,
        "tool_ts_end": tool_ts_start + latency_ms / 1000.0,
    }


def _profile_row(*, kv_size_mb: float, p95_ms: float) -> dict[str, object]:
    return {
        "engine": "fixture-engine",
        "mechanism": "swap",
        "device": "fixture-gpu",
        "kv_size_mb": kv_size_mb,
        "direction": "out",
        "memory_path": "gpu-to-cpu",
        "concurrency": "single",
        "samples": 10,
        "p50_ms": p95_ms * 0.5,
        "p90_ms": p95_ms * 0.9,
        "p95_ms": p95_ms,
        "p99_ms": p95_ms * 1.2,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
