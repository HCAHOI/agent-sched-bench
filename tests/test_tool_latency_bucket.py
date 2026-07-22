from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.exploration.evaluate_tool_latency_buckets import main as evaluate_buckets_main
from trace_collect.kv_profile_sweep import KVSwapProfileEntry
from trace_collect.tool_latency_bucket import (
    bucket_edges_from_profile,
    evaluate_latency_buckets,
    latency_bucket,
)


def test_bucket_decisions_use_same_tool_history_and_global_fallback() -> None:
    rows = [
        _latency_row("seed-global", "seed-tool", 50.0, tool_ts_start=0.0),
        _latency_row("first-zircon", "zircon-saw", 900.0, tool_ts_start=1.0),
        _latency_row("second-zircon", "zircon-saw", 700.0, tool_ts_start=2.0),
    ]

    summary = evaluate_latency_buckets(rows, bucket_edges_ms=[100.0, 500.0])

    decisions = {row["sample_id"]: row for row in summary["decisions"]}
    assert summary["row_count"] == 3
    assert summary["bucket_count"] == 3

    assert decisions["seed-global"]["prediction_source"] == "cold_start"
    assert decisions["seed-global"]["history_count"] == 0
    assert decisions["seed-global"]["predicted_bucket"] is None
    assert decisions["seed-global"]["probability_by_bucket"] is None
    assert decisions["seed-global"]["label_bucket"] == 0

    assert decisions["first-zircon"]["prediction_source"] == "global_history"
    assert decisions["first-zircon"]["history_count"] == 1
    assert decisions["first-zircon"]["probability_by_bucket"] == [1.0, 0.0, 0.0]
    assert decisions["first-zircon"]["predicted_bucket"] == 0
    assert decisions["first-zircon"]["label_bucket"] == 2

    assert decisions["second-zircon"]["prediction_source"] == "tool_history"
    assert decisions["second-zircon"]["history_count"] == 1
    assert decisions["second-zircon"]["probability_by_bucket"] == [0.0, 0.0, 1.0]
    assert decisions["second-zircon"]["predicted_bucket"] == 2
    assert decisions["second-zircon"]["label_bucket"] == 2


def test_latency_equal_to_edge_falls_into_lower_bucket() -> None:
    edges = [100.0, 500.0]
    assert latency_bucket(100.0, edges) == 0
    assert latency_bucket(100.0001, edges) == 1
    assert latency_bucket(500.0, edges) == 1
    assert latency_bucket(500.0001, edges) == 2
    assert latency_bucket(0.0, edges) == 0


def test_probability_ties_resolve_to_lowest_bucket() -> None:
    rows = [
        _latency_row("hist-fast", "tie-probe", 50.0, tool_ts_start=0.0),
        _latency_row("hist-slow", "tie-probe", 900.0, tool_ts_start=1.0),
        _latency_row("scored", "tie-probe", 300.0, tool_ts_start=2.0),
    ]

    summary = evaluate_latency_buckets(rows, bucket_edges_ms=[100.0, 500.0])

    decisions = {row["sample_id"]: row for row in summary["decisions"]}
    scored = decisions["scored"]
    assert scored["prediction_source"] == "tool_history"
    assert scored["probability_by_bucket"] == [0.5, 0.0, 0.5]
    assert scored["predicted_bucket"] == 0
    assert scored["label_bucket"] == 1


def test_bucket_metrics_report_confusion_and_directional_errors() -> None:
    rows = [
        _latency_row("seed", "metric-probe", 50.0, tool_ts_start=0.0),
        _latency_row("under", "metric-probe", 900.0, tool_ts_start=1.0),
        _latency_row("exact", "metric-probe", 60.0, tool_ts_start=2.0),
    ]

    summary = evaluate_latency_buckets(rows, bucket_edges_ms=[100.0, 500.0])

    metrics = summary["metrics"]
    assert metrics["row_count"] == 3
    assert metrics["evaluated_count"] == 2
    assert metrics["cold_start_count"] == 1
    assert metrics["label_counts"] == [2, 0, 1]
    # "under" predicts bucket 0 against label 2; "exact" predicts 0 against 0.
    assert metrics["accuracy"] == 0.5
    assert metrics["mean_abs_bucket_error"] == 1.0
    assert metrics["underestimate_rate"] == 0.5
    assert metrics["overestimate_rate"] == 0.0
    assert metrics["confusion"][2][0] == 1
    assert metrics["confusion"][0][0] == 1
    assert summary["metrics_by_tool"]["metric-probe"]["row_count"] == 3


def test_bucket_edges_from_profile_dedupes_and_adds_guard() -> None:
    entries = [
        _profile_entry(kv_size_mb=256.0, p95_ms=100.0),
        _profile_entry(kv_size_mb=512.0, p95_ms=200.0),
        _profile_entry(kv_size_mb=1024.0, p95_ms=200.0),
    ]

    edges = bucket_edges_from_profile(entries, quantile="p95", guard_ms=50.0)

    assert edges == [150.0, 250.0]


def test_bucket_edges_from_profile_rejects_negative_guard() -> None:
    entries = [_profile_entry(kv_size_mb=256.0, p95_ms=100.0)]

    with pytest.raises(ValueError, match="guard_ms must be finite and non-negative"):
        bucket_edges_from_profile(entries, quantile="p95", guard_ms=-1.0)


@pytest.mark.parametrize("edge", [float("nan"), float("inf"), 0.0, -5.0])
def test_bucket_evaluator_rejects_invalid_edges(edge: float) -> None:
    rows = [_latency_row("good", "tool-a", 100.0, tool_ts_start=1.0)]

    with pytest.raises(ValueError, match="bucket edges must be finite and positive"):
        evaluate_latency_buckets(rows, bucket_edges_ms=[edge])


def test_bucket_evaluator_rejects_empty_edges() -> None:
    rows = [_latency_row("good", "tool-a", 100.0, tool_ts_start=1.0)]

    with pytest.raises(ValueError, match="at least one bucket edge is required"):
        evaluate_latency_buckets(rows, bucket_edges_ms=[])


def test_causal_walk_validates_sample_id_before_tool_name() -> None:
    # Pins the shared walk's field-validation order for doubly-malformed rows.
    row = _latency_row("bad", "tool-a", 100.0, tool_ts_start=1.0)
    row["sample_id"] = 123
    row["tool_name"] = 456

    with pytest.raises(ValueError, match="'sample_id' must be a string"):
        evaluate_latency_buckets([row], bucket_edges_ms=[100.0])


def test_bucket_evaluator_rejects_min_tool_history_below_one() -> None:
    rows = [_latency_row("good", "tool-a", 100.0, tool_ts_start=1.0)]

    with pytest.raises(ValueError, match="min_tool_history must be >= 1"):
        evaluate_latency_buckets(rows, bucket_edges_ms=[100.0], min_tool_history=0)


def test_evaluate_tool_latency_buckets_cli_derives_edges_from_kv_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    latencies_path = tmp_path / "latencies.jsonl"
    profile_path = tmp_path / "kv_profile.jsonl"
    summary_path = tmp_path / "summary.json"
    decisions_path = tmp_path / "decisions.jsonl"
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
            "evaluate_tool_latency_buckets.py",
            "--latencies",
            str(latencies_path),
            "--kv-profile",
            str(profile_path),
            "--quantile",
            "p95",
            "--guard-ms",
            "50",
            "--output",
            str(summary_path),
            "--decisions-output",
            str(decisions_path),
        ],
    )

    evaluate_buckets_main()

    assert f"Evaluated 2 rows into 3 buckets -> {summary_path}" in capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    decisions = _read_jsonl(decisions_path)
    assert "decisions" not in summary
    assert summary["bucket_edges_ms"] == [200.0, 300.0]
    assert summary["row_count"] == 2
    assert decisions == [
        {
            "history_count": 0,
            "label_bucket": 0,
            "latency_ms": 100.0,
            "predicted_bucket": None,
            "prediction_source": "cold_start",
            "probability_by_bucket": None,
            "sample_id": "first",
            "tool_name": "fixture-tool",
            "tool_ts_start": 1.0,
        },
        {
            "history_count": 1,
            "label_bucket": 1,
            "latency_ms": 300.0,
            "predicted_bucket": 0,
            "prediction_source": "tool_history",
            "probability_by_bucket": [1.0, 0.0, 0.0],
            "sample_id": "second",
            "tool_name": "fixture-tool",
            "tool_ts_start": 2.0,
        },
    ]


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


def _profile_entry(*, kv_size_mb: float, p95_ms: float) -> KVSwapProfileEntry:
    return KVSwapProfileEntry.from_mapping(
        _profile_row(kv_size_mb=kv_size_mb, p95_ms=p95_ms),
        source="fixture",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
