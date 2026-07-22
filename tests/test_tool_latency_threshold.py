from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.exploration.evaluate_tool_latency_thresholds import main as evaluate_thresholds_main
from trace_collect.tool_latency_threshold import evaluate_latency_thresholds


def test_threshold_decisions_use_same_tool_history_and_global_fallback() -> None:
    rows = [
        _latency_row("seed-global", "seed-tool", 100.0, tool_ts_start=0.0),
        _latency_row("first-zircon", "zircon-saw", 900.0, tool_ts_start=1.0),
        _latency_row("second-zircon", "zircon-saw", 700.0, tool_ts_start=2.0),
    ]

    summary = evaluate_latency_thresholds(
        rows,
        thresholds_ms=[500.0],
        probability_cutoff=0.5,
    )

    decisions = {row["sample_id"]: row for row in summary["decisions"]}
    assert summary["row_count"] == 3
    assert summary["decision_count"] == 3

    assert decisions["seed-global"]["prediction_source"] == "cold_start"
    assert decisions["seed-global"]["history_count"] == 0
    assert decisions["seed-global"]["probability_exceeds_threshold"] is None
    assert decisions["seed-global"]["predicted_exceeds_threshold"] is None

    assert decisions["first-zircon"]["prediction_source"] == "global_history"
    assert decisions["first-zircon"]["history_count"] == 1
    assert decisions["first-zircon"]["probability_exceeds_threshold"] == 0.0
    assert decisions["first-zircon"]["predicted_exceeds_threshold"] is False
    assert decisions["first-zircon"]["label_exceeds_threshold"] is True

    assert decisions["second-zircon"]["prediction_source"] == "tool_history"
    assert decisions["second-zircon"]["history_count"] == 1
    assert decisions["second-zircon"]["probability_exceeds_threshold"] == 1.0
    assert decisions["second-zircon"]["predicted_exceeds_threshold"] is True
    assert decisions["second-zircon"]["label_exceeds_threshold"] is True


def test_exact_threshold_latency_is_not_label_or_survival_exceedance() -> None:
    rows = [
        _latency_row(
            "equal-threshold-history",
            "caliper-plane",
            500.0,
            tool_ts_start=0.0,
        ),
        _latency_row("later-same-tool", "caliper-plane", 750.0, tool_ts_start=1.0),
    ]

    summary = evaluate_latency_thresholds(
        rows,
        thresholds_ms=[500.0],
        probability_cutoff=0.5,
    )

    decisions = {row["sample_id"]: row for row in summary["decisions"]}
    assert decisions["equal-threshold-history"]["label_exceeds_threshold"] is False
    assert decisions["later-same-tool"]["prediction_source"] == "tool_history"
    assert decisions["later-same-tool"]["history_count"] == 1
    assert decisions["later-same-tool"]["probability_exceeds_threshold"] == 0.0
    assert decisions["later-same-tool"]["predicted_exceeds_threshold"] is False


def test_same_trace_same_start_rows_do_not_enter_each_others_history() -> None:
    rows = [
        _latency_row("global-fast", "history-seed", 100.0, tool_ts_start=0.0),
        _latency_row("same-start-slow", "neutrino-drill", 900.0, tool_ts_start=1.0),
        _latency_row("same-start-fast", "neutrino-drill", 50.0, tool_ts_start=1.0),
        _latency_row("after-same-start", "neutrino-drill", 20.0, tool_ts_start=2.0),
    ]

    summary = evaluate_latency_thresholds(
        rows,
        thresholds_ms=[500.0],
        probability_cutoff=0.5,
    )

    decisions = {row["sample_id"]: row for row in summary["decisions"]}
    for sample_id, label in [
        ("same-start-slow", True),
        ("same-start-fast", False),
    ]:
        assert decisions[sample_id]["prediction_source"] == "global_history"
        assert decisions[sample_id]["history_count"] == 1
        assert decisions[sample_id]["probability_exceeds_threshold"] == 0.0
        assert decisions[sample_id]["predicted_exceeds_threshold"] is False
        assert decisions[sample_id]["label_exceeds_threshold"] is label

    assert decisions["after-same-start"]["prediction_source"] == "tool_history"
    assert decisions["after-same-start"]["history_count"] == 2
    assert decisions["after-same-start"]["probability_exceeds_threshold"] == 0.5
    assert decisions["after-same-start"]["predicted_exceeds_threshold"] is True


def test_overlapping_tool_history_waits_until_tool_ts_end() -> None:
    rows = [
        _latency_row("global-fast", "history-seed", 100.0, tool_ts_start=0.0),
        _latency_row("long-before", "comet-hammer", 10_000.0, tool_ts_start=1.0),
        _latency_row(
            "overlap-before-long-finishes",
            "comet-hammer",
            100.0,
            tool_ts_start=2.0,
        ),
        _latency_row("after-long-finishes", "comet-hammer", 100.0, tool_ts_start=12.0),
    ]

    summary = evaluate_latency_thresholds(
        rows,
        thresholds_ms=[1_000.0],
        probability_cutoff=0.5,
    )

    decisions = {row["sample_id"]: row for row in summary["decisions"]}
    assert decisions["long-before"]["prediction_source"] == "global_history"
    assert decisions["long-before"]["history_count"] == 1
    assert decisions["long-before"]["probability_exceeds_threshold"] == 0.0

    assert decisions["overlap-before-long-finishes"]["prediction_source"] == "global_history"
    assert decisions["overlap-before-long-finishes"]["history_count"] == 1
    assert decisions["overlap-before-long-finishes"]["probability_exceeds_threshold"] == 0.0
    assert decisions["overlap-before-long-finishes"]["predicted_exceeds_threshold"] is False

    assert decisions["after-long-finishes"]["prediction_source"] == "tool_history"
    assert decisions["after-long-finishes"]["history_count"] == 2
    assert decisions["after-long-finishes"]["probability_exceeds_threshold"] == 0.5
    assert decisions["after-long-finishes"]["predicted_exceeds_threshold"] is True



def test_source_trace_boundary_flushes_completed_tool_history_despite_timestamp_reset() -> None:
    rows = [
        _latency_row(
            "trace-a-long-tool",
            "boundary-probe",
            10_000.0,
            tool_ts_start=100.0,
            source_trace="trace-a",
        ),
        _latency_row(
            "trace-b-first-tool",
            "boundary-probe",
            100.0,
            tool_ts_start=0.0,
            source_trace="trace-b",
        ),
    ]

    summary = evaluate_latency_thresholds(
        rows,
        thresholds_ms=[1_000.0],
        probability_cutoff=0.5,
    )

    decisions = {row["sample_id"]: row for row in summary["decisions"]}
    assert summary["row_count"] == 2
    assert summary["decision_count"] == 2

    assert decisions["trace-a-long-tool"]["prediction_source"] == "cold_start"
    assert decisions["trace-a-long-tool"]["history_count"] == 0

    assert decisions["trace-b-first-tool"]["prediction_source"] == "tool_history"
    assert decisions["trace-b-first-tool"]["history_count"] == 1
    assert decisions["trace-b-first-tool"]["probability_exceeds_threshold"] == 1.0
    assert decisions["trace-b-first-tool"]["predicted_exceeds_threshold"] is True
    assert decisions["trace-b-first-tool"]["label_exceeds_threshold"] is False

@pytest.mark.parametrize("field", ["sample_id", "source_trace", "tool_name"])
def test_threshold_evaluator_rejects_non_string_identity_fields(field: str) -> None:
    row = _latency_row("bad-identity", "tool-a", 100.0, tool_ts_start=1.0)
    row[field] = 123

    with pytest.raises(ValueError, match=rf"{field!r} must be a string"):
        evaluate_latency_thresholds([row], thresholds_ms=[50.0])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("latency_ms", float("nan")),
        ("latency_ms", float("inf")),
        ("tool_ts_start", float("nan")),
        ("tool_ts_start", float("inf")),
        ("tool_ts_end", float("nan")),
        ("tool_ts_end", float("-inf")),
    ],
)
def test_threshold_evaluator_rejects_non_finite_numeric_rows(
    field: str,
    value: float,
) -> None:
    row = _latency_row("non-finite", "tool-a", 100.0, tool_ts_start=1.0)
    row[field] = value

    with pytest.raises(ValueError, match=rf"{field!r} must be finite"):
        evaluate_latency_thresholds([row], thresholds_ms=[50.0])


def test_threshold_evaluator_rejects_tool_end_before_start() -> None:
    row = _latency_row("bad-time-range", "tool-a", 100.0, tool_ts_start=5.0)
    row["tool_ts_end"] = 4.999

    with pytest.raises(ValueError, match="tool_ts_end < tool_ts_start"):
        evaluate_latency_thresholds([row], thresholds_ms=[50.0])


@pytest.mark.parametrize("threshold", [float("nan"), float("inf")])
def test_threshold_evaluator_rejects_non_finite_thresholds(threshold: float) -> None:
    rows = [_latency_row("good", "tool-a", 100.0, tool_ts_start=1.0)]

    with pytest.raises(ValueError, match="thresholds must be finite and positive"):
        evaluate_latency_thresholds(rows, thresholds_ms=[threshold])


@pytest.mark.parametrize("cutoff", [float("nan"), float("inf")])
def test_threshold_evaluator_rejects_non_finite_probability_cutoff(cutoff: float) -> None:
    rows = [_latency_row("good", "tool-a", 100.0, tool_ts_start=1.0)]

    with pytest.raises(ValueError, match="probability_cutoff must be finite"):
        evaluate_latency_thresholds(
            rows,
            thresholds_ms=[50.0],
            probability_cutoff=cutoff,
        )


def test_evaluate_tool_latency_thresholds_cli_writes_summary_and_decisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    latencies_path = tmp_path / "latencies.jsonl"
    summary_path = tmp_path / "summary.json"
    decisions_path = tmp_path / "decisions.jsonl"
    _write_jsonl(
        latencies_path,
        [
            _latency_row("first", "fixture-tool", 100.0, tool_ts_start=1.0),
            _latency_row("second", "fixture-tool", 300.0, tool_ts_start=2.0),
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_tool_latency_thresholds.py",
            "--latencies",
            str(latencies_path),
            "--thresholds-ms",
            "200",
            "--probability-cutoff",
            "0.5",
            "--output",
            str(summary_path),
            "--decisions-output",
            str(decisions_path),
        ],
    )

    evaluate_thresholds_main()

    assert f"Evaluated 2 decisions over 2 rows -> {summary_path}" in capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    decisions = _read_jsonl(decisions_path)
    assert "decisions" not in summary
    assert summary["row_count"] == 2
    assert summary["decision_count"] == 2
    assert summary["metrics_by_threshold"]["200.0"]["evaluated_count"] == 1

    assert decisions == [
        {
            "history_count": 0,
            "label_exceeds_threshold": False,
            "latency_ms": 100.0,
            "prediction_source": "cold_start",
            "predicted_exceeds_threshold": None,
            "probability_exceeds_threshold": None,
            "sample_id": "first",
            "threshold_ms": 200.0,
            "tool_name": "fixture-tool",
            "tool_ts_start": 1.0,
        },
        {
            "history_count": 1,
            "label_exceeds_threshold": True,
            "latency_ms": 300.0,
            "prediction_source": "tool_history",
            "predicted_exceeds_threshold": False,
            "probability_exceeds_threshold": 0.0,
            "sample_id": "second",
            "threshold_ms": 200.0,
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


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
