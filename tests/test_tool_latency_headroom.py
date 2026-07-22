from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.exploration.analyze_utility_band_headroom import main as headroom_main
from trace_collect.tool_latency_headroom import analyze_utility_headroom
from trace_collect.tool_latency_utility_clock import trigger_policy_utility_ms


def test_trigger_policy_utility_matches_metric_boundaries() -> None:
    assert _utility(latency_ms=100.0, trigger_ms=100.0) == 0.0
    assert _utility(latency_ms=150.0, trigger_ms=100.0) == 0.0
    assert _utility(latency_ms=200.0, trigger_ms=100.0) == 100.0
    assert _utility(latency_ms=50.0, trigger_ms=0.0) == -50.0
    assert _utility(latency_ms=150.0, trigger_ms=0.0) == 100.0


def test_headroom_identity_and_policy_capture() -> None:
    rows = [
        _row("short", "task-a", latency_ms=50.0, mean_trigger_ms=100.0),
        _row("band", "task-b", latency_ms=150.0, mean_trigger_ms=0.0),
        _row("edge", "task-c", latency_ms=200.0, mean_trigger_ms=0.0),
        _row("tail", "task-d", latency_ms=250.0, mean_trigger_ms=0.0),
    ]

    result = _analyze(rows)

    point = result["points"]["100.0"]
    assert point["positive_count"] == 3
    assert point["band_count"] == 1
    assert point["far_tail_count"] == 2
    assert point["oracle_ms"] == 300.0
    assert point["deadline_headroom_ms"] == 100.0
    assert point["rho_headroom_over_oracle"] == pytest.approx(1.0 / 3.0)
    assert point["identity_residual_ms"] == 0.0
    assert point["policies"]["deadline_only"]["net_saved_ms"] == 200.0
    mean = point["policies"]["mean_hazard"]
    assert mean["net_saved_ms"] == 300.0
    assert mean["delta_vs_deadline_ms"] == 100.0
    assert mean["captured_fraction_of_deadline_headroom"] == 1.0
    robust = point["policies"]["robust_clock"]
    assert robust["delta_vs_deadline_ms"] == 0.0
    assert robust["captured_fraction_of_deadline_headroom"] == 0.0


def test_task_cluster_bootstrap_is_reproducible() -> None:
    rows = [
        _row("short", "task-a", latency_ms=50.0, mean_trigger_ms=100.0),
        _row("band", "task-b", latency_ms=150.0, mean_trigger_ms=0.0),
        _row("tail", "task-c", latency_ms=250.0, mean_trigger_ms=0.0),
    ]

    first = _analyze(rows)
    second = _analyze(rows)

    assert (
        first["points"]["100.0"]["task_cluster_bootstrap"]
        == second["points"]["100.0"]["task_cluster_bootstrap"]
    )


@pytest.mark.parametrize(
    "latency_ms",
    [200.0, 50.0],
    ids=["zero-band-headroom", "no-positives"],
)
def test_headroom_reports_undefined_ratio_intervals(
    latency_ms: float,
) -> None:
    rows = [
        _row(
            "edge-or-short",
            "task-a",
            latency_ms=latency_ms,
            mean_trigger_ms=100.0,
        )
    ]
    result = _analyze(rows)

    point = result["points"]["100.0"]
    assert point["deadline_headroom_ms"] == 0.0
    assert (
        point["policies"]["mean_hazard"]["captured_fraction_of_deadline_headroom"]
        is None
    )
    interval = point["task_cluster_bootstrap"]["mean_hazard_captured_fraction"]
    assert interval == {
        "lower": None,
        "upper": None,
        "valid_replicates": 0,
    }
    if point["positive_count"] == 0:
        assert point["rho_headroom_over_oracle"] is None
        assert point["task_cluster_bootstrap"]["rho_headroom_over_oracle"] == {
            "lower": None,
            "upper": None,
            "valid_replicates": 0,
        }


def test_headroom_rejects_inconsistent_label() -> None:
    row = _row("bad", "task-a", latency_ms=150.0, mean_trigger_ms=0.0)
    row["label_exceeds_threshold"] = False

    with pytest.raises(ValueError, match="inconsistent label"):
        _analyze([row])


def test_headroom_rejects_duplicate_sample_cost() -> None:
    row = _row("duplicate", "task-a", latency_ms=150.0, mean_trigger_ms=0.0)

    with pytest.raises(ValueError, match="duplicate sample/cost"):
        _analyze([row, dict(row)])


def test_headroom_cli_writes_full_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    decisions_path = tmp_path / "decisions.jsonl"
    output_path = tmp_path / "headroom.json"
    decisions_path.write_text(
        json.dumps(
            _row("band", "task-a", latency_ms=150.0, mean_trigger_ms=0.0),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_utility_band_headroom.py",
            str(decisions_path),
            "--expected-costs-ms",
            "100",
            "--bootstrap-replicates",
            "100",
            "--bootstrap-seed",
            "7",
            "--confidence-level",
            "0.95",
            "--output",
            str(output_path),
        ],
    )

    headroom_main()

    assert "Analyzed 1 samples across 1 cost points" in capsys.readouterr().out
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["bootstrap"]["replicates"] == 100
    assert result["bootstrap"]["seed"] == 7
    assert result["input_paths"] == [str(decisions_path.resolve())]


def _utility(*, latency_ms: float, trigger_ms: float) -> float:
    return trigger_policy_utility_ms(
        latency_ms,
        trigger_ms,
        threshold_ms=100.0,
        kv_cost_ms=100.0,
    )


def _analyze(rows: list[dict[str, object]]) -> dict[str, object]:
    return analyze_utility_headroom(
        rows,
        expected_costs_ms=[100.0],
        bootstrap_replicates=250,
        bootstrap_seed=11,
        confidence_level=0.95,
    )


def _row(
    sample_id: str,
    task_id: str,
    *,
    latency_ms: float,
    mean_trigger_ms: float,
) -> dict[str, object]:
    threshold_ms = 100.0
    return {
        "sample_id": sample_id,
        "task_id": task_id,
        "tool_name": "exec",
        "latency_ms": latency_ms,
        "kv_cost_ms": 100.0,
        "threshold_ms": threshold_ms,
        "label_exceeds_threshold": latency_ms > threshold_ms,
        "deadline_trigger_ms": threshold_ms,
        "mean_hazard_trigger_ms": mean_trigger_ms,
        "robust_trigger_ms": threshold_ms,
    }
