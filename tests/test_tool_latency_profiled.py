from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.evaluate_profiled_latency_thresholds import main as evaluate_profiled_main
from trace_collect.tool_latency_profiled import (
    build_latency_prior,
    evaluate_profiled_latency_thresholds,
)
from trace_collect.tool_latency_threshold import evaluate_latency_thresholds


def _profile_rows() -> list[dict[str, object]]:
    # Per-tool prior: probe -> [900, 900]; other -> [50]; global -> [50, 900, 900].
    return [
        _latency_row("p1", "probe", 900.0, tool_ts_start=0.0, source_trace="trace-p"),
        _latency_row("p2", "probe", 900.0, tool_ts_start=1.0, source_trace="trace-p"),
        _latency_row("p3", "other", 50.0, tool_ts_start=2.0, source_trace="trace-p"),
    ]


def test_prior_only_uses_tool_prior_with_global_fallback_and_no_cold_start() -> None:
    eval_rows = [
        _latency_row("probe-hit", "probe", 900.0, tool_ts_start=0.0),
        _latency_row("unseen", "newtool", 50.0, tool_ts_start=1.0),
    ]

    summary = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=_profile_rows(),
        thresholds_ms=[100.0],
        predictor="prior_only",
    )

    decisions = {row["sample_id"]: row for row in summary["decisions"]}
    assert summary["profile_row_count"] == 3
    assert summary["profile_tool_count"] == 2

    probe_hit = decisions["probe-hit"]
    assert probe_hit["prior_source"] == "prior_tool"
    assert probe_hit["prior_count"] == 2
    assert probe_hit["probability_exceeds_threshold"] == 1.0
    assert probe_hit["predicted_exceeds_threshold"] is True
    assert probe_hit["online_source"] is None

    unseen = decisions["unseen"]
    assert unseen["prior_source"] == "prior_global"
    assert unseen["prior_count"] == 3
    assert unseen["probability_exceeds_threshold"] == pytest.approx(2 / 3)
    assert unseen["predicted_exceeds_threshold"] is True
    assert unseen["label_exceeds_threshold"] is False

    metrics = summary["metrics_by_threshold"]["100.0"]
    assert metrics["decided_count"] == 2
    assert metrics["cold_start_count"] == 0
    assert metrics["abstain_count"] == 0


def test_online_only_matches_unprofiled_threshold_eval() -> None:
    eval_rows = [
        _latency_row("seed-global", "seed-tool", 100.0, tool_ts_start=0.0),
        _latency_row("first-zircon", "zircon-saw", 900.0, tool_ts_start=1.0),
        _latency_row("second-zircon", "zircon-saw", 700.0, tool_ts_start=2.0),
    ]

    profiled = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=_profile_rows(),
        thresholds_ms=[500.0],
        predictor="online_only",
    )
    baseline = evaluate_latency_thresholds(eval_rows, thresholds_ms=[500.0])

    profiled_by_id = {row["sample_id"]: row for row in profiled["decisions"]}
    for base_row in baseline["decisions"]:
        row = profiled_by_id[base_row["sample_id"]]
        assert row["predicted_exceeds_threshold"] == base_row["predicted_exceeds_threshold"]
        assert row["probability_exceeds_threshold"] == base_row["probability_exceeds_threshold"]
        assert row["label_exceeds_threshold"] == base_row["label_exceeds_threshold"]
        assert row["online_source"] == base_row["prediction_source"]
        assert row["prior_source"] is None


def test_blended_pools_prior_and_online_counts() -> None:
    profile_rows = [
        _latency_row("p1", "probe", 900.0, tool_ts_start=0.0, source_trace="trace-p"),
        _latency_row("p2", "probe", 900.0, tool_ts_start=1.0, source_trace="trace-p"),
    ]
    eval_rows = [
        _latency_row("no-online", "probe", 50.0, tool_ts_start=0.0),
        _latency_row("one-online", "probe", 900.0, tool_ts_start=1.0),
    ]

    summary = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=profile_rows,
        thresholds_ms=[100.0],
        predictor="blended",
        probability_cutoff=0.6,
    )

    decisions = {row["sample_id"]: row for row in summary["decisions"]}
    # Prior alone: 2 pseudo-observations at survival 1.0.
    no_online = decisions["no-online"]
    assert no_online["probability_exceeds_threshold"] == 1.0
    assert no_online["effective_count"] == 2.0
    assert no_online["predicted_exceeds_threshold"] is True
    assert no_online["online_count"] == 0
    # Pooled with one online miss (50ms): (2 * 1.0 + 0) / 3.
    one_online = decisions["one-online"]
    assert one_online["probability_exceeds_threshold"] == pytest.approx(2 / 3)
    assert one_online["effective_count"] == 3.0
    assert one_online["predicted_exceeds_threshold"] is True

    # No cold starts for blended even with empty online history.
    metrics = summary["metrics_by_threshold"]["100.0"]
    assert metrics["cold_start_count"] == 0


def test_blended_prior_strength_caps_prior_weight() -> None:
    profile_rows = [
        _latency_row("p1", "probe", 900.0, tool_ts_start=0.0, source_trace="trace-p"),
        _latency_row("p2", "probe", 900.0, tool_ts_start=1.0, source_trace="trace-p"),
    ]
    eval_rows = [
        _latency_row("no-online", "probe", 50.0, tool_ts_start=0.0),
        _latency_row("one-online", "probe", 900.0, tool_ts_start=1.0),
    ]

    summary = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=profile_rows,
        thresholds_ms=[100.0],
        predictor="blended",
        prior_strength=1.0,
        probability_cutoff=0.6,
    )

    decisions = {row["sample_id"]: row for row in summary["decisions"]}
    # Prior scaled to 1 pseudo-observation: (1 * 1.0 + 0) / (1 + 1) = 0.5 < 0.6.
    one_online = decisions["one-online"]
    assert one_online["probability_exceeds_threshold"] == 0.5
    assert one_online["effective_count"] == 2.0
    assert one_online["predicted_exceeds_threshold"] is False


def test_wilson_abstain_band_abstains_on_thin_evidence() -> None:
    thin_profile = [
        _latency_row("p1", "probe", 900.0, tool_ts_start=0.0, source_trace="trace-p"),
        _latency_row("p2", "probe", 900.0, tool_ts_start=1.0, source_trace="trace-p"),
    ]
    rich_profile = [
        _latency_row(f"p{i}", "probe", 900.0, tool_ts_start=float(i), source_trace="trace-p")
        for i in range(20)
    ]
    eval_rows = [_latency_row("scored", "probe", 900.0, tool_ts_start=0.0)]

    thin = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=thin_profile,
        thresholds_ms=[100.0],
        predictor="prior_only",
        abstain_confidence=0.95,
    )
    rich = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=rich_profile,
        thresholds_ms=[100.0],
        predictor="prior_only",
        abstain_confidence=0.95,
    )

    # Survival estimate is 1.0 both times; only the evidence differs.
    (thin_decision,) = thin["decisions"]
    assert thin_decision["probability_exceeds_threshold"] == 1.0
    assert thin_decision["predicted_exceeds_threshold"] is None
    assert thin_decision["abstained"] is True
    assert thin_decision["ci_low"] < 0.5 <= thin_decision["ci_high"]
    thin_metrics = thin["metrics_by_threshold"]["100.0"]
    assert thin_metrics["abstain_count"] == 1
    assert thin_metrics["abstain_rate"] == 1.0
    assert thin_metrics["decided_count"] == 0

    (rich_decision,) = rich["decisions"]
    assert rich_decision["predicted_exceeds_threshold"] is True
    assert rich_decision["abstained"] is False
    assert rich_decision["ci_low"] >= 0.5


def test_fractional_prior_strength_yields_valid_wilson_band() -> None:
    profile_rows = [
        _latency_row("p1", "probe", 900.0, tool_ts_start=0.0, source_trace="trace-p"),
        _latency_row("p2", "probe", 900.0, tool_ts_start=1.0, source_trace="trace-p"),
    ]
    eval_rows = [_latency_row("scored", "probe", 900.0, tool_ts_start=0.0)]

    summary = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=profile_rows,
        thresholds_ms=[100.0],
        predictor="blended",
        prior_strength=0.5,
        abstain_confidence=0.95,
    )

    (decision,) = summary["decisions"]
    assert decision["probability_exceeds_threshold"] == 1.0
    assert decision["effective_count"] == 0.5
    assert 0.0 <= decision["ci_low"] <= decision["ci_high"] <= 1.0
    # Half a pseudo-observation is far too thin to clear the cutoff.
    assert decision["abstained"] is True


def test_wilson_band_predicts_false_when_upper_bound_misses_cutoff() -> None:
    profile_rows = [
        _latency_row(f"p{i}", "probe", 50.0, tool_ts_start=float(i), source_trace="trace-p")
        for i in range(20)
    ]
    eval_rows = [_latency_row("scored", "probe", 900.0, tool_ts_start=0.0)]

    summary = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=profile_rows,
        thresholds_ms=[100.0],
        predictor="prior_only",
        abstain_confidence=0.95,
    )

    (decision,) = summary["decisions"]
    assert decision["probability_exceeds_threshold"] == 0.0
    assert decision["ci_high"] < 0.5
    assert decision["predicted_exceeds_threshold"] is False
    assert decision["abstained"] is False


def test_min_tool_history_gates_prior_and_online_fallback_symmetrically() -> None:
    profile_rows = [
        _latency_row("p1", "probe", 900.0, tool_ts_start=0.0, source_trace="trace-p"),
        _latency_row("p2", "other", 50.0, tool_ts_start=1.0, source_trace="trace-p"),
        _latency_row("p3", "other", 50.0, tool_ts_start=2.0, source_trace="trace-p"),
    ]
    eval_rows = [
        _latency_row("first", "probe", 900.0, tool_ts_start=0.0),
        _latency_row("second", "probe", 900.0, tool_ts_start=1.0),
    ]

    summary = evaluate_profiled_latency_thresholds(
        eval_rows,
        profile_rows=profile_rows,
        thresholds_ms=[100.0],
        predictor="blended",
        min_tool_history=2,
    )

    decisions = {row["sample_id"]: row for row in summary["decisions"]}
    # Prior has one probe sample (< 2) and online has at most one (< 2):
    # both sides fall back to their global distributions together.
    assert decisions["first"]["prior_source"] == "prior_global"
    assert decisions["first"]["online_source"] == "cold_start"
    assert decisions["second"]["prior_source"] == "prior_global"
    assert decisions["second"]["online_source"] == "global_history"


def test_empty_eval_rows_are_rejected() -> None:
    with pytest.raises(ValueError, match="no latency rows supplied"):
        evaluate_profiled_latency_thresholds(
            [],
            profile_rows=_profile_rows(),
            thresholds_ms=[100.0],
            predictor="prior_only",
        )


def test_shared_traces_between_profile_and_eval_are_rejected() -> None:
    eval_rows = [
        _latency_row("leak", "probe", 900.0, tool_ts_start=0.0, source_trace="trace-p"),
    ]

    with pytest.raises(ValueError, match="disjoint traces.*trace-p"):
        evaluate_profiled_latency_thresholds(
            eval_rows,
            profile_rows=_profile_rows(),
            thresholds_ms=[100.0],
            predictor="prior_only",
        )


def test_empty_profile_rows_are_rejected() -> None:
    with pytest.raises(ValueError, match="empty latency prior"):
        build_latency_prior([])


def test_profiled_eval_rejects_invalid_parameters() -> None:
    eval_rows = [_latency_row("good", "probe", 100.0, tool_ts_start=0.0)]

    with pytest.raises(ValueError, match="unknown predictor"):
        evaluate_profiled_latency_thresholds(
            eval_rows,
            profile_rows=_profile_rows(),
            thresholds_ms=[100.0],
            predictor="oracle",
        )
    for strength in [0.0, -1.0, float("inf"), float("nan")]:
        with pytest.raises(ValueError, match="prior_strength must be finite and positive"):
            evaluate_profiled_latency_thresholds(
                eval_rows,
                profile_rows=_profile_rows(),
                thresholds_ms=[100.0],
                predictor="blended",
                prior_strength=strength,
            )
    for confidence in [0.0, 1.0, 1.5, float("nan")]:
        with pytest.raises(ValueError, match="abstain_confidence must be finite"):
            evaluate_profiled_latency_thresholds(
                eval_rows,
                profile_rows=_profile_rows(),
                thresholds_ms=[100.0],
                predictor="prior_only",
                abstain_confidence=confidence,
            )


def test_evaluate_profiled_cli_compares_predictors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path = tmp_path / "profile.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    summary_path = tmp_path / "summary.json"
    decisions_path = tmp_path / "decisions.jsonl"
    _write_jsonl(profile_path, _profile_rows())
    _write_jsonl(
        eval_path,
        [
            _latency_row("first", "probe", 900.0, tool_ts_start=1.0),
            _latency_row("second", "probe", 50.0, tool_ts_start=2.0),
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_profiled_latency_thresholds.py",
            "--profile-latencies",
            str(profile_path),
            "--eval-latencies",
            str(eval_path),
            "--thresholds-ms",
            "100",
            "--output",
            str(summary_path),
            "--decisions-output",
            str(decisions_path),
        ],
    )

    evaluate_profiled_main()

    assert f"Evaluated 3 predictors, 6 decisions -> {summary_path}" in capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert sorted(summary["by_predictor"]) == ["blended", "online_only", "prior_only"]
    for predictor_summary in summary["by_predictor"].values():
        assert "decisions" not in predictor_summary
        assert predictor_summary["row_count"] == 2

    decisions = _read_jsonl(decisions_path)
    assert len(decisions) == 6
    assert {row["predictor"] for row in decisions} == {
        "prior_only",
        "online_only",
        "blended",
    }
    # online_only has one cold start; the profiled predictors decide every row.
    online_first = next(
        row
        for row in decisions
        if row["predictor"] == "online_only" and row["sample_id"] == "first"
    )
    assert online_first["predicted_exceeds_threshold"] is None
    prior_first = next(
        row
        for row in decisions
        if row["predictor"] == "prior_only" and row["sample_id"] == "first"
    )
    assert prior_first["predicted_exceeds_threshold"] is True


def _latency_row(
    sample_id: str,
    tool_name: str,
    latency_ms: float,
    *,
    tool_ts_start: float,
    source_trace: str = "trace-e",
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
