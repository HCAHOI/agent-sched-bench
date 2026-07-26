from __future__ import annotations

import argparse
from dataclasses import replace

import pytest

from scripts.evaluation.evaluate_clause_latency_buckets import (
    ProxyCall,
    ScoredRow,
    _metrics,
    _parse_bucket_edges,
    _proxy_latency_observations,
    _validate_partition,
    evaluate,
)
from tool_resource.labels import ResourceCallSample
from tool_resource.runtime_kb import LatencyBuckets


def _sample(
    sample_id: str,
    task_id: str,
    *,
    start: float,
    end: float,
    command: str,
    censored: bool = False,
) -> ResourceCallSample:
    return ResourceCallSample(
        sample_id=sample_id,
        source_trace="trace.jsonl",
        task_id=task_id,
        agent_id="agent",
        instance_id="instance",
        iteration=0,
        action_id=sample_id,
        tool_name="exec",
        tool_args={"command": command},
        tool_ts_start=start,
        tool_ts_end=end,
        censored=censored,
        cpu_core_seconds=999.0,
        cpu_core_seconds_eligible=True,
        cpu_core_seconds_kind="legacy",
        peak_cpu_cores=99.0,
        peak_cpu_cores_eligible=True,
        peak_cpu_clipped_sample_count=1,
        peak_memory_mb=9999.0,
        peak_memory_mb_eligible=True,
        ambient_memory_mb=None,
        ambient_memory_mb_eligible=False,
        memory_window_sample_count=1,
        ambient_before_mb=None,
        ambient_before_age_s=None,
    )


def _call(
    sample_id: str,
    task_id: str,
    *,
    start: float,
    end: float,
    command: str,
    clauses: tuple[dict, ...],
    segments: tuple[tuple[float, float], ...],
) -> ProxyCall:
    return ProxyCall(
        sample=_sample(sample_id, task_id, start=start, end=end, command=command),
        command=command,
        clauses=clauses,
        parse_failed=False,
        mapping_evidence="bin_exact",
        segment_times_ms=segments,
    )


def _row(label: int, prediction: int | None, reason: str | None = None) -> ScoredRow:
    return ScoredRow(
        sample_id="sample",
        task_id="owner__repo-1",
        repo="owner__repo",
        command="pytest",
        label_bucket=label,
        predicted_bucket=prediction,
        layer=None if prediction is None else "public",
        key_kind=None if prediction is None else "bin",
        evidence_count=0 if prediction is None else 1,
        unavailable_reason=reason,
        mapping_evidence="bin_exact",
    )


def test_metrics_keep_uncomposed_rows_in_coverage_and_bucket_confusion() -> None:
    metrics = _metrics(
        [_row(0, 0), _row(1, 0), _row(1, 2), _row(2, None, "compound")],
        3,
    )

    assert metrics["eligible"] == 4
    assert metrics["known"] == 3
    assert metrics["coverage"] == 0.75
    assert metrics["accuracy"] == pytest.approx(1 / 3)
    assert metrics["mean_abs_bucket_error"] == pytest.approx(2 / 3)
    assert metrics["underestimate_rate"] == pytest.approx(1 / 3)
    assert metrics["overestimate_rate"] == pytest.approx(1 / 3)
    assert metrics["confusion"] == [[1, 0, 0], [1, 0, 1], [0, 0, 0]]
    assert metrics["unavailable_reasons"] == {"compound": 1}


def test_partition_overlap_fails_closed() -> None:
    with pytest.raises(ValueError, match="fit/eval task overlap"):
        _validate_partition(["owner__repo-1"], ["owner__repo-1"])


def test_proxy_observation_excludes_cd_and_never_attaches_cpu_or_rss() -> None:
    call = _call(
        "fit",
        "owner__repo-1",
        start=10.0,
        end=20.0,
        command="cd /tmp && a",
        clauses=(
            {"bin": "cd", "argv": ["cd", "/tmp"]},
            {"bin": "a", "argv": ["a"]},
        ),
        segments=((0.0, 1.0), (1.0, 6000.0)),
    )

    observation = _proxy_latency_observations(call, "owner__repo")[0]

    assert observation.bin == "a"
    assert observation.latency_ms == 5999.0
    assert observation.ts_end == pytest.approx(20.0)
    assert observation.peak_cpu_cores is None
    assert observation.sampled_peak_rss_mb is None


def test_overlapping_call_cannot_see_clause_until_outer_call_finishes() -> None:
    fit_calls = [
        _call(
            "fit",
            "fit__repo-1",
            start=-2.0,
            end=-1.95,
            command="a",
            clauses=({"bin": "a", "argv": ["a"]},),
            segments=((0.0, 50.0),),
        )
    ]
    eval_calls = [
        _call(
            "long",
            "eval__repo-1",
            start=0.0,
            end=10.0,
            command="a",
            clauses=({"bin": "a", "argv": ["a"]},),
            segments=((0.0, 1500.0),),
        ),
        _call(
            "overlap",
            "eval__repo-1",
            start=2.0,
            end=2.05,
            command="a",
            clauses=({"bin": "a", "argv": ["a"]},),
            segments=((0.0, 50.0),),
        ),
    ]

    _, rows = evaluate(
        fit_calls,
        eval_calls,
        LatencyBuckets((1000.0,)),
        {"fixture": True},
    )

    overlap = {row.sample_id: row for row in rows}["overlap"]
    assert overlap.predicted_bucket == 0
    assert overlap.layer == "public"


def test_evaluator_uses_right_open_labels_and_leaves_compounds_uncomposed() -> None:
    fit_calls = [
        _call(
            "fit",
            "fit__repo-1",
            start=0.0,
            end=0.05,
            command="a",
            clauses=({"bin": "a", "argv": ["a"]},),
            segments=((0.0, 50.0),),
        )
    ]
    eval_calls = [
        _call(
            "edge",
            "eval__repo-1",
            start=10.0,
            end=10.125,
            command="a",
            clauses=({"bin": "a", "argv": ["a"]},),
            segments=((0.0, 125.0),),
        ),
        _call(
            "compound",
            "eval__repo-1",
            start=20.0,
            end=20.2,
            command="a; b",
            clauses=(
                {"bin": "a", "argv": ["a"]},
                {"bin": "b", "argv": ["b"]},
            ),
            segments=((0.0, 100.0), (100.0, 200.0)),
        ),
    ]

    result, rows = evaluate(
        fit_calls,
        eval_calls,
        LatencyBuckets((125.0,)),
        {"fixture": True},
    )

    by_id = {row.sample_id: row for row in rows}
    assert by_id["edge"].label_bucket == 1
    assert by_id["edge"].predicted_bucket == 0
    assert by_id["compound"].predicted_bucket is None
    assert by_id["compound"].unavailable_reason == "compound_command_uncomposed"
    assert result["claim_bearing"] is False
    assert result["bucket_intervals"] == [
        {"bucket_id": 0, "lower_ms": 0.0, "upper_ms": 125.0},
        {"bucket_id": 1, "lower_ms": 125.0, "upper_ms": None},
    ]
    assert "balanced_accuracy" not in result["metrics"]


def test_cli_bucket_edges_are_explicit_and_validated() -> None:
    assert _parse_bucket_edges("100,1000").edges_ms == (100.0, 1000.0)
    with pytest.raises(argparse.ArgumentTypeError, match="strictly increasing"):
        _parse_bucket_edges("100,100")


def test_censored_proxy_call_produces_no_latency_observation() -> None:
    call = _call(
        "censored",
        "owner__repo-1",
        start=0.0,
        end=1.0,
        command="a",
        clauses=({"bin": "a", "argv": ["a"]},),
        segments=((0.0, 1000.0),),
    )
    call = replace(call, sample=replace(call.sample, censored=True))

    assert _proxy_latency_observations(call, "owner__repo") == []
    assert (
        _proxy_latency_observations(
            replace(
                call, sample=replace(call.sample, censored=False), parse_failed=True
            ),
            "owner__repo",
        )
        == []
    )
