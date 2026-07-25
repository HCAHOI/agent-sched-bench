from __future__ import annotations

from dataclasses import replace

import pytest

from scripts.evaluation.evaluate_clause_resource_kb import (
    ProxyCall,
    ScoredRow,
    _latency_false_negative_modes,
    _metrics,
    _proxy_observations,
    _validate_partition,
)
from tool_resource.labels import ResourceCallSample


def _row(
    truth: bool,
    prediction: bool | None,
    target: str = "cpu_heavy_2cores",
) -> ScoredRow:
    return ScoredRow(
        policy="baseline",
        target=target,
        sample_id="sample",
        task_id="owner__repo-1",
        repo="owner__repo",
        command="pytest",
        truth=truth,
        prediction=prediction,
        layer="public",
        key_kind="bin",
        evidence_count=1,
        command_structure="single",
        mapping_evidence="bin_exact",
    )


def test_metrics_keep_unknown_out_of_confusion_but_in_coverage() -> None:
    metrics = _metrics(
        [_row(True, True), _row(False, False), _row(True, None), _row(False, True)]
    )
    assert metrics == {
        "eligible": 4,
        "positive": 2,
        "prevalence": 0.5,
        "known": 3,
        "unknown": 1,
        "coverage": 0.75,
        "balanced_accuracy": 0.75,
        "recall": 1.0,
        "precision": 0.5,
        "tp": 1,
        "tn": 1,
        "fp": 1,
        "fn": 0,
    }


def test_partition_overlap_fails_closed() -> None:
    with pytest.raises(ValueError, match="fit/eval task overlap"):
        _validate_partition(["owner__repo-1"], ["owner__repo-1"])


def test_latency_false_negative_mechanism_counts_short_segment_sum() -> None:
    sample = ResourceCallSample(
        sample_id="sample",
        source_trace="trace.jsonl",
        task_id="owner__repo-1",
        agent_id="agent",
        instance_id="instance",
        iteration=0,
        action_id="action",
        tool_name="exec",
        tool_args={"command": "a; b"},
        tool_ts_start=0.0,
        tool_ts_end=8.0,
        censored=False,
        cpu_core_seconds=None,
        cpu_core_seconds_eligible=False,
        cpu_core_seconds_kind="missing",
        peak_cpu_cores=None,
        peak_cpu_cores_eligible=False,
        peak_cpu_clipped_sample_count=0,
        peak_memory_mb=None,
        peak_memory_mb_eligible=False,
        ambient_memory_mb=None,
        ambient_memory_mb_eligible=False,
        memory_window_sample_count=0,
        ambient_before_mb=None,
        ambient_before_age_s=None,
    )
    call = ProxyCall(
        sample=sample,
        command="a; b",
        clauses=({"bin": "a", "argv": ["a"]}, {"bin": "b", "argv": ["b"]}),
        mapping_evidence="bin_exact",
        segment_times_ms=((0.0, 2000.0), (2000.0, 6000.0)),
    )
    row = _row(True, False, "latency_long_5000ms")
    assert _latency_false_negative_modes(
        [row], [call], "latency_long_5000ms"
    ) == {
        "single_segment_exceeds": 0,
        "short_sequential_segments_sum_exceeds": 1,
        "pipeline_overlap_unresolved": 0,
        "command_envelope_only": 0,
        "mapping_unavailable": 0,
    }

    pipeline_call = ProxyCall(
        sample=sample,
        command="a | b",
        clauses=(
            {"bin": "a", "argv": ["a"], "in_pipe": True},
            {"bin": "b", "argv": ["b"], "in_pipe": True},
        ),
        mapping_evidence="bin_exact",
        segment_times_ms=call.segment_times_ms,
    )
    assert _latency_false_negative_modes(
        [row], [pipeline_call], "latency_long_5000ms"
    )["pipeline_overlap_unresolved"] == 1

    call_level_sample = replace(
        sample,
        peak_cpu_cores=3.0,
        peak_cpu_cores_eligible=True,
    )
    leading_cd_call = ProxyCall(
        sample=call_level_sample,
        command="cd /tmp && a",
        clauses=(
            {"bin": "cd", "argv": ["cd", "/tmp"]},
            {"bin": "a", "argv": ["a"]},
        ),
        mapping_evidence="bin_exact",
        segment_times_ms=((0.0, 1.0), (1.0, 6000.0)),
    )
    observation = _proxy_observations(
        leading_cd_call, "owner__repo", exclude_leading_cd=True
    )[0]
    assert observation.ts_end == call_level_sample.tool_ts_end
