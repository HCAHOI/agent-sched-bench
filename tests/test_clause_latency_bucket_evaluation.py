from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scripts.evaluation.evaluate_clause_latency_buckets import (
    ProxyCall,
    ScoredRow,
    _fit_public_kbs,
    _metrics,
    _parser,
    _proxy_latency_observations,
    _score,
    _serialize_virtual_deployment,
    _trace_finalization_timestamp,
    _validate_partition,
    evaluate,
    evaluate_representation,
)
from tests.test_tool_resource_services import (
    _DirectTransport,
    _FakeCollector,
    _open_run,
    _open_trace,
    _run_call,
)
from tool_resource.labels import ResourceCallSample
from tool_resource.resource_agentd import ResourceService
from tool_resource.runtime_kb import ClauseResourceKB
from tool_resource.store import ObservationStore
from tool_resource.telemetryd import TelemetryService
from trace_collect.trace_data import TraceData


def _sample(
    sample_id: str,
    task_id: str,
    *,
    start: float,
    end: float,
    command: str,
    censored: bool = False,
    source_trace: str = "trace.jsonl",
) -> ResourceCallSample:
    return ResourceCallSample(
        sample_id=sample_id,
        source_trace=source_trace,
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
    trace_finalized_at: float,
    source_trace: str = "trace.jsonl",
) -> ProxyCall:
    return ProxyCall(
        sample=_sample(
            sample_id,
            task_id,
            start=start,
            end=end,
            command=command,
            source_trace=source_trace,
        ),
        command=command,
        clauses=clauses,
        parse_failed=False,
        mapping_evidence="bin_exact",
        segment_times_ms=segments,
        trace_finalized_at=trace_finalized_at,
    )


def _row(label: int, prediction: int | None, reason: str | None = None) -> ScoredRow:
    probability = None
    if prediction is not None:
        probability = tuple(1.0 if bucket == prediction else 0.0 for bucket in range(9))
    return ScoredRow(
        sample_id="sample",
        task_id="owner__repo-1",
        repo="owner__repo",
        command="pytest",
        label_bucket=label,
        probability_by_bucket=probability,
        layer=None if prediction is None else "public",
        key_kind=None if prediction is None else "bin",
        evidence_count=0 if prediction is None else 1,
        fallback_path=None if prediction is None else ("public:bin",),
        unavailable_reason=reason,
        mapping_evidence="bin_exact",
    )


def _envelope(
    observation_id: str,
    *,
    scope: str,
    command: str,
    end: float,
    latency_ms: float,
) -> dict[str, Any]:
    argv = command.split()
    return {
        "observation_id": observation_id,
        "run_id": "seed",
        "trace_id": "seed-trace",
        "call_id": observation_id,
        "workspace_scope": scope,
        "canonicalizer_version": "mvdan-sh-v1",
        "command_digest": "a" * 64,
        "observation_interval": {"start": end - latency_ms / 1000.0, "end": end},
        "normalized_measurements": [
            {
                "bin": argv[0],
                "argv": argv,
                "ts_start": end - latency_ms / 1000.0,
                "ts_end": end,
                "latency_ms": latency_ms,
                "availability": {"latency": "ok"},
            }
        ],
        "telemetry_eligible": True,
        "ingest_eligible": True,
        "rejection_reasons": [],
    }


def test_metrics_use_cumulative_pmf_at_each_fixed_boundary() -> None:
    cumulative = replace(
        _row(1, 0),
        probability_by_bucket=(0.4, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    metrics = _metrics(
        [
            _row(0, 0),
            cumulative,
            _row(0, 1),
            _row(1, 0),
            _row(2, None, "compound"),
        ],
    )

    assert metrics["by_boundary"][0] == {
        "boundary_ms": 500.0,
        "accuracy": 0.5,
        "eligible_examples": 4,
        "positive_count": 2,
        "positive_rate": 0.5,
        "true_positive": 1,
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
    }
    assert "mean_abs_bucket_error" not in metrics


def test_exact_bucket_metrics_use_lowest_argmax_on_ties() -> None:
    tie = replace(
        _row(1, 0),
        probability_by_bucket=(0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )

    exact = _metrics([tie, _row(2, 2), _row(2, None, "unavailable")])["exact_bucket"]

    assert exact == {
        "exact_bucket_accuracy": 0.5,
        "eligible_examples": 2,
    }


def test_partition_overlap_fails_closed() -> None:
    with pytest.raises(ValueError, match="fit/eval task overlap"):
        _validate_partition(["owner__repo-1"], ["owner__repo-1"])


def test_trace_finalization_requires_a_timestamped_final_summary(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    metadata = {"type": "trace_metadata", "trace_format_version": 5}
    trace_path.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                metadata,
                {"type": "summary", "agent_id": "agent", "ts": 12.5},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    assert _trace_finalization_timestamp(TraceData.load(trace_path)) == 12.5

    trace_path.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                metadata,
                {"type": "summary", "agent_id": "agent", "elapsed_s": 12.5},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="faithful CloseTrace ordering is unavailable"):
        _trace_finalization_timestamp(TraceData.load(trace_path))


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
        trace_finalized_at=21.0,
    )

    observation = _proxy_latency_observations(call, "owner__repo")[0]

    assert observation.bin == "a"
    assert observation.latency_ms == 5999.0
    assert observation.ts_end == pytest.approx(20.0)
    assert observation.peak_cpu_cores is None
    assert observation.sampled_peak_rss_mb is None


def test_trace_stays_unpublished_until_its_actual_finalization() -> None:
    fit_calls = [
        _call(
            "fit",
            "fit__repo-1",
            start=-2.0,
            end=-1.95,
            command="a",
            clauses=({"bin": "a", "argv": ["a"]},),
            segments=((0.0, 50.0),),
            trace_finalized_at=-1.0,
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
            trace_finalized_at=12.0,
        ),
        _call(
            "overlap",
            "eval__repo-1",
            start=2.0,
            end=2.05,
            command="a",
            clauses=({"bin": "a", "argv": ["a"]},),
            segments=((0.0, 50.0),),
            trace_finalized_at=12.0,
        ),
        _call(
            "while-trace-open",
            "eval__repo-1",
            start=10.5,
            end=10.55,
            command="a",
            clauses=({"bin": "a", "argv": ["a"]},),
            segments=((0.0, 50.0),),
            trace_finalized_at=14.0,
            source_trace="next.jsonl",
        ),
        _call(
            "after-trace-close",
            "eval__repo-1",
            start=13.0,
            end=13.05,
            command="a",
            clauses=({"bin": "a", "argv": ["a"]},),
            segments=((0.0, 50.0),),
            trace_finalized_at=14.0,
            source_trace="next.jsonl",
        ),
    ]

    _, rows = evaluate(
        fit_calls,
        eval_calls,
        {"fixture": True},
    )

    overlap = {row.sample_id: row for row in rows}["overlap:clause:0"]
    assert overlap.probability_by_bucket is not None
    assert overlap.probability_by_bucket[0] == 1.0
    assert overlap.layer == "public"
    while_trace_open = {row.sample_id: row for row in rows}["while-trace-open:clause:0"]
    assert while_trace_open.layer == "public"
    after_trace_close = {row.sample_id: row for row in rows}[
        "after-trace-close:clause:0"
    ]
    assert after_trace_close.layer == "repo"
    assert after_trace_close.evidence_count == 2


def test_fit_keeps_active_repository_history_out_of_public_and_local_state() -> None:
    fit_calls = [
        _call(
            "same-repo",
            "eval__repo-9",
            start=0.0,
            end=0.75,
            command="a",
            clauses=({"bin": "a", "argv": ["a"]},),
            segments=((0.0, 750.0),),
            trace_finalized_at=0.8,
            source_trace="same.jsonl",
        ),
        _call(
            "cross-repo",
            "other__repo-1",
            start=1.0,
            end=1.05,
            command="a",
            clauses=({"bin": "a", "argv": ["a"]},),
            segments=((0.0, 50.0),),
            trace_finalized_at=1.1,
            source_trace="cross.jsonl",
        ),
    ]
    eval_calls = [
        _call(
            "query",
            "eval__repo-1",
            start=10.0,
            end=10.05,
            command="a",
            clauses=({"bin": "a", "argv": ["a"]},),
            segments=((0.0, 50.0),),
            trace_finalized_at=10.1,
        )
    ]

    _, rows = evaluate(fit_calls, eval_calls, {"fixture": True})

    assert rows[0].layer == "public"
    assert rows[0].evidence_count == 1
    assert rows[0].probability_by_bucket is not None
    assert rows[0].probability_by_bucket[0] == 1.0


def test_representation_diagnostic_reuses_signature_without_repo_leakage() -> None:
    def call(
        sample_id: str,
        task_id: str,
        command: str,
        latency_ms: float,
        start: float,
        trace: str,
        closed_at: float,
    ) -> ProxyCall:
        argv = command.split()
        return _call(
            sample_id,
            task_id,
            start=start,
            end=start + latency_ms / 1000.0,
            command=command,
            clauses=({"bin": argv[0], "argv": argv},),
            segments=((0.0, latency_ms),),
            trace_finalized_at=closed_at,
            source_trace=trace,
        )

    fit_calls = [
        call("same", "eval__repo-9", "a x", 70000.0, -100.0, "same", -29.0),
        call("exact", "other__repo-1", "a x", 750.0, -10.0, "exact", -8.0),
        call(
            "dynamic",
            "other__repo-2",
            "a /tmp/run-1 101",
            750.0,
            -7.0,
            "dynamic",
            -6.0,
        ),
        call("bin-y", "another__repo-1", "a y", 50.0, -5.0, "bin-y", -4.0),
        call("bin-w", "another__repo-2", "a w", 50.0, -3.0, "bin-w", -2.0),
    ]
    eval_calls = [
        call("hit", "eval__repo-1", "a x", 750.0, 0.0, "first", 2.0),
        call(
            "canonical-hit",
            "eval__repo-1",
            "a /tmp/run-2 202",
            750.0,
            0.5,
            "first",
            2.0,
        ),
        call("miss", "eval__repo-1", "a z", 50.0, 1.0, "first", 2.0),
        call("local", "eval__repo-1", "a x", 750.0, 3.0, "second", 4.0),
    ]

    result = evaluate_representation(fit_calls, eval_calls, {"fixture": True})

    assert result["row_identity"] == {
        "identical_mapped_row_ids_and_labels": True,
        "mapped_row_count": 4,
        "eligible_row_count": 4,
    }
    raw = result["candidates"]["raw_argv"]
    canonical = result["candidates"]["canonical_argv"]
    assert set(result["metrics"]) == {
        "public_binary_with_local_updates",
        "raw_argv_hierarchy_with_local_updates",
        "canonical_argv_hierarchy_with_local_updates",
    }
    assert {
        metrics["exact_bucket"]["eligible_examples"]
        for metrics in result["metrics"].values()
    } == {4}
    assert raw["canonicalizer_version"] is None
    assert canonical["canonicalizer_version"] == "generic-argv-v2-shape"
    assert canonical["hierarchy"] == [
        "repo_raw_argv:exact_clause",
        "repo_raw_argv:argv_prefix_depth_4",
        "repo_raw_argv:argv_prefix_depth_3",
        "repo_raw_argv:argv_prefix_depth_2",
        "repo:bin",
        "public_canonical_argv:exact_clause",
        "public_canonical_argv:argv_prefix_depth_4",
        "public_canonical_argv:argv_prefix_depth_3",
        "public_canonical_argv:argv_prefix_depth_2",
        "public:bin",
        "public:global",
    ]
    assert raw["hit_count"] == 1
    assert canonical["hit_count"] == 3
    assert raw["public_fallback_opportunity_count"] == 3
    assert canonical["public_fallback_opportunity_count"] == 3
    assert raw["support_bands"] == {"1": 1, "2-4": 0, "5+": 0}
    assert canonical["support_bands"] == {"1": 1, "2-4": 2, "5+": 0}
    assert raw["paired_exact_bucket_transitions"]["candidate_hits"] == {
        "bin_correct_candidate_correct": 0,
        "bin_correct_candidate_wrong": 0,
        "bin_wrong_candidate_correct": 1,
        "bin_wrong_candidate_wrong": 0,
    }
    assert canonical["paired_exact_bucket_transitions"]["candidate_hits"] == {
        "bin_correct_candidate_correct": 1,
        "bin_correct_candidate_wrong": 0,
        "bin_wrong_candidate_correct": 1,
        "bin_wrong_candidate_wrong": 1,
    }
    assert result["diagnostics"]["raw_argv"]["prediction_provenance_counts"] == {
        "public:bin": 2,
        "public_raw_argv:exact_clause": 1,
        "repo:exact_clause": 1,
    }
    assert result["diagnostics"]["canonical_argv"]["prediction_provenance_counts"] == {
        "public_canonical_argv:exact_clause": 3,
        "repo:exact_clause": 1,
    }


def test_serialized_virtual_deployment_uses_input_trace_order() -> None:
    raw = [
        _call(
            "first-early",
            "eval__repo-1",
            start=100.0,
            end=100.75,
            command="a",
            clauses=({"bin": "a", "argv": ["a"]},),
            segments=((0.0, 750.0),),
            trace_finalized_at=math.nan,
            source_trace="manifest-first.jsonl",
        ),
        _call(
            "first-late",
            "eval__repo-1",
            start=110.0,
            end=110.05,
            command="a",
            clauses=({"bin": "a", "argv": ["a"]},),
            segments=((0.0, 50.0),),
            trace_finalized_at=math.nan,
            source_trace="manifest-first.jsonl",
        ),
        _call(
            "second",
            "eval__repo-2",
            start=1.0,
            end=1.05,
            command="a",
            clauses=({"bin": "a", "argv": ["a"]},),
            segments=((0.0, 50.0),),
            trace_finalized_at=math.nan,
            source_trace="manifest-second.jsonl",
        ),
    ]
    calls = _serialize_virtual_deployment(raw)

    assert calls[0].sample.tool_ts_start == 0.0
    assert calls[1].sample.tool_ts_start == 10.0
    assert calls[0].trace_finalized_at == calls[1].trace_finalized_at
    assert calls[1].trace_finalized_at < calls[2].sample.tool_ts_start

    fit = _call(
        "public",
        "other__repo-1",
        start=-2.0,
        end=-1.95,
        command="a",
        clauses=({"bin": "a", "argv": ["a"]},),
        segments=((0.0, 50.0),),
        trace_finalized_at=-1.0,
    )
    _, rows = evaluate([fit], calls, {"fixture": True})
    by_id = {row.sample_id: row for row in rows}
    assert by_id["first-early:clause:0"].layer == "public"
    assert by_id["first-late:clause:0"].layer == "public"
    assert by_id["second:clause:0"].layer == "repo"
    assert by_id["second:clause:0"].evidence_count == 2


def test_empty_cross_repo_public_evidence_is_unavailable(tmp_path: Path) -> None:
    same_repo_fit = _call(
        "same-repo",
        "eval__repo-9",
        start=9.0,
        end=10.0,
        command="a",
        clauses=({"bin": "a", "argv": ["a"]},),
        segments=((0.0, 750.0),),
        trace_finalized_at=10.1,
    )
    query = _call(
        "query",
        "eval__repo-1",
        start=10.0,
        end=10.05,
        command="a",
        clauses=({"bin": "a", "argv": ["a"]},),
        segments=((0.0, 50.0),),
        trace_finalized_at=10.1,
    )

    result, rows = evaluate([same_repo_fit], [query], {"fixture": True})

    assert rows[0].probability_by_bucket is None
    assert rows[0].unavailable_reason == (
        "ValueError: no public global clause latency node"
    )
    assert result["diagnostics"]["unavailable_clause_examples"] == 1
    assert result["diagnostics"]["censored_exec_call_count"] == 0
    assert result["diagnostics"]["unavailable_reasons"] == {
        "ValueError: no public global clause latency node": 1
    }
    assert result["metrics"]["by_boundary"][0]["eligible_examples"] == 0
    assert result["metrics"]["by_boundary"][0]["accuracy"] is None

    store = ObservationStore(tmp_path / "empty.sqlite3")
    snapshot = store.create_snapshot()
    service = ResourceService(store, None)  # type: ignore[arg-type]
    run = _open_run(
        service,
        behavior="predict",
        snapshot=snapshot,
        scope="eval__repo",
    )
    trace = _open_trace(service, run["run_token"])
    online = service.dispatch(
        "BeginCall",
        {
            "trace_token": trace["trace_token"],
            "call_id": "query",
            "command": "a",
            "query_timestamp": query.sample.tool_ts_start,
        },
    )
    assert online["fallback_reason"] == rows[0].unavailable_reason
    service.close()


def test_offline_and_agentd_match_one_restored_timestamped_stream(
    tmp_path: Path,
) -> None:
    base = time.time()
    fit = _call(
        "public",
        "other__repo-1",
        start=base - 10.05,
        end=base - 10.0,
        command="a",
        clauses=({"bin": "a", "argv": ["a"]},),
        segments=((0.0, 50.0),),
        trace_finalized_at=base - 9.0,
        source_trace="public.jsonl",
    )
    learned_early = _call(
        "learned-early",
        "owner__repo-1",
        start=base,
        end=base + 0.75,
        command="a",
        clauses=({"bin": "a", "argv": ["a"]},),
        segments=((0.0, 750.0),),
        trace_finalized_at=base + 2.5,
        source_trace="first.jsonl",
    )
    same_trace_late = _call(
        "same-trace-late",
        "owner__repo-1",
        start=base + 1.0,
        end=base + 2.0,
        command="a",
        clauses=({"bin": "a", "argv": ["a"]},),
        segments=((0.0, 50.0),),
        trace_finalized_at=base + 2.5,
        source_trace="first.jsonl",
    )
    while_trace_open = _call(
        "while-trace-open",
        "owner__repo-1",
        start=base + 2.1,
        end=base + 2.15,
        command="a",
        clauses=({"bin": "a", "argv": ["a"]},),
        segments=((0.0, 50.0),),
        trace_finalized_at=base + 3.2,
        source_trace="second.jsonl",
    )
    after_close = _call(
        "after-close",
        "owner__repo-1",
        start=base + 2.6,
        end=base + 2.65,
        command="a",
        clauses=({"bin": "a", "argv": ["a"]},),
        segments=((0.0, 50.0),),
        trace_finalized_at=base + 3.2,
        source_trace="second.jsonl",
    )
    after_restore = _call(
        "after-restore",
        "owner__repo-1",
        start=base + 4.0,
        end=base + 4.05,
        command="a",
        clauses=({"bin": "a", "argv": ["a"]},),
        segments=((0.0, 50.0),),
        trace_finalized_at=base + 5.0,
        source_trace="third.jsonl",
    )
    compound = _call(
        "compound",
        "owner__repo-1",
        start=base + 4.5,
        end=base + 4.6,
        command="a; b",
        clauses=(
            {"bin": "a", "argv": ["a"]},
            {"bin": "b", "argv": ["b"]},
        ),
        segments=((0.0, 50.0), (50.0, 100.0)),
        trace_finalized_at=base + 5.0,
        source_trace="third.jsonl",
    )
    offline, _ = _fit_public_kbs(
        [fit],
        [
            learned_early,
            same_trace_late,
            while_trace_open,
            after_close,
            after_restore,
            compound,
        ],
    )
    rows = _score(
        offline,
        [learned_early, same_trace_late, while_trace_open, after_close],
    )
    offline["owner__repo"] = ClauseResourceKB.from_json_obj(
        json.loads(json.dumps(offline["owner__repo"].to_json_obj()))
    )
    rows.extend(_score(offline, [after_restore, compound]))
    offline_by_id = {row.sample_id: row for row in rows}

    store = ObservationStore(tmp_path / "observations.sqlite3")
    store.insert_observation(
        _envelope(
            "public",
            scope="other__repo",
            command="a",
            end=fit.sample.tool_ts_end,
            latency_ms=50.0,
        )
    )
    store.promote_observations({"public"})
    snapshot = store.create_snapshot()

    call_times = {
        call.sample.sample_id: (
            call.sample.tool_ts_start,
            call.sample.tool_ts_end,
            latency_ms,
        )
        for call, latency_ms in (
            (learned_early, 750.0),
            (same_trace_late, 50.0),
            (while_trace_open, 50.0),
            (after_close, 50.0),
            (after_restore, 50.0),
            (compound, 100.0),
        )
    }

    class TimestampedCollector(_FakeCollector):
        def finish_tool_call(
            self,
            token: dict[str, Any],
            *,
            replay_response: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            result = super().finish_tool_call(
                token,
                replay_response=replay_response,
            )
            start, end, latency_ms = call_times[token["call_id"]]
            result["clauses"][0].update(
                ts_start=start,
                ts_end=end,
                latency_ms=latency_ms,
            )
            return result

    telemetry = TelemetryService(
        collector_factory=TimestampedCollector,
        state_dir=tmp_path / "telemetry",
    )
    service = ResourceService(store, _DirectTransport(telemetry))
    run = _open_run(
        service,
        behavior="observe_predict_learn",
        snapshot=snapshot,
        run_id="golden",
        scope="owner__repo",
    )
    runtime_kb_id = id(service._runs[run["run_token"]].kb)
    online_by_id: dict[str, dict[str, Any]] = {}
    first_trace = _open_trace(service, run["run_token"], trace_id="first")
    for call in (learned_early, same_trace_late):
        begin, _ = _run_call(
            service,
            first_trace["trace_token"],
            call_id=call.sample.sample_id,
            command=call.command,
            query_timestamp=call.sample.tool_ts_start,
        )
        online_by_id[call.sample.sample_id] = begin["prediction"]
    second_trace = _open_trace(service, run["run_token"], trace_id="second")
    begin, _ = _run_call(
        service,
        second_trace["trace_token"],
        call_id=while_trace_open.sample.sample_id,
        command=while_trace_open.command,
        query_timestamp=while_trace_open.sample.tool_ts_start,
    )
    online_by_id[while_trace_open.sample.sample_id] = begin["prediction"]
    service.dispatch(
        "CloseTrace",
        {
            "trace_token": first_trace["trace_token"],
            "workload_status": "completed",
        },
    )
    begin, _ = _run_call(
        service,
        second_trace["trace_token"],
        call_id=after_close.sample.sample_id,
        command=after_close.command,
        query_timestamp=after_close.sample.tool_ts_start,
    )
    online_by_id[after_close.sample.sample_id] = begin["prediction"]
    service.dispatch(
        "CloseTrace",
        {
            "trace_token": second_trace["trace_token"],
            "workload_status": "completed",
        },
    )
    third_trace = _open_trace(service, run["run_token"], trace_id="third")
    for call in (after_restore, compound):
        begin, _ = _run_call(
            service,
            third_trace["trace_token"],
            call_id=call.sample.sample_id,
            command=call.command,
            query_timestamp=call.sample.tool_ts_start,
        )
        online_by_id[call.sample.sample_id] = begin["prediction"]

    for call in (
        learned_early,
        same_trace_late,
        while_trace_open,
        after_close,
        after_restore,
    ):
        online = online_by_id[call.sample.sample_id]
        offline_row = offline_by_id[f"{call.sample.sample_id}:clause:0"]
        clause = online["prediction"]
        assert offline_row.probability_by_bucket == (
            None if clause is None else clause["probability_by_bucket"]
        )
        assert offline_row.layer == (None if clause is None else clause["scope"])
        assert offline_row.key_kind == (None if clause is None else clause["key_kind"])
        assert offline_row.evidence_count == (
            0 if clause is None else clause["evidence_count"]
        )
        assert offline_row.fallback_path == (
            None if clause is None else clause["fallback_path"]
        )
        assert offline_row.unavailable_reason == online["unavailable_reason"]
        assert id(service._runs[run["run_token"]].kb) == runtime_kb_id

    assert offline_by_id["learned-early:clause:0"].layer == "public"
    assert offline_by_id["same-trace-late:clause:0"].layer == "public"
    assert offline_by_id["while-trace-open:clause:0"].layer == "public"
    assert offline_by_id["after-close:clause:0"].evidence_count == 2
    assert offline_by_id["after-close:clause:0"].probability_by_bucket[:2] == (
        0.5,
        0.5,
    )
    assert offline_by_id["after-restore:clause:0"].evidence_count == 4
    assert offline_by_id["after-restore:clause:0"].probability_by_bucket[:2] == (
        0.75,
        0.25,
    )
    assert offline_by_id["compound:clause:0"].probability_by_bucket is not None
    assert offline_by_id["compound:clause:1"].probability_by_bucket is not None
    assert online_by_id["compound"]["unavailable_reason"] == (
        "compound_command_uncomposed"
    )
    service.close()
    telemetry.close()


def test_evaluator_scores_each_mapped_clause_without_composition() -> None:
    fit_calls = [
        _call(
            "fit",
            "fit__repo-1",
            start=0.0,
            end=0.05,
            command="a",
            clauses=({"bin": "a", "argv": ["a"]},),
            segments=((0.0, 50.0),),
            trace_finalized_at=1.0,
        )
    ]
    eval_calls = [
        _call(
            "edge",
            "eval__repo-1",
            start=10.0,
            end=10.5,
            command="a",
            clauses=({"bin": "a", "argv": ["a"]},),
            segments=((0.0, 500.0),),
            trace_finalized_at=23.0,
        ),
        _call(
            "compound",
            "eval__repo-1",
            start=20.0,
            end=22.0,
            command="a; b",
            clauses=(
                {"bin": "a", "argv": ["a"]},
                {"bin": "b", "argv": ["b"]},
            ),
            segments=((0.0, 100.0), (100.0, 1600.0)),
            trace_finalized_at=23.0,
        ),
    ]

    result, rows = evaluate(
        fit_calls,
        eval_calls,
        {"fixture": True},
    )

    by_id = {row.sample_id: row for row in rows}
    assert by_id["edge:clause:0"].label_bucket == 0
    assert by_id["edge:clause:0"].probability_by_bucket is not None
    assert by_id["compound:clause:0"].label_bucket == 0
    assert by_id["compound:clause:1"].label_bucket == 2
    assert by_id["compound:clause:0"].probability_by_bucket is not None
    assert by_id["compound:clause:1"].probability_by_bucket is not None
    assert len(rows) == 3
    assert result["diagnostics"]["payload_clause_count"] == 3
    assert result["diagnostics"]["mapped_uncensored_clause_count"] == 3
    assert result["diagnostics"]["unavailable_clause_examples"] == 0
    assert result["claim_bearing"] is False
    assert result["bucket_edges_ms"] == [
        500.0,
        1000.0,
        2000.0,
        4000.0,
        8000.0,
        16000.0,
        32000.0,
        64000.0,
    ]
    assert result["bucket_intervals"][0] == {
        "bucket_id": 0,
        "lower_ms": 0.0,
        "lower_inclusive": True,
        "upper_ms": 500.0,
        "upper_inclusive": True,
    }
    assert result["bucket_intervals"][-1] == {
        "bucket_id": 8,
        "lower_ms": 64000.0,
        "lower_inclusive": False,
        "upper_ms": None,
        "upper_inclusive": False,
    }
    assert "balanced_accuracy" not in result["metrics"]


def test_cli_has_no_bucket_override() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["--bucket-edges-ms", "100,1000"])


def test_censored_proxy_call_produces_no_latency_observation() -> None:
    call = _call(
        "censored",
        "owner__repo-1",
        start=0.0,
        end=1.0,
        command="a",
        clauses=({"bin": "a", "argv": ["a"]},),
        segments=((0.0, 1000.0),),
        trace_finalized_at=2.0,
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
