#!/usr/bin/env python3
"""Replay the canonical clause-KB on development-exposed proxy traces.

This proxy is not canonical clause telemetry and cannot produce a claim-bearing
result.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tool_resource.features import parse_command_clauses, repo_of  # noqa: E402
from tool_resource.labels import (  # noqa: E402
    ResourceCallSample,
    load_resource_corpus,
)
from tool_resource.runtime_kb import (  # noqa: E402
    CANONICAL_LATENCY_BUCKETS,
    ClauseObservation,
    ClauseResourceKB,
)
from trace_collect.trace_data import TraceData  # noqa: E402

_FIT_ROOT = Path(
    "traces/swe-rebench/qwen3.7-max/offline-gated-confirm-100-v2-pacct-perbin"
)
_EVAL_ROOT = Path("traces/fresh-277-segtimeline")
_FIT_MANIFEST = Path("configs/corpora/swe-100.json")
_EVAL_MANIFEST = Path("configs/corpora/swe-277.json")
_VIRTUAL_TRACE_GAP_S = 1.0


@dataclass(frozen=True)
class ProxyCall:
    sample: ResourceCallSample
    command: str
    clauses: tuple[Mapping[str, Any], ...]
    parse_failed: bool
    mapping_evidence: str
    segment_times_ms: tuple[tuple[float, float], ...]
    trace_finalized_at: float


@dataclass(frozen=True)
class ScoredRow:
    sample_id: str
    task_id: str
    repo: str
    command: str
    label_bucket: int
    probability_by_bucket: tuple[float, ...] | None
    layer: str | None
    key_kind: str | None
    evidence_count: int
    fallback_path: tuple[str, ...] | None
    unavailable_reason: str | None
    mapping_evidence: str


def _load_proxy_calls(
    root: Path,
    manifest: Path,
    *,
    limit_tasks: int | None = None,
) -> tuple[list[ProxyCall], list[str], int]:
    samples_by_task, loaded_task_ids = load_resource_corpus(root, manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    task_ids = [task_id.strip() for task_id in payload["task_ids"]]
    if set(task_ids) != set(loaded_task_ids):
        raise ValueError(f"{manifest}: loaded task IDs differ from manifest order")
    if limit_tasks is not None:
        if limit_tasks < 1:
            raise ValueError("--limit-eval-tasks must be positive")
        task_ids = task_ids[:limit_tasks]
    calls: list[ProxyCall] = []
    parse_failures = 0
    for task_id in task_ids:
        action_data = _action_data_by_key(samples_by_task[task_id])
        for sample in samples_by_task[task_id]:
            command = (sample.tool_args or {}).get("command")
            if sample.tool_name != "exec" or not isinstance(command, str):
                continue
            parsed = parse_command_clauses(command)
            parse_failed = bool(parsed["parse_failed"])
            parse_failures += parse_failed
            clauses = tuple(parsed["clauses"])
            key = (sample.agent_id, sample.iteration, sample.action_id)
            mapping_evidence, segment_times_ms = _segment_evidence(
                clauses, action_data[key].get("segment_timeline")
            )
            calls.append(
                ProxyCall(
                    sample=sample,
                    command=command,
                    clauses=clauses,
                    parse_failed=parse_failed,
                    mapping_evidence=mapping_evidence,
                    segment_times_ms=segment_times_ms,
                    trace_finalized_at=math.nan,
                )
            )
    return _serialize_virtual_deployment(calls), task_ids, parse_failures


def _action_data_by_key(
    samples: Sequence[ResourceCallSample],
) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    by_trace: dict[str, list[ResourceCallSample]] = defaultdict(list)
    for sample in samples:
        by_trace[sample.source_trace].append(sample)
    output: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for trace_path, trace_samples in by_trace.items():
        trace = TraceData.load(Path(trace_path))
        _final_trace_summary(trace)
        actions = {
            (
                str(action["agent_id"]),
                int(action["iteration"]),
                str(action["action_id"]),
            ): action.get("data") or {}
            for action in trace.actions
            if action.get("action_type") == "tool_exec"
        }
        for sample in trace_samples:
            key = (sample.agent_id, sample.iteration, sample.action_id)
            if key not in actions:
                raise ValueError(f"{sample.sample_id}: missing raw tool action")
            if key in output:
                raise ValueError(f"duplicate tool action key: {key}")
            output[key] = actions[key]
    return output


def _final_trace_summary(trace: TraceData) -> Mapping[str, Any]:
    final_record: Mapping[str, Any] | None = None
    with trace.path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                final_record = json.loads(line)
    if final_record is None or final_record.get("type") != "summary":
        raise ValueError(
            f"{trace.path}: final record is not a successful trace summary"
        )
    return final_record


def _trace_finalization_timestamp(trace: TraceData) -> float:
    """Return the timestamp on the source trace's final summary event."""

    final_record = _final_trace_summary(trace)
    timestamp = final_record.get("ts")
    if (
        not isinstance(timestamp, (int, float))
        or isinstance(timestamp, bool)
        or not math.isfinite(timestamp)
    ):
        raise ValueError(
            f"{trace.path}: final trace summary has no finite ts; "
            "faithful CloseTrace ordering is unavailable"
        )
    return float(timestamp)


def _serialize_virtual_deployment(calls: Sequence[ProxyCall]) -> list[ProxyCall]:
    """Place source traces in input order on a deterministic synthetic clock."""

    by_trace: dict[str, list[ProxyCall]] = {}
    for call in calls:
        by_trace.setdefault(call.sample.source_trace, []).append(call)
    serialized: list[ProxyCall] = []
    next_trace_start = 0.0
    for trace_calls in by_trace.values():
        source_start = min(call.sample.tool_ts_start for call in trace_calls)
        shift = next_trace_start - source_start
        shifted = [
            replace(
                call,
                sample=replace(
                    call.sample,
                    tool_ts_start=call.sample.tool_ts_start + shift,
                    tool_ts_end=call.sample.tool_ts_end + shift,
                ),
            )
            for call in trace_calls
        ]
        trace_closed_at = (
            max(call.sample.tool_ts_end for call in shifted) + _VIRTUAL_TRACE_GAP_S
        )
        serialized.extend(
            replace(call, trace_finalized_at=trace_closed_at) for call in shifted
        )
        next_trace_start = trace_closed_at + _VIRTUAL_TRACE_GAP_S
    return serialized


def _segment_evidence(
    clauses: Sequence[Mapping[str, Any]], timeline: Any
) -> tuple[str, tuple[tuple[float, float], ...]]:
    if not isinstance(timeline, Mapping):
        return "missing", ()
    segments = timeline.get("segments")
    if not isinstance(segments, list):
        return "malformed", ()
    runtime_bins: list[str] = []
    times: list[tuple[float, float]] = []
    for segment in segments:
        if not isinstance(segment, Mapping) or not isinstance(
            segment.get("command_text"), str
        ):
            return "malformed", ()
        start = segment.get("t_start_ms")
        end = segment.get("t_end_ms")
        if (
            not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0.0
            or end < start
        ):
            return "invalid_timing", ()
        parsed = parse_command_clauses(segment["command_text"])
        if parsed["parse_failed"] or len(parsed["clauses"]) != 1:
            return "segment_parse_mismatch", ()
        runtime_bins.append(str(parsed["clauses"][0]["bin"]))
        times.append((float(start), float(end)))
    static_bins = [str(clause["bin"]) for clause in clauses]
    if len(runtime_bins) != len(static_bins):
        return "count_mismatch", ()
    if runtime_bins != static_bins:
        return "bin_mismatch", ()
    return "bin_exact", tuple(times)


def _payload_clauses(
    call: ProxyCall,
) -> tuple[tuple[Mapping[str, Any], ...], int]:
    clauses = list(call.clauses)
    structural = 0
    while len(clauses) > 1 and str(clauses[0]["bin"]) == "cd":
        clauses.pop(0)
        structural += 1
    return tuple(clauses), structural


def _call_unavailable_reason(call: ProxyCall) -> str | None:
    if call.sample.censored:
        return "censored"
    if call.parse_failed:
        return "parse_failed"
    if call.mapping_evidence != "bin_exact":
        return f"segment_evidence_{call.mapping_evidence}"
    return None


def _proxy_latency_observations(call: ProxyCall, repo: str) -> list[ClauseObservation]:
    """Legacy segment latency only; never attach proxy CPU or RSS."""

    if _call_unavailable_reason(call) is not None:
        return []
    clauses, structural = _payload_clauses(call)
    times = call.segment_times_ms[structural:]
    return [
        ClauseObservation(
            repo=repo,
            bin=str(clause["bin"]),
            argv=tuple(clause["argv"]),
            ts_start=call.sample.tool_ts_start + start_ms / 1000.0,
            # Offline replay learns mapping only when the outer call finishes.
            ts_end=call.sample.tool_ts_end,
            latency_ms=end_ms - start_ms,
        )
        for clause, (start_ms, end_ms) in zip(clauses, times, strict=True)
    ]


def _validate_partition(
    fit_tasks: Sequence[str], eval_tasks: Sequence[str]
) -> dict[str, int]:
    fit = set(fit_tasks)
    evaluate = set(eval_tasks)
    overlap = fit & evaluate
    if overlap:
        raise ValueError(f"fit/eval task overlap: {sorted(overlap)[:3]}")
    fit_repos = {repo_of(task) for task in fit}
    eval_repos = {repo_of(task) for task in evaluate}
    return {
        "fit_task_count": len(fit),
        "eval_task_count": len(evaluate),
        "task_overlap_count": 0,
        "fit_repo_count": len(fit_repos),
        "eval_repo_count": len(eval_repos),
        "repo_overlap_count": len(fit_repos & eval_repos),
    }


def _score(
    kb_by_repo: Mapping[str, ClauseResourceKB],
    calls: Sequence[ProxyCall],
) -> list[ScoredRow]:
    _validate_trace_finalizations(calls)
    trace_observations: dict[tuple[str, str], list[ClauseObservation]] = defaultdict(
        list
    )
    trace_close_times: dict[tuple[str, str], float] = {}
    trace_call_counts: Counter[tuple[str, str]] = Counter()
    for call in calls:
        repo = repo_of(call.sample.task_id)
        trace = (repo, call.sample.source_trace)
        trace_call_counts[trace] += 1
        trace_observations[trace].extend(_proxy_latency_observations(call, repo))
        trace_close_times.setdefault(trace, call.trace_finalized_at)
    close_events = [(ts_end, trace) for trace, ts_end in trace_close_times.items()]
    heapq.heapify(close_events)
    processed_calls: Counter[tuple[str, str]] = Counter()

    def close_trace(trace: tuple[str, str]) -> None:
        for observation in trace_observations[trace]:
            kb_by_repo[trace[0]].observe_completed_clause(observation)

    rows: list[ScoredRow] = []
    for call in sorted(
        calls, key=lambda item: (item.sample.tool_ts_start, item.sample.sample_id)
    ):
        repo = repo_of(call.sample.task_id)
        trace = (repo, call.sample.source_trace)
        deferred_closes: list[tuple[float, tuple[str, str]]] = []
        while close_events and close_events[0][0] <= call.sample.tool_ts_start:
            close_event = heapq.heappop(close_events)
            closing_trace = close_event[1]
            if processed_calls[closing_trace] == trace_call_counts[closing_trace]:
                close_trace(closing_trace)
            else:
                deferred_closes.append(close_event)
        for close_event in deferred_closes:
            heapq.heappush(close_events, close_event)
        kb = kb_by_repo[repo]
        for clause_index, observation in enumerate(
            _proxy_latency_observations(call, repo)
        ):
            prediction_error = None
            try:
                prediction = kb.predict_clause_latency_bucket(
                    repo,
                    observation.bin,
                    observation.argv,
                    CANONICAL_LATENCY_BUCKETS,
                    ts_start=call.sample.tool_ts_start,
                )
            except ValueError as exc:
                if str(exc) != "no public global clause latency node":
                    raise
                prediction = None
                prediction_error = f"ValueError: {exc}"
            rows.append(
                ScoredRow(
                    sample_id=f"{call.sample.sample_id}:clause:{clause_index}",
                    task_id=call.sample.task_id,
                    repo=repo,
                    command=call.command,
                    label_bucket=CANONICAL_LATENCY_BUCKETS.bucket_id(
                        observation.latency_ms
                    ),
                    probability_by_bucket=(
                        None if prediction is None else prediction.probability_by_bucket
                    ),
                    layer=None if prediction is None else prediction.scope,
                    key_kind=None if prediction is None else prediction.key_kind,
                    evidence_count=(
                        0 if prediction is None else prediction.evidence_count
                    ),
                    fallback_path=(
                        None if prediction is None else prediction.fallback_path
                    ),
                    unavailable_reason=prediction_error,
                    mapping_evidence=call.mapping_evidence,
                )
            )
        processed_calls[trace] += 1
    while close_events:
        _, trace = heapq.heappop(close_events)
        close_trace(trace)
    return rows


def _fit_public_kbs(
    fit_calls: Sequence[ProxyCall],
    eval_calls: Sequence[ProxyCall],
) -> tuple[dict[str, ClauseResourceKB], list[ClauseObservation]]:
    _validate_trace_finalizations(fit_calls)
    observations = [
        observation
        for call in fit_calls
        for observation in _proxy_latency_observations(
            call,
            repo_of(call.sample.task_id),
        )
    ]
    kb_by_repo: dict[str, ClauseResourceKB] = {}
    for repo in sorted({repo_of(call.sample.task_id) for call in eval_calls}):
        public = [
            observation for observation in observations if observation.repo != repo
        ]
        kb = ClauseResourceKB.fit_public(public) if public else ClauseResourceKB()
        kb_by_repo[repo] = kb
    return kb_by_repo, observations


def _validate_trace_finalizations(calls: Sequence[ProxyCall]) -> None:
    finalized_at: dict[str, float] = {}
    for call in calls:
        if (
            not math.isfinite(call.trace_finalized_at)
            or call.trace_finalized_at < call.sample.tool_ts_end
        ):
            raise ValueError(
                f"{call.sample.source_trace}: invalid trace finalization timestamp"
            )
        previous = finalized_at.setdefault(
            call.sample.source_trace,
            call.trace_finalized_at,
        )
        if previous != call.trace_finalized_at:
            raise ValueError(
                f"{call.sample.source_trace}: inconsistent trace finalization "
                "timestamps"
            )


def _metrics(rows: Sequence[ScoredRow]) -> dict[str, Any]:
    known = [row for row in rows if row.probability_by_bucket is not None]
    return {
        "by_boundary": [
            _boundary_metrics(known, boundary_index, boundary_ms)
            for boundary_index, boundary_ms in enumerate(
                CANONICAL_LATENCY_BUCKETS.edges_ms
            )
        ],
    }


def _diagnostics(
    rows: Sequence[ScoredRow],
    *,
    calls: Sequence[ProxyCall],
) -> dict[str, Any]:
    prediction_unavailable = [row for row in rows if row.probability_by_bucket is None]
    parsed_static_clause_count = sum(len(call.clauses) for call in calls)
    payload_clause_count = 0
    structural_clause_count = 0
    unavailable_reasons: Counter[str] = Counter(
        row.unavailable_reason
        for row in prediction_unavailable
        if row.unavailable_reason
    )
    for call in calls:
        payload, structural = _payload_clauses(call)
        payload_clause_count += len(payload)
        structural_clause_count += structural
        reason = _call_unavailable_reason(call)
        if reason is not None:
            unavailable_reasons[reason] += len(payload)
    eligible_examples = sum(row.probability_by_bucket is not None for row in rows)
    unavailable_clause_examples = payload_clause_count - eligible_examples
    if parsed_static_clause_count != structural_clause_count + payload_clause_count:
        raise ValueError("static clause reconciliation failed")
    if unavailable_clause_examples != sum(unavailable_reasons.values()):
        raise ValueError("unavailable clause reconciliation failed")
    return {
        "parsed_static_clause_count": parsed_static_clause_count,
        "structural_leading_cd_clause_count": structural_clause_count,
        "payload_clause_count": payload_clause_count,
        "mapped_uncensored_clause_count": len(rows),
        "unavailable_clause_examples": unavailable_clause_examples,
        "unavailable_reasons": dict(sorted(unavailable_reasons.items())),
        "censored_exec_call_count": sum(call.sample.censored for call in calls),
        "prediction_provenance_counts": dict(
            sorted(
                Counter(
                    f"{row.layer}:{row.key_kind}"
                    for row in rows
                    if row.probability_by_bucket is not None
                ).items()
            )
        ),
    }


def _boundary_metrics(
    rows: Sequence[ScoredRow],
    boundary_index: int,
    boundary_ms: float,
) -> dict[str, Any]:
    true_positive = true_negative = false_positive = false_negative = 0
    for row in rows:
        assert row.probability_by_bucket is not None
        predicted_positive = sum(row.probability_by_bucket[boundary_index + 1 :]) > 0.5
        positive = row.label_bucket > boundary_index
        if predicted_positive and positive:
            true_positive += 1
        elif predicted_positive:
            false_positive += 1
        elif positive:
            false_negative += 1
        else:
            true_negative += 1
    positives = true_positive + false_negative
    correct = true_positive + true_negative
    return {
        "boundary_ms": boundary_ms,
        "accuracy": correct / len(rows) if rows else None,
        "eligible_examples": len(rows),
        "positive_count": positives,
        "positive_rate": positives / len(rows) if rows else None,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def evaluate(
    fit_calls: Sequence[ProxyCall],
    eval_calls: Sequence[ProxyCall],
    provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], list[ScoredRow]]:
    kb_by_repo, observations = _fit_public_kbs(fit_calls, eval_calls)
    rows = _score(kb_by_repo, eval_calls)
    return (
        {
            "status": "development_diagnostic_proxy_not_canonical_telemetry",
            "claim_bearing": False,
            "objective": "clause_latency_bucket_prediction",
            "bucket_edges_ms": list(CANONICAL_LATENCY_BUCKETS.edges_ms),
            "bucket_intervals": [
                {
                    "bucket_id": bucket,
                    "lower_ms": (
                        0.0
                        if bucket == 0
                        else CANONICAL_LATENCY_BUCKETS.edges_ms[bucket - 1]
                    ),
                    "lower_inclusive": bucket == 0,
                    "upper_ms": (
                        CANONICAL_LATENCY_BUCKETS.edges_ms[bucket]
                        if bucket < len(CANONICAL_LATENCY_BUCKETS.edges_ms)
                        else None
                    ),
                    "upper_inclusive": bucket < len(CANONICAL_LATENCY_BUCKETS.edges_ms),
                }
                for bucket in range(CANONICAL_LATENCY_BUCKETS.bucket_count)
            ],
            "fit_clause_observation_count": len(observations),
            "eval_exec_call_count": len(eval_calls),
            "metrics": _metrics(rows),
            "diagnostics": _diagnostics(rows, calls=eval_calls),
            "provenance": dict(provenance),
        },
        rows,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-root", type=Path, default=_FIT_ROOT)
    parser.add_argument("--eval-root", type=Path, default=_EVAL_ROOT)
    parser.add_argument("--fit-manifest", type=Path, default=_FIT_MANIFEST)
    parser.add_argument("--eval-manifest", type=Path, default=_EVAL_MANIFEST)
    parser.add_argument("--limit-eval-tasks", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dump-rows", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    fit_calls, fit_tasks, fit_parse_failures = _load_proxy_calls(
        args.fit_root, args.fit_manifest
    )
    eval_calls, eval_tasks, eval_parse_failures = _load_proxy_calls(
        args.eval_root,
        args.eval_manifest,
        limit_tasks=args.limit_eval_tasks,
    )
    provenance = {
        **_validate_partition(fit_tasks, eval_tasks),
        "fit_root": str(args.fit_root.resolve()),
        "eval_root": str(args.eval_root.resolve()),
        "fit_manifest": str(args.fit_manifest.resolve()),
        "eval_manifest": str(args.eval_manifest.resolve()),
        "fit_parse_failures": fit_parse_failures,
        "eval_parse_failures": eval_parse_failures,
        "proxy_adapter": (
            "legacy bash-xtrace exact-bin segment latency only; no proxy CPU/RSS"
        ),
        "scoring_unit": "mapped_non_structural_clause",
        "deployment_semantics": "serialized_virtual_deployment",
        "task_order": "manifest task_ids order; no shuffle",
        "trace_order": "one source trace per task in manifest order",
        "trace_update": (
            "predict every mapped clause in all calls without intra-trace learning, "
            "then successful CloseTrace before the next trace"
        ),
        "virtual_clock": (
            "deterministic monotonic shift preserving source call order and "
            "durations; not historical concurrency or a historical close timestamp"
        ),
        "virtual_trace_gap_s": _VIRTUAL_TRACE_GAP_S,
        "edge_source": "canonical_objective",
    }
    result, rows = evaluate(fit_calls, eval_calls, provenance)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        sys.stdout.write(payload)
    else:
        args.out.write_text(payload, encoding="utf-8")
    if args.dump_rows is not None:
        args.dump_rows.write_text(
            "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
