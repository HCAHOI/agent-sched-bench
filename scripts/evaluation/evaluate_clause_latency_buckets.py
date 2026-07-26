#!/usr/bin/env python3
"""Diagnose clause-KB latency buckets on development-exposed legacy traces.

Numeric bucket edges are required explicitly. This proxy is not canonical
Stage-2 evidence and cannot produce a claim-bearing result.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tool_resource.features import parse_command_clauses, repo_of
from tool_resource.labels import ResourceCallSample, load_resource_corpus
from tool_resource.runtime_kb import (
    ClauseObservation,
    ClauseResourceKB,
    LatencyBuckets,
)
from trace_collect.trace_data import TraceData

_FIT_ROOT = Path(
    "traces/swe-rebench/qwen3.7-max/offline-gated-confirm-100-v2-pacct-perbin"
)
_EVAL_ROOT = Path("traces/fresh-277-segtimeline")
_FIT_MANIFEST = Path("configs/corpora/swe-100.json")
_EVAL_MANIFEST = Path("configs/corpora/swe-277.json")


@dataclass(frozen=True)
class ProxyCall:
    sample: ResourceCallSample
    command: str
    clauses: tuple[Mapping[str, Any], ...]
    parse_failed: bool
    mapping_evidence: str
    segment_times_ms: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class ScoredRow:
    sample_id: str
    task_id: str
    repo: str
    command: str
    label_bucket: int
    predicted_bucket: int | None
    layer: str | None
    key_kind: str | None
    evidence_count: int
    unavailable_reason: str | None
    mapping_evidence: str


def _load_proxy_calls(
    root: Path, manifest: Path
) -> tuple[list[ProxyCall], list[str], int]:
    samples_by_task, task_ids = load_resource_corpus(root, manifest)
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
                )
            )
    return calls, task_ids, parse_failures


def _action_data_by_key(
    samples: Sequence[ResourceCallSample],
) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    by_trace: dict[str, list[ResourceCallSample]] = defaultdict(list)
    for sample in samples:
        by_trace[sample.source_trace].append(sample)
    output: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for trace_path, trace_samples in by_trace.items():
        actions = {
            (
                str(action["agent_id"]),
                int(action["iteration"]),
                str(action["action_id"]),
            ): action.get("data") or {}
            for action in TraceData.load(Path(trace_path)).actions
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


def _proxy_latency_observations(call: ProxyCall, repo: str) -> list[ClauseObservation]:
    """Legacy segment latency only; never attach proxy CPU or RSS."""

    if (
        call.parse_failed
        or call.mapping_evidence != "bin_exact"
        or call.sample.censored
    ):
        return []
    clauses = list(call.clauses)
    times = list(call.segment_times_ms)
    while len(clauses) > 1 and str(clauses[0]["bin"]) == "cd":
        clauses.pop(0)
        times.pop(0)
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
    kb: ClauseResourceKB,
    calls: Sequence[ProxyCall],
    buckets: LatencyBuckets,
) -> list[ScoredRow]:
    rows: list[ScoredRow] = []
    for call in sorted(
        calls, key=lambda item: (item.sample.tool_ts_start, item.sample.sample_id)
    ):
        repo = repo_of(call.sample.task_id)
        prediction = kb.predict_command_latency_bucket_from_clauses(
            repo,
            call.clauses,
            call.sample.tool_ts_start,
            buckets,
            command=call.command,
            parse_failed=call.parse_failed,
        )
        if not call.sample.censored:
            clause_prediction = prediction.prediction
            latency_ms = (call.sample.tool_ts_end - call.sample.tool_ts_start) * 1000.0
            rows.append(
                ScoredRow(
                    sample_id=call.sample.sample_id,
                    task_id=call.sample.task_id,
                    repo=repo,
                    command=call.command,
                    label_bucket=buckets.bucket_id(latency_ms),
                    predicted_bucket=(
                        None
                        if clause_prediction is None
                        else clause_prediction.bucket_id
                    ),
                    layer=(
                        None if clause_prediction is None else clause_prediction.scope
                    ),
                    key_kind=(
                        None
                        if clause_prediction is None
                        else clause_prediction.key_kind
                    ),
                    evidence_count=(
                        0
                        if clause_prediction is None
                        else clause_prediction.evidence_count
                    ),
                    unavailable_reason=prediction.unavailable_reason,
                    mapping_evidence=call.mapping_evidence,
                )
            )
        for observation in _proxy_latency_observations(call, repo):
            kb.observe_completed_clause(observation)
    return rows


def _metrics(rows: Sequence[ScoredRow], bucket_count: int) -> dict[str, Any]:
    known = [row for row in rows if row.predicted_bucket is not None]
    confusion = [[0] * bucket_count for _ in range(bucket_count)]
    for row in known:
        confusion[row.label_bucket][row.predicted_bucket] += 1
    correct = sum(row.label_bucket == row.predicted_bucket for row in known)
    abs_errors = [
        abs(row.label_bucket - row.predicted_bucket)
        for row in known
        if row.predicted_bucket is not None
    ]
    under = sum(row.predicted_bucket < row.label_bucket for row in known)
    over = sum(row.predicted_bucket > row.label_bucket for row in known)
    return {
        "eligible": len(rows),
        "known": len(known),
        "unknown": len(rows) - len(known),
        "coverage": len(known) / len(rows) if rows else None,
        "label_counts": [
            sum(row.label_bucket == bucket for row in rows)
            for bucket in range(bucket_count)
        ],
        "accuracy": correct / len(known) if known else None,
        "mean_abs_bucket_error": (
            sum(abs_errors) / len(abs_errors) if abs_errors else None
        ),
        "underestimate_rate": under / len(known) if known else None,
        "overestimate_rate": over / len(known) if known else None,
        "confusion": confusion,
        "unavailable_reasons": dict(
            sorted(
                Counter(
                    row.unavailable_reason
                    for row in rows
                    if row.unavailable_reason is not None
                ).items()
            )
        ),
    }


def evaluate(
    fit_calls: Sequence[ProxyCall],
    eval_calls: Sequence[ProxyCall],
    buckets: LatencyBuckets,
    provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], list[ScoredRow]]:
    observations = [
        observation
        for call in fit_calls
        for observation in _proxy_latency_observations(call, "public")
    ]
    rows = _score(ClauseResourceKB.fit_public(observations), eval_calls, buckets)
    return (
        {
            "status": "development_diagnostic_legacy_proxy_not_canonical_stage2",
            "claim_bearing": False,
            "objective": "latency_bucket_prediction",
            "bucket_edges_ms": list(buckets.edges_ms),
            "bucket_intervals": [
                {
                    "bucket_id": bucket,
                    "lower_ms": 0.0 if bucket == 0 else buckets.edges_ms[bucket - 1],
                    "upper_ms": (
                        buckets.edges_ms[bucket]
                        if bucket < len(buckets.edges_ms)
                        else None
                    ),
                }
                for bucket in range(buckets.bucket_count)
            ],
            "fit_clause_observation_count": len(observations),
            "eval_exec_call_count": len(eval_calls),
            "metrics": _metrics(rows, buckets.bucket_count),
            "provenance": dict(provenance),
        },
        rows,
    )


def _parse_bucket_edges(value: str) -> LatencyBuckets:
    try:
        edges = tuple(float(part) for part in value.split(","))
        return LatencyBuckets(edges)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-root", type=Path, default=_FIT_ROOT)
    parser.add_argument("--eval-root", type=Path, default=_EVAL_ROOT)
    parser.add_argument("--fit-manifest", type=Path, default=_FIT_MANIFEST)
    parser.add_argument("--eval-manifest", type=Path, default=_EVAL_MANIFEST)
    parser.add_argument(
        "--bucket-edges-ms",
        required=True,
        type=_parse_bucket_edges,
        help=(
            "Explicit comma-separated positive boundaries. No canonical numeric "
            "default exists; freeze an approved log-spaced grid before claims."
        ),
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dump-rows", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    fit_calls, fit_tasks, fit_parse_failures = _load_proxy_calls(
        args.fit_root, args.fit_manifest
    )
    eval_calls, eval_tasks, eval_parse_failures = _load_proxy_calls(
        args.eval_root, args.eval_manifest
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
        "edge_source": "explicit_cli_required",
    }
    result, rows = evaluate(fit_calls, eval_calls, args.bucket_edges_ms, provenance)
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
