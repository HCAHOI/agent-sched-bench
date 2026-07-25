#!/usr/bin/env python3
"""Evaluate ClauseResourceKB on the development-exposed legacy trace proxy."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tool_resource.features import parse_command_clauses, repo_of
from tool_resource.labels import ResourceCallSample, load_resource_corpus
from tool_resource.runtime_kb import (
    CLASSIFIER_TARGETS,
    FLAG_TARGETS,
    ClauseObservation,
    ClauseResourceKB,
    CommandPrediction,
)
from trace_collect.trace_data import TraceData

_FIT_ROOT = Path(
    "traces/swe-rebench/qwen3.7-max/"
    "offline-gated-confirm-100-v2-pacct-perbin"
)
_EVAL_ROOT = Path("traces/fresh-277-segtimeline")
_FIT_MANIFEST = Path("configs/corpora/swe-100.json")
_EVAL_MANIFEST = Path("configs/corpora/swe-277.json")


@dataclass(frozen=True)
class ProxyCall:
    sample: ResourceCallSample
    command: str
    clauses: tuple[Mapping[str, Any], ...]
    mapping_evidence: str


@dataclass(frozen=True)
class ScoredRow:
    target: str
    sample_id: str
    task_id: str
    repo: str
    command: str
    truth: bool
    prediction: bool | None
    layer: str
    key_kind: str
    evidence_count: int
    command_structure: str
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
            if parsed["parse_failed"]:
                parse_failures += 1
            clauses = tuple(parsed["clauses"])
            key = (sample.agent_id, sample.iteration, sample.action_id)
            calls.append(
                ProxyCall(
                    sample=sample,
                    command=command,
                    clauses=clauses,
                    mapping_evidence=_mapping_evidence(
                        clauses, action_data[key].get("segment_timeline")
                    ),
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


def _mapping_evidence(
    clauses: Sequence[Mapping[str, Any]], timeline: Any
) -> str:
    if not isinstance(timeline, Mapping):
        return "missing"
    segments = timeline.get("segments")
    if not isinstance(segments, list):
        return "malformed"
    runtime_bins: list[str] = []
    for segment in segments:
        if not isinstance(segment, Mapping) or not isinstance(
            segment.get("command_text"), str
        ):
            return "malformed"
        parsed = parse_command_clauses(segment["command_text"])
        if parsed["parse_failed"] or len(parsed["clauses"]) != 1:
            return "segment_parse_mismatch"
        runtime_bins.append(str(parsed["clauses"][0]["bin"]))
    static_bins = [str(clause["bin"]) for clause in clauses]
    if len(runtime_bins) != len(static_bins):
        return "count_mismatch"
    return "bin_exact" if runtime_bins == static_bins else "bin_mismatch"


def _proxy_observation(call: ProxyCall, repo: str) -> ClauseObservation | None:
    if (
        len(call.clauses) != 1
        or call.mapping_evidence != "bin_exact"
        or call.sample.censored
    ):
        return None
    clause = call.clauses[0]
    sample = call.sample
    return ClauseObservation(
        repo=repo,
        bin=str(clause["bin"]),
        argv=tuple(clause["argv"]),
        ts_start=sample.tool_ts_start,
        ts_end=sample.tool_ts_end,
        latency_ms=(sample.tool_ts_end - sample.tool_ts_start) * 1000.0,
        peak_cpu_cores=(
            sample.peak_cpu_cores if sample.peak_cpu_cores_eligible else None
        ),
        sampled_peak_rss_mb=(
            sample.peak_memory_mb if sample.peak_memory_mb_eligible else None
        ),
    )


def _validate_partition(
    fit_tasks: Sequence[str], eval_tasks: Sequence[str]
) -> dict[str, int]:
    fit = set(fit_tasks)
    evaluate = set(eval_tasks)
    overlap = fit & evaluate
    if overlap:
        raise ValueError(f"fit/eval task overlap: {sorted(overlap)[:3]}")
    return {
        "fit_task_count": len(fit),
        "eval_task_count": len(evaluate),
        "task_overlap_count": 0,
        "fit_repo_count": len({repo_of(task) for task in fit}),
        "eval_repo_count": len({repo_of(task) for task in evaluate}),
        "repo_overlap_count": len(
            {repo_of(task) for task in fit} & {repo_of(task) for task in evaluate}
        ),
    }


def _truth(call: ProxyCall, target: str) -> bool | None:
    sample = call.sample
    source, threshold = FLAG_TARGETS[target]
    if source == "latency_ms":
        value = (sample.tool_ts_end - sample.tool_ts_start) * 1000.0
        return None if sample.censored else value > threshold
    if source == "peak_cpu_cores":
        return (
            sample.peak_cpu_cores > threshold
            if sample.peak_cpu_cores_eligible and sample.peak_cpu_cores is not None
            else None
        )
    if source == "sampled_peak_rss_mb":
        return (
            sample.peak_memory_mb > threshold
            if sample.peak_memory_mb_eligible and sample.peak_memory_mb is not None
            else None
        )
    raise ValueError(f"unsupported target source: {source}")


def _provenance(
    prediction: CommandPrediction, target: str
) -> tuple[str, str, int]:
    flags = prediction.targets[target].clause_flags
    scopes = sorted({flag.scope for flag in flags if flag.scope is not None})
    kinds = sorted({flag.key_kind for flag in flags if flag.key_kind is not None})
    evidence = [flag.evidence_count for flag in flags if flag.evidence_count]
    return (
        "+".join(scopes) or "none",
        "+".join(kinds) or "none",
        min(evidence, default=0),
    )


def _structure(clauses: Sequence[Mapping[str, Any]]) -> str:
    if not clauses:
        return "empty"
    if len(clauses) == 1:
        return "single"
    if any(bool(clause.get("in_pipe")) for clause in clauses):
        return "pipeline"
    return "sequential"


def _score(
    kb: ClauseResourceKB, calls: Sequence[ProxyCall]
) -> list[ScoredRow]:
    rows: list[ScoredRow] = []
    for call in sorted(
        calls, key=lambda item: (item.sample.tool_ts_start, item.sample.sample_id)
    ):
        repo = repo_of(call.sample.task_id)
        prediction = kb.predict_command_from_clauses(
            repo,
            call.clauses,
            call.sample.tool_ts_start,
            command=call.command,
        )
        for target in CLASSIFIER_TARGETS:
            truth = _truth(call, target)
            if truth is None:
                continue
            layer, key_kind, evidence = _provenance(prediction, target)
            rows.append(
                ScoredRow(
                    target=target,
                    sample_id=call.sample.sample_id,
                    task_id=call.sample.task_id,
                    repo=repo,
                    command=call.command,
                    truth=truth,
                    prediction=prediction.targets[target].flag,
                    layer=layer,
                    key_kind=key_kind,
                    evidence_count=evidence,
                    command_structure=_structure(call.clauses),
                    mapping_evidence=call.mapping_evidence,
                )
            )
        observation = _proxy_observation(call, repo)
        if observation is not None:
            kb.observe_completed_clause(observation)
    return rows


def _metrics(rows: Sequence[ScoredRow]) -> dict[str, Any]:
    known = [row for row in rows if row.prediction is not None]
    tp = sum(row.truth and row.prediction is True for row in known)
    tn = sum(not row.truth and row.prediction is False for row in known)
    fp = sum(not row.truth and row.prediction is True for row in known)
    fn = sum(row.truth and row.prediction is False for row in known)
    positives = sum(row.truth for row in rows)
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {
        "eligible": len(rows),
        "positive": positives,
        "prevalence": positives / len(rows) if rows else None,
        "known": len(known),
        "unknown": len(rows) - len(known),
        "coverage": len(known) / len(rows) if rows else None,
        "balanced_accuracy": (
            (recall + specificity) / 2
            if recall is not None and specificity is not None
            else None
        ),
        "recall": recall,
        "precision": tp / (tp + fp) if tp + fp else None,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _evidence_bucket(count: int) -> str:
    if count == 0:
        return "0"
    if count == 1:
        return "1"
    if count < 5:
        return "2-4"
    if count < 20:
        return "5-19"
    return "20+"


def _bucket_metrics(
    rows: Sequence[ScoredRow], field: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[ScoredRow]] = defaultdict(list)
    for row in rows:
        value = getattr(row, field)
        key = _evidence_bucket(value) if field == "evidence_count" else str(value)
        grouped[key].append(row)
    return {key: _metrics(grouped[key]) for key in sorted(grouped)}


def evaluate(
    fit_calls: Sequence[ProxyCall],
    eval_calls: Sequence[ProxyCall],
    provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], list[ScoredRow]]:
    fit_observations = [
        observation
        for call in fit_calls
        if (observation := _proxy_observation(call, "public")) is not None
    ]
    kb = ClauseResourceKB.fit_public(fit_observations)
    rows = _score(kb, eval_calls)
    targets: dict[str, Any] = {}
    for target in CLASSIFIER_TARGETS:
        target_rows = [row for row in rows if row.target == target]
        targets[target] = {
            "overall": _metrics(target_rows),
            "buckets": {
                field: _bucket_metrics(target_rows, field)
                for field in (
                    "layer",
                    "key_kind",
                    "evidence_count",
                    "command_structure",
                    "mapping_evidence",
                )
            },
        }
    return (
        {
            "status": "diagnostic_legacy_proxy_not_canonical_stage2",
            "provenance": dict(provenance),
            "fit_clause_observation_count": len(fit_observations),
            "eval_exec_call_count": len(eval_calls),
            "targets": targets,
        },
        rows,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-root", type=Path, default=_FIT_ROOT)
    parser.add_argument("--eval-root", type=Path, default=_EVAL_ROOT)
    parser.add_argument("--fit-manifest", type=Path, default=_FIT_MANIFEST)
    parser.add_argument("--eval-manifest", type=Path, default=_EVAL_MANIFEST)
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
    partition = _validate_partition(fit_tasks, eval_tasks)
    provenance = {
        **partition,
        "fit_root": str(args.fit_root.resolve()),
        "eval_root": str(args.eval_root.resolve()),
        "fit_parse_failures": fit_parse_failures,
        "eval_parse_failures": eval_parse_failures,
        "primary": (
            "command-level latency/CPU/memory binary balanced accuracy; "
            "frozen per-clause three-valued OR"
        ),
        "proxy_adapter": {
            "latency_ms": "legacy replay command envelope on exact single-clause rows",
            "peak_cpu_cores": (
                "legacy cgroup peak; fit replay is pacct-on and approximately "
                "3% inflated, not Stage-2 per-clause peak_cpu_cores"
            ),
            "sampled_peak_rss_mb": (
                "legacy whole-container sampled peak memory, not Stage-2 "
                "per-clause sampled_peak_rss_mb"
            ),
        },
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
