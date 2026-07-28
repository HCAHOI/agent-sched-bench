#!/usr/bin/env python3
"""Evaluate clause latency buckets on canonical eBPF clause telemetry.

The former bash-xtrace proxy lane was removed once collection moved entirely to
eBPF clause telemetry; git history holds it if it is ever needed again.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

# Canonical clause telemetry is already loaded by the resource-class evaluator;
# reuse that loader rather than re-deriving the artifact shape here.
from scripts.evaluation.evaluate_clause_resource_classes import (  # noqa: E402
    Row,
    load_rows,
)
from tool_resource_eval.labels import repo_of  # noqa: E402
from tool_resource.runtime_kb import (  # noqa: E402
    CANONICAL_LATENCY_BUCKETS,
    ClauseResourceKB,
)


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


def _exact_bucket_metrics(rows: Sequence[ScoredRow]) -> dict[str, Any]:
    correct = sum(_exact_bucket_correct(row) for row in rows)
    return {
        "exact_bucket_accuracy": correct / len(rows) if rows else None,
        "eligible_examples": len(rows),
    }


def _argmax_bucket(row: ScoredRow) -> int:
    """Predicted bucket: highest probability, lowest bucket id on ties."""

    assert row.probability_by_bucket is not None
    return max(
        range(CANONICAL_LATENCY_BUCKETS.bucket_count),
        key=row.probability_by_bucket.__getitem__,
    )


def _exact_bucket_correct(row: ScoredRow) -> bool:
    return _argmax_bucket(row) == row.label_bucket


def _bucket_intervals() -> list[dict[str, Any]]:
    """Report the [0, b0], (b_{i-1}, b_i], tail semantics the KB implements."""

    edges = CANONICAL_LATENCY_BUCKETS.edges_ms
    return [
        {
            "bucket_id": bucket,
            "lower_ms": 0.0 if bucket == 0 else edges[bucket - 1],
            "lower_inclusive": bucket == 0,
            "upper_ms": edges[bucket] if bucket < len(edges) else None,
            "upper_inclusive": bucket < len(edges),
        }
        for bucket in range(CANONICAL_LATENCY_BUCKETS.bucket_count)
    ]


def _telemetry_scored_rows(
    fit_rows: Sequence[Row], eval_rows: Sequence[Row]
) -> list[ScoredRow]:
    """Score canonical clause telemetry with the runtime KB in causal task order.

    Public priors are fit leave-one-repo-out so the fit corpus never carries the
    evaluated repository. Within a task every clause is predicted before any of
    that task's observations settle, so there is no intra-task leakage; settled
    observations become repo-local evidence for later tasks only.
    """

    eval_repos = {row.repo for row in eval_rows}
    kb_by_repo = {
        repo: ClauseResourceKB.fit_public(
            row.observation(0.0, 1.0) for row in fit_rows if row.repo != repo
        )
        for repo in sorted(eval_repos)
    }
    by_task: dict[tuple[int, str], list[Row]] = defaultdict(list)
    for row in eval_rows:
        by_task[(row.manifest_index, row.task_id)].append(row)
    scored: list[ScoredRow] = []
    for task_ordinal, task_key in enumerate(sorted(by_task)):
        rows = by_task[task_key]
        # Query strictly before this task's observations settle; settle strictly
        # before the next task queries.
        query_ts = float(task_ordinal * 2 + 1)
        settle_ts = query_ts + 1.0
        for clause_index, row in enumerate(rows):
            # Raises when no evidence node exists; never falls back to synthetic.
            prediction = kb_by_repo[row.repo].predict_clause_latency_bucket(
                row.repo,
                row.bin,
                row.argv,
                CANONICAL_LATENCY_BUCKETS,
                ts_start=query_ts,
            )
            scored.append(
                ScoredRow(
                    sample_id=f"{row.task_id}:{row.manifest_index}:{clause_index}",
                    task_id=row.task_id,
                    repo=row.repo,
                    command=" ".join(row.argv),
                    label_bucket=CANONICAL_LATENCY_BUCKETS.bucket_id(row.latency_ms),
                    probability_by_bucket=prediction.probability_by_bucket,
                    layer=prediction.scope,
                    key_kind=prediction.key_kind,
                    evidence_count=prediction.evidence_count,
                    fallback_path=prediction.fallback_path,
                    unavailable_reason=None,
                    mapping_evidence="canonical_clause_telemetry",
                )
            )
        for row in rows:
            kb_by_repo[row.repo].observe_completed_clause(
                row.observation(query_ts, settle_ts)
            )
    if not scored:
        raise ValueError("no eligible clause telemetry observations to score")
    return scored


def _telemetry_metrics(rows: Sequence[ScoredRow]) -> dict[str, Any]:
    bucket_count = CANONICAL_LATENCY_BUCKETS.bucket_count
    label_counts: Counter[int] = Counter(row.label_bucket for row in rows)
    majority_bucket, majority_count = min(
        label_counts.most_common(),
        key=lambda item: (-item[1], item[0]),
    )
    confusion = [[0] * bucket_count for _ in range(bucket_count)]
    predicted_counts: Counter[int] = Counter()
    for row in rows:
        predicted = _argmax_bucket(row)
        confusion[row.label_bucket][predicted] += 1
        predicted_counts[predicted] += 1
    per_bucket = []
    for bucket in range(bucket_count):
        support = label_counts.get(bucket, 0)
        predicted_total = predicted_counts.get(bucket, 0)
        hit = confusion[bucket][bucket]
        per_bucket.append(
            {
                "bucket_id": bucket,
                "label_count": support,
                "label_share": support / len(rows),
                "predicted_count": predicted_total,
                "correct": hit,
                "recall": hit / support if support else None,
                "precision": hit / predicted_total if predicted_total else None,
            }
        )
    return {
        **_exact_bucket_metrics(rows),
        "majority_bucket": majority_bucket,
        "majority_bucket_baseline_accuracy": majority_count / len(rows),
        "confusion_label_by_prediction": confusion,
        "per_bucket": per_bucket,
        "scope_counts": dict(sorted(Counter(row.layer for row in rows).items())),
        "key_kind_counts": dict(sorted(Counter(row.key_kind for row in rows).items())),
        "fallback_path_counts": dict(
            sorted(Counter(":".join(row.fallback_path or ()) for row in rows).items())
        ),
    }


def evaluate_clause_telemetry(
    fit_rows: Sequence[Row],
    eval_rows: Sequence[Row],
    provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], list[ScoredRow]]:
    """Evaluate latency buckets on canonical eBPF clause telemetry."""

    rows = _telemetry_scored_rows(fit_rows, eval_rows)
    return (
        {
            "status": "development_exposed_canonical_clause_telemetry",
            "claim_bearing": False,
            "objective": "clause_latency_bucket_prediction",
            "bucket_edges_ms": list(CANONICAL_LATENCY_BUCKETS.edges_ms),
            "bucket_intervals": _bucket_intervals(),
            "fit_clause_observation_count": len(fit_rows),
            "eval_clause_observation_count": len(eval_rows),
            "metrics": _telemetry_metrics(rows),
            "provenance": dict(provenance),
        },
        rows,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # Canonical eBPF clause telemetry: one aggregated replay JSONL per corpus,
    # the same artifact evaluate_clause_resource_classes.py consumes.
    parser.add_argument("--telemetry-fit", type=Path, required=True)
    parser.add_argument("--telemetry-eval", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dump-rows", type=Path)
    return parser


def _write(args: argparse.Namespace, result: Mapping[str, Any], rows: Sequence[Any]) -> None:
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


def _run_clause_telemetry(args: argparse.Namespace) -> None:
    fit_rows = load_rows(args.telemetry_fit)
    eval_rows = load_rows(args.telemetry_eval)
    fit_tasks = sorted({row.task_id for row in fit_rows})
    eval_tasks = sorted({row.task_id for row in eval_rows})
    provenance = {
        **_validate_partition(fit_tasks, eval_tasks),
        "fit_telemetry": str(args.telemetry_fit.resolve()),
        "eval_telemetry": str(args.telemetry_eval.resolve()),
        "evidence_source": (
            "canonical eBPF clause telemetry; eligible_for_kb call and clause gates"
        ),
        "scoring_unit": "kb_eligible_clause",
        "public_prior": "leave_one_repo_out over the fit corpus",
        "task_order": "(manifest_index, task_id) ascending",
        "trace_update": (
            "predict every clause of a task before any of that task's observations "
            "settle; settled clauses are repo-local evidence for later tasks only"
        ),
        "unavailable_policy": (
            "missing evidence raises; no synthetic, imputed, or majority fallback"
        ),
        "edge_source": "canonical_objective",
    }
    result, rows = evaluate_clause_telemetry(fit_rows, eval_rows, provenance)
    _write(args, result, rows)


def main() -> None:
    _run_clause_telemetry(_parser().parse_args())


if __name__ == "__main__":
    main()
