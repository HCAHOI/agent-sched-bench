#!/usr/bin/env python3
"""Evaluate causal completed-call adaptation for peak-CPU classification."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import gzip
import heapq
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tool_resource.labels import load_resource_corpus  # noqa: E402
from tool_time.command import shell_command_prefix_tokens  # noqa: E402
from tool_time.statistics import resample_task_totals  # noqa: E402


_DEFAULT_TRACE_ROOT = Path("traces/swe-rebench/qwen3.7-max/fresh-seed42-skip150-n200")
_DEFAULT_MANIFEST = Path("configs/corpora/swe-277.json")
_THRESHOLD_CORES = 2.0
_BOOTSTRAP_REPLICATES = 50_000
_SEED = 0
_CONFIDENCE_LEVEL = 0.95
_REPO_SUFFIX = re.compile(r"-\d+$")
_TIERS = ("a", "b", "c", "cold")


@dataclass(frozen=True)
class CpuRow:
    sample_id: str
    task_id: str
    repo: str
    command: str
    observed: float
    label: bool
    static_label: bool
    tool_ts_start: float
    tool_ts_end: float


@dataclass(frozen=True)
class Decision:
    row: CpuRow
    tier: str
    blended_label: bool


def repo_key(task_id: str) -> str:
    """Strip only a trailing numeric instance suffix."""

    return _REPO_SUFFIX.sub("", task_id)


def command_head(command: str) -> tuple[str, ...]:
    """Return the first two normalized tokens after leading ``cd X &&``."""

    return tuple(shell_command_prefix_tokens(command, skip_leading_cd=True)[:2])


def load_joined_rows(
    dump_path: Path,
    trace_root: Path,
    manifest: Path,
) -> tuple[list[CpuRow], dict[str, int]]:
    """Load CPU dump rows and attach every timestamp by exact sample ID."""

    dumped = _load_cpu_dump(dump_path)
    samples_by_task, task_ids = load_resource_corpus(trace_root, manifest)
    timestamps: dict[str, tuple[str, float, float]] = {}
    for task_id in task_ids:
        for sample in samples_by_task[task_id]:
            if sample.sample_id in timestamps:
                raise ValueError(f"duplicate corpus sample_id: {sample.sample_id}")
            timestamps[sample.sample_id] = (
                sample.task_id,
                sample.tool_ts_start,
                sample.tool_ts_end,
            )

    missing = [row["sample_id"] for row in dumped if row["sample_id"] not in timestamps]
    if missing:
        raise ValueError(
            f"timestamp join failed for {len(missing)}/{len(dumped)} CPU rows; "
            f"first missing sample_id: {missing[0]}"
        )

    rows: list[CpuRow] = []
    for row in dumped:
        sample_task_id, start, end = timestamps[row["sample_id"]]
        if sample_task_id != row["task_id"]:
            raise ValueError(
                f"{row['sample_id']}: dump task_id {row['task_id']!r} "
                f"does not match corpus task_id {sample_task_id!r}"
            )
        rows.append(
            CpuRow(
                **row,
                tool_ts_start=start,
                tool_ts_end=end,
            )
        )
    return rows, {
        "cpu_dump_row_count": len(dumped),
        "joined_row_count": len(rows),
        "corpus_call_count": len(timestamps),
    }


def prequential_decisions(rows: Sequence[CpuRow]) -> list[Decision]:
    """Blend static labels with the latest strictly completed matching call."""

    pending: list[tuple[float, str, CpuRow]] = []
    task_exact: dict[tuple[str, str], CpuRow] = {}
    repo_exact: dict[tuple[str, str], CpuRow] = {}
    repo_head: dict[tuple[str, tuple[str, ...]], CpuRow] = {}
    decisions: list[Decision] = []

    for row in sorted(rows, key=lambda item: (item.tool_ts_start, item.sample_id)):
        while pending and pending[0][0] < row.tool_ts_start:
            _, _, completed = heapq.heappop(pending)
            task_exact[(completed.task_id, completed.command)] = completed
            repo_exact[(completed.repo, completed.command)] = completed
            head = command_head(completed.command)
            if head:
                repo_head[(completed.repo, head)] = completed

        candidate = task_exact.get((row.task_id, row.command))
        tier = "a"
        if candidate is None:
            candidate = repo_exact.get((row.repo, row.command))
            tier = "b"
        if candidate is None:
            head = command_head(row.command)
            candidate = repo_head.get((row.repo, head)) if head else None
            tier = "c"
        if candidate is None:
            blended_label = row.static_label
            tier = "cold"
        else:
            blended_label = candidate.observed > _THRESHOLD_CORES
        decisions.append(Decision(row=row, tier=tier, blended_label=blended_label))
        heapq.heappush(pending, (row.tool_ts_end, row.sample_id, row))
    return decisions


def repo_clustered_uncertainty(
    decisions: Sequence[Decision],
    *,
    replicates: int = _BOOTSTRAP_REPLICATES,
    seed: int = _SEED,
) -> dict[str, Any]:
    """Bootstrap paired blended/static confusion counts by repository."""

    by_repo: dict[str, list[Decision]] = defaultdict(list)
    for decision in decisions:
        by_repo[decision.row.repo].append(decision)
    keys = sorted(by_repo)
    contributions = np.vstack(
        [
            np.concatenate(
                [
                    _confusion_counts(
                        [decision.row.label for decision in by_repo[key]],
                        [decision.blended_label for decision in by_repo[key]],
                    ),
                    _confusion_counts(
                        [decision.row.label for decision in by_repo[key]],
                        [decision.row.static_label for decision in by_repo[key]],
                    ),
                ]
            )
            for key in keys
        ]
    )
    totals = resample_task_totals(
        contributions,
        replicates=replicates,
        seed=seed,
    )
    blended_draws = _balanced_accuracy_draws(totals, 0)
    static_draws = _balanced_accuracy_draws(totals, 4)
    difference_draws = blended_draws - static_draws
    point_totals = contributions.sum(axis=0, keepdims=True)
    blended_point = float(_balanced_accuracy_draws(point_totals, 0)[0])
    static_point = float(_balanced_accuracy_draws(point_totals, 4)[0])

    return {
        "method": "repo-clustered confusion-count bootstrap",
        "cluster_count": len(keys),
        "confidence_level": _CONFIDENCE_LEVEL,
        "bootstrap_replicates": replicates,
        "seed": seed,
        "blended": _interval(blended_draws, blended_point),
        "blended_minus_static": _interval(
            difference_draws, blended_point - static_point
        ),
    }


def evaluate(rows: Sequence[CpuRow]) -> dict[str, Any]:
    """Return the registered point estimates, tier diagnostics, and CIs."""

    if not rows:
        raise ValueError("no peak_cpu_cores rows supplied")
    decisions = prequential_decisions(rows)
    overall = _decision_metrics(decisions, coverage=1.0)
    by_tier = {
        tier: _decision_metrics(
            [decision for decision in decisions if decision.tier == tier],
            coverage=sum(decision.tier == tier for decision in decisions)
            / len(decisions),
        )
        for tier in _TIERS
    }
    uncertainty = repo_clustered_uncertainty(decisions)
    ci_low = uncertainty["blended_minus_static"]["ci_low"]
    return {
        "threshold_cores": _THRESHOLD_CORES,
        "overall": overall,
        "by_match_tier": by_tier,
        "repo_clustered_bootstrap": uncertainty,
        "registered_criterion": {
            "definition": ("blended-minus-static BA repo-clustered CI lower bound > 0"),
            "evaluated": True,
            "passed": bool(ci_low > 0.0),
        },
    }


def _load_cpu_dump(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            raw = json.loads(line)
            if raw.get("target") != "peak_cpu_cores":
                continue
            sample_id = _text(raw.get("sample_id"), f"line {line_number} sample_id")
            if sample_id in seen:
                raise ValueError(f"duplicate CPU sample_id: {sample_id}")
            seen.add(sample_id)
            task_id = _text(raw.get("task_id"), f"{sample_id} task_id")
            command = _text(raw.get("command"), f"{sample_id} command")
            observed = _number(raw.get("observed"), f"{sample_id} observed")
            threshold = raw["thresholds"]["2"]
            label = _binary(threshold["binary_label"], f"{sample_id} binary_label")
            static_label = _binary(
                threshold["model_predicted_label"],
                f"{sample_id} model_predicted_label",
            )
            if label != (observed > _THRESHOLD_CORES):
                raise ValueError(
                    f"{sample_id}: binary_label disagrees with observed > 2"
                )
            computed_repo = repo_key(task_id)
            if raw.get("repo") != computed_repo:
                raise ValueError(
                    f"{sample_id}: dump repo {raw.get('repo')!r} "
                    f"does not match task-derived repo {computed_repo!r}"
                )
            rows.append(
                {
                    "sample_id": sample_id,
                    "task_id": task_id,
                    "repo": computed_repo,
                    "command": command,
                    "observed": observed,
                    "label": label,
                    "static_label": static_label,
                }
            )
    if not rows:
        raise ValueError("dump contains no peak_cpu_cores rows")
    return rows


def _decision_metrics(
    decisions: Sequence[Decision],
    *,
    coverage: float,
) -> dict[str, Any]:
    labels = [decision.row.label for decision in decisions]
    blended = [decision.blended_label for decision in decisions]
    static = [decision.row.static_label for decision in decisions]
    blended_metrics = _classification_metrics(labels, blended)
    static_metrics = _classification_metrics(labels, static)
    blended_ba = blended_metrics["balanced_accuracy"]
    static_ba = static_metrics["balanced_accuracy"]
    return {
        "row_count": len(decisions),
        "row_coverage": coverage,
        "blended": blended_metrics,
        "static": static_metrics,
        "blended_minus_static_balanced_accuracy": (
            blended_ba - static_ba
            if blended_ba is not None and static_ba is not None
            else None
        ),
    }


def _classification_metrics(
    labels: Sequence[bool],
    predictions: Sequence[bool],
) -> dict[str, Any]:
    counts = _confusion_counts(labels, predictions)
    balanced_accuracy = _balanced_accuracy(counts)
    return {
        "balanced_accuracy": balanced_accuracy,
        "predicted_heavy_count": int(counts[0] + counts[3]),
        "confusion": {
            "true_positive": int(counts[0]),
            "false_negative": int(counts[1]),
            "true_negative": int(counts[2]),
            "false_positive": int(counts[3]),
        },
    }


def _confusion_counts(
    labels: Sequence[bool],
    predictions: Sequence[bool],
) -> np.ndarray:
    truth = np.asarray(labels, dtype=bool)
    predicted = np.asarray(predictions, dtype=bool)
    return np.asarray(
        [
            np.sum(truth & predicted),
            np.sum(truth & ~predicted),
            np.sum(~truth & ~predicted),
            np.sum(~truth & predicted),
        ],
        dtype=float,
    )


def _balanced_accuracy(counts: np.ndarray) -> float | None:
    positive = counts[0] + counts[1]
    negative = counts[2] + counts[3]
    if positive == 0.0 or negative == 0.0:
        return None
    return float(0.5 * (counts[0] / positive + counts[2] / negative))


def _balanced_accuracy_draws(values: np.ndarray, start: int) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return 0.5 * (
            values[:, start] / (values[:, start] + values[:, start + 1])
            + values[:, start + 2] / (values[:, start + 2] + values[:, start + 3])
        )


def _interval(draws: np.ndarray, point: float) -> dict[str, float]:
    finite = draws[np.isfinite(draws)]
    if not finite.size:
        raise ValueError("all bootstrap balanced-accuracy draws are undefined")
    alpha = 1.0 - _CONFIDENCE_LEVEL
    low, high = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"point": point, "ci_low": float(low), "ci_high": float(high)}


def _text(value: Any, source: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source} must be non-empty text")
    return value


def _number(value: Any, source: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{source} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{source} must be finite and non-negative")
    return number


def _binary(value: Any, source: str) -> bool:
    if isinstance(value, bool):
        return value
    if (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and value in (0, 1)
    ):
        return bool(value)
    raise ValueError(f"{source} must be binary")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, default=_DEFAULT_TRACE_ROOT)
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    args = parser.parse_args()

    rows, join = load_joined_rows(args.rows, args.trace_root, args.manifest)
    result = evaluate(rows)
    result["inputs"] = {
        "rows": str(args.rows),
        "trace_root": str(args.trace_root),
        "manifest": str(args.manifest),
    }
    result["timestamp_join"] = join
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
