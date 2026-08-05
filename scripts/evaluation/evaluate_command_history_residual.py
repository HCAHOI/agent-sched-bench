#!/usr/bin/env python3
"""Fit and score the frozen generic command-history residual calibrator."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from scripts.evaluation.evaluate_clause_latency_buckets import (  # noqa: E402
    _accuracy_delta,
    _phase_changes,
    _sidecar_hard_metrics,
)
from tool_resource.clause_parser import parse_command_clauses  # noqa: E402
from tool_resource.runtime_kb import (  # noqa: E402
    CANONICAL_RESOURCE_BUCKET_EDGES,
    RESOURCE_BUCKET_LABELS,
    _structured_argv_parts,
)

VERSION = "command-history-residual-v1"
TARGETS = ("latency", "peak_cpu_cores", "sampled_peak_rss_mb")
DISK = "disk_read_write_bytes_total"
BUCKETS = {"latency": 5, **{target: 3 for target in CANONICAL_RESOURCE_BUCKET_EDGES}}
HASH_DIM = 256
PMF_FLOOR = 1e-6
L2 = 1e-4
SPLIT_MANIFEST = _ROOT / "analysis/development/sqlglot-relational-task-split.json"
FROZEN_FIT_ROWS = (
    _ROOT
    / "analysis/results/tool-resource-5-3-3-3-20260804"
    / "sqlglot20-80-current-fit-v1/rows.jsonl"
)
FROZEN_VALIDATION_ROWS = (
    _ROOT
    / "analysis/results/tool-resource-5-3-3-3-20260804"
    / "sqlglot50-full-test-phase-validation-v1/rows.jsonl"
)


@dataclass(frozen=True)
class Row:
    sample_id: str
    task_id: str
    command: str
    labels: Mapping[str, int | None]
    current: Mapping[str, Any]
    pmfs: Mapping[str, tuple[float, ...] | None]


@dataclass(frozen=True)
class Shape:
    vector: tuple[float, ...]
    executables: tuple[str, ...]


def _load_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            value = json.loads(raw)
            sample_id = value.get("sample_id")
            task_id = value.get("task_id")
            command = value.get("command")
            current_value = value.get("current_dynamic")
            if not all(isinstance(item, str) for item in (sample_id, task_id, command)):
                raise ValueError(f"{path}: row lacks string identity")
            if sample_id in seen or not isinstance(current_value, dict):
                raise ValueError(f"{path}: duplicate row or missing Current")
            seen.add(sample_id)
            current = dict(current_value)
            nested_pmfs = current.pop("probability_by_bucket", None)
            pmfs_value = value.get("current_probability_by_bucket", nested_pmfs)
            labels_value = value.get("labels")
            if labels_value is None:
                labels_value = {
                    "latency": value.get("latency_label"),
                    **dict(value.get("resource_labels") or {}),
                }
            if not isinstance(pmfs_value, dict) or not isinstance(labels_value, dict):
                raise ValueError(f"{path}: row lacks PMFs or labels")
            pmfs: dict[str, tuple[float, ...] | None] = {}
            labels: dict[str, int | None] = {}
            for target, buckets in BUCKETS.items():
                pmf = pmfs_value.get(target)
                label = labels_value.get(target)
                if label is not None and (
                    not isinstance(label, int) or not 0 <= label < buckets
                ):
                    raise ValueError(f"{path}: invalid {target} PMF or label")
                if pmf is None:
                    if current.get(target) is not None:
                        raise ValueError(f"{path}: {target} PMF missing for a hard prediction")
                    pmfs[target] = None
                elif (
                    not isinstance(pmf, list)
                    or len(pmf) != buckets
                    or any(not isinstance(item, (int, float)) or item < 0 for item in pmf)
                    or not math.isclose(sum(pmf), 1.0, abs_tol=1e-9)
                ):
                    raise ValueError(f"{path}: invalid {target} PMF or label")
                else:
                    pmfs[target] = tuple(float(item) for item in pmf)
                labels[target] = label
            rows.append(Row(sample_id, task_id, command, labels, current, pmfs))
    if not rows:
        raise ValueError(f"{path}: no rows")
    task_order = list(dict.fromkeys(row.task_id for row in rows))
    for task_id in task_order:
        indices = [index for index, row in enumerate(rows) if row.task_id == task_id]
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError(f"{path}: task rows are not contiguous")
    return rows


def _bounded(value: int) -> float:
    return min(10.0, math.log1p(value))


def _hash(features: Sequence[str]) -> list[float]:
    vector = [0.0] * HASH_DIM
    for feature in features:
        digest = int.from_bytes(
            hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little"
        )
        vector[digest % HASH_DIM] += 1.0 if digest >> 63 else -1.0
    return vector


def command_shape(command: str) -> Shape:
    parsed = parse_command_clauses(command)
    clauses = parsed.get("clauses") if isinstance(parsed, dict) else None
    clauses = clauses if isinstance(clauses, list) else []
    categorical = [f"parse_failed:{bool(parsed.get('parse_failed'))}"]
    executables: list[str] = []
    argc = positionals = loop = pipe = subst = 0
    for clause in clauses:
        if not isinstance(clause, dict):
            continue
        argv = clause.get("argv")
        if not isinstance(argv, list) or not argv:
            continue
        executable = Path(str(clause.get("bin") or argv[0])).name
        executables.append(executable)
        categorical.append(f"bin:{executable}")
        _subcommand, options, shaped_positionals = _structured_argv_parts(
            tuple(str(item) for item in argv)
        )
        categorical.extend(
            f"option:{option.split('=', 1)[0]}" for option in options
        )
        categorical.append(f"positional_slots:{len(shaped_positionals)}")
        argc += len(argv)
        positionals += len(shaped_positionals)
        loop += clause.get("in_loop") is True
        pipe += clause.get("in_pipe") is True
        subst += clause.get("in_subst") is True
    edges = parsed.get("control_edges", []) if isinstance(parsed, dict) else []
    for edge in edges if isinstance(edges, list) else []:
        if isinstance(edge, dict):
            categorical.append(f"edge:{edge.get('operator', '<UNKNOWN>')}")
    numeric = [
        _bounded(len(command.encode())),
        _bounded(len(clauses)),
        _bounded(argc),
        _bounded(positionals),
        _bounded(loop),
        _bounded(pipe),
        _bounded(subst),
        _bounded(len(edges) if isinstance(edges, list) else 0),
    ]
    return Shape(
        tuple([*_hash(categorical), *numeric]), tuple(sorted(set(executables)))
    )


def _history(values: Sequence[int], buckets: int) -> list[float]:
    counts = Counter(values)
    total = len(values)
    shares = [counts[bucket] / total if total else 0.0 for bucket in range(buckets)]
    last = [float(bool(values) and values[-1] == bucket) for bucket in range(buckets)]
    return [_bounded(total), *shares, *last]


def dataset(rows: Sequence[Row], target: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    buckets = BUCKETS[target]
    features: list[list[float]] = []
    bases: list[tuple[float, ...]] = []
    labels: list[int] = []
    current_task: str | None = None
    all_history: list[int] = []
    by_executable: dict[tuple[str, ...], list[int]] = defaultdict(list)
    by_exact: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        if row.task_id != current_task:
            current_task = row.task_id
            all_history = []
            by_executable = defaultdict(list)
            by_exact = defaultdict(list)
        shape = command_shape(row.command)
        features.append(
            [
                *shape.vector,
                *_history(all_history, buckets),
                *_history(by_executable[shape.executables], buckets),
                *_history(by_exact[row.command], buckets),
            ]
        )
        bases.append(
            row.pmfs[target]
            if row.pmfs[target] is not None
            else tuple(1.0 / buckets for _ in range(buckets))
        )
        label = row.labels[target]
        labels.append(-1 if label is None else label)
        if label is not None:
            all_history.append(label)
            by_executable[shape.executables].append(label)
            by_exact[row.command].append(label)
    return (
        torch.tensor(features, dtype=torch.float64),
        torch.tensor(bases, dtype=torch.float64),
        torch.tensor(labels, dtype=torch.int64),
    )


def fit_model(
    features: torch.Tensor, bases: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    eligible = labels >= 0
    x, base, y = features[eligible], bases[eligible], labels[eligible]
    buckets, width = base.shape[1], x.shape[1]
    weight = torch.zeros((buckets, width), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(buckets, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weight, bias],
        lr=1.0,
        max_iter=200,
        history_size=20,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )

    def loss() -> torch.Tensor:
        logits = base.clamp_min(PMF_FLOOR).log() + x @ weight.T + bias
        return F.cross_entropy(logits, y) + L2 * weight.square().mean()

    initial = float(loss().detach())

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        value = loss()
        value.backward()
        return value

    optimizer.step(closure)
    final = loss().detach()
    if (
        not torch.isfinite(weight).all()
        or not torch.isfinite(bias).all()
        or not torch.isfinite(final)
    ):
        raise FloatingPointError("residual fit produced non-finite parameters")
    return weight.detach(), bias.detach(), initial, float(final)


def _predict(
    features: torch.Tensor,
    bases: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    pmfs = torch.softmax(
        bases.clamp_min(PMF_FLOOR).log() + features @ weight.T + bias, dim=1
    )
    if not torch.isfinite(pmfs).all() or not torch.allclose(
        pmfs.sum(dim=1), torch.ones(len(pmfs), dtype=torch.float64), atol=1e-9
    ):
        raise FloatingPointError("residual prediction produced an invalid PMF")
    return pmfs


def _fail_closed_metrics(
    rows: Sequence[Mapping[str, Any]], target: str
) -> dict[str, Any]:
    metric = _sidecar_hard_metrics(rows, target)
    confusion = metric["confusion_label_by_prediction"]
    eligible_key = "eligible_examples" if target == "latency" else "eligible_n"
    accuracy_key = "exact_class_accuracy" if target == "latency" else "accuracy"
    eligible = metric[eligible_key]
    correct = sum(confusion[index][index] for index in range(len(confusion)))
    within_one = sum(
        count
        for label, predictions in enumerate(confusion)
        for prediction, count in enumerate(predictions)
        if abs(label - prediction) <= 1
    )
    severe = metric["prediction_unavailable"] + sum(
        count
        for label, predictions in enumerate(confusion)
        for prediction, count in enumerate(predictions)
        if label - prediction >= 2
    )
    return {
        **metric,
        accuracy_key: correct / eligible,
        "within_one_bucket_accuracy": within_one / eligible,
        "severe_underprediction_rate": severe / eligible,
    }


def run(fit_rows: Sequence[Row], validation_rows: Sequence[Row]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    models: dict[str, Any] = {}
    predictions: dict[str, torch.Tensor] = {}
    fit_started = time.perf_counter()
    for target in TARGETS:
        fit_x, fit_base, fit_y = dataset(fit_rows, target)
        weight, bias, initial, final = fit_model(fit_x, fit_base, fit_y)
        validation_x, validation_base, _validation_y = dataset(validation_rows, target)
        predictions[target] = _predict(validation_x, validation_base, weight, bias)
        models[target] = {
            "training_rows": int((fit_y >= 0).sum()),
            "feature_width": fit_x.shape[1],
            "initial_loss": initial,
            "final_loss": final,
            "weight": weight.tolist(),
            "bias": bias.tolist(),
        }
    fit_seconds = time.perf_counter() - fit_started

    predict_started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(validation_rows):
        candidate = dict(row.current)
        candidate_pmfs = {
            target: None if pmf is None else list(pmf)
            for target, pmf in row.pmfs.items()
        }
        for target in TARGETS:
            pmf = predictions[target][index].tolist()
            hard = max(range(len(pmf)), key=pmf.__getitem__)
            candidate[target] = hard if target == "latency" else RESOURCE_BUCKET_LABELS[hard]
            candidate_pmfs[target] = pmf
        rows.append(
            {
                "sample_id": row.sample_id,
                "task_id": row.task_id,
                "command": row.command,
                "labels": dict(row.labels),
                "current_dynamic": dict(row.current),
                "current_probability_by_bucket": {
                    target: None if pmf is None else list(pmf)
                    for target, pmf in row.pmfs.items()
                },
                "candidate": candidate,
                "candidate_probability_by_bucket": candidate_pmfs,
            }
        )
    prediction_seconds = time.perf_counter() - predict_started

    all_targets = (*TARGETS, DISK)
    metrics = {target: _fail_closed_metrics(rows, target) for target in all_targets}
    current_rows = [
        {
            **row,
            "candidate": row["current_dynamic"],
            "candidate_probability_by_bucket": row["current_probability_by_bucket"],
        }
        for row in rows
    ]
    current = {
        target: _fail_closed_metrics(current_rows, target) for target in all_targets
    }
    changes = {target: _phase_changes(rows, target) for target in all_targets}
    deltas = {
        target: _accuracy_delta(
            metrics[target]["exact_class_accuracy" if target == "latency" else "accuracy"],
            current[target]["exact_class_accuracy" if target == "latency" else "accuracy"],
        )
        for target in all_targets
    }
    helpful = sum(changes[target]["helpful"] for target in TARGETS)
    harmful = sum(changes[target]["harmful"] for target in TARGETS)
    helpful_tasks = {
        task_id
        for target in TARGETS
        for task_id in changes[target]["helpful_task_ids"]
    }
    severe_ok = all(
        metrics[target]["severe_underprediction_rate"] is not None
        and current[target]["severe_underprediction_rate"] is not None
        and metrics[target]["severe_underprediction_rate"]
        <= current[target]["severe_underprediction_rate"]
        for target in TARGETS
    )
    disk_identical = all(
        row["candidate"][DISK] == row["current_dynamic"][DISK]
        and row["candidate_probability_by_bucket"][DISK]
        == row["current_probability_by_bucket"][DISK]
        for row in rows
    )
    gain_ok = all(deltas[target] is not None and deltas[target] >= 5.0 for target in TARGETS)
    go = gain_ok and severe_ok and helpful > harmful and len(helpful_tasks) >= 10 and disk_identical
    return {
        "schema": VERSION,
        "status": "development_validation_go" if go else "development_validation_no_go",
        "claim_bearing": False,
        "config": {
            "hash": "signed-blake2b-64",
            "hash_dimensions": HASH_DIM,
            "pmf_floor": PMF_FLOOR,
            "l2": L2,
            "optimizer": "torch-float64-lbfgs-strong-wolfe",
            "seed": 0,
            "disk_policy": "bit_identical_current",
            "current_unavailable_policy": "uniform_candidate_base_and_fail_closed_error",
        },
        "models": models,
        "targets": {
            target: {
                "current": current[target],
                "candidate": metrics[target],
                "delta_percentage_points": deltas[target],
                "changes": changes[target],
            }
            for target in all_targets
        },
        "gate": {
            "go": go,
            "latency_cpu_rss_meet_gain": gain_ok,
            "minimum_gain_percentage_points_each": 5.0,
            "no_severe_underprediction_regression": severe_ok,
            "helpful": helpful,
            "harmful": harmful,
            "helpful_tasks": len(helpful_tasks),
            "minimum_helpful_tasks": 10,
            "disk_bit_identical": disk_identical,
        },
        "cost": {
            "fit_and_tensor_prediction_seconds": fit_seconds,
            "row_materialization_seconds": prediction_seconds,
            "validation_rows": len(rows),
        },
    }, rows


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-rows", type=Path, required=True)
    parser.add_argument("--validation-rows", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("output directory already exists")
    if (
        args.fit_rows.resolve() != FROZEN_FIT_ROWS.resolve()
        or args.validation_rows.resolve() != FROZEN_VALIDATION_ROWS.resolve()
    ):
        raise ValueError("row paths differ from the frozen development protocol")
    fit_rows = _load_rows(args.fit_rows)
    validation_rows = _load_rows(args.validation_rows)
    fit_tasks = list(dict.fromkeys(row.task_id for row in fit_rows))
    validation_tasks = list(dict.fromkeys(row.task_id for row in validation_rows))
    split = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    if (
        len(fit_rows) != 1420
        or len(fit_tasks) != 80
        or len(validation_rows) != 1044
        or len(validation_tasks) != 50
        or fit_tasks != split.get("development", [])[20:]
        or validation_tasks != split.get("validation")
    ):
        raise ValueError("rows differ from the frozen 80-fit/50-validation protocol")
    result, rows = run(fit_rows, validation_rows)
    result["inputs"] = {
        "fit_rows": str(args.fit_rows.resolve()),
        "validation_rows": str(args.validation_rows.resolve()),
        "git_sha": _git_sha(),
    }
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
