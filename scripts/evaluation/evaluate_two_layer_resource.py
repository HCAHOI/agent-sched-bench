#!/usr/bin/env python3
"""Evaluate a causal repo/public resource backoff lattice."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import heapq
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from scripts.evaluation.evaluate_prequential_binary import (  # noqa: E402
    CpuRow,
    Decision,
    _classification_metrics,
    _load_cpu_dump,
    prequential_decisions,
    repo_clustered_uncertainty,
    repo_key,
)
from scripts.evaluation.evaluate_resource_prediction import (  # noqa: E402
    _clustered_skill,
)
from tool_resource.labels import ResourceCallSample, load_resource_corpus  # noqa: E402
from tool_resource.metrics import ecdf_quantile, pinball_loss  # noqa: E402
from tool_time.command import make_row_command_prefix_keys  # noqa: E402


_DEFAULT_FIT_ROOT = Path("traces/swe-rebench/qwen3.7-max/offline-gated-confirm-100-v2")
_DEFAULT_FIT_MANIFEST = Path("configs/corpora/swe-100.json")
_DEFAULT_EVAL_ROOT = Path("traces/swe-rebench/qwen3.7-max/fresh-seed42-skip150-n200")
_DEFAULT_EVAL_MANIFEST = Path("configs/corpora/swe-277.json")
_CPU_THRESHOLD = 2.0
_P90 = 0.9
_TARGETS = ("latency_ms", "peak_cpu_cores", "peak_memory_mb")
_PREFIX_KEYS = make_row_command_prefix_keys("command", max_depth=4)
_BOOTSTRAP_REPLICATES = 50_000
_SEED = 0

NodeKey = tuple[str, str]


@dataclass(frozen=True)
class SelectedNode:
    values: Sequence[float]
    scope: str
    granularity: str


@dataclass(frozen=True)
class Forecast:
    sample_id: str
    repo: str
    target: str
    observed: float
    p90: float
    public_p90: float
    selected_scope: str
    selected_granularity: str
    selected_count: int
    predicted_heavy: bool | None


class BackoffLattice:
    """Frozen public nodes plus causally accumulated per-repository nodes."""

    def __init__(self) -> None:
        self.public_nodes: dict[tuple[str, NodeKey], list[float]] = defaultdict(list)
        self.public_global: dict[str, list[float]] = defaultdict(list)
        self.repo_nodes: dict[tuple[str, str, NodeKey], list[float]] = defaultdict(list)

    @classmethod
    def from_fit_samples(
        cls,
        samples: Sequence[ResourceCallSample],
    ) -> BackoffLattice:
        lattice = cls()
        for sample in samples:
            lattice.add_public_observation(sample)
        missing = [target for target in _TARGETS if not lattice.public_global[target]]
        if missing:
            raise ValueError(f"fit corpus has no eligible labels for {missing}")
        return lattice

    def add_public_observation(self, sample: ResourceCallSample) -> None:
        for target, value in _target_values(sample).items():
            self.public_global[target].append(value)
            for key, _ in _node_keys(sample):
                self.public_nodes[(target, key)].append(value)

    def add_repo_observation(self, sample: ResourceCallSample) -> None:
        repo = repo_key(sample.task_id)
        for target, value in _target_values(sample).items():
            for key, _ in _node_keys(sample):
                self.repo_nodes[(repo, target, key)].append(value)

    def select(
        self,
        sample: ResourceCallSample,
        target: str,
        *,
        include_repo: bool = True,
    ) -> SelectedNode:
        repo = repo_key(sample.task_id)
        for key, granularity in _node_keys(sample):
            if include_repo:
                repo_values = self.repo_nodes.get((repo, target, key))
                if repo_values:
                    return SelectedNode(repo_values, "repo", granularity)
            public_values = self.public_nodes.get((target, key))
            if public_values:
                return SelectedNode(public_values, "public", granularity)
        return SelectedNode(self.public_global[target], "public", "global")


def exceedance_probability(values: Sequence[float], threshold: float) -> float:
    """Return the empirical strict exceedance fraction."""

    if not values:
        raise ValueError("exceedance values must be non-empty")
    return sum(value > threshold for value in values) / len(values)


def predict_heavy(values: Sequence[float], threshold: float = _CPU_THRESHOLD) -> bool:
    """Apply the registered majority-exceedance decision rule."""

    return exceedance_probability(values, threshold) >= 0.5


def prequential_forecasts(
    lattice: BackoffLattice,
    eval_samples: Sequence[ResourceCallSample],
) -> list[Forecast]:
    """Forecast globally ordered calls before adding strictly completed rows."""

    pending: list[tuple[float, str, ResourceCallSample]] = []
    forecasts: list[Forecast] = []
    for sample in sorted(
        eval_samples,
        key=lambda item: (item.tool_ts_start, item.sample_id),
    ):
        while pending and pending[0][0] < sample.tool_ts_start:
            _, _, completed = heapq.heappop(pending)
            lattice.add_repo_observation(completed)

        for target, observed in _target_values(sample).items():
            selected = lattice.select(sample, target)
            public = lattice.select(sample, target, include_repo=False)
            forecasts.append(
                Forecast(
                    sample_id=sample.sample_id,
                    repo=repo_key(sample.task_id),
                    target=target,
                    observed=observed,
                    p90=ecdf_quantile(selected.values, _P90),
                    public_p90=ecdf_quantile(public.values, _P90),
                    selected_scope=selected.scope,
                    selected_granularity=selected.granularity,
                    selected_count=len(selected.values),
                    predicted_heavy=(
                        predict_heavy(selected.values)
                        if target == "peak_cpu_cores"
                        else None
                    ),
                )
            )
        heapq.heappush(
            pending,
            (sample.tool_ts_end, sample.sample_id, sample),
        )
    return forecasts


def paired_ba_uncertainty(
    rows: Sequence[CpuRow],
    candidate_labels: Sequence[bool],
    baseline_labels: Sequence[bool],
    *,
    replicates: int = _BOOTSTRAP_REPLICATES,
    seed: int = _SEED,
) -> dict[str, Any]:
    """Reuse the paired repository confusion bootstrap for two label vectors."""

    if len(rows) != len(candidate_labels) or len(rows) != len(baseline_labels):
        raise ValueError("paired BA inputs must have equal lengths")
    decisions = [
        Decision(
            row=replace(row, static_label=bool(baseline)),
            tier="two_layer",
            blended_label=bool(candidate),
        )
        for row, candidate, baseline in zip(
            rows,
            candidate_labels,
            baseline_labels,
            strict=True,
        )
    ]
    return repo_clustered_uncertainty(
        decisions,
        replicates=replicates,
        seed=seed,
    )


def evaluate(
    fit_samples: Sequence[ResourceCallSample],
    eval_samples: Sequence[ResourceCallSample],
    dump_path: Path,
) -> dict[str, Any]:
    """Run the registered binary comparison and secondary p90 reads."""

    overlapping_tasks = {sample.task_id for sample in fit_samples} & {
        sample.task_id for sample in eval_samples
    }
    if overlapping_tasks:
        raise ValueError(f"fit/eval task overlap: {sorted(overlapping_tasks)[:1]}")
    lattice = BackoffLattice.from_fit_samples(fit_samples)
    forecasts = prequential_forecasts(lattice, eval_samples)
    cpu_forecasts = {
        row.sample_id: row for row in forecasts if row.target == "peak_cpu_cores"
    }
    dump_rows = _load_cpu_dump(dump_path)
    dump_by_id = {row["sample_id"]: row for row in dump_rows}
    if set(cpu_forecasts) != set(dump_by_id):
        missing_dump = sorted(set(cpu_forecasts) - set(dump_by_id))
        missing_eval = sorted(set(dump_by_id) - set(cpu_forecasts))
        raise ValueError(
            "CPU forecast/dump sample IDs differ: "
            f"missing_dump={missing_dump[:1]}, missing_eval={missing_eval[:1]}"
        )

    eval_by_id = {sample.sample_id: sample for sample in eval_samples}
    if len(eval_by_id) != len(eval_samples):
        raise ValueError("eval corpus contains duplicate sample IDs")
    cpu_rows = [
        _cpu_row(dump_row, eval_by_id[dump_row["sample_id"]]) for dump_row in dump_rows
    ]
    latest_by_id = {
        decision.row.sample_id: decision.blended_label
        for decision in prequential_decisions(cpu_rows)
    }
    two_layer_labels = [
        bool(cpu_forecasts[row.sample_id].predicted_heavy) for row in cpu_rows
    ]
    latest_labels = [latest_by_id[row.sample_id] for row in cpu_rows]
    mlp_labels = [row.static_label for row in cpu_rows]
    cold_mlp_labels = [
        (two_layer if cpu_forecasts[row.sample_id].selected_scope == "repo" else mlp)
        for row, two_layer, mlp in zip(
            cpu_rows,
            two_layer_labels,
            mlp_labels,
            strict=True,
        )
    ]
    labels = [row.label for row in cpu_rows]
    uncertainty = paired_ba_uncertainty(
        cpu_rows,
        two_layer_labels,
        latest_labels,
    )
    registered_ci = uncertainty["blended_minus_static"]
    two_layer_metrics = _classification_metrics(labels, two_layer_labels)
    latest_metrics = _classification_metrics(labels, latest_labels)
    cold_mlp_metrics = _classification_metrics(labels, cold_mlp_labels)

    primary = {
        "row_count": len(cpu_rows),
        "two_layer": two_layer_metrics,
        "latest_observation_blend": latest_metrics,
        "two_layer_minus_latest_observation_balanced_accuracy": registered_ci["point"],
        "repo_clustered_bootstrap": {
            "method": uncertainty["method"],
            "cluster_count": uncertainty["cluster_count"],
            "confidence_level": uncertainty["confidence_level"],
            "bootstrap_replicates": uncertainty["bootstrap_replicates"],
            "seed": uncertainty["seed"],
            "two_layer": uncertainty["blended"],
            "two_layer_minus_latest_observation": registered_ci,
        },
        "selected_scope_counts": dict(
            sorted(
                Counter(row.selected_scope for row in cpu_forecasts.values()).items()
            )
        ),
        "selected_granularity_counts": dict(
            sorted(
                Counter(
                    row.selected_granularity for row in cpu_forecasts.values()
                ).items()
            )
        ),
    }
    secondary = {
        "public_selected_falls_back_to_mlp": {
            "status": "SECONDARY - NO REGISTERED GATE",
            "mlp_fallback_row_count": sum(
                row.selected_scope == "public" for row in cpu_forecasts.values()
            ),
            "metrics": cold_mlp_metrics,
            "minus_latest_observation_balanced_accuracy": (
                cold_mlp_metrics["balanced_accuracy"]
                - latest_metrics["balanced_accuracy"]
            ),
        },
        "prequential_p90_pinball": _p90_metrics(forecasts),
    }
    return {
        "registered_criterion": {
            "definition": (
                "two-layer minus latest-observation blend BA repo-clustered "
                "95% CI lower bound > 0"
            ),
            "evaluated": True,
            "passed": bool(registered_ci["ci_low"] > 0.0),
        },
        "primary_cpu_peak_binary_tau_2": primary,
        "secondary": secondary,
        "counts": {
            "fit_call_count": len(fit_samples),
            "eval_call_count": len(eval_samples),
            "forecast_rows_by_target": dict(
                sorted(Counter(row.target for row in forecasts).items())
            ),
        },
    }


def _p90_metrics(forecasts: Sequence[Forecast]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for target in _TARGETS:
        rows = [row for row in forecasts if row.target == target]
        losses_by_repo: dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(2, dtype=float)
        )
        two_layer_losses: list[float] = []
        public_losses: list[float] = []
        for row in rows:
            two_loss = pinball_loss(row.observed, row.p90, _P90)
            public_loss = pinball_loss(row.observed, row.public_p90, _P90)
            two_layer_losses.append(two_loss)
            public_losses.append(public_loss)
            losses_by_repo[row.repo] += np.asarray([two_loss, public_loss])
        result[target] = {
            "row_count": len(rows),
            "two_layer_mean_pinball": float(np.mean(two_layer_losses)),
            "public_only_mean_pinball": float(np.mean(public_losses)),
            "repo_clustered_skill": _clustered_skill(losses_by_repo),
            "selected_scope_counts": dict(
                sorted(Counter(row.selected_scope for row in rows).items())
            ),
            "selected_granularity_counts": dict(
                sorted(Counter(row.selected_granularity for row in rows).items())
            ),
        }
    return result


def _node_keys(sample: ResourceCallSample) -> list[tuple[NodeKey, str]]:
    prefixes = _PREFIX_KEYS(
        {"tool_name": sample.tool_name, "tool_args": sample.tool_args}
    )
    nodes = [
        (("prefix", key), f"command_prefix_depth_{depth}")
        for depth, key in reversed(tuple(enumerate(prefixes, start=1)))
    ]
    nodes.append((("tool", sample.tool_name), "tool_name"))
    return nodes


def _target_values(sample: ResourceCallSample) -> dict[str, float]:
    values: dict[str, float] = {}
    if not sample.censored:
        values["latency_ms"] = (sample.tool_ts_end - sample.tool_ts_start) * 1000.0
    if sample.peak_cpu_cores_eligible and sample.peak_cpu_cores is not None:
        values["peak_cpu_cores"] = float(sample.peak_cpu_cores)
    if sample.peak_memory_mb_eligible and sample.peak_memory_mb is not None:
        values["peak_memory_mb"] = float(sample.peak_memory_mb)
    return values


def _cpu_row(dump_row: dict[str, Any], sample: ResourceCallSample) -> CpuRow:
    if sample.task_id != dump_row["task_id"]:
        raise ValueError(
            f"{sample.sample_id}: dump task_id differs from eval corpus task_id"
        )
    if not math.isclose(
        float(sample.peak_cpu_cores),
        dump_row["observed"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{sample.sample_id}: dump CPU label differs from eval corpus")
    tool_args = sample.tool_args or {}
    if tool_args.get("command") != dump_row["command"]:
        raise ValueError(f"{sample.sample_id}: dump command differs from eval corpus")
    return CpuRow(
        sample_id=sample.sample_id,
        task_id=sample.task_id,
        repo=repo_key(sample.task_id),
        command=dump_row["command"],
        observed=dump_row["observed"],
        label=dump_row["label"],
        static_label=dump_row["static_label"],
        tool_ts_start=sample.tool_ts_start,
        tool_ts_end=sample.tool_ts_end,
    )


def _flatten(
    samples_by_task: dict[str, list[ResourceCallSample]],
    task_ids: Sequence[str],
) -> list[ResourceCallSample]:
    return [sample for task_id in task_ids for sample in samples_by_task[task_id]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--fit-root", type=Path, default=_DEFAULT_FIT_ROOT)
    parser.add_argument("--fit-manifest", type=Path, default=_DEFAULT_FIT_MANIFEST)
    parser.add_argument("--eval-root", type=Path, default=_DEFAULT_EVAL_ROOT)
    parser.add_argument("--eval-manifest", type=Path, default=_DEFAULT_EVAL_MANIFEST)
    args = parser.parse_args()

    fit_by_task, fit_task_ids = load_resource_corpus(
        args.fit_root,
        args.fit_manifest,
    )
    eval_by_task, eval_task_ids = load_resource_corpus(
        args.eval_root,
        args.eval_manifest,
    )
    result = evaluate(
        _flatten(fit_by_task, fit_task_ids),
        _flatten(eval_by_task, eval_task_ids),
        args.rows,
    )
    result["inputs"] = {
        "rows": str(args.rows),
        "fit_root": str(args.fit_root),
        "fit_manifest": str(args.fit_manifest),
        "eval_root": str(args.eval_root),
        "eval_manifest": str(args.eval_manifest),
    }
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
