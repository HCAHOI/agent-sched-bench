#!/usr/bin/env python3
"""Evaluate clause-level CPU/RSS/Disk Heavy/Light predictions on replay traces."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tool_resource_eval.labels import repo_of  # noqa: E402
from tool_resource.runtime_kb import (  # noqa: E402
    CANONICAL_LATENCY_BUCKETS,
    CANONICAL_RESOURCE_HEAVY_THRESHOLDS,
    SHRINKAGE_ALPHA_GRID,
    SHORT_NULL_LIGHT_MAX_LATENCY_MS,
    STRUCTURED_ARGV_REPRESENTATION,
    ClauseObservation,
    ClauseResourceKB,
)

_RESOURCE_FIELDS = {
    "peak_cpu_cores": "peak_cpu_cores",
    "sampled_peak_rss_mb": "sampled_peak_rss_mb",
    "disk_read_write_bytes_total": "disk_read_write_bytes_total",
}


@dataclass(frozen=True)
class Row:
    task_id: str
    repo: str
    manifest_index: int
    bin: str
    argv: tuple[str, ...]
    latency_ms: float
    peak_cpu_cores: float | None
    sampled_peak_rss_mb: float | None
    disk_read_write_bytes_total: float | None

    def observation(self, ts_start: float, ts_end: float) -> ClauseObservation:
        return ClauseObservation(
            repo=self.repo,
            bin=self.bin,
            argv=self.argv,
            ts_start=ts_start,
            ts_end=ts_end,
            latency_ms=self.latency_ms,
            peak_cpu_cores=self.peak_cpu_cores,
            sampled_peak_rss_mb=self.sampled_peak_rss_mb,
            disk_read_write_bytes_total=self.disk_read_write_bytes_total,
            impute_short_null_resources_as_light=True,
        )


@dataclass(frozen=True)
class CandidateSSelection:
    alpha: float
    latency_result_path: str
    fit_path: str
    eval_path: str
    fit_row_count: int
    eval_row_count: int

    def __post_init__(self) -> None:
        if self.alpha not in SHRINKAGE_ALPHA_GRID:
            raise ValueError(f"shrinkage alpha must be one of {SHRINKAGE_ALPHA_GRID}")


def load_candidate_s_selection(
    latency_result_path: Path,
    *,
    fit_path: Path,
    eval_path: Path,
    fit_row_count: int,
    eval_row_count: int,
) -> CandidateSSelection:
    result = json.loads(latency_result_path.read_text(encoding="utf-8"))
    selection = result.get("selection", {}).get("candidate_s_alpha")
    provenance = result.get("provenance")
    if not isinstance(selection, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("latency result has no Candidate S fit selection")
    alpha = selection.get("selected_alpha")
    if (
        not isinstance(alpha, (int, float))
        or isinstance(alpha, bool)
        or float(alpha) not in SHRINKAGE_ALPHA_GRID
    ):
        raise ValueError("latency result selected alpha is outside the fixed grid")
    if selection.get("alpha_grid") != list(SHRINKAGE_ALPHA_GRID):
        raise ValueError("latency result alpha grid differs from the fixed grid")
    if (
        selection.get("selection_target") != "three_class_latency_accuracy"
        or selection.get("tie_break") != "larger_alpha"
        or selection.get("outer_labels_used") is not False
    ):
        raise ValueError("latency result Candidate S selection contract differs")
    if result.get("bucket_edges_ms") != list(CANONICAL_LATENCY_BUCKETS.edges_ms):
        raise ValueError("latency result bucket edges differ from the canonical objective")
    if provenance.get("candidate_s", {}).get("enabled") is not True:
        raise ValueError("latency result does not declare Candidate S enabled")
    resolved_fit = fit_path.resolve()
    resolved_eval = eval_path.resolve()
    if Path(str(provenance.get("fit_telemetry"))).resolve() != resolved_fit:
        raise ValueError("latency result fit input differs from resource fit input")
    if Path(str(provenance.get("eval_telemetry"))).resolve() != resolved_eval:
        raise ValueError("latency result eval input differs from resource eval input")
    if (
        result.get("fit_clause_observation_count") != fit_row_count
        or selection.get("fit_row_count") != fit_row_count
        or result.get("eval_clause_observation_count") != eval_row_count
    ):
        raise ValueError("latency and resource result row counts differ")
    if result.get("row_identity", {}).get("identical_row_ids_and_labels") is not True:
        raise ValueError("latency result did not reconcile outer rows and labels")
    return CandidateSSelection(
        alpha=float(alpha),
        latency_result_path=str(latency_result_path.resolve()),
        fit_path=str(resolved_fit),
        eval_path=str(resolved_eval),
        fit_row_count=fit_row_count,
        eval_row_count=eval_row_count,
    )


def _number(value: Any) -> float | None:
    return None if value is None else float(value)


def load_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            data = record.get("data")
            if not isinstance(data, dict):
                continue
            telemetry = data.get("clause_telemetry")
            if (
                not isinstance(telemetry, dict)
                or telemetry.get("eligible_for_kb") is not True
            ):
                continue
            task_id = data.get("task_instance_id")
            manifest_index = data.get("manifest_index")
            if not isinstance(task_id, str) or not isinstance(manifest_index, int):
                raise ValueError(f"{path}: clause telemetry row lacks task identity")
            for clause in telemetry.get("clauses", []):
                if (
                    not isinstance(clause, dict)
                    or clause.get("eligible_for_kb") is not True
                ):
                    continue
                latency = clause.get("latency_ms")
                argv = clause.get("argv")
                if latency is None or not isinstance(argv, list) or not argv:
                    continue
                disk = clause.get("disk_io")
                disk_total = (
                    disk.get("read_write_bytes_total")
                    if isinstance(disk, dict)
                    else None
                )
                rows.append(
                    Row(
                        task_id=task_id,
                        repo=repo_of(task_id),
                        manifest_index=manifest_index,
                        bin=str(clause["bin"]),
                        argv=tuple(str(value) for value in argv),
                        latency_ms=float(latency),
                        peak_cpu_cores=_number(clause.get("peak_cpu_cores")),
                        sampled_peak_rss_mb=_number(clause.get("sampled_peak_rss_mb")),
                        disk_read_write_bytes_total=_number(disk_total),
                    )
                )
    if not rows:
        raise ValueError(f"{path}: no eligible clause telemetry rows")
    return rows


def _label(row: Row, resource: str) -> tuple[bool | None, str]:
    value = getattr(row, _RESOURCE_FIELDS[resource])
    if value is not None:
        heavy = value > CANONICAL_RESOURCE_HEAVY_THRESHOLDS[resource]
        return heavy, "observed_heavy" if heavy else "observed_light"
    if row.latency_ms < SHORT_NULL_LIGHT_MAX_LATENCY_MS:
        return False, "short_null_imputed_light"
    return None, "null_unavailable"


def _empty_confusion() -> dict[str, Any]:
    return {
        "provenance_counts": Counter(),
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "prediction_unavailable": 0,
    }


def _finalize_resource_metric(
    raw: dict[str, Any],
    label_source_counts: Counter[str],
) -> dict[str, Any]:
    tp, tn, fp, fn = (raw[key] for key in ("tp", "tn", "fp", "fn"))
    predicted_n = tp + tn + fp + fn
    label_eligible_n = sum(
        count
        for source, count in label_source_counts.items()
        if source != "null_unavailable"
    )
    if predicted_n + raw["prediction_unavailable"] != label_eligible_n:
        raise AssertionError("resource confusion matrix does not reconcile")
    heavy = label_source_counts["observed_heavy"]
    return {
        "eligible_n": label_eligible_n,
        "prediction_available": predicted_n,
        "observed_heavy": label_source_counts["observed_heavy"],
        "observed_light": label_source_counts["observed_light"],
        "short_null_imputed_light": label_source_counts[
            "short_null_imputed_light"
        ],
        "null_unavailable": label_source_counts["null_unavailable"],
        "heavy_count": heavy,
        "heavy_rate": heavy / label_eligible_n if label_eligible_n else None,
        "accuracy": (
            (tp + tn) / label_eligible_n
            if label_eligible_n and raw["prediction_unavailable"] == 0
            else None
        ),
        "available_only_accuracy": (
            (tp + tn) / predicted_n if predicted_n else None
        ),
        "majority_light_accuracy": (
            1.0 - heavy / label_eligible_n if label_eligible_n else None
        ),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "prediction_unavailable": raw["prediction_unavailable"],
        "label_source_counts": dict(label_source_counts),
        "provenance_counts": dict(raw["provenance_counts"]),
    }


def evaluate(
    fit_rows: list[Row],
    eval_rows: list[Row],
    *,
    candidate_s_selection: CandidateSSelection | None = None,
) -> dict[str, Any]:
    shrinkage_alpha = (
        None if candidate_s_selection is None else candidate_s_selection.alpha
    )
    if candidate_s_selection is not None and (
        candidate_s_selection.fit_row_count != len(fit_rows)
        or candidate_s_selection.eval_row_count != len(eval_rows)
    ):
        raise ValueError("Candidate S selection row counts differ from evaluator rows")
    fit_tasks = {row.task_id for row in fit_rows}
    eval_tasks = {row.task_id for row in eval_rows}
    overlap = fit_tasks & eval_tasks
    if overlap:
        raise ValueError(f"fit/eval task overlap: {sorted(overlap)[:3]}")
    fit_repos = {row.repo for row in fit_rows}
    eval_repos = {row.repo for row in eval_rows}
    current_kbs = {
        repo: ClauseResourceKB.fit_public(
            row.observation(0.0, 1.0) for row in fit_rows if row.repo != repo
        )
        for repo in sorted(eval_repos)
    }
    candidate_kbs = {
        repo: ClauseResourceKB.fit_public(
            (row.observation(0.0, 1.0) for row in fit_rows if row.repo != repo),
            representation=STRUCTURED_ARGV_REPRESENTATION,
        )
        for repo in sorted(eval_repos)
    }
    shrinkage_kbs = (
        {
            repo: ClauseResourceKB.fit_public(
                (row.observation(0.0, 1.0) for row in fit_rows if row.repo != repo),
                representation=STRUCTURED_ARGV_REPRESENTATION,
                shrinkage_alpha=shrinkage_alpha,
            )
            for repo in sorted(eval_repos)
        }
        if shrinkage_alpha is not None
        else {}
    )
    by_task: dict[tuple[int, str], list[Row]] = defaultdict(list)
    for row in eval_rows:
        by_task[(row.manifest_index, row.task_id)].append(row)

    arm_names = ["current", "candidate_r"]
    if shrinkage_alpha is not None:
        arm_names.append("candidate_s")
    metrics = {
        resource: {
            "label_source_counts": Counter(),
            "arms": {arm: _empty_confusion() for arm in arm_names},
        }
        for resource in CANONICAL_RESOURCE_HEAVY_THRESHOLDS
    }
    for task_ordinal, key in enumerate(sorted(by_task)):
        rows = by_task[key]
        query_ts = float(task_ordinal * 2 + 1)
        current_kb = current_kbs[rows[0].repo]
        candidate_kb = candidate_kbs[rows[0].repo]
        shrinkage_kb = shrinkage_kbs.get(rows[0].repo)
        for row in rows:
            predictions_by_arm = {
                "current": current_kb.predict_clause_resource_classes(
                    row.repo, row.bin, row.argv, ts_start=query_ts
                ),
                "candidate_r": candidate_kb.predict_clause_resource_classes(
                    row.repo, row.bin, row.argv, ts_start=query_ts
                ),
            }
            if shrinkage_kb is not None:
                predictions_by_arm["candidate_s"] = (
                    shrinkage_kb.predict_clause_resource_classes(
                        row.repo,
                        row.bin,
                        row.argv,
                        ts_start=query_ts,
                    )
                )
            for resource in CANONICAL_RESOURCE_HEAVY_THRESHOLDS:
                label, source = _label(row, resource)
                metric = metrics[resource]
                metric["label_source_counts"][source] += 1
                if label is None:
                    continue
                for arm, predictions in predictions_by_arm.items():
                    prediction = predictions[resource]
                    arm_metric = metric["arms"][arm]
                    if prediction is None:
                        arm_metric["prediction_unavailable"] += 1
                        continue
                    arm_metric["provenance_counts"][
                        f"{prediction.scope}:{prediction.key_kind}:"
                        f"{prediction.canonicalizer_version}:"
                        f"{prediction.arbitration}"
                    ] += 1
                    predicted = prediction.label == "heavy"
                    if label and predicted:
                        arm_metric["tp"] += 1
                    elif label:
                        arm_metric["fn"] += 1
                    elif predicted:
                        arm_metric["fp"] += 1
                    else:
                        arm_metric["tn"] += 1
        close_ts = query_ts + 0.5
        for row in rows:
            observation = row.observation(query_ts, close_ts)
            current_kb.observe_completed_clause(observation)
            candidate_kb.observe_completed_clause(observation)
            if shrinkage_kb is not None:
                shrinkage_kb.observe_completed_clause(observation)

    current_metrics: dict[str, Any] = {}
    arm_metrics: dict[str, dict[str, Any]] = {
        arm: {} for arm in arm_names if arm != "current"
    }
    for resource, raw in metrics.items():
        current = _finalize_resource_metric(
            raw["arms"]["current"],
            raw["label_source_counts"],
        )
        current_metrics[resource] = {
            "threshold": CANONICAL_RESOURCE_HEAVY_THRESHOLDS[resource],
            **current,
        }
        for arm in arm_metrics:
            candidate = _finalize_resource_metric(
                raw["arms"][arm],
                raw["label_source_counts"],
            )
            arm_metrics[arm][resource] = {
                "threshold": CANONICAL_RESOURCE_HEAVY_THRESHOLDS[resource],
                **candidate,
                "current_accuracy": current["accuracy"],
                "accuracy_minus_current_percentage_points": (
                    None
                    if candidate["accuracy"] is None
                    or current["accuracy"] is None
                    else 100.0 * (candidate["accuracy"] - current["accuracy"])
                ),
                "accuracy_minus_majority_light_percentage_points": (
                    None
                    if candidate["accuracy"] is None
                    else 100.0
                    * (
                        candidate["accuracy"]
                        - candidate["majority_light_accuracy"]
                    )
                ),
            }
    result = {
        "artifact_type": "development_exposed_serialized_virtual_resource_classification",
        "claim_bearing": False,
        "fit": {
            "row_count": len(fit_rows),
            "task_count": len({row.task_id for row in fit_rows}),
            "repo_count": len(fit_repos),
        },
        "eval": {
            "row_count": len(eval_rows),
            "task_count": len({row.task_id for row in eval_rows}),
            "repo_count": len(eval_repos),
        },
        "row_identity": {
            "identical_label_rows_across_arms": True,
            "fit_eval_task_overlap_count": 0,
        },
        "label_policy": {
            "heavy_is_strictly_greater_than_threshold": True,
            "short_null_light_max_latency_ms_exclusive": SHORT_NULL_LIGHT_MAX_LATENCY_MS,
            "long_null": "unavailable",
            "cpu_unit": "cores",
            "memory_unit": "decimal_MB",
            "disk_unit": "read_plus_write_bytes_from_linux_task_io_accounting",
        },
        "candidate_r": {
            "representation": STRUCTURED_ARGV_REPRESENTATION,
            "stable_subcommand_min_distinct_fit_repositories": 3,
            "stable_subcommand_uses_labels": False,
            "arbitration": "same hard first-nonempty selection as current",
        },
        "metrics": current_metrics,
        "candidates": {
            arm: {"metrics": metrics_by_resource}
            for arm, metrics_by_resource in arm_metrics.items()
        },
    }
    if shrinkage_alpha is not None:
        result["candidate_s"] = {
            "representation": STRUCTURED_ARGV_REPRESENTATION,
            "arbitration": "deepest local plus deepest public posterior",
            "shrinkage_alpha": shrinkage_alpha,
            "alpha_source": {
                "latency_result": candidate_s_selection.latency_result_path,
                "fit_path": candidate_s_selection.fit_path,
                "eval_path": candidate_s_selection.eval_path,
                "fit_row_count": candidate_s_selection.fit_row_count,
                "eval_row_count": candidate_s_selection.eval_row_count,
                "verified": True,
            },
            "same_alpha_for_all_targets": True,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--latency-result",
        type=Path,
        help="Candidate S latency result that owns and proves the fit-selected alpha",
    )
    args = parser.parse_args()
    fit_rows = load_rows(args.fit)
    eval_rows = load_rows(args.eval)
    selection = (
        None
        if args.latency_result is None
        else load_candidate_s_selection(
            args.latency_result,
            fit_path=args.fit,
            eval_path=args.eval,
            fit_row_count=len(fit_rows),
            eval_row_count=len(eval_rows),
        )
    )
    result = evaluate(
        fit_rows,
        eval_rows,
        candidate_s_selection=selection,
    )
    result["fit"]["path"] = str(args.fit)
    result["eval"]["path"] = str(args.eval)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
