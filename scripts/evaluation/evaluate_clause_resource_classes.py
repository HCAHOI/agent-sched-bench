#!/usr/bin/env python3
"""Evaluate clause-level CPU/RSS/Disk Heavy/Light predictions on replay traces."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tool_resource.features import repo_of  # noqa: E402
from tool_resource.runtime_kb import (  # noqa: E402
    CANONICAL_RESOURCE_HEAVY_THRESHOLDS,
    SHORT_NULL_LIGHT_MAX_LATENCY_MS,
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


def evaluate(fit_rows: list[Row], eval_rows: list[Row]) -> dict[str, Any]:
    fit_repos = {row.repo for row in fit_rows}
    eval_repos = {row.repo for row in eval_rows}
    kbs = {
        repo: ClauseResourceKB.fit_public(
            row.observation(0.0, 1.0) for row in fit_rows if row.repo != repo
        )
        for repo in eval_repos
    }
    by_task: dict[tuple[int, str], list[Row]] = defaultdict(list)
    for row in eval_rows:
        by_task[(row.manifest_index, row.task_id)].append(row)

    metrics = {
        resource: {
            "label_source_counts": Counter(),
            "provenance_counts": Counter(),
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "prediction_unavailable": 0,
        }
        for resource in CANONICAL_RESOURCE_HEAVY_THRESHOLDS
    }
    for task_ordinal, key in enumerate(sorted(by_task)):
        rows = by_task[key]
        query_ts = float(task_ordinal * 2 + 1)
        kb = kbs[rows[0].repo]
        for row in rows:
            predictions = kb.predict_clause_resource_classes(
                row.repo, row.bin, row.argv, ts_start=query_ts
            )
            for resource, prediction in predictions.items():
                label, source = _label(row, resource)
                metric = metrics[resource]
                metric["label_source_counts"][source] += 1
                if label is None:
                    continue
                if prediction is None:
                    metric["prediction_unavailable"] += 1
                    continue
                metric["provenance_counts"][
                    f"{prediction.scope}:{prediction.key_kind}"
                ] += 1
                predicted = prediction.label == "heavy"
                if label and predicted:
                    metric["tp"] += 1
                elif label:
                    metric["fn"] += 1
                elif predicted:
                    metric["fp"] += 1
                else:
                    metric["tn"] += 1
        close_ts = query_ts + 0.5
        for row in rows:
            kb.observe_completed_clause(row.observation(query_ts, close_ts))

    output_metrics: dict[str, Any] = {}
    for resource, raw in metrics.items():
        tp, tn, fp, fn = (raw[key] for key in ("tp", "tn", "fp", "fn"))
        eligible = tp + tn + fp + fn
        heavy = tp + fn
        if eligible + raw["prediction_unavailable"] != sum(
            count
            for source, count in raw["label_source_counts"].items()
            if source != "null_unavailable"
        ):
            raise AssertionError(f"{resource}: confusion matrix does not reconcile")
        output_metrics[resource] = {
            "threshold": CANONICAL_RESOURCE_HEAVY_THRESHOLDS[resource],
            "eligible_n": eligible,
            "heavy_count": heavy,
            "heavy_rate": heavy / eligible if eligible else None,
            "accuracy": (tp + tn) / eligible if eligible else None,
            "majority_light_accuracy": 1.0 - heavy / eligible if eligible else None,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "prediction_unavailable": raw["prediction_unavailable"],
            "label_source_counts": dict(raw["label_source_counts"]),
            "provenance_counts": dict(raw["provenance_counts"]),
        }
    return {
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
        "label_policy": {
            "heavy_is_strictly_greater_than_threshold": True,
            "short_null_light_max_latency_ms_exclusive": SHORT_NULL_LIGHT_MAX_LATENCY_MS,
            "long_null": "unavailable",
            "cpu_unit": "cores",
            "memory_unit": "decimal_MB",
            "disk_unit": "read_plus_write_bytes_from_linux_task_io_accounting",
        },
        "metrics": output_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(load_rows(args.fit), load_rows(args.eval))
    result["fit"]["path"] = str(args.fit)
    result["eval"]["path"] = str(args.eval)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
