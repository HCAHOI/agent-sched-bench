#!/usr/bin/env python3
"""Score the frozen survival-conditioned Disk component."""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from scripts.evaluation.evaluate_clause_latency_buckets import (  # noqa: E402
    _accuracy_delta,
    _phase_changes,
)
from scripts.evaluation.evaluate_command_history_residual import (  # noqa: E402
    BUCKETS,
    DISK,
    SPLIT_MANIFEST,
    Row,
    _fail_closed_metrics,
    _load_rows,
)
from scripts.evaluation.evaluate_command_outcome_memory import (  # noqa: E402
    ALL_TARGETS,
    FROZEN_FIT_ROWS,
    FROZEN_VALIDATION_ROWS,
    _hard,
    run as run_exact_memory,
)
from scripts.evaluation.evaluate_early_physical_bounds import (  # noqa: E402
    CPU_UPDATE_P95_S,
    FROZEN_TRACE_ROOT as FROZEN_VALIDATION_TRACE_ROOT,
    LATENCY_EDGES_MS,
    SAMPLE_AVAILABILITY_PAD_S,
    SAMPLE_INTERVAL_S,
    _aligned_actions,
    _project,
)
from scripts.evaluation.evaluate_semantic_work_units import (  # noqa: E402
    run as run_semantic_work_units,
)
from tool_resource.runtime_kb import RESOURCE_BUCKET_LABELS  # noqa: E402
from trace_collect.resource_timeline import (  # noqa: E402
    RESOURCE_TIMELINE_SCHEMA_VERSION,
)

VERSION = "survival-disk-v1"
ARMS = ("semantic_exact", "survival_disk")
FROZEN_FIT_TRACE_ROOT = (
    _ROOT
    / "traces/swe-rebench/gpt-5.6-sol"
    / "sqlglot-100-c2-fast-requested-ebpf-a0419d9-20260803"
)


def _number(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) and result >= 0 else None


def _decision(action: Mapping[str, Any]) -> dict[str, Any] | None:
    data = action.get("data")
    if not isinstance(data, dict):
        return None
    timeline = data.get("resource_timeline")
    if (
        not isinstance(timeline, dict)
        or timeline.get("version") != RESOURCE_TIMELINE_SCHEMA_VERSION
    ):
        return None
    interval = _number(timeline.get("sample_interval_s"))
    if interval is None or not math.isclose(interval, SAMPLE_INTERVAL_S, abs_tol=1e-9):
        return None
    samples = timeline.get("samples")
    if not isinstance(samples, list) or not samples or not isinstance(samples[0], dict):
        return None
    sample = samples[0]
    start = _number(action.get("ts_start"))
    end = _number(action.get("ts_end"))
    dt = _number(sample.get("dt_s"))
    offset = _number(sample.get("offset_s"))
    if None in (start, end, dt, offset) or dt < SAMPLE_INTERVAL_S:
        return None
    effective = offset + SAMPLE_AVAILABILITY_PAD_S + CPU_UPDATE_P95_S
    if end <= start + effective:
        return None
    return {
        "effective_offset_s": effective,
        "latency_floor": bisect_left(LATENCY_EDGES_MS, 1000.0 * effective),
    }


def _disk_index(value: Any) -> int | None:
    if value is None:
        return None
    return RESOURCE_BUCKET_LABELS.index(str(value))


def _copy_semantic_exact(
    semantic_row: Mapping[str, Any], exact_row: Mapping[str, Any]
) -> dict[str, Any]:
    semantic = semantic_row["arms"]["semantic_work_units"]
    exact_disk = exact_row["arms"]["exact"]
    candidate = dict(semantic["candidate"])
    pmfs = {
        target: None if value is None else list(value)
        for target, value in semantic["candidate_probability_by_bucket"].items()
    }
    provenance = {
        target: dict(value) for target, value in semantic["provenance"].items()
    }
    candidate[DISK] = exact_disk["candidate"][DISK]
    value = exact_disk["candidate_probability_by_bucket"][DISK]
    pmfs[DISK] = None if value is None else list(value)
    provenance[DISK] = dict(exact_disk["provenance"][DISK])
    return {
        "candidate": candidate,
        "candidate_probability_by_bucket": pmfs,
        "provenance": provenance,
    }


def run(
    fit_rows: Sequence[Row],
    validation_rows: Sequence[Row],
    fit_actions: Mapping[str, Mapping[str, Any]],
    validation_actions: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _semantic_result, semantic_rows = run_semantic_work_units(fit_rows, validation_rows)
    _exact_result, exact_rows = run_exact_memory(fit_rows, validation_rows)
    disk_values: dict[int, list[int]] = defaultdict(list)
    fit_live = 0
    for row in fit_rows:
        if _decision(fit_actions[row.sample_id]) is None:
            continue
        fit_live += 1
        current = _disk_index(row.current[DISK])
        label = row.labels[DISK]
        if current is not None and label is not None:
            disk_values[current].append(label)
    disk_pmfs = {
        current: [
            Counter(values).get(index, 0) / len(values)
            for index in range(BUCKETS[DISK])
        ]
        for current, values in disk_values.items()
    }

    started = time.perf_counter()
    sidecar: list[dict[str, Any]] = []
    arm_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    validation_live = latency_changes = disk_overrides = 0
    latency_violations: list[dict[str, Any]] = []
    for row, semantic_row, exact_row in zip(
        validation_rows, semantic_rows, exact_rows, strict=True
    ):
        if not (
            row.sample_id == semantic_row["sample_id"] == exact_row["sample_id"]
        ):
            raise ValueError("base rows are not aligned")
        arms = {
            arm: _copy_semantic_exact(semantic_row, exact_row) for arm in ARMS
        }
        decision = _decision(validation_actions[row.sample_id])
        if decision is not None:
            validation_live += 1
            early = arms["survival_disk"]
            floor = decision["latency_floor"]
            label = row.labels["latency"]
            if label is not None and floor > label:
                latency_violations.append(
                    {
                        "sample_id": row.sample_id,
                        "lower_bound_bucket": floor,
                        "label": label,
                    }
                )
            candidate, pmf, changed = _project(
                "latency",
                early["candidate"]["latency"],
                early["candidate_probability_by_bucket"]["latency"],
                floor,
            )
            if changed:
                latency_changes += 1
                early["candidate"]["latency"] = candidate
                early["candidate_probability_by_bucket"]["latency"] = pmf
                early["provenance"]["latency"] = {
                    "source": "elapsed_lower_bound",
                    "base_source": early["provenance"]["latency"]["source"],
                    "lower_bound_bucket": floor,
                    "effective_offset_s": decision["effective_offset_s"],
                }
            current_disk = _disk_index(row.current[DISK])
            survival_pmf = disk_pmfs.get(current_disk)
            if survival_pmf is not None:
                disk_overrides += 1
                early["candidate"][DISK] = _hard(DISK, survival_pmf)
                early["candidate_probability_by_bucket"][DISK] = list(survival_pmf)
                early["provenance"][DISK] = {
                    "source": "survival_by_current_disk",
                    "current_disk_bucket": current_disk,
                    "support": len(disk_values[current_disk]),
                    "effective_offset_s": decision["effective_offset_s"],
                }
        output = {
            "sample_id": row.sample_id,
            "task_id": row.task_id,
            "command": row.command,
            "labels": dict(row.labels),
            "current_dynamic": dict(row.current),
            "current_probability_by_bucket": {
                target: None if value is None else list(value)
                for target, value in row.pmfs.items()
            },
            "decision": decision,
            "arms": arms,
        }
        sidecar.append(output)
        for arm in ARMS:
            arm_rows[arm].append(
                {
                    **{key: value for key, value in output.items() if key not in {"arms", "decision"}},
                    "candidate": arms[arm]["candidate"],
                    "candidate_probability_by_bucket": arms[arm][
                        "candidate_probability_by_bucket"
                    ],
                }
            )
    elapsed = time.perf_counter() - started

    current_rows = [
        {
            **{key: value for key, value in row.items() if key not in {"arms", "decision"}},
            "candidate": row["current_dynamic"],
            "candidate_probability_by_bucket": row["current_probability_by_bucket"],
        }
        for row in sidecar
    ]
    current = {
        target: _fail_closed_metrics(current_rows, target) for target in ALL_TARGETS
    }
    metrics: dict[str, Any] = {}
    for arm in ARMS:
        metrics[arm] = {}
        for target in ALL_TARGETS:
            candidate = _fail_closed_metrics(arm_rows[arm], target)
            accuracy_key = "exact_class_accuracy" if target == "latency" else "accuracy"
            metrics[arm][target] = {
                "current": current[target],
                "candidate": candidate,
                "delta_percentage_points": _accuracy_delta(
                    candidate[accuracy_key], current[target][accuracy_key]
                ),
                "changes": _phase_changes(arm_rows[arm], target),
            }

    primary = metrics["survival_disk"]
    disk_gain_ok = primary[DISK]["delta_percentage_points"] >= 5.0
    disk_severe_ok = (
        primary[DISK]["candidate"]["severe_underprediction_rate"]
        <= primary[DISK]["current"]["severe_underprediction_rate"]
    )
    disk_changes = primary[DISK]["changes"]
    retained_ok = all(
        primary[target]["delta_percentage_points"] >= 5.0
        for target in ("latency", "sampled_peak_rss_mb")
    )
    semantic_identity = all(
        survival["candidate"][target] == semantic["candidate"][target]
        and survival["candidate_probability_by_bucket"][target]
        == semantic["candidate_probability_by_bucket"][target]
        for target in ("peak_cpu_cores", "sampled_peak_rss_mb")
        for survival, semantic in zip(
            arm_rows["survival_disk"], arm_rows["semantic_exact"], strict=True
        )
    )
    row_identity = all(
        [row["sample_id"] for row in arm_rows[arm]]
        == [row["sample_id"] for row in current_rows]
        and [row["labels"] for row in arm_rows[arm]]
        == [row["labels"] for row in current_rows]
        and [row["current_dynamic"] for row in arm_rows[arm]]
        == [row["current_dynamic"] for row in current_rows]
        and [row["current_probability_by_bucket"] for row in arm_rows[arm]]
        == [row["current_probability_by_bucket"] for row in current_rows]
        for arm in ARMS
    )
    go = (
        disk_gain_ok
        and disk_severe_ok
        and disk_changes["helpful"] > disk_changes["harmful"]
        and len(disk_changes["helpful_task_ids"]) >= 10
        and retained_ok
        and semantic_identity
        and not latency_violations
        and row_identity
    )
    return {
        "schema": VERSION,
        "status": "development_component_go" if go else "development_component_no_go",
        "claim_bearing": False,
        "protocol": {
            "decision_signal": "no_finish_event_at_frozen_first_window_time",
            "resource_values_read": False,
            "latency": "semantic_then_elapsed_floor",
            "cpu_rss": "semantic-work-units-v1",
            "disk": "survival_by_preexecution_current_class_then_exact_then_current",
            "validation_updates": False,
        },
        "fit": {
            "rows": len(fit_rows),
            "live_at_decision": fit_live,
            "disk_groups": {
                RESOURCE_BUCKET_LABELS[current]: {
                    "support": len(disk_values[current]),
                    "label_counts": [
                        Counter(disk_values[current]).get(index, 0)
                        for index in range(BUCKETS[DISK])
                    ],
                    "pmf": pmf,
                }
                for current, pmf in sorted(disk_pmfs.items())
            },
        },
        "coverage": {
            "validation_rows": len(validation_rows),
            "live_at_decision": validation_live,
            "latency_projection_changes": latency_changes,
            "disk_survival_overrides": disk_overrides,
        },
        "physical_validity": {
            "latency_violations": len(latency_violations),
            "rows": latency_violations,
        },
        "arms": metrics,
        "gate": {
            "go": go,
            "minimum_disk_gain_percentage_points": 5.0,
            "disk_meets_gain": disk_gain_ok,
            "disk_no_severe_underprediction_regression": disk_severe_ok,
            "disk_helpful": disk_changes["helpful"],
            "disk_harmful": disk_changes["harmful"],
            "disk_helpful_tasks": len(disk_changes["helpful_task_ids"]),
            "minimum_disk_helpful_tasks": 10,
            "latency_and_rss_retain_five_points": retained_ok,
            "cpu_rss_bit_identical_to_semantic": semantic_identity,
            "elapsed_lower_bound_valid": not latency_violations,
            "row_identity": row_identity,
            "cpu_still_unresolved": True,
        },
        "cost": {
            "projection_and_scoring_seconds": elapsed,
            "offline_reused_existing_timeline": True,
        },
    }, sidecar


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-rows", type=Path, required=True)
    parser.add_argument("--validation-rows", type=Path, required=True)
    parser.add_argument("--fit-trace-root", type=Path, required=True)
    parser.add_argument("--validation-trace-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("output directory already exists")
    if (
        args.fit_rows.resolve() != FROZEN_FIT_ROWS.resolve()
        or args.validation_rows.resolve() != FROZEN_VALIDATION_ROWS.resolve()
        or args.fit_trace_root.resolve() != FROZEN_FIT_TRACE_ROOT.resolve()
        or args.validation_trace_root.resolve()
        != FROZEN_VALIDATION_TRACE_ROOT.resolve()
    ):
        raise ValueError("inputs differ from the frozen development protocol")
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
    fit_actions = _aligned_actions(fit_rows, args.fit_trace_root)
    validation_actions = _aligned_actions(validation_rows, args.validation_trace_root)
    result, rows = run(fit_rows, validation_rows, fit_actions, validation_actions)
    result["inputs"] = {
        "fit_rows": str(args.fit_rows.resolve()),
        "validation_rows": str(args.validation_rows.resolve()),
        "fit_trace_root": str(args.fit_trace_root.resolve()),
        "validation_trace_root": str(args.validation_trace_root.resolve()),
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
