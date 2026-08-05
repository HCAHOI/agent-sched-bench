#!/usr/bin/env python3
"""Score the frozen early-execution physical lower-bound component."""

from __future__ import annotations

import argparse
from bisect import bisect_left
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
    TARGETS,
    Row,
    _fail_closed_metrics,
    _load_rows,
)
from scripts.evaluation.evaluate_command_outcome_memory import (  # noqa: E402
    ALL_TARGETS,
    FROZEN_FIT_ROWS,
    FROZEN_VALIDATION_ROWS,
    _hard,
    _task_groups,
)
from scripts.evaluation.evaluate_semantic_work_units import (  # noqa: E402
    run as run_semantic_work_units,
)
from tool_resource.runtime_kb import RESOURCE_BUCKET_LABELS  # noqa: E402
from tool_resource_eval.early_cpu_reservation import (  # noqa: E402
    CPU_UPDATE_P95_S,
    SAMPLE_AVAILABILITY_PAD_S,
    SAMPLE_INTERVAL_S,
)
from trace_collect.resource_timeline import valid_resource_timeline  # noqa: E402
from trace_collect.trace_data import TraceData  # noqa: E402

VERSION = "early-physical-bounds-v1"
ARMS = ("semantic_only", "early_bounds")
LATENCY_EDGES_MS = (500.0, 2000.0, 8000.0, 30000.0)
CPU_EDGES = (2.0, 4.0)
FROZEN_TRACE_ROOT = (
    _ROOT
    / "traces/swe-rebench/gpt-5.6-sol"
    / "sqlglot-prev100-c2-fast-requested-ebpf-20260804"
)


def _number(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) and result >= 0 else None


def _hard_index(target: str, value: int | str | None) -> int | None:
    if value is None:
        return None
    if target == "latency":
        return int(value)
    return RESOURCE_BUCKET_LABELS.index(str(value))


def _project(
    target: str,
    candidate: int | str | None,
    pmf: Sequence[float] | None,
    floor: int,
) -> tuple[int | str | None, list[float] | None, bool]:
    current = _hard_index(target, candidate)
    if current is not None and current >= floor:
        return candidate, None if pmf is None else list(pmf), False
    projected = [0.0] * BUCKETS[target] if pmf is None else list(pmf)
    for index in range(floor):
        projected[index] = 0.0
    total = sum(projected)
    if total > 0.0:
        projected = [value / total for value in projected]
    else:
        projected[floor] = 1.0
    return _hard(target, projected), projected, True


def _prefix(action: Mapping[str, Any]) -> dict[str, Any] | None:
    timeline = valid_resource_timeline(action.get("data", {}).get("resource_timeline"))
    if timeline is None:
        return None
    interval = _number(timeline.get("sample_interval_s"))
    if interval is None or not math.isclose(interval, SAMPLE_INTERVAL_S, abs_tol=1e-9):
        return None
    sample = next(
        (
            item
            for item in timeline["samples"]
            if isinstance(item, dict)
            and (_number(item.get("dt_s")) or 0.0) >= SAMPLE_INTERVAL_S
        ),
        None,
    )
    if sample is None:
        return None
    start = _number(action.get("ts_start"))
    end = _number(action.get("ts_end"))
    offset = _number(sample.get("offset_s"))
    dt = _number(sample.get("dt_s"))
    cpu = _number(sample.get("cpu_core_s"))
    quota = _number(sample.get("cpu_quota_cores"))
    if None in (start, end, offset, dt, cpu, quota) or not quota:
        return None
    effective = offset + SAMPLE_AVAILABILITY_PAD_S + CPU_UPDATE_P95_S
    if end <= start + effective:
        return None
    cpu_rate = min(cpu / dt, quota)
    return {
        "effective_offset_s": effective,
        "latency_floor": bisect_left(LATENCY_EDGES_MS, 1000.0 * effective),
        "cpu_floor": bisect_left(CPU_EDGES, cpu_rate),
        "cpu_rate_cores": cpu_rate,
    }


def _aligned_actions(
    validation_rows: Sequence[Row], trace_root: Path
) -> dict[str, dict[str, Any]]:
    aligned: dict[str, dict[str, Any]] = {}
    for task_rows in _task_groups(validation_rows):
        task_id = task_rows[0].task_id
        attempt = trace_root / task_id / "attempt_1"
        tool_calls = json.loads((attempt / "tool_calls.json").read_text(encoding="utf-8"))
        commands = {
            str(item["id"]): item.get("input", {}).get("command")
            for item in tool_calls
            if item.get("tool") == "exec"
        }
        actions = [
            action
            for action in TraceData.load(attempt / "trace.jsonl").actions
            if action.get("data", {}).get("tool_name") == "exec"
        ]
        cursor = 0
        for row in task_rows:
            match = next(
                (
                    index
                    for index in range(cursor, len(actions))
                    if commands.get(str(actions[index].get("action_id"))) == row.command
                ),
                None,
            )
            if match is None:
                raise ValueError(f"cannot align {row.sample_id} to retained action")
            aligned[row.sample_id] = actions[match]
            cursor = match + 1
    if len(aligned) != len(validation_rows):
        raise ValueError("action alignment is not one-to-one")
    return aligned


def _copy_arm(row: Mapping[str, Any]) -> dict[str, Any]:
    source = row["arms"]["semantic_work_units"]
    return {
        "candidate": dict(source["candidate"]),
        "candidate_probability_by_bucket": {
            target: None if value is None else list(value)
            for target, value in source["candidate_probability_by_bucket"].items()
        },
        "provenance": {
            target: dict(value) for target, value in source["provenance"].items()
        },
    }


def run(
    fit_rows: Sequence[Row],
    validation_rows: Sequence[Row],
    actions: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _semantic_result, semantic_rows = run_semantic_work_units(fit_rows, validation_rows)
    started = time.perf_counter()
    sidecar: list[dict[str, Any]] = []
    arm_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    live = latency_changes = cpu_changes = 0
    violations: list[dict[str, Any]] = []
    for row, semantic_row in zip(validation_rows, semantic_rows, strict=True):
        if row.sample_id != semantic_row["sample_id"]:
            raise ValueError("semantic rows are not aligned")
        arms = {arm: _copy_arm(semantic_row) for arm in ARMS}
        prefix = _prefix(actions[row.sample_id])
        if prefix is not None:
            live += 1
            for target, floor in (
                ("latency", prefix["latency_floor"]),
                ("peak_cpu_cores", prefix["cpu_floor"]),
            ):
                early = arms["early_bounds"]
                candidate, pmf, changed = _project(
                    target,
                    early["candidate"][target],
                    early["candidate_probability_by_bucket"][target],
                    floor,
                )
                label = row.labels[target]
                if label is not None and floor > label:
                    violations.append(
                        {
                            "sample_id": row.sample_id,
                            "target": target,
                            "lower_bound_bucket": floor,
                            "label": label,
                        }
                    )
                if not changed:
                    continue
                early["candidate"][target] = candidate
                early["candidate_probability_by_bucket"][target] = pmf
                early["provenance"][target] = {
                    "source": "early_physical_lower_bound",
                    "base_source": early["provenance"][target]["source"],
                    "lower_bound_bucket": floor,
                    "effective_offset_s": prefix["effective_offset_s"],
                    **(
                        {"prefix_cpu_rate_cores": prefix["cpu_rate_cores"]}
                        if target == "peak_cpu_cores"
                        else {}
                    ),
                }
                latency_changes += target == "latency"
                cpu_changes += target == "peak_cpu_cores"
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
            "prefix": prefix,
            "arms": arms,
        }
        sidecar.append(output)
        for arm in ARMS:
            arm_rows[arm].append(
                {
                    **{key: value for key, value in output.items() if key not in {"arms", "prefix"}},
                    "candidate": arms[arm]["candidate"],
                    "candidate_probability_by_bucket": arms[arm][
                        "candidate_probability_by_bucket"
                    ],
                }
            )
    elapsed = time.perf_counter() - started

    current_rows = [
        {
            **{key: value for key, value in row.items() if key not in {"arms", "prefix"}},
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

    primary = metrics["early_bounds"]
    gain_ok = all(
        primary[target]["delta_percentage_points"] >= 5.0 for target in TARGETS
    )
    severe_ok = all(
        primary[target]["candidate"]["severe_underprediction_rate"]
        <= primary[target]["current"]["severe_underprediction_rate"]
        for target in TARGETS
    )
    helpful = sum(primary[target]["changes"]["helpful"] for target in TARGETS)
    harmful = sum(primary[target]["changes"]["harmful"] for target in TARGETS)
    helpful_tasks = {
        task_id
        for target in TARGETS
        for task_id in primary[target]["changes"]["helpful_task_ids"]
    }
    identity_ok = all(
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
    disk_identity = all(
        arm_row["candidate"][DISK] == current_row["candidate"][DISK]
        and arm_row["candidate_probability_by_bucket"][DISK]
        == current_row["candidate_probability_by_bucket"][DISK]
        for arm in ARMS
        for arm_row, current_row in zip(arm_rows[arm], current_rows, strict=True)
    )
    go = (
        gain_ok
        and severe_ok
        and helpful > harmful
        and len(helpful_tasks) >= 10
        and not violations
        and identity_ok
        and disk_identity
    )
    return {
        "schema": VERSION,
        "status": "development_component_go" if go else "development_component_no_go",
        "claim_bearing": False,
        "protocol": {
            "base": "semantic-work-units-v1",
            "sample_interval_s": SAMPLE_INTERVAL_S,
            "sample_availability_pad_s": SAMPLE_AVAILABILITY_PAD_S,
            "actuation_p95_s": CPU_UPDATE_P95_S,
            "changed_targets": ["latency", "peak_cpu_cores"],
            "semantic_only_target": "sampled_peak_rss_mb",
            "unchanged_target": DISK,
            "projection": "truncate_below_observed_floor_else_point_mass",
        },
        "coverage": {
            "validation_rows": len(validation_rows),
            "live_at_decision": live,
            "latency_projection_changes": latency_changes,
            "cpu_projection_changes": cpu_changes,
        },
        "physical_validity": {
            "violations": len(violations),
            "rows": violations,
        },
        "arms": metrics,
        "gate": {
            "go": go,
            "minimum_gain_percentage_points_each_target": 5.0,
            "all_targets_meet_gain": gain_ok,
            "no_severe_underprediction_regression": severe_ok,
            "helpful": helpful,
            "harmful": harmful,
            "helpful_tasks": len(helpful_tasks),
            "minimum_helpful_tasks": 10,
            "physical_lower_bounds_valid": not violations,
            "row_identity": identity_ok,
            "disk_bit_identical_to_current": disk_identity,
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
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("output directory already exists")
    if (
        args.fit_rows.resolve() != FROZEN_FIT_ROWS.resolve()
        or args.validation_rows.resolve() != FROZEN_VALIDATION_ROWS.resolve()
        or args.trace_root.resolve() != FROZEN_TRACE_ROOT.resolve()
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
    actions = _aligned_actions(validation_rows, args.trace_root)
    result, rows = run(fit_rows, validation_rows, actions)
    result["inputs"] = {
        "fit_rows": str(args.fit_rows.resolve()),
        "validation_rows": str(args.validation_rows.resolve()),
        "trace_root": str(args.trace_root.resolve()),
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
