#!/usr/bin/env python3
"""Evaluate the frozen delayed interval-CPU feedback controller."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from scripts.evaluation.evaluate_kv_prediction_actionability import (  # noqa: E402
    DEVELOPMENT_RUN,
    SPLIT,
)
from tool_resource_eval.early_cpu_reservation import (  # noqa: E402
    CPU_UPDATE_DELAY_S,
    SAMPLE_INTERVAL_S,
    feedback_action_row,
)
from trace_collect.trace_data import TraceData  # noqa: E402


VERSION = "interval-cpu-feedback-v1"
ARMS = ("fixed8", "probe_then_two", "feedback")
MINIMUM_COVERAGE = 0.40
MINIMUM_TASKS = 20
MINIMUM_RESERVATION_REDUCTION = 0.25
MAXIMUM_SERVICE_INFLATION = 0.05


def _file_identity(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}


def _valid_attempt(task_id: str) -> tuple[Path, Path, dict[str, Any]] | None:
    attempt = DEVELOPMENT_RUN / task_id / "attempt_1"
    artifact_path = attempt / "resource_observations.json"
    trace_path = attempt / "trace.jsonl"
    if not artifact_path.is_file() or not trace_path.is_file():
        return None
    artifact_bytes = artifact_path.read_bytes()
    artifact_identity = {
        "path": str(artifact_path.resolve()),
        "bytes": len(artifact_bytes),
        "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
    }
    artifact = json.loads(artifact_bytes)
    required = {
        "collection_validity": "valid",
        "workload_execution": "completed",
        "telemetry_quality": "ok",
        "cleanup": "ok",
    }
    return (
        (trace_path, artifact_path, artifact_identity)
        if all(artifact.get(key) == value for key, value in required.items())
        else None
    )


def run() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task_ids = list(json.loads(SPLIT.read_text(encoding="utf-8"))["development"])
    rows: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    valid_tasks: list[str] = []
    source_files: dict[str, dict[str, Any]] = {}
    exec_actions = 0
    for task_id in task_ids:
        attempt = _valid_attempt(task_id)
        if attempt is None:
            exclusions["task_not_evidence_valid"] += 1
            continue
        trace_path, artifact_path, artifact_identity = attempt
        valid_tasks.append(task_id)
        source_files[task_id] = {
            "trace": _file_identity(trace_path),
            "resource_observations": artifact_identity,
        }
        for action in TraceData.load(trace_path).actions:
            if action.get("data", {}).get("tool_name") != "exec":
                continue
            exec_actions += 1
            row, reason = feedback_action_row(action)
            row.update({"task_id": task_id, "trace": str(trace_path.resolve())})
            rows.append(row)
            exclusions[reason] += reason != "eligible"

    if source_files != {
        task_id: {
            "trace": _file_identity(Path(files["trace"]["path"])),
            "resource_observations": _file_identity(
                Path(files["resource_observations"]["path"])
            ),
        }
        for task_id, files in source_files.items()
    }:
        raise ValueError("formal trace inputs changed while they were being read")

    if not rows:
        raise ValueError("development pool contains no exec actions")
    eligible = [row for row in rows if row["eligible"]]
    totals = {
        arm: {
            metric: sum(float(row["arms"][arm][metric]) for row in rows)
            for metric in ("service_s", "reserved_cpu_core_s", "added_service_s")
        }
        for arm in ARMS
    }
    request_counts = {
        arm: dict(
            sorted(
                sum(
                    (
                        Counter(
                            {
                                key: int(value)
                                for key, value in row["arms"][arm]["request_counts"].items()
                            }
                        )
                        for row in rows
                    ),
                    Counter(),
                ).items()
            )
        )
        for arm in ARMS
    }
    recorded = sum(float(row["recorded_duration_s"]) for row in rows)
    fixed = totals["fixed8"]
    feedback = totals["feedback"]
    probe = totals["probe_then_two"]
    coverage = len(eligible) / len(rows)
    eligible_tasks = len({str(row["task_id"]) for row in eligible})
    fixed_error = abs(fixed["service_s"] - recorded) / recorded
    reservation_reduction = 1.0 - (
        feedback["reserved_cpu_core_s"] / fixed["reserved_cpu_core_s"]
    )
    feedback_inflation = feedback["added_service_s"] / recorded
    probe_inflation = probe["added_service_s"] / recorded
    valid_pages = all(set(request_counts[arm]) <= {"2", "4", "8"} for arm in ARMS)
    work_conserved = all(
        row["timeline_cpu_core_s"] is None
        or all(
            float(row["arms"][arm]["reserved_cpu_core_s"])
            + 1e-9
            >= float(row["timeline_cpu_core_s"])
            for arm in ARMS
        )
        for row in rows
    )
    gate = {
        "coverage_at_least_40_percent": coverage >= MINIMUM_COVERAGE,
        "at_least_20_eligible_tasks": eligible_tasks >= MINIMUM_TASKS,
        "feedback_reservation_reduction_at_least_25_percent": reservation_reduction
        >= MINIMUM_RESERVATION_REDUCTION,
        "feedback_service_inflation_at_most_5_percent": feedback_inflation
        <= MAXIMUM_SERVICE_INFLATION,
        "feedback_better_than_probe_then_two": feedback_inflation < probe_inflation,
        "fixed8_reconstruction_error_at_most_0_1_percent": fixed_error <= 0.001,
        "requests_are_2_4_8": valid_pages,
        "all_timeline_work_conserved": work_conserved,
        "all_exec_actions_retained": len(rows) == exec_actions,
    }
    gate["go"] = all(gate.values())
    result = {
        "schema": VERSION,
        "status": "development_go_to_real_controller" if gate["go"] else "development_no_go_interval_feedback",
        "claim_bearing": False,
        "protocol": {
            "task_pool": "original SQLGlot development100 evidence-valid tasks",
            "sample_interval_s": SAMPLE_INTERVAL_S,
            "observation_and_update_delay_s": CPU_UPDATE_DELAY_S,
            "pages_cores": [2, 4, 8],
            "fallback": "fixed8",
        },
        "coverage": {
            "declared_tasks": len(task_ids),
            "evidence_valid_tasks": len(valid_tasks),
            "exec_actions": len(rows),
            "eligible_actions": len(eligible),
            "eligible_fraction": coverage,
            "eligible_tasks": eligible_tasks,
            "exclusions": dict(sorted(exclusions.items())),
        },
        "source_files": source_files,
        "totals": totals,
        "request_counts": request_counts,
        "comparisons": {
            "fixed8_reconstruction_error_fraction": fixed_error,
            "feedback_reservation_reduction_vs_fixed8": reservation_reduction,
            "feedback_service_inflation": feedback_inflation,
            "probe_then_two_service_inflation": probe_inflation,
        },
        "gate": gate,
        "limitations": [
            "This is an aggregate single-command replay, not a concurrent scheduler result.",
            "Within each sampled interval, CPU work is assumed uniform.",
            "Interval-average demand can hide sub-0.5-second bursts.",
            "The corpus is development-exposed and cannot support confirmation.",
        ],
    }
    return result, rows


def _require_committed_inputs() -> None:
    paths = (
        Path(__file__).resolve(),
        (_ROOT / "src/tool_resource_eval/early_cpu_reservation.py").resolve(),
        (
            _ROOT
            / "scripts/evaluation/evaluate_kv_prediction_actionability.py"
        ).resolve(),
        SPLIT.resolve(),
    )
    for path in paths:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            cwd=_ROOT,
            check=True,
            capture_output=True,
        )
    subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *(str(path) for path in paths)],
        cwd=_ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("output directory already exists")
    _require_committed_inputs()
    result, rows = run()
    result["inputs"] = {
        "development_run": str(DEVELOPMENT_RUN.resolve()),
        "split": str(SPLIT.resolve()),
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
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
