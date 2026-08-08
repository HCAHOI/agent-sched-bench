#!/usr/bin/env python3
"""Combine BeginCall CPU predictions with causal interval feedback."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from scripts.evaluation.evaluate_kv_prediction_actionability import (  # noqa: E402
    FIT_ROWS,
    PHASE_ARTIFACT,
    PHASE_RESULT,
    PHASE_ROWS,
    SPLIT,
    VALIDATION_RUN,
)
from scripts.evaluation.evaluate_resource_admission_predictors import (  # noqa: E402
    _static_predictions,
    _validation_commands,
)
from tool_resource_eval.early_cpu_reservation import (  # noqa: E402
    CPU_UPDATE_DELAY_S,
    SAMPLE_INTERVAL_S,
    feedback_action_row,
    model_cpu_page_policy,
)
from trace_collect.trace_data import TraceData  # noqa: E402


VERSION = "prediction-seeded-cpu-feedback-v1"
MAXIMUM_SERVICE_INFLATION = 0.05
MINIMUM_FEEDBACK_RESERVATION_REDUCTION = 0.25
MINIMUM_PREDICTION_INCREMENTAL_REDUCTION = 0.01
MAXIMUM_PREDICTION_INCREMENTAL_SERVICE = 0.005
FEEDBACK_ARMS = (
    "feedback8",
    "clause_kb_feedback",
    "task_aware_feedback",
)


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("feedback summary requires command rows")
    arm_names = tuple(rows[0]["arms"])
    if any(tuple(row["arms"]) != arm_names for row in rows):
        raise ValueError("feedback rows have different arms")
    recorded = sum(float(row["recorded_duration_s"]) for row in rows)
    if recorded <= 0.0:
        raise ValueError("feedback summary requires positive recorded service")
    fixed_reserved = sum(
        float(row["arms"]["fixed8"]["reserved_cpu_core_s"]) for row in rows
    )
    arms = {}
    for arm in arm_names:
        service = sum(float(row["arms"][arm]["service_s"]) for row in rows)
        reserved = sum(
            float(row["arms"][arm]["reserved_cpu_core_s"]) for row in rows
        )
        inflation = (service - recorded) / recorded
        arms[arm] = {
            "service_s": service,
            "reserved_cpu_core_s": reserved,
            "service_inflation": inflation,
            "reservation_reduction": 1.0 - reserved / fixed_reserved,
            "service_budget_pass": inflation <= MAXIMUM_SERVICE_INFLATION,
        }
    eligible = [
        arm
        for arm in FEEDBACK_ARMS
        if arm in arms and arms[arm]["service_budget_pass"]
    ]
    if not eligible:
        raise ValueError("no feedback arm meets the service budget")
    selected = min(
        eligible,
        key=lambda arm: (arms[arm]["reserved_cpu_core_s"], arms[arm]["service_s"]),
    )
    return {
        "recorded_service_s": recorded,
        "arms": arms,
        "eligible_feedback_arms": eligible,
        "selected_arm": selected,
        "selection_rule": "minimum reserved CPU core-seconds within 5% service inflation",
    }


def _decision(summary: Mapping[str, Any]) -> dict[str, Any]:
    arms = summary["arms"]
    feedback = arms["feedback8"]
    selected_name = str(summary["selected_arm"])
    selected = arms[selected_name]
    feedback_go = (
        feedback["reservation_reduction"]
        >= MINIMUM_FEEDBACK_RESERVATION_REDUCTION
        and feedback["service_inflation"] <= MAXIMUM_SERVICE_INFLATION
    )
    incremental_reduction = (
        selected["reservation_reduction"] - feedback["reservation_reduction"]
    )
    incremental_service = (
        selected["service_inflation"] - feedback["service_inflation"]
    )
    prediction_seed_go = (
        selected_name != "feedback8"
        and incremental_reduction >= MINIMUM_PREDICTION_INCREMENTAL_REDUCTION
        and incremental_service <= MAXIMUM_PREDICTION_INCREMENTAL_SERVICE
    )
    status = (
        "development_promising_prediction_seed"
        if feedback_go and prediction_seed_go
        else "development_promising_feedback_only"
        if feedback_go
        else "development_no_go_cpu_feedback"
    )
    return {
        "feedback_go": feedback_go,
        "prediction_seed_go": prediction_seed_go,
        "selected_arm": selected_name,
        "prediction_incremental_reservation_reduction": incremental_reduction,
        "prediction_incremental_service_inflation": incremental_service,
        "status": status,
    }


def _page(
    predictions: Mapping[str, tuple[float, float]], command_id: str
) -> tuple[int, bool]:
    request = predictions.get(command_id)
    if request is None:
        return 8, False
    page = int(request[0])
    if page not in (2, 4, 8) or page != request[0]:
        raise ValueError(f"{command_id}: invalid predicted CPU page {request[0]!r}")
    return page, True


def _assert_alignment(
    authoritative_ids: set[str],
    clause_kb_predictions: Mapping[str, tuple[float, float]],
    task_aware_predictions: Mapping[str, tuple[float, float]],
    trace_ids: set[str],
) -> None:
    if set(clause_kb_predictions) != authoritative_ids:
        raise ValueError("Clause-KB predictions differ from authoritative command rows")
    if set(task_aware_predictions) != authoritative_ids:
        raise ValueError("Task-Aware predictions differ from authoritative command rows")
    missing_from_trace = authoritative_ids - trace_ids
    if missing_from_trace:
        raise ValueError(
            f"{len(missing_from_trace)} predicted commands are absent from traces"
        )


def _file_identity(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest,
    }


def _command_row(
    action: dict[str, Any],
    *,
    task_id: str,
    trace_path: Path,
    clause_kb_predictions: Mapping[str, tuple[float, float]],
    task_aware_predictions: Mapping[str, tuple[float, float]],
) -> tuple[dict[str, Any], str]:
    base, reason = feedback_action_row(action)
    call_id = str(base["tool_call_id"])
    if not call_id:
        raise ValueError(f"{trace_path}: exec action lacks tool_call_id")
    command_id = f"{task_id}:{call_id}"
    clause_page, clause_present = _page(clause_kb_predictions, command_id)
    task_page, task_present = _page(task_aware_predictions, command_id)
    feedback8, feedback_reason = model_cpu_page_policy(
        action, initial_page=8, feedback=True
    )
    if feedback_reason != reason:
        raise ValueError(f"{command_id}: feedback eligibility implementations differ")
    clause_static, _ = model_cpu_page_policy(
        action, initial_page=clause_page, feedback=False
    )
    clause_feedback, _ = model_cpu_page_policy(
        action, initial_page=clause_page, feedback=True
    )
    task_static, _ = model_cpu_page_policy(
        action, initial_page=task_page, feedback=False
    )
    task_feedback, _ = model_cpu_page_policy(
        action, initial_page=task_page, feedback=True
    )
    return {
        "task_id": task_id,
        "command_id": command_id,
        "trace": str(trace_path.resolve()),
        "recorded_duration_s": base["recorded_duration_s"],
        "feedback_eligible": base["eligible"],
        "feedback_exclusion": None if reason == "eligible" else reason,
        "timeline_cpu_core_s": base["timeline_cpu_core_s"],
        "initial_pages": {
            "clause_kb": clause_page,
            "task_aware": task_page,
        },
        "prediction_request_present": {
            "clause_kb": clause_present,
            "task_aware": task_present,
        },
        "arms": {
            "fixed8": base["arms"]["fixed8"],
            "feedback8": feedback8,
            "clause_kb_static": clause_static,
            "clause_kb_feedback": clause_feedback,
            "task_aware_static": task_static,
            "task_aware_feedback": task_feedback,
        },
    }, reason


def run() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validation_ids = list(json.loads(SPLIT.read_text(encoding="utf-8"))["validation"])
    if len(validation_ids) != 50:
        raise ValueError("validation split differs from the exposed 50-task population")
    clause_predictions, task_predictions, prediction_source = _static_predictions(
        validation_ids
    )
    source_files = {
        task_id: {
            "trace": _file_identity(
                VALIDATION_RUN / task_id / "attempt_1/trace.jsonl"
            ),
            "resource_observations": _file_identity(
                VALIDATION_RUN / task_id / "attempt_1/resource_observations.json"
            ),
        }
        for task_id in validation_ids
    }
    command_rows, _raw_counts, traces = _validation_commands(validation_ids)
    if {
        task_id: str(path.resolve()) for task_id, path in traces.items()
    } != {
        task_id: files["trace"]["path"] for task_id, files in source_files.items()
    }:
        raise ValueError("authoritative command rows used unexpected trace paths")
    authoritative_ids = {
        f"{task_id}:{call_id}" for task_id, call_id in command_rows
    }

    rows: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    seen_commands: set[str] = set()
    for task_id in validation_ids:
        trace_path = traces[task_id]
        for action in TraceData.load(trace_path).actions:
            if action.get("data", {}).get("tool_name") != "exec":
                continue
            row, reason = _command_row(
                action,
                task_id=task_id,
                trace_path=trace_path,
                clause_kb_predictions=clause_predictions,
                task_aware_predictions=task_predictions,
            )
            if row["command_id"] in seen_commands:
                raise ValueError(f"duplicate exec command {row['command_id']}")
            seen_commands.add(row["command_id"])
            rows.append(row)
            exclusions[reason] += reason != "eligible"
    if not rows:
        raise ValueError("validation traces contain no exec actions")
    _assert_alignment(
        authoritative_ids,
        clause_predictions,
        task_predictions,
        seen_commands,
    )
    if source_files != {
        task_id: {
            kind: _file_identity(Path(identity["path"]))
            for kind, identity in files.items()
        }
        for task_id, files in source_files.items()
    }:
        raise ValueError("trace inputs changed while they were being read")

    summary = _summarize(rows)
    decision = _decision(summary)
    feedback_rows = [row for row in rows if row["feedback_eligible"]]
    recorded = summary["recorded_service_s"]
    feedback_time = sum(float(row["recorded_duration_s"]) for row in feedback_rows)
    request_mapping_coverage = {
        name: {
            "commands": sum(
                bool(row["prediction_request_present"][name]) for row in rows
            ),
            "recorded_service_s": sum(
                float(row["recorded_duration_s"])
                for row in rows
                if row["prediction_request_present"][name]
            ),
        }
        for name in ("clause_kb", "task_aware")
    }
    for metrics in request_mapping_coverage.values():
        metrics["command_fraction"] = metrics["commands"] / len(rows)
        metrics["service_fraction"] = metrics["recorded_service_s"] / recorded

    fixed = summary["arms"]["fixed8"]
    fixed_service_error = abs(fixed["service_s"] - recorded) / recorded
    fixed_reservation_error = abs(
        fixed["reserved_cpu_core_s"] - 8.0 * recorded
    ) / (8.0 * recorded)
    work_conserved = all(
        row["timeline_cpu_core_s"] is None
        or all(
            float(arm["reserved_cpu_core_s"]) + 1e-9
            >= float(row["timeline_cpu_core_s"])
            for arm in row["arms"].values()
        )
        for row in rows
    )
    integrity = {
        "all_exec_commands_unique": len(seen_commands) == len(rows),
        "fixed8_service_reconstruction_error_at_most_0_1_percent": fixed_service_error
        <= 0.001,
        "fixed8_reservation_reconstruction_error_at_most_0_1_percent": fixed_reservation_error
        <= 0.001,
        "all_timeline_cpu_work_conserved": work_conserved,
    }
    if not all(integrity.values()):
        raise ValueError(f"feedback evaluation integrity failure: {integrity}")

    return {
        "schema": VERSION,
        "status": decision["status"],
        "claim_bearing": False,
        "protocol": {
            "task_pool": "development-exposed SQLGlot validation50",
            "command_unit": "every timed exec action; absent evidence falls back to fixed8",
            "cpu_pages": [2, 4, 8],
            "sample_interval_s": SAMPLE_INTERVAL_S,
            "observation_and_update_delay_s": CPU_UPDATE_DELAY_S,
            "selection": summary["selection_rule"],
            "feedback_gate": {
                "minimum_reservation_reduction": MINIMUM_FEEDBACK_RESERVATION_REDUCTION,
                "maximum_service_inflation": MAXIMUM_SERVICE_INFLATION,
            },
            "prediction_incremental_gate": {
                "minimum_reservation_reduction": MINIMUM_PREDICTION_INCREMENTAL_REDUCTION,
                "maximum_service_inflation": MAXIMUM_PREDICTION_INCREMENTAL_SERVICE,
            },
        },
        "coverage": {
            "tasks": len(validation_ids),
            "exec_commands": len(rows),
            "feedback_eligible_commands": len(feedback_rows),
            "feedback_eligible_command_fraction": len(feedback_rows) / len(rows),
            "feedback_eligible_service_fraction": feedback_time / recorded,
            "feedback_exclusions": dict(sorted(exclusions.items())),
            "prediction_request_mapping": request_mapping_coverage,
            "prediction_source": prediction_source,
        },
        "source_files": source_files,
        "summary": summary,
        "decision": decision,
        "integrity": integrity,
        "limitations": [
            "This is a single-command counterfactual model, not a concurrent scheduler result.",
            "CPU demand is treated as uniform within each 0.5-second source interval.",
            "Validation tasks and predictor components are development-exposed.",
            "The controller cost includes the measured 91.32 ms update p95 plus a 50 ms sample-availability pad, but not a separate eBPF collection CPU charge.",
        ],
    }, rows


def _require_committed_inputs() -> None:
    paths = (
        Path(__file__).resolve(),
        (_ROOT / "src/tool_resource_eval/early_cpu_reservation.py").resolve(),
        (_ROOT / "scripts/evaluation/evaluate_resource_admission_predictors.py").resolve(),
        (_ROOT / "scripts/evaluation/evaluate_kv_prediction_actionability.py").resolve(),
        SPLIT.resolve(),
        FIT_ROWS.resolve(),
        PHASE_ROWS.resolve(),
        PHASE_RESULT.resolve(),
        PHASE_ARTIFACT.resolve(),
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
        "validation_run": str(VALIDATION_RUN.resolve()),
        "split": str(SPLIT.resolve()),
        "phase_rows": str(PHASE_ROWS.resolve()),
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
