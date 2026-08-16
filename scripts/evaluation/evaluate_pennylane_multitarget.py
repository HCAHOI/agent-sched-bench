#!/usr/bin/env python3
"""Evaluate the fixed command predictor on frozen PennyLane task streams."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from scripts.evaluation.evaluate_clause_latency_buckets import (  # noqa: E402
    _empirical_pmf,
    evaluate_prequential_commands,
)
from scripts.evaluation.evaluate_clause_resource_classes import (  # noqa: E402
    CommandRow,
    load_rows,
    load_run_rows,
)
from scripts.evaluation.evaluate_command_prequential import (  # noqa: E402
    _load_exec_events,
)
from scripts.evaluation.evaluate_doc_tool_semantics import (  # noqa: E402
    _arm,
    _arm_metrics,
    _changes,
    _labels,
    _paired_macro_bootstrap,
    _task_aware,
)
from tool_resource.runtime_kb import (  # noqa: E402
    CANONICAL_LATENCY_BUCKETS,
    CANONICAL_RESOURCE_BUCKET_EDGES,
)

SCHEMA = "pennylane-multitarget-transfer-v1"
TARGETS = ("latency", *CANONICAL_RESOURCE_BUCKET_EDGES)
ARMS = ("majority", "clause_kb", "task_aware")
RUN_DIR = _ROOT / (
    "traces/swe-rebench/gpt-5.6-sol/"
    "pennylane-all76-clean-ebpf-20260816"
)
DEVELOPMENT_SPLIT = _ROOT / "analysis/development/pennylane-survival-action-split.json"
VALIDATION_SPLIT = _ROOT / "analysis/development/offline-tool-semantics-splits.json"
PUBLIC_TELEMETRY = (
    _ROOT
    / "traces/swe-rebench/qwen3.7-max/swe100-full-5be74da-20260726/"
    "simulate_cloud_model_c2_20260726T005356962.jsonl",
    _ROOT
    / "traces/swe-rebench/qwen3.7-max/swe277-full-5be74da-20260726/"
    "simulate_cloud_model_c2_20260726T024552768.jsonl",
)
EXPOSED_VALIDATION_TASK = "PennyLaneAI__pennylane-5846"


def _split(role: str) -> tuple[list[str], int, Path]:
    if role == "development":
        path = DEVELOPMENT_SPLIT
        payload = json.loads(path.read_text(encoding="utf-8"))
        fit = [str(row["task_id"]) for row in payload["fit"]]
        scored = [str(row["task_id"]) for row in payload["replay"]]
    else:
        path = VALIDATION_SPLIT
        cohort = json.loads(path.read_text(encoding="utf-8"))["cohorts"][
            "pennylane"
        ]
        fit = list(map(str, cohort["warmup"]))
        frozen_validation = list(map(str, cohort["validation"]))
        if frozen_validation.count(EXPOSED_VALIDATION_TASK) != 1:
            raise ValueError("the exposed validation task is not uniquely frozen")
        scored = [
            task_id
            for task_id in frozen_validation
            if task_id != EXPOSED_VALIDATION_TASK
        ]
        if len(scored) != 15:
            raise ValueError("the amended validation population must contain 15 tasks")
    task_ids = fit + scored
    if len(task_ids) != len(set(task_ids)) or not fit or not scored:
        raise ValueError(f"{path}: invalid {role} task split")
    return task_ids, len(fit), path


def _write_results_view(path: Path, task_ids: Sequence[str]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "instance_id": task_id,
                    "attempt_dir": f"{task_id}/attempt_1",
                    "success": True,
                },
                sort_keys=True,
            )
            + "\n"
            for task_id in task_ids
        ),
        encoding="utf-8",
    )


def _majority_pmfs(commands: Sequence[CommandRow]) -> dict[str, list[float]]:
    labels = {target: [] for target in TARGETS}
    for row in commands:
        for target, label in _labels(row).items():
            if label is not None:
                labels[target].append(label)
    if any(not values for values in labels.values()):
        raise ValueError("warmup lacks an eligible label for one or more targets")
    return {
        target: _empirical_pmf(
            Counter(values),
            CANONICAL_LATENCY_BUCKETS.bucket_count if target == "latency" else 3,
        )
        for target, values in labels.items()
    }


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _gate(
    metrics: Mapping[str, Mapping[str, Any]], changes: Mapping[str, Any]
) -> dict[str, bool]:
    candidate = metrics["task_aware"]
    baseline = metrics["clause_kb"]
    checks = {
        "higher_equal_weight_accuracy": candidate["equal_weight_accuracy"]
        > baseline["equal_weight_accuracy"],
        "more_helpful_than_harmful": changes["helpful"] > changes["harmful"],
        "no_more_severe_underprediction": candidate[
            "equal_weight_severe_underprediction_rate"
        ]
        <= baseline["equal_weight_severe_underprediction_rate"],
        "helpful_on_at_least_two_tasks": len(changes["helpful_task_ids"]) >= 2,
    }
    return {**checks, "go": all(checks.values())}


def evaluate(role: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task_ids, warmup_count, split_path = _split(role)
    with tempfile.TemporaryDirectory() as temporary:
        results_view = Path(temporary) / "results.jsonl"
        _write_results_view(results_view, task_ids)
        loaded_ids, clauses, commands = load_run_rows(
            RUN_DIR, results_path=results_view
        )
        events = _load_exec_events(RUN_DIR, task_ids, results_path=results_view)
    if loaded_ids != task_ids:
        raise ValueError("loaded tasks differ from the frozen split")

    public = [row for path in PUBLIC_TELEMETRY for row in load_rows(path)]
    raw_public_count = len(public)
    public = [row for row in public if row.repo != "PennyLaneAI__pennylane"]
    if not public or {row.task_id for row in public} & set(task_ids):
        raise ValueError("public evidence is empty or overlaps PennyLane")

    provenance = {
        "role": role,
        "run_dir": str(RUN_DIR.resolve()),
        "split": str(split_path.resolve()),
        "public_telemetry": [str(path.resolve()) for path in PUBLIC_TELEMETRY],
        "public_rows_before_repo_filter": raw_public_count,
        "public_rows_after_repo_filter": len(public),
        "git_sha": _git_sha(),
    }
    _baseline_result, baseline_rows = evaluate_prequential_commands(
        public,
        task_ids,
        clauses,
        commands,
        provenance,
        warmup_task_count=warmup_count,
    )
    by_task: dict[str, list[CommandRow]] = {task_id: [] for task_id in task_ids}
    for row in commands:
        by_task[row.task_id].append(row)
    warmup = [row for task_id in task_ids[:warmup_count] for row in by_task[task_id]]
    scored = [row for task_id in task_ids[warmup_count:] for row in by_task[task_id]]
    baseline_by_sample = {str(row["sample_id"]): row for row in baseline_rows}
    if len(baseline_by_sample) != len(scored):
        raise ValueError("Clause-KB scored rows differ from the command stream")
    candidate_by_sample = _task_aware(
        warmup,
        scored,
        baseline_by_sample,
        events_by_task=events,
    )
    majority = _majority_pmfs(warmup)

    rows = []
    for command in scored:
        sample_id = f"{command.task_id}:{command.call_index}"
        baseline = baseline_by_sample[sample_id]
        candidate = candidate_by_sample[sample_id]
        baseline_pmfs = baseline["current_dynamic"]["probability_by_bucket"]
        rows.append(
            {
                "sample_id": sample_id,
                "task_id": command.task_id,
                "command": command.command,
                "labels": _labels(command),
                "arms": {
                    "majority": _arm(majority),
                    "clause_kb": _arm(
                        {target: baseline_pmfs.get(target) for target in TARGETS}
                    ),
                    "task_aware": _arm(
                        candidate["candidate_probability_by_bucket"],
                        provenance=candidate["provenance"],
                    ),
                },
            }
        )

    metrics = {arm: _arm_metrics(rows, arm) for arm in ARMS}
    changes = _changes(rows, "task_aware", "clause_kb")
    return {
        "schema": SCHEMA,
        "status": "development_exposed" if role == "development" else "validation",
        "claim_bearing": role == "validation",
        "protocol": {
            "evaluation_unit": "eligible exec command",
            "targets": {"latency_buckets": 5, "cpu_rss_disk_buckets": 3},
            "warmup_tasks": warmup_count,
            "scored_tasks": len(task_ids) - warmup_count,
            "causal_update": "whole-task settlement",
            "candidate": "fixed Task-Aware Command Predictor",
            "baseline": "Clause-KB",
        },
        "coverage": {
            "tasks": len(task_ids),
            "clauses": len(clauses),
            "commands": len(commands),
            "scored_commands": len(rows),
            "trace_fallback_tasks": sum(
                not (RUN_DIR / task_id / "attempt_1/tool_calls.json").exists()
                for task_id in task_ids
            ),
        },
        "arms": metrics,
        "comparison": {
            "task_aware_minus_clause_kb": {
                "accuracy_difference": metrics["task_aware"][
                    "equal_weight_accuracy"
                ]
                - metrics["clause_kb"]["equal_weight_accuracy"],
                "changes": changes,
                "paired_task_bootstrap": _paired_macro_bootstrap(
                    rows, "task_aware", "clause_kb"
                ),
            }
        },
        "gate": _gate(metrics, changes),
        "provenance": provenance,
    }, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("development", "validation"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"output already exists: {args.out_dir}")
    result, rows = evaluate(args.role)
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
