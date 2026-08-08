#!/usr/bin/env python3
"""Test the frozen CPU feedback policy on valid SWE100 and SWE277 traces."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from tool_resource_eval.early_cpu_reservation import (  # noqa: E402
    CPU_UPDATE_DELAY_S,
    SAMPLE_INTERVAL_S,
    feedback_action_row,
)
from tool_resource_eval.labels import repo_of  # noqa: E402
from trace_collect.trace_data import TraceData  # noqa: E402


VERSION = "cpu-feedback-generality-v1"
RUNS = {
    "swe100": _ROOT
    / "traces/swe-rebench/qwen3.7-max/swe100-full-5be74da-20260726",
    "swe277": _ROOT
    / "traces/swe-rebench/qwen3.7-max/swe277-full-5be74da-20260726",
}
MINIMUM_RESERVATION_REDUCTION = 0.25
MAXIMUM_SERVICE_INFLATION = 0.05
MINIMUM_BOOTSTRAP_REDUCTION = 0.20
MINIMUM_TASKS = 100
MINIMUM_REPOS = 10
BOOTSTRAP_DRAWS = 10_000


def _eligible_status(status: Mapping[str, Any]) -> bool:
    return (
        status.get("success") is True
        and status.get("collection_validity") == "valid"
        and status.get("telemetry_quality") == "ok"
        and status.get("telemetry_integrity_failed") is False
        and status.get("replay_execution") == "completed"
    )


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _selected_inputs() -> tuple[list[tuple[str, str, Path]], dict[str, Any]]:
    selected: list[tuple[str, str, Path]] = []
    identities = []
    counts: dict[str, dict[str, int]] = {}
    for corpus, run in RUNS.items():
        statuses = sorted(run.glob("*/attempt_1/openclaw_host_replay_status.json"))
        accepted = 0
        for status_path in statuses:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if not _eligible_status(status):
                continue
            task_id = status_path.parents[1].name
            trace_path = status_path.with_name("trace.jsonl")
            if not trace_path.is_file():
                raise FileNotFoundError(trace_path)
            selected.append((corpus, task_id, trace_path))
            identities.append(
                {
                    "corpus": corpus,
                    "task_id": task_id,
                    "status_sha256": _sha256(status_path),
                    "trace_sha256": _sha256(trace_path),
                }
            )
            accepted += 1
        counts[corpus] = {"status_files": len(statuses), "selected_tasks": accepted}
    task_ids = [task_id for _corpus, task_id, _trace in selected]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("SWE100 and SWE277 valid task populations overlap")
    digest = hashlib.sha256(
        json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return selected, {
        "aggregate_sha256": digest,
        "counts": counts,
        "task_ids": task_ids,
    }


def _task_row(corpus: str, task_id: str, trace_path: Path) -> dict[str, Any]:
    totals = {
        field: 0.0
        for field in (
            "recorded_service_s",
            "fixed_service_s",
            "fixed_reserved_cpu_core_s",
            "feedback_service_s",
            "feedback_reserved_cpu_core_s",
            "timeline_cpu_core_s",
        )
    }
    exclusions: Counter[str] = Counter()
    exec_commands = eligible_commands = 0
    for action in TraceData.load(trace_path).actions:
        data = action.get("data")
        if not isinstance(data, dict) or data.get("tool_name") != "exec":
            continue
        exec_commands += 1
        row, reason = feedback_action_row(action)
        exclusions[reason] += reason != "eligible"
        eligible_commands += bool(row["eligible"])
        fixed = row["arms"]["fixed8"]
        feedback = row["arms"]["feedback"]
        totals["recorded_service_s"] += float(row["recorded_duration_s"])
        totals["fixed_service_s"] += float(fixed["service_s"])
        totals["fixed_reserved_cpu_core_s"] += float(
            fixed["reserved_cpu_core_s"]
        )
        totals["feedback_service_s"] += float(feedback["service_s"])
        totals["feedback_reserved_cpu_core_s"] += float(
            feedback["reserved_cpu_core_s"]
        )
        cpu_work = row["timeline_cpu_core_s"]
        if cpu_work is not None:
            totals["timeline_cpu_core_s"] += float(cpu_work)
            if float(feedback["reserved_cpu_core_s"]) + 1e-9 < float(cpu_work):
                raise ValueError(f"{task_id}: feedback failed to conserve CPU work")
    if exec_commands and totals["recorded_service_s"] <= 0.0:
        raise ValueError(f"{trace_path}: exec commands have no recorded service")
    return {
        "corpus": corpus,
        "task_id": task_id,
        "repo": repo_of(task_id),
        "trace": str(trace_path.resolve()),
        "exec_commands": exec_commands,
        "feedback_eligible_commands": eligible_commands,
        "feedback_exclusions": dict(sorted(exclusions.items())),
        **dict(totals),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("feedback generality summary requires task rows")
    fields = (
        "recorded_service_s",
        "fixed_service_s",
        "fixed_reserved_cpu_core_s",
        "feedback_service_s",
        "feedback_reserved_cpu_core_s",
        "timeline_cpu_core_s",
    )
    totals = {field: sum(float(row[field]) for row in rows) for field in fields}
    recorded = totals["recorded_service_s"]
    fixed_reserved = totals["fixed_reserved_cpu_core_s"]
    if recorded <= 0.0 or fixed_reserved <= 0.0:
        raise ValueError("feedback totals must be positive")
    return {
        "tasks": len(rows),
        "repos": len({str(row["repo"]) for row in rows}),
        "exec_commands": sum(int(row["exec_commands"]) for row in rows),
        "feedback_eligible_commands": sum(
            int(row["feedback_eligible_commands"]) for row in rows
        ),
        **totals,
        "reservation_reduction": 1.0
        - totals["feedback_reserved_cpu_core_s"] / fixed_reserved,
        "service_inflation": (totals["feedback_service_s"] - recorded) / recorded,
    }


def _bootstrap(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = np.asarray(
        [
            (
                float(row["recorded_service_s"]),
                float(row["fixed_reserved_cpu_core_s"]),
                float(row["feedback_service_s"]),
                float(row["feedback_reserved_cpu_core_s"]),
            )
            for row in rows
        ],
        dtype=float,
    )
    rng = np.random.default_rng(0)
    indices = rng.integers(0, len(values), (BOOTSTRAP_DRAWS, len(values)))
    draws = values[indices].sum(axis=1)
    reductions = 1.0 - draws[:, 3] / draws[:, 1]
    inflations = (draws[:, 2] - draws[:, 0]) / draws[:, 0]
    return {
        "draws": BOOTSTRAP_DRAWS,
        "cluster_unit": "task",
        "reservation_reduction_ci95": [
            float(value) for value in np.quantile(reductions, [0.025, 0.975])
        ],
        "service_inflation_ci95": [
            float(value) for value in np.quantile(inflations, [0.025, 0.975])
        ],
    }


def _decision(
    *,
    pooled: Mapping[str, Any],
    corpora: Mapping[str, Mapping[str, Any]],
    pooled_intervals: Mapping[str, Sequence[float]],
    task_count: int,
    repo_count: int,
) -> dict[str, bool]:
    gate = {
        "pooled_reservation_reduction_at_least_25_percent": float(
            pooled["reservation_reduction"]
        )
        >= MINIMUM_RESERVATION_REDUCTION,
        "pooled_service_inflation_at_most_5_percent": float(
            pooled["service_inflation"]
        )
        <= MAXIMUM_SERVICE_INFLATION,
        "each_corpus_reservation_reduction_at_least_25_percent": all(
            float(row["reservation_reduction"]) >= MINIMUM_RESERVATION_REDUCTION
            for row in corpora.values()
        ),
        "each_corpus_service_inflation_at_most_5_percent": all(
            float(row["service_inflation"]) <= MAXIMUM_SERVICE_INFLATION
            for row in corpora.values()
        ),
        "task_bootstrap_reduction_lower_bound_at_least_20_percent": float(
            pooled_intervals["reservation_reduction_ci95"][0]
        )
        >= MINIMUM_BOOTSTRAP_REDUCTION,
        "task_bootstrap_service_upper_bound_at_most_5_percent": float(
            pooled_intervals["service_inflation_ci95"][1]
        )
        <= MAXIMUM_SERVICE_INFLATION,
        "at_least_100_tasks": task_count >= MINIMUM_TASKS,
        "at_least_10_repos": repo_count >= MINIMUM_REPOS,
    }
    gate["go"] = all(gate.values())
    return gate


def run() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected, source_identity = _selected_inputs()
    rows = [_task_row(corpus, task_id, trace) for corpus, task_id, trace in selected]
    _selected_again, source_identity_after = _selected_inputs()
    if source_identity_after != source_identity:
        raise ValueError("SWE trace inputs changed while they were being evaluated")
    by_corpus = {
        corpus: _aggregate([row for row in rows if row["corpus"] == corpus])
        for corpus in RUNS
    }
    intervals = {
        corpus: _bootstrap([row for row in rows if row["corpus"] == corpus])
        for corpus in RUNS
    }
    pooled = _aggregate(rows)
    pooled_intervals = _bootstrap(rows)
    decision = _decision(
        pooled=pooled,
        corpora=by_corpus,
        pooled_intervals=pooled_intervals,
        task_count=len(rows),
        repo_count=int(pooled["repos"]),
    )
    integrity = {
        "source_population_stable_while_reading": source_identity_after
        == source_identity,
        "task_ids_disjoint_across_corpora": len(rows)
        == len({row["task_id"] for row in rows}),
        "fixed8_reconstructs_recorded_service": abs(
            float(pooled["fixed_service_s"])
            - float(pooled["recorded_service_s"])
        )
        <= max(1e-6, float(pooled["recorded_service_s"]) * 1e-12),
        "all_timeline_cpu_work_conserved": float(
            pooled["feedback_reserved_cpu_core_s"]
        )
        + 1e-9
        >= float(pooled["timeline_cpu_core_s"]),
    }
    if not all(integrity.values()):
        raise ValueError(f"feedback generality integrity failure: {integrity}")
    return {
        "schema": VERSION,
        "status": (
            "development_promising_cross_repo_feedback"
            if decision["go"]
            else "development_no_go_cross_repo_feedback"
        ),
        "claim_bearing": False,
        "protocol": {
            "populations": {name: str(path.resolve()) for name, path in RUNS.items()},
            "selection": "all tasks with completed replay and valid, ok telemetry",
            "policy": "fixed8 first interval, then causal 2/4/8 page updates",
            "sample_interval_s": SAMPLE_INTERVAL_S,
            "observation_and_update_delay_s": CPU_UPDATE_DELAY_S,
            "minimum_reservation_reduction": MINIMUM_RESERVATION_REDUCTION,
            "maximum_service_inflation": MAXIMUM_SERVICE_INFLATION,
            "minimum_bootstrap_reduction": MINIMUM_BOOTSTRAP_REDUCTION,
        },
        "source_identity": source_identity,
        "corpora": by_corpus,
        "corpus_intervals": intervals,
        "pooled": pooled,
        "pooled_intervals": pooled_intervals,
        "decision": decision,
        "integrity": integrity,
        "limitations": [
            "SWE100 and SWE277 are development-exposed Qwen replay traces.",
            "This is a per-command counterfactual, not a concurrent physical scheduler.",
            "The model charges the measured update delay but not separate collector CPU cost.",
            "Only telemetry-valid tasks are in scope; invalid collection is excluded before outcomes.",
        ],
    }, rows


def _require_committed_inputs() -> None:
    paths = (
        Path(__file__).resolve(),
        (_ROOT / "src/tool_resource_eval/early_cpu_reservation.py").resolve(),
        (_ROOT / "src/tool_resource_eval/labels.py").resolve(),
        (_ROOT / "src/trace_collect/resource_timeline.py").resolve(),
        (_ROOT / "src/trace_collect/trace_data.py").resolve(),
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
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    }
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "task_rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
