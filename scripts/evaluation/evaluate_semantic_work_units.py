#!/usr/bin/env python3
"""Score the frozen causal semantic work-unit hierarchy."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

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
from scripts.evaluation.evaluate_pytest_target_overlap import (  # noqa: E402
    _weighted_pmf,
    run as run_pytest_overlap,
)
from tool_resource.clause_parser import parse_command_clauses  # noqa: E402
from tool_resource.pip_semantics import parse_pip_install  # noqa: E402

VERSION = "semantic-work-units-v1"
ARMS = ("pytest_only", "semantic_work_units")


def _pip_query(
    command: str,
) -> tuple[tuple[str, str, tuple[str, ...]], frozenset[str]] | None:
    parsed = parse_command_clauses(command)
    clauses = parsed.get("clauses") if isinstance(parsed, dict) else None
    if parsed.get("parse_failed") or not isinstance(clauses, list) or len(clauses) != 1:
        return None
    signature = parse_pip_install(tuple(str(value) for value in clauses[0].get("argv", ())))
    if signature is None:
        return None
    partition = (signature.interpreter, signature.invocation, signature.flags)
    return partition, frozenset(signature.package_names)


def _copy_arm(row: dict[str, Any], source: str) -> dict[str, Any]:
    arm = row["arms"][source]
    candidate = dict(arm["candidate"])
    pmfs = {
        target: None if value is None else list(value)
        for target, value in arm["candidate_probability_by_bucket"].items()
    }
    provenance = {
        target: dict(value) for target, value in arm["provenance"].items()
    }
    candidate[DISK] = row["current_dynamic"][DISK]
    current_disk_pmf = row["current_probability_by_bucket"][DISK]
    pmfs[DISK] = None if current_disk_pmf is None else list(current_disk_pmf)
    provenance[DISK] = {"source": "current", "support": 0}
    return {
        "candidate": candidate,
        "candidate_probability_by_bucket": pmfs,
        "provenance": provenance,
    }


def run(
    fit_rows: Sequence[Row], validation_rows: Sequence[Row]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _pytest_result, pytest_rows = run_pytest_overlap(fit_rows, validation_rows)
    pytest_by_id = {row["sample_id"]: row for row in pytest_rows}
    history: dict[
        str,
        dict[tuple[str, str, tuple[str, ...]], list[tuple[frozenset[str], int]]],
    ] = {target: defaultdict(list) for target in TARGETS}

    def absorb(rows: Sequence[Row]) -> None:
        for row in rows:
            query = _pip_query(row.command)
            if query is None:
                continue
            partition, packages = query
            for target in TARGETS:
                label = row.labels[target]
                if label is not None:
                    history[target][partition].append((packages, label))

    absorb(fit_rows)
    started = time.perf_counter()
    sidecar: list[dict[str, Any]] = []
    arm_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    eligible = nonexact = with_overlap = 0
    for task_rows in _task_groups(validation_rows):
        for row in task_rows:
            base = pytest_by_id[row.sample_id]
            arms = {
                arm: _copy_arm(base, "target_overlap")
                for arm in ARMS
            }
            query = _pip_query(row.command)
            if query is not None:
                eligible += 1
                partition, packages = query
                if all(
                    arms["semantic_work_units"]["provenance"][target]["source"]
                    == "current"
                    for target in TARGETS
                ):
                    nonexact += 1
                row_overlap = False
                for target in TARGETS:
                    if arms["semantic_work_units"]["provenance"][target]["source"] != "current":
                        continue
                    weighted = _weighted_pmf(
                        packages,
                        history[target].get(partition, []),
                        BUCKETS[target],
                    )
                    if weighted is None:
                        continue
                    row_overlap = True
                    pmf, support, weight_sum = weighted
                    arms["semantic_work_units"]["candidate"][target] = _hard(target, pmf)
                    arms["semantic_work_units"]["candidate_probability_by_bucket"][target] = pmf
                    arms["semantic_work_units"]["provenance"][target] = {
                        "source": "pip_package_overlap",
                        "support": support,
                        "weight_sum": weight_sum,
                    }
                with_overlap += row_overlap
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
                "arms": arms,
            }
            sidecar.append(output)
            for arm in ARMS:
                arm_rows[arm].append(
                    {
                        **{key: value for key, value in output.items() if key != "arms"},
                        "candidate": arms[arm]["candidate"],
                        "candidate_probability_by_bucket": arms[arm][
                            "candidate_probability_by_bucket"
                        ],
                    }
                )
        absorb(task_rows)
    elapsed = time.perf_counter() - started

    current_rows = [
        {
            **{key: value for key, value in row.items() if key != "arms"},
            "candidate": row["current_dynamic"],
            "candidate_probability_by_bucket": row["current_probability_by_bucket"],
        }
        for row in sidecar
    ]
    current = {
        target: _fail_closed_metrics(current_rows, target)
        for target in ALL_TARGETS
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

    primary = metrics["semantic_work_units"]
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
        and identity_ok
        and disk_identity
    )
    return {
        "schema": VERSION,
        "status": "development_component_go" if go else "development_component_no_go",
        "claim_bearing": False,
        "protocol": {
            "initial_evidence": "80 committed fit tasks",
            "update": "whole-task settlement",
            "primary_hierarchy": [
                "exact_complete_command",
                "pytest_target_jaccard",
                "pip_package_jaccard",
                "current",
            ],
            "changed_targets": list(TARGETS),
            "unchanged_target": DISK,
            "minimum_positive_jaccard": 0.0,
            "support_threshold": 1,
            "tie_break": "lower_bucket",
        },
        "coverage": {
            "pip_eligible_rows": eligible,
            "pip_without_prior_hierarchy_evidence": nonexact,
            "pip_rows_with_overlap_prediction": with_overlap,
        },
        "arms": metrics,
        "gate": {
            "go": go,
            "minimum_gain_percentage_points_each_changed_target": 5.0,
            "all_changed_targets_meet_gain": gain_ok,
            "no_severe_underprediction_regression": severe_ok,
            "helpful": helpful,
            "harmful": harmful,
            "helpful_tasks": len(helpful_tasks),
            "minimum_helpful_tasks": 10,
            "row_identity": identity_ok,
            "disk_bit_identical_to_current": disk_identity,
        },
        "cost": {
            "pip_scoring_and_causal_updates_seconds": elapsed,
            "validation_rows": len(validation_rows),
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
