#!/usr/bin/env python3
"""Score the frozen causal pytest target-overlap memory."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path, PurePosixPath
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
    run as run_exact_memory,
)
from tool_resource.clause_parser import parse_command_clauses  # noqa: E402
from tool_resource.pytest_semantics import (  # noqa: E402
    PytestSignature,
    is_pytest_invocation,
    parse_pytest,
)

VERSION = "pytest-target-overlap-v1"
ARMS = ("exact", "collapsed_signature", "target_overlap")
_VALUE_OPTIONS = {
    "--maxfail",
    "-n",
    "--numprocesses",
    "--dist",
    "-k",
    "-m",
    "--tb",
    "--color",
    "--code-highlight",
    "-r",
    "--junitxml",
    "--junit-prefix",
    "--durations",
    "--durations-min",
}


def _normalize_path(value: str) -> str:
    normalized = str(PurePosixPath(value.removesuffix("/")))
    return normalized.removeprefix("./") or "."


def _pytest_query(command: str) -> tuple[PytestSignature, frozenset[str]] | None:
    parsed = parse_command_clauses(command)
    clauses = parsed.get("clauses") if isinstance(parsed, dict) else None
    if parsed.get("parse_failed") or not isinstance(clauses, list) or len(clauses) != 1:
        return None
    argv = tuple(str(value) for value in clauses[0].get("argv", ()))
    signature = parse_pytest(argv)
    if signature is None or not is_pytest_invocation(argv):
        return None
    executable = PurePosixPath(argv[0]).name.lower()
    index = 1 if executable in {"pytest", "py.test"} else 3
    targets: list[str] = []
    while index < len(argv):
        word = argv[index]
        if word in _VALUE_OPTIONS:
            index += 2
            continue
        if word.startswith("-"):
            index += 1
            continue
        targets.append(word)
        index += 1
    units: set[str] = set()
    for target in targets:
        if "::" in target:
            path, _separator, _node = target.partition("::")
            path = _normalize_path(path)
            units.add(f"file:{path}")
            units.add(f"node:{path}::{target.partition('::')[2]}")
        elif target.removesuffix("/").lower().endswith(".py"):
            units.add(f"file:{_normalize_path(target)}")
        else:
            units.add(f"directory:{_normalize_path(target)}")
    return signature, frozenset(units)


def _partition(signature: PytestSignature) -> PytestSignature:
    return replace(signature, target_shapes=())


def _weighted_pmf(
    query: frozenset[str],
    observations: Sequence[tuple[frozenset[str], int]],
    buckets: int,
) -> tuple[list[float], int, float] | None:
    totals = [0.0] * buckets
    count = 0
    weight_sum = 0.0
    for history, label in observations:
        union = query | history
        weight = len(query & history) / len(union) if union else 0.0
        if weight > 0.0:
            totals[label] += weight
            count += 1
            weight_sum += weight
    if not count:
        return None
    return [value / weight_sum for value in totals], count, weight_sum


def run(
    fit_rows: Sequence[Row], validation_rows: Sequence[Row]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _exact_result, exact_rows = run_exact_memory(fit_rows, validation_rows)
    by_signature: dict[str, dict[PytestSignature, list[int]]] = {
        target: defaultdict(list) for target in ALL_TARGETS
    }
    by_partition: dict[
        str, dict[PytestSignature, list[tuple[frozenset[str], int]]]
    ] = {target: defaultdict(list) for target in ALL_TARGETS}

    def absorb(rows: Sequence[Row]) -> None:
        for row in rows:
            query = _pytest_query(row.command)
            if query is None or not query[1]:
                continue
            signature, units = query
            for target in ALL_TARGETS:
                label = row.labels[target]
                if label is not None:
                    by_signature[target][signature].append(label)
                    by_partition[target][_partition(signature)].append((units, label))

    absorb(fit_rows)
    started = time.perf_counter()
    sidecar: list[dict[str, Any]] = []
    arm_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    exact_by_id = {row["sample_id"]: row for row in exact_rows}
    for task_rows in _task_groups(validation_rows):
        for row in task_rows:
            exact_row = exact_by_id[row.sample_id]
            exact_arm = exact_row["arms"]["exact"]
            candidates = {
                arm: dict(exact_arm["candidate"])
                for arm in ARMS
            }
            pmfs = {
                arm: {
                    target: (
                        None if value is None else list(value)
                    )
                    for target, value in exact_arm[
                        "candidate_probability_by_bucket"
                    ].items()
                }
                for arm in ARMS
            }
            provenance = {
                arm: {
                    target: dict(exact_arm["provenance"][target])
                    for target in ALL_TARGETS
                }
                for arm in ARMS
            }
            query = _pytest_query(row.command)
            if query is not None and query[1]:
                signature, units = query
                for target in ALL_TARGETS:
                    if exact_arm["provenance"][target]["source"] != "current":
                        continue
                    collapsed = by_signature[target].get(signature, [])
                    if collapsed:
                        counts = Counter(collapsed)
                        collapsed_pmf = [
                            counts[index] / len(collapsed)
                            for index in range(BUCKETS[target])
                        ]
                        candidates["collapsed_signature"][target] = _hard(
                            target, collapsed_pmf
                        )
                        pmfs["collapsed_signature"][target] = collapsed_pmf
                        provenance["collapsed_signature"][target] = {
                            "source": "collapsed_signature",
                            "support": len(collapsed),
                        }
                    weighted = _weighted_pmf(
                        units,
                        by_partition[target].get(_partition(signature), []),
                        BUCKETS[target],
                    )
                    if weighted is not None:
                        overlap_pmf, support, weight_sum = weighted
                        candidates["target_overlap"][target] = _hard(
                            target, overlap_pmf
                        )
                        pmfs["target_overlap"][target] = overlap_pmf
                        provenance["target_overlap"][target] = {
                            "source": "pytest_target_overlap",
                            "support": support,
                            "weight_sum": weight_sum,
                        }
            base = {
                "sample_id": row.sample_id,
                "task_id": row.task_id,
                "command": row.command,
                "labels": dict(row.labels),
                "current_dynamic": dict(row.current),
                "current_probability_by_bucket": {
                    target: None if value is None else list(value)
                    for target, value in row.pmfs.items()
                },
            }
            sidecar.append(
                {
                    **base,
                    "arms": {
                        arm: {
                            "candidate": candidates[arm],
                            "candidate_probability_by_bucket": pmfs[arm],
                            "provenance": provenance[arm],
                        }
                        for arm in ARMS
                    },
                }
            )
            for arm in ARMS:
                arm_rows[arm].append(
                    {
                        **base,
                        "candidate": candidates[arm],
                        "candidate_probability_by_bucket": pmfs[arm],
                    }
                )
        absorb(task_rows)
    elapsed = time.perf_counter() - started

    current_rows = [
        {
            **row,
            "candidate": row["current_dynamic"],
            "candidate_probability_by_bucket": row[
                "current_probability_by_bucket"
            ],
        }
        for row in arm_rows["target_overlap"]
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
            accuracy_key = (
                "exact_class_accuracy" if target == "latency" else "accuracy"
            )
            metrics[arm][target] = {
                "current": current[target],
                "candidate": candidate,
                "delta_percentage_points": _accuracy_delta(
                    candidate[accuracy_key], current[target][accuracy_key]
                ),
                "changes": _phase_changes(arm_rows[arm], target),
            }

    primary = metrics["target_overlap"]
    gain_ok = all(
        primary[target]["delta_percentage_points"] >= 5.0
        for target in ALL_TARGETS
    )
    severe_ok = all(
        primary[target]["candidate"]["severe_underprediction_rate"]
        <= primary[target]["current"]["severe_underprediction_rate"]
        for target in ALL_TARGETS
    )
    helpful = sum(primary[target]["changes"]["helpful"] for target in ALL_TARGETS)
    harmful = sum(primary[target]["changes"]["harmful"] for target in ALL_TARGETS)
    helpful_tasks = {
        task_id
        for target in ALL_TARGETS
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
    go = (
        gain_ok
        and severe_ok
        and helpful > harmful
        and len(helpful_tasks) >= 10
        and identity_ok
    )
    return {
        "schema": VERSION,
        "status": "development_validation_go" if go else "development_validation_no_go",
        "claim_bearing": False,
        "protocol": {
            "initial_evidence": "80 committed fit tasks",
            "update": "whole-task settlement",
            "primary_hierarchy": [
                "exact_complete_command",
                "pytest_target_jaccard",
                "current",
            ],
            "minimum_positive_jaccard": 0.0,
            "support_threshold": 1,
            "tie_break": "lower_bucket",
        },
        "arms": metrics,
        "gate": {
            "go": go,
            "minimum_gain_percentage_points_each": 5.0,
            "all_targets_meet_gain": gain_ok,
            "no_severe_underprediction_regression": severe_ok,
            "helpful": helpful,
            "harmful": harmful,
            "helpful_tasks": len(helpful_tasks),
            "minimum_helpful_tasks": 10,
            "row_identity": identity_ok,
        },
        "cost": {
            "overlap_scoring_and_causal_updates_seconds": elapsed,
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
