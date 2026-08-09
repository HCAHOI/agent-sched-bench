#!/usr/bin/env python3
"""Measure hindsight headroom for existing pip/pytest prediction semantics."""

from __future__ import annotations

import argparse
from copy import deepcopy
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from scripts.evaluation.evaluate_command_outcome_memory import ALL_TARGETS
from scripts.evaluation.evaluate_doc_tool_semantics import (
    _GENERATION_OUTPUT,
    _history_row,
    _labels,
    _load_development_stream,
    _arm_metrics,
    _changes,
    _paired_macro_bootstrap,
)
from scripts.evaluation.evaluate_pytest_target_overlap import run as run_pytest_overlap
from scripts.evaluation.evaluate_semantic_work_units import run as run_semantic_work_units
from tool_resource.clause_parser import parse_command_clauses
from tool_resource.pip_semantics import parse_pip_install
from tool_resource.pytest_semantics import is_pytest_invocation
from tool_resource.runtime_kb import RESOURCE_BUCKET_LABELS


def _command_tools(command: str) -> tuple[str, ...]:
    parsed = parse_command_clauses(command)
    if parsed.get("parse_failed"):
        return ()
    found: set[str] = set()
    for clause in parsed.get("clauses", []):
        argv = tuple(str(value) for value in clause.get("argv", ()))
        if parse_pip_install(argv) is not None:
            found.add("pip")
        if is_pytest_invocation(argv):
            found.add("pytest")
    return tuple(tool for tool in ("pip", "pytest") if tool in found)


def _external_arm(row: Mapping[str, Any], name: str) -> dict[str, Any]:
    arm = row["arms"][name]
    return {
        "prediction": deepcopy(arm["candidate"]),
        "probability_by_bucket": deepcopy(
            arm["candidate_probability_by_bucket"]
        ),
        "provenance": deepcopy(arm["provenance"]),
    }


def _truth_prediction(target: str, label: int) -> int | str:
    return label if target == "latency" else RESOURCE_BUCKET_LABELS[label]


def _static_oracle(
    baseline: Mapping[str, Any],
    candidate_pool: Mapping[str, Sequence[Mapping[str, Any]]],
    labels: Mapping[str, int | None],
) -> dict[str, Any]:
    selected = deepcopy(baseline)
    selected["provenance"] = {}
    for target in ALL_TARGETS:
        label = labels[target]
        selected_source = "task_aware"
        available_sources = ["task_aware", *[str(item["source"]) for item in candidate_pool[target]]]
        if label is not None and baseline["probability_by_bucket"][target] is not None:
            truth = _truth_prediction(target, label)
            if baseline["prediction"][target] != truth:
                for candidate in candidate_pool[target]:
                    if candidate["prediction"] != truth:
                        continue
                    selected["prediction"][target] = candidate["prediction"]
                    selected["probability_by_bucket"][target] = deepcopy(
                        candidate["probability_by_bucket"]
                    )
                    selected_source = str(candidate["source"])
                    break
        selected["provenance"][target] = {
            "source": "hindsight_static_candidate_selector",
            "selected_source": selected_source,
            "candidate_sources": available_sources,
        }
    return selected


def _candidate_pool(
    candidates: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    pool = {target: [] for target in ALL_TARGETS}
    for source, candidate in candidates:
        for target in ALL_TARGETS:
            pmf = candidate["probability_by_bucket"][target]
            if pmf is None:
                continue
            provenance_source = (
                candidate.get("provenance", {}).get(target, {}).get("source")
            )
            if source == "exact_complete_command" and provenance_source != "exact":
                continue
            if source == "pytest_collapsed_signature" and provenance_source != "collapsed_signature":
                continue
            if source == "pytest_target_overlap" and provenance_source != "pytest_target_overlap":
                continue
            if source == "pip_package_overlap" and provenance_source != source:
                continue
            pool[target].append(
                {
                    "source": source,
                    "prediction": candidate["prediction"][target],
                    "probability_by_bucket": deepcopy(pmf),
                }
            )
    return pool


def _perfect_tool_oracle(
    baseline: Mapping[str, Any], labels: Mapping[str, int | None]
) -> dict[str, Any]:
    selected = deepcopy(baseline)
    selected["provenance"] = {}
    for target in ALL_TARGETS:
        label = labels[target]
        pmf = baseline["probability_by_bucket"][target]
        if label is not None and pmf is not None:
            one_hot = [0.0] * len(pmf)
            one_hot[label] = 1.0
            selected["prediction"][target] = _truth_prediction(target, label)
            selected["probability_by_bucket"][target] = one_hot
        selected["provenance"][target] = {"source": "hindsight_perfect_tool"}
    return selected


def build_oracle_rows(
    baseline_rows: Sequence[Mapping[str, Any]],
    pytest_rows: Sequence[Mapping[str, Any]],
    semantic_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join causal component rows and add the two frozen hindsight arms."""

    if len({len(baseline_rows), len(pytest_rows), len(semantic_rows)}) != 1:
        raise ValueError("component row counts disagree")
    output: list[dict[str, Any]] = []
    for baseline_row, pytest_row, semantic_row in zip(
        baseline_rows, pytest_rows, semantic_rows, strict=True
    ):
        for key in ("sample_id", "task_id", "command", "labels"):
            if baseline_row[key] != pytest_row[key] or baseline_row[key] != semantic_row[key]:
                raise ValueError(f"component rows disagree on {key}")
        tools = _command_tools(str(baseline_row["command"]))
        task_aware = deepcopy(baseline_row["arms"]["task_aware"])
        if not tools:
            static = deepcopy(task_aware)
            perfect = deepcopy(task_aware)
        else:
            candidates: list[tuple[str, Mapping[str, Any]]] = [
                ("clause_kb", baseline_row["arms"]["clause_kb"]),
                ("exact_complete_command", _external_arm(pytest_row, "exact")),
            ]
            if "pytest" in tools:
                candidates.extend(
                    (
                        ("pytest_collapsed_signature", _external_arm(pytest_row, "collapsed_signature")),
                        ("pytest_target_overlap", _external_arm(pytest_row, "target_overlap")),
                    )
                )
            if "pip" in tools:
                candidates.append(
                    ("pip_package_overlap", _external_arm(semantic_row, "semantic_work_units"))
                )
            candidate_pool = _candidate_pool(candidates)
            static = _static_oracle(task_aware, candidate_pool, baseline_row["labels"])
            perfect = _perfect_tool_oracle(task_aware, baseline_row["labels"])
        if not tools:
            candidate_pool = {target: [] for target in ALL_TARGETS}
        output.append(
            {
                key: deepcopy(baseline_row[key])
                for key in ("sample_id", "task_id", "command", "labels")
            }
            | {
                "tools": list(tools),
                "static_candidates": candidate_pool,
                "arms": {
                    "task_aware": task_aware,
                    "static_candidate_oracle": static,
                    "perfect_tool_oracle": perfect,
                },
            }
        )
    return output


def _materialize_candidates(
    fit_rows: Sequence[Any], validation_rows: Sequence[Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the existing task-causal expert evaluators in frozen row order."""

    _pytest_result, pytest_rows = run_pytest_overlap(fit_rows, validation_rows)
    _semantic_result, semantic_rows = run_semantic_work_units(fit_rows, validation_rows)
    return pytest_rows, semantic_rows


def _coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    semantic_sources = {
        "pip": {"pip_package_overlap"},
        "pytest": {"pytest_collapsed_signature", "pytest_target_overlap"},
    }
    result: dict[str, Any] = {}
    for tool, sources in semantic_sources.items():
        selected = [row for row in rows if tool in row["tools"]]
        non_exact = [
            row
            for row in selected
            if not any(
                candidate["source"] == "exact_complete_command"
                for target in ALL_TARGETS
                for candidate in row["static_candidates"][target]
            )
        ]
        distinct = [
            row
            for row in selected
            if any(
                candidate["source"] in sources
                and candidate["prediction"]
                != row["arms"]["task_aware"]["prediction"][target]
                for target in ALL_TARGETS
                for candidate in row["static_candidates"][target]
            )
        ]
        tasks = {row["task_id"] for row in selected}
        non_exact_tasks = {row["task_id"] for row in non_exact}
        result[tool] = {
            "commands": len(selected),
            "tasks": len(tasks),
            "non_exact_commands": len(non_exact),
            "non_exact_tasks": len(non_exact_tasks),
            "distinct_semantic_candidate_commands": len(distinct),
            "gate_pass": (
                len(non_exact) >= 20
                and len(tasks) >= 10
                and bool(distinct)
            ),
        }
    return result


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the frozen metrics and abandon decision to oracle rows."""

    arms = ("task_aware", "static_candidate_oracle", "perfect_tool_oracle")
    metrics = {arm: _arm_metrics(rows, arm) for arm in arms}

    def comparison(
        arm: str, *, active_tool: str | None = None
    ) -> dict[str, Any]:
        selected_arm = arm
        selected_rows: Sequence[Mapping[str, Any]] = rows
        if active_tool is not None:
            selected_arm = "tool_marginal_oracle"
            selected_rows = [
                {
                    **row,
                    "arms": {
                        **row["arms"],
                        selected_arm: row["arms"][
                            arm if active_tool in row["tools"] else "task_aware"
                        ],
                    },
                }
                for row in rows
            ]
        bootstrap = _paired_macro_bootstrap(
            selected_rows, selected_arm, "task_aware"
        )
        changes = _changes(selected_rows, selected_arm, "task_aware")
        gain_pp = 100.0 * bootstrap["point_estimate"]
        return {
            "macro_gain_percentage_points": gain_pp,
            "bootstrap": bootstrap,
            "changes": changes,
            "materially_positive": (
                gain_pp >= 1.0
                and bootstrap["interval_95"][0] > 0.0
                and len(changes["helpful_task_ids"]) >= 10
            ),
        }

    comparisons = {arm: comparison(arm) for arm in arms[1:]}
    coverage = _coverage(rows)
    per_tool = {}
    tool_decisions = {}
    for tool in ("pip", "pytest"):
        selected = [row for row in rows if tool in row["tools"]]
        tool_comparisons = {
            arm: comparison(arm, active_tool=tool) for arm in arms[1:]
        }
        if not coverage[tool]["gate_pass"]:
            tool_decision = "uninformative_coverage"
        elif tool_comparisons["static_candidate_oracle"]["materially_positive"]:
            tool_decision = "keep_static_family"
        elif tool_comparisons["perfect_tool_oracle"]["materially_positive"]:
            tool_decision = "close_static_retain_richer_state_or_compound_modeling"
        else:
            tool_decision = "close_tool_specific_modeling_on_sqlglot"
        tool_decisions[tool] = tool_decision
        per_tool[tool] = {
            "commands": len(selected),
            "metrics": {arm: _arm_metrics(selected, arm) for arm in arms} if selected else None,
            "marginal_full_cohort_comparisons": tool_comparisons,
        }
    decisions = set(tool_decisions.values())
    decision = next(iter(decisions)) if len(decisions) == 1 else "mixed_by_tool"
    non_tool_identity = all(
        row["arms"]["static_candidate_oracle"] == row["arms"]["task_aware"]
        and row["arms"]["perfect_tool_oracle"] == row["arms"]["task_aware"]
        for row in rows
        if not row["tools"]
    )
    availability_identity = all(
        (row["arms"][arm]["probability_by_bucket"][target] is None)
        == (row["arms"]["task_aware"]["probability_by_bucket"][target] is None)
        for row in rows
        for arm in arms[1:]
        for target in ALL_TARGETS
    )
    return {
        "schema": "pip-pytest-upper-bound-v1",
        "claim_bearing": False,
        "development_exposed": True,
        "coverage": coverage,
        "metrics": metrics,
        "comparisons": comparisons,
        "per_tool": per_tool,
        "tool_decisions": tool_decisions,
        "integrity": {
            "non_tool_rows_bit_identical": non_tool_identity,
            "prediction_availability_identical": availability_identity,
        },
        "decision": decision,
    }


def _load_frozen_components(
    scored_task_limit: int | None,
) -> tuple[list[dict[str, Any]], list[Any], list[Any], dict[str, Any]]:
    task_ids, _clauses, commands, _events, split = _load_development_stream()
    warmup_ids = list(split["cohorts"]["sqlglot"]["development_warmup"])
    scored_ids = list(split["cohorts"]["sqlglot"]["development_scored"])
    if task_ids != warmup_ids + scored_ids:
        raise ValueError("development stream differs from the frozen split")
    if scored_task_limit is not None:
        if not 0 < scored_task_limit <= len(scored_ids):
            raise ValueError("profile scored-task limit is invalid")
        scored_ids = scored_ids[:scored_task_limit]

    commands_by_task: dict[str, list[Any]] = defaultdict(list)
    for row in commands:
        commands_by_task[row.task_id].append(row)
    warmup_commands = [
        row for task_id in warmup_ids for row in commands_by_task[task_id]
    ]
    scored_commands = [
        row for task_id in scored_ids for row in commands_by_task[task_id]
    ]

    baseline_path = _GENERATION_OUTPUT / "rows.jsonl"
    baseline_all = [
        json.loads(line) for line in baseline_path.read_text(encoding="utf-8").splitlines()
    ]
    selected_ids = set(scored_ids)
    baseline_rows = [row for row in baseline_all if row["task_id"] in selected_ids]
    expected_ids = [f"{row.task_id}:{row.call_index}" for row in scored_commands]
    if [row["sample_id"] for row in baseline_rows] != expected_ids:
        raise ValueError("frozen Task-Aware rows differ from the scored command stream")
    for baseline, command in zip(baseline_rows, scored_commands, strict=True):
        if baseline["command"] != command.command or baseline["labels"] != _labels(command):
            raise ValueError("frozen Task-Aware command or labels differ")

    fit_rows = [_history_row(row) for row in warmup_commands]
    validation_rows = []
    for command, baseline in zip(scored_commands, baseline_rows, strict=True):
        clause = baseline["arms"]["clause_kb"]
        validation_rows.append(
            _history_row(
                command,
                clause["prediction"],
                clause["probability_by_bucket"],
            )
        )
    return baseline_rows, fit_rows, validation_rows, {
        "warmup_tasks": len(warmup_ids),
        "scored_tasks": len(scored_ids),
        "scored_commands": len(scored_commands),
        "baseline_rows": str(baseline_path.resolve()),
    }


def _evaluate(scored_task_limit: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.monotonic()
    baseline_rows, fit_rows, validation_rows, inputs = _load_frozen_components(
        scored_task_limit
    )
    pytest_rows, semantic_rows = _materialize_candidates(fit_rows, validation_rows)
    rows = build_oracle_rows(baseline_rows, pytest_rows, semantic_rows)
    result = summarize(rows)
    result["protocol"] = {
        "baseline": "frozen_task_aware",
        "targets": {"latency_buckets": 5, "cpu_rss_disk_buckets": 3},
        "causal_update": "whole_task_settlement",
        "minimum_macro_gain_percentage_points": 1.0,
        "minimum_helpful_tasks": 10,
        "bootstrap": {"unit": "task", "draws": 2000, "seed": 0},
    }
    result["inputs"] = {
        **inputs,
        "protocol": str(
            (
                Path(__file__).resolve().parents[2]
                / "analysis/development/pip-pytest-upper-bound-protocol.md"
            ).resolve()
        ),
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    }
    result["cost"] = {"evaluation_seconds": time.monotonic() - started}
    return result, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-scored-tasks", type=int)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    if (args.profile_scored_tasks is None) == (args.out_dir is None):
        parser.error("choose exactly one of --profile-scored-tasks or --out-dir")
    result, rows = _evaluate(args.profile_scored_tasks)
    if args.profile_scored_tasks is not None:
        print(
            json.dumps(
                {
                    "profile_only": True,
                    "scored_tasks": result["inputs"]["scored_tasks"],
                    "scored_commands": result["inputs"]["scored_commands"],
                    "evaluation_seconds": result["cost"]["evaluation_seconds"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.out_dir.exists():
        raise FileExistsError("output directory already exists")
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


__all__ = ["build_oracle_rows", "summarize"]


if __name__ == "__main__":
    main()
