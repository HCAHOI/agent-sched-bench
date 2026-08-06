#!/usr/bin/env python3
"""Test frozen command latency PMFs as CacheWise KV eviction signals."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from scripts.evaluation.evaluate_clause_resource_classes import (  # noqa: E402
    load_run_rows,
)
from scripts.evaluation.evaluate_command_history_residual import (  # noqa: E402
    Row,
    _load_rows,
)
from scripts.evaluation.evaluate_semantic_work_units import (  # noqa: E402
    run as run_semantic_work_units,
)
from tool_resource.runtime_kb import CANONICAL_LATENCY_BUCKETS  # noqa: E402
from tool_resource_eval.cachewise_kv_factorial import (  # noqa: E402
    BLOCK_SIZE_TOKENS,
    CAPACITY_BLOCKS,
    CAPACITY_TOKENS,
    LOAD,
    SEEDS,
    Program,
    Session,
    _bootstrap,
    _program,
    simulate,
)
from tool_resource_eval.cachewise_reproduction import (  # noqa: E402
    Gap,
    _histories,
    _predict,
    fit_clusters,
    load_gaps,
)
from trace_collect.tool_gap_extractor import extract_tool_gap_windows  # noqa: E402


VERSION = "kv-prediction-actionability-v1"
ARMS = ("lru", "tool", "c100", "current", "sota", "next_reuse_oracle")
MINIMUM_REDUCTION = 0.10
DEVELOPMENT_RUN = (
    _ROOT
    / "traces/swe-rebench/gpt-5.6-sol"
    / "sqlglot-100-c2-fast-requested-ebpf-a0419d9-20260803"
)
VALIDATION_RUN = (
    _ROOT
    / "traces/swe-rebench/gpt-5.6-sol"
    / "sqlglot-prev100-c2-fast-requested-ebpf-20260804"
)
RESULTS = _ROOT / "analysis/results/tool-resource-5-3-3-3-20260804"
SPLIT = _ROOT / "analysis/development/sqlglot-relational-task-split.json"
FIT_ROWS = RESULTS / "sqlglot20-80-current-fit-v1/rows.jsonl"
PHASE_ROWS = RESULTS / "sqlglot50-full-test-phase-validation-v1/rows.jsonl"
PHASE_RESULT = RESULTS / "sqlglot50-full-test-phase-validation-v1/result.json"
PHASE_ARTIFACT = RESULTS / "sqlglot100-full-test-phase-fresh-fit-v3/artifact.json"
CallKey = tuple[str, str]
DecisionKey = tuple[str, int]


def _remaining_from_pmf(
    pmf: Sequence[float],
    elapsed_s: float,
    durations_by_bucket: Sequence[np.ndarray],
) -> float | None:
    """Decode a bucket PMF into conditional empirical remaining time."""

    numerator = denominator = 0.0
    for probability, durations in zip(pmf, durations_by_bucket, strict=True):
        survivors = durations[durations > elapsed_s]
        if not len(survivors):
            continue
        survival = len(survivors) / len(durations)
        denominator += probability * survival
        numerator += probability * survival * float(
            np.mean(survivors - elapsed_s)
        )
    return None if denominator == 0.0 else numerator / denominator


def _trace_path(run: Path, task_id: str) -> Path:
    path = run / task_id / "attempt_1/trace.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _gap_windows(path: Path) -> list[tuple[Gap, str]]:
    gaps = load_gaps(path)
    windows = extract_tool_gap_windows(path)
    if len(gaps) != len(windows):
        raise ValueError(f"{path}: CacheWise gaps and tool windows differ")
    output = []
    for gap, window in zip(gaps, windows, strict=True):
        if window.tool_count != 1 or len(window.tool_call_ids) != 1:
            raise ValueError(f"{path}: expected one tool call per gap")
        if not math.isclose(
            gap.end - gap.start, window.available_gap_ms / 1000.0, abs_tol=1e-6
        ):
            raise ValueError(f"{path}: gap duration differs from tool window")
        output.append((gap, window.tool_call_ids[0]))
    call_ids = [call_id for _gap, call_id in output]
    if len(set(call_ids)) != len(call_ids):
        raise ValueError(f"{path}: one tool call ID appears in multiple gaps")
    return output


def _dev100_static_rows(values: Sequence[Mapping[str, Any]]) -> list[Row]:
    """Load the dev100 snapshot stored under the evaluator's legacy arm name."""

    rows = []
    for value in values:
        frozen = dict(value["frozen_at_80"])
        pmfs = frozen.pop("probability_by_bucket")
        for target in pmfs:
            frozen.setdefault(target, None)
        rows.append(
            Row(
                str(value["sample_id"]),
                str(value["task_id"]),
                str(value["command"]),
                dict(value["labels"]),
                frozen,
                {
                    target: None if pmf is None else tuple(float(item) for item in pmf)
                    for target, pmf in pmfs.items()
                },
            )
        )
    return rows


def _static_predictions(
    validation_ids: Sequence[str],
) -> tuple[
    dict[CallKey, tuple[float, ...]],
    dict[CallKey, tuple[float, ...]],
    dict[str, Any],
]:
    values = [json.loads(line) for line in PHASE_ROWS.read_text().splitlines()]
    task_order = list(dict.fromkeys(str(value["task_id"]) for value in values))
    call_keys = [
        (str(value["task_id"]), str(value["call_id"])) for value in values
    ]
    phase_result = json.loads(PHASE_RESULT.read_text())
    if (
        task_order != list(validation_ids)
        or len(values) != 1044
        or len(set(call_keys)) != len(call_keys)
        or phase_result.get("schema") != "full-test-phase-fresh-evaluation-v1"
        or phase_result.get("role") != "validation"
        or phase_result.get("development_run") != str(DEVELOPMENT_RUN.resolve())
        or phase_result.get("row_identity")
        != {"commands": 1044, "evidence_valid_tasks": 50, "identical_rows": True}
    ):
        raise ValueError("phase rows differ from the exposed validation population")
    fit = _load_rows(FIT_ROWS)
    # evaluate_prequential_commands keeps the historical key `frozen_at_80`
    # for every warm-up size. This artifact's reviewed caller used dev100 and
    # warmup_task_count=100; the result provenance above locks that source.
    base = _dev100_static_rows(values)
    base_by_task: dict[str, list[Row]] = defaultdict(list)
    for row in base:
        base_by_task[row.task_id].append(row)
    semantic_by_sample: dict[str, dict[str, Any]] = {}
    for task_id in validation_ids:
        _result, task_rows = run_semantic_work_units(fit, base_by_task[task_id])
        semantic_by_sample.update((row["sample_id"], row) for row in task_rows)
    if set(semantic_by_sample) != {row.sample_id for row in base}:
        raise ValueError("static semantic rows differ from the validation population")

    phase_pmf = tuple(
        float(value)
        for value in json.loads(PHASE_ARTIFACT.read_text())["pmfs"]["latency"]
    )
    phase_hard = max(range(len(phase_pmf)), key=phase_pmf.__getitem__)
    current: dict[CallKey, tuple[float, ...]] = {}
    sota: dict[CallKey, tuple[float, ...]] = {}
    labels: dict[CallKey, int] = {}
    phase_raised = unavailable = 0
    for value, base_row in zip(values, base, strict=True):
        sample_id = base_row.sample_id
        current_pmf = base_row.pmfs["latency"]
        semantic = semantic_by_sample[sample_id]["arms"]["semantic_work_units"]
        semantic_pmf_value = semantic["candidate_probability_by_bucket"]["latency"]
        semantic_pmf = (
            None
            if semantic_pmf_value is None
            else tuple(float(item) for item in semantic_pmf_value)
        )
        call_key = (base_row.task_id, str(value["call_id"]))
        if current_pmf is not None:
            current[call_key] = current_pmf
        labels[call_key] = int(base_row.labels["latency"])
        if semantic_pmf is None:
            unavailable += 1
            continue
        selected = semantic_pmf
        semantic_hard = max(range(len(selected)), key=selected.__getitem__)
        current_hard = base_row.current["latency"]
        if (
            value["full_test_phase"] == 2
            and current_hard is not None
            and phase_hard > int(current_hard)
            and phase_hard > semantic_hard
        ):
            selected = phase_pmf
            phase_raised += 1
        sota[call_key] = selected

    def accuracy(predictions: Mapping[CallKey, Sequence[float]]) -> float:
        return sum(
            max(range(len(pmf)), key=pmf.__getitem__) == labels[call_key]
            for call_key, pmf in predictions.items()
        ) / len(labels)

    return current, sota, {
        "rows": len(values),
        "current_available": len(current),
        "sota_available": len(sota),
        "sota_unavailable": unavailable,
        "phase_raised": phase_raised,
        "current_exact_accuracy": accuracy(current),
        "sota_exact_accuracy": accuracy(sota),
    }


def _cachewise_predictor(
    arm: str,
    global_history: np.ndarray,
    tool_history: dict[str, np.ndarray],
    clusters: dict[int, Any],
    label_cache: dict[tuple[int, str, str], int],
):
    def predict(session: Session, now_s: float) -> float:
        if session.arrival_s <= now_s or session.turn_index == 0:
            return 0.0
        gap = session.program.turns[session.turn_index - 1].gap
        if gap is None or session.gap_started_s is None:
            return 0.0
        return _predict(
            gap,
            max(0.0, now_s - session.gap_started_s),
            arm,
            global_history,
            tool_history,
            clusters,
            label_cache,
        )

    return predict


def _pmf_predictor(
    pmfs: Mapping[DecisionKey, tuple[float, ...]],
    durations_by_bucket: Sequence[np.ndarray],
    fallback,
):
    def predict(session: Session, now_s: float) -> float:
        if session.arrival_s <= now_s or session.turn_index == 0:
            return 0.0
        gap = session.program.turns[session.turn_index - 1].gap
        if gap is None or session.gap_started_s is None:
            return 0.0
        elapsed = max(0.0, now_s - session.gap_started_s)
        pmf = pmfs.get((session.program.task_id, session.turn_index - 1))
        remaining = (
            None
            if pmf is None
            else _remaining_from_pmf(pmf, elapsed, durations_by_bucket)
        )
        return fallback(session, now_s) if remaining is None else remaining

    return predict


def _oracle_predictor(session: Session, now_s: float) -> float:
    return max(0.0, session.arrival_s - now_s)


def _relative_reduction(baseline: float, candidate: float) -> float | None:
    return None if baseline == 0.0 else (baseline - candidate) / baseline


def run() -> dict[str, Any]:
    split = json.loads(SPLIT.read_text())
    development_ids = list(split["development"])
    validation_ids = list(split["validation"])
    if len(development_ids) != 100 or len(validation_ids) != 50:
        raise ValueError("task split differs from the frozen protocol")

    development_task_ids, _clauses, development_commands = load_run_rows(
        DEVELOPMENT_RUN
    )
    if development_task_ids != development_ids:
        raise ValueError("development task order differs from the frozen protocol")

    development_windows = {
        task_id: _gap_windows(_trace_path(DEVELOPMENT_RUN, task_id))
        for task_id in development_ids
    }
    fit_gaps = [
        gap
        for task_id in development_ids
        for gap, _call_id in development_windows[task_id]
    ]
    global_history, tool_history = _histories(fit_gaps)
    clusters, occupied = fit_clusters(fit_gaps, cluster_counts=(100,))
    label_cache: dict[tuple[int, str, str], int] = {}

    command_by_call = {
        (row.task_id, row.call_id): row for row in development_commands
    }
    if len(command_by_call) != len(development_commands):
        raise ValueError("development commands contain duplicate call IDs")
    durations: list[list[float]] = [
        [] for _ in range(CANONICAL_LATENCY_BUCKETS.bucket_count)
    ]
    mapped_development: set[CallKey] = set()
    for task_id in development_ids:
        for gap, call_id in development_windows[task_id]:
            call_key = (task_id, call_id)
            command = command_by_call.get(call_key)
            if command is None:
                continue
            bucket = CANONICAL_LATENCY_BUCKETS.bucket_id(command.duration_ms)
            durations[bucket].append(gap.end - gap.start)
            mapped_development.add(call_key)
    if (
        len(mapped_development) != sum(len(values) for values in durations)
        or any(not values for values in durations)
    ):
        raise ValueError("development gap decoder is duplicated or misses a bucket")
    duration_arrays = tuple(np.asarray(values) for values in durations)

    current_by_call, sota_by_call, prediction_coverage = _static_predictions(
        validation_ids
    )
    programs: dict[str, Program] = {}
    current_by_decision: dict[DecisionKey, tuple[float, ...]] = {}
    sota_by_decision: dict[DecisionKey, tuple[float, ...]] = {}
    mapped_validation: set[CallKey] = set()
    gap_count = 0
    for task_id in validation_ids:
        path = _trace_path(VALIDATION_RUN, task_id)
        programs[task_id] = _program(path)
        for gap_index, (_gap, call_id) in enumerate(_gap_windows(path)):
            gap_count += 1
            call_key = (task_id, call_id)
            decision_key = (task_id, gap_index)
            if call_key in current_by_call:
                current_by_decision[decision_key] = current_by_call[call_key]
                mapped_validation.add(call_key)
            if call_key in sota_by_call:
                sota_by_decision[decision_key] = sota_by_call[call_key]
    if mapped_validation != set(current_by_call):
        raise ValueError("validation Current PMFs do not map one-to-one to gaps")
    program_decisions = {
        (program.task_id, turn_index)
        for program in programs.values()
        for turn_index, turn in enumerate(program.turns)
        if turn.gap is not None
    }
    if (
        len(program_decisions) != gap_count
        or len(current_by_decision) != len(current_by_call)
        or not set(current_by_decision) <= program_decisions
    ):
        raise ValueError("predictor gap keys differ from simulator program gaps")

    c100 = _cachewise_predictor(
        "c100", global_history, tool_history, clusters, label_cache
    )
    predictors = {
        "tool": _cachewise_predictor(
            "tool", global_history, tool_history, clusters, label_cache
        ),
        "c100": c100,
        "current": _pmf_predictor(current_by_decision, duration_arrays, c100),
        "sota": _pmf_predictor(sota_by_decision, duration_arrays, c100),
        "next_reuse_oracle": _oracle_predictor,
    }

    schedule_rows = []
    for seed in SEEDS:
        selected = sorted(programs)
        np.random.default_rng(seed).shuffle(selected)
        selected = selected[:LOAD]
        arms = {
            "lru": simulate(
                [programs[task_id] for task_id in selected],
                scheduler="fcfs",
                eviction="lru",
                global_history=global_history,
                tool_history=tool_history,
                clusters=clusters,
                label_cache=label_cache,
            )
        }
        for arm, predictor in predictors.items():
            arms[arm] = simulate(
                [programs[task_id] for task_id in selected],
                scheduler="fcfs",
                eviction="predicted",
                global_history=global_history,
                tool_history=tool_history,
                clusters=clusters,
                label_cache=label_cache,
                remaining_predictor=predictor,
            )
        schedule_rows.append({"seed": seed, "task_ids": selected, "arms": arms})

    metrics = ("recomputed_prefix_blocks", "evicted_blocks", "eviction_events")
    means = {
        arm: {
            metric: float(np.mean([row["arms"][arm][metric] for row in schedule_rows]))
            for metric in metrics
        }
        for arm in ARMS
    }
    c100_primary = means["c100"]["recomputed_prefix_blocks"]
    comparisons = {}
    for arm in ("current", "sota", "next_reuse_oracle"):
        deltas = [
            float(row["arms"][arm]["recomputed_prefix_blocks"])
            - float(row["arms"]["c100"]["recomputed_prefix_blocks"])
            for row in schedule_rows
        ]
        comparisons[arm] = {
            "candidate": arm,
            "baseline": "c100",
            "metric": "recomputed_prefix_blocks; lower is better",
            "relative_reduction_of_means": _relative_reduction(
                c100_primary, means[arm]["recomputed_prefix_blocks"]
            ),
            **_bootstrap(deltas),
        }
    sota_ci_high = comparisons["sota"]["ci95_paired_seed_bootstrap"][1]
    oracle_headroom = comparisons["next_reuse_oracle"]["relative_reduction_of_means"]
    sota_gain = comparisons["sota"]["relative_reduction_of_means"]
    gate = {
        "c100_nonzero": c100_primary > 0.0,
        "oracle_reduction_at_least_10_percent": oracle_headroom is not None
        and oracle_headroom >= MINIMUM_REDUCTION,
        "sota_reduction_at_least_10_percent": sota_gain is not None
        and sota_gain >= MINIMUM_REDUCTION,
        "sota_paired_ci_below_zero": sota_ci_high < 0.0,
        "sota_strictly_better_than_current": means["sota"]["recomputed_prefix_blocks"]
        < means["current"]["recomputed_prefix_blocks"],
        "sota_evicted_blocks_no_worse_than_c100": means["sota"]["evicted_blocks"]
        <= means["c100"]["evicted_blocks"],
    }
    gate["go"] = all(gate.values())
    return {
        "schema": VERSION,
        "status": "development_go_to_live_kv" if gate["go"] else "development_no_go",
        "claim_bearing": False,
        "protocol": {
            "scheduler": "fcfs",
            "arms": list(ARMS),
            "load": LOAD,
            "seeds": list(SEEDS),
            "capacity_tokens": CAPACITY_TOKENS,
            "capacity_blocks": CAPACITY_BLOCKS,
            "block_size_tokens": BLOCK_SIZE_TOKENS,
            "prediction_fallback": "c100",
            "validation_updates": False,
            "primary": "mean recomputed_prefix_blocks; lower is better",
            "guardrail": "mean evicted_blocks no worse than c100",
            "minimum_relative_reduction": MINIMUM_REDUCTION,
        },
        "coverage": {
            "development_gaps": len(fit_gaps),
            "development_eligible_commands": len(development_commands),
            "decoder_eligible_gaps": len(mapped_development),
            "decoder_commands_without_next_llm_gap": len(development_commands)
            - len(mapped_development),
            "decoder_gap_counts_by_bucket": [len(values) for values in durations],
            "validation_tasks": len(programs),
            "validation_gaps": gap_count,
            "validation_predictor_gaps": len(current_by_decision),
            **prediction_coverage,
        },
        "cluster_model": {
            "c100_tool_batch_keys": len(clusters[100]),
            "c100_occupied_clusters": sum(occupied[100].values()),
        },
        "mean_metrics": means,
        "comparisons": comparisons,
        "gate": gate,
        "schedule_results": schedule_rows,
        "limitations": [
            "The validation workload and predictor selection are development-exposed.",
            "The next-reuse oracle is greedy, hindsight-only, and not globally optimal.",
            "Cache misses do not feed back into service time, so latency metrics are not utility evidence.",
            "Capacity is a paper-scale estimate rather than a measured live vLLM allocation.",
        ],
    }


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("output directory already exists")
    result = run()
    result["inputs"] = {
        "development_run": str(DEVELOPMENT_RUN.resolve()),
        "validation_run": str(VALIDATION_RUN.resolve()),
        "fit_rows": str(FIT_ROWS.resolve()),
        "phase_rows": str(PHASE_ROWS.resolve()),
        "phase_result": str(PHASE_RESULT.resolve()),
        "phase_artifact": str(PHASE_ARTIFACT.resolve()),
        "split": str(SPLIT.resolve()),
        "git_sha": _git_sha(),
    }
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
