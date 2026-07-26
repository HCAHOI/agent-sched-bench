#!/usr/bin/env python3
"""Evaluate completed-task profile updates against frozen baselines.

SWE-ReBench-100 initializes one prefix profile and fixed robust-clock gate.
Five parallel Fresh-277 lanes each warm up on four folds and score the held-out
fold under one predeclared seed-0 task order. No result from this script is a
deployment certificate.
"""

from __future__ import annotations

import argparse
import atexit
import datetime as dt
import json
import math
import os
import platform
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_GLOBS = ("src/**/*.py", "scripts/**/*.py")
PREQUENTIAL_ARTIFACT_SCHEMA_VERSION = 4


def _source_snapshot_records() -> Iterable[dict[str, Any]]:
    """Archive the executable source tree for dirty-tree reproducibility."""

    paths = sorted(
        {
            path
            for pattern in _SOURCE_GLOBS
            for path in _REPO_ROOT.glob(pattern)
            if path.is_file()
        }
    )
    for path in paths:
        relative_path = str(path.relative_to(_REPO_ROOT))
        yield {
            "record_type": "source_snapshot",
            "path": relative_path,
            "encoding": "utf-8",
            "content": path.read_bytes().decode("utf-8"),
        }


_SOURCE_SNAPSHOT_AT_IMPORT = tuple(_source_snapshot_records())
sys.path.insert(0, str(_REPO_ROOT))

from tool_time.offline_evaluation import (  # noqa: E402
    evaluate_offline_probe_clock,
)
from tool_time.policy import (  # noqa: E402
    trigger_policy_utility_ms,
)
from tool_time.prequential import (  # noqa: E402
    benchmark_profile_updates,
    evaluate_prequential_updates,
)
from trace_collect.tool_gap_extractor import discover_trace_files  # noqa: E402
from trace_collect.tool_latency_dataset import (  # noqa: E402
    ToolLatencySample,
    extract_many_tool_latency_samples,
    read_tool_latency_corpus_manifest,
    require_explicit_trace_task_ids,
)

_DYNAMIC_ARMS = ("task",)
_PANEL_NAMES = (
    "frozen_100",
    "fresh4_static",
    "warmup_snapshot",
    "task",
)
_INITIAL_MANIFEST = (
    "configs/corpora/swe-100.json"
)
_DEVELOPMENT_MANIFEST = (
    "configs/corpora/swe-277.json"
)
_INITIAL_COLLECTION = "swe-rebench-qwen3.7-max-seed42-offset50-100-complete-v2"
_DEVELOPMENT_COLLECTION = "swe-rebench-qwen3.7-max-fresh-seed42-skip150-n277"
_INITIAL_TASK_COUNT = 100
_DEVELOPMENT_TASK_COUNT = 277
_OUTER_FOLDS = 5
_OUTPUTS = {
    "json": "analysis/results/prequential-task-update-task-only/prequential-task-update.json",
    "markdown": "analysis/results/prequential-task-update-task-only/prequential-task-update.md",
}
_PAIR_FIELDS = {
    "frozen_100_vs_deadline": ("threshold_ms", "frozen_100_trigger_ms"),
    "fresh4_static_vs_deadline": ("threshold_ms", "fresh4_static_trigger_ms"),
    "warmup_snapshot_vs_deadline": ("threshold_ms", "warmup_snapshot_trigger_ms"),
    "task_vs_deadline": ("threshold_ms", "task_trigger_ms"),
    "fresh4_static_vs_frozen_100": (
        "frozen_100_trigger_ms",
        "fresh4_static_trigger_ms",
    ),
    "warmup_snapshot_vs_frozen_100": (
        "frozen_100_trigger_ms",
        "warmup_snapshot_trigger_ms",
    ),
    "task_vs_frozen_100": ("frozen_100_trigger_ms", "task_trigger_ms"),
    "task_vs_warmup_snapshot": (
        "warmup_snapshot_trigger_ms",
        "task_trigger_ms",
    ),
}
_DYNAMIC_FIELDS = (
    "trigger_ms",
    "candidate_trigger_ms",
    "candidate_margin_normalized",
    "prior_source",
    "prior_group_key",
    "prior_task_count",
    "model_version",
    "profile_row_count",
    "profile_task_count",
    "score_panel_runtime_ms",
)


class _ZstdJsonlWriter:
    """Stream complete records without retaining the million-row sidecar."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        if self.staging.exists():
            raise FileExistsError(f"stale sidecar staging file: {self.staging}")
        self.process = subprocess.Popen(
            ["zstd", "-q", "-f", "-o", str(self.staging), "-"],
            stdin=subprocess.PIPE,
            text=True,
        )
        if self.process.stdin is None:
            raise RuntimeError("zstd stdin pipe was not created")
        self.count = 0
        self._finished = False
        atexit.register(self.abort)

    def write_many(self, rows: Iterable[Mapping[str, Any]]) -> None:
        if self._finished:
            raise RuntimeError("cannot write to a finished sidecar")
        assert self.process.stdin is not None
        for row in rows:
            self.process.stdin.write(
                json.dumps(
                    row,
                    default=list,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
            self.count += 1

    def finish(self) -> None:
        if self._finished:
            raise RuntimeError("sidecar already finished")
        assert self.process.stdin is not None
        self.process.stdin.close()
        return_code = self.process.wait()
        if return_code != 0:
            self.staging.unlink(missing_ok=True)
            raise RuntimeError(f"zstd sidecar writer exited with {return_code}")
        self.staging.replace(self.path)
        self._finished = True
        atexit.unregister(self.abort)

    def abort(self) -> None:
        if self._finished:
            return
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait()
        self.staging.unlink(missing_ok=True)


def _cleanup_partial_outputs(paths: Iterable[Path]) -> None:
    """Remove named partials and their PID-scoped writer staging files."""

    for path in paths:
        path.unlink(missing_ok=True)
        for staging in path.parent.glob(f".{path.name}.*.tmp"):
            staging.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/prequential_profile_update.yaml"),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Maximum worker processes across independent fold-arm evaluations.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    initial_manifest_path = _resolve(Path(config["initialization_manifest"]))
    development_manifest_path = _resolve(Path(config["development_manifest"]))
    initial_manifest, initial_task_ids, initial_samples = _load_corpus(
        initial_manifest_path
    )
    development_manifest, development_task_ids, development_samples = _load_corpus(
        development_manifest_path
    )
    if initial_manifest["collection_id"] != _INITIAL_COLLECTION:
        raise ValueError("initialization manifest collection_id changed")
    if development_manifest["collection_id"] != _DEVELOPMENT_COLLECTION:
        raise ValueError("development manifest collection_id changed")
    if len(initial_task_ids) != _INITIAL_TASK_COUNT:
        raise ValueError("initialization corpus must contain exactly 100 tasks/traces")
    if len(development_task_ids) != _DEVELOPMENT_TASK_COUNT:
        raise ValueError("development corpus must contain exactly 277 tasks/traces")
    overlap = set(initial_task_ids) & set(development_task_ids)
    if overlap:
        raise ValueError(
            f"initialization and development tasks overlap: {sorted(overlap)}"
        )

    out_json = _resolve(Path(config["outputs"]["json"]))
    out_md = _resolve(Path(config["outputs"]["markdown"]))
    records_path = out_json.with_name(out_json.stem + "-records.jsonl.zst")
    fold_paths = [
        out_json.with_name(out_json.stem + f"-f{fold}-records.jsonl.zst")
        for fold in range(1, _OUTER_FOLDS + 1)
    ]
    sidecar_partial_paths = [
        path.with_name(path.name + ".partial") for path in [records_path, *fold_paths]
    ]
    out_json_partial = out_json.with_name(out_json.name + ".partial")
    out_md_partial = out_md.with_name(out_md.name + ".partial")
    partial_paths = [*sidecar_partial_paths, out_json_partial, out_md_partial]
    for output in (out_json, out_md, records_path, *fold_paths, *partial_paths):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output}")

    fit_costs = [float(value) for value in config["fit_kv_costs_ms"]]
    score_costs = [float(value) for value in config["score_kv_costs_ms"]]
    for label, manifest in (
        ("initialization", initial_manifest),
        ("development", development_manifest),
    ):
        if fit_costs != [float(value) for value in manifest["costs_ms"]]:
            raise ValueError(
                f"fit_kv_costs_ms must exactly match the {label} manifest panel"
            )
    profile_rows = _sample_rows(initial_samples, set(initial_task_ids))
    development_rows = _sample_rows(development_samples, set(development_task_ids))
    static = evaluate_offline_probe_clock(
        development_rows,
        profile_rows=profile_rows,
        kv_costs_ms=fit_costs,
        guard_ms=float(config["guard_ms"]),
        inner_folds=int(initial_manifest["inner_folds"]),
        min_tool_history=int(config["min_tool_history"]),
        min_profile_tasks=int(config["min_profile_tasks"]),
        command_field=str(config["command_field"]),
        max_prefix_depth=int(config["max_prefix_depth"]),
        skip_leading_cd=bool(config["skip_leading_cd"]),
        restore_cost_fraction=float(config["restore_cost_fraction"]),
        include_calibration_trace=True,
    )
    selected_guard = static["robust_calibration"]["selected_guard_normalized"]
    calibration_trace = static["calibration_trace"]
    frozen_score_rows = [
        row
        for row in static["decisions"]
        if float(row["kv_cost_ms"]) in set(score_costs)
    ]
    run_started = dt.datetime.now().isoformat(timespec="seconds")

    initial_partial = partial_paths[0]
    writer = _ZstdJsonlWriter(initial_partial)
    try:
        writer.write_many(
            [
                {
                    "record_type": "run_metadata",
                    "schema_version": PREQUENTIAL_ARTIFACT_SCHEMA_VERSION,
                    "status": "development_only_exploratory",
                    "certificate": False,
                    "run_started": run_started,
                    "config": config,
                    "initialization_collection_id": initial_manifest["collection_id"],
                    "development_collection_id": development_manifest["collection_id"],
                }
            ]
        )
        writer.write_many(_SOURCE_SNAPSHOT_AT_IMPORT)
        writer.write_many(
            {"record_type": "initial_policy_decision", **row}
            for row in static["decisions"]
        )
        writer.write_many(
            {"record_type": "initial_calibration_fold", **fold}
            for fold in calibration_trace["folds"]
        )
        writer.write_many(
            {"record_type": "initial_calibration_probe_decision", **row}
            for row in calibration_trace["probe_decisions"]
        )
        writer.finish()
    except BaseException:
        writer.abort()
        raise

    fold_args = [
        (
            fold,
            config,
            profile_rows,
            development_rows,
            development_task_ids,
            frozen_score_rows,
            selected_guard,
            partial_paths[fold],
        )
        for fold in range(1, _OUTER_FOLDS + 1)
    ]
    try:
        fold_results = _run_folds(fold_args, workers=args.workers)
    except BaseException:
        _cleanup_partial_outputs(partial_paths)
        raise
    fold_results.sort(key=lambda result: int(result["outer_fold"]))
    test_sets = [set(result["test_tasks"]) for result in fold_results]
    if set().union(*test_sets) != set(development_task_ids) or sum(
        len(test) for test in test_sets
    ) != len(development_task_ids):
        _cleanup_partial_outputs(partial_paths)
        raise AssertionError("outer test folds do not partition Fresh-277 exactly")
    combined_run = _aggregate_fold_run(
        fold_results,
        run_name="primary",
        costs=score_costs,
    )
    sidecars = [
        {
            "role": "initialization",
            "file": records_path.name,
            "record_count": writer.count,
        },
        *[
            {
                "role": f"outer_fold_{result['outer_fold']}",
                "file": fold_paths[index].name,
                "record_count": int(result["records_sidecar_record_count"]),
            }
            for index, result in enumerate(fold_results)
        ],
    ]
    payload = {
        "schema_version": PREQUENTIAL_ARTIFACT_SCHEMA_VERSION,
        "protocol": {
            "date": config["protocol_date"],
            "status": config["status"],
            "certificate": False,
            "outer_folds": _OUTER_FOLDS,
            "causal_contract": (
                "SWE-ReBench-100 initializes one profile and fixed robust gate. "
                "Each Fresh-277 outer lane causally warms up on four folds, then "
                "scores its held-out fold exactly once. The held-out task arm "
                "publishes completed observations only at task boundaries; "
                "warmup_snapshot freezes at test entry; fresh4_static is a "
                "same-gate matched reference."
            ),
            "primary_order": "one global PCG64 seed-0 permutation, filtered per fold",
            "inference": (
                "exact paired totals for one predeclared adaptive task order; no "
                "bootstrap CI or sign-flip p-value for path-dependent decisions"
            ),
        },
        "provenance": {
            "generated": run_started,
            "git_sha": _git_sha(),
            "config": str(config_path.relative_to(_REPO_ROOT)),
            "initialization_manifest": str(
                initial_manifest_path.relative_to(_REPO_ROOT)
            ),
            "initialization_collection_id": initial_manifest["collection_id"],
            "initialization_task_count": len(initial_task_ids),
            "initialization_call_count": len(profile_rows),
            "development_manifest": str(
                development_manifest_path.relative_to(_REPO_ROOT)
            ),
            "development_collection_id": development_manifest["collection_id"],
            "development_task_count": len(development_task_ids),
            "development_call_count": len(development_rows),
            "timing_environment": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "logical_cpu_count": os.cpu_count(),
                "numpy": np.__version__,
                "pyyaml": yaml.__version__,
                "requested_worker_count": args.workers,
                "effective_worker_count": min(
                    args.workers,
                    _OUTER_FOLDS * len(_PANEL_NAMES) * len(score_costs),
                ),
                "parallel_unit": "outer_fold_arm_score_cost",
            },
            "source_snapshot": {
                "records_sidecar": records_path.name,
            },
        },
        "config": config,
        "records_sidecars": sidecars,
        "initial_policy": {
            "selected_guard_normalized": selected_guard,
            "calibration": static["calibration"],
            "robust_calibration": static["robust_calibration"],
            "calibration_fold_count": len(calibration_trace["folds"]),
            "calibration_probe_decision_count": len(
                calibration_trace["probe_decisions"]
            ),
        },
        "folds": fold_results,
        "primary": {"run": combined_run},
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    published: list[Path] = []
    try:
        out_json_partial.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        out_md_partial.write_text(_render_markdown(payload), encoding="utf-8")
        destinations = [records_path, *fold_paths, out_json, out_md]
        for partial, final in zip(partial_paths, destinations, strict=True):
            partial.replace(final)
            published.append(final)
    except BaseException:
        _cleanup_partial_outputs(partial_paths)
        for path in published:
            path.unlink(missing_ok=True)
        raise
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    for sidecar in sidecars:
        print(f"wrote {out_json.parent / sidecar['file']}")


def _run_folds(
    fold_args: Sequence[tuple[Any, ...]], *, workers: int
) -> list[dict[str, Any]]:
    """Evaluate independent fold-arm-cost cells in one flat process pool."""

    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        return [_run_fold(args) for args in fold_args]

    contexts = [_prepare_fold(args) for args in fold_args]
    score_cost_count = len(contexts[0]["config"]["score_kv_costs_ms"])
    max_workers = min(workers, len(contexts) * len(_PANEL_NAMES) * score_cost_count)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        benchmarks = list(pool.map(_benchmark_fold, contexts))
        jobs = [
            (context, timing, arm, float(cost))
            for context, timing in zip(contexts, benchmarks, strict=True)
            for arm in _PANEL_NAMES
            for cost in context["config"]["score_kv_costs_ms"]
        ]
        arm_outputs = list(pool.map(_evaluate_fold_arm, jobs))

    outputs_by_fold: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (context, _, arm, _), output in zip(jobs, arm_outputs, strict=True):
        outputs_by_fold[int(context["outer_fold"])][arm].append(output)
    return [
        _finalize_fold(
            context,
            timing,
            {
                arm: _merge_cost_arm_outputs(outputs)
                for arm, outputs in outputs_by_fold[int(context["outer_fold"])].items()
            },
        )
        for context, timing in zip(contexts, benchmarks, strict=True)
    ]


def _run_fold(args: tuple[Any, ...]) -> dict[str, Any]:
    """Serial reference path using the same fold components."""

    context = _prepare_fold(args)
    benchmarks = _benchmark_fold(context)
    costs = [float(value) for value in context["config"]["score_kv_costs_ms"]]
    arms = {
        arm: _merge_cost_arm_outputs(
            [_evaluate_fold_arm((context, benchmarks, arm, cost)) for cost in costs]
        )
        for arm in _PANEL_NAMES
    }
    return _finalize_fold(context, benchmarks, arms)


def _prepare_fold(args: tuple[Any, ...]) -> dict[str, Any]:
    (
        outer_fold,
        config,
        initialization_rows,
        development_rows,
        development_task_ids,
        frozen_score_rows,
        selected_guard,
        records_path,
    ) = args
    test_tasks, warmup_tasks = _fold_task_sets(
        development_task_ids, outer_fold=outer_fold
    )
    test_rows = [row for row in development_rows if str(row["task_id"]) in test_tasks]
    warmup_rows = [
        row for row in development_rows if str(row["task_id"]) in warmup_tasks
    ]
    primary_order = list(development_task_ids)
    np.random.default_rng(int(config["task_order_seed"])).shuffle(primary_order)
    return {
        "outer_fold": outer_fold,
        "config": config,
        "initialization_rows": initialization_rows,
        "test_rows": test_rows,
        "warmup_rows": warmup_rows,
        "base_profile": [*initialization_rows, *warmup_rows],
        "test_tasks": test_tasks,
        "warmup_tasks": warmup_tasks,
        "primary_test_order": [
            task_id for task_id in primary_order if task_id in test_tasks
        ],
        "primary_warmup_order": [
            task_id for task_id in primary_order if task_id in warmup_tasks
        ],
        "expected_frozen": [
            row for row in frozen_score_rows if str(row["task_id"]) in test_tasks
        ],
        "selected_guard": selected_guard,
        "records_path": records_path,
    }


def _benchmark_fold(context: Mapping[str, Any]) -> dict[str, list[Any]]:
    config = context["config"]

    def benchmark(
        profile: list[dict[str, Any]],
        rows: list[dict[str, Any]],
        order: list[str],
    ) -> list[Any]:
        return benchmark_profile_updates(
            profile,
            rows,
            task_order=order,
            command_field=str(config["command_field"]),
            max_prefix_depth=int(config["max_prefix_depth"]),
            skip_leading_cd=bool(config["skip_leading_cd"]),
        )

    return {
        "warmup": benchmark(
            context["initialization_rows"],
            context["warmup_rows"],
            context["primary_warmup_order"],
        ),
        "test": benchmark(
            context["base_profile"],
            context["test_rows"],
            context["primary_test_order"],
        ),
    }


def _evaluate_fold_arm(
    args: tuple[Mapping[str, Any], Mapping[str, Sequence[Any]], str, float],
) -> dict[str, Any]:
    context, benchmarks, arm, score_cost = args
    if arm not in _PANEL_NAMES:
        raise ValueError(f"unknown arm: {arm}")
    if score_cost not in {
        float(value) for value in context["config"]["score_kv_costs_ms"]
    }:
        raise ValueError(f"unknown score cost: {score_cost}")
    config = context["config"]
    score_costs = [score_cost]

    def evaluate(
        rows: list[dict[str, Any]],
        *,
        profile: list[dict[str, Any]],
        order: list[str],
        mode: str,
        runtimes: Mapping[str, float],
    ) -> dict[str, Any]:
        return evaluate_prequential_updates(
            rows,
            profile_rows=profile,
            task_order=order,
            update_mode=mode,
            update_runtime_ms=runtimes,
            kv_costs_ms=score_costs,
            guard_ms=float(config["guard_ms"]),
            selected_guard_normalized=context["selected_guard"],
            min_tool_history=int(config["min_tool_history"]),
            min_profile_tasks=int(config["min_profile_tasks"]),
            command_field=str(config["command_field"]),
            max_prefix_depth=int(config["max_prefix_depth"]),
            skip_leading_cd=bool(config["skip_leading_cd"]),
            restore_cost_fraction=float(config["restore_cost_fraction"]),
        )

    if arm == "frozen_100":
        test = evaluate(
            context["test_rows"],
            profile=context["initialization_rows"],
            order=context["primary_test_order"],
            mode="frozen",
            runtimes={},
        )
        _assert_frozen_identity(context["expected_frozen"], test["decisions"])
        return {
            "test": test,
            "warmup": evaluate(
                context["warmup_rows"],
                profile=context["initialization_rows"],
                order=context["primary_warmup_order"],
                mode="frozen",
                runtimes={},
            ),
        }
    if arm == "fresh4_static":
        return {
            "test": evaluate(
                context["test_rows"],
                profile=context["warmup_rows"],
                order=context["primary_test_order"],
                mode="frozen",
                runtimes={},
            )
        }
    if arm == "warmup_snapshot":
        return {
            "test": evaluate(
                context["test_rows"],
                profile=context["base_profile"],
                order=context["primary_test_order"],
                mode="frozen",
                runtimes={},
            )
        }

    warmup_runtimes = {
        benchmark.sample_id: benchmark.runtime_ms for benchmark in benchmarks["warmup"]
    }
    test_runtimes = {
        benchmark.sample_id: benchmark.runtime_ms for benchmark in benchmarks["test"]
    }
    return {
        "warmup": evaluate(
            context["warmup_rows"],
            profile=context["initialization_rows"],
            order=context["primary_warmup_order"],
            mode=arm,
            runtimes=warmup_runtimes,
        ),
        "test": evaluate(
            context["test_rows"],
            profile=context["base_profile"],
            order=context["primary_test_order"],
            mode=arm,
            runtimes=test_runtimes,
        ),
    }


def _merge_cost_arm_outputs(
    outputs: Sequence[Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Reassemble independently scored costs in canonical per-sample order."""

    if not outputs:
        raise ValueError("cost-split arm produced no outputs")
    phases = set(outputs[0])
    if any(set(output) != phases for output in outputs):
        raise AssertionError("cost-split arm phases differ")

    merged: dict[str, Any] = {}
    for phase in outputs[0]:
        results = [output[phase] for output in outputs]
        stable = {key: value for key, value in results[0].items() if key != "decisions"}
        if any(
            {key: value for key, value in result.items() if key != "decisions"}
            != stable
            for result in results[1:]
        ):
            raise AssertionError("cost-split arm state differs across score costs")

        orders = [
            [str(row["sample_id"]) for row in result["decisions"]] for result in results
        ]
        if any(order != orders[0] for order in orders[1:]):
            raise AssertionError("cost-split decision sample order differs")
        if len(orders[0]) != len(set(orders[0])):
            raise AssertionError("cost-split output repeats a sample")

        runtime_by_sample = {
            sample_id: math.fsum(
                float(result["decisions"][index]["score_panel_runtime_ms"])
                for result in results
            )
            for index, sample_id in enumerate(orders[0])
        }
        decisions: list[dict[str, Any]] = []
        for index, sample_id in enumerate(orders[0]):
            for result in results:
                row = dict(result["decisions"][index])
                row["score_panel_runtime_ms"] = runtime_by_sample[sample_id]
                decisions.append(row)
        merged[phase] = {**stable, "decisions": decisions}
    return merged


def _finalize_fold(
    context: Mapping[str, Any],
    benchmarks: Mapping[str, Sequence[Any]],
    arms: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(arms) != set(_PANEL_NAMES):
        raise ValueError("fold arm panel is incomplete")
    outer_fold = int(context["outer_fold"])
    config = context["config"]
    score_costs = [float(value) for value in config["score_kv_costs_ms"]]
    references = {arm: arms[arm]["test"] for arm in _PANEL_NAMES[:3]}
    warmup_panels = {
        "frozen_100": arms["frozen_100"]["warmup"],
        **{arm: arms[arm]["warmup"] for arm in _DYNAMIC_ARMS},
    }
    dynamic = {arm: arms[arm]["test"] for arm in _DYNAMIC_ARMS}
    warmup_decisions = _merge_arm_decisions(
        warmup_panels,
        order_run="primary-warmup",
        base_panel="frozen_100",
    )
    decisions = _merge_arm_decisions(
        {**references, **dynamic},
        order_run="primary",
        base_panel="frozen_100",
    )

    warmup_task_final = _final_state(warmup_panels["task"])
    base_profile = context["base_profile"]
    if warmup_task_final[:2] != (
        len(base_profile),
        len({str(row["task_id"]) for row in base_profile}),
    ):
        raise AssertionError("warmup final state differs from batch profile")
    test_task_final = _final_state(dynamic["task"])
    expected_test_rows = len(base_profile) + len(context["test_rows"])
    expected_test_tasks = len(
        {str(row["task_id"]) for row in [*base_profile, *context["test_rows"]]}
    )
    if test_task_final[:2] != (expected_test_rows, expected_test_tasks):
        raise AssertionError("test final state differs from completed-task profile")

    warmup_benchmarks = benchmarks["warmup"]
    test_benchmarks = benchmarks["test"]
    records_path = context["records_path"]
    writer = _ZstdJsonlWriter(records_path)
    try:
        writer.write_many(
            [
                {
                    "record_type": "fold_metadata",
                    "schema_version": PREQUENTIAL_ARTIFACT_SCHEMA_VERSION,
                    "outer_fold": outer_fold,
                    "fold_count": _OUTER_FOLDS,
                    "status": "development_only_exploratory",
                    "certificate": False,
                    "config": config,
                    "selected_guard_normalized": context["selected_guard"],
                    "warmup_task_count": len(context["warmup_tasks"]),
                    "test_task_count": len(context["test_tasks"]),
                    "warmup_tasks": sorted(context["warmup_tasks"]),
                    "test_tasks": sorted(context["test_tasks"]),
                    "primary_warmup_order": context["primary_warmup_order"],
                    "primary_test_order": context["primary_test_order"],
                }
            ]
        )
        writer.write_many(
            {
                "record_type": "test_reference_decision",
                "outer_fold": outer_fold,
                "panel": panel,
                **row,
            }
            for panel, result in references.items()
            for row in result["decisions"]
        )
        writer.write_many(
            {
                "record_type": "warmup_update_benchmark",
                "outer_fold": outer_fold,
                **benchmark.to_json_obj(),
            }
            for benchmark in warmup_benchmarks
        )
        writer.write_many(
            {
                "record_type": "warmup_decision",
                "outer_fold": outer_fold,
                **row,
            }
            for row in warmup_decisions
        )
        for arm in _DYNAMIC_ARMS:
            writer.write_many(
                {
                    "record_type": "warmup_profile_update",
                    "outer_fold": outer_fold,
                    **row,
                }
                for row in warmup_panels[arm]["updates"]
            )
        writer.write_many(
            {
                "record_type": "test_update_benchmark",
                "outer_fold": outer_fold,
                "order_run": "primary",
                **benchmark.to_json_obj(),
            }
            for benchmark in test_benchmarks
        )
        writer.write_many(
            {
                "record_type": "test_decision",
                "outer_fold": outer_fold,
                **row,
            }
            for row in decisions
        )
        for arm in _DYNAMIC_ARMS:
            writer.write_many(
                {
                    "record_type": "test_profile_update",
                    "outer_fold": outer_fold,
                    "order_run": "primary",
                    **row,
                }
                for row in dynamic[arm]["updates"]
            )
        writer.finish()
    except BaseException:
        writer.abort()
        raise

    run_summary = {
        "order_run": "primary",
        "test_task_order": context["primary_test_order"],
        "point_estimates": _paired_point_estimates(decisions, costs=score_costs),
        "trigger_changes": _trigger_change_summary(decisions, costs=score_costs),
        "timing": _timing_summary(
            [benchmark.to_json_obj() for benchmark in test_benchmarks],
            decisions,
            panels=_PANEL_NAMES,
        ),
    }
    return {
        "outer_fold": outer_fold,
        "warmup_task_count": len(context["warmup_tasks"]),
        "test_task_count": len(context["test_tasks"]),
        "test_tasks": sorted(context["test_tasks"]),
        "warmup_summary": {
            "primary_order": context["primary_warmup_order"],
            "point_estimates": _paired_point_estimates(
                warmup_decisions,
                costs=score_costs,
                pairs={
                    "task_vs_frozen_100": (
                        "frozen_100_trigger_ms",
                        "task_trigger_ms",
                    ),
                },
            ),
        },
        "runs": [run_summary],
        "records_sidecar_record_count": writer.count,
    }


def _final_state(result: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(result["final_profile_row_count"]),
        int(result["final_profile_task_count"]),
        int(result["final_model_version"]),
    )


def _fold_task_sets(
    task_ids: Sequence[str], *, outer_fold: int
) -> tuple[set[str], set[str]]:
    if not 1 <= outer_fold <= _OUTER_FOLDS:
        raise ValueError(f"outer_fold must be in [1, {_OUTER_FOLDS}]")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("fold task_ids contain duplicates")
    test = {
        task_id
        for index, task_id in enumerate(task_ids)
        if index % _OUTER_FOLDS == outer_fold - 1
    }
    return test, set(task_ids) - test


def _aggregate_fold_run(
    folds: Sequence[Mapping[str, Any]],
    *,
    run_name: str,
    costs: Sequence[float],
) -> dict[str, Any]:
    runs = [
        next(run for run in fold["runs"] if run["order_run"] == run_name)
        for fold in folds
    ]
    comparisons: dict[str, Any] = {}
    for comparison in _PAIR_FIELDS:
        by_cost: dict[str, Any] = {}
        for cost in costs:
            points = [run["point_estimates"][comparison][str(cost)] for run in runs]
            task_count = sum(int(point["task_count"]) for point in points)
            paired_delta = sum(float(point["paired_delta_ms"]) for point in points)
            by_cost[str(cost)] = {
                "kv_cost_ms": cost,
                "call_count": sum(int(point["call_count"]) for point in points),
                "task_count": task_count,
                "paired_delta_ms": paired_delta,
                "mean_task_delta_ms": paired_delta / task_count,
                "positive_task_count": sum(
                    int(point["positive_task_count"]) for point in points
                ),
                "negative_task_count": sum(
                    int(point["negative_task_count"]) for point in points
                ),
                "zero_task_count": sum(
                    int(point["zero_task_count"]) for point in points
                ),
            }
        comparisons[comparison] = by_cost
    return {
        "order_run": run_name,
        "point_estimates": comparisons,
        "fold_runs": runs,
    }


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment config must be a mapping")
    required = {
        "schema_version",
        "protocol_date",
        "status",
        "initialization_manifest",
        "development_manifest",
        "outer_folds",
        "arms",
        "fit_kv_costs_ms",
        "score_kv_costs_ms",
        "guard_ms",
        "restore_cost_fraction",
        "min_tool_history",
        "min_profile_tasks",
        "command_field",
        "max_prefix_depth",
        "skip_leading_cd",
        "task_order_seed",
        "outputs",
    }
    if set(payload) != required:
        raise ValueError(
            "experiment config fields differ from the frozen schema: "
            f"missing={sorted(required - set(payload))}, "
            f"unexpected={sorted(set(payload) - required)}"
        )
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != PREQUENTIAL_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported experiment config schema")
    if payload["protocol_date"] != "2026-07-21":
        raise ValueError("protocol_date must be 2026-07-21")
    if payload["status"] != "development_only_exploratory":
        raise ValueError("experiment must remain development_only_exploratory")
    if payload["initialization_manifest"] != _INITIAL_MANIFEST:
        raise ValueError("initialization_manifest differs from the frozen source")
    if payload["development_manifest"] != _DEVELOPMENT_MANIFEST:
        raise ValueError("development_manifest differs from the frozen source")
    if payload["outer_folds"] != _OUTER_FOLDS:
        raise ValueError(f"outer_folds must be exactly {_OUTER_FOLDS}")
    if payload["arms"] != list(_PANEL_NAMES):
        raise ValueError(f"arms must be exactly {_PANEL_NAMES}")
    if payload["fit_kv_costs_ms"] != [
        500,
        1000,
        1500,
        2000,
        2500,
        3000,
        3500,
        4000,
        4500,
        5000,
    ]:
        raise ValueError("fit_kv_costs_ms must match the frozen ten-cell panel")
    if payload["score_kv_costs_ms"] != [3500, 5000]:
        raise ValueError("score_kv_costs_ms must be exactly [3500, 5000]")
    if type(payload["guard_ms"]) is not int or payload["guard_ms"] != 0:
        raise ValueError("guard_ms must be the frozen integer 0")
    restore_fraction = payload["restore_cost_fraction"]
    if (
        isinstance(restore_fraction, bool)
        or not isinstance(restore_fraction, (int, float))
        or float(restore_fraction) != 0.94
    ):
        raise ValueError("restore_cost_fraction must be the frozen value 0.94")
    for field, expected in (
        ("min_tool_history", 1),
        ("min_profile_tasks", 1),
        ("max_prefix_depth", 4),
    ):
        if type(payload[field]) is not int or payload[field] != expected:
            raise ValueError(f"{field} must be the frozen integer {expected}")
    if payload["command_field"] != "command":
        raise ValueError("the frozen prefix profile requires command_field=command")
    if payload["skip_leading_cd"] is not False:
        raise ValueError("the frozen prefix profile requires skip_leading_cd=false")
    if type(payload["task_order_seed"]) is not int or payload["task_order_seed"] != 0:
        raise ValueError("task_order_seed must be the frozen integer 0")
    if payload["outputs"] != _OUTPUTS:
        raise ValueError("outputs must remain under the development namespace")
    return payload


def _load_corpus(
    manifest_path: Path,
) -> tuple[
    dict[str, Any],
    list[str],
    dict[str, list[ToolLatencySample]],
]:
    manifest = read_tool_latency_corpus_manifest(manifest_path, repo_root=_REPO_ROOT)
    task_ids = list(manifest["task_ids"])
    if len(task_ids) != manifest["expected_task_count"]:
        raise ValueError("manifest task count differs from its pinned task list")
    trace_paths = discover_trace_files([Path(manifest["trace_root"])])
    if len(trace_paths) != manifest["expected_task_count"]:
        raise ValueError("manifest trace count differs from expected_task_count")
    task_by_trace = require_explicit_trace_task_ids(trace_paths)
    samples_by_task = _extract_samples_by_task(trace_paths, task_by_trace)
    if set(samples_by_task) != set(task_ids):
        raise ValueError(
            "extracted tasks differ from pinned task list: "
            f"missing={sorted(set(task_ids) - set(samples_by_task))}, "
            f"unexpected={sorted(set(samples_by_task) - set(task_ids))}"
        )
    return manifest, task_ids, samples_by_task


def _extract_samples_by_task(
    trace_paths: Sequence[Path], task_by_trace: Mapping[str, str]
) -> dict[str, list[ToolLatencySample]]:
    samples_by_task: dict[str, list[ToolLatencySample]] = defaultdict(list)
    for sample in extract_many_tool_latency_samples(trace_paths):
        resolved_source = str(Path(sample.source_trace).resolve())
        expected_task = task_by_trace.get(resolved_source)
        if expected_task is None or sample.task_id != expected_task:
            raise ValueError(
                "sample task differs from explicit trace metadata: "
                f"{sample.source_trace}: {sample.task_id!r} != {expected_task!r}"
            )
        samples_by_task[sample.task_id].append(sample)
    return dict(samples_by_task)


def _sample_rows(
    samples_by_task: Mapping[str, Sequence[ToolLatencySample]], task_ids: set[str]
) -> list[dict[str, Any]]:
    return [
        sample.to_json_obj()
        for task_id in sorted(task_ids)
        for sample in samples_by_task[task_id]
    ]


def _assert_frozen_identity(
    static_rows: Sequence[Mapping[str, Any]],
    dynamic_rows: Sequence[Mapping[str, Any]],
) -> None:
    dynamic = {
        (str(row["sample_id"]), float(row["kv_cost_ms"])): float(row["trigger_ms"])
        for row in dynamic_rows
    }
    static = {
        key: float(row["offline_gated_robust_trigger_ms"])
        for row in static_rows
        if (key := (str(row["sample_id"]), float(row["kv_cost_ms"]))) in dynamic
    }
    if set(static) != set(dynamic):
        raise AssertionError(
            "frozen prequential scorer produced a different decision panel"
        )
    mismatches = [
        key
        for key in static
        if not math.isclose(static[key], dynamic[key], rel_tol=0.0, abs_tol=1e-9)
    ]
    if mismatches:
        raise AssertionError(
            f"frozen prequential scorer changed {len(mismatches)} certified triggers"
        )


def _merge_arm_decisions(
    by_arm: Mapping[str, Mapping[str, Any]],
    *,
    order_run: str,
    base_panel: str,
) -> list[dict[str, Any]]:
    maps = {
        arm: {
            (str(row["sample_id"]), float(row["kv_cost_ms"])): row
            for row in result["decisions"]
        }
        for arm, result in by_arm.items()
    }
    keys = set(maps[base_panel])
    if any(set(panel) != keys for panel in maps.values()):
        raise AssertionError("prequential panels produced different decision keys")
    output: list[dict[str, Any]] = []
    for key in sorted(keys):
        base = maps[base_panel][key]
        positioned = maps["task"][key] if "task" in maps else base
        merged = {
            "sample_id": base["sample_id"],
            "task_id": base["task_id"],
            "source_trace": base["source_trace"],
            "order_run": order_run,
            "task_position": positioned["task_position"],
            "tool_name": base["tool_name"],
            "tool_ts_start": base["tool_ts_start"],
            "tool_ts_end": base["tool_ts_end"],
            "latency_ms": base["latency_ms"],
            "kv_cost_ms": base["kv_cost_ms"],
            "threshold_ms": base["threshold_ms"],
            "restore_cost_ms": base["restore_cost_ms"],
        }
        for arm, panel in maps.items():
            row = panel[key]
            for field in _DYNAMIC_FIELDS:
                merged[f"{arm}_{field}"] = row[field]
        output.append(merged)
    return output


def _paired_point_estimates(
    decisions: Sequence[Mapping[str, Any]],
    *,
    costs: Sequence[float],
    pairs: Mapping[str, tuple[str, str]] = _PAIR_FIELDS,
) -> dict[str, Any]:
    """Exact descriptive utility deltas for one realized adaptive order."""

    output: dict[str, Any] = {}
    for name, (baseline_field, treatment_field) in pairs.items():
        by_cost: dict[str, Any] = {}
        for cost in costs:
            task_deltas: dict[str, float] = defaultdict(float)
            rows = [row for row in decisions if float(row["kv_cost_ms"]) == cost]
            for row in rows:
                kwargs = {
                    "threshold_ms": float(row["threshold_ms"]),
                    "kv_cost_ms": cost,
                    "restore_cost_ms": float(row["restore_cost_ms"]),
                }
                baseline = trigger_policy_utility_ms(
                    float(row["latency_ms"]),
                    float(row[baseline_field]),
                    **kwargs,
                )
                treatment = trigger_policy_utility_ms(
                    float(row["latency_ms"]),
                    float(row[treatment_field]),
                    **kwargs,
                )
                task_deltas[str(row["task_id"])] += treatment - baseline
            values = np.asarray(list(task_deltas.values()), dtype=float)
            by_cost[str(cost)] = {
                "kv_cost_ms": cost,
                "call_count": len(rows),
                "task_count": len(task_deltas),
                "paired_delta_ms": float(np.sum(values)),
                "mean_task_delta_ms": float(np.mean(values)),
                "median_task_delta_ms": float(np.median(values)),
                "positive_task_count": int(np.count_nonzero(values > 1e-9)),
                "negative_task_count": int(np.count_nonzero(values < -1e-9)),
                "zero_task_count": int(np.count_nonzero(np.abs(values) <= 1e-9)),
            }
        output[name] = by_cost
    return output


def _trigger_change_summary(
    decisions: Sequence[Mapping[str, Any]], *, costs: Sequence[float]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for arm in _DYNAMIC_ARMS:
        by_cost: dict[str, Any] = {}
        for cost in costs:
            rows = [row for row in decisions if float(row["kv_cost_ms"]) == cost]
            changed = sum(
                not math.isclose(
                    float(row[f"{arm}_trigger_ms"]),
                    float(row["frozen_100_trigger_ms"]),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                for row in rows
            )
            by_cost[str(cost)] = {
                "call_count": len(rows),
                "changed_trigger_count": changed,
                "changed_trigger_fraction": changed / len(rows) if rows else None,
            }
        output[f"{arm}_vs_frozen_100"] = by_cost
    return output


def _timing_summary(
    benchmarks: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    *,
    panels: Sequence[str],
) -> dict[str, Any]:
    update_ms = np.asarray([float(row["runtime_ms"]) for row in benchmarks])
    score_by_arm: dict[str, list[float]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in decisions:
        sample_id = str(row["sample_id"])
        for arm in panels:
            key = (arm, sample_id)
            if key in seen:
                continue
            seen.add(key)
            score_by_arm[arm].append(float(row[f"{arm}_score_panel_runtime_ms"]))
    return {
        "update_runtime_ms": _distribution(update_ms),
        "score_panel_runtime_ms": {
            arm: _distribution(np.asarray(values, dtype=float))
            for arm, values in score_by_arm.items()
        },
        "task_update_publication": "all observations publish after task resolution",
    }


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("timing distribution must be finite and non-empty")
    quantiles = np.quantile(values, [0.5, 0.9, 0.95, 0.99], method="linear")
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "p50": float(quantiles[0]),
        "p90": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "p99": float(quantiles[3]),
        "max": float(np.max(values)),
    }


def _render_markdown(result: Mapping[str, Any]) -> str:
    protocol = result["protocol"]
    provenance = result["provenance"]
    primary = result["primary"]["run"]
    lines = [
        f"# Five-fold completed-task profile-update screen ({protocol['date']})",
        "",
        "**DEVELOPMENT-ONLY / EXPLORATORY. This is not an online activation gate, "
        "deployment certificate, or unopened-stream result.**",
        "",
        f"`{provenance['initialization_collection_id']}` initializes one profile "
        f"and fixed gate ({provenance['initialization_task_count']} tasks, "
        f"{provenance['initialization_call_count']} calls). "
        f"`{provenance['development_collection_id']}` supplies the five-fold "
        f"prequential stream ({provenance['development_task_count']} tasks, "
        f"{provenance['development_call_count']} calls). The task sets are disjoint.",
        "",
        "Each Fresh-277 task is scored exactly once in its held-out fold. Its lane "
        "first consumes the other four folds. `warmup snapshot` freezes at test "
        "entry; `task` publishes past observations only after each completed task. "
        "`fresh4 static` is a same-100-derived-gate matched reference.",
        "",
        "Positive utility delta favors the treatment named by the comparison.",
        "",
        "## Primary exact point estimates",
        "",
        "| Comparison | KV ms | utility delta ms | mean/task ms | + / - / 0 tasks |",
        "|---|---:|---:|---:|---:|",
    ]
    for comparison, points in primary["point_estimates"].items():
        for point in points.values():
            lines.append(
                f"| {comparison.replace('_', ' ')} | {point['kv_cost_ms']:.0f} | "
                f"{point['paired_delta_ms']:.2f} | "
                f"{point['mean_task_delta_ms']:.2f} | "
                f"{point['positive_task_count']} / {point['negative_task_count']} / "
                f"{point['zero_task_count']} |"
            )
    lines.extend(
        [
            "",
            "## Timing and publication",
            "",
            "All observations publish after their logical task resolves.",
            "",
            "| Fold | update p50 / p95 / p99 / max ms |",
            "|---:|---:|",
        ]
    )
    for fold in result["folds"]:
        update = fold["runs"][0]["timing"]["update_runtime_ms"]
        lines.append(
            f"| {fold['outer_fold']} | {update['p50']:.4f} / "
            f"{update['p95']:.4f} / {update['p99']:.4f} / "
            f"{update['max']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Host update timing is a mechanism measurement, not a deployment-latency "
            "guarantee.",
            "",
            "No bootstrap CI or sign-flip p-value is reported: held-out-fold updates "
            "make later task decisions path-dependent.",
            "",
            "Complete calibration, fold membership, decisions, update timings, "
            "task-boundary publication markers, model versions, and source snapshots "
            "are stored in:",
            "",
        ]
    )
    for sidecar in result["records_sidecars"]:
        lines.append(
            f"- `{sidecar['file']}` — {sidecar['record_count']} records"
        )
    lines.append("")
    return "\n".join(lines)


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (_REPO_ROOT / path).resolve()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


if __name__ == "__main__":
    main()
