#!/usr/bin/env python3
"""Run the frozen offline-gated robust confirmation on one fresh trace set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable

from trace_collect.tool_gap_extractor import discover_trace_files
from trace_collect.tool_latency_confirmation import paired_task_cluster_bootstrap
from trace_collect.tool_latency_dataset import (
    ToolLatencySample,
    extract_many_tool_latency_samples,
    write_tool_latency_jsonl,
)
from trace_collect.tool_latency_offline_probe import (
    aggregate_offline_probe_cv,
    evaluate_offline_probe_clock,
)
from trace_collect.trace_data import TraceData


FROZEN_COSTS_MS = [
    500.0,
    1000.0,
    1500.0,
    2000.0,
    2500.0,
    3000.0,
    3500.0,
    4000.0,
    4500.0,
    5000.0,
]
FROZEN_CONFIG = {
    "fold_count": 5,
    "inner_folds": 4,
    "costs_ms": FROZEN_COSTS_MS,
    "guard_ms": 0.0,
    "min_tool_history": 1,
    "min_profile_tasks": 1,
    "command_field": "command",
    "max_prefix_depth": 4,
    "skip_leading_cd": False,
    "bootstrap": {
        "replicates": 50_000,
        "confidence_level": 0.95,
        "seed": 0,
        "family_method": "bonferroni_percentile",
    },
}
REQUIRED_EXCLUDED_TRACE_ROOTS = [
    "traces/swe-rebench/qwen3.7-max/20260624T162037",
    "traces/terminal-bench/zai-org-GLM-5.2/20260709T171830",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen gated-robust confirmation from a data manifest."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    shard = parser.add_mutually_exclusive_group()
    shard.add_argument(
        "--only-fold",
        type=int,
        default=None,
        help="Run ONLY this outer fold's body (write its f{N}_* files) and exit "
        "before any shared provenance or cross-fold aggregation. Lets K folds "
        "run as K concurrent processes into one output root.",
    )
    shard.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip the fold loop; write the shared provenance/all.jsonl, "
        "reconstruct all_decisions from the cv/f*_decisions.jsonl the fold "
        "processes wrote, then run the aggregation + bootstrap + hashes.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_confirmation(
        args.manifest,
        output_root=args.output_root,
        only_fold=args.only_fold,
        aggregate_only=args.aggregate_only,
    )


def run_confirmation(
    manifest_path: Path,
    *,
    output_root: Path,
    only_fold: int | None = None,
    aggregate_only: bool = False,
) -> None:
    """Validate fresh inputs, run fixed OOF policies, and bootstrap by task.

    Orchestration modes leave the numerics untouched. The bare call runs every
    fold then aggregates. ``only_fold=N`` runs just fold N's body, writes its
    ``f{N}_*`` split/data/cv files, and returns before any shared provenance or
    cross-fold aggregation, so K folds can run as K concurrent processes into
    one output root. ``aggregate_only`` skips the fold loop, reconstructs
    ``all_decisions`` from the ``cv/f*_decisions.jsonl`` those processes wrote
    (in fold order, re-stamping ``outer_fold`` exactly as the loop does), then
    runs the shared provenance + aggregation + bootstrap. Only the full run and
    the aggregate pass write the shared provenance (which hashes the ~GB
    development traces) and ``all.jsonl`` / ``all_tasks.txt``; fold processes
    never touch them, so concurrent folds neither race nor redundantly hash.
    """

    if only_fold is not None and aggregate_only:
        raise ValueError("--only-fold and --aggregate-only are mutually exclusive")
    sharded = only_fold is not None or aggregate_only

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = manifest_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists() and not sharded:
        raise FileExistsError(f"refusing to mix stale output: {output_root}")
    manifest = _read_manifest(manifest_path, repo_root=repo_root)
    trace_root = Path(manifest["trace_root"])
    task_ids_path = Path(manifest["task_ids_file"])
    if _paths_overlap(output_root, trace_root):
        raise ValueError("output_root must not overlap trace_root")
    fold_count = manifest["fold_count"]
    if only_fold is not None and not 1 <= only_fold <= fold_count:
        raise ValueError(f"--only-fold must be in [1, {fold_count}], got {only_fold}")

    trace_paths = discover_trace_files([trace_root])
    if not trace_paths:
        raise ValueError(f"no trace.jsonl files found under {trace_root}")
    development_trace_paths = _required_development_trace_paths(repo_root)
    output_root.mkdir(parents=True, exist_ok=sharded)
    # Shared provenance + content-overlap gate hash the ~GB development traces;
    # write them once (full run or aggregate pass), never per fold process.
    write_shared = only_fold is None
    if write_shared:
        _write_provenance(
            output_root,
            repo_root=repo_root,
            manifest_path=manifest_path,
            task_ids_path=task_ids_path,
            trace_paths=trace_paths,
            development_trace_paths=development_trace_paths,
        )
        _reject_trace_content_overlap(trace_paths, development_trace_paths)
    task_ids = _read_task_ids(task_ids_path)
    if len(task_ids) != manifest["expected_task_count"]:
        raise ValueError(
            "manifest expected_task_count differs from frozen task_ids_file: "
            f"{manifest['expected_task_count']} != {len(task_ids)}"
        )
    task_by_trace = _require_explicit_trace_task_ids(trace_paths)
    samples = extract_many_tool_latency_samples(trace_paths)
    samples_by_task: dict[str, list[ToolLatencySample]] = {}
    for sample in samples:
        expected_task = task_by_trace.get(str(Path(sample.source_trace).resolve()))
        if expected_task is None or sample.task_id != expected_task:
            raise ValueError(
                "extracted sample task_id differs from explicit trace metadata: "
                f"{sample.source_trace}: {sample.task_id!r} != {expected_task!r}"
            )
        samples_by_task.setdefault(sample.task_id, []).append(sample)
    extracted_tasks = set(samples_by_task)
    declared_tasks = set(task_ids)
    if extracted_tasks != declared_tasks:
        raise ValueError(
            "extracted logical tasks differ from frozen task_ids_file: "
            f"missing={sorted(declared_tasks - extracted_tasks)}, "
            f"unexpected={sorted(extracted_tasks - declared_tasks)}"
        )

    split_root = output_root / "splits"
    data_root = output_root / "data"
    cv_root = output_root / "cv"
    split_root.mkdir(exist_ok=sharded)
    data_root.mkdir(exist_ok=sharded)
    cv_root.mkdir(exist_ok=sharded)
    if write_shared:
        (split_root / "all_tasks.txt").write_text(
            "".join(f"{task_id}\n" for task_id in task_ids),
            encoding="utf-8",
        )
        write_tool_latency_jsonl(samples, data_root / "all.jsonl")

    all_decisions: list[dict[str, Any]] = []
    if aggregate_only:
        for fold in range(1, fold_count + 1):
            decisions_path = cv_root / f"f{fold}_decisions.jsonl"
            for line in decisions_path.read_text(encoding="utf-8").splitlines():
                all_decisions.append(
                    {**json.loads(line), "outer_fold": f"f{fold}"}
                )
    else:
        folds = [only_fold] if only_fold is not None else list(range(1, fold_count + 1))
        for fold in folds:
            decisions = _run_single_fold(
                fold,
                fold_count=fold_count,
                task_ids=task_ids,
                declared_tasks=declared_tasks,
                samples_by_task=samples_by_task,
                manifest=manifest,
                split_root=split_root,
                data_root=data_root,
                cv_root=cv_root,
            )
            all_decisions.extend(
                {**decision, "outer_fold": f"f{fold}"} for decision in decisions
            )
        if only_fold is not None:
            print(f"Wrote fold {only_fold} of {fold_count} -> {output_root}")
            return

    pooled = aggregate_offline_probe_cv(cv_root, expected_fold_count=fold_count)
    _write_json(cv_root / "pooled_results.json", pooled)
    bootstrap = manifest["bootstrap"]
    uncertainty = paired_task_cluster_bootstrap(
        all_decisions,
        costs_ms=manifest["costs_ms"],
        replicates=bootstrap["replicates"],
        confidence_level=bootstrap["confidence_level"],
        seed=bootstrap["seed"],
    )
    uncertainty["collection"] = {
        "collection_id": manifest["collection_id"],
        "trace_root": str(trace_root),
        "task_ids_file": str(task_ids_path),
        "expected_task_count": manifest["expected_task_count"],
    }
    uncertainty["fold_robust_calibrations"] = pooled["fold_robust_calibrations"]
    _write_json(output_root / "paired_task_cluster_uncertainty.json", uncertainty)
    _verify_hash_inventory(output_root / "provenance/input_hashes.sha256")
    _write_hash_inventory(output_root, output_root / "result_hashes.sha256")
    print(
        f"Confirmed {len(task_ids)} tasks from {manifest['collection_id']} -> "
        f"{output_root}"
    )


def _run_single_fold(
    fold: int,
    *,
    fold_count: int,
    task_ids: list[str],
    declared_tasks: set[str],
    samples_by_task: dict[str, list[ToolLatencySample]],
    manifest: dict[str, Any],
    split_root: Path,
    data_root: Path,
    cv_root: Path,
) -> list[dict[str, Any]]:
    """Run one outer fold's body and write its ``f{fold}_*`` files.

    Identical to the sequential loop body; factored out so both the full run
    and a single ``--only-fold`` process execute the exact same fold logic.
    """

    eval_tasks = {
        task_id
        for index, task_id in enumerate(task_ids)
        if index % fold_count == fold - 1
    }
    profile_tasks = declared_tasks - eval_tasks
    _write_task_set(split_root / f"f{fold}_eval.txt", eval_tasks)
    _write_task_set(split_root / f"f{fold}_profile.txt", profile_tasks)
    eval_samples = _samples_for_tasks(samples_by_task, eval_tasks)
    profile_samples = _samples_for_tasks(samples_by_task, profile_tasks)
    write_tool_latency_jsonl(eval_samples, data_root / f"f{fold}_eval.jsonl")
    write_tool_latency_jsonl(profile_samples, data_root / f"f{fold}_profile.jsonl")
    result = evaluate_offline_probe_clock(
        [sample.to_json_obj() for sample in eval_samples],
        profile_rows=[sample.to_json_obj() for sample in profile_samples],
        kv_costs_ms=manifest["costs_ms"],
        guard_ms=manifest["guard_ms"],
        inner_folds=manifest["inner_folds"],
        min_tool_history=manifest["min_tool_history"],
        min_profile_tasks=manifest["min_profile_tasks"],
        command_field=manifest["command_field"],
        max_prefix_depth=manifest["max_prefix_depth"],
        skip_leading_cd=manifest["skip_leading_cd"],
    )
    decisions = result.pop("decisions")
    _write_json(cv_root / f"f{fold}_summary.json", result)
    _write_jsonl(cv_root / f"f{fold}_decisions.jsonl", decisions)
    return decisions


def _read_manifest(path: Path, *, repo_root: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("confirmation manifest must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("confirmation manifest schema_version must be 1")
    collection_id = payload.get("collection_id")
    if not isinstance(collection_id, str) or not collection_id.strip():
        raise ValueError("confirmation manifest collection_id must be non-empty")
    expected_task_count = payload.get("expected_task_count")
    if (
        not isinstance(expected_task_count, int)
        or isinstance(expected_task_count, bool)
        or expected_task_count < FROZEN_CONFIG["fold_count"]
    ):
        raise ValueError("expected_task_count must be an integer >= fold_count")
    attestation = payload.get("freshness_attestation")
    required_attestations = {
        "not_used_for_method_development",
        "not_smoke_or_synthetic",
        "complete_fixed_task_set",
    }
    if not isinstance(attestation, dict) or any(
        attestation.get(field) is not True for field in required_attestations
    ):
        raise ValueError("all freshness_attestation fields must be true")

    trace_root = _resolve_manifest_path(payload, "trace_root", repo_root=repo_root)
    task_ids_file = _resolve_manifest_path(
        payload,
        "task_ids_file",
        repo_root=repo_root,
    )
    if not trace_root.is_dir():
        raise ValueError(f"trace_root is not a directory: {trace_root}")
    if not task_ids_file.is_file():
        raise ValueError(f"task_ids_file is not a file: {task_ids_file}")
    excluded = payload.get("excluded_trace_roots")
    if not isinstance(excluded, list) or not excluded:
        raise ValueError("excluded_trace_roots must be a non-empty list")
    excluded_paths = []
    for value in excluded:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("excluded_trace_roots entries must be non-empty strings")
        path_value = Path(value).expanduser()
        excluded_paths.append(
            (repo_root / path_value).resolve()
            if not path_value.is_absolute()
            else path_value.resolve()
        )
    required_excluded_paths = {
        (repo_root / relative_path).resolve()
        for relative_path in REQUIRED_EXCLUDED_TRACE_ROOTS
    }
    if not required_excluded_paths.issubset(set(excluded_paths)):
        missing = sorted(
            str(path) for path in required_excluded_paths - set(excluded_paths)
        )
        raise ValueError(
            f"excluded_trace_roots omits required development sources: {missing}"
        )
    for excluded_path in excluded_paths:
        if _paths_overlap(trace_root, excluded_path):
            raise ValueError(f"trace_root overlaps excluded source: {excluded_path}")

    for key, frozen_value in FROZEN_CONFIG.items():
        manifest_value = payload.get(key)
        _validate_frozen_field_type(key, manifest_value)
        if key == "costs_ms" and isinstance(manifest_value, list):
            manifest_value = [float(value) for value in manifest_value]
        if manifest_value != frozen_value:
            raise ValueError(
                f"manifest field {key!r} differs from frozen protocol: "
                f"{manifest_value!r} != {frozen_value!r}"
            )
    return {
        **payload,
        "trace_root": str(trace_root),
        "task_ids_file": str(task_ids_file),
        "costs_ms": FROZEN_COSTS_MS,
    }


def _validate_frozen_field_type(field: str, value: Any) -> None:
    integer_fields = {
        "fold_count",
        "inner_folds",
        "min_tool_history",
        "min_profile_tasks",
        "max_prefix_depth",
    }
    if field in integer_fields:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"manifest field {field!r} must be an integer")
        return
    if field == "costs_ms":
        if not isinstance(value, list) or any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            for item in value
        ):
            raise ValueError("manifest field 'costs_ms' must contain finite numbers")
        return
    if field == "guard_ms":
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError("manifest field 'guard_ms' must be a finite number")
        return
    if field == "skip_leading_cd":
        if not isinstance(value, bool):
            raise ValueError("manifest field 'skip_leading_cd' must be a bool")
        return
    if field == "command_field":
        if not isinstance(value, str):
            raise ValueError("manifest field 'command_field' must be a string")
        return
    if field == "bootstrap":
        if not isinstance(value, dict):
            raise ValueError("manifest field 'bootstrap' must be an object")
        for integer_field in ("replicates", "seed"):
            item = value.get(integer_field)
            if not isinstance(item, int) or isinstance(item, bool):
                raise ValueError(f"bootstrap {integer_field} must be an integer")
        confidence = value.get("confidence_level")
        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not math.isfinite(float(confidence))
        ):
            raise ValueError("bootstrap confidence_level must be a finite number")
        if not isinstance(value.get("family_method"), str):
            raise ValueError("bootstrap family_method must be a string")
        return
    raise AssertionError(f"unhandled frozen manifest field: {field}")


def _resolve_manifest_path(
    payload: dict[str, Any],
    field: str,
    *,
    repo_root: Path,
) -> Path:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest field {field!r} must be a non-empty path")
    path = Path(value).expanduser()
    return (repo_root / path).resolve() if not path.is_absolute() else path.resolve()


def _read_task_ids(path: Path) -> list[str]:
    task_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if not task_ids or any(not task_id for task_id in task_ids):
        raise ValueError("task_ids_file must contain non-empty task IDs")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task_ids_file contains duplicate task IDs")
    return sorted(task_ids)


def _require_explicit_trace_task_ids(trace_paths: list[Path]) -> dict[str, str]:
    task_by_trace: dict[str, str] = {}
    for trace_path in trace_paths:
        trace = TraceData.load(trace_path)
        task_id = str(trace.metadata.get("instance_id") or "").strip()
        if not task_id:
            raise ValueError(f"trace lacks explicit metadata instance_id: {trace_path}")
        task_by_trace[str(trace_path.resolve())] = task_id
    return task_by_trace


def _required_development_trace_paths(repo_root: Path) -> list[Path]:
    paths: list[Path] = []
    for relative_root in REQUIRED_EXCLUDED_TRACE_ROOTS:
        paths.extend(discover_trace_files([repo_root / relative_root]))
    return sorted(set(paths))


def _reject_trace_content_overlap(
    trace_paths: list[Path],
    development_trace_paths: list[Path],
) -> None:
    development_by_hash = {_sha256(path): path for path in development_trace_paths}
    for trace_path in trace_paths:
        matching_source = development_by_hash.get(_sha256(trace_path))
        if matching_source is not None:
            raise ValueError(
                "fresh trace duplicates a development trace by content: "
                f"{trace_path} == {matching_source}"
            )


def _samples_for_tasks(
    samples_by_task: dict[str, list[ToolLatencySample]],
    task_ids: set[str],
) -> list[ToolLatencySample]:
    return [
        sample for task_id in sorted(task_ids) for sample in samples_by_task[task_id]
    ]


def _write_task_set(path: Path, task_ids: set[str]) -> None:
    path.write_text(
        "".join(f"{task_id}\n" for task_id in sorted(task_ids)),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _write_provenance(
    output_root: Path,
    *,
    repo_root: Path,
    manifest_path: Path,
    task_ids_path: Path,
    trace_paths: list[Path],
    development_trace_paths: list[Path],
) -> None:
    provenance = output_root / "provenance"
    provenance.mkdir()
    shutil.copy2(manifest_path, provenance / "manifest.json")
    shutil.copy2(task_ids_path, provenance / "task_ids.txt")
    protocol = (
        repo_root
        / "analysis/tool-time-offline-gated-robust-confirmation-20260712/protocol.md"
    )
    shutil.copy2(protocol, provenance / "protocol.md")
    _run_to_file(["git", "rev-parse", "HEAD"], provenance / "git_head.txt", repo_root)
    _run_to_file(
        ["git", "status", "--short", "--branch"],
        provenance / "git_status.txt",
        repo_root,
    )
    _run_to_file(
        ["git", "diff", "--binary"], provenance / "working_tree.diff", repo_root
    )
    _run_to_file(
        ["git", "diff", "--cached", "--binary"],
        provenance / "index.diff",
        repo_root,
    )
    _run_to_file(
        [sys.executable, "--version"],
        provenance / "python_version.txt",
        repo_root,
        include_stderr=True,
    )
    _run_to_file(
        [sys.executable, "-m", "pip", "freeze"],
        provenance / "pip_freeze.txt",
        repo_root,
    )
    _write_json(
        provenance / "invocation.json",
        {"argv": sys.argv, "python": sys.executable},
    )

    source_paths = _source_snapshot_paths(repo_root)
    snapshot_root = provenance / "source_snapshot"
    for source_path in source_paths:
        destination = snapshot_root / source_path.relative_to(repo_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
    (provenance / "source_snapshot_files.txt").write_text(
        "".join(f"{path.relative_to(repo_root)}\n" for path in source_paths),
        encoding="utf-8",
    )
    _write_hashes(
        [
            manifest_path,
            task_ids_path,
            protocol,
            *source_paths,
            *trace_paths,
            *development_trace_paths,
        ],
        provenance / "input_hashes.sha256",
    )


def _source_snapshot_paths(repo_root: Path) -> list[Path]:
    relative_paths = [
        "scripts/run_offline_gated_robust_confirmation.py",
        "src/trace_collect/causal_history.py",
        "src/trace_collect/classification_metrics.py",
        "src/trace_collect/cli_helpers.py",
        "src/trace_collect/command_features.py",
        "src/trace_collect/latency_validation.py",
        "src/trace_collect/tool_gap_extractor.py",
        "src/trace_collect/tool_latency_confirmation.py",
        "src/trace_collect/tool_latency_dataset.py",
        "src/trace_collect/tool_latency_offline_probe.py",
        "src/trace_collect/tool_latency_profiled.py",
        "src/trace_collect/tool_latency_utility_clock.py",
        "src/trace_collect/trace_data.py",
        "tests/test_tool_latency_confirmation.py",
        "tests/test_tool_latency_offline_probe.py",
        "tests/test_tool_latency_utility_clock.py",
    ]
    return [repo_root / relative_path for relative_path in relative_paths]


def _run_to_file(
    command: list[str],
    path: Path,
    cwd: Path,
    *,
    include_stderr: bool = False,
) -> None:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT if include_stderr else subprocess.PIPE,
    )
    path.write_text(result.stdout, encoding="utf-8")


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _write_hash_inventory(root: Path, output: Path) -> None:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path != output
        and path.suffix != ".pyc"
        and "__pycache__" not in path.parts
    )
    _write_hashes(paths, output)


def _write_hashes(paths: Iterable[Path], output: Path) -> None:
    unique_paths = sorted({path.resolve() for path in paths})
    output.write_text(
        "".join(f"{_sha256(path)}  {path}\n" for path in unique_paths),
        encoding="utf-8",
    )


def _verify_hash_inventory(path: Path) -> None:
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            expected, raw_path = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid hash record") from exc
        input_path = Path(raw_path)
        if not input_path.is_file():
            raise ValueError(f"hashed input disappeared during run: {input_path}")
        if _sha256(input_path) != expected:
            raise ValueError(f"hashed input changed during run: {input_path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
