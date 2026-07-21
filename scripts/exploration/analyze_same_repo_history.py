#!/usr/bin/env python3
"""Development-only matched screen for same-repository latency history.

The treatment conditions the existing robust command-prefix prior on the task's
repository.  Every held-out task falls back to its stored outer-fold
``warmup_snapshot`` decision unless a fold-fit, deadline-relative margin guard
accepts a repository candidate.  The five primary baseline arms are never
rerun.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter_ns
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))


from scripts.exploration.analyze_prequential_profile_updates import (  # noqa: E402
    _SOURCE_HASHES_AT_IMPORT,
    _ZstdJsonlWriter,
    _cleanup_partial_outputs,
    _git_sha,
    _inventory_digest,
    _load_corpus,
    _paired_point_estimates,
    _resolve,
    _sample_rows,
    _sha256,
    _source_snapshot_records,
    _source_tree_hashes,
    _verify_declared_inventory,
    _verify_file_hash,
    _verify_runtime_inventory,
)
from trace_collect.command_features import (  # noqa: E402
    make_row_command_prefix_keys,
)
from trace_collect.tool_latency_offline_probe import select_probe_guard  # noqa: E402
from trace_collect.tool_latency_profiled import (  # noqa: E402
    LatencyPrior,
    build_latency_prior,
    latency_prior_hierarchy,
)
from trace_collect.tool_latency_utility_clock import (  # noqa: E402
    robust_prior_nodes,
    robust_utility_trigger_stats,
    trigger_policy_utility_ms,
)

_CONFIG_FIELDS = {
    "schema_version",
    "protocol_date",
    "status",
    "primary_result",
    "primary_sidecars",
    "trace_inventories",
    "arm",
    "baseline",
    "score_kv_costs_ms",
    "restore_cost_fraction",
    "minimum_distinct_prior_tasks",
    "guard_objective",
    "repository_identity",
    "fallback",
    "verdict_rule",
    "outputs",
}
_EXPECTED_SIDECAR_ROLES = {"initialization", *(f"outer_fold_{i}" for i in range(1, 6))}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/same_repo_history.yaml"),
    )
    return parser


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(config, dict) or set(config) != _CONFIG_FIELDS:
        actual = set(config) if isinstance(config, dict) else set()
        raise ValueError(
            "same-repo config fields differ from the frozen schema: "
            f"missing={sorted(_CONFIG_FIELDS - actual)}, "
            f"unexpected={sorted(actual - _CONFIG_FIELDS)}"
        )
    expected = {
        "schema_version": 1,
        "protocol_date": "2026-07-21",
        "status": "development_only_post_primary_followup",
        "arm": "same_repo",
        "baseline": "warmup_snapshot",
        "score_kv_costs_ms": [3500, 5000],
        "restore_cost_fraction": 0.94,
        "minimum_distinct_prior_tasks": 2,
        "guard_objective": "candidate_vs_deadline",
        "repository_identity": "swe_rebench_task_id_owner_repo_final_numeric_issue",
        "fallback": "exact_outer_fold_warmup_snapshot",
        "verdict_rule": (
            "drop_if_either_cost_paired_delta_is_negative_otherwise_keep_mechanism_only"
        ),
    }
    for field, value in expected.items():
        if config[field] != value:
            raise ValueError(f"{field} differs from the reviewed protocol")
    if set(config["primary_sidecars"]) != _EXPECTED_SIDECAR_ROLES:
        raise ValueError("primary_sidecars must pin initialization and all five folds")
    if set(config["trace_inventories"]) != {"initialization", "development"}:
        raise ValueError("trace_inventories must pin both corpora")
    if set(config["primary_result"]) != {"file", "sha256"}:
        raise ValueError("primary_result must contain file and sha256")
    if set(config["outputs"]) != {"json", "markdown"}:
        raise ValueError("outputs must contain json and markdown")
    return config, hashlib.sha256(raw).hexdigest()


def _repository_identity(task_id: str) -> dict[str, str | bool | None]:
    """Parse owner/repository from the inference-time SWE-ReBench task ID."""

    owner, separator, remainder = task_id.partition("__")
    repository, issue_separator, issue = remainder.rpartition("-")
    if (
        separator != "__"
        or issue_separator != "-"
        or not owner
        or not repository
        or not issue.isdecimal()
    ):
        return {
            "task_id": task_id,
            "repository_supported": False,
            "repository_raw": None,
            "repository_canonical": None,
            "repository_parse_error": "expected owner__repository-<numeric issue>",
        }
    raw = f"{owner}__{repository}"
    return {
        "task_id": task_id,
        "repository_supported": True,
        "repository_raw": raw,
        "repository_canonical": raw.casefold(),
        "repository_parse_error": None,
    }


def _read_zstd_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    process = subprocess.Popen(
        ["zstd", "-q", "-d", "-c", str(path)],
        stdout=subprocess.PIPE,
        text=True,
    )
    if process.stdout is None:
        raise RuntimeError("zstd stdout pipe was not created")
    try:
        for line_number, line in enumerate(process.stdout, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: record is not an object")
            yield row
    finally:
        process.stdout.close()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"zstd reader exited with {return_code}: {path}")


def _load_primary_inputs(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], list[dict[str, Any]]]:
    primary_path = _resolve(Path(str(config["primary_result"]["file"])))
    if _sha256(primary_path) != config["primary_result"]["sha256"]:
        raise ValueError("primary result SHA-256 differs from the frozen config")
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    sidecars = {row["role"]: row for row in primary["records_sidecars"]}
    if set(sidecars) != _EXPECTED_SIDECAR_ROLES:
        raise ValueError("primary result sidecar roles are incomplete")
    folds: dict[int, dict[str, Any]] = {}
    input_sidecars: list[dict[str, Any]] = []
    for role, declared in sorted(sidecars.items()):
        path = primary_path.with_name(str(declared["file"]))
        expected_hash = str(config["primary_sidecars"][role])
        actual_hash = _sha256(path)
        if actual_hash != expected_hash or actual_hash != declared["sha256"]:
            raise ValueError(f"primary sidecar hash mismatch: {role}")
        input_record = {
            "record_type": "input_primary_sidecar",
            "role": role,
            "file": str(path),
            "sha256": actual_hash,
            "record_count": int(declared["record_count"]),
        }
        input_sidecars.append(input_record)
        if role == "initialization":
            continue
        fold = int(role.rsplit("_", 1)[1])
        metadata: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        record_count = 0
        for row in _read_zstd_jsonl(path):
            record_count += 1
            if row.get("record_type") == "fold_metadata":
                metadata.append(row)
            elif row.get("record_type") == "test_decision":
                decisions.append(row)
        if record_count != declared["record_count"]:
            raise ValueError(f"primary sidecar record count mismatch: {role}")
        if len(metadata) != 1 or int(metadata[0]["outer_fold"]) != fold:
            raise ValueError(f"primary fold metadata mismatch: {role}")
        folds[fold] = {
            "metadata": metadata[0],
            "baseline_decisions": decisions,
            "sidecar": input_record,
        }
    if set(folds) != set(range(1, 6)):
        raise ValueError("primary fold inputs are incomplete")
    return primary, folds, input_sidecars


def _rows_by_task(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        output.setdefault(str(row["task_id"]), []).append(row)
    return output


def _task_identities(task_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    return {task_id: _repository_identity(task_id) for task_id in sorted(task_ids)}


def _rows_by_repository(
    rows_by_task: Mapping[str, Sequence[dict[str, Any]]],
    identities: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for task_id, rows in rows_by_task.items():
        repository = identities[task_id]["repository_canonical"]
        if repository is not None:
            output.setdefault(str(repository), []).extend(rows)
    return output


def _score_repository_task(
    task_rows: Sequence[dict[str, Any]],
    *,
    identity: Mapping[str, Any],
    profile_rows: Sequence[dict[str, Any]],
    costs_ms: Sequence[float],
    guard_ms: float,
    restore_cost_fraction: float,
    command_field: str,
    max_prefix_depth: int,
    skip_leading_cd: bool,
    minimum_distinct_prior_tasks: int,
) -> list[dict[str, Any]]:
    """Score one task from same-repository prior tasks only."""
    eval_task_ids = {str(row["task_id"]) for row in task_rows}
    if eval_task_ids != {str(identity["task_id"])}:
        raise ValueError("task rows differ from their repository identity")

    prior_task_ids = sorted({str(row["task_id"]) for row in profile_rows})
    if str(identity["task_id"]) in prior_task_ids:
        raise ValueError("held-out task appears in its repository prior")
    repository = identity["repository_canonical"]
    if any(
        _repository_identity(task_id)["repository_canonical"] != repository
        for task_id in prior_task_ids
    ):
        raise ValueError("repository prior contains a task from another repository")
    supported = (
        identity["repository_canonical"] is not None
        and len(prior_task_ids) >= minimum_distinct_prior_tasks
    )
    row_group_keys = make_row_command_prefix_keys(
        command_field,
        max_depth=max_prefix_depth,
        skip_leading_cd=skip_leading_cd,
    )
    prior: LatencyPrior | None = None
    prior_build_runtime_ms = 0.0
    if supported:
        start_ns = perf_counter_ns()
        prior = build_latency_prior(profile_rows, row_group_keys=row_group_keys)
        prior_build_runtime_ms = (perf_counter_ns() - start_ns) / 1_000_000.0
        if sorted(prior.task_ids) != prior_task_ids:
            raise AssertionError("repository prior task partition changed during build")
    cache: dict[
        tuple[str, tuple[str, ...], float], tuple[float, float, str, str | None]
    ] = {}
    output: list[dict[str, Any]] = []
    for row in sorted(
        task_rows,
        key=lambda item: (float(item["tool_ts_start"]), str(item["sample_id"])),
    ):
        score_start_ns = perf_counter_ns()
        scored: list[dict[str, Any]] = []
        tool_name = str(row["tool_name"])
        group_keys = row_group_keys(row)
        for cost_ms in costs_ms:
            threshold_ms = cost_ms + guard_ms
            if prior is None:
                candidate_ms = threshold_ms
                margin = 0.0
                prior_source = "none"
                prior_group_key = None
            else:
                key = (tool_name, group_keys, cost_ms)
                cached = cache.get(key)
                if cached is None:
                    hierarchy = latency_prior_hierarchy(
                        prior,
                        tool_name,
                        group_keys,
                        min_tool_history=1,
                        min_profile_tasks=1,
                    )
                    node, parent = robust_prior_nodes(hierarchy)
                    stats = robust_utility_trigger_stats(
                        node,
                        parent=parent,
                        threshold_ms=threshold_ms,
                        kv_cost_ms=cost_ms,
                        restore_cost_ms=restore_cost_fraction * cost_ms,
                    )
                    cached = (
                        stats.trigger_ms,
                        stats.normalized_advantage,
                        node.source,
                        node.group_key,
                    )
                    cache[key] = cached
                candidate_ms, margin, prior_source, prior_group_key = cached
            scored.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "task_id": str(row["task_id"]),
                    "source_trace": str(row["source_trace"]),
                    "tool_name": tool_name,
                    "tool_ts_start": float(row["tool_ts_start"]),
                    "tool_ts_end": float(row["tool_ts_end"]),
                    "latency_ms": float(row["latency_ms"]),
                    "kv_cost_ms": cost_ms,
                    "threshold_ms": threshold_ms,
                    "restore_cost_ms": restore_cost_fraction * cost_ms,
                    "repository_supported": supported,
                    "repository_raw": identity["repository_raw"],
                    "repository_canonical": identity["repository_canonical"],
                    "repository_parse_error": identity["repository_parse_error"],
                    "same_repo_prior_task_ids": prior_task_ids,
                    "same_repo_prior_task_count": len(prior_task_ids),
                    "same_repo_prior_row_count": len(profile_rows),
                    "same_repo_prior_source": prior_source,
                    "same_repo_prior_group_key": prior_group_key,
                    "same_repo_candidate_trigger_ms": candidate_ms,
                    "same_repo_margin_normalized": margin,
                    "same_repo_prior_build_runtime_ms": prior_build_runtime_ms,
                }
            )
        elapsed_ms = (perf_counter_ns() - score_start_ns) / 1_000_000.0
        for decision in scored:
            decision["same_repo_score_panel_runtime_ms"] = elapsed_ms
        output.extend(scored)
    return output


def _guard_objective_trace(
    decisions: Sequence[Mapping[str, Any]],
    *,
    restore_cost_fraction: float,
    selected_guard: float | None,
) -> list[dict[str, Any]]:
    eligible: list[tuple[float, float]] = []
    for row in decisions:
        score = float(row["same_repo_margin_normalized"])
        candidate_ms = float(row["same_repo_candidate_trigger_ms"])
        threshold_ms = float(row["threshold_ms"])
        if candidate_ms >= threshold_ms or score <= 0.0:
            continue
        cost_ms = float(row["kv_cost_ms"])
        latency_ms = float(row["latency_ms"])
        delta = (
            trigger_policy_utility_ms(
                latency_ms,
                candidate_ms,
                threshold_ms=threshold_ms,
                kv_cost_ms=cost_ms,
                restore_cost_ms=restore_cost_fraction * cost_ms,
            )
            - trigger_policy_utility_ms(
                latency_ms,
                threshold_ms,
                threshold_ms=threshold_ms,
                kv_cost_ms=cost_ms,
                restore_cost_ms=restore_cost_fraction * cost_ms,
            )
        ) / cost_ms
        eligible.append((score, delta))
    guards = sorted({0.0, *(score for score, _ in eligible)})
    return [
        {
            "guard_normalized": guard,
            "objective_normalized": math.fsum(
                delta for score, delta in eligible if score > guard
            ),
            "selected": selected_guard is not None
            and math.isclose(guard, selected_guard, rel_tol=0.0, abs_tol=0.0),
        }
        for guard in guards
    ] + [
        {
            "guard_normalized": None,
            "objective_normalized": 0.0,
            "selected": selected_guard is None,
        }
    ]


def _stable_sample_key(row: Mapping[str, Any]) -> tuple[str, str, int, str]:
    """Path-independent identity from the canonical ToolLatencySample ID."""

    parts = str(row["sample_id"]).rsplit(":", 3)
    if len(parts) != 4 or not parts[0] or not parts[3]:
        raise ValueError(f"invalid ToolLatencySample ID: {row['sample_id']!r}")
    try:
        iteration = int(parts[2])
    except ValueError as exc:
        raise ValueError(
            f"invalid ToolLatencySample iteration: {row['sample_id']!r}"
        ) from exc
    return str(row["task_id"]), parts[1], iteration, parts[3]


def _apply_guard_and_baseline(
    candidates: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    *,
    selected_guard: float | None,
    outer_fold: int,
    baseline_sidecar: Mapping[str, Any],
) -> list[dict[str, Any]]:
    baseline_by_key = {
        (*_stable_sample_key(row), float(row["kv_cost_ms"])): row
        for row in baseline_rows
    }
    if len(baseline_by_key) != len(baseline_rows):
        raise ValueError(f"outer fold {outer_fold} has duplicate baseline join keys")
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        key = (*_stable_sample_key(candidate), float(candidate["kv_cost_ms"]))
        baseline = baseline_by_key.pop(key, None)
        if baseline is None:
            raise ValueError(f"outer fold {outer_fold} lacks baseline row {key}")
        for field in (
            "task_id",
            "tool_name",
            "tool_ts_start",
            "tool_ts_end",
            "latency_ms",
            "threshold_ms",
            "restore_cost_ms",
        ):
            left = candidate[field]
            right = baseline[field]
            if isinstance(left, (int, float)):
                if not math.isclose(
                    float(left), float(right), rel_tol=0.0, abs_tol=1e-9
                ):
                    raise ValueError(f"outer fold {outer_fold} {field} mismatch: {key}")
            elif left != right:
                raise ValueError(f"outer fold {outer_fold} {field} mismatch: {key}")
        candidate_ms = float(candidate["same_repo_candidate_trigger_ms"])
        threshold_ms = float(candidate["threshold_ms"])
        margin = float(candidate["same_repo_margin_normalized"])
        if not candidate["repository_supported"]:
            override = False
            reason = (
                "repository_parse_error"
                if candidate["repository_canonical"] is None
                else "fewer_than_two_prior_repository_tasks"
            )
        elif candidate_ms >= threshold_ms:
            override = False
            reason = "no_early_candidate"
        elif selected_guard is None:
            override = False
            reason = "no_positive_fold_guard_objective"
        elif margin <= selected_guard:
            override = False
            reason = "margin_not_strictly_above_fold_guard"
        else:
            override = True
            reason = "accepted_repository_candidate"
        baseline_trigger = float(baseline["warmup_snapshot_trigger_ms"])
        final_trigger = candidate_ms if override else baseline_trigger
        output.append(
            {
                "record_type": "test_decision",
                "outer_fold": outer_fold,
                "baseline_sidecar_file": baseline_sidecar["file"],
                "baseline_sidecar_sha256": baseline_sidecar["sha256"],
                "baseline_join_key": list(key),
                **candidate,
                "local_sample_id": candidate["sample_id"],
                "local_source_trace": candidate["source_trace"],
                "sample_id": baseline["sample_id"],
                "source_trace": baseline["source_trace"],
                "warmup_snapshot_trigger_ms": baseline_trigger,
                "same_repo_selected_guard_normalized": selected_guard,
                "same_repo_override": override,
                "same_repo_fallback_reason": reason,
                "same_repo_trigger_ms": final_trigger,
            }
        )
    if baseline_by_key:
        raise ValueError(
            f"outer fold {outer_fold} left {len(baseline_by_key)} baseline rows unmatched"
        )
    return output


def _fold_summary(
    decisions: Sequence[Mapping[str, Any]], costs_ms: Sequence[float]
) -> dict[str, Any]:
    points = _paired_point_estimates(
        decisions,
        costs=costs_ms,
        pairs={
            "same_repo_vs_warmup_snapshot": (
                "warmup_snapshot_trigger_ms",
                "same_repo_trigger_ms",
            )
        },
    )
    coverage: dict[str, Any] = {}
    for cost in costs_ms:
        rows = [row for row in decisions if float(row["kv_cost_ms"]) == cost]
        changed = [
            row
            for row in rows
            if not math.isclose(
                float(row["same_repo_trigger_ms"]),
                float(row["warmup_snapshot_trigger_ms"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ]
        coverage[str(cost)] = {
            "call_count": len(rows),
            "supported_call_count": sum(
                bool(row["repository_supported"]) for row in rows
            ),
            "supported_task_count": len(
                {str(row["task_id"]) for row in rows if row["repository_supported"]}
            ),
            "override_call_count": sum(bool(row["same_repo_override"]) for row in rows),
            "changed_call_count": len(changed),
            "affected_task_count": len({str(row["task_id"]) for row in changed}),
            "fallback_reason_counts": {
                reason: sum(row["same_repo_fallback_reason"] == reason for row in rows)
                for reason in sorted(
                    {str(row["same_repo_fallback_reason"]) for row in rows}
                )
            },
        }
    return {"point_estimates": points, "coverage": coverage}


def _render_markdown(result: Mapping[str, Any]) -> str:
    point_estimates = result["summary"]["point_estimates"][
        "same_repo_vs_warmup_snapshot"
    ]
    lines = [
        "# Same-repository history screen (2026-07-21)",
        "",
        "**DEVELOPMENT-ONLY / POST-PRIMARY FOLLOW-UP. No CI, p-value, deployment, "
        "or unopened-stream claim.**",
        "",
        "The sole treatment conditions the existing robust command-prefix prior "
        "on repository identity available before task start. Each outer fold fits "
        "its deadline-relative margin guard on the other four Fresh-277 folds. "
        "Unsupported or guarded-off calls reuse the exact stored `warmup_snapshot` "
        "baseline; no baseline arm was rerun.",
        "",
        f"**Verdict: {result['verdict']}** — {result['verdict_explanation']}",
        "",
        "| KV ms | paired delta ms | mean/task ms | + / - / 0 tasks | "
        "supported tasks | changed calls / affected tasks |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for cost in result["config"]["score_kv_costs_ms"]:
        point = point_estimates[str(float(cost))]
        coverage = result["summary"]["coverage"][str(float(cost))]
        lines.append(
            f"| {cost} | {point['paired_delta_ms']:.2f} | "
            f"{point['mean_task_delta_ms']:.2f} | "
            f"{point['positive_task_count']} / {point['negative_task_count']} / "
            f"{point['zero_task_count']} | {coverage['supported_task_count']} | "
            f"{coverage['changed_call_count']} / {coverage['affected_task_count']} |"
        )
    lines.extend(
        [
            "",
            "`same_trace` was not rerun: the existing within-task B1 result was "
            "-398 s near rho=1, and the completed primary call arm changed no "
            "realized utility at either headline cost. The independent design "
            "review therefore approved only this same-repository arm.",
            "",
            "Scoring times are mechanism measurements of the current exact scorer, "
            "not deployment-latency evidence.",
            "",
            f"Records: `{result['records_sidecar']['file']}` "
            f"({result['records_sidecar']['record_count']} records, SHA-256 "
            f"`{result['records_sidecar']['sha256']}`).",
            "",
        ]
    )
    return "\n".join(lines)


def _verify_trace_inventories(
    declared: Mapping[str, Mapping[str, Any]],
    initialization: Mapping[str, str],
    development: Mapping[str, str],
) -> None:
    inventories = {
        "initialization": initialization,
        "development": development,
    }
    if set(declared) != set(inventories):
        raise ValueError(
            "trace_inventories must declare initialization and development"
        )
    for role, inventory in inventories.items():
        _verify_declared_inventory(declared[role], inventory)


def _publish_staged_outputs(
    staged: Sequence[Path], destinations: Sequence[Path]
) -> None:
    if len(staged) != len(destinations):
        raise ValueError("staged output and destination counts differ")
    existing = [str(path) for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(
            f"refusing to overwrite outputs created during run: {existing}"
        )
    published: list[Path] = []
    try:
        for partial, final in zip(staged, destinations, strict=True):
            os.link(partial, final)
            published.append(final)
            partial.unlink()
    except BaseException:
        _cleanup_partial_outputs(staged)
        for path in published:
            path.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config_path = _resolve(args.config)
    config, config_hash = _load_config(config_path)
    primary, primary_folds, primary_sidecars = _load_primary_inputs(config)
    primary_config = primary["config"]
    if primary_config["score_kv_costs_ms"] != config["score_kv_costs_ms"]:
        raise ValueError("same-repo costs differ from the primary result")
    if primary_config["restore_cost_fraction"] != config["restore_cost_fraction"]:
        raise ValueError("same-repo restore fraction differs from the primary result")

    initialization_manifest_path = _resolve(
        Path(primary_config["initialization_manifest"])
    )
    development_manifest_path = _resolve(Path(primary_config["development_manifest"]))
    initialization = _load_corpus(initialization_manifest_path)
    development = _load_corpus(development_manifest_path)
    (
        initialization_manifest,
        initialization_task_ids,
        initialization_samples,
        initialization_inventory,
        initialization_manifest_hash,
        initialization_task_ids_hash,
    ) = initialization
    (
        development_manifest,
        development_task_ids,
        development_samples,
        development_inventory,
        development_manifest_hash,
        development_task_ids_hash,
    ) = development
    initialization_task_ids_path = _resolve(
        Path(initialization_manifest["task_ids_file"])
    )
    development_task_ids_path = _resolve(Path(development_manifest["task_ids_file"]))
    _verify_trace_inventories(
        config["trace_inventories"],
        initialization_inventory,
        development_inventory,
    )
    if set(initialization_task_ids) & set(development_task_ids):
        raise ValueError("initialization and development tasks overlap")
    initialization_rows = _sample_rows(
        initialization_samples, set(initialization_task_ids)
    )
    development_rows = _sample_rows(development_samples, set(development_task_ids))
    initialization_rows_by_task = _rows_by_task(initialization_rows)
    development_rows_by_task = _rows_by_task(development_rows)
    identities = _task_identities([*initialization_task_ids, *development_task_ids])
    initialization_rows_by_repository = _rows_by_repository(
        initialization_rows_by_task, identities
    )

    output_json = _resolve(Path(config["outputs"]["json"]))
    output_markdown = _resolve(Path(config["outputs"]["markdown"]))
    output_records = output_json.with_name(output_json.stem + "-records.jsonl.zst")
    staging_token = f"{os.getpid()}-{uuid.uuid4().hex}"
    output_records_partial = output_records.with_name(
        f".{output_records.name}.{staging_token}.partial"
    )
    output_markdown_partial = output_markdown.with_name(
        f".{output_markdown.name}.{staging_token}.partial"
    )
    output_json_partial = output_json.with_name(
        f".{output_json.name}.{staging_token}.partial"
    )
    staged_outputs = (
        output_records_partial,
        output_markdown_partial,
        output_json_partial,
    )
    final_outputs = (output_records, output_markdown, output_json)
    for output in (*final_outputs, *staged_outputs):
        if output.exists():
            raise FileExistsError(f"refusing to mix stale output: {output}")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)

    source_hashes = dict(_SOURCE_HASHES_AT_IMPORT)
    timing_environment = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "pyyaml": yaml.__version__,
        "effective_worker_count": 1,
        "parallel_unit": "serial_outer_fold_task",
    }
    writer = _ZstdJsonlWriter(output_records_partial)
    all_test_decisions: list[dict[str, Any]] = []
    fold_results: list[dict[str, Any]] = []
    try:
        writer.write_many(
            [
                {
                    "record_type": "run_metadata",
                    "schema_version": 1,
                    "status": config["status"],
                    "certificate": False,
                    "config": config,
                    "config_sha256": config_hash,
                    "primary_result": primary,
                    "source_sha256": source_hashes,
                    "git_sha": _git_sha(),
                    "run_started": dt.datetime.now().isoformat(timespec="seconds"),
                    "timing_environment": timing_environment,
                },
                *primary_sidecars,
                {
                    "record_type": "input_manifest",
                    "corpus_role": "initialization",
                    "manifest": initialization_manifest,
                    "manifest_sha256": initialization_manifest_hash,
                    "task_ids_sha256": initialization_task_ids_hash,
                },
                {
                    "record_type": "input_manifest",
                    "corpus_role": "development",
                    "manifest": development_manifest,
                    "manifest_sha256": development_manifest_hash,
                    "task_ids_sha256": development_task_ids_hash,
                },
            ]
        )
        writer.write_many(_source_snapshot_records(_SOURCE_HASHES_AT_IMPORT))
        writer.write_many(
            {
                "record_type": "input_trace",
                "corpus_role": role,
                "source_trace": path,
                "sha256": digest,
            }
            for role, inventory in (
                ("initialization", initialization_inventory),
                ("development", development_inventory),
            )
            for path, digest in sorted(inventory.items())
        )
        writer.write_many(
            {"record_type": "task_identity", **identity}
            for identity in identities.values()
        )

        costs_ms = [float(cost) for cost in config["score_kv_costs_ms"]]
        guard_ms = float(primary_config["guard_ms"])
        restore_fraction = float(config["restore_cost_fraction"])
        minimum_tasks = int(config["minimum_distinct_prior_tasks"])
        command_field = str(primary_config["command_field"])
        max_prefix_depth = int(primary_config["max_prefix_depth"])
        skip_leading_cd = bool(primary_config["skip_leading_cd"])

        for outer_fold in range(1, 6):
            fold_input = primary_folds[outer_fold]
            metadata = fold_input["metadata"]
            calibration_tasks = set(metadata["warmup_tasks"])
            test_tasks = set(metadata["test_tasks"])
            if calibration_tasks & test_tasks:
                raise AssertionError(f"outer fold {outer_fold} task overlap")
            if calibration_tasks | test_tasks != set(development_task_ids):
                raise AssertionError(f"outer fold {outer_fold} does not cover Fresh277")
            writer.write_many(
                [
                    {
                        "record_type": "fold_manifest",
                        "outer_fold": outer_fold,
                        "initialization_tasks": initialization_task_ids,
                        "calibration_tasks": sorted(calibration_tasks),
                        "test_tasks": sorted(test_tasks),
                        "baseline_sidecar": fold_input["sidecar"],
                    }
                ]
            )

            calibration_by_repository: dict[str, list[str]] = {}
            for task_id in calibration_tasks:
                repository = identities[task_id]["repository_canonical"]
                if repository is not None:
                    calibration_by_repository.setdefault(str(repository), []).append(
                        task_id
                    )
            fit_candidates: list[dict[str, Any]] = []
            fit_started = perf_counter_ns()
            for task_id in sorted(calibration_tasks):
                identity = identities[task_id]
                repository = identity["repository_canonical"]
                profile_rows = list(
                    initialization_rows_by_repository.get(str(repository), [])
                    if repository is not None
                    else []
                )
                if repository is not None:
                    for profile_task in calibration_by_repository.get(
                        str(repository), []
                    ):
                        if profile_task != task_id:
                            profile_rows.extend(development_rows_by_task[profile_task])
                rows = _score_repository_task(
                    development_rows_by_task[task_id],
                    identity=identity,
                    profile_rows=profile_rows,
                    costs_ms=costs_ms,
                    guard_ms=guard_ms,
                    restore_cost_fraction=restore_fraction,
                    command_field=command_field,
                    max_prefix_depth=max_prefix_depth,
                    skip_leading_cd=skip_leading_cd,
                    minimum_distinct_prior_tasks=minimum_tasks,
                )
                for row in rows:
                    if task_id in row["same_repo_prior_task_ids"]:
                        raise AssertionError(
                            f"outer fold {outer_fold} calibration task leaked into profile"
                        )
                    row["record_type"] = "guard_fit_decision"
                    row["outer_fold"] = outer_fold
                    row["fit_task_id"] = task_id
                fit_candidates.extend(rows)
            guard_fit_runtime_ms = (perf_counter_ns() - fit_started) / 1_000_000.0
            guard_started = perf_counter_ns()
            guard = select_probe_guard(
                fit_candidates,
                score_field="same_repo_margin_normalized",
                candidate_field="same_repo_candidate_trigger_ms",
                restore_cost_fraction=restore_fraction,
            )
            guard_runtime_ms = (perf_counter_ns() - guard_started) / 1_000_000.0
            guard_trace = _guard_objective_trace(
                fit_candidates,
                restore_cost_fraction=restore_fraction,
                selected_guard=guard["selected_guard_normalized"],
            )
            writer.write_many(fit_candidates)
            writer.write_many(
                {
                    "record_type": "guard_objective",
                    "outer_fold": outer_fold,
                    **row,
                }
                for row in guard_trace
            )

            test_by_repository: dict[str, list[str]] = {}
            for task_id in calibration_tasks:
                repository = identities[task_id]["repository_canonical"]
                if repository is not None:
                    test_by_repository.setdefault(str(repository), []).append(task_id)
            test_candidates: list[dict[str, Any]] = []
            test_started = perf_counter_ns()
            for task_id in sorted(test_tasks):
                identity = identities[task_id]
                repository = identity["repository_canonical"]
                profile_rows = list(
                    initialization_rows_by_repository.get(str(repository), [])
                    if repository is not None
                    else []
                )
                if repository is not None:
                    for profile_task in test_by_repository.get(str(repository), []):
                        profile_rows.extend(development_rows_by_task[profile_task])
                rows = _score_repository_task(
                    development_rows_by_task[task_id],
                    identity=identity,
                    profile_rows=profile_rows,
                    costs_ms=costs_ms,
                    guard_ms=guard_ms,
                    restore_cost_fraction=restore_fraction,
                    command_field=command_field,
                    max_prefix_depth=max_prefix_depth,
                    skip_leading_cd=skip_leading_cd,
                    minimum_distinct_prior_tasks=minimum_tasks,
                )
                for row in rows:
                    if task_id in row["same_repo_prior_task_ids"]:
                        raise AssertionError(
                            f"outer fold {outer_fold} test task leaked into profile"
                        )
                test_candidates.extend(rows)
            test_score_runtime_ms = (perf_counter_ns() - test_started) / 1_000_000.0
            test_decisions = _apply_guard_and_baseline(
                test_candidates,
                fold_input["baseline_decisions"],
                selected_guard=guard["selected_guard_normalized"],
                outer_fold=outer_fold,
                baseline_sidecar=fold_input["sidecar"],
            )
            writer.write_many(test_decisions)
            fold_summary = _fold_summary(test_decisions, costs_ms)
            fold_result = {
                "outer_fold": outer_fold,
                "calibration_task_count": len(calibration_tasks),
                "test_task_count": len(test_tasks),
                "guard": guard,
                "guard_objective_trace": guard_trace,
                "timing_ms": {
                    "guard_fit_candidate_scoring": guard_fit_runtime_ms,
                    "guard_selection": guard_runtime_ms,
                    "test_candidate_scoring": test_score_runtime_ms,
                },
                **fold_summary,
            }
            fold_results.append(fold_result)
            writer.write_many([{"record_type": "fold_summary", **fold_result}])
            all_test_decisions.extend(test_decisions)

        if len(all_test_decisions) != len(development_rows) * len(
            config["score_kv_costs_ms"]
        ):
            raise AssertionError(
                "same-repo test decisions do not cover every Fresh277 call"
            )
        summary = _fold_summary(
            all_test_decisions,
            [float(cost) for cost in config["score_kv_costs_ms"]],
        )
        point_estimates = summary["point_estimates"]["same_repo_vs_warmup_snapshot"]
        negative_costs = [
            cost
            for cost in config["score_kv_costs_ms"]
            if point_estimates[str(float(cost))]["paired_delta_ms"] < 0.0
        ]
        verdict = "DROP" if negative_costs else "KEEP"
        verdict_explanation = (
            f"negative paired utility at costs {negative_costs}"
            if negative_costs
            else "no headline cost has negative paired utility; mechanism work only"
        )

        _verify_file_hash(config_path, config_hash)
        _verify_file_hash(initialization_manifest_path, initialization_manifest_hash)
        _verify_file_hash(development_manifest_path, development_manifest_hash)
        _verify_file_hash(initialization_task_ids_path, initialization_task_ids_hash)
        _verify_file_hash(development_task_ids_path, development_task_ids_hash)
        _verify_file_hash(
            _resolve(Path(config["primary_result"]["file"])),
            config["primary_result"]["sha256"],
        )
        for sidecar in primary_sidecars:
            _verify_file_hash(Path(sidecar["file"]), str(sidecar["sha256"]))
        if _source_tree_hashes() != source_hashes:
            raise ValueError("local source tree changed during the run")
        final_initialization_inventory = _load_corpus(initialization_manifest_path)[3]
        final_development_inventory = _load_corpus(development_manifest_path)[3]
        if final_initialization_inventory != initialization_inventory:
            raise ValueError("initialization trace corpus changed during the run")
        if final_development_inventory != development_inventory:
            raise ValueError("development trace corpus changed during the run")
        _verify_trace_inventories(
            config["trace_inventories"],
            final_initialization_inventory,
            final_development_inventory,
        )
        _verify_runtime_inventory(final_initialization_inventory)
        _verify_runtime_inventory(final_development_inventory)

        writer.finish()
        result = {
            "schema_version": 1,
            "status": config["status"],
            "certificate": False,
            "verdict": verdict,
            "verdict_explanation": verdict_explanation,
            "config": config,
            "folds": fold_results,
            "summary": summary,
            "records_sidecar": {
                "file": output_records.name,
                "sha256": _sha256(output_records_partial),
                "record_count": writer.count,
            },
            "provenance": {
                "generated": dt.datetime.now().isoformat(timespec="seconds"),
                "git_sha": _git_sha(),
                "config_sha256": config_hash,
                "source_sha256": source_hashes,
                "primary_result_sha256": config["primary_result"]["sha256"],
                "primary_sidecars": primary_sidecars,
                "initialization_manifest_sha256": initialization_manifest_hash,
                "initialization_task_ids_sha256": initialization_task_ids_hash,
                "development_manifest_sha256": development_manifest_hash,
                "development_task_ids_sha256": development_task_ids_hash,
                "initialization_trace_inventory_sha256": _inventory_digest(
                    initialization_inventory
                ),
                "development_trace_inventory_sha256": _inventory_digest(
                    development_inventory
                ),
                "initialization_trace_count": len(initialization_inventory),
                "development_trace_count": len(development_inventory),
                "initialization_task_count": len(initialization_task_ids),
                "development_task_count": len(development_task_ids),
                "development_call_count": len(development_rows),
                "timing_environment": timing_environment,
            },
        }
        output_json_partial.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        output_markdown_partial.write_text(_render_markdown(result), encoding="utf-8")
        _publish_staged_outputs(staged_outputs, final_outputs)
    except BaseException:
        writer.abort()
        _cleanup_partial_outputs(staged_outputs)
        raise

    print(f"verdict={verdict}")
    for cost in config["score_kv_costs_ms"]:
        point = point_estimates[str(float(cost))]
        print(
            f"kv{cost}: delta={point['paired_delta_ms']:.2f}ms "
            f"tasks=+{point['positive_task_count']}/-{point['negative_task_count']}"
        )
    print(f"wrote {output_json}")
    print(f"wrote {output_markdown}")
    print(f"wrote {output_records}")


if __name__ == "__main__":
    main()
