"""Cross-benchmark transfer of frozen offline-gated trigger policies (E2).

The certification in scripts/run_offline_gated_robust_confirmation.py fits the
offline-probe guard and outer triggers on a frozen SWE-ReBench profile. This
module asks a different question: do those *frozen* fitting rules transfer to a
target benchmark whose task distribution the method never saw during that fit?

The whole frozen SWE-ReBench collection (``data/all.jsonl``, all 100 tasks) is
used as a single profile. Target traces supply disjoint evaluation tasks. For
each restore fraction the offline-probe guard is re-selected on the profile
(cross-fitted over its inner folds) and applied frozen to the target eval
tasks; no target information enters any fit stage. Every fit and scoring stage
shares one ``restore_cost_fraction``, exactly as in the certification and in
:func:`trace_collect.restore_cost_analysis.run_mode_b_refit`.

This is a *sensitivity* probe, not a fresh certification: the target corpora
were previously touched during method development, so a positive result is not
independent evidence. That caveat is stamped into the payload
(``exposure_note``) and the summary.

Output payloads keep ``schema_version`` 1 with additive keys only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trace_collect.restore_cost_analysis import (
    _validate_fractions,
    _write_json,
    _write_jsonl,
    fraction_key,
    render_summary_markdown,
)
from trace_collect.tool_gap_extractor import discover_trace_files
from trace_collect.tool_latency_confirmation import paired_task_cluster_bootstrap
from trace_collect.tool_latency_dataset import (
    extract_many_tool_latency_samples,
    read_tool_latency_jsonl,
)
from trace_collect.tool_latency_offline_probe import evaluate_offline_probe_clock


# The single frozen outer fold used to stamp every transfer decision. There is
# no cross-validation here: one profile, one disjoint target eval set.
TRANSFER_OUTER_FOLD = "transfer"

# (name, treatment field, baseline field, enforce gated-treatment invariant).
# Each early policy is measured against never firing early (the fixed
# deadline). The gate invariant is disabled because none of these baselines is
# the gate's own fallback pair.
TRANSFER_COMPARISONS: tuple[tuple[str, str, str, bool], ...] = (
    (
        "gated_vs_deadline",
        "offline_gated_robust_trigger_ms",
        "deadline_trigger_ms",
        False,
    ),
    ("robust_vs_deadline", "robust_trigger_ms", "deadline_trigger_ms", False),
    (
        "mean_hazard_vs_deadline",
        "mean_hazard_trigger_ms",
        "deadline_trigger_ms",
        False,
    ),
)

_MANIFEST_CONFIG_FIELDS = (
    "inner_folds",
    "costs_ms",
    "guard_ms",
    "min_tool_history",
    "min_profile_tasks",
    "command_field",
    "max_prefix_depth",
    "skip_leading_cd",
)

_EXPOSURE_NOTE = (
    "The target corpora were previously used during method development, so "
    "these results measure transfer sensitivity of the frozen fitting rules, "
    "not a fresh independent certification."
)


def run_transfer_evaluation(
    confirmation_root: Path,
    *,
    output_root: Path,
    restore_cost_fractions: list[float],
    replicates: int,
    confidence_level: float,
    seed: int,
    eval_trace_root: Path | None = None,
    eval_latencies: Path | None = None,
) -> dict[str, Any]:
    """Fit the frozen guard on the SWE-ReBench profile, score a target corpus.

    Exactly one of ``eval_trace_root`` (canonical ``trace.jsonl`` files to
    extract) or ``eval_latencies`` (a pre-extracted tool-latency JSONL) must be
    given; the latter reads the same schema
    :func:`trace_collect.tool_latency_dataset.write_tool_latency_jsonl` writes.
    """

    _validate_fractions(restore_cost_fractions)
    confirmation_root = confirmation_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to mix stale output: {output_root}")
    config = _read_manifest_config(confirmation_root)
    costs = [float(cost) for cost in config["costs_ms"]]

    profile_rows = read_tool_latency_jsonl(confirmation_root / "data" / "all.jsonl")
    profile_tasks = {str(row["task_id"]) for row in profile_rows}

    eval_rows, eval_source = _load_eval_rows(
        eval_trace_root=eval_trace_root,
        eval_latencies=eval_latencies,
    )
    eval_tasks = {str(row["task_id"]) for row in eval_rows}
    eval_trace_count = len({str(row["source_trace"]) for row in eval_rows})

    overlap = profile_tasks & eval_tasks
    if overlap:
        raise AssertionError(
            "profile and eval task ids must be disjoint across benchmarks; "
            f"overlap: {sorted(overlap)}"
        )

    output_root.mkdir(parents=True)
    comparisons: dict[str, dict[str, Any]] = {
        name: {
            "treatment_trigger_field": treatment_field,
            "baseline_trigger_field": baseline_field,
            "enforce_gated_treatment": enforce_gated,
            "by_restore_cost_fraction": {},
        }
        for name, treatment_field, baseline_field, enforce_gated in (
            TRANSFER_COMPARISONS
        )
    }
    calibrations_by_fraction: dict[str, dict[str, Any]] = {}
    for fraction in restore_cost_fractions:
        key = fraction_key(fraction)
        fraction_root = output_root / f"rho_{key}"
        fraction_root.mkdir()
        result = evaluate_offline_probe_clock(
            eval_rows,
            profile_rows=profile_rows,
            kv_costs_ms=costs,
            guard_ms=config["guard_ms"],
            inner_folds=config["inner_folds"],
            min_tool_history=config["min_tool_history"],
            min_profile_tasks=config["min_profile_tasks"],
            command_field=config["command_field"],
            max_prefix_depth=config["max_prefix_depth"],
            skip_leading_cd=config["skip_leading_cd"],
            restore_cost_fraction=fraction,
        )
        fold_decisions = result.pop("decisions")
        decisions = [
            {**row, "outer_fold": TRANSFER_OUTER_FOLD} for row in fold_decisions
        ]
        _write_json(fraction_root / "transfer_summary.json", result)
        _write_jsonl(fraction_root / "transfer_decisions.jsonl", decisions)
        calibrations_by_fraction[key] = {
            "calibration": result["calibration"],
            "robust_calibration": result["robust_calibration"],
        }
        for name, treatment_field, baseline_field, enforce_gated in (
            TRANSFER_COMPARISONS
        ):
            comparisons[name]["by_restore_cost_fraction"][key] = (
                paired_task_cluster_bootstrap(
                    decisions,
                    costs_ms=costs,
                    replicates=replicates,
                    confidence_level=confidence_level,
                    seed=seed,
                    baseline_trigger_field=baseline_field,
                    treatment_trigger_field=treatment_field,
                    restore_cost_fraction=fraction,
                    enforce_gated_treatment=enforce_gated,
                )
            )

    result = {
        "schema_version": 1,
        "mode": "cross_benchmark_transfer",
        "confirmation_root": str(confirmation_root),
        "eval_trace_root": str(eval_source),
        "eval_trace_count": eval_trace_count,
        "eval_task_count": len(eval_tasks),
        "eval_row_count": len(eval_rows),
        "profile_row_count": len(profile_rows),
        "profile_task_count": len(profile_tasks),
        "costs_ms": costs,
        "restore_cost_fractions": restore_cost_fractions,
        "exposure_note": _EXPOSURE_NOTE,
        "bootstrap": {
            "replicates": replicates,
            "confidence_level": confidence_level,
            "seed": seed,
        },
        "calibrations_by_fraction": calibrations_by_fraction,
        "comparisons": comparisons,
    }
    _write_json(output_root / "cross_benchmark_transfer.json", result)
    (output_root / "summary.md").write_text(
        render_summary_markdown(
            result,
            title="Cross-benchmark transfer (E2)",
            intro_lines=(
                "The offline-probe guard is fitted on the whole frozen",
                "SWE-ReBench profile and applied frozen to a disjoint target",
                "benchmark; both policies in every contrast share one restore",
                "fraction. " + _EXPOSURE_NOTE,
            ),
        ),
        encoding="utf-8",
    )
    return result


def _load_eval_rows(
    *,
    eval_trace_root: Path | None,
    eval_latencies: Path | None,
) -> tuple[list[dict[str, Any]], Path]:
    """Return target eval rows and the resolved source path.

    Fails fast unless exactly one input is given and it yields rows.
    """

    if (eval_trace_root is None) == (eval_latencies is None):
        raise ValueError(
            "provide exactly one of eval_trace_root or eval_latencies"
        )
    if eval_trace_root is not None:
        eval_trace_root = eval_trace_root.resolve()
        trace_paths = discover_trace_files([eval_trace_root])
        if not trace_paths:
            raise ValueError(f"no trace.jsonl files found under {eval_trace_root}")
        samples = extract_many_tool_latency_samples(trace_paths)
        rows = [sample.to_json_obj() for sample in samples]
        return rows, eval_trace_root
    assert eval_latencies is not None
    eval_latencies = eval_latencies.resolve()
    rows = read_tool_latency_jsonl(eval_latencies)
    if not rows:
        raise ValueError(f"no tool latency rows found in {eval_latencies}")
    return rows, eval_latencies


def _read_manifest_config(confirmation_root: Path) -> dict[str, Any]:
    """Read the frozen config fields the transfer fit reuses verbatim."""

    manifest_path = confirmation_root / "provenance" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest must be a JSON object: {manifest_path}")
    missing = [field for field in _MANIFEST_CONFIG_FIELDS if field not in manifest]
    if missing:
        raise ValueError(f"manifest lacks frozen config fields: {missing}")
    return {field: manifest[field] for field in _MANIFEST_CONFIG_FIELDS}


__all__ = [
    "TRANSFER_COMPARISONS",
    "TRANSFER_OUTER_FOLD",
    "run_transfer_evaluation",
]
