"""Threshold-sweep diagnostics and failure decomposition for utility clocks."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from trace_collect.tool_latency_headroom import (
    POLICY_TRIGGER_FIELDS,
    analyze_utility_headroom,
    load_decision_rows,
)
from trace_collect.tool_latency_utility_clock import trigger_policy_utility_ms


def analyze_threshold_sweep_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load a frozen sweep manifest and analyze every named corpus."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("sweep manifest must be a JSON object")
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported sweep manifest schema_version")

    costs_ms = _positive_float_list(manifest, "expected_costs_ms")
    case_thresholds_ms = _positive_float_list(manifest, "case_thresholds_ms")
    if not set(case_thresholds_ms).issubset(costs_ms):
        raise ValueError("case_thresholds_ms must be included in expected_costs_ms")
    bootstrap = _required_mapping(manifest, "bootstrap")
    replicates = int(bootstrap["replicates"])
    seed = int(bootstrap["seed"])
    confidence_level = float(bootstrap["confidence_level"])
    corpora = _required_mapping(manifest, "corpora")
    if not corpora:
        raise ValueError("sweep manifest must name at least one corpus")

    corpus_results: dict[str, Any] = {}
    resolved_inputs: dict[str, list[str]] = {}
    for corpus, serialized_paths in sorted(corpora.items()):
        if not isinstance(corpus, str) or not corpus:
            raise ValueError("corpus names must be non-empty strings")
        if not isinstance(serialized_paths, list) or not serialized_paths:
            raise ValueError(f"corpus {corpus!r} must list decision paths")
        paths = [
            _resolve_manifest_path(manifest_path, serialized_path)
            for serialized_path in serialized_paths
        ]
        rows = load_decision_rows(paths)
        corpus_results[corpus] = analyze_threshold_sweep(
            rows,
            expected_costs_ms=costs_ms,
            case_thresholds_ms=case_thresholds_ms,
            bootstrap_replicates=replicates,
            bootstrap_seed=seed,
            confidence_level=confidence_level,
        )
        resolved_inputs[corpus] = [str(path) for path in paths]

    return {
        "schema_version": 1,
        "manifest_path": str(manifest_path.resolve()),
        "expected_costs_ms": costs_ms,
        "case_thresholds_ms": case_thresholds_ms,
        "bootstrap": {
            "unit": "task_id",
            "replicates": replicates,
            "seed": seed,
            "confidence_level": confidence_level,
            "interval": "percentile",
        },
        "input_paths": resolved_inputs,
        "corpora": corpus_results,
    }


def analyze_threshold_sweep(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_costs_ms: Sequence[float],
    case_thresholds_ms: Sequence[float],
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Analyze one corpus across costs and decompose selected case points."""
    base = analyze_utility_headroom(
        rows,
        expected_costs_ms=expected_costs_ms,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    _validate_case_fields(rows)
    by_cost: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cost[float(row["kv_cost_ms"])].append(row)

    for cost_ms, cost_rows in sorted(by_cost.items()):
        point = base["points"][str(cost_ms)]
        decomposition = {
            policy: _decompose_policy(cost_rows, policy=policy)
            for policy in ("mean_hazard", "robust_clock")
        }
        for policy, policy_decomposition in decomposition.items():
            expected_delta_ms = point["policies"][policy]["delta_vs_deadline_ms"]
            if not math.isclose(
                policy_decomposition["delta_vs_deadline_ms"],
                expected_delta_ms,
                rel_tol=1e-12,
                abs_tol=1e-6,
            ):
                raise AssertionError(
                    f"{policy} decomposition mismatch at cost {cost_ms}"
                )
        point["decomposition"] = decomposition

    case_studies: dict[str, Any] = {}
    for threshold_ms in case_thresholds_ms:
        cost_rows = by_cost[float(threshold_ms)]
        case_studies[str(float(threshold_ms))] = {
            "overall": {
                policy: _decompose_policy(cost_rows, policy=policy)
                for policy in ("mean_hazard", "robust_clock")
            },
            "robust_clock_groupings": {
                "tool_name": _group_summaries(
                    cost_rows,
                    key_fn=lambda row: str(row["tool_name"]),
                ),
                "task_id": _group_summaries(
                    cost_rows,
                    key_fn=lambda row: str(row["task_id"]),
                ),
                "robust_source": _group_summaries(
                    cost_rows,
                    key_fn=lambda row: str(row["robust_source"]),
                ),
                "robust_task_count": _group_summaries(
                    cost_rows,
                    key_fn=lambda row: str(int(row["robust_task_count"])),
                ),
                "robust_node": _group_summaries(
                    cost_rows,
                    key_fn=_robust_node_key,
                ),
            },
        }
    base["case_studies"] = case_studies
    return base


def plot_threshold_sweep(result: Mapping[str, Any], output_dir: Path) -> list[Path]:
    """Write cross-corpus threshold and decomposition line charts."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "threshold-sweep plotting requires the existing figures extra; "
            "run `uv sync --extra dev --extra figures`"
        ) from exc

    corpora = _required_mapping(result, "corpora")
    if not corpora:
        raise ValueError("threshold sweep result contains no corpora")
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap("tab10").colors

    overview, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for index, (corpus, corpus_result) in enumerate(sorted(corpora.items())):
        points = _ordered_points(corpus_result)
        thresholds = [point["threshold_ms"] for point in points]
        color = colors[index % len(colors)]
        rho = [_percent_or_nan(point["rho_headroom_over_oracle"]) for point in points]
        axes[0].plot(thresholds, rho, marker="o", label=corpus, color=color)
        lower = [
            _percent_or_nan(
                point["task_cluster_bootstrap"]["rho_headroom_over_oracle"]["lower"]
            )
            for point in points
        ]
        upper = [
            _percent_or_nan(
                point["task_cluster_bootstrap"]["rho_headroom_over_oracle"]["upper"]
            )
            for point in points
        ]
        axes[0].fill_between(thresholds, lower, upper, color=color, alpha=0.12)
        robust_capture = [
            _percent_or_nan(
                point["policies"]["robust_clock"][
                    "captured_fraction_of_deadline_headroom"
                ]
            )
            for point in points
        ]
        axes[1].plot(
            thresholds,
            robust_capture,
            marker="o",
            label=corpus,
            color=color,
        )

    axes[0].set_title("Exact deadline headroom")
    axes[0].set_ylabel("Headroom / oracle (%)")
    axes[1].set_title("Robust clock headroom capture")
    axes[1].set_ylabel("Delta vs deadline / headroom (%)")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    for axis in axes:
        axis.set_xlabel("Threshold = action cost (ms)")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    overview_png = output_dir / "threshold_sweep_overview.png"
    overview_pdf = output_dir / "threshold_sweep_overview.pdf"
    overview.savefig(overview_png, dpi=180)
    overview.savefig(overview_pdf)
    plt.close(overview)

    decomposition, axes_array = plt.subplots(
        1,
        len(corpora),
        figsize=(5.2 * len(corpora), 4.5),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, (corpus, corpus_result) in zip(
        axes_array[0], sorted(corpora.items()), strict=True
    ):
        points = _ordered_points(corpus_result)
        thresholds = [point["threshold_ms"] for point in points]
        robust = [point["decomposition"]["robust_clock"] for point in points]
        axis.plot(
            thresholds,
            [_percent_or_nan(row["band_gain_fraction_of_headroom"]) for row in robust],
            marker="o",
            label="Band-positive gain",
        )
        axis.plot(
            thresholds,
            [
                _percent_or_nan(row["short_penalty_fraction_of_headroom"])
                for row in robust
            ],
            marker="o",
            label="Early-short penalty",
        )
        axis.plot(
            thresholds,
            [_percent_or_nan(row["captured_fraction_of_headroom"]) for row in robust],
            marker="o",
            linewidth=2.2,
            label="Net capture",
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(corpus)
        axis.set_xlabel("Threshold = action cost (ms)")
        axis.set_ylabel("Fraction of exact headroom (%)")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    decomposition_png = output_dir / "robust_capture_decomposition.png"
    decomposition_pdf = output_dir / "robust_capture_decomposition.pdf"
    decomposition.savefig(decomposition_png, dpi=180)
    decomposition.savefig(decomposition_pdf)
    plt.close(decomposition)
    return [overview_png, overview_pdf, decomposition_png, decomposition_pdf]


def _decompose_policy(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy: str,
) -> dict[str, Any]:
    if policy not in POLICY_TRIGGER_FIELDS or policy == "deadline_only":
        raise ValueError(f"unsupported decomposition policy: {policy}")
    band_gain_values: list[float] = []
    short_penalties: list[float] = []
    far_tail_deltas: list[float] = []
    headroom_values: list[float] = []
    early_trigger_count = 0
    early_short_count = 0
    band_count = 0
    positive_count = 0
    for row in rows:
        latency_ms = float(row["latency_ms"])
        threshold_ms = float(row["threshold_ms"])
        cost_ms = float(row["kv_cost_ms"])
        trigger_ms = float(row[POLICY_TRIGGER_FIELDS[policy]])
        deadline_ms = trigger_policy_utility_ms(
            latency_ms,
            threshold_ms,
            threshold_ms=threshold_ms,
            kv_cost_ms=cost_ms,
        )
        policy_ms = trigger_policy_utility_ms(
            latency_ms,
            trigger_ms,
            threshold_ms=threshold_ms,
            kv_cost_ms=cost_ms,
        )
        delta_ms = policy_ms - deadline_ms
        is_positive = latency_ms > threshold_ms
        is_band = is_positive and latency_ms < threshold_ms + cost_ms
        if is_positive:
            positive_count += 1
        if is_band:
            band_count += 1
            headroom_values.append(2.0 * (cost_ms - (latency_ms - threshold_ms)))
            if delta_ms < -1e-9:
                raise AssertionError(
                    "early trigger lost utility on a band-positive call"
                )
            band_gain_values.append(delta_ms)
        elif not is_positive:
            if delta_ms > 1e-9:
                raise AssertionError("early trigger gained utility on a short call")
            short_penalties.append(-delta_ms)
        else:
            far_tail_deltas.append(delta_ms)
        if latency_ms > trigger_ms and trigger_ms < threshold_ms:
            early_trigger_count += 1
            if not is_positive:
                early_short_count += 1

    band_gain_ms = math.fsum(band_gain_values)
    short_penalty_ms = math.fsum(short_penalties)
    far_tail_delta_ms = math.fsum(far_tail_deltas)
    if not math.isclose(far_tail_delta_ms, 0.0, rel_tol=0.0, abs_tol=1e-6):
        raise AssertionError(f"far-tail delta must be zero, got {far_tail_delta_ms}")
    headroom_ms = math.fsum(headroom_values)
    delta_ms = band_gain_ms - short_penalty_ms
    return {
        "call_count": len(rows),
        "positive_count": positive_count,
        "band_count": band_count,
        "early_trigger_count": early_trigger_count,
        "early_trigger_on_short_count": early_short_count,
        "deadline_headroom_ms": headroom_ms,
        "band_gain_ms": band_gain_ms,
        "short_exposure_penalty_ms": short_penalty_ms,
        "far_tail_delta_ms": far_tail_delta_ms,
        "delta_vs_deadline_ms": delta_ms,
        "band_gain_fraction_of_headroom": _ratio_or_none(band_gain_ms, headroom_ms),
        "short_penalty_fraction_of_headroom": _ratio_or_none(
            short_penalty_ms, headroom_ms
        ),
        "captured_fraction_of_headroom": _ratio_or_none(delta_ms, headroom_ms),
    }


def _group_summaries(
    rows: Sequence[Mapping[str, Any]],
    *,
    key_fn: Callable[[Mapping[str, Any]], str],
) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    return [
        {"group": group, **_decompose_policy(group_rows, policy="robust_clock")}
        for group, group_rows in sorted(groups.items())
    ]


def _validate_case_fields(rows: Sequence[Mapping[str, Any]]) -> None:
    required = {"robust_source", "robust_task_count", "robust_group_key"}
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"case-study row is missing fields: {sorted(missing)}")
        if not str(row["robust_source"]):
            raise ValueError("robust_source must be non-empty")
        task_count = int(row["robust_task_count"])
        if task_count <= 0 or float(row["robust_task_count"]) != task_count:
            raise ValueError("robust_task_count must be a positive integer")
        _robust_node_key(row)


def _robust_node_key(row: Mapping[str, Any]) -> str:
    source = str(row["robust_source"])
    group_key = row["robust_group_key"]
    if source == "prior_global":
        if group_key is not None:
            raise ValueError("prior_global rows must not have a robust_group_key")
        return "prior_global:*"
    if source == "prior_tool":
        if group_key is not None:
            raise ValueError("prior_tool rows must not have a robust_group_key")
        return f"prior_tool:{row['tool_name']}"
    if source == "prior_group":
        if not isinstance(group_key, str) or not group_key:
            raise ValueError("prior_group rows require a non-empty robust_group_key")
        return f"prior_group:{group_key}"
    raise ValueError(f"unsupported robust_source: {source!r}")


def _ordered_points(corpus_result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    points = _required_mapping(corpus_result, "points")
    return [points[key] for key in sorted(points, key=float)]


def _percent_or_nan(value: float | None) -> float:
    return 100.0 * value if value is not None else float("nan")


def _ratio_or_none(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _positive_float_list(manifest: Mapping[str, Any], field: str) -> list[float]:
    value = manifest.get(field)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    output = [float(item) for item in value]
    if len(set(output)) != len(output):
        raise ValueError(f"{field} must contain unique values")
    if not all(math.isfinite(item) and item > 0.0 for item in output):
        raise ValueError(f"{field} must contain positive finite values")
    return output


def _required_mapping(value: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    output = value.get(field)
    if not isinstance(output, dict):
        raise ValueError(f"{field} must be an object")
    return output


def _resolve_manifest_path(manifest_path: Path, serialized_path: Any) -> Path:
    if not isinstance(serialized_path, str) or not serialized_path:
        raise ValueError("decision paths must be non-empty strings")
    path = Path(serialized_path)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


__all__ = [
    "analyze_threshold_sweep",
    "analyze_threshold_sweep_manifest",
    "plot_threshold_sweep",
]
