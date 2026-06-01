"""Plot per-segment within-segment attention mean/std as a segment×iter grid.

Companion to ``plot_sparse_segment_grid.py``. Instead of the sparse-filtered
retained-mass diagnostic, this reads the recorded per-head within-segment
statistics (``head_span_*`` arrays in ``attention.npz``) and renders a 2×2 grid:

  - Top row:    within-segment attention MEAN   (prefill | decode)
  - Bottom row: within-segment attention STD    (prefill | decode)

``head_span_mean_*`` is, per (layer, head, segment), the mean post-softmax
attention weight over the tokens INSIDE that segment, averaged over the sampled
query rows. ``head_span_var_*`` is the matching within-segment population
variance. We reduce over the selected layers and all query heads (and, for
decode, over decode steps) into one MEAN and one STD per (segment, iter, phase).

These arrays are only populated when the run was recorded with
``--per-head-stats-layers`` (RecordingConfig.per_head_stats_layers non-empty).
An iter whose arrays are empty raises rather than silently rendering blank.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.recoding_figures.recording_loader import (  # noqa: E402
    IterationRecord,
    find_attempt_dirs,
    load_head_span_stats,
    load_iteration_records,
)
from scripts.recoding_figures.plot_sparse_segment_grid import (  # noqa: E402
    _artifact_relative_path,
    _integer_tick_positions,
    _json_ready,
    _load_json_required,
    _nanargmax_or_none,
    _nanmean_or_none,
    _portable_plot_summary,
    _role_counts,
    _safe_name,
    _segment_plot_label,
    _segment_turn_boundaries,
    _segments_for_record,
    _write_csv,
)

DECODE_REDUCE_MODES = ("pool_steps", "last_step")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="attempt, task, or run directories")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-orphans", action="store_true")
    parser.add_argument("--max-iters", type=int)
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help=(
            "Comma-separated subset of recorded head-stats layers to aggregate "
            "(e.g. 0,24,47). Default: all recorded layers."
        ),
    )
    parser.add_argument(
        "--decode-reduce",
        choices=DECODE_REDUCE_MODES,
        default="pool_steps",
        help=(
            "How to collapse decode steps: pool_steps treats every valid "
            "(layer, step, head) as one observation; last_step uses only the "
            "latest decode step per layer. Default pool_steps."
        ),
    )
    parser.add_argument(
        "--split-by-task",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write one grid per task. Default: true.",
    )
    args = parser.parse_args()

    layers = _parse_layer_arg(args.layers)
    summary = build_head_span_segment_grids(
        inputs=args.inputs,
        output_dir=args.output_dir,
        layers=layers,
        include_orphans=args.include_orphans,
        max_iters=args.max_iters,
        split_by_task=args.split_by_task,
        decode_reduce=args.decode_reduce,
    )
    print(json.dumps(_json_ready(summary), indent=2, sort_keys=True))


def _parse_layer_arg(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(part) for part in value.split(",") if part.strip() != ""]


def build_head_span_segment_grids(
    *,
    inputs: Sequence[Path],
    output_dir: Path,
    layers: Sequence[int] | None = None,
    include_orphans: bool = False,
    max_iters: int | None = None,
    split_by_task: bool = True,
    decode_reduce: str = "pool_steps",
) -> dict[str, Any]:
    """Build within-segment mean/std grids for one or more attempt paths."""
    if decode_reduce not in DECODE_REDUCE_MODES:
        raise ValueError(
            f"decode_reduce must be one of {DECODE_REDUCE_MODES}, got {decode_reduce!r}"
        )
    records = load_iteration_records(
        inputs,
        include_orphans=include_orphans,
        max_iters=max_iters,
    )
    attempt_dirs = find_attempt_dirs(inputs)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups: list[tuple[str, list[IterationRecord], Path]]
    if split_by_task:
        by_task: dict[str, list[IterationRecord]] = {}
        for record in records:
            by_task.setdefault(record.task, []).append(record)
        groups = [
            (task, task_records, output_dir / _safe_name(task))
            for task, task_records in sorted(by_task.items())
        ]
    else:
        groups = [("all_tasks", records, output_dir)]

    group_summaries = []
    layers_used_global: list[int] | None = None
    for label, group_records, group_dir in groups:
        trajectory_rows, layer_rows, layers_used = head_span_segment_rows(
            group_records,
            layers=layers,
            decode_reduce=decode_reduce,
        )
        if not trajectory_rows:
            raise ValueError(f"{label}: no head-span segment observations were found")
        if layers_used_global is None:
            layers_used_global = layers_used
        elif layers_used != layers_used_global:
            raise ValueError(
                f"{label}: layer set {layers_used} differs from other groups "
                f"{layers_used_global}; recordings are inconsistent"
            )
        summary_rows = _head_span_summary_rows(trajectory_rows)
        group_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(group_dir / "segment_head_span_trajectory.csv", trajectory_rows)
        _write_csv(group_dir / "segment_head_span_by_layer.csv", layer_rows)
        _write_csv(group_dir / "segment_head_span_summary.csv", summary_rows)
        plot_summary = _plot_head_span_grid(
            trajectory_rows,
            summary_rows,
            group_dir / "segment_head_span_grid",
            layers_used=layers_used,
            decode_reduce=decode_reduce,
        )
        plot_summary = _portable_plot_summary(plot_summary, artifact_root=output_dir)
        group_summary = {
            "label": label,
            "output_dir": _artifact_relative_path(group_dir, output_dir),
            "runtime_output_dir": str(group_dir),
            "n_records": len(group_records),
            "n_segments": len(summary_rows),
            "n_trajectory_rows": len(trajectory_rows),
            "n_layer_rows": len(layer_rows),
            "layers_used": layers_used,
            "role_counts": _role_counts(summary_rows),
            "plot": plot_summary,
        }
        (group_dir / "summary.json").write_text(
            json.dumps(_json_ready(group_summary), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (group_dir / "summary.md").write_text(
            _summary_markdown(group_summary, decode_reduce=decode_reduce),
            encoding="utf-8",
        )
        group_summaries.append(group_summary)

    run_summary = {
        "inputs": [str(path) for path in inputs],
        "attempt_dirs": [str(path) for path in attempt_dirs],
        "metric": "within_segment_attention_mean_std",
        "layers_used": layers_used_global,
        "decode_reduce": decode_reduce,
        "split_by_task": split_by_task,
        "groups": group_summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_json_ready(run_summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return run_summary


def reduce_head_span_cell(means: np.ndarray, variances: np.ndarray) -> dict[str, Any]:
    """Reduce per-(layer,head[,step]) within-segment stats into one cell.

    ``means``/``variances`` are flattened observation arrays for ONE segment in
    ONE phase. NaN entries (segment had zero kept key tokens for that
    layer/head/step) are dropped. Returns the equal-weight mean of the recorded
    within-segment means, the pooled within-segment std = sqrt(mean(variance))
    (average variances THEN sqrt — never the reverse), the cross-head std of the
    means (a head-disagreement diagnostic), and contributor counts.
    """
    means = np.asarray(means, dtype=np.float64).ravel()
    variances = np.asarray(variances, dtype=np.float64).ravel()
    n_total = int(means.size)
    # The writer co-fills mean and variance under one per-segment kept-keys
    # gate (hooks.py _build_head_span_arrays), so their NaN masks are identical.
    # Require both finite anyway so n_contributors is the exact denominator of
    # both the mean and the pooled std — no silent mean/var contributor drift.
    finite = np.isfinite(means) & np.isfinite(variances)
    n_contrib = int(finite.sum())
    if n_contrib == 0:
        return {
            "mean": None,
            "std": None,
            "var_pooled": None,
            "cross_head_std": None,
            "n_contributors": 0,
            "n_nan_contributors": n_total,
        }
    finite_means = means[finite]
    finite_vars = variances[finite]
    mean = float(finite_means.mean())
    var_pooled = float(finite_vars.mean())
    std = float(np.sqrt(max(var_pooled, 0.0)))
    cross_head_std = float(finite_means.std(ddof=0)) if n_contrib >= 2 else 0.0
    return {
        "mean": mean,
        "std": std,
        "var_pooled": var_pooled,
        "cross_head_std": cross_head_std,
        "n_contributors": n_contrib,
        "n_nan_contributors": n_total - n_contrib,
    }


def head_span_segment_rows(
    records: Sequence[IterationRecord],
    *,
    layers: Sequence[int] | None = None,
    decode_reduce: str = "pool_steps",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    """Return trajectory rows, per-layer rows, and the effective layer list."""
    trajectory: list[dict[str, Any]] = []
    by_layer: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    layers_used: list[int] | None = None

    for record in sorted(records, key=lambda item: (item.task, item.call_idx)):
        segments_payload = _load_json_required(record.iter_dir / "segments.json")
        segments = list(segments_payload.get("segments", []))
        segment_items = _segments_for_record(segments, record=record, metadata=metadata)
        if not segment_items:
            continue

        stats = load_head_span_stats(record.iter_dir)
        available = [int(layer) for layer in stats["head_stats_layers"].tolist()]
        if not available:
            raise ValueError(
                f"{record.iter_dir}: attention.npz has no per-head span stats "
                "(head_stats_layers is empty). Re-record with "
                "--per-head-stats-layers to populate head_span_* arrays."
            )
        selected_layers = list(layers) if layers is not None else list(available)
        missing = [layer for layer in selected_layers if layer not in available]
        if missing:
            raise ValueError(
                f"{record.iter_dir}: requested layers not recorded: {missing} "
                f"(available: {available})"
            )
        if layers_used is None:
            layers_used = selected_layers
        elif selected_layers != layers_used:
            raise ValueError(
                f"{record.iter_dir}: layer set {selected_layers} differs from "
                f"earlier iters {layers_used}; recordings are inconsistent"
            )
        positions = [available.index(layer) for layer in selected_layers]

        n_segments = int(stats["head_span_mean_prefill"].shape[2])
        sel = _select_layers(stats, positions)
        for item in segment_items:
            s = int(item["segment_idx"])
            if s >= n_segments:
                raise ValueError(
                    f"{record.iter_dir}: segment index {s} exceeds head_span "
                    f"segment axis {n_segments}"
                )
            base = _trajectory_base(item, record=record)
            base["n_layers"] = len(selected_layers)
            base["layers_used"] = list(selected_layers)
            for phase in ("prefill", "decode"):
                means, variances, kept = _phase_observations(
                    sel, segment_idx=s, phase=phase, decode_reduce=decode_reduce
                )
                cell = reduce_head_span_cell(means, variances)
                trajectory.append(
                    {**base, "phase": phase, "kept_token_count_total": kept, **_cell_columns(cell)}
                )
            for pos, layer in zip(positions, selected_layers):
                layer_sel = _select_layers(stats, [pos])
                for phase in ("prefill", "decode"):
                    means, variances, kept = _phase_observations(
                        layer_sel, segment_idx=s, phase=phase, decode_reduce=decode_reduce
                    )
                    cell = reduce_head_span_cell(means, variances)
                    by_layer.append(
                        {
                            **base,
                            "layer": int(layer),
                            "phase": phase,
                            "kept_token_count_total": kept,
                            **_cell_columns(cell),
                        }
                    )

    return trajectory, by_layer, list(layers_used or [])


def _select_layers(stats: dict[str, np.ndarray], positions: Sequence[int]) -> dict[str, np.ndarray]:
    idx = np.asarray(list(positions), dtype=np.int64)
    return {
        "mean_p": stats["head_span_mean_prefill"][idx].astype(np.float64),
        "var_p": stats["head_span_var_prefill"][idx].astype(np.float64),
        "kept_p": stats["head_span_kept_token_count_prefill"][idx].astype(np.int64),
        "mean_d": stats["head_span_mean_decode"][idx].astype(np.float64),
        "var_d": stats["head_span_var_decode"][idx].astype(np.float64),
        "kept_d": stats["head_span_kept_token_count_decode"][idx].astype(np.int64),
        "step_d": stats["head_span_decode_step"][idx].astype(np.int64),
    }


def _phase_observations(
    sel: dict[str, np.ndarray],
    *,
    segment_idx: int,
    phase: str,
    decode_reduce: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    s = segment_idx
    if phase == "prefill":
        means = sel["mean_p"][:, :, s]
        variances = sel["var_p"][:, :, s]
        kept = int(sel["kept_p"][:, s].sum())
        return means.ravel(), variances.ravel(), kept

    # decode: [L, T, H] for segment s
    mean_d = sel["mean_d"][:, :, :, s]
    var_d = sel["var_d"][:, :, :, s]
    step_d = sel["step_d"]  # [L, T]
    kept_d = sel["kept_d"]  # [L, T, S]
    if mean_d.shape[1] == 0:
        return np.empty(0), np.empty(0), 0
    valid = step_d >= 0  # [L, T]
    kept = int(kept_d[valid][:, s].sum()) if bool(valid.any()) else 0
    if decode_reduce == "pool_steps":
        means = np.where(valid[:, :, None], mean_d, np.nan)
        variances = np.where(valid[:, :, None], var_d, np.nan)
        return means.ravel(), variances.ravel(), kept
    # last_step: one observation per (layer, head) at the latest valid step
    n_layers, _, n_heads = mean_d.shape
    means = np.full((n_layers, n_heads), np.nan, dtype=np.float64)
    variances = np.full((n_layers, n_heads), np.nan, dtype=np.float64)
    for layer_pos in range(n_layers):
        valid_t = np.where(valid[layer_pos])[0]
        if valid_t.size == 0:
            continue
        t_last = int(valid_t[int(np.argmax(step_d[layer_pos, valid_t]))])
        means[layer_pos] = mean_d[layer_pos, t_last]
        variances[layer_pos] = var_d[layer_pos, t_last]
    return means.ravel(), variances.ravel(), kept


def _trajectory_base(item: dict[str, Any], *, record: IterationRecord) -> dict[str, Any]:
    first_seen_call = int(item["first_seen_call"])
    return {
        "segment_ordinal": item["segment_ordinal"],
        "segment_id": item["segment_id"],
        "identity_source": item["identity_source"],
        "task": item["task"],
        "role": item["role"],
        "tool_call_id": item.get("tool_call_id"),
        "tool_name": item.get("tool_name"),
        "first_seen_call": first_seen_call,
        "first_seen_call_inferred": item["first_seen_call_inferred"],
        "message_index": item["message_index"],
        "has_content": item["has_content"],
        "has_tool_calls": item["has_tool_calls"],
        "initial_token_count": item["initial_token_count"],
        "token_count": item["token_count"],
        "observed_call_idx": int(record.call_idx),
        "age": int(record.call_idx) - first_seen_call,
    }


def _cell_columns(cell: dict[str, Any]) -> dict[str, Any]:
    return {
        "within_segment_attention_mean": cell["mean"],
        "within_segment_attention_std": cell["std"],
        "within_segment_attention_var_pooled": cell["var_pooled"],
        "cross_head_std_of_mean": cell["cross_head_std"],
        "n_contributors": cell["n_contributors"],
        "n_nan_contributors": cell["n_nan_contributors"],
    }


def _head_span_summary_rows(trajectory_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in trajectory_rows:
        grouped.setdefault(str(row["segment_id"]), []).append(dict(row))

    summary_rows: list[dict[str, Any]] = []
    for rows in grouped.values():
        first = min(
            rows,
            key=lambda row: (
                str(row["task"]),
                int(row["first_seen_call"]),
                int(row["message_index"]),
            ),
        )
        out = {
            "segment_ordinal": first["segment_ordinal"],
            "segment_id": first["segment_id"],
            "identity_source": first["identity_source"],
            "task": first["task"],
            "role": first["role"],
            "tool_call_id": first.get("tool_call_id"),
            "tool_name": first.get("tool_name"),
            "first_seen_call": first["first_seen_call"],
            "first_seen_call_inferred": first["first_seen_call_inferred"],
            "message_index": first["message_index"],
            "has_content": first["has_content"],
            "has_tool_calls": first["has_tool_calls"],
            "initial_token_count": first["initial_token_count"],
            "max_observed_age": max(int(row["age"]) for row in rows),
        }
        for phase in ("prefill", "decode"):
            phase_rows = [row for row in rows if row["phase"] == phase]
            out.update(_head_span_phase_summary(phase, phase_rows))
        summary_rows.append(out)

    return sorted(
        summary_rows,
        key=lambda row: (
            str(row["task"]),
            int(row["first_seen_call"]),
            int(row["message_index"]),
            str(row["role"]),
        ),
    )


def _head_span_phase_summary(phase: str, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    keys = {
        f"{phase}_observed_calls": len(rows),
        f"mean_{phase}_attention_mean": None,
        f"peak_{phase}_attention_mean": None,
        f"peak_{phase}_mean_age": None,
        f"mean_{phase}_attention_std": None,
        f"peak_{phase}_attention_std": None,
        f"peak_{phase}_std_age": None,
    }
    if not rows:
        return keys
    means = _column(rows, "within_segment_attention_mean")
    stds = _column(rows, "within_segment_attention_std")
    ages = np.asarray([int(row["age"]) for row in rows], dtype=np.float64)
    mean_peak = _nanargmax_or_none(means)
    std_peak = _nanargmax_or_none(stds)
    keys[f"mean_{phase}_attention_mean"] = _nanmean_or_none(means)
    keys[f"mean_{phase}_attention_std"] = _nanmean_or_none(stds)
    if mean_peak is not None:
        keys[f"peak_{phase}_attention_mean"] = float(means[mean_peak])
        keys[f"peak_{phase}_mean_age"] = int(ages[mean_peak])
    if std_peak is not None:
        keys[f"peak_{phase}_attention_std"] = float(stds[std_peak])
        keys[f"peak_{phase}_std_age"] = int(ages[std_peak])
    return keys


def _column(rows: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray(
        [np.nan if row.get(key) is None else float(row[key]) for row in rows],
        dtype=np.float64,
    )


def _plot_head_span_grid(
    trajectory_rows: Sequence[dict[str, Any]],
    summary_rows: Sequence[dict[str, Any]],
    output_stem: Path,
    *,
    layers_used: Sequence[int],
    decode_reduce: str,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    missing_color = "#cfcfcf"
    cell_boundary_color = "#ffffff"
    major_boundary_color = "#222222"
    phases = ("prefill", "decode")
    segment_order = [str(row["segment_id"]) for row in summary_rows]
    segment_to_row = {segment_id: idx for idx, segment_id in enumerate(segment_order)}
    observed_iters = [int(row["observed_call_idx"]) for row in trajectory_rows]
    min_iter = min(observed_iters)
    max_iter = max(observed_iters)
    iter_values = list(range(min_iter, max_iter + 1))
    iter_to_col = {iter_idx: col for col, iter_idx in enumerate(iter_values)}

    mean_matrices = _fill_matrices(
        trajectory_rows,
        segment_to_row=segment_to_row,
        iter_to_col=iter_to_col,
        phases=phases,
        key="within_segment_attention_mean",
    )
    std_matrices = _fill_matrices(
        trajectory_rows,
        segment_to_row=segment_to_row,
        iter_to_col=iter_to_col,
        phases=phases,
        key="within_segment_attention_std",
    )

    labels = [_segment_plot_label(row) for row in summary_rows]
    mean_cmap = LinearSegmentedColormap.from_list(
        "asb_head_span_mean",
        ["#fffaf0", "#fee391", "#fdae61", "#e34a33", "#7f0000"],
    )
    std_cmap = LinearSegmentedColormap.from_list(
        "asb_head_span_std",
        ["#f7fcf0", "#bae4bc", "#7bccc4", "#2b8cbe", "#084081"],
    )
    mean_cmap.set_bad(missing_color)
    std_cmap.set_bad(missing_color)
    mean_vmax = _percentile_vmax(mean_matrices)
    std_vmax = _percentile_vmax(std_matrices)

    width = max(11.0, min(18.0, 6.8 + 0.42 * len(iter_values)))
    height = max(8.5, min(22.0, 2.8 + 0.30 * len(segment_order)))
    fig, axes = plt.subplots(
        2, 2, figsize=(width, height), sharex=True, sharey=True, constrained_layout=False
    )
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(
        left=0.09, right=0.88, bottom=0.075, top=0.955, hspace=0.055, wspace=0.035
    )
    turn_boundaries = _segment_turn_boundaries(summary_rows)
    images = []
    for col, phase in enumerate(phases):
        images.append(
            axes[0, col].imshow(
                mean_matrices[phase], aspect="auto", cmap=mean_cmap, vmin=0.0, vmax=mean_vmax
            )
        )
        images.append(
            axes[1, col].imshow(
                std_matrices[phase], aspect="auto", cmap=std_cmap, vmin=0.0, vmax=std_vmax
            )
        )
        axes[0, col].set_title(f"{phase}: within-segment attention mean")
        axes[1, col].set_title(f"{phase}: within-segment attention std (pooled)")
        axes[1, col].set_xlabel("recording iter / LLM call index")
    for row_idx in range(2):
        axes[row_idx, 0].set_ylabel("segment")
        axes[row_idx, 0].set_yticks(range(len(labels)))
        axes[row_idx, 0].set_yticklabels(labels, fontsize=5.5)
        axes[row_idx, 1].tick_params(axis="y", length=0)
    for ax in axes.ravel():
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", length=0)
        tick_cols = _integer_tick_positions(iter_values, max_labels=32)
        ax.set_xticks(tick_cols)
        ax.set_xticklabels([str(iter_values[col]) for col in tick_cols])
        ax.set_xlim(-0.5, len(iter_values) - 0.5)
        ax.set_facecolor(missing_color)
        ax.set_xticks(np.arange(-0.5, len(iter_values) + 0.5, 1.0), minor=True)
        ax.set_yticks(np.arange(-0.5, len(labels) + 0.5, 1.0), minor=True)
        ax.grid(which="minor", color=cell_boundary_color, linewidth=0.28, alpha=0.72)
        ax.tick_params(which="minor", bottom=False, left=False)
        first_major = min_iter - (min_iter % 5)
        for iter_idx in range(first_major, max_iter + 6, 5):
            if iter_idx < min_iter:
                continue
            ax.axvline(
                iter_to_col.get(iter_idx, len(iter_values)) - 0.5,
                color=major_boundary_color,
                linewidth=0.45,
                alpha=0.45,
            )
        for row_boundary in turn_boundaries:
            ax.axhline(
                row_boundary - 0.5, color=major_boundary_color, linewidth=0.55, alpha=0.60
            )
    cbar_mean = fig.colorbar(images[0], ax=axes[0, :], fraction=0.025, pad=0.012)
    cbar_mean.set_label("mean per-token attention weight in segment")
    cbar_std = fig.colorbar(images[1], ax=axes[1, :], fraction=0.025, pad=0.012)
    cbar_std.set_label("pooled within-segment std = sqrt(mean_{heads,layers} var)")
    fig.text(
        0.01,
        0.018,
        "Columns are recording iters / full LLM-call contexts. Top row = mean "
        "per-token post-softmax attention weight inside the segment (equal-weight "
        f"over all query heads and layers {list(layers_used)}); magnitude is "
        "confounded by segment size and context length. Bottom row = pooled "
        "within-segment std = sqrt(equal-weight mean of per-head per-token "
        "variances); it does NOT include cross-head mean spread (see "
        f"cross_head_std_of_mean in the CSV). Decode reduction: {decode_reduce}. "
        "Gray = no kept key tokens in the segment for any selected layer/head "
        "(not visible yet, or fully evicted under an active sparse/KV policy).",
        fontsize=7,
        color="#555555",
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {
        "grid_png": str(output_stem.with_suffix(".png")),
        "grid_pdf": str(output_stem.with_suffix(".pdf")),
        "n_segments": len(segment_order),
        "x_axis": "recording_iter",
        "min_iter": min_iter,
        "max_iter": max_iter,
        "n_iters": len(iter_values),
        "mean_vmax_percentile_95": mean_vmax,
        "std_vmax_percentile_95": std_vmax,
        "layers_used": list(layers_used),
        "decode_reduce": decode_reduce,
        "missing_color": missing_color,
    }


def _fill_matrices(
    trajectory_rows: Sequence[dict[str, Any]],
    *,
    segment_to_row: dict[str, int],
    iter_to_col: dict[int, int],
    phases: Sequence[str],
    key: str,
) -> dict[str, np.ndarray]:
    """Scatter one metric column into per-phase (segment × iter) matrices."""
    n_rows = len(segment_to_row)
    n_cols = len(iter_to_col)
    matrices = {
        phase: np.full((n_rows, n_cols), np.nan, dtype=np.float64) for phase in phases
    }
    for row in trajectory_rows:
        phase = str(row["phase"])
        if phase not in matrices:
            continue
        row_idx = segment_to_row.get(str(row["segment_id"]))
        col_idx = iter_to_col.get(int(row["observed_call_idx"]))
        if row_idx is None or col_idx is None:
            continue
        value = row.get(key)
        if value is not None:
            matrices[phase][row_idx, col_idx] = float(value)
    return matrices


def _percentile_vmax(matrices: dict[str, np.ndarray]) -> float:
    finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices.values()])
    vmax = float(np.percentile(finite, 95)) if finite.size else 1.0
    return max(vmax, 1e-6)


def _summary_markdown(summary: dict[str, Any], *, decode_reduce: str) -> str:
    lines = [
        "# Within-Segment Attention Mean/STD",
        "",
        f"- Label: `{summary['label']}`",
        f"- Output: `{summary['output_dir']}`",
        f"- Records: `{summary['n_records']}`",
        f"- Segments analyzed: `{summary['n_segments']}`",
        f"- Layers aggregated: `{summary['layers_used']}`",
        f"- Decode reduction: `{decode_reduce}`",
        "",
        "## Figure",
        "",
        "- Top row: within-segment per-token attention MEAN (mean over heads/layers).",
        "- Bottom row: pooled within-segment STD = sqrt(mean over heads/layers of recorded variance).",
        "- Left column: prefill. Right column: decode.",
        "- X-axis: recording iter / LLM call index; each column is one full context.",
        "",
        "## Caveat",
        "",
        "Stats come from sampled query rows (prefill cap = max_prefill_queries) and "
        "only the layers recorded via --per-head-stats-layers. Gray cells mean the "
        "segment had zero kept key tokens for every selected layer/head at that iter "
        "(not visible yet, or fully evicted under an active sparse/KV policy).",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
