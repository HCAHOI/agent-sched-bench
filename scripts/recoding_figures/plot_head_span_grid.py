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
    load_block_head_span_stats,
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
    _safe_div,
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
    parser.add_argument(
        "--mode",
        choices=("segment", "block", "block_segment", "block_position", "auto"),
        default="auto",
        help=(
            "segment = within-segment role buckets (head_span_*). block = "
            "per-selected-block buckets sink|r1..rR|recent (block_span_*, "
            "block_topk decode only). block_segment = selected block_topk "
            "middle tokens mapped back onto segment rows (decode only). "
            "block_position = selected blocks on the absolute KV token axis "
            "(block x call) with segment bands (decode only). auto = "
            "block if block_span_* populated else segment. Default auto."
        ),
    )
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
    parser.add_argument(
        "--per-layer",
        action="store_true",
        help=(
            "Also emit segment_head_span_per_layer.pdf: one page per recorded "
            "layer (same 2x2 grid), to compare the within-segment mean/std grid "
            "across layers. Color scales are per-page (per layer)."
        ),
    )
    parser.add_argument(
        "--per-head",
        action="store_true",
        help=(
            "Also emit a per-head PDF: one page per query head, each page "
            "showing the same grid as the main figure but restricted to that "
            "single head (no cross-head pooling). Works for both segment and "
            "block modes. Color scales are per-page (per head)."
        ),
    )
    args = parser.parse_args()

    layers = _parse_layer_arg(args.layers)
    mode = args.mode
    if mode == "auto":
        mode = "block" if _has_block_span(args.inputs, include_orphans=args.include_orphans, max_iters=args.max_iters) else "segment"
    if mode == "block":
        summary = build_block_head_span_grids(
            inputs=args.inputs,
            output_dir=args.output_dir,
            layers=layers,
            include_orphans=args.include_orphans,
            max_iters=args.max_iters,
            split_by_task=args.split_by_task,
            per_head=args.per_head,
        )
    elif mode == "block_segment":
        summary = build_block_segment_head_span_grids(
            inputs=args.inputs,
            output_dir=args.output_dir,
            layers=layers,
            include_orphans=args.include_orphans,
            max_iters=args.max_iters,
            split_by_task=args.split_by_task,
            per_layer=args.per_layer,
            per_head=args.per_head,
        )
    elif mode == "block_position":
        summary = build_block_position_grids(
            inputs=args.inputs,
            output_dir=args.output_dir,
            layers=layers,
            include_orphans=args.include_orphans,
            max_iters=args.max_iters,
            split_by_task=args.split_by_task,
            per_layer=args.per_layer,
            per_head=args.per_head,
        )
    else:
        summary = build_head_span_segment_grids(
            inputs=args.inputs,
            output_dir=args.output_dir,
            layers=layers,
            include_orphans=args.include_orphans,
            max_iters=args.max_iters,
            split_by_task=args.split_by_task,
            decode_reduce=args.decode_reduce,
            per_layer=args.per_layer,
            per_head=args.per_head,
        )
    print(json.dumps(_json_ready(summary), indent=2, sort_keys=True))


def _has_block_span(
    inputs: Sequence[Path], *, include_orphans: bool, max_iters: int | None
) -> bool:
    """True if any iter has populated block_span_* arrays (non-empty layer axis)."""
    records = load_iteration_records(
        inputs, include_orphans=include_orphans, max_iters=max_iters
    )
    for record in records:
        try:
            stats = load_block_head_span_stats(record.iter_dir)
        except (KeyError, FileNotFoundError, OSError, ValueError):
            # Mixed directory: some iters may pre-date block_span fields, or
            # have a truncated/corrupt npz. Skip rather than treating the entire
            # input as segment-mode.
            continue
        if int(stats["block_span_layers"].size) > 0:
            return True
    return False


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
    per_layer: bool = False,
    per_head: bool = False,
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
        per_layer_pdf = None
        if per_layer:
            per_layer_pdf = _artifact_relative_path(
                _render_per_layer_pdf(
                    group_records,
                    layers_used=layers_used,
                    group_dir=group_dir,
                    decode_reduce=decode_reduce,
                ),
                output_dir,
            )
        per_head_pdf = None
        if per_head:
            per_head_pdf = _artifact_relative_path(
                _render_per_head_pdf_segment(
                    group_records,
                    layers_used=layers_used,
                    group_dir=group_dir,
                    decode_reduce=decode_reduce,
                ),
                output_dir,
            )
        group_summary = {
            "label": label,
            "output_dir": _artifact_relative_path(group_dir, output_dir),
            "runtime_output_dir": str(group_dir),
            "n_records": len(group_records),
            "n_segments": len(summary_rows),
            "n_trajectory_rows": len(trajectory_rows),
            "n_layer_rows": len(layer_rows),
            "layers_used": layers_used,
            "per_layer_pdf": per_layer_pdf,
            "per_head_pdf": per_head_pdf,
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


def _bucket_labels(r_max: int) -> list[str]:
    """Fixed bucket column labels: sink | r1..rR_max | recent."""
    return ["sink", *[f"r{r}" for r in range(1, r_max + 1)], "recent"]


def block_head_span_rows(
    records: Sequence[IterationRecord],
    *,
    layers: Sequence[int] | None = None,
    head: int | None = None,
) -> tuple[list[dict[str, Any]], list[int], int]:
    """Pool per-selected-block decode stats into (layer, bucket) cells.

    Returns (rows, layers_used, r_max). Each row is one (layer, bucket).

    ``head=None`` pools all query heads (default, existing behaviour unchanged).
    ``head=h`` restricts to query head ``h`` only — used by the per-head PDF
    where each page shows one head without cross-head averaging.
    """
    pooled_mean: dict[tuple[int, int], list[np.ndarray]] = {}
    pooled_var: dict[tuple[int, int], list[np.ndarray]] = {}
    kept_total: dict[tuple[int, int], int] = {}
    layers_used: list[int] | None = None
    geom_ref: tuple[int, int, int, int] | None = None  # (block_size, sink, recent, C)
    r_max = 0

    for record in sorted(records, key=lambda item: (item.task, item.call_idx)):
        try:
            stats = load_block_head_span_stats(record.iter_dir)
        except (KeyError, FileNotFoundError, OSError, ValueError):
            # Mixed directory: skip iters predating block_span fields or with a
            # truncated/corrupt npz, mirroring _has_block_span. Newer block
            # recordings still plot; build_block_head_span_grids raises a
            # controlled error if nothing usable remains.
            continue
        available = [int(layer) for layer in stats["block_span_layers"].tolist()]
        if not available:
            continue
        selected_layers = list(layers) if layers is not None else list(available)
        missing = [layer for layer in selected_layers if layer not in available]
        if missing:
            raise ValueError(
                f"{record.iter_dir}: requested layers not recorded in block_span: "
                f"{missing} (available: {available})"
            )
        if layers_used is None:
            layers_used = selected_layers
        elif selected_layers != layers_used:
            raise ValueError(
                f"{record.iter_dir}: block_span layer set {selected_layers} differs "
                f"from earlier iters {layers_used}; recordings are inconsistent"
            )
        mean_d = stats["block_span_mean_decode"].astype(np.float64)  # [L,T,H,C]
        var_d = stats["block_span_var_decode"].astype(np.float64)
        step_d = stats["block_span_decode_step"]  # [L,T]
        kept_d = stats["block_span_kept_token_count_decode"]  # [L,T,C]
        C = int(mean_d.shape[3]) if mean_d.ndim == 4 else 0
        # Bucket columns are pooled by raw index and the `recent` bucket sits at
        # the last column (C-1). Pooling across recordings with different block
        # geometry would misalign `recent` (and rank spans) under a wrong label,
        # silently corrupting the grid — so require identical geometry instead of
        # max()-ing r_max across heterogeneous layouts.
        geom = (
            int(stats["block_span_block_size"]),
            int(stats["block_span_sink_size"]),
            int(stats["block_span_recent_window"]),
            C,
        )
        if geom_ref is None:
            geom_ref = geom
            r_max = C - 2 if C >= 2 else 0
        elif geom != geom_ref:
            raise ValueError(
                f"{record.iter_dir}: block_span geometry (block_size, sink, "
                f"recent, C)={geom} differs from earlier iters {geom_ref}; "
                "pooling across incompatible bucket layouts would misalign columns"
            )
        for layer in selected_layers:
            li = available.index(layer)
            valid = step_d[li] >= 0  # [T]
            if not bool(valid.any()):
                continue
            for col in range(C):
                key = (int(layer), col)
                if head is None:
                    # pool all heads: [T_valid, H] -> ravel
                    obs_mean = mean_d[li, valid, :, col].ravel()
                    obs_var = var_d[li, valid, :, col].ravel()
                else:
                    # single head: [T_valid] (head slice keeps 1-D)
                    obs_mean = mean_d[li, valid, head, col]
                    obs_var = var_d[li, valid, head, col]
                pooled_mean.setdefault(key, []).append(obs_mean)
                pooled_var.setdefault(key, []).append(obs_var)
                kept_total[key] = kept_total.get(key, 0) + int(kept_d[li, valid, col].sum())

    used = list(layers_used or [])
    rows: list[dict[str, Any]] = []
    for layer in used:
        for col in range(r_max + 2):
            key = (int(layer), col)
            means = (
                np.concatenate(pooled_mean[key]) if key in pooled_mean else np.empty(0)
            )
            variances = (
                np.concatenate(pooled_var[key]) if key in pooled_var else np.empty(0)
            )
            cell = reduce_head_span_cell(means, variances)
            rows.append(
                {
                    "layer": int(layer),
                    "bucket_col": col,
                    "bucket_label": _bucket_labels(r_max)[col],
                    "kept_token_count_total": int(kept_total.get(key, 0)),
                    **_cell_columns(cell),
                }
            )
    return rows, used, r_max


def build_block_head_span_grids(
    *,
    inputs: Sequence[Path],
    output_dir: Path,
    layers: Sequence[int] | None = None,
    include_orphans: bool = False,
    max_iters: int | None = None,
    split_by_task: bool = True,
    per_head: bool = False,
) -> dict[str, Any]:
    """Build per-selected-block within-block mean/std grids (block_topk decode)."""
    records = load_iteration_records(
        inputs, include_orphans=include_orphans, max_iters=max_iters
    )
    attempt_dirs = find_attempt_dirs(inputs)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    r_max_global: int | None = None
    layers_used_global: list[int] | None = None
    for label, group_records, group_dir in groups:
        rows, layers_used, r_max = block_head_span_rows(group_records, layers=layers)
        if not rows:
            raise ValueError(f"{label}: no block_span observations were found")
        if layers_used_global is None:
            layers_used_global = layers_used
            r_max_global = r_max
        group_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(group_dir / "block_head_span_by_layer.csv", rows)
        plot_summary = _plot_block_head_span_grid(
            rows,
            group_dir / "block_head_span_grid",
            layers_used=layers_used,
            r_max=r_max,
        )
        plot_summary = _portable_plot_summary(plot_summary, artifact_root=output_dir)
        per_head_pdf = None
        if per_head:
            per_head_pdf = _artifact_relative_path(
                _render_per_head_pdf_block(
                    group_records,
                    layers_used=layers_used,
                    group_dir=group_dir,
                    r_max=r_max,
                ),
                output_dir,
            )
        group_summary = {
            "label": label,
            "output_dir": _artifact_relative_path(group_dir, output_dir),
            "runtime_output_dir": str(group_dir),
            "n_records": len(group_records),
            "n_layers": len(layers_used),
            "r_max": r_max,
            "layers_used": layers_used,
            "per_head_pdf": per_head_pdf,
            "plot": plot_summary,
        }
        (group_dir / "summary.json").write_text(
            json.dumps(_json_ready(group_summary), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        group_summaries.append(group_summary)

    run_summary = {
        "inputs": [str(path) for path in inputs],
        "attempt_dirs": [str(path) for path in attempt_dirs],
        "metric": "within_block_attention_mean_std",
        "mode": "block",
        "layers_used": layers_used_global,
        "r_max": r_max_global,
        "split_by_task": split_by_task,
        "groups": group_summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_json_ready(run_summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return run_summary


def build_block_segment_head_span_grids(
    *,
    inputs: Sequence[Path],
    output_dir: Path,
    layers: Sequence[int] | None = None,
    include_orphans: bool = False,
    max_iters: int | None = None,
    split_by_task: bool = True,
    per_layer: bool = False,
    per_head: bool = False,
) -> dict[str, Any]:
    """Build segment grids for block_topk-selected middle tokens.

    Unlike ``build_block_head_span_grids``, the y-axis remains the prompt
    segment axis. The block_topk-selected middle tokens are attributed back to
    the segments they occupy, so the figure answers "which segments would the
    selected blocks cover?" while retaining the per-head attention statistics.
    """
    records = load_iteration_records(
        inputs, include_orphans=include_orphans, max_iters=max_iters
    )
    attempt_dirs = find_attempt_dirs(inputs)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        trajectory_rows, layer_rows, layers_used = block_segment_head_span_rows(
            group_records,
            layers=layers,
        )
        if not trajectory_rows:
            raise ValueError(
                f"{label}: no block-segment observations were found"
            )
        if layers_used_global is None:
            layers_used_global = layers_used
        elif layers_used != layers_used_global:
            raise ValueError(
                f"{label}: layer set {layers_used} differs from other groups "
                f"{layers_used_global}; recordings are inconsistent"
            )
        summary_rows = _block_segment_summary_rows(trajectory_rows)
        group_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(
            group_dir / "block_segment_head_span_trajectory.csv",
            trajectory_rows,
        )
        _write_csv(group_dir / "block_segment_head_span_by_layer.csv", layer_rows)
        _write_csv(
            group_dir / "block_segment_head_span_summary.csv",
            summary_rows,
        )
        plot_summary = _plot_block_segment_head_span_grid(
            trajectory_rows,
            summary_rows,
            group_dir / "block_segment_head_span_grid",
            layers_used=layers_used,
        )
        plot_summary = _portable_plot_summary(plot_summary, artifact_root=output_dir)
        per_layer_pdf = None
        if per_layer:
            per_layer_pdf = _artifact_relative_path(
                _render_per_layer_pdf_block_segment(
                    group_records,
                    layers_used=layers_used,
                    group_dir=group_dir,
                ),
                output_dir,
            )
        per_head_pdf = None
        if per_head:
            per_head_pdf = _artifact_relative_path(
                _render_per_head_pdf_block_segment(
                    group_records,
                    layers_used=layers_used,
                    group_dir=group_dir,
                ),
                output_dir,
            )
        group_summary = {
            "label": label,
            "output_dir": _artifact_relative_path(group_dir, output_dir),
            "runtime_output_dir": str(group_dir),
            "n_records": len(group_records),
            "n_segments": len(summary_rows),
            "n_trajectory_rows": len(trajectory_rows),
            "n_layer_rows": len(layer_rows),
            "layers_used": layers_used,
            "per_layer_pdf": per_layer_pdf,
            "per_head_pdf": per_head_pdf,
            "role_counts": _role_counts(summary_rows),
            "plot": plot_summary,
        }
        (group_dir / "summary.json").write_text(
            json.dumps(_json_ready(group_summary), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (group_dir / "summary.md").write_text(
            _block_segment_summary_markdown(group_summary),
            encoding="utf-8",
        )
        group_summaries.append(group_summary)

    run_summary = {
        "inputs": [str(path) for path in inputs],
        "attempt_dirs": [str(path) for path in attempt_dirs],
        "metric": "block_topk_selected_middle_tokens_by_segment",
        "mode": "block_segment",
        "layers_used": layers_used_global,
        "split_by_task": split_by_task,
        "groups": group_summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_json_ready(run_summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return run_summary


def block_segment_head_span_rows(
    records: Sequence[IterationRecord],
    *,
    layers: Sequence[int] | None = None,
    head: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    """Return segment rows using block_topk-selected middle-token stats.

    ``head=None`` pools all query heads for attention mean/std. Selection share
    is head-independent because the recorded block_topk keep set is shared
    across heads for a layer/step.
    """
    trajectory: list[dict[str, Any]] = []
    by_layer: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    layers_used: list[int] | None = None

    for record in sorted(records, key=lambda item: (item.task, item.call_idx)):
        segments_payload = _load_json_required(record.iter_dir / "segments.json")
        segments = list(segments_payload.get("segments", []))
        segment_items = _segments_for_record(
            segments,
            record=record,
            metadata=metadata,
        )
        if not segment_items:
            continue

        try:
            stats = load_block_head_span_stats(record.iter_dir)
        except (KeyError, FileNotFoundError, OSError, ValueError) as exc:
            raise ValueError(
                f"{record.iter_dir}: missing block-segment stats; re-record with "
                "--per-head-block-stats under block_topk"
            ) from exc
        available = [int(layer) for layer in stats["block_span_layers"].tolist()]
        if not available:
            raise ValueError(
                f"{record.iter_dir}: attention.npz has no block_span layers"
            )
        selected_layers = list(layers) if layers is not None else list(available)
        missing = [layer for layer in selected_layers if layer not in available]
        if missing:
            raise ValueError(
                f"{record.iter_dir}: requested layers not recorded in block_span: "
                f"{missing} (available: {available})"
            )
        if layers_used is None:
            layers_used = selected_layers
        elif selected_layers != layers_used:
            raise ValueError(
                f"{record.iter_dir}: block_span layer set {selected_layers} differs "
                f"from earlier iters {layers_used}; recordings are inconsistent"
            )
        positions = [available.index(layer) for layer in selected_layers]
        sel = _select_block_segment_layers(stats, positions)

        n_segments = int(sel["seg_mean_d"].shape[3]) if sel["seg_mean_d"].ndim == 4 else 0
        for item in segment_items:
            s = int(item["segment_idx"])
            if s >= n_segments:
                raise ValueError(
                    f"{record.iter_dir}: segment index {s} exceeds block_span "
                    f"segment axis {n_segments}"
                )
            base = _trajectory_base(item, record=record)
            base["n_layers"] = len(selected_layers)
            base["layers_used"] = list(selected_layers)
            means, variances, kept, total_kept, valid_steps = (
                _block_segment_decode_observations(sel, segment_idx=s, head=head)
            )
            cell = reduce_head_span_cell(means, variances)
            trajectory.append(
                {
                    **base,
                    "phase": "decode",
                    "selected_block_token_count_total": kept,
                    "selected_block_token_count_denominator": total_kept,
                    "selected_block_token_share": _safe_div(kept, total_kept),
                    "valid_layer_decode_steps": valid_steps,
                    "kept_token_count_total": kept,
                    **_cell_columns(cell),
                }
            )

            for pos, layer in enumerate(selected_layers):
                layer_sel = _select_block_segment_layer_position(sel, pos)
                means, variances, kept, total_kept, valid_steps = (
                    _block_segment_decode_observations(
                        layer_sel, segment_idx=s, head=head
                    )
                )
                cell = reduce_head_span_cell(means, variances)
                by_layer.append(
                    {
                        **base,
                        "layer": int(layer),
                        "phase": "decode",
                        "selected_block_token_count_total": kept,
                        "selected_block_token_count_denominator": total_kept,
                        "selected_block_token_share": _safe_div(kept, total_kept),
                        "valid_layer_decode_steps": valid_steps,
                        "kept_token_count_total": kept,
                        **_cell_columns(cell),
                    }
                )

    return trajectory, by_layer, list(layers_used or [])


def _select_block_segment_layers(
    stats: dict[str, np.ndarray],
    positions: Sequence[int],
) -> dict[str, np.ndarray]:
    idx = np.asarray(list(positions), dtype=np.int64)
    return {
        "seg_mean_d": stats["block_span_seg_mean_decode"][idx].astype(np.float64),
        "seg_var_d": stats["block_span_seg_var_decode"][idx].astype(np.float64),
        "seg_kept_d": stats["block_span_seg_kept_token_count_decode"][idx].astype(np.int64),
        "step_d": stats["block_span_decode_step"][idx].astype(np.int64),
    }


def _select_block_segment_layer_position(
    sel: dict[str, np.ndarray],
    position: int,
) -> dict[str, np.ndarray]:
    return {
        "seg_mean_d": sel["seg_mean_d"][position:position + 1],
        "seg_var_d": sel["seg_var_d"][position:position + 1],
        "seg_kept_d": sel["seg_kept_d"][position:position + 1],
        "step_d": sel["step_d"][position:position + 1],
    }


def _block_segment_decode_observations(
    sel: dict[str, np.ndarray],
    *,
    segment_idx: int,
    head: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Extract selected-block observations for one segment in decode."""
    s = segment_idx
    if head is None:
        mean_d = sel["seg_mean_d"][:, :, :, s]  # [L, T, H]
        var_d = sel["seg_var_d"][:, :, :, s]
    else:
        mean_d = sel["seg_mean_d"][:, :, head:head + 1, s]  # [L, T, 1]
        var_d = sel["seg_var_d"][:, :, head:head + 1, s]
    step_d = sel["step_d"]
    kept_d = sel["seg_kept_d"]  # [L, T, S]
    if mean_d.shape[1] == 0:
        return np.empty(0), np.empty(0), 0, 0, 0
    valid = step_d >= 0
    if not bool(valid.any()):
        return np.empty(0), np.empty(0), 0, 0, 0
    means = np.where(valid[:, :, None], mean_d, np.nan)
    variances = np.where(valid[:, :, None], var_d, np.nan)
    kept = int(kept_d[:, :, s][valid].sum())
    total_kept = int(kept_d[valid].sum())
    return means.ravel(), variances.ravel(), kept, total_kept, int(valid.sum())


def _plot_block_head_span_grid(
    rows: Sequence[dict[str, Any]],
    output_stem: Path,
    *,
    layers_used: Sequence[int],
    r_max: int,
) -> dict[str, Any]:
    """Render a layer x bucket grid: top = within-block mean, bottom = std."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    labels = _bucket_labels(r_max)
    n_buckets = len(labels)
    layer_to_row = {int(layer): idx for idx, layer in enumerate(layers_used)}
    n_layers = len(layers_used)

    mean_mat = np.full((n_layers, n_buckets), np.nan, dtype=np.float64)
    std_mat = np.full((n_layers, n_buckets), np.nan, dtype=np.float64)
    for row in rows:
        li = layer_to_row.get(int(row["layer"]))
        col = int(row["bucket_col"])
        if li is None or col >= n_buckets:
            continue
        if row["within_segment_attention_mean"] is not None:
            mean_mat[li, col] = float(row["within_segment_attention_mean"])
        if row["within_segment_attention_std"] is not None:
            std_mat[li, col] = float(row["within_segment_attention_std"])

    # Drop bucket columns with no data in any layer (e.g. ranks never selected).
    have_data = ~np.all(np.isnan(mean_mat), axis=0)
    keep_cols = [c for c in range(n_buckets) if have_data[c]]
    if keep_cols:
        mean_mat = mean_mat[:, keep_cols]
        std_mat = std_mat[:, keep_cols]
        labels = [labels[c] for c in keep_cols]

    missing_color = "#cfcfcf"
    mean_cmap = LinearSegmentedColormap.from_list(
        "asb_block_mean", ["#fffaf0", "#fee391", "#fdae61", "#e34a33", "#7f0000"]
    )
    std_cmap = LinearSegmentedColormap.from_list(
        "asb_block_std", ["#f7fcf0", "#bae4bc", "#7bccc4", "#2b8cbe", "#084081"]
    )
    mean_cmap.set_bad(missing_color)
    std_cmap.set_bad(missing_color)

    def _vmax(mat: np.ndarray) -> float:
        finite = mat[np.isfinite(mat)]
        return max(float(np.percentile(finite, 95)) if finite.size else 1.0, 1e-6)

    mean_vmax = _vmax(mean_mat)
    std_vmax = _vmax(std_mat)

    width = max(6.0, min(16.0, 2.5 + 0.5 * len(labels)))
    height = max(4.0, min(18.0, 2.0 + 0.45 * max(1, n_layers)))
    fig, axes = plt.subplots(2, 1, figsize=(width, height), sharex=True)
    fig.patch.set_facecolor("white")
    ylabels = [str(int(layer)) for layer in layers_used]
    for ax, mat, cmap, vmax, title in (
        (axes[0], mean_mat, mean_cmap, mean_vmax, "within-block attention mean"),
        (axes[1], std_mat, std_cmap, std_vmax, "within-block attention std (pooled)"),
    ):
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.set_ylabel("layer")
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7, rotation=90)
        ax.set_facecolor(missing_color)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    axes[1].set_xlabel("bucket (sink | selection rank | recent)")
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {
        "grid_png": str(output_stem.with_suffix(".png")),
        "grid_pdf": str(output_stem.with_suffix(".pdf")),
        "n_layers": n_layers,
        "buckets": labels,
        "mean_vmax_percentile_95": mean_vmax,
        "std_vmax_percentile_95": std_vmax,
        "missing_color": missing_color,
    }


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
    head: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    """Return trajectory rows, per-layer rows, and the effective layer list.

    ``head=None`` pools all query heads (default, existing behaviour unchanged).
    ``head=h`` restricts observations to query head ``h`` only; useful for the
    per-head PDF where each page shows one head without cross-head averaging.
    """
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
                    sel, segment_idx=s, phase=phase, decode_reduce=decode_reduce,
                    head=head,
                )
                cell = reduce_head_span_cell(means, variances)
                trajectory.append(
                    {**base, "phase": phase, "kept_token_count_total": kept, **_cell_columns(cell)}
                )
            for pos, layer in zip(positions, selected_layers):
                layer_sel = _select_layers(stats, [pos])
                for phase in ("prefill", "decode"):
                    means, variances, kept = _phase_observations(
                        layer_sel, segment_idx=s, phase=phase, decode_reduce=decode_reduce,
                        head=head,
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
    head: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract mean/var observations for one (segment, phase).

    ``head=None`` pools all query heads (existing behaviour, unchanged).
    ``head=h`` selects a single query head; the returned arrays are the same
    shape as the pooled case except the head axis has size 1, so
    ``reduce_head_span_cell`` receives per-step/layer observations for that
    head alone.
    """
    s = segment_idx
    if phase == "prefill":
        # mean_p / var_p: [L, H, S]
        if head is None:
            means = sel["mean_p"][:, :, s]       # [L, H]
            variances = sel["var_p"][:, :, s]
        else:
            means = sel["mean_p"][:, head:head + 1, s]     # [L, 1]
            variances = sel["var_p"][:, head:head + 1, s]
        kept = int(sel["kept_p"][:, s].sum())
        return means.ravel(), variances.ravel(), kept

    # decode: mean_d / var_d: [L, T, H, S]
    if head is None:
        mean_d = sel["mean_d"][:, :, :, s]   # [L, T, H]
        var_d = sel["var_d"][:, :, :, s]
    else:
        mean_d = sel["mean_d"][:, :, head:head + 1, s]  # [L, T, 1]
        var_d = sel["var_d"][:, :, head:head + 1, s]
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
    # last_step: one observation per (layer, [head]) at the latest valid step
    n_layers, _, n_h = mean_d.shape
    means = np.full((n_layers, n_h), np.nan, dtype=np.float64)
    variances = np.full((n_layers, n_h), np.nan, dtype=np.float64)
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


def _render_per_layer_pdf(
    group_records,
    *,
    layers_used: Sequence[int],
    group_dir: Path,
    decode_reduce: str,
) -> Path:
    """Render one within-segment grid per layer into a multi-page PDF.

    One page per recorded layer, same 2x2 layout, so layers can be compared by
    flipping pages. Reuses the standard single-layer render (layers=[L]) then
    stitches the PNGs into a multi-page PDF. Color scales are per-page (per
    layer) — compare patterns across pages; use the per-layer CV/summary CSVs
    for absolute magnitude.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    per_layer_dir = group_dir / "per_layer"
    pages: list[tuple[int, Path]] = []
    for layer in layers_used:
        rows, _layer_rows, _used = head_span_segment_rows(
            group_records, layers=[layer], decode_reduce=decode_reduce
        )
        if not rows:
            continue
        summary = _head_span_summary_rows(rows)
        stem = per_layer_dir / f"layer_{int(layer):02d}" / "segment_head_span_grid"
        _plot_head_span_grid(rows, summary, stem, layers_used=[layer], decode_reduce=decode_reduce)
        pages.append((int(layer), stem.with_suffix(".png")))

    pdf_path = group_dir / "segment_head_span_per_layer.pdf"
    with PdfPages(pdf_path) as pdf:
        for layer, png in pages:
            img = plt.imread(png)
            height, width = img.shape[0], img.shape[1]
            fig = plt.figure(figsize=(width / 150.0, height / 150.0 + 0.4))
            ax = fig.add_axes([0, 0, 1, 0.96])
            ax.imshow(img)
            ax.axis("off")
            fig.suptitle(
                f"layer {layer} - within-segment mean (top) / std (bottom), prefill | decode",
                fontsize=11,
                y=0.995,
            )
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
    return pdf_path


def _n_query_heads_segment(records: Sequence[IterationRecord]) -> int:
    """Read query-head count from the first available npz (head_span shape[1])."""
    for record in records:
        try:
            stats = load_head_span_stats(record.iter_dir)
        except (KeyError, FileNotFoundError, OSError):
            continue
        # head_span_mean_prefill: [L_s, query_head, S]
        h = int(stats["head_span_mean_prefill"].shape[1])
        if h > 0:
            return h
    return 0


def _n_query_heads_block(records: Sequence[IterationRecord]) -> int:
    """Read query-head count from the first available block npz (block_span shape[2])."""
    for record in records:
        try:
            stats = load_block_head_span_stats(record.iter_dir)
        except (KeyError, FileNotFoundError, OSError, ValueError):
            continue
        # block_span_mean_decode: [L_s, T_max, query_head, C]
        arr = stats["block_span_mean_decode"]
        if arr.ndim == 4:
            h = int(arr.shape[2])
            if h > 0:
                return h
    return 0


def _render_per_head_pdf_segment(
    group_records: Sequence[IterationRecord],
    *,
    layers_used: Sequence[int],
    group_dir: Path,
    decode_reduce: str,
) -> Path:
    """Render one within-segment grid per query head into a multi-page PDF.

    One page per query head, same 2×2 layout as the main segment grid but
    restricted to that head (no cross-head pooling). Mirrors
    ``_render_per_layer_pdf``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    n_heads = _n_query_heads_segment(group_records)
    per_head_dir = group_dir / "per_head"
    pages: list[tuple[int, Path]] = []
    for h in range(n_heads):
        rows, _layer_rows, _used = head_span_segment_rows(
            group_records, layers=list(layers_used), decode_reduce=decode_reduce, head=h
        )
        if not rows:
            continue
        summary = _head_span_summary_rows(rows)
        stem = per_head_dir / f"head_{h:02d}" / "segment_head_span_grid"
        _plot_head_span_grid(rows, summary, stem, layers_used=layers_used, decode_reduce=decode_reduce)
        pages.append((h, stem.with_suffix(".png")))

    pdf_path = group_dir / "segment_head_span_per_head.pdf"
    with PdfPages(pdf_path) as pdf:
        for h, png in pages:
            img = plt.imread(png)
            height, width = img.shape[0], img.shape[1]
            fig = plt.figure(figsize=(width / 150.0, height / 150.0 + 0.4))
            ax = fig.add_axes([0, 0, 1, 0.96])
            ax.imshow(img)
            ax.axis("off")
            fig.suptitle(
                f"head {h:02d} - within-segment mean (top) / std (bottom), prefill | decode",
                fontsize=11,
                y=0.995,
            )
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
    return pdf_path


def _render_per_head_pdf_block(
    group_records: Sequence[IterationRecord],
    *,
    layers_used: Sequence[int],
    group_dir: Path,
    r_max: int,
) -> Path:
    """Render one block-span grid per query head into a multi-page PDF."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    n_heads = _n_query_heads_block(group_records)
    per_head_dir = group_dir / "per_head"
    pages: list[tuple[int, Path]] = []
    for h in range(n_heads):
        rows, _used, _rm = block_head_span_rows(
            group_records, layers=list(layers_used), head=h
        )
        if not rows:
            continue
        stem = per_head_dir / f"head_{h:02d}" / "block_head_span_grid"
        _plot_block_head_span_grid(rows, stem, layers_used=layers_used, r_max=r_max)
        pages.append((h, stem.with_suffix(".png")))

    pdf_path = group_dir / "block_head_span_per_head.pdf"
    with PdfPages(pdf_path) as pdf:
        for h, png in pages:
            img = plt.imread(png)
            height, width = img.shape[0], img.shape[1]
            fig = plt.figure(figsize=(width / 150.0, height / 150.0 + 0.4))
            ax = fig.add_axes([0, 0, 1, 0.96])
            ax.imshow(img)
            ax.axis("off")
            fig.suptitle(
                f"head {h:02d} - within-block mean (top) / std (bottom)",
                fontsize=11,
                y=0.995,
            )
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
    return pdf_path


def _block_segment_summary_rows(
    trajectory_rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
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
        decode_rows = [row for row in rows if row["phase"] == "decode"]
        out.update(_head_span_phase_summary("decode", decode_rows))
        shares = _column(decode_rows, "selected_block_token_share")
        share_peak = _nanargmax_or_none(shares)
        ages = np.asarray([int(row["age"]) for row in decode_rows], dtype=np.float64)
        out["mean_decode_selected_block_token_share"] = _nanmean_or_none(shares)
        out["peak_decode_selected_block_token_share"] = (
            float(shares[share_peak]) if share_peak is not None else None
        )
        out["peak_decode_selected_block_token_share_age"] = (
            int(ages[share_peak]) if share_peak is not None else None
        )
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


def _render_per_layer_pdf_block_segment(
    group_records: Sequence[IterationRecord],
    *,
    layers_used: Sequence[int],
    group_dir: Path,
) -> Path:
    """Render one block-segment grid per recorded layer into a PDF."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    per_layer_dir = group_dir / "per_layer"
    pages: list[tuple[int, Path]] = []
    for layer in layers_used:
        rows, _layer_rows, _used = block_segment_head_span_rows(
            group_records, layers=[layer]
        )
        if not rows:
            continue
        summary = _block_segment_summary_rows(rows)
        stem = per_layer_dir / f"layer_{int(layer):02d}" / "block_segment_head_span_grid"
        _plot_block_segment_head_span_grid(rows, summary, stem, layers_used=[layer])
        pages.append((int(layer), stem.with_suffix(".png")))

    pdf_path = group_dir / "block_segment_head_span_per_layer.pdf"
    with PdfPages(pdf_path) as pdf:
        for layer, png in pages:
            img = plt.imread(png)
            height, width = img.shape[0], img.shape[1]
            fig = plt.figure(figsize=(width / 150.0, height / 150.0 + 0.4))
            ax = fig.add_axes([0, 0, 1, 0.96])
            ax.imshow(img)
            ax.axis("off")
            fig.suptitle(
                f"layer {layer} - block_topk selected tokens mapped to segments",
                fontsize=11,
                y=0.995,
            )
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
    return pdf_path


def _render_per_head_pdf_block_segment(
    group_records: Sequence[IterationRecord],
    *,
    layers_used: Sequence[int],
    group_dir: Path,
) -> Path:
    """Render one block-segment grid per query head into a PDF."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    n_heads = _n_query_heads_block(group_records)
    per_head_dir = group_dir / "per_head"
    pages: list[tuple[int, Path]] = []
    for h in range(n_heads):
        rows, _layer_rows, _used = block_segment_head_span_rows(
            group_records, layers=list(layers_used), head=h
        )
        if not rows:
            continue
        summary = _block_segment_summary_rows(rows)
        stem = per_head_dir / f"head_{h:02d}" / "block_segment_head_span_grid"
        _plot_block_segment_head_span_grid(rows, summary, stem, layers_used=layers_used)
        pages.append((h, stem.with_suffix(".png")))

    pdf_path = group_dir / "block_segment_head_span_per_head.pdf"
    with PdfPages(pdf_path) as pdf:
        for h, png in pages:
            img = plt.imread(png)
            height, width = img.shape[0], img.shape[1]
            fig = plt.figure(figsize=(width / 150.0, height / 150.0 + 0.4))
            ax = fig.add_axes([0, 0, 1, 0.96])
            ax.imshow(img)
            ax.axis("off")
            fig.suptitle(
                f"head {h:02d} - block_topk selected tokens mapped to segments",
                fontsize=11,
                y=0.995,
            )
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
    return pdf_path


def _plot_block_segment_head_span_grid(
    trajectory_rows: Sequence[dict[str, Any]],
    summary_rows: Sequence[dict[str, Any]],
    output_stem: Path,
    *,
    layers_used: Sequence[int],
) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    missing_color = "#cfcfcf"
    cell_boundary_color = "#ffffff"
    major_boundary_color = "#222222"
    segment_order = [str(row["segment_id"]) for row in summary_rows]
    segment_to_row = {segment_id: idx for idx, segment_id in enumerate(segment_order)}
    observed_iters = [int(row["observed_call_idx"]) for row in trajectory_rows]
    min_iter = min(observed_iters)
    max_iter = max(observed_iters)
    iter_values = list(range(min_iter, max_iter + 1))
    iter_to_col = {iter_idx: col for col, iter_idx in enumerate(iter_values)}

    share_matrix = _fill_block_segment_matrix(
        trajectory_rows,
        segment_to_row=segment_to_row,
        iter_to_col=iter_to_col,
        key="selected_block_token_share",
    )
    mean_matrix = _fill_block_segment_matrix(
        trajectory_rows,
        segment_to_row=segment_to_row,
        iter_to_col=iter_to_col,
        key="within_segment_attention_mean",
    )
    std_matrix = _fill_block_segment_matrix(
        trajectory_rows,
        segment_to_row=segment_to_row,
        iter_to_col=iter_to_col,
        key="within_segment_attention_std",
    )

    labels = [_segment_plot_label(row) for row in summary_rows]
    share_cmap = LinearSegmentedColormap.from_list(
        "asb_block_segment_share",
        ["#f7fbff", "#deebf7", "#9ecae1", "#3182bd", "#08519c"],
    )
    mean_cmap = LinearSegmentedColormap.from_list(
        "asb_block_segment_mean",
        ["#fffaf0", "#fee391", "#fdae61", "#e34a33", "#7f0000"],
    )
    std_cmap = LinearSegmentedColormap.from_list(
        "asb_block_segment_std",
        ["#f7fcf0", "#bae4bc", "#7bccc4", "#2b8cbe", "#084081"],
    )
    for cmap in (share_cmap, mean_cmap, std_cmap):
        cmap.set_bad(missing_color)
    share_vmax = _percentile_vmax({"decode": share_matrix})
    mean_vmax = _percentile_vmax({"decode": mean_matrix})
    std_vmax = _percentile_vmax({"decode": std_matrix})

    width = max(11.0, min(18.0, 6.8 + 0.42 * len(iter_values)))
    height = max(8.5, min(24.0, 3.2 + 0.34 * len(segment_order)))
    fig, axes = plt.subplots(
        3, 1, figsize=(width, height), sharex=True, sharey=True, constrained_layout=False
    )
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(
        left=0.13, right=0.88, bottom=0.085, top=0.965, hspace=0.08
    )
    panels = [
        (
            axes[0],
            share_matrix,
            share_cmap,
            0.0,
            share_vmax,
            "decode: selected block token share by segment",
            "share of selected middle tokens",
        ),
        (
            axes[1],
            mean_matrix,
            mean_cmap,
            0.0,
            mean_vmax,
            "decode: attention mean inside selected block tokens",
            "mean per-token attention weight",
        ),
        (
            axes[2],
            std_matrix,
            std_cmap,
            0.0,
            std_vmax,
            "decode: attention std inside selected block tokens",
            "pooled within-segment std",
        ),
    ]
    turn_boundaries = _segment_turn_boundaries(summary_rows)
    for ax, matrix, cmap, vmin, vmax, title, cbar_label in panels:
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_ylabel("segment")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=5.5)
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
                row_boundary - 0.5,
                color=major_boundary_color,
                linewidth=0.55,
                alpha=0.60,
            )
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.012)
        cbar.set_label(cbar_label)
    axes[-1].set_xlabel("recording iter / LLM call index")
    fig.text(
        0.01,
        0.022,
        "Decode-only block_topk diagnostic. Top row maps selected middle-token "
        "blocks back onto prompt segments: each cell is selected tokens in that "
        "segment divided by all selected middle tokens for the aggregated "
        f"layer/step records (layers {list(layers_used)}). Middle/bottom rows "
        "show attention mean/std over those selected tokens; gray means no "
        "selected block token from that segment.",
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
        "selected_share_vmax_percentile_95": share_vmax,
        "mean_vmax_percentile_95": mean_vmax,
        "std_vmax_percentile_95": std_vmax,
        "layers_used": list(layers_used),
        "missing_color": missing_color,
    }


def _fill_block_segment_matrix(
    trajectory_rows: Sequence[dict[str, Any]],
    *,
    segment_to_row: dict[str, int],
    iter_to_col: dict[int, int],
    key: str,
) -> np.ndarray:
    matrix = np.full((len(segment_to_row), len(iter_to_col)), np.nan, dtype=np.float64)
    for row in trajectory_rows:
        if str(row["phase"]) != "decode":
            continue
        row_idx = segment_to_row.get(str(row["segment_id"]))
        col_idx = iter_to_col.get(int(row["observed_call_idx"]))
        if row_idx is None or col_idx is None:
            continue
        value = row.get(key)
        if value is not None:
            matrix[row_idx, col_idx] = float(value)
    return matrix


def _block_segment_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# BlockTopK Selected Blocks by Segment",
        "",
        f"- Label: `{summary['label']}`",
        f"- Output: `{summary['output_dir']}`",
        f"- Records: `{summary['n_records']}`",
        f"- Segments analyzed: `{summary['n_segments']}`",
        f"- Layers aggregated: `{summary['layers_used']}`",
        "",
        "## Figure",
        "",
        "- Top row: share of block_topk selected middle tokens in each segment.",
        "- Middle row: attention mean over those selected tokens.",
        "- Bottom row: pooled attention std over those selected tokens.",
        "- X-axis: recording iter / LLM call index; block_topk selection is decode-only.",
        "",
    ]
    return "\n".join(lines)


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


# Role -> gutter color for segment bands in the block_position figure. Mirrors
# the canonical role set in recording_loader.ROLE_ORDER; "other" is the fallback.
_ROLE_COLORS = {
    "system": "#8c6bb1",
    "user": "#41ab5d",
    "assistant_message": "#fb9a29",
    "assistant_call": "#d94801",
    "tool_result": "#2171b5",
    "gen_prompt": "#969696",
    "generation": "#cb181d",
    "meta": "#bcbddc",
    "other": "#bdbdbd",
}


def build_block_position_grids(
    *,
    inputs: Sequence[Path],
    output_dir: Path,
    layers: Sequence[int] | None = None,
    include_orphans: bool = False,
    max_iters: int | None = None,
    split_by_task: bool = True,
    per_layer: bool = False,
    per_head: bool = False,
) -> dict[str, Any]:
    """Build absolute-KV-position grids of block_topk-selected blocks.

    Companion to ``build_block_segment_head_span_grids``: instead of pooling the
    selected middle tokens onto the segment axis, the y-axis is the absolute KV
    block (token) position, so the figure shows WHERE on the token axis (and
    inside which segment band) each selected block sits. Decode-only.
    """
    records = load_iteration_records(
        inputs, include_orphans=include_orphans, max_iters=max_iters
    )
    attempt_dirs = find_attempt_dirs(inputs)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        trajectory_rows, layer_rows, layers_used, meta = block_position_rows(
            group_records, layers=layers
        )
        if not trajectory_rows:
            raise ValueError(f"{label}: no block-position observations were found")
        if layers_used_global is None:
            layers_used_global = layers_used
        elif layers_used != layers_used_global:
            raise ValueError(
                f"{label}: layer set {layers_used} differs from other groups "
                f"{layers_used_global}; recordings are inconsistent"
            )
        group_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(group_dir / "block_position_trajectory.csv", trajectory_rows)
        _write_csv(group_dir / "block_position_by_layer.csv", layer_rows)
        plot_summary = _plot_block_position_grid(
            trajectory_rows,
            group_dir / "block_position_grid",
            layers_used=layers_used,
            meta=meta,
        )
        plot_summary = _portable_plot_summary(plot_summary, artifact_root=output_dir)
        per_layer_pdf = None
        if per_layer:
            per_layer_pdf = _artifact_relative_path(
                _render_per_layer_pdf_block_position(
                    group_records, layers_used=layers_used, group_dir=group_dir
                ),
                output_dir,
            )
        per_head_pdf = None
        if per_head:
            per_head_pdf = _artifact_relative_path(
                _render_per_head_pdf_block_position(
                    group_records, layers_used=layers_used, group_dir=group_dir
                ),
                output_dir,
            )
        group_summary = {
            "label": label,
            "output_dir": _artifact_relative_path(group_dir, output_dir),
            "runtime_output_dir": str(group_dir),
            "n_records": len(group_records),
            "n_blocks": meta["n_blocks"],
            "block_size": meta["block_size"],
            "sink_size": meta["sink_size"],
            "recent_window": meta["recent_window"],
            "r_max": meta["r_max"],
            "n_trajectory_rows": len(trajectory_rows),
            "n_layer_rows": len(layer_rows),
            "layers_used": layers_used,
            "per_layer_pdf": per_layer_pdf,
            "per_head_pdf": per_head_pdf,
            "plot": plot_summary,
        }
        (group_dir / "summary.json").write_text(
            json.dumps(_json_ready(group_summary), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (group_dir / "summary.md").write_text(
            _block_position_summary_markdown(group_summary),
            encoding="utf-8",
        )
        group_summaries.append(group_summary)

    run_summary = {
        "inputs": [str(path) for path in inputs],
        "attempt_dirs": [str(path) for path in attempt_dirs],
        "metric": "block_topk_selected_block_absolute_position",
        "mode": "block_position",
        "layers_used": layers_used_global,
        "split_by_task": split_by_task,
        "groups": group_summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_json_ready(run_summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return run_summary


def block_position_rows(
    records: Sequence[IterationRecord],
    *,
    layers: Sequence[int] | None = None,
    head: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int], dict[str, Any]]:
    """Per-(call, selected block) rows on the absolute KV block axis.

    For each call, every block_topk-selected MIDDLE block is attributed to its
    absolute KV position (``block_id`` -> token range ``[b*bs, b*bs+bs)``) and
    its owning segment (``seg_lo`` from ``block_span_selected_block_seg_range``).
    ``head=None`` pools query heads; ``head=h`` restricts to one. Returns
    ``(trajectory_rows, layer_rows, layers_used, meta)`` where ``meta`` carries
    geometry, the global block-axis extent, per-call key lengths (for the recent
    staircase) and the latest call's segment bands.
    """
    trajectory: list[dict[str, Any]] = []
    by_layer: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    layers_used: list[int] | None = None
    geom_ref: tuple[int, int, int, int] | None = None
    block_size = sink_size = recent_window = r_max = 0
    call_key_len: dict[int, int] = {}
    max_key_len = 0
    band_source_call = -1
    band_items: list[dict[str, Any]] = []

    for record in sorted(records, key=lambda item: (item.task, item.call_idx)):
        segments_payload = _load_json_required(record.iter_dir / "segments.json")
        segments = list(segments_payload.get("segments", []))
        segment_items = _segments_for_record(segments, record=record, metadata=metadata)
        if not segment_items:
            continue
        try:
            stats = load_block_head_span_stats(record.iter_dir)
        except (KeyError, FileNotFoundError, OSError, ValueError):
            # Mixed directory: skip iters predating block_span fields or with a
            # truncated/corrupt npz, mirroring block_head_span_rows.
            continue
        available = [int(layer) for layer in stats["block_span_layers"].tolist()]
        if not available:
            continue
        selected_layers = list(layers) if layers is not None else list(available)
        missing = [layer for layer in selected_layers if layer not in available]
        if missing:
            raise ValueError(
                f"{record.iter_dir}: requested layers not recorded in block_span: "
                f"{missing} (available: {available})"
            )
        if layers_used is None:
            layers_used = selected_layers
        elif selected_layers != layers_used:
            raise ValueError(
                f"{record.iter_dir}: block_span layer set {selected_layers} differs "
                f"from earlier iters {layers_used}; recordings are inconsistent"
            )
        mean_arr = stats["block_span_mean_decode"]
        C = int(mean_arr.shape[3]) if mean_arr.ndim == 4 else 0
        geom = (
            int(stats["block_span_block_size"]),
            int(stats["block_span_sink_size"]),
            int(stats["block_span_recent_window"]),
            C,
        )
        if geom_ref is None:
            geom_ref = geom
            block_size, sink_size, recent_window, _ = geom
            r_max = C - 2 if C >= 2 else 0
        elif geom != geom_ref:
            raise ValueError(
                f"{record.iter_dir}: block_span geometry {geom} differs from "
                f"earlier iters {geom_ref}; pooling incompatible bucket layouts "
                "would misalign the block axis"
            )
        positions = [available.index(layer) for layer in selected_layers]
        idx = np.asarray(positions, dtype=np.int64)
        mean_d = stats["block_span_mean_decode"][idx].astype(np.float64)
        var_d = stats["block_span_var_decode"][idx].astype(np.float64)
        sel_id = stats["block_span_selected_block_id"][idx].astype(np.int64)
        seg_range = stats["block_span_selected_block_seg_range"][idx].astype(np.int64)
        step_d = stats["block_span_decode_step"][idx].astype(np.int64)
        idx2item = {int(it["segment_idx"]): it for it in segment_items}

        key_len = max((int(it["token_end"]) for it in segment_items), default=0)
        call_key_len[int(record.call_idx)] = key_len
        max_key_len = max(max_key_len, key_len)
        if int(record.call_idx) >= band_source_call:
            band_source_call = int(record.call_idx)
            band_items = list(segment_items)

        agg, n_valid = _block_position_aggregate(
            mean_d, var_d, sel_id, seg_range, step_d, head=head
        )
        for block_id, cell in agg.items():
            trajectory.append(
                _block_position_row(
                    block_id, cell, n_valid, block_size, r_max,
                    record=record, idx2item=idx2item,
                    selected_layers=selected_layers, layer=None,
                )
            )
        for li, layer in enumerate(selected_layers):
            sl = slice(li, li + 1)
            agg_l, n_valid_l = _block_position_aggregate(
                mean_d[sl], var_d[sl], sel_id[sl], seg_range[sl], step_d[sl], head=head
            )
            for block_id, cell in agg_l.items():
                by_layer.append(
                    _block_position_row(
                        block_id, cell, n_valid_l, block_size, r_max,
                        record=record, idx2item=idx2item,
                        selected_layers=selected_layers, layer=int(layer),
                    )
                )

    used = list(layers_used or [])
    n_blocks = (-(-max_key_len // block_size)) if block_size > 0 else 0
    bands = _block_position_bands(band_items, block_size) if block_size > 0 else []
    meta = {
        "block_size": block_size,
        "sink_size": sink_size,
        "recent_window": recent_window,
        "r_max": r_max,
        "n_blocks": int(n_blocks),
        "call_key_len": {int(k): int(v) for k, v in call_key_len.items()},
        "bands": bands,
        "band_source_call": band_source_call,
    }
    return trajectory, by_layer, used, meta


def _block_position_aggregate(
    mean_d: np.ndarray,
    var_d: np.ndarray,
    sel_id: np.ndarray,
    seg_range: np.ndarray,
    step_d: np.ndarray,
    *,
    head: int | None,
) -> tuple[dict[int, dict[str, Any]], int]:
    """Per-block aggregates over one call's (already layer-sliced) arrays.

    Inputs: ``mean_d``/``var_d`` ``[L,T,H,C]``, ``sel_id`` ``[L,T,R_max]``,
    ``seg_range`` ``[L,T,R_max,2]``, ``step_d`` ``[L,T]``. Returns
    ``({block_id: aggregates}, n_valid_layer_steps)`` for every block selected at
    least once. Aggregation mirrors ``reduce_head_span_cell``: ``mean`` = mean of
    finite per-head means, ``var_pooled`` = mean of finite per-head variances,
    ``std`` = sqrt(var_pooled), ``cross_head_std`` = population std of the finite
    means, ``n_contributors`` = finite (head,layer,step,rank) count. The
    bincount path computes ``cross_head_std`` with a one-pass formula instead of
    the reference two-pass ``std(ddof=0)``; the two are algebraically identical
    and agree to ~1e-6 for post-softmax attention in [0, 1] (the only inputs
    here), trading the reference's large-offset stability for vectorization.
    """
    r_max = int(sel_id.shape[2]) if sel_id.ndim == 3 else 0
    valid = step_d >= 0  # [L, T]
    n_valid = int(valid.sum())
    if r_max == 0 or n_valid == 0:
        return {}, n_valid
    if head is None:
        m_sel = mean_d[:, :, :, 1 : r_max + 1]  # [L, T, H, R]
        v_sel = var_d[:, :, :, 1 : r_max + 1]
    else:
        m_sel = mean_d[:, :, head : head + 1, 1 : r_max + 1]  # [L, T, 1, R]
        v_sel = var_d[:, :, head : head + 1, 1 : r_max + 1]
    bsel = sel_id[valid]  # [Nv, R]
    segr = seg_range[valid]  # [Nv, R, 2]
    msel = m_sel[valid]  # [Nv, H, R]
    vsel = v_sel[valid]
    n_steps, n_h, _ = msel.shape

    # hits: count (step, rank) selections per block (head-independent).
    flat_b = bsel.reshape(-1)
    sel_mask = flat_b >= 0
    if not bool(sel_mask.any()):
        return {}, n_valid
    n_blocks = int(flat_b[sel_mask].max()) + 1
    hits = np.bincount(flat_b[sel_mask], minlength=n_blocks)
    seglo = segr[:, :, 0].reshape(-1)
    seghi = segr[:, :, 1].reshape(-1)
    arr_lo = np.full(n_blocks, np.iinfo(np.int64).max, dtype=np.int64)
    arr_hi = np.full(n_blocks, -1, dtype=np.int64)
    np.minimum.at(arr_lo, flat_b[sel_mask], seglo[sel_mask])
    np.maximum.at(arr_hi, flat_b[sel_mask], seghi[sel_mask])

    # per-head contributor stats: broadcast the block id over the head axis so
    # every (step, head, rank) observation is attributed to its block.
    b_bcast = np.broadcast_to(bsel[:, None, :], (n_steps, n_h, r_max)).reshape(-1)
    m_flat = msel.reshape(-1)
    v_flat = vsel.reshape(-1)
    in_sel = b_bcast >= 0
    fin = in_sel & np.isfinite(m_flat) & np.isfinite(v_flat)
    # total_obs counts this block's selected (step,head,rank) occurrences (not a
    # global observation count), so n_nan_contributors = selected-but-NaN heads.
    total_obs = np.bincount(b_bcast[in_sel], minlength=n_blocks)
    n_contrib = np.bincount(b_bcast[fin], minlength=n_blocks)
    sum_m = np.bincount(b_bcast[fin], weights=m_flat[fin], minlength=n_blocks)
    sum_m2 = np.bincount(b_bcast[fin], weights=m_flat[fin] ** 2, minlength=n_blocks)
    sum_v = np.bincount(b_bcast[fin], weights=v_flat[fin], minlength=n_blocks)

    out: dict[int, dict[str, Any]] = {}
    for block in np.nonzero(hits)[0]:
        b = int(block)
        nc = int(n_contrib[b])
        if nc > 0:
            mean = float(sum_m[b] / nc)
            var_pooled = float(sum_v[b] / nc)
            std = float(np.sqrt(max(var_pooled, 0.0)))
            cross = (
                float(np.sqrt(max(sum_m2[b] / nc - mean * mean, 0.0)))
                if nc >= 2
                else 0.0
            )
        else:
            mean = std = var_pooled = cross = None
        out[b] = {
            "n_hits": int(hits[b]),
            "seg_lo": int(arr_lo[b]),
            "seg_hi": int(arr_hi[b]),
            "mean": mean,
            "std": std,
            "var_pooled": var_pooled,
            "cross_head_std": cross,
            "n_contributors": nc,
            "n_nan_contributors": int(total_obs[b]) - nc,
        }
    return out, n_valid


def _block_position_row(
    block_id: int,
    cell: dict[str, Any],
    n_valid: int,
    block_size: int,
    r_max: int,
    *,
    record: IterationRecord,
    idx2item: dict[int, dict[str, Any]],
    selected_layers: Sequence[int],
    layer: int | None,
) -> dict[str, Any]:
    seg_lo = int(cell["seg_lo"])
    seg_hi = int(cell["seg_hi"])
    owner = idx2item.get(seg_lo, {})
    row = {
        "task": record.task,
        "observed_call_idx": int(record.call_idx),
        "block_id": int(block_id),
        "token_start": int(block_id) * block_size,
        "token_end": int(block_id) * block_size + block_size,
        "seg_lo": seg_lo,
        "seg_hi": seg_hi,
        "straddles": bool(seg_lo != seg_hi),
        "segment_id": owner.get("segment_id"),
        "segment_ordinal": owner.get("segment_ordinal"),
        "role": owner.get("role"),
        "tool_name": owner.get("tool_name"),
        "selection_freq": (cell["n_hits"] / n_valid) if n_valid else None,
        "n_hits": int(cell["n_hits"]),
        "n_valid_layer_steps": int(n_valid),
        "r_max": int(r_max),
        "mean_attn": cell["mean"],
        "std_attn": cell["std"],
        "var_pooled": cell["var_pooled"],
        "cross_head_std": cell["cross_head_std"],
        "n_contributors": int(cell["n_contributors"]),
        "n_nan_contributors": int(cell["n_nan_contributors"]),
        "n_layers": len(selected_layers),
        "layers_used": list(selected_layers),
    }
    if layer is not None:
        row["layer"] = int(layer)
    return row


def _block_position_bands(
    segment_items: Sequence[dict[str, Any]], block_size: int
) -> list[dict[str, Any]]:
    """Segment bands on the block axis from one call's segments."""
    bands: list[dict[str, Any]] = []
    for item in sorted(segment_items, key=lambda x: int(x["token_start"])):
        token_start = int(item["token_start"])
        token_end = int(item["token_end"])
        if token_end <= token_start:
            continue
        bands.append(
            {
                "block_lo": token_start // block_size,
                "block_hi": -(-token_end // block_size),  # ceil
                "label": _segment_plot_label(item),
                "role": str(item.get("role") or "other"),
                "segment_ordinal": int(item.get("segment_ordinal", 0)),
            }
        )
    return bands


def _plot_block_position_grid(
    trajectory_rows: Sequence[dict[str, Any]],
    output_stem: Path,
    *,
    layers_used: Sequence[int],
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Render the block(token) x call selection map with segment bands."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    missing_color = "#cfcfcf"
    block_size = int(meta["block_size"])
    sink_size = int(meta["sink_size"])
    recent_window = int(meta["recent_window"])
    r_max = int(meta["r_max"])
    n_blocks = int(meta["n_blocks"])
    call_key_len = {int(k): int(v) for k, v in meta["call_key_len"].items()}
    bands = meta["bands"]
    if not trajectory_rows or n_blocks <= 0 or block_size <= 0:
        raise ValueError("block_position grid: no selected-block observations")

    observed_iters = [int(row["observed_call_idx"]) for row in trajectory_rows]
    min_iter = min(observed_iters)
    max_iter = max(observed_iters)
    iter_values = list(range(min_iter, max_iter + 1))
    iter_to_col = {iter_idx: col for col, iter_idx in enumerate(iter_values)}
    n_cols = len(iter_values)

    freq_matrix = np.full((n_blocks, n_cols), np.nan, dtype=np.float64)
    mean_matrix = np.full((n_blocks, n_cols), np.nan, dtype=np.float64)
    for row in trajectory_rows:
        block_id = int(row["block_id"])
        col = iter_to_col.get(int(row["observed_call_idx"]))
        if col is None or not (0 <= block_id < n_blocks):
            continue
        if row.get("selection_freq") is not None:
            freq_matrix[block_id, col] = float(row["selection_freq"])
        if row.get("mean_attn") is not None:
            mean_matrix[block_id, col] = float(row["mean_attn"])

    freq_cmap = LinearSegmentedColormap.from_list(
        "asb_block_position_freq",
        ["#f7fbff", "#deebf7", "#9ecae1", "#3182bd", "#08519c"],
    )
    mean_cmap = LinearSegmentedColormap.from_list(
        "asb_block_position_mean",
        ["#fffaf0", "#fee391", "#fdae61", "#e34a33", "#7f0000"],
    )
    for cmap in (freq_cmap, mean_cmap):
        cmap.set_bad(missing_color)
    freq_vmax = _percentile_vmax({"decode": freq_matrix})
    mean_vmax = _percentile_vmax({"decode": mean_matrix})

    width = max(11.0, min(20.0, 6.8 + 0.5 * n_cols))
    height = max(8.5, min(26.0, 2.5 + 0.012 * n_blocks))
    fig, axes = plt.subplots(1, 2, figsize=(width, height), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.2, right=0.93, bottom=0.09, top=0.92, wspace=0.07)

    sink_blocks = -(-sink_size // block_size)
    block_ticks = _integer_tick_positions(list(range(n_blocks)), max_labels=24)
    panels = [
        (
            axes[0], freq_matrix, freq_cmap, freq_vmax,
            "selection frequency", "selected steps / valid steps", True,
        ),
        (
            axes[1], mean_matrix, mean_cmap, mean_vmax,
            "attention mean (selected)", "mean per-token attention", False,
        ),
    ]
    for ax, matrix, cmap, vmax, title, cbar_label, with_labels in panels:
        im = ax.imshow(
            matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax, origin="upper"
        )
        ax.set_title(title)
        ax.set_xlabel("recording iter / LLM call index")
        ax.set_xlim(-0.5, n_cols - 0.5)
        ax.set_ylim(n_blocks - 0.5, -0.5)
        tick_cols = _integer_tick_positions(iter_values, max_labels=32)
        ax.set_xticks(tick_cols)
        ax.set_xticklabels([str(iter_values[col]) for col in tick_cols], fontsize=7)
        ax.set_yticks(block_ticks)
        ax.set_yticklabels([str(b * block_size) for b in block_ticks], fontsize=6)
        ax.set_facecolor(missing_color)
        if sink_blocks > 0:
            ax.axhspan(-0.5, sink_blocks - 0.5, facecolor="#fdbf6f", alpha=0.18, zorder=0)
        for iter_idx, col in iter_to_col.items():
            key_len = call_key_len.get(iter_idx)
            if not key_len:
                continue
            frontier = min(-(-key_len // block_size), n_blocks)
            recent_lo = max(0, (key_len - recent_window) // block_size)
            if frontier > recent_lo:
                ax.add_patch(
                    Rectangle(
                        (col - 0.5, recent_lo - 0.5), 1.0, frontier - recent_lo,
                        facecolor="#b2df8a", alpha=0.16, edgecolor="none", zorder=0.5,
                    )
                )
        _draw_block_position_bands(ax, bands, with_labels=with_labels)
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.012)
        cbar.set_label(cbar_label)
    axes[0].set_ylabel(f"KV token position (block * {block_size})")
    fig.suptitle(
        "block_topk selected blocks on the absolute KV token axis "
        f"(layers {list(layers_used)}; block_size={block_size}, "
        f"sink={sink_size}, recent={recent_window})",
        fontsize=10,
        y=0.985,
    )
    fig.text(
        0.01,
        0.012,
        "Y = absolute KV position (each cell = one block of "
        f"{block_size} tokens; token 0 / sink at top). X = LLM call (decode steps "
        "pooled). Left = fraction of valid (layer, step) that selected the block; "
        "right = mean attention over the block when selected. Orange band = sink, "
        "green staircase = recent window (both always kept, not block_topk-chosen). "
        "Gray = block never selected. Segment bands labeled at left.",
        fontsize=6.5,
        color="#555555",
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {
        "grid_png": str(output_stem.with_suffix(".png")),
        "grid_pdf": str(output_stem.with_suffix(".pdf")),
        "x_axis": "recording_iter",
        "min_iter": min_iter,
        "max_iter": max_iter,
        "n_iters": n_cols,
        "n_blocks": n_blocks,
        "block_size": block_size,
        "sink_size": sink_size,
        "recent_window": recent_window,
        "r_max": r_max,
        "freq_vmax_percentile_95": freq_vmax,
        "mean_vmax_percentile_95": mean_vmax,
        "layers_used": list(layers_used),
        "missing_color": missing_color,
    }


def _draw_block_position_bands(
    ax: Any, bands: Sequence[dict[str, Any]], *, with_labels: bool
) -> None:
    """Overlay segment boundaries, a role-colored left gutter, and labels."""
    from matplotlib.patches import Rectangle

    major = "#222222"
    n = len(bands)
    if n == 0:
        return
    label_rows = set(_integer_tick_positions(list(range(n)), max_labels=30))
    trans = ax.get_yaxis_transform()  # x: axes fraction, y: data coords
    for i, band in enumerate(bands):
        block_lo = int(band["block_lo"])
        block_hi = int(band["block_hi"])
        ax.axhline(block_lo - 0.5, color=major, linewidth=0.4, alpha=0.5, zorder=2)
        if not with_labels:
            continue
        color = _ROLE_COLORS.get(band["role"], _ROLE_COLORS["other"])
        ax.add_patch(
            Rectangle(
                (-0.045, block_lo - 0.5), 0.02, max(1, block_hi - block_lo),
                transform=trans, clip_on=False, facecolor=color, edgecolor="none",
                zorder=3,
            )
        )
        if i in label_rows:
            ax.text(
                -0.05,
                (block_lo + block_hi) / 2.0 - 0.5,
                band["label"],
                transform=trans,
                ha="right",
                va="center",
                fontsize=4.5,
                color="#333333",
                clip_on=False,
            )
    ax.axhline(
        int(bands[-1]["block_hi"]) - 0.5, color=major, linewidth=0.4, alpha=0.5, zorder=2
    )


def _render_per_layer_pdf_block_position(
    group_records: Sequence[IterationRecord],
    *,
    layers_used: Sequence[int],
    group_dir: Path,
) -> Path:
    """Render one block-position grid per recorded layer into a PDF."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    per_layer_dir = group_dir / "per_layer"
    pages: list[tuple[int, Path]] = []
    for layer in layers_used:
        rows, _layer_rows, _used, meta = block_position_rows(
            group_records, layers=[layer]
        )
        if not rows:
            continue
        stem = per_layer_dir / f"layer_{int(layer):02d}" / "block_position_grid"
        _plot_block_position_grid(rows, stem, layers_used=[layer], meta=meta)
        pages.append((int(layer), stem.with_suffix(".png")))

    pdf_path = group_dir / "block_position_per_layer.pdf"
    with PdfPages(pdf_path) as pdf:
        for layer, png in pages:
            img = plt.imread(png)
            height, width = img.shape[0], img.shape[1]
            fig = plt.figure(figsize=(width / 150.0, height / 150.0 + 0.4))
            ax = fig.add_axes([0, 0, 1, 0.96])
            ax.imshow(img)
            ax.axis("off")
            fig.suptitle(
                f"layer {layer} - block_topk selected blocks on KV token axis",
                fontsize=11,
                y=0.995,
            )
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
    return pdf_path


def _render_per_head_pdf_block_position(
    group_records: Sequence[IterationRecord],
    *,
    layers_used: Sequence[int],
    group_dir: Path,
) -> Path:
    """Render one block-position grid per query head into a PDF."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    n_heads = _n_query_heads_block(group_records)
    per_head_dir = group_dir / "per_head"
    pages: list[tuple[int, Path]] = []
    for h in range(n_heads):
        rows, _layer_rows, _used, meta = block_position_rows(
            group_records, layers=list(layers_used), head=h
        )
        if not rows:
            continue
        stem = per_head_dir / f"head_{h:02d}" / "block_position_grid"
        _plot_block_position_grid(rows, stem, layers_used=layers_used, meta=meta)
        pages.append((h, stem.with_suffix(".png")))

    pdf_path = group_dir / "block_position_per_head.pdf"
    with PdfPages(pdf_path) as pdf:
        for h, png in pages:
            img = plt.imread(png)
            height, width = img.shape[0], img.shape[1]
            fig = plt.figure(figsize=(width / 150.0, height / 150.0 + 0.4))
            ax = fig.add_axes([0, 0, 1, 0.96])
            ax.imshow(img)
            ax.axis("off")
            fig.suptitle(
                f"head {h:02d} - block_topk selected blocks on KV token axis",
                fontsize=11,
                y=0.995,
            )
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
    return pdf_path


def _block_position_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# BlockTopK Selected Blocks on the Absolute KV Token Axis",
        "",
        f"- Label: `{summary['label']}`",
        f"- Output: `{summary['output_dir']}`",
        f"- Records: `{summary['n_records']}`",
        f"- Blocks (Y extent): `{summary['n_blocks']}` (block_size=`{summary['block_size']}`)",
        f"- Sink: `{summary['sink_size']}`; recent window: `{summary['recent_window']}`; r_max: `{summary['r_max']}`",
        f"- Layers aggregated: `{summary['layers_used']}`",
        "",
        "## Figure",
        "",
        "- Y axis: absolute KV token position (each cell = one block); token 0 / sink at top.",
        "- X axis: recording iter / LLM call index; decode steps pooled per call.",
        "- Left panel: selection frequency = valid (layer,step) selecting the block / all valid (layer,step).",
        "- Right panel: mean attention over the block's kept tokens when selected.",
        "- Orange band = sink, green staircase = recent window (always kept, not block_topk-chosen).",
        "- Segment bands labeled at left; gray = block never selected.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
