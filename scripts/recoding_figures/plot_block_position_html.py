"""Render block_position grids as an offline Canvas HTML viewer.

The static block_position PNG is a useful thumbnail, but the full block axis can
exceed one thousand rows. This companion writes a self-contained HTML app with
two synchronized Canvas heatmaps, layer/head selectors, domain-specific zoom
presets, segment focusing, and hover text from selected_blocks_detok.csv when
the detokenization step has already run.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
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
    load_iteration_records,
)
from scripts.recoding_figures.plot_head_span_grid import (  # noqa: E402
    _ROLE_COLORS,
    _parse_layer_arg,
    block_position_rows_by_head,
)
from scripts.recoding_figures.plot_sparse_segment_grid import (  # noqa: E402
    _artifact_relative_path,
    _json_ready,
    _safe_name,
)

MISSING_COLOR = "#cfcfcf"
SINK_COLOR = "#f28e2b"
SINK_EDGE = "#b85f00"
RECENT_COLOR = "#2ca02c"
RECENT_EDGE = "#1b7f2a"
AUTO_WORKER_CAP = 4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="attempt, task, or run dirs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=str, default=None, help="e.g. 0,9,18,27,38,47")
    parser.add_argument("--include-orphans", action="store_true")
    parser.add_argument("--max-iters", type=int)
    parser.add_argument(
        "--split-by-task",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write one HTML file per task. Default: true.",
    )
    parser.add_argument(
        "--max-html-mib",
        type=float,
        default=150.0,
        help=(
            "Fail after writing if a task HTML exceeds this size. This guards "
            "against silently dropping per-head data to save space. Default 150."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Parallel task HTML builders. Use 1 for serial execution. 0 means "
            f"auto = min(task groups, CPU count, {AUTO_WORKER_CAP}). Default: 1."
        ),
    )
    # Kept as a no-op for compatibility with the previous Plotly writer.
    parser.add_argument("--include-plotlyjs", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    summary = build_block_position_html(
        inputs=args.inputs,
        output_dir=args.output_dir,
        layers=_parse_layer_arg(args.layers),
        include_orphans=args.include_orphans,
        max_iters=args.max_iters,
        split_by_task=args.split_by_task,
        max_html_mib=args.max_html_mib,
        workers=args.workers,
    )
    print(json.dumps(_json_ready(summary), indent=2, sort_keys=True))


def build_block_position_html(
    inputs: Sequence[Path],
    output_dir: Path,
    *,
    layers: Sequence[int] | None = None,
    include_orphans: bool = False,
    max_iters: int | None = None,
    split_by_task: bool = True,
    max_html_mib: float = 150.0,
    workers: int = 1,
) -> dict[str, Any]:
    """Build offline Canvas block_position HTML artifacts."""
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

    jobs = [
        (label, group_records, group_dir, output_dir, layers, max_html_mib)
        for label, group_records, group_dir in groups
    ]
    n_workers = _resolve_workers(workers, len(jobs))
    if n_workers > 1:
        print(
            f"building {len(jobs)} block_position HTML group(s) with {n_workers} workers",
            file=sys.stderr,
            flush=True,
        )
        group_summaries_by_label: dict[str, dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_build_group_html_job, job): job[0] for job in jobs}
            for future in as_completed(futures):
                label = futures[future]
                summary = future.result()
                group_summaries_by_label[label] = summary
                print(
                    f"finished {label}: {summary['html_size_mib']:.1f} MiB HTML",
                    file=sys.stderr,
                    flush=True,
                )
        group_summaries = [group_summaries_by_label[label] for label, _, _ in groups]
    else:
        group_summaries = [_build_group_html_job(job) for job in jobs]

    run_summary = {
        "inputs": [str(path) for path in inputs],
        "attempt_dirs": [str(path) for path in attempt_dirs],
        "artifact": "block_position_canvas_html",
        "split_by_task": split_by_task,
        "workers": n_workers,
        "groups": group_summaries,
    }
    (output_dir / "block_position_html_summary.json").write_text(
        json.dumps(_json_ready(run_summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return run_summary


def _resolve_workers(requested: int, n_groups: int) -> int:
    """Resolve task-level worker count without oversubscribing group outputs."""
    if n_groups <= 1:
        return 1
    if requested < 0:
        raise ValueError(f"--workers must be >= 0, got {requested}")
    if requested == 1:
        return 1
    if requested == 0:
        return max(1, min(n_groups, os.cpu_count() or 1, AUTO_WORKER_CAP))
    return max(1, min(requested, n_groups))


def _build_group_html_job(
    job: tuple[str, Sequence[IterationRecord], Path, Path, Sequence[int] | None, float],
) -> dict[str, Any]:
    label, group_records, group_dir, artifact_root, layers, max_html_mib = job
    return _build_group_html(
        label,
        group_records,
        group_dir,
        artifact_root,
        layers=layers,
        max_html_mib=max_html_mib,
    )


def _build_group_html(
    label: str,
    records: Sequence[IterationRecord],
    group_dir: Path,
    artifact_root: Path,
    *,
    layers: Sequence[int] | None,
    max_html_mib: float,
) -> dict[str, Any]:
    trajectory_rows, layer_rows, per_head_trajectory, per_head_layer, layers_used, meta = (
        block_position_rows_by_head(records, layers=layers)
    )
    if not trajectory_rows:
        raise ValueError(f"{label}: no block-position observations were found")

    group_dir.mkdir(parents=True, exist_ok=True)
    detok = _load_detok(group_dir / "selected_blocks_detok.csv")
    call_values = _call_values(trajectory_rows)
    n_blocks = int(meta["n_blocks"])
    block_size = int(meta["block_size"])
    call_to_col = {call_idx: col for col, call_idx in enumerate(call_values)}
    block_to_row = {block_id: block_id for block_id in range(n_blocks)}

    n_heads = len(per_head_trajectory)
    head_keys = ["all"] + [str(head) for head in range(n_heads)]
    layer_keys = ["all"] + [str(int(layer)) for layer in layers_used]

    freq_views: dict[str, dict[str, Any]] = {
        "all": _sparse_from_rows(
            trajectory_rows,
            block_to_row=block_to_row,
            call_to_col=call_to_col,
            key="selection_freq",
        )
    }
    layer_to_rows: dict[int, list[dict[str, Any]]] = {
        int(layer): [] for layer in layers_used
    }
    for row in layer_rows:
        layer_val = int(row.get("layer", -1))
        layer_bucket = layer_to_rows.get(layer_val)
        if layer_bucket is not None:
            layer_bucket.append(row)

    for layer in layers_used:
        layer_key = str(int(layer))
        freq_views[layer_key] = _sparse_from_rows(
            layer_to_rows[int(layer)],
            block_to_row=block_to_row,
            call_to_col=call_to_col,
            key="selection_freq",
        )

    attn_views: dict[str, dict[str, dict[str, Any]]] = {key: {} for key in layer_keys}
    attn_views["all"]["all"] = _sparse_from_rows(
        trajectory_rows,
        block_to_row=block_to_row,
        call_to_col=call_to_col,
        key="mean_attn",
    )
    for layer in layers_used:
        layer_key = str(int(layer))
        attn_views[layer_key]["all"] = _sparse_from_rows(
            layer_to_rows[int(layer)],
            block_to_row=block_to_row,
            call_to_col=call_to_col,
            key="mean_attn",
        )
    for head in range(n_heads):
        head_key = str(head)
        head_rows = per_head_trajectory[head] if head < len(per_head_trajectory) else []
        head_by_layer = per_head_layer[head] if head < len(per_head_layer) else {}
        attn_views["all"][head_key] = _sparse_from_rows(
            head_rows,
            block_to_row=block_to_row,
            call_to_col=call_to_col,
            key="mean_attn",
        )
        for layer in layers_used:
            layer_key = str(int(layer))
            attn_views[layer_key][head_key] = _sparse_from_rows(
                head_by_layer.get(int(layer), []),
                block_to_row=block_to_row,
                call_to_col=call_to_col,
                key="mean_attn",
            )

    payload = {
        "label": label,
        "nBlocks": n_blocks,
        "nCalls": len(call_values),
        "blockSize": block_size,
        "sinkSize": int(meta["sink_size"]),
        "recentWindow": int(meta["recent_window"]),
        "rMax": int(meta["r_max"]),
        "callValues": list(call_values),
        "callKeyLen": {str(int(k)): int(v) for k, v in meta["call_key_len"].items()},
        "layers": [{"key": "all", "label": "all layers"}]
        + [{"key": str(int(layer)), "label": f"layer {int(layer)}"} for layer in layers_used],
        "heads": [{"key": key, "label": "all heads" if key == "all" else f"head {key}"}
                  for key in head_keys],
        "segments": _segment_payload(meta.get("bands") or [], block_size),
        "cellMeta": _cell_meta(trajectory_rows, detok, call_to_col=call_to_col),
        "freq": freq_views,
        "attn": attn_views,
        "colors": {
            "missing": MISSING_COLOR,
            "sink": SINK_COLOR,
            "sinkEdge": SINK_EDGE,
            "recent": RECENT_COLOR,
            "recentEdge": RECENT_EDGE,
        },
    }

    html_path = group_dir / "block_position.html"
    html_path.write_text(_html_document(payload), encoding="utf-8")
    size_mib = html_path.stat().st_size / 1024 / 1024
    if size_mib > max_html_mib:
        raise ValueError(
            f"{label}: block_position.html is {size_mib:.1f} MiB, exceeding "
            f"--max-html-mib={max_html_mib}. Per-head data was not dropped; "
            "raise the limit explicitly if this size is acceptable."
        )

    group_summary = {
        "label": label,
        "output_dir": _artifact_relative_path(group_dir, artifact_root),
        "runtime_output_dir": str(group_dir),
        "html": _artifact_relative_path(html_path, artifact_root),
        "runtime_html": str(html_path),
        "html_size_mib": size_mib,
        "n_records": len(records),
        "n_blocks": n_blocks,
        "n_calls": len(call_values),
        "block_size": block_size,
        "sink_size": int(meta["sink_size"]),
        "recent_window": int(meta["recent_window"]),
        "r_max": int(meta["r_max"]),
        "layers_used": list(layers_used),
        "n_heads": n_heads,
        "n_freq_views": len(freq_views),
        "n_attn_views": sum(len(per_layer) for per_layer in attn_views.values()),
        "detok_rows_loaded": len(detok),
        "renderer": "canvas",
    }
    (group_dir / "block_position_html_summary.json").write_text(
        json.dumps(_json_ready(group_summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return group_summary


def _sparse_from_rows(
    rows: Sequence[dict[str, Any]],
    *,
    block_to_row: dict[int, int],
    call_to_col: dict[int, int],
    key: str,
) -> dict[str, Any]:
    n_cols = len(call_to_col)
    flat_values: dict[int, float] = {}
    for row in rows:
        row_idx = block_to_row.get(int(row["block_id"]))
        col_idx = call_to_col.get(int(row["observed_call_idx"]))
        if row_idx is None or col_idx is None:
            continue
        value = row.get(key)
        if value is None:
            continue
        idx = int(row_idx) * n_cols + int(col_idx)
        flat_values[idx] = round(float(value), 10)

    indices = sorted(flat_values.keys())
    values = [flat_values[idx] for idx in indices]
    vmax = _percentile_values(values)
    return {"indices": indices, "values": values, "vmax": vmax}


def _percentile_values(values: Sequence[float]) -> float:
    if not values:
        return 1.0
    return max(float(np.percentile(np.asarray(values, dtype=np.float64), 95)), 1e-12)


def _segment_payload(
    bands: Sequence[dict[str, Any]], block_size: int
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, band in enumerate(bands):
        role = str(band.get("role") or "other")
        block_lo = int(band["block_lo"])
        block_hi = int(band["block_hi"])
        out.append(
            {
                "idx": idx,
                "blockLo": block_lo,
                "blockHi": block_hi,
                "tokenStart": block_lo * block_size,
                "tokenEnd": block_hi * block_size,
                "label": str(band.get("label") or f"segment {idx}"),
                "role": role,
                "color": _ROLE_COLORS.get(role, _ROLE_COLORS["other"]),
            }
        )
    return out


def _cell_meta(
    rows: Sequence[dict[str, Any]],
    detok: dict[tuple[int, int], dict[str, Any]],
    *,
    call_to_col: dict[int, int],
) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    n_cols = len(call_to_col)
    for row in rows:
        call_idx = int(row["observed_call_idx"])
        block_id = int(row["block_id"])
        col = call_to_col.get(call_idx)
        if col is None:
            continue
        linear = block_id * n_cols + col
        detok_row = detok.get((call_idx, block_id), {})
        out[str(linear)] = [
            block_id,
            int(row["token_start"]),
            int(row["token_end"]),
            _display_value(row.get("role") or detok_row.get("role")),
            _segment_text(row),
            _compact_text(detok_row.get("decoded_text")),
        ]
    return out


def _html_document(payload: dict[str, Any]) -> str:
    data_json = _json_for_script(payload)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape_text(str(payload["label"]))} block_position</title>
<style>
:root {{
  --bg: #f6f7f9;
  --panel: #ffffff;
  --border: #c9d1dc;
  --text: #172033;
  --muted: #667085;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font: 13px/1.35 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--text);
  background: var(--bg);
}}
.toolbar {{
  position: sticky;
  top: 0;
  z-index: 10;
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  min-height: 58px;
  padding: 9px 14px;
  background: rgba(255,255,255,0.97);
  border-bottom: 1px solid var(--border);
  box-shadow: 0 1px 4px rgba(16, 24, 40, 0.08);
}}
.title {{
  font-weight: 650;
  margin-right: 8px;
  white-space: nowrap;
}}
label {{ color: var(--muted); font-size: 12px; }}
select, input, button {{
  height: 30px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: #fff;
  color: var(--text);
  padding: 0 8px;
}}
input {{ width: 82px; }}
button {{ cursor: pointer; }}
button.active {{ background: #203864; color: #fff; border-color: #203864; }}
.hint {{ color: var(--muted); font-size: 12px; }}
.app {{
  height: calc(100vh - 58px);
  min-height: 680px;
  display: grid;
  grid-template-columns: 235px minmax(720px, 1fr) 330px;
  gap: 10px;
  padding: 10px;
}}
.rail, .details, .chart-card {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  min-height: 0;
  overflow: hidden;
}}
.rail-head, .details-head {{
  padding: 9px 10px;
  font-weight: 650;
  border-bottom: 1px solid var(--border);
}}
#segmentCanvas {{
  display: block;
  width: 100%;
  height: calc(100% - 38px);
  cursor: pointer;
}}
.charts {{
  min-width: 0;
  min-height: 0;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}}
.chart-card {{
  display: grid;
  grid-template-rows: 36px minmax(0, 1fr);
}}
.chart-head {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 7px 10px;
  border-bottom: 1px solid var(--border);
  font-weight: 650;
}}
.legend {{
  width: 160px;
  height: 10px;
  border-radius: 999px;
  border: 1px solid rgba(0,0,0,0.12);
}}
.legend.freq {{ background: linear-gradient(90deg,#f7fbff,#9ecae1,#08519c); }}
.legend.attn {{ background: linear-gradient(90deg,#fffaf0,#fdae61,#7f0000); }}
canvas.heatmap {{
  display: block;
  width: 100%;
  height: 100%;
  cursor: grab;
}}
canvas.heatmap.boxing {{ cursor: crosshair; }}
.details-body {{
  padding: 10px;
  overflow: auto;
  height: calc(100% - 38px);
}}
.kv {{
  display: grid;
  grid-template-columns: 92px 1fr;
  gap: 5px 8px;
  margin-bottom: 10px;
}}
.kv div:nth-child(odd) {{ color: var(--muted); }}
.text-box {{
  white-space: pre-wrap;
  word-break: break-word;
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 8px;
  background: #fbfcfe;
  min-height: 120px;
}}
.tooltip {{
  position: fixed;
  z-index: 20;
  pointer-events: none;
  display: none;
  max-width: 420px;
  padding: 8px 10px;
  background: rgba(17, 24, 39, 0.94);
  color: #fff;
  border-radius: 7px;
  font-size: 12px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.22);
}}
@media (max-width: 1300px) {{
  .app {{ grid-template-columns: 190px minmax(640px, 1fr); }}
  .details {{ display: none; }}
}}
</style>
</head>
<body>
<div class="toolbar">
  <div class="title" id="title"></div>
  <label>Layer <select id="layerSelect"></select></label>
  <label>Head <select id="headSelect"></select></label>
  <button id="panBtn" class="active" type="button">Pan</button>
  <button id="boxBtn" type="button">Box zoom</button>
  <button id="fullBtn" type="button">Full</button>
  <button id="recentBtn" type="button">Recent</button>
  <button id="lastCallBtn" type="button">Last call</button>
  <label>Call <input id="callStart" type="number"> - <input id="callEnd" type="number"></label>
  <label>Token <input id="tokenStart" type="number"> - <input id="tokenEnd" type="number"></label>
  <button id="applyRangeBtn" type="button">Apply</button>
  <label>Segment <select id="segmentSelect"></select></label>
  <button id="clearBtn" type="button">Clear selection</button>
  <span class="hint">Head changes attention only; frequency is head-independent.</span>
</div>
<div class="app">
  <div class="rail">
    <div class="rail-head">Segments</div>
    <canvas id="segmentCanvas"></canvas>
  </div>
  <div class="charts">
    <div class="chart-card">
      <div class="chart-head"><span>selection frequency</span><span class="legend freq"></span></div>
      <canvas id="freqCanvas" class="heatmap"></canvas>
    </div>
    <div class="chart-card">
      <div class="chart-head"><span>attention mean (selected)</span><span class="legend attn"></span></div>
      <canvas id="attnCanvas" class="heatmap"></canvas>
    </div>
  </div>
  <div class="details">
    <div class="details-head">Cell details</div>
    <div class="details-body" id="detailsBody"></div>
  </div>
</div>
<div class="tooltip" id="tooltip"></div>
<script>
const DATA = {data_json};
const MATRIX_CACHE_LIMIT = 2;
const state = {{
  layer: "all",
  head: "all",
  mode: "pan",
  x0: 0,
  x1: DATA.nCalls,
  y0: 0,
  y1: DATA.nBlocks,
  drag: null,
  pinned: null,
  hover: null,
  cache: new Map(),
}};

const freqCanvas = document.getElementById("freqCanvas");
const attnCanvas = document.getElementById("attnCanvas");
const segmentCanvas = document.getElementById("segmentCanvas");
const tooltip = document.getElementById("tooltip");
const detailsBody = document.getElementById("detailsBody");
const layerSelect = document.getElementById("layerSelect");
const headSelect = document.getElementById("headSelect");
const segmentSelect = document.getElementById("segmentSelect");
const tokenStart = document.getElementById("tokenStart");
const tokenEnd = document.getElementById("tokenEnd");
const callStart = document.getElementById("callStart");
const callEnd = document.getElementById("callEnd");

function init() {{
  document.getElementById("title").textContent =
    `${{DATA.label}} · ${{DATA.nCalls}} calls · ${{DATA.nBlocks}} blocks`;
  for (const layer of DATA.layers) {{
    layerSelect.add(new Option(layer.label, layer.key));
  }}
  for (const head of DATA.heads) {{
    headSelect.add(new Option(head.label, head.key));
  }}
  segmentSelect.add(new Option("choose segment", ""));
  for (const segment of DATA.segments) {{
    segmentSelect.add(new Option(segment.label, String(segment.idx)));
  }}
  wireControls();
  setFull();
}}

function wireControls() {{
  layerSelect.addEventListener("change", () => {{
    state.layer = layerSelect.value;
    render();
  }});
  headSelect.addEventListener("change", () => {{
    state.head = headSelect.value;
    render();
  }});
  document.getElementById("panBtn").onclick = () => setMode("pan");
  document.getElementById("boxBtn").onclick = () => setMode("box");
  document.getElementById("fullBtn").onclick = setFull;
  document.getElementById("recentBtn").onclick = setRecent;
  document.getElementById("lastCallBtn").onclick = setLastCall;
  document.getElementById("applyRangeBtn").onclick = applyRanges;
  document.getElementById("clearBtn").onclick = () => {{
    state.pinned = null;
    updateDetails(state.hover);
  }};
  segmentSelect.addEventListener("change", () => {{
    const idx = Number(segmentSelect.value);
    if (Number.isFinite(idx)) zoomSegment(idx);
  }});
  for (const canvas of [freqCanvas, attnCanvas]) {{
    canvas.addEventListener("wheel", onWheel, {{ passive: false }});
    canvas.addEventListener("mousedown", onMouseDown);
    canvas.addEventListener("mousemove", onMouseMove);
    canvas.addEventListener("mouseup", onMouseUp);
    canvas.addEventListener("mouseleave", onMouseLeave);
    canvas.addEventListener("click", onClick);
  }}
  segmentCanvas.addEventListener("click", onSegmentClick);
  window.addEventListener("resize", resizeCanvases);
  resizeCanvases();
}}

function setMode(mode) {{
  state.mode = mode;
  document.getElementById("panBtn").classList.toggle("active", mode === "pan");
  document.getElementById("boxBtn").classList.toggle("active", mode === "box");
  for (const canvas of [freqCanvas, attnCanvas]) {{
    canvas.classList.toggle("boxing", mode === "box");
  }}
}}

function resizeCanvases() {{
  for (const canvas of [freqCanvas, attnCanvas, segmentCanvas]) {{
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    canvas.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);
  }}
  render();
}}

function setFull() {{
  state.x0 = 0;
  state.x1 = DATA.nCalls;
  state.y0 = 0;
  state.y1 = DATA.nBlocks;
  syncInputs();
  render();
}}

function setRecent() {{
  const lastCall = DATA.callValues[DATA.callValues.length - 1];
  const keyLen = DATA.callKeyLen[String(lastCall)] || DATA.nBlocks * DATA.blockSize;
  const recentLo = Math.max(0, Math.floor((keyLen - DATA.recentWindow) / DATA.blockSize));
  const frontier = Math.min(DATA.nBlocks, Math.ceil(keyLen / DATA.blockSize));
  state.x0 = Math.max(0, DATA.nCalls - 8);
  state.x1 = DATA.nCalls;
  state.y0 = Math.max(0, recentLo - 8);
  state.y1 = Math.min(DATA.nBlocks, frontier + 8);
  syncInputs();
  render();
}}

function setLastCall() {{
  state.x0 = Math.max(0, DATA.nCalls - 1);
  state.x1 = DATA.nCalls;
  syncInputs();
  render();
}}

function applyRanges() {{
  const c0 = DATA.callValues.indexOf(Number(callStart.value));
  const c1 = DATA.callValues.indexOf(Number(callEnd.value));
  if (c0 >= 0 && c1 >= c0) {{
    state.x0 = c0;
    state.x1 = c1 + 1;
  }}
  const t0 = Number(tokenStart.value);
  const t1 = Number(tokenEnd.value);
  if (Number.isFinite(t0) && Number.isFinite(t1) && t1 > t0) {{
    state.y0 = clamp(Math.floor(t0 / DATA.blockSize), 0, DATA.nBlocks - 1);
    state.y1 = clamp(Math.ceil(t1 / DATA.blockSize), state.y0 + 1, DATA.nBlocks);
  }}
  syncInputs();
  render();
}}

function syncInputs() {{
  callStart.value = DATA.callValues[Math.floor(state.x0)] ?? DATA.callValues[0];
  callEnd.value = DATA.callValues[Math.max(0, Math.ceil(state.x1) - 1)] ?? DATA.callValues.at(-1);
  tokenStart.value = Math.floor(state.y0) * DATA.blockSize;
  tokenEnd.value = Math.ceil(state.y1) * DATA.blockSize;
}}

function zoomSegment(idx) {{
  const segment = DATA.segments[idx];
  if (!segment) return;
  state.y0 = Math.max(0, segment.blockLo - 4);
  state.y1 = Math.min(DATA.nBlocks, segment.blockHi + 4);
  syncInputs();
  render();
}}

function onSegmentClick(event) {{
  const y = event.offsetY;
  const block = state.y0 + (y / Math.max(1, segmentCanvas.clientHeight)) * (state.y1 - state.y0);
  const segment = DATA.segments.find(s => block >= s.blockLo && block < s.blockHi);
  if (segment) {{
    segmentSelect.value = String(segment.idx);
    zoomSegment(segment.idx);
  }}
}}

function onWheel(event) {{
  event.preventDefault();
  const canvas = event.currentTarget;
  const plot = plotRect(canvas);
  const mx = clamp((event.offsetX - plot.left) / plot.width, 0, 1);
  const my = clamp((event.offsetY - plot.top) / plot.height, 0, 1);
  const factor = event.deltaY < 0 ? 0.82 : 1.22;
  if (event.shiftKey) {{
    zoomRange("x", state.x0 + mx * (state.x1 - state.x0), factor);
  }} else {{
    zoomRange("y", state.y0 + my * (state.y1 - state.y0), factor);
  }}
  syncInputs();
  render();
}}

function zoomRange(axis, center, factor) {{
  const minSpan = axis === "x" ? 1 : 4;
  const max = axis === "x" ? DATA.nCalls : DATA.nBlocks;
  const a0 = axis === "x" ? state.x0 : state.y0;
  const a1 = axis === "x" ? state.x1 : state.y1;
  let span = clamp((a1 - a0) * factor, minSpan, max);
  let next0 = center - (center - a0) / (a1 - a0) * span;
  let next1 = next0 + span;
  if (next0 < 0) {{ next1 -= next0; next0 = 0; }}
  if (next1 > max) {{ next0 -= next1 - max; next1 = max; }}
  if (axis === "x") {{ state.x0 = next0; state.x1 = next1; }}
  else {{ state.y0 = next0; state.y1 = next1; }}
}}

function onMouseDown(event) {{
  const cell = eventToCell(event.currentTarget, event);
  state.drag = {{
    canvas: event.currentTarget,
    startX: event.offsetX,
    startY: event.offsetY,
    lastX: event.offsetX,
    lastY: event.offsetY,
    x0: state.x0,
    x1: state.x1,
    y0: state.y0,
    y1: state.y1,
    cell,
  }};
}}

function onMouseMove(event) {{
  const canvas = event.currentTarget;
  if (state.drag && state.drag.canvas === canvas) {{
    if (state.mode === "pan") {{
      const plot = plotRect(canvas);
      const dx = (event.offsetX - state.drag.lastX) / plot.width * (state.x1 - state.x0);
      const dy = (event.offsetY - state.drag.lastY) / plot.height * (state.y1 - state.y0);
      panBy(-dx, -dy);
      state.drag.lastX = event.offsetX;
      state.drag.lastY = event.offsetY;
      syncInputs();
      render();
    }} else {{
      render();
      drawBox(canvas, state.drag.startX, state.drag.startY, event.offsetX, event.offsetY);
    }}
    return;
  }}
  const cell = eventToCell(canvas, event);
  state.hover = cell;
  showTooltip(cell, event.clientX, event.clientY);
  updateDetails(state.pinned || cell);
  renderCrosshair(cell);
}}

function onMouseUp(event) {{
  if (!state.drag) return;
  const drag = state.drag;
  state.drag = null;
  if (state.mode === "box") {{
    const plot = plotRect(drag.canvas);
    const xA = clamp(Math.min(drag.startX, event.offsetX), plot.left, plot.left + plot.width);
    const xB = clamp(Math.max(drag.startX, event.offsetX), plot.left, plot.left + plot.width);
    const yA = clamp(Math.min(drag.startY, event.offsetY), plot.top, plot.top + plot.height);
    const yB = clamp(Math.max(drag.startY, event.offsetY), plot.top, plot.top + plot.height);
    if (xB - xA > 8 && yB - yA > 8) {{
      const old = {{x0: state.x0, x1: state.x1, y0: state.y0, y1: state.y1}};
      state.x0 = old.x0 + ((xA - plot.left) / plot.width) * (old.x1 - old.x0);
      state.x1 = old.x0 + ((xB - plot.left) / plot.width) * (old.x1 - old.x0);
      state.y0 = old.y0 + ((yA - plot.top) / plot.height) * (old.y1 - old.y0);
      state.y1 = old.y0 + ((yB - plot.top) / plot.height) * (old.y1 - old.y0);
    }}
  }}
  syncInputs();
  render();
}}

function onMouseLeave() {{
  tooltip.style.display = "none";
  state.hover = null;
  render();
}}

function onClick(event) {{
  const cell = eventToCell(event.currentTarget, event);
  if (cell) {{
    state.pinned = cell;
    updateDetails(cell);
  }}
}}

function panBy(dx, dy) {{
  const xSpan = state.x1 - state.x0;
  const ySpan = state.y1 - state.y0;
  state.x0 = clamp(state.x0 + dx, 0, DATA.nCalls - xSpan);
  state.x1 = state.x0 + xSpan;
  state.y0 = clamp(state.y0 + dy, 0, DATA.nBlocks - ySpan);
  state.y1 = state.y0 + ySpan;
}}

function render() {{
  drawHeatmap(freqCanvas, "freq");
  drawHeatmap(attnCanvas, "attn");
  drawSegments();
  updateDetails(state.pinned || state.hover);
}}

function drawHeatmap(canvas, metric) {{
  const ctx = canvas.getContext("2d");
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const plot = plotRect(canvas);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = DATA.colors.missing;
  ctx.fillRect(plot.left, plot.top, plot.width, plot.height);
  drawOverlays(ctx, plot);

  const matrix = matrixFor(metric);
  const xStart = Math.max(0, Math.floor(state.x0));
  const xEnd = Math.min(DATA.nCalls, Math.ceil(state.x1));
  const yStart = Math.max(0, Math.floor(state.y0));
  const yEnd = Math.min(DATA.nBlocks, Math.ceil(state.y1));
  const cellW = plot.width / (state.x1 - state.x0);
  const cellH = plot.height / (state.y1 - state.y0);
  for (let y = yStart; y < yEnd; y++) {{
    const py = plot.top + (y - state.y0) * cellH;
    const ph = Math.max(1, cellH + 0.35);
    for (let x = xStart; x < xEnd; x++) {{
      const value = matrix.values[y * DATA.nCalls + x];
      if (!Number.isFinite(value)) continue;
      const px = plot.left + (x - state.x0) * cellW;
      ctx.fillStyle = colorFor(metric, value / matrix.vmax);
      ctx.fillRect(px, py, Math.max(1, cellW + 0.35), ph);
    }}
  }}
  drawAxes(ctx, plot, metric);
  drawSelection(ctx, plot);
}}

function drawOverlays(ctx, plot) {{
  const cellW = plot.width / (state.x1 - state.x0);
  const cellH = plot.height / (state.y1 - state.y0);
  const sinkBlocks = Math.ceil(DATA.sinkSize / DATA.blockSize);
  if (sinkBlocks > state.y0) {{
    const y0 = plot.top;
    const y1 = plot.top + (Math.min(sinkBlocks, state.y1) - state.y0) * cellH;
    ctx.fillStyle = rgba(DATA.colors.sink, 0.30);
    ctx.fillRect(plot.left, y0, plot.width, Math.max(0, y1 - y0));
  }}
  for (let x = Math.floor(state.x0); x < Math.ceil(state.x1); x++) {{
    if (x < 0 || x >= DATA.nCalls) continue;
    const call = DATA.callValues[x];
    const keyLen = DATA.callKeyLen[String(call)];
    if (!keyLen) continue;
    const frontier = Math.min(DATA.nBlocks, Math.ceil(keyLen / DATA.blockSize));
    const recentLo = Math.max(0, Math.floor((keyLen - DATA.recentWindow) / DATA.blockSize));
    if (frontier <= state.y0 || recentLo >= state.y1) continue;
    const px = plot.left + (x - state.x0) * cellW;
    const py = plot.top + (Math.max(recentLo, state.y0) - state.y0) * cellH;
    const ph = (Math.min(frontier, state.y1) - Math.max(recentLo, state.y0)) * cellH;
    ctx.fillStyle = rgba(DATA.colors.recent, 0.34);
    ctx.fillRect(px, py, Math.max(1, cellW), Math.max(1, ph));
  }}
  ctx.strokeStyle = "rgba(34,34,34,0.35)";
  ctx.lineWidth = 1;
  for (const segment of DATA.segments) {{
    if (segment.blockLo < state.y0 || segment.blockLo > state.y1) continue;
    const y = plot.top + (segment.blockLo - state.y0) * cellH;
    ctx.beginPath();
    ctx.moveTo(plot.left, y);
    ctx.lineTo(plot.left + plot.width, y);
    ctx.stroke();
  }}
}}

function drawAxes(ctx, plot, metric) {{
  ctx.fillStyle = "#344563";
  ctx.font = "11px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const xTicks = tickValues(state.x0, state.x1, 8);
  for (const x of xTicks) {{
    const idx = Math.round(x);
    if (idx < 0 || idx >= DATA.nCalls) continue;
    const px = plot.left + (idx + 0.5 - state.x0) / (state.x1 - state.x0) * plot.width;
    ctx.fillText(String(DATA.callValues[idx]), px, 4);
  }}
  if (metric === "freq") {{
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const yTicks = tickValues(state.y0, state.y1, 9);
    for (const y of yTicks) {{
      const block = Math.round(y);
      if (block < 0 || block > DATA.nBlocks) continue;
      const py = plot.top + (block - state.y0) / (state.y1 - state.y0) * plot.height;
      ctx.fillText(String(block * DATA.blockSize), plot.left - 6, py);
    }}
  }}
  ctx.strokeStyle = "#667085";
  ctx.lineWidth = 1;
  ctx.strokeRect(plot.left, plot.top, plot.width, plot.height);
}}

function drawSegments() {{
  const canvas = segmentCanvas;
  const ctx = canvas.getContext("2d");
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, width, height);
  const visible = state.y1 - state.y0;
  for (const segment of DATA.segments) {{
    if (segment.blockHi < state.y0 || segment.blockLo > state.y1) continue;
    const y0 = (Math.max(segment.blockLo, state.y0) - state.y0) / visible * height;
    const y1 = (Math.min(segment.blockHi, state.y1) - state.y0) / visible * height;
    ctx.fillStyle = segment.color;
    ctx.fillRect(6, y0, 8, Math.max(1, y1 - y0));
    ctx.strokeStyle = "rgba(0,0,0,0.18)";
    ctx.beginPath();
    ctx.moveTo(18, y0);
    ctx.lineTo(width, y0);
    ctx.stroke();
    if (y1 - y0 >= 13) {{
      ctx.fillStyle = "#344054";
      ctx.font = "11px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      ctx.textBaseline = "middle";
      ctx.textAlign = "left";
      ctx.fillText(segment.label, 22, (y0 + y1) / 2, width - 26);
    }}
  }}
}}

function drawSelection(ctx, plot) {{
  const cell = state.pinned || state.hover;
  if (!cell) return;
  const cellW = plot.width / (state.x1 - state.x0);
  const cellH = plot.height / (state.y1 - state.y0);
  const px = plot.left + (cell.col - state.x0) * cellW;
  const py = plot.top + (cell.block - state.y0) * cellH;
  ctx.strokeStyle = state.pinned ? "#ffbf00" : "#203864";
  ctx.lineWidth = 2;
  ctx.strokeRect(px, py, Math.max(2, cellW), Math.max(2, cellH));
}}

function drawCrosshair(cell) {{
  render();
  if (!cell) return;
  for (const canvas of [freqCanvas, attnCanvas]) {{
    const ctx = canvas.getContext("2d");
    const plot = plotRect(canvas);
    const x = plot.left + (cell.col + 0.5 - state.x0) / (state.x1 - state.x0) * plot.width;
    const y = plot.top + (cell.block + 0.5 - state.y0) / (state.y1 - state.y0) * plot.height;
    ctx.strokeStyle = "rgba(32,56,100,0.35)";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(plot.left, y); ctx.lineTo(plot.left + plot.width, y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x, plot.top); ctx.lineTo(x, plot.top + plot.height); ctx.stroke();
  }}
}}

function drawBox(canvas, x0, y0, x1, y1) {{
  const ctx = canvas.getContext("2d");
  ctx.strokeStyle = "#203864";
  ctx.fillStyle = "rgba(32,56,100,0.12)";
  ctx.lineWidth = 2;
  ctx.fillRect(Math.min(x0,x1), Math.min(y0,y1), Math.abs(x1-x0), Math.abs(y1-y0));
  ctx.strokeRect(Math.min(x0,x1), Math.min(y0,y1), Math.abs(x1-x0), Math.abs(y1-y0));
}}

function matrixFor(metric) {{
  const key = metric === "freq" ? `freq:${{state.layer}}` : `attn:${{state.layer}}:${{state.head}}`;
  if (state.cache.has(key)) {{
    const cached = state.cache.get(key);
    state.cache.delete(key);
    state.cache.set(key, cached);
    return cached;
  }}
  const source = metric === "freq" ? DATA.freq[state.layer] : DATA.attn[state.layer][state.head];
  const values = new Float32Array(DATA.nBlocks * DATA.nCalls);
  values.fill(Number.NaN);
  for (let i = 0; i < source.indices.length; i++) values[source.indices[i]] = source.values[i];
  const matrix = {{ values, vmax: source.vmax || 1 }};
  state.cache.set(key, matrix);
  while (state.cache.size > MATRIX_CACHE_LIMIT) {{
    const oldest = state.cache.keys().next().value;
    state.cache.delete(oldest);
  }}
  return matrix;
}}

function eventToCell(canvas, event) {{
  const plot = plotRect(canvas);
  if (event.offsetX < plot.left || event.offsetX > plot.left + plot.width ||
      event.offsetY < plot.top || event.offsetY > plot.top + plot.height) return null;
  const col = Math.floor(state.x0 + (event.offsetX - plot.left) / plot.width * (state.x1 - state.x0));
  const block = Math.floor(state.y0 + (event.offsetY - plot.top) / plot.height * (state.y1 - state.y0));
  if (col < 0 || col >= DATA.nCalls || block < 0 || block >= DATA.nBlocks) return null;
  return cellDetails(block, col);
}}

function cellDetails(block, col) {{
  const idx = block * DATA.nCalls + col;
  const meta = DATA.cellMeta[String(idx)] || [block, block * DATA.blockSize, (block + 1) * DATA.blockSize, "", "", ""];
  const freq = matrixFor("freq").values[idx];
  const attn = matrixFor("attn").values[idx];
  return {{
    block,
    col,
    call: DATA.callValues[col],
    tokenStart: meta[1],
    tokenEnd: meta[2],
    role: meta[3],
    segment: meta[4],
    text: meta[5],
    freq,
    attn,
  }};
}}

function showTooltip(cell, clientX, clientY) {{
  if (!cell) {{
    tooltip.style.display = "none";
    return;
  }}
  tooltip.innerHTML = detailHtml(cell, true);
  tooltip.style.left = `${{Math.min(window.innerWidth - 430, clientX + 14)}}px`;
  tooltip.style.top = `${{Math.min(window.innerHeight - 180, clientY + 14)}}px`;
  tooltip.style.display = "block";
}}

function updateDetails(cell) {{
  if (!cell) {{
    detailsBody.innerHTML = "<div class='hint'>Hover or click a colored cell.</div>";
    return;
  }}
  detailsBody.innerHTML = detailHtml(cell, false);
}}

function detailHtml(cell, compact) {{
  const freq = Number.isFinite(cell.freq) ? cell.freq.toPrecision(5) : "";
  const attn = Number.isFinite(cell.attn) ? cell.attn.toPrecision(5) : "";
  const text = escapeHtml(cell.text || "");
  return `
    <div class="kv">
      <div>call</div><div>${{cell.call}}</div>
      <div>block</div><div>${{cell.block}}</div>
      <div>tokens</div><div>[${{cell.tokenStart}}, ${{cell.tokenEnd}})</div>
      <div>role</div><div>${{escapeHtml(cell.role || "")}}</div>
      <div>segment</div><div>${{escapeHtml(cell.segment || "")}}</div>
      <div>frequency</div><div>${{freq}}</div>
      <div>attention</div><div>${{attn}}</div>
    </div>
    ${{compact ? `<div>${{text}}</div>` : `<div class="text-box">${{text}}</div>`}}
  `;
}}

function plotRect(canvas) {{
  const left = canvas === freqCanvas ? 54 : 16;
  return {{
    left,
    top: 22,
    width: Math.max(1, canvas.clientWidth - left - 10),
    height: Math.max(1, canvas.clientHeight - 48),
  }};
}}

function colorFor(metric, t) {{
  t = clamp(t, 0, 1);
  if (metric === "freq") return ramp(t, [[247,251,255],[158,202,225],[8,81,156]]);
  return ramp(t, [[255,250,240],[253,174,97],[127,0,0]]);
}}

function ramp(t, stops) {{
  const scaled = t * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(scaled));
  const f = scaled - i;
  const a = stops[i], b = stops[i + 1];
  return `rgb(${{Math.round(a[0] + (b[0]-a[0])*f)}},${{Math.round(a[1] + (b[1]-a[1])*f)}},${{Math.round(a[2] + (b[2]-a[2])*f)}})`;
}}

function rgba(hex, alpha) {{
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${{(n >> 16) & 255}}, ${{(n >> 8) & 255}}, ${{n & 255}}, ${{alpha}})`;
}}

function tickValues(a, b, maxTicks) {{
  const span = b - a;
  const rough = span / maxTicks;
  const pow = Math.pow(10, Math.floor(Math.log10(Math.max(1, rough))));
  const step = [1,2,5,10].map(x => x * pow).find(x => x >= rough) || pow * 10;
  const out = [];
  for (let v = Math.ceil(a / step) * step; v <= b; v += step) out.push(v);
  return out;
}}

function clamp(v, lo, hi) {{ return Math.max(lo, Math.min(hi, v)); }}
function escapeHtml(v) {{
  return String(v).replace(/[&<>"']/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[c]));
}}

init();
</script>
</body>
</html>
"""


def _json_for_script(value: Any) -> str:
    return json.dumps(_json_ready(value), ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )


def _escape_text(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _load_detok(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    if not path.is_file():
        return {}
    out: dict[tuple[int, int], dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                call_idx = int(row["observed_call_idx"])
                block_id = int(row["block_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if not _truthy(row.get("align_ok")):
                continue
            out[(call_idx, block_id)] = row
    return out


def _call_values(rows: Sequence[dict[str, Any]]) -> list[int]:
    observed = [int(row["observed_call_idx"]) for row in rows]
    if not observed:
        raise ValueError("block_position HTML: no observed calls")
    return list(range(min(observed), max(observed) + 1))


def _segment_text(row: dict[str, Any]) -> str:
    lo = row.get("seg_lo")
    hi = row.get("seg_hi")
    role = row.get("role")
    tool = row.get("tool_name")
    parts = [f"seg {lo}" if lo == hi else f"seg {lo}-{hi}"]
    if role:
        parts.append(str(role))
    if tool:
        parts.append(f"tool:{tool}")
    return " / ".join(parts)


def _compact_text(value: Any, *, limit: int = 260) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).split())
    if len(text) > limit:
        text = text[: max(0, limit - 3)] + "..."
    return text


def _display_value(value: Any) -> str:
    return "" if value is None else str(value)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


if __name__ == "__main__":
    main()
