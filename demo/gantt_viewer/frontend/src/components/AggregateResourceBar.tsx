import { createEffect, createMemo, createSignal, Show } from "solid-js";

import type { ResourceSample, TracePayload } from "../api/client";
import { RESOURCE_METRIC_COLORS, findNearestSample } from "../canvas/hit";
import { displayColor } from "../theme/displayColor";
import { canvasTimeRange } from "../state/signals";
import type { ClockMode, ResourceMetric, ThemeMode, TimeMode } from "../state/signals";

interface AggregateResourceBarProps {
  traces: TracePayload[];
  visibility: Record<string, boolean>;
  resourceMetric: ResourceMetric;
  resourceMetricSecondary: ResourceMetric;
  showResourceChart: boolean;
  timeMode: TimeMode;
  clockMode: ClockMode;
  themeMode: ThemeMode;
}

const BAR_HEIGHT = 80;
const CHART_PAD = 4;

function extractValue(sample: ResourceSample, metric: ResourceMetric): number {
  switch (metric) {
    case "cpu": return sample.cpu_percent;
    case "memory": return sample.memory_mb;
    case "disk_io": return (sample.disk_read_mb ?? 0) + (sample.disk_write_mb ?? 0);
    case "net_io": return (sample.net_rx_mb ?? 0) + (sample.net_tx_mb ?? 0);
    case "none": return 0;
  }
}

function metricUnit(metric: ResourceMetric): string {
  switch (metric) {
    case "cpu": return "%";
    case "memory": return "MB";
    case "disk_io": return "MB";
    case "net_io": return "MB";
    case "none": return "";
  }
}

function selectTime(sample: ResourceSample, timeMode: TimeMode, clockMode: ClockMode): number {
  if (clockMode === "real") {
    const realT = timeMode === "sync" ? sample.t_real : sample.t_real_abs;
    if (typeof realT === "number" && Number.isFinite(realT)) return realT;
  }
  return timeMode === "sync" ? sample.t : sample.t_abs;
}

/** Binary-search the nearest sample, then linearly interpolate. */
function interpolateSample(
  timeline: ResourceSample[],
  times: number[],
  targetT: number,
): ResourceSample {
  if (targetT <= times[0]) return timeline[0];
  if (targetT >= times[times.length - 1]) return timeline[timeline.length - 1];

  let lo = 0;
  let hi = times.length - 1;
  while (lo < hi - 1) {
    const mid = (lo + hi) >> 1;
    if (times[mid] <= targetT) lo = mid;
    else hi = mid;
  }

  const t0 = times[lo];
  const t1 = times[hi];
  const frac = t1 === t0 ? 0 : (targetT - t0) / (t1 - t0);
  const s0 = timeline[lo];
  const s1 = timeline[hi];

  const lerp = (a: number, b: number) => a + (b - a) * frac;
  return {
    t: targetT,
    t_abs: targetT,
    t_real: targetT,
    t_real_abs: targetT,
    cpu_percent: lerp(s0.cpu_percent, s1.cpu_percent),
    memory_mb: lerp(s0.memory_mb, s1.memory_mb),
    disk_read_mb: lerp(s0.disk_read_mb ?? 0, s1.disk_read_mb ?? 0),
    disk_write_mb: lerp(s0.disk_write_mb ?? 0, s1.disk_write_mb ?? 0),
    net_rx_mb: lerp(s0.net_rx_mb ?? 0, s1.net_rx_mb ?? 0),
    net_tx_mb: lerp(s0.net_tx_mb ?? 0, s1.net_tx_mb ?? 0),
  };
}

function aggregateTimelines(
  traces: TracePayload[],
  visibility: Record<string, boolean>,
  timeMode: TimeMode,
  clockMode: ClockMode,
): ResourceSample[] {
  const valid = traces.filter(
    (t) => visibility[t.id] !== false && t.resource_timeline?.length,
  );
  if (valid.length === 0) return [];

  // Clip each trace's resource timeline to its last action end time
  function selectSpanEnd(span: { end: number; end_abs: number; end_real?: number | null; end_real_abs?: number | null }): number {
    if (clockMode === "real") {
      const r = timeMode === "sync" ? span.end_real : span.end_real_abs;
      if (typeof r === "number" && Number.isFinite(r)) return r;
    }
    return timeMode === "sync" ? span.end : span.end_abs;
  }

  const clippedTimelines = valid.map((t) => {
    let lastEnd = -Infinity;
    for (const lane of t.lanes) {
      for (const span of lane.spans) lastEnd = Math.max(lastEnd, selectSpanEnd(span));
    }
    if (!Number.isFinite(lastEnd)) return t.resource_timeline!;
    const cutoff = lastEnd + 60;
    return t.resource_timeline!.filter((s) => selectTime(s, timeMode, clockMode) <= cutoff);
  }).filter((tl) => tl.length > 0);

  if (clippedTimelines.length === 0) return [];
  if (clippedTimelines.length === 1) return clippedTimelines[0];

  // Pre-compute per-trace time arrays
  const traceTimes = clippedTimelines.map((tl) =>
    tl.map((s) => selectTime(s, timeMode, clockMode)),
  );

  // Collect all unique timestamps, down-sample to ~500 points max for perf
  const allTimes = new Set<number>();
  for (const times of traceTimes) {
    for (const t of times) allTimes.add(t);
  }
  let sorted = [...allTimes].sort((a, b) => a - b);
  if (sorted.length > 500) {
    const step = Math.ceil(sorted.length / 500);
    sorted = sorted.filter((_, i) => i % step === 0 || i === sorted.length - 1);
  }

  return sorted.map((t) => {
    let cpu = 0;
    let mem = 0;
    let dr = 0;
    let dw = 0;
    let nr = 0;
    let nt = 0;
    for (let i = 0; i < clippedTimelines.length; i++) {
      const s = interpolateSample(clippedTimelines[i], traceTimes[i], t);
      cpu += s.cpu_percent;
      mem += s.memory_mb;
      dr += s.disk_read_mb ?? 0;
      dw += s.disk_write_mb ?? 0;
      nr += s.net_rx_mb ?? 0;
      nt += s.net_tx_mb ?? 0;
    }
    return {
      t,
      t_abs: t,
      t_real: t,
      t_real_abs: t,
      cpu_percent: cpu,
      memory_mb: mem,
      disk_read_mb: dr,
      disk_write_mb: dw,
      net_rx_mb: nr,
      net_tx_mb: nt,
    };
  });
}

function drawMetricOverlay(
  ctx: CanvasRenderingContext2D,
  timeline: ResourceSample[],
  metric: ResourceMetric,
  timeMode: TimeMode,
  clockMode: ClockMode,
  y: number,
  h: number,
  pad: number,
  canvasW: number,
  side: "left" | "right",
  externalTimeMin?: number,
  externalTimeRange?: number,
): void {
  const values = timeline.map((s) => extractValue(s, metric));
  let vMin = Number.POSITIVE_INFINITY;
  let vMax = Number.NEGATIVE_INFINITY;
  for (const v of values) {
    if (v < vMin) vMin = v;
    if (v > vMax) vMax = v;
  }
  if (vMin === vMax) vMax = vMin + 1;
  const vRange = vMax - vMin;
  const innerH = h - pad * 2;
  const baselineY = y + pad + innerH;

  // Time → X mapping: use external (shared) range if provided
  const times = timeline.map((s) => selectTime(s, timeMode, clockMode));
  let tMin: number;
  let tRange: number;
  if (externalTimeMin != null && externalTimeRange != null) {
    tMin = externalTimeMin;
    tRange = externalTimeRange;
  } else {
    tMin = times[0];
    const tMax = times[times.length - 1];
    tRange = tMax - tMin || 1;
  }
  const timeToX = (t: number) => ((t - tMin) / tRange) * canvasW;

  const points = times.map((t, i) => ({
    x: timeToX(t),
    y: y + pad + innerH - ((values[i] - vMin) / vRange) * innerH,
  }));

  if (points.length === 0) return;

  const color = displayColor(RESOURCE_METRIC_COLORS[metric] ?? "#94A3B8");

  // Area fill
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
  ctx.lineTo(points[points.length - 1].x, baselineY);
  ctx.lineTo(points[0].x, baselineY);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.globalAlpha = side === "left" ? 0.25 : 0.18;
  ctx.fill();
  ctx.globalAlpha = 1;

  // Stroke
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.globalAlpha = 0.8;
  ctx.stroke();
  ctx.globalAlpha = 1;
  ctx.lineWidth = 1;

  // Y-axis labels
  ctx.fillStyle = color;
  ctx.font = '9px "JetBrains Mono", monospace';
  const unit = metricUnit(metric);
  if (side === "left") {
    ctx.textAlign = "left";
    ctx.fillText(`${vMax.toFixed(1)}${unit}`, 4, y + pad + 8);
    ctx.fillText(`${vMin.toFixed(1)}${unit}`, 4, y + h - pad - 2);
  } else {
    ctx.textAlign = "right";
    ctx.fillText(`${vMax.toFixed(1)}${unit}`, canvasW - 4, y + pad + 8);
    ctx.fillText(`${vMin.toFixed(1)}${unit}`, canvasW - 4, y + h - pad - 2);
  }
}

export default function AggregateResourceBar(props: AggregateResourceBarProps) {
  let canvasEl!: HTMLCanvasElement;
  let wrapEl!: HTMLDivElement;

  const [expanded, setExpanded] = createSignal(false);
  const [hoverInfo, setHoverInfo] = createSignal<{ x: number; y: number; sample: ResourceSample } | null>(null);

  const aggregated = createMemo(() =>
    aggregateTimelines(props.traces, props.visibility, props.timeMode, props.clockMode),
  );

  const hasAnyMetric = () => props.resourceMetric !== "none" || props.resourceMetricSecondary !== "none";
  const hasData = () => aggregated().length > 0 && props.showResourceChart && hasAnyMetric();

  function render() {
    if (!canvasEl || !wrapEl) return;
    const data = aggregated();
    if (data.length === 0) return;

    const dpr = window.devicePixelRatio || 1;
    const w = wrapEl.clientWidth;
    const h = BAR_HEIGHT;
    if (w <= 0) return;
    canvasEl.width = w * dpr;
    canvasEl.height = h * dpr;
    canvasEl.style.width = `${w}px`;
    canvasEl.style.height = `${h}px`;

    const ctx = canvasEl.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    // Background
    const isDark = props.themeMode === "dark";
    ctx.fillStyle = isDark ? "#1a1f2e" : "#f0f2f5";
    ctx.fillRect(0, 0, w, h);

    // Use shared time range from main canvas for alignment
    const tr = canvasTimeRange();
    const tMin = tr?.minTime;
    const tRange = tr?.totalRange;

    // Primary metric
    if (props.resourceMetric !== "none") {
      drawMetricOverlay(ctx, data, props.resourceMetric, props.timeMode, props.clockMode, 0, h, CHART_PAD, w, "left", tMin, tRange);
    }

    // Secondary metric (skip if same or none)
    if (props.resourceMetricSecondary !== "none" && props.resourceMetricSecondary !== props.resourceMetric) {
      drawMetricOverlay(ctx, data, props.resourceMetricSecondary, props.timeMode, props.clockMode, 0, h, CHART_PAD, w, "right", tMin, tRange);
    }
  }

  function scheduleRender() {
    requestAnimationFrame(() => render());
  }

  createEffect(() => {
    if (!expanded() || !hasData()) return;
    // Touch reactive deps
    aggregated();
    canvasTimeRange();
    props.resourceMetric;
    props.resourceMetricSecondary;
    props.themeMode;
    scheduleRender();
  });

  return (
    <>
      <Show when={hasData()}>
        <button
          class="aggregate-resource-toggle"
          onClick={() => setExpanded((v) => !v)}
          title={expanded() ? "Collapse aggregate" : "Expand aggregate"}
        >
          {expanded() ? "\u25BC" : "\u25B2"}
        </button>
      </Show>
      <Show when={hasData() && expanded()}>
        <div
          class="aggregate-resource-bar"
          ref={(el) => { wrapEl = el; scheduleRender(); }}
          onMouseMove={(e) => {
            const data = aggregated();
            if (!data.length || !wrapEl) return;
            const rect = wrapEl.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const w = rect.width;
            // Use shared time range for alignment
            const tr = canvasTimeRange();
            let tMin: number;
            let tRange: number;
            if (tr) {
              tMin = tr.minTime;
              tRange = tr.totalRange;
            } else {
              const times = data.map((s) => selectTime(s, props.timeMode, props.clockMode));
              tMin = times[0];
              tRange = (times[times.length - 1] - tMin) || 1;
            }
            const time = tMin + (mx / w) * tRange;
            const sample = findNearestSample(data, time, props.timeMode, props.clockMode);
            setHoverInfo({ x: e.clientX, y: e.clientY, sample });
          }}
          onMouseLeave={() => setHoverInfo(null)}
        >
          <canvas ref={canvasEl} />
        </div>
        <Show when={hoverInfo()}>
          {(info) => (
            <div
              class="aggregate-hover-tooltip"
              style={{
                left: `${info().x + 12}px`,
                top: `${info().y - 80}px`,
              }}
            >
              <strong>Aggregate</strong>
              <div>CPU: {info().sample.cpu_percent.toFixed(1)}%</div>
              <div>Mem: {info().sample.memory_mb.toFixed(1)} MB</div>
              {info().sample.disk_read_mb != null && (
                <div>Disk R/W: {(info().sample.disk_read_mb ?? 0).toFixed(1)} / {(info().sample.disk_write_mb ?? 0).toFixed(1)} MB</div>
              )}
              {info().sample.net_rx_mb != null && (
                <div>Net Rx/Tx: {(info().sample.net_rx_mb ?? 0).toFixed(1)} / {(info().sample.net_tx_mb ?? 0).toFixed(1)} MB</div>
              )}
            </div>
          )}
        </Show>
      </Show>
    </>
  );
}
