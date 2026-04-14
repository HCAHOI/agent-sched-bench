import type { Registries, ResourceSample, TracePayload } from "../api/client";
import { displayColor } from "../theme/displayColor";
import { formatAbsTime } from "./time";

export interface SpanHit {
  item: TracePayload["lanes"][number]["spans"][number];
  kind: "span";
  laneAgentId: string;
  traceId: string;
  traceLabel: string;
}

export interface MarkerHit {
  item: TracePayload["lanes"][number]["markers"][number];
  kind: "marker";
  laneAgentId: string;
  traceId: string;
  traceLabel: string;
}

export interface LaneHit {
  kind: "lane";
  laneAgentId: string;
  trace: TracePayload;
  traceId: string;
  traceLabel: string;
}

export interface ResourceHit {
  kind: "resource";
  traceId: string;
  traceLabel: string;
  timeline: ResourceSample[];
  metric: string;
  metricSecondary?: string;
  chartY: number;
  chartH: number;
  chartPad: number;
  vMin: number;
  vMax: number;
  vMinSecondary?: number;
  vMaxSecondary?: number;
  /** Time range for X→time reverse mapping */
  timeMin: number;
  timeRange: number;
  canvasWidth: number;
  /** Populated during hover (not click) */
  hoveredSample?: ResourceSample;
  hoveredTime?: number;
}

export type Hit = LaneHit | MarkerHit | ResourceHit | SpanHit;

export interface HitCard {
  hit: Hit;
  x: number;
  y: number;
}

export function sameHit(a: Hit | null, b: Hit | null): boolean {
  if (!a || !b || a.kind !== b.kind || a.traceId !== b.traceId) {
    return false;
  }
  if (a.kind === "lane" && b.kind === "lane") {
    return a.laneAgentId === b.laneAgentId;
  }
  if (a.kind === "resource" && b.kind === "resource") {
    return a.traceId === b.traceId && a.metric === b.metric && a.metricSecondary === b.metricSecondary;
  }
  if (a.kind === "span" && b.kind === "span") {
    return (
      a.laneAgentId === b.laneAgentId &&
      a.item.type === b.item.type &&
      a.item.iteration === b.item.iteration &&
      a.item.start_abs === b.item.start_abs &&
      a.item.end_abs === b.item.end_abs
    );
  }
  if (a.kind === "marker" && b.kind === "marker") {
    return (
      a.laneAgentId === b.laneAgentId &&
      a.item.type === b.item.type &&
      a.item.event === b.item.event &&
      a.item.t_abs === b.item.t_abs
    );
  }
  return false;
}

export const RESOURCE_METRIC_COLORS: Record<string, string> = {
  cpu: "#00E5FF",
  memory: "#76FF03",
  disk_io: "#FF6D00",
  net_io: "#AB47BC",
};

const RESOURCE_METRIC_LABELS: Record<string, string> = {
  cpu: "CPU %",
  memory: "Memory MB",
  disk_io: "Disk I/O MB",
  net_io: "Network I/O MB",
};

export function hitAccent(hit: Hit, registries: Registries | null): string {
  if (hit.kind === "lane") {
    return displayColor("#00E5FF");
  }
  if (hit.kind === "resource") {
    return displayColor(RESOURCE_METRIC_COLORS[hit.metric] ?? "#94A3B8");
  }
  if (hit.kind === "span") {
    return displayColor(registries?.spans[hit.item.type]?.color ?? "#94A3B8");
  }
  const marker = registries?.markers[hit.item.event] ?? registries?.markers[hit.item.type];
  return displayColor(marker?.color ?? "#94A3B8");
}

export function hitTitle(hit: Hit, registries: Registries | null): string {
  if (hit.kind === "lane") {
    return `Trace ${hit.traceLabel}`;
  }
  if (hit.kind === "resource") {
    const primary = RESOURCE_METRIC_LABELS[hit.metric] ?? hit.metric;
    const secondary = hit.metricSecondary ? (RESOURCE_METRIC_LABELS[hit.metricSecondary] ?? hit.metricSecondary) : null;
    return secondary && secondary !== primary ? `Resource · ${primary} + ${secondary}` : `Resource · ${primary}`;
  }
  if (hit.kind === "span") {
    const label = registries?.spans[hit.item.type]?.label ?? hit.item.type;
    return `${label} · Iter ${hit.item.iteration}`;
  }
  return `${hit.item.event} · Iter ${hit.item.iteration}`;
}

export function hitRows(hit: Hit): Array<[string, string]> {
  if (hit.kind === "resource") {
    const s = hit.hoveredSample;
    if (s) {
      const rows: Array<[string, string]> = [
        ["trace", hit.traceLabel],
        ["CPU", `${s.cpu_percent.toFixed(1)}%`],
        ["Memory", `${s.memory_mb.toFixed(1)} MB`],
      ];
      if (s.disk_read_mb != null || s.disk_write_mb != null) {
        rows.push(["Disk R/W", `${(s.disk_read_mb ?? 0).toFixed(1)} / ${(s.disk_write_mb ?? 0).toFixed(1)} MB`]);
      }
      if (s.net_rx_mb != null || s.net_tx_mb != null) {
        rows.push(["Net Rx/Tx", `${(s.net_rx_mb ?? 0).toFixed(1)} / ${(s.net_tx_mb ?? 0).toFixed(1)} MB`]);
      }
      if (s.context_switches != null) {
        rows.push(["Ctx Switches", String(s.context_switches)]);
      }
      return rows;
    }
    // Fallback: no hover sample (e.g. pinned card)
    const rows: Array<[string, string]> = [
      ["trace", hit.traceLabel],
      [RESOURCE_METRIC_LABELS[hit.metric] ?? hit.metric, `${hit.vMin.toFixed(1)} – ${hit.vMax.toFixed(1)}`],
    ];
    if (hit.metricSecondary && hit.metricSecondary !== hit.metric) {
      rows.push(
        [RESOURCE_METRIC_LABELS[hit.metricSecondary] ?? hit.metricSecondary, `${(hit.vMinSecondary ?? 0).toFixed(1)} – ${(hit.vMaxSecondary ?? 0).toFixed(1)}`],
      );
    }
    rows.push(["samples", String(hit.timeline.length)]);
    return rows;
  }
  if (hit.kind === "lane") {
    const metadata = hit.trace.metadata;
    const rows: Array<[string, string]> = [
      ["trace", hit.traceLabel],
      ["agent", hit.laneAgentId],
      ["scaffold", metadata.scaffold],
      ["model", metadata.model ?? "?"],
      ["instance", metadata.instance_id || "?"],
      ["actions", String(metadata.n_actions)],
      ["iterations", String(metadata.n_iterations)],
      ["events", String(metadata.n_events)],
      ["elapsed", `${(metadata.elapsed_s ?? 0).toFixed(2)} s`],
      ["lanes", String(hit.trace.lanes.length)],
    ];
    if (metadata.mode) {
      rows.splice(4, 0, ["mode", metadata.mode]);
    }
    return rows;
  }

  if (hit.kind === "span") {
    const rows: Array<[string, string]> = [
      ["trace", hit.traceLabel],
      ["agent", hit.laneAgentId],
      ["duration", `${((hit.item.end_abs - hit.item.start_abs) * 1000).toFixed(1)} ms`],
      ["start", formatAbsTime(hit.item.start_abs)],
      ["relative", `+${hit.item.start.toFixed(3)}s`],
    ];
    for (const [key, value] of Object.entries(hit.item.detail ?? {})) {
      rows.push([key, formatDetailValue(value)]);
    }
    return rows;
  }

  const rows: Array<[string, string]> = [
    ["trace", hit.traceLabel],
    ["agent", hit.laneAgentId],
    ["event", hit.item.event],
    ["time", formatAbsTime(hit.item.t_abs)],
    ["relative", `+${hit.item.t.toFixed(3)}s`],
  ];
  for (const [key, value] of Object.entries(hit.item.detail ?? {})) {
    rows.push([key, formatDetailValue(value)]);
  }
  return rows;
}

type TimeMode = "sync" | "abs";
type ClockMode = "wall" | "real";

function selectResourceTime(sample: ResourceSample, timeMode: TimeMode, clockMode: ClockMode): number {
  if (clockMode === "real") {
    const realT = timeMode === "sync" ? sample.t_real : sample.t_real_abs;
    if (typeof realT === "number" && Number.isFinite(realT)) return realT;
  }
  return timeMode === "sync" ? sample.t : sample.t_abs;
}

export function findNearestSample(
  timeline: ResourceSample[],
  targetTime: number,
  timeMode: TimeMode,
  clockMode: ClockMode,
): ResourceSample {
  if (timeline.length === 0) return timeline[0];
  let best = 0;
  let bestDist = Math.abs(selectResourceTime(timeline[0], timeMode, clockMode) - targetTime);
  // Binary search for efficiency on large timelines
  let lo = 0;
  let hi = timeline.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const t = selectResourceTime(timeline[mid], timeMode, clockMode);
    const dist = Math.abs(t - targetTime);
    if (dist < bestDist) {
      bestDist = dist;
      best = mid;
    }
    if (t < targetTime) lo = mid + 1;
    else if (t > targetTime) hi = mid - 1;
    else break;
  }
  return timeline[best];
}

export function formatDetailValue(value: unknown): string {
  if (Array.isArray(value)) {
    return value.map((item) => String(item)).join("\n");
  }
  if (value && typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}
