import type { Registries, TracePayload } from "../api/client";

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

export type Hit = LaneHit | MarkerHit | SpanHit;

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

export function hitAccent(hit: Hit, registries: Registries | null): string {
  if (hit.kind === "lane") {
    return "#00E5FF";
  }
  if (hit.kind === "span") {
    return registries?.spans[hit.item.type]?.color ?? "#94A3B8";
  }
  const marker = registries?.markers[hit.item.event] ?? registries?.markers[hit.item.type];
  return marker?.color ?? "#94A3B8";
}

export function hitTitle(hit: Hit, registries: Registries | null): string {
  if (hit.kind === "lane") {
    return `Trace ${hit.traceLabel}`;
  }
  if (hit.kind === "span") {
    const label = registries?.spans[hit.item.type]?.label ?? hit.item.type;
    return `${label} · Iter ${hit.item.iteration}`;
  }
  return `${hit.item.event} · Iter ${hit.item.iteration}`;
}

export function hitRows(hit: Hit): Array<[string, string]> {
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
    ["relative", `+${hit.item.t.toFixed(3)}s`],
  ];
  for (const [key, value] of Object.entries(hit.item.detail ?? {})) {
    rows.push([key, formatDetailValue(value)]);
  }
  return rows;
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
