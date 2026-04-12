import type { SnapshotBootstrapData, TraceDescriptor, TracePayload } from "../api/client";
import {
  DEFAULT_CLOCK_MODE,
  DEFAULT_THEME_MODE,
  DEFAULT_TIME_MODE,
  DEFAULT_VIEW_MODE,
  DEFAULT_ZOOM,
  type ClockMode,
  type ThemeMode,
  type TimeMode,
  type ViewMode,
} from "../state/signals";

const SNAPSHOT_BOOTSTRAP_ELEMENT_ID = "gantt-viewer-snapshot-bootstrap";

export const SNAPSHOT_DEFAULTS: {
  clockMode: ClockMode;
  themeMode: ThemeMode;
  timeMode: TimeMode;
  viewMode: ViewMode;
  zoom: number;
} = {
  clockMode: DEFAULT_CLOCK_MODE,
  themeMode: DEFAULT_THEME_MODE,
  timeMode: DEFAULT_TIME_MODE,
  viewMode: DEFAULT_VIEW_MODE,
  zoom: DEFAULT_ZOOM,
};

declare global {
  interface Window {
    __GANTT_VIEWER_BOOTSTRAP__?: SnapshotBootstrapData;
  }
}

function readBootstrapScriptPayload(): unknown {
  if (typeof document === "undefined") {
    return null;
  }
  const element = document.getElementById(SNAPSHOT_BOOTSTRAP_ELEMENT_ID);
  if (!(element instanceof HTMLScriptElement) || element.textContent === null) {
    return null;
  }
  try {
    return JSON.parse(element.textContent);
  } catch {
    return null;
  }
}

function isTracePayloadArray(value: unknown): value is TracePayload[] {
  return Array.isArray(value) && value.every((trace) => typeof trace?.id === "string");
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((entry) => typeof entry === "string");
}

function isSnapshotBootstrapData(value: unknown): value is SnapshotBootstrapData {
  if (typeof value !== "object" || value === null) {
    return false;
  }

  const candidate = value as Partial<SnapshotBootstrapData>;
  const payload = candidate.payload;
  return (
    candidate.mode === "snapshot" &&
    typeof payload === "object" &&
    payload !== null &&
    typeof payload.registries === "object" &&
    payload.registries !== null &&
    isTracePayloadArray(payload.traces) &&
    isStringArray(candidate.trace_ids) &&
    isStringArray(candidate.visible_trace_ids)
  );
}

export function readSnapshotBootstrap(): SnapshotBootstrapData | null {
  const globalBootstrap = typeof window === "undefined" ? null : window.__GANTT_VIEWER_BOOTSTRAP__;
  if (isSnapshotBootstrapData(globalBootstrap)) {
    return globalBootstrap;
  }

  const scriptBootstrap = readBootstrapScriptPayload();
  if (!isSnapshotBootstrapData(scriptBootstrap)) {
    return null;
  }

  if (typeof window !== "undefined") {
    window.__GANTT_VIEWER_BOOTSTRAP__ = scriptBootstrap;
  }
  return scriptBootstrap;
}

export function snapshotDescriptorsFromTraces(traces: TracePayload[]): TraceDescriptor[] {
  return traces.map((trace, index) => ({
    id: trace.id,
    label: trace.label,
    mtime: index,
    path: `snapshot:${trace.id}`,
    size_bytes: 0,
    source_format: "trace",
  }));
}

export function visibilityFromTraceIds(
  traceIds: string[],
  visibleTraceIds: string[] = traceIds,
): Record<string, boolean> {
  const visibleSet = new Set(visibleTraceIds);
  return Object.fromEntries(traceIds.map((traceId) => [traceId, visibleSet.has(traceId)]));
}
