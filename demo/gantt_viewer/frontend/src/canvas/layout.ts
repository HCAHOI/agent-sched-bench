import type { TracePayload } from "../api/client";
import type { ViewMode } from "../state/signals";

export const LANE_H = 60;
export const LANE_H_CONCISE = 26;
export const MARKER_H = 6;
export const SPAN_H = 18;
export const SPAN_PAD = 4;
export const TIME_AXIS_H = 28;

export interface LaneRow {
  laneAgentId: string;
  traceId: string;
  traceLabel: string;
  trace: TracePayload;
}

export function effectiveLaneH(viewMode: ViewMode): number {
  return viewMode === "concise" ? LANE_H_CONCISE : LANE_H;
}

export function flattenVisibleLanes(
  traces: TracePayload[],
  visible: Record<string, boolean>,
): LaneRow[] {
  return traces.flatMap((trace) => {
    if (visible[trace.id] === false) {
      return [];
    }
    return trace.lanes.map((lane) => ({
      laneAgentId: lane.agent_id,
      traceId: trace.id,
      traceLabel: trace.label,
      trace,
    }));
  });
}

export function computeTotalContentHeight(
  traces: TracePayload[],
  visible: Record<string, boolean>,
  viewMode: ViewMode,
  minHeight: number,
): number {
  return Math.max(
    TIME_AXIS_H + flattenVisibleLanes(traces, visible).length * effectiveLaneH(viewMode),
    minHeight,
  );
}
