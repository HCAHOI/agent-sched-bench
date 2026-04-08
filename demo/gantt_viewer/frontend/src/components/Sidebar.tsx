import { For, Show, createMemo } from "solid-js";

import type { TracePayload } from "../api/client";
import type { HitCard } from "../canvas/hit";
import { flattenVisibleLanes, TIME_AXIS_H, effectiveLaneH } from "../canvas/layout";
import type { ViewMode } from "../state/signals";

interface SidebarProps {
  onPinLane: (card: HitCard) => void;
  scrollTop: number;
  traces: TracePayload[];
  viewMode: ViewMode;
  visibility: Record<string, boolean>;
}

export default function Sidebar(props: SidebarProps) {
  const rows = createMemo(() => flattenVisibleLanes(props.traces, props.visibility));

  return (
    <aside class="sidebar-shell">
      <div
        class="sidebar-track"
        style={{ transform: `translateY(-${props.scrollTop}px)` }}
      >
        <div class="sidebar-time" style={{ height: `${TIME_AXIS_H}px` }}>
          Time
        </div>

        <Show
          fallback={<div class="sidebar-empty">Loaded lane labels appear here.</div>}
          when={rows().length > 0}
        >
          <For each={rows()}>
            {(row) => (
              <button
                class="lane-label"
                classList={{ concise: props.viewMode === "concise" }}
                onClick={(event) =>
                  props.onPinLane({
                    hit: {
                      kind: "lane",
                      laneAgentId: row.laneAgentId,
                      trace: row.trace,
                      traceId: row.traceId,
                      traceLabel: row.traceLabel,
                    },
                    x: event.clientX,
                    y: event.clientY,
                  })
                }
                style={{ height: `${effectiveLaneH(props.viewMode)}px` }}
                type="button"
              >
                <strong>{row.traceLabel}</strong>
                <span>{row.trace.metadata.scaffold}</span>
                <span>{row.laneAgentId}</span>
              </button>
            )}
          </For>
        </Show>
      </div>
    </aside>
  );
}
