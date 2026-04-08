import { For, Show, createEffect, createMemo } from "solid-js";

import type { TracePayload } from "../api/client";
import type { HitCard } from "../canvas/hit";
import { flattenVisibleLanes } from "../canvas/layout";

interface SidebarProps {
  onPinLane: (card: HitCard) => void;
  onScroll?: (scrollTop: number) => void;
  scrollTop: number;
  traces: TracePayload[];
  visibility: Record<string, boolean>;
}

export default function Sidebar(props: SidebarProps) {
  const rows = createMemo(() => flattenVisibleLanes(props.traces, props.visibility));
  let shellEl!: HTMLElement;
  let suppressNextScroll = false;

  createEffect(() => {
    const target = props.scrollTop;
    if (!shellEl || shellEl.scrollTop === target) {
      return;
    }
    suppressNextScroll = true;
    shellEl.scrollTop = target;
  });

  const handleScroll = () => {
    if (suppressNextScroll) {
      suppressNextScroll = false;
      return;
    }
    props.onScroll?.(shellEl.scrollTop);
  };

  return (
    <aside class="sidebar-shell" onScroll={handleScroll} ref={shellEl}>
      <div class="sidebar-track">
        <div class="sidebar-time">TIME</div>
        <Show
          fallback={<div class="sidebar-empty">Load traces from the bar above.</div>}
          when={rows().length > 0}
        >
          <For each={rows()}>
            {(row) => (
              <button
                class="lane-label"
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
