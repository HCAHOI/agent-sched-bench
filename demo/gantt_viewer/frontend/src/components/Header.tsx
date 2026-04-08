import type { Accessor } from "solid-js";

import type { TimeMode, ViewMode } from "../state/signals";

interface HeaderProps {
  loadedCount: Accessor<number>;
  summary: Accessor<string>;
  onTimeModeChange: (mode: TimeMode) => void;
  onViewModeChange: (mode: ViewMode) => void;
  timeMode: Accessor<TimeMode>;
  viewMode: Accessor<ViewMode>;
  zoom: Accessor<number>;
}

export default function Header(props: HeaderProps) {
  return (
    <header class="toolbar-card">
      <div>
        <p class="eyebrow">Dynamic Gantt Viewer</p>
        <h1>Timeline workspace</h1>
        <p class="lede compact">
          Render payloads on an imperative canvas while Solid owns the controls.
        </p>
        <p class="toolbar-meta">{props.summary()}</p>
      </div>

      <div class="toolbar-groups">
        <div class="toggle-group">
          <span class="metric-label">time</span>
          <button
            classList={{ active: props.timeMode() === "sync" }}
            onClick={() => props.onTimeModeChange("sync")}
            type="button"
          >
            Sync
          </button>
          <button
            classList={{ active: props.timeMode() === "abs" }}
            onClick={() => props.onTimeModeChange("abs")}
            type="button"
          >
            Abs
          </button>
        </div>

        <div class="toggle-group">
          <span class="metric-label">layout</span>
          <button
            classList={{ active: props.viewMode() === "layered" }}
            onClick={() => props.onViewModeChange("layered")}
            type="button"
          >
            Layered
          </button>
          <button
            classList={{ active: props.viewMode() === "concise" }}
            onClick={() => props.onViewModeChange("concise")}
            type="button"
          >
            Concise
          </button>
        </div>

        <div class="metric-block">
          <span class="metric-label">zoom</span>
          <strong>{Math.round(props.zoom() * 100)}%</strong>
        </div>

        <div class="metric-block">
          <span class="metric-label">loaded</span>
          <strong>{props.loadedCount()}</strong>
        </div>
      </div>
    </header>
  );
}
