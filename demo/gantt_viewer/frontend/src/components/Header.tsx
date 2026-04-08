import { For } from "solid-js";
import type { Accessor } from "solid-js";

import { ZOOM_PRESETS } from "../canvas/CanvasRenderer";
import type { TimeMode, ViewMode } from "../state/signals";

interface HeaderProps {
  loadedCount: Accessor<number>;
  summary: Accessor<string>;
  onTimeModeChange: (mode: TimeMode) => void;
  onViewModeChange: (mode: ViewMode) => void;
  onZoomChange: (factor: number) => void;
  timeMode: Accessor<TimeMode>;
  viewMode: Accessor<ViewMode>;
  zoom: Accessor<number>;
}

function formatZoom(factor: number): string {
  return factor >= 1 ? `${factor}x` : `${factor}x`;
}

export default function Header(props: HeaderProps) {
  const currentPresetValue = () => {
    const z = props.zoom();
    const match = ZOOM_PRESETS.find((p) => Math.abs(p - z) < 0.001);
    return match !== undefined ? String(match) : "";
  };

  return (
    <header class="toolbar-card">
      <span class="toolbar-title">TRACE GANTT</span>
      <span class="toolbar-meta-inline" title={props.summary()}>
        {props.summary()}
      </span>
      <label class="zoom-select-wrap" title="Zoom presets (Ctrl+wheel for free zoom)">
        <span class="zoom-select-label">zoom</span>
        <select
          class="zoom-select"
          value={currentPresetValue()}
          onChange={(event) => {
            const value = event.currentTarget.value;
            if (value) props.onZoomChange(Number(value));
          }}
        >
          {currentPresetValue() === "" && (
            <option value="" disabled>
              {Math.round(props.zoom() * 100)}%
            </option>
          )}
          <For each={ZOOM_PRESETS}>
            {(preset) => <option value={String(preset)}>{formatZoom(preset)}</option>}
          </For>
        </select>
      </label>
      <span class="toolbar-zoom">
        loaded {props.loadedCount()}
      </span>
      <div class="toggle-group">
        <button
          classList={{ active: props.timeMode() === "sync" }}
          onClick={() => props.onTimeModeChange("sync")}
          type="button"
        >
          SYNC
        </button>
        <button
          classList={{ active: props.timeMode() === "abs" }}
          onClick={() => props.onTimeModeChange("abs")}
          type="button"
        >
          ABS
        </button>
      </div>
      <div class="toggle-group">
        <button
          classList={{ active: props.viewMode() === "layered" }}
          onClick={() => props.onViewModeChange("layered")}
          type="button"
        >
          LAYER
        </button>
        <button
          classList={{ active: props.viewMode() === "concise" }}
          onClick={() => props.onViewModeChange("concise")}
          type="button"
        >
          CONCISE
        </button>
      </div>
    </header>
  );
}
