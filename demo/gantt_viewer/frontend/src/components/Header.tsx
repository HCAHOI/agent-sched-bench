import { For, Show, type Accessor } from "solid-js";

import { ZOOM_PRESETS } from "../canvas/CanvasRenderer";
import type { ClockMode, ThemeMode, TimeMode, ViewMode } from "../state/signals";

interface HeaderProps {
  clockMode: Accessor<ClockMode>;
  exportDisabled: Accessor<boolean>;
  onClockModeChange: (mode: ClockMode) => void;
  onExport: () => Promise<void> | void;
  onThemeModeChange: (mode: ThemeMode) => void;
  onTimeModeChange: (mode: TimeMode) => void;
  onViewModeChange: (mode: ViewMode) => void;
  themeMode: Accessor<ThemeMode>;
  onZoomChange: (factor: number) => void;
  timeMode: Accessor<TimeMode>;
  viewMode: Accessor<ViewMode>;
  zoom: Accessor<number>;
  snapshotMode: boolean;
}

function getPresetValue(zoom: number): string {
  const matchingPreset = ZOOM_PRESETS.find((preset) => Math.abs(preset - zoom) < 0.001);
  return matchingPreset === undefined ? "" : String(matchingPreset);
}

export default function Header(props: HeaderProps) {
  const currentPresetValue = () => getPresetValue(props.zoom());
  const currentZoomLabel = () => `${Math.round(props.zoom() * 100)}%`;

  function handleZoomChange(event: Event): void {
    const value = (event.currentTarget as HTMLSelectElement).value;
    if (value !== "") {
      props.onZoomChange(Number(value));
    }
  }

  return (
    <header class="toolbar-card">
      <span class="toolbar-title">TRACE GANTT</span>
      <div class="toolbar-spacer" />
      <label class="zoom-select-wrap" title="Zoom presets (Ctrl+wheel for free zoom)">
        <span class="zoom-select-label">zoom</span>
        <select
          class="zoom-select"
          value={currentPresetValue()}
          onChange={handleZoomChange}
        >
          {currentPresetValue() === "" && (
            <option value="" disabled>
              {currentZoomLabel()}
            </option>
          )}
          <For each={ZOOM_PRESETS}>
            {(preset) => <option value={String(preset)}>{`${preset}x`}</option>}
          </For>
        </select>
      </label>
      <Show when={!props.snapshotMode}>
        <button
          class="primary-btn toolbar-export-btn"
          disabled={props.exportDisabled()}
          onClick={() => void props.onExport()}
          type="button"
        >
          Export
        </button>
      </Show>
      <div class="toggle-group">
        <button
          classList={{ active: props.themeMode() === "dark" }}
          onClick={() => props.onThemeModeChange("dark")}
          type="button"
        >
          DARK
        </button>
        <button
          classList={{ active: props.themeMode() === "light" }}
          onClick={() => props.onThemeModeChange("light")}
          type="button"
        >
          LIGHT
        </button>
      </div>
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
          classList={{ active: props.clockMode() === "wall" }}
          onClick={() => props.onClockModeChange("wall")}
          type="button"
        >
          WALL
        </button>
        <button
          classList={{ active: props.clockMode() === "real" }}
          onClick={() => props.onClockModeChange("real")}
          type="button"
        >
          REAL
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
