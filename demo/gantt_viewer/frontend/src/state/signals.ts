import { createSignal } from "solid-js";

import type { Registries, TraceDescriptor, TracePayload } from "../api/client";
import type { HitCard } from "../canvas/hit";

export type TimeMode = "sync" | "abs";
export type ViewMode = "layered" | "concise";
export type ThemeMode = "dark" | "light";
export type ClockMode = "wall" | "real";
export type ResourceMetric = "cpu" | "memory" | "disk_io" | "net_io";

export const DEFAULT_TIME_MODE: TimeMode = "sync";
export const DEFAULT_VIEW_MODE: ViewMode = "layered";
export const DEFAULT_THEME_MODE: ThemeMode = "dark";
export const DEFAULT_CLOCK_MODE: ClockMode = "wall";
export const DEFAULT_ZOOM = 1;

const STORAGE_KEYS = {
  clockMode: "gantt.clockMode",
  themeMode: "gantt.themeMode",
  viewMode: "gantt.viewMode",
} as const;

function readStoredMode<TMode extends string>(
  key: string,
  expectedValue: TMode,
  fallback: TMode,
): TMode {
  if (typeof window === "undefined") {
    return fallback;
  }

  return window.localStorage.getItem(key) === expectedValue ? expectedValue : fallback;
}

function initialViewMode(): ViewMode {
  return readStoredMode(STORAGE_KEYS.viewMode, "concise", DEFAULT_VIEW_MODE);
}

export function initialClockMode(): ClockMode {
  return readStoredMode(STORAGE_KEYS.clockMode, "real", DEFAULT_CLOCK_MODE);
}

export function initialThemeMode(): ThemeMode {
  return readStoredMode(STORAGE_KEYS.themeMode, "light", DEFAULT_THEME_MODE);
}

export const [timeMode, setTimeMode] = createSignal<TimeMode>(DEFAULT_TIME_MODE);
export const [viewMode, setViewMode] = createSignal<ViewMode>(initialViewMode());
export const [themeMode, setThemeMode] = createSignal<ThemeMode>(initialThemeMode());
export const [clockMode, setClockMode] = createSignal<ClockMode>(initialClockMode());
export const [zoom, setZoom] = createSignal(DEFAULT_ZOOM);
export const [descriptors, setDescriptors] = createSignal<TraceDescriptor[]>([]);
export const [registries, setRegistries] = createSignal<Registries | null>(null);
export const [loadedTraces, setLoadedTraces] = createSignal<TracePayload[]>([]);
export const [visibility, setVisibility] = createSignal<Record<string, boolean>>({});
export const [loadingIds, setLoadingIds] = createSignal<Record<string, boolean>>({});
export const [hoverCard, setHoverCard] = createSignal<HitCard | null>(null);
export const [pinnedCard, setPinnedCard] = createSignal<HitCard | null>(null);
export const [scrollTop, setScrollTop] = createSignal(0);
export const [appError, setAppError] = createSignal<string | null>(null);
export const [resourceMetric, setResourceMetric] = createSignal<ResourceMetric>("cpu");
export const [showResourceChart, setShowResourceChart] = createSignal(true);

export function __resetSignalsForTests(): void {
  setTimeMode(DEFAULT_TIME_MODE);
  setViewMode(DEFAULT_VIEW_MODE);
  setThemeMode(DEFAULT_THEME_MODE);
  setClockMode(DEFAULT_CLOCK_MODE);
  setZoom(DEFAULT_ZOOM);
  setDescriptors([]);
  setRegistries(null);
  setLoadedTraces([]);
  setVisibility({});
  setLoadingIds({});
  setHoverCard(null);
  setPinnedCard(null);
  setScrollTop(0);
  setAppError(null);
  setResourceMetric("cpu");
  setShowResourceChart(true);
}
