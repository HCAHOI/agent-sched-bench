import { createSignal } from "solid-js";

import type { Registries, TraceDescriptor, TracePayload } from "../api/client";
import type { HitCard } from "../canvas/hit";

export type TimeMode = "sync" | "abs";
export type ViewMode = "layered" | "concise";
export type ThemeMode = "dark" | "light";

function initialViewMode(): ViewMode {
  if (typeof window === "undefined") {
    return "layered";
  }
  const persisted = window.localStorage.getItem("gantt.viewMode");
  return persisted === "concise" ? "concise" : "layered";
}

export function initialThemeMode(): ThemeMode {
  if (typeof window === "undefined") {
    return "dark";
  }
  const persisted = window.localStorage.getItem("gantt.themeMode");
  return persisted === "light" ? "light" : "dark";
}

export const [timeMode, setTimeMode] = createSignal<TimeMode>("sync");
export const [viewMode, setViewMode] = createSignal<ViewMode>(initialViewMode());
export const [themeMode, setThemeMode] = createSignal<ThemeMode>(initialThemeMode());
export const [zoom, setZoom] = createSignal(1);
export const [descriptors, setDescriptors] = createSignal<TraceDescriptor[]>([]);
export const [registries, setRegistries] = createSignal<Registries | null>(null);
export const [loadedTraces, setLoadedTraces] = createSignal<TracePayload[]>([]);
export const [visibility, setVisibility] = createSignal<Record<string, boolean>>({});
export const [loadingIds, setLoadingIds] = createSignal<Record<string, boolean>>({});
export const [hoverCard, setHoverCard] = createSignal<HitCard | null>(null);
export const [pinnedCard, setPinnedCard] = createSignal<HitCard | null>(null);
export const [scrollTop, setScrollTop] = createSignal(0);
export const [appError, setAppError] = createSignal<string | null>(null);
