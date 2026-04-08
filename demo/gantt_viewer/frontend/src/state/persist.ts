import { createEffect } from "solid-js";

import { viewMode } from "./signals";

let started = false;

export function enablePersistence(): void {
  if (started || typeof window === "undefined") {
    return;
  }
  started = true;

  createEffect(() => {
    window.localStorage.setItem("gantt.viewMode", viewMode());
  });

  // Global CSS hooks for viewMode — lets .sidebar and canvas share lane
  // height through a single CSS variable instead of threading a prop.
  createEffect(() => {
    if (typeof document === "undefined") return;
    const concise = viewMode() === "concise";
    document.body.classList.toggle("view-concise", concise);
    document.documentElement.style.setProperty("--lane-h", concise ? "26px" : "60px");
  });
}

export function __resetPersistenceForTests(): void {
  started = false;
}
