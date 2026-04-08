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
}
