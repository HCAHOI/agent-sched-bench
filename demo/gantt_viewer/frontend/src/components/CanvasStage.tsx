import { createEffect, onCleanup, onMount } from "solid-js";

import type { GanttPayload } from "../api/client";
import { CanvasRenderer } from "../canvas/CanvasRenderer";
import type { HitCard } from "../canvas/hit";
import type { TimeMode, ViewMode } from "../state/signals";

interface CanvasStageProps {
  onClick: (card: HitCard | null) => void;
  onHover: (card: HitCard | null) => void;
  onScroll: (scrollTop: number) => void;
  onZoom: (factor: number) => void;
  payload: GanttPayload | null;
  timeMode: TimeMode;
  viewMode: ViewMode;
  visibility: Record<string, boolean>;
  zoom: number;
}

export default function CanvasStage(props: CanvasStageProps) {
  let canvasEl!: HTMLCanvasElement;
  let wrapEl!: HTMLDivElement;
  let renderer: CanvasRenderer | null = null;

  onMount(() => {
    renderer = new CanvasRenderer(canvasEl, wrapEl);
    renderer.addEventListener("hover", (event) =>
      props.onHover((event as CustomEvent<HitCard | null>).detail),
    );
    renderer.addEventListener("click", (event) =>
      props.onClick((event as CustomEvent<HitCard | null>).detail),
    );
    renderer.addEventListener("scroll", (event) =>
      props.onScroll((event as CustomEvent<{ scrollTop: number }>).detail.scrollTop),
    );
    renderer.addEventListener("zoom", (event) =>
      props.onZoom((event as CustomEvent<{ factor: number }>).detail.factor),
    );
  });

  createEffect(() => {
    renderer?.setPayload(props.payload);
  });

  createEffect(() => {
    renderer?.setTimeMode(props.timeMode);
  });

  createEffect(() => {
    renderer?.setViewMode(props.viewMode);
  });

  createEffect(() => {
    renderer?.setVisibility(props.visibility);
  });

  createEffect(() => {
    renderer?.setZoom(props.zoom);
  });

  onCleanup(() => {
    renderer?.destroy();
    renderer = null;
  });

  return (
    <div class="canvas-panel">
      <div class="canvas-wrap" ref={wrapEl}>
        <canvas ref={canvasEl} />
      </div>
    </div>
  );
}
