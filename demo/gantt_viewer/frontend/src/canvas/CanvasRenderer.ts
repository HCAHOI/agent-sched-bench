import type { GanttPayload } from "../api/client";
import { sameHit, type Hit, type HitCard } from "./hit";
import { computeTotalContentHeight, effectiveLaneH, MARKER_H, SPAN_H, SPAN_PAD, TIME_AXIS_H } from "./layout";
import { formatTimeLabel, niceStep } from "./time";

type TimeMode = "sync" | "abs";
type ViewMode = "layered" | "concise";

interface RenderHitBox {
  h: number;
  hit: Hit;
  w: number;
  x: number;
  y: number;
}

const GRID_COLOR = "#141a24";
const LANE_BG = "#0a0e14";
const LANE_BORDER = "#1e2633";
const UNKNOWN_SPAN_COLOR = "#6b7280";
const TEXT_COLOR = "#c5c8d4";
const AXIS_TEXT_COLOR = "#6b7280";
const SPAN_LABEL_COLOR = "#0a0e14";
const ZOOM_MAX = 32;
const ZOOM_MIN = 0.25;
const ZOOM_STEP = 1.25;
export const ZOOM_PRESETS = [0.25, 0.5, 1, 2, 4, 8, 16, 32] as const;

export class CanvasRenderer extends EventTarget {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private hitBoxes: RenderHitBox[] = [];
  private keydownHandler: (event: KeyboardEvent) => void;
  private payload: GanttPayload | null = null;
  private rafId: number | null = null;
  private resizeRafId: number | null = null;
  private resizeObserver: ResizeObserver;
  private timeMode: TimeMode = "sync";
  private viewMode: ViewMode = "layered";
  private visibility: Record<string, boolean> = {};
  private wrap: HTMLDivElement;
  private zoomFactor = 1;

  constructor(canvas: HTMLCanvasElement, wrap: HTMLDivElement) {
    super();
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      throw new Error("2d canvas context unavailable");
    }

    this.canvas = canvas;
    this.ctx = ctx;
    this.wrap = wrap;

    this.resizeObserver = new ResizeObserver(() => this.scheduleResize());
    this.resizeObserver.observe(this.wrap);

    this.canvas.addEventListener("mousemove", (event) => this.onMouseMove(event));
    this.canvas.addEventListener("mouseleave", () => this.emitHover(null));
    this.canvas.addEventListener("click", (event) => this.onClick(event));
    this.canvas.addEventListener(
      "wheel",
      (event) => this.onWheel(event),
      { passive: false },
    );
    this.wrap.addEventListener("scroll", () => {
      this.dispatchEvent(
        new CustomEvent("scroll", {
          detail: {
            scrollLeft: this.wrap.scrollLeft,
            scrollTop: this.wrap.scrollTop,
          },
        }),
      );
    });
    this.keydownHandler = (event) => this.onKeyDown(event);
    document.addEventListener("keydown", this.keydownHandler);

    this.resizeCanvasImmediate();
  }

  destroy(): void {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    if (this.resizeRafId !== null) {
      cancelAnimationFrame(this.resizeRafId);
      this.resizeRafId = null;
    }
    this.resizeObserver.disconnect();
    document.removeEventListener("keydown", this.keydownHandler);
  }

  setPayload(payload: GanttPayload | null): void {
    this.payload = payload;
    this.scheduleResize();
  }

  setTimeMode(mode: TimeMode): void {
    if (this.timeMode === mode) {
      return;
    }
    this.timeMode = mode;
    this.queueRender();
  }

  setViewMode(mode: ViewMode): void {
    if (this.viewMode === mode) {
      return;
    }
    this.viewMode = mode;
    this.scheduleResize();
  }

  setZoom(factor: number, cursorFrac?: number): void {
    const clamped = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, factor));
    const currentWidth = this.canvas.clientWidth || this.wrap.clientWidth || 1;
    const anchorFrac =
      cursorFrac !== undefined
        ? (this.wrap.scrollLeft + cursorFrac * this.wrap.clientWidth) / currentWidth
        : 0.5;

    if (clamped === this.zoomFactor && cursorFrac === undefined) {
      return;
    }

    this.zoomFactor = clamped;
    // Synchronous resize for cursor-anchored zoom: scrollLeft needs the
    // new logical width, which scheduleResize would only produce next frame.
    this.resizeCanvasImmediate();

    const nextWidth = this.canvas.clientWidth || this.wrap.clientWidth || 1;
    const offset =
      cursorFrac !== undefined
        ? cursorFrac * this.wrap.clientWidth
        : this.wrap.clientWidth / 2;
    this.wrap.scrollLeft = anchorFrac * nextWidth - offset;
    this.dispatchEvent(
      new CustomEvent("zoom", {
        detail: {
          factor: this.zoomFactor,
        },
      }),
    );
  }

  setVisibility(map: Record<string, boolean>): void {
    this.visibility = map;
    this.scheduleResize();
  }

  rerender(): void {
    this.queueRender();
  }

  hitTest(mx: number, my: number): Hit | null {
    for (let index = this.hitBoxes.length - 1; index >= 0; index -= 1) {
      const box = this.hitBoxes[index];
      if (mx >= box.x && mx <= box.x + box.w && my >= box.y && my <= box.y + box.h) {
        return box.hit;
      }
    }
    return null;
  }

  locateHit(hit: Hit): HitCard | null {
    const box = this.hitBoxes.find((candidate) => sameHit(candidate.hit, hit));
    if (!box) {
      return null;
    }
    const rect = this.canvas.getBoundingClientRect();
    return {
      hit: box.hit,
      x: rect.left + box.x + box.w / 2,
      y: rect.top + box.y + box.h / 2,
    };
  }

  private emitHover(card: HitCard | null): void {
    this.dispatchEvent(new CustomEvent("hover", { detail: card }));
  }

  private onClick(event: MouseEvent): void {
    const hit = this.hitFromEvent(event);
    this.dispatchEvent(
      new CustomEvent("click", {
        detail: hit
          ? {
              hit,
              x: event.clientX,
              y: event.clientY,
            }
          : null,
      }),
    );
  }

  private onKeyDown(event: KeyboardEvent): void {
    const target = event.target as HTMLElement | null;
    if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) {
      return;
    }
    if (event.key === "+" || event.key === "=") {
      event.preventDefault();
      this.setZoom(this.zoomFactor * ZOOM_STEP);
    } else if (event.key === "-" || event.key === "_") {
      event.preventDefault();
      this.setZoom(this.zoomFactor / ZOOM_STEP);
    } else if (event.key === "0") {
      event.preventDefault();
      this.setZoom(1);
    }
  }

  private onMouseMove(event: MouseEvent): void {
    const hit = this.hitFromEvent(event);
    this.emitHover(
      hit
        ? {
            hit,
            x: event.clientX,
            y: event.clientY,
          }
        : null,
    );
  }

  private onWheel(event: WheelEvent): void {
    if (!event.ctrlKey && !event.metaKey) {
      return;
    }
    event.preventDefault();
    const rect = this.canvas.getBoundingClientRect();
    const cursorX = event.clientX - rect.left;
    const cursorFrac = this.canvas.clientWidth > 0 ? cursorX / this.canvas.clientWidth : 0.5;
    this.setZoom(
      this.zoomFactor * (event.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP),
      cursorFrac,
    );
  }

  private hitFromEvent(event: MouseEvent): Hit | null {
    const rect = this.canvas.getBoundingClientRect();
    return this.hitTest(event.clientX - rect.left, event.clientY - rect.top);
  }

  private queueRender(): void {
    if (this.rafId !== null) {
      return;
    }
    this.rafId = requestAnimationFrame(() => {
      this.rafId = null;
      this.render();
    });
  }

  private scheduleResize(): void {
    if (this.resizeRafId !== null) {
      return;
    }
    this.resizeRafId = requestAnimationFrame(() => {
      this.resizeRafId = null;
      this.resizeCanvasImmediate();
    });
  }

  private resizeCanvasImmediate(): void {
    const dpr = window.devicePixelRatio || 1;
    const logicalWidth = Math.max(this.wrap.clientWidth * this.zoomFactor, this.wrap.clientWidth || 1);
    const logicalHeight = computeTotalContentHeight(
      this.payload?.traces ?? [],
      this.visibility,
      this.viewMode,
      this.wrap.clientHeight || 320,
    );

    // canvas.width/height reset the backing buffer (and clear it) even on a
    // no-op write. Gate the writes so rapid re-renders from the same viewport
    // size don't thrash the GPU texture at high zoom.
    const nextBufW = Math.round(logicalWidth * dpr);
    const nextBufH = Math.round(logicalHeight * dpr);
    if (this.canvas.width !== nextBufW) {
      this.canvas.width = nextBufW;
    }
    if (this.canvas.height !== nextBufH) {
      this.canvas.height = nextBufH;
    }
    this.canvas.style.width = `${logicalWidth}px`;
    this.canvas.style.height = `${logicalHeight}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.queueRender();
  }

  private render(): void {
    const width = this.canvas.clientWidth || this.wrap.clientWidth || 1;
    const height = this.canvas.clientHeight || this.wrap.clientHeight || 1;

    this.ctx.clearRect(0, 0, width, height);
    this.hitBoxes = [];

    const traces = (this.payload?.traces ?? []).filter(
      (trace) => this.visibility[trace.id] !== false,
    );
    if (traces.length === 0) {
      this.ctx.fillStyle = AXIS_TEXT_COLOR;
      this.ctx.font = '12px "JetBrains Mono", monospace';
      this.ctx.fillText("Load a trace to render the timeline.", 20, 48);
      return;
    }

    let minTime = Number.POSITIVE_INFINITY;
    let maxTime = Number.NEGATIVE_INFINITY;

    for (const trace of traces) {
      for (const lane of trace.lanes) {
        for (const span of lane.spans) {
          const start = this.timeMode === "sync" ? span.start : span.start_abs;
          const end = this.timeMode === "sync" ? span.end : span.end_abs;
          minTime = Math.min(minTime, start);
          maxTime = Math.max(maxTime, end);
        }
        for (const marker of lane.markers) {
          const point = this.timeMode === "sync" ? marker.t : marker.t_abs;
          minTime = Math.min(minTime, point);
          maxTime = Math.max(maxTime, point);
        }
      }
    }

    if (!Number.isFinite(minTime)) {
      minTime = 0;
      maxTime = 1;
    }

    const range = maxTime - minTime || 1;
    const margin = range * 0.02;
    // Sync mode anchors t=0 on the trace start — a negative left margin
    // would display "-Xs" ticks that have no referent.
    if (this.timeMode === "sync") {
      minTime = 0;
    } else {
      minTime -= margin;
    }
    maxTime += margin;
    const totalRange = maxTime - minTime || 1;
    const laneHeight = effectiveLaneH(this.viewMode);
    const gridStep = niceStep(totalRange, width / 80);
    const timeToX = (value: number) => ((value - minTime) / totalRange) * width;

    this.ctx.strokeStyle = GRID_COLOR;
    this.ctx.lineWidth = 1;
    for (
      let tick = Math.ceil(minTime / gridStep) * gridStep;
      tick <= maxTime;
      tick += gridStep
    ) {
      const x = timeToX(tick);
      this.ctx.beginPath();
      this.ctx.moveTo(x, 0);
      this.ctx.lineTo(x, height);
      this.ctx.stroke();
    }

    this.ctx.fillStyle = AXIS_TEXT_COLOR;
    this.ctx.font = '10px "JetBrains Mono", "Menlo", monospace';
    this.ctx.textAlign = "center";
    for (
      let tick = Math.ceil(minTime / gridStep) * gridStep;
      tick <= maxTime;
      tick += gridStep
    ) {
      this.ctx.fillText(formatTimeLabel(this.timeMode, tick), timeToX(tick), TIME_AXIS_H - 6);
    }

    let laneY = TIME_AXIS_H;
    for (const trace of traces) {
      for (const lane of trace.lanes) {
        this.ctx.fillStyle = LANE_BG;
        this.ctx.fillRect(0, laneY, width, laneHeight);

        this.ctx.strokeStyle = LANE_BORDER;
        this.ctx.beginPath();
        this.ctx.moveTo(0, laneY + laneHeight);
        this.ctx.lineTo(width, laneY + laneHeight);
        this.ctx.stroke();

        const spansByIteration = new Map<number, typeof lane.spans>();
        for (const span of lane.spans) {
          const bucket = spansByIteration.get(span.iteration) ?? [];
          bucket.push(span);
          spansByIteration.set(span.iteration, bucket);
        }

        for (const spans of spansByIteration.values()) {
          spans.sort((left, right) => {
            const leftOrder = this.payload?.registries.spans[left.type]?.order ?? 99;
            const rightOrder = this.payload?.registries.spans[right.type]?.order ?? 99;
            return leftOrder - rightOrder;
          });

          spans.forEach((span, index) => {
            const start = this.timeMode === "sync" ? span.start : span.start_abs;
            const end = this.timeMode === "sync" ? span.end : span.end_abs;
            const x0 = timeToX(start);
            const x1 = timeToX(end);
            const widthPx = Math.max(x1 - x0, 3);
            const yOffset =
              laneY +
              SPAN_PAD +
              (this.viewMode === "concise" ? 0 : index) * (SPAN_H + 2);
            const color =
              this.payload?.registries.spans[span.type]?.color ?? UNKNOWN_SPAN_COLOR;

            this.ctx.fillStyle = color;
            this.ctx.globalAlpha = 0.88;
            this.ctx.fillRect(x0, yOffset, widthPx, SPAN_H);
            this.ctx.globalAlpha = 1;

            if (widthPx > 22) {
              this.ctx.fillStyle = SPAN_LABEL_COLOR;
              this.ctx.textAlign = "left";
              this.ctx.fillText(String(span.iteration), x0 + 4, yOffset + 12);
            }

            this.hitBoxes.push({
              h: SPAN_H,
              hit: {
                item: span,
                kind: "span",
                laneAgentId: lane.agent_id,
                traceId: trace.id,
                traceLabel: trace.label,
              },
              w: widthPx,
              x: x0,
              y: yOffset,
            });
          });
        }

        for (const marker of lane.markers) {
          const point = this.timeMode === "sync" ? marker.t : marker.t_abs;
          const x = timeToX(point);
          const markerReg =
            this.payload?.registries.markers[marker.event] ??
            this.payload?.registries.markers[marker.type];
          const color = markerReg?.color ?? UNKNOWN_SPAN_COLOR;
          const symbol = markerReg?.symbol ?? "dot";
          const centerY = laneY + laneHeight - MARKER_H - 2;

          this.ctx.fillStyle = color;
          this.ctx.globalAlpha = 0.8;
          if (symbol === "diamond") {
            this.ctx.save();
            this.ctx.translate(x, centerY);
            this.ctx.rotate(Math.PI / 4);
            this.ctx.fillRect(-3, -3, 6, 6);
            this.ctx.restore();
          } else if (symbol === "flag") {
            this.ctx.fillRect(x - 1, centerY - 6, 2, 10);
            this.ctx.fillRect(x + 1, centerY - 6, 6, 4);
          } else if (symbol === "cross") {
            this.ctx.fillRect(x - 4, centerY - 1, 8, 2);
            this.ctx.fillRect(x - 1, centerY - 4, 2, 8);
          } else {
            this.ctx.beginPath();
            this.ctx.arc(x, centerY, 3, 0, Math.PI * 2);
            this.ctx.fill();
          }
          this.ctx.globalAlpha = 1;

          this.hitBoxes.push({
            h: 12,
            hit: {
              item: marker,
              kind: "marker",
              laneAgentId: lane.agent_id,
              traceId: trace.id,
              traceLabel: trace.label,
            },
            w: 12,
            x: x - 6,
            y: centerY - 6,
          });
        }

        laneY += laneHeight;
      }
    }

    this.dispatchEvent(new CustomEvent("render"));
  }
}
