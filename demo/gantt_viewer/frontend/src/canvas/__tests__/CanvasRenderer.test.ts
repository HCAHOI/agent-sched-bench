import type { GanttPayload } from "../../api/client";
import { CanvasRenderer } from "../CanvasRenderer";

function createMockContext(): CanvasRenderingContext2D {
  const noop = () => {};
  return {
    arc: noop,
    beginPath: noop,
    clearRect: noop,
    fill: noop,
    fillRect: noop,
    fillText: noop,
    lineTo: noop,
    moveTo: noop,
    restore: noop,
    rotate: noop,
    save: noop,
    setTransform: noop,
    stroke: noop,
    translate: noop,
  } as unknown as CanvasRenderingContext2D;
}

function setElementBox(element: HTMLElement, width: number, height: number): void {
  Object.defineProperty(element, "clientWidth", { configurable: true, value: width });
  Object.defineProperty(element, "clientHeight", { configurable: true, value: height });
}

function payloadFixture(): GanttPayload {
  return {
    registries: {
      markers: {
        message_dispatch: { color: "#76FF03", label: "Message Dispatch", symbol: "diamond" },
      },
      spans: {
        llm: { color: "#00E5FF", label: "LLM Call", order: 0 },
        tool: { color: "#FF6D00", label: "Tool Exec", order: 1 },
      },
    },
    traces: [
      {
        id: "trace-1",
        label: "trace-1",
        metadata: {
          elapsed_s: 2,
          instance_id: "demo",
          max_iterations: 2,
          mode: "import",
          model: "test",
          n_actions: 2,
          n_events: 1,
          n_iterations: 1,
          scaffold: "openclaw",
        },
        t0: 1000,
        lanes: [
          {
            agent_id: "agent-1",
            markers: [
              {
                detail: {},
                event: "message_dispatch",
                iteration: 0,
                t: 0.8,
                t_abs: 1000.8,
                type: "scheduling",
              },
            ],
            spans: [
              {
                detail: { llm_content: "hello" },
                end: 0.4,
                end_abs: 1000.4,
                iteration: 0,
                start: 0,
                start_abs: 1000,
                type: "llm",
              },
              {
                detail: { tool_name: "bash" },
                end: 1.0,
                end_abs: 1001.0,
                iteration: 0,
                start: 0.5,
                start_abs: 1000.5,
                type: "tool",
              },
            ],
          },
        ],
      },
    ],
  };
}

describe("CanvasRenderer", () => {
  it("locates and hit-tests a rendered span", () => {
    const canvas = document.createElement("canvas");
    const wrap = document.createElement("div");
    const context = createMockContext();

    vi.spyOn(canvas, "getContext").mockImplementation(() => context);
    vi.spyOn(canvas, "getBoundingClientRect").mockImplementation(() => ({
      bottom: 360,
      height: 320,
      left: 10,
      right: 650,
      top: 20,
      width: 640,
      x: 10,
      y: 20,
      toJSON: () => ({}),
    }));
    setElementBox(canvas, 640, 320);
    setElementBox(wrap, 640, 320);

    const renderer = new CanvasRenderer(canvas, wrap);
    const payload = payloadFixture();

    renderer.setPayload(payload);
    (renderer as unknown as { render: () => void }).render();
    const hit = renderer.hitTest(24, 40);
    expect(hit).not.toBeNull();
    if (!hit) {
      throw new Error("expected span hit");
    }
    const card = renderer.locateHit(hit);

    expect(card).not.toBeNull();
    const localHit = renderer.hitTest(card!.x - 10, card!.y - 20);
    expect(localHit?.kind).toBe("span");
    renderer.destroy();
  });

  it("clamps zoom and emits the resulting factor", () => {
    const canvas = document.createElement("canvas");
    const wrap = document.createElement("div");
    const context = createMockContext();
    vi.spyOn(canvas, "getContext").mockImplementation(() => context);
    vi.spyOn(canvas, "getBoundingClientRect").mockImplementation(() => ({
      bottom: 360,
      height: 320,
      left: 0,
      right: 640,
      top: 0,
      width: 640,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    }));
    setElementBox(canvas, 640, 320);
    setElementBox(wrap, 640, 320);

    const renderer = new CanvasRenderer(canvas, wrap);
    const factors: number[] = [];
    renderer.addEventListener("zoom", (event) => {
      factors.push((event as CustomEvent<{ factor: number }>).detail.factor);
    });

    renderer.setZoom(100);

    expect(factors.at(-1)).toBe(32);
    renderer.destroy();
  });

  it("reanchors a span hit after switching to concise layout", () => {
    const canvas = document.createElement("canvas");
    const wrap = document.createElement("div");
    const context = createMockContext();
    vi.spyOn(canvas, "getContext").mockImplementation(() => context);
    vi.spyOn(canvas, "getBoundingClientRect").mockImplementation(() => ({
      bottom: 360,
      height: 320,
      left: 0,
      right: 640,
      top: 0,
      width: 640,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    }));
    setElementBox(canvas, 640, 320);
    setElementBox(wrap, 640, 320);

    const renderer = new CanvasRenderer(canvas, wrap);
    const payload = payloadFixture();

    renderer.setPayload(payload);
    (renderer as unknown as { render: () => void }).render();
    const secondSpan = renderer.hitTest(340, 58);
    expect(secondSpan).not.toBeNull();
    if (!secondSpan) {
      throw new Error("expected second span hit");
    }
    const layered = renderer.locateHit(secondSpan);
    renderer.setViewMode("concise");
    (renderer as unknown as { render: () => void }).render();
    const concise = renderer.locateHit(secondSpan);

    expect(layered).not.toBeNull();
    expect(concise).not.toBeNull();
    expect(concise!.y).toBeLessThan(layered!.y);
    renderer.destroy();
  });
});
