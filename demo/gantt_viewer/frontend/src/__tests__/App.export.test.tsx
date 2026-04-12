import { render } from "solid-js/web";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Registries, TraceDescriptor, TracePayload } from "../api/client";
import {
  __resetSignalsForTests,
  descriptors as descriptorState,
  loadedTraces,
  setLoadedTraces,
  setRegistries,
  setClockMode,
  setThemeMode,
  setTimeMode,
  setViewMode,
  setVisibility,
  setZoom,
  themeMode,
  timeMode,
  viewMode,
  visibility,
  zoom,
} from "../state/signals";

const apiClient = vi.hoisted(() => ({
  exportSnapshotHtml: vi.fn(),
  getPayload: vi.fn(),
  getTraces: vi.fn(),
  unregisterTraces: vi.fn(),
  uploadTrace: vi.fn(),
}));

const urlMocks = vi.hoisted(() => ({
  createObjectURL: vi.fn(() => "blob:download"),
  revokeObjectURL: vi.fn(),
}));

const persist = vi.hoisted(() => ({
  enableDisplaySync: vi.fn(),
  enablePersistence: vi.fn(),
}));

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return {
    ...actual,
    exportSnapshotHtml: apiClient.exportSnapshotHtml,
    getPayload: apiClient.getPayload,
    getTraces: apiClient.getTraces,
    unregisterTraces: apiClient.unregisterTraces,
    uploadTrace: apiClient.uploadTrace,
  };
});

vi.mock("../components/CanvasStage", () => ({
  default: () => <div data-testid="canvas-stage" />,
}));
vi.mock("../components/Legend", () => ({
  default: () => <div data-testid="legend" />,
}));
vi.mock("../components/Sidebar", () => ({
  default: () => <div data-testid="sidebar" />,
}));
vi.mock("../components/Tooltip", () => ({
  default: () => <div data-testid="tooltip" />,
}));
vi.mock("../state/persist", () => ({
  enableDisplaySync: persist.enableDisplaySync,
  enablePersistence: persist.enablePersistence,
}));

import App from "../App";

function flush(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function setSnapshotBootstrap(value: unknown): void {
  Object.defineProperty(window, "__GANTT_VIEWER_BOOTSTRAP__", {
    configurable: true,
    value,
    writable: true,
  });
}

function setSnapshotBootstrapScript(value: unknown): void {
  const existing = document.getElementById("gantt-viewer-snapshot-bootstrap");
  existing?.remove();
  const element = document.createElement("script");
  element.id = "gantt-viewer-snapshot-bootstrap";
  element.type = "application/json";
  element.textContent = JSON.stringify(value);
  document.body.append(element);
}

const baseRegistries: Registries = {
  markers: {
    done: { color: "#0f0", label: "Done", symbol: "circle" },
  },
  spans: {
    work: { color: "#00f", label: "Work", order: 1 },
  },
};

const baseTrace: TracePayload = {
  id: "trace-a",
  label: "Trace A",
  lanes: [],
  metadata: {
    instance_id: "instance-a",
    n_actions: 2,
    n_events: 2,
    n_iterations: 1,
    scaffold: "openclaw",
  },
  t0: 0,
};

const descriptors: TraceDescriptor[] = [
  {
    id: baseTrace.id,
    label: baseTrace.label,
    mtime: 0,
    path: "/tmp/trace-a.jsonl",
    size_bytes: 10,
    source_format: "trace",
  },
];

function createSnapshotTrace(id: string, label: string, instanceId: string): TracePayload {
  return {
    ...baseTrace,
    id,
    label,
    metadata: {
      ...baseTrace.metadata,
      instance_id: instanceId,
    },
  };
}

function createSnapshotBootstrap(traces: TracePayload[]) {
  const traceIds = traces.map((trace) => trace.id);
  return {
    mode: "snapshot" as const,
    payload: {
      errors: [],
      registries: baseRegistries,
      traces,
    },
    trace_ids: traceIds,
    visible_trace_ids: traceIds,
  };
}

function traceChipLabels(host: HTMLElement): string[] {
  return Array.from(host.querySelectorAll(".trace-chip-main"), (button) => button.textContent ?? "");
}

function mountApp() {
  const host = document.createElement("div");
  document.body.append(host);
  const dispose = render(() => <App />, host);
  return { dispose, host };
}

beforeEach(() => {
  setSnapshotBootstrap(undefined);
  __resetSignalsForTests();
  apiClient.getTraces.mockReset();
  apiClient.getPayload.mockReset();
  apiClient.uploadTrace.mockReset();
  apiClient.exportSnapshotHtml.mockReset();
  apiClient.unregisterTraces.mockReset();
  persist.enableDisplaySync.mockReset();
  persist.enablePersistence.mockReset();
  apiClient.getTraces.mockResolvedValue({ traces: descriptors, registries: baseRegistries });
  apiClient.getPayload.mockResolvedValue({ traces: [], registries: baseRegistries });
  apiClient.uploadTrace.mockResolvedValue(undefined);
  apiClient.exportSnapshotHtml.mockResolvedValue("<html>snapshot</html>");
  apiClient.unregisterTraces.mockResolvedValue({ missing_ids: [], removed_ids: [baseTrace.id] });
  vi.stubGlobal("URL", {
    ...URL,
    createObjectURL: urlMocks.createObjectURL,
    revokeObjectURL: urlMocks.revokeObjectURL,
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
  document.body.innerHTML = "";
});

describe("App export flow", () => {
  it("keeps live toolbar controls and downloads one html snapshot", async () => {
    setRegistries(baseRegistries);
    setLoadedTraces([baseTrace]);
    setVisibility({ [baseTrace.id]: true });
    const clickedLinks: HTMLAnchorElement[] = [];
    const clickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(function (this: HTMLAnchorElement) {
        clickedLinks.push(this);
      });

    const { dispose, host } = mountApp();
    try {
      await flush();
      const exportButton = host.querySelector("button.toolbar-export-btn") as HTMLButtonElement;

      expect(host.textContent).toContain("+ JSONL");
      expect(host.textContent).toContain("Load all");
      expect(traceChipLabels(host)).toEqual([baseTrace.label]);
      expect(exportButton.disabled).toBe(false);
      expect(persist.enableDisplaySync).not.toHaveBeenCalled();
      expect(persist.enablePersistence).toHaveBeenCalledOnce();

      window.dispatchEvent(new Event("dragenter", { bubbles: true }));
      await flush();
      expect(host.textContent).toContain("Drop JSONL");

      exportButton.click();
      await flush();

      expect(apiClient.exportSnapshotHtml).toHaveBeenCalledWith({
        registries: baseRegistries,
        traces: [baseTrace],
      });
      expect(clickSpy).toHaveBeenCalledTimes(1);
      expect(urlMocks.createObjectURL).toHaveBeenCalledTimes(1);
      const link = clickedLinks[0];
      expect(link.download).toBe("trace-gantt-export.html");
    } finally {
      dispose();
    }
  });

  it("hides live-only controls in snapshot mode and preserves exported trace order", async () => {
    const snapshotTraceB = createSnapshotTrace("trace-b", "Trace B", "instance-b");
    setSnapshotBootstrap(createSnapshotBootstrap([snapshotTraceB, baseTrace]));
    setThemeMode("light");
    setClockMode("real");
    setTimeMode("abs");
    setViewMode("concise");
    setZoom(2);

    const { dispose, host } = mountApp();
    try {
      const activeButtons = Array.from(host.querySelectorAll(".toggle-group button.active"));
      expect(activeButtons.map((button) => button.textContent)).toEqual(["DARK", "SYNC", "WALL", "LAYER"]);

      await flush();

      expect(traceChipLabels(host)).toEqual(["Trace B", "Trace A"]);
      expect(host.textContent).not.toContain("+ JSONL");
      expect(host.textContent).not.toContain("Load all");
      expect(host.querySelector("button.toolbar-export-btn")).toBeNull();

      window.dispatchEvent(new Event("dragenter", { bubbles: true }));
      await flush();
      expect(host.textContent).not.toContain("Drop JSONL");

      expect(apiClient.getTraces).not.toHaveBeenCalled();
      expect(apiClient.getPayload).not.toHaveBeenCalled();
      expect(descriptorState().map((descriptor) => descriptor.id)).toEqual(["trace-b", "trace-a"]);
      expect(loadedTraces().map((trace) => trace.id)).toEqual(["trace-b", "trace-a"]);
      expect(visibility()).toEqual({ "trace-b": true, "trace-a": true });
      expect(themeMode()).toBe("dark");
      expect(timeMode()).toBe("sync");
      expect(viewMode()).toBe("layered");
      expect(zoom()).toBe(1);
      expect(persist.enableDisplaySync).toHaveBeenCalledOnce();
      expect(persist.enablePersistence).not.toHaveBeenCalled();
    } finally {
      dispose();
    }
  });

  it("boots from the embedded snapshot script payload without calling live APIs", async () => {
    const snapshotTraceB = createSnapshotTrace("trace-b", "Trace B", "instance-b");
    setSnapshotBootstrapScript(createSnapshotBootstrap([snapshotTraceB, baseTrace]));

    const { dispose, host } = mountApp();
    try {
      await flush();

      expect(traceChipLabels(host)).toEqual(["Trace B", "Trace A"]);
      expect(apiClient.getTraces).not.toHaveBeenCalled();
      expect(apiClient.getPayload).not.toHaveBeenCalled();
      expect(persist.enableDisplaySync).toHaveBeenCalledOnce();
      expect(persist.enablePersistence).not.toHaveBeenCalled();
      expect(window.__GANTT_VIEWER_BOOTSTRAP__).toMatchObject({
        mode: "snapshot",
        trace_ids: ["trace-b", "trace-a"],
      });
    } finally {
      dispose();
    }
  });

  it("disables Export with zero loaded traces", async () => {
    setRegistries(baseRegistries);
    const { dispose, host } = mountApp();
    try {
      await flush();
      const exportButton = host.querySelector("button.toolbar-export-btn") as HTMLButtonElement;
      expect(exportButton.disabled).toBe(true);
      exportButton.click();
      await flush();
      expect(apiClient.exportSnapshotHtml).not.toHaveBeenCalled();
    } finally {
      dispose();
    }
  });

  it("logs export failures and re-enables the button without downloading", async () => {
    setRegistries(baseRegistries);
    setLoadedTraces([baseTrace]);
    setVisibility({ [baseTrace.id]: true });
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => undefined);
    apiClient.exportSnapshotHtml.mockRejectedValue(new Error("boom"));

    const { dispose, host } = mountApp();
    try {
      await flush();
      const exportButton = host.querySelector("button.toolbar-export-btn") as HTMLButtonElement;

      exportButton.click();
      expect(exportButton.disabled).toBe(true);

      await flush();

      expect(clickSpy).not.toHaveBeenCalled();
      expect(consoleSpy).toHaveBeenCalledOnce();
      expect(exportButton.disabled).toBe(false);
    } finally {
      dispose();
    }
  });

  it("unregisters traces when remove is clicked and clears the chip", async () => {
    setRegistries(baseRegistries);
    setLoadedTraces([baseTrace]);
    setVisibility({ [baseTrace.id]: true });

    const { dispose, host } = mountApp();
    try {
      await flush();

      const removeButton = host.querySelector("button.trace-remove") as HTMLButtonElement;
      expect(removeButton).not.toBeNull();

      removeButton.click();
      await flush();

      expect(apiClient.unregisterTraces).toHaveBeenCalledWith([baseTrace.id]);
      expect(traceChipLabels(host)).toEqual([]);
      expect(host.textContent).not.toContain(baseTrace.label);
    } finally {
      dispose();
    }
  });
});
