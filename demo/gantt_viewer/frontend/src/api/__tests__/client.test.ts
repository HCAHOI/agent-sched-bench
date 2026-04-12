import { afterEach, describe, expect, it, vi } from "vitest";

import { exportSnapshotHtml, unregisterTraces, type ExportSnapshotRequest } from "../client";

const snapshot: ExportSnapshotRequest = {
  registries: {
    markers: {
      done: { color: "#0f0", label: "Done", symbol: "circle" },
    },
    spans: {
      work: { color: "#00f", label: "Work", order: 1 },
    },
  },
  traces: [
    {
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
    },
  ],
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("exportSnapshotHtml", () => {
  it("posts the snapshot wrapper to the html export endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      text: vi.fn().mockResolvedValue("<html />"),
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(exportSnapshotHtml(snapshot)).resolves.toBe("<html />");
    expect(fetchMock).toHaveBeenCalledWith("/api/export/html", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ snapshot }),
    });
  });

  it("throws on non-ok responses", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 422,
      }),
    );

    await expect(exportSnapshotHtml(snapshot)).rejects.toThrow(
      "POST /api/export/html failed: 422",
    );
  });
});

describe("unregisterTraces", () => {
  it("posts ids to the unregister endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue({ missing_ids: [], removed_ids: ["trace-a"] }),
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(unregisterTraces(["trace-a"])).resolves.toEqual({
      missing_ids: [],
      removed_ids: ["trace-a"],
    });
    expect(fetchMock).toHaveBeenCalledWith("/api/traces/unregister", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ ids: ["trace-a"] }),
    });
  });
});
