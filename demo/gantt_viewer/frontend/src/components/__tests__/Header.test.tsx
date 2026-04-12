import { render } from "solid-js/web";
import { afterEach, describe, expect, it, vi } from "vitest";

import Header from "../Header";

function mountHeader(options?: { exportDisabled?: boolean; snapshotMode?: boolean }) {
  const host = document.createElement("div");
  document.body.append(host);
  const onExport = vi.fn();
  const dispose = render(
    () => (
      <Header
        clockMode={() => "wall"}
        exportDisabled={() => options?.exportDisabled ?? false}
        onClockModeChange={() => undefined}
        onExport={onExport}
        onThemeModeChange={() => undefined}
        onTimeModeChange={() => undefined}
        onViewModeChange={() => undefined}
        themeMode={() => "dark"}
        onZoomChange={() => undefined}
        snapshotMode={options?.snapshotMode ?? false}
        timeMode={() => "sync"}
        viewMode={() => "layered"}
        zoom={() => 1}
      />
    ),
    host,
  );
  return { dispose, host, onExport };
}

function headerText(host: HTMLElement): string {
  return host.textContent ?? "";
}

afterEach(() => {
  document.body.innerHTML = "";
});

describe("Header", () => {
  it("removes summary text and places Export after zoom", () => {
    const { dispose, host } = mountHeader();
    try {
      const header = host.querySelector("header");
      const exportButton = host.querySelector("button.toolbar-export-btn");
      const themeGroup = host.querySelectorAll(".toggle-group")[0];

      expect(header?.textContent).toContain("TRACE GANTT");
      expect(header?.textContent).not.toContain("actions total");
      expect(header?.textContent).not.toContain("loaded ");
      expect(headerText(host)).toContain("DARK");
      expect(headerText(host)).toContain("LIGHT");
      expect(headerText(host)).toContain("SYNC");
      expect(headerText(host)).toContain("ABS");
      expect(headerText(host)).toContain("WALL");
      expect(headerText(host)).toContain("REAL");
      expect(headerText(host)).toContain("LAYER");
      expect(headerText(host)).toContain("CONCISE");
      expect(exportButton?.previousElementSibling?.className).toContain("zoom-select-wrap");
      expect(themeGroup?.compareDocumentPosition(exportButton!)! & Node.DOCUMENT_POSITION_PRECEDING).toBe(
        Node.DOCUMENT_POSITION_PRECEDING,
      );
    } finally {
      dispose();
    }
  });

  it("disables Export when requested", () => {
    const { dispose, host } = mountHeader({ exportDisabled: true });
    try {
      const exportButton = host.querySelector("button.toolbar-export-btn") as HTMLButtonElement;
      expect(exportButton.disabled).toBe(true);
    } finally {
      dispose();
    }
  });

  it("hides Export in snapshot mode", () => {
    const { dispose, host } = mountHeader({ snapshotMode: true });
    try {
      expect(host.querySelector("button.toolbar-export-btn")).toBeNull();
      expect(headerText(host)).toContain("DARK");
      expect(headerText(host)).toContain("SYNC");
      expect(headerText(host)).toContain("WALL");
    } finally {
      dispose();
    }
  });
});
