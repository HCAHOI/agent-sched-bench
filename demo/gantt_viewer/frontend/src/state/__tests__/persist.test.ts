import { createRoot } from "solid-js";

import { __resetPersistenceForTests, enablePersistence } from "../persist";
import { setViewMode } from "../signals";

describe("persist", () => {
  beforeEach(() => {
    __resetPersistenceForTests();
    setViewMode("layered");
    window.localStorage.clear();
  });

  it("writes the view mode into localStorage", async () => {
    createRoot((dispose) => {
      enablePersistence();
      setViewMode("concise");
      queueMicrotask(dispose);
    });

    await Promise.resolve();
    expect(window.localStorage.getItem("gantt.viewMode")).toBe("concise");
  });
});
