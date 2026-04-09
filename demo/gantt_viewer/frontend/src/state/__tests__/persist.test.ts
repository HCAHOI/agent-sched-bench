import { createRoot } from "solid-js";

import { __resetPersistenceForTests, enablePersistence } from "../persist";
import { initialThemeMode, setThemeMode, setViewMode } from "../signals";

describe("persist", () => {
  beforeEach(() => {
    __resetPersistenceForTests();
    setViewMode("layered");
    setThemeMode("dark");
    window.localStorage.clear();
    document.documentElement.classList.remove("theme-light");
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

  it("writes the theme mode into localStorage and toggles the root class", async () => {
    createRoot((dispose) => {
      enablePersistence();
      setThemeMode("light");
      queueMicrotask(dispose);
    });

    await Promise.resolve();
    expect(window.localStorage.getItem("gantt.themeMode")).toBe("light");
    expect(document.documentElement.classList.contains("theme-light")).toBe(true);
  });

  it("restores light mode from localStorage", () => {
    window.localStorage.setItem("gantt.themeMode", "light");
    expect(initialThemeMode()).toBe("light");
  });
});
