import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { assignTracks } from "../tracks";

const __dirname_poly = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(__dirname_poly, "../../../../../../");

interface Action {
  type: string;
  action_type?: string;
  iteration?: number;
  ts_start?: number;
  ts_end?: number;
}

function loadActionsByIteration(tracePath: string, iteration: number): Action[] {
  const text = readFileSync(tracePath, "utf-8");
  const actions: Action[] = [];
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    const rec = JSON.parse(line);
    if (rec.type === "action" && rec.iteration === iteration) {
      actions.push(rec);
    }
  }
  return actions;
}

const DRB_TRACE = resolve(
  REPO_ROOT,
  "traces/deep-research-bench/qwen-plus-latest/20260416T053932/51/attempt_1/trace.jsonl",
);
const BC_TRACE = resolve(
  REPO_ROOT,
  "traces/browsecomp/qwen-plus-latest/20260416T054501/0/attempt_1/trace.jsonl",
);

const start = (a: Action) => a.ts_start ?? 0;
const end = (a: Action) => a.ts_end ?? 0;

describe("assignTracks against real research-agent smoke test traces", () => {
  it("places all 5 sequential searches from DRB on track 0", () => {
    const searches = loadActionsByIteration(DRB_TRACE, 1);
    expect(searches.length).toBe(5);
    const tracks = assignTracks(searches, start, end);
    expect(tracks).toEqual([0, 0, 0, 0, 0]);
  });

  it("distributes DRB concurrent fetches across >= 3 distinct tracks", () => {
    const fetches = loadActionsByIteration(DRB_TRACE, 2);
    expect(fetches.length).toBe(5);
    const tracks = assignTracks(fetches, start, end);
    expect(new Set(tracks).size).toBeGreaterThanOrEqual(3);
  });

  it("places all 5 sequential searches from BrowseComp on track 0", () => {
    const searches = loadActionsByIteration(BC_TRACE, 1);
    expect(searches.length).toBe(5);
    const tracks = assignTracks(searches, start, end);
    expect(tracks).toEqual([0, 0, 0, 0, 0]);
  });

  it("distributes BrowseComp concurrent fetches across >= 3 distinct tracks", () => {
    const fetches = loadActionsByIteration(BC_TRACE, 2);
    expect(fetches.length).toBe(5);
    const tracks = assignTracks(fetches, start, end);
    expect(new Set(tracks).size).toBeGreaterThanOrEqual(3);
  });
});
