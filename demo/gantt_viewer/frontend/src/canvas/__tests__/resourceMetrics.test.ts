import { describe, expect, it } from "vitest";

import type { ResourceSample } from "../../api/client";
import { cumulativeValueAt, formatMetricValue, resourceMetricUnit, resourceMetricValueAt, resourceMetricValues } from "../resourceMetrics";

describe("cumulativeValueAt", () => {
  // Monotonic write counter: 100 MB at t=10, 160 at t=20, 160 at t=30.
  const tl = [sample(10, 0, 0, 100), sample(20, 0, 0, 160), sample(30, 0, 0, 160)];

  it("interpolates the counter between samples", () => {
    expect(cumulativeValueAt(tl, 15, "disk_write_mb")).toBeCloseTo(130, 6);
  });

  it("yields an exact window delta regardless of sample alignment", () => {
    // Span [12, 28] straddles sample boundaries: 112 -> 160 = 48 MB.
    const start = cumulativeValueAt(tl, 12, "disk_write_mb");
    const end = cumulativeValueAt(tl, 28, "disk_write_mb");
    expect(end - start).toBeCloseTo(48, 6);
  });

  it("clamps to endpoints outside the timeline", () => {
    expect(cumulativeValueAt(tl, 0, "disk_write_mb")).toBe(100);
    expect(cumulativeValueAt(tl, 99, "disk_write_mb")).toBe(160);
    expect(cumulativeValueAt([], 5, "disk_write_mb")).toBe(0);
  });
});

describe("formatMetricValue", () => {
  it("uses fixed 3-decimal precision regardless of magnitude", () => {
    expect(formatMetricValue(0.018)).toBe("0.018");
    expect(formatMetricValue(0.0078)).toBe("0.008");
    expect(formatMetricValue(45.2)).toBe("45.200");
    expect(formatMetricValue(823.5712)).toBe("823.571");
  });

  it("falls back to 0.000 for non-finite input", () => {
    expect(formatMetricValue(0)).toBe("0.000");
    expect(formatMetricValue(Number.NaN)).toBe("0.000");
  });
});

function sample(
  tAbs: number,
  memoryMb: number,
  diskReadMb: number,
  diskWriteMb: number,
): ResourceSample {
  return {
    t: tAbs,
    t_abs: tAbs,
    t_real: tAbs,
    t_real_abs: tAbs,
    cpu_percent: 0,
    memory_mb: memoryMb,
    disk_read_mb: diskReadMb,
    disk_write_mb: diskWriteMb,
    net_rx_mb: 0,
    net_tx_mb: 0,
  };
}

describe("resourceMetrics", () => {
  it("computes disk throughput from adjacent cumulative samples", () => {
    const timeline = [
      sample(0, 100, 10, 20),
      sample(2, 110, 14, 26),
      sample(5, 120, 20, 35),
    ];

    expect(resourceMetricValueAt(timeline, 0, "disk_read")).toBe(0);
    expect(resourceMetricValueAt(timeline, 1, "disk_read")).toBeCloseTo(2);
    expect(resourceMetricValueAt(timeline, 1, "disk_write")).toBeCloseTo(3);
    expect(resourceMetricValueAt(timeline, 1, "disk_total")).toBeCloseTo(5);

    expect(resourceMetricValueAt(timeline, 2, "disk_read")).toBeCloseTo(2);
    expect(resourceMetricValueAt(timeline, 2, "disk_write")).toBeCloseTo(3);
    expect(resourceMetricValues(timeline, "disk_total")).toEqual([0, 5, 5]);
  });

  it("keeps cpu and memory as instantaneous metrics", () => {
    const timeline = [
      sample(0, 100, 0, 0),
      sample(1, 140, 0, 0),
    ];

    expect(resourceMetricValueAt(timeline, 1, "memory")).toBe(140);
    expect(resourceMetricUnit("memory")).toBe("MB");
    expect(resourceMetricUnit("disk_total")).toBe("MB/s");
  });

  it("uses sample-provided memory bandwidth values directly", () => {
    const timeline = [
      {
        ...sample(0, 100, 0, 0),
        memory_total_mb_s: null,
        memory_read_mb_s: null,
        memory_write_mb_s: null,
      },
      {
        ...sample(1, 120, 0, 0),
        memory_total_mb_s: 200.0,
        memory_read_mb_s: 120.0,
        memory_write_mb_s: 80.0,
      },
    ];

    expect(resourceMetricValueAt(timeline, 0, "mem_total")).toBeNull();
    expect(resourceMetricValueAt(timeline, 1, "mem_total")).toBe(200.0);
    expect(resourceMetricValueAt(timeline, 1, "mem_read")).toBe(120.0);
    expect(resourceMetricValueAt(timeline, 1, "mem_write")).toBe(80.0);
    expect(resourceMetricUnit("mem_total")).toBe("MB/s");
  });
});
