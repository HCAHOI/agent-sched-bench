import { describe, expect, it } from "vitest";

import type { ResourceSample } from "../../api/client";
import { resourceMetricUnit, resourceMetricValueAt, resourceMetricValues } from "../resourceMetrics";

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
});
