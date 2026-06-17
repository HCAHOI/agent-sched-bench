import type { ResourceSample } from "../api/client";
import type { ResourceMetric } from "../state/signals";

function sampleTime(sample: ResourceSample): number {
  if (typeof sample.t_abs === "number" && Number.isFinite(sample.t_abs)) {
    return sample.t_abs;
  }
  return sample.t;
}

function throughputMbPerS(
  current: number | null | undefined,
  previous: number | null | undefined,
  dtSeconds: number,
): number {
  if (!Number.isFinite(dtSeconds) || dtSeconds <= 0) {
    return 0;
  }
  const delta = (current ?? 0) - (previous ?? 0);
  return Math.max(0, delta) / dtSeconds;
}

/**
 * Format a resource value (MB or MB/s) at fixed 3-decimal precision. Fixed
 * precision keeps readings comparable and copyable for quantitative analysis;
 * we deliberately avoid magnitude-adaptive rounding.
 */
export function formatMetricValue(value: number): string {
  return Number.isFinite(value) ? value.toFixed(3) : "0.000";
}

/** Cumulative-counter fields that can be interpolated over a time window. */
type CumulativeField = "disk_read_mb" | "disk_write_mb" | "net_rx_mb" | "net_tx_mb";

/**
 * Interpolate a cumulative counter value at an absolute epoch time. The
 * timeline is assumed sorted by t_abs (as served). Querying the counter at a
 * span's start and end and subtracting gives the exact volume over that
 * window — immune to sampling-interval gaps, since it reads a monotonic
 * counter rather than counting samples that fall inside the window.
 */
export function cumulativeValueAt(
  timeline: ResourceSample[],
  tAbs: number,
  field: CumulativeField,
): number {
  if (timeline.length === 0) {
    return 0;
  }
  const first = timeline[0];
  const last = timeline[timeline.length - 1];
  if (tAbs <= first.t_abs) {
    return first[field] ?? 0;
  }
  if (tAbs >= last.t_abs) {
    return last[field] ?? 0;
  }
  for (let i = 1; i < timeline.length; i += 1) {
    const right = timeline[i];
    if (right.t_abs >= tAbs) {
      const left = timeline[i - 1];
      const range = right.t_abs - left.t_abs;
      const frac = range === 0 ? 0 : (tAbs - left.t_abs) / range;
      const lv = left[field] ?? 0;
      const rv = right[field] ?? 0;
      return lv + (rv - lv) * frac;
    }
  }
  return last[field] ?? 0;
}

export function resourceMetricUnit(metric: ResourceMetric): string {
  switch (metric) {
    case "cpu":
      return "%";
    case "memory":
      return "MB";
    case "mem_total":
    case "mem_read":
    case "mem_write":
    case "disk_total":
    case "disk_read":
    case "disk_write":
      return "MB/s";
    case "none":
      return "";
  }
}

export function resourceMetricValueAt(
  timeline: ResourceSample[],
  index: number,
  metric: ResourceMetric,
): number | null {
  const sample = timeline[index];
  if (!sample) {
    return null;
  }
  switch (metric) {
    case "cpu":
      return sample.cpu_percent;
    case "memory":
      return sample.memory_mb;
    case "mem_total":
      return sample.memory_total_mb_s ?? null;
    case "mem_read":
      return sample.memory_read_mb_s ?? null;
    case "mem_write":
      return sample.memory_write_mb_s ?? null;
    case "disk_total":
    case "disk_read":
    case "disk_write": {
      if (index === 0) {
        return 0;
      }
      const previous = timeline[index - 1];
      const dtSeconds = sampleTime(sample) - sampleTime(previous);
      if (metric === "disk_total") {
        const read = throughputMbPerS(sample.disk_read_mb, previous.disk_read_mb, dtSeconds);
        const write = throughputMbPerS(sample.disk_write_mb, previous.disk_write_mb, dtSeconds);
        return read + write;
      }
      if (metric === "disk_read") {
        return throughputMbPerS(sample.disk_read_mb, previous.disk_read_mb, dtSeconds);
      }
      return throughputMbPerS(sample.disk_write_mb, previous.disk_write_mb, dtSeconds);
    }
    case "none":
      return 0;
  }
}

export function resourceMetricValues(
  timeline: ResourceSample[],
  metric: ResourceMetric,
): Array<number | null> {
  return timeline.map((_sample, index) => resourceMetricValueAt(timeline, index, metric));
}
