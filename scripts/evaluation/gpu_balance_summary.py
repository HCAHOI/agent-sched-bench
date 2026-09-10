"""Summarize cross-GPU load balance from a two-instance run's paired GPU samples.

Input is results/<run>/server/gpu-paired.csv (nvidia-smi at 200 ms, both GPUs
per query) and server/routing.jsonl, whose first dispatch and last finish bound
the workload window. Consecutive GPU-0/GPU-1 rows form a pair when at most
100 ms apart; a pair covers the time until the next pair unless that gap
exceeds 500 ms. Busy means utilization >= 50%, idle <= 10%. Definitions match
the 2026-09-09 gpu-balance-summary.json files under results/.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

PAIR_SKEW_S = 0.1
MAX_GAP_S = 0.5
BUSY_PCT = 50
IDLE_PCT = 10
GAP_PP = 20


def parse_ts(text: str) -> float:
    return datetime.strptime(text.strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()


def load_pairs(csv_path: Path) -> tuple[list[tuple[float, float, float]], int, int]:
    rows = []
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            rows.append((parse_ts(row["timestamp_utc"]), int(row["gpu_index"]), float(row["utilization_pct"])))
    pairs, skew = [], 0
    i = 0
    while i + 1 < len(rows):
        (t0, g0, u0), (t1, g1, u1) = rows[i], rows[i + 1]
        if {g0, g1} == {0, 1}:
            if abs(t1 - t0) <= PAIR_SKEW_S:
                pairs.append((min(t0, t1), u0 if g0 == 0 else u1, u1 if g1 == 1 else u0))
            else:
                skew += 1
            i += 2
        else:
            i += 1
    return pairs, len(rows), skew


def workload_window(routing: Path) -> tuple[float, float]:
    first = last = None
    with routing.open() as fh:
        for line in fh:
            row = json.loads(line)
            if row["event"] == "dispatch":
                first = row["timestamp_s"] if first is None else min(first, row["timestamp_s"])
            elif row["event"] == "finish":
                last = row["timestamp_s"] if last is None else max(last, row["timestamp_s"])
    assert first is not None and last is not None
    return first, last


def longest_run(flags: list[bool], durations: list[float]) -> float:
    best = cur = 0.0
    for f, d in zip(flags, durations):
        cur = cur + d if f else 0.0
        best = max(best, cur)
    return best


def summarize(run: Path) -> dict:
    pairs, rows, skew = load_pairs(run / "server" / "gpu-paired.csv")
    start, end = workload_window(run / "server" / "routing.jsonl")
    window = [p for p in pairs if start <= p[0] <= end]
    durations, kept, long_gap = [], [], 0
    for cur, nxt in zip(window, window[1:]):
        gap = nxt[0] - cur[0]
        if gap > MAX_GAP_S:
            long_gap += 1
            continue
        durations.append(gap)
        kept.append(cur)
    covered = sum(durations)
    u0 = [p[1] for p in kept]
    u1 = [p[2] for p in kept]
    diff = [abs(a - b) for a, b in zip(u0, u1)]
    both_busy = [a >= BUSY_PCT and b >= BUSY_PCT for a, b in zip(u0, u1)]
    both_idle = [a <= IDLE_PCT and b <= IDLE_PCT for a, b in zip(u0, u1)]
    one_idle = [(a >= BUSY_PCT and b <= IDLE_PCT) or (b >= BUSY_PCT and a <= IDLE_PCT) for a, b in zip(u0, u1)]
    over_gap = [d > GAP_PP for d in diff]

    def weighted(flags: list[bool]) -> float:
        return 100 * sum(d for f, d in zip(flags, durations) if f) / covered

    return {
        "run": run.name,
        "window_start_unix_s": start, "window_end_unix_s": end, "window_s": end - start,
        "collected_rows": rows, "paired_samples_in_window": len(window),
        "paired_coverage_pct": 100 * covered / (end - start),
        "excluded_skew_pairs": skew, "excluded_long_gap_pairs": long_gap,
        "utilization_pct": {
            "gpu0_mean": statistics.fmean(u0), "gpu1_mean": statistics.fmean(u1),
            "mean_absolute_simultaneous_difference": statistics.fmean(diff),
            "p95_absolute_simultaneous_difference": sorted(diff)[int(0.95 * (len(diff) - 1))],
        },
        "temporal_balance": {
            "both_busy_pct": weighted(both_busy), "both_idle_pct": weighted(both_idle),
            "one_busy_other_idle_pct": weighted(one_idle),
            "one_busy_other_idle_seconds": sum(d for f, d in zip(one_idle, durations) if f),
            "longest_one_busy_other_idle_s": longest_run(one_idle, durations),
            "gpu_difference_over20pp_pct": weighted(over_gap),
            "longest_gpu_difference_over20pp_s": longest_run(over_gap, durations),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, help="write the JSON for the last run here")
    a = p.parse_args()
    for run in a.runs:
        s = summarize(run)
        t, u = s["temporal_balance"], s["utilization_pct"]
        print(f"{s['run']}: window {s['window_s'] / 60:.1f} min, coverage {s['paired_coverage_pct']:.2f}%, "
              f"util {u['gpu0_mean']:.1f}/{u['gpu1_mean']:.1f}%, both busy {t['both_busy_pct']:.2f}%, "
              f"one idle {t['one_busy_other_idle_pct']:.3f}% ({t['one_busy_other_idle_seconds']:.1f} s, longest "
              f"{t['longest_one_busy_other_idle_s']:.2f} s), >20pp {t['gpu_difference_over20pp_pct']:.1f}% "
              f"(longest {t['longest_gpu_difference_over20pp_s']:.2f} s)")
    if a.out:
        a.out.write_text(json.dumps(s, indent=1) + "\n")


if __name__ == "__main__":
    main()
