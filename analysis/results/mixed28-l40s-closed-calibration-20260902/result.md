# Fixed-concurrency FCFS calibration

Date: 2026-09-02

## Decision

Use 16 concurrent agent sessions for the next closed-load comparison. A second
24-session calibration is unnecessary: 16 sessions already kept FCFS fully
occupied while preserving the trace-recorded tool gaps.

This 15-minute diagnostic selected the formal workload configuration. The
process was intentionally stopped after the observation window.

## Configuration

- GPU and model: one NVIDIA L40S with
  `Qwen/Qwen3-4B-Instruct-2507-FP8`.
- Serving method: CacheWise fork with CacheWise scheduling disabled.
- Workload: 28 existing L40S traces, all immediately ready; 16 worker slots.
- Tools: source-trace duration sleep and recorded results.
- vLLM: batch limit 8, 131,072-token context, prefix caching, eager execution,
  and GPU memory utilization 0.95.
- Observation window: 2026-09-02 12:16:08-12:31:08 UTC.
- Manifest: `analysis/development/mixed28-l40s-closed-calibration-v1/manifest.yaml`.
- Remote run directory:
  `/home/Ubuntu/agent-sched-bench/results/calibration-qwen3-4b-fcfs-c16-trace-sleep-v2-20260902/cachewise-disabled-r1`.

## Observations

| Metric | Value |
|---|---:|
| vLLM samples | 90 |
| Waiting present | 90/90 (100%) |
| Mean / maximum waiting requests | 8.38 / 12 |
| Mean running requests | 5.67 |
| Mean requests in vLLM | 14.04 |
| Mean / maximum KV-cache use | 38.93% / 62.8% |
| Completed HTTP 200 generations | 445 |
| GPU utilization mean | 99.00% |
| GPU zero-utilization samples | 1/875 (0.11%) |
| Device-memory activity mean | 60.08% |
| GPU power mean | 329.33 W |
| KV blocks stored / removed | 304,222 / 287,051 |
| KV tokens removed | 4,592,816 |

The mean waiting count was 8.51 in the first half and 8.24 in the second half.
The fixed 16-session population kept it bounded throughout the window. As tool
sleeps became more common, the service still had 4-7 running and 5-9 waiting
requests in the sampled snapshots, with 100% GPU use.

The partial task traces contain 429 completed per-request records. Their TTFT
P50/P95/P99/maximum was 13.83/39.76/44.14/49.47 seconds. The server completed
16 additional requests before the intentional stop reached their task action
writer, so these request statistics are diagnostic only.

KV event collection ended cleanly with consecutive event batches and no cache
clear. The runner recorded exit 143 because the observation window ended by
design. All 16 calibration containers were then removed; the GPU returned to
P8 with zero memory allocated.

The earlier start attempt used an old manifest whose first trace was absent on
the L40S. It stopped before any generation request and contributed no samples
to this decision.
