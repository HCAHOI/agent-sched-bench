# Serving metrics real smoke

Date: 2026-09-01

## Question

Validate the formal serving telemetry with a real model and an existing trace before
starting paired scheduling runs. The smoke uses one SQLGlot task, 29 forced-length
LLM requests, 28 replayed tool calls, an NVIDIA L40S, and
`Qwen/Qwen3-4B-Instruct-2507-FP8`. CacheWise's vLLM fork runs with its scheduling
policy disabled.

## Timing defect found and fixed

The first run completed 57/57 actions and 29/29 requests, then failed while building
the final metrics file. An immediate-ready queue already created a run-start clock,
but omitted that clock and the per-task completion interval from its throughput
summary. Commit `433fcf4` records those values for immediate-ready queues and adds an
end-to-end regression test; remote commit `80380b1` contains the same change.

First-run artifacts remain at:

`/home/Ubuntu/agent-sched-bench/results/qwen3-4b-serving-metrics-smoke-20260901`

They are diagnostic artifacts. The current accepted end-to-end smoke includes DCGM.

## Accepted end-to-end smoke

Artifacts:

`/home/Ubuntu/agent-sched-bench/results/qwen3-4b-serving-metrics-dcgm-smoke-20260901`

- Simulator exit: 0
- Cell exit: 0
- Tasks: 1/1 successful
- Actions: 57/57 successful
- Requests: 29/29 returned the requested length
- Scheduled makespan: 223.244 s
- HTTP 4xx/5xx, CUDA errors, Xid, OOM, and sampler errors: 0

## Metrics verified across recorded artifacts

| Metric | Accepted value |
|---|---:|
| Prompt tokens | 422,573 |
| Cached prompt tokens | 394,176 |
| Full-run cached prompt token ratio | 93.280% |
| Generation tokens | 3,810 |
| Whole-run generation throughput | 17.067 tokens/s |
| First-request TTFT | 0.142 s |
| Subsequent-request TTFT p95 | 0.293 s |
| TPOT median / p95 | 21.398 / 21.869 ms |
| Decode throughput median | 46.733 tokens/s |
| Preemptions | 0 |

The per-request token sums exactly equal the start-to-finish Prometheus counter
deltas. The headline cache statistic is
`sum(request cached prompt tokens) / sum(request prompt tokens)`. It does not read
vLLM's recent-1000 display value.

KV collection contains 289 consecutive batches (`seq=0..288`), 308 events, and
1,997 stored blocks. Tail replay completed after two empty rounds, with zero missing
sequence ranges and zero cache clears. This single-task run generated zero removed
blocks and zero preemptions, so a multi-task run must exercise and preserve those
event paths.

## DCGM installation and GPU memory telemetry

The L40S host now has `datacenter-gpu-manager-4-cuda13` 4.6.1 installed, with
`nvidia-dcgm` enabled and active. GPU setup detects the CUDA major version, installs
the matching DCGM 4 package only when absent, enables the service, and verifies GPU 0
field 1005. Re-running the full GPU setup completed in six seconds and skipped the
package installation.

Inside the scheduled task window, field 1005 produced 224 valid samples with a
maximum gap of 1.006 s. The DRAM activity ratio averaged 16.575%, reached p95
58.444%, and peaked at 68.885%. All raw rows identify GPU 0, field 1005, and status
`OK`; the collector also saved a final sample 11.675 s after the scheduled task end.

NVIDIA defines field 1005 as the fraction of cycles in each DCGM interval during
which data was sent to or received from device memory. It is a continuous
memory-interface activity measure, not transferred GB/s. The existing `nvidia-smi`
`utilization.memory` series remains alongside it: 217 in-window samples, maximum gap
1.036 s, mean 21.493%, p95 64.2%, and maximum 73%.

Source: [NVIDIA DCGM profiling metrics](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html).

Implementation commits are `d716bea` locally and `02090e8` on the L40S host.

## Decision

The formal runner now has a passing real-model path for full-run cache counters,
per-request timing, complete KV events, NVML sampling, and DCGM field 1005. Every
later method run must produce `serving_metrics.json`, `request_metrics.jsonl`,
complete KV event state, both Prometheus snapshots, `gpu.csv`, and `dcgm.csv`;
missing, invalid, or interrupted telemetry fails that run.
