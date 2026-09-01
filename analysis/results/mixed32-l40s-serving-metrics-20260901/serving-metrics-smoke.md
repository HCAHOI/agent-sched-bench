# Serving metrics real smoke

Date: 2026-09-01

## Question

Validate the formal serving telemetry with a real model and an existing trace before
starting the paired scheduling runs. The smoke uses one SQLGlot task, 29 forced-length
LLM requests, 28 replayed tool calls, an NVIDIA L40S, and
`Qwen/Qwen3-4B-Instruct-2507-FP8`. CacheWise's vLLM fork runs with its scheduling
policy disabled.

## Timing defect found and fixed

The first run completed 57/57 actions and 29/29 requests, then failed while building
the final metrics file. An immediate-ready queue already created a run-start clock,
but omitted that clock and the per-task completion interval from its throughput
summary. Commit `433fcf4` records those values for immediate-ready queues and adds an
end-to-end regression test; remote commit `80380b1` contains the same change.

First-run artifacts are retained at:

`/home/Ubuntu/agent-sched-bench/results/qwen3-4b-serving-metrics-smoke-20260901`

They are diagnostic artifacts. The accepted smoke is the second run.

## Accepted smoke

Artifacts:

`/home/Ubuntu/agent-sched-bench/results/qwen3-4b-serving-metrics-smoke-r2-20260901`

- Simulator exit: 0
- Cell exit: 0
- Tasks: 1/1 successful
- Actions: 57/57 successful
- Requests: 29/29 returned the requested length
- Scheduled makespan: 224.235 s
- HTTP 4xx/5xx, CUDA errors, Xid, OOM, and GPU sampler errors: 0

## Metrics verified across recorded artifacts

| Metric | Accepted value |
|---|---:|
| Prompt tokens | 422,573 |
| Cached prompt tokens | 394,176 |
| Full-run cached prompt token ratio | 93.280% |
| Generation tokens | 3,810 |
| Whole-run generation throughput | 16.991 tokens/s |
| First-request TTFT | 0.138 s |
| Subsequent-request TTFT p95 | 0.292 s |
| Preemptions | 0 |

The per-request token sums exactly equal the start-to-finish Prometheus counter
deltas. The headline cache statistic is
`sum(request cached prompt tokens) / sum(request prompt tokens)`. It does not read
vLLM's recent-1000 display value.

KV collection contains 289 consecutive batches (`seq=0..288`), 308 events, and
1,997 stored blocks. Tail replay completed after two empty rounds, with zero missing
sequence ranges and zero cache clears. This single-task run generated zero removed
blocks and zero preemptions, so the later multi-task run must exercise and preserve
those event paths.

GPU telemetry contains 219 samples inside the measured run, with a maximum sample
gap of 1.029 s. GPU utilization averaged 28.20%. GPU memory activity averaged
20.36%, reached p95 64%, and peaked at 72%. Memory activity means the percentage of
each sampling period during which global device memory was read or written. A GB/s
measurement requires a separate hardware counter that this host does not expose.

## Decision

The formal runner now has a passing real-model telemetry path. Every later method
run must produce `serving_metrics.json`, `request_metrics.jsonl`, complete KV event
state, both Prometheus snapshots, and gap-checked GPU samples; a missing or
inconsistent artifact fails that run.
