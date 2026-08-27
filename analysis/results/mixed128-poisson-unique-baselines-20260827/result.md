# Unique-128 paper-baseline stress result

## Result

On 128 distinct coding-agent traces arriving as a Poisson process, GPU/KV-aware
scheduling becomes a first-order effect.  Relative to stock vLLM FCFS,
ThunderAgent reduces mean task completion time by 53.4% and increases throughput
by 57.2%.  The CacheWise reproduction records a 59.2% mean-task reduction and a
68.3% throughput increase, but its result is confounded by its different vLLM
fork.  The single-GPU SAGA KV subset is harmful: mean task completion increases
34.4%, throughput falls 16.8%, and every one of the 128 tasks becomes slower.

ThunderAgent and CacheWise improve task completion by concentrating service on
selected programs.  That creates a fairness cost: request p99 time-to-first-token
increases from 445 seconds under FCFS to 1,525 and 2,299 seconds respectively.
The main open problem is therefore to retain program-level completion gains while
bounding starvation, not merely to maximize prefix-cache hits.

## Workload and validity

- Development stress workload; not held-out confirmation.
- 64 PennyLane and 64 SQLGlot tasks, all task identifiers distinct.
- Poisson arrivals with rate 0.1 task/s; all tasks arrive within 1,337 seconds.
- One A100 80 GB; replay concurrency 128; vLLM `max_num_seqs=8`.
- 11,934 actions: 6,031 LLM requests and 5,903 trace-timed tool calls.
- All four arms complete 128/128 tasks with exact action and provider-request
  sequence matches and no OOM, XID, CUDA, HTTP 5xx, or replay error.
- Each arm receives the same 163,613,947 prompt tokens and 1,117,277 requested
  and returned tokens.  The per-request message/token/length digest is identical.
- Aggregate tool time is 48.3287 hours in every arm and differs by less than one
  second, so the large deltas are not explained by tool-runtime variation.

JCT is measured from a task's scheduled arrival to its completion.  Makespan is
from Poisson time zero to the last completion.  TTFT includes client/proxy wait.
GPU utilization and energy are integrated from Poisson time zero to makespan.

## Physical comparison

Lower is better for JCT, makespan, TTFT, and energy.  Higher is better for
throughput and prefix-cache hit rate.

| Method | Mean JCT, min | p95 JCT, min | Makespan, h | Tasks/h | Mean TTFT, s | p99 TTFT, s | Final prefix hit | Mean GPU util. | Energy, kWh |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FCFS | 257.2 | 464.2 | 8.91 | 14.36 | 269.0 | 445.2 | 13.6% | 89.4% | 2.45 |
| ThunderAgent | 119.8 | 239.9 | 5.67 | 22.58 | 108.2 | 1,524.9 | 77.0% | 74.1% | 1.36 |
| CacheWise reproduction | 104.8 | 214.0 | 5.30 | 24.17 | 91.6 | 2,299.0 | 38.4% | 70.4% | 1.21 |
| SAGA KV subset | 345.6 | 571.7 | 10.71 | 11.96 | 374.1 | 583.7 | 12.7% | 74.1% | 2.61 |

| Method | Mean-JCT delta | Makespan delta | Throughput delta | Mean-TTFT delta | p99-TTFT delta | Tasks faster than FCFS |
|---|---:|---:|---:|---:|---:|---:|
| ThunderAgent | -53.4% | -36.4% | +57.2% | -59.8% | +242.5% | 117/128 |
| CacheWise reproduction | -59.2% | -40.6% | +68.3% | -66.0% | +416.4% | 116/128 |
| SAGA KV subset | +34.4% | +20.1% | -16.8% | +39.1% | +31.1% | 0/128 |

## Mechanism evidence

### ThunderAgent

The official scheduler performs 3,158 program pauses, 3,237 resumes, and 128
releases.  Backend mean waiting falls from 53.8 to 6.2 requests and final prefix
hit rises from 13.6% to 77.0%.  The workload contains no duplicated task, so the
gain comes from normal within-agent conversation-prefix reuse.  Its proxy queue
is not included in the backend waiting counter; the comparable end-to-end TTFT
shows the resulting starvation tail.

### CacheWise reproduction

All 6,031 causal session-policy updates succeed.  Update cost is 87 ms/request
on average, 188 ms at p95, and 526 seconds in aggregate.  Mean inference and
task completion improve substantially, but a small set of requests waits up to
the multi-minute tail.  CacheWise uses the authors' different vLLM fork and the
generated-tool-to-existing-KV attachment is a causal reproduction of an
unpublished hook.  A policy-disabled control on that exact fork is required
before attributing the full gain to CacheWise.

### SAGA KV subset

The causal profile covers 5,900/6,031 requests, but coverage does not produce
reuse: final prefix hit is 12.7%, backend mean waiting is 61.6 requests, and all
128 tasks regress.  Policy updates cost 369 ms/request on average and 2,227
seconds in aggregate.  The profile's `exec` p95 is 105 seconds while the
evaluation workload's observed p95 is 331 seconds; fast-tool TTLs are only
6--10 ms.  This supports, but does not prove, a state-definition failure: raw
tool duration is not the same as time until the next LLM request.  The retained
runner did not snapshot SAGA's eviction and hard-fallback counters before
shutdown, so exact victim-level attribution is unavailable.

This is the published single-GPU WA-LRU/adaptive-TTL subset, not full SAGA.  It
does not include the private AFS/AEG implementation, periodic scheduling,
multi-GPU migration, or CUDA paths.

## Interpretation

The earlier low-pressure results and this stress result are consistent.  At
32 tasks and concurrency 16, ThunderAgent changed mean task time by -0.61% and
makespan by -0.09%.  In Unique-128, the FCFS queue reaches 119 requests and KV
occupancy reaches 99.9%; only then do program/KV policies have enough opportunity
to produce a large effect.

The current workload is more realistic than duplicated-task stress because all
task IDs are unique, but it is deliberately overloaded and uses only two
repositories.  Tool execution is trace-timed rather than physical, so this
result supports GPU/KV scheduling claims only.  Each method has one physical
repetition; system-level run-to-run uncertainty is not yet measured.

## Next decisions

1. Run a policy-disabled control on the exact CacheWise fork.
2. Treat ThunderAgent as the strong baseline and target its average-JCT gain
   while bounding request starvation.
3. Model causal return-to-LLM time, rather than raw tool duration, before another
   SAGA-derived run; validate on fresh tasks rather than this exposed workload.
4. Persist eviction, TTL-expiry, hard-fallback, and per-session residency
   counters in future physical runs.
5. Agentix and Continuum have not yet been evaluated on this Unique-128 workload;
   their older low-pressure results are not directly comparable.

## Evidence and recovery

- Workload: `analysis/development/mixed128-poisson-v1/manifest.yaml`
- FCFS: `/home/Ubuntu/mixed128-poisson-unique-fcfs-thunder-v2-20260825/fcfs-r1`
- ThunderAgent: `/home/Ubuntu/mixed128-poisson-unique-thunder-rerun-v1-20260826/thunderagent-r1`
- CacheWise: `/home/Ubuntu/mixed128-poisson-unique-cachewise-v1-20260826/cachewise-r1`
- SAGA: `/home/Ubuntu/mixed128-poisson-unique-saga-kv-v1-20260826/saga-r1`
- Code commits: FCFS `9229b37`, ThunderAgent/CacheWise `5af1787`, SAGA `facc4f6`
- The frozen SAGA profiles and training manifest are retained beside this file.
- The complete raw run archive is stored outside Git under
  `outputs/traces/mixed128-poisson-unique-baselines-20260827.tar.zst`.
- Archive SHA-256:
  `2c80b3b545cb57ee8ce9672099b1a48e9bc0ea52670ec1dff8c31f2e1cf0d307`.
