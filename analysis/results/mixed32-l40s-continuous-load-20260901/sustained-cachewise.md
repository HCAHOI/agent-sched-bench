# 32-task sustained-load CacheWise

Date: 2026-09-01

## Result

This run exposed a severe starvation failure under sustained load. One measured
task, `PennyLaneAI__pennylane-6398`, waited 1,800 s for its 45th LLM response
without receiving a first token and then hit the client timeout. The failure
stopped replacement admission and cancelled the 16 background tasks active at
that time, so the full-run JCT and throughput are not comparable with sustained
FCFS.

| Diagnostic only | Value |
|---|---:|
| Successful measured tasks | 31 / 32 |
| Mean JCT over all terminal events | 45.071 min |
| P95 JCT over all terminal events | 137.077 min |
| Tasks / h over all terminal events | 10.011 |
| Tasks / h over successful tasks | 9.699 |

The apparent 1.879x rate over FCFS includes the failed task's early terminal
time and more than two hours with no replacement background. It is not a
CacheWise speedup result.

## Configuration

- GPU and model: one NVIDIA L40S with
  `Qwen/Qwen3-4B-Instruct-2507-FP8`.
- Measured workload: the same fixed 32-task manifest, physical tool execution,
  concurrency 32, and scheduled arrivals used by sustained FCFS.
- Sustained load: replacement delay mean 50 s and seed 42; measurements include
  only the original 32 tasks.
- vLLM: CacheWise policy enabled, batch limit 8, 131,072-token context, prefix
  caching, eager execution, and GPU memory utilization 0.95.
- CPU allocation: vLLM cores 0-2; task containers cores 3-11 with at most two
  cores per container.
- Run directory:
  `/home/Ubuntu/agent-sched-bench/results/mixed32-l40s-sustained-real-tools-qwen3-4b-cachewise-20260901`.
- Run code: commit `61d9848279665b070fb0efc2b19360648bb62728`.

## Failure evidence

- The failed request had a 44,671-token prompt and a requested completion of
  416 tokens. It ran from 14:27:21.519 to 14:57:21.614 UTC and produced no
  first SSE response block.
- During the same 1,800 s, the server completed 1,144 other requests; 1,136
  began later and completed before this request. The GPU averaged 99.89%
  utilization, while vLLM reported 5-23 waiting requests and up to eight
  running requests. The service remained active throughout the timeout.
- CacheWise orders waiting work by additional KV blocks, then arrival time and
  request ID, without an age-based override. Repeated smaller requests can
  therefore continue passing a large request under permanent backlog.
- The preceding tool call lasted 2.187 s. Its fallback duration prediction was
  about 200.8 s when attached and about 514.2 s when the tool returned. During
  the 2.29 s between the prior response and this request, the server recorded
  861 removed and 865 stored KV blocks. These observations support a two-step
  explanation: the fallback estimate made the hot prefix easy to evict, then
  the waiting order kept the enlarged request behind smaller work.
- The identical request completed under sustained FCFS with 256 cached prompt
  tokens, 175.4 s TTFT, and 216.1 s total latency.
- Before the failure, 36 replacement tasks had started and 20 completed. At
  failure, 16 active replacements were cancelled and no new replacements were
  admitted. The remaining measured long tail consequently ran under a lighter
  workload.
- All 32 measured task status files and combined trace data were written. The
  simulator then remained asleep after the earlier failure instead of exiting;
  it was terminated after confirming zero live task workers and containers.
  The resulting simulator and cell exit codes are 143.
- The sustained replay cleanup defect was fixed in commit `7a1f614`; the remote
  equivalent is `b025b43`. A focused regression test passes.

This run establishes a CacheWise failure case for long-running programs under
sustained backlog. Continuum reproduction and ThunderAgent are evaluated next
with the same workload.
