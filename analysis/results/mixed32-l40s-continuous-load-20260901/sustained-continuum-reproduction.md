# 32-task sustained-load Continuum reproduction

Date: 2026-09-01

## Result

This run exposed a starvation failure for the first request of a replacement
program. `tobymao__sqlglot-2450__replacement-0003` received no first token for
1,800 seconds and hit the client timeout. The event stopped replacement
admission and removed the 28 active background tasks, while the remaining
measured tasks continued under a much lighter load. Full-run JCT and throughput
therefore cannot support a Continuum speedup claim.

| Diagnostic only | Value |
|---|---:|
| Successful measured tasks | 32 / 32 |
| Mean JCT | 48.367 min |
| P95 JCT | 145.927 min |
| Tasks / h | 9.150 |
| Apparent rate vs. sustained FCFS | 1.717x |

The diagnostic values use the original 32 task arrivals and terminal action
times. More than one hour of the measured tail ran after the replacement load
had stopped.

## Configuration

- GPU and model: one NVIDIA L40S with
  `Qwen/Qwen3-4B-Instruct-2507-FP8`.
- Measured workload: the same fixed 32-task manifest, physical tool execution,
  concurrency 32, and scheduled arrivals used by sustained FCFS.
- Sustained load: replacement delay mean 50 s and seed 42; measurements include
  only the original 32 tasks.
- Serving: Continuum reproduction, batch limit 8, 131,072-token context, prefix
  caching, eager execution, and GPU memory utilization 0.95.
- Prefill profile:
  `/home/Ubuntu/agent-sched-bench/results/continuum-prefill-qwen3-4b-l40s-20260901/prefill.json`.
- CPU allocation: vLLM cores 0-2; task containers cores 3-11 with at most two
  cores per container.
- Run directory:
  `/home/Ubuntu/agent-sched-bench/results/mixed32-l40s-sustained-real-tools-qwen3-4b-continuum-reproduction-r2-20260901`.
- Run code: commit `6206688c323d5669012c36faf9efefca15e1f465`.

## Failure evidence

- The failed request ran from 19:44:10.007 to 20:14:10.024 UTC and ended with
  `ReadTimeout` before the first response block. The matching original program
  request reports 864 prompt tokens, so input size does not explain the wait.
- During the same window, the service completed about 477 other requests. Across
  180 ten-second samples, waiting requests ranged from 9 to 25 and running
  requests ranged from 1 to 8. The service logged no CUDA, HTTP 5xx, or engine
  exception.
- Continuum's retention benefit begins after a program has completed a turn and
  entered a tool call. A new program's first request has no retained program
  state, so this replacement received no protection from that mechanism.
- Sixty-nine replacement tasks started. Forty completed successfully, one
  failed, and 28 active tasks were cancelled after the failure.
- All 32 measured tasks completed successfully with matching action sequences.
  The cell exited with code 1 because the background request failed, and no
  formal throughput summary was emitted.

ThunderAgent is evaluated next on the same workload. The later tool-sleep arm
will use the faster complete sustained-load method.
