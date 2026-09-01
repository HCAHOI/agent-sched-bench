# 32-task finite-batch FCFS

Date: 2026-09-01

## Result

The finite-batch exact-fork FCFS completed all 32 tasks successfully.

| Metric | Value |
|---|---:|
| Mean JCT | 47.033 min |
| P95 JCT | 134.660 min |
| Tasks / h | 9.900 |
| Scheduled-arrival makespan | 11,636.870 s (3.232 h) |

This result is retained for a descriptive comparison with the sustained-load
FCFS. The sustained-load FCFS remains the baseline for the sustained CacheWise
comparison.

## Configuration

- GPU and model: one NVIDIA L40S with
  `Qwen/Qwen3-4B-Instruct-2507-FP8`.
- Workload: the fixed 32-task manifest, physical tool execution, concurrency
  32, and scheduled arrivals ending at 1,489.506 s.
- vLLM: CacheWise fork with its scheduling policy disabled, batch limit 8,
  131,072-token context, prefix caching, eager execution, and GPU memory
  utilization 0.95.
- CPU allocation: vLLM cores 0-2; task containers cores 3-11 with at most two
  cores per container.
- Run directory:
  `/home/Ubuntu/agent-sched-bench/results/mixed32-l40s-real-tools-qwen3-4b-fcfs-r2-20260901`.
- Run code: commit `5605abe3a405e2eb8081687964ba4e676bd62a6f`.

## Checks and secondary measurements

- 32/32 tasks succeeded; 3,076 actions contained 1,554 LLM calls and 1,522
  tool calls. Failed actions: 0.
- All 32 action sequences matched their source traces. All 1,554 generations
  returned the requested output length and ended with `finish_reason=length`.
- Simulator and cell exit codes were 0. All 1,554 chat requests returned HTTP
  200; the logs contain no CUDA, Xid, out-of-memory, traceback, or HTTP error.
- Mean / max waiting requests: 1.723 / 15. Peak KV usage: 97.2%.
- Whole-run request cached-token ratio: 74.084%. The cumulative vLLM prefix
  lookup counter ratio was 54.502%.
- P50 / P95 / P99 TTFT: 0.653 / 54.576 / 68.401 s.
- Mean GPU utilization: 48.91%. Mean device-memory activity: 33.73%.
- Mean / P95 / maximum measured DRAM throughput: 238.81 / 726.19 /
  775.95 GB/s.

The next step is a short plumbing run for sustained replacement load, followed
by the formal sustained-load FCFS using the same 32-task manifest.
