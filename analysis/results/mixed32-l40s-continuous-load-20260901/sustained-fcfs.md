# 32-task sustained-load FCFS

Date: 2026-09-01

## Result

The sustained-load exact-fork FCFS completed all 32 measured tasks
successfully while replacement tasks kept the serving queue busy.

| Metric | Value |
|---|---:|
| Mean JCT | 91.894 min |
| P95 JCT | 298.929 min |
| Tasks / h | 5.328 |
| Scheduled-arrival makespan | 21,621.224 s (6.006 h) |

The finite-batch FCFS recorded 9.900 Tasks / h, 47.033 min Mean JCT, and
134.660 min P95 JCT. Sustained load reduced throughput to 0.538x, raised Mean
JCT to 1.954x, and raised P95 JCT to 2.220x. It therefore supplied materially
more scheduling opportunity than the finite batch.

## Configuration

- GPU and model: one NVIDIA L40S with
  `Qwen/Qwen3-4B-Instruct-2507-FP8`.
- Measured workload: the fixed 32-task manifest, physical tool execution,
  concurrency 32, and scheduled arrivals ending at 1,489.506 s.
- Sustained load: a completed slot started the same source trace in a fresh
  container after an exponential delay with 50 s mean and seed 42. Measured
  metrics include only the original 32 tasks.
- vLLM: CacheWise fork with its scheduling policy disabled, batch limit 8,
  131,072-token context, prefix caching, eager execution, and GPU memory
  utilization 0.95.
- CPU allocation: vLLM cores 0-2; task containers cores 3-11 with at most two
  cores per container.
- Run directory:
  `/home/Ubuntu/agent-sched-bench/results/mixed32-l40s-sustained-real-tools-qwen3-4b-fcfs-20260901`.
- Run code: commit `61d9848279665b070fb0efc2b19360648bb62728`.

## Checks and secondary measurements

- 32/32 measured tasks succeeded; their 3,076 actions contained 1,554 LLM
  calls and 1,522 tool calls. Every action sequence matched its source trace.
- All 4,393 measured and background generations returned the requested output
  length and ended with `finish_reason=length`.
- Simulator and cell exit codes were 0. Logs contain no CUDA, Xid,
  out-of-memory, traceback, or HTTP error.
- Replacement tasks: 92 started, 61 completed, and 31 were cancelled when the
  measured cohort ended.
- Mean / max waiting requests: 21.800 / 30. Peak KV usage: 95.5%.
- Measured-request cached-token ratio: 24.656%. The whole-run ratio including
  background requests was 11.557%.
- Measured-request P50 / P95 / P99 TTFT: 41.052 / 175.594 / 205.013 s.
  Across all requests they were 130.856 / 194.117 / 220.980 s.
- Mean GPU utilization: 99.64%. Mean device-memory activity: 46.16%.
- Mean / P95 / maximum measured DRAM throughput: 318.61 / 485.54 /
  774.37 GB/s.
- A single `docker stats` poll timed out after five seconds among 313,844
  container samples. Task, request, GPU, and serving outputs completed.
- Physical execution produced 25 additional failed tool-return markers across
  ten measured tasks relative to the source traces. All tasks still completed,
  and the replay contract retained their full action sequences. A later sleep
  comparison will reuse the physical CacheWise tool times and returned results.

The next node is the same sustained workload with the original CacheWise
scheduling policy enabled.
