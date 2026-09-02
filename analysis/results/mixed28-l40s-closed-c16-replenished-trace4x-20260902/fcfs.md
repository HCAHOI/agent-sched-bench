# FCFS with replenished background and 4x tool replay

Date: 2026-09-02

## Configuration

- GPU and model: one NVIDIA L40S with
  `Qwen/Qwen3-4B-Instruct-2507-FP8`.
- Serving stack: CacheWise vLLM fork with CacheWise scheduling disabled.
- Workload: 28 measured tasks ready at time zero, 16 active session slots.
- Background: completed sessions restart after an exponential delay with a
  10-second mean; background work stops with the measured cohort.
- Tools: recorded results with source durations divided by four.
- vLLM: batch limit 8, 131,072-token context, prefix caching, eager execution,
  and GPU memory utilization 0.95.
- Run commit: `1e08ae0`.
- Remote artifact directory:
  `/home/Ubuntu/agent-sched-bench/results/mixed28-l40s-closed-c16-replenished-trace4x-qwen3-4b-fcfs-20260902/cachewise-disabled-r1`.

## Result

| Metric | FCFS |
|---|---:|
| Completed measured tasks | 28/28 |
| Mean JCT | 63.541 min |
| P95 JCT | 126.717 min |
| Makespan | 9,796.408 s |
| Tasks/h | 10.289 |
| Recorded requests | 2,041 |
| TTFT P50 / P95 / P99 | 59.183 / 97.037 / 110.539 s |
| Request cached-prompt-token ratio | 5.795% |
| Prefix lookup token hit ratio | 4.253% |
| GPU utilization mean | 99.917% |
| DRAM read+write mean / P95 / maximum | 318.949 / 449.880 / 768.182 GB/s |

The measured cohort issued 1,235 LLM calls and 1,207 tool calls. Its recorded
tool sleep totaled 1.780 task-hours. Total LLM request latency, including time
waiting inside vLLM, was 21.997 task-hours. The configured tool budget is close
to the earlier low-load LLM total of 1.846 task-hours; queueing under 16 active
sessions raised the observed LLM share to 92.5% of action time.

Background execution started 29 sessions, completed 14, and cancelled 15 when
the measured cohort ended. All measured action sequences matched their source
traces, with zero failed actions. GPU and KV collection completed successfully.

The 1.20x target requires CacheWise to reach at least 12.347 Tasks/h, equivalent
to a makespan of at most 8,163.673 seconds on the same measured cohort.
