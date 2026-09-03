# Native Priority with replenished background

Date: 2026-09-03

## Configuration

The run used the same L40S, Qwen3-4B FP8 model, 28 measured tasks, 16 active
sessions, replenished background, and 4x tool replay as FCFS. The vLLM backend
used its native priority scheduler.

- Run commit: `b8b436f`.
- Remote artifact directory:
  `/home/Ubuntu/agent-sched-bench/results/mixed28-l40s-closed-c16-replenished-trace4x-qwen3-4b-native-priority-20260903/native-priority-r1`.

## Result

| Metric | FCFS | Native Priority |
|---|---:|---:|
| Successful measured tasks | 28/28 | 28/28 |
| Mean JCT | 63.541 min | 49.436 min |
| P95 JCT | 126.717 min | 97.937 min |
| Scheduled makespan | 9,796.408 s | 7,676.315 s |
| Tasks/h | 10.289 | 13.131 |
| Tasks/h ratio | 1.000x | 1.276x |
| TTFT P50 / P95 / P99 | 59.183 / 97.037 / 110.539 s | 15.768 / 55.734 / 461.652 s |
| First-request TTFT P50 / P95 / P99 | Not recorded | 247.816 / 1,429.708 / 1,524.338 s |
| Request cached-prompt-token ratio | 5.795% | 30.962% |
| Prefix lookup token hit ratio | 4.253% | 31.504% |
| Preemptions | 0 | 277 |
| KV tokens removed | 48,499,104 | 36,542,864 |
| GPU utilization mean | 99.917% | 98.149% |
| DRAM read+write mean / P95 / maximum | 318.949 / 449.880 / 768.182 GB/s | 379.625 / 727.609 / 752.912 GB/s |

All 28 measured action sequences completed with zero failed actions and all
1,235 measured LLM calls recorded. Background execution started 27 sessions,
completed 12, and cancelled 15 when the measured cohort ended. The simulator
and cell both exited with code 0. One completed background response was not
persisted before cancellation during its following tool sleep; measured-task
request accounting is complete.

The measured cohort accumulated 23.070 task-hours of JCT. Applying the
whole-run server queue/inference split to measured request latency gives:

| JCT component | FCFS | Native Priority |
|---|---:|---:|
| Waiting, including admission | 21.361 h (72.0%) | 13.266 h (57.5%) |
| Active inference | 6.458 h (21.8%) | 7.996 h (34.7%) |
| Tool sleep | 1.780 h (6.0%) | 1.780 h (7.7%) |
| Other | 0.054 h (0.2%) | 0.027 h (0.1%) |

Native Priority exceeded the 1.20x target by reducing accumulated waiting by
8.094 task-hours. It also produced a heavy first-request tail: first-request
TTFT reached 1,560.340 seconds, close to the 1,800-second client timeout. No
request timed out and all measured tasks completed, so this run is valid while
the tail remains an important limitation.
