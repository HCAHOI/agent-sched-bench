# Continuum Public with replenished background

Date: 2026-09-02

## Configuration

The run used the same L40S, Qwen3-4B FP8 model, 28 measured tasks, 16 active
sessions, replenished background, and 4x tool replay as FCFS. This is the
released Continuum Public preview with its fixed two-second policy, not the
unreleased estimator described by the paper.

- Run commit: `1311fab`.
- Remote artifact directory:
  `/home/Ubuntu/agent-sched-bench/results/mixed28-l40s-closed-c16-replenished-trace4x-qwen3-4b-continuum-public-20260902/continuum-public-r1`.

## Result

| Metric | FCFS | Continuum Public |
|---|---:|---:|
| Successful measured tasks | 28/28 | 28/28 |
| Mean JCT | 63.541 min | 32.332 min |
| P95 JCT | 126.717 min | 71.079 min |
| Scheduled makespan | 9,796.408 s | 4,702.364 s |
| Tasks/h | 10.289 | 21.436 |
| Tasks/h ratio | 1.000x | 2.083x |
| TTFT P50 / P95 / P99 | 59.183 / 97.037 / 110.539 s | 1.871 / 50.449 / 286.432 s |
| Request cached-prompt-token ratio | 5.795% | 74.465% |
| Prefix lookup token hit ratio | 4.253% | 58.440% |
| Preemptions | 0 | 188 |
| KV tokens removed | 48,499,104 | 15,325,440 |
| GPU utilization mean | 99.917% | 96.968% |
| DRAM read+write mean / P95 / maximum | 318.949 / 449.880 / 768.182 GB/s | 485.237 / 735.942 / 750.192 GB/s |

All 28 measured action sequences matched their source traces, with zero failed
actions and all 1,235 measured LLM calls recorded. The server completed 1,932
requests including background work. One completed background response was
cancelled during its following tool sleep before the trace writer persisted it;
the measured cohort is complete. Background execution started 27 sessions,
completed 12, and cancelled 15 when the measured cohort ended.

The measured cohort accumulated 15.088 task-hours of JCT. Applying the
whole-run server split to its LLM time gives this comparison:

| JCT component | FCFS | Continuum Public |
|---|---:|---:|
| Waiting, including admission | 21.361 h (72.0%) | 6.532 h (43.3%) |
| Active inference | 6.458 h (21.8%) | 6.749 h (44.7%) |
| Tool sleep | 1.780 h (6.0%) | 1.780 h (11.8%) |
| Other | 0.054 h (0.2%) | 0.027 h (0.2%) |

The improvement comes from 14.828 fewer accumulated task-hours waiting; active
inference time did not fall. P99 TTFT increased because some newly introduced
sessions waited much longer, but no request reached the 1,800-second timeout.

The workload and request outputs completed successfully (`simulate-exit-code`
0). The cell exit code was 1 because an extra CUPTI sample produced during
shutdown lasted outside the collector's expected 0.5--1.5-second interval.
The recorded DRAM CSV covers the complete scheduled window with a maximum gap
of 1.000 seconds, so the table uses a clearly named postprocessed summary. The
collector has been changed to skip such irregular samples instead of failing
the cell.
