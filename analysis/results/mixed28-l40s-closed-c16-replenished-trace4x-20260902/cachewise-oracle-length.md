# CacheWise + Oracle Length with replenished background

Date: 2026-09-02

## Configuration

This run used the same L40S, Qwen3-4B FP8 model, 28 measured tasks, 16 active
sessions, replenished background, and 4x tool replay as the paired FCFS run.
The Oracle score used the source output length and the additional KV blocks
required at scheduling time. Coefficients came from the paired FCFS service
metrics on the same machine and model:

- Prefill: 0.205500 ms per newly computed prompt token.
- Decode: 90.257846 ms per generated token.
- Remote artifact directory:
  `/home/Ubuntu/agent-sched-bench/results/mixed28-l40s-closed-c16-replenished-trace4x-qwen3-4b-cachewise-oracle-length-20260902/cachewise-oracle-length-r1`.

## Result

The run was stopped after a measured-task failure made the comparison invalid.
At that point, 26 measured tasks had succeeded,
`PennyLaneAI__pennylane-4161` had failed, and one measured task remained
active. The failed task's next LLM request received no first output for 1,800
seconds. Its remaining 43 source tool calls therefore could not be replayed.

During the 30-minute wait, vLLM completed 743 other requests. The 180 service
samples in that interval recorded mean running requests of 6.82, mean waiting
requests of 7.15, and a maximum of 13 waiting. Before the failure, the Oracle
made 3,111 recorded choices and differed from original CacheWise on 1,712 of
them. The service reported no OOM or HTTP 500 response.

| Metric | CacheWise + Oracle Length |
|---|---:|
| Successful measured tasks before stop | 26/28 |
| Request timeout | 1,800 s |
| Other requests completed during timeout | 743 |
| Mean / maximum waiting requests during timeout | 7.15 / 13 |
| Mean running requests during timeout | 6.82 |
| Oracle choices / choices changed | 3,111 / 1,712 |
| Mean JCT / P95 JCT / Tasks/h | Not reported for failed run |
| TTFT / cache ratio / KV eviction | Final summary unavailable |

Adding source output length changed scheduling frequently but did not prevent
an old request from waiting indefinitely. The next experiment is Continuum
Public on the same workload.
