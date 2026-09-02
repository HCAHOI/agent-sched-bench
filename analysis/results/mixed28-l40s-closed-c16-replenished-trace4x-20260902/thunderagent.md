# ThunderAgent with replenished background

Date: 2026-09-03

## Configuration

The run used the same L40S, Qwen3-4B FP8 model, 28 measured tasks, 16 active
sessions, replenished background, and 4x tool replay as FCFS. ThunderAgent used
its official proxy with one vLLM backend.

- Run commit: `b8b436f`.
- Remote artifact directory:
  `/home/Ubuntu/agent-sched-bench/results/mixed28-l40s-closed-c16-replenished-trace4x-qwen3-4b-thunderagent-20260902/thunderagent-r1`.

## Result

The run was stopped after a measured-task failure made the comparison invalid.
At that point 26 measured tasks had succeeded,
`PennyLaneAI__pennylane-4161` had failed, and one measured task remained
active. The failed task's LLM request waited for 1,800 seconds. ThunderAgent
then logged `wait timeout` and forced the program to resume, at the same moment
the client request timed out. Its remaining 14 source tool calls could not be
replayed.

During that request's 30-minute wait, the backend completed 933 other requests.
The 180 service samples in the interval recorded mean running requests of 7.62,
mean waiting requests of 2.24, and a maximum of 6 waiting. Proxy and backend
successful-request counts remained equal, and there was no OOM or HTTP 500
response.

| Metric | ThunderAgent |
|---|---:|
| Successful measured tasks before stop | 26/28 |
| Request timeout | 1,800 s |
| Other requests completed during timeout | 933 |
| Mean / maximum waiting requests during timeout | 2.24 / 6 |
| Mean running requests during timeout | 7.62 |
| Mean JCT / P95 JCT / Tasks/h | Not reported for failed run |
| TTFT / cache ratio / KV eviction | Final summary unavailable |

ThunderAgent repeatedly paused and resumed this long-running program, but its
configured wait-time recovery did not finish before the client timeout. The
next experiment is Native Priority on the same workload.
