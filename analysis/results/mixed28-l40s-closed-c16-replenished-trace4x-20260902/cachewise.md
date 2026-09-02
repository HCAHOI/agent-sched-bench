# CacheWise with replenished background and 4x tool replay

Date: 2026-09-02

## Configuration

The model, GPU, 28 measured tasks, 16 active-session limit, background
replenishment, 4x tool replay, vLLM settings, and 1,800-second request timeout
match the paired FCFS run. CacheWise used the official predictor and scheduling
policy. Tool-duration curves were divided by four to match replayed tool time.

- Run commit: `1311fab`.
- Remote artifact directory:
  `/home/Ubuntu/agent-sched-bench/results/mixed28-l40s-closed-c16-replenished-trace4x-qwen3-4b-cachewise-scaled-20260902/cachewise-r1`.

## Result

This run is invalid for the throughput comparison. It ended with cell and
simulator exit code 1: 27 measured tasks succeeded and
`PennyLaneAI__pennylane-2603` failed after one LLM request received no first
output for 1,800 seconds. Its final 16 source tool calls were consequently not
replayed.

During that request's 30-minute wait, vLLM completed 850 other requests. Across
the 180 ten-second service samples in the same interval, mean running requests
were 7.37 and mean waiting requests were 6.39, with a maximum of 9 waiting.
The GPU and service remained active and reported no OOM or HTTP 500 response.
This is consistent with the CacheWise waiting policy repeatedly selecting
shorter cache work while the older request had no age limit.

| Metric | CacheWise |
|---|---:|
| Successful measured tasks | 27/28 |
| Request timeout | 1,800 s |
| Other requests completed during timeout | 850 |
| Mean / maximum waiting requests during timeout | 6.39 / 9 |
| Mean running requests during timeout | 7.37 |
| Mean JCT / P95 JCT / Tasks/h | Not reported for failed run |
| TTFT / cache ratio / KV eviction | Final summary unavailable |

The failed run cannot support a speedup value against FCFS. It also does not
justify implementing common tool-gap skipping: the observed problem is request
starvation under sustained replenishment. The next experiment is
CacheWise + Oracle Length on the same workload.
