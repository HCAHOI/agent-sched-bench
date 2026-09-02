# 28-task Poisson background FCFS

Date: 2026-09-02

## Result

The exact-fork FCFS run completed all 28 measured tasks successfully under the
fixed low-rate background workload.

| Metric | Value |
|---|---:|
| Measured Mean JCT | 22.403 min |
| Measured P95 JCT | 59.827 min |
| Measured Tasks / h | 5.742 |
| Measured scheduled-arrival makespan | 17,556.266 s (4.877 h) |
| Recorded-request P99 TTFT | 5.285 s (`n=1,705`) |

Tasks / h uses the interval from the first scheduled arrival through the final
measured completion. The simulator's measurement wall time was 17,567.553 s
(4.880 h), including 11.287 s of setup before the first scheduled arrival; the
corresponding rate is 5.738 Tasks / h.

## Configuration and artifacts

- GPU and model: one NVIDIA L40S with
  `Qwen/Qwen3-4B-Instruct-2507-FP8`.
- Measured workload: 28 tasks with seed-42 Poisson arrivals scaled to 7 tasks/h;
  the last measured arrival was at 13,885.714 s.
- Background workload: 19 planned tasks with seed-43 Poisson arrivals scaled to
  1.5 tasks/h over 12 hours. Background prompts carry a session marker and reuse
  measured-task images.
- vLLM: CacheWise fork with scheduling disabled, batch limit 8, 131,072-token
  context, prefix caching, eager execution, and GPU memory utilization 0.95.
- CPU and tools: vLLM cores 0-2; physical tool containers on cores 3-11 with at
  most two cores per container; total task concurrency 32.
- Run code: commit `154570b`; model process 2026-09-02 02:04:46-06:59:46 UTC.
- Run directory:
  `/home/Ubuntu/agent-sched-bench/results/mixed28-l40s-poisson-background-real-tools-qwen3-4b-fcfs-20260902/cachewise-disabled-r1`.
- Main summaries: `output/throughput_summary.json` and
  `serving_metrics.json`; per-request data: `request_metrics.jsonl`.

## Completeness checks

- Measured cohort: 28/28 successful, zero failed actions, 2,442 actions formed
  by 1,235 LLM requests and 1,207 tool calls.
- Background: 10 started, 8 completed, 2 were cancelled when measurement
  finished, and 9 had not reached their scheduled arrival.
- The action traces contain 3,374 finalized actions: 1,705 LLM requests and
  1,669 tool calls. Their LLM split is 1,235 measured and 470 background
  requests.
- The server completed 1,707 chat requests: 1,235 measured and 472 background.
  The two additional background responses belong to the tasks cancelled during
  teardown: `PennyLaneAI__pennylane-5857__replica-002` completed `llm_28`, and
  `PennyLaneAI__pennylane-2603__replica-002` completed `llm_34`. Each partial
  OpenClaw trace records that LLM start followed by its next tool start.
  Cancellation during the tool preceded the end-of-iteration action writer, so
  these completed LLM calls do not appear in the combined action trace or
  `request_metrics.jsonl`.
- All 1,705 recorded generations returned the requested token count with
  `finish_reason=length`; every request used priority 0.
- Simulator and cell exit codes were 0. The vLLM log contained no HTTP 5xx,
  CUDA, Xid, out-of-memory, or traceback error. The model process returned HTTP
  200 for all 1,707 workload requests.
- Container resource collection covered all 38 started tasks with 28,373
  samples, zero dropped samples, and zero collection errors.

## Request and serving telemetry

| View | Requests | P99 TTFT | Prompt tokens | Completion tokens | Cached-prompt-token ratio |
|---|---:|---:|---:|---:|---:|
| Measured, recorded | 1,235 | 4.822 s | 36,103,360 | 228,197 | 94.751% |
| Background, recorded | 470 | 5.780 s | 16,083,161 | 89,914 | 94.824% |
| All recorded | 1,705 | 5.285 s | 52,186,521 | 318,111 | 94.774% |
| All server-completed | 1,707 | Unavailable | 52,250,755 | 318,304 | 94.778% |

The two server-completed calls omitted from per-request records had 30,841 and
33,393 prompt tokens, plus 111 and 82 completion tokens. Their sums of 64,234
prompt tokens and 193 completion tokens exactly equal the server-counter
surplus over the 1,705 recorded requests; the cached-prompt-token surplus is
63,232. The reported all-recorded P99 TTFT therefore covers the 1,705 finalized
per-request records. The two omitted TTFT observations cannot be reconstructed
from the cumulative counters.

- Whole-run generation throughput was 18.131 tokens/s. The cumulative vLLM
  prefix lookup token ratio was 94.778%.
- Mean GPU utilization was 32.08%; median utilization was 0% and P95 was 100%.
  Mean device-memory activity was 26.36%.
- Mean / P95 / maximum GPU memory read-plus-write throughput from CUPTI was
  182.81 / 650.93 / 745.23 GB/s.
- vLLM's 996 periodic samples recorded 0.004 waiting requests on average, a
  maximum of 1 waiting request, and 61.5% peak KV-cache use.
- KV telemetry recorded 2,757,392 removed tokens, zero preemptions, and zero
  recomputed prompt tokens.

The waiting-queue measurements show that this FCFS arm rarely presented more
than one queued request to the scheduler. The paired CacheWise arm therefore
tests this exact low-contention operating point. Under the fixed 1.20 Tasks / h
requirement, CacheWise must reach at least 6.890 Tasks / h, equivalent to a
scheduled-arrival makespan no greater than 14,630.222 s.
