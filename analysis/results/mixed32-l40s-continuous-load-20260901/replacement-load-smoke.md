# Sustained replacement load smoke

Date: 2026-09-01

This two-task physical-tool run checked the sustained-load plumbing before the
formal 32-task experiment. It is not a performance result.

- Both measured SQLGlot tasks succeeded with 58 measured LLM calls and 56 tool
  calls; simulator and cell exit codes were 0.
- The first completed slot started
  `tobymao__sqlglot-4208__replacement-0001` after an exponential delay.
- The original and replacement executions used different fresh container IDs.
- The replacement marker was present in the messages sent to vLLM: the stored
  request-message digest matched the marked source messages.
- The replacement produced 22 requests before the second measured task ended.
  It was then cancelled and its container was removed.
- The throughput summary retained exactly the two measured tasks. Request
  telemetry labeled 58 measured requests and 22 replacement requests.
- Serving telemetry completed successfully, including request, prefix-cache,
  KV-event, GPU, and DRAM-throughput outputs. No service, CUDA, container, or
  request errors were found.

Run directory:
`/home/Ubuntu/agent-sched-bench/results/mixed2-l40s-sustained-load-smoke-20260901`.

The sustained-load implementation is ready for the formal FCFS run using the
fixed 32-task manifest, a 50-second mean replacement delay, and seed 42.
