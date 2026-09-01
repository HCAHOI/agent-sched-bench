# Continuous replacement load implementation

Date: 2026-09-01

This node adds the sustained-load option for the 32-task physical-tool workload.
The original 32 tasks remain the measured cohort. After the final planned original
arrival, each completed slot replays the same source trace in a fresh container
after an exponential delay. The configured mean delay is 50 seconds and the seed
is recorded by the runner.

Replacement sessions receive a unique run ID and prepend that ID to the existing
system prompt sent to vLLM. This preserves reuse within one session while preventing
exact prompt reuse across cycles. When the measured cohort finishes, pending and
running replacements stop; failed measured or replacement tasks stop the run.

Mean JCT, p95 JCT, and Tasks/h continue to use only the original cohort. Serving
records include observed replacement requests and label whether each request belongs
to the measured cohort. The frozen 32-task traces contain 1,554 LLM calls, all with
a string system prompt suitable for the session marker.

Verification: 198 relevant tests passed, shell syntax and lint checks passed, and an
independent focused recheck found no remaining issue. No formal sustained-load result
was produced in this node; the existing finite-batch FCFS remains in progress.
