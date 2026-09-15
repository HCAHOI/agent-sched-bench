# Mixed24 L40S load ladder

**Status 2026-09-15: stopped at L0. The pre-registered gate was read and not
met. L1 and L2 were never launched, so the ladder did not exit by its own
rule.** The pre-registration below is unchanged; see Outcome.

This ladder is fixed before reading the original CacheWise result. Every level
uses the same 24 sessions, order, model, tool containers, and random seed. Only
the Poisson arrival timeline changes.

| Level | Nominal load | Arrival rate | Arrival scale | Last arrival |
|---|---:|---:|---:|---:|
| L0 | 1.25x | 0.004390159 task/s | 1.0 | 5890.426913 s |
| L1 | 2x | 0.0070242544 task/s | 0.625 | 3681.516821 s |
| L2 | 4x | 0.0140485088 task/s | 0.3125 | 1840.758410 s |

The loop exits at the first level where all 24 tasks succeed and CacheWise
average task throughput is at least 1.30 times the exact-fork disabled-policy
control in tasks/hour. This is equivalent to a CacheWise makespan no greater
than the control makespan divided by 1.30. L2 is the final level. All attempted
levels remain in the report.

Images are pulled before the experiment clock and retained across runs.
Containers are created only when each task reaches its scheduled arrival.

## Outcome

L0 ran both arms on 2026-08-31 UTC. Receipts:
[`fcfs.md`](../../results/mixed24-l40s-cachewise-loop-l0-20260901/fcfs.md) and
[`cachewise.md`](../../results/mixed24-l40s-cachewise-loop-l0-20260901/cachewise.md).
Every number below is copied from those receipts; the ratio and the overshoot
are receipt values, not arithmetic done here.

| L0 quantity | Value |
|---|---:|
| FCFS control throughput | 6.917 Task/h |
| FCFS control makespan | 12,490.135 s |
| Gate: 1.30x control throughput | 8.993 Task/h |
| Gate: equivalent makespan ceiling | 9,607.796 s |
| CacheWise throughput | 8.368 Task/h |
| CacheWise makespan | 10,324.710 s |
| CacheWise over control | 1.210x |
| Makespan above the ceiling | 716.914 s |

**The gate was not met.** CacheWise reached 1.210x against a required 1.30x,
and its makespan was 716.914 s above the ceiling.

Integrity was clean in both arms, so the shortfall is not a run defect: 24/24
tasks succeeded, 1,127 LLM requests ran with output token counts matching
returned counts, every vLLM chat request returned HTTP 200, and no 5xx, CUDA
out-of-memory, XID, or traceback appeared in the logs. The CacheWise receipt
classifies itself as one exploratory run supplying descriptive evidence, and
that framing is unchanged here.

**The ladder stopped here.** The CacheWise receipt names L1 FCFS as the next
step, and L1 and L2 were never launched. No record in this repository states
why. What the repository does show is that subsequent work went to the
two-instance 2xL40S line:
[`MILESTONE-2-Multi-Instance.md`](../../../millstone/MILESTONE-2-Multi-Instance.md)
is dated 2026-09-10 and closes that exploration phase, after L0 finished on
2026-08-31.

The L1 and L2 manifests remain in the repository and are unused, with no
results:
[`../mixed24-l40s-poisson-real-tools-l1-2x/manifest.yaml`](../mixed24-l40s-poisson-real-tools-l1-2x/manifest.yaml)
and
[`../mixed24-l40s-poisson-real-tools-l2-4x/manifest.yaml`](../mixed24-l40s-poisson-real-tools-l2-4x/manifest.yaml).
