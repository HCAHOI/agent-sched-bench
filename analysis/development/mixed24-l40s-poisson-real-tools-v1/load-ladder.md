# Mixed24 L40S load ladder

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
