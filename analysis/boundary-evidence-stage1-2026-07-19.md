# Candidate C Stage-1 boundary-evidence kill test

> **FINAL - complete corpus**
>
> EXPLORATORY. Durations replayed on our own hardware (our_hardware); totals and boundary times both from the segment-timeline re-run (no-mixing rule). Generated 2026-07-19T18:02:04.

**Verdict: KILL**

## Census

| quantity | value |
| --- | --- |
| input chains | 4824 |
| analysable chains (>=2 seg, has raw_total) | 4026 |
| tasks | 277 |
| boundary events | 4867 |
| events supported (both arms) | 4367 |
| events scored (log-score defined) | 4367 |
| events dropped: thin E | 1 |
| events dropped: thin E+B | 499 |
| support coverage | 89.7% |

## Metric (ii): decision divergence via hazard_recheck_ms

Certified operating point: rho=0.94, guard 0ms (threshold==kv). Pooled divergence: 72.915% over 43670 decisions.

| kv cost ms | decisions | divergence |
| --- | --- | --- |
| 500 | 4367 | 77.994% |
| 1000 | 4367 | 83.284% |
| 1500 | 4367 | 83.055% |
| 2000 | 4367 | 42.432% |
| 2500 | 4367 | 87.428% |
| 3000 | 4367 | 57.156% |
| 3500 | 4367 | 63.842% |
| 4000 | 4367 | 77.353% |
| 4500 | 4367 | 79.597% |
| 5000 | 4367 | 77.009% |

## Metric (i): paired log-score gain (E+B over E)

Mean gain -0.0137 nats; 95% task-clustered bootstrap CI [-0.4148, 0.1994] over 277 tasks / 4367 events. CI covers zero: True.

## Kill readout

KILL if pooled divergence < 1.0% OR the log-score gain CI is not strictly above zero; SURVIVE only if both hold.

- divergence below bar: False
- CI covers zero: True
- gain CI strictly above zero: False
- **verdict: KILL**
