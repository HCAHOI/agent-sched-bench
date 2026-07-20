# A0 adjudication: k=2 re-check DP vs certified k=1

> **FINAL - complete corpus**
>
> EXPLORATORY. Original-trace latencies via the frozen manifest (swe-rebench-qwen3.7-max-fresh-seed42-skip150-n277). Generated 2026-07-20T06:00:08 (git 25b5ca9fb78a613d59bb04fe397207a46575d1f3).

**Verdict: COLLAPSE CONFIRMED**

Nodes: 670 distinct selected prior sample sets x 10 kv cells = 6700 cells at guard 0ms (threshold==kv), rho=0.94.

## Value equivalence (unpriced)

- max |k2_opt - k1| value gap: 0.000e+00 ms (tolerance 1.0e-06)
- max |paired delta| (induced decisions): 0.000e+00 ms
- value-tied churn cells (different trigger, zero value delta): 5485
- offending cells: 0

## Per-check overhead: domination

At overhead 1 ms/check: k=2 never strictly beats k=1 (max priced gap 0.000e+00 ms, holds=True). Genuine two-check (defer) policies strictly dominated in 1869 cells.
