# Fresh-Sample Offline-Gated Robust Confirmation

Date: 2026-07-12

Status: method and inference protocol frozen before a confirmation trace root
is supplied. No confirmation result exists yet.

## Question

On one untouched real agent workload, what is the paired task-cluster
uncertainty of `offline_gated_robust_clock - robust_clock` at each fixed action
cost?

## Freshness Boundary

The confirmation manifest must identify one trace root that:

- was not used to design, debug, calibrate, or select the current method;
- was not included in SWE-Rebench, Terminal-Bench GLM-5.2, or
  ScienceAgentBench analyses reported on 2026-07-11;
- contains real completed agent operations rather than mocks, simulations, or
  smoke-only traces;
- is included in full according to a manifest `task_ids_file` fixed before
  latency extraction. The extracted logical task IDs must match it exactly;
  tasks cannot be silently dropped after observing tool calls.
- has a non-empty canonical trace-metadata `instance_id` for every trace. Path
  fallback is forbidden because logical tasks, not files, define folds and
  bootstrap clusters.

Existing development corpora and the small BrowseComp, DeepResearch, and
Terminal qwen smoke runs in this workspace are ineligible. Dataset identity is
recorded for provenance but is never passed to the policy.

## Frozen Policy Evaluation

- Five deterministic outer folds over complete logical tasks, sorted by
  `task_id` and assigned round-robin.
- Four deterministic inner task folds inside each outer profile.
- Costs: 500 through 5,000 ms in 500 ms increments; guard 0 ms.
- Command field `command`, prefix depth 4, leading `cd` retained.
- Prior eligibility: one call and one task, matching development.
- Arms: deadline, mean hazard, robust clock, rejected point-margin control, and
  offline-gated robust clock.
- One dimensionless robust guard is learned per outer fold and shared across
  all tools and costs in that fold.

No hyperparameter, cost, task subset, command grouping, or arm may change after
the trace manifest is supplied.

## Primary Paired Estimand

For logical task `j` and cost `c`, sum call-level utility differences:

```
d[j,c] = sum_calls_in_task_j (
    utility(offline_gated_robust_clock) - utility(robust_clock)
)
```

The observed workload-total paired effect is `D[c] = sum_j d[j,c]`. Repeated
calls retain their real systems cost, while uncertainty treats each logical
task as one independent cluster.

## Frozen Uncertainty Procedure

- Resampling unit: complete logical task.
- Replicates: 50,000.
- PRNG: NumPy `Generator(PCG64)`, seed 0.
- Each replicate samples exactly `N` task IDs with replacement from the `N`
  observed tasks and sums their full contribution vectors across all costs.
- Pointwise interval: two-sided 95% percentile interval.
- Simultaneous interval: Bonferroni-adjusted two-sided 95% family interval
  across the ten costs. Each cost therefore uses percentile quantiles 0.0025
  and 0.9975.
- Inference is conditional on the fitted outer-fold policies; folds are not
  retrained inside bootstrap replicates.

At a cost, label the paired result:

- `positive` only if the simultaneous lower bound is strictly above zero;
- `harmful` only if the simultaneous upper bound is strictly below zero;
- `inconclusive` otherwise.

Zeros are included in `inconclusive`. No pooled cross-cost sum, favorable
subset, pointwise-only interval, or development result can override these
labels. There is no binary whole-method success declaration.

## Required Reporting

For every cost preserve and report:

- observed paired milliseconds and normalized paired effect (`ms / cost`);
- pointwise and simultaneous intervals;
- positive/harmful/inconclusive label;
- robust and gated early-fire and early-short-fire counts;
- every task's paired contribution, largest positive/negative tasks, and fold
  membership;
- learned guards and fold-level paired effects.

The full task contribution matrix and bootstrap configuration remain in the
machine-readable output. Task concentration is a limitation, not evidence to
discard a task.

## Integrity and Review Gate

The runner must reject stale output, incomplete cost panels, task overlap,
policy triggers outside robust/deadline, a manifest task-count mismatch, and a
trace root overlapping any declared excluded source. It must snapshot and hash
the manifest, trace inputs, source, environment, raw fold decisions, pooled
results, and uncertainty output.

Path exclusion is supplemented by exact content hashes against every canonical
trace in the three required development roots. A copied development trace is
rejected even when renamed or moved. Phase-2 review must still inspect semantic
collection lineage and task overlap that byte hashes cannot detect.

All manifest, task-list, trace, protocol, and source inputs are hashed before
trace parsing. The same inventory is revalidated after evaluation and before
the result inventory is written; any input mutation aborts the run.

No confirmation command may run until an independent reviewer approves the
manifest-driven split, unchanged policy configuration, paired contribution
math, task-cluster resampling, simultaneous interval, tests, provenance, and
the actual fresh-data manifest.
