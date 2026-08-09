# CPU-Idle Short-Null RSS Sensitivity

## Status and question

This is an openly post-outcome amendment on development-exposed SQLGlot data.
The conservative RSS-safety result is unchanged: it treated every null source
RSS as 16,000 MB and failed. Visible before this amendment were the Clause-KB
and Task-Aware completion reductions (33.392% and 25.984%) and their exposure
totals (12,390 and 12,439).

The amended question is narrower:

> If short RSS nulls caused only by insufficient sampling follow the existing
> short-null Low policy, does either frozen predictor retain useful CPU-idle
> backfill without any exposure against measured or long-unknown RSS?

The result is a sensitivity analysis. It cannot establish physical safety of
the short-null assumption.

## Frozen population and execution

- The same exposed 50 SQLGlot tasks, 32 orders, load 40, 8 CPU cores, and
  16,000 MB RSS as `cpu-idle-rss-safety-v1`.
- The same Serial-8, source-RSS FCFS, Clause-KB FCFS, and Task-Aware FCFS arms.
- The same frozen predictions, FCFS order, one speculative command, strict
  normal CPU priority, CPU profiles, work conservation, and bootstrap.
- Prediction-time inputs and reservations do not change. This amendment
  changes only source RSS accounting and the source-RSS comparison arm.

## Frozen short-null source policy

Apply the following rule independently to each matched eligible clause:

1. A numeric RSS observation keeps its measured value.
2. RSS null maps to 500 MB only when both are true:
   - memory availability is exactly `unknown:insufficient_rss_samples`;
   - clause latency is strictly below 500 ms.
3. Every other null remains unavailable.

Compose clauses using the canonical physical rules: sequential stages use max
and pipeline stages use sum. Cap the result at 16,000 MB. Any remaining null,
unmatched command, ambiguous structure, or invalid telemetry falls back to
16,000 MB. No binary, package, task, or command-name exception is allowed.

The 500 ms boundary and 500 MB reservation already belong to the canonical
prediction contract; they are not selected from the RSS-safety outcome.

Report separately:

- confirmed source-capacity exposures after this policy;
- commands and speculative overlaps involving short-null imputation;
- the number of imputed clauses and commands;
- every non-imputed fallback reason.

## Frozen decision gate

Evaluate Clause-KB and Task-Aware independently. An arm is eligible for one
controlled short-command memory calibration only if all are true:

1. mean per-order completion improves at least 5% versus Serial-8;
2. it captures at least 50% of the amended source-RSS FCFS reduction;
3. the paired order-bootstrap upper endpoint versus Serial-8 is below zero;
4. service inflation is at most 5%;
5. makespan regression versus Serial-8 is at most 1%;
6. confirmed source-capacity exposures are zero; and
7. predicted capacity, physical CPU, and CPU-work checks all pass.

If both pass, retain the arm with the larger mean completion reduction as the
development candidate. This selection rule is openly informed by the prior
result and is not confirmation. If neither passes, stop CPU-idle prediction on
exposed SQLGlot. Do not tune the threshold, fallback, predictor, reservation,
candidate order, or speculative width.

Passing authorizes only a controlled calibration that crosses duration and
memory independently, including a short high-RSS process. It does not
authorize physical `cpu.idle` scheduling or a safety claim.

## Implementation and verification boundary

Reuse the existing evaluator and simulator. Preserve availability provenance
when deriving the amended source map; do not alter the canonical telemetry or
KB schema. Add focused tests for the exact reason and strict 500 ms boundary,
compound composition, long-null fallback, and unchanged conservative default.
Obtain independent review before the single formal run. The prior result
artifact remains immutable.
