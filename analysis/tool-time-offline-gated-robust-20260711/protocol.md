# Offline-Gated Robust Utility Clock

Date: 2026-07-11

Status: frozen before implementation completion and before result generation.
Exploratory follow-up on three analyst-exposed corpora; not a confirmation
experiment.

## Question

Can sampled offline tool calls safely filter robust-clock early actions while
preserving deployment-specific calibration, without allowing a point-estimate
node to bypass task-cluster robustness?

## Motivation Fixed Before This Run

The preceding nested point-margin experiment exposed one failure mechanism:
a global call-weighted guard admitted a six-call/two-task `exec:export` node,
then one unseen task repeated the mistake many times. This follow-up tests a
general mechanism correction. It must not inspect dataset identity, command
names, exposed task names, or the failing cost region.

The new arm is a strict sub-policy of the existing robust clock. It cannot use
the earlier mean-hazard trigger after robust evidence has chosen to wait.

## Robust Candidate and Score

For one selected hierarchy node, the robust model family contains its full
empirical utility curve, every non-empty leave-one-task-out curve, its parent
curve, and every non-empty parent leave-one-task-out curve. Let `k_r` be the
existing robust trigger: the earliest candidate that every curve strictly
prefers to every later candidate and to no action. If no such early trigger
exists, `k_r = T` and the row is ineligible for calibration.

For an eligible robust trigger, define one dimensionless score:

```
s_r = min_d [U_d(k_r) - max(0, max_{k > k_r} U_d(k))] / c
```

where `d` ranges over the complete robust model family and `c` is action cost.
The score and trigger use profile calls only. A positive score is guaranteed
by robust-trigger construction; its magnitude measures the weakest supported
advantage over waiting.

## Nested Offline Calibration

Within each outer profile:

1. Partition complete logical tasks into the same four deterministic inner
   folds, balanced only by task ID and row count.
2. Fit the hierarchy on three inner folds and produce `k_r` and `s_r` for the
   held-out calls in the fourth.
3. For each eligible call/cost, compute realized normalized delta of `k_r`
   versus deadline. These held-out labels are used only for guard calibration.
4. Candidate guards are zero and the distinct positive OOF robust scores.
   A separate `never early` candidate has objective zero.
5. Select one guard maximizing the sum of realized delta divided by action
   cost. Objectives within absolute `1e-12` are treated as numerical ties;
   ties select the more conservative guard, and `never early` wins a zero tie.
6. Refit on the complete outer profile. Use `k_r` only when `k_r < T` and
   `s_r > guard`; otherwise use `T`.

One dimensionless guard is shared across every tool and all ten costs in one
deployment sample. Different folds or corpora may learn different values only
through their sampled task statistics. The implementation receives no corpus
or dataset field.

## Fixed Data and Configuration

- Existing five task-disjoint outer folds for SWE-Rebench qwen3.7-max,
  Terminal-Bench GLM-5.2, and ScienceAgentBench Verified qwen3.7-max.
- Costs: 500 through 5,000 ms in 500 ms increments. All points are reported.
- Guard: 0 ms; command field `command`; prefix depth 4; leading `cd` retained.
- Prior eligibility: one call and one task, matching existing baselines.
- Inner folds: four.

No value from either earlier threshold sweep is an input or guard candidate.

## Arms and Endpoints

Arms: `deadline_only`, `mean_hazard`, `robust_clock`, the rejected
`offline_probe_guard` mechanism control, and
`offline_gated_robust_clock`.

At every corpus/cost report raw held-out outer-OOF:

- net utility and delta versus deadline;
- captured exact deadline headroom;
- band gain and short-exposure penalty;
- early-trigger and early-trigger-on-short counts.

Report both calibration records for every fold and verify that every new-arm
trigger equals either the baseline robust trigger or the deadline. Pool raw
milliseconds only after strict fold/task/cost validation.

## Interpretation and Adoption Boundary

This arm was motivated by an exposed failure, so these corpora can establish
only whether the mechanism behaves as implemented. There is no binary success
criterion on this exposed development panel. Every corpus, fold, and cost is
reported descriptively; no favorable point, subset, pooled sum, or tolerance
will be used to declare empirical success after inspection.

The sparse-node case and strict-subpolicy property are mechanical audits, not
evidence of generalization. The analysis will report whether robust actions
were filtered or retained, their paired raw utility deltas versus both
deadline and robust clock, and fold heterogeneity without assigning a pass
label.

Any systems or generalization claim requires a frozen follow-up on a fresh
deployment/bridge sample.

## Review Gate

No real panel output may run until an independent reviewer approves the robust
score equivalence, strict-subpolicy invariant, task-OOF boundary, calibration
tie behavior, aggregation, provenance, tests, and this protocol.
