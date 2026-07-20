# Nested Offline-Probe Utility Guard

Date: 2026-07-11

Status: frozen before result generation. Exploratory development on three
analyst-exposed corpora; not a confirmation experiment.

## Question

Can sampled offline tool calls learn a deployment-specific early-action guard
that transfers to held-out tasks, without using dataset identity or
hand-setting a threshold from exposed evaluation results?

## Generalization Contract

- The policy implementation receives profile/eval rows and action costs. It
  does not receive a corpus or dataset name.
- Every outer fold independently learns its guard only from that fold's
  profile tasks. Different corpora and folds may produce different guards as
  a consequence of their sampled latency statistics.
- No value from the 100 ms evaluation sweep is used as a guard candidate or
  feature.
- Outer eval task latency, label, task history, corpus identity, and result
  metrics are unavailable during calibration.

## Fixed Outer Data

- The same five task-disjoint folds for SWE-Rebench qwen3.7-max,
  Terminal-Bench GLM-5.2, and ScienceAgentBench Verified qwen3.7-max.
- Action proxy costs: 500 through 5,000 ms in 500 ms increments.
- Guard: 0 ms; command field `command`; prefix depth 4; leading `cd` retained.
- Prior eligibility remains one sample and one task, matching the utility-clock
  development baseline.

All ten action costs are retained. They are inputs to one shared normalized
guard, not independently tuned operating points.

## Nested Offline Probe

Within each outer profile:

1. Partition complete logical tasks into four deterministic inner folds.
   Tasks are greedily balanced by call count using only task identity and row
   count, never latency values.
2. For each inner fold, fit the existing hierarchical empirical prior on the
   other three folds and score calls in the held-out fold.
3. For every call/cost, retain the mean-utility candidate trigger, normalized
   predicted band gain, short penalty, margin, and profile-only conditional
   probabilities of short/boundary/far regions among candidate survivors.
4. Compute the candidate's realized delta versus deadline on the held-out
   probe call. This label is used only for offline guard calibration.

Every outer-profile task appears exactly once as an inner probe task. The full
outer profile is then refit for outer-eval scoring after guard selection.

## Guard Selection

The score is predicted mean utility improvement over deadline divided by the
action cost. One dimensionless guard is shared across all tools and all ten
costs in the deployment sample.

- Candidate guards are exactly zero and the distinct positive OOF predicted
  margins observed in the offline probe; there is no manual threshold grid.
- At guard `g`, an early candidate is used only when its predicted normalized
  margin is strictly greater than `g`.
- Select the guard maximizing the sum of realized OOF delta divided by cost.
  Cost normalization prevents larger proxy costs from receiving mechanical
  calibration weight solely due to units.
- Utility ties select the more conservative, larger guard.
- A separate `never early` candidate has objective zero and wins ties. It is
  serialized as a null guard.

This is empirical calibration, not a guarantee. Call-total utility remains the
estimand, so repeated calls carry their real repeated cost.

## Arms and Endpoints

Arms: `deadline_only`, `mean_hazard`, `robust_clock`, and
`offline_probe_guard`.

Pool raw outer-OOF decisions and milliseconds. At every corpus/cost report:

- net utility and delta versus deadline;
- captured exact deadline headroom;
- boundary-band gain and early-short exposure penalty;
- early-trigger and early-trigger-on-short counts.

Also report every fold's selected guard, inner-probe objective, accepted count,
and worst accepted task delta. Guard differences are outcomes, not
corpus-specific configuration.

The aggregator must accept exactly `f1..f5`, reject stale or missing fold
files, verify outer task disjointness, require a complete common cost panel for
every sample, and exactly reproduce each fold summary from raw decisions. The
runner refuses to start when any corpus output directory already exists.

## Interpretation

- Existing three corpora and thresholds are exposed development data.
- A positive result motivates preregistered confirmation on a fresh deployment
  sample; it does not establish a universal guard.
- A negative result is retained and diagnoses whether one global calibrated
  guard is too coarse for node/task heterogeneity.

## Review Gate

No real outer-fold result may run until an independent reviewer approves the
nested split, no-leakage boundary, score/region math, guard ERM, conservative
tie behavior, pooled recomputation, tests, protocol, and runner.
