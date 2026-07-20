# DACD mandatory review record

Reviewer: `/root/dacd_research_review`

## Round 1: REQUEST_CHANGES

- OOF folds could use different triggers under one context rule, followed by a
  different full-profile refit trigger.
- Student-t inference was invalid for dependent cross-fit losses and differed
  from the preregistered task-cluster bootstrap.
- Aggregation did not require or recompute probe artifacts.

Resolution: replaced cross-fit/refit with an honest task split, froze exact
trigger identity, deployed the unchanged fit model, and implemented simultaneous
task-cluster bootstrap certification.

## Round 2: REQUEST_CHANGES

- Aggregation could not prove that probe and held-out decisions came from the
  declared hashed inputs.
- The implementation hash manifest was not canonical.

Resolution: aggregation now reloads hashed inputs, recomputes the balanced
split and both frozen-policy decision sets, compares exact decisions by
`(sample_id, cost)`, and requires a canonical implementation dependency set.
The demonstrated stale-probe and swapped-eval attacks became regression tests.

## Round 3: APPROVE

No remaining blocking findings. Reviewer confirmed the honest split, exact
trigger mapping, task-cluster bootstrap, fail-closed policy, complete artifact
recomputation, and canonical source hashes. Focused verification: 60 passed;
`git diff --check` clean.
