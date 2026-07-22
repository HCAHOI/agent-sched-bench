# Checkpoint — 2026-07-22

## Current position

Branch: `dev/kv-swap-profile-sweep-test`  
Checkpoint base: `d4a1c19`

The development-only prequential mechanism screen and its minimal same-history
follow-up are complete, copied locally, integrity-checked, and independently
reviewed. The next lane is implementation performance. Do not rerun or tune the
completed scientific screens while optimizing the scorer.

## Banked development evidence

The primary screen uses 100 disjoint initialization tasks (4,640 calls) and a
five-fold Fresh-277 stream (13,410 calls). Every Fresh task is held out exactly
once. One predeclared global PCG64 seed-0 task permutation is filtered into each
fold; there is no order sweep, CI, p-value, deployment certificate, or
unopened-stream claim.

Exact point estimates:

| Comparison | KV 3500 | KV 5000 | Status |
|---|---:|---:|---|
| `warmup_snapshot` vs deadline | +47.540 s | +151.043 s | headline mechanism candidate |
| `task` vs `warmup_snapshot` | +0.362 s (1/0/276 tasks) | +9.050 s (12/4/261) | KEEP for mechanism/confirmation |
| `call` vs `task` | 0 | 0 | no separate arm |

`call` state is not accidentally identical to `task`: it differs on
13,030/13,410 calls. Only one KV3500 trigger changes, on a call that reaches
neither trigger, and KV5000 changes no trigger. The realized-utility null is
therefore genuine.

Minimal history follow-up:

- `same_trace`: DROP without rerun. The prior within-task B1 result is about
  -398 s near the measured restore-cost regime, and `call` adds no realized
  utility over `task`.
- `same_repo`: DROP under its frozen either-cost-negative rule. Relative to the
  exact `warmup_snapshot`, KV3500 is -9.760 s (0/2/275 tasks) and KV5000 is
  +7.819 s (2/2/273). Only 61/277 tasks have sufficient repository history and
  only seven tasks are affected. Do not rescue this arm with post-hoc guard or
  cost tuning.

Current policy interpretation:

1. Retain `warmup_snapshot` as the simplest headline mechanism candidate.
2. Retain completed-task updates (`task`) as the only adaptive candidate. It is
   promising at KV5000 but remains single-order development evidence.
3. Do not retain separate `call`, `same_trace`, or `same_repo` policy variants.
4. The deployment lifecycle remains: initialize from disjoint development
   data, predict from state available before task *t*, update only after task
   *t* resolves, and revoke to fallback on a predeclared harmful-only monitor.

Primary artifacts:

- `analysis/development/prequential-profile-update-2026-07-21.{json,md}`
- `analysis/development/prequential-profile-update-2026-07-21-run-state.json`
- producer commit `e6cca14`; remote-result checkpoint `6a3ecd8`

Same-repository artifacts:

- `analysis/development/same-repo-history-2026-07-21.{json,md}`
- records SHA-256
  `888b01d727811cd8ac57f25cde7b00394effad0022ad866a85a63c369c07d7ab`
- implementation commit `b1f46ce`; result commit `d4a1c19`

## Performance blocker

Task-boundary profile publication is already cheap:

- update p50/p95/p99/max: 0.0715/0.1541/0.5130/2.5922 ms.

Exact robust trigger scoring is not deployable:

- score p50/p95/p99/max: 5.846/15.990/16.725/18.019 s, summed over
  the two development cost panels.

Seconds-scale scoring is unacceptable. Parallel experiment scheduling reduced
batch wall time but did not fix this per-decision path.

The hotspot is:

```text
robust_utility_trigger_stats
  -> _node_utility_curves
     -> _utility_sum
        -> _utility_matrix  # materializes N samples × C candidates
```

in `src/trace_collect/tool_latency_utility_clock.py`. `_score_row` in
`src/trace_collect/tool_latency_prequential.py` already caches a trigger by
prior-node state and cost, but cache misses still execute the exact matrix
scorer.

## Optimization contract

Optimization must preserve the existing estimator before considering an
approximate method:

- Candidate set remains exactly `{0, threshold, L, L - KV}` within bounds.
- Preserve the selected node, its parent, the full curve, and every non-empty
  leave-one-task-out curve.
- Preserve the earliest trigger unanimously strictly better than every later
  choice; ties wait.
- Preserve restore-cost accounting, strict fire inequalities, deterministic
  ordering, and fail-closed validation.
- No new datasets, task orders, broad sweeps, policy arms, or scientific
  hyperparameters.
- Completed result artifacts are frozen; performance work must not rewrite
  their conclusions.

### Step 1 — exact sufficient-statistic scorer

Replace the per-curve `N × C` utility matrix with sorted latency values plus
prefix counts/sums. The utility is piecewise linear with breakpoints at `L` and
`L - KV`, so each curve can be evaluated exactly by binary search or a sorted
sweep without sampling. Optimize total/parent/leave-one-task-out aggregation
only after the per-curve replacement is proven equivalent.

Required proof:

1. Exhaustive comparison with the current scorer over all frozen prior nodes,
   both costs, parents, and leave-one-task-out curves.
2. Exact trigger identity and identical strict accept/fallback decisions. Any
   floating-point delta must be bounded and shown unable to cross a decision
   boundary; byte-identical decision artifacts remain the preferred bar.
3. Focused unit tests for `L`, `L-KV`, threshold, ties, short-call restore,
   empty LOO remainder, and parent disagreement.
4. End-to-end benchmarks on real profiles, not a synthetic microbenchmark.
5. Minimum gate: p99 and max score latency must both leave the seconds regime
   on the same 8-core reference environment. This is only an optimization
   floor, not a deployment SLO.

### Step 2 — cache and incremental state

Once the exact sum path is fast, exploit the `task` contract: model state is
constant within a task. Reuse exact trigger statistics for repeated selected
nodes/costs and rebuild or increment indexes only after a completed task is
published. Measure cache hit/miss latency separately.

### Step 3 — approximation only if exact remains insufficient

Do not start with top-k or unbounded random sampling:

- top-k overweights long calls and can omit short-call exposure/restore harm;
- random sampling can miss the single harmful task that controls the robust
  unanimous rule;
- both introduce sample-size, seed, and error-tolerance choices.

If exact prefix sums plus caching still miss the latency requirement, use a
separate, reviewed approximation with deterministic error bounds and exact
fallback near the guard boundary. Report decision disagreement and utility
regret against the frozen exact oracle. Approximation is a method change and
must not be mixed into the exact optimization commit.

## Immediate next action

Profile and replace `_utility_sum` with an exact prefix-sum/sweep evaluator,
then run the equivalence proof before touching robust selection or sampling.
Do not spend another experiment run to measure code that has not passed the
mandatory independent review gate.

## Repository hygiene

The working tree currently contains extensive unrelated staged, unstaged, and
untracked reorganization from another active lane. Do not reset, clean, amend,
or include those changes. The commit containing this checkpoint must contain
only `analysis/CHECKPOINT-20260722.md`; scorer optimization starts in a later
commit.
