# Checkpoint — 2026-07-22

## Current position

Branch: `dev/kv-swap-profile-sweep-test`  
Checkpoint base: `d4a1c19`

The development-only prequential mechanism screen and its minimal same-history
follow-up remain frozen. The exact scorer optimization and the follow-on bounded
command-token cache are complete and independently reviewed. The completed
Fresh-277 replay remains the frozen scientific result; cache validation used
only the controlled 50+50 development benchmark.

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

## Exact scorer optimization — complete

The old scorer materialized a calls × candidates utility matrix for every
full-node and leave-one-task-out curve. The optimized scorer preserves the
candidate set and utility algebra under the numerical contract below while
using:

1. sorted latency prefix sums plus `searchsorted` for exact utility sums;
2. streamed full-node and leave-one-task-out curves, avoiding a task ×
   candidates result matrix;
3. immediate rejection of candidate prefixes that can never become viable;
4. a bounded 4 × candidates matrix only for task lists of at most four rows,
   where measured setup cost is lower than the prefix path; and
5. direct scalar utility evaluation instead of allocating a 1 × 1 NumPy
   matrix.
The numerical contract was fixed before the performance benchmarks: a utility
advantage must exceed a conservative floating-point accumulation bound;
advantages inside that bound wait as ties. This intentionally replaces raw
`> 0.0` only inside the derived error envelope, preventing summation order from
turning a numerical tie into an early action. It is the only policy-level
change, is not a tunable hyperparameter, and changed no decision in the frozen
or optimized Fresh-277 artifacts.

For $N$ calls, $T$ tasks, and $C$ candidates, one utility sum now uses
$O(N + C)$ scratch and $O(N + C\log N)$ work instead of an $O(NC)$ matrix.
Including exact leave-one-task-out curves, the robust scan is
$O(N + TC\log N + C\log C)$ in the worst case, but streams one curve at a time
with $O(N + C)$ working memory. Candidate suffix rejection reduces realized
work without changing that worst-case bound.

### Controlled 50+50 benchmark

One outer fold: 50 initialization tasks, 40 warmup tasks, 10 held-out tasks,
all five arms, both score costs, eight workers on the same remote environment.

| Implementation | Fold wall time | Relative to matrix |
|---|---:|---:|
| Original matrix scorer | 1121.76 s | 1.00× |
| Exact prefix sums | 206.26 s | 5.44× |
| Streamed LOO curves | 176.86 s | 6.34× |
| Candidate-suffix rejection | 125.02 s | 8.97× |
| Final exact fast paths | **102.88 s** | **10.90×** |

Final peak RSS was 178,328 KiB. The final and exact-prefix artifacts contain
14,356 records with identical summaries, triggers, utilities, and decisions
after removing runtime, publication timestamp, and artifact-hash metadata.

### Full Fresh-277 replay

The complete frozen protocol (100 initialization tasks, five Fresh-277 outer
folds, 13,410 development calls, five arms, two score costs) completed with
eight workers in **987.59 s (16m 28s)**. The frozen run took 7h12m, an observed
26.25× wall-time reduction despite using 8 workers instead of 25; this is an
end-to-end historical comparison, not a controlled hardware benchmark.

The latency objective is met. Values below are milliseconds, summed over the
two score panels:

| Arm | Version | p50 | p95 | p99 | max |
|---|---|---:|---:|---:|---:|
| `task` | frozen | 0.0268 | 12695.32 | 16236.04 | 18118.13 |
| `task` | optimized | **0.0209** | **70.65** | **83.94** | **108.30** |
| `call` | frozen | 5846.24 | 15990.87 | 16725.15 | 18019.08 |
| `call` | optimized | **16.85** | **86.69** | **89.77** | **115.48** |

All six optimized sidecar hashes validate. Streaming comparison covered
596,598 non-metadata records:

- zero trigger, utility, score, or policy-decision changes;
- eight rows differ only in `call_prior_task_count` by one;
- 88 rows differ only in call publication version/count/hash metadata; and
- aggregate readiness is 12,843/13,133 instead of 12,847/13,133.

Those remaining fields are scheduler/runtime observations, not scorer outputs.
Readiness is a wall-clock race between asynchronous publication and the next
eligible call, so scorer speed and worker contention can move those counters
even when the resulting call decisions are unchanged.
The scientific point estimates and conclusions remain unchanged. The optimized
replay is isolated under
`analysis/development/optimized-replay-2026-07-22/`; it did not overwrite the
reviewed artifacts.
The replay scorer is bound to source SHA-256
`174da20e6e29da074c1db1c3c00f7fd70d6cf9d93e910d11ecc7561da0c207fd`.
The commit candidate differs from that replayed file only by public docstrings
clarifying the already-executed numerical tie contract; no executable line
changed after the replay.

### Verification and review

- numerical-contract differential: 30,000/30,000 triggers match; maximum
  normalized-margin delta $3.39\times10^{-15}$;
- direct scalar utility: 100,000/100,000 random cases exactly match the shared
  utility matrix;
- 195 combined scorer, prequential, confirmation, pressure-headroom, and
  hazard-confirmation tests pass;
- independent review validated the near-tie numerical contract, task-imbalance
  roundoff bound, suffix-index alignment, full replay integrity, and the
  separation of controlled from historical speedups.
- final result review initially flagged raw strict-positive preservation;
  resolved by documenting the earlier human-approved error-bound tie contract
  and the post-replay doc-only source delta. Final status: CLEAN, safe to commit.

## Bounded command-token cache — complete

`command_features._normalized_tokens` now uses a process-local
`functools.lru_cache(maxsize=4096)` and returns an immutable tuple. Public key
builders still allocate fresh containers, so callers cannot mutate cached
state. The bound covers the controlled workload's 2,227 distinct commands
without retaining unbounded trace text; it changes no scoring or policy logic
or configuration at fixed causal inputs.

### Controlled 50+50 benchmark

All runs used one outer fold: 50 initialization tasks, 40 warmup tasks, 10
held-out tasks, all five arms, and both score costs on the same idle remote
environment. Three unprofiled live-timing A/B runs gave:

| Implementation | Elapsed seconds | Median | Relative |
|---|---|---:|---:|
| Cache disabled | 64.47, 65.29, 65.43 | 65.29 | 1.00× |
| Bounded cache | 57.29, 56.74, 57.30 | **57.29** | **1.140×** |

Median peak RSS increased from 178,176 KiB to 182,400 KiB (+4,224 KiB).
The cache recorded 96,480 hits and 2,227 misses in the profiled fold.
`_normalized_tokens` fell from 98,707 parses / 18.110 s cumulative to 2,227
parses / 0.494 s. End-to-end profiled elapsed time fell from 102.88 s to
77.58 s; the unprofiled median above is the deployment-relevant speedup.

Two different equivalence checks answer different questions:

1. **Estimator identity at fixed causal inputs.** A separately hash-pinned
   availability panel replayed identical update runtimes into cache-off and
   cache-on runs. Summaries match and 0/14,356 non-timing/hash records differ.
2. **Live self-timing behavior.** The real A/B feeds measured update runtimes
   back into causal publication readiness. Faster scoring made one additional
   prior row ready at five calls, changing ten candidate-margin fields (two
   costs each). Candidate and selected triggers, utilities, scores, policy
   decisions, point estimates, and summaries remain unchanged. These live
   sidecars are deliberately not described as bit-identical.

The focused command-feature and downstream scorer suite passes (92 tests).
Independent review found the cache implementation and both evidence controls
clean and safe to commit.

### Remaining acceleration space

Do not revive the prior approximate hazard rewrite: it changed 13/30,000
tie-breaks for only 6% gain. Candidate construction remains a possible later
lane, but the bounded cache is sufficient for the current request.

## Repository hygiene

The working tree still contains extensive unrelated changes from other active
lanes. The scorer commit must be allowlisted to the utility-clock source, its
two focused tests, and this checkpoint. Raw benchmark/replay artifacts remain
local evidence and must not be mixed with the frozen scientific artifacts.
