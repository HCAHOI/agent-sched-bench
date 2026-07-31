# Research directions — current state

Rewritten 2026-07-31 after the CacheWise/TraceLab full-paper comparison. This
states what is true now; prior survey framing is in git history. Detailed paper
boundaries and the frozen reproduction protocol are in
`analysis/offline/related-work.md`.

## Established negative result: single-call KV swap timing

The previous loop closed the per-call duration-prediction lane for the measured
swap-back-expensive regime. The result is internally consistent:

| Policy class | Best share of the oracle budget |
|---|---|
| Constant trigger, fitted out-of-fold | negative; beats deadline in 1 of 20 cells |
| Constant trigger, in-sample oracle | at most 7.2% |
| Per-key empirical, in-sample | 29--75%; all overfit |
| Per-key empirical, leave-one-out | negative |
| Per-call oracle | 100% by construction |

The usable budget exists only for `kv < L < 2*kv`. `T=kv` is near-optimal among
constants because that is where the restore penalty vanishes. On the measured
model/device, reload beats recompute by 38.4x and the effective restore charge
is `rho=0.94`; the large 3,500/5,000 ms cells were 62--89 requests' worth of KV,
not one request. These facts reject another trigger estimator under the same
cost model; they do not reject ordering multiple paused sessions under real
memory pressure.

## What the expanded related work changes

The occupied directions are now clearer:

- Autellix, ThunderAgent, SAGA, and Murakkab cover program/workflow-aware LLM
  scheduling.
- Continuum, KVFlow, CacheWise, PEEK, and the policy-runtime case study cover
  tool-gap KV lifetime, workflow-aware placement, paused-session eviction, or
  prefix-aware queueing.
- Parrot, Pie, and AgentCgroup cover declared semantics, programmable serving,
  or declared call-level resources.
- SpecBox, PASTE, Seer, Crab, and DeltaBox cover environment/tool speculation,
  prefix prefetch, and sandbox state.

The remaining measurement advantage is narrower: our traces preserve raw tool
arguments and can observe shell clauses, `execve` children, and counters below
the outer call. TraceLab's public release cannot test that feature because it
removes raw arguments. CacheWise does test arguments, but only as one serialized
outer payload with TF-IDF/KMeans.

## Current directions

| Direction | Status | Next decision |
|---|---|---|
| CacheWise outer-argument reproduction on SWE | **STOPPED / NO-GO** | C100 raises mean regret by 0.587 s versus tool-name; pair-cluster bootstrap CI [+0.120, +1.188] s. It does not reproduce the paper's granularity gain. |
| Clause-aware extension of CacheWise | **NOT AUTHORIZED by this gate** | The predeclared prerequisite failed. Do not add clause features or a post-hoc support threshold to rescue this comparison. |
| Clause-granular resource control | **Near-dead under the old consumer** | Corrected payoff was 10.6% on 12.3% of calls; `memory.max` is a cap, not a reservation, and lowering it below usage OOM-kills. Reopen only with a different measured consumer. |
| Pipeline-dominated shells | **Open measurement fact, no consumer** | 87.7% of multi-clause exec calls are pipeline-dominated. Do not build a predictor until a decision depends on it. |
| Heterogeneous multi-tenant composition | **Blocked offline; live run not authorized** | Existing SWE traces cannot establish the 30--50-session pressure regime, and the required predictor gate failed. |

## Stop/go boundary for the current run

The CacheWise reproduction is not a new canonical tool-resource predictor and
does not change the locked latency/resource objective. It reuses canonical
outer `tool_exec` records and makes no new collection.

Go to one live serving experiment only if the predeclared 95% unordered-task-pair
cluster bootstrap interval for `mean_regret(C100) - mean_regret(tool_name)` is
strictly below zero. Otherwise stop: do not tune tokenization, choose `C` after
the fact, substitute synthetic concurrency, or descend to shell clauses to
rescue the result. `C=20/50`, top-1 agreement, and call-duration fit explain the
mechanism but cannot override the primary gate.

Even a pass supports only "outer arguments improve pairwise reuse ordering on
development-exposed SWE traces." It does not support CacheWise's end-to-end JCT
claim or confirmation, because fresh-277 reaches only concurrency two and some
repositories overlap the fit corpus.

The gate did not pass. Tool-name mean regret is 0.780 s; C20/C50/C100 are
0.857/1.179/1.366 s. C100's top-1 rate is almost unchanged (94.07% versus
93.98%) because it makes nearly balanced numbers of helpful and harmful choice
changes, but its rare harmful changes are much larger. Post-hoc inspection
locates the tail in repository-specific test commands whose fitted argument
clusters end far earlier than the held-out calls. This is evidence against
whole-argument TF-IDF/KMeans as a safe remaining-time tail estimator here, not
authorization to tune on fresh-277.

## Corrections retained from the prior loop

The earlier investigation self-corrected a transcription error (`1947` for
`1089`), a roughly 5x envelope overestimate from summing concurrent pipeline
clauses, an inaccurate verification line, negative reasoning-text/runtime-
activity hypotheses, in-sample per-key overfit, the refuted restore-overcharge
hypothesis, and the 62--89-request cost-scale artifact. Those corrections remain
load-bearing context for the closed single-call KV lane.
