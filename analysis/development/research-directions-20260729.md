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
| Causal within-`exec` progress counters | **STOPPED / NO-GO** | Progress raises mean regret by 0.505 s; pair-cluster bootstrap CI [+0.257, +0.795] s. The default counter model's lower average error hides a harmful underprediction bias at the ranking boundary. |
| Synthetic-c32 CacheWise diagnostic | **STOPPED / NO-GO** | Across 32 fixed arrival permutations, C100 raises mean regret by 1.847 s; the schedule-bootstrap CI is [+1.087, +2.646] s. Synthetic overlap does not rescue argument clustering. |
| TB synthetic-c32 transfer | **PASSED GATE, SINGLE-TASK FRAGILE** | C100 lowers aggregate regret by 72.299 s, but excluding one post-hoc-identified task flips the effect to +26.722 s. Do not generalize or scale this result. |
| Exact-timeout task-stability gate | **STOPPED / NO-GO** | Aggregate regret improves by 17.271 s, but deleting `mixed-integer-programming` flips the effect to +4.012 s. Exact timeout is not task-stable. |
| Same-repository CacheWise diagnostic | **STOPPED / NO-GO** | Repo-C100 improves over repo-tool by 0.413 s on 199 eligible events, but the task-pair CI reaches +0.018 s and deleting `sqlfluff` flips the effect to +0.003 s. Hard repo-local history also raises regret from 0.665 s to 33.734 s versus pooled tool history. |
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

## Frozen causal progress-counter gate

Frozen 2026-07-31 before fitting or reading any counter-model result. This is a
new development-exposed question motivated by the CacheWise failure, not an
amendment that can reverse that result.

The mechanism question is whether live, workload-neutral cgroup counters reveal
repository-specific long-running work that static command similarity misses.
The existing timeline has complete `exec`-call presence in both corpora; 737 of
3,003 SWE-100 calls and 2,411 of 8,976 fresh-277 calls have at least two samples.
Each sample is an approximately 0.5 s interval delta, with a final partial
interval on exit, for CPU core-seconds, CPU opportunity/quota, and network
RX/TX. It is container-cgroup state during the `exec` interval, not
clause-exclusive attribution.

Timestamp amendment record: independent review rejected aligning recorder
offsets from the eventual action end because that cannot be reproduced at the
ranking timestamp. Before fresh-277 was opened, availability was changed to
tool-action start plus recorder offset plus a 10 ms pad, above SWE-100's 3.784
ms maximum total pre/post-recorder overhead. The first fresh load then failed
before model fitting or evaluation: only the overhead distribution was visible,
with 3 of 8,976 calls above 10 ms and a 25.846 ms maximum; no regret output was
produced. The pad was openly amended to 50 ms. The run fails closed if any
eligible single-`exec` action exceeds it. Action end is used only for this
whole-run validity audit, never as a feature or eligibility timestamp; any
resulting verdict is therefore development-exposed and requires independent
validation.

Protocol:

- Fit remains SWE-100; evaluation remains fresh-277. No new collection or
  confirmation corpus is accessed.
- Training rows come from every observed sampler endpoint in a gap containing
  exactly one `exec` call. Features and labels use no later sample: elapsed gap
  time, sample count, cumulative CPU, mean/recent CPU rate, CPU fraction of
  quota and current quota, cumulative network bytes, and recent network rates;
  network quantities are transformed with `log1p`. The label is observed
  remaining gap time at that sampler endpoint.
- The candidate is one default `HistGradientBoostingRegressor` with squared
  error and `random_state=0`, fitted to raw remaining seconds. There is no model,
  feature, threshold, or hyperparameter sweep.
- Evaluation retains all 2,410 observed-overlap ranking events and the exact
  tool-name conditional-remaining baseline. At a ranking timestamp, the
  candidate may use only the latest counter sample whose endpoint is no later
  than that timestamp; new/unsupported/non-single-`exec` gaps fall back to the
  baseline. Parallel calls are never double-counted as independent cgroup
  signals.
- The primary comparison is full-event
  `mean_regret(progress) - mean_regret(tool_name)`, with the same unordered
  task-pair cluster bootstrap. The gate passes only if the 95% interval is
  strictly below zero. Intervention coverage, top-1 agreement, and regret-tail
  quantiles are diagnostics.

A failure closes this existing-counter direction. A pass justifies an
independent validation decision; because fresh-277 is already exposed, it does
not by itself authorize a live systems claim. Time-resolved disk/RSS, process
phase, clause parsing, support thresholds, and argument text are outside this
gate and cannot rescue it after the result is visible.

Outcome: **NO-GO**. On all 2,410 ranking events, tool-name survival has 0.780 s
mean regret and the progress model has 1.284 s; the paired difference is
+0.505 s with a 95% task-pair cluster-bootstrap interval of [+0.257, +0.795]
s. Progress changes 71 choices: all move from the still-running gap selected by
the tool baseline to the newly arrived gap, and 63 are harmful versus 8
helpful. Harm totals 1,249.2 s while help totals 32.9 s. On the 2,212 candidate
rows with causal progress, mean absolute error improves from 57.43 to 55.24 s,
but mean signed error flips from +26.25 s to -16.83 s. The counters therefore
make the default estimator less pessimistic about active long work, which is
precisely wrong for this ordering decision despite slightly better pointwise
error. This closes the frozen CPU/network-counter model; it is not evidence
against uncollected time-resolved disk, RSS, or process-phase signals.

## Frozen synthetic-c32 CacheWise diagnostic

Frozen 2026-07-31 before reading any synthetic-concurrency model result. This
is a separate, development-only counterfactual and cannot reverse the negative
observed-overlap result above.

- Fit is unchanged SWE-100. Evaluation is unchanged fresh-277, whose outcomes
  are already development-exposed.
- The target is one operating point corresponding to CacheWise's 30--50
  session regime: nominal concurrency 32. SWE-100's median span from first to
  last tool gap is 357.303430 s, so the fixed open-loop interarrival is
  `357.303430 / 32 = 11.165732` s. No evaluation duration sets the arrival rate.
- Evaluation session IDs are sorted and independently shuffled with NumPy
  `default_rng(seed)` for the 32 predeclared seeds 0--31. In each schedule,
  session `i` arrives at `i * 11.165732` s. Within each session, all tool-gap
  offsets and durations are preserved relative to its first tool-gap start; no
  outcome or argument changes any schedule.
- The predictor, arguments, TF-IDF limit, C20/C50/C100 fits, conditional-
  remaining fallback, tie break, and regret are exactly those of the reviewed
  reproduction. The primary comparison remains full-event
  `mean_regret(C100) - mean_regret(tool_name)`.
- The primary estimate is the mean of the 32 per-schedule full-event
  `mean_regret(C100) - mean_regret(tool_name)` differences. Its 95% interval is
  a deterministic 10,000-draw bootstrap over these paired schedule-level
  differences. This quantifies arrival-permutation sensitivity conditional on
  the fixed traces; it is not task-population uncertainty. The gate passes only
  if the upper endpoint is below zero. C20/C50, top-1 agreement, realized
  concurrency, and tail regret are diagnostics and cannot select another
  operating point.

Review amendment before any model result: the first draft used one seed and a
trigger-task cluster bootstrap. Schedule-only inspection had exposed 12,665
seed-0 ranking events, mean/maximum live-session concurrency 23.7/36, and
maximum active-gap concurrency 16. Independent review correctly found that an
active task crosses trigger-task clusters, invalidating that interval. It was
replaced by the 32 schedule repetitions above before fitting or evaluating any
synthetic result.

Only one c32 arrival-rate operating point is run. A negative result stops this
synthetic direction; a positive result only says that argument clustering helps
victim ordering in expectation over this fixed family of counterfactual arrival
orders. The simulation holds isolated-trace durations fixed and has no resource
contention, KV-capacity measurement, eviction/recomputation feedback, eviction
count, or end-to-end JCT claim.

Outcome: **NO-GO**. The 32 schedules each contain 12,439--12,722 ranking
events. Mean live-session concurrency is 22.2, schedule maxima are 36--42, and
the largest simultaneous active-gap candidate set is 19. Tool-name survival
has 57.961 s mean regret and C100 has 59.808 s; the mean paired schedule-level
difference is +1.847 s with a 95% schedule-bootstrap interval of
[+1.087, +2.646] s. C100 is harmful in 24 schedules and helpful in 8, with
per-schedule differences spanning -2.789 to +7.459 s; its top-1 agreement also
drops from 44.60% to 43.67%. C20 and C50 are worse by 2.853 s and 5.166 s,
respectively. The failure therefore survives a large increase in ranking-event
coverage and many fixed arrival orders; it is not explained by the natural SWE
traces having only pairwise observed overlap. This closes the synthetic-c32
direction without changing the narrower claim boundary: no real contention,
KV pressure, eviction feedback, or JCT was measured.

## Frozen TB synthetic-c32 transfer

Frozen 2026-07-31 before fitting or reading any TB synthetic result. This is a
cross-corpus transfer diagnostic requested after the SWE synthetic NO-GO, not a
post-hoc replacement for that result.

- The authoritative source is `traces/terminal-bench/tb-all/canonical` plus the
  existing GLM-100 `task_ids.txt` selection manifest. Its 100 task IDs form
  evaluation; the disjoint 139 canonical task IDs form fit. Tasks without any
  extractable LLM-to-LLM tool gap contribute no row: 136 fit tasks provide
  1,888 gaps and 99 evaluation tasks provide 1,227 gaps.
- The unchanged synthetic protocol uses nominal c32, seeds 0--31, sorted task
  IDs shuffled with `default_rng(seed)`, and all within-session gap offsets and
  durations preserved. TB-139's median first-to-last-gap span is 115.715042 s,
  fixing the open-loop interarrival at `115.715042 / 32 = 3.616095` s without
  using a TB-100 duration.
- The unchanged CacheWise arms, TF-IDF/KMeans fits, conditional-remaining
  fallback, tie break, regret, 32 paired schedule-level differences, 10,000-
  draw schedule bootstrap, and strict upper-CI-below-zero gate remain in force.
  C100 versus tool-name is primary; no result from SWE or TB selects a new C,
  seed, arrival rate, feature, or fallback.

A negative result stops this transfer. A positive result only establishes
heterogeneity under the fixed TB counterfactual and requires independent/live
validation. The same limits remain: isolated durations are fixed, and no real
contention, KV pressure, eviction/recomputation feedback, eviction count, or
JCT is simulated.

Outcome: **the frozen aggregate gate passes, but the result is not robust across
tasks**. Across 32 schedules, tool-name survival has 194.633 s mean regret and
C100 has 122.334 s. The paired difference is -72.299 s with a 95% schedule-
bootstrap interval of [-90.844, -53.768] s; 29 schedules are helpful and 3 are
harmful. Each schedule contains 1,126--1,207 ranking events, mean live-session
concurrency is 9.0, maxima are 42--54, and the largest active-gap candidate set
is 23. C100 raises top-1 oracle agreement from 28.23% to 46.68%. C50's lower
113.224 s regret is diagnostic only and cannot replace the frozen C100 primary.

Post-hoc mechanism diagnosis shows that the net improvement is dominated by
`hdfs-deployment`, which contains two approximately 600 s download `exec` gaps.
C100 repeatedly recognizes these as long survivors from command text and the
explicit 600 s timeout while the tool-name baseline pools them with mostly
short `exec` calls. Filtering this task from each already-frozen schedule while
retaining every other task's arrival time reverses the result: C100 is worse by
26.722 s, with a 95% schedule-bootstrap interval of [+17.355, +37.789] s, and is
harmful in 28 of 32 schedules. This sensitivity was selected after seeing the
aggregate result, so it is a diagnostic rather than a second confirmatory gate.
It prevents a corpus-wide or live-system claim: TB demonstrates a real
inference-time long-command signal, not yet a robust CacheWise policy.

## Frozen exact-timeout task-stability gate

Frozen 2026-07-31 after the TB C100 aggregate result and the post-hoc
`hdfs-deployment` deletion were visible, but before fitting or reading any
timeout-only result. This is a development-only mechanism test, not an
independent confirmation.

- Fit and evaluation remain the manifest-fixed TB-139/TB-100 split. The 32
  frozen synthetic schedules, arrival rate, conditional-remaining estimator,
  fallback, tie break, regret, and schedule bootstrap are unchanged.
- The only new arm conditions history on `(tool batch, exact timeout tuple)`
  when at least one raw tool-argument dictionary contains a positive finite
  numeric `timeout`. Missing or unusable timeout values fall back to tool-name,
  and exhausted timeout histories use the same tool-name/global fallback. No
  command text, TF-IDF/KMeans label, timeout bucket, threshold, or fitted model
  is used.
- The aggregate primary remains the mean of 32 paired schedule-level
  `mean_regret(timeout) - mean_regret(tool_name)` differences, with the frozen
  10,000-draw bootstrap. The aggregate gate requires its 95% upper endpoint to
  be below zero.
- Task stability is an exhaustive delete-one-task check over every TB-100 task
  with an extracted gap. Each task is removed from each already-frozen schedule
  without shifting any other arrival. For every deletion, compute the mean of
  the 32 paired schedule-level differences. The stability gate requires the
  worst of these task-deletion means to remain strictly below zero. Both the
  aggregate and stability gates must pass.

A failure stops this timeout-only direction; command-derived features cannot
rescue it in this gate. A pass would show that a simple inference-time deadline
signal is not dependent on one evaluation task, but would still require an
independent corpus and real contention/KV evaluation before a systems claim.

Outcome: **NO-GO on the joint task-stability gate**. Across the 32 frozen
schedules, tool-name survival has 194.633 s mean regret and exact-timeout
conditioning has 177.362 s. The paired difference is -17.271 s with a 95%
schedule-bootstrap interval of [-29.739, -4.296] s, so the aggregate sub-gate
passes; 20 schedules are helpful and 12 are harmful. Mean top-1 oracle
agreement rises from 28.23% to 31.07%.

The exhaustive task-deletion gate fails. `mixed-integer-programming` is the
only one of 99 deletions that reverses the mean sign: without it, timeout is
worse by 4.012 s and only 13 of 32 schedules are helpful. That task contains a
300.011 s solver call with an exact 300 s timeout. Post-hoc choice attribution
shows that the 600 s signature supplies the largest aggregate benefit, while
the 300 s signature is net harmful unless this full-timeout solver is present.
The timeout field is therefore a useful coarse ceiling but not a stable proxy
for remaining work. Per the frozen stop rule, do not add command-text features
to rescue this exposed gate.

## Frozen same-repository CacheWise diagnostic

Frozen 2026-08-03 after all CacheWise, progress, synthetic-concurrency, TB, and
timeout results above were visible, but before fitting or reading any
repository-local model result. This is a development-only mechanism test of the
hypothesis that the paper's random session split benefits from workload-family
stability. It cannot reverse the original SWE NO-GO.

The fixed SWE-100 fit corpus contains 100 tasks from 94 repositories; fresh-277
contains 277 tasks from 213 repositories. Thirty-six repositories overlap,
covering 40 fit tasks and 73 evaluation tasks. Of those 36 repositories, 32
have only one fit task and four have two. No task ID appears in both corpora.

Protocol:

- The repository key strips only the trailing `-<issue number>` from a SWE task
  ID. A ranking event is eligible only when every candidate repository has at
  least one fit task. This selection uses task metadata, not durations or model
  outcomes.
- The pooled tool-name and pooled C100 arms are unchanged controls on the same
  eligible rows. The repo-tool and repo-C100 arms use only other fit tasks from
  that candidate's repository. An exhausted repo cluster falls back to the
  repo tool-name distribution, then the repo-global distribution; it never
  borrows a pooled duration.
- The sole primary comparison is
  `mean_regret(repo-C100) - mean_regret(repo-tool)`. Its gate requires the upper
  endpoint of the unchanged 10,000-draw unordered-task-pair cluster bootstrap
  to be below zero.
- Stability deletes every event involving each evaluation repository in turn.
  The largest remaining mean primary delta must stay below zero. Both gates
  must pass before targeted same-repository collection is justified.

Independent bounded review found no critical or major issue. Before the formal
run, its two minors were fixed: deletion reporting now covers all evaluation
repositories, including those absent from eligible events, and the self-check
directly verifies that repo-C100 reads only its repository's durations.

Outcome: **NO-GO**. Only 199 natural ranking events across 15 task pairs are
eligible; seven events from one task pair place two sessions from the same
repository in direct competition. On the eligible rows, pooled tool-name has
0.665 s mean regret and pooled C100 has 1.254 s. Hard repo-local histories are
far worse: repo-tool has 33.734 s mean regret and 44.22% top-1 agreement, versus
92.46% for pooled tool-name. This is direct evidence that one or two prior tasks
per repository do not preserve enough remaining-time tail support.

Repo-C100 reduces repo-tool regret from 33.734 s to 33.321 s. The primary delta
is -0.413 s with a 95% task-pair cluster-bootstrap interval of
[-1.689, +0.018] s, so the primary gate fails. It changes 17 choices: nine help
by 98.296 s total and eight harm by 16.090 s total. The favorable magnitude is
not repository-stable. Removing the 30 eligible events involving `sqlfluff`
leaves a +0.003 s delta; those events contribute -82.715 s, slightly more than
the full -82.206 s net benefit. The stability gate therefore also fails.

The evidence does not establish that a high-support same-repository deployment
would fail: that regime was not present, because 32 of 36 overlapping
repositories contribute only one fit task. It does establish that repository
identity alone does not make the existing CacheWise estimator deployable, and
that the small within-repository C100 advantage is again concentrated in a
single repository. Per the frozen stop rule, do not start a targeted collection
to rescue this result. A future study of history depth or hierarchical pooling
would be a new, openly post-result hypothesis and would require a separately
frozen method plus independent tasks; it cannot be tuned on these 199 events.

## Frozen SQLGlot factorial KV mechanism diagnostic

Frozen 2026-08-04 after the SQLGlot48 natural-c2 and synthetic-c32 C100 results
were visible, but before any KV-capacity simulation result on SQLGlot100 was
computed. This is an openly post-result, development-only mechanism diagnostic.
It asks how much of CacheWise's paper result remains when prefix-aware scheduling
and C100 eviction act together on our same-repository traces. It cannot reproduce
live vLLM throughput or session completion time.

The paper compares two independent mechanisms: prefix-aware scheduling chooses
the queued request needing the fewest additional resident blocks, while C100
replaces LRU with whole-tool-argument TF-IDF/KMeans conditional-remaining-time
eviction. At load 40, the paper reports a 1.85--2.66x session-time improvement
from prefix scheduling; predictive eviction adds about 1.7--2x over prefix-aware
baselines; the complete system reduces evicted blocks by 2--2.6x. Those paper
numbers and the positive SQLGlot48 C100 result are visible motivation, not gates
selected from the new simulation.

Protocol:

- Reuse the exact 24 fit tasks in
  `configs/corpora/swe-sqlglot-48-gpt56-ebpf.json`; no evaluation task updates
  C100. Evaluation candidates are the other 76 successful canonical traces in
  the packaged SQLGlot100 `results.jsonl`.
- For seeds 0--31, permute sorted evaluation task IDs with NumPy
  `default_rng(seed)`, select the first 40, and launch their first requests at
  time zero in that order. Preserve each selected trace's LLM service durations,
  prompt/completion token counts, and post-LLM tool-gap durations. A session's
  next request becomes eligible only after its recorded gap completes.
- Model 16-token KV blocks and a fixed capacity of 800,000 tokens (50,000
  blocks). This is a paper-scale approximation, not a measured vLLM block count:
  Qwen2.5-Coder-32B BF16 has 64 layers, 8 KV heads, and head dimension 128, or
  262,144 KV bytes/token; 800,000 tokens plus roughly 64 GB of weights consume
  about 274 GB of the paper's combined 282 GB HBM, leaving about 8 GB for other
  runtime state. Do not tune capacity after reading the result.
- Reusable prefix is the complete-block overlap between the preceding sequence
  and current prompt. Capacity eviction removes suffix blocks from one inactive
  session at a time. A completed session releases its blocks. Recorded LLM
  service time is fixed across arms; cache misses do not feed back into service
  duration, so simulated latency is descriptive only.
- Evaluate exactly four arms on identical sessions: `fcfs_lru`, `prefix_lru`,
  `fcfs_c100`, and `prefix_c100`. FCFS uses initial/causal arrival order. Prefix
  scheduling selects the queued request requiring the fewest additional blocks.
  LRU evicts the least-recently accessed resident session. C100 reuses the
  existing 5,000-term TF-IDF, KMeans `C=100`, conditional-survivor mean, and
  tool/global fallback without retuning.
- Primary metrics are evicted blocks and eviction-induced reusable-prefix miss
  blocks. Report the mean paired effects of prefix alone (`prefix_lru -
  fcfs_lru`), C100 alone (`fcfs_c100 - fcfs_lru`), incremental C100 under prefix
  scheduling (`prefix_c100 - prefix_lru`), and the full baseline-to-combined
  eviction ratio. Use a 10,000-draw paired seed bootstrap only to show
  conditional schedule/cohort uncertainty.
- Classify the result as paper-sized only if both mechanisms reduce mean evicted
  blocks in their matching comparisons and `fcfs_lru / prefix_c100 >= 2.0`, the
  paper's lower reported eviction reduction. A smaller effect is a measured gap,
  not permission to change capacity, load, cluster count, split, or simulator.

Any positive result justifies at most one separately approved live-GPU smoke.
The simulator omits continuous batching, chunked prefill, transfer contention,
cache-miss feedback, and exact vLLM block allocation. Therefore its latency
output cannot be compared numerically with the paper's 2.7--3.5x end-to-end
claim; the defensible comparison is block eviction under the frozen model.

## Corrections retained from the prior loop

The earlier investigation self-corrected a transcription error (`1947` for
`1089`), a roughly 5x envelope overestimate from summing concurrent pipeline
clauses, an inaccurate verification line, negative reasoning-text/runtime-
activity hypotheses, in-sample per-key overfit, the refuted restore-overcharge
hypothesis, and the 62--89-request cost-scale artifact. Those corrections remain
load-bearing context for the closed single-call KV lane.
