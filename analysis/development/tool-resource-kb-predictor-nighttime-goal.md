# Nighttime Goal — Three-Bucket Tool-Resource KB and Predictor

**Date:** 2026-07-28

**Time budget:** the original eight-hour window, followed by a human-authorized
two-hour recovery window ending before the 2026-07-29 meeting

**Mode:** unattended development exploration; no confirmation claim

**Repository:** `/home/chiyu/workspace/agent-sched-bench`

**Branch:** `exp/tool-resource-3bucket-swe`

**Base commit:** `776d2857b6325992262da59abce88b256404fed9`

This file is the execution contract for the nighttime goal. Re-read it at the
start of every continuation and after context compaction. Human instructions
issued after this file override it. Otherwise, do not relax its data,
causality, metric, review, or completion gates after seeing results.

## 1. Goal

Replace the current nine-bin latency objective with one three-bin clause
latency PMF:

```text
short:   [0, 2000] ms
middle:  (2000, 8000] ms
long:    (8000, +inf) ms
```

The predictor returns:

```text
(P(short), P(middle), P(long))
```

The hard predicted class is the highest-probability bucket. Ties select the
shorter bucket deterministically.

Explore whether a minimum structured argv representation plus support-aware
public/repository evidence combination can outperform both:

1. the current canonical predictor under the same three-bin labels; and
2. the same-evaluation majority-class baseline.

Retain the existing independent resource targets and thresholds:

```text
peak_cpu_cores > 2.0
sampled_peak_rss_mb > 500 decimal MB
disk_read_write_bytes_total > 104857600 bytes
```

The resource short-null policy remains unchanged: only an explicitly marked
null observation with `latency_ms < 500` is imputed Light.

This is best-effort development work. A reviewed NO-GO with a concrete
mechanism diagnosis is a complete outcome. Repeatedly changing candidates
until one wins is not.

The data scope is SWE-ReBench only. Do not evaluate, inspect, collect, or use
another benchmark during this goal.

### 2026-07-29 recovery amendment

Visible before this amendment: all baseline, Candidate R, and Candidate S
outer development results. Candidate S improved latency and Disk in both
orientations but regressed CPU and RSS; its latency-selected alpha was 16.
Candidate C lacked an authorized context with 90% coverage. The prior agent
incorrectly treated those facts as a reason to end after about one hour,
despite the human's intent to continue mechanism exploration.

Use the remaining two-hour recovery window for one bounded, fit-only
falsification before proposing another architecture:

1. Use only the fixed alpha grid `1, 4, 16, 64`.
2. For SWE100 and SWE277 separately, assign repositories to the existing five
   deterministic repository-grouped folds. Preserve manifest task order and
   trace-close causal updates inside validation.
3. Produce pooled out-of-fold predictions for current raw hard-backoff and
   structured posterior shrinkage at every alpha.
4. On the identical out-of-fold rows, report exact latency accuracy and each
   resource accuracy against current and the same-fold majority or
   majority-Light baseline.
5. A shared alpha is feasible only if it is strictly better than both current
   and majority for latency, CPU, RSS, and Disk in both cohorts. If multiple
   alphas are feasible, select the one with the largest minimum percentage-
   point margin across those eight gates; ties select the larger alpha.
6. Do not read or rerun another outer result to choose the alpha. These folds
   are development-exposed diagnostics, not fresh validation.

If no alpha is feasible, do not scan more weights. Diagnose whether the
failure is driven by false positives, false negatives, representation, or the
shared-arbitration constraint, and produce a meeting-ready decision artifact.
If one is feasible, freeze the selection rule, add the minimum reusable
implementation, obtain bounded independent review, and only then run a
development-only comparison. The recovery window is a hard deadline: prefer
one complete, committed falsification over unfinished breadth.

The shared-alpha diagnostic completed after this amendment and before the
following human instruction. No alpha was feasible: latency and Disk passed
for all four values, CPU required alpha 4 across both cohorts, and RSS gained
no true positives while adding false positives for every alpha.

### 2026-07-29 multi-head amendment

Visible before this amendment: the full shared-alpha fit-fold result described
above and all earlier outer results. The human explicitly authorized different
predictors for different targets, analogous to task-specific heads on a shared
model. The earlier same-alpha and no-second-head rules were experimental scope
controls, not architectural truths, and are superseded here.

Keep one causal KB/service boundary and share canonicalization, evidence
eligibility, timestamps, snapshots, and online/offline entry points. Permit
target-specific heads above that shared stem. The minimum evidence-supported
hybrid is:

- latency: structured posterior head, alpha 16;
- CPU: structured posterior head, alpha 4;
- Disk: structured posterior head, alpha 16 (all fixed-grid values passed, so
  reuse the reviewed latency value rather than introduce another choice);
- RSS: retain current raw hard-backoff until a separate RSS head passes.

Before runtime integration, run exactly one fit-only RSS-head falsification:

1. Use the existing `generic-argv-v3-role` privacy-safe semantic tokens and
   fit-only stable-subcommand vocabulary.
2. Hash token counts deterministically into 256 signed dimensions with SHA-256;
   do not expose raw paths, IDs, or opaque values.
3. Train one RSS-only MLP per repository fold: `256 -> 32 ReLU -> 1`, seed 0,
   Adam learning rate `1e-3`, weight decay `1e-4`, 50 full-batch epochs, and
   `BCEWithLogitsLoss(pos_weight=sqrt(Light/Heavy))`.
4. Use a fixed 0.5 Heavy threshold. Do not sweep architecture, seed, loss
   weight, epoch count, or threshold after reading results.
5. Public training repositories exclude validation repositories. Report pooled
   out-of-fold TP/TN/FP/FN and exact accuracy on the same eligible RSS labels.
6. GO only if accuracy is strictly greater than both current and
   majority-Light in each cohort, with at least one true positive and no
   unavailable prediction.

This is a development-only architecture probe, not runtime integration or a
confirmation claim. A failure ends MLP tuning for this recovery window.

The fixed argv-only MLP then returned all-Light predictions in both cohorts.
An analysis-only threshold oracle also preferred all Light, so this is
insufficient ranking signal rather than a missed 0.5 calibration threshold.
A mechanism audit, visible before the following amendment, found repeatable
Python test/install argv associations but low positive precision; actual
latency separated Heavy from Light strongly but is postexecution-only.

Run exactly one final fit-only feature falsification, without changing the
network or training recipe:

1. Keep the same five outer repository folds, RSS labels, 256 hashed argv
   dimensions, `32`-unit ReLU head, seed, optimizer, loss, epochs, and fixed
   threshold.
2. Add the three probabilities from the alpha-16 structured latency head,
   producing 259 input dimensions. These are predictions available before
   execution; actual latency, CPU, Disk, and RSS remain forbidden features.
3. For each outer validation fold, fit its latency KB only on outer-training
   repositories. Generate MLP-training latency features with four nested
   repository folds inside the outer-training set: each row's PMF comes from a
   KB whose public corpus excludes both its inner group and the outer
   validation group.
4. Within each predicted repository, preserve task order and predict every
   clause in a task before settling any clause from that task. Pool outer
   out-of-fold TP/TN/FP/FN; never average fold accuracies.
5. Use one KB per public/target split because local state is already keyed by
   repository, and run at most four outer folds concurrently with one Torch
   thread per worker.
6. GO only if accuracy is strictly greater than current and majority-Light in
   each cohort, with at least one true positive and no unavailable prediction.
   A failure ends RSS-head experimentation in this recovery window.

The augmented probe was also NO-GO: it recovered no Heavy row in either
cohort and added four false positives in SWE277. The alpha-16 latency PMF had
moderate Heavy/Light rank signal, but its argmax remained Short for 34/42
SWE100 Heavy rows and 109/121 SWE277 Heavy rows. At the roughly one-percent
Heavy base rate, the fixed accuracy objective requires new positive
predictions to exceed 50% precision; the repeatable argv families reach only
about 2-5%.

The recovery decision is therefore architectural, not a runtime integration:
keep one causal evidence/service stem with target-specific heads, use the
already development-GO alpha-16 structured posterior as the latency candidate,
use structured hard-backoff Candidate R as the Disk candidate because the
existing outer results are at least as good as alpha-16 in both orientations,
and retain current raw hard-backoff for CPU and RSS. Alpha 4 remains a
fit-only CPU research candidate, not deployment evidence. No additional RSS
MLP, loss, seed, or threshold tuning is authorized from these exposed data.

### 2026-07-29 contrastive clause-retrieval amendment

Visible before this amendment: both RSS MLP failures, the Heavy/Light mechanism
audit, and a historical 2026-07-23 BERT summary reporting no robust marginal
for the older tool-level whole-container memory target. That historical target,
telemetry, and unit differ from the current clause-level sampled-RSS objective,
so the result is a caution rather than a substitute for this test.

The human explicitly authorized a pretrained embedding model plus contrastive
adaptation for canonicalized clauses. This supersedes invariant 10 only for
one development-only, offline RSS retrieval falsification. Do not add a runtime
dependency, ANN index, model download, or service integration. Use the locally
cached `distilbert-base-uncased` weights through an ephemeral
`transformers==4.57.6` environment; do not modify `pyproject.toml` or
`uv.lock`.

Use one joint deterministic five-fold repository partition over the union of
the two locked SWE inputs; a repository must remain in one fold even if it
appears in both inputs. Report each cohort separately and pooled. For every
fold:

1. Render each clause as the training-vocabulary `generic-argv-v3-role` token
   stream with explicit role markers. Validation labels and repositories never
   affect the stable-subcommand vocabulary.
2. B0 is frozen DistilBERT last-hidden-state mean pooling under the attention
   mask, L2 normalized, maximum 128 wordpieces. Report truncation coverage.
3. Construct exactly one training triplet per RSS-Heavy anchor. The positive
   is the closest B0 Heavy example from another repository, preferring the same
   bin when available. The hard negative is the closest B0 Light example from
   another repository, also preferring the same bin. Mining uses outer-training
   labels only.
4. B1 adds a 768-to-128 linear projection and unfreezes only DistilBERT's final
   transformer block. Train the fixed triplets for five epochs, batch size 32,
   seed 0, cosine triplet margin 0.2, AdamW with encoder learning rate `2e-5`,
   projection learning rate `1e-3`, and weight decay `1e-4`. There is no
   checkpoint, epoch, margin, mining, or learning-rate selection.
5. Retrieve the five nearest outer-training clauses by cosine similarity.
   Predict Heavy only when at least three neighbors are Heavy. The primary arm
   preserves current raw hard-backoff positives and adds retrieval positives;
   also report standalone retrieval for mechanism diagnosis.
6. B1 is GO only if the primary hybrid strictly beats current and all-Light
   accuracy in each cohort, has at least one additional true positive, and
   adds fewer false positives than true positives. B0 is a frozen baseline,
   not an alternative selection opportunity.

This test asks whether task-adapted embedding geometry creates a small
high-purity Heavy neighborhood. Failure ends embedding and contrastive tuning
on these exposed corpora. Success permits a reviewed reusable implementation;
it is not confirmation evidence.

The completed full-population result was NO-GO. B1 retrieval pooled one true
positive and ten false positives; preserving current raw positives produced
one true positive and eleven false positives. SWE100 added one true positive
and three false positives, while SWE277 added no true positives and seven
false positives. B1 accuracy was below current and majority-Light in both
cohorts. Frozen B0 was also NO-GO and had fewer false positives than B1, so
contrastive adaptation expanded the wrong neighborhoods. Do not tune epochs,
margin, mining, `k`, or vote threshold on these exposed rows.

### 2026-07-29 landmark and speculative-execution amendment

Visible before this amendment: the Heavy/Light mechanism audit, both RSS MLP
failures, the in-progress full-population contrastive retrieval folds, and the
following latency and conditional-label diagnostics. No landmark-conditioned
model result existed when the protocol below was frozen.

The prediction unit remains one eligible clause. Across the two locked SWE
inputs there are 18,224 such clauses. The observed strict latency CDF is:

| threshold | clauses below threshold | fraction |
|---|---:|---:|
| 50 ms | 10,035 | 55.06% |
| 100 ms | 10,468 | 57.44% |
| 250 ms | 11,309 | 62.06% |
| 500 ms | 12,713 | 69.76% |

Actions take nonzero time, so resource prediction should target the population
for which an action can still matter. Introduce a deployment landmark `h`,
initially considering 100 and 500 ms. The clause may be treated speculatively
as if it will survive to `h`: compute a prediction during the window, but apply
it only if the same clause is still running at `h`. A clause that ends earlier
completes normally and its speculative result is discarded. Survival at the
actual landmark is a causal application gate, not an inference feature known
at clause start and not hindsight.

The empirical landmark populations and canonical resource labels are:

| population | clauses | CPU Heavy | RSS Heavy | Disk Heavy |
|---|---:|---:|---:|---:|
| `T > 100 ms` | 7,756 (42.56%) | 678/5,008 (13.54%) | 163/5,980 (2.73%) | 427/7,704 (5.54%) |
| `T > 500 ms` | 5,511 (30.24%) | 678/2,763 (24.54%) | 163/3,735 (4.36%) | 421/5,459 (7.71%) |

At 500 ms the canonical short-null-imputed Light rows are gone. Missing
post-landmark resource labels remain unavailable: 2,748 CPU, 1,776 RSS, and 52
Disk rows in the joint corpus. Do not convert them to Light.

An action is useful only when the clause remains alive after the action
finishes. For fixed action latency `L_action`, the observed opportunity counts
are:

| landmark | `L_action=100 ms` | `L_action=250 ms` | `L_action=500 ms` |
|---|---:|---:|---:|
| 100 ms | 7,180 (92.57% of survivors) | 6,190 (79.81%) | 5,171 (66.67%) |
| 500 ms | 5,171 (93.83% of survivors) | 4,736 (85.94%) | 4,183 (75.90%) |

Waiting to 500 ms therefore does lose some opportunities relative to a 100 ms
decision; it cannot be described as lossless. It also reduces the expensive
resource-head population, removes short-null label mass, and aligns with the
first nominal 500 ms eBPF CPU/RSS sample.

Use this speculative state machine:

```text
t = 0:
    bind clause_id, command features, and the intended KB snapshot
    schedule the landmark timer
    launch the conditional predictor asynchronously

t = h:
    if the clause ended: discard any result
    if alive and the result is valid and ready: apply the action
    if alive but the result is late: apply it when ready only if the clause
        is still alive, or use the declared no-op/current fallback at h
    if alive but the result is invalid or unavailable: no-op/current fallback
```

The human corrected the earlier delayed-launch proposal after it was written:
the predictor starts at clause start for every clause. Prediction latency
therefore determines whether a result is ready at `h`, not when prediction
starts. If the result is ready before `h`, cache it; if the clause ends before
`h`, discard it. If the result becomes ready after `h`, the earliest possible
application time is that completion time and the clause must still be alive.
Equivalently, application is gated at `max(h, prediction_completion_time)`;
`min(h, prediction_completion_time)` is only the first checkpoint. Charge all
launched work, including predictions discarded for the 57.44% and 69.76% of
clauses that end by the 100 and 500 ms landmarks. Measure end-to-end latency
only for a predictor that remains a candidate, to quantify deadline coverage;
do not benchmark a rejected representation merely to choose a launch time.

The static conditional head may be computed later but must use the frozen
start-time command features and declared KB snapshot. A separate dynamic
refinement may use telemetry with `sample_ts <= decision_ts`; never silently
mix post-start telemetry into the static head. Prediction compute must be
resource-isolated from the workload so it cannot change the clause latency,
CPU, or RSS being predicted.

For action utility, the landmark target should describe the future after `h`,
not a peak that already happened:

```text
P(
    remaining_duration,
    future_cpu_peak,
    future_rss_peak,
    future_disk_delta
  | clause_alive_at_h, evidence_available_by_h
)
```

The practical staged architecture is:

1. `t=0`: cheap latency/survival head for early planning;
2. near 100 ms: optional lightweight preparation or command
   embedding/retrieval if its measured p99 fits the window;
3. near 500 ms: CPU/RSS/Disk decision using survival and, only when valid,
   the first eBPF telemetry.

Train and score each landmark head only on clauses alive at that landmark, but
charge every launched and discarded prediction to system cost. Preserve whole
repository folds, task settlement barriers, identical eligible rows across
arms, and cohort-separated reporting. The primary deployment gate must be
action utility at the operating point, including predictor cost, action
latency, useful remaining-runtime coverage, and harmful false actions; higher
conditional classification accuracy alone is insufficient.

Before changing predictor structure, run one fixed-predictor landmark
falsification on the two development-exposed SWE cohorts:

1. Evaluate exactly `h=100 ms` and `h=500 ms`, with strict survival
   `latency_ms > h`. Use five deterministic repository-grouped folds separately
   in each cohort, manifest task order, and task-close causal updates.
2. Fix both arms to the current raw exact/prefix hard-backoff KB. The
   unconditional arm fits public evidence and admits later local evidence from
   all eligible clauses. The conditional arm differs only by fitting and
   admitting evidence from clauses with `latency_ms > h`.
3. At replay inference, invoke both arms for every eligible clause at clause
   start, before any observation from the task settles. Score and apply only
   on the identical survivor rows. Report all launched, discarded, survivor,
   and resource-label-unavailable counts.
4. Keep the canonical total-latency `2000/8000 ms` buckets and whole-clause
   CPU/RSS/Disk labels for this controlled first probe. These aggregate
   resource labels can test population conditioning but are not future-after-h
   targets; the available artifacts do not contain the time-resolved resource
   values needed to claim future action utility.
5. For latency report exact accuracy, class counts, confusion, and the
   same-survivor majority. For every resource report label-source counts,
   TP/TN/FP/FN, exact accuracy, and same-survivor majority-Light. Reconcile
   identical survivor row identities and labels across arms.
6. A target/landmark is a development GO only if the conditional arm is
   strictly more accurate than both the unconditional arm and the
   same-survivor majority in both cohorts. Otherwise it is NO-GO for that
   target. Do not tune the landmark, KB representation, thresholds, or
   resource labels after reading these results.
7. Also report the already fixed useful-opportunity counts for action
   latencies 100, 250, and 500 ms. Do not select one global landmark or claim
   scheduler utility without a concrete action cost/benefit model and
   time-resolved post-landmark targets.

The completed full-population contrastive experiment remains an unchanged
NO-GO representation baseline. Do not reinterpret it as landmark evidence;
landmark conditioning changes the decision population and must be frozen and
evaluated as a separate mechanism.

The fixed-predictor landmark falsification subsequently completed under
protocol commit `f01fff9`, after an independent pre-result review returned GO.
All 18,224 clauses launched both predictions at clause start. The 100 ms gate
applied to 7,756 survivors and discarded 10,468 results; the 500 ms gate
applied to 5,511 survivors and discarded 12,713 results.

No 100 ms target passed the frozen two-cohort gate. At 500 ms, only Disk
passed: conditional accuracy versus unconditional current versus
same-survivor majority-Light was `92.3140% / 92.1488% / 91.9835%` on SWE100
and `93.0337% / 93.0101% / 92.3747%` on SWE277. SWE100 changed from
`TP/TN/FP/FN = 4/1111/2/93` to `7/1110/3/90`; SWE277 changed from
`34/3918/7/290` to `35/3918/7/289`. The improvement is therefore real under
the frozen exact-accuracy gate but small in absolute terms.

The 500 ms latency and RSS heads failed, while CPU was cohort-inconsistent:
conditional CPU lost `3.19 pp` to unconditional current on SWE100 and gained
`2.29 pp` on SWE277. Treat 500 ms conditionalization as a development GO only
for the current Disk head, not as a global landmark selection. The result uses
whole-clause Disk I/O and does not establish future-after-500-ms Disk delta or
scheduler utility; those require time-resolved post-landmark telemetry and a
concrete action cost/benefit model. The durable result is
`analysis/results/tool-resource-nighttime-3bucket-20260728/landmark-conditional-current-fit-only.json`.

## 2. Authority and required amendment

Before reading any new three-bin result, rewrite
`analysis/development/tool-resource-canonical-objective.md` so it records this
human-authorized 2026-07-28 amendment:

- latency boundaries are exactly `2000` and `8000` ms;
- the product is one normalized three-bin PMF;
- the primary latency score is exact three-class accuracy;
- the comparator is the same-evaluation majority-class accuracy;
- class counts/rates and the 3x3 label-by-prediction confusion matrix are
  required reconciliation diagnostics;
- the old eight-boundary accuracies and nine-bin exact accuracy are historical
  diagnostics and must not select the new candidate;
- CPU, RSS, Disk, short-null, causal visibility, compound-command, and evidence
  eligibility semantics remain unchanged;
- all new SWE results are development-exposed and non-confirmatory.

Do not describe this amendment as pre-registered: the older nine-bin and
representation/arbitration results were already visible.

`analysis/development/tool-resource-service-architecture.md` remains
authoritative and must not be edited.

## 3. Non-negotiable invariants

1. Offline evaluation and `resource-agentd` call the same
   `ClauseResourceKB` prediction, canonicalization, arbitration, update, and
   bucket code. Do not create a separate executable offline predictor.
2. An observation is visible only when
   `observation.ts_end < query.ts_start`. Reject backdated queries.
3. New causal evidence enters the run-local KB only after successful trace
   finalization and only when `ingest_eligible is True`.
4. Public evidence is frozen and excludes the active repository. Repository
   evidence is causal and scope-local.
5. Missing, invalid, ambiguous, lossy, cleanup-invalid, or ineligible evidence
   is unavailable, never a zero or negative label except for the existing
   short-null resource policy.
6. Compound commands remain `compound_command_uncomposed`. Do not add OR,
   addition, max, or another command-composition heuristic.
7. Prediction/telemetry failure never changes, delays, or replaces workload
   execution or workload status.
8. Do not use command results, observed duration, post-execution filesystem
   state, expected answers, or another hindsight signal as a feature.
9. Use only the two locked SWE-ReBench inputs in Section 5.
10. Do not add a dependency, ANN index, embedding model, neural model, LLM
    call, per-dataset rule, or per-binary prediction table.

## 4. Branch and Git isolation

This checkout already uses the dedicated experiment branch:

```text
repository: /home/chiyu/workspace/agent-sched-bench
branch:     exp/tool-resource-3bucket-swe
base:       776d2857b6325992262da59abce88b256404fed9
```

The pre-existing `dev/cpu-only` history and all unrelated changes are
user-owned. Do not create another worktree for this goal.

Before editing or resuming:

1. `cd /home/chiyu/workspace/agent-sched-bench`;
2. verify `git status --short --branch`, `git rev-parse HEAD`, and
   `git worktree list --porcelain`;
3. stop if the branch is not `exp/tool-resource-3bucket-swe`;
4. keep unrelated files and any active collection artifacts untouched;
5. use one write lane per file;
6. never use `git add -A`;
7. stage only explicit task-owned paths and inspect `git diff --cached`;
8. do not push.

## 5. Data boundary

### 5.1 Locked SWE inputs

Use exactly these already collected, finalized, development-exposed
SWE-ReBench clause-level eBPF traces:

```text
traces/swe-rebench/qwen3.7-max/swe100-full-5be74da-20260726/simulate_cloud_model_c2_20260726T005356962.jsonl
traces/swe-rebench/qwen3.7-max/swe277-full-5be74da-20260726/simulate_cloud_model_c2_20260726T024552768.jsonl
```

The 2026-07-28 result JSONs retain their original `/tmp` provenance paths, but
the same named source artifacts now live under the stable trace paths above.
They are not missing. At plan revision, the SWE100 trace was 437 MB/9,016 JSONL
records and the SWE277 trace was 2.0 GB/26,736 JSONL records.

Before implementation, audit both inputs and record:

```text
call and clause eligibility
bin
argv
latency_ms
peak_cpu_cores
sampled_peak_rss_mb
disk_io.read_write_bytes_total
task identity
repository identity
manifest/task order
```

Reconcile the audit with the previous extraction counts: SWE100 has 4,458 rows,
99 tasks, and 93 repositories; SWE277 has 13,766 rows, 277 tasks, and 213
repositories. A difference must be explained before scoring. Do not silently
substitute a stale `/tmp` path, legacy proxy data, an active trace, a partial
subset, or another cohort.

Use both SWE fit/evaluation orientations. Within an evaluation repository,
preserve manifest task order for causal local updates. Do not start a new
collection during this goal.

### 5.2 Split and freshness rules

- Fit/evaluation task IDs must be disjoint.
- Public fit evidence must exclude the evaluated repository.
- Every call in one trace is predicted before successful trace finalization
  releases that trace's observations.
- Row identity and three-bin labels must be identical across all compared arms.
- Fit-derived vocabulary or shrinkage parameters may use fit folds only.
- Outer development results are read once after the candidate and its
  selection rule are frozen.
- Existing result artifacts are immutable.

## 6. Baselines and diagnostic upper bounds

Run the following on the exact same eligible rows:

1. **Majority:** always predict the most common evaluation bucket. This is a
   deliberately strong same-evaluation constant baseline.
2. **Current:** current raw repo exact/prefix/bin first-hit backoff plus public
   bin/global, relabelled only through the fixed `2000/8000` ms buckets.
3. **Public-only:** frozen public evidence with no evaluation-local updates.
4. **Local-only diagnostic:** use selected local evidence where available;
   report coverage separately.
5. **Current/public oracle:** hindsight-select the correct arm per row. This is
   an analysis-only upper bound and must never enter runtime.
6. **Node oracle:** hindsight-select among exact, structured, bin, and global
   candidates. This estimates whether representation/arbitration has enough
   ceiling to justify more work.

Oracles are not methods. Label them `oracle: true`, never compare them as
deployable candidates, and never use their row-level selections as features.

If the oracle cannot materially exceed `max(current, majority)`, stop
architecture expansion and report the ceiling. Do not spend the remaining time
on a model sweep.

## 7. Candidate sequence

Change one mechanism axis at a time. Every iteration states its hypothesis
before reading its result.

### Candidate R — structured argv representation

**Hypothesis:** raw ordered prefixes fragment equivalent option orderings and
overweight early token position, while public bin/global discards useful argv
semantics. A role-aware signature will improve three-class accuracy without
storing arbitrary opaque values in public keys.

Keep repository-local raw exact keys. Replace raw ordered-prefix generalization
in the candidate arm with one versioned `generic-argv-v3-role` signature:

```text
normalized bin
fit-approved stable subcommand, otherwise its shape
sorted option multiset with repetitions
option value shape and numeric sign/order-of-magnitude
ordered positional value shapes
explicit "--" boundary
```

Rules:

- normalize `argv[0]` to `bin`;
- do not sort positional arguments;
- option order may be normalized, but option multiplicity is retained;
- preserve long/short option names;
- preserve numeric sign and base-10 order of magnitude;
- map paths, URLs, UUIDs, hex IDs, random-looking values, and opaque values to
  typed placeholders;
- never place arbitrary raw paths, IDs, secrets, or temporary names in public
  keys;
- a plain subcommand literal is public only when it matches
  `[A-Za-z][A-Za-z0-9_-]{0,31}` and appears for the same binary in at least
  three distinct fit repositories; this vocabulary uses no labels;
- all other literals become a shape token;
- record the canonicalizer version in prediction provenance and snapshots.

First compare structured public evidence against public bin/global with local
updates disabled. Then compare the full current arm against the structured arm
with the same existing hard arbitration. This isolates representation from
shrinkage.

### Candidate S — public/local posterior shrinkage

Run this only after Candidate R is implemented and reviewed, or when the
current/public oracle demonstrates a decision-changing arbitration ceiling.

**Hypothesis:** first-hit repo selection treats one local observation as
infinitely more trustworthy than public evidence. A public prior with causal
local updates will improve three-class accuracy while still adapting online.

For each query and target:

1. select the deepest non-empty local node;
2. select the deepest non-empty public node independently;
3. ensure the scopes contain disjoint observations;
4. compute:

```text
posterior = (local_counts + alpha * public_pmf) / (local_n + alpha)
```

When one scope is absent, use the other. Do not sum exact, signature, bin, and
global nodes together: those contain duplicated observations.

Select one global `alpha` from the fixed set:

```text
1, 4, 16, 64
```

Use repository-grouped inner fit folds and three-class latency accuracy only.
Never select `alpha` on an outer evaluation result. Break an exact tie in favor
of the larger `alpha`, the more conservative public prior. Use the selected
`alpha` unchanged for latency, CPU, RSS, and Disk; do not tune per target,
binary, repository, SWE cohort, or bucket.

The previous `local_n <= 4` replacement gate already failed and must not be
reintroduced.

### Conditional Candidate C — one inference-time context

Do not begin this by default. It is allowed only if the reviewed residual
analysis shows all of the following:

- the structured/shrinkage candidate has meaningful oracle ceiling but fails;
- one context is available before execution;
- it has at least 90% coverage in both fit and evaluation data;
- it varies enough to identify an effect;
- using it does not name a benchmark or task.

Choose at most one of CPU quota/machine class, cwd role, or a genuinely
pre-execution input-size feature. State the feature and falsifiable hypothesis
before scoring. If no feature satisfies the gate, stop. Do not collect new
telemetry to manufacture one during this goal.

## 8. Metrics and decision gates

### 8.1 Latency

For each arm and evaluation orientation/fold, report:

```text
eligible_examples
short/middle/long counts and rates
predicted class counts
3x3 confusion_label_by_prediction
three_class_accuracy
majority_class
majority_class_accuracy
accuracy_minus_majority_percentage_points
accuracy_minus_current_percentage_points
prediction-unavailable count
scope/key-kind/support provenance counts
```

Primary development GO:

1. identical eligible row IDs and labels across candidate, current, and
   majority;
2. candidate three-class accuracy is strictly greater than both current and
   same-evaluation majority in each of the two SWE fit/evaluation
   orientations;
3. the sign of `candidate - max(current, majority)` is positive under a paired
   repository-cluster bootstrap; report the interval, but do not turn an
   exploratory interval into a confirmation claim;
4. no unavailable prediction is converted to a synthetic majority fallback.

There is no hidden average over the historical eight boundaries. Do not use the
old boundary metrics, exact nine-bin metric, Brier/NLL, bucket MAE, q-error,
precision/recall, or a hand-selected subset to choose the candidate.

### 8.2 CPU, RSS, and Disk

For each target separately report:

```text
eligible n
observed Heavy
observed Light
short-null-imputed Light
null-unavailable
Heavy count/rate
TP/TN/FP/FN
accuracy
same-evaluation majority-Light accuracy
current accuracy
candidate deltas
```

A resource target has development skill only if candidate accuracy is strictly
greater than both current and majority-Light in every outer
orientation/fold-union. Do not combine the three targets into one score.

A latency candidate may be reported as promising when its latency gate passes
but a resource target does not. It may not replace the shared runtime
arbitration if any resource target materially regresses from current. In that
case, finish with a target-specific NO-GO and leave runtime arbitration
unchanged; do not add target-specific similarity or a second model.

### 8.3 Operational gates

Measure, on representative loaded state:

- snapshot build/restore succeeds;
- online and offline PMFs/provenance are identical for the same timestamped
  stream;
- p50/p95 prediction latency does not regress materially from current;
- state size is reported;
- invalid/missing evidence and telemetry failure remain workload-neutral.

These are implementation gates, not prediction scores.

## 9. Review and validation

Before any changed predictor/evaluator code is used for formal development
results, spawn one fresh independent reviewer with a bounded brief:

- exact changed files;
- the three-bin and majority contract;
- structured signature privacy/identity rules;
- public/local disjointness and shrinkage formula;
- causal visibility and trace-close update;
- online/offline parity;
- result-row reconciliation.

The reviewer reports and stops. Fix critical/major findings and request focused
re-review. Minor fixes need no re-review unless behavior changes.

Run only checks that can falsify this work:

```text
tests/test_clause_resource_kb.py
tests/test_clause_latency_bucket_evaluation.py
relevant prediction/service cases in tests/test_tool_resource_services.py
ruff on changed Python files
git diff --check
```

Add or update one golden test that feeds the same timestamped event stream
through offline and online entry points and asserts identical three-bin PMFs,
provenance, causal visibility, and restored state.

Do not run a full suite unless the changed import/protocol surface creates a
specific cross-module risk. Do not run a real replay or collection merely to
validate pure KB logic.

## 10. Allowed files

Task-owned edits are limited to the minimum subset of:

```text
analysis/development/tool-resource-canonical-objective.md
analysis/development/tool-resource-kb-predictor-nighttime-goal.md
src/tool_resource/runtime_kb.py
src/tool_resource/profile.py
src/tool_resource/resource_agentd.py
src/tool_resource/README.md
scripts/evaluation/evaluate_clause_latency_buckets.py
scripts/evaluation/evaluate_clause_resource_classes.py
tests/test_clause_resource_kb.py
tests/test_clause_latency_bucket_evaluation.py
tests/test_tool_resource_services.py
analysis/results/tool-resource-nighttime-3bucket-20260728/
```

Do not edit a listed file unless the selected implementation actually requires
it. Do not create a new framework, model hierarchy, config file, evaluator
family, status document, or helper script when the existing core/evaluators can
hold the change.

## 11. Protected scope

Do not modify:

```text
analysis/development/tool-resource-service-architecture.md
src/tool_resource/telemetry.py
src/tool_resource/telemetryd.py
src/tool_resource/telemetry_protocol.py
src/tool_resource/clause_parser.py
src/tool_resource/clause_bridge.py
src/tool_resource/store.py
src/tool_resource/client.py
src/tool_resource/resource_protocol.py
src/trace_collect/
src/agents/
configs/benchmarks/
traces/
existing analysis/results directories
```

If a necessary change crosses this boundary, stop that candidate and report
the exact interface need. Do not expand scope autonomously.

## 12. Eight-hour execution schedule

Use this as a budget, not a requirement to consume time:

| Elapsed | Work |
|---|---|
| 0:00–0:30 | Re-read locks; verify the dedicated branch; audit the two locked SWE inputs; write the objective amendment before new results. |
| 0:30–1:15 | Implement the fixed three-bin contract and baseline metrics; reconcile rows; measure majority/current/public/oracle ceilings. |
| 1:15–2:45 | Implement Candidate R and focused tests; smoke without reading formal metrics. |
| 2:45–3:15 | Independent bounded review; fix/re-review major findings. |
| 3:15–4:00 | Read Candidate R development result once; apply its gate and diagnose. |
| 4:00–5:15 | If authorized by the decision tree, implement Candidate S and its causal/serialization tests. |
| 5:15–5:45 | Focused review of changed behavior. |
| 5:45–6:30 | Freeze the candidate and read outer development results once. |
| 6:30–7:15 | Mechanism diagnosis; run Conditional Candidate C only if its preconditions hold. |
| 7:15–8:00 | Operational checks, explicit-path staging audit, coherent commits, compact decision artifact, and final handoff. |

No single offline evaluation is expected to exceed 30 minutes. If it does,
stop, profile, and diagnose rather than waiting or shrinking the cohort.

## 13. Checkpoint and restart protocol

One checkpoint is one independently judgeable step, not one command or an
unfinished intermediate state. The expected checkpoints are:

1. three-bucket contract and reconciled baselines;
2. Candidate R implementation, focused tests, and review;
3. Candidate R two-orientation SWE evaluation and GO/NO-GO;
4. Candidate S implementation, focused tests, and review, if authorized;
5. Candidate S two-orientation SWE evaluation and GO/NO-GO, if authorized;
6. Conditional Candidate C implementation/evaluation, only if authorized;
7. final operational checks and decision artifact.

At each completed checkpoint:

1. preserve frozen result artifacts;
2. record exact inputs, configuration, elapsed time, row counts, and gate
   outcome in the result JSON, not a new status Markdown;
3. run the focused check that falsifies the change;
4. stage explicit paths only;
5. inspect staged diff and conflict markers;
6. commit with the repository commit format and the result-bearing body below;
7. do not leave a completed checkpoint uncommitted;
8. do not commit every command, a failed partial edit, or an unfinished
   experiment;
9. do not push.

Use this commit body contract:

```text
[type] Brief checkpoint description

- Stage: checkpoint or candidate name
- Result: key test or metric result; use "not read" when formal metrics remain frozen
- Decision: GO, NO-GO, continue to named next gate, or implementation-ready
- Verification: focused tests and reviewer verdict when review is required
- Artifact: authoritative result path, or "none" for a pre-evaluation implementation checkpoint
```

For an evaluation checkpoint, include both SWE orientations and the candidate,
current, and majority accuracies in `Result`. Keep the full confusion matrices,
row counts, configuration, provenance, and uncertainty in the immutable result
JSON rather than expanding the commit message. A message containing only
`checkpoint`, `progress`, or `WIP` does not satisfy this contract.

After interruption or compaction:

1. re-read this file and both canonical locks;
2. inspect Git/worktree state and the latest result decision artifact;
3. verify whether a process is actually live before restarting;
4. continue from the last passed gate;
5. never rerun a completed outer evaluation merely because context was lost.

## 14. Stop conditions

Stop the affected line immediately when:

- row IDs or labels differ across compared arms;
- an evaluator reimplements core prediction semantics;
- an observation becomes visible at `ts_end == query.ts_start` or earlier than
  trace finalization permits;
- public fit includes the evaluated repository;
- a candidate needs hindsight, expected answers, dataset identity, or an
  unapproved protected-file change;
- reviewer finds an unresolved major/critical issue;
- the oracle ceiling cannot change the decision;
- Candidate R/S fails its gate and residual evidence does not authorize
  Conditional Candidate C;
- a run materially exceeds its estimate or stalls;

Do not substitute a smoke, partial cohort, stale aggregate, or legacy proxy for
a passing formal development result.

## 15. Completion criteria

The goal is complete through either path.

### Path A — reviewed candidate

- the canonical objective records the open three-bin amendment;
- runtime, offline evaluation, profile validation, docs, and tests agree on
  boundaries `2000/8000`;
- one reviewed candidate passes the latency development GO;
- resource targets are each reported against current and majority-Light;
- any target that fails is explicitly NO-GO, without a fabricated claim;
- online/offline golden parity, causal visibility, snapshot restore, focused
  tests, Ruff, and `git diff --check` pass;
- one compact immutable decision artifact contains inputs, counts, metrics,
  uncertainty, provenance, reviewer verdict, and limitations;
- every completed checkpoint has one scoped result-bearing commit;
- commits contain only task-owned files and nothing was pushed.

### Path B — reviewed NO-GO

- the three-bin contract and baseline are implemented and verified on both
  locked SWE inputs;
- every actually evaluated candidate has identical rows/labels and a recorded
  gate outcome;
- the negative result identifies whether the limiting mechanism is
  representation, arbitration, data coverage, or lack of predictive signal;
- runtime arbitration remains at the last reviewed valid implementation;
- existing artifacts and protected files remain untouched;
- every completed checkpoint has one scoped result-bearing commit;
- task-owned commits/checks are clean and nothing was pushed.

An eight-hour timeout alone is not completion. A passing smoke, one orientation,
one favorable target, or a result artifact without reviewed executable parity
is not completion.
