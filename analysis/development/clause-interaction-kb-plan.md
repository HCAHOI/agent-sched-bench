# Task-Local Resource Prediction Research Record

**Effective:** 2026-08-05

**Status:** the offline-agent and deterministic task-phase directions are
closed. The parameter-free causal within-task telemetry overlay also returned
NO-GO. The 50-task validation partition is development-exposed; the 50-task
final partition remains unread and unauthorized. The generic linear residual
calibrator and causal full-command outcome memory also returned NO-GO. Disk has
a separately frozen cache-state mechanism test that is not authorized to run.
The pytest target-overlap test also returned NO-GO; it isolated a strong
CPU/RSS mechanism but not a Disk mechanism. Final remains closed.
Adding pip package-set overlap also returned component NO-GO: RSS passed the
five-point gate, but latency and CPU did not. The next frozen component test
added a parameter-free early-execution lower-bound correction and returned
NO-GO: elapsed time was valid and useful, but cgroup CPU was not a lower bound
for the canonical clause-owned CPU target. Final remains closed.

This record extends `tool-resource-canonical-objective.md`, which remains the
authority for targets, bucket boundaries, causal visibility, eligible command
rows, telemetry validity, Current, and compound-command composition. Git and
the frozen artifacts retain superseded protocols; this file states the current
research decisions.

## 1. Settled evidence

### 1.1 Current and semantic KBs

Current is the causal raw exact/prefix/binary `ClauseResourceKB`. On the exposed
SQLGlot 80/20 replay it beats the constant majority for latency, CPU, RSS, and
Disk. Dynamic task-final updates change exact accuracy by only +0.287, 0.000,
+0.422, and -0.287 percentage points respectively.

Changing the KB representation is not the main opportunity:

- pip package-set Jaccard matching improved an exposed latency replay, but
  changed only six commands across six tasks on any hard target and had no
  scheduler consumer;
- pytest semantic matching improved four commands, all under one signature,
  below its frozen diversity gate;
- causal package/cache state was not identifiable in the existing traces; and
- trie, lattice, prefix, generic argv, and CacheWise-style variants did not
  produce a robust workload-level gain.

These results close “find a better lookup structure over the same command
evidence.” A new structure is not a new signal.

### 1.2 Early execution and scheduling

For the existing EAR CPU-page action set, a hindsight early-execution oracle
reduced post-decision reservation by 5.686%, below the frozen 10% gate. The 176
commands that still needed eight cores accounted for 89.84% of the control
reservation. More accurate prefix prediction cannot rescue that consumer's
limited action ceiling, so Phase 2 and scheduler integration remain closed.

### 1.3 Deterministic task phase

The only 5/3/3/3 candidate that passed its exposed development continuation
gate is `full-test-third-or-later-v1`. It recognizes an unselected single-clause
pytest or exact `make test`, counts earlier completed full-suite calls in the
same task, and applies a monotone upward PMF correction only on the third and
later call. It reads command history, not prior outputs, labels, telemetry, or
the current result.

On the exposed 80/20 replay:

| Target | Current | Phase feature | Delta |
|---|---:|---:|---:|
| Latency | 77.937% | 79.083% | +1.146 pp |
| CPU | 90.476% | 93.333% | +2.857 pp |
| RSS | 91.561% | 94.093% | +2.532 pp |
| Disk | 81.375% | 81.375% | bit-identical |

All 16 changed target predictions were helpful, none harmful, across five
tasks. Severe underpredictions changed from 4 to 0 for latency and from 7 to 1
for both CPU and RSS. This feature was designed after the residual cases were
visible, so the result is post-hoc development evidence, not a claim.

Artifacts:

- `analysis/results/tool-resource-5-3-3-3-20260804/sqlglot80-20-full-test-phase.json`
- `analysis/results/tool-resource-5-3-3-3-20260804/sqlglot80-20-full-test-phase.rows.jsonl`

## 2. Offline-agent direction — closed

### 2.1 What was tested

The common hypothesis was that an offline agent could interpret task-local
command/results history, identify environment blockers and remediations, and
compile that interpretation into a zero-agent-cost prediction-time feature.
The experiments progressively removed agent freedom so that prompt or code
generation failures could be separated from the underlying signal.

| Stage | Agent action | Observed outcome | Recorded generation cost |
|---|---|---|---:|
| Residual discovery | Read exposed errors and propose a mechanism | Correctly identified runner/dependency readiness; opportunity analysis only | not a prediction run |
| Per-query blind classifier | Classify each full-test query from prior results | Implemented but not used as evidence; no formal result artifact, and per-query calls violate the later task-level cost boundary | not run |
| Frozen Python extractor | Select residuals, then emit one local feature function | Exact accuracy unchanged; 5 helpful and 5 harmful changes, helpful in only 2 tasks | 213,257 input / 2,077 output tokens |
| Relational Python extractor | Emit scope and blocker/remediation span functions | Structural NO-GO: generated source violated the frozen auditable-source rule | 59,872 / 3,047 |
| Declarative regex | Emit schema-bounded scope/blocker/remediation patterns | Structural NO-GO: blocker captured only one distinct identifier | 56,982 / 1,664 |
| Family and scope | Emit family scopes plus relations | Structural NO-GO: generated `\z` did not compile in Python 3.12 | 32,742 / 1,812 |
| Finite typed catalog | Select one host-built executable configuration | Selected the credible `unittest` missing-module to install relation, but no supported hard-PMF relation/scope contrast existed | 14,111 / 113 |

The final arm is decisive about the architecture question. The host generated,
validated, ranked, and executed every candidate; the agent could only select an
ID. It chose `C006`, not the support-only `C000`, and its local runtime passed at
0.694 ms p95. Under `no-target`, however, `blocked` and `closure_candidate` had
the same hard mode for latency, CPU, RSS, and Disk. The second scope had only
one task per observed state. Removing arbitrary code and regex generation
therefore removed engineering failures but did not reveal predictive resource
separation.

### 2.2 Conclusion

Observed:

- the agent can identify a semantically credible blocker/remediation story;
- a successful install does not imply complete dependency closure;
- even exact named closure states can share the same resource bucket; and
- the useful deterministic phase feature depends only on repeated workflow
  position, not semantic interpretation of output.

Inference:

> In these SQLGlot traces, blocker/remediation semantics explain why a command
> can proceed, but not enough of how much work the proceeding command performs.

The agent direction is closed for this project. Do not add another prompt,
critic, repair loop, DSL, generated parser, tool adapter, or online agent
integration. Reopen it only if a new inference-time signal first demonstrates
resource separation without an agent and the agent has a named decision that a
deterministic selector cannot make.

Authoritative artifacts:

- `analysis/results/tool-resource-5-3-3-3-20260804/sqlglot80-20-offline-agent-extractor-v1`
- `analysis/results/tool-resource-5-3-3-3-20260804/sqlglot100-relational-state-v1`
- `analysis/results/tool-resource-5-3-3-3-20260804/sqlglot100-declarative-relational-v1`
- `analysis/results/tool-resource-5-3-3-3-20260804/sqlglot100-declarative-family-state-v1`
- `analysis/results/tool-resource-5-3-3-3-20260804/sqlglot100-typed-catalog-state-v1`

## 3. Evidence boundary

The original 100 SQLGlot tasks are fully development-exposed. The additional
100 task IDs were assigned before outcome access into 50 validation and 50
final tasks by the committed manifest:

`analysis/development/sqlglot-relational-task-split.json`

No agent arm opened either partition. The frozen task-phase method consumed
validation and returned NO-GO. Validation is now development evidence. Final
remains unavailable because the exact method did not pass its gate; no final
command, output, label, telemetry, or derived score has been inspected.

## 4. Deterministic task-phase replication — closed

The method, PMFs, coverage rule, validation gate, and final authorization were
frozen before reserved access. Validation had 41 raw third-or-later commands
across 24 tasks, well above the three-task coverage gate, and scored 1,044
identical command rows from 50 evidence-valid tasks.

The evaluated `full-test-third-or-later-v1` candidate was unchanged:

- the existing `parse_pytest` definition of an unselected full suite and exact
  `make test` are the only recognized commands;
- phase is the causal count of earlier completed recognized commands in the
  same task, capped at third-or-later;
- fit one PMF for third-or-later commands using all 100 development tasks;
- apply only a monotone upward override for latency, CPU, and RSS;
- leave Disk and every other command bit-identical to frozen Current; and
- perform no update within validation or final.

This is deliberately tool-aware and narrow. A same-repository success would
establish repeatability of the phase mechanism, not generality across tools or
repositories.

The development fit contains 44 third-or-later commands across 34 tasks.
Their latency labels are `{0: 3, 4: 41}`, CPU labels are `{0: 3, 2: 41}`, RSS
labels are `{0: 3, 2: 41}`, and Disk labels are `{0: 25, 1: 19}`. Thus the
frozen hard prediction is the highest bucket for latency, CPU, and RSS. Disk is
not fit or overridden. These counts are fit evidence, not a fresh result.

Validation produced 17 helpful and four harmful target changes across seven
helpful tasks. CPU improved from 82.887% to 83.780% (+0.893 pp), RSS from
80.978% to 81.658% (+0.679 pp), and Disk remained bit-identical at 82.044%.
Available-only latency improved from 74.976% to 75.168% (+0.192 pp), but one of
1,044 latency rows was unavailable for both Current and candidate. Complete
latency accuracy was therefore undefined. Under the frozen fail-closed
no-regression gate, validation was NO-GO. Even the available-only diagnostic is
far below the five-point research target, so changing the availability policy
would not rescue the mechanism. Final stays closed.

Artifacts:

- `analysis/results/tool-resource-5-3-3-3-20260804/sqlglot100-full-test-phase-fresh-fit-v3/artifact.json`
- `analysis/results/tool-resource-5-3-3-3-20260804/sqlglot50-full-test-phase-validation-v1`

Two pre-result amendments preserved the frozen method and gate. The first
validation invocation exposed only `KeyError: peak_cpu_cores`; the wrapper was
changed to represent omitted unavailable Current predictions explicitly as
`None`. The unchanged retry exposed only a `None >= None` scoring exception;
the shared gate was changed to fail closed when either arm lacks a complete
primary metric. Neither invocation produced a result artifact or exposed a
validation score, row, label, or method comparison. Each repair was reviewed
and bound to a new development-fit artifact before the next invocation.

## 5. Post-validation signal audit and next direction

### 5.1 What existing traces can support

All 1,792 development commands align exactly with the preceding OpenClaw tool
history. Ninety-five of 100 tasks used `edit_file`; 167 of 172 full-suite calls
had a known touched-path set. The signal exists, but cumulative touched-file
count plus tool family harmed the exposed 80/20 replay by 1.719, 0.952, 1.266,
and 3.152 points for latency, CPU, RSS, and Disk. Workset size alone is not the
missing mechanism.

An interactive 25-task fit / 25-task test diagnostic on the now-exposed
validation partition suggested that Current plus structured command features
and causal task-local history could improve latency, CPU, and RSS. No
executable evaluator or machine-readable artifact was retained, so those
numbers cannot select a method or authorize final access. They motivate only a
simpler, reproducible test of whether completed-call telemetry carries signal.

### 5.2 Why Disk is different

Within-task raw-exact reuse changed Disk with ten helpful and ten harmful cases;
command identity, structured work units, edit history, and prior resource
buckets do not determine physical I/O. The retained interface has no
pre-command page-cache residency, installed-package inventory, or requested
wheel-cache inventory. Final eBPF Disk bytes exist only after execution.

The validation error decomposition confirms that this is not just a command
modeling problem. Of 1,008 rows with both a Disk label and Current prediction,
181 are wrong: 96 underpredict and 85 overpredict. Sixty-nine underpredictions
are read-dominant and 27 are write-dominant. Medium truth is mixed (162
read-dominant, 109 write-dominant), while 119 of 122 High rows are
write-dominant. A fixed causal model adding cumulative prior read/write bytes to
Current changed exposed 25/25 accuracy by only +0.198 points, with 12 helpful
and 11 harmful changes. A repeat-full-suite state harmed by 0.496 points, and
conditioning repeat state on `PytestSignature` harmed by 1.290 points.

Current's shell composer is not losing bytes at bucket boundaries: it composes
deterministic draws from the raw empirical clause values and sums Disk over the
shell graph. Single-clause and compound commands account for 92 and 89 errors,
respectively. Another composer or aggregate history feature is therefore not
the next experiment.

The available probes are feasible but measure different state. Reading
container cgroup-v2 `memory.stat` from a persistent Python process cost 0.048 ms
p95 over 1,000 host trials, but it is aggregate rather than command-specific.
`fincore` over 572 files totaling 12.39 MiB cost 13 ms p95 over 20 trials and
uses the same `mincore` residency mechanism needed for a command footprint.
The container cgroup mount is read-only on this Linux 5.15 host and exposes no
`memory.reclaim`; global cache eviction remains forbidden. Per-file
`POSIX_FADV_DONTNEED` is available.

### 5.3 Frozen development test: causal completed-call overlay

`causal-call-overlay-v1` changes no parser, representation, threshold, or
cross-task KB state. It has no fit phase, learned parameters, or agent call.

Use the original, fully exposed 100-task SQLGlot run and its existing ordered
80-task warm-up / 20-task test split. After warm-up, Current and the candidate
start from byte-identical KB snapshots. Within each test task, score eligible
commands in retained call order with strictly increasing synthetic query times:

- Current remains unchanged and sees the task only after whole-task settlement;
- after scoring a command, the candidate alone receives that command's eligible
  clauses with an end time strictly before the next command query;
- the candidate may use those clauses only for latency, CPU, and RSS predictions
  of later commands in the same task;
- Disk is copied bit-for-bit from Current; and
- after task settlement, both arms must contain identical cross-task evidence
  before the next task begins.

Both arms use the existing raw exact/argv-prefix/bin hierarchy and physical
compound-command composition. The candidate never sees the current command's
telemetry, output, duration, or label. It records the selected scope and support
for every changed target. Eligible command IDs, labels, availability, order,
and Current predictions must match the committed baseline exactly.

The retained traces are successful, final-valid sessions. This replay is thus
an accuracy ceiling for a deployable ephemeral overlay: online use would also
have to wait for the previous `FinishCall`, keep the observation outside the
persistent KB, and discard the task-local state if session finalization later
fails. No runtime change is authorized by this test.

Continue to a runtime cost/validity audit only if all are true on the fixed 20
development test tasks:

1. latency, CPU, and RSS exact accuracy each improve by at least 5.0 percentage
   points over Current;
2. their severe-underprediction rates do not increase;
3. across those targets, helpful changed predictions exceed harmful ones and
   helpful changes cover at least five tasks; and
4. Disk rows and PMFs are bit-identical to Current.

A miss stops this exact overlay without adding a support threshold, changing
the hierarchy, or selecting a subset. The result remains development-only and
cannot open the final 50 tasks.

The fixed replay returned NO-GO on 349 commands:

| Target | Current | Overlay | Delta |
|---|---:|---:|---:|
| Latency | 77.937% | 78.797% | +0.860 pp |
| CPU | 90.476% | 90.952% | +0.476 pp |
| RSS | 91.561% | 91.983% | +0.422 pp |
| Disk | 81.375% | 81.375% | bit-identical |

Across latency, CPU, and RSS, 11 changed target predictions were helpful and
six harmful, with helpful changes in five tasks. All three
severe-underprediction rates increased. The dominant failure was not a missing
support threshold: exact-command observations themselves alternated between
resource modes. In `tobymao__sqlglot-4519`, one `python3 -m pytest` call was
correctly lowered from the highest latency/CPU/RSS buckets, but that low
observation then lowered the next identical command, whose truth returned to
the highest buckets. The overlay therefore lagged a changing execution state
by one command.

This closes direct last-observation persistence. A later method must condition
on the requested work and recent state jointly; it cannot merely publish
current-task telemetry into the existing hierarchy.

Artifacts:

- `analysis/results/tool-resource-5-3-3-3-20260804/sqlglot80-20-causal-call-overlay-v1/result.json`
- `analysis/results/tool-resource-5-3-3-3-20260804/sqlglot80-20-causal-call-overlay-v1/rows.jsonl`

### 5.4 Frozen Disk mechanism test: file footprint x residency

This is a controlled development experiment on already-exposed task IDs. It
does not consume the final partition and cannot establish task generalization.
It tests the narrower causal hypothesis that physical read bytes require both a
command-specific file footprint and current page residency.

Selection uses only the original 100 development tasks. A task is eligible when
its first successful, single-clause command parsed by the frozen
`PytestSignature` grammar exists. Success is used only to ensure that repeated
execution is meaningful; no Disk label or Current error selects a task. Seed 42
shuffles the 81 eligible task IDs and fixes the first 12:

```text
tobymao__sqlglot-4459  tobymao__sqlglot-4004
tobymao__sqlglot-4390  tobymao__sqlglot-4430
tobymao__sqlglot-4165  tobymao__sqlglot-3975
tobymao__sqlglot-4438  tobymao__sqlglot-3891
tobymao__sqlglot-4519  tobymao__sqlglot-4148
tobymao__sqlglot-4696  tobymao__sqlglot-4393
```

For each task, replay only the causal prefix before the selected command and
commit that filesystem state as a temporary prepared image. Run the selected
command once under `strace -f -e trace=%file`; this discovery run is not scored.
Normalize existing regular-file paths in first-access order, excluding
`/proc`, `/sys`, and `/dev`, and stop at 4,096 paths or 512 MiB total file size.
No path name, resource label, or result-dependent rule is added.

Starting from the same prepared image, run each condition twice in a fresh
container:

- **cold:** apply `POSIX_FADV_DONTNEED` to every template file, then measure its
  resident-page fraction with `mincore`;
- **warm:** sequentially read every template file, then measure the same
  resident-page fraction.

Execute the original command unchanged after the probe and collect the existing
eBPF physical read/write bytes. Condition order is counterbalanced by seed 42;
concurrency is one because page cache is host-global. Charge discovery,
preparation, intervention, probe, image storage, and command time separately.
Delete temporary images only after the result artifact is committed.

The predeclared gate is:

1. at least 10/12 tasks have a bounded discovery template, matching command exit
   codes in all four measured runs, and valid telemetry;
2. probe p95 is below 100 ms and at least 8/12 tasks show a warm-minus-cold
   resident fraction of at least 0.50 in both repetitions;
3. paired physical read bytes are lower when warm in at least 8/12 tasks; and
4. at least 4/12 tasks cross to a lower Disk bucket when warm, with at most one
   task crossing in the opposite direction.

Passing only authorizes design of a state-aware predictor; it is not itself a
five-point accuracy result. A failure stops page-residency work without adding
paths, changing bounds, or choosing another task sample. Preparation plus 60
target executions requires 72 container starts and 12 prepared images; it is
estimated at 1--3 hours and up to roughly 60 GB before shared-layer
deduplication. This run is not authorized yet and must wait until the
independent PennyLane collection is finished.

Do not use global `drop_caches`, reinterpret physical Disk as logical bytes,
tune on the exposed 25/25 diagnostic, or open the final partition. Mid-execution
forecasting remains closed unless a new consumer can still act after the
observation time.

## 6. Frozen development test: command-conditioned residual calibration

### 6.1 Question and evidence split

`command-history-residual-v1` tests whether recent state is useful only when
interpreted together with the requested work. It is a supervised linear
calibrator, not a new KB, parser, agent, or tool-specific rule.

Fit on the original SQLGlot100 in retained task order: the first 20 tasks form
only the causal Current warm-up, and commands from the remaining 80 tasks form
the training rows. Evaluate exactly once on the 50 already-exposed validation
task IDs in `sqlglot-relational-task-split.json`. Current for validation is
frozen after all 100 development tasks, as in the committed task-phase
validation result. Do not fit, select, early-stop, or update on validation. The
50 final-test task IDs remain unread and cannot be opened by a validation miss.

Validation outcomes were exposed by the earlier task-phase evaluation, so this
is still development evidence. The new candidate's source, weights,
predictions, and score have not existed when this protocol is committed.

### 6.2 Frozen inference features

For each target independently, use only:

1. that target's Current PMF;
2. generic static command structure: command byte length, clause and argument
   counts, shell loop/pipeline/substitution flags, executable basenames, option
   names, positional-slot counts, and control-edge kinds; and
3. target labels from eligible commands already completed in the same task:
   total history, same executable-set history, and same exact-command history,
   represented by count, class histogram, and last class.

Raw positional literals, repository/task IDs, output text, exit status, current
telemetry, and future calls are excluded. Exact command bytes may be used only
to join a query to its own earlier history; they are never model features.
Categorical features use signed BLAKE2b hashing into 256 fixed dimensions.
Numeric counts use `log1p` and are capped at 10; class histograms are normalized
shares plus `log1p` support. Feature extraction is identical for fit and
validation and resets at each task boundary.

### 6.3 Frozen model and prediction

Fit one affine residual per target for latency, CPU, and RSS:

```text
candidate PMF = softmax(log(max(Current PMF, 1e-6)) + W x + b)
```

Use the already-declared CPU PyTorch dependency in float64. Initialize `W` and
`b` to zero and optimize unweighted cross-entropy plus
`1e-4 * mean(W**2)` with deterministic CPU LBFGS: learning rate 1,
`max_iter=200`, `history_size=20`, `tolerance_grad=1e-7`,
`tolerance_change=1e-9`, and strong-Wolfe line search. Seed is 0. There is no
class weighting, hyperparameter search, calibration threshold, early stopping,
or model selection. Rows with an unavailable target label are omitted only for
that target. If Current has no PMF for an otherwise labelled command, use the
uniform PMF as the residual base and require the candidate to produce a finite
PMF. For the paired comparison, a Current unavailable prediction contributes
zero correct and remains explicit in its unavailable count; it does not make
the entire baseline accuracy undefined.

Hard predictions use the canonical lower-index argmax. Disk hard predictions
and PMFs are copied bit-for-bit from Current. Record the frozen weights, feature
schema, training counts, command rows, per-target confusion matrices, changes,
and wall-clock fit/prediction cost.

### 6.4 Gate

Validation is GO only if all are true:

1. latency, CPU, and RSS exact accuracy each improve by at least 5.0 percentage
   points over the identical Current rows;
2. no target's severe-underprediction rate increases;
3. helpful changed predictions exceed harmful changes and helpful changes cover
   at least ten validation tasks; and
4. Disk hard predictions and PMFs are bit-identical to Current.

Only GO authorizes a separately committed final-test invocation with the exact
same source and frozen weights. NO-GO closes this linear residual candidate;
do not change the hash width, feature list, regularization, optimizer, fit split,
or gate after seeing validation.

Before any real residual fit or validation score, implementation preflight
reached the already-recorded validation row whose Current latency PMF is
unavailable. The original “preserve prediction availability” wording made every
candidate unable to pass by construction. This protocol was therefore openly
amended to the uniform-base and fail-closed-correct-count rule above. The row,
its label, and Current unavailability were already exposed by the task-phase
NO-GO; no residual weight, prediction, or target score was visible.

The frozen validation returned NO-GO on 1,044 command rows:

| Target | Current | Residual | Delta |
|---|---:|---:|---:|
| Latency | 74.904% | 73.372% | -1.533 pp |
| CPU | 82.887% | 83.036% | +0.149 pp |
| RSS | 80.978% | 82.065% | +1.087 pp |
| Disk | 82.044% | 82.044% | bit-identical |

Fit plus tensor prediction took 4.990 seconds on CPU. Across latency, CPU, and
RSS, 192 changes were helpful and 201 harmful. Severe-underprediction rates did
not regress, but none of the three targets approached the five-point gate.
Latency alone changed 288 rows and had 102 helpful versus 118 harmful changes;
the model traded exact modes for fewer severe underpredictions.

The label support and task metadata explain why model capacity is not the next
knob. Fit contains only 4 CPU-Medium and 5 RSS-Medium commands, versus 28 and 87
in validation. Joining the committed task IDs to the cached
`nebius/SWE-rebench` input metadata shows that all 80 fit tasks were created
from 2024-08-28 through 2025-04-25, while all 50 validation tasks span
2023-09-27 through 2024-07-15. The cohorts have zero date overlap and use
different repository versions and environment-setup commits.

Observed: a nominal classifier fit on the later workspace generation cannot
recover middle buckets that barely exist in its labels. Inference: the split is
backward temporal extrapolation, so command and history features omit a real
deployment variable: the checked-out workspace and environment generation.
Do not add timestamp, task ID, version, image ID, class weights, or an ordinal
loss to rescue this result. The next method must first show that physical,
inference-time workspace measurements carry the missing signal.

Artifacts:

- `analysis/results/tool-resource-5-3-3-3-20260804/sqlglot20-80-current-fit-v1`
- `analysis/results/tool-resource-5-3-3-3-20260804/sqlglot50-command-history-residual-v1`

## 7. Frozen development test: causal full-command outcome memory

### 7.1 Question and exposed motivation

Current stores exact and prefix evidence for individual clauses, then composes
their empirical values into a command prediction. It does not retain the final
latency/CPU/RSS/Disk class of the complete command. This test asks whether that
unit mismatch loses repeatable command-level interactions.

An explicitly hindsight diagnostic on the exposed validation rows motivated
the test but is not evidence for it. Among rows whose exact command appeared in
another validation task, leave-one-task-out exact-command modes exceeded
Current by 14.076, 7.764, 12.389, and 5.521 percentage points for latency, CPU,
RSS, and Disk. Those gains apply only to the covered subsets and use tasks that
may occur later in deployment order. A name-free parsed command-shape oracle
changed full validation accuracy by +4.023, +3.869, +5.842, and +1.687 points.
These diagnostics make a causal replay worth seconds; they do not authorize a
final claim.

### 7.2 Frozen candidate and ablations

Use the committed 80 fit-task command rows as initial evidence and score the 50
development-exposed validation tasks in retained order. All commands in one
task are predicted before any label from that task becomes visible. At task
settlement, eligible complete-command labels enter two empirical memories for
each target:

1. exact command bytes; and
2. the existing `command_shape` representation: parsed executable basenames,
   option names, positional-slot and shell-structure counts, with no positional
   literals, task ID, output, or telemetry.

The primary `command-outcome-memory-v1` hierarchy uses exact-command evidence,
then command-shape evidence, then the row's unchanged Current PMF. Any nonempty
empirical evidence is used; there is no support threshold, smoothing,
shrinkage, or learned parameter. The empirical class histogram is the PMF and
the lower-index class wins a tie. Record exact-only and shape-only ablations
under the same causal settlement rule. Evaluation rows, labels, availability,
and order must be identical across arms.

This is an evaluation-only mechanism test. It does not change
`ClauseResourceKB`, telemetry promotion, or runtime storage. It may use aggregate
labels already retained by the evaluator only after the entire source task has
settled successfully.

### 7.3 Gate

The primary arm is GO only if all are true:

1. exact accuracy improves by at least 5.0 percentage points for each of
   latency, CPU, RSS, and Disk;
2. no target's severe-underprediction rate increases;
3. helpful changes exceed harmful changes and helpful changes cover at least
   ten validation tasks; and
4. the two ablations use byte-identical row IDs, labels, and Current PMFs.

A miss closes causal full-command outcome memory without a support threshold,
different key, target-specific arbitration, or replay reorder. It cannot open
the final 50 tasks. Runtime and result generation are expected to take under
ten seconds and require no container or agent call.

The frozen replay returned NO-GO on 1,044 validation commands:

| Target | Current | Exact only | Exact + shape | Primary delta |
|---|---:|---:|---:|---:|
| Latency | 74.904% | 77.969% | 77.778% | +2.874 pp |
| CPU | 82.887% | 84.970% | 85.119% | +2.232 pp |
| RSS | 80.978% | 84.783% | 84.918% | +3.940 pp |
| Disk | 82.044% | 83.829% | 83.234% | +1.190 pp |

The primary arm made 134 helpful and 48 harmful target changes across 35
helpful tasks, with no severe-underprediction regression, but no target reached
five points. Scoring and causal updates took 1.159 seconds.

The mechanism split is informative. Exact-command evidence covered 363/1,044
latency rows, 335/672 CPU rows, 359/736 RSS rows, and 349/1,008 Disk rows. Within
those covered rows it improved Current by 8.82, 4.18, 7.80, and 5.16 points.
Rows with neither exact nor shape evidence remained at only 71.52%, 72.82%,
72.14%, and 81.15% Current accuracy. The shape fallback increased CPU/RSS
slightly but reduced latency by 2.86 points and Disk by 8.57 points on its own
covered rows.

Observed: complete-command outcomes are a better unit than independent clause
composition when the exact invocation recurs, but exact recurrence covers too
little of the stream. Inference: the remaining problem is generalization to a
previously unseen requested work unit, plus physical state for Disk. Another
coarser command key would repeat the harmful shape backoff rather than add a
new signal. Final remains closed.

Artifacts:

- `analysis/results/tool-resource-5-3-3-3-20260804/sqlglot50-command-outcome-memory-v1/result.json`
- `analysis/results/tool-resource-5-3-3-3-20260804/sqlglot50-command-outcome-memory-v1/rows.jsonl`

## 8. Frozen development test: pytest target overlap

### 8.1 Question and pre-outcome coverage

The full-command result shows that aggregate outcomes transfer when an exact
invocation recurs, while the name-free shape is too coarse. The smallest
intermediate representation already has a physical meaning for pytest: the set
of requested test files and node IDs. This test asks whether overlapping test
targets extend exact-command coverage without pooling unrelated work.

A label-free scan of the already-exposed validation commands found 150
single-clause invocations accepted by the existing `parse_pytest`
grammar. Seventy-one were not exact complete-command matches at their causal
query time; 52 of those shared at least one target unit with an earlier settled
fit or validation task. No target label or candidate prediction was used for
this count.

### 8.2 Frozen candidate and ablations

Use the same 80 initial fit tasks and causal 50-task validation order as Section
7. Start from the Section 7 exact-complete-command arm. When one target has no
exact-command evidence, the primary `pytest-target-overlap-v1` arm may replace
Current only for a single executable clause accepted by `parse_pytest` and with
at least one explicit target.

Target units are deterministic:

- a file argument contributes `file:<normalized path>`;
- a node ID contributes both its file unit and its complete node ID; and
- a directory contributes `directory:<normalized path>`.

Paths are normalized lexically by removing `.` components and trailing slashes;
they are not opened and names are not rewritten. Match only observations with
identical non-target pytest modifiers: maxfail, workers, distribution mode,
collect-only, last-failed, failed-first, stepwise, and `-k`/`-m` expression
shape. Direct and `python -m pytest` invocations share this semantic partition.

For every historical observation with positive target-unit Jaccard overlap,
weight its complete-command class label by that Jaccard value. Normalize the
weighted class totals into a PMF and use the lower class on ties. There is no
minimum similarity, support threshold, alpha, target-count multiplier, or
per-resource rule. Empty-target full suites, compounds, parser refusals, and
zero-overlap queries fall back to Current. As before, all rows in a task predict
before that task's labels enter either exact or overlap memory.

Record two causal ablations on identical rows:

1. exact complete-command memory only; and
2. exact memory followed by the prior name-free exact `PytestSignature`, which
   retains target shapes/counts but removes target identities.

### 8.3 Gate

The target-overlap arm is GO only if all are true:

1. latency, CPU, RSS, and Disk exact accuracy each improve by at least 5.0
   percentage points over Current;
2. no target's severe-underprediction rate increases;
3. helpful changes exceed harmful changes and cover at least ten validation
   tasks; and
4. all arms have byte-identical row IDs, labels, Current hard predictions, and
   Current PMFs.

A miss closes target-overlap without changing unit expansion, similarity,
partitions, or arbitration. It cannot open final. Runtime is expected below ten
seconds with no container, filesystem probe, dependency, or agent call.

The frozen replay returned NO-GO:

| Target | Current | Collapsed signature | Target overlap | Primary delta |
|---|---:|---:|---:|---:|
| Latency | 74.904% | 78.448% | 78.736% | +3.831 pp |
| CPU | 82.887% | 86.012% | 87.054% | +4.167 pp |
| RSS | 80.978% | 85.462% | 87.228% | +6.250 pp |
| Disk | 82.044% | 83.135% | 83.135% | +1.091 pp |

The primary arm made 188 helpful and 63 harmful changes across 44 helpful
tasks, with no severe-underprediction regression. Runtime was 1.082 seconds.
The overlap carrier itself covered only 52 latency/RSS and 48 CPU/Disk rows,
but there it improved Current by 15.38, 29.17, and 34.62 points for latency,
CPU, and RSS. On the same Disk carrier it reduced accuracy by 14.58 points.

Observed: explicit pytest work units separate compute and memory load far more
cleanly than the name-free signature, but target identity does not identify
physical I/O state. Inference: semantic work-unit adapters may form the
CPU/RSS/latency half of a predictor, while Disk needs a separate state signal.
This exact pytest arm is closed by the all-target gate and cannot be repaired or
selectively enabled after seeing this result. A new combined method, if any,
must be separately frozen and must leave this NO-GO intact.

Artifacts:

- `analysis/results/tool-resource-5-3-3-3-20260804/sqlglot50-pytest-target-overlap-v1/result.json`
- `analysis/results/tool-resource-5-3-3-3-20260804/sqlglot50-pytest-target-overlap-v1/rows.jsonl`

## 9. Frozen component test: semantic requested-work units

### 9.1 Scope and pre-outcome coverage

The canonical targets are independent. Section 8 showed that explicit pytest
targets predict latency/CPU/RSS but anti-predict physical Disk on their carrier.
This phase therefore tests only the requested-work component: latency, CPU, and
RSS may change, while Disk hard predictions and PMFs must remain bit-identical
to Current. Passing would not complete the four-target objective or open final;
it would retain one half of a later combined predictor.

Keep the exact-complete-command and pytest-target hierarchy from Section 8 and
add the only other existing semantic work parser, `parse_pip_install`. A
label-free causal scan found 62 accepted single-clause pip commands in
validation, 23 without exact complete-command evidence, and 20 of those with a
positive package-name overlap to earlier fit or settled validation tasks.

### 9.2 Frozen candidate

For latency, CPU, and RSS, `semantic-work-units-v1` uses:

1. exact complete-command outcome evidence;
2. pytest target-unit Jaccard from Section 8;
3. pip canonical-package-name Jaccard; then
4. unchanged Current.

The pytest and pip parsers are fail-closed and apply only to single-clause
commands, so their work-unit arms cannot conflict. Pip observations match only
the same interpreter basename, direct versus module invocation, and frozen
flags. Versions do not become units: each canonical package name appears once,
and every positive-overlap historical complete-command label is weighted by
package-set Jaccard. Normalize weighted class totals to a PMF and use the lower
class on ties. There is no state inference, similarity threshold, alpha,
package-specific rule, or target-specific tuning among latency/CPU/RSS.

All validation commands in one task predict before that task's aggregate labels
enter exact or semantic memory. Record the Section 8 exact-plus-pytest arm as an
ablation on identical rows. Copy Disk directly from Current for both arms.

### 9.3 Component gate

This component is GO only if all are true:

1. latency, CPU, and RSS exact accuracy each improve by at least 5.0 percentage
   points over Current;
2. none of their severe-underprediction rates increases;
3. helpful changed predictions exceed harmful changes across those targets and
   helpful changes cover at least ten validation tasks; and
4. Disk hard predictions and PMFs are bit-identical to Current for every row.

A miss closes this exact work-unit combination without modifying either parser,
overlap, weighting, or target routing. A pass only authorizes retaining the
component while a separately preregistered physical Disk mechanism is tested.
Final remains closed in both cases. Expected runtime is below ten seconds with
no container, dependency, filesystem probe, or agent call.

### 9.4 Result and decision

The frozen replay returned component NO-GO:

| Target | Current | Pytest-only ablation | Semantic work units | Primary delta |
|---|---:|---:|---:|---:|
| Latency | 74.904% | 78.736% | 78.927% | +4.023 pp |
| CPU | 82.887% | 87.054% | 87.054% | +4.167 pp |
| RSS | 80.978% | 87.228% | 87.228% | +6.250 pp |
| Disk | 82.044% | 82.044% | 82.044% | +0.000 pp |

Across the three changed targets, the primary made 154 helpful and 38 harmful
changes over 40 helpful tasks; severe underprediction did not regress. Disk
hard predictions and PMFs were bit-identical to Current. Scoring and causal
updates took 0.602 seconds for 1,044 validation rows.

Observed: 20 pip rows had positive causal package overlap. Relative to the
pytest-only ablation, pip changed four latency decisions (three helpful, one
harmful) and no CPU or RSS hard decisions. All 20 CPU labels were Low; 17 of 20
RSS labels were Low, so Current already selected the same dominant classes.
The package overlap added only +0.192 latency points and no CPU/RSS points.

Inference: semantic package identity is not the missing signal for this gate.
The remaining errors require either a broader source of requested-work evidence
or inference-time physical/environment state; tuning the exposed pip arm would
not answer either question. This exact hierarchy is closed and does not
authorize final access.

Artifacts:

- `analysis/results/tool-resource-5-3-3-3-20260804/sqlglot50-semantic-work-units-v1/result.json`
- `analysis/results/tool-resource-5-3-3-3-20260804/sqlglot50-semantic-work-units-v1/rows.jsonl`

## 10. Frozen component test: early physical lower bounds

### 10.1 Question and pre-outcome coverage

The remaining latency/CPU errors often alternate across identical commands, so
another history key cannot establish the current execution mode. This phase
asks a narrower causal question: after the first complete telemetry interval,
can facts that the current execution has already established safely correct an
underprediction?

Use the existing validation traces and the frozen Section 9 hierarchy as the
pre-decision arm. A label-free scan found valid 0.5-second action timelines for
all 1,044 rows. At the existing Phase-1 effective decision time, 416 commands
were still running; 225 first-window CPU rates exceeded two cores and 53
exceeded four cores. No resource label or early-arm prediction was read in that
scan.

### 10.2 Frozen candidate

For each command, take the first timeline sample whose `dt_s >= 0.5`. The
effective decision offset is its endpoint plus the already measured 0.050 s
availability pad and 0.09132007875 s actuation p95. Apply the candidate only if
the command has not finished by that offset. Online, this is the absence of a
finish event; replay may use `ts_end` only to reproduce that availability.

Define two observed lower-bound buckets:

- latency: the bucket containing the effective elapsed time;
- CPU: the bucket containing `min(cpu_core_s / dt_s, cpu_quota_cores)` for the
  first complete sample.

Starting from `semantic-work-units-v1`, change a target only when its current
hard prediction is unavailable or below the observed lower-bound bucket. Zero
PMF mass below the lower bound and renormalize; if no mass remains, use a point
mass at the lower-bound bucket. Leave the prediction unchanged when it is
already at or above the bound. RSS remains the Section 9 prediction. Disk hard
prediction and PMF remain bit-identical to Current.

This arm has no learned parameter, tool parser, command family, outcome, future
sample, final peak, or final duration. Record the Section 9 arm as the ablation
on identical rows. A replay validity check requires every applied latency/CPU
lower bound to be no higher than its final measured bucket; a violation means
the retained timeline and canonical target do not have compatible physical
scope and invalidates the arm rather than permitting a repair.

### 10.3 Component gate

The component is GO only if:

1. latency, CPU, and RSS each improve by at least 5.0 percentage points over
   Current;
2. none of their severe-underprediction rates increases;
3. helpful changed predictions exceed harmful changes and cover at least ten
   validation tasks;
4. no applied lower bound exceeds the final target bucket; and
5. Disk hard predictions and PMFs are bit-identical to Current.

A miss closes this exact early-bound arm without changing the decision time,
window, projection, or target routing. A pass retains a latency/CPU/RSS
component only; it cannot open final until a separately frozen Disk mechanism
passes. The replay uses retained traces and is expected below ten seconds.

### 10.4 Result and decision

The frozen replay returned component NO-GO:

| Target | Current | Semantic-only ablation | Early bounds | Primary delta |
|---|---:|---:|---:|---:|
| Latency | 74.904% | 78.927% | 80.747% | +5.843 pp |
| CPU | 82.887% | 87.054% | 78.423% | -4.464 pp |
| RSS | 80.978% | 87.228% | 87.228% | +6.250 pp |
| Disk | 82.044% | 82.044% | 82.044% | +0.000 pp |

Of 416 commands still running at the decision time, the arm changed 23 latency
and 177 CPU projections. The elapsed-time correction passed its physical check
and moved latency beyond the five-point target. CPU failed the physical check:
96 first-window cgroup buckets exceeded the final clause-owned eBPF bucket,
across 40 tasks. Sixty-nine were Medium-prefix versus Low truth, 24 High-prefix
versus Low truth, and three High-prefix versus Medium truth. CPU accuracy fell
despite a lower severe-underprediction rate. Disk remained bit-identical.

Observed: the violations span Python, pytest, pip, apt, and compound commands;
they are not one parser or one task. The action timeline measures all CPU in the
tool container cgroup, while the canonical label measures only the current
clause's owned exec lineage on aligned windows. The former is therefore a
different scope, not a conservative prefix of the latter.

Inference: elapsed wall time is a usable early lower bound, but the retained
cgroup timeline cannot stand in for canonical CPU. This exact arm is closed;
thresholding or selecting only the favorable commands after this result would
hide the scope error. A future early CPU method requires prefix counters from
the same canonical eBPF lineage. No such prefix profile is retained in these
traces, so this result does not authorize final access.

Artifacts:

- `analysis/results/tool-resource-5-3-3-3-20260804/sqlglot50-early-physical-bounds-v1/result.json`
- `analysis/results/tool-resource-5-3-3-3-20260804/sqlglot50-early-physical-bounds-v1/rows.jsonl`
