# Task-Local Resource Prediction Research Record

**Effective:** 2026-08-05

**Status:** the offline-agent and deterministic task-phase directions are
closed. The 50-task validation partition is development-exposed; the 50-task
final partition remains unread and unauthorized. Existing pre-command traces
support a task-local calibrator for latency, CPU, and RSS, but not Disk. The
next step is cache-state probe feasibility, not an authorized collection.

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

A fixed 25-task fit / 25-task test diagnostic on the now-exposed validation
partition used Current PMFs, deterministic pip/pytest work atoms, generic
executable/subcommand/flag and compound-action features, and fixed L2 logistic
regression. Adding only structured command features improved latency, CPU, and
RSS by 3.839, 7.121, and 11.173 points but harmed Disk by 0.794. Adding causal
task-local prior exact/semantic/tool buckets raised those changes to 6.334,
8.978, 12.849, and 0.000 points. A single default Random Forest check raised
Disk by only 1.587 points. No parameter or threshold sweep was run.

This is development-exposed diagnostic evidence, not a confirmation result.
It supports a simple architecture for the first three targets: Current is the
global prior, while an ephemeral task-local calibrator consumes only completed
earlier commands and is never published into the cross-task KB. It does not
support a static Disk claim.

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

### 5.3 Frozen next mechanism test: file footprint x residency

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
