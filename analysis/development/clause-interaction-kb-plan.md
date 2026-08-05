# Task-Local Resource Prediction Research Record

**Effective:** 2026-08-05

**Status:** the offline-agent direction is closed. The additional SQLGlot
validation and final partitions remain scientifically unread. The next proposed
experiment is a fresh replication of the already-fixed deterministic full-test
phase feature; this document does not authorize reserved-outcome access.

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

No agent arm opened either partition. Validation may be consumed by one newly
preregistered non-agent method. Final remains unavailable unless that exact
method passes its validation gate. IDs, process status, file counts, and
collection completion metadata may be inspected before then; commands, outputs,
labels, telemetry, and derived scores may not.

## 4. Frozen next experiment: replicate deterministic task phase

This is the shortest experiment that can change the next decision. It tests
whether the one positive development mechanism repeats on new tasks from the
same repository. It does not test a new KB or a learned semantic model.

The protocol below was frozen on 2026-08-05 before reading any command,
output, label, or telemetry from either reserved role. The additional run's
completion metadata may have been inspected, but neither role has been scored.

### 4.1 Fixed candidate and development fit

Reuse `full-test-third-or-later-v1` unchanged:

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

The development-only fit contains 44 third-or-later commands across 34 tasks.
Their latency labels are `{0: 3, 4: 41}`, CPU labels are `{0: 3, 2: 41}`, RSS
labels are `{0: 3, 2: 41}`, and Disk labels are `{0: 25, 1: 19}`. Thus the
frozen hard prediction is the highest bucket for latency, CPU, and RSS. Disk is
not fit or overridden. These counts are fit evidence, not a fresh result.

### 4.2 Smallest valid protocol

Before any reserved command or outcome is read, commit the evaluator and this
record. Then create and separately commit one development-fit artifact binding
the exact development fingerprint, public inputs, host commit, split manifest,
PMFs, coverage rule, and validation/final authorization. Validation cannot run
against an uncommitted artifact. The label-free validation coverage rule is:
among completed tasks with accepted workload output and valid telemetry,
third-or-later recognized full-suite commands must occur in at least three
validation tasks, the minimum number that could satisfy the task-level gain
gate. Otherwise stop without scoring labels.

On identical eligible command rows, validation passes only if:

- exact accuracy is no lower than Current for latency, CPU, RSS, or Disk;
- severe underprediction is no higher for any target;
- helpful target changes outnumber harmful changes; and
- helpful changes span at least three validation tasks.

Only validation GO authorizes the byte-identical final evaluation. Final uses
the same independent gate. The validation result and row sidecar must be
committed unchanged before final can run. No detector, phase boundary, target,
PMF, public evidence, Current snapshot, eligibility rule, or gate may change
after validation access. The fresh evaluator reuses the existing command
evaluator and freezes Current after the 100 development tasks; it performs no
within-role learning.

Amendment, 2026-08-05: the first validation invocation passed coverage and
loaded the frozen scoring path, then crashed before producing any metric or
result artifact because Current omits resource keys when its prediction is
unavailable while the shared scorer requires explicit `None`. The only visible
information was `KeyError: peak_cpu_cores`; no validation score, row, label, or
method comparison was inspected. The wrapper now canonicalizes omitted Current
resource predictions and PMFs to `None`. This does not change any available
prediction, fit PMF, detector, override, row eligibility, or gate. Because the
host commit changes, the repaired evaluator requires a new dev-only fit
artifact and independent review before retrying the unchanged validation role.

Second amendment, 2026-08-05: the unchanged retry again produced no result
artifact or metric. After explicit unavailable keys reached the shared scorer,
both Current and candidate had incomplete exact accuracy for at least one
target, and the scorer crashed on `None >= None`. The only newly visible
information was this exception. The shared gate now treats any incomplete
Current or candidate primary accuracy or severe-underprediction metric as
fail-closed NO-GO instead of comparing nulls. Numeric comparisons are unchanged.
This is the specified interpretation of “no lower/no higher”: unavailable
metrics cannot demonstrate either condition. A newly committed dev-only fit
artifact is again required before the unchanged validation retry.

## 5. Other angles, in priority order

1. **Pre-command workspace workset.** The strongest genuinely new hypothesis is
   that actual work depends on changed-file volume/type, selected-test count,
   dependency/cache inventory, and build/test graph impact. First audit whether
   development traces contain a complete causal snapshot; do not infer these
   features from future command output. If the snapshot is absent, quantify the
   exact fields and probe cost before requesting new collection.
2. **Mid-execution forecasting.** Early counters may predict the final bucket,
   but the existing EAR CPU-reservation consumer already has a 5.686% hindsight
   ceiling. Reopen only for a different named consumer whose action can still
   change after the observation point.
3. **Generic supervised history model.** Old cross-repository tabular results
   were mixed and used superseded targets. Do not train another broad model
   until either task phase replicates or workspace features show a measurable
   information gain; otherwise it obscures the mechanism and spends the final
   reserve on model selection.

Do not spend the untouched SQLGlot partitions to compare these ideas in
parallel. Run the deterministic phase replication first because it is already
fixed, cheap, and falsifiable; a validation NO-GO ends that branch before any
second method is designed against those outcomes.
