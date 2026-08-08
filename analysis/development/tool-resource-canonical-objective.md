# Tool-Resource Prediction — Canonical Objective and Lock

**Effective:** 2026-08-08
**Scope:** current contract and decisions only

This file is the authority for tool-resource targets, evaluation semantics,
evidence boundaries, and scheduler-facing decisions. It intentionally omits
superseded protocols and chronological experiment narration. Git and frozen
artifacts retain that history.

## 1. Current decisions

| Area | Current decision |
|---|---|
| Prediction | Keep the target-specific Task-Aware Command Predictor below as the selected development candidate. It is exposed evidence, not confirmation or deployed code. |
| Runtime feedback | **KEEP causal command-level eBPF feedback as the current promising resource-action candidate.** It is counterfactual development evidence, not an integrated scheduler. |
| CPU execution | Keep equal-share burstable execution as the simple physical CPU-sharing baseline, not as a proven method. |
| Saturated quartet result | Frozen claim verdict is NO-GO because 8/12, not 9/12, groups improved. This is not a module-retirement decision. |
| Fresh counterbalanced queue | **Inconclusive because 7/1,818 paired tool calls changed terminal class.** Timing-only improvement was 0.929% with 2/4 queues improving, below the frozen gate even before the quality failure. |
| Prediction contribution to feedback | Not established. Task-Aware seeding did not improve the frozen reservation/service operating point over feedback alone. |
| CPU scheduling | A physical reservation-aware admission test must compare feedback against fixed-eight and equal-share burstable controls. |
| KB representation | Raw exact/prefix/binary `ClauseResourceKB` is the implemented Clause-KB baseline. Trie/lattice/semantic-key replacement is closed as a contribution. |
| Offline agent | Closed. More prompts, critics, DSLs, or generated adapters are not authorized. |
| KV scheduling | Closed under the current CacheWise/C100 simulator and action model. |
| Peak-class admission | Closed. Peak CPU classes are the wrong target for sustaining command throughput. |
| Runtime integration | No predictor, feedback controller, or scheduler is currently integrated or authorized for production. |
| Next research step | Test whether feedback-released reservation changes physical admission and queue completion on development-exposed workloads; consume fresh evidence only after that mechanism works. |

`status: no_go` in a result artifact answers only that artifact's frozen claim
gate. It never authorizes deleting an implementation that this table marks
KEEP.

## 2. Prediction objective

The evaluation unit is one eligible `exec` command. Clauses are internal
evidence; they are never scored as separate rows.

### Latency

Predict a normalized PMF over:

```text
[0, 500] ms
(500, 2000] ms
(2000, 8000] ms
(8000, 30000] ms
(30000, +inf) ms
```

The hard prediction is the highest-probability bucket; ties select the shorter
bucket. Primary metric is exact five-class command accuracy. An unavailable
prediction counts as incorrect. Also report eligible count, label/prediction
counts, confusion matrix, constant-majority accuracy, within-one-bucket
accuracy, and severe underprediction rate (truth at least two buckets above
prediction).

### CPU, RSS, and Disk

Predict each target independently as Low, Medium, or High. Boundaries belong to
the lower bucket:

```text
CPU  = [0, 2], (2, 4], (4, +inf) peak cores
RSS  = [0, 500], (500, 2000], (2000, +inf) decimal MB
Disk = [0, 1048576], (1048576, 104857600], (104857600, +inf) bytes
```

An observation explicitly marked null by policy is imputed Low only when
command latency is below 500 ms; other nulls are unavailable. Ties select the
lower bucket. Primary metric is exact three-class command accuracy. Also report
eligible count, label/prediction counts, confusion matrix, majority and
constant-Low accuracy, within-one-bucket accuracy, and severe underprediction.

Old 3/2/2/2 targets, nine-bin latency, balanced accuracy, Brier/NLL, q-error,
and hand-selected subsets cannot select a candidate.

## 3. Causal and physical contract

Offline replay and online serving share this interface:

```python
predict(repo, command, parsed_clauses, ts_start) -> predictions + provenance
observe(completed_clause_observations) -> None
```

- An observation is visible only when `observation.ts_end < query.ts_start`.
- A task becomes learnable only after successful whole-task finalization.
- The current task and failed or unfinalized tasks are invisible to cross-task
  learning.
- Frozen cross-repository evidence never updates during evaluation.
- Every arm uses identical command IDs, labels, availability, order, and shell
  structure.
- Invalid, ambiguous, lossy, or cleanup-invalid telemetry is withheld;
  independently valid siblings may remain eligible.
- Static prediction cannot use the current command's output or telemetry.
- Early-execution prediction may use only samples available by its frozen
  decision timestamp. Hindsight state is diagnostic only.

Compound commands compose physical values, never bucket IDs:

| Shell relation | Latency | CPU | RSS | Disk |
|---|---|---|---|---|
| Sequential | sum | max | max | sum |
| Concurrent pipeline | max | sum | sum | sum |

Unsupported or ambiguous structure remains unavailable. Boolean OR is never a
composition rule.

## 4. Prediction architectures

The **Clause-KB baseline** is the implemented `ClauseResourceKB`. It uses raw
exact argv, argv-prefix, and binary/global backoff with frozen public evidence
and causal repository-local task-final updates. The trie is an implementation
detail, not the research contribution.

The **Task-Aware Command Predictor** is the selected meeting-facing development
candidate. It uses one common `BeginCall` decision time and target-specific
heads. Majority is the constant fit-set mode for each target.

| Target | Eligible | Majority | Clause-KB baseline | Task-Aware Predictor | Gain vs Clause-KB | Selected head |
|---|---:|---:|---:|---:|---:|---|
| Latency | 1,044 | 58.429% | 74.904% | 79.023% | +4.119 pp | command-work match + repeated-test phase |
| CPU | 672 | 82.440% | 82.887% | 87.500% | +4.613 pp | command-work match + repeated-test phase |
| RSS | 736 | 81.250% | 80.978% | 87.500% | +6.522 pp | command-work match + repeated-test phase |
| Disk | 1,008 | 61.012% | 82.044% | 83.829% | +1.786 pp | exact command history |

Equal-weight four-target accuracy is 70.783% for Majority, 80.203% for the
Clause-KB baseline, and 84.463% for the Task-Aware Command Predictor. The
candidate first matches complete commands, then equivalent requested pip or
pytest work, and otherwise falls back to Clause-KB. For latency, CPU, and RSS,
a third-or-later repeated full-suite test may only raise the predicted bucket;
Disk uses exact command history. This configuration was selected post-hoc on
exposed validation data. Only RSS clears the existing five-point per-target
gate; it does not authorize confirmation or runtime integration. Artifact:
`sqlglot50-multitarget-sota-v1/` under the result root in Section 8.

Runtime ownership remains fixed:

- `resource-agentd`: parsing, prediction, causal state, persistence, and
  orchestration; always unprivileged.
- `telemetryd`: privileged eBPF/cgroup collection and finalized observations.
- collectors, replay, and schedulers: thin unprivileged clients.

The detailed privilege, IPC, and lifecycle contract lives in
`tool-resource-service-architecture.md`.

## 5. Scheduler evidence and decisions

All rows below are development-exposed.

| Question | Result | Decision |
|---|---|---|
| CacheWise/C100 KV victim selection | Task-Aware Command Predictor improved C100 recomputation 1.481%; block-aware hindsight upper bound improved 9.620% | Closed for current simulator/action model |
| CPU+RSS admission oracle | Hindsight reservations reduced mean makespan 10.868% | Action space exists |
| Peak-class predictor admission | Task-Aware Command Predictor was 9.630% slower than Clause-KB and both created many unverifiable capacity exposures | Direct hard-class admission closed |
| Physical under-reservation | 2 cores were 3.928x slower than 8; 2 GiB `memory.high` did not finish within 3,600 s versus 18.493 s baseline | Under-reservation cost is real and must be charged |
| CPU-throughput target oracle | Throughput class reduced peak-class makespan 19.03% | Target has hindsight headroom |
| Static two-core request | 8.452% mean regret versus throughput oracle | Prediction or feedback still has headroom |
| Ideal burstable two-core model | 8.881% faster than hard two-core; every seed improved | Authorized physical test |
| Two-task physical paired replay | 14.490% mean improvement; 13/13 pairs improved; CI [9.049%, 22.832%] | Positive exposed mechanism evidence |
| Four-task saturated replay | 12.966% mean improvement; CI [2.947%, 25.527%]; 8/12 groups improved | Frozen claim NO-GO; **module decision KEEP** |
| Rolling four-task queue | Burstable full wall time was 8,952.374 s versus 8,688.964 s hard-two, 3.031% slower | Framework-invalid; neither confirms nor rejects burstable |
| Exact-tool ABBA diagnostic | Four SQLGlot 3734 runs preserved all 67 actions and all 33 tool-call terminal classes; descriptive burstable wall time was 4.697% lower | Framework repair works on this task; no method verdict |
| Fresh 48-task counterbalanced queue | 7/1,818 paired calls changed terminal class; timing-only mean was +0.929%, with 2/4 queues improving | Inconclusive on quality; timing-only result also misses the 5% and 3/4 gate |
| Prediction-seeded feedback | Feedback alone reduced reservation 37.076% at 1.032% service inflation; Task-Aware seeding changed this to 37.635% and 1.274% | Feedback promising; incremental prediction gate failed |
| Cross-repository feedback | On 259 SWE100/277 tasks, reservation fell 45.010% at 1.543% service inflation; task-bootstrap intervals were [38.286%, 50.959%] and [1.256%, 1.820%] | Development GO for causal feedback; physical scheduling utility untested |
| Prediction-weighted CPU shares | Task-Aware reduced mean completion by 0.0056%; a true-demand oracle reduced it by only 0.0527% | Closed: recorded-duration floor leaves no useful share-allocation headroom |
| Ready-queue priority | Exact-duration aging oracle reduced mean completion 3.991% with 0.301% makespan cost | Below the frozen 5% gate; retain only as a diagnosed tradeoff |

### Current CPU action frontier

Equal-share burstable execution remains the simplest physical baseline: tasks
retain equal two-core guarantees but may borrow idle CPU. The two-task result
was positive; the saturated four-task claim missed its direction gate by one
group. The older rolling result was framework-invalid because outcome and
lifecycle drift were not part of its validity check.

The repaired 48-task experiment used four disjoint 12-task queues, immediate
refill, counterbalanced arm order, exact source tool calls, complete lifecycle
timing, and explicit terminal-class quality checks. Seven of 1,818 paired calls
still changed terminal class across arms, involving five tasks, so timing was
ineligible. Descriptively, burstable changed queue wall time by -16.575%,
+18.855%, -5.281%, and +6.716%: mean +0.929%, median +0.717%, and only two
queues improved. The first two burstable arms also overlapped unquantified
offline host work; the generated result and run note forbid post-hoc correction.
Thus burstable remains a control, not a supported scheduling claim.

The strongest current action is instead causal command-level feedback. Each
command begins at eight cores; after the first complete 0.5-second eBPF/cgroup
sample, demand maps to a 2, 4, or 8-core page and can update causally. On the
SQLGlot development set, feedback alone reduced reserved CPU-core-seconds by
37.076% at 1.032% service inflation. Seeding it with Clause-KB or Task-Aware
predictions did not pass the incremental gate.

Without changing the policy, the SWE100/277 generality run covered 259 valid
tasks, 205 repositories, and 8,237 exec commands. Feedback reduced reservation
45.010% at 1.543% service inflation; both corpora independently passed the
frozen gate. Of 258 tasks with exec work, 244 reduced reservation. This is a
counterfactual per-command result, not a concurrent physical scheduler: it
shows broad signal and a viable operating point, but not that released capacity
improves queue completion. The next experiment must connect those causal page
updates to actual admission and charge controller/collector cost.

## 6. Closed directions

- **Lookup structure alone:** trie, lattice, generic argv, pip/pytest semantic
  keys, and command-state variants changed too few action-relevant decisions.
- **Offline agent:** generated Python, relational spans, bounded regex, and a
  typed catalog found no resource-separating state. The final finite-choice arm
  removed arbitrary code and still returned no useful contrast.
- **Task-local last-value state:** repeated command load alternates; a causal
  overlay creates one-command lag rather than stable state.
- **Continuous latency survival:** most apparent gain came from the elapsed-time
  physical floor; command-equal accuracy gain was only 0.229 points.
- **Peak CPU as admission demand:** instantaneous peak does not equal the CPU
  quota required to preserve throughput.
- **KV victim selection:** C100 already removes most LRU recomputation, leaving
  insufficient predictor headroom in the current model.

Closed means do not tune or retry on the exposed SQLGlot tasks. A genuinely new
signal, action, or fresh workload may motivate a separately frozen protocol.

## 7. Evidence boundary

- The original SQLGlot100, the 50-task validation partition, and every result
  summarized here are development-exposed.
- The nominal SQLGlot final partition is now fully development-exposed: 48
  tasks were consumed by the counterbalanced physical replay, and two older
  tasks had already appeared in result artifacts.
- SWE100 and SWE277 are development-exposed. Their feedback result tests
  cross-repository breadth but is not confirmation.
- Existing SQLGlot tasks end at 2025-04-25; no ready newer public SQLGlot image
  cohort was found.
- Confirmation requires genuinely fresh tasks, a new time period, or another
  suitable repository. Criteria must be frozen before outcome access.
- No result-dependent task selection, threshold tuning, package/test-name
  outcome rule, or hindsight state is allowed.
- New collection or a run expected to exceed 30 minutes requires explicit
  approval after a smoke, resource estimate, and decision gate.

## 8. Authoritative artifacts

Result root:
`analysis/results/tool-resource-5-3-3-3-20260804/`

- Clause-KB baseline and Task-Aware Command Predictor: `sqlglot50-multitarget-sota-v1/result.json`
- CPU+RSS oracle: `sqlglot100-resource-admission-oracle-v1/result.json`
- Predictor admission: `sqlglot50-resource-admission-predictors-v1/result.json`
- Under-reservation calibration: `resource-underreservation-calibration-v1/result.json`
- Long-memory calibration: `resource-underreservation-memory-completion-v1/result.json`
- Throughput oracle: `sqlglot50-cpu-throughput-oracle-v1/result.json`
- Two-core baseline: `sqlglot50-two-core-throughput-baseline-v1/result.json`
- Burstable model: `sqlglot50-burstable-two-core-fluid-v1/result.json`
- Two-task physical matrix: `sqlglot26-paired-cpu-borrowing-compatible-v2/result.json`
- Saturated four-task matrix: `sqlglot48-quartet-cpu-borrowing-contract-v2/result.json`
- Rolling queue: `sqlglot48-rolling-cpu-borrowing-contract-v1/result.json`
- Rolling outcome audit: `sqlglot48-rolling-cpu-borrowing-contract-v1/outcome-audit.json`
- Rolling validity amendment: `sqlglot48-rolling-cpu-borrowing-contract-v1/validity-amendment.json`
- Exact-tool repeatability diagnostic: `sqlglot3734-exact-tool-abba-diagnostic-v1/result.json`
- Fresh counterbalanced physical replay: `sqlglot-final48-counterbalanced-rolling-exact-v1/result.json`
- Physical replay interference record: `sqlglot-final48-counterbalanced-rolling-exact-v1/run-notes.json`
- Prediction-seeded CPU feedback: `sqlglot50-prediction-seeded-cpu-feedback-v1/result.json`
- Prediction-weighted CPU shares: `sqlglot50-prediction-weighted-cpu-shares-v1/result.json`
- SWE100/277 feedback generality: `swe100-277-cpu-feedback-generality-v1/result.json`

Task split authority:
`analysis/development/sqlglot-relational-task-split.json`.

## 9. Non-negotiable task contract

```text
Evaluation unit = eligible exec command; clauses are internal evidence.
Latency buckets = [0,500], (500,2000], (2000,8000], (8000,30000],
                  (30000,+inf) ms.
Primary latency metric = exact five-class accuracy on identical rows.
Resource buckets = CPU edges 2/4 cores; RSS edges 500/2000 MB;
                   Disk edges 1/100 MiB.
Unavailable hard predictions count as incorrect.
Short-null resource policy = Low only when explicitly marked and <500 ms.
Causal visibility = observation end before query start, after task settlement.
Compound commands = physical sequential/pipeline composition, never Boolean OR.
Clause-KB baseline = unchanged raw exact/prefix/binary control.
Task-Aware Command Predictor = selected development candidate, not deployed.
Causal eBPF feedback = promising resource-action candidate, not integrated.
Equal-share burstable CPU execution = retained physical control baseline.
Protocol NO-GO != permission to delete a retained baseline.
No result-dependent tuning, hindsight state, or dataset-specific outcome rule.
No new collection, runtime integration, or scheduler claim without a separate
approved protocol.
```
