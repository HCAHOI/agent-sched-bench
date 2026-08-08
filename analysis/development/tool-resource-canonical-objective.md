# Tool-Resource Prediction — Canonical Objective and Lock

**Effective:** 2026-08-08
**Scope:** current contract and decisions only

This file is the authority for tool-resource targets, evaluation semantics,
evidence boundaries, and scheduler-facing decisions. It intentionally omits
superseded protocols and chronological experiment narration. Git and frozen
artifacts retain that history; `clause-interaction-kb-plan.md` contains the
long-form research record.

## 1. Current decisions

| Area | Current decision |
|---|---|
| Prediction | Keep the target-specific development SOTA below. It is exposed development evidence, not confirmation. |
| CPU execution | **KEEP equal-share burstable execution as the current development CPU-sharing baseline.** |
| Saturated quartet result | Frozen claim verdict is NO-GO because 8/12, not 9/12, groups improved. This is not a module-retirement decision. |
| Rolling queue result | **Framework-invalid and inconclusive.** The artifact mechanically recorded NO-GO, but outcome drift and fixed arm order prevent a method verdict. |
| CPU scheduling | Every subsequent CPU scheduler must compare against equal-share burstable execution unless fresh evidence supersedes it. |
| KB representation | Raw exact/prefix/binary `ClauseResourceKB` remains Current. Trie/lattice/semantic-key replacement is closed as a contribution. |
| Offline agent | Closed. More prompts, critics, DSLs, or generated adapters are not authorized. |
| KV scheduling | Closed under the current CacheWise/C100 simulator and action model. |
| Peak-class admission | Closed. Peak CPU classes are the wrong target for sustaining command throughput. |
| Runtime integration | No predictor or scheduler candidate currently authorizes production integration. |
| Next research step | A future CPU-sharing claim requires exact source tool calls, outcome-quality accounting, counterbalanced arm order, and complete lifecycle timing. |

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

## 4. Current predictor

`ClauseResourceKB` is Current. It uses raw exact argv, argv-prefix, and
binary/global backoff with frozen public evidence and causal repository-local
task-final updates. The trie is an implementation detail, not the research
contribution.

The current development reference uses one common `BeginCall` decision time
with target-specific heads:

| Target | Eligible | Current | Development SOTA | Delta | Selected head |
|---|---:|---:|---:|---:|---|
| Latency | 1,044 | 74.904% | 79.023% | +4.119 pp | semantic work units + monotone test phase |
| CPU | 672 | 82.887% | 87.500% | +4.613 pp | semantic work units + monotone test phase |
| RSS | 736 | 80.978% | 87.500% | +6.522 pp | semantic work units + monotone test phase |
| Disk | 1,008 | 82.044% | 83.829% | +1.786 pp | exact complete command |

Equal-weight four-target accuracy is 84.463% versus 80.203% for Current. This
configuration was selected post-hoc on exposed validation data. Only RSS clears
the existing five-point per-target gate; it does not authorize confirmation or
runtime integration. Artifact:
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
| CacheWise/C100 KV victim selection | SOTA improved C100 recomputation 1.481%; block-aware hindsight upper bound improved 9.620% | Closed for current simulator/action model |
| CPU+RSS admission oracle | Hindsight reservations reduced mean makespan 10.868% | Action space exists |
| Peak-class predictor admission | SOTA was 9.630% slower than Current and both created many unverifiable capacity exposures | Direct hard-class admission closed |
| Physical under-reservation | 2 cores were 3.928x slower than 8; 2 GiB `memory.high` did not finish within 3,600 s versus 18.493 s baseline | Under-reservation cost is real and must be charged |
| CPU-throughput target oracle | Throughput class reduced peak-class makespan 19.03% | Target has hindsight headroom |
| Static two-core request | 8.452% mean regret versus throughput oracle | Prediction or feedback still has headroom |
| Ideal burstable two-core model | 8.881% faster than hard two-core; every seed improved | Authorized physical test |
| Two-task physical paired replay | 14.490% mean improvement; 13/13 pairs improved; CI [9.049%, 22.832%] | Positive exposed mechanism evidence |
| Four-task saturated replay | 12.966% mean improvement; CI [2.947%, 25.527%]; 8/12 groups improved | Frozen claim NO-GO; **module decision KEEP** |
| Rolling four-task queue | Burstable full wall time was 8,952.374 s versus 8,688.964 s hard-two, 3.031% slower | Framework-invalid; neither confirms nor rejects burstable |

### Current CPU-sharing baseline

The saturated replay admitted four tasks with two-core guarantees, filling the
eight-core host. Hard-two enforced each two-core quota. Burstable-two retained
the same cpuset and equal shares but removed the hard quota.

Burstable-two reduced mean task completion from 572.915 to 517.502 seconds
and p95 from 1,868.606 to 1,116.854 seconds. Two of 48 tasks slowed by more than
10%; none slowed by more than 25%. All 96 task-arm executions preserved actions
and CPU controls with zero source-success timeout or OOM; a post-run audit found
none of the 96 containers remaining.

The frozen direction gate missed by one group, so the artifact remains NO-GO
for a claim of consistent group-level improvement. The large positive effects
and small negative effects nevertheless justify retaining equal-share
burstable execution as the development baseline. It is not yet a production
default and has no temporal-generalization claim.

Post-hoc mechanism attribution found that group improvement tracked the hard-arm
tail and idle-guarantee fractions (Spearman 0.734 for each). The two dominant
groups kept nearly the same total CPU work across arms while long package builds
used up to eight cores and shortened by 1,207 and 1,075 seconds. Current
`BeginCall` predictions missed both carriers, and the original development set
contained no `--force-reinstall` example, so prediction-guided task grouping is
not authorized.

The completed rolling gate used the same 48 exposed tasks, concurrency four,
immediate refill, and full queue wall time. Burstable was 3.031% slower despite
reducing mean task runtime by 6.490% and p95 by 20.976%; 34/48 individual tasks
were faster. The fixed burstable-first order also charged 1,196.7 seconds more
aggregate recorded container-startup time to that arm. More importantly, replay
fixed the recorded actions but physically re-executed commands: 7/1,765 tool
calls across three tasks changed coarse terminal class between source or arms,
including a 600-second source/burstable timeout that became a 57.8-second
nonzero exit under hard-two. Four calls across two tasks differed directly
between arms.

The original `result.json` therefore retains its mechanically produced
`status: no_go`, but the 2026-08-08 validity amendment supersedes that as a
scientific interpretation. The experiment is inconclusive because its
`all_validity_checks_passed` field omitted material outcome and lifecycle
validity dimensions. It is not negative evidence about burstable CPU sharing.

A post-hoc diagnostic retained only the 36 source-clean tasks with identical
source/burstable/hard terminal classes. Reconstructing a fixed-duration
four-worker queue made burstable 9.716% faster for execution alone and 4.524%
faster after adding recorded container-startup time. The latter omits artifact
restore, finalization, container stop, and image cleanup. This is neither a
physical rerun nor a confirmatory subset and supplies no method verdict. It
instead diagnoses why the original gate is not interpretable. The next
protocol must preserve tool calls exactly, treat outcome differences as quality
rather than filter them away, counterbalance arm order, and separate replay,
startup, and the remaining lifecycle costs.

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
- The nominal final50 resource labels and traces were not scored, but collector
  metadata for two final tasks was accidentally read. That partition is no
  longer untouched confirmation evidence and remains unauthorized absent an
  explicit new protocol.
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

- Current predictor: `sqlglot50-multitarget-sota-v1/result.json`
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
Current = unchanged raw exact/prefix/binary control.
Equal-share burstable CPU execution = retained development scheduling baseline.
Protocol NO-GO != permission to delete a retained baseline.
No result-dependent tuning, hindsight state, or dataset-specific outcome rule.
No new collection, runtime integration, or scheduler claim without a separate
approved protocol.
```
