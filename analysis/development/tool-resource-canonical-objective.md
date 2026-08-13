# Tool-Resource Prediction — Current Objective and Decisions

**Effective:** 2026-08-13
**Scope:** current scientific contract, retained results, and stop conditions

This is the authority for tool-resource targets, evaluation semantics, evidence
boundaries, and current decisions. It is not an experiment diary. Completed
protocols live in their result artifacts and Git history; the runtime boundary
lives in `tool-resource-service-architecture.md`.

## 1. Current state

| Area | Decision | What the evidence supports |
|---|---|---|
| Prediction | **KEEP Task-Aware Command Predictor** as the selected development candidate. | On exposed SQLGlot50 it improves equal-weight four-target accuracy from 80.203% for Clause-KB to 84.463%. It is not confirmation or deployed code. |
| Baseline | **KEEP Clause-KB** as the normal learned baseline. | It is the implemented raw exact/prefix/binary `ClauseResourceKB`; trie is an implementation detail, not a contribution. |
| Runtime feedback | **KEEP causal command-level eBPF feedback** as a measured mechanism and control. | It reduced reserved CPU-core-seconds 45.010% at 1.543% service inflation on 259 SWE100/277 tasks. Tested admission consumers did not pass their utility gates. |
| CPU action | **KEEP equal-share burstable execution and prediction-free CPU-idle FCFS as controls.** | Burstable has positive small-pair evidence but no valid general scheduling claim. CPU-idle FCFS has strong replay headroom but lacks causal memory safety. |
| Tool semantics | **KEEP the pytest worker-count × target-scope result; close pip-specific and generic KB-structure work.** | The pytest representation improved RSS accuracy/High recall to 87.799%/54.762% from 82.536%/2.381%, but its static admission consumer failed. |
| Memory action | **KEEP temporal RSS packing as action-space evidence only.** | A hindsight PennyLane oracle improved mean completion 48.823% versus static peak packing. No causal safe scheduler is established. |
| GPU/KV | **Close CacheWise/C100 victim selection under the current simulator.** Tool-gap retention remains unresolved, not active. | Predictor gain was 1.481%; a hindsight upper bound was 9.620%. GPU action experiments exposed tail/action-activation problems. |
| Runtime integration | **None authorized.** | No predictor, feedback controller, or scheduler is integrated for production. |
| PennyLane collection | **Do not run high-load PennyLane on this 16 GB host.** | Existing runs already OOM; `concurrency=1` does not constrain task-internal `pytest -n auto` workers. Use existing valid traces or an explicitly approved high-memory CPU node. |
| Immediate work | **Establish the joint GPU–tool scheduling ceiling from existing traces and measured GPU costs; no collection or evaluator is currently authorized.** | CPU-only analysis is a cheap preflight, not the intended systems contribution. Raw local PennyLane traces have been deleted. |

`status: no_go` answers one frozen claim. It does not authorize deleting a
component marked **KEEP** above.

## 2. Canonical prediction contract

The evaluation unit is one eligible `exec` command. Clauses are internal
evidence and are never scored as separate rows.

### Targets

| Target | Buckets | Primary metric |
|---|---|---|
| Latency | `[0,500]`, `(500,2000]`, `(2000,8000]`, `(8000,30000]`, `(30000,+inf)` ms | exact five-class command accuracy |
| CPU | `[0,2]`, `(2,4]`, `(4,+inf)` peak cores | exact three-class command accuracy |
| RSS | `[0,500]`, `(500,2000]`, `(2000,+inf)` decimal MB | exact three-class command accuracy |
| Disk | `[0,1 MiB]`, `(1,100 MiB]`, `(100 MiB,+inf)` bytes | exact three-class command accuracy |

Hard prediction is the highest-probability bucket; ties choose the lower
bucket. Unavailable prediction counts as incorrect. Report eligible count,
label/prediction counts, confusion matrix, majority accuracy,
within-one-bucket accuracy, and severe underprediction. Resource reports also
include constant-Low accuracy.

An explicitly null CPU/RSS/Disk observation is imputed Low only when command
latency is below 500 ms. Other nulls are unavailable. The old 3/2/2/2 targets,
nine-bin latency, balanced accuracy, Brier/NLL, q-error, and hand-selected
subsets cannot select a candidate.

### Causality and physical composition

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
  independently valid sibling calls may remain eligible.
- Static prediction cannot use current-command output or telemetry.
- Early-execution prediction may use only samples available at its frozen
  decision timestamp. Hindsight state is diagnostic only.

Compound commands compose physical values, never bucket IDs:

| Shell relation | Latency | CPU | RSS | Disk |
|---|---|---|---|---|
| Sequential | sum | max | max | sum |
| Concurrent pipeline | max | sum | sum | sum |

Unsupported or ambiguous structure remains unavailable. Boolean OR is never a
composition rule.

## 3. Selected predictor

The **Clause-KB baseline** uses raw exact argv, argv-prefix, and binary/global
backoff with frozen public evidence and causal repository-local task-final
updates.

The **Task-Aware Command Predictor** uses one common `BeginCall` decision time
and target-specific heads. It first matches complete commands, then equivalent
requested pip or pytest work, and otherwise falls back to Clause-KB. Latency,
CPU, and RSS may raise a third-or-later repeated full-suite test prediction;
Disk uses exact command history. Majority is the constant fit-set mode.

| Target | Eligible | Majority | Clause-KB | Task-Aware | Gain | Selected head |
|---|---:|---:|---:|---:|---:|---|
| Latency | 1,044 | 58.429% | 74.904% | 79.023% | +4.119 pp | command-work match + repeated-test phase |
| CPU | 672 | 82.440% | 82.887% | 87.500% | +4.613 pp | command-work match + repeated-test phase |
| RSS | 736 | 81.250% | 80.978% | 87.500% | +6.522 pp | command-work match + repeated-test phase |
| Disk | 1,008 | 61.012% | 82.044% | 83.829% | +1.786 pp | exact command history |

Equal-weight accuracy is 70.783% Majority, 80.203% Clause-KB, and 84.463%
Task-Aware. This configuration was selected post-hoc on exposed data. Only RSS
clears the existing five-point target gate; no result authorizes confirmation
or runtime integration.

## 4. Research position and next decisions

### Paper-level question

> Can an agent scheduler use tool-command understanding to coordinate GPU
> inference, KV state, and distributed CPU/tool execution across alternating
> agent phases, improving completion time and utilization without unsafe
> interference?

This replaces “find a better KB data structure” as the organizing question.
The intended chain is:

```text
command + docs-derived factors + settled traces -> tool time/resource forecast
GPU queue/KV state + CPU-worker state + feedback -> joint phase decision
  -> admit another agent / run or queue tool / keep or offload KV / place task
  -> task completion, GPU tail, tool service, utilization, and safety
```

The LM, if used, runs once before deployment and emits bounded execution
factors. It never predicts cost directly and has zero prediction-time cost.
The KB learns measured cost from traces. The scheduler consumes predictions;
classification accuracy alone is not the claim. CPU-only backfill is one
action component, not the paper's boundary.

### What we have established

1. **Prediction signal exists.** Task-Aware improves the exposed SQLGlot
   four-target average from 80.203% to 84.463%. A documented interaction—pytest
   worker count together with requested test scope—improves RSS High recall
   from 2.381% to 54.762% on exposed PennyLane results.
2. **Runtime signal exists broadly.** Causal CPU feedback reduced logical
   reservation 45.010% at 1.543% service inflation on 259 SWE tasks from 205
   repositories. This is a per-command counterfactual, not a scheduler result.
3. **Useful actions exist in hindsight.** Prediction-free CPU-idle FCFS
   improved SQLGlot mean completion 14.620% with hindsight RSS safety. A
   PennyLane temporal-RSS oracle improved mean completion 48.823% over static
   peak packing. Both expose headroom; neither is deployable evidence.
4. **GPU/tool phase coupling is already visible.** Raising static active-task
   concurrency from four to eight reduced mean task JCT 21.77%, but worsened
   paired p99 TTFT to 1.262x and 1.131x. Separating task concurrency eight from
   LLM concurrency four reduced mean JCT 13.465%, but a host-side semaphore
   queue inflated end-to-end p99 TTFT by 15.984x–22.231x. These are exposed
   trade-offs and a queue-design failure, not evidence against joint scheduling.
5. **Point classes are not enough for safe overlap.** Better RSS accuracy did
   not produce a better static admission policy. Mean CPU demand admitted too
   much burst overlap; peak demand admitted too little. The missing object is a
   forecast of phase completion and time-varying resource use, not another
   bucket mapping.

### Primary direction: joint GPU–tool phase scheduling

Agents alternate between GPU inference and CPU/tool execution. The scheduler
should not reserve one fixed end-to-end concurrency slot for both phases. It
should separately control:

- **GPU:** ready-request admission and priority; stock KV retention versus the
  existing five-second reactive offload baseline;
- **tool workers:** run/wait admission under CPU, RSS, and Disk pressure;
- **agent admission:** temporary extra tasks while incumbents are in tool
  phases, without creating a burst of simultaneous returning GPU requests;
- **distributed placement:** assign whole task containers to CPU workers that
  share one GPU service. Per-command workspace migration is not required.

Proceed as a decision tree:

1. **Joint oracle before infrastructure.** Replay the existing agent phase
   trajectories with measured GPU inference/KV costs and recorded tool
   profiles. Compare the best GPU-only action, best tool-only action, and a
   joint oracle. The joint direction advances only if coordination adds value
   beyond both single-resource controls at matched GPU-tail and tool-service
   operating points. CPU-only simulation is acceptable for this gate; it is
   not physical evidence.
2. **Small causal action set.** If the joint oracle passes, begin with only
   three decisions: admit another agent or wait; run a ready tool or wait; keep
   KV or use the existing five-second reactive offload. Do not begin with RL,
   PD separation, per-command remote migration, or a general cluster manager.
3. **Action-specific forecasts.** Predict the distribution of active tools'
   return times and conservative CPU/RSS trajectories from settled histories.
   Use causal eBPF/cgroup samples to update surviving tools and locate their
   current phase. Use Task-Aware and feedback-only as separate controls;
   abstain when evidence is unavailable.
4. **Tool understanding where it changes the action.** Add offline
   documentation factors only for command families that account for oracle
   starts or prediction failures. The representation should express execution
   policy, requested-work scope, and mode; measured traces still determine
   resource cost. Demonstrate on three or four tools only after label-blind
   coverage shows they matter.
5. **Physical single-GPU validation.** Use one GPU server and one or more CPU
   tool workers. Compare fixed concurrency, feedback-only, and joint predicted
   scheduling. Report a Pareto frontier rather than inventing one fixed SLO:
   mean/tail task JCT, GPU TTFT/throughput, tool service inflation, KV transfer,
   OOM/capacity violations, and utilization.
6. **Distributed and fresh confirmation.** Only a physical single-GPU gain
   authorizes multiple CPU nodes or a fresh same-repository collection. Whole
   task containers remain on their assigned CPU worker and call the shared GPU;
   the experiment measures whether prediction improves load balance and phase
   overlap, not remote-filesystem engineering.

The first deliverable is a joint-oracle table over existing traces: absolute
GPU/tool utilization and task completion for GPU-only, tool-only, and joint
actions. It requires no new trace collection or LM call. A GPU is needed only
after this gate passes.

### Secondary directions

- **Disk-aware tool placement** is part of the distributed branch only if the
  audit finds repeated Disk-High commands and measurable same-device
  interference. Disk prediction
  is already relatively accurate, but no action headroom has been measured.
- **PD separation** is conditional. Add it only if the repaired joint scheduler
  still shows prefill bursts harming decode/TTFT; the prior phase-aware result
  was dominated by host-side queueing, so it cannot justify PD infrastructure.
- **Predictive early KV timing** remains closed as a standalone line. The joint
  scheduler may use stock caching and the causal five-second action; it must not
  claim that earlier learned triggers already work.

### Closed mechanisms

- Replacing the trie with lattice, poset, generic argv, or semantic keys as a
  contribution: too few action-relevant changes.
- Trace-conditioned agents and generated arbitrary code: no separating state;
  keep the negative result.
- pip-specific semantic work: missed the contrast gate and had only a 0.392
  point perfect-tool ceiling.
- Peak CPU class as hard admission demand: peak is not throughput-preserving
  quota.
- Prediction-weighted CPU shares and shortest-ready ordering: hindsight
  headroom was below the frozen threshold.
- Static hard-RSS CPU-idle admission on exposed SQLGlot/Zarr/PennyLane:
  classification accuracy did not produce a safe action.
- CacheWise/C100 victim selection under the current model: too little residual
  recomputation headroom.
- Further CPU reservation/page tuning: EAR-style work-conserving elasticity and
  causal feedback already capture most of that action; prediction seeding added
  no meaningful benefit.

## 5. Result ledger

All rows are development-exposed. Numbers are descriptive only where the
registered validity or action gate failed.

### Prediction and tool understanding

| Experiment | Key result | Decision |
|---|---|---|
| SQLGlot multitarget predictor | Task-Aware 84.463% equal-weight accuracy vs Clause-KB 80.203% | Retain selected predictor |
| pip/pytest upper bound | pytest hindsight routing +1.941 points overall; 93/118 corrections restored Clause-KB. pip perfect-tool ceiling +0.392 points | Retain pytest question; deprioritize pip |
| Docs-only tool compiler v1 | Only `make` produced a valid spec; 1,792 predictions copied Clause-KB | Closed as uninformative, not a semantics impossibility result |
| Plugin-aware pytest ToolSpec v1/v2 | One-shot generation failed structural contracts before labels | Compiler reliability blocker; no prediction verdict |
| xdist worker scaling | 84.211% RSS accuracy but 2.381% High recall; all carriers Medium | Worker count alone NO-GO |
| xdist worker × scope | 87.799% accuracy and 54.762% High recall; 22 helpful/0 harmful carrier changes | Retain prediction mechanism |

### CPU actions

| Experiment | Key result | Decision |
|---|---|---|
| Two-task burstable replay | 14.490% mean improvement; 13/13 pairs improved | Positive mechanism evidence |
| Saturated four-task replay | 12.966% mean improvement; 8/12 groups improved, below 9/12 gate | Claim NO-GO; retain control |
| Fresh 48-task queue | 7/1,818 paired calls changed terminal class; timing-only mean +0.929%, 2/4 queues improved | Inconclusive validity and below timing gate |
| Prediction-seeded feedback | Feedback 37.076% reservation reduction at 1.032% service inflation; Task-Aware seed 37.635%/1.274% | Incremental prediction gate failed |
| SWE feedback generality | 45.010% reservation reduction at 1.543% service inflation | Retain feedback mechanism |
| Hard-page feedback admission | 6.584% mean-completion gain but 12.936% service inflation | Action NO-GO |
| Feedback with CPU borrowing | 1.982% gain; 7.352% service inflation | Action NO-GO |
| CPU-idle FCFS oracle | 14.620% completion and 17.226% makespan improvement; 3.073% service inflation | Retain action-space evidence |
| Hindsight shortest-safe ordering | 0.137% worse than FCFS | Close latency ordering consumer |
| Predicted RSS safety | 2,638 Clause-KB and 2,702 Task-Aware confirmed exposure events after short-null amendment | Unsafe; no further exposed-data exceptions |
| Prediction-weighted shares | Task-Aware +0.0056%; demand oracle +0.0527% | Closed |
| Exact-duration ready priority | 3.991% mean-completion gain, below 5% gate | Closed |

### PennyLane CPU and memory mechanisms

| Experiment | Key result | Decision |
|---|---|---|
| Pairwise mean CPU | 16.462% completion gain, 6.321% service inflation | Mean hides bursts; NO-GO |
| Static exact peak | 2.176% completion gain, effectively zero inflation | Safe but too conservative |
| Full phase-shape oracle | 10.699% completion gain, effectively zero inflation | Phase headroom exists |
| One-sided fit envelope | 6.707% gain, 0.0835% inflation | Candidate profile transfer works with foreground oracle |
| Two-sided fit envelope | 4.461% gain, 1.757% inflation | Causal foreground transfer misses gate |
| Reactive current + exact peak | 3.853% gain, 3.457% inflation | Causal foreground consumer misses gate |
| Temporal RSS oracle | 48.823% mean-completion gain vs static peak; max 15,864 MB, zero sampled violations | Strong hindsight action-space evidence |
| Scope-conditioned xdist admission | 20.526% gain but 7.261% service inflation and 22 modeled exposures | Prediction GO does not imply action GO |
| Finite fit-envelope admission | 28.244% gain; exposures fell 22→16; 7.935% inflation | Better trade-off, still not safe |

### KV and tool-gap actions

| Experiment | Key result | Decision |
|---|---|---|
| CacheWise/C100 | Predictor recomputation gain 1.481%; hindsight upper bound 9.620% | Closed under current simulator/action |
| Profile-guided retention | +116.507 GiB-s but +1,290.791 ms stall; pre-restore affected too few tasks | Early-clock/pre-restore NO-GO |
| Static survival on SWE | Semantic arms added stall; robust clocks changed zero actions | Cross-repository learned trigger NO-GO |
| PennyLane exact recurrence | First loan advanced by 5 s for 8/26 tasks, all recurring apt setup | Narrow action activation only |
| Static GPU concurrency 4→8 | Mean task JCT -21.77%, but p99 TTFT ratios 1.262/1.131 | Throughput-tail trade-off; no promotion |
| Phase-aware shadow admission | Mean task JCT -13.465%, but end-to-end p99 TTFT ratios 22.231/15.984 | Host admission queue dominates; NO-GO |
| Predictive tool-gap loan | Only one distinct early action, below four-action gate | Insufficient activation; no performance verdict |
| Load-32 hard-pin control | Deadlocked/stalled before a valid result | Invalid control, no method verdict |

## 6. Current operational boundaries

### Deleted PennyLane command-window replay

The synchronized 2603 smoke passed measurement plumbing: all 33 commands had
finite cgroup-memory peaks, zero read failures, and at least five samples; the
peak range was 9.936896–6,122.881024 MB. This validated the measurement path,
not host capacity for the high-load cohort.

Before cleanup, the subsequent exposed replay15 run was **not valid evidence**:

- 15 completed workload artifacts: 11 valid, four invalid;
- invalid tasks: 2654, 3278, 3381, and 3386;
- every invalid artifact reports overlapping exec tool calls and withholds all
  calls fail-closed;
- high-load xdist commands also reproduce the already-known OOM risk on this
  16 GB, no-swap host.

All raw local PennyLane traces and matching temporary collection directories
were deleted on 2026-08-13. Historical machine-readable results listed in
Section 8 remain, but the replay15 artifacts do not. Therefore the frozen
cgroup-memory evaluator cannot run. Do not rerun PennyLane locally. A future
recollection requires explicit approval and a dedicated high-memory CPU node;
`concurrency=1` is insufficient because xdist creates task-internal parallel
workers.

### Deferred, not authorized

- Load-8 live `keep` versus five-second `deadline/reactive` GPU action.
- Load-32 stock evictable prefix-cache control versus `deadline/reactive`.
- High-memory rerun of the four invalid PennyLane calibration tasks.

These are not active work. Any revival requires a short current protocol,
hardware/cost estimate, and explicit approval before launch.

## 7. Evidence boundary

- Original SQLGlot100, SQLGlot50 validation, and the nominal final partition
  are development-exposed.
- SWE100 and SWE277 are development-exposed; their feedback result tests breadth
  but is not confirmation.
- The 41-task PennyLane fit/replay split is development-exposed. Its raw local
  traces were deleted; retained result artifacts are historical evidence but
  no longer provide local byte-level reproduction.
- The separate PennyLane warmup16 and validation16 remain untouched and
  uncollected. They stay closed while the calibration gate is invalid.
- Zarr development19 and validation10 are exposed; final12 remains untouched.
- Confirmation requires genuinely fresh tasks, a new time period, or another
  suitable repository with criteria frozen before outcome access.
- No result-dependent task selection, threshold tuning, package/test-name
  outcome rule, or hindsight state is allowed.
- New collection or a run expected above 30 minutes requires explicit approval
  after a smoke, resource estimate, and decision gate.

## 8. Authoritative artifacts

### Core prediction and CPU action root

`analysis/results/tool-resource-5-3-3-3-20260804/`

- `sqlglot50-multitarget-sota-v1/result.json`
- `swe100-277-cpu-feedback-generality-v1/result.json`
- `sqlglot50-cpu-feedback-admission-v1/result.json`
- `sqlglot50-cpu-feedback-borrowing-v1/result.json`
- `sqlglot50-cpu-idle-backfill-oracle-v1/result.json`
- `sqlglot50-cpu-idle-rss-safety-v1/result.json`
- `sqlglot50-cpu-idle-short-null-v1/result.json`
- `sqlglot26-paired-cpu-borrowing-compatible-v2/result.json`
- `sqlglot48-quartet-cpu-borrowing-contract-v2/result.json`
- `sqlglot-final48-counterbalanced-rolling-exact-v1/result.json`

### Tool semantics and survival

- `analysis/results/pip-pytest-upper-bound-sqlglot-v1/result.json`
- `analysis/results/offline-tool-semantics-sqlglot-v1/result.json`
- `analysis/results/static-survival-gap-action-swe177-20260810/result.json`
- `analysis/results/survival-work-state-action-swe177-20260810/result.json`
- `analysis/results/survival-robust-clock-swe177-20260810/result.json`
- `analysis/results/task-pareto-survival-action-pennylane41-20260810/result.json`
- `analysis/results/first-loan-actionability-pennylane41-20260810/result.json`

### PennyLane action results

- `analysis/results/pennylane-pairwise-cpu-backfill-20260810/result.json`
- `analysis/results/pennylane-pairwise-peak-backfill-20260810/result.json`
- `analysis/results/pennylane-phase-shape-backfill-20260810/result.json`
- `analysis/results/pennylane-clause-kb-phase-envelope-20260810/result.json`
- `analysis/results/pennylane-two-sided-phase-envelope-20260811/result.json`
- `analysis/results/pennylane-reactive-current-peak-20260811/result.json`
- `analysis/results/pennylane-temporal-rss-packing-ceiling-development-v1/result.json`
- `analysis/results/pennylane-xdist-rss-positive-control-v1/result.json`
- `analysis/results/pennylane-xdist-rss-positive-control-v2/result.json`
- `analysis/results/pennylane-xdist-rss-admission-v1/result.json`
- `analysis/results/pennylane-xdist-rss-fit-envelope-admission-v1/result.json`

### GPU action results

- `analysis/results/gpu-tool-gap-actions-a100-instruct-20260809/result.json`
- `analysis/results/gpu-tool-gap-actions-a100-instruct-20260809/load32-hard-pin-stall.json`
- `analysis/results/fixed-trajectory-static-ceiling-20260810/result.json`
- `analysis/results/fixed-trajectory-phase-aware-admission-20260810/result.json`
- `analysis/results/predictive-tool-gap-loan-20260810/result.json`

Task split authorities:

- `analysis/development/sqlglot-relational-task-split.json`
- `analysis/development/pennylane-survival-action-split.json`
- `analysis/development/offline-tool-semantics-splits.json`

## 9. Non-negotiable checklist

```text
Evaluation unit = eligible exec command; clauses are internal evidence.
Prediction targets = latency 5 buckets; CPU/RSS/Disk 3 buckets.
Unavailable hard predictions count as incorrect.
Short-null resource policy = Low only when explicitly marked and <500 ms.
Causal visibility = observation end before query start, after task settlement.
Compound commands = physical sequential/pipeline composition, never Boolean OR.
Clause-KB = unchanged raw exact/prefix/binary control.
Task-Aware = selected development candidate, not deployed.
Causal eBPF feedback = retained mechanism/control, not integrated.
Prediction GO does not imply action GO.
Protocol NO-GO does not imply deleting a retained baseline.
No result-dependent tuning, hindsight state, or dataset-specific outcome rule.
Offline LM input = pinned public docs/help only; prediction-time LM cost zero.
Scheduler claims require an observable action, measured costs, and physical safety.
No PennyLane high-load collection on the local 16 GB host.
No new collection or runtime integration without a separate approved protocol.
```

## 10. Development document map

| Role | Files |
|---|---|
| Current authority | this file; `tool-resource-service-architecture.md` |
| Frozen protocol provenance retained because evaluators/results reference it | `cpu-feedback-admission-protocol.md`, `cpu-feedback-borrowing-protocol.md`, `cpu-idle-speculative-backfill-protocol.md`, `cpu-idle-rss-safety-protocol.md`, `cpu-idle-short-null-amendment.md`, `pip-pytest-upper-bound-protocol.md` |
| Machine-readable split/source inputs | JSON and pinned documentation snapshots in this directory |

Completed implementation plans are not current documents and are not kept
here. Git history retains them.
