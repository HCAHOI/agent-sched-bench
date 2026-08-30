# Tool-resource prediction: canonical objective and decisions

**Effective:** 2026-08-30

**Scope:** current targets, causal semantics, retained decisions, evidence
exposure, and launch boundaries

This file is authoritative for tool-resource work. It is not an experiment
diary. Frozen result receipts control their own numbers and provenance;
[`tool-resource-service-architecture.md`](tool-resource-service-architecture.md)
controls the runtime interface. The broader research frontier is
[`../ROADMAP.md`](../ROADMAP.md).

## Current decisions

| Area | Decision | Supported boundary |
|---|---|---|
| Prediction | **KEEP Task-Aware Command Predictor** as the selected development candidate. | Exposed SQLGlot50 equal-weight four-target accuracy is 84.463% versus 80.203% for Clause-KB. Preregistered PennyLane transfer is 78.326% versus 75.680% on 15 scored tasks. It is not deployed. |
| Learned baseline | **KEEP Clause-KB** as the normal baseline. | Production semantics are raw exact argv, argv-prefix, binary, then global backoff. Trie storage is an implementation detail, not a contribution. |
| Runtime feedback | **KEEP causal command-level eBPF feedback** as a measured mechanism and control. | On 259 SWE100/277 tasks it reduced logical CPU reservation 45.010% at 1.543% service inflation. Tested admission consumers failed their utility gates. |
| CPU action | **KEEP equal-share burstable execution and prediction-free CPU-idle FCFS as controls.** | Burstable has positive small-pair evidence but no general scheduling claim. CPU-idle FCFS has replay headroom but no causal memory-safety proof. |
| Tool semantics | **KEEP pytest worker count × target scope; CLOSE pip-specific and generic KB-structure work.** | The pytest representation improved RSS accuracy/High recall to 87.799%/54.762% from 82.536%/2.381%; its static admission consumer still failed. |
| Memory action | **KEEP temporal RSS packing as action-space evidence only.** | A hindsight PennyLane oracle improved mean completion 48.823% over static peak packing. No causal safe scheduler follows from it. |
| Joint action | **KEEP joint phase coordination as action-space evidence; CLOSE CPU-only bucket carriers.** | The frozen PennyLane hindsight arm reached 14,100.143 s mean completion versus 21,705.486 s for the best tool-only arm. Both causal bucket carriers deadlocked; this is not a physical GPU result. |
| Tool-container state | **CLOSE parking and remote snapshot placement for the current PennyLane action model.** | Instantaneous free parking worsened mean completion 2.664% and released too little CPU/RSS resource-time. |
| KV duration prediction | **CLOSE CacheWise/C100 victim selection and predictive tool-gap lending under their tested mappings.** | C100 predictor recomputation gain was 1.481% versus a 9.620% hindsight bound; the physical hard bucket could not activate before feedback. This does not close phase-level KV retention. |
| Program/KV scheduling | **KEEP official ThunderAgent as the strong high-pressure baseline.** | On development-exposed Unique-128 it reduced mean JCT 53.4% but raised p99 TTFT from 445.2 s to 1,524.9 s. The residual target is bounded foreground-return latency, not generic program scheduling. |
| Runtime integration | **None authorized.** | No predictor, feedback controller, lease controller, or compaction policy is integrated for production. |
| PennyLane corpus | **Use the completed 76-task high-memory-node corpus; do not rerun it on this 16 GB host.** | All tasks retain evidence-valid clause aggregates; six replay-only attempts use the canonical trace-to-tool-call fallback. |
| Immediate work | **Close faithful baselines before a new controller.** | First obtain the exact-fork CacheWise policy-disabled control and reconcile applicable official/public/reproduction baselines. The frozen 6–9 A100-hour native-priority protocol remains unlaunched and needs explicit approval plus reconciliation with the newer Unique-128 evidence. |

`status: no_go` answers one frozen claim. It never authorizes deleting a
component marked **KEEP**.

## Canonical prediction contract

The evaluation unit is one eligible `exec` command. Parsed clauses are internal
evidence and are never scored as separate rows.

| Target | Buckets | Primary metric |
|---|---|---|
| Latency | `[0,500]`, `(500,2000]`, `(2000,8000]`, `(8000,30000]`, `(30000,+inf)` ms | exact five-class command accuracy |
| CPU | `[0,2]`, `(2,4]`, `(4,+inf)` peak cores | exact three-class command accuracy |
| RSS | `[0,500]`, `(500,2000]`, `(2000,+inf)` decimal MB | exact three-class command accuracy |
| Disk | `[0,1 MiB]`, `(1,100 MiB]`, `(100 MiB,+inf)` bytes | exact three-class command accuracy |

Hard prediction is the highest-probability bucket; ties choose the lower bucket.
Unavailable prediction counts as incorrect. Reports include eligible count,
label/prediction counts, confusion matrix, majority accuracy, within-one-bucket
accuracy, and severe underprediction; resource reports also include constant-Low
accuracy.

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
- A task becomes learnable only after successful whole-task finalization. The
  current task and failed or unfinalized tasks remain invisible.
- Frozen cross-repository evidence never updates during evaluation.
- Every arm uses identical command IDs, labels, availability, order, and shell
  structure.
- Invalid, ambiguous, lossy, or cleanup-invalid telemetry is withheld;
  independently valid sibling calls may remain eligible.
- Static prediction cannot use current-command output or telemetry. An
  early-execution arm may read only samples complete by its frozen decision
  timestamp. Hindsight state is diagnostic only.

Compound commands compose physical values, never bucket IDs:

| Shell relation | Latency | CPU | RSS | Disk |
|---|---|---|---|---|
| Sequential | sum | max | max | sum |
| Concurrent pipeline | max | sum | sum | sum |

Unsupported or ambiguous structure remains unavailable. Boolean OR is never a
composition rule.

## Selected predictor

The **Clause-KB baseline** uses raw exact argv, argv-prefix, and binary/global
backoff with frozen public evidence and causal repository-local task-final
updates.

The **Task-Aware Command Predictor** uses one `BeginCall` decision time and
target-specific heads. It first matches complete commands, then equivalent
requested pip or pytest work, and otherwise falls back to Clause-KB. Latency,
CPU, and RSS may raise a third-or-later repeated full-suite test prediction;
Disk uses exact command history. Majority is the fit-set constant mode.

| Target | Eligible | Majority | Clause-KB | Task-Aware | Gain | Selected head |
|---|---:|---:|---:|---:|---:|---|
| Latency | 1,044 | 58.429% | 74.904% | 79.023% | +4.119 pp | command-work match + repeated-test phase |
| CPU | 672 | 82.440% | 82.887% | 87.500% | +4.613 pp | command-work match + repeated-test phase |
| RSS | 736 | 81.250% | 80.978% | 87.500% | +6.522 pp | command-work match + repeated-test phase |
| Disk | 1,008 | 61.012% | 82.044% | 83.829% | +1.786 pp | exact command history |

Equal-weight accuracy is 70.783% Majority, 80.203% Clause-KB, and 84.463%
Task-Aware. Task-Aware was selected post hoc on exposed SQLGlot data. Only RSS
clears the existing five-point target gate; no result authorizes confirmation
or runtime integration.

## Result ledger

All rows below are development-exposed. A hindsight or invalid-gate row is
action-space evidence, not a deployable result.

### Prediction and tool understanding

| Result | Observation | Decision |
|---|---|---|
| SQLGlot multitarget | Task-Aware 84.463% versus Clause-KB 80.203% | KEEP selected predictor |
| PennyLane transfer | 78.326% versus 75.680%; task-bootstrap gain CI +0.746 to +4.515 pp | Transfer gate GO; validation consumed |
| pytest worker × scope | RSS 87.799% accuracy and 54.762% High recall; 22 helpful/0 harmful carrier changes | KEEP mechanism; action claim absent |
| pip perfect-tool ceiling | +0.392 pp | CLOSE pip-specific work |
| Generic docs compiler | Only `make` produced a valid spec; 1,792 predictions copied Clause-KB | CLOSE implementation route, not semantic possibility |

### CPU, memory, and joint actions

| Result | Observation | Decision |
|---|---|---|
| Two-task burstable replay | Mean completion -14.490%; 13/13 pairs improved | KEEP positive mechanism control |
| Saturated four-task replay | Mean -12.966%, but only 8/12 groups improved versus 9/12 gate | Claim NO-GO; KEEP control |
| SWE causal feedback | Reservation -45.010% at +1.543% service | KEEP feedback mechanism |
| Hard-page / borrowing admissions | Completion improved 6.584% at 12.936% service inflation and 1.982% at 7.352% | Action NO-GO |
| CPU-idle FCFS hindsight | Mean completion -14.620%, makespan -17.226%, service +3.073% | KEEP action-space evidence |
| Prediction-weighted shares / exact-duration priority | At most +0.0527% and 3.991% mean gain | CLOSE these consumers |
| Temporal RSS hindsight | Mean completion -48.823% over static peak; zero sampled violations | KEEP action-space evidence |
| Scope-conditioned xdist admission | Mean -20.526% but service +7.261% and 22 modeled exposures | Action NO-GO |
| Joint phase-packing hindsight | Mean 14,100.143 s versus tool-only 21,705.486 s; max active 11; zero modeled violations | KEEP joint action-space evidence |
| Hard-page and finite-bound causal joint carriers | Both deadlocked at 307.167 s with 0/35 complete; serial control completed | CLOSE CPU-only carrier tuning |
| Free perfect parking | Mean +2.664%; CPU-core-time -3.178%, RSS-time -1.316% | CLOSE parking and remote snapshot branch |

### GPU/KV and phase actions

| Result | Observation | Decision |
|---|---|---|
| CacheWise/C100 simulator | Predictor recomputation +1.481%; hindsight bound +9.620% | CLOSE this predictor/action mapping |
| Static concurrency 4→8 | Mean JCT -21.77%; p99 TTFT ratios 1.262/1.131 | Throughput-tail trade-off only |
| Host phase admission | Mean JCT -13.465%; end-to-end p99 ratios 22.231/15.984 | Queue-design NO-GO |
| PennyLane physical feedback | Mean JCT -32.748%/-33.102%, makespan -42.774%/-42.443%; p99 1.048x/1.115x | KEEP Pareto mechanism; repeated tail-safe claim NO-GO |
| Predictive physical loan | No predictor-triggered loan; fifth bucket starts at 30 s below the 34.683–34.791 s budget | CLOSE registered mapping, not Task-Aware generally |
| Revocable lease replay | Mean -27.039%, makespan -33.733%, retained 89.427% of permanent-loan gain; overlap -10.309% versus 25% gate | Mechanism evidence; frozen NO-GO for physical promotion |
| ThunderAgent, low pressure | Official baseline made mean JCT 0.55% worse; no pause/resume and peak KV below 39% | Valid official-baseline run, no opportunity at this load |
| ThunderAgent, Unique-128 | Mean JCT -53.4%, throughput +57.2%, p99 TTFT 445.2→1,524.9 s | KEEP strong high-pressure baseline; residual tail gap |
| CacheWise reproduction, Unique-128 | Mean JCT -59.2%, throughput +68.3%, p99 TTFT 2,299.0 s | Exact-fork policy-disabled control required |
| SAGA KV subset, Unique-128 | Mean JCT +34.4%; 0/128 tasks faster | Subset NO-GO on this workload, not full-SAGA verdict |

Primary receipts are indexed in [`../README.md`](../README.md#result-entry-points).

## Frozen native-priority protocol

This protocol is retained because its gates and amendments are integrity
records. It is **not launched** and is no longer automatically the next run.

- Cohort: seed-42 order `3182, 5835, 4366, 4251, 1405, 6062, 5623, 6939`, all
  `PennyLaneAI__pennylane-`; first four foreground, last four borrowers.
- Population: 972 actions, 490 LLM calls, 482 tool calls, 370 shell execs.
- Hardware/model: one A100 80 GB at 250 W,
  `NousResearch/Meta-Llama-3.1-8B-Instruct`, 131,072-token limit, container CPU
  cap two, empty prefix cache per cell.
- Arms/order: `fixed -> feedback -> priority-feedback`, then reverse. Fixed and
  feedback send priority zero; priority-feedback sends borrowers at priority
  one. No host admission cap; queueing is included in TTFT.
- Inputs: [`pennylane-native-priority-v1/manifest.yaml`](pennylane-native-priority-v1/manifest.yaml)
  and [`pennylane-native-priority-v1/resource-profile.yaml`](pennylane-native-priority-v1/resource-profile.yaml).
- Runner/scorer: [`../../scripts/evaluation/run_pennylane_native_priority.sh`](../../scripts/evaluation/run_pennylane_native_priority.sh)
  and [`../../scripts/evaluation/evaluate_pennylane_native_priority.py`](../../scripts/evaluation/evaluate_pennylane_native_priority.py).
- Frozen gate: both repetitions valid; feedback and priority-feedback each
  improve mean JCT at least 5% and makespan strictly versus fixed;
  priority-feedback retains at least 80% of feedback's mean-JCT gain; its p95
  and p99 TTFT are at most 1.05x fixed; its p99 is below feedback in both
  repetitions and geometric-mean p99 ratio versus feedback is at most 0.95.
  Any failure stops this branch.

**Pre-outcome clarification, 2026-08-17.** The exact model namespace, token and
power limits, run root
`/home/Ubuntu/pennylane-native-priority-physical-v1-20260818`, result path
`analysis/results/pennylane-native-priority-physical-development-v1/result.json`,
inputs, driver, scorer, numbered cell order, priority policy, and readiness-only
probe above were fixed before outcomes. No generative warmup may seed the cache;
no run root may be overwritten.

**Execution amendment, 2026-08-19, before any priority-effect result was read.**
Cell 3 replay completed but had one 4.66 s GPU-telemetry gap above the frozen
3 s validity limit. Effect gates and the limit did not change. Because each
cell has an independent fresh server and telemetry stream, the driver now runs
and validates all six cells even after one invalid cell, retains failures, and
withholds analysis until all six claim-bearing cells are valid.

The run costs about 6–9 A100 hours. Launch requires a fresh allocation,
reconciliation with the Unique-128 baseline plan, and explicit approval.

## Integrity amendments and settled gates

- **Physical feedback replication, post-outcome amendment, 2026-08-17.** After
  the first three cells were visible and the second predictor activation gate
  became unreachable, the user authorized stopping predictor work and running
  only `feedback-r2` then `fixed-r2`. Tasks, model, hardware, fresh-server
  lifecycle, and telemetry stayed fixed. Every repeated-mechanism condition
  passed except feedback-r2 p99 (1.115x > 1.05), so the tail-safe claim is
  NO-GO. This amendment does not change the predictive-loan NO-GO.
- **Revocable lease, frozen before outcome.** Physical promotion required at
  least 5% mean gain, lower makespan, at least 80% retention of permanent-loan
  gain, at least 25% lower LLM overlap, and no higher peak simultaneous prompt
  tokens. Overlap fell only 10.309%; the gate remains NO-GO. The post-outcome
  diagnosis that fixed-four supplied 73.655% of total overlap is explanatory,
  not a gate amendment.
- **Short-null resource amendment.** Explicit null resource labels are Low only
  below 500 ms; all other nulls remain unavailable. Exposed-data exceptions are
  closed. See [`cpu-idle-short-null-amendment.md`](cpu-idle-short-null-amendment.md).
- **PennyLane transfer exclusion.** Validation task
  `PennyLaneAI__pennylane-5846` was excluded without replacement only after a
  replay-format diagnostic exposed its status, first aggregate, and trace
  excerpt; 15 preregistered tasks were scored once and are consumed.

## Evidence boundary

- Original SQLGlot100, SQLGlot50 validation, and the nominal final SQLGlot
  partition are development-exposed.
- SWE100 and SWE277 are development-exposed; feedback breadth is not
  confirmation.
- The 41-task PennyLane fit/replay split and all 70 ordinary PennyLane
  trajectories used by joint phase packing are development-exposed. Six
  replay-only attempts were excluded by frozen format criteria.
- PennyLane warmup16 supplied fit evidence. The 15 scored validation tasks are
  consumed under the exclusion above.
- The Unique-128 PennyLane/SQLGlot stress workload and every policy result on it
  are development-exposed; tools were trace-timed, so it supports GPU/KV claims
  only. CacheWise also uses a different vLLM fork.
- Zarr development19 and validation10 are exposed; final12 remains untouched.
- Confirmation requires genuinely fresh tasks, a new time period, or another
  suitable repository, with criteria frozen before outcome access.
- No result-dependent task selection, threshold tuning, package/test-name
  outcome rule, hindsight feature, or synthetic concurrency may repair a
  failed gate.
- New collection or a run expected above 30 minutes requires a smoke, resource
  estimate, decision gate, and explicit approval.

## Authoritative artifacts

- Core prediction/CPU results:
  [`../results/tool-resource-5-3-3-3-20260804/`](../results/tool-resource-5-3-3-3-20260804/)
- PennyLane prediction:
  [`../results/pennylane-multitarget-transfer-development-v1/result.json`](../results/pennylane-multitarget-transfer-development-v1/result.json)
  and [`../results/pennylane-multitarget-transfer-validation-v1/result.json`](../results/pennylane-multitarget-transfer-validation-v1/result.json)
- Joint/causal actions:
  [`../results/pennylane-joint-phase-packing-v1/result.json`](../results/pennylane-joint-phase-packing-v1/result.json),
  [`../results/pennylane-causal-joint-tool-admission-v1/result.json`](../results/pennylane-causal-joint-tool-admission-v1/result.json),
  and [`../results/pennylane-finite-bound-joint-admission-v1/result.json`](../results/pennylane-finite-bound-joint-admission-v1/result.json)
- Physical phase actions:
  [`../results/pennylane-physical-gap-loan-development-v1/result.json`](../results/pennylane-physical-gap-loan-development-v1/result.json),
  [`../results/pennylane-revocable-tool-lease-development-v1/result.json`](../results/pennylane-revocable-tool-lease-development-v1/result.json),
  and [`../results/pennylane-perfect-container-parking-development-v1/result.json`](../results/pennylane-perfect-container-parking-development-v1/result.json)
- Paper baselines:
  [`../results/paper-baseline-physical-20260820/result.json`](../results/paper-baseline-physical-20260820/result.json),
  [`../results/pennylane-paper-baseline-suite-physical-v1.md`](../results/pennylane-paper-baseline-suite-physical-v1.md),
  and [`../results/mixed128-poisson-unique-baselines-20260827/result.md`](../results/mixed128-poisson-unique-baselines-20260827/result.md)
- Split authorities:
  [`sqlglot-relational-task-split.json`](sqlglot-relational-task-split.json),
  [`pennylane-survival-action-split.json`](pennylane-survival-action-split.json),
  and [`offline-tool-semantics-splits.json`](offline-tool-semantics-splits.json)

## Non-negotiable checklist

```text
Evaluation unit = eligible exec command; clauses are internal evidence.
Targets = latency 5 buckets; CPU/RSS/Disk 3 buckets.
Unavailable hard predictions count as incorrect.
Short-null = Low only when explicitly null and <500 ms.
Causal visibility = observation end before query start, after task settlement.
Compound commands = physical sequential/pipeline composition, never Boolean OR.
Clause-KB = unchanged raw exact/prefix/binary control.
Task-Aware = selected exposed-data candidate, not deployed.
Causal eBPF feedback = retained mechanism/control, not integrated.
Prediction GO does not imply action GO.
Protocol NO-GO does not imply deleting a retained baseline.
No result-dependent tuning, hindsight state, or dataset-specific outcome rule.
Scheduler claims require observable actions, charged costs, and physical safety.
No PennyLane high-load collection on the local 16 GB host.
No new collection, long run, or runtime integration without separate approval.
```
