# Tool-Resource Prediction — Canonical Objective and Lock

**Effective:** 2026-08-09
**Scope:** current contract and decisions only

This file is the authority for tool-resource targets, evaluation semantics,
evidence boundaries, and scheduler-facing decisions. It intentionally omits
superseded protocols and chronological experiment narration. Git and frozen
artifacts retain that history.

## 1. Current decisions

| Area | Current decision |
|---|---|
| Prediction | Keep the target-specific Task-Aware Command Predictor below as the selected development candidate. It is exposed evidence, not confirmation or deployed code. |
| Runtime feedback | **KEEP causal command-level eBPF feedback as a measured mechanism and control.** Neither tested admission action passed its frozen utility gate, so it is not a runtime scheduler candidate. |
| CPU execution | Keep equal-share burstable execution as a physical control. Phase oracle and one-sided transfer expose headroom, but both tested causal foreground actions miss the completion gate. Close further CPU-policy tuning on the exposed PennyLane wave. |
| Saturated quartet result | Frozen claim verdict is NO-GO because 8/12, not 9/12, groups improved. This is not a module-retirement decision. |
| Fresh counterbalanced queue | **Inconclusive because 7/1,818 paired tool calls changed terminal class.** Timing-only improvement was 0.929% with 2/4 queues improving, below the frozen gate even before the quality failure. |
| Prediction contribution to feedback | Not established. Task-Aware seeding did not improve the frozen reservation/service operating point over feedback alone. |
| CPU scheduling | Short-null handling reduced conservative exposure counts about 78%, but Clause-KB and Task-Aware still produced 2,638 and 2,702 confirmed exposure events. Both hard-RSS safety arms remain closed on exposed SQLGlot; the prediction-free action ceiling is retained. |
| KB representation | Raw exact/prefix/binary `ClauseResourceKB` is the implemented Clause-KB baseline. Trie/lattice/semantic-key replacement is closed as a contribution. |
| Offline tool knowledge | Trace-conditioned generation remains closed. The frozen docs-only v1 protocol also stops before fresh collection: only `make` produced a valid specification, and its SQLGlot arm never formed a non-exact contrast. This is an uninformative protocol result, not evidence that documentation semantics cannot work. |
| Manual tool semantics | **KEEP only the pytest candidate-routing question.** A hindsight selector over existing causal candidates improved full-cohort four-target accuracy by 1.941 points for pytest, but 93/118 corrections merely restored Clause-KB; adding more pytest rules is not supported. pip missed the frozen contrast gate and its perfect-tool ceiling was only 0.392 points, so deprioritize pip-specific work. |
| KV victim selection | Closed under the current CacheWise/C100 simulator and action model. |
| GPU tool-gap action | The profile-only early clock and pre-restore are closed by the Section 5.1 offline gate. The load-8 live action test remains unresolved. The load-32 hard-pin control in Section 5.3 was invalid; Section 5.4 compares the unchanged five-second action with stock evictable prefix caching. |
| Static survival KV action | Cross-repository semantic transfer remains NO-GO. On fixed-history PennyLane, exact recurrence safely advanced the first physical loan for 8/26 tasks by 5 s each; every changed command was recurring `apt-get` setup. This passes the development actionability gate but does not yet establish live completion benefit. |
| Peak-class admission | Closed. Peak CPU classes are the wrong target for sustaining command throughput. |
| Runtime integration | No predictor, feedback controller, or scheduler is currently integrated or authorized for production. |
| Predictive tool-gap loan | The frozen six-cell run passed registered execution validity but activated only one distinct early action, below the required four. Status is `insufficient_action_activation`, not a performance verdict; no tuning or fresh confirmation is authorized. |
| Next research step | On the same development-exposed PennyLane cohort, audit whether task RSS rises are attributable to tool-command intervals and visible at command admission. Use that timing evidence to freeze the smallest causal Clause/KB action; do not claim the temporal oracle as deployable or open final12. |

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
| Complete-work/state survival action | Exact command: +668.981 GiB-s release, +1,236.944 ms stall; work and work+state: +794.288 GiB-s, +1,848.373 ms. State changed zero actions. | All learned arms fail the no-added-stall gate; representation-only branch closed on exposed SWE |
| Task-stable robust survival clock | Exact and work clocks both changed zero actions and were bit-identical to five-second feedback. | NO-GO for this action objective; latency utility cannot select memory-time-only improvements |
| PennyLane task-Pareto survival action | Exact task-Pareto: +389.756 GiB-s, zero stall, 11 tasks; work task-Pareto: +1,438.416 GiB-s, +3,611.076 ms stall, 26 tasks. Exact-immediate control: +660.664 GiB-s, zero stall, 16 tasks. | Primary semantic-transfer claim NO-GO; retain exact recurrence only as a narrow fresh/live candidate |
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
| Hard-page feedback admission | Versus static Task-Aware requests, mean per-order task completion improved 6.584%; all 32 orders improved, queue time fell 6.702%, and makespan fell 6.810% by ratio of means | Positive admission mechanism; frozen physical-advance gate failed because service inflation was 12.936%, above 5% |
| Feedback admission with CPU borrowing | Mean per-order task completion improved 1.982%; 27/32 orders improved and the paired absolute-delta interval was [-278.536, -133.809] s. By ratio of means, queue fell 2.140%, service fell 0.877%, but makespan rose 0.917% | Frozen gate failed: effect was below 5% and service inflation was 7.352%; stop before PSI calibration |
| CPU-idle FCFS backfill | Versus Serial-8, mean per-order completion improved 14.620%; all 32 orders improved, makespan improved 17.226%, and service inflation was 3.073% | Action gate passed under strict-priority CPU and hindsight RSS safety; requires causal RSS safety and physical calibration |
| CPU-idle shortest-safe selection | Hindsight shortest selection changed completion by -0.137% versus FCFS; 17/32 orders improved and the paired interval [-51.549, 70.048] s crossed zero | Ordering predictor has no headroom; do not build a latency/shortest selector |
| CPU-idle predicted RSS safety | Conservative null handling produced 12,390/12,439 Clause-KB/Task-Aware exposures. The open short-null amendment reduced these to 2,638/2,702, but both remained nonzero; utility was 33.392%/25.984% | Amendment gate still failed; do not calibrate physical `cpu.idle` or add another exposed-data exception |
| PennyLane phase-aware CPU backfill | Full-profile oracle: 10.699% completion gain at zero service cost. Fit-only candidate envelope: 6.707% at 0.0835%. Two-sided envelope: 4.461% at 1.757%. Reactive-current plus oracle peak: 3.853% at 3.457%. | Phase headroom and candidate transfer are real, but both causal foreground actions miss the gate; close tuning on exposed PennyLane |
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

The strongest measured mechanism is causal command-level feedback. Each
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
shows broad signal and a viable operating point.

The frozen dynamic-admission replay connected those page updates to an
eight-core FCFS-ready queue over the 50 exposed SQLGlot tasks. Relative to
static Task-Aware reservations, feedback improved mean per-order task
completion 6.584%; all 32 fixed workload orders improved and the paired
order-bootstrap interval for the absolute delta was [-857.949, -626.791]
seconds. By ratio of means, queue time fell 6.702%, makespan fell 6.810%,
service time fell 9.354%, and reserved CPU-core-seconds fell 30.171%.

The physical-advance gate nevertheless failed: candidate service remained
12.936% above recorded eight-core service, versus the frozen 5% ceiling. This
failure was consistent across orders (minimum inflation 7.921%). Expansion
pressure explained the lost work: 82.030% of expansion attempts were denied by
the saturated hard-reservation pool, while feedback recovered 47.393% of the
static predictor's service overhead. A hard page therefore turned
capacity-accounting contention into throughput loss.

The separately frozen borrowing replay removed that confound: logical pages
controlled admission only, while every admitted command received equal-weight,
work-conserving CPU. Feedback still improved mean completion in 27/32 orders;
the mean per-order relative reduction was 1.982% and the paired absolute-delta
interval was [-278.536, -133.809] seconds. By ratio of means, queue time fell
2.140% and command service fell 0.877%, but makespan rose 0.917%. Logical
reserved core-seconds fell 32.619% without changing maximum concurrency.

This was a clean NO-GO for the tested action, not for the feedback module.
Candidate service remained 7.352% above recorded eight-core service, and only
6/32 orders were within the 5% ceiling. The mean effect also missed 5%. CPU
work was conserved and no capacity violation occurred. Mechanistically,
borrowing had already removed most hard-page loss from the static baseline, so
feedback mainly changed queue order; its completion delta was almost entirely
aligned with queue-time delta across orders, while the extra overlap still paid
physical contention cost. The simulation used an ideal causal backlog bit, so
a real PSI signal cannot rescue this frozen hypothesis. Do not run PSI
calibration, physical replay, or post-hoc page tuning from this result.

Strict-priority CPU-idle backfill tested a different action: one normal command
had first claim on all eight cores, while one speculative command used only
residual CPU. With a shared hindsight RSS-fit filter, prediction-free FCFS
backfill improved mean per-order task completion 14.620%; all 32 orders
improved and the paired absolute-delta interval was [-2127.184, -1870.621]
seconds. By ratio of means, makespan improved 17.226% and service inflation was
3.073%. CPU work was conserved and no modeled capacity violation occurred.

Candidate ordering is not the missing predictor consumer. Hindsight shortest
RSS-safe selection was 0.137% worse than FCFS on mean per-order completion;
only 17/32 orders improved and its paired interval crossed zero.

Replacing hindsight fit with frozen hard-RSS predictions increased overlap but
did not establish safety. Task-Aware improved mean completion 25.984% in all 32
orders, with a paired absolute-delta interval of [-3702.319, -3437.197]
seconds, 10.525% lower makespan, and 4.561% service inflation. It nevertheless
created 12,439 conservative source-RSS capacity exposures. Clause-KB was also
unsafe, with 12,390 exposures, and improved completion more: 33.392%. Thus
Task-Aware failed both the zero-exposure gate and its required one-point
incremental contribution. CPU work and predicted capacity were valid in every
arm; the failure is RSS evidence and action quality, not replay integrity.

The conservative failure was partly a mismatch between prediction labels and
physical evidence. Source RSS was unavailable for 680/1,196 commands. Every
one of the 527 explicit CPU-and-RSS null commands was marked
`insufficient_rss_samples`, not invalid telemetry; 450 whole commands lasted
under 500 ms.

An openly post-outcome amendment therefore applied the canonical short-null
rule per clause: only `insufficient_rss_samples` clauses below 500 ms received
the Low upper bound of 500 MB; all other null, unmatched, and invalid sources
remained 16 GB. This imputed 1,254 clauses in 521 commands. The amended
source-RSS oracle improved completion 34.765% with zero exposure. Clause-KB
captured 33.392% improvement at 4.783% service inflation, while Task-Aware
captured 25.984% at 4.561%. All 32 orders improved for both.

The special case was useful but insufficient. Confirmed exposure totals fell
78.709% for Clause-KB, from 12,390 to 2,638, and 78.278% for Task-Aware, from
12,439 to 2,702. Each arm failed only its zero-exposure gate; all utility,
service, makespan, CPU-work, and predicted-capacity checks passed. The replay
also contained 11,930 and 11,941 overlaps involving physically unverified
short-null commands. Static attribution found 123 Clause-KB and 110 Task-Aware
under-reservations outside the imputed command set, across all 50 tasks. The
remaining error is therefore not removable by reclassifying short nulls.
Do not add another exception on this exposed cohort.

### 5.1 Frozen GPU tool-gap retention protocol

**Question.** After an agent finishes an LLM turn and starts a tool, can a
profile-only command-duration model free its prompt KV earlier, and can a
causal pre-restore hide the next-turn reload, without harming the other
tenants? This is a retention-timing action, not the closed C100 victim selector.

**Evidence boundary.** The 50 tasks under
`traces/swe-rebench/qwen3.7-max/20260624T162037` are the only fitting corpus.
The disjoint 277 tasks under
`traces/swe-rebench/qwen3.7-max/fresh-seed42-skip150-n200` are an exposed
development replay; their outcomes only score frozen decisions. Tool name and
command become visible at that tool's replay event boundary. Source offsets,
tool duration, the next event, and the next LLM arrival are never predictor
inputs. The current turn's block-aligned prompt size is observable.

**Physical costs.** Use only the measured Llama-3.1-8B-Instruct/A100-80GB
artifacts in
`analysis/serving/tool-time-rho-measurement-a100-instruct-20260809/`. Map prompt size to
the smallest measured token point at least as large; do not extrapolate above
131,072 tokens. The measured D2H and H2D values are separate costs. Recompute
is a measured comparison, not a live arm: host round-trip is 6.19x--10.25x
faster than prefix-cache-free prefill at the six shared 1K--32K points.

**2026-08-09 plumbing amendment, before replay outcome access.** The original
serving config named the base Llama-3.1-8B tokenizer, which has no chat template;
the evaluator stopped after profile fitting and emitted no result. Serving and
tokenization are therefore corrected to Llama-3.1-8B-Instruct, the model used
by the existing validated connector experiments. The architecture and KV
layout are unchanged, but both physical artifacts are remeasured under the
exact corrected model identifier before evaluation. No gate or data split
changes.

**2026-08-09 transcript amendment, before replay outcome access.** The
Instruct template then rejected source histories containing parallel tool
calls; the evaluator had scored one task and still emitted no result. A shared
offline/live adapter now preserves every call and matching adjacent result in
source order, parses JSON argument strings into the object form required by
the official template, and serializes each parallel group as consecutive
single-call/result pairs. Malformed or incomplete groups fail closed. This is
a deterministic model-interface correction; no gate, split, timing decision,
or tool-event order changes.

**Frozen arms.** Timing is fixed 5-second deadline versus the existing robust
command clock fitted only on the 50 profile tasks. Restore is reactive at the
next LLM arrival versus the existing exact pre-restore stopping rule fitted on
the same profile. This gives a 2x2 attribution: deadline/robust timing by
reactive/pre-restore. No static demo trigger table, target-call outcome, raw
future trace span, or artificial `5000 * rho` restore timer is allowed.

**Offline advance gate.** Report, without combining them into an arbitrary
scalar, released GiB-seconds, critical-path transfer stall, changed gaps and
tasks, and concentration by command family. Robust/reactive advances only if
it releases strictly more GiB-seconds than deadline/reactive with no greater
total critical-path stall, and changes at least 20 tasks. Pre-restore enters
the live 2x2 only if it changes at least 20 tasks and hides at least 5% of the
same timing arm's reactive reload stall; its early-residency GiB-seconds remain
an explicit cost. Exact realized durations are scorer-only. Any oracle is
reported separately.

**Live gate.** First run a bounded plumbing smoke. A development A/B is valid
only under real connector transfers and unmodified tool-event order, with
actual trigger/offload/restore timestamps, bytes and blocks, co-tenant
admission while blocks are free, early-residency block-time, paired next-turn
TTFT/program JCT, co-tenant tail latency, and exact request/output parity. If
the selected workload produces no offload-caused co-tenant admission or
preemption change, stop: a latency-only difference cannot support a scheduling
claim. A main candidate must reduce mean completed-program JCT by at least 5%
versus deadline/reactive, preserve outputs and terminal status, and not worsen
p99 TTFT by more than 5%. Fresh confirmation remains required for a paper
claim.

**Result and decision.** The formal artifact contains 50 profile tasks, 277
disjoint replay tasks, 13,406 tool calls, and 12,771 scored gaps. The robust
clock released 116.507 additional GiB-seconds over deadline/reactive, but added
1,290.791 ms of critical-path stall; it changed 90 tasks but failed the
no-higher-stall gate. Deadline and robust pre-restore changed only 16 tasks and
hid 110.962 ms, respectively 0.118% and 0.116% of reactive reload stall, far
below the five-percent gate. The 621/625 reported plans produced only 27 actual
starts; two partially hid reload and 25 completed too early under the frozen
conservative accounting. All four gates are NO-GO, so the live 2x2 is not
implemented. Artifact:
`analysis/results/gpu-tool-gap-actions-a100-instruct-20260809/result.json`.

The mechanism is a narrow-boundary failure, not absence of an action. Robust
timing moved 166 already-triggering gaps slightly earlier and added five
offloads near the five-second boundary. The earlier long calls gained capacity,
but the five boundary mistakes paid enough D2H/H2D stall to dominate. In
contrast, deadline/reactive itself offloaded 644 gaps across 218 tasks and
released 37,151.024 GiB-seconds at 94,231.430 ms total transfer stall. That
absolute value motivates the separately frozen action-existence test below;
it does not rescue either failed predictive increment.

### 5.2 Frozen live action-existence A/B

**Question.** Under real A100 memory pressure, does the causal observation
"this revealed tool call has survived five seconds" create useful scheduling
capacity, independent of the failed profile clock and pre-restore?

**Arms and workload.** Compare `keep` against `deadline/reactive` on the full
277-task development-exposed workload, seed 0, load 8, Llama-3.1-8B-Instruct,
and the measured A100-80GB staged-transfer path. `keep` retains complete prompt
blocks until the next turn. `deadline/reactive` starts D2H only after a
causally revealed tool call survives the fixed 5,000 ms deadline and restores
at the next request. No robust clock, pre-restore, Continuum, ThunderAgent,
task subset, changed tool timing, or outcome-based task order is allowed.

The primary order is fixed ABBA: `keep`, `deadline`, `deadline`, `keep`. Each
cell replays all 277 programs in the same manifest order. This is 1,108
program-cell executions, 9,805,672 completion tokens, and a 2.965-hour
recorded-gap lower bound before generation and prefill. A bounded action-positive
smoke may use subset flags solely to validate plumbing and estimate wall time;
its timing is never evidence.

**Action-causality evidence.** Every cell records exact request/program joins,
output tokens and terminal status. The deadline arm additionally records
monotonic D2H/H2D start and completion, the exact old-prefix blocks freed and
free-pool delta, and every newly admitted request's allocated block IDs. A
causal reuse exists only when an owner's blocks-free event precedes a different
program's admission, the block-ID sets intersect, and that admission precedes
the owner's restore start. Transfer completion alone is not capacity reuse.

**GO/NO-GO gate.** Pair the first `deadline` cell with the preceding first
`keep` cell and the second `deadline` cell with the following second `keep`
cell. All four cells must complete with identical request sets, input prompts,
output token IDs, finish reasons, and program terminal status. Each deadline
repetition independently requires at least 20 distinct owner programs with a
causal block-reuse event; counts are never pooled across repetitions.

For each deadline repetition, define the affected co-tenant cohort as the
distinct admitted request keys `(program_index, turn_index)` appearing in its
causal reuse events. The paired keep cohort is exactly those same request keys
in that repetition's paired keep cell, regardless of whether keep records an
admission event. Compute p99 by the repository's existing linear-interpolation
percentile. In each pair separately, deadline must not exceed 1.05 times keep
for either all-request p99 TTFT or affected-cohort p99 TTFT.

Average each program's JCT across the two repetitions of its arm, then compute
the mean paired program delta. Deadline must reduce that mean by at least 5%
relative to keep, and its mean JCT must be lower in each of the two repetition
pairs separately. Report absolute JCT, TTFT, transfers, bytes,
freed/reused block-time, reuse owners and affected request keys, plus every
per-repetition delta. Any failed validity, per-repetition action-causality,
effect, direction, or tail gate stops this action without threshold or
task-selection changes. This remains development evidence; a paper claim
requires fresh confirmation.

**2026-08-09 preflight amendment.** The allowed two-program smoke completed 47
requests and three offload/restore pairs but observed no exact co-tenant block
reuse. A prompt-only diagnostic found at most 294,328 tokens within each fixed
eight-program chunk, versus 459,392 KV tokens. This is not a simultaneous-demand
upper bound: slow programs can overlap later chunks, and generation KV is not
included. Section 5.2 remains unexecuted and unresolved; neither a method nor a
load-8 operating-point verdict follows from this smoke.

### 5.3 Frozen high-concurrency action A/B

This separate development experiment keeps every Section 5.2 arm, workload,
instrument, ABBA order, validity check, and GO/NO-GO threshold unchanged, but
sets load to 32. This exploratory operating point was selected before any
load-32 outcome because it admits four times as many active programs; as a
prompt-only diagnostic, all nine fixed 32-program chunks sum above A100 KV
capacity (644,808--912,306 tokens). This raises the chance of real pressure but
does not assert it. The four cells still execute 1,108 programs and 9,805,672
completion tokens; their recorded-gap lower bound is 0.741 hours before
generation and prefill.

A non-evidentiary first-32-program smoke may run at most 40 turns. It
authorizes the formal ABBA only if at least one exact event satisfies offload
completion, actual block release, different-program allocation of an
intersecting block ID, and allocation before owner restore. It must also finish
all submitted requests and complete every observed restore. Failure stops
without changing workload order, task selection, capacity, or policy.

The smoke passed, but the formal first `keep` cell exposed an invalid control
at this operating point. The cell reached 12 free blocks and then produced no
scheduler event for 543.418 seconds, longer than the largest active recorded
gap (300.122 seconds), while GPU utilization was zero and the engine thread
busy-spun. All 32 resident prefixes had no expiry. Because this control held
finished blocks with positive references, a waiting next turn could not obtain
the blocks required to run and therefore could not consume the prefix that
would release those references. The cell was terminated without a result.
Section 5.3 consequently fails its four-complete-cell validity requirement; no
performance number or deadline-action verdict is drawn from it. The evidence
is frozen in
`analysis/results/gpu-tool-gap-actions-a100-instruct-20260809/load32-hard-pin-stall.json`.

### 5.4 Frozen stock-cache action A/B

This is a separate development experiment frozen before observing any
stock-cache outcome. It changes only the invalid Section 5.3 control. `cache`
uses vLLM's enabled prefix cache without manual retention: a finished request
returns its blocks with reference count zero, so a matching next turn can hit
them while pressure may evict them and force recomputation. It performs no
D2H/H2D transfer. `deadline/reactive` is unchanged: causally revealed tool
calls surviving 5,000 ms trigger D2H, and the next matching turn restores from
host.

The workload remains all 277 development-exposed programs in manifest order,
seed 0, load 32, Llama-3.1-8B-Instruct, the same A100-80GB staged-transfer
configuration, and ABBA order `cache`, `deadline`, `deadline`, `cache`. The
request/prompt/output/terminal-status validity checks, exact causal block-reuse
definition, two independent 20-owner action gates, per-pair 1.05x overall and
affected-cohort p99 TTFT ceilings, per-pair JCT direction gates, and aggregate
5% mean program-JCT reduction gate are exactly those in Section 5.2. No robust
clock, pre-restore, subset, changed tool timing, capacity change, or
outcome-dependent task order is allowed.

Before the formal ABBA, one non-evidentiary `cache` smoke uses the same first 32
programs and at most 40 turns. It must complete the same 1,090 submitted
requests with no failed program and no KV transfer. The already completed
deadline smoke supplies only unchanged-path plumbing evidence: 56 exact causal
reuse events from 4 owner programs and every observed restore completed. If the
cache smoke fails, stop. If it passes, run the four formal cells from scratch;
any failed Section 5.2 validity, action, tail, direction, or effect gate stops
without tuning. This remains development evidence and needs fresh confirmation
for a paper claim.

**2026-08-09 thermal preflight amendment, before any completed formal cell.**
The initial `cache-first` process at the A100's default 300 W limit reached the
documented 85 C maximum operating temperature and repeatedly reported software
thermal slowdown. It emitted no cell result and is excluded. While that same
non-evidentiary load remained active, lowering the fixed power limit to 250 W
reduced the device to 74--76 C within four minutes, with both software and
hardware thermal-slowdown flags inactive. The formal ABBA therefore restarts
all four cells from scratch at 250 W. One-minute telemetry records power limit,
draw, temperatures, SM clock, and both thermal flags; any formal sample with a
thermal-slowdown flag makes timing comparisons invalid. Workload, order,
policies, outputs, and every Section 5.2 gate remain unchanged. The default
300 W limit is restored after the run.

### 5.5 Frozen fixed-trajectory CPU/GPU bridge

**Question.** Can the existing exact-tool replay charge real shared-GPU Llama
work without letting generated text change the recorded trajectory, and does
static higher agent concurrency leave enough physical headroom to justify an
observation-driven admission scheduler?

Each source `llm_call` sends its recorded `messages_in` to one shared local
Llama-3.1-8B-Instruct vLLM server with temperature zero, `ignore_eos`, and the
recorded completion-token count. The generated token IDs and TTFT are recorded
but the text is discarded; the provider returns the source `raw_response`, so
the existing OpenClaw runner executes the exact source action IDs, tool-call
IDs, arguments, order, and task-container state. This is fixed-trajectory
shadow inference, not autonomous-agent correctness. The recorded completion
count is workload specification and is never policy input.

The only authorized first run is a non-evidentiary smoke over the first eight
SQLGlot100 manifest tasks, once at maximum active-task counts four and eight.
Both use fresh two-core-capped task containers, the same cached image digests,
the same fixed task order, required eBPF finalization, one otherwise-unshared
A100 at 250 W, and a fresh vLLM process. The smoke passes only if every expected
LLM call emits exactly its requested token count, every exact source action is
emitted in order, all tool arguments match, all telemetry is valid, every
container reports the two-core cap, and there is no framework error, OOM, or
thermal slowdown. Smoke timing is not evidence.

The four-versus-eight contrast is explicitly a **static concurrency ceiling**,
not an observation or prediction contribution. The opt-in staged replay path
now prepares every task before one common ready time, admits tasks FIFO under a
single active-task cap, and records admission wait and ready-to-terminal JCT.
Its 2026-08-10 first-eight cap-four plumbing smoke exited zero: all eight
container startups ended 100--129 ms before common ready, reconstructed maximum
active concurrency was four, all 480 actions and 43,439 requested completion
tokens were preserved, source terminal-class differences were zero, and 161 of
162 eBPF calls were eligible. The one withheld call was the already-known
SQLGlot-3799 unexecuted static branch. This validates the harness only; timing
is non-evidentiary.

The formal static comparison then ran in ABBA order `4, 8, 8, 4`, with a fresh
vLLM/KV state, identical excluded warm-up, and fresh pinned-image containers in
each cell. All four cells were valid: each completed eight tasks and 480 exact
source actions, returned all 43,439 requested shadow tokens, enforced the
declared active-task cap and two-core container cap, retained the same 161/162
eligible eBPF coverage, and recorded no OOM or thermal slowdown. Byte-identical
tool output was not required because replay output never enters the shadow
prompt.

Cap eight lowered common-ready makespan in both pairs, from 638.08 to 567.50 s
and from 647.23 to 570.80 s. It lowered pooled mean ready-to-terminal JCT by
21.77%, and paired p95 JCT ratios were 0.786 and 0.799. The registered decision
still failed because paired p99 LLM TTFT ratios were 1.262 and 1.131, above the
1.05 ceiling. This was not one outlier: cap eight increased TTFT for 188/244
and 170/244 aligned requests. For the same 12,457-token SQLGlot-3820 request,
TTFT was 603/597 ms with two or three overlapping LLM calls at cap four, versus
1,107/1,081 ms with eight overlaps at cap eight. The measured result is a
completion-time versus LLM-tail tradeoff; shared-GPU contention is the likely
mechanism, not a directly observed cause.

The excluded warm-up is one non-streaming chat-completions request after vLLM
readiness and before replay: user text `Reply with exactly one word: ready`,
temperature zero, seed zero, `max_tokens=32`, and `ignore_eos=true`; its request
and response are retained outside the replay output. Pairing is fixed as cells
one/two and four/three; the aggregate JCT comparison pools all 16 task-cell
observations in each arm.

The frozen decision tree required every static gate to pass before the proposed
phase-aware admission action, so this run does not authorize that experiment.
It closes only unconditional cap-eight promotion under this protocol; it does
not invalidate the staged replay harness or erase the measured JCT gain. A
future open amendment may preregister a tail-aware action on fresh tasks, but
cannot reinterpret this static result as a GO. Predictor contribution remains
unclaimed until another pre-outcome protocol names a prediction-dependent
action. All SQLGlot evidence in this section is development-exposed; the full
machine-readable result is in
`analysis/results/fixed-trajectory-static-ceiling-20260810/result.json`.

### 5.6 Phase-aware shadow-request admission — development NO-GO

This experiment was registered before implementation or outcome access and
does not alter the Section 5.5 static NO-GO. The control admitted four tasks
and four shadow requests; the candidate admitted eight tasks while retaining a
four-request cap. The host acquired one cross-process slot immediately before
each real HTTP request, released it in `finally`, and charged the wait to
end-to-end TTFT. No prediction, task identity, tool property, future duration,
or vLLM-internal state affected admission.

The fixed development ABBA was `task4/llm4, task8/llm4, task8/llm4,
task4/llm4` on the already exposed SQLGlot first eight. Every cell preserved
all 480 actions, 244 LLM calls, 236 tool calls, and 43,439 requested/returned
completion tokens. Exact source/replay action identity held for all tasks;
request occupancy peaked at four and returned to zero. Each cell had 161/162
eBPF calls eligible, with the same SQLGlot-3799 `unmatched_static_clause`
withheld fail-closed. All 32 task executions succeeded with valid collection,
two-core containers, no telemetry-integrity failure, no OOM, and no thermal
slowdown.

The candidate reduced common-ready makespan by 3.541% and 9.063% in the two
pairs and reduced pooled mean task JCT by 13.465%. Its paired p95 task-JCT
ratios were 0.883 and 0.839. It nevertheless failed the frozen tail gate:
p99 end-to-end TTFT ratios were 22.231 and 15.984, versus the allowed 1.05.
Candidate request-slot wait p99 was 16.717 s and 11.490 s, while server p99
TTFT remained 0.753 s and 0.769 s. Host admission queueing therefore dominates
the tail failure; the smaller server-service movement does not explain its
magnitude.

The action redistributes completion time rather than improving every task: the
first four manifest tasks became 9.97%--30.31% slower, while the later four
became 23.82%--43.56% faster. The throughput/JCT opportunity is retained as
mechanism evidence and the implementation remains available, but the frozen
joint gate is **NO-GO**. No alternate slot count, task order, or exposed subset
may be tried, and this result does not authorize fresh confirmation or a
prediction contribution claim.

A pre-outcome metadata-only audit had reserved the following 16 SQLGlot IDs.
They remain untouched: no patch, result, or outcome field was displayed or
used.

```text
tobymao__sqlglot-884   tobymao__sqlglot-1367
tobymao__sqlglot-1083  tobymao__sqlglot-1944
tobymao__sqlglot-1825  tobymao__sqlglot-1744
tobymao__sqlglot-1394  tobymao__sqlglot-1340
tobymao__sqlglot-960   tobymao__sqlglot-570
tobymao__sqlglot-1325  tobymao__sqlglot-772
tobymao__sqlglot-2226  tobymao__sqlglot-1089
tobymao__sqlglot-1670  tobymao__sqlglot-1782
```

The failed development gate means these tasks must not be collected or opened
for this action. The full machine-readable result is in
`analysis/results/fixed-trajectory-phase-aware-admission-20260810/result.json`.

### 5.7 Predictive tool-gap loan — insufficient activation

The frozen six-cell `Fixed, Feedback, Predictor, Predictor, Feedback, Fixed`
run completed on the registered eight SQLGlot tasks. Every cell preserved 664
actions, 336 LLM calls, 328 tool calls, 50,972 requested and returned completion
tokens, exact source tool identity and arguments, two-core containers, valid
eBPF collection with zero loss or integrity errors, and no logged OOM or
thermal slowdown. The 16 IDs reserved in Section 5.6 remain untouched.

Both Predictor cells made one early loan, but it was the same SQLGlot-3765
`make test` action in both repetitions; the frozen gate required four distinct
actions. Its hard latency bucket had a 30 s lower edge against causal budgets
of 11.636/11.589 s, but the tool actually ended after 4.242/4.377 s. It
therefore would not have triggered feedback. Relative to paired Feedback, the
action admitted task 2619 only 0.76/0.21 s earlier; FIFO borrower remapping
admitted task 3333 56.11/55.04 s earlier.

Performance is descriptive only. Predictor makespan was 2,157.268 s versus
2,108.062 s Feedback and 2,152.259 s Fixed in pair one, then 1,012.330 s versus
2,154.300 s and 2,155.777 s in pair two. Pooled mean task JCT was 603.629 s,
11.414% below Feedback and 26.835% below Fixed, but pairwise p99 end-to-end TTFT
ratios versus Fixed were 1.152 and 1.079, both above 1.05. The pair-two outlier
was dominated by two task-2337 pytest calls that exited after 37.79/26.28 s
instead of timing out near 600 s as in the other five cells; six tool calls
changed terminal class across cells. Terminal-class identity was not a frozen
validity condition, so registered validity passes, but these numbers cannot
identify a causal performance effect.

The status is `insufficient_action_activation`, not GO or performance NO-GO.
Do not tune the rule, reuse another exposed subset, or open fresh confirmation.
Formal execution took 12,023 s, the smoke 2,180 s, and the six formal GPU traces
integrated to 381.619 Wh; host energy was not instrumented. Full evidence is in
`analysis/results/predictive-tool-gap-loan-20260810/result.json`.

### 5.8 Static command-survival action — development NO-GO

Commit `d445b4c` froze the CPU-only evaluator, arms, and gate before the full
outcome was read. It used 2,605 eligible clauses from 81/82 telemetry-valid
SWE100 tasks as leave-target-repository-out public evidence, then replayed
8,273 gaps and 5,799 exec calls from 177 telemetry-valid SWE277 tasks with
whole-task-final causal updates. Prediction-time agent and GPU costs were zero;
the scorer reused the recorded gap trajectories and measured A100 KV transfer
costs.

The primary expected-Pareto arm made 26 early decisions. Relative to five-second
feedback it released 203.424 additional GiB-seconds, but added 1,450.528 ms of
critical-path stall and changed only 15 tasks, failing two of three frozen gate
conditions. A realized `exec > 5 s` oracle released 4,425.676 additional
GiB-seconds, reduced stall by 774.066 ms, and changed 136 tasks, so the action
space remains open. The observed failure is evidence quality: ten primary
actions actually ended within five seconds; 25/26 used composed
`shell_execution_graph` evidence, and every selected history assigned survival
probability one. Fast already-satisfied installs and narrow pytest selections
are concrete counterexamples. This concentration does not by itself prove
composition is the sole cause; environment state and requested work are also
plausible missing variables. Do not tune the Pareto rule on this exposed result.

### 5.9 Complete-work and causal-state attribution — development NO-GO

Commit `5ebdc45` froze exact complete-command, exact apt/pip/pytest work, and
exact work-plus-causal-state histories before outcomes. The first invocation
stopped before producing an artifact because the evaluator required raw terminal
tool actions with no following LLM gap to appear in `TraceProgram`. Commit
`65e3fcf` fixed only that alignment: the modeled prefix remains identity-checked,
two terminal raw actions in each population are excluded and counted, and no
method, action, population, or gate changed before the sole completed run.

The run used 2,435 exec observations from 82 SWE100 tasks as
leave-target-repository-out public evidence and causally replayed 8,273 gaps and
5,799 exec calls from 177 SWE277 tasks. Local histories settled only after a
whole task. Prediction-time agent and GPU costs were zero; the CPU evaluator ran
95.820 seconds. Relative to five-second feedback, exact command released
668.981 additional GiB-seconds but added 1,236.944 ms stall across 80 tasks.
Work signature released 794.288 GiB-seconds but added 1,848.373 ms across 94
tasks. Work plus state was bit-identical to work signature: 26 queries retrieved
different histories and one differed in availability, but none crossed the
binary action boundary. Every learned arm therefore failed only the registered
no-added-stall condition. Their task-bootstrap stall intervals were also wholly
positive: [480.157, 2,146.915] ms for exact command and [805.820, 3,059.905] ms
for both work arms.

A post-outcome read-only attribution localizes the failure. The 95 work-signature
actions whose commands actually survived five seconds added 760.469 GiB-seconds
with zero stall. The 12 false positives added only 33.819 GiB-seconds but all
1,848.373 ms of stall. Apt-family rows supplied about 731.4 of 794.3
GiB-seconds with zero stall; the displayed Python-heavy families supplied about
1.683 seconds and `apt-get+which` another 0.080 of the 1.848 seconds of added
stall. Thus the observed problem is not that
early release lacks value: it is that exact and semantic duration transfer still
cannot make a zero-added-stall action from finite, variable histories. The
inference is that the modeled invocation/install state is too weak; this run
does not establish which missing runtime variable would fix it. Do not tune a
support threshold on these exposed outcomes. A future test must preregister
risk-aware uncertainty or abstention and use genuinely fresh task outcomes.

The result records dated trace paths, ordered task counts, omitted terminal
actions, and committed evaluator SHA. The ignored trace directories themselves
have no immutable content digest, so preserving those collection directories is
required for byte-level reproduction.

### 5.10 Robust survival-clock development NO-GO

This protocol was fixed before computing either robust-clock arm. It reuses the
Section 5.9 populations, task order, complete-command/work keys, whole-task-final
updates, five-second feedback control, measured A100 transfer costs, scorer, and
three-part gate. It adds no support or probability threshold. State is excluded
because Section 5.9 showed that it changed zero actions.

The two new arms are exact-command and work-signature robust clocks. For the
selected causal history, the existing `robust_utility_trigger_stats` chooses a
continuous trigger in [0, 5,000] ms only when the full history and every
non-empty leave-one-task-out history unanimously prefer it to every later
trigger; fewer than two independent tasks therefore falls back to feedback.
There is no parent fallback. A second deterministic guard requires that the
same full and leave-one-task-out histories predict strictly positive released
GiB-seconds and non-positive stall versus five-second feedback at that trigger.
Otherwise the arm waits five seconds. Current exact/work immediate actions and
the realized-survival oracle remain attribution controls.

The primary work-robust arm advances only if it releases strictly more
GiB-seconds than feedback, adds no aggregate critical-path stall, and changes at
least 20 tasks. Exact robust uses the same gate as an ablation. If both pass,
prefer exact unless work strictly Pareto-dominates it on release and stall. A
failed primary closes this robust representation/action combination on exposed
SWE without trying thresholds or alternate trigger rules. A passing development
arm only authorizes a separately frozen evaluation on genuinely fresh task
outcomes; it is not confirmation itself.

**Result and decision.** The formal development run used the frozen 82-task fit
population and 177-task replay population. Both robust arms chose the
five-second feedback time for every available query, so each was bit-identical
to feedback: zero additional GiB-seconds, zero added stall, and zero changed
tasks. Both therefore failed the release and 20-task activation gates. The run
took 113.125 CPU seconds and used no GPU or prediction-time agent calls. No
fresh split was opened.

A post-outcome read-only attribution separates three causes. Exact history had
255 queries with fewer than two independent tasks and 905 utility fallbacks;
work history had 173 thin-history queries and 1,206 utility fallbacks. Only two
work queries reached an early raw trigger and were rejected by the Pareto guard,
so relaxing that guard could not meet the 20-task gate. Among the full-history
utility failures, 47 exact and 88 work queries were pure far-tail histories:
earlier and five-second triggers tied on latency utility even though earlier
release had positive memory-time value without added stall. Thus the observed
dominant blocker is an objective mismatch, not task stability or the Pareto
guard. This does not authorize fitting a memory-time trigger on the same exposed
SWE outcomes; that action needs a new preregistration and genuinely fresh data.

### 5.11 Frozen direct memory-time development protocol

This protocol was fixed before loading or scoring the selected PennyLane action
outcomes. The development cohort is every locally complete GPT-5.6 PennyLane
trace: the oldest 15 task IDs are settled fit history and the next 26 are
causally replayed in numeric PR order. Exact trace paths and IDs are frozen in
`pennylane-survival-action-split.json`. These old traces are development-exposed,
but have not been used by the survival-action artifacts above. The separate
PennyLane 16-task warm-up and 16-task validation cohorts in
`offline-tool-semantics-splits.json` remain uncollected and untouched.

The new trigger directly matches the action objective. For exact-command and
work-signature histories separately, require at least two settled tasks. The
candidate trigger is zero when there is no historical duration at or below five
seconds, otherwise the largest such duration. Accept it only when every task's
history predicts strictly positive released GiB-seconds and zero added stall
versus five-second feedback under the measured A100 transfer costs. Otherwise
wait five seconds. There is no similarity, parent fallback, state feature,
support threshold beyond the two-task identifiability requirement, or runtime
agent. KV size uses the Llama-3.1-8B-Instruct chat-template prompt plus the
just-finished recorded completion, retains complete 16-token blocks, and uses
the existing measured A100 transfer profile. The evaluator pins the public
RedHatAI W4A16 mirror at revision `6a426ef8` and verifies that its tokenizer,
chat-template config, and special-token blobs equal the official Meta listing.
Five-second feedback, immediate exact/work actions, the exact-key direct
trigger, and realized survival are controls.

The primary work-signature arm advances only if it releases strictly more
GiB-seconds than feedback, adds no aggregate critical-path stall, and changes at
least 8 of the 26 replay tasks. Exact uses the same gate as an ablation; if both
pass, prefer exact unless work strictly Pareto-dominates it on release and
stall. Failure closes this action on the frozen PennyLane development cohort
without changing the split, trigger, or gate. Passing only authorizes collection
and separately frozen evaluation of the untouched PennyLane 16+16 cohorts.

**Result and decision.** The frozen run causally replayed 1,383 gaps and 1,024
exec calls from 26 tasks after 637 fit observations from 15 earlier tasks. The
primary work-signature arm released 1,438.416 additional GiB-seconds but added
3,611.076 ms stall: 86 early calls were truly longer than five seconds and 19
were not. It changed all 26 tasks but failed the no-added-stall gate, so the
registered result is `development_no_go` with no selected arm.

The exact-command ablation passed its own frozen arm gate: 20/20 early calls
were truly long, yielding 389.756 additional GiB-seconds with zero added stall
across 11 tasks; its task-bootstrap release interval was [204.668, 598.219].
However, the predeclared exact-immediate control also had zero added stall and
released more, 660.664 GiB-seconds across 16 tasks. Thus this run supports safe
same-repository exact recurrence, not incremental value from the new
task-Pareto trigger. Exact task-Pareto captured 3.4% and exact-immediate 5.8% of
the oracle's 11,314.367 GiB-second incremental headroom, so neither yet
establishes scheduling utility.

A post-outcome read-only case attribution found that retained work false
positives pooled `tests/drawer` with unrelated directory suites and pooled
unrelated two-node selections by target shape. This supports semantic scope
conflation as one failure mechanism, but the artifact retains only 13 of 19
false-positive examples and does not prove that it caused all aggregate stall.
Do not add scope rules on this exposed cohort. The evaluator used zero GPU and
zero prediction-time agent calls and completed in 197.842 seconds wall time.

### 5.12 Frozen first-loan actionability check

This development check was fixed after the Section 5.11 aggregate result was
visible but before inspecting task-level first-loan outcomes. It tests the
stronger exact-immediate control because that arm released more memory-time
than exact task-Pareto with the same zero added stall. This is an explicit
post-development choice, not a confirmation claim.

The 15 Section 5.11 fit tasks are the only history. All 26 replay tasks query
that fixed settled history; replay tasks never update one another, matching a
concurrent arrival wave. Compare five-second feedback with
exact-immediate-plus-feedback. Exact-immediate acts at tool start only when the
unchanged `expected_early_action` rule accepts the complete raw command from
fit history; every other exec falls back to five seconds. There is no work
signature, prefix, state, threshold, or agent.

For each replay task, order actions on its recorded timeline and keep only the
first loan each policy would emit. A loan becomes usable after the measured
A100 swap-out completes, not at the trigger instant. The scorer retains the
recorded next-request time and charges measured swap-in stall. If a policy
emits no loan, its next admission opportunity is the recorded task-terminal
capacity release. Report the number of tasks whose first usable loan advances,
total/median/p95 admission advance under an always-nonempty waiting queue,
freed-KV sizes, and the full
pairwise coverage of other replay tasks' first recorded prompt-plus-completion
KV demand. Coverage is descriptive: it cannot rescue an inactive or unsafe
policy, and admission advance is an upper bound rather than a completion-time
claim.

The actionability gate passes only if exact-immediate-plus-feedback advances
the first usable loan for at least 8 of 26 tasks and its aggregate critical-path
stall is no higher than feedback. A pass authorizes a separately frozen fresh
live A/B with byte-aware admission and mean completion/makespan metrics; a
failure stops this exact action without changing the history, trigger, task
order, or gate. No fixed SLO is assumed.

**Result and decision.** The reviewed formal evaluator used the fixed 15-task
fit history and scored 26 replay tasks without updates. Exact-immediate advanced
the first physical loan for exactly 8 tasks, meeting the frozen activation
threshold. Every advance was 5,000 ms, for a 40,000 ms admission upper bound;
candidate and feedback aggregate stall were identical at 3,763.355 ms. The
gate therefore passes and authorizes a separately frozen fresh live test. The
run took 212.900 s wall time with zero GPU and zero prediction-time agent calls.

All eight changed first loans were recurring `apt-get update && apt-get install`
commands. Six used the same `python3-pip` command with six fit observations;
the other two setup commands had one exact fit observation each. They freed
between 1.324 and 4.000 GiB, enough for every other replay task's first recorded
prompt-plus-completion demand: 200/200 lender-borrower pairs fit. Independent
review reproduced the split, counts, advances, stalls, capacity edges, gate,
and zero overlap with the untouched 16+16 IDs.

The pass is about action activation, not task completion. A post-outcome upper
bound under the existing four-foreground/four-waiter loan runtime is at most
four distinct 5 s borrower advances, or 2.5 s off the eight-task mean JCT even
if every foreground task hits and every advance propagates fully. The live
action must therefore account for bytes and should measure waiting-request
admission/TTFT directly; an expensive task-JCT run alone is unlikely to resolve
such a small ceiling.

### 5.13 Frozen PennyLane pairwise CPU-work ceiling

This exposed-development branch asks whether the long PennyLane commands make
CPU-idle backfill actionable under a workload unlike SQLGlot. Of the 26 replay
tasks in Section 5.11, exactly 15 have evidence-valid command telemetry; the
other 11 are excluded solely by the pre-existing telemetry validity gate. The
15 tasks contain 570 exec commands with complete eight-core CPU timelines. No
reserved PennyLane task is read.

Preliminary diagnosis fixed the next question. With observed RSS as a hindsight
safety filter, unrestricted FCFS backfill reduced mean task completion by
23.484% and makespan by 17.191%, but inflated aggregate command service by
9.553%. Hindsight shortest-safe selection still inflated service by 8.102%.
Canonical latency-bucket-5 plus non-High peak CPU reduced completion only
0.940%. A continuous hindsight score based on duration and average CPU did
contain 5%/5% operating points, but fit-history exact commands covered only 16
of 330 RSS-safe commands. The unchanged work signature collapsed 196 covered
commands to 13 scores; its apparent top-160 pass split a tied score by task ID
and is invalid. Thresholding whole ties produced no pass. The repository's
fixed CacheWise TF-IDF representation plus default Ridge also produced no
5%/5% point. These are diagnostics, not candidate-selection claims.

The remaining mechanism question is pairwise compatibility. For every command,
mean CPU demand is total clipped CPU core-seconds divided by recorded command
duration. At each backfill decision, the oracle admits the first RSS-safe ready
command whose mean demand plus the running foreground command's mean demand is
at most the physical eight-core capacity. It otherwise follows the unchanged
strict-priority simulator: foreground work has first claim, speculative work
uses only residual CPU, task programs and command order are unchanged, and all
CPU work is conserved. Full realized profiles and observed RSS make this a
hindsight action-space ceiling, not a predictor.

Compare Serial-8, unrestricted RSS-safe FCFS, and pairwise-mean FCFS on all 15
valid tasks in one concurrent task-start wave while preserving every task's
recorded delay before its first command becomes ready. The primary gate
requires pairwise-mean FCFS to reduce mean task completion by at least 5%, keep
aggregate service inflation at or below 5%, and introduce zero logical/physical
capacity or CPU-work violations. A pass authorizes a separately frozen causal
predictor using fit-task command CPU work plus runtime foreground feedback. A
failure closes CPU-idle backfill for the present PennyLane action model. It does
not amend any earlier survival, work-signature, or SQLGlot result.

Pre-outcome correction (2026-08-10): independent review found that the proposed
32 list shuffles could not change scheduling order because all recorded
first-command ready times are distinct and the simulator correctly sorts by
ready time before its tie-break. No formal outcome had been run or read. Rather
than invent an external arrival process or detach first-command delay from its
task, the frozen evaluation therefore uses the single physically defined wave
above and removes the vacuous every-order gate.

**Result and decision.** Pairwise-mean FCFS reduced mean task completion by
16.462% and makespan by 16.730%, with zero modeled capacity or CPU-work
violations, but aggregate command service increased by 6.321%. This exceeds
the frozen 5% ceiling, so the present pairwise-mean action model is NO-GO and
does not authorize a causal predictor. Unrestricted FCFS was faster but worse:
23.484% completion reduction at 9.553% service inflation. The scored population
was the fixed 15 valid tasks and 570 commands; 11 pre-invalid tasks were
excluded and no reserved task was read.

Post-outcome diagnosis attributes the miss to phase bursts hidden by the mean.
Of 268 admitted speculative commands, 101 stretched; eight commands account
for half of the 1,981.260 added service seconds. For the 101 stretched pairs,
the median sum of whole-command mean demand was 6.396 cores, while the median
sum of realized peak demand was 11.813 cores; every pair's realized peak sum
exceeded eight cores. This is mechanism evidence for testing a fixed exact-peak
ceiling, not a retrospective change to the failed gate.

Dependency prewarming is not part of this experiment. The repeated
`python3-pip` setup gain was real (about 11--12 s per invocation), but the task
images already expose `/opt/conda/bin` while the collector deliberately
replaces the image PATH. Treat that setup recurrence as a runtime confounder,
not a research claim; changing PATH would require a separate trace-regime
decision.

### 5.14 Frozen PennyLane pairwise peak ceiling

Section 5.13 failed because whole-command mean CPU hid realized bursts. Before
building another predictor, test whether a static per-command peak can express
a useful and safe action. Reuse the same fixed 15-task, 570-command population,
single concurrent task-start wave, observed RSS safety filter, strict-priority
simulator, and Serial-8 baseline. No reserved task is read.

The exact-peak oracle assigns each command the maximum instantaneous core
demand in its recorded CPU-work profile and admits a pair only when foreground
plus candidate peaks sum to at most eight cores. The canonical-bucket oracle
first maps that same peak to the frozen CPU classes in Section 2, then uses the
class upper bounds Low=2, Medium=4, and High=8 cores for the identical pairwise
test. The latter tests the expressiveness of the current predictor output, not
prediction accuracy. Both peak values and RSS safety remain hindsight oracles.

For each oracle independently, GO requires at least 5% lower mean task
completion than Serial-8, at most 5% aggregate command-service inflation, and
zero logical/physical capacity or CPU-work violations. If exact peak fails,
close static pairwise command summaries. If exact peak passes but canonical
buckets fail, retain the mechanism result but do not build the current
three-class predictor action. Only both passing authorizes a separately frozen
causal evaluation using fit-task predictions and runtime foreground feedback.
No peak threshold, bucket boundary, candidate order, or service gate may change
after the outcome is read.

**Result and decision.** Exact-peak admission had effectively zero service
inflation and no violations, but reduced mean task completion by only 2.176%
and makespan by 3.525%. Canonical 2/4/8 bucket upper bounds were more
conservative: 1.131% completion and 2.372% makespan reduction, also at
effectively zero service inflation. Both miss the frozen 5% completion gate,
so this experiment does not authorize a peak/bucket predictor. Together with
Section 5.13, the result exposes a static-summary tradeoff: whole-command mean
admits useful work but hides bursts, while whole-command peak preserves service
but assumes worst phases coincide. The next mechanism question is therefore
within-command phase alignment, not another scalar threshold.

### 5.15 Frozen PennyLane phase-shape ceiling

The exact-peak result assumes two commands' worst CPU bursts coincide. Test the
smallest alternative that removes only that assumption. Reuse the Section 5.14
population, single task-start wave, observed RSS filter, FCFS candidate order,
strict foreground priority, and Serial-8 baseline without changing any command,
delay, or CPU work.

At each backfill decision, the phase-shape oracle aligns the running
foreground command's remaining sampled CPU-work profile with a ready
candidate's profile from its first sample. It admits the candidate only if the
sum of their sampled core demands never exceeds eight cores before either
command finishes. Profiles use the existing 0.5 s telemetry intervals; partial
foreground progress is retained exactly. The check is repeated at every later
admission. This is a hindsight compatibility ceiling that assumes an isolated
profile remains stable when shifted in time, not a causal predictor or a live
feedback policy.

GO requires at least 5% lower mean task completion than Serial-8, at most 5%
aggregate command-service inflation, and zero logical/physical capacity or
CPU-work violations. Exact-peak FCFS remains the fixed scalar control. A pass
authorizes a separately frozen feasibility test for predicting phase envelopes
from fit-task clause and timeline evidence plus locating the foreground phase
from causal runtime feedback. A failure closes this single-start, pairwise
CPU-backfill action under the present PennyLane wave; it does not rule out a
preemptive or differently timed action. No phase smoothing, tolerance,
candidate reordering, or gate may change after the outcome is read.

**Result and decision.** Phase-shape FCFS reduced mean task completion by
10.699% and makespan by 6.680% with effectively zero aggregate service
inflation and no capacity or CPU-work violation. It admitted 301 speculative
commands, versus 196 for exact peak, while preserving all 570 commands and
their recorded work. The frozen gate passes. This authorizes only the causal
prediction-feasibility test above: phase profiles and observed RSS were
hindsight, all 15 scored tasks were development-exposed, and the simulator
assumes isolated sampled demand remains stable after a time shift. It does not
establish predictor accuracy, live interference safety, or cross-repository
generality.

### 5.16 Frozen one-sided causal phase-envelope feasibility

A post-Section-5.15 coverage audit found that raw exact commands and the
tool-specific work key cover only 16 and 114 of 330 RSS-safe replay commands.
The existing generic Clause-KB argv hierarchy covers 302/330 and 277/301 of
the phase oracle's starts. This audit exposed only coverage and a
hindsight-compatibility restriction; it did not evaluate a predicted profile.

Test candidate-profile transfer while leaving foreground future shape and RSS
safety as explicit hindsight controls. Build history solely from the fixed 15
fit tasks. Parse every command with the existing shell parser and form
whole-command signatures from control operators, clause structural context,
binary names, and argv tails. Query the current raw hierarchy in fixed order:
exact, shared prefix depths 4, 3, and 2, then binary. Interpreter paths are
excluded exactly as in Clause-KB. A compound command matches only when every
clause and its shell structure match at the same level.

For the most-specific fit-supported signature, form one deterministic candidate
profile: take the pointwise maximum CPU rate across every matching fit profile
on their absolute 0.5 s timeline, split at all observed segment boundaries, and
end at the longest fit duration. Completed fit profiles contribute zero after
completion. There is no smoothing, normalized-time alignment, support cutoff,
exemplar selection, or replay update. An unmatched command is not eligible for
speculation. If a replay command outlasts its envelope, the predictor supplies
no synthetic tail; any resulting interference is charged by the physical
actual-profile replay and reported.

At each admission, align the predicted candidate envelope with the running
foreground's actual remaining profile and require their summed rate never to
exceed eight cores over the predicted overlap. The unchanged simulator executes
the actual candidate profile, so prediction errors appear as service inflation.
Compare this arm with Serial-8, exact peak, and the full-profile phase oracle on
the same 15 replay tasks and one wave. GO requires at least 5% lower mean task
completion than Serial-8, at most 5% aggregate service inflation, and zero
capacity or CPU-work violations. A pass authorizes only the next test replacing
foreground-future hindsight with causal feedback; a failure closes this
Clause-KB phase-envelope representation. Replay results may not change keys,
envelope construction, candidate order, or gates.

**Result and decision.** The fit-only candidate-envelope arm reduced mean task
completion by 6.707% and makespan by 3.172%, with 0.0835% aggregate service
inflation and no capacity or CPU-work violation. It matched 302/330 RSS-safe
commands and made 287 speculative starts. Forty commands accumulated 26.175 s
of added service; 13 of those outlasted their fit envelope. The frozen gate
passes and therefore authorizes only replacing the actual foreground future
profile. Candidate prediction is causal, but foreground shape and RSS safety
remain hindsight controls; this is not yet a deployable scheduler result.

### 5.17 Frozen two-sided causal CPU-envelope feasibility

Remove foreground-future hindsight without changing the Clause-KB
representation. Build the same fit-only envelopes from the fixed 15 fit tasks
for every matched replay command. The coverage audit found envelopes for
398/570 commands and for the foreground in 250/287 one-sided speculative
starts. These counts are exposed diagnostics, not a result gate.

At each admission, locate the running foreground solely by elapsed wall time
since its observed command start and slice its fit envelope at that offset.
Align the predicted remaining foreground envelope with the ready candidate's
fit envelope from time zero. Admit only when their predicted sum never exceeds
eight cores. If either command has no fit envelope, or the foreground is still
running after its envelope ends, fail closed. Do not consult the foreground's
actual profile index, remaining segment, current replay CPU rate, or future
samples. A command promoted after speculative execution uses the same causal
wall-time position; no hidden correction restores its isolated phase.

The physical simulator still executes both replay profiles, so every two-sided
prediction error is charged as service inflation. Candidate eligibility remains
the observed RSS-safe set; RSS is the sole remaining hindsight admission
control. Compare with Serial-8, the full-profile oracle, and the one-sided
candidate-envelope arm in the same wave. GO requires at least 5% lower mean
task completion, at most 5% aggregate service inflation, and zero capacity or
CPU-work violations. A pass authorizes a separately frozen RSS replacement and
then physical validation; a failure stops this phase-envelope scheduler. No
profile tail, feedback correction, key, order, or gate may change after reading
the outcome.

**Result and decision.** The two-sided arm reduced mean task completion by
4.461% and makespan by 2.868%, with 1.757% service inflation and no capacity or
CPU-work violation. It therefore fails the frozen completion gate and does not
authorize RSS replacement. Relative to the one-sided control, speculative
starts fell from 287 to 116, while added service rose from 26.175 s to
550.804 s and mean completion worsened by 567.716 s. Absolute-time foreground
transfer both rejects useful alignments and misaligns some admitted phases.
Close this phase-envelope scheduler under the present single-wave action model;
do not add a post-outcome alignment or feedback correction on these replay
tasks. This does not negate the 10.699% hindsight phase-shape headroom or rule
out a preemptive/differently timed action.

Post-outcome leave-one-task-out sensitivity supports that mechanism boundary.
The phase oracle exceeded 5% in 15/15 subsets (median 10.912%); the one-sided
fit-envelope arm did so in 14/15 (median 8.455%, minimum 4.929%); the two-sided
arm did so in only 1/15 (median 2.719%, minimum -0.213%). No method or gate was
changed from these diagnostics.

### 5.18 Frozen reactive-current/peak CPU ceiling

Test a different action that makes no foreground-future prediction. Reuse the
fixed 15-task PennyLane replay wave, strict foreground priority, FCFS order,
actual CPU-work profiles, and observed RSS-safe candidate set. A candidate may
be considered only after the running foreground has produced a complete 0.5 s
CPU interval whose end plus the existing 0.14132007875 s observation/update
delay is no later than the decision. Use the latest such foreground rate and
the candidate's exact whole-command peak; admit only when their sum is at most
eight cores. Reevaluate at later simulator events. A foreground command that
started speculatively is ineligible because its earlier observed rate may have
been throttled.

The foreground signal is causal, but candidate peak and RSS safety are
hindsight oracles. Compare Serial-8, pairwise exact peak, full phase shape, and
reactive-current/peak FCFS in the same wave. GO requires at least 5% lower mean
task completion than Serial-8, at most 5% aggregate service inflation, and zero
capacity or CPU-work violations. A pass authorizes replacing candidate peak
with the frozen fit envelope; a failure closes this reactive-current/static-
candidate action. Do not change the sample interval, delay, peak definition,
candidate order, or gates after reading the outcome.

**Result and decision.** Reactive-current/peak FCFS reduced mean task completion
by 3.853% and makespan by 7.358%, with 3.457% service inflation and no capacity
or CPU-work violation. It made 230 speculative starts but missed the frozen 5%
completion floor, so candidate-peak prediction is not authorized. Together
with Section 5.17, this closes both tested causal foreground consumers on the
exposed PennyLane wave: absolute-time envelope transfer is too brittle, while
latest-sample feedback plus a static candidate bound admits work without enough
mean-completion benefit. This does not close runtime feedback generally or a
separately frozen preemptive/differently timed action on untouched tasks.

### 5.19 Zarr RSS-predicted CPU-idle backfill: validation NO-GO

The frozen same-repository experiment fit on 19 development tasks and collected
all ten validation tasks with Codex GPT-5.6 fast, maximum 100 iterations,
concurrency one, required eBPF telemetry, and image cleanup. All ten collections
were valid. The evaluator preserved the frozen split hash, task order, 8-core and
16,000 MB capacities, FCFS action, predictor heads, reservation mapping, and
no-update boundary. It scored 192 physical commands, including 178 prediction
rows and 133 RSS-label-eligible rows. The 12 final tasks were not read.

The primary Task-Aware hard-RSS arm reduced mean task completion from 3,233.662
to 2,180.489 seconds, a 32.569% reduction, and reduced makespan by 13.907%. It
captured more than half of the 51.695% source-bound-oracle reduction, made 81
speculative starts across all ten tasks, conserved CPU work, and added no
service time. These are positive action results within the replay model.

The preregistered verdict is nevertheless **NO-GO** for two independent reasons.
Clause-KB reduced mean completion by 33.683%, so Task-Aware was 1.114 percentage
points worse instead of at least one point better. Task-Aware also produced
three modeled source-bound exposure events instead of zero. No physical or
aggregate simulator capacity violation occurred, but the frozen safety gate
counts the modeled exposures and therefore fails. Final12 remains untouched.

The failure is not explained by lower command-class accuracy. Task-Aware RSS
accuracy was 78.947% versus Clause-KB's 69.173%. It changed 17 reservations: 13
became less conservative and corrected a label error, while four pytest calls
moved from 2,000 to 16,000 MB. Ten of the useful corrections were short final
`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` calls. We infer that these late,
short corrections offered little task-level leverage; the four 16,000 MB pytest
reservations, by construction, disabled backfill during longer calls. This
demonstrates the frozen protocol's intended distinction: better per-command
accuracy can still produce worse scheduling utility.

The three Task-Aware modeled exposure events were shared with Clause-KB. Two
were multiline dependency probes with unavailable RSS source evidence: the
source policy assigned the 16,000 MB full fallback while both predictors
reserved 500 MB. The third was recorded when an editable pip install with a
231 MB source bound was admitted, so that event reflects aggregate overlap, not
a pip underprediction. These are not measured OOMs; they show that a point class
cannot certify the frozen source bound when telemetry is unavailable. The
highest-supported-bucket ablation reduced exposures to two but only reduced mean
completion by 24.419%, so it does not rescue the candidate.

Decision: close this Task-Aware Zarr candidate and do not tune it on validation
or collect final12. Retain the demonstrated CPU-idle backfill opportunity and
the result that action utility differs from classification accuracy. Any
successor must separately freeze an uncertainty-aware reservation or
action-trained objective using other exposed development evidence.

### 5.20 Frozen post-hoc exact-supported demotion development

The validation result exposed that Task-Aware's upward changes hurt the action
while weak baseline evidence caused all modeled exposures. The next candidate
is therefore a new, explicitly post-hoc development policy; it does not repair
or reinterpret Section 5.19.

Start from Clause-KB hard RSS and compute the Task-Aware highest-supported RSS
bucket. The proposed reservation is the lower of those two values; Task-Aware
may relax a reservation but never raise it. A proposed reservation below
16,000 MB is authorized only when the identical complete command has a non-null
RSS label in at least two independent fit tasks and the largest such historical
bucket does not exceed the proposal. Otherwise reserve 16,000 MB. This uses no
target outcome, current-task state, tool name, command literal, or tunable
similarity threshold.

Evaluate one fixed candidate on two exposed cohorts: leave-one-task-out over the
19 fit tasks, and fixed-fit scoring over the ten now-exposed validation tasks.
Use the unchanged Serial-8, source-bound oracle, Clause-KB, capacities, command
profiles, FCFS order, and short-null source policy from Section 5.19. Report the
ungated demotion policy only as the already-exposed mechanism control.

Development GO requires the candidate to pass on both cohorts: at least 5%
mean-completion reduction versus Serial-8; at least one percentage point more
reduction than Clause-KB; lower makespan than Serial-8; at most 5% service
inflation; zero modeled source-bound, capacity, physical-capacity, or CPU-work
violations; and at least 20 speculative starts across five tasks. Do not alter
the support count, evidence rule, mapping, arms, or gate after reading results.
Only a pass authorizes writing a fresh protocol for final12; it does not itself
authorize collecting or reading final12.

**Result and decision.** The candidate failed both exposed cohorts. On the
19-task leave-one-task-out development cohort, it reduced mean completion by
0.864% versus Serial-8, compared with 29.597% for Clause-KB, and made 18
speculative starts instead of 188. On exposed validation10, it reduced mean
completion by 5.607%, compared with 33.683% for Clause-KB, and made 12 starts
instead of 75. It had zero modeled exposure, capacity, physical-capacity, and
CPU-work violations, but missed the frozen improvement-over-Clause and
20-start gates in both cohorts; development also missed the 5% completion
gate.

The mechanism is evidence sparsity, not harmful authorized demotion. The gate
blocked 205/245 proposed sub-capacity development reservations and 89/117
validation reservations. Every authorized demotion was the same harness
completion command, `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat
patch.txt`: 17 development rows and ten validation rows. Exact whole-command
recurrence therefore certifies a ubiquitous low-work marker but does not cover
the varied commands that create scheduling opportunity. The ungated control
retained broad coverage and reached 35.861%/40.262% mean-completion reduction,
but on validation it retained the same three modeled source-bound exposures as
Clause-KB.

Decision: close this exact-supported policy and do not collect or inspect
final12. Retain the causal question exposed by the controls: a useful successor
must distinguish unsafe source-unavailable cases without reducing the ordinary
Clause-KB action set to exact recurring commands. This result does not close
RSS-aware backfill or the demotion-only action; it closes this exact-evidence
safety mechanism.

### 5.21 Frozen post-hoc compound-evidence safety gate

This candidate was designed after Section 5.20 and after inspecting the three
exposed validation overlap events. Two events began with different multiline
dependency probes whose three-clause RSS predictions combined two exact clause
matches with one prefix match; their final command source was unavailable. The
third began with a measured 231 MB editable install while one of those probes
was already running. This diagnosis is development-exposed and cannot support
a claim by itself.

Start from the unchanged Clause-KB hard RSS reservation for every command. If
that prediction is a shell execution graph with more than one clause and any
selected constituent match is not `exact_clause`, replace its reservation with
16,000 MB. Otherwise preserve the Clause-KB reservation exactly. Apply the
reservation symmetrically: such a command cannot start as backfill beside a
foreground command, and when it is the foreground its full reservation blocks
backfill. The decision uses only parsed command structure and fit-derived
Clause-KB provenance available at `BeginCall`; it uses no tool name, command
literal, target outcome, current-task telemetry, or tunable similarity rule.

Evaluate one fixed candidate on the same exposed cohorts as Section 5.20:
leave-one-task-out over development19 and fixed-fit validation10. Keep Serial-8,
the source-bound oracle, Clause-KB, capacities, profiles, FCFS order, and source
policy unchanged. Report one mechanism control that assigns 16,000 MB to every
multi-clause command, so a positive result cannot be attributed merely to
blocking compound commands.

Development GO requires the candidate to pass on both cohorts: at least 5%
mean-completion reduction versus Serial-8; at least 90% of Clause-KB's
mean-completion reduction; lower makespan than Serial-8; at most 5% service
inflation; zero modeled source-bound, capacity, physical-capacity, or CPU-work
violations; at least 80% of Clause-KB's speculative starts; and at least 20
starts across five tasks. Do not alter the guard, controls, or gates after
reading results. Only a pass authorizes a separately reviewed final12 protocol;
it does not authorize collecting or reading final12.

**Result and decision.** The guard removed every modeled exposure and preserved
more action than the exact-command policy, but it remained too broad. On
development19 it changed 75 of 89 sub-capacity compound reservations, retained
113/188 Clause-KB starts, and reduced mean completion by 19.249% versus 29.597%
for Clause-KB. On validation10 it changed 36 of 43, retained 51/75 starts, and
reduced mean completion by 23.029% versus 33.683%. It passed the absolute 5%
completion, makespan, service, exposure, capacity, CPU-work, and task-coverage
gates in both cohorts, but failed both the frozen 90%-of-Clause benefit and
80%-of-Clause start-retention gates.

The all-compound control reduced mean completion by 20.943% on development and
19.173% on validation. Selective provenance guarding therefore helped on
validation but did not uniformly dominate the simpler structural control; FCFS
interactions make additional starts non-monotonic in completion time. The main
failure is coverage: prefix/backoff evidence inside compound agent commands is
ordinary, not a narrow marker for unavailable RSS.

Decision: close this static compound-provenance guard and do not inspect or
collect final12. Do not tune another prefix depth or compound exception on the
same exposed cases. The next question is whether already-collected telemetry
contains a causal in-flight RSS signal that can verify the *running overlap*
instead of rejecting commands before execution. If it does not, specify the
smallest fresh collection needed before implementing that action.

### 5.22 Frozen PennyLane time-varying RSS packing ceiling

The audit after Section 5.21 found that Zarr cannot test a reactive memory
action: its task-container peak RSS is at most 1,257 MB on development and 652
MB on validation, far below the 16,000 MB capacity. The existing PennyLane
replay cohort does contain the required pressure and temporal variation. All 15
evidence-valid replay tasks have task-container RSS samples at approximately
two-second intervals; 13 contain commands reaching at least 2,000 MB. Among
the 44 such commands, the median time-average RSS is 55.3% of sampled peak.
These audit statistics were visible before freezing this protocol and make
this cohort development-exposed.

Before designing another predictor, test whether that variation can change a
memory admission decision at all. Treat each of the 15 evidence-valid replay
tasks in `pennylane-survival-action-split.json` as one job arriving at time
zero, ordered by task ID. Use its complete `resources.json` sample sequence as
a step-held RSS profile and its first-to-last sample interval as fixed service
time. This deliberately ignores CPU contention and command rescheduling: it is
a memory-only hindsight ceiling, not a deployable scheduler or completion-time
claim.

Compare four deterministic FCFS arms at 16,000 MB:

1. `serial`: admit only one task at a time.
2. `static_peak`: admit every ready task whose sampled lifetime peak fits with
   the lifetime peaks reserved by running tasks.
3. `temporal_oracle`: admit a ready task only when its complete future sampled
   profile fits with the remaining sampled profiles of all running tasks for
   their full overlap.
4. `unconstrained`: start all tasks at time zero; this control must exceed the
   capacity at least once or the memory constraint is not action-relevant.

No resampling, interpolation, predictor, Clause-KB rule, threshold tuning, or
new collection is allowed. Report mean task completion, makespan, maximum
concurrency, starts before the first completion, and maximum sampled aggregate
RSS. A capacity violation means sampled aggregate RSS greater than 16,000 MB.
The temporal arm passes only if it reduces mean completion by at least 5%
relative to `static_peak`, strictly reduces makespan, starts at least one
additional task before the first completion, and has zero sampled capacity
violations while `unconstrained` has at least one. Failure closes time-varying
RSS packing on this cohort. A pass authorizes only a separately frozen causal
action using prediction and/or observations; it does not validate the oracle,
open final12, or authorize fresh collection.

**Result and decision.** The ceiling passed every frozen gate. Static lifetime
peak reservation admitted three tasks before the first completion, reached
five-way concurrency, and produced 6,202.222 s mean completion and 18,877.186 s
makespan. The temporal oracle admitted eight tasks before the first completion,
reached eight-way concurrency, and produced 3,174.106 s mean completion and
9,891.043 s makespan: a 48.823% mean-completion reduction versus static peak.
Its maximum sampled aggregate RSS was 15,864.092 MB with zero sampled capacity
violations. The unconstrained control reached 47,523.476 MB and exceeded
capacity at 12,771 sampled transition points, so the constraint is
action-relevant rather than vacuous.

This establishes substantial temporal memory-packing headroom on the exposed
PennyLane cohort, not a scheduler result. The oracle knows every task's future
RSS, fixes isolated task duration, and ignores CPU contention. The next step is
therefore an action-timing audit: determine whether memory rises occur inside
tool intervals and whether a command-boundary controller could reserve them
before they occur. Only that evidence may define a causal Clause/KB admission
candidate.

## 6. Closed directions

- **Lookup structure alone:** trie, lattice, generic argv, pip/pytest semantic
  keys, and command-state variants changed too few action-relevant decisions.
- **Trace-conditioned offline agent:** generated Python, relational spans,
  bounded regex, and a typed catalog found no resource-separating state. The
  final finite-choice arm removed arbitrary code and still returned no useful
  contrast. This result is not weakened or repaired.
- **Docs-only compiler v1:** the frozen generation and SQLGlot development run
  were protocol-valid, but only one of four generated specifications passed
  validation. All 1,792 docs and generic predictions copied Clause-KB: public
  evidence contained no `make`, while all 47 recognized scored clauses were
  the warm-up-exact command `make test`. The protocol is closed without fresh
  collection; it did not identify the documentation representation itself.
- **Task-local last-value state:** repeated command load alternates; a causal
  overlay creates one-command lag rather than stable state.
- **Continuous latency survival:** most apparent gain came from the elapsed-time
  physical floor; command-equal accuracy gain was only 0.229 points.
- **Peak CPU as admission demand:** instantaneous peak does not equal the CPU
  quota required to preserve throughput.
- **Feedback-driven admission pages:** hard pages failed the service-cost gate;
  work-conserving pages failed both effect and service-cost gates. Retain the
  causal feedback implementation as evidence and a control, but do not tune
  this action on exposed SQLGlot.
- **CPU-idle candidate ordering:** exact-duration hindsight selection did not
  improve FCFS idle backfill. Do not build a latency/shortest selector on the
  exposed SQLGlot tasks.
- **Static hard-RSS CPU-idle safety:** a separately frozen short-null amendment
  removed about 78% of conservative exposures but left 2,638/2,702 confirmed
  events. Do not tune reservations or relax null handling again on exposed
  SQLGlot.
- **KV victim selection:** C100 already removes most LRU recomputation, leaving
  insufficient predictor headroom in the current model.
- **Profile-guided GPU tool-gap timing:** the robust clock increased aggregate
  stall, and pre-restore changed too few tasks while hiding about 0.12% of
  reactive stall. Do not implement or tune those arms on exposed replay data.

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
- The 41 collected PennyLane tasks in the Section 5.11 split are
  development-exposed. The separate PennyLane 16+16 cohorts remain untouched
  and uncollected.
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
- Dynamic hard-page feedback admission: `sqlglot50-cpu-feedback-admission-v1/result.json`
- Feedback admission with CPU borrowing: `sqlglot50-cpu-feedback-borrowing-v1/result.json`
- CPU-idle backfill oracle: `sqlglot50-cpu-idle-backfill-oracle-v1/result.json`
- CPU-idle predicted RSS safety: `sqlglot50-cpu-idle-rss-safety-v1/result.json`
- CPU-idle short-null sensitivity: `sqlglot50-cpu-idle-short-null-v1/result.json`
- GPU tool-gap offline gate:
  `../gpu-tool-gap-actions-a100-instruct-20260809/result.json`
- Predictive tool-gap loan:
  `../predictive-tool-gap-loan-20260810/result.json`
- Static command-survival action:
  `../static-survival-gap-action-swe177-20260810/result.json`
- Complete-work and causal-state survival attribution:
  `../survival-work-state-action-swe177-20260810/result.json`
- Task-stable robust survival clock:
  `../survival-robust-clock-swe177-20260810/result.json`
- PennyLane task-Pareto survival action:
  `../task-pareto-survival-action-pennylane41-20260810/result.json`

Docs-only compiler development artifact:
`analysis/results/offline-tool-semantics-sqlglot-v1/result.json`.

Task split authority:
`analysis/development/sqlglot-relational-task-split.json`.

PennyLane survival-action development split:
`analysis/development/pennylane-survival-action-split.json`.

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
Causal eBPF feedback = retained measured mechanism/control, not integrated.
Equal-share burstable CPU execution = retained physical control baseline.
CPU-idle FCFS backfill = exposed action headroom, not deployable RSS safety.
Static hard-RSS CPU-idle admission = closed on exposed SQLGlot.
Protocol NO-GO != permission to delete a retained baseline.
No result-dependent tuning, hindsight state, or dataset-specific outcome rule.
Offline LM input = versioned public tool docs and --help only; zero runtime LM.
Documentation supplies representation, never an action or measured cost.
Scheduler work requires observable pre-execution state and a physical oracle.
GPU tool-gap timing sees each command only at its tool-event boundary.
GPU action costs come from same-model, same-device measurements, never 5s*rho.
No new collection, runtime integration, or scheduler claim without a separate
approved protocol.
```
