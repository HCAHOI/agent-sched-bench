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
| CPU execution | Keep equal-share burstable execution as a physical control. Strict-priority CPU-idle backfill retains exposed action headroom, but neither tested hard-RSS predictor established safe admission, so physical `cpu.idle` calibration is not authorized. |
| Saturated quartet result | Frozen claim verdict is NO-GO because 8/12, not 9/12, groups improved. This is not a module-retirement decision. |
| Fresh counterbalanced queue | **Inconclusive because 7/1,818 paired tool calls changed terminal class.** Timing-only improvement was 0.929% with 2/4 queues improving, below the frozen gate even before the quality failure. |
| Prediction contribution to feedback | Not established. Task-Aware seeding did not improve the frozen reservation/service operating point over feedback alone. |
| CPU scheduling | Short-null handling reduced conservative exposure counts about 78%, but Clause-KB and Task-Aware still produced 2,638 and 2,702 confirmed exposure events. Both hard-RSS safety arms remain closed on exposed SQLGlot; the prediction-free action ceiling is retained. |
| KB representation | Raw exact/prefix/binary `ClauseResourceKB` is the implemented Clause-KB baseline. Trie/lattice/semantic-key replacement is closed as a contribution. |
| Offline tool knowledge | Trace-conditioned generation remains closed. The frozen docs-only v1 protocol also stops before fresh collection: only `make` produced a valid specification, and its SQLGlot arm never formed a non-exact contrast. This is an uninformative protocol result, not evidence that documentation semantics cannot work. |
| Manual tool semantics | **KEEP only the pytest candidate-routing question.** A hindsight selector over existing causal candidates improved full-cohort four-target accuracy by 1.941 points for pytest, but 93/118 corrections merely restored Clause-KB; adding more pytest rules is not supported. pip missed the frozen contrast gate and its perfect-tool ceiling was only 0.392 points, so deprioritize pip-specific work. |
| KV victim selection | Closed under the current CacheWise/C100 simulator and action model. |
| GPU tool-gap action | The profile-only early clock and pre-restore are closed by the Section 5.1 offline gate. The load-8 live action test remains unresolved. The load-32 hard-pin control in Section 5.3 was invalid; Section 5.4 compares the unchanged five-second action with stock evictable prefix caching. |
| Peak-class admission | Closed. Peak CPU classes are the wrong target for sustaining command throughput. |
| Runtime integration | No predictor, feedback controller, or scheduler is currently integrated or authorized for production. |
| Next research step | Run the Section 5.4 load-32 stock-cache versus `deadline/reactive` A/B only if its bounded cache smoke completes. Do not implement the failed robust clock or pre-restore. Admit a new CPU experiment only if it supplies the missing causal RSS-safety signal or a genuinely different action. |

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
| Hard-page feedback admission | Versus static Task-Aware requests, mean per-order task completion improved 6.584%; all 32 orders improved, queue time fell 6.702%, and makespan fell 6.810% by ratio of means | Positive admission mechanism; frozen physical-advance gate failed because service inflation was 12.936%, above 5% |
| Feedback admission with CPU borrowing | Mean per-order task completion improved 1.982%; 27/32 orders improved and the paired absolute-delta interval was [-278.536, -133.809] s. By ratio of means, queue fell 2.140%, service fell 0.877%, but makespan rose 0.917% | Frozen gate failed: effect was below 5% and service inflation was 7.352%; stop before PSI calibration |
| CPU-idle FCFS backfill | Versus Serial-8, mean per-order completion improved 14.620%; all 32 orders improved, makespan improved 17.226%, and service inflation was 3.073% | Action gate passed under strict-priority CPU and hindsight RSS safety; requires causal RSS safety and physical calibration |
| CPU-idle shortest-safe selection | Hindsight shortest selection changed completion by -0.137% versus FCFS; 17/32 orders improved and the paired interval [-51.549, 70.048] s crossed zero | Ordering predictor has no headroom; do not build a latency/shortest selector |
| CPU-idle predicted RSS safety | Conservative null handling produced 12,390/12,439 Clause-KB/Task-Aware exposures. The open short-null amendment reduced these to 2,638/2,702, but both remained nonzero; utility was 33.392%/25.984% | Amendment gate still failed; do not calibrate physical `cpu.idle` or add another exposed-data exception |
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

Docs-only compiler development artifact:
`analysis/results/offline-tool-semantics-sqlglot-v1/result.json`.

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
