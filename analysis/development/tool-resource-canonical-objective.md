# Tool-Resource Prediction — Canonical Objective and Lock

**Effective:** 2026-08-04; amended to 5/3/3/3 classes after the earlier
3/2/2/2-class SQLGlot results were visible
**Status:** development-only; validation returned NO-GO; final resource labels
remain unopened, but collector result metadata was exposed on 2026-08-06

This is the authoritative contract for tool-resource prediction. It records
current truth, not the implementation history. Detailed experiment procedures
live in `clause-interaction-kb-plan.md`; frozen result artifacts and git retain
superseded protocols.

## 1. Prediction objective

The evaluation unit is one eligible `exec` command. Clauses are internal
evidence used to predict that command; they are never scored as separate rows.

For latency, predict one normalized probability mass function over:

```text
[0, 500] ms
(500, 2000] ms
(2000, 8000] ms
(8000, 30000] ms
(30000, +inf) ms
```

The hard prediction is the highest-probability bucket; an exact tie selects the
shorter bucket. The primary metric is exact five-class command accuracy. An
unavailable hard prediction counts as incorrect while remaining reported as
unavailable; it does not remove the labelled command or make the aggregate
accuracy undefined. This fail-closed scoring amendment was recorded before the
first residual-model fit or validation score, after the prior task-phase run
had already exposed one unavailable validation prediction.
Compare it with one constant majority class computed over the identical
evaluation commands. Report eligible count, label and prediction counts, the
5x5 confusion matrix, majority accuracy, within-one-bucket accuracy, severe
underprediction rate (truth at least two buckets above prediction), and deltas
from majority and Current.

Predict CPU peak, sampled RSS, and Disk I/O independently as Low, Medium, or
High. Boundaries belong to the lower bucket:

```text
CPU  = [0, 2], (2, 4], (4, +inf) peak cores
RSS  = [0, 500], (500, 2000], (2000, +inf) decimal MB
Disk = [0, 1048576], (1048576, 104857600], (104857600, +inf) bytes
```

For these resource targets only, an observation explicitly marked by policy as
null with command latency below 500 ms is imputed Low. Other nulls are
unavailable. The hard prediction is the highest-probability bucket with ties
selecting the lower bucket. Report eligible, label and prediction counts, the
3x3 confusion matrix, exact accuracy, majority accuracy, constant-Low accuracy,
within-one-bucket accuracy, and severe underprediction rate.

Old nine-bin latency accuracy, the superseded 2000/8000-only result, legacy
3500/5000 ms boundaries, balanced accuracy, Brier/NLL, q-error, and
hand-selected subsets cannot select a candidate.

## 2. Causal and physical contract

Offline replay and online serving use the same predictor semantics:

```python
predict(repo, command, parsed_clauses, ts_start) -> predictions + provenance
observe(completed_clause_observations) -> None
```

- An observation is visible only when `observation.ts_end < query.ts_start`.
- A task's observations enter the KB only after successful whole-task
  finalization. The current task and failed or unfinalized tasks are invisible.
- Frozen cross-repository evidence does not update during evaluation;
  repository-local evidence updates causally between settled tasks.
- Every arm uses identical command IDs, labels, availability, order, and
  compound-command structure.
- Telemetry marked invalid, ambiguous, lossy, or cleanup-invalid is withheld;
  independently valid sibling calls may remain eligible.
- A static pre-execution arm cannot use the current command's observations or
  output. An early-execution arm may use only samples timestamped no later than
  its frozen decision time; the remaining execution and final output stay
  hidden. Hindsight execution modes are diagnostics only.

Compound commands compose empirical clause values according to shell structure:

- sequential stages: latency and Disk sum; CPU and RSS take the maximum;
- concurrent pipelines: latency takes the maximum; CPU and RSS sum;
- Disk sums across all clauses.

Never compose bucket IDs or resource class labels with Boolean OR. Unsupported or
ambiguous structures remain unavailable.

## 3. Current architecture and research boundary

`ClauseResourceKB` is the Current control. It uses raw exact argv, argv-prefix,
and binary/global backoff with frozen public evidence and causal repo-local
updates. The trie is not the proposed research contribution.

The current **development reference SOTA** is a target-specific multi-head
predictor at one common `BeginCall` decision time. Latency, CPU, and RSS use
semantic work-unit evidence, with the full-test phase allowed only to raise the
predicted bucket. Disk uses exact complete-command outcome memory. Cross-task
evidence updates only after whole-task settlement; the current task's outcomes
remain hidden, while the phase head sees only the count of earlier completed
full-test commands. The heads share the same 1,044 validation commands, labels,
Current predictions, and bucket definitions.

| Target | Eligible | Current | Development SOTA | Delta | Selected head |
|---|---:|---:|---:|---:|---|
| Latency | 1,044 | 74.904% | 79.023% | +4.119 pp | semantic work units + monotone test phase |
| CPU | 672 | 82.887% | 87.500% | +4.613 pp | semantic work units + monotone test phase |
| RSS | 736 | 80.978% | 87.500% | +6.522 pp | semantic work units + monotone test phase |
| Disk | 1,008 | 82.044% | 83.829% | +1.786 pp | exact complete command |

Equal-weighting the four target accuracies gives 84.463% versus 80.203% for
Current, a descriptive +4.260-point gain. This configuration was selected
post-hoc from already-exposed validation results: it is a reproducible
development reference, not confirmation. Only RSS clears the existing
five-point per-target gate, so the research verdict remains NO-GO and the final
partition remains unauthorized. The separate Disk survival result reaches 85.218% only
after an approximately 0.64-second execution prefix; it is an optional later
update, not part of this pre-command SOTA.
The machine-readable result and command rows are in
`analysis/results/tool-resource-5-3-3-3-20260804/sqlglot50-multitarget-sota-v1/`.

A frozen follow-up asked whether repeated pytest argument transitions add
information beyond this SOTA. On the exposed validation partition, the joint
occurrence/transition residual reduced the equal-weight latency/CPU/RSS
accuracy from 84.674% to 82.646%. Relative to SOTA it changed predictions in
24 tasks, with 15 helpful and 62 harmful target changes; latency, CPU, and RSS
fell by 1.149, 2.083, and 2.853 percentage points, while Disk and non-pytest
rows remained bit-identical. The gate returned NO-GO and did not authorize the
final partition. Post-hoc trace inspection suggests, but does not establish,
the mechanism: the same pytest transition can be a dependency-driven fast
failure or a real suite execution, so argument sequence alone aliases distinct
environment states. The reviewed artifact is in
`analysis/results/tool-resource-5-3-3-3-20260804/sqlglot50-pytest-transition-residual-v1/`.

The completed semantic-signature question was:

> Within many tasks from the same repository, can a deterministic semantic
> work signature reuse evidence that raw argv matching misses?

That route is closed by the frozen results below: the surviving representation
changed too few decisions, causal environment state was not identifiable, and
no scheduler consumer existed.

The early-execution CPU-reservation question is also closed for the current EAR
action set. It used each command's own timestamped execution prefix rather than
cross-task similarity, but its hindsight action-space upper bound did not pass
the frozen utility gate.

A later development-exposed latency diagnostic tested continuous updates while
a command is alive. Command-duration survival passed its time-weighted mechanism
gate, moving exact accuracy from 60.260% to 90.669%, but clause-level progress
added only 0.147 points and increased severe-or-unavailable time. A post-hoc
elapsed-floor control reached 86.832%, explaining 26.572 of the formal arm's
30.409-point gain. That floor is a physical constraint, not a prediction, so a
runtime API that merely raised the hard class to the next possible bucket was
retracted before use. The corrected strict-survival predictor recomputed the
five-bucket PMF from raw historical durations greater than elapsed time and made
an exhausted empirical tail unavailable. Its initially frozen wall-time metric
was subsequently amended, after results were visible, because the canonical
evaluation unit is a command and no named consumer incurs time-proportional
error cost. Command-equal correct-time fraction moved only from 79.502% to
79.731% (+0.229 points), with 22 positive, 26 negative, and 2 tied task deltas.
The 75.827% versus 60.260% wall-time-weighted result is retained as a secondary
long-command diagnostic, not the primary predictor claim. Strict finite-sample
survival and clause inference are closed; this result has no scheduler-utility
claim and does not authorize the unread final partition.

The deterministic `full-test-third-or-later-v1` feature passed its reviewed,
post-hoc development continuation rule. On the exposed 80/20 replay it improved
latency, CPU, and RSS by 1.146, 2.857, and 2.532 percentage points, with Disk
bit-identical. Its detector, development PMFs, coverage rule, validation gate,
and final authorization were then fixed before reserved access. On the 50-task
validation partition it improved available-only latency by 0.192 points, CPU
by 0.893 points, and RSS by 0.679 points, with Disk bit-identical. One unavailable
latency prediction made the complete primary metric undefined, so the frozen
fail-closed gate returned NO-GO. The available-only gain is also far below the
five-point research target. The final partition therefore remains closed.

The offline-agent direction is complete and closed. It progressed from a
per-query blind classifier to a frozen Python feature, exact relational spans,
schema-bounded regex, family/scope extraction, and finally selection from a
fully host-built typed catalog. The frozen Python feature left all four exact
accuracies unchanged, with five helpful and five harmful target changes and
helpful changes in only two tasks. Intermediate relational candidates failed
their preregistered structural validators without repair or retry.

The final typed-catalog arm removed arbitrary code, regex, and parser
generation. The agent selected the semantically credible `unittest`
missing-module to successful-install relation rather than support-only `C000`,
and local extraction passed its runtime gate. Nevertheless, supported
`blocked` and `closure_candidate` states had identical hard PMF modes for all
four targets, and the second scope lacked task support. This is evidence that
the blocker/remediation story can be semantically correct without separating
resource outcomes. Exact costs, intermediate failures, and artifacts are in
`clause-interaction-kb-plan.md`.

Taken together, the agent direction is closed. The final finite-choice arm
removed arbitrary code, regex, and parser generation yet still found no
resource-separating relation/scope contrast. More prompts, critics, DSLs, or
agent adapters are not authorized.

The original 100 SQLGlot tasks are development-only fit evidence. The
additional 100 tasks were split by task ID, independent of completion state,
into 50 validation and 50 final-test tasks before any reserved output or label
was read. No agent arm opened either partition. The frozen task-phase method
consumed validation and returned NO-GO, making validation development-exposed;
it did not authorize final. Exact IDs and stage permissions live only in
`analysis/development/sqlglot-relational-task-split.json` and the current
research record.

Two pre-result validation invocations failed in plumbing without producing a
result artifact: omitted unavailable Current keys first raised
`KeyError: peak_cpu_cores`, then explicit null metrics reached an invalid
numeric comparison. The reviewed repairs canonicalized unavailable predictions
to `None` and made incomplete primary metrics fail closed. The concise amendment
record in `clause-interaction-kb-plan.md` states what was visible before each
unchanged retry.

Post-validation diagnostics use development-exposed data only. One interactive
25-task fit / 25-task test diagnostic reported gains from combining Current,
structured command features, and causal task-local history, but it retained no
executable evaluator or machine-readable artifact. It is therefore a lead, not
evidence for a method or a final-test authorization.

The parameter-free `causal-call-overlay-v1` then tested whether an otherwise
identical Current KB should temporarily observe valid clauses from earlier
completed commands in the same task. On the exposed 80/20 replay it improved
latency, CPU, and RSS by only 0.860, 0.476, and 0.422 percentage points. Their
severe-underprediction rates all increased; Disk was bit-identical by design.
The frozen gate returned NO-GO. Repeated command load can alternate within a
task, so treating the last measured mode as persistent creates a one-command
lag rather than a stable state estimate. The exact result and changed-case
diagnosis are in `clause-interaction-kb-plan.md`.

The generic `command-history-residual-v1` candidate then combined Current with
parsed command structure and causal task history. On the exposed validation
tasks it changed latency by -1.533 points, CPU by +0.149, RSS by +1.087, and
left Disk bit-identical. Its frozen five-point gate returned NO-GO. It reduced
severe underprediction but overrode too many correct Current modes: 192 changed
target predictions were helpful and 201 harmful.

This validation also exposed a cohort problem that a stronger classifier cannot
repair. Every 80-task fit item is newer than every 50-task validation item:
their creation ranges are 2024-08-28--2025-04-25 and
2023-09-27--2024-07-15, respectively, with different repository versions and
environment setup commits. CPU/RSS Medium support shifts from 4/5 fit commands
to 28/87 validation commands. This is backward temporal extrapolation across
workspace generations, not an IID same-repository split. The next research
question is whether inference-time workspace/environment measurements explain
that shift; task ID, timestamp, version, or image identity alone are not valid
predictive shortcuts.

The causal `command-outcome-memory-v1` test then checked a smaller unit
mismatch: Current memorizes clause outcomes but not the aggregate outcome of a
complete command. Exact complete-command evidence was useful when available,
but the frozen exact-then-shape hierarchy improved latency, CPU, RSS, and Disk
by only 2.874, 2.232, 3.940, and 1.190 points. The shape fallback harmed
latency and Disk on its own covered rows. The five-point gate returned NO-GO;
final remains closed. This narrows the next question to inference-time evidence
about previously unseen requested work and physical environment state, rather
than another coarser command key.

The separately frozen pytest target-overlap test used an intermediate work unit
rather than a coarser generic key: Jaccard overlap between explicit test
file/node targets. It improved latency, CPU, and RSS by 3.831, 4.167, and 6.250
points, but Disk by only 1.091 points, so the all-target gate returned NO-GO.
On the overlap carrier alone, compute/memory gains were large while Disk
accuracy fell by 14.58 points. This is evidence for semantic requested-work
units for latency/CPU/RSS and against using them as a proxy for physical I/O
state. Final remains closed.

Because the four canonical targets are independent, the next component test did
not ask semantic work units to proxy physical I/O. It combined exact
complete-command outcomes, pytest target overlap, and pip package-name overlap
for latency/CPU/RSS while copying Disk from Current. It returned NO-GO:
latency, CPU, and RSS improved by 4.023, 4.167, and 6.250 points respectively,
while Disk remained bit-identical. Pip overlap itself changed only four latency
hard decisions and no CPU/RSS hard decisions. This closes the exact semantic
hierarchy without opening final.

The next frozen component keeps that hierarchy and tests an early-execution
physical lower bound. At the unchanged Phase-1 decision time, elapsed time and
the first complete 0.5-second CPU interval may only raise a prediction that is
below the already observed bucket. RSS remains semantic-only and Disk remains
Current. The component must put latency, CPU, and RSS each at least five points
above Current, introduce no lower-bound/final-label inconsistency, and cannot
open final without an independent Disk mechanism. It returned NO-GO. Latency
reached +5.843 points and RSS remained +6.250, but CPU fell by 4.464 points and
produced 96 physical-scope violations. The action timeline is cgroup-wide;
canonical CPU is clause-owned eBPF lineage. Their buckets are not nested, so
the cgroup prefix is not a valid lower bound and cannot be repaired by a
threshold or favorable subset.

The next frozen development component discarded the cgroup value and retained
only the causal survival event at the same decision time. Combining this with
exact-command Disk evidence improved Disk by 3.175 points: 56 helpful and 24
harmful changes, with severe underprediction falling to zero. It nevertheless
missed the frozen five-point gate. The same arm retained elapsed-time latency at
+5.843 points, semantic CPU at +4.167, and semantic RSS at +6.250. This closes
survival calibration without opening final.

The next frozen development-only check composes existing predictions without
refitting: third-or-later full-suite phase may raise semantic latency/CPU/RSS,
elapsed time may further raise latency, and survival supplies Disk. This tests
component complementarity only; it cannot repair a Disk miss by changing the
survival rule or open final from the already-exposed validation partition. The
composition returned NO-GO: latency, CPU, RSS, and Disk changed by +5.939,
+4.613, +6.522, and +3.175 points. Phase added three net-correct CPU rows, but
CPU remained roughly three correct rows short of the gate and Disk was
unchanged from survival.

Two post-composition selectors are also closed. A first-40/next-40 causal
calibration keyed by the four Current hard predictions improved CPU by only
0.456 points and changed no Disk prediction; its gate stopped before reading
the adjacent-validation file. A task-clustered five-fold confidence gate on the
exposed 50 validation tasks then selected composition corrections solely from
the difference in winning PMF probability. It preserved latency/RSS but
returned +4.167 CPU and +2.282 Disk points. Fold thresholds rejected more
helpful than harmful Disk corrections. These results close hard-state coupling,
PMF-confidence thresholding, and further selector tuning on exposed rows.

Physical Disk prediction requires evidence about pre-command cache residency
or equivalent environment state. The controlled file-footprint by
page-residency experiment on 12 already-exposed tasks completed on 2026-08-06.
Its official joint-validity gate returned NO-GO: five sub-one-second executions
had valid Disk but unavailable canonical peak CPU, leaving nine complete tasks.
Those nine all showed the intended residency and physical-read contrast, four
crossed to a lower Disk bucket, and none crossed upward. A post-hoc Disk-only
availability check found 12/12 residency/read contrasts and 7/12 lower buckets,
but cannot amend the frozen verdict. The same run's clause-owned 500 ms CPU
prefix made one helpful and zero harmful changes, below its three-task gate.
No completed candidate authorizes runtime integration, global cache eviction,
or final-partition access.

The fixed full-command targets in Section 1 remain the static-predictor
baseline. Their first causal 80-task warm-up / 20-task test replay contains 349
test commands. It is development-exposed, not confirmation evidence:

| Target | Eligible | Label counts low to high | Majority | Frozen at 80 | Dynamic update | Within one | Severe under |
|---|---:|---:|---:|---:|---:|---:|---:|
| Latency | 349 | 214 / 58 / 20 / 28 / 29 | 61.318% | 77.650% | 77.937% | 91.977% | 1.146% |
| CPU | 210 | 181 / 0 / 29 | 86.190% | 90.476% | 90.476% | 90.476% | 3.333% |
| RSS | 237 | 208 / 2 / 27 | 87.764% | 91.139% | 91.561% | 91.561% | 2.954% |
| Disk | 349 | 243 / 85 / 21 | 69.628% | 81.662% | 81.375% | 99.713% | 0.000% |

Dynamic updates changed exact accuracy by +0.287, 0.000, +0.422, and -0.287
percentage points respectively. The Current predictor beats the constant
majority on all four targets, but this corpus does not substantiate the chosen
CPU and RSS middle levels: CPU has no Medium labels and RSS has only two.
Changing those boundaries after this result would be a visible amendment, not
a pre-registered comparison. The machine-readable result and command rows are:

- `analysis/results/tool-resource-5-3-3-3-20260804/sqlglot80-20-current-baseline.json`
- `analysis/results/tool-resource-5-3-3-3-20260804/sqlglot80-20-current-baseline.rows.jsonl`

Phase 0 was only an observability and coverage audit. The action set, decision
time, future-work target, cost metric, and gate below were committed before the
formal oracle result was read.

The frozen Phase 1 consumer is the existing EAR shared CPU lease pool. Its
checked-in Docker policy at EAR commit `244d1f58` uses integer pages
`{1, 2, 4, 8}` on this eight-core host. For each eligible SQLGlot command, the
hindsight oracle acts at its first complete sample endpoint plus 0.14132007875
seconds, nominally about 0.641 seconds, and chooses the smallest page no lower
than any later 0.5-second average CPU demand. The primary action cost is reserved
CPU core-seconds after that time. Always reserving eight cores is the control.
The oracle was eligible on 467 commands across all 100 SQLGlot tasks. It chose a
smaller page for 291 commands across 99 tasks with zero modeled false shrink,
but reduced post-decision reservation only from 144,661.88 to 136,436.48 CPU
core-seconds, a 5.686% reduction below the frozen 10% gate. The mechanism is
time-weighted: the 176 commands that still required eight cores accounted for
89.84% of control reservation, while the numerous shrinkable commands were too
short to dominate cost. This is a hindsight capacity upper bound and does not
establish sub-interval safety, latency gain, or scheduling gain. Phase 2 stops.

The runtime boundary remains unchanged: `resource-agentd` owns parsing,
prediction, state, persistence, and orchestration; `telemetryd` owns privileged
collection and finalized observations. `src/tool_resource/` imports nothing
from the rest of the repository. Offline trace adapters live in
`src/tool_resource_eval/`. Phase 0 does not change runtime integration.

## 4. Settled semantic evidence under the superseded 3/2/2/2 target

The results in this section were visible before the 5/3/3/3 amendment. They
remain useful representation diagnostics but are not scores for the current
target and cannot set its class boundaries or confirmation criteria.

### Phase A — pip representation: complete

On the fixed 1,792 SQLGlot commands, Current scored 87.500% overall and
81.513% on 119 pip commands. Canonical exact scored 87.388% overall and 79.832%
on pip. Jaccard semantic matching scored 87.667% overall and 84.034% on pip.
Against canonical exact, seven non-exact carriers changed: six helpfully and
one harmfully, with positive net gains across six tasks and five signatures.
Non-pip predictions were identical. The frozen Phase A gate therefore selected
Jaccard semantic matching. State-aware and semantic-only predictions were
identical and state did not contribute to selection.

Authoritative artifacts:

- `analysis/results/tool-resource-clause-interactions-20260804/pip-carrier-audit-latency.json`
- `analysis/results/tool-resource-clause-interactions-20260804/pip-carrier-audit-latency-rows.jsonl`
- `analysis/results/tool-resource-clause-interactions-20260804/semantic-work-signature.html`

### Phase B — second tool in the same SQLGlot corpus: complete, no-go

The fixed second tool is pytest. Its semantic evidence may come only from
causally settled earlier SQLGlot tasks. Current retains its unchanged frozen
public prior; the candidate falls back to Current when no repo-local semantic
evidence exists. No SWE semantic evidence is admitted.

Before reading pytest labels, require all of:

- at least 100 parsed pytest commands;
- coverage across at least 20 SQLGlot tasks; and
- at least 20 commands whose semantic signature has causal earlier-task
  evidence but whose Current exact clause key has no earlier-task match.

The pre-label coverage gate passed with 339 parsed pytest commands across 88
tasks and 125 non-exact carriers. The frozen latency run then improved Current
from 1,568/1,792 (87.500%) to 1,572/1,792 (87.723%) overall, and from 284/398
(71.357%) to 288/398 (72.362%) on pytest commands. Seven hard predictions
changed: four helpfully, none harmfully, and three neutrally. Non-pytest PMFs
were bit-identical.

The result is nevertheless a frozen no-go: positive net gain covered four
tasks but only one semantic signature, the single-file pytest shape, below the
required two signatures. Phase B therefore stops without changing the parser,
tool, thresholds, or gate, and does not transfer to CPU, RSS, or Disk.

### Phase C — state identifiability: complete, no-go

Among 26 pip package-set signatures, ten repeated across tasks, none appeared
under two causal pre-query states, and only one appeared under multiple
output-derived execution modes across tasks; the frozen requirement was eight.
Semantic-only and the selected causal-state arm remained PMF-identical at
100/119 correct. Even a hindsight execution-mode leave-one-task-out oracle fell
to 98/119, with 11 helpful and 13 harmful changes. State therefore lacks both
the required contrast and an oracle improvement ceiling. The controlled state
collection is not prepared or requested.

### Phase E — scheduler actionability: complete, no-go

The surviving pip representation changed five latency decisions across five
tasks, four Disk decisions across four tasks, and no CPU or RSS decisions.
After deduplication, only six commands across six tasks changed any hard target;
the frozen gate required 20 commands across ten tasks. No repository consumer
outside the tool-resource service currently maps these predictions to an
allocation, admission, ordering, or timeout action. Scheduler implementation
and scheduling-utility claims therefore stop. CacheWise timeout remains a
ceiling, not a remaining-work signal.

### KV eviction actionability diagnostic: complete, no-go

The advisor subsequently authorized one development-only action diagnostic
with CacheWise as the baseline. It tests whether the existing latency PMFs
improve a named consumer, KV victim selection, without changing the predictor
or opening the final partition. The already-exposed validation 50 are the
workload; all predictor and decoder evidence comes from completed development
tasks.

The existing trace-driven simulator remains fixed at FCFS, 40 sessions,
800,000 KV tokens, 16-token blocks, and seeds 0--31. Prefix scheduling is
excluded because it removed all eviction pressure in the earlier factorial.
The arms are LRU, CacheWise tool-name survival, CacheWise C100, frozen Current,
frozen development SOTA, and a hindsight next-reuse oracle. Current and SOTA
replace C100 only for an eligible exec gap with an available latency PMF; all
other gaps use identical C100 fallback. C100 and the tool-name histories are
fit on the 100 development traces. Current is the fixed KB snapshot after
those development tasks, with no validation update. The semantic head retains
its fixed 80-task fit, and the test-phase head uses only earlier command names
within the same task.

Each PMF is converted to remaining time by a frozen empirical decoder. For
each latency bucket, development gaps whose eligible command has that bucket
form the within-bucket duration distribution. At elapsed time `e`, the decoder
returns the PMF-weighted mean of `duration - e`, conditional on `duration > e`.
If that empirical mixture has no survivor, the arm uses the identical C100
fallback rather than extrapolating a tail.
The oracle uses the observed next-reuse interval and is an upper-bound policy,
not a deployable feature or globally optimal cache solution.

The primary metric is mean eviction-induced recomputed prefix blocks; lower is
better. Evicted blocks are the physical-cost guardrail. A live experiment is
authorized only if all of the following hold: C100 has nonzero primary cost;
the oracle improves it by at least 10%; SOTA improves it by at least 10%; the
upper endpoint of the paired-seed bootstrap interval for SOTA minus C100 is
below zero; SOTA is strictly better than Current; and SOTA does not increase
mean evicted blocks over C100. Otherwise the result is NO-GO for live KV
integration. LRU and tool-name are descriptive baselines. Simulated latency is
not a utility metric because cache misses do not feed back into service time.

A pre-result unit audit found 1,842 single-call tool gaps: 1,044 map one-to-one
to every eligible validation exec command, 152 are ineligible exec calls, and
646 are non-exec calls. Eligible command duration and whole-gap duration share
the same latency bucket for 1,038/1,044 rows; the extra gap overhead has 23.16
ms median and 41.17 ms p95. This supports the decoder alignment but is
development-exposed preflight evidence, not a validation claim. Final resource
artifacts are not used. This protocol was frozen before the formal simulator
outcome; two pre-output plumbing failures only removed commands with no KV
decision gap and normalized sparse unavailable fields.

The completed run returned NO-GO. Mean recomputed prefix blocks were 259,557
for LRU, 45,349 for C100, 44,910 for Current, 44,678 for SOTA, and 41,255 for
the hindsight next-reuse oracle. SOTA reduced C100 by 671 blocks or 1.4805%;
the paired-seed 95% bootstrap interval for the absolute delta was
[-944.06, -402.74], so the direction was stable but far below the 10% utility
gate. The oracle improved C100 by only 9.0272% and also failed the gate. SOTA
improved Current by 232 blocks, or 0.5167%, without increasing evicted blocks.

The mechanism evidence is twofold. First, C100 already removed 82.53% of LRU's
recomputation, leaving little next-reuse-ordering headroom even for hindsight.
Second, with cross-validation-task settlement disabled as concurrent causality
requires, Current and SOTA exact latency accuracy were 74.904% and 75.766%.
The serial replay's 4.119-point SOTA gain therefore shrank to 0.862 points;
79.1% of that descriptive gain depended on between-task updates unavailable at
concurrent batch start. This result stops live KV integration. It does not show
that latency PMFs are useless to every scheduler; it shows that improving this
already-strong C100 victim policy is not the promising consumer. The reviewed
machine-readable result is in
`analysis/results/tool-resource-5-3-3-3-20260804/sqlglot50-kv-prediction-actionability-v1/`.

### Block-aware KV oracle: complete, no-go

The positive 1.4805% SOTA result and its live-integration NO-GO remain settled.
One final development-only upper-bound diagnostic tests whether the prior
greedy next-reuse oracle understated the KV eviction action space. It reuses
the exact 32 task selections, FCFS schedule, 40-session load, 800,000-token
capacity, recorded service/gap durations, and C100 metrics from the committed
actionability result. It adds one hindsight-only block-aware Belady arm; no
predictor is changed or rescored.

Because cache misses do not feed back into service time, each schedule's
request order is fixed before eviction. For every resident session, blocks
beyond the next turn's reusable prefix have no next use and are evicted first.
If no such suffix remains, the arm evicts reusable suffix blocks from the
session whose next request occurs latest in that fixed order. Partial eviction
stops at the non-reusable suffix boundary before choosing another victim. Under
the simulator's unit-block cost and contiguous-prefix representation, this is
the farthest-next-use policy for the modeled block objects. It is an offline
upper bound, not an online policy; the current request remains protected under
the same simulator constraint as every prior arm.

The primary metric remains mean eviction-induced recomputed prefix blocks;
mean evicted blocks is the physical-cost guardrail. The result is GO only if
the schedules and request counts are identical to the committed baseline,
Belady reduces C100 recomputation by at least 10%, the upper endpoint of the
paired-seed 95% bootstrap interval for Belady minus C100 is below zero, and
Belady does not increase mean evicted blocks. GO authorizes only design of a
causal approximation. NO-GO closes KV victim selection for the current
simulator. The existing result, thresholds, and greedy-oracle interpretation
cannot be amended by this diagnostic.

The completed result remained below the frozen gate. C100 averaged 45,349
recomputed prefix blocks, the greedy next-reuse oracle 41,255, and block-aware
Belady 40,986. Belady reduced C100 by 4,363 blocks or 9.6201%, with a paired
seed-bootstrap 95% interval of [-4,846.36, -3,900.12] blocks. It passed request
identity, paired-direction, and evicted-block guardrails, but missed the 10%
minimum by 0.380 percentage points. The threshold is unchanged.

Belady recovered only 269 additional blocks, or 0.652%, beyond the greedy
oracle. It also increased mean eviction events from 619 to 875 while reducing
total evicted blocks, a cost not charged by the simulator. The evidence
therefore preserves a small positive benefit from block-aware suffix handling
but closes KV victim selection under the current action model. The reviewed
artifact is in
`analysis/results/tool-resource-5-3-3-3-20260804/sqlglot50-kv-block-belady-oracle-v1/`.

### KV oracle mechanism decomposition: complete, no-go

The 9.6201% Belady result remains a NO-GO and is not rescored. A final 2x2
development diagnostic attributes its improvement using two hindsight factors:
next-use order is either exact gap-arrival time or actual fixed FCFS request
rank, and suffix handling either ignores or first exhausts blocks absent from
the next reusable prefix. The four arms are therefore arrival/no-suffix (the
committed greedy oracle), rank/no-suffix, arrival/suffix, and rank/suffix (the
committed Belady result). Only the two missing arms are simulated; the exact
32 schedules, requests, capacity, service model, and committed corner results
are reused byte-for-byte.

Lower recomputed prefix blocks remains primary. The rank main effect is the
average improvement from arrival to rank with suffix off and on. The suffix
main effect is the average improvement from no-suffix to suffix under arrival
and rank. Their interaction and eviction-block/event changes are descriptive.
A factor authorizes one causal-proxy experiment only if its main effect is
positive, at least 1% of C100 recomputation, and at least 20% of the total
C100-to-Belady improvement. If both pass, only the larger factor proceeds; if
neither passes, the apparent block-aware opportunity is too diffuse or small
to pursue. This decomposition cannot reopen live integration or amend any
earlier gate.

The completed decomposition authorizes neither factor. Replacing exact
gap-arrival order with exact FCFS request rank saved a 262.422-block main
effect, or 0.579% of C100 recomputation and 6.015% of the C100-to-Belady gap;
the paired-seed bootstrap interval was [166.203, 370.268] blocks. Dead-suffix
handling saved only 6.422 blocks, or 0.014% of C100 and 0.147% of the gap, with
interval [3.578, 9.844]. Both directions are positive, but both miss the
frozen 1% and 20% minimums.

The committed arrival/no-suffix oracle already accounts for 93.84% of the
C100-to-Belady improvement. Its 9.0272% C100 reduction was the prior frozen
NO-GO, so this is mechanism attribution rather than a new integration case:
the small remaining Belady benefit comes mostly from knowing exact request
order, not from block-aware suffix handling. KV victim selection is closed
under the current simulator and action model. The independently reviewed
artifact is in
`analysis/results/tool-resource-5-3-3-3-20260804/sqlglot50-kv-oracle-decomposition-v1/`.

### CPU+RSS command admission oracle: complete, go

KV eviction is closed. The next development-only action-space test asks whether
the existing command CPU/RSS targets could support a concrete consumer at all:
a single-host admission controller that starts ready `exec` commands only when
their reserved CPU and RSS fit within 8 cores and 16,000 MB. It uses the 100
already-exposed SQLGlot tasks from
`sqlglot-100-c2-fast-requested-ebpf-a0419d9-20260803`, with the same 32 seeds and
the same sorted-pool shuffle then first-40 selection procedure as the KV
diagnostics. No final task or predictor output is read in this phase.

Each task is a closed serial program. The replay preserves recorded time before
the first `exec`, between successive `exec` calls, after the final `exec`, and
each command's recorded duration. Delaying one command shifts the rest of that
task; other tasks continue independently. Ready commands use deterministic FCFS
order with work-conserving backfill. Non-`exec` work contributes recorded think
time but consumes no modeled host reservation.

The fixed-high control reserves the full host for every command, so at most one
command executes at a time. The hindsight oracle composes retained clause
telemetry using the canonical physical rule: sequential stages take the maximum
and concurrent pipeline members sum. Any missing CPU or RSS value reserves the
full host for that target; the canonical short-null Low label is not treated as
a measured physical ceiling. Invalid structure or an unmatched command also
reserves the full host. Composed values are bounded by the physical host
capacity, so missing telemetry never creates packing headroom. The oracle is not
causal and cannot be used as a predictor result. This stricter fallback was
fixed before any schedule outcome was read.

Mean batch makespan is primary; mean task completion time and total command
queue time are secondary. GO requires at least 10% lower mean makespan, a
paired-seed bootstrap interval for oracle-minus-control makespan strictly below
zero, at least 20 distinct commands across 10 tasks starting while another
command runs, identical commands and recorded durations in both arms, and no
modeled CPU/RSS reservation-capacity violation. GO authorizes only evaluation of
Current and development SOTA as admission inputs. NO-GO closes command-level
CPU+RSS admission for this action model. The replay does not model idle
container memory, Disk/network contention, or performance interference, so even
GO is only an upper bound and cannot authorize runtime integration.

The completed oracle passed the frozen action-space gate. Across the 32
40-task schedules, fixed-high mean makespan was 7,618.831 seconds and the
oracle mean was 6,790.848 seconds: an absolute reduction of 827.983 seconds
(13.8 minutes), or 10.8676%. The paired schedule-bootstrap interval for the
oracle-minus-control delta was [-857.835, -796.253] seconds. All 32 schedules
improved; individual relative reductions ranged from 7.29% to 12.81%, and 25
of 32 exceeded 10%.

The oracle started 369 distinct commands from 87 tasks while another command
was running, with mean maximum concurrency 6.19. Of 1,916 commands, only 459
used fully observed CPU/RSS bounds; 1,333 had at least one null target and 124
unmatched commands fell back conservatively. The two arms retained identical
commands and service durations and had no modeled capacity violation.

This GO is narrow. Mean task completion improved by only 131.784 seconds
(2.35%) and total queue time by 2.52%; some individual schedules regressed on
those secondary metrics. Recorded durations do not include performance
interference from modeled co-admission. The result authorizes one development
comparison of causal Current and SOTA reservation inputs, not runtime
integration or a scheduling-performance claim. The independently reviewed
artifact is in
`analysis/results/tool-resource-5-3-3-3-20260804/sqlglot100-resource-admission-oracle-v1/`.

### CPU+RSS predictor admission: frozen

The oracle GO authorizes one predictor action test on the 50 already-exposed
SQLGlot validation tasks; the final partition remains closed. The exact 32
seeds, sorted-pool shuffle, first-40 selection, 8-core/16,000-MB capacity,
closed-task timing replay, and FCFS-ready work-conserving backfill remain fixed.
All four arms evaluate identical commands and recorded durations: fixed-high,
the conservative telemetry oracle, Current, and development SOTA.

Current is regenerated from its dev100-frozen state with no validation-task
settlement. SOTA uses the frozen semantic work-unit evidence and the same-task
third-or-later full-test phase; it also receives no cross-validation-task
updates. Only information available by each command's `BeginCall` may choose a
reservation. CPU Low/Medium/High request 2/4/8 cores; RSS Low/Medium/High request
500/2,000/16,000 MB. An unavailable or unmatched target requests that target's
full host capacity. No threshold, class mapping, PMF quantile, fallback, or
feature is tuned in this phase.

Admission uses each arm's requested reservation. Separately, concurrent demand
is checked against the same conservative telemetry requirement used by the
oracle. Because a null target is full-host fallback, exceeding capacity means
the schedule is not safety-verifiable from retained evidence; it is not a claim
that physical overload was directly observed. Such exposure cannot be hidden by
unchanged recorded runtimes.

The primary comparison is SOTA-minus-Current mean batch makespan. GO requires:
the validation-pool oracle still reduces fixed-high makespan by at least 10%;
SOTA reduces Current makespan by at least 5%; its paired-seed bootstrap interval
is strictly below zero; it captures at least half of the oracle headroom over
fixed-high; at least 20 commands across 10 tasks receive a different CPU or RSS
request than Current; SOTA creates zero conservative capacity exposures; and all
arms retain identical commands and service durations. Mean task completion,
queue time, Current exposures, and requested resource-time are secondary.

GO authorizes only a real interference experiment with an explicit failure
policy. NO-GO closes hard-class Current/SOTA as direct admission requests; it
does not authorize threshold tuning, a safety critic, or a new predictor on the
exposed validation tasks.

## 5. Development-exposure record

- On 2026-08-06, the KV decision-unit preflight mistakenly parsed the reserved
  run's complete `results.jsonl` before filtering to validation IDs, and its
  first two rows were printed. This exposed final-task collector outcomes and
  proxies including success, elapsed time, iteration count, and model patch.
  Final-task traces, resource observations, canonical bucket labels, and
  evaluation scores were not opened. Nevertheless, the final 50 are no longer
  an untouched confirmation partition and cannot support a confirmatory claim.
  All subsequent access is restricted to paths constructed directly from the
  already-exposed validation IDs; fresh tasks are required for confirmation.
- The Phase A numbers above were visible before Phase B was corrected from a
  cross-repository SWE test to a same-repository SQLGlot test.
- At the SQLGlot-only scope correction, no SQLGlot pytest coverage count or
  label had been read. Coverage was then frozen and passed before the Phase B
  labels above were read.
- Earlier SWE100/277 trie, generic-argv, resource, and concurrency diagnostics
  are development-exposed negative evidence. They motivate controlling
  repository relatedness but do not evaluate the SQLGlot semantic candidate.
- Before any early-execution future-work label was inspected, the decision-time
  formula was corrected from `max(sample interval, update latency)` to their
  serial sum plus the frozen 50 ms sample-availability pad. The live update
  latencies were visible when this correction was recorded.
- The Phase 1 CPU-reservation protocol and gate were committed before its future
  CPU samples were evaluated. Independent review then corrected complete-window
  and missing-counter handling before the formal result was run; no result was
  visible during those fixes.
- The first SQLGlot command-level replay made with an older public aggregate
  retained invalid downstream-pipeline evidence. It remains explicitly named
  `*.pre-public-structure-fix.invalid`; only the corrected artifact is
  reportable.
- The superseded 2000/8000 ms objective and binary resource thresholds were
  selected after older SWE diagnostics were visible. All resulting scores are
  development-only.
- The 500/2000/8000/30000 ms, 2/4-core, 500/2000-MB, and 1/100-MiB boundaries
  were fixed after all earlier SQLGlot 3/2/2/2-class results and the CacheWise
  diagnostics were visible. They preserve existing physical cutoffs and add
  allocation-relevant levels; they were not selected from 5/3/3/3 accuracy.
- The first 5/3/3/3 SQLGlot replay was read only after objective commit
  `ef1e7e4`, implementation commit `9075235`, focused verification, and bounded
  independent review. Its empty CPU Medium class and two-row RSS Medium class
  are now development-exposed and cannot be used for an unreported threshold
  change.
- The seven dangerous 5/3/3/3 errors, their raw outputs, and a task-phase table
  were inspected before the task-local full-test phase method and its
  continuation rule were frozen. Its original 80/20 replay is therefore only
  post-hoc development evidence. The adjacent earlier SQLGlot collection had
  zero completed result rows when its held-out protocol was frozen; no result
  from that collection had been read.
- Agent-guided residual opportunity counts were visible before the blind
  causal-state schema and gate were frozen. The resulting 80/20 replay is also
  post-hoc development evidence. The adjacent collection remained
  scientifically unread at this amendment.
- The offline generated-extractor protocol was frozen after the post-hoc blind
  state result was visible but before its catalog, agent selection, generated
  source, or 20-task score existed. It is development-only and consumed no
  adjacent task output.
- The first generated source and both prompts were frozen before candidate PMFs
  or final-20 scores were read. Its first replay stopped before scoring because
  regex source text was mistakenly treated as a required literal training
  substring. The validator was corrected and independently reviewed to measure
  matches of the exact frozen regex while rejecting package/test/file argument
  dependence; the same source and prompt hashes were then replayed with no
  additional model call. The resulting no-go numbers above are the first score.
- The relational closure protocol was frozen after that no-go and its bimodal
  mechanism diagnosis were visible. Before the freeze, only process state,
  file names/counts, task-ID disjointness, and timestamps of the adjacent
  SQLGlot collection were inspected; no command output, resource label, or
  evaluation result from it was read. The new arm is therefore an explicit
  development amendment evaluated only on the already-running task-disjoint
  collection, not a reanalysis of the exposed 80/20 split.
- On 2026-08-05, before implementation or reserved-outcome access, the
  all-adjacent-task evaluation scope was amended to preserve a final same-repo
  reserve: the 100 requested task IDs were assigned 50/50 to validation and
  final test using seed 20260805 without using completion status. Only process
  and filesystem metadata had been inspected. The final partition cannot be
  opened unless the frozen method passes validation, and validation cannot
  change the source, fit, contrast, or gate.
- Before typed-catalog generation, a label-free implementation preflight showed
  that accidentally inheriting the prior raw-prompt first-twelve-task cap left
  zero complete configurations, whereas the recorded feasibility count of eight
  used all qualifying development episodes. The typed-catalog protocol was
  openly amended to use all development episodes because its prompt contained
  only aggregate templates and counts. No resource label or reserved artifact
  was read.

## 6. Non-negotiable task contract

```text
Evaluation unit = eligible exec command; clauses are internal evidence.
Latency buckets = [0,500], (500,2000], (2000,8000], (8000,30000],
                  (30000,+inf) ms.
Primary latency metric = exact five-class accuracy on identical rows.
Resource buckets = CPU edges 2/4 cores; RSS edges 500/2000 MB;
                   Disk edges 1/100 MiB.
Resource hard labels = Low/Medium/High by PMF argmax; lower bucket wins ties.
Short-null resource policy = Low only when explicitly marked and <500 ms.
Causal visibility = observation end before query start, after task settlement.
Compound commands = physical stage/pipeline composition, never Boolean OR.
Current = unchanged raw exact/prefix/binary control.
Early-execution Phase 1 = frozen NO-GO at 5.686% versus the 10% gate.
Current-command prefix was visible only up to the frozen decision timestamp.
Phase 2 predictor and runtime control were not run.
No result-dependent tuning, package/test-name outcome rules, or hindsight state.
No new collection, runtime integration, or scheduler implementation without a
separate approved protocol.
```
