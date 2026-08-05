# Tool-Resource Prediction — Canonical Objective and Lock

**Effective:** 2026-08-04; amended to 5/3/3/3 classes after the earlier
3/2/2/2-class SQLGlot results were visible
**Status:** development-only; the reserved 50-task validation partition was
evaluated and returned NO-GO; the 50-task final partition remains unread

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

Physical Disk prediction requires evidence about pre-command cache
residency or equivalent environment state. The controlled file-footprint by
page-residency mechanism test on 12 already-exposed development tasks remains
frozen in the same record. No completed candidate authorizes runtime integration,
the long residency run, global cache eviction, or final-partition access.

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

## 5. Development-exposure record

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
