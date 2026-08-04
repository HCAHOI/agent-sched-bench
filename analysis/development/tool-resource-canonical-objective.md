# Tool-Resource Prediction — Canonical Objective and Lock

**Effective:** 2026-08-04; amended to 5/3/3/3 classes after the earlier
3/2/2/2-class SQLGlot results were visible
**Status:** development-only; no confirmation corpus has been evaluated

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
shorter bucket. The primary metric is exact five-class command accuracy.
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

The active development arm is a blind task-local state replay layered on the
reviewed full-test phase correction. For this named arm only, a fixed agent may
read earlier completed command text, exit status, and bounded result excerpts
from the current task, but never timing, telemetry, labels, the current result,
or future calls. It assigns a frozen environment state; the ordinary evaluator
fits state-conditioned PMFs from the first 80 tasks and scores the final 20.
This is an explicit amendment to the rule that the current task was entirely
invisible. Resource observations still settle only after whole-task
finalization, and the agent never receives them. The exact schema, support rule,
and gate live in `clause-interaction-kb-plan.md`. No runtime integration is
authorized.

The reviewed post-hoc 80/20 replay passed its development continuation rule.
Latency accuracy changed from 77.937% to 79.083%, CPU from 90.476% to 93.333%,
RSS from 91.561% to 94.093%, and Disk remained 81.375% with bit-identical PMFs.
Severe underpredictions fell from 4 to 0 for latency and from 7 to 1 for both
CPU and RSS. All 16 changed target predictions were helpful across five tasks;
none were harmful. This does not establish generalization. The unread adjacent
SQLGlot100 collection is the frozen task-held-out replication.

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
  post-hoc development evidence. It cannot change the frozen held-out primary,
  whose collection remained scientifically unread at this amendment.

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
