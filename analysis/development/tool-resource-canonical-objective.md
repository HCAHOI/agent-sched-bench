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

The blind task-local state replay is a frozen development predecessor layered on
the reviewed full-test phase correction. For this named arm only, a fixed agent may
read earlier completed command text, exit status, and bounded result excerpts
from the current task, but never timing, telemetry, labels, the current result,
or future calls. Every query uses a separate tool-free model context, preventing
cross-query leakage. It assigns a frozen environment state; the ordinary
evaluator fits state-conditioned PMFs from the first 80 tasks and scores the
final 20.
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
none were harmful. This does not establish generalization. At that time the
unread adjacent SQLGlot100 collection was assigned as one task-held-out
replication; Section 3's current 50/50 amendment supersedes that access plan.

A separate completed development-only amendment tested a lower-cost agent boundary: two
bounded, tool-free calls inspect only the first 80 exposed tasks and emit one
frozen causal feature function. The function runs locally on future commands,
emits a signature rather than a bucket, and leaves PMF fitting and unsupported
fallback to the ordinary evaluator. It may discover its own scope, but the
framework names no tool or hand-selected residual family. The fixed input,
output, cost, support, and continuation rules are in
`clause-interaction-kb-plan.md`. This arm consumed no adjacent task output.

That lower-cost arm is now complete and is a development no-go. Its single
agent-generated causal signature covered 31 warm-up commands in 15 tasks and
six test commands in three tasks. Exact accuracy stayed unchanged at 77.937%
latency, 90.476% CPU, 91.561% RSS, and 81.375% Disk. Severe underprediction fell
slightly for latency, CPU, and RSS, but the ten target changes split evenly into
five helpful and five harmful and helpful changes covered only two tasks. The
mechanism collapsed partial and complete dependency remediation: 14/15 warm-up
support tasks contained both a short retry and a later real full-suite execution
under the same signature. The method therefore stops without regeneration and
consumed no adjacent task output.

A preregistered relational extension tested the narrower failure mechanism
without regenerating or rescoring that 80/20 run. One offline, tool-free agent
call receives only the prior arm's selected label-free development prefixes and
emits a frozen pure extractor. Its stateless functions enumerate scope,
blocker-result spans, and remediation-command spans one string at a time; the
host, not the agent, scans every causal event, constructs exact-token edges, and
uniquely derives blocked, partial remediation, closure candidate, verified
closure, and newly surfaced blocker states. A deterministic verifier enforces
causal graph/state invariants, and a mandatory same-carrier
collapsed-state ablation isolates whether the relations add value; empirical
resource PMFs remain the ordinary evaluator's job. There
is no runtime agent, iterative repair, critic model, graph framework, or new
dependency. The exact structural, coverage, cost, and fresh-data gates are
frozen in `clause-interaction-kb-plan.md`.

That relational candidate is a structural development NO-GO. Its only allowed
agent call emitted a generic test-suite/blocker/install parser, but the source
assigned matching literals to local variables and failed the frozen auditable-
source validator before execution, PMF fit, or primary-contrast selection. The
call used 59,872 input and 3,047 output tokens. It was not repaired or
regenerated. This result says the single generated candidate did not satisfy
the frozen representation contract; it is not evidence that blocker-remediation
relations lack predictive value. No adjacent validation or final-test command,
output, resource label, or telemetry was consumed.

The separately preregistered declarative relational arm is also a structural
NO-GO. It did not weaken, repair, or regenerate the rejected Python candidate.
Its one offline call returned a bounded JSON rule for short `pytest` commands,
`No module named <id>` blockers, and `python3-<id>` remediation. The blocker
pattern appeared in 15 development tasks but captured only one distinct
identifier, `pytest`, below the frozen minimum of two. The arm therefore stopped
before graph replay, PMF fit, primary-pair selection, or reserved scoring. The
call used 56,982 input and 1,664 output tokens; prediction-time agent cost would
have been zero. No validation or final outcome was opened, and no retry or rule
repair is allowed. Exact protocol and artifacts remain in Section 9 of
`clause-interaction-kb-plan.md` and
`analysis/results/tool-resource-5-3-3-3-20260804/sqlglot100-declarative-relational-v1`.

The family-and-scope relational arm also ended at its frozen structural gate.
Its one offline call received the bounded event-chain inventory and returned a
strict JSON `pytest` family with targeted and suite scopes plus missing-module
and install relations. Both scope regexes used `\z`, which does not compile in
the committed Python 3.12 host, so the response was rejected before support
checks, graph replay, PMF fitting, or contrast selection. The call used 32,742
input and 1,812 output tokens in 45.002 seconds; no repair or retry is allowed.
This is evidence against this single generated candidate, not against the
family/scope/state hypothesis. Prediction-time agent cost would have been zero,
and validation/final remain unread. The independently reviewed contract and
artifact are Section 10 of `clause-interaction-kb-plan.md` and
`analysis/results/tool-resource-5-3-3-3-20260804/sqlglot100-declarative-family-state-v1`.

A separate typed-catalog selection arm is now preregistered and has not called
an agent. The host alone mines a finite catalog of executable family, work-scope,
and causal token-template configurations from label-free development episodes;
the agent can return only one catalog ID. `C000` is a deterministic support-only
selector, so fresh evaluation can distinguish agent selection from catalog
construction. No generated regex, code, parser, literal, threshold, state, or
prediction is permitted, and prediction-time agent cost remains zero. The
complete construction, cost, structural, ablation, and fresh gates are Section
11 of `clause-interaction-kb-plan.md`; validation and final remain unread.

SQLGlot is not excluded from this experiment. The original 100 SQLGlot tasks
are development-only fit evidence; the additional 100 requested tasks were
split by task ID, independent of completion state, into 50 validation and 50
final-test tasks before any reserved output or label was read. Validation may
only gate whether the byte-identical method consumes final test; it cannot tune
the method or its PMFs. The earlier all-100 blind-state held-out wording is
superseded before outcome access. Exact IDs and stage permissions live only in
`analysis/development/sqlglot-relational-task-split.json` and
`clause-interaction-kb-plan.md`.

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
  used all qualifying development episodes. Section 11 was openly amended to
  use all development episodes because its prompt contains only aggregate
  templates and counts. No resource label or reserved artifact was read.

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
