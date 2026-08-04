# Tool-Resource Prediction — Canonical Objective and Lock

**Effective:** 2026-08-04
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
[0, 2000] ms
(2000, 8000] ms
(8000, +inf) ms
```

The hard prediction is the highest-probability bucket; an exact tie selects the
shorter bucket. The primary metric is exact three-class command accuracy.
Compare it with one constant majority class computed over the identical
evaluation commands. Report eligible count, label and prediction counts, the
3x3 confusion matrix, majority accuracy, and deltas from majority and Current.

Predict CPU peak, sampled RSS, and Disk I/O independently as Heavy or Light:

```text
CPU Heavy  = peak_cpu_cores > 2.0
RSS Heavy  = sampled_peak_rss_mb > 500 decimal MB
Disk Heavy = read_bytes + write_bytes > 104857600 bytes
```

For these resource targets only, an observation explicitly marked by policy as
null with command latency below 500 ms is imputed Light. Other nulls are
unavailable. Report eligible, Heavy/Light and unavailable counts, TP/TN/FP/FN,
accuracy, majority accuracy, and constant-Light accuracy.

Old nine-bin latency accuracy, legacy 3500/5000 ms boundaries, balanced
accuracy, Brier/NLL, bucket MAE, q-error, and hand-selected subsets cannot
select a candidate.

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
- The command and its output cannot reveal its own label or query-time state.
  Hindsight execution modes are diagnostics only.

Compound commands compose empirical clause values according to shell structure:

- sequential stages: latency and Disk sum; CPU and RSS take the maximum;
- concurrent pipelines: latency takes the maximum; CPU and RSS sum;
- Disk sums across all clauses.

Never compose bucket IDs or Heavy/Light labels with Boolean OR. Unsupported or
ambiguous structures remain unavailable.

## 3. Current architecture and research boundary

`ClauseResourceKB` is the Current control. It uses raw exact argv, argv-prefix,
and binary/global backoff with frozen public evidence and causal repo-local
updates. The trie is not the proposed research contribution.

The active research question is narrower:

> Within many tasks from the same repository, can a deterministic semantic
> work signature reuse evidence that raw argv matching misses?

The active development corpus is the 100-task SQLGlot trace set. It was chosen
because task relatedness within one repository is a controlled condition.
SWE100/277 remain development-exposed historical diagnostics and may remain in
Current's already-frozen public prior, but they must not select, tune, or test a
semantic work-signature candidate. Other repositories and new collection are
outside the current semantic experiment.

All SQLGlot results are development-exposed. They may falsify mechanisms and
guide the next fixed comparison, but cannot support a confirmation claim.

The runtime boundary remains unchanged: `resource-agentd` owns parsing,
prediction, state, persistence, and orchestration; `telemetryd` owns privileged
collection and finalized observations. `src/tool_resource/` imports nothing
from the rest of the repository. Offline trace adapters live in
`src/tool_resource_eval/`. Runtime integration is not part of the current
offline semantic experiment.

## 4. Current decision tree

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

State identifiability, controlled state intervention, and scheduler
actionability remain downstream gates in `clause-interaction-kb-plan.md`. They
cannot authorize new collection or scheduler implementation by themselves.

## 5. Development-exposure record

- The Phase A numbers above were visible before Phase B was corrected from a
  cross-repository SWE test to a same-repository SQLGlot test.
- At the SQLGlot-only scope correction, no SQLGlot pytest coverage count or
  label had been read. Coverage was then frozen and passed before the Phase B
  labels above were read.
- Earlier SWE100/277 trie, generic-argv, resource, and concurrency diagnostics
  are development-exposed negative evidence. They motivate controlling
  repository relatedness but do not evaluate the SQLGlot semantic candidate.
- The first SQLGlot command-level replay made with an older public aggregate
  retained invalid downstream-pipeline evidence. It remains explicitly named
  `*.pre-public-structure-fix.invalid`; only the corrected artifact is
  reportable.
- The 2000/8000 ms objective and resource thresholds were selected after older
  SWE diagnostics were visible. All results under the current objective are
  development-only.

## 6. Non-negotiable task contract

```text
Evaluation unit = eligible exec command; clauses are internal evidence.
Latency buckets = [0,2000], (2000,8000], (8000,+inf) ms.
Primary latency metric = exact three-class accuracy on identical rows.
Resource thresholds = CPU >2 cores; RSS >500 MB; Disk >100 MiB.
Short-null resource policy = Light only when explicitly marked and <500 ms.
Causal visibility = observation end before query start, after task settlement.
Compound commands = physical stage/pipeline composition, never Boolean OR.
Current = unchanged raw exact/prefix/binary control.
Active semantic development corpus = SQLGlot100 only.
No result-dependent tuning, package/test-name outcome rules, or hindsight state.
No new collection, runtime integration, or scheduler implementation without a
separate approved protocol.
```
