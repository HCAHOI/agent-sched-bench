# pip/pytest Modeling Upper-Bound Protocol

**Frozen:** 2026-08-09
**Status:** run once and recorded on 2026-08-09; aggregate decision
`mixed_by_tool`. See Outcome below.
**Purpose:** development decision on already exposed SQLGlot traces; not a
confirmatory result.

**Amendment, 2026-08-09:** independent implementation review found that a
global verdict could let an uncovered tool determine the covered tool's
decision. Before any 100-task oracle result was produced, the decision was
made per tool using its marginal full-cohort arm, as specified below. Only a
one-task plumbing smoke without aggregate outcomes was visible.

## Question

Can explicit pip/pytest modeling materially improve the current Task-Aware
Command Predictor in the best case? If not, stop this line of work.

The tested static family is deliberately narrow: existing deterministic
pip/pytest parsers, complete-command memory, semantic-signature matching,
package/target overlap, causal task-final observations, and the current host
composition. This experiment does not test installed-package state, cache
state, network state, command output, or runtime telemetry.

## Evidence boundary

Use the existing SQLGlot docs-v1 cohort: 100 warm-up tasks followed by 100
scored tasks in the frozen order. All of it is development-exposed. At freeze
time, the existing semantic-work-unit results and an uncommitted exploratory
approximation of oracle headroom were visible. Consequently, the result may
guide development but cannot support a fresh validation claim.

Every arm has identical command IDs, target labels, availability, task order,
and 5/3/3/3 bucket definitions. Scored-task observations become visible only
after whole-task settlement. Hindsight is used only by the two explicitly
named oracle selectors.

## Arms

### Baseline: Task-Aware

Reproduce the frozen Task-Aware prediction already evaluated by
`evaluate_doc_tool_semantics.py`. This is the primary baseline; Clause-KB is
not the comparison because Task-Aware already contains the useful pip/pytest
semantic heads.

### Static-candidate oracle

For each scored pip/pytest command and target, form a label-independent pool
from predictions already produced by existing causal code:

- Task-Aware;
- Clause-KB fallback;
- complete-command memory;
- pytest collapsed signature and target overlap, when available; and
- pip package overlap, when available.

No parser, similarity, support, weighting, or phase rule may change. The
oracle may inspect the current label only to select one hard bucket from this
pre-existing pool. If no candidate is correct, it retains Task-Aware. All
non-pip/pytest rows remain bit-identical to Task-Aware.

This is the upper bound of routing the current static candidate family; it is
not a deployable predictor.

### Perfect-tool oracle

For any command containing a parsed pip-install or pytest clause, replace each
available Task-Aware target with its true bucket. Leave every other command
bit-identical to Task-Aware. This is an intentionally unattainable ceiling on
the total error mass available to any pip/pytest-specific model, including a
richer state-aware model.

## Coverage gate

Before scoring oracle outcomes, report for pip and pytest separately:

- at least 20 scored non-exact commands;
- at least 10 scored tasks containing the tool; and
- at least one causally available static semantic candidate distinct from the
  Task-Aware hard prediction.

A tool that misses this gate is `uninformative_coverage`, not a method
failure. The other tool may still be evaluated.

## Metrics and fixed decision

Report Latency, CPU, RSS, and Disk exact command accuracy; their equal-weight
mean; task-cluster bootstrap 95% intervals with 2,000 draws and seed 0;
helpful corrections; severe underpredictions; affected tasks; and pip/pytest
breakdowns. Absolute accuracy and percentage-point gain over Task-Aware are
both required. Materiality is decided separately for pip and pytest: the
selected tool receives its oracle arm while every other row remains
Task-Aware, so the gain remains an overall full-cohort effect rather than a
within-tool accuracy.

A ceiling is materially positive only if it:

1. improves equal-weight four-target accuracy by at least 1.0 percentage
   point;
2. has a task-cluster bootstrap 95% lower bound above zero; and
3. corrects predictions in at least 10 scored tasks.

Decision for each tool that passes coverage:

| Static-candidate oracle | Perfect-tool oracle | Decision |
|---|---|---|
| positive | any | Keep the current static family; routing/aggregation has headroom. |
| not positive | positive | Close static argv/parser/matcher work; only richer state or compound-work modeling retains headroom. |
| not positive | not positive | Close that tool's special modeling on SQLGlot. |
| insufficient coverage | any | Do not infer failure for that tool. |

The perfect-tool oracle cannot rescue or validate the static family. Neither
oracle is a predictor result. If pip and pytest receive different decisions,
the aggregate status is `mixed_by_tool`; an uncovered tool never changes the
other tool's verdict.

## Implementation and checks

Reuse existing loaders, parsers, evaluators, metrics, and bootstrap code. Add
one small evaluator and one focused test covering candidate-pool identity,
causal update order, non-tool identity, and oracle selection. Run an
independent bounded review before producing the result artifact. Do not
collect new traces or add dependencies.

## Outcome

Receipt:
[`../results/pip-pytest-upper-bound-sqlglot-v1/result.json`](../results/pip-pytest-upper-bound-sqlglot-v1/result.json),
over 1,792 scored commands in 100 scored tasks after 100 warm-up tasks. The
aggregate decision is `mixed_by_tool`, as the per-tool rule above requires.

**Coverage gate.** pytest passed; pip did not.

| Tool | Scored non-exact commands (≥20) | Scored tasks with the tool (≥10) | Distinct static semantic candidates (≥1) | Gate |
|---|---:|---:|---:|---|
| pip | 44 | 58 | 0 | not met |
| pytest | 202 | 100 | 68 | met |

**pip: `uninformative_coverage`.** No causally available static semantic
candidate ever differed from the Task-Aware hard prediction, so the third
coverage condition failed. The coverage gate above calls such a tool
`uninformative_coverage` rather than a method failure, and the decision table's
last row says no failure may be inferred for it. Its marginal full-cohort oracle
arms were recorded but are not a verdict: the static-candidate oracle gained
0.034 percentage points (95% interval [0.000, +0.107] pp, 2 helpful and 0
harmful changes in 1 task) and the perfect-tool oracle gained 0.392 percentage
points (95% interval [+0.170, +0.676] pp, 25 helpful and 0 harmful changes in 13
tasks). Neither reaches the 1.0-percentage-point materiality threshold, and the
receipt marks both `materially_positive: false`.

**pytest: `keep_static_family`.** The static-candidate oracle gained 1.941
percentage points of equal-weight four-target accuracy (95% interval [+1.326,
+2.660] pp, 118 helpful and 0 harmful changes in 44 tasks), clearing all three
materiality conditions. The perfect-tool oracle gained 5.494 percentage points
(95% interval [+4.535, +6.466] pp, 345 helpful and 0 harmful changes in 81
tasks). A positive static-candidate ceiling selects the first row of the
decision table: keep the current static family, because routing and aggregation
still have headroom.
