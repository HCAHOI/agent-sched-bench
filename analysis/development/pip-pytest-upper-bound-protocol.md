# pip/pytest Modeling Upper-Bound Protocol

**Frozen:** 2026-08-09
**Purpose:** development decision on already exposed SQLGlot traces; not a
confirmatory result.

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
- at least 10 scored tasks; and
- at least one causally available static semantic candidate distinct from the
  Task-Aware hard prediction.

A tool that misses this gate is `uninformative_coverage`, not a method
failure. The other tool may still be evaluated.

## Metrics and fixed decision

Report Latency, CPU, RSS, and Disk exact command accuracy; their equal-weight
mean; task-cluster bootstrap 95% intervals with 2,000 draws and seed 0;
helpful corrections; severe underpredictions; affected tasks; and pip/pytest
breakdowns. Absolute accuracy and percentage-point gain over Task-Aware are
both required.

A ceiling is materially positive only if it:

1. improves equal-weight four-target accuracy by at least 1.0 percentage
   point;
2. has a task-cluster bootstrap 95% lower bound above zero; and
3. corrects predictions in at least 10 scored tasks.

Decision:

| Static-candidate oracle | Perfect-tool oracle | Decision |
|---|---|---|
| positive | any | Keep the current static family; routing/aggregation has headroom. |
| not positive | positive | Close static argv/parser/matcher work; only richer state or compound-work modeling retains headroom. |
| not positive | not positive | Close pip/pytest-specific modeling on SQLGlot. |
| insufficient coverage | any | Do not infer failure for the uncovered tool. |

The perfect-tool oracle cannot rescue or validate the static family. Neither
oracle is a predictor result.

## Implementation and checks

Reuse existing loaders, parsers, evaluators, metrics, and bootstrap code. Add
one small evaluator and one focused test covering candidate-pool identity,
causal update order, non-tool identity, and oracle selection. Run an
independent bounded review before producing the result artifact. Do not
collect new traces or add dependencies.
