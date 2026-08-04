# Semantic Work-Signature Research Plan

**Effective:** 2026-08-04

**Status:** development-only decision tree

**Scope:** same-repository semantic signatures, state identifiability, and
scheduler actionability on SQLGlot100

This plan extends `tool-resource-canonical-objective.md`. It does not change
the fixed targets, thresholds, causal visibility, task-settlement barrier,
command-level evaluation unit, or physical compound-command composition.

## 1. Research question

Can a command be represented as the work it requests, and can the predictor
determine how much of that work remains in the environment?

Proceed through the smallest decision tree that can answer that question:

1. Use SQLGlot100 to distinguish simple pip normalization from package-set
   overlap.
2. If the pip mechanism survives, test exactly one second tool, `pytest`, in
   the same SQLGlot100 causal task stream.
3. Independently ask whether causal state is identifiable in SQLGlot100;
   request a controlled collection only if a meaningful oracle gap exists.
4. Stop at a scheduler actionability gate unless a real consumer would change
   an action.

All existing corpora are development-exposed.

## 2. Phase A — pip carrier audit

Add one `canonical-exact` arm to the existing pip evaluator:

- **Current:** existing raw exact/prefix/bin predictor.
- **Canonical exact:** normalized interpreter, invocation, behavior flags, and
  complete requirements must match exactly; package order does not affect the
  key.
- **Jaccard semantic:** existing normalized package-name overlap.

All three arms use identical command rows, frozen public evidence, `alpha =
16`, task-final causal updates, and Current fallback when semantic evidence is
absent. State is report-only and cannot select the representation.

Output all 119 pip commands and identify the 15 commands that used non-exact
semantic evidence. For each carrier report task, normalized signature, each
arm's prediction, label, helpful/harmful change, and contributing evidence.

Select exactly once:

- Select Jaccard only if it is strictly more accurate than canonical exact on
  both all commands and the pip subset, its changed carriers have more helpful
  than harmful changes, and its net gain spans at least two tasks and two
  semantic signatures.
- Otherwise select canonical exact only if it satisfies the same conditions
  relative to Current.
- Otherwise stop the entire semantic-method route.

Do not change `alpha`, flags, support thresholds, or package-specific rules.
Obtain one bounded independent review before reading the formal result, then
commit the implementation and result artifact.

## 3. Phase B — fixed second tool: pytest (complete, no-go)

Run this phase only if Phase A passes. Failure does not authorize switching to
`apt` or another tool.

Add a minimal `PytestSignature` module, not a generic plugin framework:

- normalize direct `pytest` and `python -m pytest` invocations;
- represent targets only by counts and shapes (`file`, `directory`, `nodeid`,
  or none), without repository or test names;
- normalize `-x` to `maxfail=1` and retain explicit `--maxfail`;
- retain worker count or `auto`, `--dist`, collect-only, last-failed,
  failed-first, and stepwise modes;
- retain only Boolean structure and atom count for `-k` and `-m`, replacing
  identifiers with placeholders;
- return `None` and fall back to Current for an unknown result-affecting option
  or `--` passthrough;
- use exact semantic-signature matching only, with no new similarity parameter.

Replay SQLGlot100 causally in its existing manifest/task order. Current keeps
its unchanged frozen public prior. The pytest candidate may use semantic
evidence only from settled earlier SQLGlot tasks and falls back to Current when
none exists. Before reading pytest labels, require at least 100 parsed pytest
commands, coverage across at least 20 tasks, and 20 commands whose semantic
signature has causal earlier-task evidence but whose Current exact clause key
has no earlier-task match; otherwise stop.

Latency passes only if the pytest subset and overall accuracy are both strictly
higher than Current, changed commands are net helpful, positive net gains span
at least two tasks and two semantic signatures, and every non-pytest PMF is
bit-identical. Report paired
task-cluster bootstrap uncertainty without tuning on it. Only after this gate
passes, transfer the unchanged representation to CPU, RSS, and Disk without
target-specific parser or weight changes.

Observed after freezing the gate: coverage passed at 339 parsed commands, 88
tasks, and 125 non-exact carriers. Accuracy improved by four commands overall
and on the pytest subset, with four helpful, zero harmful, and three neutral
hard changes. The gain spanned four tasks but only one positive-net semantic
signature, so the frozen two-signature gate failed. Resource transfer and any
replacement second tool are stopped.

## 4. Phase C — state identifiability (complete, no-go)

This phase requires the representation selected by Phase A but is independent
of Phase B. Use SQLGlot100 only. Measure:

- how many independent tasks repeat each semantic signature;
- whether the same signature appears under different pre-query causal states;
- whether those states cross latency or resource buckets;
- how many Current errors a hindsight execution mode can fix under
  leave-one-task-out evaluation; and
- the gap between causal state and the hindsight mode oracle.

Prepare a controlled collection only if at least eight package sets occur in
two or more execution modes across independent tasks, the oracle mode nets at
least ten hard-error fixes over semantic-only prediction, and the gain is not
concentrated in one task or package set. Otherwise record that the existing
workload has no identifiable state contrast and stop state modeling.

Observed: 26 package-set signatures included ten repeated across tasks, zero
with multiple causal pre-query states, and one with multiple execution modes.
The semantic-only and selected causal-state arms were identical at 100/119;
the hindsight execution-mode leave-one-task-out oracle scored 98/119, with 11
helpful and 13 harmful changes. The gate failed, so state modeling and the
paired collection stop here.

## 5. Phase D — paired state intervention (not reached)

This is the only allowed new collection. It is not pre-authorized: after a
smoke and resource estimate, pause for explicit approval before the expected
one-to-three-hour run.

With seed 42, select 12 real successful package sets from SQLGlot100 spanning
sizes 1, 2, and 3+, excluding local, VCS, and path installs. Execute each set
twice in each of four isolated states, for 96 real executions total:

1. pip absent;
2. pip present, empty cache, requested packages not installed;
3. requested wheels cached, requested packages not installed;
4. requested top-level packages already installed.

Randomize order with seed 42 and keep image, index/network configuration, and
eBPF telemetry fixed. Before the query, state may inspect only pip
availability, installed distributions, and requested-wheel cache inventory.
It may not inspect current output, the final dependency graph, or download
volume. Measure and charge state-probe latency and resources.

The primary comparison is leave-one-package-set-out semantic-only versus
semantic-plus-state three-bucket accuracy. Continue only if at least 3 of 12
sets cross a latency bucket across states, state-aware accuracy improves by at
least five percentage points, changed predictions are net helpful, and probe
p95 is below 100 ms. A failure ends the state route; do not add states or
replace package sets.

## 6. Phase E — scheduler actionability (complete, no-go)

Do not implement a scheduler. For every surviving candidate, count changed
latency/resource hard decisions and independent tasks, then identify whether a
real repository consumer would take a different action.

Write a separate scheduler protocol only if a named consumer changes action on
at least 20 commands across at least 10 tasks and false-action and
missed-action costs can be stated. Otherwise stop at predictor representation
and make no scheduling-utility claim. CacheWise timeout is a ceiling, not a
remaining-work proxy, and its ranking is not a default consumer.

Observed: latency changed five commands across five tasks, Disk changed four
across four, and CPU/RSS changed none. The union is six commands across six
tasks, below both frozen thresholds. No repository consumer outside the
tool-resource service maps these predictions to a scheduling action. No
scheduler protocol or implementation is authorized.

## 7. Interfaces, artifacts, and acceptance

- Reuse the causal command evaluator; do not create another evaluator
  framework.
- Keep a narrow `PytestSignature` parser in its own module only if Phase B is
  reached. Do not add agent-generated adapters, daemons, snapshots, runtime
  integration, or new dependencies.
- Extend the existing aggregate loader if Phase B needs command duration, raw
  output, or clauses; do not add a second trace reader.
- Emit one machine-readable result and the necessary row sidecar per phase,
  and update the existing self-contained HTML summary.
- Verify identical command IDs, labels, and availability across arms; require
  non-target-tool PMFs to be bit-identical.
- Preserve `observation.ts_end < query.ts_start`, task settlement before
  learning, and the current latency/CPU/RSS/Disk compound-composition rules.
- Add focused tests for pip order normalization and partial overlap. If Phase B
  is reached, test invocation equivalence, options, selection-expression shape,
  and fail-closed parsing.
- Obtain one bounded independent review after each substantial evaluator/parser
  phase and before its output becomes evidence.
- Commit each completed phase, including a negative gate result. Do not run an
  unrelated full suite.

SQLGlot100 is development-exposed. SWE100/277 are not semantic-method evidence
in this plan. Fresh-repository confirmation is deferred until the actionability
gate yields a minimum useful effect and a task-cluster power/precision analysis
justifies consuming it.
