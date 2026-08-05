# Task-Local Resource-State Research Plan

**Effective:** 2026-08-04

**Status:** family-and-scope relational candidate ended at its structural gate;
same-repo validation/final task partitions remain scientifically unread

This plan extends `tool-resource-canonical-objective.md`. The earlier semantic
KB and early-execution routes are complete negative results. The final question
was narrower:

> Can a label-free offline agent define reusable command families, current work
> scope, and dependency state that improve causal command-resource prediction?

## 1. Evidence and hypothesis

The fixed SQLGlot 80/20 replay has seven unique severe latency or High-to-Low
CPU/RSS errors. All seven are single-clause `pytest` or `make test` full-suite
commands. In every case, earlier full-suite attempts in the same task failed
quickly because the interpreter or dependencies were unavailable; after setup,
the suite ran for 92--138 seconds with High CPU and RSS.

This is observed evidence. The hypothesis is that a task-local retry phase can
separate these modes. The 20 test tasks and their labels are already
development-exposed, so the first replay can only develop or falsify the method.

## 2. Frozen development arm

Current remains the control. The candidate recognizes only a single-clause:

- pytest invocation with no path, node, `-k`/`-m`, max-fail/exit-first,
  collection, last-failed, or stepwise selection; or
- exact conventional `make test`, ignoring leading environment assignments.

Before each command, count earlier completed full-suite exec calls in the same
task, across both invocation forms. The count is derived from command text only;
previous output, exit status, duration, telemetry, and future calls are hidden.

The first 80 tasks fit one empirical bucket PMF for third-or-later full-suite
calls. On the final 20 tasks, the candidate may replace Current only when:

1. the current call is the third or later full-suite call;
2. the learned hard bucket is higher than Current; and
3. the target is latency, CPU, or RSS.

Disk remains bit-identical to Current. Earlier test phases and all non-test
commands remain bit-identical to Current. This monotone arm is intentionally a
dangerous-underprediction correction, not a replacement KB.

The development continuation rule was chosen after the seven error cases and
the 80/20 phase table were visible. Continue only if the formal causal replay:

- does not reduce exact accuracy for latency, CPU, RSS, or Disk;
- does not increase severe underprediction for any target;
- has more helpful than harmful changed predictions; and
- improves commands from at least three test tasks.

The reviewed formal replay passed. Across the identical 349 commands, latency
accuracy rose from 77.937% to 79.083%, CPU from 90.476% to 93.333%, and RSS from
91.561% to 94.093%. Severe underpredictions fell from 4 to 0, 7 to 1, and 7 to
1 respectively. Disk stayed at 81.375% with bit-identical PMFs. Six commands
changed at least one target; all 16 target-level changes were helpful, none were
harmful, across five tasks. This remains post-hoc development evidence.

Artifacts:

- `analysis/results/tool-resource-5-3-3-3-20260804/sqlglot80-20-full-test-phase.json`
- `analysis/results/tool-resource-5-3-3-3-20260804/sqlglot80-20-full-test-phase.rows.jsonl`

## 3. Agent-guided residual discovery

A read-only Codex `gpt-5.6-sol` pass inspected only the development-exposed
SQLGlot100 rows and their causal trace prefixes. The adjacent SQLGlot100
held-out collection was explicitly forbidden and remained unread. The agent
proposed five generic trace-history patterns; a separate mechanical
reconciliation verified that every cited support row is a real residual error.

The selected hypothesis is scope-conditioned test-environment readiness. It
distinguishes an unavailable runner, a runnable but collection-blocked suite,
and a suite whose dependency closure is satisfied. Its 11 cited error commands
span six tasks and contain 32 latency/CPU/RSS target errors; nine matched correct
controls span nine tasks. These are opportunity counts, not repaired
predictions. No candidate replay or accuracy gain is claimed.

This pass establishes that attempt count is only a proxy for environment state,
but it does not yet define an executable predictor. Before the follow-up, a
small generic result-category schema and the amendment allowing earlier
completed exit status and result categories were frozen. The adjacent task
outputs were not used in this discovery.

Artifact:

- `analysis/results/tool-resource-5-3-3-3-20260804/sqlglot-agent-pattern-discovery.json`

## 4. Blind causal-state replay

This development-only arm tests whether an agent can turn prior command results
into a useful state without seeing the resource labels it will be judged on.
For every recognized full-suite query in all 100 existing SQLGlot tasks, the
agent receives an opaque query ID, the current command, and the causally earlier
completed exec commands with their exit status and bounded raw result excerpt
(first and last 2,000 characters). It receives no task identity, durations,
telemetry, bucket labels, current result, or future event. The fixed model is
Codex `gpt-5.6-sol`; its output is exactly one of:

- `unavailable`: the selected runner or interpreter cannot start and no later
  successful remediation is visible;
- `collection_blocked`: the runner starts, but the latest suite evidence stops
  in import, collection, or setup and no later remediation is visible;
- `ready`: a prior suite reached test execution, or every named earlier
  runner/collection blocker has a later successful explicit remediation; or
- `unknown`: evidence is absent, conflicting, or does not meet those rules.

Each query is classified in a separate ephemeral Codex process; no model
context contains another query. The fixed request uses the requested fast tier,
medium reasoning, and no temperature parameter because the Codex provider does
not expose one. Any tool call by the classifier invalidates that query. The
runner retains the structured event transcript, rejects pre-existing output,
and binds the replay artifact to the exact canonical packet bytes.

The agent emits only state, cited prior-event indices, and a short rationale; it
does not predict latency, CPU, RSS, or Disk. The first 80 tasks fit an empirical
bucket PMF per non-unknown state for latency, CPU, and RSS. A state is usable for
a target only with eligible evidence from at least three distinct warm-up tasks.
On the final 20 tasks, a usable state PMF replaces the reviewed full-test-phase
PMF in either direction; otherwise that arm is unchanged. Disk and every
non-full-suite prediction remain bit-identical to the full-test-phase arm.

The replay continues only if, relative to the full-test-phase arm, latency, CPU,
and RSS exact accuracy are each no lower; severe underprediction is no higher
for any; target-level helpful changes outnumber harmful changes; and helpful
changes cover at least three test tasks. Identical eligible rows and the stated
PMF-identity checks are validity conditions. These criteria were frozen after
the discovery counts were visible, so any result remains post-hoc development
evidence.

## 5. Offline-generated causal extractor

This development-only arm tested whether an agent is useful as a batch feature
program generator rather than a per-command predictor. It consumed no adjacent
task output.

The first 80 development tasks are exposed through a generic catalog. Every
eligible command has opaque task and sample IDs, command text, parsed clauses,
the four labels, and Current's four causal hard predictions. PMFs are omitted
because they do not change which hard residuals the discovery step can target
and would dominate its context. The catalog
builder contains no executable, option, package, test, or result-text allowlist.
The first agent call sees the complete catalog and may select at most 12 samples
from at most 12 tasks for one hypothesis, or abstain. The second call sees only
those selected rows plus their causal task prefixes: earlier
command text, exit status, and a first/last result excerpt bounded to 500 total
characters per event. Both calls use Codex `gpt-5.6-sol`, requested fast tier,
medium reasoning, no tools, and separate ephemeral contexts. Their combined
serialized prompts must not exceed 800,000 UTF-8 bytes. Actual tokens and wall
time are recorded; the byte limit is an engineering cost ceiling, not a success
criterion.

The output is exactly one frozen Python `extract(query)` function or an
abstention. At prediction time `query` contains the current command, parsed
clauses, and only causally earlier events from the same task. The function may
return `rule_id`, `state`, and `evidence_event_indices`, or `None`; it cannot
return a resource bucket or PMF. It receives no task/repository identity,
timing, telemetry, labels, Current prediction, current result, future event,
filesystem, network, clock, or randomness. The host validates and runs it with
restricted builtins and fits the four empirical PMFs independently. A
signature-target is usable only with eligible support from at least five
distinct warm-up tasks; otherwise that target falls back to Current.

The source is generated once and frozen before the final 20 development tasks
are scored. Candidate and Current use identical command rows, labels, order,
availability, compound composition, and dynamic task-final updates. The
candidate may replace Current in either direction for any of latency, CPU, RSS,
or Disk when support exists. Continue only if:

- exact accuracy is no lower for every target;
- severe underprediction is no higher for every target;
- target-level helpful changes outnumber harmful changes; and
- helpful changes cover at least three test tasks.

Report all commands, plus selected-scope coverage, abstentions, generated-rule
support, agent token cost, and local extractor p50/p95 latency. An abstention or
failed gate is a complete negative result; do not regenerate the source, change
the prompt or support threshold, or report only its selected subset. A positive
result is still development evidence. To attribute the mechanism, also replay
the same frozen source with an empty causal history and compare against the
existing non-agent full-test phase arm; neither diagnostic selects the
candidate.

Implementation amendment before any candidate PMF or final-20 score was read:
the first frozen source used a regular expression, while the validator had
incorrectly required the regex source text itself to occur in five tasks. Regex
support is instead counted by applying that frozen expression to warm-up text;
package/test tokens inside it remain forbidden. The exact source and both prompt
hashes stay unchanged, and replay performs no additional model call.

The frozen replay is complete and is a no-go. The agent autonomously selected
the repeated `collection failure -> successful install -> full-suite retry`
family and generated one local causal rule. It fired on 31 warm-up commands
across 15 tasks and six test commands across three tasks. However, 14 of the 15
warm-up tasks contained both a 0.5--2 second retry and a later over-30-second
execution under the same generated signature, usually in that order. The fitted
state was therefore intrinsically bimodal rather than a remaining-work state.

All four exact accuracies were unchanged from Current: latency 77.937%, CPU
90.476%, RSS 91.561%, and Disk 81.375%. Severe underprediction fell from 1.146%
to 0.860% for latency, 3.333% to 2.857% for CPU, and 2.954% to 2.532% for RSS,
but every rescued low-to-high error was paired with an incorrect high
overprediction. Across targets there were five helpful and five harmful changes
covering only two helpful tasks, failing both frozen positive-net and three-task
requirements. Empty-history replay produced no signature, confirming that the
rule used causal history rather than command text alone.

The two agent calls consumed 213,257 input and 2,077 output tokens in 45.43
seconds, within the 541,465-byte combined prompt bound. The frozen local rule's
p95 was 0.001843 ms over 1,792 queries, so runtime agent cost was zero. The
negative mechanism result is that recognizing “an install happened” does not
establish dependency closure; useful follow-up would have to relate the named
collection blocker to the exact remediation or otherwise distinguish partial
from complete remediation. This run must not be regenerated on the exposed
split.

## 6. Same-repository evidence isolation

SQLGlot is deliberately both the fit and evaluation repository because it is
currently the only repository with enough independent tasks. Repository
membership is not excluded. Leakage is controlled only by disjoint task IDs and
stage permissions recorded in the authoritative manifest:

`analysis/development/sqlglot-relational-task-split.json`

The manifest was generated without reading reserved task output, labels, or
telemetry: lexicographically sort the 100 requested adjacent task IDs, shuffle
with Python `random.Random(20260805)`, assign the first 50 to validation and the
remaining 50 to final test. Completion status did not affect assignment. The
three disjoint roles are:

- `development` — the original 100 SQLGlot tasks. Their commands, outputs,
  resource labels, and previous evaluation results are fully exposed. They may
  generate the source, fit PMFs, and select the single primary state contrast,
  but they cannot support a validation or generalization claim.
- `validation` — 50 additional task IDs. Before source, evaluator, tests,
  structural gate, and primary contrast are reviewed and committed, only IDs,
  process status, file existence/counts, and collection-completion metadata may
  be read. Then this partition may be scored once with the frozen artifact. It
  may decide whether to consume final test, but cannot change prompt, source,
  parser semantics, state reducer, PMFs, support, coverage, targets, or gates.
- `final_test` — the other 50 additional task IDs. Until validation passes, the
  same metadata-only rule applies. A validation no-go leaves all final command
  output, labels, telemetry, and scores unread. A validation go authorizes one
  unchanged final evaluation; no refit on validation tasks is allowed.

Current and every candidate use the same frozen public evidence and all 100
development tasks as settled same-repository prior. Neither arm updates within
validation or final test in the primary comparison. Any public, development,
validation, and final task-ID overlap is a hard validity failure. Report every
requested reserved task as accepted, failed, unfinished, or telemetry-invalid;
never silently exclude it.

The earlier plan to use all 100 adjacent tasks as one blind-state held-out
primary is superseded before any adjacent outcome was read. The blind classifier
remains a development method, not a claim-bearing use of final test. This is a
task-held-out same-repository experiment, not temporal or cross-repository
generalization: the adjacent tasks precede the development tasks by creation
time.

## 7. Relational closure arm

The offline-generated extractor's no-go is evidence against the state
`an install succeeded`, not against offline agent-generated features. That state
merged a short retry with a later real suite execution. The next experiment
tests one causal hypothesis only:

> Does retaining which blocker a successful action addresses, and whether a
> later verifier confirms or refutes closure, separate those execution modes?

### 7.1 Design preflight

Provenance and object-centric event-log standards represent typed events,
objects, and qualified relations; planning formalisms represent actions through
preconditions and effects. Agent workflow libraries add graph execution, while
ReAct, Reflexion, Self-Refine, and CRITIC add iterative model feedback. The
needed experiment is smaller than any of those systems: an ordered in-memory
event graph plus a deterministic state reducer is sufficient. No graph library,
workflow runtime, online agent loop, or LLM critic is added. A critic would have
no independent fresh signal before the command finishes; static source and
graph invariants provide cheaper external verification.

The agent remains useful only as an offline compiler for semantic extraction.
There is exactly one tool-free Codex `gpt-5.6-sol` call at requested fast tier
and medium reasoning. It receives the previous arm's already-selected twelve
development examples and their causal prefixes, with labels and Current
predictions removed. It also receives the relational output contract below. It
does not receive any adjacent-collection data. The serialized prompt is capped
at 200,000 UTF-8 bytes. An abstention, invalid source, or failed structural gate
is a no-go; there is no prompt repair, regeneration, or second candidate.

### 7.2 Frozen event graph and states

The generated source does not emit a query graph or state. It defines exactly
three stateless pure functions:

- `scope(current_command, parsed_clauses) -> rule_id | None`;
- `blocker_spans(result_excerpt) -> list[[start, end]]`; and
- `remediation_spans(command) -> list[[start, end]]`.

Each span refers to the single input string supplied to that function. The host
calls `blocker_spans` on every causally earlier failed event whose
whitespace-collapsed command equals the current command, and calls
`remediation_spans` on every causally earlier successful event. The parser sees
one string at a time, not the task prefix, so it cannot omit a blocker based on
which later remediation happens to match. A failed verifier with no returned
blocker span contributes no state.

The host case-folds each cited span, trims surrounding punctuation, and
collapses each run of non-alphanumeric characters to one hyphen. A remediation
edge exists only when its normalized successful-command span exactly equals an
earlier normalized blocker span. This exact-token relation is conservative:
aliases are missed rather than guessed. The host rejects empty or path-like
identifiers, overlapping or out-of-range spans, duplicate occurrences, and
non-causal events. It enumerates every span returned by the frozen parser; the
agent cannot choose graph nodes or edges per query.

A verifier is any prior event whose command equals the current command after
collapsing whitespace. This prevents an unrelated successful command from
certifying closure; equivalent-but-differently-spelled invocations are
conservatively missed.

The host derives one state with the following ordered, mutually exclusive
rules:

1. `newly_surfaced`: a nonzero verifier after a successful remediation
   introduces a different active blocker;
2. `partial_remediation`: at least one blocker is resolved and one remains
   active, without satisfying rule 1;
3. `closure_verified`: all recognized blockers are resolved and a later
   verifier exits zero;
4. `closure_candidate`: all recognized blockers are resolved, but rule 3 is not
   met;
5. `blocked`: at least one blocker is active and none is resolved.

No graph returns a state if it satisfies none of these rules. A verifier is used
only for ordering plus its recorded exit status; the host does not accept an
agent assertion that a nonzero verifier “progressed.”

`closure_candidate` is deliberately not treated as complete closure. The
extractor cannot use the current result to promote the current query. Empty
history produces no graph, and blank strings must produce no spans or scope.
The existing generated-source sandbox, opaque-ID rejection, and
package/test/file-specific literal checks remain in force.

### 7.3 Development fit and structural gate

After the source is frozen, replay it over all 100 development-exposed SQLGlot
tasks. Do not score or optimize an 80/20 split. Fit one empirical PMF per
`(rule_id, state, target)` for latency, CPU, RSS, and Disk, using a state-target
only when it has eligible observations from at least five distinct development
tasks. This retains Section 5's support threshold unchanged.

Before consuming fresh labels, continue only if all of these mechanical checks
pass:

- at least one pair of non-null states under the same `rule_id` is each observed
  in five development tasks and has different hard PMF modes for at least one
  target;
- `closure_candidate` never satisfies the graph invariant for
  `closure_verified`; and
- every non-null output passes the causal and relational verifier.

These checks establish that the representation can change a prediction; they
do not select a threshold or source variant. Before fresh data is opened, choose
one primary contrast mechanically from all qualifying same-rule state pairs:
maximize the smaller state task-support, then combined task-support, then break
ties lexicographically by `(rule_id, sorted state names)`. The primary target is
the first of latency, CPU, RSS, and Disk whose fitted hard modes differ for that
pair. Record the selected pair and target in the frozen artifact.

The one frozen generation attempt failed this structural gate before any fresh
task result was read. The agent proposed a generic test-suite scope plus named
missing-command/module blockers and install arguments, but its source assigned
matching literals to local variables and therefore violated the precommitted
auditable-source restriction. The validator rejected it before execution, PMF
fit, or primary-contrast selection. The call used 59,872 input tokens and 3,047
output tokens; no second prompt, repair, or candidate is allowed. This is a
development structural NO-GO for the relational arm. Validation and final-test
commands, outputs, labels, and telemetry remain unconsumed, so Sections 7.4 and
8 retain the frozen protocol but are not executed for this candidate.

### 7.4 Fresh evaluation and frozen gate

The original 100 tasks are the fixed fit corpus. Validation and final test are
the exact same-repository task-ID partitions in Section 6. The fitted source,
PMFs, collapsed-state ablation, primary contrast, and Current repository KB stay
byte-identical across both evaluations. No within-reserved update is part of
this experiment.

On each reserved partition separately, before labels are read, require
label-free carrier coverage of at least 20 commands across five tasks. Each
state in the frozen primary contrast must appear on at least five commands from
three tasks. Failure is a coverage no-go and labels for that partition remain
unscored. If coverage passes, the candidate replaces frozen Current
independently for any target with a usable development state PMF; every
unsupported command-target falls back to bit-identical Current.

Fit a mandatory collapsed-state ablation from the identical development
outputs, replacing `(rule_id, state)` with `rule_id` while preserving the same
carrier, commands, evidence, support threshold, targets, and fallback. This is
the direct control for “the new extractor found a better carrier, but the
relations did no work.”

Validation continues to final test only if, relative to frozen Current on
identical eligible command rows and labels, all of these gates pass:

- exact accuracy is no lower for latency, CPU, RSS, or Disk;
- severe underprediction is no higher for any target and lower for at least
  two targets;
- target-level helpful changes outnumber harmful changes; and
- helpful changes cover at least three fresh tasks;
- exact accuracy is no lower than the collapsed-state ablation for any target;
  and
- on the frozen primary target and only the fresh commands in the frozen
  primary state pair, the relational arm has strictly more correct predictions
  than the collapsed-state ablation.

The final-test GO gate is identical and independent. The overall method is GO
only if both validation and final test pass; a final coverage failure is a
no-go. Report accuracy and changes by relational state and by the frozen
same-rule state pair, but no subset can select the method. Once validation is
read, those 50 tasks are development-exposed; the final 50 remain untouched
until the transition above is satisfied.

## 8. Implementation and checks

- Reuse the existing command loader, pytest parser, Current replay, bucket
  labels, and metrics. Add no KB class or runtime integration.
- Raw exec order comes from accepted final trace actions. Every scored
  full-suite command must have a matching causal phase.
- The output contains one result JSON and one command-row sidecar. Changed rows
  record phase, Current PMF, candidate PMF, truth, and helpful/harmful status.
- Focused tests cover direct/module pytest, `make test`, exclusions, cross-form
  counting, monotone fallback, identical rows, and Disk identity.
- A bounded independent review is required before the formal development
  artifact is treated as evidence.
- Commit the method before reading the held-out collection. Do not tune the
  detector, third-attempt boundary, targets, or gate from held-out results.
- The generated-extractor arm reuses this evaluator and raw-event loader; it
  adds no KB class, runtime service, online agent call, rule DSL, or registry.
- The relational arm reuses Section 5's selected evidence, query builder,
  source sandbox, Current evaluator, PMF fitter, and hard metrics. Its only new
  reusable logic is relation-record validation and frozen-fit/fresh-test
  orchestration.
- Focused tests cover span normalization, unsuccessful and mismatched
  remediation, every ordered state rule, dangling and non-causal edges,
  empty/neutral history, same-rule support, the collapsed-state ablation,
  fit-only PMFs, frozen Current identity, and fresh task disjointness.
  Independent review checks leakage and gate implementation before any
  adjacent-collection output or label is opened.
- The evaluator accepts the split manifest and one explicit role. It rejects a
  task outside that role, any development/reserved overlap, validation before a
  frozen artifact, final test without a passing immutable validation result, or
  fit/source hashes that differ between stages.

## 9. Declarative relational arm

**Frozen before generation or reserved-outcome access: 2026-08-05.** This is a
new arm, not a repair, reinterpretation, or continuation of the rejected Python
candidate in Section 7. Its question is unchanged:

> Can blocker identity, exact remediation, and verified or incomplete closure
> separate resource modes that command text and an unstructured “install
> happened” feature merge?

**Outcome: structural NO-GO.** The single allowed call produced one generic
Python-test rule: scope on short `pytest` commands, extract `No module named
<id>` blockers, and treat successful `python3-<id>` commands as remediation.
Independent development replay found blocker matches in 15 tasks but only one
distinct captured identifier, `pytest`, below the frozen minimum of two. The
candidate therefore failed before graph replay, PMF fit, primary-pair selection,
or reserved coverage/scoring. The call used 143,623 prompt bytes, 56,982 input
tokens, 1,664 output tokens, and 30.812 seconds. It was not repaired or retried;
validation and final remain unopened.

### 9.1 One-shot agent contract and cost

There is exactly one new tool-free Codex `gpt-5.6-sol` call at requested fast
tier and medium reasoning. It receives the same twelve label-free development
examples and causal prefixes used by Section 7, with exact runtime parsed-clause
objects. It receives no labels, Current predictions, durations, telemetry,
reserved task IDs, or reserved files. The serialized prompt is at most 200,000
UTF-8 bytes. An abstention, invalid specification, failed support check, or
failed structural gate is a complete NO-GO: no prompt repair, schema change,
second generation, or hand translation is allowed.

The response is one strict JSON object with exactly these fields:

```text
abstain: boolean
rule_id: string
scope_patterns: array[string]
blocker_patterns: array[string]
remediation_patterns: array[string]
explanation: string
```

On abstention, `rule_id` and all arrays are empty. Otherwise `rule_id` matches
`[a-z][a-z0-9_-]{0,39}`; there are 1--4 scope patterns of at most 256 ASCII
characters and 1--6 blocker plus 1--6 remediation patterns of at most 512 ASCII
characters each. The agent cannot emit code, flags, buckets, states, weights,
thresholds, relations, or per-query decisions.

### 9.2 Bounded regex and span semantics

The host compiles every pattern with fixed ASCII and case-insensitive flags.
Scope patterns have no capture groups and are full-matched against the
whitespace-collapsed current command. The host additionally requires one
successfully parsed, non-pipeline, non-loop, non-substitution clause.
Blocker/remediation patterns each have exactly one named capture `id`, no other
group, and are applied with `finditer` to one failed-verifier result excerpt or
one successful command respectively.

The allowed regex grammar contains only ASCII literals, character classes and
ranges, standard character categories, string/word anchors, and bounded repeats
whose upper bound is at most 128. Alternation, wildcard dot, lookaround,
backreferences, conditional groups, inline flags, unbounded repeats, nested
repeats, and every group except `(?P<id>...)` are rejected. Across all
variable-width repeats in one pattern, the product of possible repeat counts
may not exceed 1,024; this rejects bounded-repeat backtracking explosions before
execution. Inputs are bounded
to 8,192 command characters, the existing 500-character result excerpt, and 256
prior events. Each pattern may emit at most 16 spans per input. Each query may
contain at most 64 blocker nodes and 64 remediation edges. Development overflow
is structural NO-GO; fresh overflow produces explicit per-query Current
fallback and is reported.

Anchors are allowed only at the two outer pattern boundaries; `\B` and anchors
inside a capture or repeat are rejected. A case-insensitive character class
whose choices all encode one literal is also rejected. These restrictions stop
zero-width and case-class spellings from hiding fixed literals.

Every captured span is passed through Section 7's exact host validator:
2--80 normalized characters, no path separators, no overlap, and no duplicate
normalized identifier within an input. A pattern must match at least five
development tasks. Every blocker/remediation pattern must capture at least two
distinct normalized identifiers across those tasks. Scope and remediation
patterns are rejected when their match depends on a task-specific positional
argument after mechanical argument masking. Decoded fixed regex literals from
all three pattern classes are also checked against development positional
arguments and opaque IDs, so escaped spellings do not bypass the check.

### 9.3 Host graph, fit, and ablations

The deterministic graph and ordered states are exactly Section 7.2: blocker
mentions arise only from prior failed exact-command verifiers, remediation
mentions only from prior successful commands, and an edge requires exact
normalized identifier equality. The host alone derives `newly_surfaced`,
`partial_remediation`, `closure_verified`, `closure_candidate`, or `blocked`.
Prediction-time agent calls are zero.

Replay the frozen specification over all 100 development tasks and fit three
independent arms with the same five-distinct-task target support:

1. **relational:** PMF per `(rule_id, state, target)`;
2. **collapsed-state:** on the identical non-null relation carrier, PMF per
   `(rule_id, target)`; and
3. **scope-only:** on every command matching scope regardless of history, PMF
   per `(rule_id, target)`.

Current is unchanged and remains the fallback. The structural gate requires a
same-rule pair of relational states, each supported by five development tasks,
with different hard PMF modes for at least one target; all graph and bound
checks; empty/blank-history neutrality; and host extraction p95 no greater than
5 ms. Primary-pair selection and target ordering are exactly Section 7.3. All
three PMF tables, the primary pair, exact JSON bytes, prompt/transcript, split
hash, the byte identities of the prior twelve-example artifact and catalog,
public-input identities, and a fingerprint over task IDs plus full clause and
command rows are frozen before reserved scoring. The artifact also records the
clean committed host-code revision and result-affecting paths; later evaluation
rejects any committed or working-tree change under those paths.
The prior-example and public-telemetry paths and hashes are committed in the
split manifest before the call, rather than chosen by the generation command.
The split manifest, this protocol, and the canonical objective must all be
unchanged files in the same commit as the reviewed host before the call starts.
Structural GO artifacts and their transcript/graph sidecars must themselves be
reviewed, committed, and unchanged before validation can load them.

### 9.4 Fresh SQLGlot protocol and gate

The task IDs and permissions remain exactly Section 6's committed manifest.
Validation and final test use the identical source-free specification, graph
code, PMFs, Current evidence, coverage rule, command rows, labels, and fallback.
No reserved task updates any arm. The label-free coverage gate remains at least
20 relational commands in five tasks, with each frozen primary state appearing
in five commands across three tasks. Coverage failure leaves that partition's
labels unscored.

Validation authorizes final test only if, relative to frozen Current:

- exact accuracy is no lower for latency, CPU, RSS, or Disk;
- severe underprediction is no higher for any target and lower for at least two;
- target-level helpful changes outnumber harmful changes and helpful changes
  cover at least three validation tasks;
- relational exact accuracy is no lower than both collapsed-state and
  scope-only for every target; and
- on the frozen primary target and primary state pair, relational has strictly
  more correct predictions than each ablation.

Final test uses the identical independent gate. Overall GO requires both
validation and final GO. Validation NO-GO or coverage failure stops without
opening final. A validation GO must be independently reviewed and committed
unchanged; final authorization verifies the committed file bytes, `gate.go`,
coverage, scored-label flag, row identity, artifact hash, and split hash. If the
committed validation row sidecar does not match its recorded hash, authorization
also fails. If the
collector has no top-level completion artifact, evaluation waits: partial task
directories and attempt counts are metadata, not evidence.

### 9.5 Required checks

Focused tests cover JSON cardinality and abstention consistency; every forbidden
regex construct; capture count/name; repeat and input bounds; literal and
cross-task support; exact spans; relation overflow; all five states; scope-only
and collapsed carriers; fit-only PMFs; Current identity; split disjointness; and
validation-to-final input/hash authorization. An independent bounded review of
the protocol and implementation is required before the single generation.
Unchanged reviewed code may then produce development and fresh results; a final
bounded review checks artifacts, gates, and data access before result commit.

## 10. Family-and-scope relational arm

**Frozen before generation or reserved-outcome access: 2026-08-05.** This is a
new development arm, not a repair, retry, or reinterpretation of Sections 7 or
9. Their artifacts and NO-GO verdicts stay unchanged. Validation and final-test
commands, outputs, labels, and telemetry remain unread.

The visible label-free diagnostic motivating this amendment used a deliberately
permissive lexical oracle, not a candidate method. With exact-command verifiers
it produced 146 non-null states over 59 development tasks, including no
`closure_verified` state. Replacing only verifier equality with a generic
direct/module invocation family produced 308 states over 92 tasks, including
35 `closure_verified` commands in 33 tasks. This is an optimistic upper bound
with false-positive relations; it may justify the representation but cannot set
patterns, PMFs, thresholds, or a success verdict.

The causal question is:

> Does retaining both the current command's work scope and family-level
> blocker/remediation closure separate a short verification run from later real
> work without a prediction-time model call?

### 10.1 Label-free evidence and single agent call

The host constructs evidence only from the 100 development task IDs in the
committed split. It reads raw command text, exit status, bounded result excerpts,
and parsed clauses; it does not read durations, telemetry, resource labels,
Current predictions, reserved task IDs, or reserved files.

For evidence compression, a simple one-clause command has an invocation key:
use argv[2] when argv has at least three entries and argv[1] is exactly `-m`;
otherwise use `argv[0].rsplit("/", 1)[-1]`. Normalize that value by case-folding,
replacing every maximal run outside `[0-9a-z]` with one hyphen, and stripping
outer hyphens; an empty result is not a key. A key is a candidate when at least
five tasks contain a failed invocation followed by a later invocation with that
key. Retain at most four candidates by descending qualifying-task count, then
lexicographic key. For each candidate retain the first twelve qualifying tasks
in the committed development order. An episode starts at the first failed
candidate invocation that has a later verifier and ends at the last candidate
invocation; it contains every candidate invocation and every successful
intervening command, preserving order.

The exact string trim uses marker `\n<omitted>\n`. A string at or below its limit
is unchanged; otherwise its head has `floor((limit - marker_length) / 2)`
characters and its tail fills the remaining characters after the marker.
Commands use limit 800 and failed-verifier excerpts use limit 500. An episode
with more than 48 selected events keeps exactly the first and last 24; shorter
episodes keep every selected event. Candidate and task aliases are assigned in
the retained order as `F000...` and `T000...`; original task and sample IDs are
absent. After alias assignment, case-insensitive occurrences of a retained task
ID inside command or result text are replaced by that task's alias; occurrences
of any other development task ID are replaced by `<task>`. The serialized prompt
must not exceed 180,000 UTF-8 bytes; overflow or fewer than two candidate keys is
a pre-generation structural NO-GO.

There is exactly one tool-free Codex `gpt-5.6-sol` call at requested fast tier
and medium reasoning. There is no critic, repair, retry, or candidate sweep.
The agent receives the compressed evidence and contract and returns one strict
JSON object. Prediction-time agent cost is zero.

### 10.2 Declarative schema and bounds

The response contains exactly:

```text
abstain: boolean
family_id: string
scopes: array[{scope_id: string, patterns: array[string]}]
blocker_patterns: array[string]
remediation_patterns: array[string]
explanation: string
```

On abstention all strings and arrays except `explanation` are empty. Otherwise
IDs match `[a-z][a-z0-9_-]{0,39}`; `explanation` is 1--1,000 characters; there
are 2--4 distinct scopes, each with 1--4 command patterns of 1--256 ASCII
characters, and at most eight scope patterns in total. There are 1--6 blocker
and 1--6 remediation patterns of 1--512 ASCII characters. Fixed flags, allowed
regex grammar, repeat ambiguity, span validation, input limits, graph limits,
and 5 ms p95 runtime limit are exactly Section 9.2.

Every command pattern is full-matched and must support five development tasks.
Every blocker/remediation pattern must support five tasks and capture at least
two distinct identifiers. A command may match at most one scope. Decoded fixed
literals remain subject to opaque-ID and development-specific positional-token
checks. Opaque literals include every complete development ID, every evidence
alias, and every alphabetic run of at least four characters in the repository
owner/name portion of a development ID. Generic argument shape and count are
allowed because they define work scope. The agent cannot emit buckets, states,
relations, weights, thresholds, tool names outside its patterns, or per-query
decisions.

### 10.3 Deterministic host semantics and development gate

The union of scope patterns defines one command family. The current command
must match exactly one `scope_id`. Any earlier command matching any scope in the
same family is a verifier; this is the only change from exact-command equality.
Blocker spans come only from failed family verifiers. Remediation spans come
only from successful prior commands, and an edge still requires exact
normalized identifier equality. A successful family verifier after the last
matching remediation may establish closure. The host alone derives the five
ordered states in Section 7.2. The query result and all future events remain
invisible.

Fit development PMFs with the unchanged five-task target support for three arms
on the identical non-null full-relation carrier; commands without a relation
state fall back to byte-identical Current in every arm:

1. **family-only:** collapse scope and state to `(family_id, target)`;
2. **scope-only:** collapse state to `(family_id, scope_id, target)`; and
3. **full:** `(family_id, scope_id, dependency_state, target)`.

Before any reserved outcome access, the candidate must expose both contrasts:

- one **relation contrast**: under the same scope, two dependency states each
  occur in five tasks and have different hard PMF modes for at least one target;
- one **scope contrast**: under the same dependency state, two scopes each
  occur in five tasks and have different hard PMF modes for at least one target.

For each contrast maximize the smaller task support, then combined support,
then lexicographic `(family_id, fixed dimension, sorted varying dimension)`;
select its first differing target in latency, CPU, RSS, Disk order. Freeze both
primary contrasts. Every graph/bound check must pass, empty history must return
no state, and extraction p95 must not exceed 5 ms. Failure is a complete
development structural NO-GO with no regeneration.

### 10.4 Fresh evaluation and decision gate

The split, Current evidence, fit corpus, no-update rule, task accounting, and
validation-to-final authorization remain Section 6. Validation coverage requires
20 full-arm commands in five tasks; each primary relation state under its fixed
scope and each primary scope under its fixed dependency state must appear in
five commands across three tasks. Coverage is checked without labels; failure
leaves labels unscored.

Validation authorizes final only when full, relative to Current, has no lower
exact accuracy for any target, no higher severe underprediction for any target
and lower severe underprediction for at least two, more helpful than harmful
target changes covering three tasks, no lower accuracy than family-only or
scope-only for any target, and both of these paired requirements:

- full is strictly more correct than scope-only on the frozen relation contrast
  and its primary target; and
- scope-only is strictly more correct than family-only on the frozen scope
  contrast and its primary target.

Final uses the identical independent gate; overall GO requires both partitions
to pass.

The prompt, schema, response, spec, compressed evidence, PMFs, primary contrast,
split and input hashes, committed host revision, and generation cost are frozen
in one development artifact. Protocol and implementation receive bounded
independent review before generation. A structural GO artifact is reviewed and
committed before any validation access; a validation GO is reviewed and
committed before final access.

The single allowed call completed in 45.002 seconds using 32,742 input and
1,812 output tokens (zero cached input tokens). It returned a `pytest` family
with targeted and suite scopes plus missing-module/install relations, but both
scope patterns used the Python-incompatible `\z` anchor. The committed Python
3.12 host rejected the response with `scope regex does not compile` before
support checks, graph replay, PMF fitting, or contrast selection. This is the
frozen structural NO-GO for this candidate: no repair or retry is allowed, and
validation/final remain unread. The independently checked artifact is
`analysis/results/tool-resource-5-3-3-3-20260804/sqlglot100-declarative-family-state-v1`.
