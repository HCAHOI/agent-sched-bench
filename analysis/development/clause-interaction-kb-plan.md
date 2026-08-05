# Task-Local Resource-State Research Plan

**Effective:** 2026-08-04

**Status:** relational closure protocol frozen before implementation; same-repo
validation/final task partitions fixed while the adjacent collection remains
scientifically unread

This plan extends `tool-resource-canonical-objective.md`. The earlier semantic
KB and early-execution routes are complete negative results. The active question
is narrower:

> Can command history inside the current task distinguish a full test suite that
> will really execute from the same command failing quickly during setup?

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
