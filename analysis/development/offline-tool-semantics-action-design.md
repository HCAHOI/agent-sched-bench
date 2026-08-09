# Offline Tool Semantics and Action Design

**Status:** approved research design; no collection or runtime integration is
authorized by this document.

## Claim under test

A language model can compile versioned public tool documentation into bounded
command semantics once before deployment. Those semantics may improve causal
latency, CPU, RSS, and Disk prediction across repeated agent tasks in the same
repository, with zero prediction-time LM cost. A scheduling claim requires a
second result: the improved prediction must change a real action whose
counterfactual value was established independently.

The causal chain is deliberately separated:

```text
docs -> semantic representation
history + observable state -> outcome distribution
outcome distribution + action cost -> scheduler decision
```

Documentation never reveals current cache contents, dependency closure,
network behavior, resource labels, or action utility.

## Offline compiler

Each supported tool receives at most one schema-constrained LM generation from
its exact-version official synopsis, option reference, examples, and `--help`.
The LM cannot read commands, traces, task or repository identifiers, outputs,
resource labels, prediction errors, or existing generated specifications.
The frozen budget is at most 64,000 input tokens and 64 KiB of JSON per tool;
record model/version, temperature, token usage, wall time, and source snapshots.
A label-blind census selects one documented tool version per tool before
generation; other versions fail closed to Clause-KB.

The output is a small JSON `ToolSpec`. It may express only:

- invocation aliases and operations;
- argument roles: work item, selector, execution policy, output, or opaque;
- alias, unordered-collection, fixed-value-equivalence, scope-order,
  requires, and excludes relations.

It cannot contain code, regex, resource directions, thresholds, learned
weights, cache contents, or actions. Deterministic host code validates and
interprets the specification. Unknown options, plugins, versions, or invalid
specifications fall back to the unchanged Clause-KB. There is no critic,
repair generation, prompt sweep, or human patch after outcomes are visible.

## Prediction study

The fixed tools are `pip install`, `pytest`, `git`, and `make`. A label-blind
coverage check requires at least 50 eligible commands across 10 tasks per
tool. A tool that misses coverage is reported as unsupported and is not
replaced after outcomes are read.

The fixed comparisons use identical command rows and labels. Every historical
method also shares the same evidence, causal task-final updates, and physical
compound composition:

1. constant fit-set Majority reference;
2. whole-command raw prefix;
3. current Clause-KB exact/prefix/binary baseline;
4. current Task-Aware Command Predictor;
5. generic argument partial order without documentation;
6. documentation-derived semantics with the same partial-order estimator.

The two partial-order arms also share the same matcher, support, and backoff.
`Doc semantics - generic partial order` isolates documentation knowledge;
comparisons with Clause-KB and Task-Aware measure deployment value. Evaluation
uses the canonical latency 5-class and CPU/RSS/Disk 3-class command targets.
Report per-tool results, equal-weight four-target accuracy, severe
underprediction, changed-command helpful/harmful counts, and causal cold-start
curves after 5, 10, 20, and 40 settled tasks.

Existing SQLGlot traces are development-only. Before any new outcome access,
freeze exact task IDs from metadata:

- SQLGlot: existing 200 valid tasks for development and plumbing only;
- PennyLane: 32 never-traced tasks, ordered by `created_at`, split 16 causal
  warm-up and 16 validation;
- DVC: 65 never-traced tasks, ordered by `created_at`, split 32 causal warm-up
  and 33 untouched final tasks;
- tox: optional predeclared long-load stress case only after a separate smoke
  and cost estimate; it cannot rescue a failed confirmation.

The validation gate requires the task-cluster-bootstrap 95% lower bound for the
equal-weight accuracy difference against generic partial order to exceed zero,
helpful above harmful changed predictions, no worse severe underprediction,
non-negative direction for at least three supported tools, and gains from
multiple tasks. Documentation semantics must also have non-negative
equal-weight accuracy against Task-Aware. Only that gate may authorize opening
DVC final outcomes.

## From prediction to action

Prediction accuracy alone is not actionability. For an action `a`, outcome
`Y`, and live system state `s`, a scheduler needs an independently defined
cost and chooses:

```text
argmin_a E[cost(a, Y, s) | command]
```

The first candidate is cache-affinity placement because EAR already handles
CPU and RSS resizing efficiently but cannot undo a cold placement. At command
start:

- `ToolSpec` and argv produce a semantic work key;
- the runtime, not the LM, reports each worker's compatible tool-owned cache;
- causal warm/cold measurements estimate service distributions;
- the scheduler compares queue delay plus expected service and setup cost.

Only tool-native caches with a verifiable compatibility key are eligible.
There is no command-output reuse, inferred dependency closure, or unsafe
workspace sharing. A tool such as `pytest` may support prediction without
supporting this action.

Before scheduler implementation, a physical paired oracle must cover at least
20 eligible long commands from 10 development tasks selected before the paired
outcomes exist, preserve terminal class and workspace outcome, reduce mean task
completion time by at least 5%, have a paired task-clustered 95% upper bound
below zero for completion time, and regress makespan by no more than 1%.
Failure stops cache-affinity integration; no policy tuning follows.

If the oracle passes, compare cache-oblivious placement, repository-only
affinity, raw-prefix/CacheWise-style affinity, documentation semantics, and the
oracle under the same EAR executor. The end-to-end claim requires all three
links: documentation changes predictions, changed predictions change actions,
and changed actions improve the physical outcome.

## Explicit exclusions

- Do not modify or seed EAR CPU/RSS resizing.
- Do not use static reservation exposure as a scheduler gate.
- Do not reopen the prior trace-conditioned agent, KV simulator, hard-page
  admission, CPU-share weighting, or shortest-command ordering results.
- Do not start a collection expected to exceed 30 minutes without a smoke,
  resource estimate, frozen protocol, and explicit approval.
