# Task-Local Full-Test Phase Research Plan

**Effective:** 2026-08-04

**Status:** development replay passed; held-out SQLGlot collection running and unread

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

## 3. Held-out replication

Development passed and the implementation is frozen. After collection
completion, fit state on all 100 existing SQLGlot tasks and evaluate every
evidence-valid task from the unread collection:

`traces/swe-rebench/gpt-5.6-sol/sqlglot-prev100-c2-fast-requested-ebpf-20260804`

Current and the candidate receive the same original 100 tasks as prior evidence.
The candidate's phase PMFs and Current's repository KB are frozen for the primary
replication; within-new-collection updates are secondary only. Report all failed
or telemetry-invalid collection tasks rather than silently excluding them.

The held-out gate is fixed before any result from that collection is read:

- Disk predictions are bit-identical;
- latency, CPU, and RSS exact accuracy are each no lower than Current;
- severe underprediction is lower for at least two of those targets and no
  higher for any;
- changed predictions are net helpful; and
- helpful changes cover at least three tasks.

This is a task-held-out same-repository replication, not a temporal deployment
claim: the new tasks precede the development tasks by creation time.

## 4. Implementation and checks

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
