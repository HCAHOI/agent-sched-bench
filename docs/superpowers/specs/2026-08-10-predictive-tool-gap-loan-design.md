# Predictive Tool-Gap Loan

## Question

Can the existing latency predictor improve agent-task completion by advancing
the same backfill action that causal elapsed-time feedback can take later,
without increasing p99 end-to-end LLM TTFT by more than 5%?

This is a new development experiment. It does not reinterpret or tune the
failed static cap-eight or phase-aware request-cap experiments. The SQLGlot
first eight are development-exposed; the previously reserved 16 SQLGlot task
IDs remain untouched unless this protocol passes unchanged.

## Action

Four prepared tasks begin as foreground tasks. Four more task containers are
prepared but do not execute. Each foreground task owns one non-recursive loan:
it may start at most one waiting task, and a task started by a loan may not
lend again. The maximum active-task count is therefore eight.

The action never gates, cancels, or delays an LLM HTTP request. Once a task is
active, its requests go directly to the unchanged vLLM server. A returning
foreground task is not preempted or queued by host admission logic.

After every foreground task has completed at least one LLM request, define the
causal request budget `B` at each tool start as the maximum full HTTP response
latency among foreground LLM requests completed before that tool start. If `B`
is not yet available, that tool cannot create a loan. `B` is frozen for that
tool call.

The selected Task-Aware Command Predictor is queried at the existing
`BeginCall` boundary. It uses only the command, parsed clauses, repository,
and evidence causally available before the command starts. Let `L` be the
lower edge of the hard latency bucket selected from its five-bucket PMF. An
unavailable prediction has no `L`.

Three arms execute the same trajectories:

1. **Fixed-4:** never lend.
2. **Feedback-loan:** lend when the running tool's elapsed time first exceeds
   its frozen `B`.
3. **Predictor+feedback-loan:** lend immediately when `L > B`; otherwise use
   the identical feedback rule.

There is no probability threshold, latency-bucket special case, recursive
loan, request semaphore, or post-start prediction repair. CPU, RSS, and Disk
predictions do not participate; adding them would mix gap discovery with the
separately failed hard-resource-admission question.

Unavailable prediction, parsing failure, or missing causal evidence falls
back to Feedback-loan. A missing waiting task is a no-op. Every decision logs
the arm, lender and borrower IDs, command/action ID, prediction provenance and
PMF, hard bucket and `L`, frozen `B`, decision time, feedback threshold-crossing
time, actual tool end, and whether the loan was early or feedback-triggered.

## Implementation Boundary

Reuse the staged replay queue, shadow-generation metrics, selected predictor,
and existing tool-start/tool-end lifecycle. Add only a parent-owned loan
controller shared by workers. The controller needs four operations:

- register the four foreground lenders and four waiting borrowers;
- record completed foreground LLM response latency;
- evaluate a tool-start prediction and arm-specific loan rule;
- receive elapsed/tool-end events and emit one auditable borrower release.

No new service, scheduler framework, predictor architecture, or vLLM change is
part of this experiment. The existing phase-aware request-slot implementation
remains available but is disabled in every arm.

## Development Protocol

Use the same SQLGlot first-eight source trajectories, manifest order,
two-core containers, eBPF, Llama-3.1-8B-Instruct A100-80GB server at 250 W,
warm-up, and fresh vLLM/KV state per cell as Section 5.6. Run the symmetric
six-cell order:

```text
Fixed, Feedback, Predictor, Predictor, Feedback, Fixed
```

Pair cells 1/2/3 and 6/5/4. Before the formal run, one bounded smoke may check
plumbing and decision logs but cannot contribute performance evidence. After
the smoke, report measured wall time and memory; the already approved formal
run is expected to take about 65 minutes.

Action activation is valid only if both Predictor cells contain at least one
early loan and at least four distinct command/action IDs create early loans
across the two cells. Failure is `insufficient_action_activation`, not a
performance NO-GO. It does not authorize changing the rule.

All six cells must preserve exact source action IDs, types, tool names,
tool-call IDs, arguments, order, requested and returned completion-token
counts, two-core container caps, telemetry validity, and absence of framework
error, OOM, or thermal slowdown. Completed-LLM and predictor evidence used by
a loan must strictly precede its decision; the feedback threshold uses only
monotonic elapsed time available at that decision. No request may contain host
admission wait.

## Frozen Decision Gate

Use common-ready task JCT and linear-interpolation percentiles. The development
result is GO only if every condition holds:

- both Predictor repetitions have lower common-ready makespan than their
  paired Feedback and Fixed cells;
- pooled mean task JCT for Predictor is at least 5% lower than both Feedback
  and Fixed;
- paired Predictor/Fixed p95 task-JCT ratios are at most 1.05;
- paired Predictor/Fixed p99 end-to-end TTFT ratios are at most 1.05;
- action activation and every validity condition pass.

Report absolute per-cell makespan, mean/p95 task JCT, server and end-to-end
TTFT, full LLM response latency, loan counts and timing advance, prediction
bucket/provenance, per-task changes, and all validity counts. Any failed gate
stops without changing thresholds, bucket handling, task order, task subset,
or vLLM settings. Only an unchanged development GO permits a separately
costed fresh confirmation.

## Interpretation

A GO supports the narrow claim that static latency prediction advances a
causal backfill action beyond elapsed-time feedback and improves task
completion without exporting cost to LLM tail latency. It does not establish
CPU/RSS/Disk action value or production integration.

A valid NO-GO separates two causes: insufficient action activation means the
workload lacks predicted gaps under the frozen rule; activated but harmful or
ineffective execution means prediction timing does not create useful action
value under this scheduler. Predictor errors and physical contention must be
reported separately from the gate verdict.
