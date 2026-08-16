# PennyLane Multi-Target Transfer Protocol

**Status:** frozen before aggregate validation outcomes
**Purpose:** test whether the fixed SQLGlot-selected Task-Aware predictor
transfers to a second same-repository workload with longer, higher-load tools.

## Evidence boundary

- Development plumbing uses the 15 fit and 26 replay tasks in
  `pennylane-survival-action-split.json`; all are development-exposed.
- Validation fits on the 16 PennyLane `warmup` tasks in
  `offline-tool-semantics-splits.json` and scores that file's 16 `validation`
  tasks, except `PennyLaneAI__pennylane-5846`.
- `5846` is excluded by an open pre-outcome amendment: its replay status, first
  clause aggregate, and a trace excerpt were inspected while diagnosing the
  replay-only artifact format. It is not replaced.
- The remaining 15 validation-task outcomes must be evaluated once. They may
  not be used to alter the predictor, targets, fallback, split, or gate.

## Frozen comparison

- Unit: one eligible `exec` command; clauses are internal evidence only.
- Targets and primary metrics are the canonical 5-bucket latency and 3-bucket
  CPU peak, sampled RSS, and Disk I/O exact accuracies.
- Arms: fit-set Majority, Clause-KB, and the unchanged Task-Aware Command
  Predictor selected on SQLGlot.
- Public evidence is SWE100/277 with all PennyLane rows excluded.
- Same-repository evidence becomes visible only after whole-task settlement;
  the current task is never training evidence.
- Null handling, compound composition, ties, and unavailable predictions follow
  `tool-resource-canonical-objective.md` without amendment.

## Validation gate

Task-Aware transfers only if all four conditions hold on the 15 scored tasks:

1. equal-weight four-target accuracy is strictly above Clause-KB;
2. changed target predictions contain more helpful than harmful cases;
3. equal-weight severe-underprediction rate does not exceed Clause-KB;
4. helpful changes occur in at least two tasks.

Report Majority and both learned arms for all four targets, command coverage,
changed predictions, and a paired task-cluster bootstrap interval. The interval
is descriptive and does not add a fifth gate.

Passing this gate retains Task-Aware as the prediction input to the joint
GPU-tool action study. Failure keeps Clause-KB and feedback-only controls; it
does not invalidate the already measured telemetry or action headroom.
