# Current Execution Plan

Updated: 2026-03-31

## Active Goal

Implement `agent-sched-bench` according to `docs/agent_benchmark_spec.md` with
checkpointed commits, independent review, and GitHub sync.

## Checkpoint Status

- `BOOTSTRAP-0`: completed
- `ENV-1`: completed (code + static verification)
- `ENV-2`: completed (code + static verification)
- `ENV-3a`: pending
- `ENV-3b`: pending
- `ENV-3c`: pending
- `ENV-4`: pending
- `ENV-5`: pending
- `AGENT-1`: pending
- `AGENT-2`: pending
- `AGENT-3`: pending
- `AGENT-4`: pending
- `HARNESS-1`: pending
- `HARNESS-2`: pending
- `HARNESS-3`: pending
- `ANALYSIS-1`: pending
- `REPLAY-1`: pending

## Execution Rules

- Pause at every checkpoint boundary for human approval unless autopilot is
  explicitly requested.
- Before each checkpoint commit, run targeted verification and an independent
  reviewer pass.
- Do not start downloads, server launches, or benchmark runs without explicit
  approval for that checkpoint.

## Latest Completed Work

- Created the new sibling repository root at
  `/Users/chiyuh/Workspace/agent-sched-bench`.
- Migrated `AGENTS.md`, `CLAUDE.md`, and `docs/agent_benchmark_spec.md`.
- Added BOOTSTRAP-0 repository scaffold, configs, scripts, and bootstrap tests.
- Resolved reviewer findings around install idempotence, dataset defaults,
  Makefile contract alignment, and fail-closed placeholders.
- Doubled the primary workload step budgets to `40/20/30` and aligned the spec
  and tests.
- Implemented `ENV-1` assets: a real Ubuntu bootstrap script, JSON environment
  reporting, and static verification tests.
- Implemented `ENV-2` assets: real model download backends, full-load artifact
  verification, `.env` propagation after success, and static verification
  tests.

## Next Checkpoint

- `ENV-3a`: implement the raw vLLM serving launch and readiness checks.

## ENV-1 Caveat

- The `ENV-1` code checkpoint is complete and reviewed, but the real acceptance
  run still has to happen on the approved Ubuntu+A100 server because this local
  machine cannot execute the server bootstrap path faithfully.

## ENV-2 Caveat

- The `ENV-2` code checkpoint is complete and reviewed, but the real acceptance
  run still requires an approved server with model access, enough disk/RAM, and
  a successful full `AutoModelForCausalLM.from_pretrained()` load.
