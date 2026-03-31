# Current Execution Plan

Updated: 2026-03-31

## Active Goal

Implement `agent-sched-bench` according to `docs/agent_benchmark_spec.md` with
checkpointed commits, independent review, and GitHub sync.

## Checkpoint Status

- `BOOTSTRAP-0`: completed
- `ENV-1`: completed (code + static verification)
- `ENV-2`: completed (code + static verification)
- `ENV-3a`: completed (code + static verification)
- `ENV-3b`: completed (code + static verification)
- `ENV-3c`: completed (code + static verification)
- `ENV-4`: completed (code + static verification)
- `ENV-5`: completed (code + static verification)
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
- Implemented `ENV-3a` assets: raw vLLM launcher, readiness/metrics/chat
  verification, serving config, and static verification tests.
- Implemented `ENV-3b` assets: Continuum clone/install/start flow, pinned-ref
  enforcement, install reporting, and program_id-aware repeated validation with
  prefix-cache-hit gating.
- Implemented `ENV-3c` assets: ThunderAgent clone/install/start flow, pinned-ref
  enforcement, install reporting, and proxy API checks for program/profile
  tracking.
- Implemented `ENV-4` assets: clean pull workflow, smoke suite entry, harness
  fail-closed sweep wrapper, and rsync-based result collection workflow.
- Implemented `ENV-5` assets: preemption-sensitive vLLM configs, preemption
  launcher flags, and metrics/log observability helpers for eviction evidence.

## Next Checkpoint

- `AGENT-1`: implement the base agent interface and StepRecord contract.

## ENV-1 Caveat

- The `ENV-1` code checkpoint is complete and reviewed, but the real acceptance
  run still has to happen on the approved Ubuntu+A100 server because this local
  machine cannot execute the server bootstrap path faithfully.

## ENV-2 Caveat

- The `ENV-2` code checkpoint is complete and reviewed, but the real acceptance
  run still requires an approved server with model access, enough disk/RAM, and
  a successful full `AutoModelForCausalLM.from_pretrained()` load.

## ENV-3a Caveat

- The `ENV-3a` code checkpoint is complete and reviewed, but real acceptance
  still requires running `scripts/serve_vllm.sh` on the approved server with
  the actual model, GPU, and vLLM runtime.

## ENV-3b Caveat

- The `ENV-3b` code checkpoint is complete and reviewed, but real acceptance
  still requires running `scripts/serve_continuum.sh` on the approved server
  with the pinned `CONTINUUM_REF`, real model, and real Continuum runtime.

## ENV-3c Caveat

- The `ENV-3c` code checkpoint is complete and reviewed, but real acceptance
  still requires running `scripts/serve_thunderagent.sh` on the approved server
  with the pinned `THUNDERAGENT_REF` and real ThunderAgent runtime.

## ENV-4 Caveat

- The `ENV-4` code checkpoint is complete and reviewed, but the clean/dirty pull
  and rsync paths still need one real server-side exercise to validate the
  operational assumptions.

## ENV-5 Caveat

- The `ENV-5` code checkpoint is complete and reviewed, but runtime confirmation
  still requires a live vLLM run with scheduler instrumentation enabled and a
  workload that actually triggers preemption or eviction.
