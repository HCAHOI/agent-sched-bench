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
- `AGENT-1`: completed (code + static verification)
- `AGENT-2`: completed (code + static verification)
- `AGENT-3`: completed (code + static verification)
- `AGENT-4`: completed (code + static verification)
- `HARNESS-1`: completed (code + static verification)
- `HARNESS-2`: completed (code + static verification)
- `HARNESS-3`: completed (code + static verification)
- `ANALYSIS-1`: completed (code + static verification)
- `REPLAY-1`: completed (code + static verification)

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
- Implemented `AGENT-1` assets: typed `StepRecord`, shared `AgentBase`, unified
  LLM call contract, trace export, and summary aggregation.
- Implemented `AGENT-2` assets: the SWE-style coding agent, shared tool-call
  parser, temp-repo sandbox path, patch application, and code-agent smoke tests.
- Implemented `AGENT-3` assets: the read-only NL2SQL data agent, real sqlite
  tools, denotation-based final evaluation, and data-agent smoke tests.
- Implemented `AGENT-4` assets: the research agent, real DuckDuckGo-style
  search/page tools, unit tests, and a live-network smoke path.
- Implemented `HARNESS-1` assets: the concurrent runner, CLI entry, signal-stop
  behavior, timeout/result semantics, and `N=2` harness tests.
- Implemented `HARNESS-2` assets: vLLM metrics polling, fail-closed snapshot
  validation, GPU util CSV parsing, and metrics collector tests.
- Implemented `HARNESS-3` assets: JSONL trace logging, run_id generation,
  duplicate-run protection, and trace logger tests.
- Implemented `ANALYSIS-1` assets: JSONL parsing, basic throughput/JCT
  summaries, simple plot generation, and heuristic inefficiency detection.
- Implemented `REPLAY-1` assets: trace replay sequencing, inter-program timing
  offsets, tool-gap waiting, and replay tests.

## Next Checkpoint

- Remaining planned checkpoints are complete.

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

## AGENT-1 Caveat

- The `AGENT-1` code checkpoint is complete and reviewed, but `_call_llm()`
  still needs a live backend integration run for wire-level confirmation.

## AGENT-2 Caveat

- The `AGENT-2` code checkpoint is complete and reviewed, but full end-to-end
  evaluation against real SWE-bench tasks and a live model backend still remains
  a runtime acceptance item.

## AGENT-3 Caveat

- The `AGENT-3` code checkpoint is complete and reviewed, but full runtime
  fidelity still requires real BIRD databases and the intended live model
  backend.

## AGENT-4 Caveat

- The `AGENT-4` code checkpoint is complete and reviewed, but live-network smoke
  still depends on external DuckDuckGo/web availability and a live model
  backend for the LLM loop.

## HARNESS-1 Caveat

- The `HARNESS-1` code checkpoint is complete and reviewed, but one real sweep
  against the live backend/agents is still needed to validate end-to-end stop
  and timeout behavior under real load.

## HARNESS-2 Caveat

- The `HARNESS-2` code checkpoint is complete and reviewed, but final acceptance
  still requires a live vLLM `/metrics` endpoint and real `nvidia-smi` sampling
  during a benchmark run.

## HARNESS-3 Caveat

- The `HARNESS-3` code checkpoint is complete and reviewed, but it still needs
  one end-to-end benchmark run to validate logger integration with real trace
  emission frequency and file lifecycle.

## ANALYSIS-1 Caveat

- The `ANALYSIS-1` code checkpoint is complete and reviewed, but full confidence
  still depends on running it over real benchmark traces rather than only unit
  fixtures.

## REPLAY-1 Caveat

- The `REPLAY-1` code checkpoint is complete and reviewed, but final acceptance
  still requires replaying against a live OpenAI-compatible serving stack under
  real timing jitter and scheduler behavior.
