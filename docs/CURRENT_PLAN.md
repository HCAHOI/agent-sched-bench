# SWE-rebench Claude Parity Planning

## Goal

Plan a complete implementation to:

1. make SWE-rebench default to the `cc_aligned` prompt
2. move OpenClaw / MiniSWE agent bodies into the task container so their
   runtime location matches Claude Code for motivation experiments

## Steps

1. Branch + context anchor
   Status: completed
   Scope:
   - Create branch `feat/swe-rebench-cc-parity`
   - Save context snapshot under `.omx/context/`

2. Grounding audit
   Status: completed
   Scope:
   - Inspect local SWE-rebench benchmark, collector, OpenClaw, MiniSWE runtime paths.
   - Inspect `../agentcgroup` Claude Code single-task runner for execution semantics.

3. Consensus plan draft
   Status: completed
   Scope:
   - Planner drafted `.omx/plans/swe-rebench-cc-alignment.md`.

4. Architect review
   Status: completed
   Scope:
   - Architect flagged that prompt default and runtime strategy must be separated,
     canonical artifact writing should stay host-owned, and container bootstrap
     must be proven rather than assumed.

5. Critic review + revision
   Status: completed
   Scope:
   - Revise the plan accordingly, then re-run review.


## Implementation progress

- Phase 1 foundation landed on branch `feat/swe-rebench-cc-parity`.
- Done: benchmark-owned prompt default plumbing (`cc_aligned` for swe-rebench), plugin runtime mode hook, host-owned manifest/results runtime metadata, task-container launcher/entrypoint skeleton, OpenClaw `tool_workspace` support, MiniSWE local-environment mode, and focused unit coverage.
- Verified locally: targeted pytest suites covering prompt resolution, runtime dispatch, task-container host helpers, collector task-container helpers, attempt metadata, openclaw runtime changes, and local patch extraction all pass; targeted ruff passes too.
- Real smoke evidence: pulled `swerebench/sweb.eval.x86_64.kinto_1776_kinto-http.py-384` and attempted task-container bootstrap. On this current macOS+Podman host, container start still fails before preflight with an amd64/arm64 OCI runtime mismatch. The earlier Python-runtime mount issue is gone after switching the in-container launcher to the repo `.venv` interpreter path.
- Pending next: continue implementation and treat the remaining real-smoke blocker as macOS/Podman host-specific, not as a Linux experiment-path blocker.
