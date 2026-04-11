# Cloud-Model Trace Replay

## Goal
Add a `cloud_model` replay mode to `python -m trace_collect.cli simulate`
that replays canonical OpenClaw traces against local tool execution while
treating LLM phases as scaled waiting time. The mode must support both a
single trace and multiple concurrent traces in one output JSONL.

## Non-goals
- Do not change `collect` behavior or trace schema.
- Do not attach vLLM or scheduler metrics in `cloud_model` mode.
- Do not add synthetic CPU busy-wait for LLM thinking.
- Do not broaden support beyond the repo-backed OpenClaw traces already
  accepted by `simulate`.

## Execution Plan
1. Update `trace_collect.cli` argument parsing:
   add `--mode {local_model,cloud_model}`, `--trace-manifest`, and
   `--replay-speed`; make `--source-trace` and `--trace-manifest` mutually
   exclusive.
2. Change `_run_simulate()` so `local_model` still resolves provider/model
   config, while `cloud_model` bypasses LLM config resolution and rejects
   `--metrics-url`.
3. Refactor `trace_collect.simulator` into shared trace/session loading plus
   two execution paths:
   `run_local_model_simulation(...)` and `run_cloud_model_replay(...)`.
4. Implement multi-trace replay scheduling inside the simulator:
   prepare one workspace per trace, start all sessions from a common replay
   zero, schedule action start times from source `ts_start / replay_speed`,
   and keep canonical action/summary emission in one output JSONL.
5. Preserve provenance additively in emitted action data:
   `replay_mode`, `replay_speed`, `source_llm_latency_ms`,
   `source_duration_ms`, and tool replay source markers.
6. Add focused tests for CLI parsing/dispatch and simulator behavior,
   including single-trace scaling, multi-trace interleaving, and the
   guarantee that `cloud_model` never creates or calls an LLM client.
7. Run targeted verification, then a fresh independent review sub-agent
   before finalizing.

## Checkpoints
- After CLI edits, confirm `cloud_model` no longer requires provider/model
  arguments and `local_model` still does.
- After simulator refactor, confirm `local_model` tests still pass unchanged.
- Before finalizing, run an independent review focused on correctness and
  research-integrity regressions in the replay path.
