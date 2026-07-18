# trace_collect (`dev/cpu-only`)

This branch is cloud-provider-only. Active subcommands:

- default: collect traces with a configured cloud/OpenAI-compatible provider
- `simulate`: replay source traces under bounded concurrency using source timing
- `gantt-serve` / `gantt-export`: viewer helpers for trace inspection

Removed from this branch: local-HF recording, KV eviction, sparse attention,
vLLM serving/metrics, local-model simulation, and GPU profiling. Do not add
`--record-internals`, `--local-hf`, `--kv-*`, `--sparse-attn*`, `--metrics-url`,
`--gpu-*`, `--vllm-*`, or `profile-gpu` back to this branch.

## Collect contract

`python -m trace_collect.cli` requires:

- `--provider` and `--model`
- provider API key via the provider env var or `--api-key`
- `--mcp-config` for OpenClaw (`none` is the explicit no-MCP opt-out)

Benchmark-specific defaults live in `configs/benchmarks/<slug>.yaml` and the
benchmark plugin layer under `src/agents/benchmarks/`.

## Simulate contract

`uv run python -m trace_collect.cli simulate` performs cloud replay only. It
does not issue LLM requests; it replays source trace timing with
`--replay-speed` and a bounded queue controlled by `--concurrency`.

## Trace integrity

Keep canonical JSONL trace fields stable: full model responses, timing, tool
outputs, run config, benchmark metadata, and task/container runtime proofs.
`tool_exec.data.resource_timeline` is optional v1 telemetry currently emitted
for OpenClaw `exec` tool intervals. It records cgroup CPU core-seconds plus
network RX/TX byte deltas. Container replay uses it, when present for a single
`exec.command`, for an online source-equivalent resource-integrated timeout;
host/no-op replay and multi-command exec preserve it as source metadata only.
The fixed v1 timeout model treats source intervals as CPU-active at >=0.05 core
and network-active at >=1024 B/s, samples replay every 0.5s, and uses a 5-60s
stall detector plus a 24h outer protocol guard.

`tool_exec.data.segment_timeline` is optional v2 telemetry emitted only on the
**simulate container replay** path (never collect). A chained exec command
(`cd X && make && pytest`) is one tool call; segment timing re-runs the
*unmodified* command under bash with xtrace redirected to a dedicated fd
(`BASH_XTRACEFD`, set via env) and `PS4='+$EPOCHREALTIME '` set as a
**script-body assignment before `set -x`** (`bash -c "PS4='+$EPOCHREALTIME '\nset
-x\n<command>"`), not via the process environment: on real task-container bash
builds, an env-inherited `PS4` is captured once at shell startup (before
`EPOCHREALTIME` is live) and never re-expanded per xtrace line, silently
freezing every timestamp empty. A script-body `PS4` assignment gets bash's
normal per-line re-expansion. This was found live (mini manifest, real docker
replay): the marshaling chain worked but every segment_timeline was
`telemetry_absent: no_segments_traced` until `PS4` moved out of `env=`. So
per-top-level-segment start timestamps are captured without touching the
command's stdout/stderr or semantics (the 2-line preamble is untraced and
semantically inert; the command itself remains verbatim at the script tail).
Shape: `{version: 2, source, segments: [{segment_index,
command_text, t_start_ms, t_end_ms}], segment_count, raw_total_ms}`, all
relative to exec start. **Duration-validity ceiling:** per-segment durations
are meaningful ONLY for sequential operators (`&&`, `;`, newline). Pipeline
members (`a | b`) run concurrently but both emit depth-1 xtrace lines, so the
parser assigns fictitious sequential bounds (first member ~0ms, last member
the whole span); loop headers re-emit per iteration. Analyses of per-segment
durations MUST filter or flag samples whose parent command contains `|` or
loop keywords (`for`/`while`/`until`) — `command_text`/parent chain are
preserved exactly so this is detectable downstream. `raw_total_ms` is the measured exec wall time recorded
alongside the segments for cross-checking (sum of segments + gaps need not
equal it — both are kept, never forced equal). When bash is absent, or the
`--no-segment-timeline` opt-out is set, the field is `{version: 2,
telemetry_absent: true, reason}` (or absent) and replay is unaffected: replay
never fails because timing hiccupped; malformed telemetry fails fast only at
extraction (`trace_collect.tool_latency_dataset.extract_segment_latency_samples`).
On by default in simulate replay because it is invisible to replayed commands
and cheap; toggled by `simulate(segment_timeline=...)` via the
`OPENCLAW_SEGMENT_TIMELINE` container env var. Known ceiling: telemetry-on execs
run under bash while `--no-segment-timeline` execs keep `/bin/sh`.
