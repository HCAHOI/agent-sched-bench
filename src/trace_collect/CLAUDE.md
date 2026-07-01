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

`python -m trace_collect.cli simulate` performs cloud replay only. It does not
issue LLM requests; it replays source trace timing with `--replay-speed` and a
bounded queue controlled by `--concurrency`.

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
