# trace_collect (`dev/cpu-only`)

This branch is cloud-provider-only. Active subcommands:

- default: collect traces with a configured cloud/OpenAI-compatible provider
- `simulate`: replay source traces under bounded concurrency using source timing
- `inspect`: inspect JSONL traces
- `gantt-serve` / `gantt-export`: viewer helpers

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
