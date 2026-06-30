# Configuration Directory Map

`dev/cpu-only` keeps only configuration used by cloud-provider trace collection,
cloud timing replay, prompts, MCP, and benchmark plugins.

## Directory inventory

| Directory | Lifecycle | Current role |
| --- | --- | --- |
| `benchmarks/` | `active-runtime` | Benchmark plugin YAMLs loaded as `configs/benchmarks/<slug>.yaml` by `trace_collect.cli`. |
| `prompts/` | `active-runtime` | Prompt templates resolved as `configs/prompts/<benchmark_slug>/<template>.md`. |
| `mcp/` | `active-runtime` | MCP server YAMLs passed through `--mcp-config`, e.g. OpenClaw `context7.yaml`. |
| `simulate/` | `curated-replay` | Replay manifests for simulation/static export flows. |
| `trace_collect/` | `legacy-compatibility` | Older trace collection/simulation configs retained for compatibility checks. |

Removed from this branch: local-HF recording/KV-policy/sparse-attention configs
and GPU recording campaign manifests.
