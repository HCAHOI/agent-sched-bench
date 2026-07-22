# Configuration map

| Directory | Role |
|---|---|
| `benchmarks/` | Benchmark plugins loaded as `configs/benchmarks/<slug>.yaml`. |
| `prompts/` | Benchmark-specific prompt templates. |
| `mcp/` | MCP server definitions passed through the trace-collection CLI. |
| `simulate/` | Curated replay and static-export configuration. |
| `trace_collect/` | Trace-collection configuration and compatibility examples. |
| `experiments/` | Development replay configuration; completed-task profile update is the retained adaptive lane. |
| `serving/` | Live-system development configuration. |

## W5 multi-tenant configuration

`serving/w5_multitenant.yaml` is the single current W5 matrix definition. It
owns policies, load levels, workload/task-list inputs, model settings, transfer
settings, and the runtime restore-cost fraction. Input task lists live under
`analysis/serving/w5-multitenant/inputs/`.

The file is not launch-ready and has no associated result. Before a GPU run:

- generate
  `analysis/serving/w5-multitenant/prefill_result_llama31_8b.json` on the target
  hardware;
- resolve the development trigger table's `rho=1.0` metadata against the
  configured runtime `restore_cost_fraction: 0.94`;
- pass the focused CPU tests and independent serving-code review.

Do not infer a result from the presence of the config, and do not silently
substitute another prefill profile, trigger table, task set, or `rho` value.
