# Configuration Directory Map

`configs/` contains several kinds of project configuration. The subdirectories are intentionally not all equivalent: some are active runtime inputs, while others are experiment ledgers, replay manifests, or legacy compatibility files kept for reproducibility.

Do not move or delete files from this tree without first checking exact path references in `src/`, `scripts/`, `tests/`, `demo/`, and campaign TSV/YAML files. Many command examples pass these paths directly.

## Lifecycle labels

| Label | Meaning |
| --- | --- |
| `active-runtime` | Loaded by current CLI/runtime code or passed directly to active commands. Keep paths stable unless compatibility shims are added. |
| `active-ledger` | Not necessarily parsed by a runner, but records experiment arms, parameters, and reproducibility context. Preserve unless the experiment record is migrated. |
| `curated-replay` | Manifests or fixtures for replay/export demos and validation. Keep with the consumer that names them. |
| `legacy-compatibility` | Older harness/config surface that still has tests or documented entrypoints. Do not delete until callers are migrated or explicitly retired. |
| `candidate-for-archive` | Low-confidence or superseded configs that need owner confirmation before archiving. |

## Directory inventory

| Directory | Lifecycle | Current role | Before changing |
| --- | --- | --- | --- |
| `benchmarks/` | `active-runtime` | Benchmark plugin YAMLs loaded as `configs/benchmarks/<slug>.yaml` by `trace_collect.cli`. | Run benchmark config/plugin tests and update docs if slugs change. |
| `prompts/` | `active-runtime` | Prompt templates resolved as `configs/prompts/<benchmark_slug>/<template>.md` with hyphens converted to underscores. | Keep benchmark slug/template names aligned with `prompt_loader`. |
| `kv_policies/` | `active-runtime` | YAML presets for `--kv-config`, using the flat `EvictionPolicyConfig` overlay schema. | Check CLI examples, campaign files, and `tests/test_kv_eviction_config.py`. |
| `sparse_attention/` | `active-runtime` | YAML presets for `--sparse-attn-config`, including recording/observe-only experiment arms. | Check campaign TSV/YAML files and `tests/test_sparse_attn_config.py`. |
| `mcp/` | `active-runtime` | MCP server YAMLs passed through `--mcp-config`, e.g. OpenClaw `context7.yaml`. | Check OpenClaw collection and terminal-bench tests. |
| `recording_campaigns/` | `active-ledger` | Pre-registered campaign manifests and parameter ledgers. These may be documentation rather than runner input. | Preserve exact invocations and rationale when migrating. |
| `simulate/` | `curated-replay` | Replay manifests for simulation/static export flows. | Check `demo/gantt_viewer` consumers before renaming. |
| `trace_collect/` | `legacy-compatibility` / `candidate-for-archive` | Older trace collection/simulation configs referenced by docs, while current benchmark plugins live under `benchmarks/`. | Confirm current consumers before archiving; update README examples if replacing. |

## Reorganization policy

1. Prefer adding lifecycle documentation before moving files.
2. Keep active runtime paths stable unless code supports old and new paths during a transition.
3. Treat campaign manifests as research artifacts: preserve rationale, exact flags, and model/config choices.
4. Archive only after an owner confirms the config is no longer needed for reproducibility.
5. When a path changes, update tests, docs, campaign examples, and any TSV/YAML invocations in the same change.
