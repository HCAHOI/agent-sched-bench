# Configuration Map

| Directory | Role |
|---|---|
| `benchmarks/` | Benchmark plugins loaded as `configs/benchmarks/<slug>.yaml`; owns dataset, image, selection, and prompt defaults. |
| `corpora/` | Frozen cohort and evaluation-protocol definitions. These describe intended IDs; they do not materialize or validate traces. |
| `prompts/` | Benchmark-specific collection prompts. |
| `mcp/` | MCP server definitions passed to trace collection. |
| `simulate/` | Replay manifest examples and replay documentation. |
| `trace_collect/` | Legacy collection examples; not the source of benchmark defaults. |

## Frozen Corpus Definitions

| File | Declared cohort | Materialized status in this checkout |
|---|---|---|
| `corpora/swe-100.json` | 100 SWE-ReBench task IDs | Referenced trace root has 99 task directories; do not use as a complete 100-task corpus without restoring and validating the missing task. |
| `corpora/swe-277.json` | 277 SWE-ReBench task IDs | Referenced `fresh-seed42-skip150-n200` root has 275 task directories; the directory name is historical and not the declared count. |
| `corpora/swe-sqlglot-48-gpt56-ebpf.json` | 24 fit + 24 evaluation SQLGlot tasks | Protocol/preregistration only; it does not name one materialized trace root. Locate the collection through `traces/README.md` and its result receipt. |

These are evidence-bound research definitions, not a live data catalog. Do not
edit task IDs to match whatever happens to be present. Restore missing data or
create a separately named amended corpus, and record what evidence was already
visible.

`swe-100.json` also contains a workstation-specific absolute `trace_root`;
`swe-277.json` uses a repository-relative root. Always resolve and validate the
paths on the machine that will run the evaluation. Simulation manifests may
also intentionally contain absolute paths because task sources and trace roots
are host-local.
