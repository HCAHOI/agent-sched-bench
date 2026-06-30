# Trace Collection Compatibility Configs

This directory holds older trace-collection configuration files. Treat it as `legacy-compatibility` plus `candidate-for-archive`, not as the primary place for new benchmark configuration.

Current benchmark collection is configured through:

- `configs/benchmarks/<slug>.yaml`
- `configs/prompts/<benchmark_slug>/<template>.md`
- CLI flags on `python -m trace_collect.cli`

## Files

| File | Status | Notes |
| --- | --- | --- |
| `swebench.yaml` | candidate-for-archive | Looks superseded by benchmark plugin YAMLs such as `configs/benchmarks/swe-bench-verified.yaml`, but do not delete without checking historical scripts and reproducibility needs. |

## Migration guidance

Before moving or deleting anything here:

1. Search for exact path references in `README.md`, `src/`, `scripts/`, `tests/`, and demo code.
2. Confirm whether the file is still needed to reproduce older traces.
3. If replacing with `configs/benchmarks/`, update docs and provide a transition note.
