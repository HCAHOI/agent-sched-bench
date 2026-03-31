# agent-sched-bench Design

## Summary

This repository implements a benchmark environment for comparing serving systems
on multi-step agent workloads. The implementation is structured as explicit
checkpoints so environment setup, agent logic, harness code, and analysis
components can be reviewed independently.

## Defaults

- Repository root: `/Users/chiyuh/Workspace/agent-sched-bench`
- Search backend: DuckDuckGo by default
- Config system: Hydra/OmegaConf
- Default workload configs point at full datasets; smoke subsets use dedicated
  `*_smoke.yaml` files
- Review gate: fresh-context reviewer before each checkpoint commit
- Long-running operations: scripted, but only executed after checkpoint approval

## Immediate Build Order

1. Bootstrap repository layout, tooling, and docs.
2. Add environment scripts and sync workflow.
3. Implement agent interfaces and concrete workloads.
4. Implement concurrent harness, metrics, and trace logging.
5. Add analysis and replay fallback paths.
