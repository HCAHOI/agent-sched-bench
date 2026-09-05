# Paper baselines

These adapters pin either authors' code or the smallest paper-derived subset
needed by the physical replay. They do not make every method a full-system
reproduction.

| Cell name | Adapter | Fidelity boundary | Preparation |
|---|---|---|---|
| `fcfs` | stock repo vLLM | Physical control with prefix caching; no paper policy | `bash scripts/setup/benchmark_server.sh --gpu` |
| `thunderagent` | `thunderagent_official.sh` | Official program-aware proxy; it does not manage tool containers | `./scripts/baselines/thunderagent_official.sh install` |
| `agentix` | `agentix_reproduction.sh` | PLAS arrival-priority subset; no ATLAS, dynamic quantum, KV-swap kernel, or multi-engine routing | `./scripts/baselines/agentix_reproduction.sh install` |
| `continuum-public` | `continuum_public.sh` | Authors' public fixed-2-second TTL fork, not the paper estimator | `./scripts/baselines/continuum_public.sh install` |
| `continuum-reproduction` | `continuum_reproduction.sh` | Public fork plus the paper TTL equation; requires a hardware-measured prefill/reload profile | `./scripts/baselines/continuum_reproduction.sh install` |
| `native-priority` | stock repo vLLM priority scheduler | Priority 1 for each program's initial request, priority 0 for causal tool returns | `bash scripts/setup/benchmark_server.sh --gpu` |
| `native-priority-aging` | `native_priority_aging.sh` | Same return priority with promotion after one `max_num_seqs=8` batch of bypasses | `./scripts/baselines/native_priority_aging.sh install` |
| `cachewise-disabled` | `cachewise_reproduction.sh serve-disabled` | Exact CacheWise fork/config control without CacheWise scheduling or policy payloads | `./scripts/baselines/cachewise_reproduction.sh install` |
| `cachewise` | `cachewise_reproduction.sh` | Authors' vLLM fork plus reconstructed scheduler hooks; the exact paper split and engine attachment hook are unpublished | Run `cachewise_official.sh install`, `cachewise_official.sh train-published`, then `cachewise_reproduction.sh install` |
| `saga` | `saga_reproduction.sh` | Single-GPU KV/arrival-priority subset; no private multi-GPU routing, migration, or prefetch system | `./scripts/baselines/saga_reproduction.sh install` and build a causal profile |

`cachewise_official.sh` is only the authors' released duration predictor. It is
not a serving baseline. `continuum-public` and `continuum-reproduction` remain
separate because only the latter adds the estimator described in the paper.
Murakkab remains related work only: its declared-DAG/MILP input contract does
not match black-box trace replay, so it has no active adapter here.

## Physical runner

The single entry point is `scripts/evaluation/run_paper_baseline.sh`. It accepts
any compatible replay manifest and runs cells sequentially on one A100 80 GB.
Every output root must be new.

```bash
export MODEL=NousResearch/Meta-Llama-3.1-8B-Instruct
export MANIFEST=/absolute/path/to/replay-manifest.yaml
export RUN_ROOT=/absolute/path/to/new-baseline-run
export CELLS='fcfs-r1 thunderagent-r1'
export CONCURRENCY=4

./scripts/evaluation/run_paper_baseline.sh --preflight
./scripts/evaluation/run_paper_baseline.sh --run
```

Add `agentix-r1`, `continuum-public-r1`, `continuum-reproduction-r1`,
`native-priority-r1`, `native-priority-aging-r1`, `cachewise-disabled-r1`,
`cachewise-r1`, or `saga-r1` only after its preparation above. Continuum
reproduction additionally needs `CONTINUUM_REPRODUCTION_PROFILE`; SAGA needs
`SAGA_PROFILE`. Use `TRACE_TOOL_REPLAY=0` for real task-container tool execution
or `1` only for an explicitly declared trace-timed replay.

Result receipts:

- [PennyLane low-pressure suite](../../analysis/results/pennylane-paper-baseline-suite-physical-v1.md)
- [Unique-128 high-pressure suite](../../analysis/results/mixed128-poisson-unique-baselines-20260827/result.md)

The dated one-off PennyLane suite driver was removed after its final state and
result receipt were retained in Git (`3b2ea8d` and `454bd13`). Recover it from
those revisions only to inspect the historical protocol, not as a current
entry point.
