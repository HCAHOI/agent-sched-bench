# Live serving development

`spike/` contains the isolated vLLM serving implementation. It does not add GPU
or vLLM dependencies to the base trace-collection package.

## Layout

- `vllm_connector/` — selective KV transfer plus pause/evict/resume integration
  for vLLM `>=0.11,<0.12`.
- `run_spike.py` — focused transfer, pause/resume, and trigger-integration
  diagnostics.
- `multitenant.py` — workload loading, policy planning, accounting, and report
  helpers for the multi-tenant harness.
- `run_multitenant.py` — one workload/policy/load execution from the W5 config.
- [`P4_EVICTION_DESIGN.md`](P4_EVICTION_DESIGN.md) — implementation contract and
  known connector/scheduler limits.

The custom connector uses vLLM's scheduler and worker connector seams. Copying
KV alone does not free blocks; the pause path saves KV, waits for worker
confirmation, preempts to release blocks, then reallocates and restores before
resuming.

## Setup

```bash
bash scripts/setup/benchmark_server.sh
source .venv/bin/activate
uv pip install -e '.[serving-spike]'
```

## Transfer defaults

The current transfer mode is `staged`. It gathers paged blocks into a contiguous
device buffer and copies through pinned host memory on a dedicated CUDA stream.

There is one staged-buffer capacity default: **128 blocks**.

- `run_spike.py`: `--staging-max-blocks 128`
- W5 config: `serving.transfer_max_blocks: 128`

The buffer is allocated outside vLLM's `gpu_memory_utilization` reservation and
is approximately 2 MB per block for the reference layout. A transfer wider than
the configured capacity fails fast. Raise the value only with measured VRAM
headroom. `strided` remains an explicit diagnostic baseline, not the default.

`run_spike.py` defaults `--block-dim 0`; the W5 Llama-3.1-8B config explicitly
uses `block_dim: 1`. Verify the real cache layout on the target vLLM build before
trusting byte counts.

## Focused GPU diagnostics

Transfer-only diagnostic; blocks are copied but not freed:

```bash
PYTHONPATH=src:. python spike/run_spike.py \
  --scenario offload --model meta-llama/Llama-3.1-8B \
  --num-load 8 --output offload_note.json
```

Pause/evict/resume diagnostic:

```bash
PYTHONPATH=src:. python spike/run_spike.py \
  --scenario pause --model meta-llama/Llama-3.1-8B \
  --num-load 8 --block-dim 1 --output pause_note.json
```

The pause report includes `pause_to_freed_ms`, `resume_to_first_token_ms`,
`blocks_freed`, and a greedy token-identity check against an uninterrupted run.
The identity claim is limited to the supported deterministic decoding path.

The `certified` scenario is an integration diagnostic over real recorded tool
durations. Its accounting is not a new certificate and its static trigger table
only approximates the full certified union.

## W5 multi-tenant harness

The single matrix definition is
[`../configs/serving/w5_multitenant.yaml`](../configs/serving/w5_multitenant.yaml).
Task selections are self-owned under
`../analysis/serving/w5-multitenant/inputs/`.

One cell is invoked as:

```bash
PYTHONPATH=src:. python spike/run_multitenant.py \
  --config configs/serving/w5_multitenant.yaml \
  --workload swe-rebench-100-development \
  --policy deadline --load 2 \
  --output w5-cell.json
```

W5 is development work and has no result. Do not launch it until both blockers
are resolved:

1. `analysis/serving/w5-multitenant/prefill_result_llama31_8b.json` is missing
   and must be measured on the target hardware.
2. The development trigger table encodes `rho=1.0`, while runtime accounting is
   configured for the measured `rho=0.94`. Regenerate the table or approve an
   explicit policy contract; do not silently relabel it.

A live smoke must also prove real memory pressure, co-tenant admission into
freed blocks, faithful resume, and complete request/policy/input provenance
before any matrix run.

## CPU verification

```bash
PYTHONPATH=src:. python -m pytest \
  tests/test_vllm_connector_spike_logic.py \
  tests/test_certified_trigger_integration.py \
  tests/test_multitenant.py -q
```

These tests cover control and accounting logic without claiming to validate CUDA
copy behavior or live vLLM scheduling.
