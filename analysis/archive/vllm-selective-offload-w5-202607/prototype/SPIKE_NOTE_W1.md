# W1 spike note — vLLM selective KV offload seam (2026-07-18)

Box: H100 PCIe 80GB, **PCIe Gen4 x16** (Ice Lake host, 2×8352Y, 20c/125GB;
Gen4 is the campaign's operating platform per user decision 2026-07-18).
vllm==0.11.2 (exact pin; 0.11.0 needs nvcc at runtime, 0.11.2 changed the
connector ctor ABI — both handled), torch 2.9.0+cu128, driver 580.126.09.
Llama-3.1-8B-Instruct, 8 co-running load requests, 3 offload + 3 control
repetitions. Raw data: `spike_note_gen4_20260718.json`.

## Seam verdict: WORKS (all three open questions answered)

1. `start_load_kv` DOES fire for the target request mid-generation — the
   worker executed our offload directive under load.
2. Scheduler→worker metadata delivery (SelectiveOffloadMeta) works.
3. KV layout on this stack: **blocks are dim 1** (dim 0 = K/V pair, size 2).
   The reviewer-mandated fail-fast guard caught the dim-0 assumption on
   first contact (exact ValueError, no garbage numbers); `--block-dim 1`
   fixed it with zero code change.

## Measured numbers (97 blocks, 203.4 MB per event)

| metric | median | p99 |
|---|---|---|
| offload | 72.0 ms (2.83 GB/s) | 169.6 ms |
| restore | 65.9 ms (3.09 GB/s) | 69.6 ms |

- restore/offload ratio ≈ 0.92 — consistent with the measured rho=0.94;
  the ratio survives platform and implementation changes, as predicted.
- **Bandwidth is ~8x below the Gen4 pinned-DMA ceiling (~25 GB/s).** Cause:
  per-layer strided copies through non-contiguous movedim views, not one
  contiguous pinned staging transfer. Platform (Gen4 vs Gen5) explains 2x
  vs the rho box's 54-57 GB/s; implementation explains the rest. This is
  P4 engineering item #1: contiguous pinned staging buffer + batched
  async copies.

## Interference on 8 co-tenants (6120 ITL samples each arm)

| | mean | p50 | p99 | max |
|---|---|---|---|---|
| with offload | 15.88 ms | 15.23 | 20.72 | **189.1** |
| without | 14.87 ms | 14.47 | 18.87 | 27.2 |

Steady-state cost is modest (+1.0 ms mean, +1.9 ms p99) but the offload
moment stalls a few tokens hard (max 189 ms vs 27 baseline). P4 item #2:
move the copy off the forward-pass critical path (async stream + rate
limit). Selective, trigger-driven offload does NOT collapse co-tenant
latency — consistent with our position vs ThunderAgent Fig 7a, pending the
full eval.

## What the spike does NOT prove (unchanged)

Offload here COPIES KV; it does not evict blocks (no memory freed) and does
not pause/resume the request. P4 item #3 (the big one): scheduler-side
eviction + pause/resume + trigger wiring. vLLM 0.11 has no public API for
this; it is net-new work on the fork.

## W2 go/descope input

Transfer path proven and timed; failure modes were all environment/config,
fixed in hours, committed. Recommend **GO** for connector-based P4 with the
three items above as the scope; the descope fallback (single-tenant
demonstration) remains available if item #3 balloons by end of W2.
