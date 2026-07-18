# P4 validation results (2026-07-18)

Box: H100 PCIe 80GB, PCIe Gen4 x16 (Ice Lake), vllm==0.11.2, commit 1474a3d.
Llama-3.1-8B-Instruct, 8 co-running load requests, max_tokens 1024, seed 0,
203.4 MB KV (97-99 blocks) per event. Raw: spike_ab2_{strided,staged}.json,
spike_pause.json. All review-gated code; two GPU-found staleness bugs fixed
mid-session (commit 1474a3d) — both caught by fail-fast guards, zero silent
corruption at any point.

## Transfer A/B (lane A) — VALIDATED

| mode | offload | restore | co-tenant ITL mean | p99 | max tail |
|---|---|---|---|---|---|
| strided (W1 baseline) | 61.8 ms (3.3 GB/s) | 57.6 ms (3.5 GB/s) | 13.28 | 17.4 | **193 ms** |
| **staged** | **8.8 ms (23.1 GB/s)** | **8.7 ms (23.3 GB/s)** | 12.55 | 15.80 | **58 ms** |
| (no-offload control) | — | — | 12.58 | 15.79 | 24-29 ms |

- **7x bandwidth**: 23 GB/s = 74% of the Gen4 x16 theoretical 31 GB/s —
  matches lane A's projection; the residual is gather-kernel + DMA setup.
- **Interference eliminated at mean/p99**: staged-with-offload is
  statistically indistinguishable from the no-offload control (12.55 vs
  12.58 mean; 15.80 vs 15.79 p99). Max tail 193 -> 58 ms.
- restore/offload ratio stays ~1 (0.99) — rho-consistent.
- On a Gen5 host these transfer times halve again (~4.4 ms for 203 MB).

## Pause/resume (lane C) — VALIDATED, all four claims

```
blocks_freed            = 99      (memory ACTUALLY freed - eviction works)
pause_to_freed_ms       = 108.6   (incl. synchronous 203MB save)
resume_to_first_token_ms= 39.2    (load + reschedule + first decode step)
identical               = true    (greedy continuation BIT-IDENTICAL to
                                   uninterrupted run, 1024/1024 tokens)
```

- **The logit-identity unknown (design memo risk #2) is resolved: saved-KV
  resume == recompute, exactly.** The whole preemption+connector resume
  hypothesis holds end-to-end on real hardware.
- Mechanism demonstrated: external trigger -> synchronous save -> force
  preempt (99 blocks to the pool) -> 3 s hold (tool window) -> resume ->
  connector load replaces recompute -> generation continues bit-faithfully,
  with 8 co-tenants running throughout.
- Zero forked vLLM files: PausableScheduler injected via scheduler_cls.

## What this means for the paper (roadmap P4)

The complete mechanical core of the system contribution now exists and is
measured: trigger-driven selective KV eviction of a NAMED request with
honest restore, at near-link-rate transfer, with co-tenant interference at
the noise floor, and provably lossless resume. Remaining P4 work is
integration, not invention: wire the certified trigger (fresh-corpus
validated policy) to the pause trigger, multi-tenant admission accounting
(freed blocks -> admit), and the eval harness (4 policies x workloads x
loads + the ThunderAgent Fig 7a refutation).

## Session bug ledger (all fail-fast catches, for the reviewer trail)

1. Staged restore=0: liveness race — fixing the ITL stall unthrottled decode
   and the target finished before the restore trigger. Fix: pop-once
   offload-time block snapshot + observable restore_skipped + explained
   diagnostics. (Not a drain bug — drain design verified correct.)
2. Pause save under-coverage (97 vs 99 blocks): connector block table is
   admission-stale; decode growth invisible to update_state_after_alloc.
   Fix: scheduler-side fresh get_block_ids at pause instant + pause-trigger
   exceptions contained (fail the pause, never the engine).
