# Prefill (KV-recompute) cost measurement — findings (2026-07-16)

The P2 reload-vs-recompute action restores a swapped-out KV cache by the cheaper
of reload (ρ·kv over PCIe) or recompute (prefill the context). The recompute
side was the campaign's one unmeasured number; the sweep used placeholder rates
{0.05..1.5 ms/tok}. This measures the real prefill curve on the deployment
model/hardware. Script: `scripts/measure_prefill_cost.py` (review-gated APPROVE).
Raw: `prefill_result.json`.

## Setup

- Model: **Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8** (exact FP8 checkpoint —
  matches the ρ=0.94 measurement; KV layout 48 layers × 4 KV heads × head_dim
  128 × float8_e4m3fn = **49152 bytes/token**).
- Hardware: **NVIDIA H100 PCIe 80GB** (same class as the ρ measurement, H100
  Gen5x16). vLLM 0.25.1 (V1), FlashAttention-3, FP8 linear on the prebuilt
  Cutlass block-scaled kernel; `enforce_eager=True`, prefix caching OFF, fresh
  random token ids per call (real prefill every request).
- Env-only adaptations (do NOT affect prefill/TTFT timing; greedy sampling):
  `VLLM_USE_FLASHINFER_SAMPLER=0` (avoids an nvcc JIT of the sampler kernel),
  CUDA_HOME left unset (keeps FP8 on the no-nvcc Cutlass path). No change to the
  measurement script.

## The curve (median prefill_ms, warmup + 10 reps)

| ctx tokens | prefill_ms | secant ms/tok |
|---|---|---|
| 8 | 78.20 | — |
| 512 | 76.80 | 0.150 |
| 1024 | 74.80 | 0.073 |
| 2048 | 80.07 | 0.039 |
| 4096 | 118.60 | 0.029 |
| 8192 | 254.81 | 0.031 |
| 16384 | 611.48 | 0.037 |
| 32768 | 1685.93 | 0.051 |

- **overhead_floor_ms = 74.80** (the ctx=8 point).
- linear_fit: slope **0.04861 ms/token**, intercept −22.52 ms, r² 0.966.

## Interpretation (this is the important part)

Prefill cost has TWO regimes and is NOT a single rate:
1. **Floor-dominated below ~2K tokens:** ~75 ms flat (ctx 8/512/1024/2048 all
   ~75–80 ms). This floor is wall-clock serving overhead (request submission,
   scheduler, eager mode, 1 decode step) — NOT recompute FLOPs.
2. **Super-linear above ~4K tokens:** c² attention takes over — 119 ms (4K) →
   1686 ms (32K). Marginal rate at long context ≈ 0.049 ms/token.

Consequences for P2:
- The linear fit's negative intercept (−22.5 ms) is a fitting artifact of the
  c²-weighted OLS; the sanity check `slope·512 + intercept = 2.4 ms` vs measured
  77 ms confirms the linear model badly under-predicts short context. So P2 must
  use the **per-context curve** (interpolated prefill_ms), not one scalar rate.
- The ~75 ms floor is a measurement/serving artifact, not pure recompute compute.
  Against the CUDA-event reload number (pure kernel time) this floor is an
  asymmetric charge on the recompute side; the honest recompute COMPUTE cost is
  ≈ prefill_ms − overhead_floor (or the marginal slope at large context). Both
  the per-point curve and the floor are recorded so the P2 wiring can reconcile.
- Whether recompute beats reload is now a MEASURED question per (kv swap cost,
  context length): reload = ρ·kv_cost (kv_cost swept 500–5000 ms); recompute =
  prefill_ms(context). At realistic agentic contexts (a few K → tens of K
  tokens) prefill is ~75 ms → ~600 ms, so recompute can undercut reload when the
  swept kv swap cost is large — but the crossover must be computed from this
  curve against the fresh corpus's actual context distribution, not asserted.

## Caveats

- `enforce_eager=True` can over-estimate prefill vs a cudagraph-captured deploy
  (bias conservative toward reload). MoE (A3B) with random token ids may shift
  expert load vs real text (second-order; activated FLOPs fixed). Wall-clock
  floor is serving overhead, reconciled via overhead_floor_ms.

## Next step ($0, review-gated)

Feed the per-context curve into the recompute sweep: extend
`tool_latency_recompute` from a single linear rate to interpolate prefill_ms(L)
from this curve (minus the serving floor), then re-run `run_recompute_restore_
sweep` at rho=0.94 against the fresh corpus's exact prompt_tokens. That converts
P2's "one unmeasured number" into a measured verdict on whether recompute helps.
