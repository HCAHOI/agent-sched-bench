# Measured KV-swap restore cost rho (2026-07-15)

First hardware measurement of the quantity the entire restore-cost campaign
swept blind: rho = swap_in_cost / swap_out_cost, the KV-cache swap-back
(host->GPU, on the critical path when a call returns) as a fraction of the
swap-out (GPU->host, hidden during the long call). Measured with
`scripts/measure_kv_swap_cost.py` (commit 3f070bf) on a rented H100.

## Hardware / method

- GPU: NVIDIA H100 80GB HBM3, PCIe **Gen5 x16** (full width), driver
  580.126.09, host wise-love-slices-fin-02 (32 vCPU, 181 GB RAM).
- KV layout from Qwen3-Coder-30B-A3B config (a coding model — matches the
  agent workload): 48 layers, 8:1 GQA (32 attn -> 4 KV heads), head_dim
  128 -> 96 KiB/token at bf16 KV, 48 KiB/token at fp8 KV.
- Primitive: pinned-host cudaMemcpy timed with CUDA events, median of
  15-30 reps after warmup. This is exactly vLLM's block-swap mechanism
  (ops.swap_blocks -> cudaMemcpyAsync); the script does not load weights,
  only the config, so rho is model-quality-independent.

## Result: rho ~= 0.94, flat across every size and dtype

| tokens | MiB | swap-out ms | swap-in ms | out GB/s | in GB/s | rho |
|---|---|---|---|---|---|---|
| 512 | 48 | 0.94 | 0.88 | 53.6 | 57.2 | 0.936 |
| 8192 | 768 | 14.96 | 13.98 | 53.8 | 57.6 | 0.935 |
| 131072 | 12288 | 238.2 | 224.3 | 54.1 | 57.4 | 0.942 |
| 294912 | 27648 | 535.0 | 505.2 | 54.2 | 57.4 | 0.944 |
| 589824 | 55296 | 1071.0 | 1010.8 | 54.1 | 57.4 | 0.944 |

**rho = 0.935-0.944, constant across 512 -> 590k tokens (48 MB -> 55 GB)
and identical at bf16 and fp8 KV.** The transfer is bandwidth-bound even at
the smallest size; swap-in (H2D) is marginally FASTER than swap-out (D2H)
on this platform, so the restore is ~94% of the swap-out cost. Sustained
~54 GB/s out / ~57.5 GB/s in is ~86-91% of Gen5 x16 theoretical — sane.

Cost-grid mapping (bf16 KV): the campaign's 500 ms swap-out operating point
= ~275k tokens (27 GB); 1000 ms = ~550k tokens (54 GB). Grid points >=1500
ms exceed a single 80 GB GPU's KV for one contiguous buffer, so they
represent aggregate/batched pool swaps — rho holds there since it is
size-independent.

## Why this is the decisive result

**Real rho ~= 0.94 sits at the pessimistic (rho=1.0) end of the swept
{0, 0.25, 0.5, 1.0}.** The honest operating point is the HARD regime, which
retroactively validates the whole campaign:

- rho=0 results were not merely optimistic — they were the WRONG regime.
  Every policy that looked good only at rho=0 (aggressive early firing, the
  within-task B1 baseline) is net-negative or collapses at rho~=1.
- At the real operating point (interpolating the sweep at rho=1.0):
  Mode B holds (+118 s), B1 within-task is net -398 s, the single gated
  GBM loses to the trie on SWE-ReBench, and the **gate-union is the
  standout** — +136 s vs deadline, +112 s vs the GBM, +18 s vs the trie,
  zero harmful cells. The union that the mechanism analysis surfaced is
  precisely the policy that wins where hardware actually places us.

The measured rho is slightly below 1.0 (0.94), so the rho=1.0 column is a
hair conservative; the qualitative conclusions are unchanged.

## Caveats / next

1. **Idle measurement.** This is the swap primitive with no competing
   traffic. E5 (inference/tool containers contending for PCIe + host
   memory bandwidth) can lower absolute bandwidth; the RATIO should be more
   robust but must be verified — next box task.
2. **Platform-specific.** Gen5 x16 discrete H100. A Gen4 A100 has ~half the
   bandwidth (absolute ms doubles; ratio direction similar). A GH200 /
   unified-memory (NVLink-C2C) system would be radically different
   (near-symmetric or swap-free) and would change the entire policy
   calculus — a deployment-dependent input, exactly as the campaign argued.
3. The measurement script's auto-provenance fields (device_name,
   pcie_link_gen/width) serialized as null — a capture bug; hardware is
   recorded manually above. Fix before the E5 run.
