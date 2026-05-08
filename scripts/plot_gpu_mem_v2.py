"""GPU memory budget over time — honest version.

The previous chart's "activations residual" (= nvidia-smi total - weights -
kv_used) was misleading. It's dominated by:
  - vLLM caching allocator headroom (pre-allocated at gpu_memory_utilization)
  - CUDA graphs (torch.compile prewarms many seq-length-shaped graphs)
Neither is "activations". Real per-step activations are in the hundreds-of-MiB
range and not separable from outside (would need deep-profile mode +
torch.cuda.memory_allocated() module hooks).

This chart shows the vLLM memory BUDGET layout instead:
  - model weights (constant)
  - KV cache pool (constant, reserved by vLLM at startup)
    - inside it: kv_used (time-varying)
  - vLLM allocator headroom + CUDA graphs (the rest of nvidia-smi total,
    effectively constant — labeled but not stacked as "activations")
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(in_path: Path, out_path: Path) -> None:
    with open(in_path) as f:
        data = json.load(f)
    baseline = data["gpu_baseline"]
    samples = data["gpu_samples"]
    summary = data["summary"]
    weights_mib = baseline["weights_mib"]
    kv_total_mib = baseline["kv_cache_total_mib"]
    model = baseline["model"]

    ts = np.array([s["ts"] for s in samples])
    ts -= ts[0]
    kv = np.array([s["kv_cache_used_mib"] or 0.0 for s in samples])
    total = np.array([s["total_pid_mib"] for s in samples])

    weights_g = weights_mib / 1024.0
    kv_total_g = kv_total_mib / 1024.0
    kv_g = kv / 1024.0
    total_g = total / 1024.0
    pool_top = weights_g + kv_total_g  # constant baseline + KV pool ceiling

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # weights (constant, blue)
    ax_top.fill_between(ts, 0, weights_g, color="#4E79A7", alpha=0.85,
                        label=f"model weights ({weights_mib:.0f} MiB, constant)")

    # KV cache USED (orange, time-varying, drawn inside the pool)
    ax_top.fill_between(ts, weights_g, weights_g + kv_g, color="#F28E2B", alpha=0.85,
                        label="KV cache used (time-varying)")

    # KV cache FREE pool (translucent — visible budget vLLM holds for KV)
    ax_top.fill_between(ts, weights_g + kv_g, pool_top, color="#F28E2B", alpha=0.15,
                        hatch="//", linewidth=0,
                        label=f"KV cache free pool (reserved {kv_total_mib:.0f} MiB total)")

    # nvidia-smi process total (the actual GPU footprint, mostly constant)
    ax_top.plot(ts, total_g, color="black", linewidth=1.4, linestyle="-",
                label=f"nvidia-smi PID total ({total_g.min():.1f}–{total_g.max():.1f} GiB)")

    # The gap between (weights + kv_total) and (nvidia-smi total) is
    # vLLM allocator headroom + torch.compile CUDA graphs — annotate it.
    headroom_g = total_g.mean() - pool_top
    ax_top.axhline(y=pool_top, color="gray", linestyle="--", linewidth=0.8)
    ax_top.text(ts[len(ts) // 2], (pool_top + total_g.mean()) / 2,
                f"vLLM allocator headroom + CUDA graphs ≈ {headroom_g:.1f} GiB\n"
                f"(NOT activations — measurable only via profile-gpu module hooks)",
                ha="center", fontsize=9, style="italic", color="dimgray",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="lightgray"))

    ax_top.set_ylabel("GPU memory (GiB)")
    ax_top.set_title(
        f"GPU memory budget over time — {model} on RTX 5090\n"
        f"trace: wemake-python-styleguide-2343 (79 LLM calls, container mode, {summary['duration_s']:.0f}s)"
    )
    ax_top.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax_top.set_ylim(0, total_g.max() * 1.05)
    ax_top.grid(alpha=0.25)

    # Bottom: KV cache used alone with percentile markers
    ax_bot.plot(ts, kv, color="#F28E2B", linewidth=0.8)
    ax_bot.fill_between(ts, 0, kv, color="#F28E2B", alpha=0.30)
    sorted_kv = np.sort(kv)
    p50 = sorted_kv[len(sorted_kv) // 2]
    p90 = sorted_kv[int(len(sorted_kv) * 0.9)]
    p99 = sorted_kv[int(len(sorted_kv) * 0.99)]
    ax_bot.axhline(y=p50, color="green", linestyle=":", linewidth=1, label=f"p50 = {p50:.0f}")
    ax_bot.axhline(y=p90, color="orange", linestyle=":", linewidth=1, label=f"p90 = {p90:.0f}")
    ax_bot.axhline(y=p99, color="red", linestyle=":", linewidth=1, label=f"p99 = {p99:.0f}")
    ax_bot.axhline(y=kv_total_mib, color="gray", linestyle="--", linewidth=0.8,
                   label=f"pool ceiling {kv_total_mib:.0f}")
    ax_bot.set_xlabel("time since simulate start (s)")
    ax_bot.set_ylabel("KV cache used (MiB)")
    ax_bot.set_title(f"KV cache utilization — peak {kv.max():.0f} MiB ({kv.max()/kv_total_mib*100:.1f}% of pool)")
    ax_bot.legend(loc="upper left", fontsize=8, ncol=4)
    ax_bot.grid(alpha=0.25)

    fig.text(0.5, 0.005,
             "Real attn/MLP activation split needs deep-profile mode "
             "(`profile-gpu` subcommand): in-process vllm.LLM(...) + module forward hooks "
             "on torch.cuda.memory_allocated() pre/post delta.",
             ha="center", fontsize=8, style="italic", color="dimgray")

    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    in_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/wemake_gpu.json")
    out_path = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/wemake_gpu_mem_v2.png")
    main(in_path, out_path)
