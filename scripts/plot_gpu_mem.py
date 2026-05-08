"""Plot GPU memory breakdown over time from a gpu_resources.json output.

Layers (stacked area, ordered from bottom to top):
  1. model weights (constant baseline from GpuBaseline)
  2. KV cache used (time-varying)
  3. activations residual (time-varying; = total_pid - weights - kv_used)

Note: attn/mlp split requires deep-profile mode (profile-gpu subcommand,
in-process vLLM + module forward hooks). The default HTTP-server path
(what produced this file) only exposes the residual.
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
    n = len(samples)

    ts = np.array([s["ts"] for s in samples])
    ts -= ts[0]  # relative seconds
    kv = np.array([s["kv_cache_used_mib"] or 0.0 for s in samples])
    acts = np.array([s["activations_mib"] or 0.0 for s in samples])
    total = np.array([s["total_pid_mib"] for s in samples])
    weights = np.full_like(total, weights_mib)

    # Convert to GiB for readability
    weights_g = weights / 1024.0
    kv_g = kv / 1024.0
    acts_g = acts / 1024.0
    total_g = total / 1024.0

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Top panel: stacked area
    ax_top.fill_between(ts, 0, weights_g, label=f"model weights ({weights_mib:.0f} MiB, const)",
                        color="#4E79A7", alpha=0.85)
    ax_top.fill_between(ts, weights_g, weights_g + kv_g,
                        label="KV cache used (vLLM Prometheus × baseline)",
                        color="#F28E2B", alpha=0.85)
    ax_top.fill_between(ts, weights_g + kv_g, weights_g + kv_g + acts_g,
                        label="activations residual (CUDA graphs + allocator headroom + true acts)",
                        color="#E15759", alpha=0.55)
    ax_top.plot(ts, total_g, color="black", linewidth=1.2, linestyle="--",
                label=f"total PID memory (nvidia-smi --query-compute-apps, peak {summary['peak_total_pid_mib']:.0f} MiB)")
    ax_top.axhline(y=(weights_mib + kv_total_mib) / 1024.0, color="gray",
                   linestyle=":", linewidth=1,
                   label=f"weights + kv_total reserved ({(weights_mib + kv_total_mib):.0f} MiB)")

    ax_top.set_ylabel("GPU memory (GiB)")
    ax_top.set_title(
        f"GPU memory breakdown over time — {model} on RTX 5090\n"
        f"trace: wemake-python-styleguide-2343 (79 LLM calls, container mode, {summary['duration_s']:.0f}s)"
    )
    ax_top.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_top.set_ylim(0, total_g.max() * 1.05)
    ax_top.grid(alpha=0.25)

    # Bottom panel: KV cache used alone (zoomed)
    ax_bot.plot(ts, kv, color="#F28E2B", linewidth=0.8)
    ax_bot.fill_between(ts, 0, kv, color="#F28E2B", alpha=0.30)
    ax_bot.axhline(y=kv_total_mib, color="gray", linestyle=":", linewidth=1,
                   label=f"kv_cache_total reserved ({kv_total_mib:.0f} MiB)")
    ax_bot.set_xlabel("time since simulate start (s)")
    ax_bot.set_ylabel("KV cache used (MiB)")
    ax_bot.set_title("KV cache used (zoomed) — peaks align with in-flight LLM calls")
    ax_bot.legend(loc="upper right", fontsize=8)
    ax_bot.grid(alpha=0.25)

    # Footer note about missing attn/mlp dimension
    fig.text(0.5, 0.005,
             "Note: attn vs MLP activation split requires deep-profile mode "
             "(`profile-gpu` subcommand with in-process vLLM + module hooks). "
             "HTTP-server path only exposes the residual.",
             ha="center", fontsize=8, style="italic", color="dimgray")

    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path} ({n} samples)")


if __name__ == "__main__":
    in_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/wemake_gpu.json")
    out_path = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/wemake_gpu_mem.png")
    main(in_path, out_path)
