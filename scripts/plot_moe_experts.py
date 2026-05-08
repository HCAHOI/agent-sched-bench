"""Plot MoE expert utilization heatmaps.

Shows:
  - Top: per-iter (layer × expert) heatmap of expert token-counts (normalized)
  - Bottom-left: per-layer routing entropy across iterations (line)
  - Bottom-right: expert "rank" stability — for each iter, sort experts by usage
    and plot top-N usage to see how skewed the routing is.
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
    n_experts = data["num_experts"]
    top_k = data["top_k"]
    n_layers = data["n_layers"]
    iters = data["iterations"]
    n_iters = len(iters)

    # tensor shape: [iter, layer, expert]
    counts = np.zeros((n_iters, n_layers, n_experts))
    masses = np.zeros((n_iters, n_layers, n_experts))
    entropy = np.zeros((n_iters, n_layers))
    for i_iter, it in enumerate(iters):
        for layer_row in it["layers"]:
            li = layer_row["layer"]
            counts[i_iter, li] = layer_row["expert_token_counts"]
            masses[i_iter, li] = layer_row["expert_prob_mass"]
            entropy[i_iter, li] = layer_row["routing_entropy_mean"]

    # Normalize counts within each (iter, layer) to relative frequency [0, 1]
    counts_norm = counts / counts.sum(axis=-1, keepdims=True)

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, n_iters, height_ratios=[3, 1.2, 1.2], hspace=0.55, wspace=0.25)

    # Row 1: heatmap per iter (layer × expert)
    vmax = float(counts_norm.max())
    for i_iter in range(n_iters):
        ax = fig.add_subplot(gs[0, i_iter])
        im = ax.imshow(counts_norm[i_iter], aspect="auto", cmap="magma", origin="lower",
                       vmin=0, vmax=vmax)
        ax.set_title(f"iter {iters[i_iter]['iter']} ({iters[i_iter]['input_tokens']}t)", fontsize=9)
        ax.set_xlabel("expert idx")
        if i_iter == 0:
            ax.set_ylabel("layer idx")
        ax.set_xticks([0, 16, 32, 48, 63])
        if i_iter == n_iters - 1:
            cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
            cbar.set_label("rel. token freq", fontsize=8)
            cbar.ax.tick_params(labelsize=7)

    # Row 2: routing entropy across layers, one line per iter
    ax_ent = fig.add_subplot(gs[1, :])
    layer_x = np.arange(n_layers)
    for i_iter, it in enumerate(iters):
        ax_ent.plot(layer_x, entropy[i_iter], marker="o", markersize=3,
                    label=f"i{it['iter']} ({it['input_tokens']}t)", alpha=0.8)
    # Reference lines: log(num_experts) = max possible entropy in nats
    max_entropy = np.log(n_experts)
    ax_ent.axhline(y=max_entropy, color="gray", linestyle=":", linewidth=1,
                   label=f"max = ln({n_experts}) = {max_entropy:.2f}")
    # top-k uniform routing entropy
    topk_uniform = -top_k * (1.0 / n_experts) * np.log(1.0 / n_experts)
    ax_ent.set_xlabel("layer idx")
    ax_ent.set_ylabel("entropy (nats)")
    ax_ent.set_title("Per-layer routing entropy (mean over tokens) — high = balanced, low = collapse")
    ax_ent.legend(fontsize=8, ncol=6, loc="lower center")
    ax_ent.grid(alpha=0.3)

    # Row 3: cumulative expert load (Lorenz curve) for one iter (last)
    ax_lor = fig.add_subplot(gs[2, :])
    last = counts_norm[-1]  # [layer, expert]
    layers_to_plot = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    for li in layers_to_plot:
        sorted_load = np.sort(last[li])[::-1]
        cumulative = np.cumsum(sorted_load)
        ax_lor.plot(np.arange(1, n_experts + 1), cumulative, marker=".",
                    markersize=3, label=f"layer {li}", alpha=0.8)
    # Reference: top-k uniform line
    topk_uniform_cum = np.zeros(n_experts)
    topk_uniform_cum[:top_k] = np.arange(1, top_k + 1) / top_k  # if perfectly balanced over top-k
    ax_lor.plot(np.arange(1, n_experts + 1), topk_uniform_cum, color="gray", linestyle=":",
                label=f"perfect top-{top_k} uniform", linewidth=1.2)
    ax_lor.set_xlabel("expert rank (most-used → least-used)")
    ax_lor.set_ylabel("cumulative token frequency")
    ax_lor.set_title(f"Expert load concentration (last iter) — closer to gray-line = more balanced")
    ax_lor.legend(fontsize=8, ncol=6, loc="lower right")
    ax_lor.grid(alpha=0.3)
    ax_lor.set_ylim(0, 1.02)

    fig.suptitle(
        f"{data['model']} — MoE expert routing on agent prompts (wemake-python-styleguide-2343, "
        f"{n_experts} experts, top-{top_k})",
        fontsize=11
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    in_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/wemake_moe_experts.json")
    out_path = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/wemake_moe_experts.png")
    main(in_path, out_path)
