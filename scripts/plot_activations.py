"""Plot per-layer attn/mlp activation distribution heatmaps from probe output."""
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
    n_layers = data["n_layers"]
    n_iters = data["n_iterations"]
    iters = data["iterations"]

    metrics = ["abs_mean", "std", "abs_max", "l2_norm", "sparsity_1e-3"]
    kinds = ["attn", "mlp"]

    # build (kind, metric) → layer×iter array
    grids: dict = {}
    for kind in kinds:
        for metric in metrics:
            arr = np.zeros((n_layers, n_iters))
            for col, it in enumerate(iters):
                for row in it["stats"]:
                    if row["kind"] == kind:
                        arr[row["layer"], col] = row[metric]
            grids[(kind, metric)] = arr

    # 2 rows (attn/mlp) × 4 cols (abs_mean / std / abs_max / sparsity)
    metrics_to_plot = ["abs_mean", "std", "abs_max", "sparsity_1e-3"]
    fig, axes = plt.subplots(2, len(metrics_to_plot), figsize=(4*len(metrics_to_plot), 8))
    iter_token_lens = [it["input_tokens"] for it in iters]
    iter_labels = [f"i{it['iter']}\n{it['input_tokens']}t" for it in iters]

    for r, kind in enumerate(kinds):
        for c, metric in enumerate(metrics_to_plot):
            ax = axes[r, c]
            arr = grids[(kind, metric)]
            cmap = "viridis" if metric != "sparsity_1e-3" else "plasma"
            im = ax.imshow(arr, aspect="auto", cmap=cmap, origin="lower")
            ax.set_title(f"{kind} — {metric}", fontsize=10)
            ax.set_xlabel("iter (tokens)" if r == 1 else "")
            ax.set_ylabel("layer idx" if c == 0 else "")
            ax.set_xticks(range(n_iters))
            ax.set_xticklabels(iter_labels, fontsize=7)
            cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
            cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"{data['model']} — per-layer attn/MLP output activation stats over {n_iters} agent iterations\n"
        f"(trace: wemake-python-styleguide-2343, capped at 4096 tokens; HF transformers forward, no vLLM)",
        fontsize=11
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    in_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/wemake_activations.json")
    out_path = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/wemake_activations.png")
    main(in_path, out_path)
