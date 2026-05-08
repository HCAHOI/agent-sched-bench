"""Plot full-sequence downsampled attention stage heatmaps."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


ROLE_COLORS = {
    "system": "#444444",
    "user": "#1f77b4",
    "assistant": "#2ca02c",
    "assistant_message": "#2ca02c",
    "assistant_call": "#17becf",
    "tool": "#d62728",
    "tool_result": "#d62728",
    "meta": "#8c8c8c",
    "gen_prompt": "#9467bd",
}

ROLE_TICK_LABELS = {
    "system": "sys",
    "user": "usr",
    "assistant": "asst",
    "assistant_message": "msg",
    "assistant_call": "call",
    "tool": "tool",
    "tool_result": "res",
    "meta": "meta",
    "gen_prompt": "gen",
}


def title_for(sample: dict) -> str:
    return (
        f"{sample['label']} / call {sample['call_index']} / iter {sample['trace_iteration']}\n"
        f"{sample['input_tokens']} toks"
    )


def require_finite_array(name: str, arr: np.ndarray) -> None:
    bad = int((~np.isfinite(arr)).sum())
    if bad:
        raise ValueError(f"{name} contains {bad} non-finite values")


def add_role_legend(fig: plt.Figure) -> None:
    handles = [
        Patch(facecolor=color, edgecolor="none", label=role)
        for role, color in ROLE_COLORS.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8)


def _require_samples(data: dict) -> list[dict]:
    samples = data.get("samples") or []
    if not samples:
        raise ValueError("attention plot input contains no samples")
    return samples


def add_segment_lines(ax, sample: dict, horizontal: bool) -> None:
    for segment in sample["segments"]:
        pos = segment["start_bin"]
        color = ROLE_COLORS.get(segment["role"], "#999999")
        ax.axvline(pos, color=color, linewidth=0.45, alpha=0.8)
        if horizontal:
            ax.axhline(pos, color=color, linewidth=0.45, alpha=0.8)


def segment_tick_positions(n_segments: int) -> np.ndarray:
    if n_segments <= 24:
        return np.arange(n_segments)
    return np.linspace(0, n_segments - 1, min(14, n_segments), dtype=int)


def segment_tick_labels(sample: dict, positions: np.ndarray) -> list[str]:
    return [
        f"{int(pos)}:{ROLE_TICK_LABELS.get(sample['segments'][int(pos)]['role'], '?')}"
        for pos in positions
    ]


def plot_prefill(data: dict, out_path: Path) -> None:
    samples = _require_samples(data)
    layers = [layer["layer"] for layer in samples[0]["layers"]]
    fig, axes = plt.subplots(
        len(samples),
        len(layers),
        figsize=(4.0 * len(layers), 3.7 * len(samples)),
        squeeze=False,
    )
    for row, sample in enumerate(samples):
        for col, layer_record in enumerate(sample["layers"]):
            ax = axes[row][col]
            arr = np.asarray(layer_record["prefill_map_downsampled"], dtype=float)
            require_finite_array("prefill_map_downsampled", arr)
            vmax = float(np.quantile(arr, 0.995))
            im = ax.imshow(
                arr,
                origin="lower",
                aspect="auto",
                cmap="magma",
                vmin=0.0,
                vmax=vmax if vmax > 0 else None,
            )
            add_segment_lines(ax, sample, horizontal=True)
            if row == 0:
                ax.set_title(f"layer {layer_record['layer']}", fontsize=10)
            if col == 0:
                ax.set_ylabel(title_for(sample), fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(
        "Full-sequence prefill attention, downsampled\n"
        "x = key token bin, y = query token bin; segment boundaries are colored",
        fontsize=12,
    )
    add_role_legend(fig)
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_prefill_segments(data: dict, out_path: Path) -> None:
    samples = _require_samples(data)
    layers = [layer["layer"] for layer in samples[0]["layers"]]
    fig, axes = plt.subplots(
        len(samples),
        len(layers),
        figsize=(4.0 * len(layers), 3.7 * len(samples)),
        squeeze=False,
    )
    for row, sample in enumerate(samples):
        positions = segment_tick_positions(len(sample["segments"]))
        labels = segment_tick_labels(sample, positions)
        for col, layer_record in enumerate(sample["layers"]):
            ax = axes[row][col]
            arr = np.asarray(layer_record["prefill_seg_to_seg"], dtype=float)
            require_finite_array("prefill_seg_to_seg", arr)
            vmax = float(np.quantile(arr, 0.995))
            im = ax.imshow(
                arr,
                origin="lower",
                aspect="auto",
                cmap="magma",
                vmin=0.0,
                vmax=vmax if vmax > 0 else None,
            )
            if row == 0:
                ax.set_title(f"layer {layer_record['layer']}", fontsize=10)
            if col == 0:
                ax.set_ylabel(title_for(sample), fontsize=8)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=90, fontsize=5)
            ax.set_yticks(positions)
            ax.set_yticklabels(labels, fontsize=5)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(
        "Prefill segment-to-segment attention mass\n"
        "x = key segment, y = query segment; labels are index:role-initial",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_decode(data: dict, out_path: Path) -> None:
    samples = _require_samples(data)
    layers = [layer["layer"] for layer in samples[0]["layers"]]
    fig, axes = plt.subplots(
        1,
        len(samples),
        figsize=(5.2 * len(samples), 4.0),
        squeeze=False,
    )
    for col, sample in enumerate(samples):
        ax = axes[0][col]
        arr = np.asarray(
            [layer["decode_attn_downsampled"] for layer in sample["layers"]],
            dtype=float,
        )
        require_finite_array("decode_attn_downsampled", arr)
        vmax = float(np.quantile(arr, 0.995))
        im = ax.imshow(
            arr,
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=vmax if vmax > 0 else None,
        )
        add_segment_lines(ax, sample, horizontal=False)
        ax.set_title(f"{title_for(sample)}\nnext token {sample['next_token_text']!r}", fontsize=8)
        ax.set_xlabel("key token bin")
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f"L{layer}" for layer in layers], fontsize=8)
        if col == 0:
            ax.set_ylabel("decode query layer")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle("Next-token decode attention to the full sequence", fontsize=12)
    add_role_legend(fig)
    fig.tight_layout(rect=[0, 0.08, 1, 0.90])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_decode_segments(data: dict, out_path: Path) -> None:
    samples = _require_samples(data)
    layers = [layer["layer"] for layer in samples[0]["layers"]]
    fig, axes = plt.subplots(
        1,
        len(samples),
        figsize=(5.6 * len(samples), 4.0),
        squeeze=False,
    )
    for col, sample in enumerate(samples):
        ax = axes[0][col]
        arr = np.asarray([layer["decode_to_seg"] for layer in sample["layers"]], dtype=float)
        require_finite_array("decode_to_seg", arr)
        vmax = float(np.quantile(arr, 0.995))
        im = ax.imshow(
            arr,
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=vmax if vmax > 0 else None,
        )
        positions = segment_tick_positions(len(sample["segments"]))
        ax.set_xticks(positions)
        ax.set_xticklabels(segment_tick_labels(sample, positions), rotation=90, fontsize=6)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f"L{layer}" for layer in layers], fontsize=8)
        ax.set_title(title_for(sample), fontsize=8)
        ax.set_xlabel("key segment")
        if col == 0:
            ax.set_ylabel("decode query layer")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(
        "Next-token decode attention mass by prompt segment\n"
        "x labels are index:role-initial",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    plot_prefill(
        data,
        args.output_prefix.with_name(args.output_prefix.name + "_prefill.png"),
    )
    plot_prefill_segments(
        data,
        args.output_prefix.with_name(args.output_prefix.name + "_prefill_segments.png"),
    )
    plot_decode(
        data,
        args.output_prefix.with_name(args.output_prefix.name + "_decode.png"),
    )
    plot_decode_segments(
        data,
        args.output_prefix.with_name(args.output_prefix.name + "_decode_segments.png"),
    )


if __name__ == "__main__":
    main()
