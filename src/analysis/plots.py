from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_throughput_vs_concurrency(frame: pd.DataFrame, output_path: Path) -> None:
    """Plot throughput against concurrency for a processed analysis table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(frame["concurrency"], frame["throughput_steps_per_min"], marker="o")
    plt.xlabel("Concurrency")
    plt.ylabel("Throughput (steps/min)")
    plt.title("Throughput vs Concurrency")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_latency_breakdown(frame: pd.DataFrame, output_path: Path) -> None:
    """Plot average LLM/tool latency breakdown per concurrency point."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.bar(frame["concurrency"], frame["avg_llm_ms"], label="LLM")
    plt.bar(frame["concurrency"], frame["avg_tool_ms"], bottom=frame["avg_llm_ms"], label="Tool")
    plt.xlabel("Concurrency")
    plt.ylabel("Average latency (ms)")
    plt.title("Latency Breakdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_prefix_cache_hit_vs_concurrency(frame: pd.DataFrame, output_path: Path) -> None:
    """Plot prefix cache hit rate against concurrency."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(frame["concurrency"], frame["prefix_cache_hit_rate"], marker="o")
    plt.xlabel("Concurrency")
    plt.ylabel("Prefix cache hit rate")
    plt.title("Prefix Cache Hit Rate vs Concurrency")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def identify_cliff_point(frame: pd.DataFrame, *, throughput_drop_threshold: float = 0.1) -> int | None:
    """Identify the first concurrency point where throughput drops materially."""
    ordered = frame.sort_values("concurrency").reset_index(drop=True)
    best_throughput = None
    for row in ordered.itertuples(index=False):
        throughput = float(row.throughput_steps_per_min)
        if best_throughput is None or throughput > best_throughput:
            best_throughput = throughput
            continue
        if best_throughput > 0 and throughput <= best_throughput * (1.0 - throughput_drop_threshold):
            return int(row.concurrency)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render simple analysis plots.")
    parser.add_argument("csv_file")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.csv_file)
    plot_throughput_vs_concurrency(frame, Path(args.output))


if __name__ == "__main__":
    main()
