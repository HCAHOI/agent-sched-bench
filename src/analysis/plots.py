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
