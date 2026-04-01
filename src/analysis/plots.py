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


def plot_throughput_comparison(frames: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Overlay throughput-vs-concurrency curves for multiple systems."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    plt.figure(figsize=(8, 5))
    for i, (system_name, frame) in enumerate(frames.items()):
        plt.plot(
            frame["concurrency"],
            frame["throughput_steps_per_min"],
            marker=markers[i % len(markers)],
            label=system_name,
        )
    plt.xlabel("Concurrency")
    plt.ylabel("Throughput (steps/min)")
    plt.title("Throughput vs Concurrency: System Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_gantt(trace_path: Path, output_path: Path) -> None:
    """Render a Gantt chart showing LLM reasoning vs tool execution per agent.

    Reads a JSONL trace file and plots one horizontal bar per agent, with
    blue segments for LLM calls and orange segments for tool execution.
    """
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))

    steps = [e for e in entries if e.get("type") == "step"]
    if not steps:
        return

    t0 = min(s["ts_start"] for s in steps)
    agents = sorted({s["agent_id"] for s in steps})
    agent_idx = {a: i for i, a in enumerate(agents)}

    fig, ax = plt.subplots(figsize=(14, max(3, len(agents) * 0.4)))

    for s in steps:
        y = agent_idx[s["agent_id"]]
        start = s["ts_start"] - t0
        llm_dur = s["llm_latency_ms"] / 1000.0
        tool_dur = (s.get("tool_duration_ms") or 0) / 1000.0

        # LLM reasoning segment
        ax.barh(y, llm_dur, left=start, height=0.6, color="#4285F4", edgecolor="none")
        # Tool execution segment (immediately after LLM)
        if tool_dur > 0:
            ax.barh(y, tool_dur, left=start + llm_dur, height=0.6, color="#EA4335", edgecolor="none")

    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels(agents, fontsize=7)
    ax.set_xlabel("Time (seconds)")
    ax.set_title(f"Agent Timeline — {trace_path.stem}")
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(color="#4285F4", label="LLM"), Patch(color="#EA4335", label="Tool")],
        loc="upper right",
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
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
    parser.add_argument(
        "--plot-type",
        choices=["throughput", "latency", "cache_hit", "comparison"],
        default="throughput",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if args.plot_type == "comparison":
        # csv_file is a comma-separated list of "name:path" pairs
        frames: dict[str, pd.DataFrame] = {}
        for entry in args.csv_file.split(","):
            name, path = entry.split(":", 1)
            frames[name] = pd.read_csv(path)
        plot_throughput_comparison(frames, output_path)
    else:
        frame = pd.read_csv(args.csv_file)
        if args.plot_type == "throughput":
            plot_throughput_vs_concurrency(frame, output_path)
        elif args.plot_type == "latency":
            plot_latency_breakdown(frame, output_path)
        elif args.plot_type == "cache_hit":
            plot_prefix_cache_hit_vs_concurrency(frame, output_path)


if __name__ == "__main__":
    main()
