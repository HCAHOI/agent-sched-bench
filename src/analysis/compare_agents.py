"""Cross-agent comparison: load two trace sets and produce metrics + plots.

Usage:
    python -m analysis.compare_agents \
        --agent-a traces/swebench_verified/.../results.jsonl \
        --agent-b traces/swebench_verified/.../results.jsonl \
        --label-a mini-swe-agent --label-b openclaw \
        --output traces/analysis/pilot-comparison
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_results(path: Path) -> list[dict[str, Any]]:
    """Load results.jsonl into a list of dicts."""
    results = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            results.append(json.loads(line))
    return results


def load_trace_steps(trace_path: Path) -> list[dict[str, Any]]:
    """Load step records from a single trace JSONL file."""
    steps = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("type") == "step":
            steps.append(rec)
    return steps


def find_trace_file(results_path: Path, instance_id: str) -> Path | None:
    """Find the trace JSONL for a given instance_id in the same run directory."""
    run_dir = results_path.parent
    candidate = run_dir / f"{instance_id}.jsonl"
    if candidate.exists():
        return candidate
    return None


# ---------------------------------------------------------------------------
# Per-task metrics
# ---------------------------------------------------------------------------


def compute_task_metrics(
    result: dict[str, Any], steps: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compute metrics for one task from its result + step records."""
    n_steps = result.get("n_steps") or len(steps)
    elapsed_s = result.get("elapsed_s", 0.0)
    total_llm_ms = result.get("total_llm_ms", 0.0)
    total_tool_ms = result.get("total_tool_ms", 0.0)
    total_tokens = result.get("total_tokens", 0)

    # If result doesn't have totals, compute from steps
    if total_llm_ms == 0.0 and steps:
        total_llm_ms = sum(s.get("llm_latency_ms", 0.0) for s in steps)
    if total_tool_ms == 0.0 and steps:
        total_tool_ms = sum(s.get("tool_duration_ms", 0.0) or 0.0 for s in steps)
    if total_tokens == 0 and steps:
        total_tokens = sum(
            (s.get("prompt_tokens", 0) or 0) + (s.get("completion_tokens", 0) or 0)
            for s in steps
        )

    # Tool usage distribution
    tool_counts: Counter[str] = Counter()
    for s in steps:
        tool_name = s.get("tool_name")
        if tool_name:
            tool_counts[tool_name] += 1

    # Token flow: prompt_tokens per step
    token_flow = [s.get("prompt_tokens", 0) or 0 for s in steps]

    # Step durations
    step_durations = []
    for s in steps:
        ts_start = s.get("ts_start", 0.0)
        ts_end = s.get("ts_end", 0.0)
        if ts_start and ts_end:
            step_durations.append(ts_end - ts_start)

    # LLM/tool time ratio
    total_time = total_llm_ms + total_tool_ms
    llm_ratio = total_llm_ms / total_time if total_time > 0 else 0.0

    # Completion token sizes per step (indicates response verbosity)
    completion_sizes = [s.get("completion_tokens", 0) or 0 for s in steps]

    return {
        "instance_id": result["instance_id"],
        "success": result.get("success", False),
        "official_resolved": result.get("official_resolved"),
        "patch_generated": result.get("patch_generated", False),
        "n_steps": n_steps,
        "elapsed_s": elapsed_s,
        "total_llm_ms": total_llm_ms,
        "total_tool_ms": total_tool_ms,
        "total_tokens": total_tokens,
        "tool_counts": dict(tool_counts),
        "tool_diversity": len(tool_counts),
        "token_flow": token_flow,
        "step_durations": step_durations,
        "llm_ratio": llm_ratio,
        "avg_completion_tokens": (
            sum(completion_sizes) / len(completion_sizes) if completion_sizes else 0.0
        ),
        "max_completion_tokens": max(completion_sizes) if completion_sizes else 0,
    }


# ---------------------------------------------------------------------------
# Aggregate comparison
# ---------------------------------------------------------------------------


def compute_aggregate(task_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-task metrics into scaffold-level summary."""
    n = len(task_metrics)
    if n == 0:
        return {}

    def avg(key: str) -> float:
        vals = [m[key] for m in task_metrics if m.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    # Merge tool counts across tasks
    combined_tools: Counter[str] = Counter()
    for m in task_metrics:
        combined_tools.update(m.get("tool_counts", {}))

    # Weighted average of completion tokens (weighted by step count, not by task)
    total_completion_tokens = sum(
        m.get("avg_completion_tokens", 0) * m.get("n_steps", 0) for m in task_metrics
    )
    total_steps = sum(m.get("n_steps", 0) for m in task_metrics)
    weighted_avg_completion = (
        total_completion_tokens / total_steps if total_steps > 0 else 0.0
    )

    # solve_rate = official_resolved (harness result), distinct from patch_rate
    solved = sum(1 for m in task_metrics if m.get("official_resolved"))

    return {
        "n_tasks": n,
        "solve_rate": solved / n,
        "patch_rate": sum(1 for m in task_metrics if m["patch_generated"]) / n,
        "avg_steps": avg("n_steps"),
        "avg_elapsed_s": avg("elapsed_s"),
        "avg_llm_ms": avg("total_llm_ms"),
        "avg_tool_ms": avg("total_tool_ms"),
        "avg_tokens": avg("total_tokens"),
        "avg_llm_ratio": avg("llm_ratio"),
        "avg_tool_diversity": avg("tool_diversity"),
        "avg_completion_tokens": weighted_avg_completion,
        "total_tool_distribution": dict(combined_tools.most_common()),
    }


def build_comparison(
    metrics_a: list[dict[str, Any]],
    metrics_b: list[dict[str, Any]],
    label_a: str,
    label_b: str,
) -> dict[str, Any]:
    """Build the full comparison structure."""
    agg_a = compute_aggregate(metrics_a)
    agg_b = compute_aggregate(metrics_b)

    # Per-task paired comparison (tasks present in both)
    ids_a = {m["instance_id"]: m for m in metrics_a}
    ids_b = {m["instance_id"]: m for m in metrics_b}
    common_ids = sorted(set(ids_a) & set(ids_b))

    paired = []
    for iid in common_ids:
        ma, mb = ids_a[iid], ids_b[iid]
        paired.append(
            {
                "instance_id": iid,
                label_a: {
                    "n_steps": ma["n_steps"],
                    "elapsed_s": ma["elapsed_s"],
                    "total_tokens": ma["total_tokens"],
                    "tool_diversity": ma["tool_diversity"],
                    "llm_ratio": ma["llm_ratio"],
                    "patch_generated": ma["patch_generated"],
                },
                label_b: {
                    "n_steps": mb["n_steps"],
                    "elapsed_s": mb["elapsed_s"],
                    "total_tokens": mb["total_tokens"],
                    "tool_diversity": mb["tool_diversity"],
                    "llm_ratio": mb["llm_ratio"],
                    "patch_generated": mb["patch_generated"],
                },
            }
        )

    return {
        "agents": {label_a: agg_a, label_b: agg_b},
        "paired_tasks": paired,
        "common_task_count": len(common_ids),
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def render_markdown(
    comparison: dict[str, Any],
    label_a: str,
    label_b: str,
) -> str:
    """Render comparison as a human-readable Markdown report."""
    lines: list[str] = []
    lines.append("# Agent Comparison Report\n")

    agg_a = comparison["agents"][label_a]
    agg_b = comparison["agents"][label_b]

    lines.append("## Summary\n")
    lines.append(f"| Metric | {label_a} | {label_b} |")
    lines.append(
        "|--------|" + "-" * (len(label_a) + 2) + "|" + "-" * (len(label_b) + 2) + "|"
    )

    metrics_display = [
        ("Tasks", "n_tasks", "d"),
        ("Solve rate", "solve_rate", ".1%"),
        ("Patch rate", "patch_rate", ".1%"),
        ("Avg steps", "avg_steps", ".1f"),
        ("Avg elapsed (s)", "avg_elapsed_s", ".1f"),
        ("Avg LLM time (ms)", "avg_llm_ms", ".0f"),
        ("Avg tool time (ms)", "avg_tool_ms", ".0f"),
        ("Avg tokens", "avg_tokens", ".0f"),
        ("Avg LLM/total ratio", "avg_llm_ratio", ".2f"),
        ("Avg tool diversity", "avg_tool_diversity", ".1f"),
        ("Avg completion tokens/step", "avg_completion_tokens", ".0f"),
    ]

    for display_name, key, fmt in metrics_display:
        va = agg_a.get(key, 0)
        vb = agg_b.get(key, 0)
        lines.append(f"| {display_name} | {va:{fmt}} | {vb:{fmt}} |")

    lines.append("\n## Tool Distribution\n")
    lines.append(f"### {label_a}")
    for tool, count in sorted(
        agg_a.get("total_tool_distribution", {}).items(), key=lambda x: -x[1]
    ):
        lines.append(f"- {tool}: {count}")
    lines.append(f"\n### {label_b}")
    for tool, count in sorted(
        agg_b.get("total_tool_distribution", {}).items(), key=lambda x: -x[1]
    ):
        lines.append(f"- {tool}: {count}")

    lines.append("\n## Per-Task Comparison\n")
    paired = comparison.get("paired_tasks", [])
    if paired:
        lines.append(
            f"| Task | {label_a} steps | {label_b} steps | {label_a} tokens | {label_b} tokens | {label_a} patch | {label_b} patch |"
        )
        lines.append("|------|" + "---|" * 6)
        for p in paired:
            a = p[label_a]
            b = p[label_b]
            lines.append(
                f"| {p['instance_id']} | {a['n_steps']} | {b['n_steps']} "
                f"| {a['total_tokens']} | {b['total_tokens']} "
                f"| {'Y' if a['patch_generated'] else 'N'} | {'Y' if b['patch_generated'] else 'N'} |"
            )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Plots (optional, requires matplotlib)
# ---------------------------------------------------------------------------


def generate_plots(
    metrics_a: list[dict[str, Any]],
    metrics_b: list[dict[str, Any]],
    label_a: str,
    label_b: str,
    output_dir: Path,
) -> list[Path]:
    """Generate comparison plots. Returns list of saved file paths."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    ids_a = {m["instance_id"]: m for m in metrics_a}
    ids_b = {m["instance_id"]: m for m in metrics_b}
    common = sorted(set(ids_a) & set(ids_b))
    short_ids = [iid.split("__")[-1][:15] for iid in common]

    # 1. Steps comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(common))
    width = 0.35
    ax.bar(
        [i - width / 2 for i in x],
        [ids_a[iid]["n_steps"] for iid in common],
        width,
        label=label_a,
    )
    ax.bar(
        [i + width / 2 for i in x],
        [ids_b[iid]["n_steps"] for iid in common],
        width,
        label=label_b,
    )
    ax.set_xlabel("Task")
    ax.set_ylabel("Steps")
    ax.set_title("Steps per Task")
    ax.set_xticks(list(x))
    ax.set_xticklabels(short_ids, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    p = plots_dir / "steps_comparison.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(p)

    # 2. Token flow overlay (per common task)
    for iid in common:
        short = iid.split("__")[-1][:20]
        fig, ax = plt.subplots(figsize=(10, 4))
        flow_a = ids_a[iid].get("token_flow", [])
        flow_b = ids_b[iid].get("token_flow", [])
        if flow_a:
            ax.plot(range(len(flow_a)), flow_a, label=label_a, alpha=0.8)
        if flow_b:
            ax.plot(range(len(flow_b)), flow_b, label=label_b, alpha=0.8)
        ax.set_xlabel("Step")
        ax.set_ylabel("Prompt Tokens")
        ax.set_title(f"Context Growth: {short}")
        ax.legend()
        fig.tight_layout()
        p = plots_dir / f"token_flow_{short}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(p)

    # 3. LLM/Tool time ratio
    if common:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(
            [i - width / 2 for i in x],
            [ids_a[iid]["llm_ratio"] for iid in common],
            width,
            label=label_a,
        )
        ax.bar(
            [i + width / 2 for i in x],
            [ids_b[iid]["llm_ratio"] for iid in common],
            width,
            label=label_b,
        )
        ax.set_xlabel("Task")
        ax.set_ylabel("LLM / (LLM + Tool) time ratio")
        ax.set_title("LLM vs Tool Time Ratio")
        ax.set_xticks(list(x))
        ax.set_xticklabels(short_ids, rotation=45, ha="right")
        ax.legend()
        fig.tight_layout()
        p = plots_dir / "llm_tool_ratio.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(p)

    # 4. Tool diversity comparison
    if common:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(
            [i - width / 2 for i in x],
            [ids_a[iid]["tool_diversity"] for iid in common],
            width,
            label=label_a,
        )
        ax.bar(
            [i + width / 2 for i in x],
            [ids_b[iid]["tool_diversity"] for iid in common],
            width,
            label=label_b,
        )
        ax.set_xlabel("Task")
        ax.set_ylabel("Unique Tools Used")
        ax.set_title("Tool Diversity")
        ax.set_xticks(list(x))
        ax.set_xticklabels(short_ids, rotation=45, ha="right")
        ax.legend()
        fig.tight_layout()
        p = plots_dir / "tool_diversity.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(p)

    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two agent scaffolds on the same SWE-bench tasks.",
    )
    parser.add_argument(
        "--agent-a", required=True, help="Path to agent A results.jsonl"
    )
    parser.add_argument(
        "--agent-b", required=True, help="Path to agent B results.jsonl"
    )
    parser.add_argument(
        "--label-a", default="mini-swe-agent", help="Display label for agent A"
    )
    parser.add_argument(
        "--label-b", default="openclaw", help="Display label for agent B"
    )
    parser.add_argument(
        "--output", default="traces/analysis/comparison", help="Output directory"
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    path_a = Path(args.agent_a)
    path_b = Path(args.agent_b)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results_a = load_results(path_a)
    results_b = load_results(path_b)
    print(
        f"Loaded {len(results_a)} results from {args.label_a}, {len(results_b)} from {args.label_b}"
    )

    # Compute per-task metrics
    metrics_a = []
    for r in results_a:
        trace_file = find_trace_file(path_a, r["instance_id"])
        steps = load_trace_steps(trace_file) if trace_file else []
        metrics_a.append(compute_task_metrics(r, steps))

    metrics_b = []
    for r in results_b:
        trace_file = find_trace_file(path_b, r["instance_id"])
        steps = load_trace_steps(trace_file) if trace_file else []
        metrics_b.append(compute_task_metrics(r, steps))

    # Build comparison
    comparison = build_comparison(metrics_a, metrics_b, args.label_a, args.label_b)

    # Write JSON
    json_path = output_dir / "comparison_summary.json"
    json_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"JSON summary: {json_path}")

    # Write Markdown
    md_path = output_dir / "comparison_summary.md"
    md_path.write_text(
        render_markdown(comparison, args.label_a, args.label_b), encoding="utf-8"
    )
    print(f"Markdown report: {md_path}")

    # Generate plots
    if not args.no_plots:
        plots = generate_plots(
            metrics_a, metrics_b, args.label_a, args.label_b, output_dir
        )
        if plots:
            print(f"Plots: {len(plots)} files in {output_dir / 'plots'}/")
        else:
            print("Plots: skipped (matplotlib not available)")


if __name__ == "__main__":
    main()
