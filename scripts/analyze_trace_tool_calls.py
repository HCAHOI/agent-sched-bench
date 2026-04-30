"""Analyze tool-call timing and output-size distributions for Gantt traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="demo/gantt_viewer/configs/terminal-bench-10.yaml",
        help="Gantt discovery YAML containing trace paths.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/trace_tool_analysis",
        help="Directory for generated CSV, JSON, Markdown, and figures.",
    )
    return parser.parse_args()


def load_trace_paths(config_path: Path) -> list[Path]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    groups = raw.get("groups")
    if not isinstance(groups, list):
        raise ValueError(f"{config_path} must define a groups list")
    paths: list[Path] = []
    for group in groups:
        group_paths = group.get("paths") if isinstance(group, dict) else None
        if not isinstance(group_paths, list):
            continue
        for item in group_paths:
            if not isinstance(item, str):
                raise ValueError(f"non-string trace path in {config_path}: {item!r}")
            paths.append(Path(item))
    if not paths:
        raise ValueError(f"{config_path} did not contain any trace paths")
    return paths


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"non-object record at {path}:{line_number}")
            records.append(record)
    return records


def trace_task_id(path: Path, records: list[dict[str, Any]]) -> str:
    for record in records:
        if record.get("type") == "trace_metadata":
            instance_id = record.get("instance_id")
            if isinstance(instance_id, str) and instance_id:
                return instance_id
    return path.parent.parent.name


def output_text(data: dict[str, Any]) -> str:
    for key in ("tool_result", "result", "stdout", "stderr", "result_preview"):
        value = data.get(key)
        if isinstance(value, str):
            return value
        if value is not None:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return ""


def args_text(data: dict[str, Any]) -> str:
    for key in ("tool_args", "args", "args_preview"):
        value = data.get(key)
        if isinstance(value, str):
            return value
        if value is not None:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return ""


def extract_tool_calls(trace_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = read_jsonl(trace_path)
    task_id = trace_task_id(trace_path, records)
    rows: list[dict[str, Any]] = []
    open_starts: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []

    for record in records:
        if record.get("type") == "event" and record.get("event") == "tool_exec_start":
            open_starts.append(record)
        elif record.get("type") == "event" and record.get("event") == "tool_exec_end":
            if open_starts:
                open_starts.pop(0)
            else:
                unmatched.append(
                    {
                        "task": task_id,
                        "trace_path": str(trace_path),
                        "kind": "end_without_start",
                        "iteration": record.get("iteration"),
                        "ts": record.get("ts"),
                    }
                )

        if record.get("type") != "action" or record.get("action_type") != "tool_exec":
            continue

        data = record.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        start = float(record.get("ts_start") or 0.0)
        end = float(record.get("ts_end") or start)
        duration_s = max(0.0, end - start)
        duration_ms = data.get("duration_ms")
        if isinstance(duration_ms, int | float) and duration_ms >= 0:
            duration_s_from_payload = float(duration_ms) / 1000.0
        else:
            duration_s_from_payload = None

        result = output_text(data)
        arguments = args_text(data)
        rows.append(
            {
                "task": task_id,
                "trace_path": str(trace_path),
                "action_id": record.get("action_id"),
                "agent_id": record.get("agent_id"),
                "iteration": record.get("iteration"),
                "tool_name": data.get("tool_name") or "unknown",
                "success": data.get("success"),
                "ts_start": start,
                "ts_end": end,
                "duration_s": duration_s,
                "duration_s_from_payload": duration_s_from_payload,
                "args_chars": len(arguments),
                "args_bytes": len(arguments.encode("utf-8")),
                "output_chars": len(result),
                "output_bytes": len(result.encode("utf-8")),
                "output_lines": result.count("\n") + (1 if result else 0),
                "output_preview": result[:240].replace("\n", "\\n"),
            }
        )

    for start_event in open_starts:
        data = start_event.get("data") or {}
        unmatched.append(
            {
                "task": task_id,
                "trace_path": str(trace_path),
                "kind": "start_without_end",
                "iteration": start_event.get("iteration"),
                "tool_name": data.get("tool_name") if isinstance(data, dict) else None,
                "ts": start_event.get("ts"),
                "args_preview": data.get("args_preview") if isinstance(data, dict) else None,
            }
        )

    return rows, unmatched


def percentile(series: pd.Series, q: float) -> float:
    if series.empty:
        return 0.0
    return float(series.quantile(q))


def describe_numeric(series: pd.Series) -> dict[str, float]:
    clean = series.dropna()
    if clean.empty:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": int(clean.count()),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "p90": percentile(clean, 0.90),
        "p95": percentile(clean, 0.95),
        "max": float(clean.max()),
    }


def grouped_summary(df: pd.DataFrame, group_key: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, group in df.groupby(group_key, dropna=False):
        out[str(key)] = {
            "count": int(len(group)),
            "duration_s": describe_numeric(group["duration_s"]),
            "output_chars": describe_numeric(group["output_chars"]),
            "total_duration_s": float(group["duration_s"].sum()),
            "total_output_chars": int(group["output_chars"].sum()),
        }
    return out


def correlation_summary(df: pd.DataFrame) -> dict[str, Any]:
    if len(df) < 2:
        return {}
    result: dict[str, Any] = {
        "pearson_duration_output": float(df["duration_s"].corr(df["output_chars"], method="pearson")),
        "spearman_duration_output": float(df["duration_s"].corr(df["output_chars"], method="spearman")),
    }
    by_tool: dict[str, dict[str, float]] = {}
    for tool_name, group in df.groupby("tool_name"):
        if len(group) < 3:
            continue
        by_tool[str(tool_name)] = {
            "n": int(len(group)),
            "pearson": float(group["duration_s"].corr(group["output_chars"], method="pearson")),
            "spearman": float(group["duration_s"].corr(group["output_chars"], method="spearman")),
        }
    result["by_tool"] = by_tool
    return result


def bin_counts(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    duration_bins = pd.cut(
        df["duration_s"],
        bins=[-0.001, 0.1, 1, 10, 60, 600, float("inf")],
        labels=["<=0.1s", "0.1-1s", "1-10s", "10-60s", "60-600s", ">600s"],
    )
    output_bins = pd.cut(
        df["output_chars"],
        bins=[-1, 0, 100, 1_000, 10_000, 100_000, float("inf")],
        labels=["0", "1-100", "101-1k", "1k-10k", "10k-100k", ">100k"],
    )
    return {
        "duration_s": {str(k): int(v) for k, v in duration_bins.value_counts(sort=False).items()},
        "output_chars": {str(k): int(v) for k, v in output_bins.value_counts(sort=False).items()},
    }


def write_figures(df: pd.DataFrame, output_dir: Path) -> list[str]:
    figure_paths: list[str] = []
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    def save(name: str) -> None:
        path = figures_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        figure_paths.append(str(path))

    plt.figure(figsize=(8, 4.5))
    plt.hist(df["duration_s"].clip(lower=1e-4), bins=40, color="#2563eb")
    plt.xscale("log")
    plt.xlabel("tool duration (s, log scale)")
    plt.ylabel("call count")
    plt.title("Tool execution duration distribution")
    save("duration_distribution.png")

    plt.figure(figsize=(8, 4.5))
    plt.hist((df["output_chars"] + 1), bins=40, color="#16a34a")
    plt.xscale("log")
    plt.xlabel("output length (chars + 1, log scale)")
    plt.ylabel("call count")
    plt.title("Tool output length distribution")
    save("output_length_distribution.png")

    by_tool = df.groupby("tool_name")["duration_s"].sum().sort_values(ascending=True)
    plt.figure(figsize=(8, max(3.0, 0.35 * len(by_tool))))
    plt.barh(by_tool.index.astype(str), by_tool.values, color="#f97316")
    plt.xlabel("total tool duration (s)")
    plt.title("Total tool time by tool type")
    save("tool_time_by_type.png")

    plt.figure(figsize=(7, 5))
    for tool_name, group in df.groupby("tool_name"):
        plt.scatter(
            group["output_chars"] + 1,
            group["duration_s"].clip(lower=1e-4),
            label=str(tool_name),
            alpha=0.65,
            s=18,
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("output length (chars + 1, log scale)")
    plt.ylabel("duration (s, log scale)")
    plt.title("Tool duration vs output length")
    plt.legend(fontsize=7, loc="best")
    save("duration_vs_output_by_tool.png")

    return figure_paths


def markdown_report(summary: dict[str, Any], df: pd.DataFrame) -> str:
    top_duration = df.sort_values("duration_s", ascending=False).head(10)
    top_output = df.sort_values("output_chars", ascending=False).head(10)
    tool_time = (
        df.groupby("tool_name")["duration_s"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    total_time = float(df["duration_s"].sum())
    lines = [
        "# Trace Tool Analysis",
        "",
        f"- Traces analyzed: {summary['n_traces']}",
        f"- Completed tool calls: {summary['n_tool_calls']}",
        f"- Incomplete tool events: {summary['n_incomplete_tool_events']}",
        f"- Total tool duration: {total_time:.2f}s",
        f"- Median duration: {summary['overall']['duration_s']['median']:.3f}s",
        f"- P95 duration: {summary['overall']['duration_s']['p95']:.3f}s",
        f"- Median output length: {summary['overall']['output_chars']['median']:.0f} chars",
        f"- P95 output length: {summary['overall']['output_chars']['p95']:.0f} chars",
        "",
        "## Tool Time Share",
        "",
        "| tool | calls | total_s | share | median_s | p95_s | median_output_chars |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    by_tool = summary["by_tool"]
    for _, row in tool_time.iterrows():
        tool = str(row["tool_name"])
        item = by_tool[tool]
        share = (item["total_duration_s"] / total_time * 100) if total_time else 0.0
        lines.append(
            "| "
            f"{tool} | {item['count']} | {item['total_duration_s']:.2f} | "
            f"{share:.1f}% | {item['duration_s']['median']:.3f} | "
            f"{item['duration_s']['p95']:.3f} | {item['output_chars']['median']:.0f} |"
        )

    lines.extend(
        [
            "",
            "## Longest Tool Calls",
            "",
            "| task | tool | iter | duration_s | output_chars |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in top_duration.iterrows():
        lines.append(
            f"| {row['task']} | {row['tool_name']} | {row['iteration']} | "
            f"{row['duration_s']:.2f} | {int(row['output_chars'])} |"
        )

    lines.extend(
        [
            "",
            "## Largest Tool Outputs",
            "",
            "| task | tool | iter | output_chars | duration_s |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in top_output.iterrows():
        lines.append(
            f"| {row['task']} | {row['tool_name']} | {row['iteration']} | "
            f"{int(row['output_chars'])} | {row['duration_s']:.2f} |"
        )

    corr = summary["correlation"]
    lines.extend(
        [
            "",
            "## Duration / Output Correlation",
            "",
            f"- Pearson(duration, output_chars): {corr.get('pearson_duration_output', 0.0):.3f}",
            f"- Spearman(duration, output_chars): {corr.get('spearman_duration_output', 0.0):.3f}",
            "",
            "## Notes",
            "",
            "- Durations are completed `tool_exec` action spans only.",
            "- Unclosed `tool_exec_start` events are reported separately and are not imputed.",
            "- Output length is measured from recorded tool result text when present.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    repo = Path.cwd()
    config_path = (repo / args.config).resolve()
    output_dir = (repo / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_paths = [(repo / path).resolve() for path in load_trace_paths(config_path)]
    missing = [path for path in trace_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing trace files: {missing}")

    rows: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    for trace_path in trace_paths:
        trace_rows, trace_incomplete = extract_tool_calls(trace_path)
        rows.extend(trace_rows)
        incomplete.extend(trace_incomplete)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("no completed tool_exec actions found")
    incomplete_df = pd.DataFrame(incomplete)

    df.to_csv(output_dir / "tool_calls.csv", index=False)
    incomplete_df.to_csv(output_dir / "incomplete_tool_events.csv", index=False)

    summary: dict[str, Any] = {
        "config": str(config_path.relative_to(repo)),
        "n_traces": len(trace_paths),
        "trace_paths": [str(path.relative_to(repo)) for path in trace_paths],
        "n_tool_calls": int(len(df)),
        "n_incomplete_tool_events": int(len(incomplete_df)),
        "overall": {
            "duration_s": describe_numeric(df["duration_s"]),
            "output_chars": describe_numeric(df["output_chars"]),
            "total_duration_s": float(df["duration_s"].sum()),
            "total_output_chars": int(df["output_chars"].sum()),
        },
        "by_tool": grouped_summary(df, "tool_name"),
        "by_task": grouped_summary(df, "task"),
        "bins": bin_counts(df),
        "correlation": correlation_summary(df),
    }
    summary["figures"] = write_figures(df, output_dir)

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(markdown_report(summary, df), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
