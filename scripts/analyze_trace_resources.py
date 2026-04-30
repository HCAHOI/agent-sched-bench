"""Classify CPU and memory consumption patterns for configured traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
        default="results/trace_resource_analysis",
        help="Output directory for metrics and Markdown report.",
    )
    return parser.parse_args()


def trace_paths_from_config(config_path: Path) -> list[Path]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    groups = raw.get("groups")
    if not isinstance(groups, list):
        raise ValueError(f"{config_path} must define a groups list")
    paths: list[Path] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        group_paths = group.get("paths")
        if not isinstance(group_paths, list):
            continue
        paths.extend(Path(item) for item in group_paths if isinstance(item, str))
    if not paths:
        raise ValueError(f"{config_path} did not define any trace paths")
    return paths


def parse_percent(value: Any) -> float:
    if isinstance(value, str):
        return float(value.rstrip("% "))
    if value is None:
        return 0.0
    return float(value)


def parse_mem_mb(sample: dict[str, Any]) -> float:
    value = sample.get("memory_mb")
    if value is not None:
        return float(value)
    raw = sample.get("mem_usage")
    if not isinstance(raw, str):
        return 0.0
    left = raw.split("/", maxsplit=1)[0].strip()
    if not left:
        return 0.0
    number = "".join(ch for ch in left if ch.isdigit() or ch == ".")
    unit = left[len(number):].strip()
    if not number:
        return 0.0
    multipliers = {
        "B": 1 / 1_000_000,
        "kB": 1 / 1_000,
        "KB": 1 / 1_000,
        "KiB": 1 / 1024,
        "MB": 1.0,
        "MiB": 1.0,
        "GB": 1_000.0,
        "GiB": 1024.0,
    }
    return float(number) * multipliers.get(unit, 1.0)


def load_task_id(trace_path: Path) -> str:
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record, dict) and record.get("type") == "trace_metadata":
                instance_id = record.get("instance_id")
                if isinstance(instance_id, str) and instance_id:
                    return instance_id
            break
    return trace_path.parent.parent.name


def weighted_mean(values: pd.Series, epochs: pd.Series) -> float:
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return float(values.iloc[0])
    deltas = epochs.shift(-1) - epochs
    median_delta = float(deltas.dropna().median()) if not deltas.dropna().empty else 1.0
    deltas.iloc[-1] = median_delta
    total = float(deltas.sum())
    if total <= 0:
        return float(values.mean())
    return float((values * deltas).sum() / total)


def classify_cpu(row: dict[str, Any]) -> str:
    avg = row["cpu_avg"]
    p50 = row["cpu_p50"]
    max_cpu = row["cpu_max"]
    frac_ge_80 = row["cpu_frac_ge_80"]
    frac_ge_200 = row["cpu_frac_ge_200"]
    if avg >= 150 or frac_ge_200 >= 0.25:
        return "sustained multi-core compute"
    if avg >= 60 and p50 >= 60 and frac_ge_80 >= 0.45:
        return "sustained single-core compute"
    if max_cpu >= 120 and avg < 60:
        return "bursty compute"
    if avg < 15 and max_cpu < 80:
        return "mostly idle/control"
    return "mixed low-to-moderate CPU"


def classify_memory(row: dict[str, Any]) -> str:
    avg = row["mem_avg_mb"]
    max_mem = row["mem_max_mb"]
    span = row["mem_range_mb"]
    growth = row["mem_nonzero_growth_mb"]
    if max_mem >= 700 or avg >= 400:
        return "high memory resident set"
    if growth >= 150 and span >= 200:
        return "memory ramp-up / cache accumulation"
    if avg >= 150 and span < 120:
        return "moderate stable memory"
    if avg < 100 and span < 120:
        return "low stable memory"
    return "variable memory"


def combined_pattern(cpu_pattern: str, mem_pattern: str) -> str:
    if cpu_pattern == "sustained multi-core compute":
        return "compute-dominated"
    if cpu_pattern == "sustained single-core compute":
        return "single-core compute"
    if cpu_pattern == "bursty compute":
        return "bursty setup/test"
    if cpu_pattern == "mostly idle/control" and mem_pattern in {
        "low stable memory",
        "moderate stable memory",
    }:
        return "control-plane / waiting"
    if mem_pattern in {"high memory resident set", "memory ramp-up / cache accumulation"}:
        return "memory-resident mixed workload"
    return "light mixed workload"


def summarize_trace(repo: Path, trace_path: Path) -> dict[str, Any]:
    resource_path = trace_path.parent / "resources.json"
    if not resource_path.is_file():
        raise FileNotFoundError(f"missing resources.json beside {trace_path}")
    data = json.loads(resource_path.read_text(encoding="utf-8"))
    samples = data.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{resource_path} has no resource samples")

    frame = pd.DataFrame(
        {
            "epoch": [float(sample["epoch"]) for sample in samples],
            "cpu": [parse_percent(sample.get("cpu_percent")) for sample in samples],
            "mem": [parse_mem_mb(sample) for sample in samples],
        }
    ).sort_values("epoch")

    nonzero_mem = frame.loc[frame["mem"] > 0, "mem"]
    if nonzero_mem.empty:
        first_nonzero = 0.0
        last_nonzero = 0.0
    else:
        first_nonzero = float(nonzero_mem.iloc[0])
        last_nonzero = float(nonzero_mem.iloc[-1])

    duration_s = float(frame["epoch"].iloc[-1] - frame["epoch"].iloc[0])
    row: dict[str, Any] = {
        "task": load_task_id(trace_path),
        "trace_path": str(trace_path.relative_to(repo)),
        "resource_path": str(resource_path.relative_to(repo)),
        "samples": int(len(frame)),
        "duration_s": duration_s,
        "cpu_avg": weighted_mean(frame["cpu"], frame["epoch"]),
        "cpu_p50": float(frame["cpu"].quantile(0.50)),
        "cpu_p90": float(frame["cpu"].quantile(0.90)),
        "cpu_p95": float(frame["cpu"].quantile(0.95)),
        "cpu_max": float(frame["cpu"].max()),
        "cpu_std": float(frame["cpu"].std(ddof=0)),
        "cpu_frac_ge_80": float((frame["cpu"] >= 80).mean()),
        "cpu_frac_ge_150": float((frame["cpu"] >= 150).mean()),
        "cpu_frac_ge_200": float((frame["cpu"] >= 200).mean()),
        "cpu_frac_le_5": float((frame["cpu"] <= 5).mean()),
        "core_seconds_est": 0.0,
        "mem_avg_mb": weighted_mean(frame["mem"], frame["epoch"]),
        "mem_p50_mb": float(frame["mem"].quantile(0.50)),
        "mem_p90_mb": float(frame["mem"].quantile(0.90)),
        "mem_p95_mb": float(frame["mem"].quantile(0.95)),
        "mem_max_mb": float(frame["mem"].max()),
        "mem_min_nonzero_mb": float(nonzero_mem.min()) if not nonzero_mem.empty else 0.0,
        "mem_first_nonzero_mb": first_nonzero,
        "mem_last_nonzero_mb": last_nonzero,
        "mem_nonzero_growth_mb": last_nonzero - first_nonzero,
        "mem_range_mb": float(frame["mem"].max() - frame["mem"].min()),
        "mem_std_mb": float(frame["mem"].std(ddof=0)),
    }
    row["core_seconds_est"] = row["cpu_avg"] / 100.0 * duration_s
    row["cpu_pattern"] = classify_cpu(row)
    row["memory_pattern"] = classify_memory(row)
    row["combined_pattern"] = combined_pattern(row["cpu_pattern"], row["memory_pattern"])
    return row


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def write_markdown(rows: pd.DataFrame, output_path: Path) -> None:
    total_core_seconds = float(rows["core_seconds_est"].sum())
    lines = [
        "# Resource Consumption Patterns",
        "",
        "Scope: 10 Terminal-Bench GLM-5.1/OpenClaw traces from `demo/gantt_viewer/configs/terminal-bench-10.yaml`.",
        "",
        "## Classification Rules",
        "",
        "- CPU percentages use Docker-style semantics: `100%` is approximately one fully utilized CPU core.",
        "- `sustained multi-core compute`: average CPU >= 150%, or at least 25% of samples >= 200%.",
        "- `sustained single-core compute`: average CPU >= 60%, median CPU >= 60%, and at least 45% of samples >= 80%.",
        "- `bursty compute`: max CPU >= 120% but average CPU < 60%.",
        "- `mostly idle/control`: average CPU < 15% and max CPU < 80%.",
        "- Memory classes use sampled container resident memory: high resident set, ramp-up/cache accumulation, moderate stable, low stable, or variable.",
        "- Memory bandwidth counters are not used because these traces report `memory_bandwidth_available=false`.",
        "",
        "## Aggregate View",
        "",
        f"- Resource traces analyzed: `{len(rows)}`",
        f"- Estimated total CPU core-seconds: `{total_core_seconds:.1f}`",
        f"- Highest average CPU: `{rows.loc[rows['cpu_avg'].idxmax(), 'task']}` at `{rows['cpu_avg'].max():.1f}%`",
        f"- Highest peak CPU: `{rows.loc[rows['cpu_max'].idxmax(), 'task']}` at `{rows['cpu_max'].max():.1f}%`",
        f"- Highest average memory: `{rows.loc[rows['mem_avg_mb'].idxmax(), 'task']}` at `{rows['mem_avg_mb'].max():.1f} MB`",
        f"- Highest peak memory: `{rows.loc[rows['mem_max_mb'].idxmax(), 'task']}` at `{rows['mem_max_mb'].max():.1f} MB`",
        "",
        "## Per-Task Classification",
        "",
        "| task | combined pattern | CPU pattern | memory pattern | avg CPU | max CPU | >=80% CPU samples | avg mem MB | max mem MB | mem growth MB |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in rows.sort_values("core_seconds_est", ascending=False).iterrows():
        lines.append(
            "| "
            f"{row['task']} | {row['combined_pattern']} | {row['cpu_pattern']} | "
            f"{row['memory_pattern']} | {row['cpu_avg']:.1f}% | {row['cpu_max']:.1f}% | "
            f"{format_pct(row['cpu_frac_ge_80'])} | {row['mem_avg_mb']:.1f} | "
            f"{row['mem_max_mb']:.1f} | {row['mem_nonzero_growth_mb']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Pattern Groups",
            "",
        ]
    )
    for pattern, group in rows.groupby("combined_pattern"):
        tasks = ", ".join(group.sort_values("task")["task"].tolist())
        core_seconds = float(group["core_seconds_est"].sum())
        lines.append(
            f"- **{pattern}**: {len(group)} task(s), `{core_seconds:.1f}` estimated core-seconds. Tasks: {tasks}."
        )

    lines.extend(
        [
            "",
            "## Observations",
            "",
            "- CPU time is highly concentrated in a small number of long-running workloads; this matches the tool-time analysis where long `exec` calls dominate.",
            "- Memory does not track CPU one-to-one. Some tasks have large outputs with tiny resource cost, while training/benchmark tasks keep resident memory high for long windows.",
            "- `cartpole-rl-training` is the clearest sustained multi-core case: high average CPU and high resident memory from repeated PyTorch/Gym training runs.",
            "- `predicate-pushdown-bench` and `causal-inference-r` are compute-heavy but have different memory profiles; predicate pushdown accumulates a larger resident set from Spark/SBT state.",
            "- Several tasks are control-plane dominated: many quick file/shell operations, low CPU, low or moderate memory, and little sustained compute.",
            "",
            "## Caveats",
            "",
            "- CPU/memory samples are container-level samples from `resources.json`, not per-process attribution.",
            "- Unclosed agent/tool terminations can leave trailing idle or cleanup samples; classifications use all recorded samples and should be read as workload-run patterns, not only successful-solution patterns.",
            "- Memory bandwidth is unavailable on these traces (`pmu_unsupported`), so this analysis only covers CPU percentage and resident memory.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo = Path.cwd()
    output_dir = (repo / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_paths = [(repo / path).resolve() for path in trace_paths_from_config(repo / args.config)]
    rows = pd.DataFrame([summarize_trace(repo, path) for path in trace_paths])
    rows.to_csv(output_dir / "resource_metrics.csv", index=False)
    (output_dir / "resource_patterns.json").write_text(
        rows.to_json(orient="records", indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(rows, output_dir / "resource_patterns.md")
    print(rows.sort_values("core_seconds_est", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
