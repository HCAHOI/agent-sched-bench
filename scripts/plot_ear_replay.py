#!/usr/bin/env python3
"""Plot one fixed/elastic EAR replay pair from their run artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixed_dir", type=Path)
    parser.add_argument("elastic_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def _memory_gib(value: str) -> float:
    match = re.match(r"([0-9.]+)([KMG]iB)", value)
    if match is None:
        raise ValueError(f"unsupported Docker memory value: {value!r}")
    amount = float(match.group(1))
    return amount * {"KiB": 1 / 1024**2, "MiB": 1 / 1024, "GiB": 1}[match.group(2)]


def _load_run(path: Path) -> dict[str, Any]:
    throughput = json.loads((path / "throughput_summary.json").read_text())
    controller = json.loads((path / "controller_summary.json").read_text())
    runtime = controller["ear_runtime"]
    events = [
        json.loads(line)
        for line in (path / "lease_events.jsonl").read_text().splitlines()
    ]
    anchor = runtime["clock_anchor"]
    acquired_events = [
        event for event in events if event["event_type"] == "acquired"
    ]
    start = min(
        (float(event["timestamp"]) for event in acquired_events),
        default=float(anchor["monotonic_s"]),
    )
    resources_by_agent = {
        resources_path.parents[1].name: json.loads(resources_path.read_text())["samples"]
        for resources_path in path.glob("*/attempt_*/resources.json")
    }
    events_by_lease: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if event.get("lease_id"):
            events_by_lease.setdefault(str(event["lease_id"]), []).append(event)
    series = []
    for lease_id, lease_events in events_by_lease.items():
        acquired = next(
            (
                event
                for event in lease_events
                if event["event_type"] == "acquired"
            ),
            None,
        )
        if acquired is None:
            continue
        agent_id = str(acquired["agent_id"])
        cpu_steps = [
            (
                float(event["timestamp"]) - start,
                0.0
                if event["event_type"] == "released"
                else float(event["cpu_cores"]),
            )
            for event in lease_events
            if event["event_type"] in {"acquired", "cpu_resized", "released"}
        ]
        memory_steps = [
            (
                float(event["timestamp"]) - start,
                0.0
                if event["event_type"] == "released"
                else float(event["memory_gb"]),
            )
            for event in lease_events
            if event["event_type"] in {"acquired", "memory_resized", "released"}
        ]
        observed = []
        for sample in resources_by_agent.get(agent_id, []):
            monotonic = (
                float(anchor["monotonic_s"])
                + float(sample["epoch"])
                - float(anchor["epoch_s"])
            )
            observed.append(
                {
                    "elapsed_s": monotonic - start,
                    "cpu_cores": float(sample["cpu_percent"].rstrip("%")) / 100.0,
                    "memory_gib": _memory_gib(
                        sample["mem_usage"].split("/", 1)[0].strip()
                    ),
                }
            )
        series.append(
            {
                "lease_id": lease_id,
                "agent_id": agent_id,
                "cpu_steps": cpu_steps,
                "memory_steps": memory_steps,
                "observed": observed,
            }
        )
    return {
        "path": str(path),
        "wall_time_s": throughput["wall_time_s"],
        "completed_traces": throughput["completed_traces"],
        "failed_traces": throughput["failed_traces"],
        "status": runtime["status"],
        "oom_kill_count": runtime["oom_kill_count"],
        "cpu_core_seconds": controller["total_reserved_cpu_core_seconds"],
        "memory_gib_seconds": controller["total_reserved_memory_gb_seconds"],
        "series": series,
    }


def main() -> None:
    args = _args()
    import matplotlib.pyplot as plt

    runs = {
        "fixed": _load_run(args.fixed_dir),
        "elastic": _load_run(args.elastic_dir),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for label, run in runs.items():
        for series in run["series"]:
            series_label = f"{label} {series['agent_id']}"
            axes[0].step(
                *zip(*series["cpu_steps"], strict=True),
                where="post",
                label=f"{series_label} limit",
            )
            axes[0].plot(
                [sample["elapsed_s"] for sample in series["observed"]],
                [sample["cpu_cores"] for sample in series["observed"]],
                alpha=0.55,
                label=f"{series_label} observed",
            )
            axes[1].step(
                *zip(*series["memory_steps"], strict=True),
                where="post",
                label=f"{series_label} limit",
            )
            axes[1].plot(
                [sample["elapsed_s"] for sample in series["observed"]],
                [sample["memory_gib"] for sample in series["observed"]],
                alpha=0.55,
                label=f"{series_label} observed",
            )
    axes[0].set_ylabel("CPU cores")
    axes[1].set_ylabel("Memory GiB")
    axes[1].set_xlabel("Seconds since lease acquisition")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(ncol=2)
    figure.suptitle("EAR replay resource timeline")
    figure.tight_layout()
    figure.savefig(args.output_dir / "ear_timeline.png", dpi=160)
    plt.close(figure)

    metrics = ("wall_time_s", "cpu_core_seconds", "memory_gib_seconds")
    normalized = {
        label: [run[metric] / runs["fixed"][metric] for metric in metrics]
        for label, run in runs.items()
    }
    figure, axis = plt.subplots(figsize=(8, 4.5))
    positions = range(len(metrics))
    width = 0.36
    axis.bar(
        [position - width / 2 for position in positions],
        normalized["fixed"],
        width,
        label="fixed",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        normalized["elastic"],
        width,
        label="elastic",
    )
    axis.axhline(1.0, color="black", linewidth=0.8)
    axis.set_xticks(list(positions), ["Wall time", "CPU core-s", "Memory GiB-s"])
    axis.set_ylabel("Ratio to fixed")
    axis.set_title("EAR fixed vs elastic replay")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "ear_fixed_vs_elastic.png", dpi=160)
    plt.close(figure)

    comparison = {
        label: {
            key: value
            for key, value in run.items()
            if key != "series"
        }
        for label, run in runs.items()
    }
    comparison["elastic_vs_fixed"] = {
        metric: runs["elastic"][metric] / runs["fixed"][metric]
        for metric in metrics
    }
    (args.output_dir / "ear_comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
