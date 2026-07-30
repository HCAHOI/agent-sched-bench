#!/usr/bin/env python3
"""Plot fixed/elastic EAR replays, optionally with an unrestricted baseline."""

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
    parser.add_argument("--free-dir", type=Path)
    return parser.parse_args()


def _memory_gib(value: str) -> float:
    match = re.match(r"([0-9.]+)([KMG]iB)", value)
    if match is None:
        raise ValueError(f"unsupported Docker memory value: {value!r}")
    amount = float(match.group(1))
    return amount * {"KiB": 1 / 1024**2, "MiB": 1 / 1024, "GiB": 1}[match.group(2)]


def _load_trace_metadata(
    path: Path,
    throughput: dict[str, Any],
) -> dict[str, Any]:
    trace_path = path / Path(throughput["trace_file"]).name
    with trace_path.open(encoding="utf-8") as fh:
        return json.loads(next(fh))


def _ear_runtime_identity(runtime: dict[str, Any]) -> dict[str, Any]:
    return {
        key: runtime.get(key)
        for key in (
            "mode",
            "policy_sha256",
            "pool",
            "ear_git_commit",
            "clock_anchor",
            "initial_lease",
            "artifacts",
        )
    }


def _validate_ear_runtime_provenance(
    path: Path,
    controller_runtime: dict[str, Any],
    throughput: dict[str, Any],
) -> None:
    trace_runtime = _load_trace_metadata(path, throughput).get("ear_runtime")
    throughput_runtime = throughput.get("ear_runtime")
    if not isinstance(trace_runtime, dict) or not isinstance(throughput_runtime, dict):
        raise TypeError(f"missing EAR runtime provenance: {path}")
    identities = [
        _ear_runtime_identity(runtime)
        for runtime in (controller_runtime, throughput_runtime, trace_runtime)
    ]
    if any(value is None for value in identities[0].values()) or not all(
        identity == identities[0] for identity in identities[1:]
    ):
        raise ValueError(f"inconsistent EAR runtime provenance: {path}")


def _load_observed_usage(path: Path, throughput: dict[str, Any]) -> dict[str, Any]:
    resource_run = throughput["container_resources"]
    summary_path = path / Path(resource_run["summary_path"]).name
    if (
        resource_run["status"] != "collected"
        or resource_run["errors"]
        or not summary_path.is_file()
    ):
        raise ValueError(f"invalid container resource monitoring artifacts: {path}")
    resource_summary = json.loads(summary_path.read_text())
    if (
        resource_summary["errors"]
        or resource_summary["dropped_error_count"]
        or not resource_summary["sampling"]["stop_complete"]
        or resource_run["sample_count"] != resource_summary["sample_count"]
        or resource_run["sampling"] != resource_summary["sampling"]
    ):
        raise ValueError(f"invalid container resource summary: {summary_path}")
    summaries = [container["summary"] for container in resource_summary["containers"]]
    if (
        resource_summary["sample_count"] <= 0
        or len(summaries) != throughput["completed_traces"]
        or sum(summary["sample_count"] for summary in summaries)
        != resource_summary["sample_count"]
        or any(
            summary["sample_count"] < 2 or summary["duration_seconds"] <= 0
            for summary in summaries
        )
    ):
        raise ValueError(f"incomplete container resource coverage: {summary_path}")
    return {
        "observed_cpu_core_seconds": sum(
            summary["cpu_percent"]["avg"] / 100 * summary["duration_seconds"]
            for summary in summaries
        ),
        "observed_memory_gib_seconds": sum(
            summary["memory_mb"]["avg"] / 1024 * summary["duration_seconds"]
            for summary in summaries
        ),
        "resource_sample_count": resource_summary["sample_count"],
        "monitored_containers": len(summaries),
        "resource_sample_interval_s": resource_summary["sampling"]["interval_s"],
    }


def _load_run(path: Path, *, observed_required: bool = False) -> dict[str, Any]:
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
    run = {
        "path": str(path),
        "wall_time_s": throughput["wall_time_s"],
        "completed_traces": throughput["completed_traces"],
        "failed_traces": throughput["failed_traces"],
        "status": runtime["status"],
        "oom_kill_count": runtime["oom_kill_count"],
        "cpu_core_seconds": controller["total_reserved_cpu_core_seconds"],
        "memory_gib_seconds": controller["total_reserved_memory_gb_seconds"],
        "series": series,
        "_attempted_traces": throughput["attempted_traces"],
        "_arm_mode": runtime["mode"],
        "_ear_runtime": runtime,
    }
    if observed_required:
        _validate_ear_runtime_provenance(path, runtime, throughput)
        run.update(_load_observed_usage(path, throughput))
    return run


def _load_free_run(path: Path) -> dict[str, Any]:
    throughput = json.loads((path / "throughput_summary.json").read_text())
    if (
        throughput.get("ear_runtime") is not None
        or (path / "controller_summary.json").exists()
        or (path / "lease_events.jsonl").exists()
        or throughput["attempted_traces"] != throughput["completed_traces"]
        or throughput["failed_traces"] != 0
        or _load_trace_metadata(path, throughput).get("ear_runtime") is not None
    ):
        raise ValueError(f"invalid unrestricted replay artifacts: {path}")
    return {
        "path": str(path),
        "wall_time_s": throughput["wall_time_s"],
        "completed_traces": throughput["completed_traces"],
        "failed_traces": throughput["failed_traces"],
        "status": "valid",
        "oom_kill_count": None,
        "cpu_core_seconds": None,
        "memory_gib_seconds": None,
        "series": [],
        "_attempted_traces": throughput["attempted_traces"],
        "_arm_mode": "free",
        **_load_observed_usage(path, throughput),
    }


def _comparison_key(path: Path) -> dict[str, Any]:
    throughput = json.loads((path / "throughput_summary.json").read_text())
    metadata = _load_trace_metadata(path, throughput)
    key = {
        key: metadata[key]
        for key in (
            "simulate_mode",
            "sandbox_backend",
            "checkpoint_backend",
            "replay_speed",
            "llm_timing_mode",
            "source_trace_entries",
            "manifest",
            "concurrency",
            "scheduler_mode",
            "network_mode",
            "workers",
            "prep_concurrency",
            "monitoring",
        )
    }
    key["llm_ttft_ms"] = metadata.get("llm_ttft_ms")
    key["llm_tpot_ms"] = metadata.get("llm_tpot_ms")
    key["manifest"] = str(Path(metadata["manifest"]).resolve())
    return key


def _validate_three_arm_comparison(
    runs: dict[str, dict[str, Any]],
    paths: dict[str, Path],
) -> None:
    expected_modes = {"fixed": "fixed", "elastic": "elastic", "free": "free"}
    if any(
        runs[label]["status"] != "valid"
        or runs[label]["_arm_mode"] != expected_modes[label]
        or runs[label]["_attempted_traces"] != runs[label]["completed_traces"]
        or runs[label]["failed_traces"] != 0
        for label in expected_modes
    ):
        raise ValueError("invalid fixed/elastic/free arm roles")
    fixed = runs["fixed"]
    elastic = runs["elastic"]
    fixed_runtime = fixed["_ear_runtime"]
    elastic_runtime = elastic["_ear_runtime"]
    if (
        not fixed_runtime["policy_sha256"]
        or fixed_runtime["policy_sha256"] != elastic_runtime["policy_sha256"]
        or not fixed_runtime["pool"]
        or fixed_runtime["pool"] != elastic_runtime["pool"]
        or not fixed_runtime["ear_git_commit"]
        or fixed_runtime["ear_git_commit"] != elastic_runtime["ear_git_commit"]
    ):
        raise ValueError("fixed/elastic EAR policy provenance does not match")
    keys = {label: _comparison_key(paths[label]) for label in expected_modes}
    if keys["elastic"] != keys["fixed"] or keys["free"] != keys["fixed"]:
        raise ValueError("fixed/elastic/free replay configurations do not match")


def main() -> None:
    args = _args()
    import matplotlib.pyplot as plt

    runs = {
        "fixed": _load_run(args.fixed_dir, observed_required=args.free_dir is not None),
        "elastic": _load_run(
            args.elastic_dir,
            observed_required=args.free_dir is not None,
        ),
    }
    if args.free_dir is not None:
        runs["free"] = _load_free_run(args.free_dir)
        _validate_three_arm_comparison(
            runs,
            {
                "fixed": args.fixed_dir,
                "elastic": args.elastic_dir,
                "free": args.free_dir,
            },
        )
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

    if args.free_dir is None:
        metrics = ("wall_time_s", "cpu_core_seconds", "memory_gib_seconds")
    else:
        metrics = (
            "wall_time_s",
            "observed_cpu_core_seconds",
            "observed_memory_gib_seconds",
        )
        tick_labels = (
            "Wall time",
            "Observed CPU core-s",
            "Observed memory GiB-s",
        )
        title = "Fixed vs elastic vs free replay"
        figure_name = "ear_fixed_elastic_free.png"
    normalized = {
        label: [run[metric] / runs["fixed"][metric] for metric in metrics]
        for label, run in runs.items()
    }
    figure, axis = plt.subplots(figsize=(8, 4.5))
    positions = range(len(metrics))
    if args.free_dir is None:
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
        tick_labels = ("Wall time", "CPU core-s", "Memory GiB-s")
        title = "EAR fixed vs elastic replay"
        figure_name = "ear_fixed_vs_elastic.png"
    else:
        width = 0.8 / len(runs)
        for index, (label, values) in enumerate(normalized.items()):
            offset = -0.4 + width / 2 + index * width
            axis.bar(
                [position + offset for position in positions],
                values,
                width,
                label=label.title(),
            )
    axis.axhline(1.0, color="black", linewidth=0.8)
    axis.set_xticks(list(positions), tick_labels)
    axis.set_ylabel("Ratio to fixed")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / figure_name, dpi=160)
    plt.close(figure)

    comparison = {
        label: {
            key: value
            for key, value in run.items()
            if key != "series" and not key.startswith("_")
        }
        for label, run in runs.items()
    }
    comparison["elastic_vs_fixed"] = {
        metric: runs["elastic"][metric] / runs["fixed"][metric]
        for metric in metrics
    }
    if "free" in runs:
        comparison["free_vs_fixed"] = {
            metric: runs["free"][metric] / runs["fixed"][metric]
            for metric in metrics
        }
    (args.output_dir / "ear_comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
