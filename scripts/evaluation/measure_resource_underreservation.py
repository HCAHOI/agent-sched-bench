#!/usr/bin/env python3
"""Measure the physical cost of CPU and memory under-reservation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import tempfile
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = (
    ROOT
    / "analysis/results/tool-resource-5-3-3-3-20260804"
    / "resource-underreservation-calibration-v1"
)
COMPLETION_OUTPUT = (
    ROOT
    / "analysis/results/tool-resource-5-3-3-3-20260804"
    / "resource-underreservation-memory-completion-v1"
)
WORKSPACE = Path("/home/chiyu/ear-workspaces/mixed-burst-real/main")
WORKLOAD = Path(
    "/home/chiyu/workspace/elastic-agent-runtime/experiments/analysis/"
    "elastic-memory-remote-20260729/workloads/code_index_burst.py"
)
IMAGE = "python:3.13-slim"
MEMORY_MAX_BYTES = 6 * 1024**3
MEMORY_HIGH_BYTES = 2 * 1024**3
TIMEOUT_S = 180.0
COMPLETION_TIMEOUT_S = 3_600.0
POLL_S = 0.1
SEED = 42
BLOCKS = 3
ARMS = {
    "baseline": (8.0, "max"),
    "cpu4": (4.0, "max"),
    "cpu2": (2.0, "max"),
    "memory_high_2g": (8.0, str(MEMORY_HIGH_BYTES)),
}


def _command(args: list[str], *, timeout: float = 120.0) -> str:
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(args)}\n"
            f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}"
        )
    return result.stdout.strip()


def _read_int_map(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                values[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return values


def _delta(after: Mapping[str, int], before: Mapping[str, int]) -> dict[str, int]:
    return {
        key: value - before.get(key, 0)
        for key, value in sorted(after.items())
    }


def _cpu_quota_cores(raw: str) -> float:
    parts = raw.split()
    if len(parts) != 2 or parts[0] == "max":
        raise ValueError(f"finite cpu.max expected, got {raw!r}")
    quota, period = (int(value) for value in parts)
    if quota <= 0 or period <= 0:
        raise ValueError(f"positive cpu.max expected, got {raw!r}")
    return quota / period


def _cgroup_path(container: str) -> Path:
    pid = int(_command(["docker", "inspect", "--format", "{{.State.Pid}}", container]))
    for line in Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8").splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
            path = Path("/sys/fs/cgroup") / parts[2].lstrip("/")
            if path.is_dir():
                return path
    raise RuntimeError(f"unified cgroup not found for container {container}")


def _set_memory_high(cgroup: Path, value: str) -> None:
    result = subprocess.run(
        ["sudo", "-n", "tee", str(cgroup / "memory.high")],
        input=f"{value}\n",
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"failed to set memory.high: {result.stderr[-1000:]}")


def _snapshot(cgroup: Path) -> dict[str, Any]:
    return {
        "cpu": _read_int_map(cgroup / "cpu.stat"),
        "memory_events": _read_int_map(cgroup / "memory.events"),
        "memory_stat": _read_int_map(cgroup / "memory.stat"),
        "memory_current_bytes": int((cgroup / "memory.current").read_text()),
    }


def _workload_summary(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "elapsed_s" in value:
            return value
    return None


def _run_order() -> list[tuple[int, str]]:
    rng = random.Random(SEED)
    order: list[tuple[int, str]] = []
    for block in range(1, BLOCKS + 1):
        arms = list(ARMS)
        rng.shuffle(arms)
        order.extend((block, arm) for arm in arms)
    return order


def _run_one(
    *, index: int, block: int, arm: str, timeout_s: float = TIMEOUT_S
) -> dict[str, Any]:
    cpu_cores, memory_high = ARMS[arm]
    name = f"asb-underreserve-{os.getpid()}-{index:02d}"
    wrapper = (
        "while [ ! -e /asb-state/start ]; do sleep 0.02; done; "
        "python /workload.py --workers 8 --bursts 1 --hold-s 1 --idle-s 0 "
        ">/asb-state/stdout 2>/asb-state/stderr; rc=$?; "
        "printf '%s\\n' \"$rc\" >/asb-state/status.tmp; "
        "mv /asb-state/status.tmp /asb-state/status; "
        "while [ ! -e /asb-state/release ]; do sleep 0.02; done; exit \"$rc\""
    )
    with tempfile.TemporaryDirectory(prefix="asb-underreserve-") as state_raw:
        state = Path(state_raw)
        _command(
            [
                "docker",
                "create",
                "--name",
                name,
                "--network",
                "none",
                "--cpus",
                str(cpu_cores),
                "--memory",
                str(MEMORY_MAX_BYTES),
                "--memory-swap",
                str(MEMORY_MAX_BYTES),
                "--oom-score-adj",
                "-1000",
                "-v",
                f"{WORKSPACE}:/workspace:ro",
                "-v",
                f"{WORKLOAD}:/workload.py:ro",
                "-v",
                f"{state}:/asb-state",
                "-w",
                "/workspace",
                IMAGE,
                "/bin/sh",
                "-c",
                wrapper,
            ]
        )
        try:
            _command(["docker", "start", name])
            cgroup = _cgroup_path(name)
            _set_memory_high(cgroup, memory_high)
            observed_cpu_max = (cgroup / "cpu.max").read_text().strip()
            observed_memory_high = (cgroup / "memory.high").read_text().strip()
            observed_memory_max = int((cgroup / "memory.max").read_text())
            observed_memory_swap_max = int(
                (cgroup / "memory.swap.max").read_text()
            )
            observed_cpu_cores = _cpu_quota_cores(observed_cpu_max)
            if observed_cpu_cores != cpu_cores:
                raise RuntimeError("cpu.max differs from requested arm")
            if observed_memory_high != memory_high:
                raise RuntimeError("memory.high differs from requested arm")
            if observed_memory_max != MEMORY_MAX_BYTES:
                raise RuntimeError("memory.max differs from frozen protocol")
            if observed_memory_swap_max != 0:
                raise RuntimeError("memory.swap.max differs from frozen protocol")

            before = _snapshot(cgroup)
            peak_current = before["memory_current_bytes"]
            started = time.perf_counter()
            (state / "start").touch()
            timed_out = False
            while not (state / "status").exists():
                sample = _snapshot(cgroup)
                peak_current = max(peak_current, sample["memory_current_bytes"])
                if time.perf_counter() - started >= timeout_s:
                    timed_out = True
                    break
                time.sleep(POLL_S)
            wall_s = time.perf_counter() - started
            after = _snapshot(cgroup)
            peak_current = max(peak_current, after["memory_current_bytes"])
            stdout = (state / "stdout").read_text(errors="replace") if (state / "stdout").exists() else ""
            stderr = (state / "stderr").read_text(errors="replace") if (state / "stderr").exists() else ""
            workload_exit = None if timed_out else int((state / "status").read_text())
            summary = _workload_summary(stdout)

            if timed_out:
                _command(["docker", "kill", name])
                container_exit = 137
            else:
                (state / "release").touch()
                container_exit = int(_command(["docker", "wait", name]))
            return {
                "index": index,
                "block": block,
                "arm": arm,
                "requested_cpu_cores": cpu_cores,
                "requested_memory_high": memory_high,
                "timeout_s": timeout_s,
                "observed_cpu_max": observed_cpu_max,
                "observed_cpu_cores": observed_cpu_cores,
                "observed_memory_high": observed_memory_high,
                "observed_memory_max_bytes": observed_memory_max,
                "observed_memory_swap_max_bytes": observed_memory_swap_max,
                "timed_out": timed_out,
                "workload_exit": workload_exit,
                "container_exit": container_exit,
                "wall_s": wall_s,
                "workload": summary,
                "cpu_delta": _delta(after["cpu"], before["cpu"]),
                "memory_events_delta": _delta(
                    after["memory_events"], before["memory_events"]
                ),
                "memory_stat_delta": _delta(
                    after["memory_stat"], before["memory_stat"]
                ),
                "sampled_peak_memory_current_bytes": peak_current,
                "stderr_tail": stderr[-2000:],
            }
        finally:
            _command(["docker", "rm", "-f", name], timeout=30)


def _elapsed(rows: list[dict[str, Any]], arm: str) -> list[float]:
    return [
        float(row["workload"]["elapsed_s"])
        for row in rows
        if row["arm"] == arm
        and not row["timed_out"]
        and row["workload_exit"] == 0
        and row["container_exit"] == 0
        and row["workload"] is not None
    ]


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm = {arm: [row for row in rows if row["arm"] == arm] for arm in ARMS}
    complete = all(len(by_arm[arm]) == BLOCKS for arm in ARMS)
    identities = {
        (
            row["workload"]["source_files"],
            row["workload"]["bursts"][0]["files"],
            row["workload"]["bursts"][0]["ast_nodes"],
        )
        for row in rows
        if not row["timed_out"]
        and row["workload_exit"] == 0
        and row["container_exit"] == 0
        and row["workload"] is not None
    }
    identical_successful_output = len(identities) == 1
    elapsed = {arm: _elapsed(rows, arm) for arm in ARMS}
    medians = {
        arm: statistics.median(values) if values else None
        for arm, values in elapsed.items()
    }
    baseline = medians["baseline"]
    ratios = {
        arm: (medians[arm] / baseline if medians[arm] is not None and baseline else None)
        for arm in ARMS
    }
    successful_cpu = all(
        not row["timed_out"]
        and row["workload_exit"] == 0
        and row["container_exit"] == 0
        and row["workload"]
        for arm in ("baseline", "cpu4", "cpu2")
        for row in by_arm[arm]
    )
    baseline_throttle = {row["block"]: row["cpu_delta"].get("throttled_usec", 0) for row in by_arm["baseline"]}
    paired_throttle_higher = all(
        row["cpu_delta"].get("throttled_usec", 0) > baseline_throttle[row["block"]]
        for row in by_arm["cpu2"]
    )
    cpu_gate = bool(
        complete
        and successful_cpu
        and identical_successful_output
        and ratios["cpu2"] is not None
        and ratios["cpu2"] >= 1.25
        and paired_throttle_higher
    )

    baseline_success = all(
        not row["timed_out"]
        and row["workload_exit"] == 0
        and row["container_exit"] == 0
        and row["workload"]
        for row in by_arm["baseline"]
    )
    memory_rows = by_arm["memory_high_2g"]
    memory_high_every_run = all(
        row["memory_events_delta"].get("high", 0) > 0 for row in memory_rows
    )
    no_memory_oom = all(
        row["memory_events_delta"].get("oom_kill", 0) == 0 for row in memory_rows
    )
    memory_timeouts = sum(bool(row["timed_out"]) for row in memory_rows)
    memory_slow = ratios["memory_high_2g"] is not None and ratios["memory_high_2g"] >= 1.10
    memory_gate = bool(
        complete
        and baseline_success
        and identical_successful_output
        and memory_high_every_run
        and no_memory_oom
        and (memory_slow or memory_timeouts >= 2)
    )

    return {
        "complete": complete,
        "elapsed_median_s": medians,
        "elapsed_ratio_to_baseline": ratios,
        "cpu_gate": cpu_gate,
        "cpu_all_runs_successful": successful_cpu,
        "cpu2_paired_throttled_usec_higher": paired_throttle_higher,
        "memory_gate": memory_gate,
        "memory_high_every_run": memory_high_every_run,
        "memory_no_oom_kill": no_memory_oom,
        "memory_timeouts": memory_timeouts,
        "successful_output_identities": [list(value) for value in sorted(identities)],
        "identical_successful_output": identical_successful_output,
        "actionability_detected": cpu_gate or memory_gate,
    }


def _host_before() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "docker_version": _command(
            ["docker", "version", "--format", "{{.Server.Version}}"]
        ),
        "loadavg": list(os.getloadavg()),
    }


def _completion_order() -> list[tuple[int, str]]:
    rng = random.Random(SEED)
    order: list[tuple[int, str]] = []
    for block in range(1, BLOCKS + 1):
        arms = ["baseline", "memory_high_2g"]
        rng.shuffle(arms)
        order.extend((block, arm) for arm in arms)
    return order


def _completion_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_elapsed = _elapsed(rows, "baseline")
    memory_elapsed = _elapsed(rows, "memory_high_2g")
    memory_rows = [row for row in rows if row["arm"] == "memory_high_2g"]
    identities = {
        (
            row["workload"]["source_files"],
            row["workload"]["bursts"][0]["files"],
            row["workload"]["bursts"][0]["ast_nodes"],
        )
        for row in rows
        if not row["timed_out"]
        and row["workload_exit"] == 0
        and row["container_exit"] == 0
        and row["workload"] is not None
    }
    valid = bool(
        len(rows) == 2 * BLOCKS
        and len(baseline_elapsed) == BLOCKS
        and len(memory_rows) == BLOCKS
        and len(identities) == 1
        and all(row["memory_events_delta"].get("oom_kill", 0) == 0 for row in rows)
        and all(
            row["memory_events_delta"].get("high", 0) > 0
            for row in memory_rows
        )
    )
    timeouts = sum(bool(row["timed_out"]) for row in memory_rows)
    if valid and len(memory_elapsed) == BLOCKS:
        characterization = "completed"
    elif valid and timeouts >= 2:
        characterization = "greater_than_3600s"
    else:
        characterization = "invalid"
    baseline_median = statistics.median(baseline_elapsed) if baseline_elapsed else None
    memory_median = (
        statistics.median(memory_elapsed)
        if characterization == "completed"
        else None
    )
    return {
        "valid": valid,
        "characterization": characterization,
        "baseline_median_s": baseline_median,
        "memory_completed_runs": len(memory_elapsed),
        "memory_timeouts": timeouts,
        "memory_successful_median_s": memory_median,
        "memory_successful_ratio_to_baseline": (
            memory_median / baseline_median
            if characterization == "completed"
            and memory_median is not None
            and baseline_median
            else None
        ),
        "successful_output_identities": [list(value) for value in sorted(identities)],
    }


def _run_memory_completion() -> None:
    if COMPLETION_OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {COMPLETION_OUTPUT}")
    COMPLETION_OUTPUT.mkdir(parents=True)
    partial = COMPLETION_OUTPUT / "partial.json"
    host_before = _host_before()
    warmup = _run_one(index=0, block=0, arm="baseline")
    if warmup["timed_out"] or warmup["workload_exit"] != 0:
        raise RuntimeError("unmeasured warm-up failed")

    rows: list[dict[str, Any]] = []
    for index, (block, arm) in enumerate(_completion_order(), start=1):
        print(f"run {index}/{2 * BLOCKS}: block={block} arm={arm}", flush=True)
        rows.append(
            _run_one(
                index=index,
                block=block,
                arm=arm,
                timeout_s=COMPLETION_TIMEOUT_S,
            )
        )
        partial.write_text(json.dumps({"runs": rows}, indent=2) + "\n")
    result = {
        "schema": "resource-underreservation-memory-completion-v1",
        "role": "post-result-descriptive-amendment",
        "protocol": {
            "seed": SEED,
            "blocks": BLOCKS,
            "arms": ["baseline", "memory_high_2g"],
            "timeout_s": COMPLETION_TIMEOUT_S,
            "poll_s": POLL_S,
            "memory_max_bytes": MEMORY_MAX_BYTES,
            "image": IMAGE,
            "workspace": str(WORKSPACE),
            "workload": str(WORKLOAD),
        },
        "host_before": host_before,
        "warmup_identity": warmup["workload"],
        "runs": rows,
        "summary": _completion_summary(rows),
    }
    (COMPLETION_OUTPUT / "result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    partial.unlink()
    print(json.dumps(result["summary"], indent=2), flush=True)


def _run_calibration() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    if not WORKLOAD.is_file() or not (WORKSPACE / "sources").is_dir():
        raise FileNotFoundError("frozen workload or workspace is missing")
    OUTPUT.mkdir(parents=True)
    partial = OUTPUT / "partial.json"

    host_before = _host_before()
    warmup = _run_one(index=0, block=0, arm="baseline")
    if warmup["timed_out"] or warmup["workload_exit"] != 0:
        raise RuntimeError("unmeasured warm-up failed")

    rows: list[dict[str, Any]] = []
    for index, (block, arm) in enumerate(_run_order(), start=1):
        print(f"run {index}/{BLOCKS * len(ARMS)}: block={block} arm={arm}", flush=True)
        rows.append(_run_one(index=index, block=block, arm=arm))
        partial.write_text(json.dumps({"runs": rows}, indent=2) + "\n")

    result = {
        "schema": "resource-underreservation-calibration-v1",
        "role": "development-mechanism-calibration",
        "protocol": {
            "seed": SEED,
            "blocks": BLOCKS,
            "arms": {arm: {"cpu_cores": values[0], "memory_high": values[1]} for arm, values in ARMS.items()},
            "timeout_s": TIMEOUT_S,
            "poll_s": POLL_S,
            "memory_max_bytes": MEMORY_MAX_BYTES,
            "image": IMAGE,
            "workspace": str(WORKSPACE),
            "workload": str(WORKLOAD),
        },
        "host_before": host_before,
        "warmup_identity": warmup["workload"],
        "runs": rows,
        "summary": _summarize(rows),
    }
    (OUTPUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    partial.unlink()
    print(json.dumps(result["summary"], indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-completion", action="store_true")
    args = parser.parse_args()
    if args.memory_completion:
        _run_memory_completion()
    else:
        _run_calibration()


if __name__ == "__main__":
    main()
