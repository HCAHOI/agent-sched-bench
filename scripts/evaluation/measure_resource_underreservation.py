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
BURST_OUTPUT = (
    ROOT
    / "analysis/results/tool-resource-5-3-3-3-20260804"
    / "resource-burst-contention-v1"
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
BURST_TIMEOUT_S = 900.0
BURST_SLEEP_S = 30.0
POLL_S = 0.1
SEED = 42
BLOCKS = 3
BURST_ARMS = ("hard_two", "burstable_two", "hard_four")
EXPECTED_AST_IDENTITY = (6964, 6964, 15_029_159)
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


def _burst_order() -> list[tuple[int, str]]:
    rng = random.Random(SEED)
    order: list[tuple[int, str]] = []
    for block in range(1, BLOCKS + 1):
        arms = list(BURST_ARMS)
        rng.shuffle(arms)
        order.extend((block, arm) for arm in arms)
    return order


def _burst_quota(arm: str, role: str) -> float | None:
    if arm == "burstable_two":
        return None
    if arm == "hard_four" and role == "ast":
        return 4.0
    return 2.0


def _run_burst_batch(*, index: int, block: int, arm: str) -> dict[str, Any]:
    if arm not in BURST_ARMS:
        raise ValueError(f"unknown burst arm: {arm}")
    name_prefix = f"asb-burst-{os.getpid()}-{index:02d}"
    jobs = [("ast", 1), ("ast", 2), ("idle", 1), ("idle", 2)]
    with tempfile.TemporaryDirectory(prefix="asb-burst-") as state_raw:
        state = Path(state_raw)
        contexts: list[dict[str, Any]] = []
        try:
            for role, slot in jobs:
                job_id = f"{role}{slot}"
                name = f"{name_prefix}-{job_id}"
                job_state = state / job_id
                job_state.mkdir()
                if role == "ast":
                    wrapper = (
                        "while [ ! -e /asb-state/start ]; do sleep 0.02; done; "
                        "python /workload.py --workers 8 --bursts 1 --hold-s 1 "
                        "--idle-s 0 >/asb-state/stdout 2>/asb-state/stderr; rc=$?; "
                        "printf '%s\\n' \"$rc\" >/asb-state/status.tmp; "
                        "mv /asb-state/status.tmp /asb-state/status; "
                        "while [ ! -e /asb-state/release ]; do sleep 0.02; done; "
                        "exit \"$rc\""
                    )
                    memory_bytes = MEMORY_MAX_BYTES
                else:
                    wrapper = (
                        "while [ ! -e /asb-state/start ]; do sleep 0.02; done; "
                        f"sleep {BURST_SLEEP_S}; "
                        f"printf '{{\"elapsed_s\": {BURST_SLEEP_S}, "
                        f"\"sleep_s\": {BURST_SLEEP_S}}}\\n' >/asb-state/stdout; "
                        "printf '0\\n' >/asb-state/status.tmp; "
                        "mv /asb-state/status.tmp /asb-state/status; "
                        "while [ ! -e /asb-state/release ]; do sleep 0.02; done"
                    )
                    memory_bytes = 128 * 1024**2
                quota = _burst_quota(arm, role)
                command = [
                    "docker",
                    "create",
                    "--name",
                    name,
                    "--network",
                    "none",
                    "--cpuset-cpus",
                    "0-7",
                    "--cpu-shares",
                    "1024",
                    "--memory",
                    str(memory_bytes),
                    "--memory-swap",
                    str(memory_bytes),
                    "--oom-score-adj",
                    "-1000",
                    "-v",
                    f"{job_state}:/asb-state",
                ]
                if quota is not None:
                    command.extend(["--cpus", str(quota)])
                if role == "ast":
                    command.extend(
                        [
                            "-v",
                            f"{WORKSPACE}:/workspace:ro",
                            "-v",
                            f"{WORKLOAD}:/workload.py:ro",
                            "-w",
                            "/workspace",
                        ]
                    )
                command.extend([IMAGE, "/bin/sh", "-c", wrapper])
                _command(command)
                _command(["docker", "start", name])
                cgroup = _cgroup_path(name)
                observed = {
                    "cpu_max": (cgroup / "cpu.max").read_text().strip(),
                    "cpu_weight": int((cgroup / "cpu.weight").read_text()),
                    "cpuset_cpus_effective": (
                        cgroup / "cpuset.cpus.effective"
                    ).read_text().strip(),
                    "memory_max_bytes": int((cgroup / "memory.max").read_text()),
                    "memory_swap_max_bytes": int(
                        (cgroup / "memory.swap.max").read_text()
                    ),
                }
                expected_cpu_max = (
                    "max 100000" if quota is None else f"{int(quota * 100_000)} 100000"
                )
                if (
                    observed["cpu_max"] != expected_cpu_max
                    or observed["cpuset_cpus_effective"] != "0-7"
                    or observed["memory_max_bytes"] != memory_bytes
                    or observed["memory_swap_max_bytes"] != 0
                ):
                    raise RuntimeError(f"container limits differ from protocol: {name}")
                before = _snapshot(cgroup)
                contexts.append(
                    {
                        "role": role,
                        "slot": slot,
                        "name": name,
                        "state": job_state,
                        "cgroup": cgroup,
                        "quota_cores": quota,
                        "observed": observed,
                        "before": before,
                        "last": before,
                        "peak": before["memory_current_bytes"],
                        "telemetry_lost": False,
                    }
                )

            if len({context["observed"]["cpu_weight"] for context in contexts}) != 1:
                raise RuntimeError("burst batch CPU weights differ")
            started = time.perf_counter()
            for context in contexts:
                (context["state"] / "start").touch()
            timed_out = False
            abrupt_exit = False
            while not all((context["state"] / "status").exists() for context in contexts):
                for context in contexts:
                    try:
                        sample = _snapshot(context["cgroup"])
                    except FileNotFoundError:
                        context["telemetry_lost"] = True
                        if not (context["state"] / "status").exists():
                            abrupt_exit = True
                    else:
                        context["last"] = sample
                        context["peak"] = max(
                            context["peak"], sample["memory_current_bytes"]
                        )
                if abrupt_exit:
                    break
                if time.perf_counter() - started >= BURST_TIMEOUT_S:
                    timed_out = True
                    break
                time.sleep(POLL_S)
            wall_s = time.perf_counter() - started

            rows = []
            for context in contexts:
                try:
                    after = _snapshot(context["cgroup"])
                except FileNotFoundError:
                    context["telemetry_lost"] = True
                    after = context["last"]
                else:
                    context["last"] = after
                    context["peak"] = max(
                        context["peak"], after["memory_current_bytes"]
                    )
                job_state = context["state"]
                stdout = (
                    (job_state / "stdout").read_text(errors="replace")
                    if (job_state / "stdout").exists()
                    else ""
                )
                stderr = (
                    (job_state / "stderr").read_text(errors="replace")
                    if (job_state / "stderr").exists()
                    else ""
                )
                rows.append(
                    {
                        "role": context["role"],
                        "slot": context["slot"],
                        "requested_quota_cores": context["quota_cores"],
                        "observed": context["observed"],
                        "workload_exit": (
                            int((job_state / "status").read_text())
                            if (job_state / "status").exists()
                            else None
                        ),
                        "workload": _workload_summary(stdout),
                        "cpu_delta": _delta(after["cpu"], context["before"]["cpu"]),
                        "memory_events_delta": _delta(
                            after["memory_events"],
                            context["before"]["memory_events"],
                        ),
                        "telemetry_lost": context["telemetry_lost"],
                        "sampled_peak_memory_current_bytes": context["peak"],
                        "stderr_tail": stderr[-2000:],
                    }
                )

            if timed_out or abrupt_exit:
                for context in contexts:
                    subprocess.run(
                        ["docker", "kill", context["name"]],
                        capture_output=True,
                        check=False,
                        timeout=30,
                    )
            else:
                for context in contexts:
                    (context["state"] / "release").touch()
            for row, context in zip(rows, contexts, strict=True):
                row["container_exit"] = int(
                    _command(["docker", "wait", context["name"]])
                )
                row["docker_oom_killed"] = (
                    _command(
                        [
                            "docker",
                            "inspect",
                            "--format",
                            "{{.State.OOMKilled}}",
                            context["name"],
                        ]
                    )
                    == "true"
                )
            return {
                "index": index,
                "block": block,
                "arm": arm,
                "timeout_s": BURST_TIMEOUT_S,
                "timed_out": timed_out,
                "abrupt_exit": abrupt_exit,
                "batch_wall_s": wall_s,
                "jobs": rows,
            }
        finally:
            for _, slot in reversed(jobs):
                for role in ("idle", "ast"):
                    name = f"{name_prefix}-{role}{slot}"
                    subprocess.run(
                        ["docker", "rm", "-f", name],
                        capture_output=True,
                        check=False,
                        timeout=30,
                    )


def _burst_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm = {arm: [row for row in rows if row["arm"] == arm] for arm in BURST_ARMS}
    ast_jobs = [
        job for row in rows for job in row["jobs"] if job["role"] == "ast"
    ]
    identities = {
        (
            job["workload"]["source_files"],
            job["workload"]["bursts"][0]["files"],
            job["workload"]["bursts"][0]["ast_nodes"],
        )
        for job in ast_jobs
        if job["workload"] is not None
    }
    valid = bool(
        all(len(by_arm[arm]) == BLOCKS for arm in BURST_ARMS)
        and len(ast_jobs) == 2 * BLOCKS * len(BURST_ARMS)
        and identities == {EXPECTED_AST_IDENTITY}
        and all(
            not row["timed_out"]
            and not row["abrupt_exit"]
            and len(row["jobs"]) == 4
            and all(
                job["workload_exit"] == 0
                and job["container_exit"] == 0
                and job["workload"] is not None
                and job["observed"]["cpuset_cpus_effective"] == "0-7"
                and not job["telemetry_lost"]
                and not job["docker_oom_killed"]
                and job["memory_events_delta"].get("oom", 0) == 0
                and job["memory_events_delta"].get("oom_kill", 0) == 0
                for job in row["jobs"]
            )
            for row in rows
        )
    )
    medians = {
        arm: statistics.median(row["batch_wall_s"] for row in arm_rows)
        if arm_rows
        else None
        for arm, arm_rows in by_arm.items()
    }
    hard_by_block = {row["block"]: row for row in by_arm["hard_two"]}
    burst_by_block = {row["block"]: row for row in by_arm["burstable_two"]}
    every_block_faster = len(hard_by_block) == len(burst_by_block) == BLOCKS and all(
        burst_by_block[block]["batch_wall_s"]
        < hard_by_block[block]["batch_wall_s"]
        for block in hard_by_block
    )
    hard_throttled = {
        row["block"]: sum(
            job["cpu_delta"].get("throttled_usec", 0)
            for job in row["jobs"]
            if job["role"] == "ast"
        )
        for row in by_arm["hard_two"]
    }
    burst_throttled = {
        row["block"]: sum(
            job["cpu_delta"].get("throttled_usec", 0)
            for job in row["jobs"]
            if job["role"] == "ast"
        )
        for row in by_arm["burstable_two"]
    }
    every_block_less_throttled = (
        len(hard_throttled) == len(burst_throttled) == BLOCKS
        and all(
            hard_throttled[block] > burst_throttled[block]
            for block in hard_throttled
        )
    )
    reduction = (
        (medians["hard_two"] - medians["burstable_two"]) / medians["hard_two"]
        if medians["hard_two"] and medians["burstable_two"] is not None
        else None
    )
    ratio_to_control = (
        medians["burstable_two"] / medians["hard_four"]
        if medians["hard_four"] and medians["burstable_two"] is not None
        else None
    )
    gate = bool(
        valid
        and every_block_faster
        and reduction is not None
        and reduction >= 0.25
        and ratio_to_control is not None
        and ratio_to_control <= 1.10
        and every_block_less_throttled
    )
    return {
        "valid": valid,
        "batch_wall_median_s": medians,
        "burstable_reduction_vs_hard_two": reduction,
        "burstable_ratio_to_hard_four": ratio_to_control,
        "every_block_burstable_faster": every_block_faster,
        "every_block_burstable_less_throttled": every_block_less_throttled,
        "successful_output_identities": [list(value) for value in sorted(identities)],
        "gate": gate,
        "status": (
            "development_go_to_fresh_sqlglot_burst_protocol"
            if gate
            else "development_no_go_physical_burst_mechanism"
        ),
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


def _run_burst_contention() -> None:
    if BURST_OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {BURST_OUTPUT}")
    if not WORKLOAD.is_file() or not (WORKSPACE / "sources").is_dir():
        raise FileNotFoundError("frozen workload or workspace is missing")
    BURST_OUTPUT.mkdir(parents=True)
    partial = BURST_OUTPUT / "partial.json"
    host_before = _host_before()
    warmup = _run_burst_batch(index=0, block=0, arm="hard_four")
    if warmup["timed_out"] or any(
        job["workload_exit"] != 0 or job["container_exit"] != 0
        for job in warmup["jobs"]
    ) or {
        (
            job["workload"]["source_files"],
            job["workload"]["bursts"][0]["files"],
            job["workload"]["bursts"][0]["ast_nodes"],
        )
        for job in warmup["jobs"]
        if job["role"] == "ast" and job["workload"] is not None
    } != {EXPECTED_AST_IDENTITY}:
        raise RuntimeError("unmeasured burst warm-up failed")

    rows: list[dict[str, Any]] = []
    for index, (block, arm) in enumerate(_burst_order(), start=1):
        print(f"run {index}/{BLOCKS * len(BURST_ARMS)}: block={block} arm={arm}", flush=True)
        rows.append(_run_burst_batch(index=index, block=block, arm=arm))
        partial.write_text(json.dumps({"runs": rows}, indent=2) + "\n")

    summary = _burst_summary(rows)
    result = {
        "schema": "resource-burst-contention-v1",
        "role": "development-physical-mechanism-test",
        "inputs": {
            "git_sha": _command(["git", "rev-parse", "HEAD"]),
            "workspace": str(WORKSPACE),
            "workload": str(WORKLOAD),
        },
        "protocol": {
            "seed": SEED,
            "blocks": BLOCKS,
            "arms": list(BURST_ARMS),
            "ast_jobs": 2,
            "idle_jobs": 2,
            "idle_s": BURST_SLEEP_S,
            "cpuset": "0-7",
            "cpu_shares": 1024,
            "ast_workers": 8,
            "timeout_s": BURST_TIMEOUT_S,
            "poll_s": POLL_S,
            "ast_memory_max_bytes": MEMORY_MAX_BYTES,
            "idle_memory_max_bytes": 128 * 1024**2,
            "image": IMAGE,
        },
        "host_before": host_before,
        "warmup": warmup,
        "runs": rows,
        "summary": summary,
        "limitations": [
            "This mechanism test uses the AST workload, not SQLGlot task commands.",
            "Idle sleep jobs isolate CPU borrowing but do not model other resource interference.",
            "Equal cgroup weights provide borrowing, not the simulator's hindsight demand caps.",
        ],
    }
    (BURST_OUTPUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    partial.unlink()
    print(json.dumps(summary, indent=2), flush=True)


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
    parser.add_argument("--burst-contention", action="store_true")
    args = parser.parse_args()
    if args.memory_completion and args.burst_contention:
        parser.error("choose one experiment")
    if args.memory_completion:
        _run_memory_completion()
    elif args.burst_contention:
        _run_burst_contention()
    else:
        _run_calibration()


if __name__ == "__main__":
    main()
