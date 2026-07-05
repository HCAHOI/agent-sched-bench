#!/usr/bin/env python3
"""Eval campaign: F3+E phase — produce comprehensive JSON report for the paper.

Runs all eval experiments and produces a JSON report with per-experiment
summaries, raw data, and measurement methodology metadata.

Usage::

    PYTHONPATH=src python scripts/experiments/eval_campaign.py --help
    PYTHONPATH=src python scripts/experiments/eval_campaign.py --list
    PYTHONPATH=src python scripts/experiments/eval_campaign.py --run density
    PYTHONPATH=src python scripts/experiments/eval_campaign.py --run all --output report.json

All experiments that need /dev/kvm delegate through ``run_on_kvm``, which
auto-detects local KVM and falls back to SSH if unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KVM_HOST_DEFAULT = "ubuntu@51.158.203.248"
_KVM_REPO_PATH = "~/workspace/agent-sched-bench"
_FC_KERNEL_PATH = Path(
    os.environ.get("FC_KERNEL_PATH", "/tmp/fc-cache/vmlinux-5.10.225")
)
_FC_ROOTFS_PATH = Path(
    os.environ.get("FC_ROOTFS_PATH", "/tmp/fc-cache/rootfs.ext4")
)

# Firecracker defaults (mirrored from dr1_fc_runner.py).
_FC_VCPU_COUNT = 2
_FC_MEM_SIZE_MIB = 1024
_FC_BINARY = "firecracker"
_VM_BOOT_TIMEOUT_S = 60

AVAILABLE_RUNS: tuple[str, ...] = (
    "density",
    "checkpoint-traffic",
    "restore-latency",
    "mismatch-rate",
    "all",
)


# ---------------------------------------------------------------------------
# KVM helper
# ---------------------------------------------------------------------------


def _has_local_kvm() -> bool:
    """Return True if /dev/kvm is accessible locally."""
    return Path("/dev/kvm").exists()


def run_on_kvm(
    cmd: list[str] | str,
    *,
    capture: bool = True,
    timeout: int | None = None,
    cwd: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run *cmd* on a KVM-capable host.

    If /dev/kvm exists locally, runs the command via ``subprocess.run``.
    Otherwise, SSH-es to ``KVM_HOST`` (env var, default 51.158.203.248) and
    executes the command there.

    Returns a ``CompletedProcess`` with ``stdout``, ``stderr``, and
    ``returncode``.
    """
    if isinstance(cmd, list):
        cmd_str = " ".join(shlex.quote(p) for p in cmd)
    else:
        cmd_str = cmd

    if _has_local_kvm():
        result = subprocess.run(
            cmd_str,
            shell=True,
            capture_output=capture,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env={**os.environ, **(extra_env or {})},
            check=False,
        )
        return result

    kvm_host = os.environ.get("KVM_HOST", _KVM_HOST_DEFAULT)
    # Wrap in bash -lc so .venv activation and PATH work on remote.
    wrapped = (
        f"cd {_KVM_REPO_PATH} && "
        f"source .venv/bin/activate && "
        f"{cmd_str}"
    )
    ssh_cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=10",
        kvm_host,
        "bash", "-lc", shlex.quote(wrapped),
    ]

    result = subprocess.run(
        ssh_cmd,
        capture_output=capture,
        text=True,
        timeout=timeout,
        cwd=cwd,
        check=False,
    )
    return result


def run_on_kvm_py(
    python_code: str,
    *,
    timeout: int | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a Python snippet on the KVM host with PYTHONPATH=src.

    The snippet is written to a temp file, copied to the KVM host if needed,
    and executed.
    """
    if _has_local_kvm():
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="eval_camp_"
        ) as fh:
            fh.write(python_code)
            tmp_path = fh.name
        try:
            env = {**os.environ, "PYTHONPATH": "src", **(extra_env or {})}
            return subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                check=False,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # Remote: scp the script, run it, scp results back.
    kvm_host = os.environ.get("KVM_HOST", _KVM_HOST_DEFAULT)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="eval_camp_"
    ) as fh:
        fh.write(python_code)
        tmp_path = fh.name
    try:
        remote_path = f"/tmp/eval_camp_{os.getpid()}.py"
        subprocess.run(
            ["scp", "-o", "StrictHostKeyChecking=no",
             "-o", "UserKnownHostsFile=/dev/null",
             tmp_path, f"{kvm_host}:{remote_path}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            kvm_host,
            f"cd {_KVM_REPO_PATH} && source .venv/bin/activate && "
            f"PYTHONPATH=src python {remote_path}",
        ]
        return subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Trace file discovery
# ---------------------------------------------------------------------------


def _discover_trace_files(trace_dir: Path) -> list[Path]:
    """Find all ``trace.jsonl`` files under *trace_dir*.

    Looks for the conventional layout ``{task_id}/attempt_1/trace.jsonl`` and
    also accepts a single ``trace.jsonl`` passed directly.
    """
    if not trace_dir.exists():
        return []
    if trace_dir.is_file():
        if trace_dir.suffix == ".jsonl" or trace_dir.name == "trace.jsonl":
            return [trace_dir]
        return []
    candidates: list[Path] = []
    for attempt_dir in sorted(trace_dir.rglob("attempt_1")):
        trace = attempt_dir / "trace.jsonl"
        if trace.is_file():
            candidates.append(trace)
    if not candidates:
        direct = trace_dir / "trace.jsonl"
        if direct.is_file():
            candidates.append(direct)
    return sorted(candidates)


# ---------------------------------------------------------------------------
# Density experiment
# ---------------------------------------------------------------------------


@dataclass
class DensityResult:
    density: int
    fc_vm_rss_kb: list[int]
    fc_vm_rss_mean_kb: float
    fc_vm_rss_median_kb: float
    fc_boot_latency_ms: list[float]
    fc_boot_latency_mean_ms: float
    docker_container_rss_kb: list[int]
    docker_container_rss_mean_kb: float
    docker_container_rss_median_kb: float
    rss_ratio_fc_vs_docker: float


def _run_density_on_host(densities: Sequence[int]) -> list[DensityResult]:
    """Execute density experiment directly on a KVM-capable host.

    For each density N, launches N Firecracker VMs (idle), measures boot
    latency and RSS, then launches N ``docker run -d alpine sleep infinity``
    containers for comparison.
    """
    results: list[DensityResult] = []

    # --- Pre-flight checks ---
    fc_bin = shutil.which("firecracker") or _FC_BINARY
    if not Path(fc_bin).exists() and not shutil.which(fc_bin):
        print("ERROR: firecracker not found on PATH", file=sys.stderr)
        return results

    kernel = _FC_KERNEL_PATH
    rootfs = _FC_ROOTFS_PATH
    if not kernel.exists():
        print(f"ERROR: kernel not found at {kernel}", file=sys.stderr)
        return results
    if not rootfs.exists():
        print(f"ERROR: rootfs not found at {rootfs}", file=sys.stderr)
        return results

    for n in densities:
        print(f"\n--- Density {n} ---", file=sys.stderr)
        fc_pids: list[int] = []
        fc_boot_ms: list[float] = []
        fc_rss_kb: list[int] = []
        fc_sockets: list[str] = []

        try:
            # Launch N Firecracker VMs sequentially.
            for i in range(n):
                api_sock = f"/tmp/fc-density-{os.getpid()}-{i}.sock"
                if os.path.exists(api_sock):
                    os.unlink(api_sock)
                fc_sockets.append(api_sock)

                t0 = time.monotonic()
                proc = subprocess.Popen(
                    [
                        fc_bin,
                        "--api-sock", api_sock,
                        "--boot-timer",
                        "--no-api",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                fc_pids.append(proc.pid)

                # Wait briefly for the process to start, then measure RSS.
                time.sleep(0.5)
                rss = _read_process_rss_kb(proc.pid)
                if rss > 0:
                    fc_rss_kb.append(rss)

                # Boot latency: time until the VM process is in 'S' (sleeping)
                # state.  We don't do full SSH readiness for idle VMs — the
                # process RSS stabilizes within ~1s.
                deadline = time.monotonic() + _VM_BOOT_TIMEOUT_S
                booted = False
                while time.monotonic() < deadline:
                    rss2 = _read_process_rss_kb(proc.pid)
                    if rss2 > 0 and abs(rss2 - rss) / max(rss, 1) < 0.05:
                        booted = True
                        break
                    time.sleep(0.5)
                elapsed_ms = (time.monotonic() - t0) * 1000
                fc_boot_ms.append(elapsed_ms)
                if not booted:
                    print(
                        f"  VM {i}: RSS may not have stabilized "
                        f"({elapsed_ms:.0f}ms)",
                        file=sys.stderr,
                    )

                print(
                    f"  VM {i}: PID={proc.pid} RSS={rss}kB "
                    f"boot={elapsed_ms:.0f}ms",
                    file=sys.stderr,
                )

            # Measure RSS again after all VMs are running.
            fc_rss_kb_snapshot: list[int] = []
            for pid in fc_pids:
                rss = _read_process_rss_kb(pid)
                if rss > 0:
                    fc_rss_kb_snapshot.append(rss)

            # --- Docker comparison ---
            docker_rss_kb: list[int] = []
            container_ids: list[str] = []
            for i in range(n):
                result = subprocess.run(
                    [
                        "docker", "run", "-d", "--rm",
                        "alpine", "sleep", "infinity",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                if result.returncode == 0:
                    cid = result.stdout.strip()
                    container_ids.append(cid)
                    time.sleep(0.5)
                    rss = _read_container_rss_kb(cid)
                    if rss > 0:
                        docker_rss_kb.append(rss)
                    print(
                        f"  Docker {i}: CID={cid[:12]} RSS={rss}kB",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"  Docker {i}: ERROR — {result.stderr.strip()}",
                        file=sys.stderr,
                    )

            # Clean up docker containers.
            for cid in container_ids:
                subprocess.run(
                    ["docker", "stop", cid],
                    capture_output=True,
                    timeout=10,
                    check=False,
                )

            # Compute summary stats.
            fc_mean_rss = (
                sum(fc_rss_kb_snapshot) / len(fc_rss_kb_snapshot)
                if fc_rss_kb_snapshot
                else 0.0
            )
            fc_median_rss = _median(fc_rss_kb_snapshot) if fc_rss_kb_snapshot else 0.0
            docker_mean_rss = (
                sum(docker_rss_kb) / len(docker_rss_kb)
                if docker_rss_kb
                else 0.0
            )
            docker_median_rss = _median(docker_rss_kb) if docker_rss_kb else 0.0
            ratio = (
                fc_mean_rss / docker_mean_rss
                if docker_mean_rss > 0
                else float("inf")
            )

            results.append(
                DensityResult(
                    density=n,
                    fc_vm_rss_kb=fc_rss_kb_snapshot,
                    fc_vm_rss_mean_kb=round(fc_mean_rss, 1),
                    fc_vm_rss_median_kb=round(fc_median_rss, 1),
                    fc_boot_latency_ms=[round(v, 1) for v in fc_boot_ms],
                    fc_boot_latency_mean_ms=round(
                        sum(fc_boot_ms) / len(fc_boot_ms), 1
                    ) if fc_boot_ms else 0.0,
                    docker_container_rss_kb=docker_rss_kb,
                    docker_container_rss_mean_kb=round(docker_mean_rss, 1),
                    docker_container_rss_median_kb=round(docker_median_rss, 1),
                    rss_ratio_fc_vs_docker=round(ratio, 3),
                )
            )

        finally:
            # Kill all FC processes.
            for pid in fc_pids:
                try:
                    os.kill(pid, 9)
                except OSError:
                    pass
            for sock in fc_sockets:
                try:
                    os.unlink(sock)
                except OSError:
                    pass

    return results


def _read_process_rss_kb(pid: int) -> int:
    """Read VmRSS (kB) from /proc/<pid>/status."""
    try:
        text = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                return int(parts[1]) if len(parts) >= 2 else 0
    except (OSError, ValueError):
        pass
    return 0


def _read_container_rss_kb(container_id: str) -> int:
    """Read RSS for a docker container via ``docker inspect`` or /proc."""
    # First try docker stats (one-shot).
    result = subprocess.run(
        [
            "docker", "stats", "--no-stream",
            "--format", "{{.MemUsage}}",
            container_id,
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        usage = result.stdout.strip()  # e.g. "1.234MiB / 16GiB"
        return _parse_docker_mem_kb(usage)

    # Fallback: find the container's PID via inspect, then read /proc.
    result2 = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Pid}}", container_id],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result2.returncode == 0 and result2.stdout.strip():
        try:
            pid = int(result2.stdout.strip())
            return _read_process_rss_kb(pid)
        except ValueError:
            pass
    return 0


def _parse_docker_mem_kb(usage: str) -> int:
    """Parse a docker memory usage string like '1.234MiB' into kB."""
    m = re.match(r"([\d.]+)\s*(KiB|MiB|GiB|kB|MB|GB|B)", usage.strip())
    if not m:
        return 0
    value = float(m.group(1))
    unit = m.group(2).lower()
    multipliers: dict[str, float] = {
        "b": 1 / 1024,
        "kb": 1,
        "kib": 1,
        "mb": 1024,
        "mib": 1024,
        "gb": 1024 * 1024,
        "gib": 1024 * 1024,
    }
    return int(value * multipliers.get(unit, 1))


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return float(s[n // 2])
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def run_density(args: argparse.Namespace) -> dict[str, Any]:
    """Run the density experiment."""
    densities = [1, 2, 4, 8, 16]

    py_code = f"""
import json, os, sys
sys.path.insert(0, "src")
# Inline the density experiment code.
{_density_py_snippet(densities)}
results = _run_density_on_host({densities!r})
print(json.dumps([r.__dict__ if hasattr(r, '__dict__') else r for r in results], default=str))
"""

    result = run_on_kvm_py(py_code, timeout=600)
    if result.returncode != 0:
        return {
            "experiment": "density",
            "error": result.stderr.strip() or "Unknown error",
            "stdout": result.stdout,
        }

    try:
        raw = json.loads(result.stdout.strip().split("\n")[-1])
    except json.JSONDecodeError:
        raw = []

    return {
        "experiment": "density",
        "methodology": {
            "fc_version": "v1.11.0",
            "kernel": str(_FC_KERNEL_PATH),
            "vcpu": _FC_VCPU_COUNT,
            "mem_mib": _FC_MEM_SIZE_MIB,
            "measurement": "RSS via /proc/<pid>/status VmRSS",
            "docker_baseline": "alpine:latest sleep infinity",
            "densities_tested": list(densities),
        },
        "results": raw,
    }


def _density_py_snippet(densities: Sequence[int]) -> str:
    """Return a self-contained Python snippet that implements density
    measurement.  Inlined for remote execution via ``run_on_kvm_py``."""
    return """
import json, os, shutil, subprocess, sys, time, re
from pathlib import Path

_FC_KERNEL_PATH = Path(os.environ.get("FC_KERNEL_PATH", "/tmp/fc-cache/vmlinux-5.10.225"))
_FC_ROOTFS_PATH = Path(os.environ.get("FC_ROOTFS_PATH", "/tmp/fc-cache/rootfs.ext4"))
_VM_BOOT_TIMEOUT_S = 60

def _read_process_rss_kb(pid):
    try:
        text = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                return int(parts[1]) if len(parts) >= 2 else 0
    except (OSError, ValueError):
        pass
    return 0

def _read_container_rss_kb(container_id):
    result = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container_id],
        capture_output=True, text=True, timeout=10, check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        usage = result.stdout.strip()
        m = re.match(r"([\\d.]+)\\s*(KiB|MiB|GiB|kB|MB|GB|B)", usage.strip())
        if m:
            value = float(m.group(1))
            unit = m.group(2).lower()
            multipliers = {"b": 1/1024, "kb": 1, "kib": 1, "mb": 1024, "mib": 1024, "gb": 1024*1024, "gib": 1024*1024}
            return int(value * multipliers.get(unit, 1))
    result2 = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Pid}}", container_id],
        capture_output=True, text=True, timeout=10, check=False,
    )
    if result2.returncode == 0 and result2.stdout.strip():
        try:
            return _read_process_rss_kb(int(result2.stdout.strip()))
        except ValueError:
            pass
    return 0

def _median(values):
    if not values: return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1: return float(s[n // 2])
    return (s[n // 2 - 1] + s[n // 2]) / 2.0

def _run_density_on_host(densities):
    results = []
    fc_bin = shutil.which("firecracker") or "firecracker"
    kernel = _FC_KERNEL_PATH
    rootfs = _FC_ROOTFS_PATH
    if not kernel.exists():
        print("ERROR: kernel not found", file=sys.stderr)
        return results
    if not rootfs.exists():
        print("ERROR: rootfs not found", file=sys.stderr)
        return results

    for n in densities:
        print(f"--- Density {n} ---", file=sys.stderr)
        fc_pids, fc_boot_ms, fc_rss_kb, fc_sockets = [], [], [], []
        try:
            for i in range(n):
                api_sock = f"/tmp/fc-density-{os.getpid()}-{i}.sock"
                if os.path.exists(api_sock):
                    os.unlink(api_sock)
                fc_sockets.append(api_sock)

                t0 = time.monotonic()
                proc = subprocess.Popen(
                    [fc_bin, "--api-sock", api_sock, "--boot-timer", "--no-api"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                fc_pids.append(proc.pid)
                time.sleep(0.5)
                rss = _read_process_rss_kb(proc.pid)
                if rss > 0:
                    fc_rss_kb.append(rss)

                deadline = time.monotonic() + _VM_BOOT_TIMEOUT_S
                while time.monotonic() < deadline:
                    rss2 = _read_process_rss_kb(proc.pid)
                    if rss2 > 0 and abs(rss2 - rss) / max(rss, 1) < 0.05:
                        break
                    time.sleep(0.5)
                elapsed_ms = (time.monotonic() - t0) * 1000
                fc_boot_ms.append(elapsed_ms)

            fc_snapshot = [_read_process_rss_kb(p) for p in fc_pids if _read_process_rss_kb(p) > 0]

            docker_rss, container_ids = [], []
            for i in range(n):
                result = subprocess.run(
                    ["docker", "run", "-d", "--rm", "alpine", "sleep", "infinity"],
                    capture_output=True, text=True, timeout=30, check=False,
                )
                if result.returncode == 0:
                    cid = result.stdout.strip()
                    container_ids.append(cid)
                    time.sleep(0.5)
                    rss = _read_container_rss_kb(cid)
                    if rss > 0:
                        docker_rss.append(rss)

            for cid in container_ids:
                subprocess.run(["docker", "stop", cid], capture_output=True, timeout=10, check=False)

            fc_mean = sum(fc_snapshot) / len(fc_snapshot) if fc_snapshot else 0.0
            fc_med = _median(fc_snapshot) if fc_snapshot else 0.0
            d_mean = sum(docker_rss) / len(docker_rss) if docker_rss else 0.0
            d_med = _median(docker_rss) if docker_rss else 0.0
            ratio = fc_mean / d_mean if d_mean > 0 else float("inf")

            results.append({
                "density": n,
                "fc_vm_rss_kb": fc_snapshot,
                "fc_vm_rss_mean_kb": round(fc_mean, 1),
                "fc_vm_rss_median_kb": round(fc_med, 1),
                "fc_boot_latency_ms": [round(v, 1) for v in fc_boot_ms],
                "fc_boot_latency_mean_ms": round(sum(fc_boot_ms) / len(fc_boot_ms), 1) if fc_boot_ms else 0.0,
                "docker_container_rss_kb": docker_rss,
                "docker_container_rss_mean_kb": round(d_mean, 1),
                "docker_container_rss_median_kb": round(d_med, 1),
                "rss_ratio_fc_vs_docker": round(ratio, 3),
            })
        finally:
            for pid in fc_pids:
                try: os.kill(pid, 9)
                except OSError: pass
            for sock in fc_sockets:
                try: os.unlink(sock)
                except OSError: pass

    return results
"""


# ---------------------------------------------------------------------------
# Checkpoint traffic experiment
# ---------------------------------------------------------------------------


def run_checkpoint_traffic(args: argparse.Namespace) -> dict[str, Any]:
    """Run the checkpoint-traffic experiment using DR1's trace analysis.

    Reads trace files from ``--trace-dir`` and computes per-turn bytes for
    three methods: file-CAS delta, estimated memory-diff, and block-delta
    (via thin_delta when available).
    """
    trace_dir = Path(
        args.trace_dir or "traces"
    ).resolve()
    trace_files = _discover_trace_files(trace_dir)
    if not trace_files:
        return {
            "experiment": "checkpoint-traffic",
            "error": f"No trace.jsonl files found under {trace_dir}",
            "num_traces": 0,
        }

    # Import DR1 logic lazily.
    try:
        from scripts.experiments.dr1_memory_diff_cost import (
            _compute_file_cas_delta_bytes,
            _estimate_dirty_bytes,
            _estimate_pause_ms,
            _iter_turns,
            _parse_jsonl,
        )
    except ImportError:
        # Fallback: inline the logic.
        return _run_checkpoint_traffic_fallback(trace_files, trace_dir)

    all_turns: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "experiment": "checkpoint-traffic",
        "methodology": {
            "file_cas_delta": "Sum of added/modified file sizes between CAS manifests at turn boundaries (see dr1_memory_diff_cost.py)",
            "memory_diff_estimate": "file_cas_delta * 1.15 (4 KiB block round-up + metadata overhead); actual KVM dirty-page tracking deferred to DR1 Stage 2",
            "block_delta": "thin_delta changed-block count * 512 byte sectors (available only on KVM hosts with dm-thin)",
        },
        "num_traces": len(trace_files),
        "num_turns_total": 0,
        "per_method": {
            "file_cas_delta_bytes": {},
            "memory_diff_bytes_est": {},
            "block_delta_bytes": {},
        },
        "per_turn_samples": [],
    }

    for tf in trace_files:
        records = _parse_jsonl(tf)
        folded: dict[str, dict[str, Any]] = {}
        for turn_idx, turn_actions in _iter_turns(records):
            cas_delta, folded = _compute_file_cas_delta_bytes(
                turn_actions, tf, folded
            )
            dirty_est = _estimate_dirty_bytes(cas_delta)
            pause_est = _estimate_pause_ms(dirty_est)

            all_turns.append({
                "trace_file": str(tf),
                "turn_number": turn_idx + 1,
                "file_cas_delta_bytes": cas_delta,
                "memory_diff_bytes_est": dirty_est,
                "block_delta_bytes": None,  # requires KVM
                "snapshot_pause_ms_est": pause_est,
            })

    # Compute summary stats.
    if all_turns:
        cas_values = [t["file_cas_delta_bytes"] for t in all_turns]
        dirty_values = [t["memory_diff_bytes_est"] for t in all_turns]
        summary["num_turns_total"] = len(all_turns)
        summary["per_method"]["file_cas_delta_bytes"] = _describe_stats(cas_values)
        summary["per_method"]["memory_diff_bytes_est"] = _describe_stats(dirty_values)
        summary["per_turn_samples"] = all_turns[:100]  # cap to avoid huge JSON

    return summary


def _run_checkpoint_traffic_fallback(
    trace_files: list[Path], trace_dir: Path
) -> dict[str, Any]:
    """Fallback: run the DR1 script as a subprocess and parse CSV output."""
    dr1_script = Path("scripts/experiments/dr1_memory_diff_cost.py")
    all_rows: list[dict[str, Any]] = []
    for tf in trace_files:
        # Run Stage 1 analysis for each trace.
        result = subprocess.run(
            [
                sys.executable, str(dr1_script),
                "--trace-dir", str(tf),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(Path(__file__).parent.parent.parent),
            env={**os.environ, "PYTHONPATH": "src"},
            check=False,
        )
        if result.returncode != 0:
            continue
        # Parse CSV from stdout (skip stderr comments and header).
        for line in result.stdout.strip().splitlines():
            if line.startswith("#") or line.startswith("turn_number"):
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                try:
                    all_rows.append({
                        "turn_number": int(parts[0]),
                        "dirty_memory_bytes": int(parts[1]),
                        "file_cas_delta_bytes": int(parts[2]),
                        "pause_ms": float(parts[3]),
                        "guest_slowdown_ms": float(parts[4]),
                    })
                except (ValueError, IndexError):
                    pass

    cas_values = [r["file_cas_delta_bytes"] for r in all_rows]
    dirty_values = [r["dirty_memory_bytes"] for r in all_rows]

    return {
        "experiment": "checkpoint-traffic",
        "methodology": {
            "note": "Computed by dr1_memory_diff_cost.py Stage 1 (analytical estimates)",
            "file_cas_delta": "Sum of added/modified file sizes between CAS manifests at turn boundaries",
            "memory_diff_estimate": "file_cas_delta * 1.15 (4 KiB block round-up + metadata overhead)",
            "block_delta": "Requires KVM host with dm-thin; not available in fallback mode",
        },
        "num_traces": len(trace_files),
        "num_turns_total": len(all_rows),
        "per_method": {
            "file_cas_delta_bytes": _describe_stats(cas_values),
            "memory_diff_bytes_est": _describe_stats(dirty_values),
            "block_delta_bytes": {"note": "Requires KVM host"},
        },
        "per_turn_samples": all_rows[:100],
    }


def _describe_stats(values: list[int]) -> dict[str, Any]:
    """Compute descriptive statistics for a list of values."""
    if not values:
        return {"count": 0, "sum": 0, "mean": 0, "median": 0,
                "p95": 0, "p99": 0, "min": 0, "max": 0}
    s = sorted(values)
    n = len(s)
    return {
        "count": n,
        "sum": sum(s),
        "mean": round(sum(s) / n, 1),
        "median": round(_median(s), 1),
        "p95": round(s[int(n * 0.95)] if n > 1 else s[0], 1),
        "p99": round(s[int(n * 0.99)] if n > 1 else s[0], 1),
        "min": s[0],
        "max": s[-1],
    }


# ---------------------------------------------------------------------------
# Restore latency experiment
# ---------------------------------------------------------------------------


def run_restore_latency(args: argparse.Namespace) -> dict[str, Any]:
    """Measure Firecracker snapshot restore latency (warm and cold)."""
    # The snippet is shipped to run_on_kvm_py; it must be self-contained.
    py_code = """
import json, os, socket, subprocess, sys, tempfile, time
from pathlib import Path

_FC_BINARY = "firecracker"
_FC_KERNEL_PATH = Path(os.environ.get("FC_KERNEL_PATH", "/tmp/fc-cache/vmlinux-5.10.225"))
_FC_ROOTFS_PATH = Path(os.environ.get("FC_ROOTFS_PATH", "/tmp/fc-cache/rootfs.ext4"))
_API_SOCK = "/tmp/fc-restore-lat.sock"
_MEM_MIB = 1024

def _fc_api_request(sock_path, method, path, body=None):
    import json
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(10)
    sock.connect(sock_path)
    try:
        headers = f"{method} {path} HTTP/1.1\\r\\nHost: localhost\\r\\n"
        body_bytes = b""
        if body is not None:
            body_str = json.dumps(body)
            body_bytes = body_str.encode("utf-8")
            headers += f"Content-Type: application/json\\r\\nContent-Length: {len(body_bytes)}\\r\\n"
        headers += "Connection: close\\r\\n\\r\\n"
        sock.sendall(headers.encode("utf-8") + body_bytes)
        chunks = []
        while True:
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                break
            if not chunk:
                break
            chunks.append(chunk)
        response = b"".join(chunks).decode("utf-8", errors="replace")
    finally:
        sock.close()
    parts = response.split("\\r\\n\\r\\n", 1)
    resp_body = parts[1] if len(parts) > 1 else ""
    header_section = parts[0]
    status_line = header_section.split("\\r\\n")[0]
    try:
        status_code = int(status_line.split(" ")[1])
    except (IndexError, ValueError):
        status_code = 0
    if resp_body.strip():
        try:
            return status_code, json.loads(resp_body)
        except json.JSONDecodeError:
            return status_code, resp_body
    return status_code, {}

def _configure_vm(sock_path):
    _fc_api_request(sock_path, "PUT", "/machine-config", {"vcpu_count": 2, "mem_size_mib": _MEM_MIB})
    _fc_api_request(sock_path, "PUT", "/boot-source", {"kernel_image_path": str(_FC_KERNEL_PATH), "boot_args": "console=ttyS0 reboot=k panic=1 pci=off root=/dev/vda rw quiet"})
    _fc_api_request(sock_path, "PUT", "/drives/rootfs", {"drive_id": "rootfs", "path_on_host": str(_FC_ROOTFS_PATH), "is_root_device": True, "is_read_only": False})

results = {"warm": {}, "cold": {}, "error": None}

try:
    # Clean up stale socket.
    if os.path.exists(_API_SOCK):
        os.unlink(_API_SOCK)

    # Phase 1: Boot VM and take full snapshot.
    proc = subprocess.Popen(
        [_FC_BINARY, "--api-sock", _API_SOCK],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 30
    while not os.path.exists(_API_SOCK):
        if time.monotonic() > deadline:
            raise TimeoutError("FC API socket did not appear")
        if proc.poll() is not None:
            raise RuntimeError(f"FC exited early (rc={proc.returncode})")
        time.sleep(0.1)

    _configure_vm(_API_SOCK)
    _fc_api_request(_API_SOCK, "PUT", "/actions", {"action_type": "InstanceStart"})
    time.sleep(2)  # Let VM boot.

    snap_dir = tempfile.mkdtemp(prefix="fc-restore-")
    mem_path = os.path.join(snap_dir, "full-mem.snap")
    state_path = os.path.join(snap_dir, "full-vmstate.snap")

    # Take full snapshot.
    t0 = time.monotonic()
    _fc_api_request(_API_SOCK, "PATCH", "/vm", {"state": "Paused"})
    pause_ms = (time.monotonic() - t0) * 1000
    _fc_api_request(_API_SOCK, "PUT", "/snapshot/create", {
        "snapshot_type": "Full", "snapshot_path": mem_path,
        "mem_file_path": mem_path, "version": "1.1.0",
    })
    _fc_api_request(_API_SOCK, "PATCH", "/vm", {"state": "Resumed"})
    full_snap_ms = (time.monotonic() - t0) * 1000

    results["full_snapshot"] = {
        "pause_ms": round(pause_ms, 3),
        "total_ms": round(full_snap_ms, 3),
    }

    # Phase 2: Warm restore — resume from snapshot with VM already running.
    t1 = time.monotonic()
    _fc_api_request(_API_SOCK, "PATCH", "/vm", {"state": "Paused"})
    warm_pause_ms = (time.monotonic() - t1) * 1000
    _fc_api_request(_API_SOCK, "PUT", "/snapshot/load", {
        "snapshot_path": mem_path, "mem_file_path": mem_path,
        "enable_diff_snapshots": True,
    })
    _fc_api_request(_API_SOCK, "PATCH", "/vm", {"state": "Resumed"})
    warm_total_ms = (time.monotonic() - t1) * 1000

    results["warm"] = {
        "pause_ms": round(warm_pause_ms, 3),
        "total_restore_ms": round(warm_total_ms, 3),
        "components": {
            "pause": round(warm_pause_ms, 3),
            "load_snapshot": round(warm_total_ms - warm_pause_ms, 3),
        },
    }

    # Phase 3: Cold restore — kill FC, re-launch, load snapshot.
    _fc_api_request(_API_SOCK, "PUT", "/actions", {"action_type": "SendCtrlAltDel"})
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)

    if os.path.exists(_API_SOCK):
        os.unlink(_API_SOCK)

    t2 = time.monotonic()
    proc2 = subprocess.Popen(
        [_FC_BINARY, "--api-sock", _API_SOCK],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline2 = time.monotonic() + 30
    while not os.path.exists(_API_SOCK):
        if time.monotonic() > deadline2:
            raise TimeoutError("FC API socket did not appear (cold)")
        if proc2.poll() is not None:
            raise RuntimeError(f"FC exited early (cold, rc={proc2.returncode})")
        time.sleep(0.1)
    launch_ms = (time.monotonic() - t2) * 1000

    _configure_vm(_API_SOCK)
    _fc_api_request(_API_SOCK, "PUT", "/snapshot/load", {
        "snapshot_path": mem_path, "mem_file_path": mem_path,
        "enable_diff_snapshots": True,
    })
    _fc_api_request(_API_SOCK, "PUT", "/actions", {"action_type": "InstanceStart"})
    cold_total_ms = (time.monotonic() - t2) * 1000

    results["cold"] = {
        "launch_ms": round(launch_ms, 3),
        "total_restore_ms": round(cold_total_ms, 3),
        "components": {
            "launch_fc": round(launch_ms, 3),
            "configure_vm": round(cold_total_ms - launch_ms, 3),
        },
    }

    # Cleanup.
    proc2.kill()
    proc2.wait(timeout=5)
    if os.path.exists(_API_SOCK):
        os.unlink(_API_SOCK)

except Exception as exc:
    results["error"] = str(exc)[:500]

print(json.dumps(results))
"""

    result = run_on_kvm_py(py_code, timeout=180)
    if result.returncode != 0:
        return {
            "experiment": "restore-latency",
            "error": result.stderr.strip() or "Unknown error",
            "stdout": result.stdout,
        }

    try:
        raw = json.loads(
            result.stdout.strip().split("\n")[-1]
            if "\n" in result.stdout.strip()
            else result.stdout.strip()
        )
    except json.JSONDecodeError:
        raw = {"error": "Failed to parse output", "raw_stdout": result.stdout[:1000]}

    return {
        "experiment": "restore-latency",
        "methodology": {
            "fc_version": "v1.11.0",
            "kernel": str(_FC_KERNEL_PATH),
            "mem_mib": 1024,
            "warm_restore": "Load snapshot while VM is running (no re-launch)",
            "cold_restore": "Kill FC, re-launch, load snapshot, resume",
            "measurement": "Wall-clock via time.monotonic() around FC API calls",
        },
        **raw,
    }


# ---------------------------------------------------------------------------
# Mismatch rate experiment
# ---------------------------------------------------------------------------


def run_mismatch_rate(args: argparse.Namespace) -> dict[str, Any]:
    """Run parity_gate.py across multiple trace files.

    Collects same-runtime (docker->docker) and cross-runtime (docker->FC)
    mismatch rates, plus the ratio.
    """
    trace_dir = Path(
        args.trace_dir or "traces"
    ).resolve()
    trace_files = _discover_trace_files(trace_dir)
    if not trace_files:
        return {
            "experiment": "mismatch-rate",
            "error": f"No trace.jsonl files found under {trace_dir}",
            "num_traces": 0,
        }

    parity_script = Path("scripts/experiments/parity_gate.py")

    per_trace: list[dict[str, Any]] = []
    all_docker_mismatches = 0
    all_fc_mismatches = 0
    all_active_tools = 0
    errors: list[str] = []

    for tf in trace_files:
        # Run parity_gate.py in dry-run mode first to get counts, then full run.
        # For efficiency, we run the full parity gate directly.
        result = subprocess.run(
            [
                sys.executable, str(parity_script),
                "--trace-jsonl", str(tf),
                "--output-dir", f"/tmp/parity-gate-{tf.stem}",
            ],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(Path(__file__).parent.parent.parent),
            env={**os.environ, "PYTHONPATH": "src"},
            check=False,
        )

        # Parse the output for summary stats.
        report = _parse_parity_output(result.stdout, result.stderr)
        if "error" in report:
            errors.append(f"{tf.name}: {report['error']}")
            continue

        per_trace.append({
            "trace_file": str(tf),
            **report,
        })
        all_docker_mismatches += report.get("docker_mismatches", 0)
        all_active_tools += report.get("active_tools", 0)
        all_fc_mismatches += report.get("fc_mismatches", 0)

    # Aggregate.
    docker_rate = all_docker_mismatches / all_active_tools if all_active_tools > 0 else 0.0
    fc_rate = all_fc_mismatches / all_active_tools if all_active_tools > 0 else 0.0
    ratio = (
        all_fc_mismatches / all_docker_mismatches
        if all_docker_mismatches > 0
        else (1.0 if all_fc_mismatches == 0 else float("inf"))
    )

    return {
        "experiment": "mismatch-rate",
        "methodology": {
            "script": "parity_gate.py",
            "description": (
                "Replays each tool_exec action through DockerBackend and "
                "FCBackend, compares normalized outputs against source using "
                "the tiered mismatch oracle"
            ),
            "runtime": "All traces under --trace-dir; no subset selection",
        },
        "num_traces": len(trace_files),
        "num_traces_with_errors": len(errors),
        "errors": errors,
        "aggregate": {
            "total_active_tools": all_active_tools,
            "docker_mismatches": all_docker_mismatches,
            "docker_mismatch_rate": round(docker_rate, 4),
            "fc_mismatches": all_fc_mismatches,
            "fc_mismatch_rate": round(fc_rate, 4),
            "fc_vs_docker_ratio": round(ratio, 4) if ratio != float("inf") else "inf",
        },
        "per_trace": per_trace,
    }


def _parse_parity_output(stdout: str, stderr: str) -> dict[str, Any]:
    """Extract parity report numbers from parity_gate.py output."""
    combined = stdout + "\n" + stderr

    report: dict[str, Any] = {}
    # If the subprocess failed, capture the first meaningful error line.
    if "ERROR:" in combined or "Traceback" in combined:
        for line in combined.splitlines():
            line = line.strip()
            if "ERROR:" in line or "Error:" in line:
                report["error"] = line[:200]
                break
        if "error" not in report:
            report["error"] = "parity_gate.py execution failed (see stderr)"
        return report

    key_map = {
        "total_tools": r"total_tools:\s*(\d+)",
        "active_tools": r"active_tools:\s*(\d+)",
        "docker_mismatches": r"docker_mismatches:\s*(\d+)",
        "docker_mismatch_rate": r"docker_mismatch_rate:\s*([\d.]+)",
        "fc_mismatches": r"fc_mismatches:\s*(\d+)",
        "fc_mismatch_rate": r"fc_mismatch_rate:\s*([\d.]+)",
        "fc_vs_docker_ratio": r"fc_vs_docker_ratio:\s*([\d.]+|inf)",
    }

    for key, pattern in key_map.items():
        m = re.search(pattern, combined)
        if m:
            val = m.group(1)
            if val == "inf":
                report[key] = float("inf")
            elif "." in val:
                report[key] = float(val)
            else:
                report[key] = int(val)

    if not report:
        # Try to parse the JSON-like key: value format.
        for line in combined.splitlines():
            line = line.strip()
            if ":" in line and not line.startswith("#") and not line.startswith("---"):
                parts = line.split(":", 1)
                key = parts[0].strip()
                val = parts[1].strip()
                try:
                    if "." in val:
                        report[key] = float(val)
                    else:
                        report[key] = int(val)
                except ValueError:
                    report[key] = val

    if not report:
        report["error"] = "Could not parse output"

    return report


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    """Run all experiments and produce a comprehensive report."""
    report: dict[str, Any] = {
        "title": "F3+E Eval Campaign Report",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "has_local_kvm": _has_local_kvm(),
        "kvm_host": os.environ.get("KVM_HOST", _KVM_HOST_DEFAULT),
        "experiments": {},
    }

    experiments = [
        ("density", run_density),
        ("checkpoint-traffic", run_checkpoint_traffic),
        ("restore-latency", run_restore_latency),
        ("mismatch-rate", run_mismatch_rate),
    ]

    for name, fn in experiments:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"  Running: {name}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        t0 = time.monotonic()
        try:
            result = fn(args)
            elapsed = time.monotonic() - t0
            result["elapsed_s"] = round(elapsed, 1)
            report["experiments"][name] = result
            print(f"  {name}: completed in {elapsed:.0f}s", file=sys.stderr)
        except Exception as exc:
            elapsed = time.monotonic() - t0
            report["experiments"][name] = {
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": round(elapsed, 1),
            }
            print(f"  {name}: FAILED — {exc}", file=sys.stderr)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="F3+E Eval Campaign — comprehensive JSON report for the paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available --run targets:
  {', '.join(AVAILABLE_RUNS)}

Examples:
  %(prog)s --list
  %(prog)s --run density
  %(prog)s --run all --output report.json
  %(prog)s --run mismatch-rate --trace-dir traces/swe-rebench
        """.strip(),
    )
    parser.add_argument(
        "--run",
        choices=list(AVAILABLE_RUNS),
        default=None,
        help="Experiment to run",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available --run targets and exit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON report to file (default: stdout)",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        help="Directory containing per-task trace.jsonl files (for checkpoint-traffic, mismatch-rate)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Pretty-print JSON output (default: True)",
    )
    parser.add_argument(
        "--no-pretty",
        action="store_false",
        dest="pretty",
        help="Disable pretty-printing",
    )
    args = parser.parse_args()

    if args.list:
        print("Available --run targets:")
        for name in AVAILABLE_RUNS:
            desc = _run_description(name)
            print(f"  {name:25s}  {desc}")
        return

    if args.run is None:
        parser.print_help()
        sys.exit(1)

    # Dispatch.
    runners = {
        "density": run_density,
        "checkpoint-traffic": run_checkpoint_traffic,
        "restore-latency": run_restore_latency,
        "mismatch-rate": run_mismatch_rate,
        "all": run_all,
    }

    runner = runners[args.run]
    report = runner(args)

    # Serialize.
    indent = 2 if args.pretty else None
    json_text = json.dumps(report, indent=indent, default=str, ensure_ascii=False)

    if args.output:
        args.output.write_text(json_text, encoding="utf-8")
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(json_text)


def _run_description(name: str) -> str:
    """Return a one-line description for a run target."""
    descriptions = {
        "density": "Scale 1/2/4/8/16 FC VMs; measure RSS per VM and boot latency vs Docker",
        "checkpoint-traffic": "Three-way checkpoint bytes/turn: file-CAS, memory-diff, block-delta",
        "restore-latency": "FC snapshot restore time (warm + cold) with component breakdown",
        "mismatch-rate": "Parity gate: same-runtime vs cross-runtime mismatch rates across all traces",
        "all": "Run all experiments and produce a comprehensive JSON report",
    }
    return descriptions.get(name, "Unknown experiment")


if __name__ == "__main__":
    main()
