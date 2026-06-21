from __future__ import annotations

import logging
import subprocess
from typing import Any


def sample_nvidia_smi_compute_apps() -> list[dict[str, Any]]:
    """All GPU compute processes via `nvidia-smi --query-compute-apps`.

    Returns rows of {pid:int, gpu_serial:str, memory_used_mib:float}.
    Empty list if nvidia-smi missing or no compute apps running.
    """
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_serial,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        logging.warning("nvidia-smi compute-apps query failed: %s", exc)
        return []
    rows: list[dict[str, Any]] = []
    for raw in output.splitlines():
        if not raw.strip():
            continue
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 3:
            continue
        try:
            rows.append({
                "pid": int(parts[0]),
                "gpu_serial": parts[1],
                "memory_used_mib": float(parts[2]),
            })
        except ValueError:
            logging.warning("nvidia-smi compute-apps malformed row: %r", raw)
            continue
    return rows


def _resolve_gpu_index(gpu_serial: str) -> int:
    """Map gpu_serial → gpu_index. Returns 0 if lookup fails (single-GPU case)."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,gpu_serial", "--format=csv,noheader,nounits"],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 0
    for raw in output.splitlines():
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) >= 2 and parts[1] == gpu_serial:
            try:
                return int(parts[0])
            except ValueError:
                return 0
    return 0


def sample_nvidia_smi_per_pid(pid: int) -> dict[str, Any] | None:
    """Memory used by a specific PID across GPUs.

    Returns {pid, gpu_index, memory_used_mib} for the FIRST matching row,
    or None if PID not present (caller decides whether that is an error).
    """
    apps = sample_nvidia_smi_compute_apps()
    for row in apps:
        if row["pid"] == pid:
            gpu_index = _resolve_gpu_index(row["gpu_serial"])
            return {
                "pid": pid,
                "gpu_index": gpu_index,
                "memory_used_mib": row["memory_used_mib"],
            }
    return None
