"""Decode Linux BSD process-accounting (``acct`` v3) records.

Per-binary workload attribution: when a coding agent runs a compound shell
command (``... | python check.py | head | wc``), the call-level ``cpu_core_s``
we already collect cannot say how much of that CPU was ``python`` versus
``head``/``wc``. Enabling BSD process accounting inside the task container
makes the kernel append one fixed-size record per process *exit* — including
sub-millisecond processes that sampling misses — carrying the command name,
CPU time, average memory, and the pid/ppid needed to attribute the batch to one
exec call.

This module is the canonical, tested decoder used host-side and for offline
analysis. The in-container replay server (``openclaw_tools._REPLAY_AGENT_SCRIPT``)
inlines a minimal copy of :func:`parse_pacct_v3` / :func:`filter_subtree` because
it runs as a standalone ``python -c`` script in an arbitrary task image with no
access to this repo; keep the two struct layouts in sync.

Record layout is ``struct acct_v3`` from ``<sys/acct.h>`` (64 bytes). Times are
``comp_t`` (13-bit mantissa, 3-bit base-8 exponent) in AHZ ticks; AHZ is a fixed
kernel constant of 100, independent of ``CLK_TCK``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

# flag, ver, tty, exitcode, uid, gid, pid, ppid, btime, etime,
# utime, stime, mem, io, rw, minflt, majflt, swaps, comm[16]
_ACCT_V3_FORMAT = "<bbHIIIIIIfHHHHHHHH16s"
_ACCT_V3_SIZE = struct.calcsize(_ACCT_V3_FORMAT)  # 64
_AHZ = 100.0


@dataclass(frozen=True, slots=True)
class PacctRecord:
    """One process-exit accounting record."""

    comm: str
    pid: int
    ppid: int
    utime_s: float
    stime_s: float
    avg_mem_kb: int
    btime: int
    exitcode: int

    def to_row(self) -> dict[str, Any]:
        """Trace-serializable per-process row (matches the ``per_process`` schema)."""
        return {
            "comm": self.comm,
            "pid": self.pid,
            "ppid": self.ppid,
            "utime_s": round(self.utime_s, 3),
            "stime_s": round(self.stime_s, 3),
            "avg_mem_kb": self.avg_mem_kb,
            "exitcode": self.exitcode,
        }


def _decode_comp_t(value: int) -> int:
    exponent = (value >> 13) & 0x7
    mantissa = value & 0x1FFF
    return mantissa << (3 * exponent)


def parse_pacct_v3(data: bytes) -> list[PacctRecord]:
    """Decode a v3 accounting buffer; a trailing partial record is dropped.

    A partial tail is expected when reading the live file mid-append, so it is
    truncated rather than treated as corruption.
    """
    usable = len(data) - (len(data) % _ACCT_V3_SIZE)
    records: list[PacctRecord] = []
    for offset in range(0, usable, _ACCT_V3_SIZE):
        fields = struct.unpack(_ACCT_V3_FORMAT, data[offset : offset + _ACCT_V3_SIZE])
        version = fields[1]
        if version != 3:
            continue
        (exitcode, pid, ppid, btime) = (fields[3], fields[6], fields[7], fields[8])
        (utime, stime, mem) = (fields[10], fields[11], fields[12])
        comm = fields[18].split(b"\0", 1)[0].decode("latin1")
        records.append(
            PacctRecord(
                comm=comm,
                pid=pid,
                ppid=ppid,
                utime_s=_decode_comp_t(utime) / _AHZ,
                stime_s=_decode_comp_t(stime) / _AHZ,
                avg_mem_kb=_decode_comp_t(mem),
                btime=btime,
                exitcode=exitcode,
            )
        )
    return records


def filter_subtree(records: list[PacctRecord], root_pid: int) -> list[PacctRecord]:
    """Keep only records in ``root_pid``'s process subtree (itself + descendants).

    ``ppid`` links are resolved within ``records`` alone — one exec call's exit
    batch. A background process left running by an earlier command, whose
    ancestor shell already exited and is therefore absent from this batch, has
    no path back to ``root_pid`` and is excluded. This is the disambiguation for
    the rare case where a lingering ``&`` job exits during a later exec's window.
    """
    kept = {root_pid}
    changed = True
    while changed:
        changed = False
        for record in records:
            if record.pid not in kept and record.ppid in kept:
                kept.add(record.pid)
                changed = True
    return [record for record in records if record.pid in kept]


def aggregate_by_comm(records: list[PacctRecord]) -> list[dict[str, Any]]:
    """Per-binary rollup (count, summed CPU, peak avg-memory), heaviest first.

    Analysis/inspection helper — the trace stores raw per-process rows; the
    per-binary cost table is derived from many calls offline.
    """
    totals: dict[str, dict[str, Any]] = {}
    for record in records:
        entry = totals.setdefault(
            record.comm,
            {"comm": record.comm, "count": 0, "utime_s": 0.0, "stime_s": 0.0, "avg_mem_kb": 0},
        )
        entry["count"] += 1
        entry["utime_s"] += record.utime_s
        entry["stime_s"] += record.stime_s
        entry["avg_mem_kb"] = max(entry["avg_mem_kb"], record.avg_mem_kb)
    for entry in totals.values():
        entry["utime_s"] = round(entry["utime_s"], 3)
        entry["stime_s"] = round(entry["stime_s"], 3)
    return sorted(totals.values(), key=lambda e: -(e["utime_s"] + e["stime_s"]))


__all__ = [
    "PacctRecord",
    "parse_pacct_v3",
    "filter_subtree",
    "aggregate_by_comm",
    "_ACCT_V3_SIZE",
]
