"""Tests for the BSD process-accounting (acct v3) decoder used for per-binary
workload attribution.

The fixture ``pacct_v3_sample.acct`` is a real accounting file captured inside a
``--cap-add SYS_PACCT`` container running
``for f in $(ls src); do python3 check.py $f | head -20 | wc -l; done`` plus one
CPU-heavy ``python3`` loop. Known contents: 10 python3, 8 head, 8 wc, 1 ls,
1 bash (28 records).
"""

from __future__ import annotations

import struct
from pathlib import Path

from trace_collect.pacct import (
    _ACCT_V3_SIZE,
    PacctRecord,
    aggregate_by_comm,
    filter_subtree,
    parse_pacct_v3,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "pacct_v3_sample.acct"


def test_record_size_is_64_bytes():
    assert _ACCT_V3_SIZE == 64


def test_parse_real_fixture_counts_and_comm():
    records = parse_pacct_v3(_FIXTURE.read_bytes())
    assert len(records) == 28
    counts = {}
    for record in records:
        counts[record.comm] = counts.get(record.comm, 0) + 1
    assert counts == {"python3": 10, "head": 8, "wc": 8, "ls": 1, "bash": 1}


def test_python_dominates_light_binaries():
    agg = {a["comm"]: a for a in aggregate_by_comm(parse_pacct_v3(_FIXTURE.read_bytes()))}
    # The whole point of attribution: python CPU >> head/wc CPU, and memory too.
    py_cpu = agg["python3"]["utime_s"] + agg["python3"]["stime_s"]
    for light in ("head", "wc", "ls"):
        light_cpu = agg[light]["utime_s"] + agg[light]["stime_s"]
        assert py_cpu > light_cpu
        assert agg["python3"]["avg_mem_kb"] > agg[light]["avg_mem_kb"]


def test_partial_trailing_record_is_dropped():
    data = _FIXTURE.read_bytes()
    # 28 whole records + 10 stray bytes (a mid-append tail) -> still 28.
    assert len(parse_pacct_v3(data + b"\x00" * 10)) == 28
    assert parse_pacct_v3(b"\x00" * 10) == []


def test_comp_t_exponent_decoding():
    # comp_t: value = mantissa << (3 * exponent). Encode 5000 ticks and check the
    # decoder recovers 5000/AHZ = 50.0 s of utime. 5000 = 625 << 3, exp=1.
    encoded = (1 << 13) | 625
    assert _decode_via_record(utime=encoded).utime_s == 50.0


def test_filter_subtree_excludes_lingering_background_job():
    # shell(100) -> python(101) -> worker(102); plus a lingering bg proc(200)
    # from an earlier command whose parent shell(199) is not in this batch.
    def rec(pid, ppid, comm):
        return PacctRecord(comm, pid, ppid, 0.0, 0.0, 0, 0, 0)

    batch = [
        rec(102, 101, "cc1"),
        rec(101, 100, "python3"),
        rec(100, 1, "bash"),
        rec(200, 199, "sleep"),  # lingering, parent absent from batch
    ]
    kept = {r.pid for r in filter_subtree(batch, root_pid=100)}
    assert kept == {100, 101, 102}


def _decode_via_record(*, utime: int) -> PacctRecord:
    """Round-trip one utime comp_t through the real struct parser."""
    fields = [0] * 19
    fields[0] = 0  # flag
    fields[1] = 3  # version
    fields[10] = utime  # utime comp_t
    fields[18] = b"x"  # comm
    packed = struct.pack("<bbHIIIIIIfHHHHHHHH16s", *fields)
    (record,) = parse_pacct_v3(packed)
    return record
