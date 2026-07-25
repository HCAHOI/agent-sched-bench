"""Regressions for the Stage-2 collector's pure analysis (bcc-free import)."""

from __future__ import annotations

import sys
from pathlib import Path

_STAGE2 = Path(__file__).resolve().parents[1] / (
    "analysis/development/clause-telemetry-ebpf-stage2-20260725"
)
sys.path.insert(0, str(_STAGE2))

import collector as C  # noqa: E402

_W = C.WINDOW_NS


def _ev(type_, ts, pid, *, tid=None, seq=C.SENTINEL, cpu_ns=0, rss=0, mm=0,
        arg_index=0, arg="", child=0, exit_code=0):
    return {
        "type": type_, "ts_ns": ts, "cgroup_id": 1, "host_pid": pid,
        "host_tid": tid if tid is not None else pid, "exec_seq": seq,
        "cpu_ns": cpu_ns, "rss_pages": rss, "mm_ptr": mm, "arg_index": arg_index,
        "arg": arg, "child_host_pid": child, "exit_code": exit_code,
    }


def test_direct_seq_and_half_open_windows() -> None:
    # pid 100 execs A (seq 0 @0) then B (seq 1 @1000); exits @2000.
    events = [
        _ev("exec_boundary", 0, 100, seq=0),
        _ev("exec_arg", 0, 100, seq=0, arg="A"),
        _ev("exec_boundary", 1000, 100, seq=1),
        _ev("exec_arg", 1000, 100, seq=1, arg="B"),
        _ev("exit_boundary", 2000, 100, seq=1, exit_code=0),
        # a resolved-seq sample carrying seq 0 must attribute to A even though ts
        # falls in B's window (direct-seq before window fallback)
        _ev("perf", 1500, 100, seq=0, cpu_ns=5),
        # a sentinel sample exactly at the boundary ts=1000 belongs to B
        # (half-open: A's window is [0,1000), B's is [1000,2000))
        _ev("perf", 1000, 100, seq=C.SENTINEL, cpu_ns=5),
        # a sentinel sample inside A's window
        _ev("perf", 500, 100, seq=C.SENTINEL, cpu_ns=5),
    ]
    clauses, fork_parent = C._clauses_and_lineage(events)
    per_clause, gaps = C._attribute(events, clauses, fork_parent)
    assert not gaps
    a = per_clause[(100, 0)]
    b = per_clause[(100, 1)]
    # A gets the direct-seq sample (ts 1500) and the in-window sentinel (ts 500)
    assert {s["ts_ns"] for s in a} == {0, 500, 1500}
    # B gets the boundary sentinel (ts 1000), its own exec + exit boundaries
    assert {s["ts_ns"] for s in b} == {1000, 2000}


def test_cpu_delta_apportioned_across_intersected_windows() -> None:
    # one tid, delta 500 cpu_ns over [400ms, 900ms) spans window 0 and window 1
    samples = [
        {"host_tid": 7, "ts_ns": 400_000_000, "cpu_ns": 1000},
        {"host_tid": 7, "ts_ns": 900_000_000, "cpu_ns": 1500},
    ]
    profile = dict(C.cpu_window_profile(samples))
    # overlap: window0 [0,500ms) covers 100ms/500ms -> 100; window1 -> 400
    assert profile == {0: 100, 1: 400}


def test_sentinel_at_terminal_end_is_half_open() -> None:
    # pid 100 execs A (seq 0 @0), exits @1000. A sentinel sample AT ts==t_end
    # (1000) must NOT attribute to A (half-open [0,1000)); the exit boundary at
    # 1000 still lands on A via direct-seq. Guards the removed terminal-inclusive
    # special case in _clause_at.
    events = [
        _ev("exec_boundary", 0, 100, seq=0),
        _ev("exec_arg", 0, 100, seq=0, arg="prog"),
        _ev("exit_boundary", 1000, 100, seq=0, exit_code=0),
        _ev("perf", 1000, 100, seq=C.SENTINEL, cpu_ns=5),  # sentinel at t_end
    ]
    clauses, fork_parent = C._clauses_and_lineage(events)
    per_clause, gaps = C._attribute(events, clauses, fork_parent)
    a = per_clause[(100, 0)]
    # A owns its exec (0) and exit (1000, direct-seq) boundaries only; the
    # sentinel at 1000 falls outside [0,1000) and becomes a coverage gap.
    assert {s["ts_ns"] for s in a} == {0, 1000}
    assert [g["ts_ns"] for g in gaps] == [1000]
    assert [g["exec_seq"] for g in gaps] == [C.SENTINEL]


def test_no_exit_marks_clause_without_causal_end() -> None:
    # terminal clause with no exit boundary -> has_causal_end False (fail closed)
    events = [
        _ev("exec_boundary", 0, 100, seq=0),
        _ev("exec_arg", 0, 100, seq=0, arg="prog"),
        _ev("perf", 500, 100, seq=0, cpu_ns=5),
    ]
    clauses, _ = C._clauses_and_lineage(events)
    assert len(clauses) == 1
    assert clauses[0].has_causal_end is False
