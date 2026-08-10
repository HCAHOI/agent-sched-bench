"""Regressions for the collector's pure analysis (BCC-free import)."""

from __future__ import annotations

import os

import pytest

from tool_resource import telemetry as C

_W = C.WINDOW_NS


def test_bpf_lifecycle_keeps_identity_until_free_and_clears_new_child() -> None:
    fork_probe = C.BPF_PROGRAM.split(
        "RAW_TRACEPOINT_PROBE(sched_process_fork)", 1
    )[1].split("TRACEPOINT_PROBE(sched, sched_process_exit)", 1)[0]
    exit_probe = C.BPF_PROGRAM.split(
        "TRACEPOINT_PROBE(sched, sched_process_exit)", 1
    )[1].split("RAW_TRACEPOINT_PROBE(sched_process_free)", 1)[0]
    free_probe = C.BPF_PROGRAM.split(
        "RAW_TRACEPOINT_PROBE(sched_process_free)", 1
    )[1].split("int on_cpu_clock", 1)[0]

    # The fork event carries no argv payload, so it is emitted on the small
    # ring; what matters here is unchanged -- the child's inherited identity is
    # cleared before the event goes out.
    assert fork_probe.index("current_seq.delete(&child_key)") < (
        fork_probe.index("events_small.ringbuf_reserve")
    )
    assert fork_probe.index("pending_seq.delete(&child_key)") < (
        fork_probe.index("events_small.ringbuf_reserve")
    )
    assert "current_seq.delete" not in exit_probe
    assert "pending_seq.delete" not in exit_probe
    assert "if (!wanted())" not in free_probe
    assert ".task_ptr = (u64)task" in free_probe
    assert "current_seq.delete(&task_key)" in free_probe
    assert "pending_seq.delete(&task_key)" in free_probe
    assert "BPF_HASH(current_seq, struct task_key_t, u64)" in C.BPF_PROGRAM
    assert "u64 argv_ptr;" in C.BPF_PROGRAM
    assert (
        "BPF_HASH(pending_seq, struct task_key_t, struct pending_exec_t)"
        in C.BPF_PROGRAM
    )


def _ev(type_, ts, pid, *, tid=None, seq=C.SENTINEL, cpu_ns=0, rss=0, mm=0,
        arg_index=0, arg="", child=0, child_tid=0, exit_code=0,
        io_read=0, io_write=0, io_cancelled=0):
    return {
        "type": type_, "ts_ns": ts, "cgroup_id": 1, "host_pid": pid,
        "host_tid": tid if tid is not None else pid, "exec_seq": seq,
        "cpu_ns": cpu_ns, "rss_pages": rss, "mm_ptr": mm, "arg_index": arg_index,
        "arg": arg, "child_host_pid": child, "child_host_tid": child_tid,
        "exit_code": exit_code, "io_read_bytes": io_read,
        "io_write_bytes": io_write,
        "io_cancelled_write_bytes": io_cancelled,
    }


def test_direct_seq_precedes_post_exec_sentinel_failure() -> None:
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
        # Sentinel samples after this PID has successfully exec'd are never
        # repaired from its window; losing current_seq after exec is fatal.
        _ev("perf", 1000, 100, seq=C.SENTINEL, cpu_ns=5),
        _ev("perf", 500, 100, seq=C.SENTINEL, cpu_ns=5),
    ]
    clauses, fork_parent = C._clauses_and_lineage(events)
    per_clause, gaps = C._attribute(events, clauses, fork_parent)
    a = per_clause[(100, 0)]
    b = per_clause[(100, 1)]
    assert {s["ts_ns"] for s in a} == {0, 1500}
    assert {s["ts_ns"] for s in b} == {1000, 2000}
    assert [gap["ts_ns"] for gap in gaps] == [1000, 500]
    assert {gap["reason"] for gap in gaps} == {
        "sentinel_after_successful_exec"
    }


def test_direct_entry_child_pre_exec_is_structural_setup() -> None:
    events = [
        _ev("fork", 10, 50, child=100, child_tid=100),
        _ev("perf", 15, 100, cpu_ns=1, rss=1, mm=10),
        _ev("exec_arg", 20, 100, seq=0, arg="prog"),
        _ev("exec_boundary", 20, 100, seq=0),
        _ev("exit_boundary", 30, 100, seq=0),
    ]
    clauses, fork_parent = C._clauses_and_lineage(events)
    per_clause, gaps = C._attribute(
        events,
        clauses,
        fork_parent,
        entry_pid=50,
    )

    assert [sample["type"] for sample in per_clause[(100, 0)]] == [
        "exec_boundary",
        "exit_boundary",
    ]
    assert [
        {
            "reason": gap["reason"],
            "fork_ancestry": gap["fork_ancestry"],
            "fork_ts_ns": gap["fork_ts_ns"],
        }
        for gap in gaps
    ] == [
        {
            "reason": "entry_fork_pre_exec_structural_setup",
            "fork_ancestry": [100, 50],
            "fork_ts_ns": 10,
        }
    ]


def test_initial_pending_exec_sample_without_fork_is_structural_setup() -> None:
    events = [
        _ev("exec_arg", 10, 100, seq=0, arg="sh"),
        _ev("perf", 15, 100, cpu_ns=1, rss=1, mm=10),
        _ev("exec_boundary", 20, 100, seq=0),
        _ev("exit_boundary", 30, 100, seq=0),
    ]
    clauses, fork_parent = C._clauses_and_lineage(events)
    per_clause, gaps = C._attribute(
        events,
        clauses,
        fork_parent,
        entry_pid=50,
    )

    assert [sample["type"] for sample in per_clause[(100, 0)]] == [
        "exec_boundary",
        "exit_boundary",
    ]
    assert len(gaps) == 1
    assert gaps[0]["reason"] == (
        "initial_exec_pending_pre_boundary_structural_setup"
    )
    assert gaps[0]["pending_exec_evidence"] == {
        "host_pid": 100,
        "host_tid": 100,
        "pending_exec_seq": 0,
        "exec_arg_start_ns": 10,
        "sample_ts_ns": 15,
        "successful_exec_boundary_ns": 20,
    }


def test_failed_pending_exec_sample_without_fork_remains_fatal() -> None:
    events = [
        _ev("exec_arg", 10, 100, seq=0, arg="missing"),
        _ev("perf", 15, 100, cpu_ns=1, rss=1, mm=10),
        _ev("failed_exec_attempt", 20, 100, seq=0, exit_code=2),
    ]
    clauses, fork_parent = C._clauses_and_lineage(events)
    _, gaps = C._attribute(events, clauses, fork_parent, entry_pid=50)

    assert len(gaps) == 1
    assert gaps[0]["reason"] == "sentinel_pre_exec_missing_fork_ancestry"
    assert "pending_exec_evidence" not in gaps[0]


def test_fork_only_intermediary_to_entry_is_structural_setup() -> None:
    events = [
        _ev("fork", 10, 50, child=60, child_tid=60),
        _ev("fork", 12, 60, child=100, child_tid=100),
        _ev("perf", 15, 100, cpu_ns=1, rss=1, mm=10),
        _ev("exec_arg", 20, 100, seq=0, arg="prog"),
        _ev("exec_boundary", 20, 100, seq=0),
        _ev("exit_boundary", 30, 100, seq=0),
    ]
    clauses, fork_parent = C._clauses_and_lineage(events)
    _, gaps = C._attribute(
        events,
        clauses,
        fork_parent,
        entry_pid=50,
    )

    assert gaps[0]["reason"] == "entry_fork_pre_exec_structural_setup"
    assert gaps[0]["fork_ancestry"] == [100, 60, 50]


def test_pre_exec_sample_inherits_active_exec_once() -> None:
    events = [
        _ev("fork", 10, 50, child=100, child_tid=100),
        _ev("exec_arg", 20, 100, seq=0, arg="root"),
        _ev("exec_boundary", 20, 100, seq=0, rss=100, mm=10),
        _ev(
            "fork",
            100_000_000,
            100,
            child=200,
            child_tid=200,
        ),
        _ev(
            "perf",
            400_000_000,
            100,
            seq=0,
            rss=100,
            mm=10,
        ),
        _ev(
            "perf",
            400_000_000,
            200,
            cpu_ns=100_000_000,
            rss=200,
            mm=10,
        ),
        _ev("exec_arg", 600_000_000, 200, seq=1, arg="child"),
        _ev(
            "exec_boundary",
            600_000_000,
            200,
            seq=1,
            cpu_ns=200_000_000,
            rss=50,
            mm=20,
        ),
        _ev(
            "exit_boundary",
            1_000_000_000,
            200,
            seq=1,
            cpu_ns=250_000_000,
            rss=50,
            mm=20,
        ),
        _ev(
            "exit_boundary",
            1_200_000_000,
            100,
            seq=0,
            rss=100,
            mm=10,
        ),
    ]
    metrics, gaps = C.analyze(
        C.RawRun(
            1,
            8.0,
            0,
            1_200_000_000,
            0,
            0,
            2,
            0,
            0,
            True,
            events,
        ),
        entry_pid=50,
    )

    assert gaps == []
    root = next(metric for metric in metrics if metric.host_pid == 100)
    child = next(metric for metric in metrics if metric.host_pid == 200)
    assert sum(cpu_ns for _, cpu_ns in root.cpu_windows) == 200_000_000
    assert sum(cpu_ns for _, cpu_ns in child.cpu_windows) == 50_000_000
    assert root.sampled_peak_rss_mb == pytest.approx(200 * C.PAGE / 1e6)
    attribution = root.provenance["sample_attribution"]
    assert attribution["inherited_owner_sample_count"] == 1
    # Rows reference deduplicated evidence tables; resolving one restores the
    # full record, including the fields recovered from the fork chain.
    inherited = C.resolve_inherited_owner_sample(
        attribution["inherited_owner_samples"][0], attribution
    )
    assert inherited["original_host_pid"] == 200
    assert inherited["original_host_tid"] == 200
    assert inherited["original_exec_seq"] == C.SENTINEL
    assert inherited["owner_host_pid"] == 100
    assert inherited["owner_exec_seq"] == 0
    assert inherited["fork_ancestry"] == [200, 100, 50]
    # Against the fixture's own fork timestamps, not against the chain the
    # resolver derived these from -- comparing a derived field to its source
    # would restate the derivation rather than test it.
    assert inherited["fork_chain_records"] == [
        {"child_id": 200, "parent_pid": 100, "ts_ns": 100_000_000},
        {"child_id": 100, "parent_pid": 50, "ts_ns": 10},
    ]
    assert inherited["fork_ts_ns"] == 100_000_000


def test_new_thread_pre_exec_sample_inherits_active_tgid_image() -> None:
    events = [
        _ev("fork", 10, 50, child=100, child_tid=100),
        _ev(
            "exec_boundary",
            20,
            100,
            seq=0,
            rss=100,
            mm=10,
        ),
        _ev("exec_arg", 20, 100, seq=0, arg="root"),
        _ev("fork", 30, 100, child=101, child_tid=101),
        _ev(
            "perf",
            40,
            100,
            tid=101,
            cpu_ns=100,
            rss=200,
            mm=10,
        ),
        _ev(
            "exit_boundary",
            60,
            100,
            tid=101,
            cpu_ns=200,
            rss=200,
            mm=10,
        ),
        _ev(
            "exit_boundary",
            1_200_000_000,
            100,
            seq=0,
            rss=100,
            mm=10,
        ),
    ]

    metrics, gaps = C.analyze(
        C.RawRun(1, 8.0, 0, 1_200_000_000, 0, 0, 1, 0, 0, True, events),
        entry_pid=50,
    )

    assert gaps == []
    root = metrics[0]
    assert sum(cpu_ns for _, cpu_ns in root.cpu_windows) == 200
    assert root.sampled_peak_rss_mb == pytest.approx(200 * C.PAGE / 1e6)
    inherited = root.provenance["sample_attribution"]
    assert inherited["inherited_owner_sample_count"] == 2
    assert {
        (
            sample["original_host_pid"],
            sample["original_host_tid"],
            tuple(sample["fork_ancestry"]),
            sample["owner_host_pid"],
        )
        for sample in (
            C.resolve_inherited_owner_sample(row, inherited)
            for row in inherited["inherited_owner_samples"]
        )
    } == {(100, 101, (101, 100, 50), 100)}
    # Both samples share one lineage, so the chain is stored exactly once.
    assert len(inherited["fork_chains"]) == 1


def test_pre_exec_ambiguous_fork_ancestry_is_fatal() -> None:
    events = [
        _ev("fork", 10, 50, child=100, child_tid=100),
        _ev("fork", 11, 51, child=100, child_tid=100),
        _ev("perf", 15, 100, cpu_ns=1),
        _ev("exec_arg", 20, 100, seq=0, arg="prog"),
        _ev("exec_boundary", 20, 100, seq=0),
        _ev("exit_boundary", 30, 100, seq=0),
    ]
    clauses, fork_parent = C._clauses_and_lineage(events)
    _, gaps = C._attribute(
        events,
        clauses,
        fork_parent,
        entry_pid=50,
    )

    assert [gap["reason"] for gap in gaps] == [
        "sentinel_pre_exec_ambiguous_fork_ancestry"
    ]


def test_pre_exec_active_owner_does_not_hide_ambiguity_above_it() -> None:
    events = [
        _ev("fork", 10, 50, child=100, child_tid=100),
        _ev("fork", 11, 51, child=100, child_tid=100),
        _ev("exec_arg", 20, 100, seq=0, arg="root"),
        _ev("exec_boundary", 20, 100, seq=0),
        _ev("fork", 30, 100, child=200, child_tid=200),
        _ev("perf", 40, 200, cpu_ns=1),
        _ev("exit_boundary", 50, 100, seq=0),
    ]
    clauses, fork_parent = C._clauses_and_lineage(events)
    _, gaps = C._attribute(
        events,
        clauses,
        fork_parent,
        entry_pid=50,
    )

    assert len(gaps) == 1
    assert gaps[0]["reason"] == "sentinel_pre_exec_ambiguous_fork_ancestry"
    assert gaps[0]["fork_ancestry"] == [200, 100]


def test_pre_exec_ancestry_edges_must_precede_the_child_fork() -> None:
    events = [
        _ev("fork", 10, 100, child=200, child_tid=200),
        _ev("fork", 30, 50, child=100, child_tid=100),
        _ev("exec_arg", 35, 100, seq=0, arg="root"),
        _ev("exec_boundary", 35, 100, seq=0),
        _ev("perf", 40, 200, cpu_ns=1),
        _ev("exit_boundary", 50, 100, seq=0),
    ]
    clauses, fork_parent = C._clauses_and_lineage(events)
    per_clause, gaps = C._attribute(
        events,
        clauses,
        fork_parent,
        entry_pid=50,
    )

    assert all(
        sample["host_pid"] != 200
        for sample in per_clause[(100, 0)]
    )
    assert gaps[0]["reason"] == "sentinel_pre_exec_missing_fork_ancestry"
    assert gaps[0]["fork_ancestry"] == [200, 100]
    assert gaps[0]["fork_chain_records"] == [
        {"child_id": 200, "parent_pid": 100, "ts_ns": 10}
    ]
    assert gaps[0]["fork_resolution_failure"] == {
        "failure_kind": "missing_generation",
        "child_id": 100,
        "timestamp_bound_ns": 10,
        "eligible_records": [],
        "rejected_records": [{"parent_pid": 50, "ts_ns": 30}],
    }


def test_repeated_same_parent_fork_generation_is_ambiguous() -> None:
    events = [
        _ev("fork", 10, 50, child=100, child_tid=100),
        _ev("fork", 11, 50, child=100, child_tid=100),
        _ev("perf", 15, 100, cpu_ns=1),
        _ev("exec_arg", 20, 100, seq=0, arg="prog"),
        _ev("exec_boundary", 20, 100, seq=0),
        _ev("exit_boundary", 30, 100, seq=0),
    ]
    clauses, fork_parent = C._clauses_and_lineage(events)
    _, gaps = C._attribute(
        events,
        clauses,
        fork_parent,
        entry_pid=50,
    )

    assert gaps[0]["reason"] == "sentinel_pre_exec_ambiguous_fork_ancestry"
    assert gaps[0]["fork_ancestry"] == [100]
    assert gaps[0]["fork_chain_records"] == []
    assert gaps[0]["fork_resolution_failure"] == {
        "failure_kind": "ambiguous_generation",
        "child_id": 100,
        "timestamp_bound_ns": 15,
        "eligible_records": [
            {"parent_pid": 50, "ts_ns": 10},
            {"parent_pid": 50, "ts_ns": 11},
        ],
        "rejected_records": [],
    }


def test_cyclic_pre_exec_ancestry_persists_the_failing_edge() -> None:
    events = [
        _ev("fork", 30, 200, child=300, child_tid=300),
        _ev("fork", 31, 300, child=200, child_tid=200),
        _ev("perf", 40, 200, cpu_ns=1),
    ]
    clauses, fork_parent = C._clauses_and_lineage(events)
    _, gaps = C._attribute(
        events,
        clauses,
        fork_parent,
        entry_pid=50,
    )

    assert gaps[0]["reason"] == "sentinel_pre_exec_ambiguous_fork_ancestry"
    assert gaps[0]["fork_ancestry"] == [200, 300]
    assert gaps[0]["fork_chain_records"] == [
        {"child_id": 200, "parent_pid": 300, "ts_ns": 31}
    ]
    assert gaps[0]["fork_resolution_failure"] == {
        "failure_kind": "cyclic_parent",
        "child_id": 300,
        "timestamp_bound_ns": 31,
        "eligible_records": [{"parent_pid": 200, "ts_ns": 30}],
        "rejected_records": [],
    }


def test_nonpositive_pre_exec_parent_persists_the_failing_edge() -> None:
    events = [
        _ev("fork", 30, 0, child=200, child_tid=200),
        _ev("perf", 40, 200, cpu_ns=1),
    ]
    clauses, fork_parent = C._clauses_and_lineage(events)
    _, gaps = C._attribute(
        events,
        clauses,
        fork_parent,
        entry_pid=50,
    )

    assert gaps[0]["fork_resolution_failure"] == {
        "failure_kind": "nonpositive_parent",
        "child_id": 200,
        "timestamp_bound_ns": 40,
        "eligible_records": [{"parent_pid": 0, "ts_ns": 30}],
        "rejected_records": [],
    }


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


def test_resolved_terminal_sample_after_end_is_identity_only() -> None:
    events = [
        _ev(
            "exec_boundary",
            0,
            100,
            seq=7,
            cpu_ns=10,
            rss=10,
            mm=1,
        ),
        _ev("exec_arg", 0, 100, seq=7, arg="prog"),
        _ev(
            "exit_boundary",
            1_000,
            100,
            seq=7,
            cpu_ns=20,
            rss=20,
            mm=1,
            io_read=100,
        ),
        _ev(
            "perf",
            1_025,
            100,
            seq=7,
            cpu_ns=1_000,
            rss=1_000,
            mm=1,
            io_read=10_000,
        ),
    ]
    metrics, gaps = C.analyze(
        C.RawRun(1, 8.0, 0, 1_025, 0, 0, 1, 0, 0, True, events)
    )

    assert gaps == []
    metric = metrics[0]
    assert metric.t_end_ns == 1_000
    assert sum(cpu_ns for _, cpu_ns in metric.cpu_windows) == 10
    assert metric.rss_bins == (
        (0, 1, 10 * C.PAGE / 1e6),
        (0, 1, 20 * C.PAGE / 1e6),
    )
    assert metric.disk_read_bytes_total == 100
    assert metric.provenance["identity_only_samples"] == [
        {
            "type": "perf",
            "ts_ns": 1_025,
            "host_pid": 100,
            "host_tid": 100,
            "exec_seq": 7,
            "reason": "outside_half_open_exec_window",
            "t_exec_ns": 0,
            "t_end_ns": 1_000,
            "offset_from_end_ns": 25,
        }
    ]


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


def test_rss_conversion_uses_host_page_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert C.PAGE == os.sysconf("SC_PAGE_SIZE")
    monkeypatch.setattr(C, "PAGE", 65_536)
    assert C.rss_bin_profile(
        [{"ts_ns": 0, "mm_ptr": 1, "rss_pages": 2}]
    ) == ((0, 1, 0.131072),)


def test_disk_io_uses_disjoint_exec_baselines_not_scalar_peaks() -> None:
    events = [
        _ev(
            "exec_boundary", 0, 100, seq=0,
            io_read=100, io_write=200, io_cancelled=10,
        ),
        _ev("exec_arg", 0, 100, seq=0, arg="A"),
        _ev(
            "perf", 500, 100, seq=0,
            io_read=140, io_write=240, io_cancelled=10,
        ),
        _ev(
            "exec_boundary", 1000, 100, seq=1,
            io_read=180, io_write=300, io_cancelled=20,
        ),
        _ev("exec_arg", 1000, 100, seq=1, arg="B"),
        _ev(
            "exit_boundary", 2000, 100, seq=1,
            io_read=230, io_write=420, io_cancelled=25,
        ),
    ]
    metrics, gaps = C.analyze(
        C.RawRun(1, 8.0, 0, 2000, 0, 0, 1, 0, 0, True, events)
    )
    assert gaps == []
    assert [
        (
            metric.disk_read_bytes_total,
            metric.disk_write_bytes_total,
            metric.disk_cancelled_write_bytes_total,
            metric.disk_io_reason,
        )
        for metric in metrics
    ] == [(80, 100, 10, "ok"), (50, 120, 5, "ok")]


def test_disk_io_new_forked_tid_uses_zero_baseline() -> None:
    events = [
        _ev("fork", 0, 50, child=100, child_tid=100),
        _ev(
            "exec_boundary", 1, 100, seq=0,
            io_read=100, io_write=200,
        ),
        _ev("exec_arg", 1, 100, seq=0, arg="A"),
        _ev("fork", 100, 100, child=101, child_tid=101),
        _ev(
            "exit_boundary", 500, 101, tid=101,
            io_read=300, io_write=400, io_cancelled=50,
        ),
        _ev(
            "exit_boundary", 1000, 100, seq=0,
            io_read=110, io_write=220,
        ),
    ]
    metrics, gaps = C.analyze(
        C.RawRun(1, 8.0, 0, 1000, 0, 0, 0, 0, 0, True, events),
        entry_pid=50,
    )
    assert gaps == []
    assert len(metrics) == 1
    assert (
        metrics[0].disk_read_bytes_total,
        metrics[0].disk_write_bytes_total,
        metrics[0].disk_cancelled_write_bytes_total,
    ) == (310, 420, 50)
    assert metrics[0].provenance["disk_io"]["zero_fork_baseline_tids"] == [101]


def _spoolable(event: dict) -> dict:
    """_ev omits the fields the packed spool record needs; default them."""
    return {
        "hiwater_pages": 0, "parent_host_pid": 0, "arg_chunk_index": 0,
        "arg_flags": 0, **event,
    }


def _stream_with_every_event_type() -> tuple[list[dict], int]:
    """One pid execs twice with argv, meta and a failed attempt, forks a child
    and a thread, samples, then exits -- so every declared event type appears."""
    events = [
        _ev("fork", 10, 100, child=101, child_tid=101),
        _ev("exec_boundary", 20, 101, seq=1, cpu_ns=10, rss=100, mm=1616),
        _ev("exec_arg", 20, 101, seq=1, arg_index=0, arg="/usr/bin/gcc"),
        _ev("exec_arg", 20, 101, seq=1, arg_index=1, arg="-c"),
        _ev("exec_meta", 20, 101, seq=1, arg="/usr/bin/gcc"),
        _ev("bprm_meta", 20, 101, seq=1, arg="/usr/bin/gcc", exit_code=2),
        _ev("interp_meta", 20, 101, seq=1, arg="/lib/ld.so"),
        _ev("perf", 40, 101, seq=1, cpu_ns=500, rss=200, mm=1616),
        _ev("fork", 50, 101, child=102, child_tid=102),
        _ev("fork", 55, 101, child=101, child_tid=9101),
        _ev("failed_exec_attempt", 60, 101, seq=1, exit_code=2, arg="/bin/nope"),
        _ev("exec_boundary", 70, 101, seq=2, cpu_ns=800, rss=150, mm=1616),
        _ev("exec_arg", 70, 101, seq=2, arg_index=0, arg="/usr/bin/ld"),
        _ev("perf", 90, 101, seq=2, cpu_ns=900, rss=250, mm=1616),
        _ev("exit_boundary", 120, 101, seq=2, cpu_ns=1000, mm=1616),
        # pid 102 execs and never exits, so its clause is bounded by the last
        # timestamp in the stream...
        _ev("exec_boundary", 130, 102, seq=3, cpu_ns=5, rss=80, mm=1632),
        _ev("exec_arg", 130, 102, seq=3, arg_index=0, arg="/bin/sleep"),
        # ...which is a perf event, a type the lineage view drops. A view that
        # reported its own maximum would end that clause at 130 instead of 500.
        _ev("perf", 500, 102, seq=3, cpu_ns=10, rss=5, mm=1632),
    ]
    return [_spoolable(event) for event in events], 500


def test_type_filtered_views_match_the_unfiltered_source(tmp_path) -> None:
    """A pass handed only its declared event types must produce exactly what it
    produces from the whole stream. This is the failure the filter can cause:
    a type missing from one of the declared sets silently drops evidence."""
    events, last_ts = _stream_with_every_event_type()
    types = {event["type"] for event in events}
    assert types == set(C.TYPE_CODES), "stream must exercise every event type"

    spool = C._EventSpool(tmp_path)
    for event in events:
        spool.append(event)
    source = C._sorted_event_source(
        spool.snapshot(), started_ns=0, ended_ns=last_ts, cgroup_id=1,
        directory=tmp_path,
    )
    try:
        assert [dict(row) for row in source.of_types(frozenset(types))] == [
            dict(row) for row in source
        ]
        assert source.max_ts_ns == last_ts

        full_argv = C._captured_argv(source)
        view_argv = C._captured_argv(source.of_types(C._ARGV_EVENT_TYPES))
        assert view_argv == full_argv

        full = C._clauses_and_lineage(source, full_argv)
        view = C._clauses_and_lineage(
            source.of_types(C._LINEAGE_EVENT_TYPES), full_argv
        )
        assert view == full
        # The unterminated clause is bounded by an event the lineage view drops.
        assert any(not clause.has_causal_end for clause in full[0])
        assert max(clause.t_end_ns for clause in full[0]) == last_ts

        assert C._fork_io_baselines(
            source.of_types(C._FORK_EVENT_TYPES), *full
        ) == C._fork_io_baselines(source, *full)
    finally:
        source.close()
        spool.close()


def _reserve_blocks() -> list[tuple[str, str]]:
    """Each ringbuf reserve..submit block in the BPF program as (ring, body)."""
    import re

    blocks = []
    for match in re.finditer(
        r"struct (event_t|event_small_t) \*e = (events|events_small)"
        r"\.ringbuf_reserve",
        C.BPF_PROGRAM,
    ):
        body = C.BPF_PROGRAM[match.start() : C.BPF_PROGRAM.index(
            "ringbuf_submit", match.start()
        )]
        record, ring = match.group(1), match.group(2)
        assert (record == "event_t") == (ring == "events"), (
            f"{ring} reserved as {record}"
        )
        blocks.append((ring, body))
    return blocks


def test_only_the_argv_ring_carries_an_argv_payload() -> None:
    """The small ring's record has no ``arg`` member and its decode never reads
    one, so an emitter that fills a payload must stay on the wide ring and a
    type _event_row unpacks a payload for must never be emitted on the small
    one. Both mistakes are silent: the first loses the argv, the second makes
    every argv word come back empty."""
    payload_types = {
        "TYPE_EXEC_ARG", "TYPE_EXEC_META", "TYPE_BPRM_META", "TYPE_INTERP_META",
    }
    # _event_row extracts a payload for exactly these type codes.
    assert {C.TYPE_CODES[C.TYPE_NAMES[code]] for code in (1, 7, 8, 9)} == {
        C.TYPE_CODES[name.removeprefix("TYPE_").lower()] for name in payload_types
    }

    blocks = _reserve_blocks()
    assert len(blocks) >= 8, "expected every emitter to be found"
    assert {ring for ring, _ in blocks} == {"events", "events_small"}

    import re

    wide = 0
    for ring, body in blocks:
        emitted = set(re.findall(r"e->type\s*=\s*[^;]*?(TYPE_\w+)", body))
        emitted |= set(re.findall(r":\s*(TYPE_\w+)", body))
        if ring == "events_small":
            assert not re.search(r"e->arg\b", body), "small-ring emitter fills a payload"
            assert not emitted & payload_types, (
                f"payload type {sorted(emitted & payload_types)} on the small ring"
            )
            continue
        wide += 1
        # The converse, and the one that silently costs rather than breaks:
        # a type with no payload sitting on the wide ring pays 640 bytes a
        # record instead of 128, which is the entire point of the split. A
        # block that sets its type from a parameter must at least fill a
        # payload to belong here.
        assert emitted or "e->arg" in body, (
            "wide-ring emitter neither names a type nor fills a payload"
        )
        assert not emitted - payload_types, (
            f"non-payload type {sorted(emitted - payload_types)} on the wide ring"
        )
    assert wide == 4, f"expected 4 payload emitters on the wide ring, found {wide}"
    # fill_counters serves only small-ring events; letting it take event_t
    # again would quietly re-widen every counter event.
    assert "static void fill_counters(struct event_small_t *e" in C.BPF_PROGRAM
    assert "char arg[ARG_BYTES];" in C.BPF_PROGRAM
    small = C.BPF_PROGRAM.split("struct event_small_t {", 1)[1].split("};", 1)[0]
    assert "arg" not in small, "small record grew a payload"
