from __future__ import annotations

import importlib.util
import os
import shlex
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from tool_resource import telemetry as C
from tool_resource.telemetry import (
    _failed_exec_attempt_records,
    analyze,
    collect_case,
)

pytestmark = pytest.mark.skipif(
    os.geteuid() != 0 or importlib.util.find_spec("bcc") is None,
    reason="live clause telemetry requires root and BCC",
)


def _collect(case: tuple[str, int, str]):
    tag, blocks, output_path = case
    output = shlex.quote(output_path)
    command = (
        f"dd if=/dev/zero of={output} bs=4096 count={blocks} conv=fsync status=none"
    )
    return tag, blocks, collect_case(command, f"io_{tag}")


def _assert_only_harness_root_pre_exec_gaps(run, gaps) -> None:
    for gap in gaps:
        assert gap["exec_seq"] == C.SENTINEL
        assert gap["reason"] in {
            "entry_fork_pre_exec_structural_setup",
            "initial_exec_pending_pre_boundary_structural_setup",
            "sentinel_pre_exec_missing_fork_ancestry",
        }
        if gap["reason"] == "sentinel_pre_exec_missing_fork_ancestry":
            failure = gap["fork_resolution_failure"]
            assert failure["failure_kind"] == "missing_generation"
            assert failure["eligible_records"] == []
        assert not any(
            event["type"] == "exec_boundary"
            and event["host_tid"] == gap["host_tid"]
            and event["ts_ns"] <= gap["ts_ns"]
            for event in run.events
        )


def test_current_rss_counter_layout_compiles_and_reports_pages() -> None:
    payload = (
        "buf=bytearray(32*1024*1024);"
        "[(buf.__setitem__(i,1)) for i in range(0,len(buf),4096)];"
        "print(sum(buf))"
    )
    run = collect_case(
        f"{shlex.quote(sys.executable)} -c {shlex.quote(payload)}",
        "rss_counter_layout",
    )

    assert run.status == 0
    assert max(
        event["rss_pages"]
        for event in run.events
        if event["type"] == "perf"
    ) > 0


def test_parallel_collectors_isolate_cgroups_and_report_task_io(
    tmp_path: Path,
) -> None:
    cases = tuple(
        (tag, blocks, str(tmp_path / f"{tag}.bin"))
        for tag, blocks in (("small", 8), ("large", 128))
    )
    with ProcessPoolExecutor(max_workers=2) as pool:
        runs = list(pool.map(_collect, cases))

    writes: dict[str, int] = {}
    for tag, blocks, run in runs:
        assert run.status == 0
        assert run.loss_count == 0
        assert {event["cgroup_id"] for event in run.events} == {run.cgroup_id}
        metrics, gaps = analyze(run)
        _assert_only_harness_root_pre_exec_gaps(run, gaps)
        dd = next(metric for metric in metrics if metric.bin == "dd")
        assert dd.disk_io_reason == "ok"
        assert dd.disk_write_bytes_total is not None
        assert dd.disk_write_bytes_total >= blocks * 4096
        assert dd.disk_read_bytes_total == 0
        writes[tag] = dd.disk_write_bytes_total

    assert writes["small"] < writes["large"]


def test_failed_execve_emits_pending_argv_and_errno() -> None:
    run = collect_case(
        "python3 -c 'import os; "
        'os.execve("/codex-missing-executable", ["missing", "--flag"], os.environ)'
        "'",
        "failed_execve",
    )
    assert run.loss_count == 0
    attempts = _failed_exec_attempt_records(run.events)
    assert len(attempts) == 1
    assert attempts[0].exec_seq >= 0
    assert attempts[0].argv == ("missing", "--flag")
    assert attempts[0].errno == 2
    metrics, gaps = analyze(run)
    _assert_only_harness_root_pre_exec_gaps(run, gaps)
    assert (
        next(metric for metric in metrics if metric.bin == "python3").normal_exit_status
        == 1
    )
    assert run.lifecycle_map_entries == {"current_seq": 0, "pending_seq": 0}


def test_argv_capture_flags_cover_truncation_cap_and_short_argv() -> None:
    payload = (
        "import subprocess;"
        "subprocess.run(['/bin/true','e'*511],check=True);"
        "subprocess.run(['/bin/true','l'*1110],check=True);"
        f"subprocess.run(['/bin/true','m'*{C.MAX_ARG_WORD_BYTES}],check=True);"
        f"subprocess.run(['/bin/true','x'*{C.MAX_ARG_WORD_BYTES + 1}],check=True);"
        f"subprocess.run(['/bin/true',*map(str,range({C.MAX_ARGS}))],check=True);"
        "subprocess.run(['/bin/true','ok'],check=True)"
    )
    run = collect_case(
        f"{shlex.quote(sys.executable)} -c {shlex.quote(payload)}",
        "argv_capture_flags",
    )

    assert run.loss_count == 0
    metrics, gaps = analyze(run)
    _assert_only_harness_root_pre_exec_gaps(run, gaps)
    true_metrics = [metric for metric in metrics if metric.bin == "true"]
    exact_buffer_edge = next(
        metric
        for metric in true_metrics
        if len(metric.argv) == 2 and metric.argv[1].startswith("e")
    )
    exact_chunked = next(
        metric
        for metric in true_metrics
        if len(metric.argv) == 2 and metric.argv[1].startswith("l")
    )
    exact_max = next(
        metric
        for metric in true_metrics
        if len(metric.argv) == 2 and metric.argv[1].startswith("m")
    )
    truncated = next(
        metric
        for metric in true_metrics
        if len(metric.argv) == 2 and metric.argv[1].startswith("x")
    )
    capped = next(metric for metric in true_metrics if metric.argv[1:3] == ("0", "1"))
    short = next(
        metric
        for metric in true_metrics
        if len(metric.argv) == 2 and metric.argv[1] == "ok"
    )

    assert len(exact_buffer_edge.argv[1]) == C.ARG_BYTES - 1
    assert exact_buffer_edge.argv_capture_flags == 0
    assert len(exact_chunked.argv[1]) == 1110
    assert exact_chunked.argv_capture_flags == 0
    assert len(exact_max.argv[1]) == C.MAX_ARG_WORD_BYTES
    assert exact_max.argv_capture_flags == 0
    assert len(truncated.argv[1]) == C.MAX_ARG_WORD_BYTES
    assert truncated.argv_capture_flags == 1 << 1
    assert len(capped.argv) == C.MAX_ARGS
    assert capped.argv_capture_flags == 1 << C.MAX_ARGS
    assert capped.exact_argc == C.MAX_ARGS + 1
    assert short.argv_capture_flags == 0


def test_successful_exec_recovers_cold_filename_and_original_argv() -> None:
    payload = """import ctypes,mmap,tempfile
script=tempfile.NamedTemporaryFile(delete=False)
script.write(b'#!/bin/sh\\nrm -- "$0"\\nexit 0\\n')
script.flush()
script.close()
__import__("os").chmod(script.name,0o755)
path_file=tempfile.TemporaryFile()
path_file.write(script.name.encode()+b"\\0")
path_file.truncate(mmap.PAGESIZE)
path_file.flush()
path_map=mmap.mmap(path_file.fileno(),mmap.PAGESIZE,access=mmap.ACCESS_COPY)
path_address=ctypes.addressof(ctypes.c_char.from_buffer(path_map))
path_map.madvise(mmap.MADV_DONTNEED)
f=tempfile.TemporaryFile()
f.write(b"cold-page-argument\\0")
f.truncate(mmap.PAGESIZE)
f.flush()
m=mmap.mmap(f.fileno(),mmap.PAGESIZE,access=mmap.ACCESS_COPY)
address=ctypes.addressof(ctypes.c_char.from_buffer(m))
m.madvise(mmap.MADV_DONTNEED)
argv=(ctypes.c_char_p*3)()
argv[0]=b"original-argv0"
argv[1]=ctypes.c_char_p(address)
envp=(ctypes.c_char_p*1)()
libc=ctypes.CDLL(None,use_errno=True)
libc.execve(ctypes.c_char_p(path_address),argv,envp)
raise OSError(ctypes.get_errno())"""
    run = collect_case(
        f"{shlex.quote(sys.executable)} -c {shlex.quote(payload)}",
        "page_cold_argv",
    )

    metrics, gaps = analyze(run)
    _assert_only_harness_root_pre_exec_gaps(run, gaps)
    assert run.loss_count == 0
    script_metric = next(
        metric for metric in metrics if metric.bin == "original-argv0"
    )
    assert script_metric.argv == (
        "original-argv0",
        "cold-page-argument",
    )
    assert script_metric.exact_argc == 2
    assert script_metric.bprm_filename == script_metric.requested_executable_path
    assert script_metric.bprm_interp == "/bin/sh"


def test_unrecoverable_exec_filename_still_counts_loss() -> None:
    payload = """import ctypes
argv=(ctypes.c_char_p*2)()
argv[0]=b"invalid-filename"
envp=(ctypes.c_char_p*1)()
libc=ctypes.CDLL(None,use_errno=True)
result=libc.execve(ctypes.c_void_p(1),argv,envp)
raise SystemExit(0 if result == -1 and ctypes.get_errno() == 14 else 1)"""
    run = collect_case(
        f"{shlex.quote(sys.executable)} -c {shlex.quote(payload)}",
        "invalid_exec_filename",
    )

    assert run.status == 0
    assert run.argv_read_failures == 1
    assert run.argv_boundary_read_failures == 0
    assert run.ringbuf_reserve_failures == 0


def test_normal_exec_exit_status_is_decoded_from_kernel_wait_status() -> None:
    run = collect_case("exit 7", "normal_exit_status")
    metrics, _ = analyze(run)

    assert run.status == 7
    shell = next(metric for metric in metrics if metric.bin == "sh")
    assert shell.normal_exit_status == 7
    assert shell.exit_signal is None


def test_terminal_scheduler_sample_keeps_identity_but_not_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(C, "SAMPLE_PERIOD_NS", 100_000)
    run = collect_case(
        "i=0; while [ $i -lt 1500 ]; do /bin/true; i=$((i+1)); done",
        "terminal_identity",
    )
    clauses, _ = C._clauses_and_lineage(run.events)
    by_key = {(clause.host_pid, clause.exec_seq): clause for clause in clauses}
    terminal_samples = [
        event
        for event in run.events
        if event["type"] == "perf"
        and event["exec_seq"] != C.SENTINEL
        and (clause := by_key.get((event["host_pid"], event["exec_seq"]))) is not None
        and event["ts_ns"] >= clause.t_end_ns
    ]

    assert run.loss_count == 0
    assert terminal_samples
    metrics, gaps = analyze(run)
    _assert_only_harness_root_pre_exec_gaps(run, gaps)
    assert not {gap["reason"] for gap in gaps} & {"sentinel_after_successful_exec"}
    affected = {(event["host_pid"], event["exec_seq"]) for event in terminal_samples}
    assert all(
        metric.provenance["identity_only_sample_count"] > 0
        for metric in metrics
        if (metric.host_pid, metric.exec_seq) in affected
    )


def test_process_free_bounds_lifecycle_maps() -> None:
    run = collect_case(
        "i=0; while [ $i -lt 256 ]; do /bin/true; i=$((i+1)); done",
        "process_free_cleanup",
    )

    assert run.loss_count == 0
    assert run.lifecycle_map_entries == {"current_seq": 0, "pending_seq": 0}


def test_parallel_exec_burst_exceeds_old_sequence_pool_without_loss() -> None:
    exec_count = 8200
    run = collect_case(
        f"seq 1 {exec_count} | xargs -P64 -I{{}} /bin/true",
        "parallel_exec_burst",
    )
    true_execs = {
        (event["host_pid"], event["exec_seq"])
        for event in run.events
        if event["type"] == "exec_arg"
        and event["arg_index"] == 0
        and event["arg"] == "/bin/true"
    }
    true_boundaries = sum(
        event["type"] == "exec_boundary"
        and (event["host_pid"], event["exec_seq"]) in true_execs
        for event in run.events
    )

    assert run.status == 0
    assert run.loss_counts == {
        "ringbuf_reserve_failures": 0,
        "argv_read_failures": 0,
        "argv_boundary_read_failures": 0,
    }
    assert true_boundaries == exec_count
    # This exceeds both the ring's instantaneous capacity and the removed
    # 8,192-entry one-shot sequence queue.
    assert len(run.events) > 32_000


def test_fork_reinitializes_child_slot_before_first_exec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(C, "SAMPLE_PERIOD_NS", 100_000)
    payload = (
        "import os,time;"
        "pid=os.fork();"
        "deadline=time.thread_time()+0.03 if pid==0 else 0;"
        "exec('while time.thread_time() < deadline: pass\\n"
        'os.execve("/bin/true", ["true"], os.environ)\' if pid==0 else '
        "'os.waitpid(pid, 0)')"
    )
    run = collect_case(
        f"{shlex.quote(sys.executable)} -c {shlex.quote(payload)}",
        "fork_slot_reinit",
    )
    child_fork = next(event for event in run.events if event["type"] == "fork")
    child_tid = child_fork["child_host_tid"]
    child_exec = next(
        event
        for event in run.events
        if event["type"] == "exec_boundary"
        and event["host_tid"] == child_tid
        and event["ts_ns"] > child_fork["ts_ns"]
    )
    pre_exec_samples = [
        event
        for event in run.events
        if event["type"] == "perf"
        and event["host_tid"] == child_tid
        and child_fork["ts_ns"] <= event["ts_ns"] < child_exec["ts_ns"]
    ]

    assert run.loss_count == 0
    assert pre_exec_samples
    assert {event["exec_seq"] for event in pre_exec_samples} == {C.SENTINEL}
    assert child_exec["exec_seq"] != C.SENTINEL
    assert run.lifecycle_map_entries == {"current_seq": 0, "pending_seq": 0}
