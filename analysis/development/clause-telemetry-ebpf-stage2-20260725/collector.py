#!/usr/bin/python3
"""Stage-2 clause telemetry: honest per-clause peak_cpu_cores + sampled_peak_rss.

Extends the Stage-1b lifecycle collector (exec/fork/exit/hiwater) with the
perf CPU-clock sampler validated by the accepted spike, and reconstructs, per
clause = (host_pid, exec_seq):

- ``peak_cpu_cores``: cumulative per-TID CPU deltas of the clause's own threads
  and non-exec descendants, aggregated into 500 ms wall windows (matching the
  resource_timeline label semantics), rate = Delta cpu_ns / window_ns, clipped
  ONLY to the observed cgroup quota; the max window rate. Never cpu_ns/wall_ns.
- ``sampled_peak_rss``: the maximum, over aligned time bins, of the SUM of
  current RSS across DISTINCT live ``mm`` address spaces in the clause lineage
  (threads sharing an mm are deduplicated; distinct mm are summed at the same
  aligned timestamp). Never a per-TID sum, never per-mm maxima summed across
  different times, never a reused lifetime hiwater.

Both carry provenance (cadence, window width, sample/coverage counts, boundary
coverage, lost-event counters, quota) and are returned ``unavailable`` with a
reason when their target-specific coverage is insufficient. ``wall_ns`` and
cumulative ``cpu_ns`` are preserved as separate raw observations.

Non-perturbing: samples emit in-kernel only for tasks in the target cgroup.
Runs a workload in a fresh cgroup-v2 scope (the host's frozen Stage-1b docker
image is unavailable, so — like the spike — a local cgroup is used; container
teardown semantics remain a Stage-1b concern). Root required.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# bcc is imported lazily inside the collection functions so the pure analysis
# (attribution, windowing, aggregation) can be imported and unit-tested without
# a bcc/BPF runtime.

SAMPLE_PERIOD_NS = 10_000_000  # ~10 ms CPU-time per perf callback
WINDOW_NS = 500_000_000  # 500 ms wall label window (resource_timeline semantics)
ALIGN_BIN_NS = 20_000_000  # 20 ms aligned bins for RSS summation
SENTINEL = 2**64 - 1
PAGE = 4096
_NPROC = os.cpu_count() or 1

TYPE_NAMES = {
    1: "exec_arg",
    2: "exec_boundary",
    3: "exit_boundary",
    4: "fork",
    5: "perf",
}

BPF_PROGRAM = r"""
#include <linux/mm_types.h>
#include <linux/sched.h>
#include <linux/sched/signal.h>
#include <uapi/linux/bpf_perf_event.h>

#define TYPE_EXEC_ARG 1
#define TYPE_EXEC_BOUNDARY 2
#define TYPE_EXIT_BOUNDARY 3
#define TYPE_FORK 4
#define TYPE_PERF 5
#define MAX_ARGS 8
#define ARG_BYTES 128

struct event_t {
    u64 timestamp_ns;
    u64 cgroup_id;
    u64 exec_seq;
    u64 cpu_ns;         /* per-task cumulative utime+stime at sample time */
    u64 rss_pages;      /* CURRENT rss = file+anon+shmem (not hiwater) */
    u64 mm_ptr;         /* address-space identity for dedup */
    u64 hiwater_pages;  /* raw lifetime hiwater (exit only), kept separate */
    u32 type;
    u32 host_pid;
    u32 host_tid;
    u32 parent_host_pid;
    u32 child_host_pid;
    u32 arg_index;
    u32 exit_code;
    char arg[ARG_BYTES];
};

BPF_RINGBUF_OUTPUT(events, 1024);
BPF_ARRAY(target_cgroup, u64, 1);
BPF_ARRAY(reserve_failures, u64, 1);
BPF_ARRAY(perf_sample_count, u64, 1);
BPF_QUEUE(exec_sequences, u64, 65536);
BPF_ARRAY(sequence_ready, u32, 1);
BPF_HASH(current_seq, u32, u64);   /* tid -> seq of the RUNNING exec image */
BPF_HASH(pending_seq, u32, u64);   /* tid -> seq of an in-flight execve */

static int wanted(void) {
    u32 zero = 0;
    u64 *t = target_cgroup.lookup(&zero);
    return t && *t && *t == bpf_get_current_cgroup_id();
}

static void lost(void) {
    u32 z = 0;
    u64 *c = reserve_failures.lookup(&z);
    if (c) __sync_fetch_and_add(c, 1);
}

static u32 parent_tgid(void) {
    struct task_struct *task = (struct task_struct *)bpf_get_current_task();
    struct task_struct *parent = 0;
    u32 tgid = 0;
    bpf_probe_read_kernel(&parent, sizeof(parent), &task->real_parent);
    if (parent) bpf_probe_read_kernel(&tgid, sizeof(tgid), &parent->tgid);
    return tgid;
}

static u64 current_rss_pages(struct task_struct *task, u64 *mm_out) {
    struct mm_struct *mm = 0;
    bpf_probe_read_kernel(&mm, sizeof(mm), &task->mm);
    *mm_out = (u64)mm;
    if (!mm) return 0;
    long file = 0, anon = 0, shmem = 0;
    bpf_probe_read_kernel(&file, sizeof(file), &mm->rss_stat.count[0].counter);
    bpf_probe_read_kernel(&anon, sizeof(anon), &mm->rss_stat.count[1].counter);
    bpf_probe_read_kernel(&shmem, sizeof(shmem), &mm->rss_stat.count[3].counter);
    long total = file + anon + shmem;
    return total < 0 ? 0 : (u64)total;
}

static u64 current_cpu_ns(struct task_struct *task) {
    u64 u = 0, s = 0;
    bpf_probe_read_kernel(&u, sizeof(u), &task->utime);
    bpf_probe_read_kernel(&s, sizeof(s), &task->stime);
    return u + s;
}

static void fill_counters(struct event_t *e, struct task_struct *task) {
    u64 mm_ptr = 0;
    e->rss_pages = current_rss_pages(task, &mm_ptr);
    e->mm_ptr = mm_ptr;
    e->cpu_ns = current_cpu_ns(task);
}

/* execve/execveat ENTRY: assign a new seq as PENDING (do NOT overwrite
 * current_seq yet, so samples between enter and a successful exec stay on the
 * OLD image), and capture argv tagged with that pending seq. Shared by both
 * execve and execveat so execveat transitions are not silently dropped. */
static int capture_enter(const char *const *argv) {
    u32 zero = 0;
    u32 *ready = sequence_ready.lookup(&zero);
    if (!ready || !*ready) return 0;
    if (!wanted()) return 0;
    u64 seq = 0;
    if (exec_sequences.pop(&seq)) return 0;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tid = pid_tgid;
    pending_seq.update(&tid, &seq);
    #pragma unroll
    for (int i = 0; i < MAX_ARGS; i++) {
        const char *a = 0;
        bpf_probe_read_user(&a, sizeof(a), &argv[i]);
        if (!a) break;
        struct event_t *e = events.ringbuf_reserve(sizeof(*e));
        if (!e) { lost(); continue; }
        __builtin_memset(e, 0, sizeof(*e));
        e->timestamp_ns = bpf_ktime_get_ns();
        e->cgroup_id = bpf_get_current_cgroup_id();
        e->exec_seq = seq;
        e->type = TYPE_EXEC_ARG;
        e->host_pid = pid_tgid >> 32;
        e->host_tid = tid;
        e->arg_index = i;
        bpf_probe_read_user_str(e->arg, sizeof(e->arg), a);
        events.ringbuf_submit(e, 0);
    }
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_enter_execve) {
    return capture_enter((const char *const *)args->argv);
}

TRACEPOINT_PROBE(syscalls, sys_enter_execveat) {
    return capture_enter((const char *const *)args->argv);
}

/* execve RETURN: only fires meaningfully on FAILURE (success does not return to
 * the old image). A failed exec must abandon its pending seq — the old image
 * keeps running under current_seq. Success is handled by sched_process_exec. */
static int on_exec_return(long ret) {
    if (ret == 0) return 0;
    if (!wanted()) return 0;
    u32 tid = bpf_get_current_pid_tgid();
    pending_seq.delete(&tid);
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_exit_execve) { return on_exec_return(args->ret); }
TRACEPOINT_PROBE(syscalls, sys_exit_execveat) { return on_exec_return(args->ret); }

/* Successful exec: promote pending -> current, then emit the boundary with the
 * NEW image's seq. Without a pending seq (missing enter) keep the prior. */
TRACEPOINT_PROBE(sched, sched_process_exec) {
    if (!wanted()) return 0;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tid = pid_tgid;
    u64 new_seq = ~0ULL;
    u64 *pending = pending_seq.lookup(&tid);
    if (pending) {
        new_seq = *pending;
        current_seq.update(&tid, &new_seq);
        pending_seq.delete(&tid);
    } else {
        u64 *cur = current_seq.lookup(&tid);
        if (cur) new_seq = *cur;
    }
    struct event_t *e = events.ringbuf_reserve(sizeof(*e));
    if (!e) { lost(); return 0; }
    __builtin_memset(e, 0, sizeof(*e));
    struct task_struct *task = (struct task_struct *)bpf_get_current_task();
    fill_counters(e, task);
    e->timestamp_ns = bpf_ktime_get_ns();
    e->cgroup_id = bpf_get_current_cgroup_id();
    e->exec_seq = new_seq;
    e->type = TYPE_EXEC_BOUNDARY;
    e->host_pid = pid_tgid >> 32;
    e->host_tid = tid;
    e->parent_host_pid = parent_tgid();
    events.ringbuf_submit(e, 0);
    return 0;
}

/* Fork lineage must be TGID-consistent with every other event (which key on
 * tgid = pid_tgid>>32). The tracepoint's parent_pid/child_pid are TIDs; using
 * parent_pid directly breaks lineage when a non-leader thread forks. Record the
 * forking task's TGID as the parent; child_pid == child TGID for a process fork
 * (and is an inert TID for a thread clone, never consulted by the tgid walk). */
TRACEPOINT_PROBE(sched, sched_process_fork) {
    if (!wanted()) return 0;
    struct event_t *e = events.ringbuf_reserve(sizeof(*e));
    if (!e) { lost(); return 0; }
    __builtin_memset(e, 0, sizeof(*e));
    e->timestamp_ns = bpf_ktime_get_ns();
    e->cgroup_id = bpf_get_current_cgroup_id();
    e->exec_seq = ~0ULL;
    e->type = TYPE_FORK;
    e->host_pid = bpf_get_current_pid_tgid() >> 32;  /* parent TGID */
    e->child_host_pid = args->child_pid;
    events.ringbuf_submit(e, 0);
    return 0;
}

TRACEPOINT_PROBE(sched, sched_process_exit) {
    if (!wanted()) return 0;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tid = pid_tgid;
    struct task_struct *task = (struct task_struct *)bpf_get_current_task();
    struct signal_struct *signal = 0;
    u64 gu = 0, gs = 0;
    u32 exit_code = 0;
    bpf_probe_read_kernel(&signal, sizeof(signal), &task->signal);
    bpf_probe_read_kernel(&exit_code, sizeof(exit_code), &task->exit_code);
    struct mm_struct *mm = 0;
    unsigned long hiwater = 0;
    bpf_probe_read_kernel(&mm, sizeof(mm), &task->mm);
    if (mm) bpf_probe_read_kernel(&hiwater, sizeof(hiwater), &mm->hiwater_rss);
    u64 *seq = current_seq.lookup(&tid);
    struct event_t *e = events.ringbuf_reserve(sizeof(*e));
    if (!e) { lost(); return 0; }
    __builtin_memset(e, 0, sizeof(*e));
    fill_counters(e, task);
    e->timestamp_ns = bpf_ktime_get_ns();
    e->cgroup_id = bpf_get_current_cgroup_id();
    e->exec_seq = seq ? *seq : ~0ULL;
    e->hiwater_pages = hiwater;
    e->type = TYPE_EXIT_BOUNDARY;
    e->host_pid = pid_tgid >> 32;
    e->host_tid = tid;
    e->parent_host_pid = parent_tgid();
    e->exit_code = exit_code;
    events.ringbuf_submit(e, 0);
    current_seq.delete(&tid);
    pending_seq.delete(&tid);
    return 0;
}

int on_cpu_clock(struct bpf_perf_event_data *ctx) {
    if (!wanted()) return 0;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tid = pid_tgid;
    struct task_struct *task = (struct task_struct *)bpf_get_current_task();
    u64 *seq = current_seq.lookup(&tid);
    struct event_t *e = events.ringbuf_reserve(sizeof(*e));
    if (!e) { lost(); return 0; }
    __builtin_memset(e, 0, sizeof(*e));
    fill_counters(e, task);
    e->timestamp_ns = bpf_ktime_get_ns();
    e->cgroup_id = bpf_get_current_cgroup_id();
    e->exec_seq = seq ? *seq : ~0ULL;
    e->type = TYPE_PERF;
    e->host_pid = pid_tgid >> 32;
    e->host_tid = tid;
    events.ringbuf_submit(e, 0);
    u32 z = 0;
    u64 *c = perf_sample_count.lookup(&z);
    if (c) __sync_fetch_and_add(c, 1);
    return 0;
}
"""


# ---------------------------------------------------------------------------
# Raw collection
# ---------------------------------------------------------------------------


def _read_usage_usec(cgroup: Path) -> int:
    for line in (cgroup / "cpu.stat").read_text().splitlines():
        if line.startswith("usage_usec"):
            return int(line.split()[1])
    raise RuntimeError("usage_usec not found")


def observed_quota_cores(cgroup: Path) -> float:
    raw = (cgroup / "cpu.max").read_text().split()
    return float(_NPROC) if raw[0] == "max" else float(int(raw[0]) / int(raw[1]))


class RssOracle(threading.Thread):
    """Independent live-RSS reference: sum of VmRSS over distinct cgroup PIDs.

    A userspace poller (analysis-only oracle, never a prediction input): at each
    tick it sums current VmRSS across distinct tgids in the cgroup and keeps the
    max. This is the ground truth ``sampled_peak_rss`` is compared against.
    """

    def __init__(self, cgroup: Path, interval_s: float = 0.002) -> None:
        super().__init__(daemon=True)
        self._cgroup = cgroup
        self._interval = interval_s
        self._halt = threading.Event()  # not _stop: Thread._stop is internal
        self.peak_sum_kb = 0
        self.samples = 0

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                pids = (self._cgroup / "cgroup.procs").read_text().split()
            except OSError:
                break
            total = 0
            for pid in pids:
                try:
                    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                        if line.startswith("VmRSS:"):
                            total += int(line.split()[1])
                            break
                except OSError:
                    continue
            if total > self.peak_sum_kb:
                self.peak_sum_kb = total
            self.samples += 1
            time.sleep(self._interval)

    def stop(self) -> None:
        self._halt.set()


@dataclass
class RawRun:
    cgroup_id: int
    quota_cores: float
    status: int
    wall_ns: int
    usage_usec: int
    reserve_failures: int
    perf_sample_count: int
    oracle_peak_rss_kb: int
    oracle_samples: int
    marker: bool
    events: list[dict[str, Any]] = field(default_factory=list)


def _new_cgroup(tag: str) -> Path:
    cg = Path(f"/sys/fs/cgroup/clause_stage2_{os.getpid()}_{tag}")
    cg.mkdir(exist_ok=False)
    return cg


def _rmdir_with_retry(cg: Path, attempts: int = 25, delay_s: float = 0.02) -> None:
    """Remove an emptied cgroup, retrying transient EBUSY; never hide failure.

    A just-emptied cgroup can briefly return EBUSY while the kernel reaps the
    last exiting task. We retry for ~0.5 s; a persistent failure is reported
    loudly to stderr (with the lingering PIDs) rather than silently swallowed.
    """

    for _ in range(attempts):
        try:
            cg.rmdir()
            return
        except FileNotFoundError:
            return
        except OSError:
            time.sleep(delay_s)
    try:
        procs = (cg / "cgroup.procs").read_text().split()
    except OSError:
        procs = ["<unreadable>"]
    print(
        f"WARNING: cgroup {cg} not removed after {attempts} attempts; "
        f"lingering pids={procs}",
        file=sys.stderr,
    )


def collect_case(command: str, tag: str, *, marker: str = "") -> RawRun:
    """Attach the sampler, run ``sh -c command`` in a fresh cgroup, analyze raw."""

    from bcc import BPF, PerfSWConfig, PerfType

    cg = _new_cgroup(tag)
    cgroup_id = cg.stat().st_ino
    bpf = BPF(text=BPF_PROGRAM)
    q = bpf["exec_sequences"]
    for seq in range(8192):
        q.push(ctypes.c_ulonglong(seq))
    bpf["sequence_ready"][ctypes.c_int(0)] = ctypes.c_uint(1)
    bpf["target_cgroup"][ctypes.c_int(0)] = ctypes.c_ulonglong(cgroup_id)

    events: list[dict[str, Any]] = []
    lock = threading.Lock()
    table = bpf["events"]

    def receive(_ctx: int, data: int, _size: int) -> int:
        e = table.event(data)
        row = {
            "type": TYPE_NAMES[int(e.type)],
            "ts_ns": int(e.timestamp_ns),
            "cgroup_id": int(e.cgroup_id),
            "exec_seq": int(e.exec_seq),
            "cpu_ns": int(e.cpu_ns),
            "rss_pages": int(e.rss_pages),
            "mm_ptr": int(e.mm_ptr),
            "hiwater_pages": int(e.hiwater_pages),
            "host_pid": int(e.host_pid),
            "host_tid": int(e.host_tid),
            "parent_host_pid": int(e.parent_host_pid),
            "child_host_pid": int(e.child_host_pid),
            "arg_index": int(e.arg_index),
            "exit_code": int(e.exit_code),
        }
        if e.type == 1:
            row["arg"] = bytes(e.arg).split(b"\0", 1)[0].decode("utf-8", "replace")
        with lock:
            events.append(row)
        return 0

    table.open_ring_buffer(receive)
    stop_poll = threading.Event()

    def poll() -> None:
        while not stop_poll.is_set():
            bpf.ring_buffer_poll(timeout=10)

    poller = threading.Thread(target=poll, daemon=True)
    poller.start()

    bpf.attach_perf_event(
        ev_type=PerfType.SOFTWARE,
        ev_config=PerfSWConfig.CPU_CLOCK,
        fn_name="on_cpu_clock",
        sample_period=SAMPLE_PERIOD_NS,
    )

    oracle = RssOracle(cg)
    usage_before = _read_usage_usec(cg)

    def _join_cgroup() -> None:
        (cg / "cgroup.procs").write_text(str(os.getpid()))

    oracle.start()
    t0 = time.monotonic_ns()
    proc = subprocess.Popen(
        ["/bin/sh", "-c", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        preexec_fn=_join_cgroup,
    )
    out, _ = proc.communicate()
    status = proc.returncode
    wall_ns = time.monotonic_ns() - t0
    time.sleep(0.25)  # drain after exit
    oracle.stop()
    oracle.join(timeout=1)
    stop_poll.set()
    poller.join(timeout=1)
    try:
        bpf.ring_buffer_consume()
    except Exception:
        pass
    usage_delta = _read_usage_usec(cg) - usage_before
    reserve_failures = bpf["reserve_failures"][ctypes.c_int(0)].value
    perf_count = bpf["perf_sample_count"][ctypes.c_int(0)].value
    quota = observed_quota_cores(cg)
    bpf.detach_perf_event(ev_type=PerfType.SOFTWARE, ev_config=PerfSWConfig.CPU_CLOCK)
    bpf.cleanup()
    _rmdir_with_retry(cg)
    with lock:
        ordered = sorted(events, key=lambda r: r["ts_ns"])
    return RawRun(
        cgroup_id=cgroup_id,
        quota_cores=quota,
        status=status,
        wall_ns=wall_ns,
        usage_usec=usage_delta,
        reserve_failures=reserve_failures,
        perf_sample_count=perf_count,
        oracle_peak_rss_kb=oracle.peak_sum_kb,
        oracle_samples=oracle.samples,
        marker=marker.encode() in out if marker else True,
        events=[e for e in ordered if e["cgroup_id"] == cgroup_id],
    )


# ---------------------------------------------------------------------------
# Clause reconstruction, attribution, and the two honest metrics
# ---------------------------------------------------------------------------


@dataclass
class Clause:
    host_pid: int
    exec_seq: int
    t_exec_ns: int
    t_end_ns: int
    bin: str
    argv: tuple[str, ...]
    lineage_parent_pid: int | None
    terminal: bool
    has_causal_end: bool  # real exit (terminal) or next same-pid exec (non-terminal)


def _clauses_and_lineage(
    events: list[dict[str, Any]],
) -> tuple[list[Clause], dict[int, int]]:
    """Build per-clause windows and the child_tgid -> parent_tgid fork map."""

    fork_parent: dict[int, int] = {}
    for e in events:
        if e["type"] == "fork" and e["child_host_pid"]:
            fork_parent.setdefault(e["child_host_pid"], e["host_pid"])

    argv_words: dict[tuple[int, int], dict[int, str]] = {}
    for e in events:
        if e["type"] == "exec_arg":
            argv_words.setdefault((e["host_pid"], e["exec_seq"]), {})[
                e["arg_index"]
            ] = e["arg"]

    def argv_of(pid: int, seq: int) -> tuple[str, ...]:
        words = argv_words.get((pid, seq), {})
        return tuple(words[i] for i in sorted(words))

    exits: dict[int, int] = {}
    for e in events:
        if e["type"] == "exit_boundary":
            exits[e["host_pid"]] = max(exits.get(e["host_pid"], 0), e["ts_ns"])

    # exec boundaries per pid, ordered -> clause windows
    execs_by_pid: dict[int, list[dict[str, Any]]] = {}
    for e in events:
        if e["type"] == "exec_boundary" and e["exec_seq"] != SENTINEL:
            execs_by_pid.setdefault(e["host_pid"], []).append(e)

    last_ts = max((r["ts_ns"] for r in events), default=0)
    clauses: list[Clause] = []
    for pid, execs in execs_by_pid.items():
        execs.sort(key=lambda r: r["ts_ns"])
        for i, e in enumerate(execs):
            terminal = i == len(execs) - 1
            if not terminal:
                t_end = execs[i + 1]["ts_ns"]  # next exec on same pid (causal)
                has_causal_end = True
            elif pid in exits:
                t_end = exits[pid]
                has_causal_end = True
            else:
                t_end = last_ts  # synthetic bound; NOT a real causal end
                has_causal_end = False
            argv = argv_of(pid, e["exec_seq"])
            clauses.append(
                Clause(
                    host_pid=pid,
                    exec_seq=e["exec_seq"],
                    t_exec_ns=e["ts_ns"],
                    t_end_ns=t_end,
                    bin=Path(argv[0]).name if argv else "",
                    argv=argv,
                    lineage_parent_pid=fork_parent.get(pid),
                    terminal=terminal,
                    has_causal_end=has_causal_end,
                )
            )
    return clauses, fork_parent


def _clause_at(clauses_on_pid: "list[Clause] | tuple", ts: int) -> "Clause | None":
    """Half-open window match: [t_exec, t_end), for terminal and non-terminal
    clauses alike. The exit-boundary sample (ts == t_end) carries its clause's
    exec_seq and is attributed by the direct-seq path, so no inclusive terminal
    end is needed here — and an inclusive end would wrongly pull a sentinel
    sample at exactly t_end onto a just-ended clause."""

    for c in clauses_on_pid:
        if c.t_exec_ns <= ts < c.t_end_ns:
            return c
    return None


def _ancestor_clause_pid(
    pid: int, ts: int, clause_by_pid: dict[int, list[Clause]], fork_parent: dict[int, int]
) -> Clause | None:
    """Nearest ancestor pid whose half-open clause window contains ts."""

    seen: set[int] = set()
    cur = pid
    while cur and cur not in seen:
        match = _clause_at(clause_by_pid.get(cur, ()), ts)
        if match is not None:
            return match
        seen.add(cur)
        cur = fork_parent.get(cur, 0)
    return None


@dataclass
class ClauseMetrics:
    host_pid: int
    exec_seq: int
    bin: str
    argv: tuple[str, ...]  # exec-image argv, evidence for the clause bridge
    lineage_parent_pid: int | None  # fork parent, for bridge lineage attribution
    terminal: bool
    has_causal_end: bool  # real exit or next same-pid exec; fail closed if False
    t_exec_ns: int  # clause-window bounds, for the bridge time-aligned merge
    t_end_ns: int
    wall_ns: int
    cpu_ns_cumulative: int  # raw, preserved separately
    exit_signal: int | None  # low 7 bits of exit_code on the terminal exit
    peak_cpu_cores: float | None
    peak_cpu_cores_reason: str
    sampled_peak_rss_mb: float | None
    sampled_peak_rss_reason: str
    # time-aligned profiles the clause bridge merges across owned images
    cpu_windows: tuple[tuple[int, int], ...]
    rss_bins: tuple[tuple[int, int, float], ...]
    provenance: dict[str, Any]


def _attribute(
    events: list[dict[str, Any]], clauses: list[Clause], fork_parent: dict[int, int]
) -> tuple[dict[tuple[int, int], list[dict[str, Any]]], list[dict[str, Any]]]:
    """Attribute perf + boundary samples to a clause; return (per_clause, gaps)."""

    clause_by_pid: dict[int, list[Clause]] = {}
    by_pid_seq: dict[tuple[int, int], Clause] = {}
    for c in clauses:
        clause_by_pid.setdefault(c.host_pid, []).append(c)
        by_pid_seq[(c.host_pid, c.exec_seq)] = c
    per_clause: dict[tuple[int, int], list[dict[str, Any]]] = {
        (c.host_pid, c.exec_seq): [] for c in clauses
    }
    gaps: list[dict[str, Any]] = []
    for e in events:
        if e["type"] not in {"perf", "exec_boundary", "exit_boundary"}:
            continue
        ts, pid, seq = e["ts_ns"], e["host_pid"], e["exec_seq"]
        target: Clause | None = None
        # 1) DIRECT-SEQ: the sample carries the exec_seq of a clause on its pid
        #    (exec/exit boundaries, and perf on the exec'ing thread) — exact,
        #    taken before any window/lineage fallback.
        if seq != SENTINEL:
            target = by_pid_seq.get((pid, seq))
        # 2) half-open window on the sample's own pid (sentinel-seq threads)
        if target is None:
            target = _clause_at(clause_by_pid.get(pid, ()), ts)
        # 3) else walk fork lineage to an ancestor clause (half-open)
        if target is None:
            target = _ancestor_clause_pid(pid, ts, clause_by_pid, fork_parent)
        if target is None:
            gaps.append(e)  # preserved, never dropped
        else:
            per_clause[(target.host_pid, target.exec_seq)].append(e)
    return per_clause, gaps


_MIN_ELIGIBLE_SPAN_NS = 1_000_000_000  # resource_timeline: clause >= 1 s
_MIN_WINDOW_SPAN_NS = 100_000_000  # ignore <100 ms trailing windows (rate noise)


def _apportion(t0: int, t1: int, cpu_ns: int) -> "list[tuple[int, float]]":
    """Split a cpu_ns delta over [t0, t1) across EVERY intersected 500 ms window,
    proportional to each window's overlap — a delta spanning a window boundary
    must not be dumped whole into one window."""

    dt = t1 - t0
    if dt <= 0:
        return []
    out: list[tuple[int, float]] = []
    w = t0 // WINDOW_NS
    while w * WINDOW_NS < t1:
        lo = max(w * WINDOW_NS, t0)
        hi = min((w + 1) * WINDOW_NS, t1)
        overlap = hi - lo
        if overlap > 0:
            out.append((w, cpu_ns * overlap / dt))
        w += 1
    return out


def cpu_window_profile(samples: list[dict[str, Any]]) -> tuple[tuple[int, int], ...]:
    """Absolute-indexed (window_idx, cpu_ns) contributions for the clause bridge.

    Windows are keyed by ``ts // WINDOW_NS`` (a common absolute grid) so the
    bridge can SUM concurrent owned images per window; each per-TID cpu delta is
    apportioned across every window it intersects.
    """

    per_tid: dict[int, list[tuple[int, int]]] = {}
    for s in samples:
        if s["cpu_ns"] > 0:
            per_tid.setdefault(s["host_tid"], []).append((s["ts_ns"], s["cpu_ns"]))
    windows: dict[int, float] = {}
    for points in per_tid.values():
        points.sort()
        for (t0, c0), (t1, c1) in zip(points, points[1:]):
            if t1 <= t0 or c1 < c0:
                continue
            for widx, part in _apportion(t0, t1, c1 - c0):
                windows[widx] = windows.get(widx, 0.0) + part
    return tuple((w, int(round(v))) for w, v in sorted(windows.items()))


def _peak_cpu_cores(
    samples: list[dict[str, Any]], clause: Clause, quota: float
) -> tuple[float | None, str, dict[str, Any]]:
    cpu_samples = [s for s in samples if s["cpu_ns"] > 0]
    profile = cpu_window_profile(samples)  # apportioned absolute windows
    span = clause.t_end_ns - clause.t_exec_ns
    prov = {
        "cpu_sample_count": len(cpu_samples),
        "cpu_windows": len(profile),
        "span_s": round(span / 1e9, 3),
    }
    # resource_timeline eligibility: clause >= 1 s AND >= 2 CPU samples.
    if span < _MIN_ELIGIBLE_SPAN_NS:
        return None, "clause_shorter_than_1s_ineligible_for_peak", prov
    if len(cpu_samples) < 2 or not profile:
        return None, "insufficient_cpu_samples", prov
    peak: float | None = None
    for widx, cpu_ns in profile:
        win_start = widx * WINDOW_NS
        win_span = min(clause.t_end_ns, win_start + WINDOW_NS) - max(
            clause.t_exec_ns, win_start
        )
        if win_span < _MIN_WINDOW_SPAN_NS:
            continue
        rate = min(cpu_ns / win_span, quota)
        peak = rate if peak is None else max(peak, rate)
    if peak is None:
        return None, "no_eligible_merged_window", prov
    return peak, "ok", prov


def rss_bin_profile(
    samples: list[dict[str, Any]],
) -> tuple[tuple[int, int, float], ...]:
    """Absolute-indexed (bin_idx, mm_ptr, rss_mb) samples for the clause bridge."""

    return tuple(
        (s["ts_ns"] // ALIGN_BIN_NS, s["mm_ptr"], s["rss_pages"] * PAGE / 1e6)
        for s in samples
        if s["rss_pages"] > 0
    )


def _sampled_peak_rss(
    samples: list[dict[str, Any]], clause: Clause
) -> tuple[float | None, str, dict[str, Any]]:
    rss_samples = [s for s in samples if s["rss_pages"] > 0]
    # aligned bins -> per bin, one RSS per distinct mm (latest), sum distinct mm
    bins: dict[int, dict[int, int]] = {}
    mm_tids: dict[int, set[int]] = {}
    for s in rss_samples:
        b = s["ts_ns"] // ALIGN_BIN_NS
        bins.setdefault(b, {})[s["mm_ptr"]] = s["rss_pages"]
        mm_tids.setdefault(s["mm_ptr"], set()).add(s["host_tid"])
    perf_rss = [s for s in rss_samples if s["type"] == "perf"]
    boundary_rss = [s for s in rss_samples if s["type"] != "perf"]
    span = max(clause.t_end_ns - clause.t_exec_ns, 1)
    # coverage: largest gap between consecutive rss samples vs the window
    ts_sorted = sorted(s["ts_ns"] for s in rss_samples)
    max_gap = 0
    edges = [clause.t_exec_ns, *ts_sorted, clause.t_end_ns]
    for a, b in zip(edges, edges[1:]):
        max_gap = max(max_gap, b - a)
    prov = {
        "rss_sample_count": len(rss_samples),
        "perf_rss_samples": len(perf_rss),
        "boundary_rss_samples": len(boundary_rss),
        "distinct_mm": len(mm_tids),
        "shared_mm_tid_counts": {
            hex(mm): len(t) for mm, t in mm_tids.items() if len(t) > 1
        },
        "max_intersample_gap_frac": round(max_gap / span, 3),
    }
    if len(rss_samples) < 2:
        return None, "insufficient_rss_samples", prov
    peak_pages = max(sum(per_mm.values()) for per_mm in bins.values())
    return peak_pages * PAGE / 1e6, "ok", prov


def analyze(run: RawRun) -> tuple[list[ClauseMetrics], list[dict[str, Any]]]:
    clauses, fork_parent = _clauses_and_lineage(run.events)
    per_clause, gaps = _attribute(run.events, clauses, fork_parent)
    metrics: list[ClauseMetrics] = []
    for c in clauses:
        samples = per_clause[(c.host_pid, c.exec_seq)]
        in_window = sum(
            1 for e in run.events
            if e["type"] in {"perf", "exec_boundary", "exit_boundary"}
            and c.t_exec_ns <= e["ts_ns"] <= c.t_end_ns
            and e["host_pid"] == c.host_pid
        )
        peak, cpu_reason, cpu_prov = _peak_cpu_cores(samples, c, run.quota_cores)
        rss, rss_reason, rss_prov = _sampled_peak_rss(samples, c)
        has_exit = any(
            e["type"] == "exit_boundary" and e["host_pid"] == c.host_pid
            for e in run.events
        )
        # Raw cumulative CPU (preserved separately, never used for the peak):
        # deterministic group sum across the terminal process's threads.
        if c.terminal:
            exits = [
                e for e in run.events
                if e["type"] == "exit_boundary" and e["host_pid"] == c.host_pid
            ]
            cpu_cum = sum(e["cpu_ns"] for e in exits)
            leader = next(
                (e for e in exits if e["host_tid"] == e["host_pid"]),
                exits[0] if exits else None,
            )
            exit_signal = (leader["exit_code"] & 0x7F) if leader else None
        else:
            cpu_cum = 0
            exit_signal = None
        metrics.append(
            ClauseMetrics(
                host_pid=c.host_pid,
                exec_seq=c.exec_seq,
                bin=c.bin,
                argv=c.argv,
                lineage_parent_pid=c.lineage_parent_pid,
                terminal=c.terminal,
                has_causal_end=c.has_causal_end,
                t_exec_ns=c.t_exec_ns,
                t_end_ns=c.t_end_ns,
                wall_ns=c.t_end_ns - c.t_exec_ns,
                cpu_ns_cumulative=cpu_cum,
                exit_signal=exit_signal,
                peak_cpu_cores=peak,
                peak_cpu_cores_reason=cpu_reason,
                sampled_peak_rss_mb=rss,
                sampled_peak_rss_reason=rss_reason,
                cpu_windows=cpu_window_profile(samples),
                rss_bins=rss_bin_profile(samples),
                provenance={
                    "cadence_ns": SAMPLE_PERIOD_NS,
                    "window_ns": WINDOW_NS,
                    "align_bin_ns": ALIGN_BIN_NS,
                    "attributed_samples": len(samples),
                    "attribution_coverage": round(
                        len(samples) / max(in_window, 1), 3
                    ),
                    "boundary_coverage": {
                        "has_exec": True,
                        "has_exit": has_exit,
                    },
                    "reserve_failures": run.reserve_failures,
                    "quota_cores": run.quota_cores,
                    "cpu": cpu_prov,
                    "rss": rss_prov,
                },
            )
        )
    return metrics, gaps
