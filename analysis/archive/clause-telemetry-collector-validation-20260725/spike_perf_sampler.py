#!/usr/bin/python3
"""Stage-2 kernel-5.15 perf CPU-clock sampler spike (rigorous edition).

Proves, under a REAL ~2-core parallel workload, the mechanism the full Stage-2
collector will rely on — and reports where it falls short rather than asserting
success:

1. attach + density   -- perf CPU-clock program attaches and samples densely
                         enough under sustained ~2 cores for >=2 s;
2. CPU reconstruction -- reconstructed per-clause CPU vs the cgroup cpu.stat
                         reference (relative error, not just a sample count);
3. lineage attribution-- worker threads (shared mm) AND forked children
                         (distinct mm) are attributed to the clause; every
                         in-window workload sample is attributed or preserved
                         as an explicit coverage gap (never dropped);
4. RSS dedup + sum     -- threads sharing one mm are deduplicated and distinct
                         live address spaces are summed at aligned timestamps;
5. overhead + loss     -- sampler-on vs sampler-off wall/CPU over repeats, and
                         ring-buffer loss under the real parallel load.

Non-perturbing: samples emit in-kernel only for tasks in a cgroup we create per
workload. Stage-1b is untouched. Run as root: sudo python3 spike_perf_sampler.py
"""

from __future__ import annotations

import ctypes
import json
import os
import statistics
import sys
import time
from pathlib import Path

from bcc import BPF, PerfType, PerfSWConfig

_HERE = Path(__file__).resolve().parent
_WORKLOAD = _HERE / "workload"
_SAMPLE_PERIOD_NS = 10_000_000  # ~10 ms CPU-time per perf callback (cadence)
_WINDOW_NS = 500_000_000  # 500 ms wall label window for peak_cpu_cores
_ALIGN_BIN_NS = 20_000_000  # 20 ms bins for time-aligned RSS aggregation
_SENTINEL = 2**64 - 1
_NPROC = os.cpu_count() or 1
_PAGE = 4096

TYPE_NAMES = {1: "perf", 2: "exec_boundary", 3: "exit_boundary", 4: "fork"}

BPF_PROGRAM = r"""
#include <linux/mm_types.h>
#include <linux/sched.h>
#include <uapi/linux/bpf_perf_event.h>

#define TYPE_PERF 1
#define TYPE_EXEC_BOUNDARY 2
#define TYPE_EXIT_BOUNDARY 3
#define TYPE_FORK 4

struct sample_t {
    u64 ts_ns;
    u64 cgroup_id;
    u64 cpu_ns;      /* cumulative per-task utime+stime */
    u64 rss_pages;   /* CURRENT rss = file+anon+shmem (not hiwater) */
    u64 exec_seq;
    u64 mm_ptr;      /* address-space identity for thread/mm dedup */
    u32 host_pid;    /* tgid */
    u32 host_tid;    /* pid */
    u32 child_pid;   /* fork child (TYPE_FORK only) */
    u32 type;
};

BPF_RINGBUF_OUTPUT(samples, 512);
BPF_ARRAY(target_cgroup, u64, 1);
BPF_ARRAY(reserve_failures, u64, 1);
BPF_ARRAY(sample_count, u64, 1);
BPF_QUEUE(exec_sequences, u64, 65536);
BPF_ARRAY(sequence_ready, u32, 1);
BPF_HASH(current_seq, u32, u64);

static int wanted(void) {
    u32 zero = 0;
    u64 *target = target_cgroup.lookup(&zero);
    return target && *target && *target == bpf_get_current_cgroup_id();
}

static void bump(void *addr) {
    u64 *count = (u64 *)addr;
    if (count) __sync_fetch_and_add(count, 1);
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
    u64 utime = 0, stime = 0;
    bpf_probe_read_kernel(&utime, sizeof(utime), &task->utime);
    bpf_probe_read_kernel(&stime, sizeof(stime), &task->stime);
    return utime + stime;
}

static int emit(struct task_struct *task, u32 type) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tid = pid_tgid;
    struct sample_t *s = samples.ringbuf_reserve(sizeof(*s));
    if (!s) { u32 z = 0; bump(reserve_failures.lookup(&z)); return 0; }
    u64 mm_ptr = 0;
    s->rss_pages = current_rss_pages(task, &mm_ptr);
    s->ts_ns = bpf_ktime_get_ns();
    s->cgroup_id = bpf_get_current_cgroup_id();
    s->cpu_ns = current_cpu_ns(task);
    u64 *seq = current_seq.lookup(&tid);
    s->exec_seq = seq ? *seq : ~0ULL;
    s->mm_ptr = mm_ptr;
    s->host_pid = pid_tgid >> 32;
    s->host_tid = tid;
    s->child_pid = 0;
    s->type = type;
    samples.ringbuf_submit(s, 0);
    u32 z = 0; bump(sample_count.lookup(&z));
    return 0;
}

int on_cpu_clock(struct bpf_perf_event_data *ctx) {
    if (!wanted()) return 0;
    return emit((struct task_struct *)bpf_get_current_task(), TYPE_PERF);
}

static int capture_exec_enter(void) {
    u32 zero = 0;
    u32 *ready = sequence_ready.lookup(&zero);
    if (!ready || !*ready) return 0;
    if (!wanted()) return 0;
    u64 seq = 0;
    if (exec_sequences.pop(&seq)) return 0;
    u32 tid = bpf_get_current_pid_tgid();
    current_seq.update(&tid, &seq);
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_enter_execve) { return capture_exec_enter(); }
TRACEPOINT_PROBE(syscalls, sys_enter_execveat) { return capture_exec_enter(); }

TRACEPOINT_PROBE(sched, sched_process_exec) {
    if (!wanted()) return 0;
    return emit((struct task_struct *)bpf_get_current_task(), TYPE_EXEC_BOUNDARY);
}

TRACEPOINT_PROBE(sched, sched_process_fork) {
    if (!wanted()) return 0;
    struct sample_t *s = samples.ringbuf_reserve(sizeof(*s));
    if (!s) { u32 z = 0; bump(reserve_failures.lookup(&z)); return 0; }
    __builtin_memset(s, 0, sizeof(*s));
    s->ts_ns = bpf_ktime_get_ns();
    s->cgroup_id = bpf_get_current_cgroup_id();
    s->exec_seq = ~0ULL;
    s->host_pid = args->parent_pid;
    s->child_pid = args->child_pid;
    s->type = TYPE_FORK;
    samples.ringbuf_submit(s, 0);
    return 0;
}

TRACEPOINT_PROBE(sched, sched_process_exit) {
    if (!wanted()) return 0;
    u32 tid = bpf_get_current_pid_tgid();
    emit((struct task_struct *)bpf_get_current_task(), TYPE_EXIT_BOUNDARY);
    current_seq.delete(&tid);
    return 0;
}
"""


def _read_usage_usec(cgroup: Path) -> int:
    for line in (cgroup / "cpu.stat").read_text().splitlines():
        if line.startswith("usage_usec"):
            return int(line.split()[1])
    raise RuntimeError("usage_usec not found")


def _observed_quota_cores(cgroup: Path) -> float:
    raw = (cgroup / "cpu.max").read_text().split()
    if raw[0] == "max":
        return float(_NPROC)
    return float(int(raw[0]) / int(raw[1]))


def _run_in_cgroup(cgroup: Path, argv: list[str]) -> tuple[int, float, int, bytes]:
    """Run argv as a child that joins cgroup then execs; return status, wall_s,
    cgroup CPU-usage delta (usec), stdout."""

    usage_before = _read_usage_usec(cgroup)
    read_fd, write_fd = os.pipe()
    t0 = time.monotonic()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        os.dup2(write_fd, 1)
        os.close(write_fd)
        (cgroup / "cgroup.procs").write_text(str(os.getpid()))
        os.execv(str(_WORKLOAD), [str(_WORKLOAD), *argv])
        os._exit(127)
    os.close(write_fd)
    out = b""
    while True:
        chunk = os.read(read_fd, 65536)
        if not chunk:
            break
        out += chunk
    os.close(read_fd)
    _, status = os.waitpid(pid, 0)
    wall_s = time.monotonic() - t0
    # let cpu.stat settle
    time.sleep(0.05)
    usage_delta = _read_usage_usec(cgroup) - usage_before
    return status, wall_s, usage_delta, out


def _new_cgroup(tag: str) -> Path:
    cg = Path(f"/sys/fs/cgroup/clause_spike_{os.getpid()}_{tag}")
    cg.mkdir(exist_ok=False)
    return cg


def _drain(bpf: BPF, seconds: float = 0.8) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        bpf.ring_buffer_poll(timeout=25)
    try:
        bpf.ring_buffer_consume()
    except Exception:
        pass


def _sampled_run(tag: str, argv: list[str]) -> dict:
    """Load+attach the sampler, run one workload, return samples + refs."""

    cg = _new_cgroup(tag)
    cgroup_id = cg.stat().st_ino
    bpf = BPF(text=BPF_PROGRAM)
    q = bpf["exec_sequences"]
    for seq in range(4096):
        q.push(ctypes.c_ulonglong(seq))
    bpf["sequence_ready"][ctypes.c_int(0)] = ctypes.c_uint(1)
    bpf["target_cgroup"][ctypes.c_int(0)] = ctypes.c_ulonglong(cgroup_id)

    samples: list[dict] = []
    table = bpf["samples"]

    def receive(_ctx: int, data: int, _size: int) -> int:
        e = table.event(data)
        samples.append(
            {
                "ts_ns": int(e.ts_ns),
                "cgroup_id": int(e.cgroup_id),
                "cpu_ns": int(e.cpu_ns),
                "rss_pages": int(e.rss_pages),
                "exec_seq": int(e.exec_seq),
                "mm_ptr": int(e.mm_ptr),
                "host_pid": int(e.host_pid),
                "host_tid": int(e.host_tid),
                "child_pid": int(e.child_pid),
                "type": TYPE_NAMES[int(e.type)],
            }
        )
        return 0

    table.open_ring_buffer(receive)
    bpf.attach_perf_event(
        ev_type=PerfType.SOFTWARE,
        ev_config=PerfSWConfig.CPU_CLOCK,
        fn_name="on_cpu_clock",
        sample_period=_SAMPLE_PERIOD_NS,
    )
    status, wall_s, usage_delta, out = _run_in_cgroup(cg, argv)
    _drain(bpf)
    reserve_failures = bpf["reserve_failures"][ctypes.c_int(0)].value
    kernel_count = bpf["sample_count"][ctypes.c_int(0)].value
    bpf.detach_perf_event(ev_type=PerfType.SOFTWARE, ev_config=PerfSWConfig.CPU_CLOCK)
    quota = _observed_quota_cores(cg)
    bpf.cleanup()
    _rmdir(cg)
    return {
        "cgroup_id": cgroup_id,
        "status": status,
        "wall_s": wall_s,
        "usage_usec": usage_delta,
        "quota_cores": quota,
        "reserve_failures": reserve_failures,
        "kernel_sample_count": kernel_count,
        "marker": b"WORKLOAD_DONE" in out,
        "samples": [s for s in samples if s["cgroup_id"] == cgroup_id],
    }


def _rmdir(cg: Path) -> None:
    try:
        cg.rmdir()
    except OSError as error:
        print(f"warning: cgroup cleanup failed for {cg}: {error}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Attribution + reconstruction (userspace; the algorithm Stage-2 will formalize)
# ---------------------------------------------------------------------------


def _clause_and_lineage(samples: list[dict]) -> tuple[int, int, dict[int, int]]:
    """Return (clause_tgid, clause_exec_seq, child_tgid->parent_tgid map).

    The clause is the workload's own exec: the exec_boundary whose tgid later
    exits, carrying a resolved exec_seq.
    """

    fork_parent: dict[int, int] = {}
    for s in samples:
        if s["type"] == "fork" and s["child_pid"]:
            fork_parent.setdefault(s["child_pid"], s["host_pid"])
    execs = [s for s in samples if s["type"] == "exec_boundary" and s["exec_seq"] != _SENTINEL]
    if not execs:
        raise RuntimeError("no resolved exec boundary found")
    clause = min(execs, key=lambda s: s["ts_ns"])
    return clause["host_pid"], clause["exec_seq"], fork_parent


def _attributed_to_clause(tgid: int, clause_tgid: int, fork_parent: dict[int, int]) -> bool:
    seen = set()
    cur = tgid
    while cur and cur not in seen:
        if cur == clause_tgid:
            return True
        seen.add(cur)
        cur = fork_parent.get(cur, 0)
    return False


def _cpu_analysis(run: dict) -> dict:
    samples = run["samples"]
    clause_tgid, clause_seq, fork_parent = _clause_and_lineage(samples)
    # window: from the clause exec boundary to its exit boundary
    exec_ts = min(
        s["ts_ns"] for s in samples
        if s["type"] == "exec_boundary" and s["host_pid"] == clause_tgid
    )
    exit_candidates = [
        s["ts_ns"] for s in samples
        if s["type"] == "exit_boundary" and s["host_pid"] == clause_tgid
    ]
    exit_ts = max(exit_candidates) if exit_candidates else max(s["ts_ns"] for s in samples)

    in_window = [
        s for s in samples
        if s["type"] in {"perf", "exec_boundary", "exit_boundary"}
        and exec_ts <= s["ts_ns"] <= exit_ts
    ]
    attributed, gaps = [], []
    for s in in_window:
        if _attributed_to_clause(s["host_pid"], clause_tgid, fork_parent):
            attributed.append(s)
        else:
            gaps.append(s)

    # reconstructed CPU = sum over attributed tasks of that task's max cpu_ns
    per_task_max: dict[int, int] = {}
    for s in attributed:
        key = s["host_tid"]
        per_task_max[key] = max(per_task_max.get(key, 0), s["cpu_ns"])
    reconstructed_cpu_s = sum(per_task_max.values()) / 1e9
    reference_cpu_s = run["usage_usec"] / 1e6
    rel_err = (
        abs(reconstructed_cpu_s - reference_cpu_s) / reference_cpu_s
        if reference_cpu_s > 0 else None
    )

    # windowed peak cores over 500 ms wall windows, clipped to observed quota
    peak_cores, windows = _windowed_peak_cores(attributed, exec_ts, exit_ts, run["quota_cores"])

    return {
        "clause_tgid": clause_tgid,
        "clause_exec_seq": clause_seq,
        "distinct_tids_attributed": len(per_task_max),
        "in_window_sample_count": len(in_window),
        "attributed_sample_count": len(attributed),
        "coverage_gap_count": len(gaps),
        "attribution_coverage": round(len(attributed) / max(len(in_window), 1), 4),
        "reconstructed_cpu_s": round(reconstructed_cpu_s, 4),
        "reference_cgroup_cpu_s": round(reference_cpu_s, 4),
        "cpu_relative_error": None if rel_err is None else round(rel_err, 4),
        "wall_s": round(run["wall_s"], 3),
        "observed_quota_cores": run["quota_cores"],
        "windowed_peak_cpu_cores": round(peak_cores, 3),
        "window_count": windows,
        "gap_examples": gaps[:3],
    }


def _windowed_peak_cores(
    attributed: list[dict], start_ns: int, end_ns: int, quota: float
) -> tuple[float, int]:
    """Aggregate descendant CPU deltas into 500 ms wall windows -> max rate."""

    # per-task ordered (ts, cpu_ns) to difference within windows
    series: dict[int, list[tuple[int, int]]] = {}
    for s in attributed:
        series.setdefault(s["host_tid"], []).append((s["ts_ns"], s["cpu_ns"]))
    n_windows = max(1, int((end_ns - start_ns) // _WINDOW_NS) + 1)
    window_cpu = [0.0] * n_windows
    for points in series.values():
        points.sort()
        for (t0, c0), (t1, c1) in zip(points, points[1:]):
            if t1 <= t0 or c1 < c0:
                continue
            # attribute this task's cpu delta to the window of the midpoint
            mid = (t0 + t1) // 2
            w = min(n_windows - 1, max(0, int((mid - start_ns) // _WINDOW_NS)))
            window_cpu[w] += (c1 - c0)
    peak = 0.0
    for cpu_ns in window_cpu:
        rate = cpu_ns / (_WINDOW_NS)  # cores = cpu_ns per wall_ns of the window
        peak = max(peak, min(rate, quota))
    return peak, n_windows


def _rss_analysis(run: dict) -> dict:
    """Dedup threads sharing mm; sum distinct live mm at aligned timestamps."""

    samples = [s for s in run["samples"] if s["rss_pages"] > 0]
    clause_tgid, _, fork_parent = _clause_and_lineage(run["samples"])
    lineage = [
        s for s in samples
        if _attributed_to_clause(s["host_pid"], clause_tgid, fork_parent)
    ]
    # bin by aligned 20 ms wall bins; within a bin take latest rss per mm
    bins: dict[int, dict[int, int]] = {}
    mm_to_tids: dict[int, set[int]] = {}
    for s in lineage:
        b = s["ts_ns"] // _ALIGN_BIN_NS
        bins.setdefault(b, {})[s["mm_ptr"]] = s["rss_pages"]
        mm_to_tids.setdefault(s["mm_ptr"], set()).add(s["host_tid"])
    best_mb, best_detail = 0.0, {}
    for _b, per_mm in bins.items():
        agg_mb = sum(per_mm.values()) * _PAGE / 1e6
        if agg_mb > best_mb and len(per_mm) >= 2:
            best_mb = agg_mb
            best_detail = {hex(mm): round(p * _PAGE / 1e6, 1) for mm, p in per_mm.items()}
    shared_mm = {mm: sorted(tids) for mm, tids in mm_to_tids.items() if len(tids) >= 2}
    return {
        "distinct_address_spaces_seen": len(mm_to_tids),
        "threads_shared_mm_dedup": {hex(mm): tids for mm, tids in shared_mm.items()},
        "max_aligned_distinct_mm_count": len(best_detail),
        "sampled_peak_rss_mb": round(best_mb, 1),
        "peak_bin_per_mm_mb": best_detail,
    }


def _overhead(argv: list[str], reps: int) -> dict:
    """Sampler-on vs sampler-off wall + cgroup CPU over repeated runs."""

    def off_run() -> tuple[float, float]:
        cg = _new_cgroup(f"off_{time.monotonic_ns()}")
        _, wall_s, usage, _ = _run_in_cgroup(cg, argv)
        _rmdir(cg)
        return wall_s, usage / 1e6

    def on_run() -> tuple[float, float, int]:
        r = _sampled_run(f"on_{time.monotonic_ns()}", argv)
        return r["wall_s"], r["usage_usec"] / 1e6, r["reserve_failures"]

    off = [off_run() for _ in range(reps)]
    on = [on_run() for _ in range(reps)]
    off_wall = statistics.mean(w for w, _ in off)
    on_wall = statistics.mean(w for w, _, _ in on)
    off_cpu = statistics.mean(c for _, c in off)
    on_cpu = statistics.mean(c for _, c, _ in on)
    return {
        "reps": reps,
        "off_wall_s_mean": round(off_wall, 3),
        "on_wall_s_mean": round(on_wall, 3),
        "wall_overhead_pct": round(100 * (on_wall - off_wall) / off_wall, 2),
        "off_cgroup_cpu_s_mean": round(off_cpu, 3),
        "on_cgroup_cpu_s_mean": round(on_cpu, 3),
        "cpu_overhead_pct": round(100 * (on_cpu - off_cpu) / off_cpu, 2),
        "max_reserve_failures_under_load": max(f for _, _, f in on),
    }


def main() -> None:
    if os.geteuid() != 0:
        raise SystemExit("run as root: sudo python3 spike_perf_sampler.py")
    if not _WORKLOAD.exists():
        raise SystemExit(f"missing workload binary; gcc -O2 -pthread -o {_WORKLOAD} workload.c")

    t0 = time.monotonic()
    cpu_threads = _sampled_run("cpu_threads", ["cpu-threads", "2", "2.5"])
    cpu_forks = _sampled_run("cpu_forks", ["cpu-forks", "2", "2.5"])
    rss = _sampled_run("rss", ["rss", "150", "2.0"])
    overhead = _overhead(["cpu-threads", "2", "2.0"], reps=3)

    threads_cpu = _cpu_analysis(cpu_threads)
    forks_cpu = _cpu_analysis(cpu_forks)
    rss_out = _rss_analysis(rss)

    # ---- pass/fail against the user's explicit bars ----
    density_ok = threads_cpu["windowed_peak_cpu_cores"] >= 1.7 and cpu_threads["wall_s"] >= 2.0
    recon_ok = (
        threads_cpu["cpu_relative_error"] is not None
        and threads_cpu["cpu_relative_error"] <= 0.15
        and forks_cpu["cpu_relative_error"] is not None
        and forks_cpu["cpu_relative_error"] <= 0.15
    )
    attribution_ok = (
        threads_cpu["attribution_coverage"] == 1.0
        and forks_cpu["attribution_coverage"] == 1.0
        and threads_cpu["distinct_tids_attributed"] >= 3   # main + 2 threads
        and forks_cpu["distinct_tids_attributed"] >= 3     # main + 2 children
    )
    rss_ok = (
        rss_out["max_aligned_distinct_mm_count"] >= 2
        and rss_out["sampled_peak_rss_mb"] >= 200.0
        and len(rss_out["threads_shared_mm_dedup"]) >= 1
    )
    overhead_ok = (
        overhead["wall_overhead_pct"] <= 10.0
        and overhead["max_reserve_failures_under_load"] == 0
    )

    proofs = {
        "1_attach_and_density": {"pass": density_ok, **threads_cpu},
        "2_cpu_reconstruction": {
            "pass": recon_ok,
            "cpu_threads": {k: threads_cpu[k] for k in
                ("reconstructed_cpu_s", "reference_cgroup_cpu_s", "cpu_relative_error")},
            "cpu_forks": {k: forks_cpu[k] for k in
                ("reconstructed_cpu_s", "reference_cgroup_cpu_s", "cpu_relative_error")},
        },
        "3_lineage_attribution": {
            "pass": attribution_ok,
            "cpu_threads": {k: threads_cpu[k] for k in
                ("attribution_coverage", "attributed_sample_count",
                 "in_window_sample_count", "coverage_gap_count",
                 "distinct_tids_attributed")},
            "cpu_forks": {k: forks_cpu[k] for k in
                ("attribution_coverage", "attributed_sample_count",
                 "in_window_sample_count", "coverage_gap_count",
                 "distinct_tids_attributed")},
        },
        "4_rss_dedup_and_sum": {"pass": rss_ok, **rss_out},
        "5_overhead_and_loss": {"pass": overhead_ok, **overhead},
    }
    result = {
        "kernel": os.uname().release,
        "nproc": _NPROC,
        "sample_period_ns": _SAMPLE_PERIOD_NS,
        "window_ns": _WINDOW_NS,
        "elapsed_s": round(time.monotonic() - t0, 1),
        "all_pass": all(p["pass"] for p in proofs.values()),
        "proofs": proofs,
    }
    (_HERE / "spike-results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    sys.exit(0 if result["all_pass"] else 1)


if __name__ == "__main__":
    main()
