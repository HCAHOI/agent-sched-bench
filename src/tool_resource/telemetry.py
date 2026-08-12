"""Clause telemetry collector runtime and finalized artifact analysis.

Combines exec/fork/exit lifecycle collection with a perf CPU-clock sampler and
reconstructs, per
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
The local collector path runs a workload in a fresh cgroup-v2 scope. Root is
required.
"""

from __future__ import annotations

import ctypes
import heapq
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import threading
import time
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
)

from tool_resource.artifact_schema import (
    CLAUSE_TELEMETRY_COLLECTOR,
    CLAUSE_TELEMETRY_SCHEMA_VERSION,
    CLAUSE_TELEMETRY_STATUS_MODEL,
)

if TYPE_CHECKING:
    from tool_resource.clause_bridge import ShellCommandLookupFailure

# bcc is imported lazily inside the collection functions so the pure analysis
# (attribution, windowing, aggregation) can be imported and unit-tested without
# a bcc/BPF runtime.

SAMPLE_PERIOD_NS = 10_000_000  # ~10 ms CPU-time per perf callback
WINDOW_NS = 500_000_000  # 500 ms wall label window (resource_timeline semantics)
ALIGN_BIN_NS = 20_000_000  # 20 ms aligned bins for RSS summation
SENTINEL = 2**64 - 1
MAX_ARGS = 64
ARG_BYTES = 512
MAX_ARG_CHUNKS = 8
MAX_ARG_WORD_BYTES = (ARG_BYTES - 1) * MAX_ARG_CHUNKS
ARG_FLAG_TRUNCATED = 1
ARG_FLAG_ARGV_CAPPED = 2
ARG_FLAG_CONTINUED = 4
PAGE = os.sysconf("SC_PAGE_SIZE")
_NPROC = os.cpu_count() or 1


def _trim_process_heap() -> None:
    """Return freed glibc arenas after a collector's analysis high-water."""

    try:
        malloc_trim = ctypes.CDLL(None).malloc_trim
    except (AttributeError, OSError):
        return
    malloc_trim.argtypes = (ctypes.c_size_t,)
    malloc_trim.restype = ctypes.c_int
    malloc_trim(0)


def _restore_unset() -> object:
    return _UNSET


class _Unset:
    def __reduce__(
        self,
    ) -> tuple[Callable[[], object], tuple[Any, ...]]:
        return _restore_unset, ()


_UNSET = _Unset()
LOSS_COUNTER_NAMES = (
    "ringbuf_reserve_failures",
    "argv_read_failures",
    "argv_boundary_read_failures",
)
ARGV_READ_FAILURE_SITE_NAMES = (
    "argv_pointer",
    "argv_string",
    "argv_cap_pointer",
    "kernel_exec_metadata",
    "missing_bprm_capture",
    "unrecovered_filename",
)

TYPE_NAMES = {
    1: "exec_arg",
    2: "exec_boundary",
    3: "exit_boundary",
    4: "fork",
    5: "perf",
    6: "failed_exec_attempt",
    7: "exec_meta",
    8: "bprm_meta",
    9: "interp_meta",
}
TYPE_CODES = {name: code for code, name in TYPE_NAMES.items()}

BPF_PROGRAM = r"""
#include <linux/binfmts.h>
#include <linux/mm_types.h>
#include <linux/sched.h>
#include <linux/sched/signal.h>
#include <linux/version.h>
#include <uapi/linux/bpf_perf_event.h>

#define TYPE_EXEC_ARG 1
#define TYPE_EXEC_BOUNDARY 2
#define TYPE_EXIT_BOUNDARY 3
#define TYPE_FORK 4
#define TYPE_PERF 5
#define TYPE_FAILED_EXEC_ATTEMPT 6
#define TYPE_EXEC_META 7
#define TYPE_BPRM_META 8
#define TYPE_INTERP_META 9
#define MAX_ARGS 64
#define ARG_BYTES 512
#define MAX_ARG_CHUNKS 8
#define ARG_FLAG_TRUNCATED 1
#define ARG_FLAG_ARGV_CAPPED 2
#define ARG_FLAG_CONTINUED 4
#define ARGV_FAILURE_POINTER 0
#define ARGV_FAILURE_STRING 1
#define ARGV_FAILURE_CAP_POINTER 2
#define ARGV_FAILURE_KERNEL_META 3
#define ARGV_FAILURE_MISSING_BPRM 4
#define ARGV_FAILURE_FILENAME 5

/* The fields every event carries. Only four of the nine event types ever fill
 * the argv payload that follows, and those four are the minority: across 370
 * collected artifacts the argv word a payload holds averages 10 bytes, while a
 * perf sample -- by far the highest-volume type, one per 10ms of CPU time --
 * carries none at all. Two ring buffers, so a sample costs 128 bytes in the
 * ring instead of 640 and the same allocation absorbs a proportionally longer
 * burst before ringbuf_reserve starts failing (any failure voids the whole
 * tool call's evidence).
 *
 * event_small_t names its fields identically to event_t, so the emitters that
 * moved to the small ring are unchanged apart from their declaration.
 */
#define EVENT_COMMON_FIELDS \
    u64 timestamp_ns; \
    u64 cgroup_id; \
    u64 exec_seq; \
    u64 cpu_ns;         /* per-task cumulative utime+stime at sample time */ \
    u64 rss_pages;      /* CURRENT rss = file+anon+shmem (not hiwater) */ \
    u64 mm_ptr;         /* address-space identity for dedup */ \
    u64 hiwater_pages;  /* raw lifetime hiwater (exit only), kept separate */ \
    u64 io_read_bytes;  /* task->ioac.read_bytes */ \
    u64 io_write_bytes; /* task->ioac.write_bytes */ \
    u64 io_cancelled_write_bytes; /* task->ioac.cancelled_write_bytes */ \
    u32 type; \
    u32 host_pid; \
    u32 host_tid; \
    u32 parent_host_pid; \
    u32 child_host_pid; \
    u32 child_host_tid; \
    u32 arg_index; \
    u32 arg_chunk_index; \
    u32 arg_flags; \
    u32 exit_code;

struct event_t {
    EVENT_COMMON_FIELDS
    char arg[ARG_BYTES];
};

struct event_small_t {
    EVENT_COMMON_FIELDS
};

/* Page counts must be powers of two. The argv ring keeps its original size so
 * no workload mix can hold fewer argv events than before this split; the small
 * ring is added capacity, and a loss in either voids the call, so the safe
 * starting point is "never worse", not "same total bytes". 4MiB + 1MiB per
 * collector: 6553 argv-carrying events as before, plus 8192 counter events
 * that used to compete for the same space at five times the width. */
BPF_RINGBUF_OUTPUT(events, 1024);
BPF_RINGBUF_OUTPUT(events_small, 256);
BPF_ARRAY(target_cgroup, u64, 1);
BPF_ARRAY(ringbuf_reserve_failures, u64, 1);
/* Diagnostic only: which ring ran out. ringbuf_reserve_failures stays the
 * total both rings contribute to, so the persisted loss schema is unchanged
 * and a loss still voids the call the same way regardless of which ring it
 * came from. */
BPF_ARRAY(ringbuf_small_reserve_failures, u64, 1);
BPF_ARRAY(argv_read_failures, u64, 1);
BPF_ARRAY(argv_read_failure_sites, u64, 6);
BPF_ARRAY(argv_boundary_read_failures, u64, 1);
BPF_ARRAY(perf_sample_count, u64, 1);
BPF_PERCPU_ARRAY(next_exec_sequence, u64, 1);
struct task_key_t {
    u32 tid;
    u32 pad;
    u64 task_ptr;
};
struct pending_exec_t {
    u64 seq;
    u64 argv_ptr;
    u64 argv_captured;
    u64 argv_capture_incomplete;
    u64 filename_read_failed;
};
BPF_HASH(current_seq, struct task_key_t, u64);
BPF_HASH(pending_seq, struct task_key_t, struct pending_exec_t);

static int wanted(void) {
    u32 zero = 0;
    u64 *t = target_cgroup.lookup(&zero);
    return t && *t && *t == bpf_get_current_cgroup_id();
}

static void lost(u64 *counter) {
    if (counter) __sync_fetch_and_add(counter, 1);
}

static void ringbuf_reserve_failed(void) {
    u32 z = 0;
    lost(ringbuf_reserve_failures.lookup(&z));
}

static void ringbuf_small_reserve_failed(void) {
    u32 z = 0;
    lost(ringbuf_reserve_failures.lookup(&z));
    lost(ringbuf_small_reserve_failures.lookup(&z));
}

static void argv_read_failed(u32 site) {
    u32 z = 0;
    lost(argv_read_failures.lookup(&z));
    lost(argv_read_failure_sites.lookup(&site));
}

static void argv_boundary_read_failed(void) {
    u32 z = 0;
    lost(argv_boundary_read_failures.lookup(&z));
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
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 2, 0)
    bpf_probe_read_kernel(&file, sizeof(file), &mm->rss_stat[0].count);
    bpf_probe_read_kernel(&anon, sizeof(anon), &mm->rss_stat[1].count);
    bpf_probe_read_kernel(&shmem, sizeof(shmem), &mm->rss_stat[3].count);
#else
    bpf_probe_read_kernel(&file, sizeof(file), &mm->rss_stat.count[0].counter);
    bpf_probe_read_kernel(&anon, sizeof(anon), &mm->rss_stat.count[1].counter);
    bpf_probe_read_kernel(&shmem, sizeof(shmem), &mm->rss_stat.count[3].counter);
#endif
    long total = file + anon + shmem;
    return total < 0 ? 0 : (u64)total;
}

static u64 current_cpu_ns(struct task_struct *task) {
    u64 u = 0, s = 0;
    bpf_probe_read_kernel(&u, sizeof(u), &task->utime);
    bpf_probe_read_kernel(&s, sizeof(s), &task->stime);
    return u + s;
}

static void fill_counters(struct event_small_t *e, struct task_struct *task) {
    u64 mm_ptr = 0;
    e->rss_pages = current_rss_pages(task, &mm_ptr);
    e->mm_ptr = mm_ptr;
    e->cpu_ns = current_cpu_ns(task);
    bpf_probe_read_kernel(
        &e->io_read_bytes, sizeof(e->io_read_bytes), &task->ioac.read_bytes
    );
    bpf_probe_read_kernel(
        &e->io_write_bytes, sizeof(e->io_write_bytes), &task->ioac.write_bytes
    );
    bpf_probe_read_kernel(
        &e->io_cancelled_write_bytes,
        sizeof(e->io_cancelled_write_bytes),
        &task->ioac.cancelled_write_bytes
    );
}

static void capture_argv(
    u64 seq, const char *const *argv, u64 pid_tgid,
    struct pending_exec_t *pending, u32 failed_exec_fallback
) {
    u32 tid = pid_tgid;
    u32 captured_args = 0;
    #pragma unroll
    for (int i = 0; i < MAX_ARGS; i++) {
        const char *a = 0;
        int pointer_read = bpf_probe_read_user(&a, sizeof(a), &argv[i]);
        if (pointer_read < 0) {
            pending->argv_capture_incomplete = 1;
            if (!failed_exec_fallback) argv_read_failed(ARGV_FAILURE_POINTER);
            break;
        }
        if (!a) break;
        captured_args++;
        #pragma unroll
        for (int chunk = 0; chunk < MAX_ARG_CHUNKS; chunk++) {
            int offset = chunk * (ARG_BYTES - 1);
            struct event_t *e = events.ringbuf_reserve(sizeof(*e));
            if (!e) {
                ringbuf_reserve_failed();
                break;
            }
            __builtin_memset(e, 0, sizeof(*e));
            e->timestamp_ns = bpf_ktime_get_ns();
            e->cgroup_id = bpf_get_current_cgroup_id();
            e->exec_seq = seq;
            e->type = TYPE_EXEC_ARG;
            e->host_pid = pid_tgid >> 32;
            e->host_tid = tid;
            e->arg_index = i;
            e->arg_chunk_index = chunk;
            int arg_size = bpf_probe_read_user_str(
                e->arg, sizeof(e->arg), a + offset
            );
            int complete = 0;
            if (arg_size < 0) {
                pending->argv_capture_incomplete = 1;
                e->arg_flags = ARG_FLAG_TRUNCATED;
                if (!failed_exec_fallback) argv_read_failed(ARGV_FAILURE_STRING);
                complete = 1;
            } else if (arg_size == sizeof(e->arg)) {
                char source_last = 0;
                int last_read = bpf_probe_read_user(
                    &source_last,
                    sizeof(source_last),
                    a + offset + sizeof(e->arg) - 1
                );
                if (last_read < 0) {
                    pending->argv_capture_incomplete = 1;
                    e->arg_flags = ARG_FLAG_TRUNCATED;
                    if (!failed_exec_fallback) argv_boundary_read_failed();
                    complete = 1;
                } else if (source_last == '\0') {
                    complete = 1;
                } else if (chunk == MAX_ARG_CHUNKS - 1) {
                    e->arg_flags = ARG_FLAG_TRUNCATED;
                    complete = 1;
                } else {
                    e->arg_flags = ARG_FLAG_CONTINUED;
                }
            } else {
                complete = 1;
            }
            events.ringbuf_submit(e, 0);
            if (complete) break;
        }
    }
    const char *extra = 0;
    if (captured_args == MAX_ARGS) {
        int pointer_read = bpf_probe_read_user(
            &extra, sizeof(extra), &argv[MAX_ARGS]
        );
        if (pointer_read < 0) {
            pending->argv_capture_incomplete = 1;
            if (!failed_exec_fallback)
                argv_read_failed(ARGV_FAILURE_CAP_POINTER);
        }
    }
    if (extra) {
        struct event_t *e = events.ringbuf_reserve(sizeof(*e));
        if (!e) {
            ringbuf_reserve_failed();
        } else {
            __builtin_memset(e, 0, sizeof(*e));
            e->timestamp_ns = bpf_ktime_get_ns();
            e->cgroup_id = bpf_get_current_cgroup_id();
            e->exec_seq = seq;
            e->type = TYPE_EXEC_ARG;
            e->host_pid = pid_tgid >> 32;
            e->host_tid = tid;
            e->arg_index = MAX_ARGS;
            e->arg_flags = ARG_FLAG_ARGV_CAPPED;
            events.ringbuf_submit(e, 0);
        }
    }
}

static void emit_kernel_exec_meta(
    u32 type, u64 seq, const char *value, u32 argc, u64 pid_tgid
) {
    if (!value) return;
    struct event_t *e = events.ringbuf_reserve(sizeof(*e));
    if (!e) {
        ringbuf_reserve_failed();
        return;
    }
    __builtin_memset(e, 0, sizeof(*e));
    e->timestamp_ns = bpf_ktime_get_ns();
    e->cgroup_id = bpf_get_current_cgroup_id();
    e->exec_seq = seq;
    e->type = type;
    e->host_pid = pid_tgid >> 32;
    e->host_tid = (u32)pid_tgid;
    e->exit_code = argc;
    int size = bpf_probe_read_kernel_str(e->arg, sizeof(e->arg), value);
    if (size < 0)
        argv_read_failed(ARGV_FAILURE_KERNEL_META);
    else if (size == sizeof(e->arg))
        e->arg_flags = ARG_FLAG_TRUNCATED;
    events.ringbuf_submit(e, 0);
}

/* execve/execveat ENTRY: assign a new seq as PENDING without replacing the
 * current image. The saved vector is read after copy_strings() has faulted
 * valid cold pages; failed exec argv remains available on return. */
static int capture_enter(const char *filename, const char *const *argv) {
    u32 zero = 0;
    if (!wanted()) return 0;
    u64 *next_sequence = next_exec_sequence.lookup(&zero);
    if (!next_sequence) return 0;
    u64 local_sequence = *next_sequence;
    *next_sequence = local_sequence + 1;
    u64 seq =
        ((u64)bpf_get_smp_processor_id() << 48)
        | (local_sequence & 0x0000ffffffffffffULL);
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tid = pid_tgid;
    struct task_key_t task_key = {
        .tid = tid,
        .task_ptr = (u64)bpf_get_current_task(),
    };
    struct pending_exec_t pending = {
        .seq = seq,
        .argv_ptr = (u64)argv,
    };
    pending_seq.update(&task_key, &pending);
    struct event_t *e = events.ringbuf_reserve(sizeof(*e));
    if (!e) {
        ringbuf_reserve_failed();
    } else {
        __builtin_memset(e, 0, sizeof(*e));
        e->timestamp_ns = bpf_ktime_get_ns();
        e->cgroup_id = bpf_get_current_cgroup_id();
        e->exec_seq = seq;
        e->type = TYPE_EXEC_META;
        e->host_pid = pid_tgid >> 32;
        e->host_tid = tid;
        int filename_size = bpf_probe_read_user_str(
            e->arg, sizeof(e->arg), filename
        );
        if (filename_size < 0) {
            pending.filename_read_failed = 1;
            pending_seq.update(&task_key, &pending);
        } else if (filename_size == sizeof(e->arg)) {
            e->arg_flags = ARG_FLAG_TRUNCATED;
        }
        events.ringbuf_submit(e, 0);
    }
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_enter_execve) {
    return capture_enter(args->filename, (const char *const *)args->argv);
}

TRACEPOINT_PROBE(syscalls, sys_enter_execveat) {
    return capture_enter(args->filename, (const char *const *)args->argv);
}

/* copy_strings() has faulted the original argv pages before bprm_execve.
 * Capture here, before a script interpreter can rewrite the final argv. */
int capture_bprm_argv(struct pt_regs *ctx) {
    if (!wanted()) return 0;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tid = pid_tgid;
    struct task_key_t task_key = {
        .tid = tid,
        .task_ptr = (u64)bpf_get_current_task(),
    };
    struct pending_exec_t *pending = pending_seq.lookup(&task_key);
    if (!pending || pending->argv_captured) return 0;
    struct linux_binprm *bprm =
        (struct linux_binprm *)PT_REGS_PARM1(ctx);
    const char *filename = 0;
    const char *interp = 0;
    int argc = 0;
    bpf_probe_read_kernel(&filename, sizeof(filename), &bprm->filename);
    bpf_probe_read_kernel(&interp, sizeof(interp), &bprm->interp);
    bpf_probe_read_kernel(&argc, sizeof(argc), &bprm->argc);
    if (filename && pending->filename_read_failed) {
        emit_kernel_exec_meta(
            TYPE_EXEC_META, pending->seq, filename, argc, pid_tgid
        );
        pending->filename_read_failed = 0;
    }
    emit_kernel_exec_meta(
        TYPE_BPRM_META, pending->seq, filename, argc, pid_tgid
    );
    emit_kernel_exec_meta(
        TYPE_INTERP_META, pending->seq, interp, argc, pid_tgid
    );
    capture_argv(
        pending->seq,
        (const char *const *)pending->argv_ptr,
        pid_tgid,
        pending,
        0
    );
    pending->argv_captured = 1;
    return 0;
}

int capture_interp_change(struct pt_regs *ctx) {
    if (!wanted()) return 0;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tid = pid_tgid;
    struct task_key_t task_key = {
        .tid = tid,
        .task_ptr = (u64)bpf_get_current_task(),
    };
    struct pending_exec_t *pending = pending_seq.lookup(&task_key);
    if (!pending) return 0;
    const char *interp = (const char *)PT_REGS_PARM1(ctx);
    emit_kernel_exec_meta(
        TYPE_INTERP_META, pending->seq, interp, 0, pid_tgid
    );
    return 0;
}

/* execve RETURN: promote a successful image or close a failed attempt. */
static int on_exec_return(long ret) {
    if (!wanted()) return 0;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tid = pid_tgid;
    struct task_key_t task_key = {
        .tid = tid,
        .task_ptr = (u64)bpf_get_current_task(),
    };
    struct pending_exec_t *pending = pending_seq.lookup(&task_key);
    if (pending) {
        if (!pending->argv_captured && ret < 0) {
            capture_argv(
                pending->seq,
                (const char *const *)pending->argv_ptr,
                pid_tgid,
                pending,
                1
            );
        } else if (!pending->argv_captured) {
            argv_read_failed(ARGV_FAILURE_MISSING_BPRM);
        }
        if (pending->filename_read_failed) {
            argv_read_failed(ARGV_FAILURE_FILENAME);
        }
        if (ret >= 0) {
            current_seq.update(&task_key, &pending->seq);
        }
        struct event_small_t *e = events_small.ringbuf_reserve(sizeof(*e));
        if (!e) {
            ringbuf_small_reserve_failed();
        } else {
            __builtin_memset(e, 0, sizeof(*e));
            e->timestamp_ns = bpf_ktime_get_ns();
            e->cgroup_id = bpf_get_current_cgroup_id();
            e->exec_seq = pending->seq;
            e->type = ret < 0 ? TYPE_FAILED_EXEC_ATTEMPT : TYPE_EXEC_BOUNDARY;
            e->host_pid = pid_tgid >> 32;
            e->host_tid = tid;
            e->parent_host_pid = parent_tgid();
            if (ret < 0) {
                e->exit_code = (u32)(-ret);  /* positive errno */
                if (pending->argv_capture_incomplete)
                    e->arg_flags = ARG_FLAG_TRUNCATED;
            } else {
                fill_counters(
                    e, (struct task_struct *)bpf_get_current_task()
                );
            }
            events_small.ringbuf_submit(e, 0);
        }
    }
    pending_seq.delete(&task_key);
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_exit_execve) { return on_exec_return(args->ret); }
TRACEPOINT_PROBE(syscalls, sys_exit_execveat) { return on_exec_return(args->ret); }

/* Fork lineage must be TGID-consistent with every other event (which key on
 * tgid = pid_tgid>>32). The tracepoint's parent_pid/child_pid are TIDs; using
 * parent_pid directly breaks lineage when a non-leader thread forks. Record the
 * forking task's TGID as the parent. child_pid is always the new TID; it is also
 * the TGID for a process fork and supplies the zero I/O baseline for both
 * process and thread children. */
RAW_TRACEPOINT_PROBE(sched_process_fork) {
    if (!wanted()) return 0;
    struct task_struct *child = (struct task_struct *)ctx->args[1];
    u32 child_tid = 0;
    bpf_probe_read_kernel(&child_tid, sizeof(child_tid), &child->pid);
    struct task_key_t child_key = {
        .tid = child_tid,
        .task_ptr = (u64)child,
    };
    current_seq.delete(&child_key);
    pending_seq.delete(&child_key);
    struct event_small_t *e = events_small.ringbuf_reserve(sizeof(*e));
    if (!e) { ringbuf_small_reserve_failed(); return 0; }
    __builtin_memset(e, 0, sizeof(*e));
    e->timestamp_ns = bpf_ktime_get_ns();
    e->cgroup_id = bpf_get_current_cgroup_id();
    e->exec_seq = ~0ULL;
    e->type = TYPE_FORK;
    e->host_pid = bpf_get_current_pid_tgid() >> 32;  /* parent TGID */
    e->child_host_pid = child_tid;
    e->child_host_tid = child_tid;
    events_small.ringbuf_submit(e, 0);
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
    struct task_key_t task_key = {
        .tid = tid,
        .task_ptr = (u64)task,
    };
    u64 *seq = current_seq.lookup(&task_key);
    struct event_small_t *e = events_small.ringbuf_reserve(sizeof(*e));
    if (!e) { ringbuf_small_reserve_failed(); return 0; }
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
    events_small.ringbuf_submit(e, 0);
    return 0;
}

/* CPU-clock can sample a terminal task after sched_process_exit. Keep its exec
 * identity until the task_struct is actually released; the analysis still
 * excludes samples outside the original half-open exec window. This hook is
 * intentionally unfiltered because the freeing task need not share the dead
 * task's cgroup. A new in-scope fork also clears both child slots defensively. */
RAW_TRACEPOINT_PROBE(sched_process_free) {
    struct task_struct *task = (struct task_struct *)ctx->args[0];
    u32 tid = 0;
    bpf_probe_read_kernel(&tid, sizeof(tid), &task->pid);
    struct task_key_t task_key = {
        .tid = tid,
        .task_ptr = (u64)task,
    };
    current_seq.delete(&task_key);
    pending_seq.delete(&task_key);
    return 0;
}

int on_cpu_clock(struct bpf_perf_event_data *ctx) {
    if (!wanted()) return 0;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tid = pid_tgid;
    struct task_struct *task = (struct task_struct *)bpf_get_current_task();
    struct task_key_t task_key = {
        .tid = tid,
        .task_ptr = (u64)task,
    };
    u64 *seq = current_seq.lookup(&task_key);
    struct event_small_t *e = events_small.ringbuf_reserve(sizeof(*e));
    if (!e) { ringbuf_small_reserve_failed(); return 0; }
    __builtin_memset(e, 0, sizeof(*e));
    fill_counters(e, task);
    e->timestamp_ns = bpf_ktime_get_ns();
    e->cgroup_id = bpf_get_current_cgroup_id();
    e->exec_seq = seq ? *seq : ~0ULL;
    e->type = TYPE_PERF;
    e->host_pid = pid_tgid >> 32;
    e->host_tid = tid;
    events_small.ringbuf_submit(e, 0);
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
        self.pid_status_reads = 0
        self.pid_status_read_failures = 0
        self.read_error: str | None = None

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                pids = (self._cgroup / "cgroup.procs").read_text().split()
            except OSError as exc:
                self.read_error = f"cgroup.procs read failed: {exc}"
                break
            total = 0
            complete = True
            for pid in pids:
                try:
                    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                        if line.startswith("VmRSS:"):
                            total += int(line.split()[1])
                            self.pid_status_reads += 1
                            break
                except (FileNotFoundError, ProcessLookupError):
                    complete = False
                    break
                except OSError:
                    self.pid_status_read_failures += 1
                    complete = False
                    break
            if complete:
                if total > self.peak_sum_kb:
                    self.peak_sum_kb = total
                self.samples += 1
            time.sleep(self._interval)

    def stop(self) -> None:
        self._halt.set()


class MemoryCurrentOracle(threading.Thread):
    """Sample absolute task-cgroup charged memory for scheduler safety."""

    def __init__(self, cgroup: Path, interval_s: float = 0.002) -> None:
        super().__init__(daemon=True)
        self._path = cgroup / "memory.current"
        self._interval = interval_s
        self._halt = threading.Event()
        self.peak_bytes = 0
        self.samples = 0
        self.read_failures = 0
        self.read_error: str | None = None

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                current = int(self._path.read_text().strip())
            except (OSError, ValueError) as exc:
                self.read_failures += 1
                self.read_error = f"memory.current read failed: {exc}"
                break
            self.peak_bytes = max(self.peak_bytes, current)
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
    ringbuf_reserve_failures: int
    perf_sample_count: int
    oracle_peak_rss_kb: int
    oracle_samples: int
    marker: bool
    events: list[dict[str, Any]] = field(default_factory=list)
    lifecycle_map_entries: dict[str, int] = field(default_factory=dict)
    argv_read_failures: int = 0
    argv_boundary_read_failures: int = 0

    @property
    def loss_count(self) -> int:
        return (
            self.ringbuf_reserve_failures
            + self.argv_read_failures
            + self.argv_boundary_read_failures
        )

    @property
    def loss_counts(self) -> dict[str, int]:
        return {
            "ringbuf_reserve_failures": self.ringbuf_reserve_failures,
            "argv_read_failures": self.argv_read_failures,
            "argv_boundary_read_failures": self.argv_boundary_read_failures,
        }


@dataclass(frozen=True)
class ToolCallToken:
    tool_call_id: str
    command: str
    started_ns: int
    ringbuf_reserve_failures: int
    perf_sample_count: int
    argv_read_failures: int = 0
    argv_boundary_read_failures: int = 0
    source_tool_call_id: str = ""
    source_command: str = ""
    source_tool_result: str = ""
    static_plan: Mapping[str, Any] | None = None


_EXIT_CODE_DIAGNOSTIC = re.compile(r"^Exit code: (?P<code>-?\d+)$")
_PROTOCOL_TIMEOUT_MARKERS = frozenset(
    {"[timeout]", "[resource_timeout]", "[resource_stall_timeout]"}
)


def _anchored_command_not_found(text: str) -> tuple[str, str] | None:
    from tool_resource.clause_bridge import parse_shell_lookup_diagnostic

    matches = [
        (head, line)
        for line in text.splitlines()
        if (head := parse_shell_lookup_diagnostic(line)) is not None
    ]
    if not matches or len({head for head, _line in matches}) != 1:
        return None
    return matches[0]


def _strict_exit_code(tool_result: str) -> int | None:
    lines = [line for line in tool_result.splitlines() if line]
    if not lines or (match := _EXIT_CODE_DIAGNOSTIC.fullmatch(lines[-1])) is None:
        return None
    return int(match.group("code"))


def _is_protocol_timeout(replay_exit_code: int | None, replay_result: str) -> bool:
    return replay_exit_code == 124 and any(
        line.strip() in _PROTOCOL_TIMEOUT_MARKERS for line in replay_result.splitlines()
    )


def _replay_tool_result(
    replay_response: Mapping[str, Any] | None,
    replay_exit_code: int | None,
) -> str:
    if replay_response is None:
        return ""
    result = str(replay_response.get("result") or "")
    if not replay_response.get("ok", False) and not result.startswith("Error"):
        result = f"Error: {result}"
    if replay_exit_code is not None:
        return f"{result}\n\nExit code: {replay_exit_code}".strip()
    return result


def shell_command_lookup_failure_evidence(
    *,
    command: str,
    source_tool_call_id: str,
    replay_tool_call_id: str,
    source_command: str,
    source_tool_result: str,
    replay_result: str,
    replay_stderr: str,
    replay_exit_code: int | None,
) -> ShellCommandLookupFailure | None:
    """Return strict source/replay command-lookup evidence, else no evidence."""

    from tool_resource.clause_bridge import (
        ShellCommandLookupFailure,
        shell_lookup_exit_semantics,
    )

    if (
        not source_tool_call_id
        or not replay_tool_call_id
        or source_command != command
        or replay_exit_code not in {0, 127}
    ):
        return None
    source_exit_code = _strict_exit_code(source_tool_result)
    if source_exit_code != replay_exit_code:
        return None
    source_match = _anchored_command_not_found(source_tool_result)
    replay_channel = "raw_stderr" if replay_stderr else "tool_result"
    replay_match = _anchored_command_not_found(
        replay_stderr if replay_stderr else replay_result
    )
    if (
        source_match is None
        or replay_match is None
        or source_match[0] != replay_match[0]
    ):
        return None
    exit_code_semantics = shell_lookup_exit_semantics(
        command,
        source_match[0],
        source_exit_code,
    )
    if exit_code_semantics is None:
        return None
    return ShellCommandLookupFailure(
        executable_head=source_match[0],
        command=command,
        source_tool_call_id=source_tool_call_id,
        replay_tool_call_id=replay_tool_call_id,
        source_exit_code=source_exit_code,
        replay_exit_code=replay_exit_code,
        source_diagnostic=source_match[1],
        replay_diagnostic=replay_match[1],
        source_channel="source_tool_result",
        replay_channel=replay_channel,
        parser="anchored_shell_command_not_found_v1",
        exit_code_semantics=exit_code_semantics,
    )


def _source_exec_fields(
    action: Mapping[str, Any] | None,
) -> tuple[str, str, str]:
    data = action.get("data") if action is not None else None
    if not isinstance(data, Mapping):
        return "", "", ""
    raw_args = data.get("tool_args")
    if isinstance(raw_args, Mapping):
        args = raw_args
    else:
        try:
            parsed = json.loads(str(raw_args or "{}"))
        except (json.JSONDecodeError, TypeError):
            parsed = {}
        args = parsed if isinstance(parsed, Mapping) else {}
    return (
        str(data.get("tool_call_id") or ""),
        str(args.get("command") or ""),
        str(data.get("tool_result", data.get("result", "")) or ""),
    )


def _new_cgroup(tag: str) -> Path:
    cg = Path(f"/sys/fs/cgroup/clause_telemetry_{os.getpid()}_{tag}")
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
    bpf.attach_kprobe(event="bprm_execve", fn_name="capture_bprm_argv")
    bpf.attach_kprobe(event="bprm_change_interp", fn_name="capture_interp_change")
    bpf["target_cgroup"][ctypes.c_int(0)] = ctypes.c_ulonglong(cgroup_id)

    events: list[dict[str, Any]] = []
    lock = threading.Lock()
    table = bpf["events"]
    small_table = bpf["events_small"]

    def receiver(source: Any) -> "Callable[[int, int, int], int]":
        def receive(_ctx: int, data: int, _size: int) -> int:
            row = _event_row(source, data)
            with lock:
                events.append(row)
            return 0

        return receive

    table.open_ring_buffer(receiver(table))
    small_table.open_ring_buffer(receiver(small_table))
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
    loss_counts = _loss_counts(bpf)
    perf_count = bpf["perf_sample_count"][ctypes.c_int(0)].value
    lifecycle_map_entries = {
        name: sum(1 for _ in bpf[name].items())
        for name in ("current_seq", "pending_seq")
    }
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
        ringbuf_reserve_failures=loss_counts["ringbuf_reserve_failures"],
        perf_sample_count=perf_count,
        oracle_peak_rss_kb=oracle.peak_sum_kb,
        oracle_samples=oracle.samples,
        marker=marker.encode() in out if marker else True,
        events=[e for e in ordered if e["cgroup_id"] == cgroup_id],
        lifecycle_map_entries=lifecycle_map_entries,
        argv_read_failures=loss_counts["argv_read_failures"],
        argv_boundary_read_failures=loss_counts["argv_boundary_read_failures"],
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
    requested_executable_path: str | None
    requested_executable_path_truncated: bool
    bprm_filename: str | None
    bprm_interp: str | None
    bprm_evidence_truncated: bool
    exact_argc: int | None
    lineage_parent_pid: int | None
    terminal: bool
    has_causal_end: bool  # real exit (terminal) or next same-pid exec (non-terminal)
    argv_capture_flags: int = 0


# The event types each pass over a call's event stream actually branches on.
# A pass handed a view narrower than its set would silently drop evidence, so
# these live beside the functions that consume them and are covered by
# tests/test_clause_telemetry_analysis.py::test_type_filtered_views_match_full.
_ARGV_EVENT_TYPES = frozenset({"exec_arg", "exec_boundary", "failed_exec_attempt"})
_LINEAGE_EVENT_TYPES = frozenset(
    {
        "fork",
        "exec_meta",
        "bprm_meta",
        "interp_meta",
        "exit_boundary",
        "exec_boundary",
    }
)
_FORK_EVENT_TYPES = frozenset({"fork"})


def _events_of_types(
    events: Any,
    types: frozenset[str],
) -> Any:
    """Restrict ``events`` to ``types`` without decoding what is filtered out.

    Only worth doing for a disk-backed source, where skipping a record skips a
    deserialization; an in-memory list is already decoded, so it is returned
    untouched and the consumer's own type test does the work.
    """

    if isinstance(events, _SortedEventSource):
        return events.of_types(types)
    return events


def _captured_argv(
    events: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[tuple[int, int], dict[int, str]],
    dict[tuple[int, int], int],
]:
    chunks: dict[
        tuple[int, int],
        dict[int, dict[int, tuple[bytes, int]]],
    ] = {}
    capture_flags: dict[tuple[int, int], int] = {}
    words: dict[tuple[int, int], dict[int, str]] = {}

    def finish_exec(key: tuple[int, int]) -> None:
        by_index = chunks.pop(key, None)
        if by_index is None:
            return
        for index, word_chunks in by_index.items():
            ordered = sorted(word_chunks)
            flags = [word_chunks[chunk][1] for chunk in ordered]
            complete = (
                len(ordered) <= MAX_ARG_CHUNKS
                and ordered == list(range(len(ordered)))
                and all(flag == ARG_FLAG_CONTINUED for flag in flags[:-1])
                and flags[-1] == 0
            )
            if not complete:
                capture_flags[key] = capture_flags.get(key, 0) | (1 << index)
            words.setdefault(key, {})[index] = b"".join(
                word_chunks[chunk][0] for chunk in ordered
            ).decode("utf-8", "replace")

    for event in events:
        if event["type"] in {"exec_boundary", "failed_exec_attempt"}:
            key = (int(event["host_pid"]), int(event["exec_seq"]))
            if int(event.get("arg_flags", 0)) & ARG_FLAG_TRUNCATED:
                capture_flags[key] = capture_flags.get(key, 0) | (1 << MAX_ARGS)
            finish_exec(key)
            continue
        if event["type"] != "exec_arg":
            continue
        key = (int(event["host_pid"]), int(event["exec_seq"]))
        index = int(event["arg_index"])
        event_flags = int(event.get("arg_flags", 0))
        if index == MAX_ARGS and event_flags & ARG_FLAG_ARGV_CAPPED:
            capture_flags[key] = capture_flags.get(key, 0) | (1 << MAX_ARGS)
            continue
        if index >= MAX_ARGS:
            continue
        chunk_index = int(event.get("arg_chunk_index", 0))
        word_chunks = chunks.setdefault(key, {}).setdefault(index, {})
        if chunk_index in word_chunks:
            capture_flags[key] = capture_flags.get(key, 0) | (1 << index)
        payload = _event_arg_payload(event)
        word_chunks[chunk_index] = (payload, event_flags)

    # Consume each buffered exec as soon as its final words exist, so finish
    # does not retain both raw bytes and decoded argv for the whole call.
    for key in list(chunks):
        finish_exec(key)
    return words, capture_flags


def _clauses_and_lineage(
    events: list[dict[str, Any]],
    captured_argv: tuple[
        dict[tuple[int, int], dict[int, str]], dict[tuple[int, int], int]
    ]
    | None = None,
) -> tuple[list[Clause], dict[int, int]]:
    """Build per-clause windows and the child_tgid -> parent_tgid fork map.

    ``captured_argv`` lets a caller that already ran :func:`_captured_argv`
    reuse it; the argv reassembly decodes every ``exec_arg`` chunk, which is the
    bulk of an event stream.

    A terminal exec that never exited is bounded by the last timestamp in the
    stream. When ``events`` is a view restricted to ``_LINEAGE_EVENT_TYPES``
    that bound has to come from the unfiltered source, or the clause would end
    early; ``_EventTypeView`` carries it as ``source_max_ts_ns``.
    """

    last_ts: int | None = getattr(events, "source_max_ts_ns", None)
    fork_parent: dict[int, int] = {}
    exact_argc: dict[tuple[int, int], int | None] = {}
    requested_paths: dict[tuple[int, int], str] = {}
    requested_path_truncated: set[tuple[int, int]] = set()
    bprm_filenames: dict[tuple[int, int], str] = {}
    bprm_interpreters: dict[tuple[int, int], str] = {}
    bprm_truncated: set[tuple[int, int]] = set()
    exits: dict[int, int] = {}
    execs_by_pid: dict[int, list[dict[str, Any]]] = {}
    for e in events:
        event_type = e["type"]
        last_ts = e["ts_ns"] if last_ts is None else max(last_ts, e["ts_ns"])
        if event_type == "fork" and e["child_host_pid"]:
            fork_parent.setdefault(e["child_host_pid"], e["host_pid"])
        elif event_type == "exec_meta":
            key = (e["host_pid"], e["exec_seq"])
            requested_paths[key] = e.get("arg", "")
            if int(e.get("arg_flags", 0)) & ARG_FLAG_TRUNCATED:
                requested_path_truncated.add(key)
        elif event_type == "bprm_meta":
            key = (e["host_pid"], e["exec_seq"])
            bprm_filenames[key] = e.get("arg", "")
            argc = int(e.get("exit_code") or 0)
            if argc > 0:
                exact_argc[key] = argc
            if int(e.get("arg_flags", 0)) & ARG_FLAG_TRUNCATED:
                bprm_truncated.add(key)
        elif event_type == "interp_meta":
            key = (e["host_pid"], e["exec_seq"])
            bprm_interpreters[key] = e.get("arg", "")
            if int(e.get("arg_flags", 0)) & ARG_FLAG_TRUNCATED:
                bprm_truncated.add(key)
        elif event_type == "exit_boundary":
            exits[e["host_pid"]] = max(
                exits.get(e["host_pid"], 0),
                e["ts_ns"],
            )
        elif event_type == "exec_boundary" and e["exec_seq"] != SENTINEL:
            execs_by_pid.setdefault(e["host_pid"], []).append(e)

    if captured_argv is None:
        # A lineage-restricted view carries no exec_arg events, so reassembling
        # argv from it would hand every clause an empty argv and a "" bin
        # without failing. A caller that passes a view must pass the argv too.
        assert not isinstance(events, _EventTypeView), (
            "_clauses_and_lineage needs precomputed argv when given a filtered view"
        )
    argv_words, argv_capture_flags = (
        _captured_argv(events) if captured_argv is None else captured_argv
    )

    def argv_of(pid: int, seq: int) -> tuple[tuple[str, ...], int]:
        words = argv_words.get((pid, seq), {})
        return (
            tuple(words[i] for i in sorted(words)),
            argv_capture_flags.get((pid, seq), 0),
        )

    # exec boundaries per pid, ordered -> clause windows
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
                # execs_by_pid is non-empty here, so the stream had a timestamp.
                assert last_ts is not None
                t_end = last_ts  # synthetic bound; NOT a real causal end
                has_causal_end = False
            argv, capture_flags = argv_of(pid, e["exec_seq"])
            clauses.append(
                Clause(
                    host_pid=pid,
                    exec_seq=e["exec_seq"],
                    t_exec_ns=e["ts_ns"],
                    t_end_ns=t_end,
                    bin=Path(argv[0]).name if argv else "",
                    argv=argv,
                    requested_executable_path=requested_paths.get((pid, e["exec_seq"])),
                    requested_executable_path_truncated=(
                        (pid, e["exec_seq"]) in requested_path_truncated
                    ),
                    bprm_filename=bprm_filenames.get((pid, e["exec_seq"])),
                    bprm_interp=bprm_interpreters.get((pid, e["exec_seq"])),
                    bprm_evidence_truncated=((pid, e["exec_seq"]) in bprm_truncated),
                    exact_argc=exact_argc.get((pid, e["exec_seq"]), len(argv)),
                    lineage_parent_pid=fork_parent.get(pid),
                    terminal=terminal,
                    has_causal_end=has_causal_end,
                    argv_capture_flags=capture_flags,
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
    pid: int,
    ts: int,
    clause_by_pid: dict[int, list[Clause]],
    fork_parent: dict[int, int],
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
    requested_executable_path: str | None
    requested_executable_path_truncated: bool
    bprm_filename: str | None
    bprm_interp: str | None
    bprm_evidence_truncated: bool
    exact_argc: int | None
    lineage_parent_pid: int | None  # fork parent, for bridge lineage attribution
    terminal: bool
    has_causal_end: bool  # real exit or next same-pid exec; fail closed if False
    t_exec_ns: int  # clause-window bounds, for the bridge time-aligned merge
    t_end_ns: int
    wall_ns: int
    cpu_ns_cumulative: int  # raw, preserved separately
    exit_signal: int | None  # low 7 bits of exit_code on the terminal exit
    normal_exit_status: int | None  # wait status high byte; unavailable on signal
    peak_cpu_cores: float | None
    peak_cpu_cores_reason: str
    sampled_peak_rss_mb: float | None
    sampled_peak_rss_reason: str
    disk_read_bytes_total: int | None
    disk_write_bytes_total: int | None
    disk_cancelled_write_bytes_total: int | None
    disk_io_reason: str
    # time-aligned profiles the clause bridge merges across owned images
    cpu_windows: tuple[tuple[int, int], ...]
    rss_bins: tuple[tuple[int, int, float], ...]
    provenance: dict[str, Any]
    argv_capture_flags: int = 0


def _attribute(
    events: list[dict[str, Any]],
    clauses: list[Clause],
    fork_parent: dict[int, int],
    *,
    entry_pid: int | None = None,
    on_attributed: Callable[[Clause, Mapping[str, Any]], None] | None = None,
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
    fork_records: dict[int, list[dict[str, Any]]] = {}
    exec_boundaries_by_tid: dict[int, list[dict[str, Any]]] = {}
    boundary_events_by_tid: dict[int, list[dict[str, Any]]] = {}
    exec_arg_start_by_tid_seq: dict[tuple[int, int], int] = {}
    for event in events:
        event_type = event["type"]
        if event_type == "fork" and event.get("child_host_pid"):
            fork_records.setdefault(event["child_host_pid"], []).append(event)
        elif event_type == "exec_boundary":
            exec_boundaries_by_tid.setdefault(event["host_tid"], []).append(event)
            boundary_events_by_tid.setdefault(event["host_tid"], []).append(event)
        elif event_type == "exit_boundary":
            boundary_events_by_tid.setdefault(event["host_tid"], []).append(event)
        elif event_type == "exec_arg" and event["arg_index"] == 0:
            key = (event["host_tid"], event["exec_seq"])
            exec_arg_start_by_tid_seq[key] = min(
                exec_arg_start_by_tid_seq.get(key, event["ts_ns"]),
                event["ts_ns"],
            )

    def pre_exec_owner(
        event: dict[str, Any],
    ) -> tuple[Clause | None, dict[str, Any], str | None]:
        ts, pid, tid = event["ts_ns"], event["host_pid"], event["host_tid"]
        if any(
            boundary["ts_ns"] <= ts for boundary in exec_boundaries_by_tid.get(tid, ())
        ):
            return None, {}, "sentinel_after_successful_exec"

        lineage_id = tid if tid != pid else pid
        # The collector can arm after the initial command process was forked.
        # A CPU-clock sample may then land after sys_enter_execve captured one
        # pending argv but before sys_exit_execve promotes that same seq.
        # It belongs to neither the not-yet-successful image nor an observable
        # fork ancestor. Preserve it as structural setup only with a unique,
        # same-TID successful boundary that strictly closes the pending window.
        # Failed execs and samples without that future boundary remain fatal.
        if not fork_records.get(lineage_id):
            pending_successes = [
                boundary
                for boundary in exec_boundaries_by_tid.get(tid, ())
                if boundary["ts_ns"] > ts
                and (
                    arg_start := exec_arg_start_by_tid_seq.get(
                        (tid, boundary["exec_seq"])
                    )
                )
                is not None
                and arg_start <= ts
            ]
            if len(pending_successes) == 1:
                boundary = pending_successes[0]
                arg_start = exec_arg_start_by_tid_seq[(tid, boundary["exec_seq"])]
                return (
                    None,
                    {
                        "pending_exec_evidence": {
                            "host_pid": pid,
                            "host_tid": tid,
                            "pending_exec_seq": boundary["exec_seq"],
                            "exec_arg_start_ns": arg_start,
                            "sample_ts_ns": ts,
                            "successful_exec_boundary_ns": boundary["ts_ns"],
                        }
                    },
                    "initial_exec_pending_pre_boundary_structural_setup",
                )

        ancestry = [lineage_id]
        current = lineage_id
        seen = {lineage_id}
        first_fork_ts: int | None = None
        ancestor_ts_bound = ts
        fork_chain_records: list[dict[str, int]] = []
        while current != entry_pid:
            all_records = fork_records.get(current, ())
            eligible = [
                record for record in all_records if record["ts_ns"] <= ancestor_ts_bound
            ]
            if len(eligible) != 1:
                reason = (
                    "sentinel_pre_exec_ambiguous_fork_ancestry"
                    if len(eligible) > 1
                    else "sentinel_pre_exec_missing_fork_ancestry"
                )
                record_rows = [
                    {
                        "parent_pid": int(record["host_pid"]),
                        "ts_ns": int(record["ts_ns"]),
                    }
                    for record in all_records
                ]
                return (
                    None,
                    {
                        "fork_ancestry": ancestry,
                        "fork_chain_records": fork_chain_records,
                        "fork_resolution_failure": {
                            "failure_kind": (
                                "ambiguous_generation"
                                if len(eligible) > 1
                                else "missing_generation"
                            ),
                            "child_id": current,
                            "timestamp_bound_ns": ancestor_ts_bound,
                            "eligible_records": sorted(
                                (
                                    row
                                    for row in record_rows
                                    if row["ts_ns"] <= ancestor_ts_bound
                                ),
                                key=lambda row: (
                                    row["ts_ns"],
                                    row["parent_pid"],
                                ),
                            ),
                            "rejected_records": sorted(
                                (
                                    row
                                    for row in record_rows
                                    if row["ts_ns"] > ancestor_ts_bound
                                ),
                                key=lambda row: (
                                    row["ts_ns"],
                                    row["parent_pid"],
                                ),
                            ),
                        },
                    },
                    reason,
                )
            fork_record = eligible[0]
            parent = fork_record["host_pid"]
            if first_fork_ts is None:
                first_fork_ts = fork_record["ts_ns"]
            if parent <= 0 or parent in seen:
                record_rows = [
                    {
                        "parent_pid": int(record["host_pid"]),
                        "ts_ns": int(record["ts_ns"]),
                    }
                    for record in all_records
                ]
                return (
                    None,
                    {
                        "fork_ancestry": ancestry,
                        "fork_chain_records": fork_chain_records,
                        "fork_resolution_failure": {
                            "failure_kind": (
                                "nonpositive_parent" if parent <= 0 else "cyclic_parent"
                            ),
                            "child_id": current,
                            "timestamp_bound_ns": ancestor_ts_bound,
                            "eligible_records": sorted(
                                (
                                    row
                                    for row in record_rows
                                    if row["ts_ns"] <= ancestor_ts_bound
                                ),
                                key=lambda row: (
                                    row["ts_ns"],
                                    row["parent_pid"],
                                ),
                            ),
                            "rejected_records": sorted(
                                (
                                    row
                                    for row in record_rows
                                    if row["ts_ns"] > ancestor_ts_bound
                                ),
                                key=lambda row: (
                                    row["ts_ns"],
                                    row["parent_pid"],
                                ),
                            ),
                        },
                    },
                    "sentinel_pre_exec_ambiguous_fork_ancestry",
                )
            ancestry.append(parent)
            fork_chain_records.append(
                {
                    "child_id": current,
                    "parent_pid": parent,
                    "ts_ns": fork_record["ts_ns"],
                }
            )
            seen.add(parent)
            current = parent
            ancestor_ts_bound = fork_record["ts_ns"]

        match = next(
            (
                active
                for ancestor in ancestry[1:]
                if (
                    active := _clause_at(
                        clause_by_pid.get(ancestor, ()),
                        ts,
                    )
                )
                is not None
            ),
            None,
        )
        if match is not None:
            endpoint = min(
                (
                    candidate
                    for candidate in boundary_events_by_tid.get(tid, ())
                    if candidate["ts_ns"] > ts
                ),
                key=lambda candidate: candidate["ts_ns"],
                default=None,
            )
            provenance = {
                "kind": "inherited_active_exec_owner",
                "original_type": event["type"],
                "original_ts_ns": ts,
                "original_host_pid": pid,
                "original_host_tid": tid,
                "original_exec_seq": event["exec_seq"],
                "owner_host_pid": match.host_pid,
                "owner_exec_seq": match.exec_seq,
                "fork_ancestry": ancestry,
                "fork_chain_records": fork_chain_records,
                "fork_ts_ns": first_fork_ts,
                "cpu_counter_support": {
                    "baseline": {
                        "ts_ns": first_fork_ts,
                        "cpu_ns": 0,
                        "source": "new_fork_zero",
                    },
                    "endpoint": (
                        {
                            "type": endpoint["type"],
                            "ts_ns": endpoint["ts_ns"],
                            "host_pid": endpoint["host_pid"],
                            "host_tid": endpoint["host_tid"],
                            "exec_seq": endpoint["exec_seq"],
                            "cpu_ns": endpoint["cpu_ns"],
                        }
                        if endpoint is not None
                        else None
                    ),
                },
            }
            return match, provenance, None

        return (
            None,
            {
                "fork_ancestry": ancestry,
                "fork_chain_records": fork_chain_records,
                "fork_ts_ns": first_fork_ts,
            },
            "entry_fork_pre_exec_structural_setup",
        )

    gaps: list[dict[str, Any]] = []
    for e in events:
        if e["type"] not in {"perf", "exec_boundary", "exit_boundary"}:
            continue
        ts, pid, tid, seq = (
            e["ts_ns"],
            e["host_pid"],
            e["host_tid"],
            e["exec_seq"],
        )
        target: Clause | None = None
        attributed_event = e
        # 1) DIRECT-SEQ: the sample carries the exec_seq of a clause on its pid
        #    (exec/exit boundaries, and perf on the exec'ing thread) — exact,
        #    taken before any window/lineage fallback.
        if seq != SENTINEL:
            target = by_pid_seq.get((pid, seq))
            if (
                target is not None
                and e["type"] == "perf"
                and not (target.t_exec_ns <= ts < target.t_end_ns)
            ):
                attributed_event = {
                    **e,
                    "metric_excluded": {
                        "reason": "outside_half_open_exec_window",
                        "t_exec_ns": target.t_exec_ns,
                        "t_end_ns": target.t_end_ns,
                        "offset_from_end_ns": ts - target.t_end_ns,
                    },
                }
        elif pid != entry_pid or tid in fork_records:
            target, provenance, reason = pre_exec_owner(e)
            if target is not None:
                attributed_event = {**e, "attribution": provenance}
            elif reason is not None:
                gaps.append({**e, **provenance, "reason": reason})
                continue
        # Non-sentinel unmatched events keep the existing causal ancestor
        # fallback. Sentinel events use only the stricter pre-exec contract.
        if target is None and seq != SENTINEL:
            target = _clause_at(clause_by_pid.get(pid, ()), ts)
            if target is None:
                target = _ancestor_clause_pid(pid, ts, clause_by_pid, fork_parent)
        if target is None:
            gaps.append(
                {
                    **e,
                    "reason": (
                        "sentinel_exec_seq_without_active_exec_image_or_owned_ancestor"
                        if seq == SENTINEL
                        else "exec_seq_without_matching_exec_image_or_owned_ancestor"
                    ),
                }
            )
        else:
            if on_attributed is None:
                per_clause[(target.host_pid, target.exec_seq)].append(attributed_event)
            else:
                on_attributed(target, attributed_event)
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


def _cpu_counter_points(
    samples: list[dict[str, Any]],
) -> dict[int, list[tuple[int, int]]]:
    per_tid: dict[int, dict[int, int]] = {}
    for sample in samples:
        if sample["cpu_ns"] > 0:
            per_tid.setdefault(sample["host_tid"], {})[sample["ts_ns"]] = sample[
                "cpu_ns"
            ]
        support = sample.get("attribution", {}).get("cpu_counter_support")
        if not isinstance(support, dict):
            continue
        for point in (support.get("baseline"), support.get("endpoint")):
            if not isinstance(point, dict):
                continue
            ts_ns, cpu_ns = point.get("ts_ns"), point.get("cpu_ns")
            if isinstance(ts_ns, int) and isinstance(cpu_ns, int):
                per_tid.setdefault(sample["host_tid"], {})[ts_ns] = cpu_ns
    return {tid: sorted(points.items()) for tid, points in per_tid.items()}


def _cpu_window_profile_from_points(
    per_tid_points: dict[int, list[tuple[int, int]]],
) -> tuple[tuple[int, int], ...]:
    windows: dict[int, float] = {}
    for points in per_tid_points.values():
        for (t0, c0), (t1, c1) in zip(points, points[1:]):
            if t1 <= t0 or c1 < c0:
                continue
            for widx, part in _apportion(t0, t1, c1 - c0):
                windows[widx] = windows.get(widx, 0.0) + part
    return tuple((w, int(round(v))) for w, v in sorted(windows.items()))


def cpu_window_profile(samples: list[dict[str, Any]]) -> tuple[tuple[int, int], ...]:
    """Absolute-indexed (window_idx, cpu_ns) contributions for the clause bridge.

    Windows are keyed by ``ts // WINDOW_NS`` (a common absolute grid) so the
    bridge can SUM concurrent owned images per window; each per-TID cpu delta is
    apportioned across every window it intersects.
    """

    return _cpu_window_profile_from_points(_cpu_counter_points(samples))


def _peak_cpu_cores(
    samples: list[dict[str, Any]],
    clause: Clause,
    quota: float,
    *,
    per_tid_points: dict[int, list[tuple[int, int]]],
    profile: tuple[tuple[int, int], ...],
) -> tuple[float | None, str, dict[str, Any]]:
    """``per_tid_points``/``profile`` are computed once per clause by ``analyze``
    and passed in: deriving them here re-walked the same samples three times."""

    cpu_samples = [s for s in samples if s["cpu_ns"] > 0]
    cpu_counter_points = sum(len(points) for points in per_tid_points.values())
    span = clause.t_end_ns - clause.t_exec_ns
    prov = {
        "cpu_sample_count": len(cpu_samples),
        "cpu_counter_point_count": cpu_counter_points,
        "cpu_windows": len(profile),
        "span_s": round(span / 1e9, 3),
    }
    # resource_timeline eligibility: clause >= 1 s AND >= 2 CPU samples.
    if span < _MIN_ELIGIBLE_SPAN_NS:
        return None, "clause_shorter_than_1s_ineligible_for_peak", prov
    if cpu_counter_points < 2 or not profile:
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


_IO_COUNTER_FIELDS = (
    "io_read_bytes",
    "io_write_bytes",
    "io_cancelled_write_bytes",
)


def _fork_io_baselines(
    events: list[dict[str, Any]],
    clauses: list[Clause],
    fork_parent: dict[int, int],
) -> dict[tuple[int, int], dict[int, int]]:
    """Map each newly forked TID to the exec image active at its fork."""

    clause_by_pid: dict[int, list[Clause]] = {}
    for clause in clauses:
        clause_by_pid.setdefault(clause.host_pid, []).append(clause)
    baselines: dict[tuple[int, int], dict[int, int]] = {
        (clause.host_pid, clause.exec_seq): {} for clause in clauses
    }
    for event in events:
        if event["type"] != "fork" or not event.get("child_host_tid"):
            continue
        target = _clause_at(clause_by_pid.get(event["host_pid"], ()), event["ts_ns"])
        if target is None:
            target = _ancestor_clause_pid(
                event["host_pid"],
                event["ts_ns"],
                clause_by_pid,
                fork_parent,
            )
        if target is not None:
            baselines[(target.host_pid, target.exec_seq)].setdefault(
                event["child_host_tid"], event["ts_ns"]
            )
    return baselines


def _task_io_totals(
    samples: list[dict[str, Any]],
    clause: Clause,
    fork_baselines: dict[int, int],
    *,
    exec_baseline_index: Mapping[tuple[int, int, int], Sequence[dict[str, Any]]],
    boundary_events_by_tid: Mapping[int, Sequence[dict[str, Any]]],
    boundary_ts_by_tid: Mapping[int, Sequence[int]],
) -> tuple[tuple[int, int, int] | None, str, dict[str, Any]]:
    """Exact task-I/O-accounting deltas for one exec image.

    The image's exec boundary is its surviving TID's baseline. A new forked
    TID starts from the kernel's zeroed task I/O accounting. The first later
    exec or exit boundary is the exact endpoint, so adjacent exec images and
    owned descendants remain disjoint; perf samples are diagnostic only.

    The baseline and endpoint lookups are indexed rather than scanned: both were
    linear in the events on the clause's pid/tid, so a pid carrying many exec
    images cost quadratic time overall.
    """

    return _task_io_totals_from_state(
        attributed_tids={event["host_tid"] for event in samples},
        perf_sample_count=sum(event["type"] == "perf" for event in samples),
        clause=clause,
        fork_baselines=fork_baselines,
        exec_baseline_index=exec_baseline_index,
        boundary_events_by_tid=boundary_events_by_tid,
        boundary_ts_by_tid=boundary_ts_by_tid,
    )


def _task_io_totals_from_state(
    *,
    attributed_tids: set[int],
    perf_sample_count: int,
    clause: Clause,
    fork_baselines: dict[int, int],
    exec_baseline_index: Mapping[
        tuple[int, int, int],
        Sequence[dict[str, Any]],
    ],
    boundary_events_by_tid: Mapping[int, Sequence[dict[str, Any]]],
    boundary_ts_by_tid: Mapping[int, Sequence[int]],
) -> tuple[tuple[int, int, int] | None, str, dict[str, Any]]:
    exec_baselines = exec_baseline_index.get(
        (clause.host_pid, clause.exec_seq, clause.t_exec_ns), ()
    )
    provenance: dict[str, Any] = {
        "source": "linux_task_io_accounting",
        "reduction": "nonnegative_per_tid_deltas_then_sum",
        "fields": {
            "read_bytes": "task->ioac.read_bytes",
            "write_bytes": "task->ioac.write_bytes",
            "cancelled_write_bytes": "task->ioac.cancelled_write_bytes",
        },
        "exec_boundary_baseline_tids": [],
        "zero_fork_baseline_tids": sorted(fork_baselines),
        "exact_endpoint_tids": [],
        "perf_sample_count": perf_sample_count,
        "counter_regression_clamps": 0,
    }
    if len(exec_baselines) != 1:
        provenance["exec_boundary_count"] = len(exec_baselines)
        return None, "missing_or_ambiguous_exec_io_baseline", provenance

    root = exec_baselines[0]
    baselines: dict[int, tuple[int, dict[str, int]]] = {
        root["host_tid"]: (
            root["ts_ns"],
            {field: int(root[field]) for field in _IO_COUNTER_FIELDS},
        )
    }
    provenance["exec_boundary_baseline_tids"] = [root["host_tid"]]
    for tid, ts_ns in fork_baselines.items():
        baselines.setdefault(
            tid,
            (ts_ns, dict.fromkeys(_IO_COUNTER_FIELDS, 0)),
        )

    missing_baselines = sorted(attributed_tids - set(baselines))
    if missing_baselines:
        provenance["missing_baseline_tids"] = missing_baselines
        return None, "missing_tid_io_baseline", provenance

    totals = dict.fromkeys(_IO_COUNTER_FIELDS, 0)
    for tid, (baseline_ts, baseline) in baselines.items():
        # Boundaries for this tid are timestamp-sorted, so the first one after
        # the baseline is the earliest endpoint — the same event the previous
        # filter-then-min produced, including its tie-break on equal timestamps.
        boundaries = boundary_events_by_tid.get(tid, ())
        index = bisect_right(boundary_ts_by_tid.get(tid, ()), baseline_ts)
        endpoint = (
            boundaries[index]
            if index < len(boundaries) and boundaries[index]["ts_ns"] <= clause.t_end_ns
            else None
        )
        if endpoint is None:
            provenance["missing_endpoint_tids"] = sorted(
                {
                    *provenance.get("missing_endpoint_tids", []),
                    tid,
                }
            )
            continue
        provenance["exact_endpoint_tids"].append(tid)
        for counter_field in _IO_COUNTER_FIELDS:
            delta = int(endpoint[counter_field]) - baseline[counter_field]
            if delta < 0:
                provenance["counter_regression_clamps"] += 1
                delta = 0
            totals[counter_field] += delta

    if provenance.get("missing_endpoint_tids"):
        return None, "missing_exact_tid_io_endpoint", provenance
    if provenance["counter_regression_clamps"]:
        return None, "io_counter_regression", provenance
    return (
        (
            totals["io_read_bytes"],
            totals["io_write_bytes"],
            totals["io_cancelled_write_bytes"],
        ),
        "ok",
        provenance,
    )


class _AttributionTables:
    """Deduplicated inherited-sample attribution evidence for one clause.

    Every pre-exec sample on the same lineage carries the same fork chain, and
    a clause with a long-running descendant accumulates thousands of them. The
    chains and CPU-counter supports are stored once here and referenced by
    index, so the per-sample rows hold only what actually varies.

    Interning per clause loses nothing: a fork chain belongs to exactly one
    lineage, which is owned by exactly one clause.
    """

    def __init__(self) -> None:
        self.fork_chains: list[Sequence[Mapping[str, int]]] = []
        self.cpu_counter_supports: list[Mapping[str, Any]] = []
        self._chain_refs: dict[tuple[Any, ...], int] = {}
        self._support_refs: dict[tuple[Any, ...], int] = {}

    @staticmethod
    def _ref(table: list[Any], refs: dict[tuple[Any, ...], int], key, value) -> int:
        existing = refs.get(key)
        if existing is not None:
            return existing
        ref = len(table)
        refs[key] = ref
        table.append(value)
        return ref

    def _fork_chain_ref(self, records: Sequence[Mapping[str, int]]) -> int:
        key = tuple(
            (record["child_id"], record["parent_pid"], record["ts_ns"])
            for record in records
        )
        return self._ref(self.fork_chains, self._chain_refs, key, records)

    def _cpu_counter_support_ref(self, support: Mapping[str, Any]) -> int:
        key = tuple(
            None if point is None else tuple(sorted(point.items()))
            for point in (support.get("baseline"), support.get("endpoint"))
        )
        return self._ref(self.cpu_counter_supports, self._support_refs, key, support)

    def row(self, attribution: Mapping[str, Any]) -> dict[str, Any]:
        """One per-sample record with its shared evidence replaced by references.

        ``fork_ancestry`` and ``fork_ts_ns`` are not stored: both are exactly
        recoverable from the referenced chain — ancestry is
        ``[chain[0].child_id] + [record.parent_pid for record in chain]`` and
        ``fork_ts_ns`` is ``chain[0].ts_ns``. An ``inherited_active_exec_owner``
        attribution is only produced after the fork walk ran, so its chain is
        never empty.
        """

        return {
            "kind": attribution["kind"],
            "original_type": attribution["original_type"],
            "original_ts_ns": attribution["original_ts_ns"],
            "original_host_pid": attribution["original_host_pid"],
            "original_host_tid": attribution["original_host_tid"],
            "original_exec_seq": attribution["original_exec_seq"],
            "owner_host_pid": attribution["owner_host_pid"],
            "owner_exec_seq": attribution["owner_exec_seq"],
            "fork_chain_ref": self._fork_chain_ref(attribution["fork_chain_records"]),
            "cpu_counter_support_ref": self._cpu_counter_support_ref(
                attribution["cpu_counter_support"]
            ),
        }


def resolve_inherited_owner_sample(
    row: Mapping[str, Any],
    sample_attribution: Mapping[str, Any],
) -> dict[str, Any]:
    """Expand one interned row back to the full attribution record."""

    chain = sample_attribution["fork_chains"][row["fork_chain_ref"]]
    return {
        **{
            key: value
            for key, value in row.items()
            if key not in {"fork_chain_ref", "cpu_counter_support_ref"}
        },
        "fork_chain_records": chain,
        "fork_ancestry": [chain[0]["child_id"], *(r["parent_pid"] for r in chain)],
        "fork_ts_ns": chain[0]["ts_ns"],
        "cpu_counter_support": sample_attribution["cpu_counter_supports"][
            row["cpu_counter_support_ref"]
        ],
    }


def analyze(
    run: RawRun,
    *,
    entry_pid: int | None = None,
    clauses_and_lineage: tuple[list[Clause], dict[int, int]] | None = None,
) -> tuple[list[ClauseMetrics], list[dict[str, Any]]]:
    """``clauses_and_lineage`` lets a caller that already built the clause list
    hand it over instead of paying for a second reconstruction of it."""

    clauses, fork_parent = (
        _clauses_and_lineage(run.events)
        if clauses_and_lineage is None
        else clauses_and_lineage
    )
    per_clause, gaps = _attribute(
        run.events,
        clauses,
        fork_parent,
        entry_pid=entry_pid,
    )
    fork_io_baselines = _fork_io_baselines(run.events, clauses, fork_parent)
    counter_events_by_pid: dict[int, list[dict[str, Any]]] = {}
    exit_events_by_pid: dict[int, list[dict[str, Any]]] = {}
    boundary_events_by_tid: dict[int, list[dict[str, Any]]] = {}
    # (host_pid, exec_seq, ts_ns) -> exec boundaries, so one clause's I/O
    # baseline is a dict hit instead of a scan of every event on its pid.
    exec_baseline_index: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for event in run.events:
        if event["type"] not in {"perf", "exec_boundary", "exit_boundary"}:
            continue
        counter_events_by_pid.setdefault(event["host_pid"], []).append(event)
        if event["type"] != "perf":
            boundary_events_by_tid.setdefault(event["host_tid"], []).append(event)
        if event["type"] == "exec_boundary":
            exec_baseline_index.setdefault(
                (event["host_pid"], event["exec_seq"], event["ts_ns"]), []
            ).append(event)
        if event["type"] == "exit_boundary":
            exit_events_by_pid.setdefault(event["host_pid"], []).append(event)
    # Sort by timestamp so the per-clause lookups below can bisect, and make the
    # in-window count independent of the order the caller supplied. The sort is
    # stable, so events sharing a timestamp keep their original relative order
    # and the endpoint "first match" tie-break is unchanged.
    #
    # That tie-break is order-dependent, and was before this change too: two
    # boundaries on one tid at the same timestamp resolve to whichever the
    # caller listed first, so `analyze` is not order-canonical. Preserving that
    # is the point here -- the scans this replaced behaved identically.
    for events_on_pid in counter_events_by_pid.values():
        events_on_pid.sort(key=lambda event: event["ts_ns"])
    for events_on_tid in boundary_events_by_tid.values():
        events_on_tid.sort(key=lambda event: event["ts_ns"])
    counter_ts_by_pid = {
        pid: [event["ts_ns"] for event in events]
        for pid, events in counter_events_by_pid.items()
    }
    boundary_ts_by_tid = {
        tid: [event["ts_ns"] for event in events]
        for tid, events in boundary_events_by_tid.items()
    }
    metrics: list[ClauseMetrics] = []
    for c in clauses:
        attributed_samples = per_clause[(c.host_pid, c.exec_seq)]
        samples = [
            sample for sample in attributed_samples if "metric_excluded" not in sample
        ]
        identity_only_samples = [
            {
                "type": sample["type"],
                "ts_ns": sample["ts_ns"],
                "host_pid": sample["host_pid"],
                "host_tid": sample["host_tid"],
                "exec_seq": sample["exec_seq"],
                **sample["metric_excluded"],
            }
            for sample in attributed_samples
            if "metric_excluded" in sample
        ]
        # Counter events on this pid are timestamp-sorted; the inclusive window
        # count is the gap between two insertion points rather than a rescan of
        # every event on the pid for every clause on it.
        window_ts = counter_ts_by_pid.get(c.host_pid, ())
        in_window = bisect_right(window_ts, c.t_end_ns) - bisect_left(
            window_ts, c.t_exec_ns
        )
        # Computed once per clause and reused by both the peak and the profile
        # emitted below; these previously cost three walks of the same samples.
        per_tid_points = _cpu_counter_points(samples)
        cpu_windows = _cpu_window_profile_from_points(per_tid_points)
        peak, cpu_reason, cpu_prov = _peak_cpu_cores(
            samples,
            c,
            run.quota_cores,
            per_tid_points=per_tid_points,
            profile=cpu_windows,
        )
        rss, rss_reason, rss_prov = _sampled_peak_rss(samples, c)
        io_totals, io_reason, io_prov = _task_io_totals(
            samples,
            c,
            fork_io_baselines[(c.host_pid, c.exec_seq)],
            exec_baseline_index=exec_baseline_index,
            boundary_events_by_tid=boundary_events_by_tid,
            boundary_ts_by_tid=boundary_ts_by_tid,
        )
        # Intern the inherited-sample evidence: the raw attribution dicts are
        # discarded with `samples` when this call returns, so only the compact
        # rows and their shared tables stay resident in the collector.
        attribution_tables = _AttributionTables()
        inherited_rows = [
            attribution_tables.row(sample["attribution"])
            for sample in samples
            if "attribution" in sample
        ]
        attribution_payload = {
            "inherited_owner_sample_count": len(inherited_rows),
            "inherited_owner_samples": inherited_rows,
            "fork_chains": attribution_tables.fork_chains,
            "cpu_counter_supports": attribution_tables.cpu_counter_supports,
        }
        exits = exit_events_by_pid.get(c.host_pid, ())
        has_exit = bool(exits)
        # Raw cumulative CPU (preserved separately, never used for the peak):
        # deterministic group sum across the terminal process's threads.
        if c.terminal:
            cpu_cum = sum(e["cpu_ns"] for e in exits)
            leader = next(
                (e for e in exits if e["host_tid"] == e["host_pid"]),
                exits[0] if exits else None,
            )
            raw_exit_code = leader["exit_code"] if leader else None
            signal = (raw_exit_code & 0x7F) if raw_exit_code is not None else 0
            exit_signal = signal or None
            normal_exit_status = (
                (raw_exit_code >> 8) & 0xFF
                if raw_exit_code is not None and signal == 0
                else None
            )
        else:
            cpu_cum = 0
            exit_signal = None
            normal_exit_status = None
        metrics.append(
            ClauseMetrics(
                host_pid=c.host_pid,
                exec_seq=c.exec_seq,
                bin=c.bin,
                argv=c.argv,
                requested_executable_path=c.requested_executable_path,
                requested_executable_path_truncated=(
                    c.requested_executable_path_truncated
                ),
                bprm_filename=c.bprm_filename,
                bprm_interp=c.bprm_interp,
                bprm_evidence_truncated=c.bprm_evidence_truncated,
                exact_argc=c.exact_argc,
                lineage_parent_pid=c.lineage_parent_pid,
                terminal=c.terminal,
                has_causal_end=c.has_causal_end,
                t_exec_ns=c.t_exec_ns,
                t_end_ns=c.t_end_ns,
                wall_ns=c.t_end_ns - c.t_exec_ns,
                cpu_ns_cumulative=cpu_cum,
                exit_signal=exit_signal,
                normal_exit_status=normal_exit_status,
                peak_cpu_cores=peak,
                peak_cpu_cores_reason=cpu_reason,
                sampled_peak_rss_mb=rss,
                sampled_peak_rss_reason=rss_reason,
                disk_read_bytes_total=(io_totals[0] if io_totals is not None else None),
                disk_write_bytes_total=(
                    io_totals[1] if io_totals is not None else None
                ),
                disk_cancelled_write_bytes_total=(
                    io_totals[2] if io_totals is not None else None
                ),
                disk_io_reason=io_reason,
                cpu_windows=cpu_windows,
                rss_bins=rss_bin_profile(samples),
                provenance={
                    "cadence_ns": SAMPLE_PERIOD_NS,
                    "window_ns": WINDOW_NS,
                    "align_bin_ns": ALIGN_BIN_NS,
                    "attributed_samples": len(samples),
                    "identity_only_sample_count": len(identity_only_samples),
                    "identity_only_samples": identity_only_samples,
                    "attribution_coverage": round(len(samples) / max(in_window, 1), 3),
                    "boundary_coverage": {
                        "has_exec": True,
                        "has_exit": has_exit,
                    },
                    "reserve_failures": run.loss_count,
                    "loss_counts": run.loss_counts,
                    "quota_cores": run.quota_cores,
                    "cpu": cpu_prov,
                    "rss": rss_prov,
                    "disk_io": io_prov,
                    "sample_attribution": attribution_payload,
                },
                argv_capture_flags=c.argv_capture_flags,
            )
        )
    return metrics, gaps


@dataclass(slots=True)
class _StreamingCpuSeries:
    previous: tuple[int, int] | None = None
    current: tuple[int, int] | None = None
    point_count: int = 0
    windows: dict[int, float] = field(default_factory=dict)
    finished: bool = False

    def add(self, ts_ns: int, cpu_ns: int) -> None:
        if self.finished:
            raise RuntimeError("CPU series received a point after finalization")
        point = (ts_ns, cpu_ns)
        if self.current is None:
            self.current = point
            self.point_count = 1
            return
        if ts_ns < self.current[0]:
            raise RuntimeError("CPU points are not timestamp ordered")
        if ts_ns == self.current[0]:
            self.current = point
            return
        if self.previous is not None:
            self._add_interval(self.previous, self.current)
        self.previous = self.current
        self.current = point
        self.point_count += 1

    def _add_interval(
        self,
        first: tuple[int, int],
        second: tuple[int, int],
    ) -> None:
        t0, c0 = first
        t1, c1 = second
        if t1 <= t0 or c1 < c0:
            return
        for window, cpu_ns in _apportion(t0, t1, c1 - c0):
            self.windows[window] = self.windows.get(window, 0.0) + cpu_ns

    def finish(self) -> None:
        if self.finished:
            return
        if self.previous is not None and self.current is not None:
            self._add_interval(self.previous, self.current)
        self.finished = True


class _StreamingClauseAccumulator:
    def __init__(self, clause: Clause) -> None:
        self.clause = clause
        self.sample_count = 0
        self.in_window_count = 0
        self.identity_only_samples: list[dict[str, Any]] = []
        self.cpu_sample_count = 0
        self.cpu_series: dict[int, _StreamingCpuSeries] = {}
        self.cpu_support_seen: set[tuple[int, int]] = set()
        self.rss_sample_count = 0
        self.perf_rss_samples = 0
        self.boundary_rss_samples = 0
        self.rss_first_ts: int | None = None
        self.rss_last_ts: int | None = None
        self.rss_max_gap = 0
        self.rss_bins: dict[tuple[int, int], int] = {}
        self.rss_profile_bins: dict[tuple[int, int], int] = {}
        self.mm_tids: dict[int, set[int]] = {}
        self.attributed_tids: set[int] = set()
        self.perf_sample_count = 0
        self.attribution_tables = _AttributionTables()
        self.inherited_rows: list[dict[str, Any]] = []

    def add_cpu_point(self, tid: int, ts_ns: int, cpu_ns: int) -> None:
        self.cpu_series.setdefault(tid, _StreamingCpuSeries()).add(
            ts_ns,
            cpu_ns,
        )

    def add(
        self,
        sample: Mapping[str, Any],
    ) -> list[tuple[int, int, int]]:
        excluded = sample.get("metric_excluded")
        if isinstance(excluded, Mapping):
            self.identity_only_samples.append(
                {
                    "type": sample["type"],
                    "ts_ns": sample["ts_ns"],
                    "host_pid": sample["host_pid"],
                    "host_tid": sample["host_tid"],
                    "exec_seq": sample["exec_seq"],
                    **excluded,
                }
            )
            return []

        self.sample_count += 1
        tid = int(sample["host_tid"])
        ts_ns = int(sample["ts_ns"])
        cpu_ns = int(sample["cpu_ns"])
        support = sample.get("attribution", {}).get("cpu_counter_support")
        support_points: list[tuple[int, int]] = []
        if isinstance(support, Mapping):
            for point in (support.get("baseline"), support.get("endpoint")):
                if not isinstance(point, Mapping):
                    continue
                support_ts = point.get("ts_ns")
                support_cpu = point.get("cpu_ns")
                if isinstance(support_ts, int) and isinstance(support_cpu, int):
                    support_points.append((support_ts, support_cpu))

        for support_ts, support_cpu in support_points:
            key = (tid, support_ts)
            if support_ts < ts_ns and key not in self.cpu_support_seen:
                self.cpu_support_seen.add(key)
                self.add_cpu_point(tid, support_ts, support_cpu)
        if cpu_ns > 0:
            self.cpu_sample_count += 1
            self.add_cpu_point(tid, ts_ns, cpu_ns)
        scheduled: list[tuple[int, int, int]] = []
        for support_ts, support_cpu in support_points:
            key = (tid, support_ts)
            if key in self.cpu_support_seen:
                continue
            self.cpu_support_seen.add(key)
            if support_ts <= ts_ns:
                self.add_cpu_point(tid, support_ts, support_cpu)
            else:
                scheduled.append((tid, support_ts, support_cpu))

        rss_pages = int(sample["rss_pages"])
        if rss_pages > 0:
            self.rss_sample_count += 1
            if sample["type"] == "perf":
                self.perf_rss_samples += 1
            else:
                self.boundary_rss_samples += 1
            if self.rss_last_ts is not None:
                self.rss_max_gap = max(
                    self.rss_max_gap,
                    ts_ns - self.rss_last_ts,
                )
            else:
                self.rss_first_ts = ts_ns
            self.rss_last_ts = ts_ns
            mm_ptr = int(sample["mm_ptr"])
            bin_key = (ts_ns // ALIGN_BIN_NS, mm_ptr)
            self.rss_bins[bin_key] = rss_pages
            self.rss_profile_bins[bin_key] = max(
                self.rss_profile_bins.get(bin_key, 0),
                rss_pages,
            )
            self.mm_tids.setdefault(mm_ptr, set()).add(tid)

        self.attributed_tids.add(tid)
        if sample["type"] == "perf":
            self.perf_sample_count += 1
        attribution = sample.get("attribution")
        if isinstance(attribution, Mapping):
            self.inherited_rows.append(self.attribution_tables.row(attribution))
        return scheduled

    def cpu_profile(self) -> tuple[tuple[int, int], ...]:
        combined: dict[int, float] = {}
        for series in self.cpu_series.values():
            series.finish()
            for window, cpu_ns in series.windows.items():
                combined[window] = combined.get(window, 0.0) + cpu_ns
        return tuple(
            (window, int(round(cpu_ns))) for window, cpu_ns in sorted(combined.items())
        )

    def cpu_result(
        self,
        quota: float,
        profile: tuple[tuple[int, int], ...],
    ) -> tuple[float | None, str, dict[str, Any]]:
        point_count = sum(series.point_count for series in self.cpu_series.values())
        span = self.clause.t_end_ns - self.clause.t_exec_ns
        provenance = {
            "cpu_sample_count": self.cpu_sample_count,
            "cpu_counter_point_count": point_count,
            "cpu_windows": len(profile),
            "span_s": round(span / 1e9, 3),
        }
        if span < _MIN_ELIGIBLE_SPAN_NS:
            return (
                None,
                "clause_shorter_than_1s_ineligible_for_peak",
                provenance,
            )
        if point_count < 2 or not profile:
            return None, "insufficient_cpu_samples", provenance
        peak: float | None = None
        for window, cpu_ns in profile:
            window_start = window * WINDOW_NS
            window_span = min(
                self.clause.t_end_ns,
                window_start + WINDOW_NS,
            ) - max(self.clause.t_exec_ns, window_start)
            if window_span < _MIN_WINDOW_SPAN_NS:
                continue
            rate = min(cpu_ns / window_span, quota)
            peak = rate if peak is None else max(peak, rate)
        if peak is None:
            return None, "no_eligible_merged_window", provenance
        return peak, "ok", provenance

    def rss_result(
        self,
    ) -> tuple[float | None, str, dict[str, Any]]:
        span = max(self.clause.t_end_ns - self.clause.t_exec_ns, 1)
        max_gap = self.rss_max_gap
        if self.rss_first_ts is not None and self.rss_last_ts is not None:
            max_gap = max(
                max_gap,
                self.rss_first_ts - self.clause.t_exec_ns,
                self.clause.t_end_ns - self.rss_last_ts,
            )
        else:
            max_gap = span
        provenance = {
            "rss_sample_count": self.rss_sample_count,
            "perf_rss_samples": self.perf_rss_samples,
            "boundary_rss_samples": self.boundary_rss_samples,
            "distinct_mm": len(self.mm_tids),
            "shared_mm_tid_counts": {
                hex(mm_ptr): len(tids)
                for mm_ptr, tids in self.mm_tids.items()
                if len(tids) > 1
            },
            "max_intersample_gap_frac": round(max_gap / span, 3),
        }
        if self.rss_sample_count < 2:
            return None, "insufficient_rss_samples", provenance
        totals: dict[int, int] = {}
        for (bin_index, _mm_ptr), pages in self.rss_bins.items():
            totals[bin_index] = totals.get(bin_index, 0) + pages
        return max(totals.values()) * PAGE / 1e6, "ok", provenance

    def rss_profile(self) -> tuple[tuple[int, int, float], ...]:
        rows = tuple(
            (bin_index, mm_ptr, pages * PAGE / 1e6)
            for (bin_index, mm_ptr), pages in self.rss_profile_bins.items()
        )
        if self.rss_sample_count >= 2 and len(rows) == 1:
            return (rows[0], rows[0])
        return rows


def _analyze_streaming(
    run: RawRun,
    *,
    entry_pid: int | None,
    clauses_and_lineage: tuple[list[Clause], dict[int, int]],
) -> tuple[list[ClauseMetrics], list[dict[str, Any]]]:
    """Analyze a stable timestamp-ordered, re-iterable event source."""

    clauses, fork_parent = clauses_and_lineage
    accumulators = {
        (clause.host_pid, clause.exec_seq): _StreamingClauseAccumulator(clause)
        for clause in clauses
    }
    clauses_by_pid: dict[int, list[Clause]] = {}
    for clause in clauses:
        clauses_by_pid.setdefault(clause.host_pid, []).append(clause)
    clause_starts_by_pid: dict[int, list[int]] = {}
    for pid, pid_clauses in clauses_by_pid.items():
        pid_clauses.sort(key=lambda clause: clause.t_exec_ns)
        clause_starts_by_pid[pid] = [clause.t_exec_ns for clause in pid_clauses]

    exit_events_by_pid: dict[int, list[dict[str, Any]]] = {}
    boundary_events_by_tid: dict[int, list[dict[str, Any]]] = {}
    exec_baseline_index: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for event in run.events:
        if event["type"] not in {"perf", "exec_boundary", "exit_boundary"}:
            continue
        pid = int(event["host_pid"])
        ts_ns = int(event["ts_ns"])
        pid_clauses = clauses_by_pid.get(pid, ())
        starts = clause_starts_by_pid.get(pid, ())
        index = bisect_right(starts, ts_ns) - 1
        for candidate_index in (index - 1, index):
            if not (0 <= candidate_index < len(pid_clauses)):
                continue
            clause = pid_clauses[candidate_index]
            if clause.t_exec_ns <= ts_ns <= clause.t_end_ns:
                accumulators[(clause.host_pid, clause.exec_seq)].in_window_count += 1
        if event["type"] != "perf":
            row = dict(event)
            boundary_events_by_tid.setdefault(
                int(event["host_tid"]),
                [],
            ).append(row)
            if event["type"] == "exec_boundary":
                exec_baseline_index.setdefault(
                    (pid, int(event["exec_seq"]), ts_ns),
                    [],
                ).append(row)
            if event["type"] == "exit_boundary":
                exit_events_by_pid.setdefault(pid, []).append(row)
    boundary_ts_by_tid = {
        tid: [event["ts_ns"] for event in events]
        for tid, events in boundary_events_by_tid.items()
    }
    fork_io_baselines = _fork_io_baselines(
        _events_of_types(run.events, _FORK_EVENT_TYPES), clauses, fork_parent
    )

    scheduled_cpu: list[tuple[int, int, tuple[int, int], int, int]] = []
    schedule_order = 0

    def flush_cpu(until_ns: int) -> None:
        while scheduled_cpu and scheduled_cpu[0][0] <= until_ns:
            _ts_ns, _order, key, tid, cpu_ns = heapq.heappop(scheduled_cpu)
            accumulators[key].add_cpu_point(tid, _ts_ns, cpu_ns)

    def on_attributed(
        clause: Clause,
        sample: Mapping[str, Any],
    ) -> None:
        nonlocal schedule_order
        flush_cpu(int(sample["ts_ns"]))
        key = (clause.host_pid, clause.exec_seq)
        for tid, ts_ns, cpu_ns in accumulators[key].add(sample):
            heapq.heappush(
                scheduled_cpu,
                (ts_ns, schedule_order, key, tid, cpu_ns),
            )
            schedule_order += 1

    _unused, gaps = _attribute(
        run.events,
        clauses,
        fork_parent,
        entry_pid=entry_pid,
        on_attributed=on_attributed,
    )
    flush_cpu(SENTINEL)

    metrics: list[ClauseMetrics] = []
    for clause in clauses:
        key = (clause.host_pid, clause.exec_seq)
        accumulator = accumulators[key]
        cpu_windows = accumulator.cpu_profile()
        peak, cpu_reason, cpu_provenance = accumulator.cpu_result(
            run.quota_cores,
            cpu_windows,
        )
        rss, rss_reason, rss_provenance = accumulator.rss_result()
        io_totals, io_reason, io_provenance = _task_io_totals_from_state(
            attributed_tids=accumulator.attributed_tids,
            perf_sample_count=accumulator.perf_sample_count,
            clause=clause,
            fork_baselines=fork_io_baselines[key],
            exec_baseline_index=exec_baseline_index,
            boundary_events_by_tid=boundary_events_by_tid,
            boundary_ts_by_tid=boundary_ts_by_tid,
        )
        attribution_payload = {
            "inherited_owner_sample_count": len(accumulator.inherited_rows),
            "inherited_owner_samples": accumulator.inherited_rows,
            "fork_chains": accumulator.attribution_tables.fork_chains,
            "cpu_counter_supports": (
                accumulator.attribution_tables.cpu_counter_supports
            ),
        }
        exits = exit_events_by_pid.get(clause.host_pid, ())
        has_exit = bool(exits)
        if clause.terminal:
            cpu_cumulative = sum(event["cpu_ns"] for event in exits)
            leader = next(
                (event for event in exits if event["host_tid"] == event["host_pid"]),
                exits[0] if exits else None,
            )
            raw_exit_code = leader["exit_code"] if leader else None
            signal = (raw_exit_code & 0x7F) if raw_exit_code is not None else 0
            exit_signal = signal or None
            normal_exit_status = (
                (raw_exit_code >> 8) & 0xFF
                if raw_exit_code is not None and signal == 0
                else None
            )
        else:
            cpu_cumulative = 0
            exit_signal = None
            normal_exit_status = None
        metrics.append(
            ClauseMetrics(
                host_pid=clause.host_pid,
                exec_seq=clause.exec_seq,
                bin=clause.bin,
                argv=clause.argv,
                requested_executable_path=clause.requested_executable_path,
                requested_executable_path_truncated=(
                    clause.requested_executable_path_truncated
                ),
                bprm_filename=clause.bprm_filename,
                bprm_interp=clause.bprm_interp,
                bprm_evidence_truncated=clause.bprm_evidence_truncated,
                exact_argc=clause.exact_argc,
                lineage_parent_pid=clause.lineage_parent_pid,
                terminal=clause.terminal,
                has_causal_end=clause.has_causal_end,
                t_exec_ns=clause.t_exec_ns,
                t_end_ns=clause.t_end_ns,
                wall_ns=clause.t_end_ns - clause.t_exec_ns,
                cpu_ns_cumulative=cpu_cumulative,
                exit_signal=exit_signal,
                normal_exit_status=normal_exit_status,
                peak_cpu_cores=peak,
                peak_cpu_cores_reason=cpu_reason,
                sampled_peak_rss_mb=rss,
                sampled_peak_rss_reason=rss_reason,
                disk_read_bytes_total=(io_totals[0] if io_totals is not None else None),
                disk_write_bytes_total=(
                    io_totals[1] if io_totals is not None else None
                ),
                disk_cancelled_write_bytes_total=(
                    io_totals[2] if io_totals is not None else None
                ),
                disk_io_reason=io_reason,
                cpu_windows=cpu_windows,
                rss_bins=accumulator.rss_profile(),
                provenance={
                    "cadence_ns": SAMPLE_PERIOD_NS,
                    "window_ns": WINDOW_NS,
                    "align_bin_ns": ALIGN_BIN_NS,
                    "attributed_samples": accumulator.sample_count,
                    "identity_only_sample_count": len(
                        accumulator.identity_only_samples
                    ),
                    "identity_only_samples": (accumulator.identity_only_samples),
                    "attribution_coverage": round(
                        accumulator.sample_count / max(accumulator.in_window_count, 1),
                        3,
                    ),
                    "boundary_coverage": {
                        "has_exec": True,
                        "has_exit": has_exit,
                    },
                    "reserve_failures": run.loss_count,
                    "loss_counts": run.loss_counts,
                    "quota_cores": run.quota_cores,
                    "cpu": cpu_provenance,
                    "rss": rss_provenance,
                    "disk_io": io_provenance,
                    "sample_attribution": attribution_payload,
                },
                argv_capture_flags=clause.argv_capture_flags,
            )
        )
    return metrics, gaps


class ClauseTelemetryIntegrityError(RuntimeError):
    """Telemetry cannot be used without hiding a coverage or lifecycle gap."""

    def __init__(
        self,
        message: str,
        *,
        artifact_payload: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.artifact_payload = dict(artifact_payload or {})


def _command_tree_provenance(
    metrics: Sequence[Clause | ClauseMetrics],
    fork_parent: Mapping[int, int],
    *,
    fork_records: Mapping[int, Sequence[Mapping[str, Any]]] | None = None,
) -> tuple[int, set[int], dict[str, Any]]:
    """Identify transitive exec roots and their one observed outside parent."""

    first_exec_by_pid: dict[int, int] = {}
    for metric in metrics:
        first_exec_by_pid[metric.host_pid] = min(
            first_exec_by_pid.get(metric.host_pid, metric.t_exec_ns),
            metric.t_exec_ns,
        )
    exec_pids = set(first_exec_by_pid)
    ancestry: list[dict[str, Any]] = []
    roots: list[int] = []
    entry_by_root: dict[int, int] = {}
    failure: str | None = None
    fork_ambiguities: list[dict[str, Any]] = []
    for pid in sorted(exec_pids):
        chain: list[int] = []
        nearest_exec_ancestor: int | None = None
        current = pid
        seen = {pid}
        ancestor_ts_bound = first_exec_by_pid[pid]
        while current in fork_parent or (
            fork_records is not None and current in fork_records
        ):
            eligible_records = (
                [
                    record
                    for record in fork_records[current]
                    if int(record["ts_ns"]) <= ancestor_ts_bound
                ]
                if fork_records is not None and current in fork_records
                else [
                    {
                        "host_pid": int(fork_parent[current]),
                        "ts_ns": ancestor_ts_bound,
                    }
                ]
            )
            if len(eligible_records) != 1:
                failure = "ambiguous_fork_ancestry"
                if not eligible_records:
                    failure = "temporally_invalid_fork_ancestry"
                evidence_records = (
                    eligible_records
                    if eligible_records
                    else list(fork_records.get(current, ()))
                )
                fork_ambiguities.append(
                    {
                        "child_pid": current,
                        "parent_candidates": sorted(
                            {int(record["host_pid"]) for record in evidence_records}
                        ),
                        "candidate_records": sorted(
                            (
                                {
                                    "parent_pid": int(record["host_pid"]),
                                    "ts_ns": int(record["ts_ns"]),
                                }
                                for record in evidence_records
                            ),
                            key=lambda record: (
                                record["ts_ns"],
                                record["parent_pid"],
                            ),
                        ),
                    }
                )
                break
            fork_record = eligible_records[0]
            parent = int(fork_record["host_pid"])
            if parent <= 0 or parent in seen:
                failure = "invalid_or_cyclic_fork_ancestry"
                break
            chain.append(parent)
            seen.add(parent)
            if nearest_exec_ancestor is None and parent in exec_pids:
                nearest_exec_ancestor = parent
            current = parent
            ancestor_ts_bound = int(fork_record["ts_ns"])
        is_root = nearest_exec_ancestor is None
        if is_root:
            roots.append(pid)
            if chain:
                entry_by_root[pid] = chain[-1]
            else:
                failure = failure or "missing_root_ancestry"
        ancestry.append(
            {
                "exec_pid": pid,
                "ancestor_chain": chain,
                "nearest_exec_ancestor_pid": nearest_exec_ancestor,
                "is_root": is_root,
            }
        )

    entries = sorted(set(entry_by_root.values()))
    if not exec_pids:
        failure = "no_exec_images"
    elif len(entries) != 1 or len(entry_by_root) != len(roots):
        failure = failure or "disconnected_command_trees"
    provenance = {
        "status": "failed" if failure else "ok",
        "reason": failure,
        "entry_pid": entries[0] if not failure else None,
        "root_pids": roots,
        "exec_ancestry": ancestry,
    }
    if fork_ambiguities:
        provenance["fork_ambiguities"] = fork_ambiguities
    if failure:
        raise ClauseTelemetryIntegrityError(
            "cannot identify one connected command tree: "
            f"reason={failure} roots={roots} entries={entries}",
            artifact_payload={"provenance": {"command_tree": provenance}},
        )
    return entries[0], set(roots), provenance


def validate_clause_telemetry_runtime(
    *,
    container_executable: str | None,
    concurrency: int,
    workers: int,
) -> None:
    """Fail before container preparation when clause telemetry is unsupported."""

    if sys.platform != "linux":
        raise ValueError("clause telemetry requires Linux")
    if os.geteuid() != 0:
        raise ValueError("clause telemetry requires root")
    if container_executable != "docker":
        raise ValueError("clause telemetry requires --container docker")
    if workers != 1:
        raise ValueError("clause telemetry requires --workers 1")
    if not Path("/sys/fs/cgroup/cgroup.controllers").is_file():
        raise ValueError("clause telemetry requires cgroup v2")
    try:
        import bcc  # noqa: F401
    except ImportError as exc:
        raise ValueError(
            "clause telemetry requires BCC Python bindings in the active interpreter"
        ) from exc


def _container_cgroup(
    container_id: str,
    container_executable: str,
) -> tuple[Path, int]:
    result = subprocess.run(
        [
            container_executable,
            "inspect",
            container_id,
            "--format",
            "{{.State.Pid}}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip().isdigit():
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"cannot resolve container host pid: {detail}")
    init_pid = int(result.stdout.strip())
    cgroup_lines = Path(f"/proc/{init_pid}/cgroup").read_text().splitlines()
    unified = next(
        (line.split(":", 2)[2] for line in cgroup_lines if line.startswith("0::")),
        None,
    )
    if unified is None:
        raise RuntimeError(f"container {container_id[:12]} has no cgroup-v2 path")
    cgroup = Path("/sys/fs/cgroup") / unified.lstrip("/")
    if not cgroup.is_dir():
        raise RuntimeError(f"container cgroup does not exist: {cgroup}")
    return cgroup, init_pid


_EVENT_FIELDS = (
    "type",
    "ts_ns",
    "cgroup_id",
    "exec_seq",
    "cpu_ns",
    "rss_pages",
    "mm_ptr",
    "hiwater_pages",
    "io_read_bytes",
    "io_write_bytes",
    "io_cancelled_write_bytes",
    "host_pid",
    "host_tid",
    "parent_host_pid",
    "child_host_pid",
    "child_host_tid",
    "arg_index",
    "arg_chunk_index",
    "arg_flags",
    "exit_code",
    "errno",
)
_EVENT_KEYS = frozenset((*_EVENT_FIELDS, "arg", "arg_raw"))
# The stored fields, without the two derived-from-payload keys, so the common
# lookup answers on one frozenset hit and skips the payload branches.
_EVENT_SLOT_KEYS = frozenset(_EVENT_FIELDS)


class EventRow(Mapping):
    """One ring-buffer event, stored compactly.

    The initial dict-to-slots change measured 636 bytes against 388 with
    realistic per-event values -- the difference was the dict table, since the
    integer values cost the same either way. A build-heavy tool call spawns
    tens of thousands of processes and every exec emits up to MAX_ARGS *
    MAX_ARG_CHUNKS argv events, so a single call can deliver millions of these.
    On a two-container collection that was gigabytes of host memory, and host
    OOM has already stopped a run.

    Arg payload is stored once as bytes. The decoded ``arg`` and hexadecimal
    ``arg_raw`` mapping values are produced only if a consumer requests them;
    normal collection writes the bytes directly into the event spool without
    constructing either duplicate string.

    It is a Mapping so every consumer keeps working through ``[]``, ``.get()``,
    ``in`` and ``**`` splat, and so do the tests that build events as plain
    dicts. ``arg``/``arg_raw`` stay genuinely absent when the kernel did not
    supply them, matching the dict this replaces -- ``event.get("arg", "")``
    must still yield ``""`` and not ``None``.

    The absent-field sentinel preserves its identity across pickling because
    runtime collection tests return rows from worker processes.
    """

    __slots__ = (*_EVENT_FIELDS, "_arg_payload")

    def __init__(
        self,
        *values: Any,
        arg_payload: bytes | object = _UNSET,
    ) -> None:
        (
            self.type,
            self.ts_ns,
            self.cgroup_id,
            self.exec_seq,
            self.cpu_ns,
            self.rss_pages,
            self.mm_ptr,
            self.hiwater_pages,
            self.io_read_bytes,
            self.io_write_bytes,
            self.io_cancelled_write_bytes,
            self.host_pid,
            self.host_tid,
            self.parent_host_pid,
            self.child_host_pid,
            self.child_host_tid,
            self.arg_index,
            self.arg_chunk_index,
            self.arg_flags,
            self.exit_code,
            self.errno,
        ) = values
        self._arg_payload = arg_payload

    def __getitem__(self, key: str) -> Any:
        # Membership first: `getattr` alone would answer `row["get"]` with the
        # bound method and raise TypeError rather than KeyError for a non-string
        # key, neither of which a dict does.
        #
        # A frozenset, not `self.__slots__`: tuple membership compares with `==`
        # without hashing, which is a linear scan on the hottest accessor in the
        # module and answers an unhashable key with KeyError where a dict raises
        # TypeError. Hashing the key restores both.
        if key in _EVENT_SLOT_KEYS:
            return getattr(self, key)
        return self._payload_item(key)

    def _payload_item(self, key: str) -> Any:
        """The two keys derived from the raw payload, plus every miss."""

        if key == "arg":
            if self._arg_payload is _UNSET:
                raise KeyError(key)
            return self._arg_payload.decode("utf-8", "replace")
        if key == "arg_raw":
            if self._arg_payload is _UNSET or self.type != "exec_arg":
                raise KeyError(key)
            return self._arg_payload.hex()
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        # The inherited Mapping.get answers a miss by raising KeyError out of
        # __getitem__ and catching it, which measured 242 ns against 30 ns for
        # a dict. Every attributed sample asks for three keys this row never
        # carries ("attribution", "metric_excluded"), so the miss is the hot
        # case and must not raise. Membership still hashes the key, so an
        # unhashable one raises TypeError exactly as dict.get does.
        if key in _EVENT_SLOT_KEYS:
            return getattr(self, key)
        if key not in _EVENT_KEYS:
            return default
        try:
            return self._payload_item(key)
        except KeyError:
            return default

    def __iter__(self) -> Any:
        yield from _EVENT_FIELDS
        if self._arg_payload is not _UNSET:
            yield "arg"
            if self.type == "exec_arg":
                yield "arg_raw"

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __repr__(self) -> str:
        return f"EventRow({dict(self)!r})"


def _event_row(table: Any, data: int) -> EventRow:
    event = table.event(data)
    event_type = TYPE_NAMES[int(event.type)]
    arg_payload: bytes | object = _UNSET
    if event.type in {1, 7, 8, 9}:
        arg_payload = bytes(event.arg).split(b"\0", 1)[0]
    return EventRow(
        event_type,
        int(event.timestamp_ns),
        int(event.cgroup_id),
        int(event.exec_seq),
        int(event.cpu_ns),
        int(event.rss_pages),
        int(event.mm_ptr),
        int(event.hiwater_pages),
        int(event.io_read_bytes),
        int(event.io_write_bytes),
        int(event.io_cancelled_write_bytes),
        int(event.host_pid),
        int(event.host_tid),
        int(event.parent_host_pid),
        int(event.child_host_pid),
        int(event.child_host_tid),
        int(event.arg_index),
        int(event.arg_chunk_index),
        int(event.arg_flags),
        int(event.exit_code),
        int(event.exit_code) if event_type == "failed_exec_attempt" else 0,
        arg_payload=arg_payload,
    )


def _event_arg_payload(event: Mapping[str, Any]) -> bytes:
    payload = getattr(event, "_arg_payload", _UNSET)
    if payload is not _UNSET:
        return payload
    raw = event.get("arg_raw")
    return (
        bytes.fromhex(raw)
        if isinstance(raw, str)
        else str(event.get("arg", "")).encode()
    )


_EVENT_SPOOL_SEGMENT_BYTES = 4 * 1024 * 1024
_EVENT_SPOOL_MERGE_FAN_IN = 32
# arrival, type, ten u64 counters/identities, nine u32 fields,
# payload-size-plus-one (zero means absent), fixed payload storage.
_EVENT_RECORD = struct.Struct(f"<QB10Q9IH{ARG_BYTES}s")
# The sort key and the window test read four leading fields. Unpacking the
# whole record for them also copies the ARG_BYTES payload, which measured
# 239 ns against 65 ns for this prefix -- and both run once per record per
# merge pass. The type code is the single byte at _EVENT_TYPE_OFFSET, so a
# type filter can reject a record without unpacking anything at all.
_EVENT_RECORD_PREFIX = struct.Struct("<QBQQ")  # arrival, type, ts_ns, cgroup_id
_EVENT_TYPE_OFFSET = 8


def _pack_event_record(event: Mapping[str, Any], arrival: int) -> bytes:
    if isinstance(event, EventRow):
        payload = None if event._arg_payload is _UNSET else event._arg_payload
        payload_marker = 0 if payload is None else len(payload) + 1
        if payload is not None and len(payload) > ARG_BYTES:
            raise ValueError("event payload exceeds spool record")
        return _EVENT_RECORD.pack(
            arrival,
            TYPE_CODES[event.type],
            event.ts_ns,
            event.cgroup_id,
            event.exec_seq,
            event.cpu_ns,
            event.rss_pages,
            event.mm_ptr,
            event.hiwater_pages,
            event.io_read_bytes,
            event.io_write_bytes,
            event.io_cancelled_write_bytes,
            event.host_pid,
            event.host_tid,
            event.parent_host_pid,
            event.child_host_pid,
            event.child_host_tid,
            event.arg_index,
            event.arg_chunk_index,
            event.arg_flags,
            event.exit_code,
            payload_marker,
            b"" if payload is None else payload,
        )
    payload = getattr(event, "_arg_payload", _UNSET)
    if payload is _UNSET:
        payload = (
            _event_arg_payload(event) if "arg" in event or "arg_raw" in event else None
        )
    payload_marker = 0 if payload is None else len(payload) + 1
    if payload is not None and len(payload) > ARG_BYTES:
        raise ValueError("event payload exceeds spool record")
    return _EVENT_RECORD.pack(
        arrival,
        TYPE_CODES[str(event["type"])],
        *(int(event[name]) for name in _EVENT_FIELDS[1:11]),
        *(int(event[name]) for name in _EVENT_FIELDS[11:20]),
        payload_marker,
        b"" if payload is None else payload,
    )


def _unpack_event_record(record: bytes) -> EventRow:
    values = _EVENT_RECORD.unpack(record)
    event_type = TYPE_NAMES[values[1]]
    payload_marker = values[21]
    payload: bytes | object = (
        _UNSET if payload_marker == 0 else values[22][: payload_marker - 1]
    )
    exit_code = values[20]
    return EventRow(
        event_type,
        *values[2:21],
        exit_code if event_type == "failed_exec_attempt" else 0,
        arg_payload=payload,
    )


def _event_record_key(record: bytes) -> tuple[int, int]:
    arrival, _type_code, ts_ns, _cgroup_id = _EVENT_RECORD_PREFIX.unpack_from(record)
    return ts_ns, arrival


def _event_record_in_window(
    record: bytes,
    *,
    started_ns: int,
    ended_ns: int,
    cgroup_id: int,
) -> bool:
    _arrival, _type_code, ts_ns, record_cgroup = _EVENT_RECORD_PREFIX.unpack_from(
        record
    )
    return started_ns <= ts_ns <= ended_ns and record_cgroup == cgroup_id


@dataclass(slots=True)
class _EventSpoolSegment:
    file: BinaryIO
    record_count: int = 0
    byte_count: int = 0
    min_ts_ns: int = 2**64 - 1
    max_ts_ns: int = 0


def _temporary_binary_file(directory: Path) -> BinaryIO:
    return tempfile.TemporaryFile(
        mode="w+b",
        buffering=PAGE,
        dir=directory,
    )


class _EventTypeView:
    """Re-iterable view of one event source restricted to some event types.

    A single tool call walks its event source several times, and most of those
    walks want a small minority of it: the fork map wants ``fork``, argv
    reassembly wants the exec events. Testing the type byte in the packed
    record costs an index; decoding one costs a struct unpack plus twenty-one
    slot stores. So the filter runs before the decode, not after it.

    ``source_max_ts_ns`` is the maximum over the WHOLE source, not over the
    retained types, because a consumer that bounds an unterminated clause by
    "the last thing that happened" must not see a shortened stream.
    """

    def __init__(self, source: "_SortedEventSource", types: frozenset[str]) -> None:
        self._source = source
        self._codes = frozenset(TYPE_CODES[name] for name in types)

    @property
    def source_max_ts_ns(self) -> int:
        # Read on demand: resolving it seeks the shared run, so doing it at
        # construction time would derail an iteration already in flight.
        return self._source.max_ts_ns

    def __iter__(self) -> Iterator[EventRow]:
        return self._source.iter_records(self._codes)


class _SortedEventSource:
    """Re-iterable stable timestamp order over one private temporary run."""

    def __init__(self, run: BinaryIO, record_count: int) -> None:
        self._run = run
        self.record_count = record_count

    @property
    def max_ts_ns(self) -> int:
        """The last record's timestamp; the run is in timestamp order."""

        if not self.record_count:
            return 0
        self._run.seek((self.record_count - 1) * _EVENT_RECORD.size)
        record = self._run.read(_EVENT_RECORD.size)
        if len(record) != _EVENT_RECORD.size:
            raise OSError("event spool sorted run is truncated")
        return _EVENT_RECORD_PREFIX.unpack_from(record)[2]

    def of_types(self, types: frozenset[str]) -> _EventTypeView:
        return _EventTypeView(self, types)

    def iter_records(self, codes: frozenset[int] | None = None) -> Iterator[EventRow]:
        """Every record, or only those whose packed type byte is in ``codes``."""

        self._run.seek(0)
        for _ in range(self.record_count):
            record = self._run.read(_EVENT_RECORD.size)
            if len(record) != _EVENT_RECORD.size:
                raise OSError("event spool sorted run is truncated")
            if codes is None or record[_EVENT_TYPE_OFFSET] in codes:
                yield _unpack_event_record(record)
        if self._run.read(1):
            raise OSError("event spool sorted run has trailing bytes")

    def __iter__(self) -> Iterator[EventRow]:
        return self.iter_records()

    def __len__(self) -> int:
        return self.record_count

    def close(self) -> None:
        self._run.close()

    def __enter__(self) -> "_SortedEventSource":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def _merge_event_runs(
    runs: Sequence[tuple[BinaryIO, int]],
    *,
    directory: Path,
) -> tuple[BinaryIO, int]:
    output = _temporary_binary_file(directory)
    readers: list[tuple[BinaryIO, int]] = []
    heap: list[tuple[int, int, int, bytes]] = []
    total = 0
    try:
        for index, (run, count) in enumerate(runs):
            run.seek(0)
            readers.append((run, count))
            if not count:
                continue
            record = run.read(_EVENT_RECORD.size)
            if len(record) != _EVENT_RECORD.size:
                raise OSError("event spool merge input is truncated")
            ts_ns, arrival = _event_record_key(record)
            heapq.heappush(heap, (ts_ns, arrival, index, record))
        consumed = [0] * len(readers)
        while heap:
            _ts_ns, _arrival, index, record = heapq.heappop(heap)
            output.write(record)
            consumed[index] += 1
            total += 1
            run, count = readers[index]
            if consumed[index] < count:
                next_record = run.read(_EVENT_RECORD.size)
                if len(next_record) != _EVENT_RECORD.size:
                    raise OSError("event spool merge input is truncated")
                ts_ns, arrival = _event_record_key(next_record)
                heapq.heappush(
                    heap,
                    (ts_ns, arrival, index, next_record),
                )
        output.flush()
        return output, total
    except BaseException:
        output.close()
        raise
    finally:
        for run, _count in runs:
            run.close()


def _sorted_event_source(
    segments: Sequence[_EventSpoolSegment],
    *,
    started_ns: int,
    ended_ns: int,
    cgroup_id: int,
    directory: Path,
) -> _SortedEventSource:
    runs: list[tuple[BinaryIO, int]] = []
    try:
        for segment in segments:
            segment.file.flush()
            segment.file.seek(0)
            records: list[bytes] = []
            for _ in range(segment.record_count):
                record = segment.file.read(_EVENT_RECORD.size)
                if len(record) != _EVENT_RECORD.size:
                    raise OSError("event spool segment is truncated")
                if _event_record_in_window(
                    record,
                    started_ns=started_ns,
                    ended_ns=ended_ns,
                    cgroup_id=cgroup_id,
                ):
                    records.append(record)
            if segment.file.read(1):
                raise OSError("event spool segment has trailing bytes")
            if not records:
                continue
            records.sort(key=_event_record_key)
            run = _temporary_binary_file(directory)
            try:
                for record in records:
                    run.write(record)
                run.flush()
            except BaseException:
                run.close()
                raise
            runs.append((run, len(records)))

        if not runs:
            return _SortedEventSource(_temporary_binary_file(directory), 0)
        while len(runs) > 1:
            merged: list[tuple[BinaryIO, int]] = []
            try:
                for start in range(0, len(runs), _EVENT_SPOOL_MERGE_FAN_IN):
                    merged.append(
                        _merge_event_runs(
                            runs[start : start + _EVENT_SPOOL_MERGE_FAN_IN],
                            directory=directory,
                        )
                    )
            except BaseException:
                for run, _count in merged:
                    run.close()
                raise
            runs = merged
        run, count = runs.pop()
        return _SortedEventSource(run, count)
    except BaseException:
        for run, _count in runs:
            run.close()
        raise


class _EventSpool:
    """Disk-backed raw event segments; memory holds only one write page."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self._segments: list[_EventSpoolSegment] = []
        self._current = _EventSpoolSegment(_temporary_binary_file(directory))
        self._arrival = 0

    def append(self, event: Mapping[str, Any]) -> None:
        record = _pack_event_record(event, self._arrival)
        self._arrival += 1
        if (
            self._current.record_count
            and self._current.byte_count + len(record) > _EVENT_SPOOL_SEGMENT_BYTES
        ):
            self._seal_current()
        written = self._current.file.write(record)
        if written != len(record):
            raise OSError("short event spool write")
        ts_ns = int(event["ts_ns"])
        self._current.record_count += 1
        self._current.byte_count += written
        self._current.min_ts_ns = min(self._current.min_ts_ns, ts_ns)
        self._current.max_ts_ns = max(self._current.max_ts_ns, ts_ns)

    def _seal_current(self) -> None:
        if self._current.record_count:
            self._current.file.flush()
            self._segments.append(self._current)
        else:
            self._current.file.close()
        self._current = _EventSpoolSegment(_temporary_binary_file(self.directory))

    def drop_before(self, started_ns: int) -> None:
        self._seal_current()
        retained: list[_EventSpoolSegment] = []
        for segment in self._segments:
            if segment.max_ts_ns < started_ns:
                segment.file.close()
            else:
                retained.append(segment)
        self._segments = retained

    def snapshot(self) -> tuple[_EventSpoolSegment, ...]:
        self._seal_current()
        return tuple(self._segments)

    def release_through(
        self,
        ended_ns: int,
        snapshot: Sequence[_EventSpoolSegment],
    ) -> None:
        consumed = {
            id(segment) for segment in snapshot if segment.max_ts_ns <= ended_ns
        }
        retained: list[_EventSpoolSegment] = []
        for segment in self._segments:
            if id(segment) in consumed:
                segment.file.close()
            else:
                retained.append(segment)
        self._segments = retained

    def close(self) -> None:
        for segment in self._segments:
            segment.file.close()
        self._segments.clear()
        self._current.file.close()


def _counter(bpf: Any, name: str) -> int:
    return int(bpf[name][ctypes.c_int(0)].value)


def _loss_counts(bpf: Any) -> dict[str, int]:
    return {name: _counter(bpf, name) for name in LOSS_COUNTER_NAMES}


def _argv_read_failure_sites(bpf: Any) -> dict[str, int]:
    table = bpf["argv_read_failure_sites"]
    return {
        name: count
        for index, name in enumerate(ARGV_READ_FAILURE_SITE_NAMES)
        if (count := int(table[ctypes.c_int(index)].value))
    }


def _loss_delta(bpf: Any, token: ToolCallToken) -> dict[str, int]:
    before = {
        "ringbuf_reserve_failures": token.ringbuf_reserve_failures,
        "argv_read_failures": token.argv_read_failures,
        "argv_boundary_read_failures": token.argv_boundary_read_failures,
    }
    return {name: _counter(bpf, name) - before[name] for name in LOSS_COUNTER_NAMES}


def _exec_image_record(metric: ClauseMetrics) -> Any:
    from tool_resource.clause_bridge import ExecImageRecord

    return ExecImageRecord(
        host_pid=metric.host_pid,
        exec_seq=metric.exec_seq,
        t_exec_ns=metric.t_exec_ns,
        t_end_ns=metric.t_end_ns,
        bin=metric.bin,
        argv=metric.argv,
        terminal=metric.terminal,
        cpu_windows=metric.cpu_windows,
        rss_bins=metric.rss_bins,
        peak_cpu_cores=metric.peak_cpu_cores,
        peak_cpu_reason=metric.peak_cpu_cores_reason,
        sampled_peak_rss_mb=metric.sampled_peak_rss_mb,
        sampled_rss_reason=metric.sampled_peak_rss_reason,
        disk_read_bytes_total=metric.disk_read_bytes_total,
        disk_write_bytes_total=metric.disk_write_bytes_total,
        disk_cancelled_write_bytes_total=metric.disk_cancelled_write_bytes_total,
        disk_io_reason=metric.disk_io_reason,
        cpu_ns_cumulative=metric.cpu_ns_cumulative,
        exit_signal=metric.exit_signal,
        normal_exit_status=metric.normal_exit_status,
        has_causal_end=metric.has_causal_end,
        argv_capture_flags=metric.argv_capture_flags,
        requested_executable_path=metric.requested_executable_path,
        requested_executable_path_truncated=(
            metric.requested_executable_path_truncated
        ),
        exact_argc=metric.exact_argc,
        argv_capped=bool(metric.argv_capture_flags & (1 << MAX_ARGS)),
        truncated_words=tuple(
            index
            for index in range(min(len(metric.argv), MAX_ARGS))
            if metric.argv_capture_flags & (1 << index)
        ),
        bprm_filename=metric.bprm_filename,
        bprm_interp=metric.bprm_interp,
        bprm_evidence_truncated=metric.bprm_evidence_truncated,
        provenance=metric.provenance,
    )


def _failed_exec_attempt_records(
    events: list[dict[str, Any]],
    captured_argv: tuple[
        dict[tuple[int, int], dict[int, str]], dict[tuple[int, int], int]
    ]
    | None = None,
) -> list[Any]:
    from tool_resource.clause_bridge import FailedExecAttempt

    argv_words, argv_capture_flags = (
        _captured_argv(events) if captured_argv is None else captured_argv
    )
    requested_paths: dict[tuple[int, int], str] = {}
    requested_path_truncated: set[tuple[int, int]] = set()
    for event in events:
        if event["type"] != "exec_meta":
            continue
        key = (event["host_pid"], event["exec_seq"])
        requested_paths[key] = event.get("arg", "")
        if int(event.get("arg_flags", 0)) & ARG_FLAG_TRUNCATED:
            requested_path_truncated.add(key)
    attempts: list[FailedExecAttempt] = []
    for event in events:
        if event["type"] != "failed_exec_attempt":
            continue
        words = argv_words.get((event["host_pid"], event["exec_seq"]), {})
        key = (event["host_pid"], event["exec_seq"])
        argv = tuple(words[index] for index in sorted(words))
        attempts.append(
            FailedExecAttempt(
                host_pid=event["host_pid"],
                exec_seq=event["exec_seq"],
                ts_ns=event["ts_ns"],
                argv=argv,
                errno=event["errno"],
                argv_capture_flags=argv_capture_flags.get(key, 0),
                requested_executable_path=requested_paths.get(key),
                requested_executable_path_truncated=key in requested_path_truncated,
            )
        )
    return attempts


def _event_type_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        event_type = str(event["type"])
        counts[event_type] = counts.get(event_type, 0) + 1
    return counts


class ClauseTelemetryCollector:
    """One BPF program, armed once, delimiting serial exec tool calls."""

    def __init__(
        self,
        *,
        container_id: str,
        container_executable: str,
        repo: str,
        artifact_path: Path,
        source_actions: Sequence[Mapping[str, Any]] = (),
        command_rss_oracle: bool = False,
        command_memory_current_oracle: bool = False,
    ) -> None:
        from bcc import BPF, PerfSWConfig, PerfType

        cgroup, init_pid = _container_cgroup(container_id, container_executable)
        self.container_id = container_id
        self.cgroup = cgroup
        self.cgroup_id = cgroup.stat().st_ino
        self.init_pid = init_pid
        self.quota_cores = observed_quota_cores(cgroup)
        self.repo = repo
        self.artifact_path = artifact_path
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        self._epoch_offset_s = time.time() - time.monotonic()
        self._spool = _EventSpool(self.artifact_path.parent)
        self._events_lock = threading.Lock()
        self._poll_lock = threading.Lock()
        self._stop_poll = threading.Event()
        self._poll_error: BaseException | None = None
        self._active: ToolCallToken | None = None
        self._command_rss_oracle_enabled = command_rss_oracle
        self._command_rss_oracle: RssOracle | None = None
        self._command_rss_oracle_error: str | None = None
        self._command_memory_current_oracle_enabled = command_memory_current_oracle
        self._command_memory_current_oracle: MemoryCurrentOracle | None = None
        self._command_memory_current_oracle_error: str | None = None
        self._closed = False
        self.state = "active"
        self._disabled_reason: str | None = None
        self._first_disabled_call: str | None = None
        self._cleanup_status = "pending"
        self._integrity_errors: list[str] = []
        self.calls: list[dict[str, Any]] = []
        self._source_exec_actions = [
            action
            for action in source_actions
            if action.get("action_type") == "tool_exec"
            and isinstance(action.get("data"), Mapping)
            and action["data"].get("tool_name") == "exec"
        ]
        self._source_exec_index = 0

        self._bpf = BPF(text=BPF_PROGRAM)
        try:
            self._bpf.attach_kprobe(event="bprm_execve", fn_name="capture_bprm_argv")
            self._bpf.attach_kprobe(
                event="bprm_change_interp", fn_name="capture_interp_change"
            )
            self._bpf["target_cgroup"][ctypes.c_int(0)] = ctypes.c_ulonglong(
                self.cgroup_id
            )
            self._table = self._bpf["events"]
            self._small_table = self._bpf["events_small"]

            def receiver(table: Any) -> "Callable[[int, int, int], int]":
                # One callback per ring; both append to the one spool, which
                # restores a single order by (ts_ns, arrival). Every emitter
                # stamps its own bpf_ktime_get_ns(), so the timestamp -- not
                # the ring a record arrived on -- is what orders the stream.
                def receive(_ctx: int, data: int, _size: int) -> int:
                    try:
                        row = _event_row(table, data)
                        with self._events_lock:
                            self._spool.append(row)
                    except BaseException as exc:
                        self._poll_error = exc
                        self._stop_poll.set()
                    return 0

                return receive

            self._table.open_ring_buffer(receiver(self._table))
            self._small_table.open_ring_buffer(receiver(self._small_table))

            def poll() -> None:
                try:
                    while not self._stop_poll.is_set():
                        with self._poll_lock:
                            self._bpf.ring_buffer_poll(timeout=10)
                except BaseException as exc:
                    if not self._stop_poll.is_set():
                        self._poll_error = exc

            self._poller = threading.Thread(
                target=poll,
                name="clause-telemetry-ring-poller",
                daemon=True,
            )
            self._poller.start()
            self._bpf.attach_perf_event(
                ev_type=PerfType.SOFTWARE,
                ev_config=PerfSWConfig.CPU_CLOCK,
                fn_name="on_cpu_clock",
                sample_period=SAMPLE_PERIOD_NS,
            )
            self._perf_type = PerfType
            self._perf_config = PerfSWConfig
        except BaseException:
            self._stop_poll.set()
            poller = getattr(self, "_poller", None)
            if poller is not None:
                poller.join(timeout=2)
            try:
                self._bpf.cleanup()
            finally:
                self._spool.close()
            self._closed = True
            self._cleanup_status = "ok"
            raise

    @classmethod
    def unavailable(
        cls,
        *,
        repo: str,
        artifact_path: Path,
        reason: str,
        container_id: str = "",
        source_actions: Sequence[Mapping[str, Any]] = (),
    ) -> "ClauseTelemetryCollector":
        """Return a disabled collector when setup cannot arm BPF."""

        collector = object.__new__(cls)
        collector.container_id = container_id
        collector.cgroup = None
        collector.cgroup_id = 0
        collector.init_pid = 0
        collector.quota_cores = 0.0
        collector.repo = repo
        collector.artifact_path = artifact_path
        collector._epoch_offset_s = time.time() - time.monotonic()
        collector._spool = None
        collector._events_lock = threading.Lock()
        collector._poll_lock = threading.Lock()
        collector._stop_poll = threading.Event()
        collector._poll_error = None
        collector._active = None
        collector._command_rss_oracle_enabled = False
        collector._command_rss_oracle = None
        collector._command_rss_oracle_error = None
        collector._command_memory_current_oracle_enabled = False
        collector._command_memory_current_oracle = None
        collector._command_memory_current_oracle_error = None
        collector._closed = False
        collector.state = "disabled"
        collector._disabled_reason = reason
        collector._first_disabled_call = None
        collector._cleanup_status = "not_started"
        collector._integrity_errors = [reason]
        collector.calls = []
        collector._source_exec_actions = [
            action
            for action in source_actions
            if action.get("action_type") == "tool_exec"
            and isinstance(action.get("data"), Mapping)
            and action["data"].get("tool_name") == "exec"
        ]
        collector._source_exec_index = 0
        return collector

    def _start_command_rss_oracle(self, tool_call_id: str) -> None:
        if not getattr(self, "_command_rss_oracle_enabled", False):
            return
        self._command_rss_oracle_error = None
        try:
            oracle = RssOracle(self.cgroup)
            oracle.start()
            self._command_rss_oracle = oracle
        except BaseException as exc:
            self._command_rss_oracle_error = (
                f"start failed: {type(exc).__name__}: {exc}"
            )

    def _finish_command_rss_oracle(
        self, tool_call_id: str, *, stop_error: str | None = None
    ) -> dict[str, Any] | None:
        if not getattr(self, "_command_rss_oracle_enabled", False):
            return None
        oracle = getattr(self, "_command_rss_oracle", None)
        self._command_rss_oracle = None
        if oracle is None:
            return {
                "status": "unavailable",
                "sampled_peak_rss_mb": None,
                "sample_count": 0,
                "cadence_ms": 2.0,
                "pid_status_read_failures": 0,
                "error": self._command_rss_oracle_error or "sampler not started",
            }
        try:
            oracle.join(timeout=1)
        except BaseException as exc:
            return {
                "status": "unavailable",
                "sampled_peak_rss_mb": None,
                "sample_count": oracle.samples,
                "cadence_ms": 2.0,
                "pid_status_read_failures": int(
                    getattr(oracle, "pid_status_read_failures", 0)
                ),
                "error": f"finish failed: {type(exc).__name__}: {exc}",
            }
        if oracle.is_alive():
            return {
                "status": "unavailable",
                "sampled_peak_rss_mb": None,
                "sample_count": oracle.samples,
                "cadence_ms": 2.0,
                "pid_status_read_failures": int(
                    getattr(oracle, "pid_status_read_failures", 0)
                ),
                "error": "sampler did not stop",
            }
        read_error = getattr(oracle, "read_error", None)
        pid_read_failures = int(getattr(oracle, "pid_status_read_failures", 0))
        valid = (
            oracle.samples > 0
            and stop_error is None
            and read_error is None
            and pid_read_failures == 0
        )
        return {
            "status": "ok" if valid else "unavailable",
            "sampled_peak_rss_mb": (oracle.peak_sum_kb / 1000.0 if valid else None),
            "sample_count": oracle.samples,
            "cadence_ms": 2.0,
            "pid_status_read_failures": pid_read_failures,
            "error": stop_error
            or read_error
            or ("no samples" if not oracle.samples else None),
        }

    def _start_command_memory_current_oracle(self) -> None:
        if not getattr(self, "_command_memory_current_oracle_enabled", False):
            return
        self._command_memory_current_oracle_error = None
        try:
            oracle = MemoryCurrentOracle(self.cgroup)
            oracle.start()
            self._command_memory_current_oracle = oracle
        except BaseException as exc:
            self._command_memory_current_oracle_error = (
                f"start failed: {type(exc).__name__}: {exc}"
            )

    def _finish_command_memory_current_oracle(
        self, *, stop_error: str | None = None
    ) -> dict[str, Any] | None:
        if not getattr(self, "_command_memory_current_oracle_enabled", False):
            return None
        oracle = getattr(self, "_command_memory_current_oracle", None)
        self._command_memory_current_oracle = None
        if oracle is None:
            return {
                "status": "unavailable",
                "sampled_peak_mb": None,
                "sample_count": 0,
                "cadence_ms": 2.0,
                "read_failures": 0,
                "error": self._command_memory_current_oracle_error
                or "sampler not started",
            }
        try:
            oracle.join(timeout=1)
        except BaseException as exc:
            return {
                "status": "unavailable",
                "sampled_peak_mb": None,
                "sample_count": oracle.samples,
                "cadence_ms": 2.0,
                "read_failures": oracle.read_failures,
                "error": f"finish failed: {type(exc).__name__}: {exc}",
            }
        if oracle.is_alive():
            return {
                "status": "unavailable",
                "sampled_peak_mb": None,
                "sample_count": oracle.samples,
                "cadence_ms": 2.0,
                "read_failures": oracle.read_failures,
                "error": "sampler did not stop",
            }
        valid = (
            oracle.samples > 0
            and stop_error is None
            and oracle.read_failures == 0
            and oracle.read_error is None
        )
        return {
            "status": "ok" if valid else "unavailable",
            "sampled_peak_mb": oracle.peak_bytes / 1_000_000 if valid else None,
            "sample_count": oracle.samples,
            "cadence_ms": 2.0,
            "read_failures": oracle.read_failures,
            "error": stop_error
            or oracle.read_error
            or ("no samples" if not oracle.samples else None),
        }

    def _start_command_oracles(self, tool_call_id: str) -> None:
        self._start_command_rss_oracle(tool_call_id)
        self._start_command_memory_current_oracle()

    def _finish_command_oracles(
        self, tool_call_id: str
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        stop_errors: list[str | None] = []
        for oracle in (
            getattr(self, "_command_rss_oracle", None),
            getattr(self, "_command_memory_current_oracle", None),
        ):
            try:
                if oracle is not None:
                    oracle.stop()
                stop_errors.append(None)
            except BaseException as exc:
                stop_errors.append(f"stop failed: {type(exc).__name__}: {exc}")
        return (
            self._finish_command_rss_oracle(tool_call_id, stop_error=stop_errors[0]),
            self._finish_command_memory_current_oracle(stop_error=stop_errors[1]),
        )

    def _disable(self, reason: str, *, tool_call_id: str | None = None) -> None:
        if self.state == "closed":
            return
        if self.state == "active":
            self.state = "disabled"
            self._disabled_reason = reason
            self._first_disabled_call = tool_call_id
            self._stop_poll.set()
        if reason not in self._integrity_errors:
            self._integrity_errors.append(reason)

    def _source_fields(self) -> tuple[str, str, str]:
        source_action = (
            self._source_exec_actions[self._source_exec_index]
            if self._source_exec_index < len(self._source_exec_actions)
            else None
        )
        self._source_exec_index += 1
        return _source_exec_fields(source_action)

    def _unavailable_call(
        self,
        token: ToolCallToken,
        *,
        reason: str | None = None,
    ) -> dict[str, Any]:
        unavailable_reason = reason or self._disabled_reason or "collector_disabled"
        summary = {
            "version": CLAUSE_TELEMETRY_SCHEMA_VERSION,
            "tool_call_id": token.tool_call_id,
            "tool_trace_ref": token.tool_call_id,
            "command": token.command,
            "telemetry_quality": "unavailable",
            "eligible_for_kb": False,
            "invalid_reasons": [
                {"kind": "collector_disabled", "detail": unavailable_reason}
            ],
            "mapping": {
                "static_clause_count": 0,
                "mappable_clause_count": 0,
                "mapped_clause_count": 0,
                "observation_clause_count": 0,
                "no_runtime_exec_count": 0,
                "coverage": 0.0,
                "gaps": [],
                "unobserved_builtins": [],
            },
            "clauses": [],
            "no_runtime_exec": [],
            "integrity": {
                "status": "unavailable",
                "errors": [f"unavailable:collector_disabled:{unavailable_reason}"],
            },
        }
        self.calls.append(summary)
        return summary

    def begin_tool_call(
        self,
        tool_call_id: str,
        command: str,
        *,
        static_plan: Mapping[str, Any] | None = None,
        source_tool_call_id: str | None = None,
        source_command: str | None = None,
        source_tool_result: str | None = None,
        started_ns: int | None = None,
    ) -> ToolCallToken:
        if (
            source_tool_call_id is None
            and source_command is None
            and source_tool_result is None
        ):
            source_tool_call_id, source_command, source_tool_result = (
                self._source_fields()
            )
        elif not all(
            isinstance(value, str)
            for value in (source_tool_call_id, source_command, source_tool_result)
        ):
            raise ValueError("source call plan fields must all be strings")
        if self.state == "active" and self._poll_error is not None:
            self._disable(
                "ring poller failed: "
                f"{type(self._poll_error).__name__}: {self._poll_error}",
                tool_call_id=tool_call_id,
            )
        if self._active is not None:
            self._disable(
                f"overlapping exec tool calls: {self._active.tool_call_id}, "
                f"{tool_call_id}",
                tool_call_id=tool_call_id,
            )
            self._finish_command_oracles(self._active.tool_call_id)
            self._unavailable_call(self._active, reason="exec delimiter desynchronized")
            self._active = None
        if not tool_call_id:
            self._disable(
                "exec tool call has no tool_call_id", tool_call_id=tool_call_id
            )
        counters = dict.fromkeys(
            (
                "ringbuf_reserve_failures",
                "perf_sample_count",
                "argv_read_failures",
                "argv_boundary_read_failures",
            ),
            0,
        )
        if self.state == "active":
            try:
                counters = {
                    "ringbuf_reserve_failures": _counter(
                        self._bpf, "ringbuf_reserve_failures"
                    ),
                    "perf_sample_count": _counter(self._bpf, "perf_sample_count"),
                    "argv_read_failures": _counter(self._bpf, "argv_read_failures"),
                    "argv_boundary_read_failures": _counter(
                        self._bpf, "argv_boundary_read_failures"
                    ),
                }
            except BaseException as exc:
                self._disable(
                    f"collector counter read failed: {type(exc).__name__}: {exc}",
                    tool_call_id=tool_call_id,
                )
        now_ns = time.monotonic_ns()
        if started_ns is None:
            started_ns = now_ns
        elif (
            not isinstance(started_ns, int)
            or isinstance(started_ns, bool)
            or started_ns <= 0
            or started_ns > now_ns
        ):
            raise ValueError("started_ns must be a past positive monotonic timestamp")
        # Drop everything the ring delivered before this call opened. The
        # finish-time slice keeps only [started_ns, ended_ns], and calls are
        # sequential, so an earlier event can belong to no call at all -- but it
        # was retained until the next finish anyway. Container background tasks
        # keep the perf sampler firing while the agent is between tool calls, so
        # that gap accumulated events destined to be discarded.
        if self._spool is not None:
            try:
                with self._events_lock:
                    self._spool.drop_before(started_ns)
            except BaseException as exc:
                self._disable(
                    f"collector spool prepare failed: {type(exc).__name__}: {exc}",
                    tool_call_id=tool_call_id,
                )
        token = ToolCallToken(
            tool_call_id=tool_call_id,
            command=command,
            started_ns=started_ns,
            ringbuf_reserve_failures=int(counters["ringbuf_reserve_failures"]),
            perf_sample_count=int(counters["perf_sample_count"]),
            argv_read_failures=int(counters["argv_read_failures"]),
            argv_boundary_read_failures=int(counters["argv_boundary_read_failures"]),
            source_tool_call_id=source_tool_call_id,
            source_command=source_command,
            source_tool_result=source_tool_result,
            static_plan=static_plan,
        )
        self._active = token
        if self.state == "active":
            self._start_command_oracles(tool_call_id)
        return token

    def finish_tool_call(
        self,
        token: ToolCallToken,
        *,
        replay_response: Mapping[str, Any] | None = None,
        ended_ns: int | None = None,
    ) -> dict[str, Any]:
        if token is not self._active:
            self._finish_command_oracles(token.tool_call_id)
            self._disable(
                f"exec delimiter mismatch for {token.tool_call_id}",
                tool_call_id=token.tool_call_id,
            )
            return self._unavailable_call(token, reason="exec delimiter desynchronized")
        now_ns = time.monotonic_ns()
        if ended_ns is None:
            ended_ns = now_ns
        elif (
            not isinstance(ended_ns, int)
            or isinstance(ended_ns, bool)
            or ended_ns < token.started_ns
            or ended_ns > now_ns
        ):
            self._finish_command_oracles(token.tool_call_id)
            self._active = None
            self._disable(
                f"invalid exec delimiter end for {token.tool_call_id}",
                tool_call_id=token.tool_call_id,
            )
            raise ValueError("ended_ns must be a valid past monotonic timestamp")
        command_window_rss, command_window_memory_current = (
            self._finish_command_oracles(token.tool_call_id)
        )
        self._active = None
        if self.state != "active":
            return self._unavailable_call(token)
        if self._poll_error is not None:
            self._disable(
                "ring poller failed: "
                f"{type(self._poll_error).__name__}: {self._poll_error}",
                tool_call_id=token.tool_call_id,
            )
            return self._unavailable_call(token)
        # The kernel timestamps events before ring delivery. Serialize one final
        # consume with the poll thread before snapshotting so every event already
        # published by the completed command crosses the call boundary.
        try:
            with self._poll_lock:
                self._bpf.ring_buffer_consume()
                if self._poll_error is not None:
                    raise self._poll_error
                loss_counts = _loss_delta(self._bpf, token)
                perf_samples = (
                    _counter(self._bpf, "perf_sample_count") - token.perf_sample_count
                )
                with self._events_lock:
                    spool_snapshot = self._spool.snapshot()
            event_source = _sorted_event_source(
                spool_snapshot,
                started_ns=token.started_ns,
                ended_ns=ended_ns,
                cgroup_id=self.cgroup_id,
                directory=self.artifact_path.parent,
            )
            with self._events_lock:
                self._spool.release_through(ended_ns, spool_snapshot)
        except BaseException as exc:
            self._disable(
                f"collector finish failed: {type(exc).__name__}: {exc}",
                tool_call_id=token.tool_call_id,
            )
            return self._unavailable_call(token)
        replay_result = (
            str(replay_response.get("result") or "")
            if replay_response is not None
            else ""
        )
        replay_stderr = (
            str(replay_response.get("stderr") or "")
            if replay_response is not None
            else ""
        )
        raw_replay_exit = (
            replay_response.get("returncode") if replay_response is not None else None
        )
        replay_exit_code = (
            raw_replay_exit
            if isinstance(raw_replay_exit, int)
            and not isinstance(raw_replay_exit, bool)
            else None
        )
        protocol_timeout = _is_protocol_timeout(
            replay_exit_code,
            replay_result,
        )
        source_exit_code = _strict_exit_code(token.source_tool_result)
        replay_tool_result = _replay_tool_result(
            replay_response,
            replay_exit_code,
        )
        control_flow_fidelity = {
            "source_action_available": bool(token.source_tool_call_id),
            "source_command_matches": token.source_command == token.command,
            "source_exit_code": source_exit_code,
            "replay_exit_code": replay_exit_code,
            "exit_code_matches": (
                source_exit_code is not None
                and replay_exit_code is not None
                and source_exit_code == replay_exit_code
            ),
            "tool_result_exact": (
                bool(token.source_tool_call_id)
                and token.source_tool_result == replay_tool_result
            ),
        }
        control_flow_fidelity["short_circuit_eligible"] = (
            control_flow_fidelity["source_command_matches"]
            and control_flow_fidelity["exit_code_matches"]
            and control_flow_fidelity["tool_result_exact"]
        )
        lookup_failure = shell_command_lookup_failure_evidence(
            command=token.command,
            source_tool_call_id=token.source_tool_call_id,
            replay_tool_call_id=token.tool_call_id,
            source_command=token.source_command,
            source_tool_result=token.source_tool_result,
            replay_result=replay_result,
            replay_stderr=replay_stderr,
            replay_exit_code=replay_exit_code,
        )
        try:
            with event_source:
                summary, violations = self._summarize_call(
                    token=token,
                    ended_ns=ended_ns,
                    events=event_source,
                    raw_event_count=len(event_source),
                    loss_counts=loss_counts,
                    perf_samples=perf_samples,
                    command_lookup_failure=lookup_failure,
                    control_flow_fidelity=control_flow_fidelity,
                    protocol_timeout=protocol_timeout,
                )
        except Exception as exc:
            message = (
                f"{token.tool_call_id}: telemetry analysis failed: "
                f"{type(exc).__name__}: {exc}"
            )
            if isinstance(exc, ClauseTelemetryIntegrityError):
                self._integrity_errors.append(message)
            else:
                self._disable(message, tool_call_id=token.tool_call_id)
            failed_call = {
                "version": CLAUSE_TELEMETRY_SCHEMA_VERSION,
                "tool_call_id": token.tool_call_id,
                "tool_trace_ref": token.tool_call_id,
                "command": token.command,
                "telemetry_quality": "invalid",
                "eligible_for_kb": False,
                "invalid_reasons": [{"kind": "analysis_failure", "detail": message}],
                "integrity": {"status": "failed", "errors": [message]},
            }
            if isinstance(exc, ClauseTelemetryIntegrityError):
                failed_call.update(exc.artifact_payload)
            if command_window_rss is not None:
                failed_call["command_window_rss"] = command_window_rss
            if command_window_memory_current is not None:
                failed_call["command_window_memory_current"] = (
                    command_window_memory_current
                )
            self.calls.append(failed_call)
            return failed_call
        if command_window_rss is not None:
            summary["command_window_rss"] = command_window_rss
        if command_window_memory_current is not None:
            summary["command_window_memory_current"] = command_window_memory_current
        self.calls.append(summary)
        for violation in violations:
            if violation not in self._integrity_errors:
                self._integrity_errors.append(violation)
        return summary

    def record_safety_guard_blocked(
        self,
        tool_call_id: str,
        command: str,
        replay_result: str,
        *,
        static_plan: Mapping[str, Any] | None = None,
        source_tool_call_id: str | None = None,
        source_command: str | None = None,
        source_tool_result: str | None = None,
    ) -> dict[str, Any]:
        """Record an exec rejected before the container runtime was entered."""

        from tool_resource.clause_bridge import SafetyGuardBlockEvidence

        token = self.begin_tool_call(
            tool_call_id,
            command,
            static_plan=static_plan,
            source_tool_call_id=source_tool_call_id,
            source_command=source_command,
            source_tool_result=source_tool_result,
        )
        ended_ns = time.monotonic_ns()
        command_window_rss, command_window_memory_current = (
            self._finish_command_oracles(tool_call_id)
        )
        self._active = None
        if self.state != "active":
            return self._unavailable_call(token)
        try:
            loss_counts = _loss_delta(self._bpf, token)
            perf_samples = (
                _counter(self._bpf, "perf_sample_count") - token.perf_sample_count
            )
        except BaseException as exc:
            self._disable(
                f"collector finish failed: {type(exc).__name__}: {exc}",
                tool_call_id=tool_call_id,
            )
            return self._unavailable_call(token)
        evidence = SafetyGuardBlockEvidence(
            command=command,
            source_command=token.source_command,
            source_tool_call_id=token.source_tool_call_id,
            replay_tool_call_id=token.tool_call_id,
            source_result=token.source_tool_result,
            replay_result=replay_result,
        )
        fidelity = {
            "source_action_available": bool(token.source_tool_call_id),
            "source_command_matches": token.source_command == command,
            "source_exit_code": None,
            "replay_exit_code": None,
            "exit_code_matches": False,
            "tool_result_exact": token.source_tool_result == replay_result,
            "short_circuit_eligible": False,
        }
        try:
            summary, violations = self._summarize_call(
                token=token,
                ended_ns=ended_ns,
                events=[],
                loss_counts=loss_counts,
                perf_samples=perf_samples,
                control_flow_fidelity=fidelity,
                safety_guard_blocked=evidence,
            )
        except Exception as exc:
            message = (
                f"{token.tool_call_id}: telemetry analysis failed: "
                f"{type(exc).__name__}: {exc}"
            )
            if isinstance(exc, ClauseTelemetryIntegrityError):
                self._integrity_errors.append(message)
            else:
                self._disable(message, tool_call_id=token.tool_call_id)
            summary = {
                "version": CLAUSE_TELEMETRY_SCHEMA_VERSION,
                "tool_call_id": token.tool_call_id,
                "tool_trace_ref": token.tool_call_id,
                "command": token.command,
                "telemetry_quality": "invalid",
                "eligible_for_kb": False,
                "invalid_reasons": [{"kind": "analysis_failure", "detail": message}],
                "integrity": {"status": "failed", "errors": [message]},
            }
            violations = [message]
        if command_window_rss is not None:
            summary["command_window_rss"] = command_window_rss
        if command_window_memory_current is not None:
            summary["command_window_memory_current"] = command_window_memory_current
        self.calls.append(summary)
        for violation in violations:
            if violation not in self._integrity_errors:
                self._integrity_errors.append(violation)
        return summary

    def _summarize_call(
        self,
        *,
        token: ToolCallToken,
        ended_ns: int,
        events: list[dict[str, Any]],
        raw_event_count: int | None = None,
        loss_counts: Mapping[str, int],
        perf_samples: int,
        command_lookup_failure: ShellCommandLookupFailure | None = None,
        control_flow_fidelity: Mapping[str, Any] | None = None,
        safety_guard_blocked: Any | None = None,
        protocol_timeout: bool = False,
    ) -> tuple[dict[str, Any], list[str]]:
        from tool_resource.clause_bridge import bridge_command

        normalized_loss_counts = {
            name: int(loss_counts.get(name, 0)) for name in LOSS_COUNTER_NAMES
        }
        loss = sum(normalized_loss_counts.values())
        run = RawRun(
            cgroup_id=self.cgroup_id,
            quota_cores=self.quota_cores,
            status=0,
            wall_ns=ended_ns - token.started_ns,
            usage_usec=0,
            ringbuf_reserve_failures=normalized_loss_counts["ringbuf_reserve_failures"],
            perf_sample_count=perf_samples,
            oracle_peak_rss_kb=0,
            oracle_samples=0,
            marker=True,
            events=events,
            argv_read_failures=normalized_loss_counts["argv_read_failures"],
            argv_boundary_read_failures=normalized_loss_counts[
                "argv_boundary_read_failures"
            ],
        )
        fork_records: dict[int, list[dict[str, Any]]] = {}
        for event in _events_of_types(events, _FORK_EVENT_TYPES):
            if event["type"] == "fork" and event["child_host_pid"]:
                fork_records.setdefault(event["child_host_pid"], []).append(event)
        fork_parent = {
            child: next(iter({record["host_pid"] for record in records}))
            for child, records in fork_records.items()
            if len({record["host_pid"] for record in records}) == 1
        }
        # Reassemble argv and reconstruct the clause list ONCE for this call:
        # analyze() and the failed-exec records below reuse both rather than
        # rebuilding them from the same events.
        captured_argv = _captured_argv(_events_of_types(events, _ARGV_EVENT_TYPES))
        clauses_and_lineage = _clauses_and_lineage(
            _events_of_types(events, _LINEAGE_EVENT_TYPES), captured_argv
        )
        clauses, _ = clauses_and_lineage
        if safety_guard_blocked is not None and not clauses and not events:
            entry_pid = 0
            root_pids: set[int] = set()
            command_tree = {
                "status": "not_applicable",
                "reason": "safety_guard_blocked_before_runtime",
                "entry_pid": None,
                "root_pids": [],
                "exec_ancestry": [],
            }
        else:
            entry_pid, root_pids, command_tree = _command_tree_provenance(
                clauses,
                fork_parent,
                fork_records=fork_records,
            )
        if isinstance(events, _SortedEventSource):
            metrics, attribution_gaps = _analyze_streaming(
                run,
                entry_pid=entry_pid,
                clauses_and_lineage=clauses_and_lineage,
            )
        else:
            metrics, attribution_gaps = analyze(
                run,
                entry_pid=entry_pid,
                clauses_and_lineage=clauses_and_lineage,
            )

        def command_descendant(pid: int) -> bool:
            current = pid
            seen: set[int] = set()
            while current and current not in seen:
                if current in root_pids:
                    return True
                seen.add(current)
                current = fork_parent.get(current, 0)
            return False

        def gap_payload(event: Mapping[str, Any]) -> dict[str, Any]:
            structural_setup_reasons = {
                "entry_fork_pre_exec_structural_setup",
                "initial_exec_pending_pre_boundary_structural_setup",
            }
            if event["reason"] in structural_setup_reasons:
                relation = event["reason"]
            elif event["host_pid"] == entry_pid:
                relation = (
                    "entry_parent"
                    if event["host_tid"] == entry_pid
                    else "entry_parent_thread"
                )
            elif command_descendant(event["host_pid"]):
                relation = "command_descendant"
            else:
                relation = "outside_entry_parent_and_command"
            lineage_id = (
                event["host_tid"]
                if event["host_tid"] != event["host_pid"]
                else event["host_pid"]
            )
            payload = {
                "type": event["type"],
                "ts_ns": event["ts_ns"],
                "host_pid": event["host_pid"],
                "host_tid": event["host_tid"],
                "exec_seq": event["exec_seq"],
                "entry_pid": entry_pid,
                "entry_parent_relation": relation,
                "fork_parent_pid": fork_parent.get(lineage_id),
                "reason": event["reason"],
            }
            if "fork_ancestry" in event:
                payload["fork_ancestry"] = list(event["fork_ancestry"])
            if "fork_chain_records" in event:
                payload["fork_chain_records"] = list(event["fork_chain_records"])
            if "fork_resolution_failure" in event:
                payload["fork_resolution_failure"] = dict(
                    event["fork_resolution_failure"]
                )
            if "fork_ts_ns" in event:
                payload["fork_ts_ns"] = event["fork_ts_ns"]
            if "pending_exec_evidence" in event:
                payload["pending_exec_evidence"] = dict(event["pending_exec_evidence"])
            return payload

        gap_evidence = [gap_payload(event) for event in attribution_gaps]
        relevant_gaps = [
            event
            for event in gap_evidence
            if event["entry_parent_relation"]
            not in {
                "entry_parent",
                "entry_parent_thread",
                "entry_fork_pre_exec_structural_setup",
                "initial_exec_pending_pre_boundary_structural_setup",
            }
        ]
        structural_gaps = [
            event
            for event in gap_evidence
            if event["entry_parent_relation"]
            in {
                "entry_parent",
                "entry_parent_thread",
                "entry_fork_pre_exec_structural_setup",
                "initial_exec_pending_pre_boundary_structural_setup",
            }
        ]
        exec_image_records = [_exec_image_record(metric) for metric in metrics]
        bridge = bridge_command(
            self.repo,
            token.command,
            exec_image_records,
            parsed_command=token.static_plan,
            failed_exec_attempts=[
                attempt
                for attempt in _failed_exec_attempt_records(events, captured_argv)
                if command_descendant(attempt.host_pid)
            ],
            command_lookup_failure=command_lookup_failure,
            safety_guard_blocked=safety_guard_blocked,
            # Short-circuit resolution answers "did this clause execute in THIS
            # replay?", which is decided entirely by replay-side kernel evidence
            # (the controller's normal_exit_status from task->exit_code) plus the
            # replay's own parse tree. It does not depend on the replay matching
            # the source trace. `control_flow_fidelity` is retained in provenance
            # as a separate replay-quality signal; gating on it here starved
            # stateful workloads (Terminal-Bench) of clause data whenever output
            # diverged, while the resolver's own conservatism already fails
            # closed on missing or contradictory runtime evidence.
            allow_control_short_circuit=True,
            entry_pid=entry_pid,
            fork_parent=fork_parent,
            epoch_offset=self._epoch_offset_s,
            loss_count=loss,
            attribution_gap_count=len(relevant_gaps),
            protocol_timeout=protocol_timeout,
            call_end_ns=ended_ns,
        )
        mapping_gaps = [
            {"kind": gap.kind, "detail": gap.detail} for gap in bridge.coverage_gaps
        ]
        clauses = [
            {
                "bin": bridged.observation.bin,
                "argv": list(bridged.observation.argv),
                "ts_start": bridged.observation.ts_start,
                "ts_end": bridged.observation.ts_end,
                "latency_ms": bridged.observation.latency_ms,
                "peak_cpu_cores": bridged.observation.peak_cpu_cores,
                "sampled_peak_rss_mb": bridged.observation.sampled_peak_rss_mb,
                "cpu_ns_cumulative": bridged.observation.cpu_ns_cumulative,
                "cpu_window_profile": [
                    {
                        "start_offset_s": start,
                        "end_offset_s": end,
                        "span_s": span,
                        "cpu_ns": cpu_ns,
                        "cpu_cores": cores,
                    }
                    for start, end, span, cpu_ns, cores in bridged.cpu_window_profile
                ],
                "in_loop": bridged.observation.in_loop,
                "in_pipe": bridged.observation.in_pipe,
                "in_subst": bridged.observation.in_subst,
                "pipeline_position": bridged.observation.pipeline_position,
                "disk_io": {
                    "read_bytes_total": bridged.disk_read_bytes_total,
                    "write_bytes_total": bridged.disk_write_bytes_total,
                    "cancelled_write_bytes_total": (
                        bridged.disk_cancelled_write_bytes_total
                    ),
                    "read_write_bytes_total": (
                        bridged.disk_read_bytes_total + bridged.disk_write_bytes_total
                        if bridged.disk_read_bytes_total is not None
                        and bridged.disk_write_bytes_total is not None
                        else None
                    ),
                    "availability": bridged.availability["disk_io"],
                },
                "availability": bridged.availability,
                "mapping_evidence": bridged.mapping_evidence,
                "owned_exec_image_count": len(bridged.owned_exec_images),
                "provenance": bridged.provenance,
            }
            for bridged in bridge.bridged
        ]

        def no_runtime_exec_row(resolved: Any) -> dict[str, Any]:
            row = {
                "bin": resolved.bin,
                "argv": list(resolved.argv),
                "availability": resolved.availability,
                "mapping_evidence": resolved.mapping_evidence,
                "attempt_count": len(resolved.attempts),
            }
            if resolved.safety_guard_blocked is not None:
                evidence = resolved.safety_guard_blocked
                row["provenance"] = {
                    "evidence_kind": "safety_guard_blocked_before_runtime",
                    "command": evidence.command,
                    "source": {
                        "tool_call_id": evidence.source_tool_call_id,
                        "command": evidence.source_command,
                        "result": evidence.source_result,
                    },
                    "replay": {
                        "tool_call_id": evidence.replay_tool_call_id,
                        "result": evidence.replay_result,
                    },
                }
                return row
            if resolved.control_short_circuit is not None:
                row["provenance"] = {
                    "evidence_kind": "shell_control_short_circuit",
                    **resolved.control_short_circuit,
                }
                return row
            if resolved.command_lookup_failure is None:
                row["errno"] = sorted({attempt.errno for attempt in resolved.attempts})
                row["provenance"] = {
                    "evidence_kind": "failed_execve",
                    "failed_exec_attempts": [
                        {
                            "host_pid": attempt.host_pid,
                            "exec_seq": attempt.exec_seq,
                            "ts_ns": attempt.ts_ns,
                            "errno": attempt.errno,
                        }
                        for attempt in resolved.attempts
                    ],
                }
                return row
            evidence = resolved.command_lookup_failure
            row["provenance"] = {
                "evidence_kind": "shell_command_lookup_failure",
                "parser": evidence.parser,
                "command": evidence.command,
                "executable_head": evidence.executable_head,
                "exit_code_semantics": evidence.exit_code_semantics,
                "source": {
                    "tool_call_id": evidence.source_tool_call_id,
                    "exit_code": evidence.source_exit_code,
                    "channel": evidence.source_channel,
                    "diagnostic": evidence.source_diagnostic,
                },
                "replay": {
                    "tool_call_id": evidence.replay_tool_call_id,
                    "exit_code": evidence.replay_exit_code,
                    "channel": evidence.replay_channel,
                    "diagnostic": evidence.replay_diagnostic,
                },
            }
            return row

        no_runtime_exec = [
            no_runtime_exec_row(resolved) for resolved in bridge.no_runtime_exec
        ]
        target_availability: dict[str, Any] = {}
        for target in ("latency", "cpu", "memory"):
            values = [
                clause["availability"][target]
                for clause in [*clauses, *no_runtime_exec]
            ]
            reasons: dict[str, int] = {}
            for value in values:
                reasons[value] = reasons.get(value, 0) + 1
            target_availability[target] = {
                "available": sum(value == "ok" for value in values),
                "total": len(values),
                "reasons": reasons,
            }
        mappable = bridge.static_clause_count - len(bridge.unobserved_builtins)
        # Count static clauses, not bridged entries: a loop body is one static
        # clause that yields one observation per iteration, so len(bridge.bridged)
        # would push coverage above 1.0.
        mapped = bridge.bridged_clause_count + len(bridge.no_runtime_exec)
        summary = {
            "version": CLAUSE_TELEMETRY_SCHEMA_VERSION,
            "tool_call_id": token.tool_call_id,
            "tool_trace_ref": token.tool_call_id,
            "command": token.command,
            "static_word_intent": [
                {
                    "clause_index": index,
                    "bin": clause["bin"],
                    "argv": clause["argv"],
                    "span": clause["span"],
                    "structural_context": clause.get("structural_context", []),
                    "word_intents": clause.get("word_intents", []),
                }
                for index, clause in enumerate(bridge.static_clauses)
            ],
            "runtime_invocations": [
                {
                    "host_pid": image.host_pid,
                    "exec_seq": image.exec_seq,
                    "requested_executable_path": image.requested_executable_path,
                    "requested_executable_path_truncated": (
                        image.requested_executable_path_truncated
                    ),
                    "argv": list(image.argv),
                    "argc": image.exact_argc,
                    "argv_capped": image.argv_capped,
                    "truncated_words": list(image.truncated_words),
                    "bprm_filename": image.bprm_filename,
                    "bprm_interp": image.bprm_interp,
                    "bprm_evidence_truncated": image.bprm_evidence_truncated,
                }
                for image in exec_image_records
            ],
            "transition_graph": bridge.transition_graph,
            "candidate_rejections": bridge.candidate_rejections,
            "mapping": {
                "static_clause_count": bridge.static_clause_count,
                "mappable_clause_count": mappable,
                "mapped_clause_count": mapped,
                "observation_clause_count": len(bridge.observations),
                "no_runtime_exec_count": len(bridge.no_runtime_exec),
                "coverage": mapped / max(mappable, 1),
                "gaps": mapping_gaps,
                "unobserved_builtins": bridge.unobserved_builtins,
            },
            "target_availability": target_availability,
            "coverage_gaps": {
                "relevant": {
                    "count": len(relevant_gaps),
                    "event_types": _event_type_counts(relevant_gaps),
                    "events": relevant_gaps,
                },
                "structural": {
                    "count": len(structural_gaps),
                    "event_types": _event_type_counts(structural_gaps),
                    "events": structural_gaps,
                },
            },
            "telemetry_loss": {
                **normalized_loss_counts,
                "total": loss,
                "perf_sample_count": perf_samples,
            },
            "ring_loss": {
                "reserve_failures": loss,
                "perf_sample_count": perf_samples,
            },
            "clauses": clauses,
            "no_runtime_exec": no_runtime_exec,
            "provenance": {
                "collector": CLAUSE_TELEMETRY_COLLECTOR,
                "repo": self.repo,
                "container_cgroup_id": self.cgroup_id,
                "quota_cores": self.quota_cores,
                "page_size_bytes": PAGE,
                "call_started_monotonic_ns": token.started_ns,
                "call_ended_monotonic_ns": ended_ns,
                "raw_event_count": (
                    len(events) if raw_event_count is None else raw_event_count
                ),
                "exec_image_count": len(metrics),
                "command_tree": command_tree,
                "source_replay_control_flow_fidelity": (
                    dict(control_flow_fidelity)
                    if control_flow_fidelity is not None
                    else {
                        "source_action_available": False,
                        "short_circuit_eligible": False,
                    }
                ),
                "cadence_ns": SAMPLE_PERIOD_NS,
                "window_ns": WINDOW_NS,
                "align_bin_ns": ALIGN_BIN_NS,
                "disk_io_semantics": "linux_task_io_accounting_total_bytes",
                "disk_io_fields": [
                    "task->ioac.read_bytes",
                    "task->ioac.write_bytes",
                    "task->ioac.cancelled_write_bytes",
                ],
            },
        }
        violations: list[str] = []
        if loss:
            causes = ",".join(
                f"{name}={count}"
                for name, count in normalized_loss_counts.items()
                if count
            )
            violations.append(f"{token.tool_call_id}: telemetry loss={loss} ({causes})")
        if relevant_gaps:
            violations.append(
                f"{token.tool_call_id}: relevant coverage gaps={len(relevant_gaps)}"
            )
        if mapping_gaps:
            kinds = sorted({gap["kind"] for gap in mapping_gaps})
            violations.append(f"{token.tool_call_id}: mapping gaps={','.join(kinds)}")
        invalid_reasons: list[dict[str, str]] = []
        if loss:
            invalid_reasons.append({"kind": "telemetry_loss", "detail": violations[0]})
        if relevant_gaps:
            invalid_reasons.append(
                {
                    "kind": "attribution_gap",
                    "detail": (
                        f"{token.tool_call_id}: relevant coverage "
                        f"gaps={len(relevant_gaps)}"
                    ),
                }
            )
        invalid_reasons.extend(mapping_gaps)
        summary["telemetry_quality"] = "invalid" if violations else "ok"
        summary["eligible_for_kb"] = not violations and bridge.data_valid
        summary["invalid_reasons"] = invalid_reasons
        for clause in summary["clauses"]:
            clause["telemetry_quality"] = summary["telemetry_quality"]
            clause["eligible_for_kb"] = summary["eligible_for_kb"]
        summary["integrity"] = {
            "status": "failed" if violations else "ok",
            "errors": violations,
        }
        return summary, violations

    def add_integrity_error(self, message: str) -> None:
        self._disable(message)

    def finalize(self, *, replay_execution: str = "completed") -> None:
        if self.state == "closed":
            return
        if replay_execution not in {"completed", "failed", "incomplete"}:
            raise ValueError(f"invalid replay execution state {replay_execution!r}")
        try:
            total_loss_counts = (
                _loss_counts(self._bpf)
                if self.state == "active"
                else dict.fromkeys(LOSS_COUNTER_NAMES, 0)
            )
        except BaseException as exc:
            total_loss_counts = dict.fromkeys(LOSS_COUNTER_NAMES, 0)
            self._disable(f"collector counter read failed: {type(exc).__name__}: {exc}")
        argv_read_failure_sites = (
            _argv_read_failure_sites(self._bpf)
            if self.state == "active" and total_loss_counts["argv_read_failures"]
            else {}
        )
        total_loss = sum(total_loss_counts.values())
        if self._active is not None:
            self._disable(
                f"unterminated exec delimiter: {self._active.tool_call_id}",
                tool_call_id=self._active.tool_call_id,
            )
            self._finish_command_oracles(self._active.tool_call_id)
            self._unavailable_call(self._active, reason="unterminated exec delimiter")
            self._active = None
        try:
            if hasattr(self, "_bpf"):
                self._close_bpf()
        except BaseException as exc:
            self._cleanup_status = "failed"
            self._disable(f"collector cleanup leak: {type(exc).__name__}: {exc}")
        if total_loss:
            causes = ",".join(
                f"{name}={count}" for name, count in total_loss_counts.items() if count
            )
            self._integrity_errors.append(
                f"collector total telemetry loss={total_loss} ({causes})"
            )
        if argv_read_failure_sites:
            sites = ",".join(
                f"{name}={count}" for name, count in argv_read_failure_sites.items()
            )
            self._integrity_errors.append(f"argv read failure sites: {sites}")
        if self._poll_error is not None:
            self._disable(
                "ring poller failed: "
                f"{type(self._poll_error).__name__}: {self._poll_error}"
            )
        prior_state = self.state
        valid_count = sum(call.get("telemetry_quality") == "ok" for call in self.calls)
        invalid_count = sum(
            call.get("telemetry_quality") == "invalid" for call in self.calls
        )
        unavailable_count = sum(
            call.get("telemetry_quality") == "unavailable" for call in self.calls
        )
        eligible_count = sum(call.get("eligible_for_kb") is True for call in self.calls)
        collector_healthy = (
            prior_state == "active"
            and self._cleanup_status == "ok"
            and total_loss == 0
            and unavailable_count == 0
        )
        telemetry_quality = "ok" if collector_healthy else "unavailable"
        formal_completeness = (
            "unavailable"
            if not collector_healthy
            else ("complete" if eligible_count == len(self.calls) else "partial")
        )
        collection_validity = "valid" if collector_healthy else "invalid"
        call_errors = {
            str(error)
            for call in self.calls
            for error in (call.get("integrity") or {}).get("errors", [])
        }
        collector_errors = (
            [error for error in self._integrity_errors if error not in call_errors]
            if collector_healthy
            else list(self._integrity_errors)
        )
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_path.write_text(
            json.dumps(
                {
                    "version": CLAUSE_TELEMETRY_SCHEMA_VERSION,
                    "mode": "clause",
                    "status_model": CLAUSE_TELEMETRY_STATUS_MODEL,
                    "container_id": self.container_id,
                    "cgroup_id": self.cgroup_id,
                    "quota_cores": self.quota_cores,
                    "calls": self.calls,
                    "telemetry_loss_total": {
                        **total_loss_counts,
                        "total": total_loss,
                    },
                    "argv_read_failure_sites": argv_read_failure_sites,
                    "ring_loss_total": total_loss,
                    "cleanup": self._cleanup_status,
                    "collector": {
                        "state": "closed",
                        "state_before_close": prior_state,
                        "health": ("healthy" if collector_healthy else "unavailable"),
                        "first_disabled_call": self._first_disabled_call,
                        "disabled_reason": self._disabled_reason,
                        "valid_call_count": valid_count,
                        "invalid_call_count": invalid_count,
                        "unavailable_call_count": unavailable_count,
                        "eligible_call_count": eligible_count,
                    },
                    "call_coverage": {
                        "total_call_count": len(self.calls),
                        "eligible_call_count": eligible_count,
                        "withheld_call_count": len(self.calls) - eligible_count,
                        "eligible_fraction": (
                            eligible_count / len(self.calls) if self.calls else 1.0
                        ),
                    },
                    "replay_execution": replay_execution,
                    "telemetry_quality": telemetry_quality,
                    "formal_completeness": formal_completeness,
                    "collection_validity": collection_validity,
                    "integrity": {
                        "status": "ok" if collector_healthy else "failed",
                        "errors": collector_errors,
                    },
                    "provenance": {
                        "collector": CLAUSE_TELEMETRY_COLLECTOR,
                        "repo": self.repo,
                        "page_size_bytes": PAGE,
                        "cadence_ns": SAMPLE_PERIOD_NS,
                        "window_ns": WINDOW_NS,
                        "align_bin_ns": ALIGN_BIN_NS,
                        "disk_io_semantics": (
                            "nonnegative_per_tid_linux_task_io_accounting_deltas"
                        ),
                        "disk_io_fields": [
                            "task->ioac.read_bytes",
                            "task->ioac.write_bytes",
                            "task->ioac.cancelled_write_bytes",
                        ],
                    },
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self.state = "closed"
        _trim_process_heap()

    def _close_bpf(self) -> None:
        if self._closed:
            return
        try:
            self._stop_poll.set()
            self._poller.join(timeout=2)
            if self._poller.is_alive():
                raise RuntimeError("ring poller did not stop")
            self._bpf.detach_perf_event(
                ev_type=self._perf_type.SOFTWARE,
                ev_config=self._perf_config.CPU_CLOCK,
            )
            self._bpf.cleanup()
        finally:
            if self._spool is not None:
                self._spool.close()
            self._closed = True
        self._cleanup_status = "ok"


__all__ = [
    "ALIGN_BIN_NS",
    "BPF_PROGRAM",
    "Clause",
    "ClauseMetrics",
    "ClauseTelemetryCollector",
    "ClauseTelemetryIntegrityError",
    "EventRow",
    "RawRun",
    "SAMPLE_PERIOD_NS",
    "SENTINEL",
    "ToolCallToken",
    "WINDOW_NS",
    "analyze",
    "cpu_window_profile",
    "resolve_inherited_owner_sample",
    "rss_bin_profile",
    "validate_clause_telemetry_runtime",
]
