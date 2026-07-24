#!/usr/bin/python3
"""One-run BCC collector for the frozen Stage-1 synthetic cases."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import threading
import time
from typing import Any

from bcc import BPF


ARG_BYTES = 1024
MAX_ARGS = 16
TYPE_NAMES = {
    1: "exec_arg",
    2: "exec_argv_end",
    3: "exec",
    4: "fork",
    5: "exit",
    6: "hiwater",
}

BPF_PROGRAM = rf"""
#include <linux/mm_types.h>
#include <linux/sched.h>
#include <linux/sched/signal.h>

#define TYPE_EXEC_ARG 1
#define TYPE_EXEC_END 2
#define TYPE_EXEC 3
#define TYPE_FORK 4
#define TYPE_EXIT 5
#define TYPE_HIWATER 6
#define FLAG_ARG_TRUNCATED 1
#define FLAG_ARG_COUNT_TRUNCATED 2
#define FLAG_MISSING_ENTER 4
#define MAX_ARGS {MAX_ARGS}
#define ARG_BYTES {ARG_BYTES}

struct event_t {{
    u64 timestamp_ns;
    u64 cgroup_id;
    u64 exec_seq;
    u64 utime_ns;
    u64 stime_ns;
    u64 hiwater_pages;
    u32 type;
    u32 host_pid;
    u32 host_tid;
    u32 parent_host_pid;
    u32 child_host_pid;
    u32 arg_index;
    u32 arg_length;
    u32 flags;
    u32 exit_code;
    u32 reserved;
    char arg[ARG_BYTES];
}};

BPF_RINGBUF_OUTPUT(events, 256);
BPF_ARRAY(target_cgroup, u64, 1);
BPF_QUEUE(exec_sequences, u64, 65536);
BPF_ARRAY(sequence_ready, u32, 1);
BPF_ARRAY(sequence_failures, u64, 1);
BPF_ARRAY(reserve_failures, u64, 1);
BPF_HASH(pending_exec, u32, u64);

static __always_inline void clear_event(struct event_t *event) {{
    event->timestamp_ns = 0;
    event->cgroup_id = 0;
    event->exec_seq = 0;
    event->utime_ns = 0;
    event->stime_ns = 0;
    event->hiwater_pages = 0;
    event->type = 0;
    event->host_pid = 0;
    event->host_tid = 0;
    event->parent_host_pid = 0;
    event->child_host_pid = 0;
    event->arg_index = 0;
    event->arg_length = 0;
    event->flags = 0;
    event->exit_code = 0;
    event->reserved = 0;
    event->arg[0] = 0;
}}

static void lost(void) {{
    u32 zero = 0;
    u64 *count = reserve_failures.lookup(&zero);
    if (count) __sync_fetch_and_add(count, 1);
}}

static int wanted(void) {{
    u32 zero = 0;
    u64 *target = target_cgroup.lookup(&zero);
    return target && *target && *target == bpf_get_current_cgroup_id();
}}

static u32 parent_tgid(void) {{
    struct task_struct *task = (struct task_struct *)bpf_get_current_task();
    struct task_struct *parent = 0;
    u32 tgid = 0;
    bpf_probe_read_kernel(&parent, sizeof(parent), &task->real_parent);
    if (parent) bpf_probe_read_kernel(&tgid, sizeof(tgid), &parent->tgid);
    return tgid;
}}

static int capture_exec(const char *const *argv) {{
    u32 zero = 0;
    u32 *ready = sequence_ready.lookup(&zero);
    if (!ready || !*ready) return 0;
    u64 seq = 0;
    if (exec_sequences.pop(&seq)) {{
        u64 *failures = sequence_failures.lookup(&zero);
        if (failures) __sync_fetch_and_add(failures, 1);
        return 0;
    }}
    if (!wanted()) return 0;

    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tid = pid_tgid;
    u32 tgid = pid_tgid >> 32;
    pending_exec.update(&tid, &seq);
    u32 argc = 0;
    u32 flags = 0;

#pragma unroll
    for (int i = 0; i < MAX_ARGS; i++) {{
        const char *argument = 0;
        bpf_probe_read_user(&argument, sizeof(argument), &argv[i]);
        if (!argument) break;
        struct event_t *event = events.ringbuf_reserve(sizeof(*event));
        if (!event) {{
            lost();
            continue;
        }}
        clear_event(event);
        event->timestamp_ns = bpf_ktime_get_ns();
        event->cgroup_id = bpf_get_current_cgroup_id();
        event->exec_seq = seq;
        event->type = TYPE_EXEC_ARG;
        event->host_pid = tgid;
        event->host_tid = tid;
        event->parent_host_pid = parent_tgid();
        event->arg_index = i;
        int length = bpf_probe_read_user_str(event->arg, sizeof(event->arg), argument);
        event->arg_length = length > 0 ? length : 0;
        if (length == ARG_BYTES) event->flags |= FLAG_ARG_TRUNCATED;
        events.ringbuf_submit(event, 0);
        argc = i + 1;
    }}
    if (argc == MAX_ARGS) {{
        const char *overflow = 0;
        bpf_probe_read_user(&overflow, sizeof(overflow), &argv[MAX_ARGS]);
        if (overflow) flags |= FLAG_ARG_COUNT_TRUNCATED;
    }}

    struct event_t *end = events.ringbuf_reserve(sizeof(*end));
    if (!end) {{
        lost();
        return 0;
    }}
    clear_event(end);
    end->timestamp_ns = bpf_ktime_get_ns();
    end->cgroup_id = bpf_get_current_cgroup_id();
    end->exec_seq = seq;
    end->type = TYPE_EXEC_END;
    end->host_pid = tgid;
    end->host_tid = tid;
    end->parent_host_pid = parent_tgid();
    end->arg_index = argc;
    end->flags = flags;
    events.ringbuf_submit(end, 0);
    return 0;
}}

TRACEPOINT_PROBE(syscalls, sys_enter_execve) {{
    return capture_exec((const char *const *)args->argv);
}}

TRACEPOINT_PROBE(syscalls, sys_enter_execveat) {{
    return capture_exec((const char *const *)args->argv);
}}

TRACEPOINT_PROBE(sched, sched_process_exec) {{
    if (!wanted()) return 0;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tid = pid_tgid;
    u64 seq = ~0ULL;
    u64 *pending = pending_exec.lookup(&tid);
    u32 flags = 0;
    if (pending) seq = *pending;
    else flags = FLAG_MISSING_ENTER;
    struct event_t *event = events.ringbuf_reserve(sizeof(*event));
    if (!event) {{
        lost();
        return 0;
    }}
    clear_event(event);
    event->timestamp_ns = bpf_ktime_get_ns();
    event->cgroup_id = bpf_get_current_cgroup_id();
    event->exec_seq = seq;
    event->type = TYPE_EXEC;
    event->host_pid = pid_tgid >> 32;
    event->host_tid = tid;
    event->parent_host_pid = parent_tgid();
    event->flags = flags;
    events.ringbuf_submit(event, 0);
    return 0;
}}

TRACEPOINT_PROBE(sched, sched_process_fork) {{
    if (!wanted()) return 0;
    struct event_t *event = events.ringbuf_reserve(sizeof(*event));
    if (!event) {{
        lost();
        return 0;
    }}
    clear_event(event);
    event->timestamp_ns = bpf_ktime_get_ns();
    event->cgroup_id = bpf_get_current_cgroup_id();
    event->type = TYPE_FORK;
    event->host_pid = args->parent_pid;
    event->child_host_pid = args->child_pid;
    events.ringbuf_submit(event, 0);
    return 0;
}}

TRACEPOINT_PROBE(sched, sched_process_exit) {{
    if (!wanted()) return 0;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tid = pid_tgid;
    u32 tgid = pid_tgid >> 32;
    if (tid != tgid) return 0;
    struct task_struct *task = (struct task_struct *)bpf_get_current_task();
    struct signal_struct *signal = 0;
    u64 task_utime = 0, task_stime = 0, group_utime = 0, group_stime = 0;
    u32 exit_code = 0;
    bpf_probe_read_kernel(&task_utime, sizeof(task_utime), &task->utime);
    bpf_probe_read_kernel(&task_stime, sizeof(task_stime), &task->stime);
    bpf_probe_read_kernel(&signal, sizeof(signal), &task->signal);
    bpf_probe_read_kernel(&exit_code, sizeof(exit_code), &task->exit_code);
    if (signal) {{
        bpf_probe_read_kernel(&group_utime, sizeof(group_utime), &signal->utime);
        bpf_probe_read_kernel(&group_stime, sizeof(group_stime), &signal->stime);
    }}
    u64 seq = ~0ULL;
    u64 *pending = pending_exec.lookup(&tid);
    if (pending) seq = *pending;
    struct event_t *event = events.ringbuf_reserve(sizeof(*event));
    if (!event) {{
        lost();
        return 0;
    }}
    clear_event(event);
    event->timestamp_ns = bpf_ktime_get_ns();
    event->cgroup_id = bpf_get_current_cgroup_id();
    event->exec_seq = seq;
    event->utime_ns = task_utime + group_utime;
    event->stime_ns = task_stime + group_stime;
    event->type = TYPE_EXIT;
    event->host_pid = tgid;
    event->host_tid = tid;
    event->parent_host_pid = parent_tgid();
    event->exit_code = exit_code;
    events.ringbuf_submit(event, 0);
    pending_exec.delete(&tid);
    return 0;
}}

int read_hiwater(struct pt_regs *ctx, struct task_struct *task) {{
    if (!wanted()) return 0;
    u32 pid = 0, tgid = 0;
    bpf_probe_read_kernel(&pid, sizeof(pid), &task->pid);
    bpf_probe_read_kernel(&tgid, sizeof(tgid), &task->tgid);
    if (pid != tgid) return 0;
    struct mm_struct *mm = 0;
    unsigned long pages = 0;
    bpf_probe_read_kernel(&mm, sizeof(mm), &task->mm);
    if (!mm) return 0;
    bpf_probe_read_kernel(&pages, sizeof(pages), &mm->hiwater_rss);
    struct event_t *event = events.ringbuf_reserve(sizeof(*event));
    if (!event) {{
        lost();
        return 0;
    }}
    clear_event(event);
    event->timestamp_ns = bpf_ktime_get_ns();
    event->cgroup_id = bpf_get_current_cgroup_id();
    event->type = TYPE_HIWATER;
    event->host_pid = tgid;
    event->host_tid = pid;
    event->hiwater_pages = pages;
    events.ringbuf_submit(event, 0);
    return 0;
}}
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _docker_command(
    payload: dict[str, Any], name: str, cidfile: Path
) -> list[str]:
    case = payload["case"]
    args = case.get("args")
    if args is None:
        args = ["-c", case["command"]]
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--cidfile",
        str(cidfile),
        "--entrypoint",
        case["entrypoint"],
        payload["image"],
        *args,
    ]


def _raw_event(table: Any, data: int) -> dict[str, Any]:
    event = table.event(data)
    raw = {
        "type": TYPE_NAMES[event.type],
        "timestamp_ns": int(event.timestamp_ns),
        "cgroup_id": int(event.cgroup_id),
        "exec_seq": int(event.exec_seq),
        "host_pid": int(event.host_pid),
        "host_tid": int(event.host_tid),
        "parent_host_pid": int(event.parent_host_pid),
        "child_host_pid": int(event.child_host_pid),
        "arg_index": int(event.arg_index),
        "arg_length": int(event.arg_length),
        "flags": int(event.flags),
        "exit_code": int(event.exit_code),
        "utime_ns": int(event.utime_ns),
        "stime_ns": int(event.stime_ns),
        "hiwater_pages": int(event.hiwater_pages),
    }
    if event.type == 1:
        raw["arg"] = bytes(event.arg).split(b"\0", 1)[0].decode(
            "utf-8", errors="surrogateescape"
        )
    return raw


def _process_records(events: list[dict[str, Any]]) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]]
]:
    attempts: dict[tuple[int, int], dict[str, Any]] = {}
    for event in events:
        if event["type"] not in {"exec_arg", "exec_argv_end"}:
            continue
        key = (event["host_pid"], event["exec_seq"])
        attempt = attempts.setdefault(
            key,
            {
                "host_pid": event["host_pid"],
                "exec_seq": event["exec_seq"],
                "args": {},
                "end": None,
            },
        )
        if event["type"] == "exec_arg":
            attempt["args"][event["arg_index"]] = event["arg"]
            attempt["arg_flags"] = attempt.get("arg_flags", 0) | event["flags"]
        else:
            attempt["end"] = event

    exec_events = []
    for event in events:
        if event["type"] != "exec":
            continue
        key = (event["host_pid"], event["exec_seq"])
        attempt = attempts.get(key, {"args": {}, "end": None, "arg_flags": 0})
        end = attempt.get("end")
        argc = None if end is None else end["arg_index"]
        argv = [attempt["args"][index] for index in sorted(attempt["args"])]
        exec_events.append(
            {
                "host_pid": event["host_pid"],
                "host_tid": event["host_tid"],
                "parent_host_pid": event["parent_host_pid"],
                "exec_seq": event["exec_seq"],
                "t_exec_ns": event["timestamp_ns"],
                "argv": argv,
                "argv_expected_count": argc,
                "argv_complete": argc is not None and len(argv) == argc,
                "argv_truncated": bool(
                    attempt.get("arg_flags", 0)
                    or (0 if end is None else end["flags"])
                ),
                "missing_enter": bool(event["flags"] & 4),
            }
        )

    exits = {
        event["host_pid"]: event for event in events if event["type"] == "exit"
    }
    peaks: dict[int, int] = {}
    for event in events:
        if event["type"] == "hiwater":
            peaks[event["host_pid"]] = max(
                peaks.get(event["host_pid"], 0), event["hiwater_pages"] * 4
            )
    parents = {
        event["child_host_pid"]: event["host_pid"]
        for event in events
        if event["type"] == "fork"
    }

    records = []
    by_pid: dict[int, list[dict[str, Any]]] = {}
    for event in exec_events:
        by_pid.setdefault(event["host_pid"], []).append(event)
    for pid, process_execs in by_pid.items():
        process_execs.sort(key=lambda row: row["t_exec_ns"])
        exit_event = exits.get(pid)
        for index, event in enumerate(process_execs):
            terminal = index == len(process_execs) - 1
            t_exit = (
                exit_event["timestamp_ns"]
                if terminal and exit_event is not None
                else None
            )
            records.append(
                {
                    **event,
                    "host_exec_index": index,
                    "terminal_image": terminal,
                    "lineage_parent_host_pid": parents.get(
                        pid, event["parent_host_pid"]
                    ),
                    "t_exit_ns": t_exit,
                    "wall_ns": (
                        None if t_exit is None else t_exit - event["t_exec_ns"]
                    ),
                    "utime_ns": (
                        exit_event["utime_ns"]
                        if terminal and exit_event is not None
                        else 0
                    ),
                    "stime_ns": (
                        exit_event["stime_ns"]
                        if terminal and exit_event is not None
                        else 0
                    ),
                    "cpu_ns": (
                        exit_event["utime_ns"] + exit_event["stime_ns"]
                        if terminal and exit_event is not None
                        else 0
                    ),
                    "exit_code_raw": (
                        exit_event["exit_code"]
                        if terminal and exit_event is not None
                        else None
                    ),
                    "signal": (
                        exit_event["exit_code"] & 0x7F
                        if terminal and exit_event is not None
                        else None
                    ),
                    "peak_rss_kb": peaks.get(pid),
                }
            )
    return exec_events, records


def collect(payload: dict[str, Any], name: str, cidfile: Path) -> dict[str, Any]:
    actual_image_id = subprocess.check_output(
        ["docker", "image", "inspect", payload["image"], "--format", "{{.Id}}"],
        text=True,
    ).strip()
    if actual_image_id != payload["image_id"]:
        raise RuntimeError(
            f"image id mismatch: {actual_image_id} != {payload['image_id']}"
        )

    bpf = BPF(text=BPF_PROGRAM)
    bpf.attach_kprobe(event="taskstats_exit", fn_name="read_hiwater")
    sequence_queue = bpf["exec_sequences"]
    for sequence in range(65536):
        sequence_queue.push(ctypes.c_ulonglong(sequence))
    bpf["sequence_ready"][ctypes.c_int(0)] = ctypes.c_uint(1)
    table = bpf["events"]
    events: list[dict[str, Any]] = []
    events_lock = threading.Lock()

    def receive(ctx: int, data: int, size: int) -> int:
        with events_lock:
            events.append(_raw_event(table, data))
        return 0

    table.open_ring_buffer(receive)
    attached_ns = time.monotonic_ns()
    attached_utc = _utc_now()
    stop_poll = threading.Event()

    def poll() -> None:
        while not stop_poll.is_set():
            bpf.ring_buffer_poll(timeout=25)

    poller = threading.Thread(target=poll)
    poller.start()

    cidfile.unlink(missing_ok=True)
    monitor_stop = threading.Event()
    monitor: dict[str, Any] = {
        "cgroup_id": None,
        "cgroup_path": None,
        "cgroup_set_monotonic_ns": None,
        "cpu_usage_usec": 0,
        "live_vmhwm_kb": 0,
        "live_vmhwm_pid": None,
        "error": None,
    }

    def set_and_monitor_cgroup() -> None:
        try:
            deadline = time.monotonic() + 10
            container_id = ""
            while time.monotonic() < deadline:
                if cidfile.exists():
                    container_id = cidfile.read_text(encoding="utf-8").strip()
                    if len(container_id) == 64:
                        break
                time.sleep(0.0005)
            if len(container_id) != 64:
                raise RuntimeError("container id did not appear")
            path = Path(
                f"/sys/fs/cgroup/system.slice/docker-{container_id}.scope"
            )
            while time.monotonic() < deadline and not path.exists():
                time.sleep(0.0005)
            cgroup_id = path.stat().st_ino
            bpf["target_cgroup"][ctypes.c_int(0)] = ctypes.c_ulonglong(cgroup_id)
            monitor.update(
                cgroup_id=cgroup_id,
                cgroup_path=str(path),
                cgroup_set_monotonic_ns=time.monotonic_ns(),
                container_id=container_id,
            )
            while not monitor_stop.is_set() and path.exists():
                try:
                    cpu = {
                        key: int(value)
                        for key, value in (
                            line.split()
                            for line in (path / "cpu.stat")
                            .read_text(encoding="utf-8")
                            .splitlines()
                        )
                    }
                    monitor["cpu_usage_usec"] = max(
                        monitor["cpu_usage_usec"], cpu["usage_usec"]
                    )
                    pids = [
                        int(value)
                        for value in (path / "cgroup.procs")
                        .read_text(encoding="utf-8")
                        .split()
                    ]
                except OSError:
                    continue
                for pid in pids:
                    try:
                        status = Path(f"/proc/{pid}/status").read_text(
                            encoding="utf-8"
                        )
                    except OSError:
                        continue
                    for line in status.splitlines():
                        if line.startswith("VmHWM:"):
                            hwm = int(line.split()[1])
                            if hwm > monitor["live_vmhwm_kb"]:
                                monitor["live_vmhwm_kb"] = hwm
                                monitor["live_vmhwm_pid"] = pid
                time.sleep(0.001)
        except Exception as error:
            monitor["error"] = repr(error)

    monitor_thread = threading.Thread(target=set_and_monitor_cgroup)
    monitor_thread.start()
    command = _docker_command(payload, name, cidfile)
    launch_ns = time.monotonic_ns()
    launch_utc = _utc_now()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    exit_ns = time.monotonic_ns()
    exit_utc = _utc_now()
    time.sleep(payload["collector"]["drain_after_exit_ms"] / 1000)
    drained_ns = time.monotonic_ns()
    monitor_stop.set()
    monitor_thread.join(timeout=2)
    stop_poll.set()
    poller.join(timeout=2)
    try:
        bpf.ring_buffer_consume()
    except Exception:
        pass

    reserve_failures = bpf["reserve_failures"][ctypes.c_int(0)].value
    sequence_failures = bpf["sequence_failures"][ctypes.c_int(0)].value
    cidfile.unlink(missing_ok=True)
    subprocess.run(
        ["docker", "rm", "-f", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    with events_lock:
        ordered_events = sorted(events, key=lambda row: row["timestamp_ns"])
    exec_events, records = _process_records(ordered_events)
    exited_pids = {
        event["host_pid"] for event in ordered_events if event["type"] == "exit"
    }
    integrity = {
        "reserve_failures": reserve_failures,
        "sequence_failures": sequence_failures,
        "successful_exec_count": len(exec_events),
        "exited_process_count": len(exited_pids),
        "exec_process_balanced": all(
            event["argv_complete"]
            and not event["missing_enter"]
            and event["host_pid"] in exited_pids
            for event in exec_events
        ),
        "argv_truncation_count": sum(
            event["argv_truncated"] for event in exec_events
        ),
        "attached_before_launch": attached_ns < launch_ns,
        "cgroup_setup_error": monitor["error"],
    }
    return {
        "schema_version": 1,
        "buffer": "ringbuf",
        "loss_counter": "bpf_reserve_failures",
        "docker_command": command,
        "stdout_hex": stdout.hex(),
        "stderr_hex": stderr.hex(),
        "exit_code": process.returncode,
        "elapsed_ns": exit_ns - launch_ns,
        "marker_present": payload["case"].get("live_marker", "").encode()
        in stdout,
        "timing": {
            "attached_monotonic_ns": attached_ns,
            "attached_utc": attached_utc,
            "launch_monotonic_ns": launch_ns,
            "launch_utc": launch_utc,
            "exit_monotonic_ns": exit_ns,
            "exit_utc": exit_utc,
            "drained_monotonic_ns": drained_ns,
        },
        "cgroup": monitor,
        "raw_events": ordered_events,
        "exec_events": exec_events,
        "process_records": records,
        "integrity": integrity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--rep", required=True, type=int)
    parser.add_argument("--name", required=True)
    parser.add_argument("--cidfile", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    if set(payload) != {"image", "image_id", "collector", "case"}:
        raise ValueError(f"invalid mechanism payload keys: {sorted(payload)}")
    result = collect(payload, args.name, args.cidfile)
    result.update(case=args.case, repetition=args.rep)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
