#!/usr/bin/python3
"""Docker smoke phase 1 (root, system python3 + bcc): collect + analyze.

Runs ONE real command inside a Docker container under the Stage-2 collector,
filters by the container cgroup, analyzes into per-exec-image ClauseMetrics, and
dumps them (plus fork lineage + entry pid + counters) as JSON for the phase-2
bridge/KB step. No full replay. Usage:
    sudo python3 docker_smoke_collect.py --image IMG --command CMD --out FILE
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from bcc import BPF, PerfSWConfig, PerfType

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))

from trace_collect import clause_telemetry as C  # noqa: E402

_HERE = Path(__file__).resolve().parent


def _receive_row(table, data):  # noqa: ANN001
    e = table.event(data)
    row = {
        "type": C.TYPE_NAMES[int(e.type)],
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
    return row


def collect_docker(command: str, image: str) -> dict:
    cidfile = _HERE / f".smoke_cid_{os.getpid()}"
    bpf = BPF(text=C.BPF_PROGRAM)
    q = bpf["exec_sequences"]
    for seq in range(8192):
        q.push(ctypes.c_ulonglong(seq))
    bpf["sequence_ready"][ctypes.c_int(0)] = ctypes.c_uint(1)

    events: list[dict] = []
    lock = threading.Lock()
    table = bpf["events"]

    def receive(_ctx, data, _size):  # noqa: ANN001
        with lock:
            events.append(_receive_row(table, data))
        return 0

    table.open_ring_buffer(receive)
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            bpf.ring_buffer_poll(timeout=10)

    poller = threading.Thread(target=poll, daemon=True)
    poller.start()
    bpf.attach_perf_event(
        ev_type=PerfType.SOFTWARE,
        ev_config=PerfSWConfig.CPU_CLOCK,
        fn_name="on_cpu_clock",
        sample_period=C.SAMPLE_PERIOD_NS,
    )

    cg = {"id": None, "quota": None, "path": None, "error": None,
          "armed_after_launch_ns": None, "init_pid": None}
    launch_ns_box = {"ns": None}
    armed_event = threading.Event()

    def monitor():
        try:
            deadline = time.monotonic() + 15
            cid = ""
            while time.monotonic() < deadline:
                if cidfile.exists():
                    cid = cidfile.read_text().strip()
                    if len(cid) == 64:
                        break
                time.sleep(0.0005)
            if len(cid) != 64:
                raise RuntimeError("container id did not appear")
            path = Path(f"/sys/fs/cgroup/system.slice/docker-{cid}.scope")
            while time.monotonic() < deadline and not path.exists():
                time.sleep(0.0002)
            cgid = path.stat().st_ino
            bpf["target_cgroup"][ctypes.c_int(0)] = ctypes.c_ulonglong(cgid)
            raw = (path / "cpu.max").read_text().split()
            quota = (
                float(os.cpu_count())
                if raw[0] == "max"
                else float(int(raw[0]) / int(raw[1]))
            )
            cg.update(id=cgid, quota=quota, path=str(path))
            # authoritative entry (container init) host pid, so gate/entry-shell
            # execs are classified as structural, never as clause telemetry
            try:
                pid = subprocess.check_output(
                    ["docker", "inspect", cid, "--format", "{{.State.Pid}}"],
                    text=True,
                ).strip()
                cg["init_pid"] = int(pid) if pid.isdigit() else None
            except Exception:  # noqa: BLE001
                cg["init_pid"] = None
            if launch_ns_box["ns"] is not None:
                cg["armed_after_launch_ns"] = time.monotonic_ns() - launch_ns_box["ns"]
            armed_event.set()  # target cgroup is armed BEFORE the gate is released
        except Exception as error:  # noqa: BLE001
            cg["error"] = repr(error)

    mon = threading.Thread(target=monitor)
    mon.start()

    cidfile.unlink(missing_ok=True)
    # Start gate: the container blocks on `read` from stdin (a shell builtin —
    # no exec, off-CPU) until we release it. We arm the Stage-2 target cgroup
    # first, then release; the ORIGINAL command (passed via env, unchanged) is
    # exec'd only after arming, so no real-command exec can be missed.
    gate = 'read _; exec sh -c "$STAGE2_CMD"'
    cmd = ["docker", "run", "--rm", "-i", "--cidfile", str(cidfile),
           "-e", "STAGE2_CMD=" + command, image, "sh", "-c", gate]
    launch_ns_box["ns"] = time.monotonic_ns()
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if not armed_event.wait(timeout=20):
        proc.kill()
        raise RuntimeError(f"gate not armed: {cg['error']}")
    release_ns = time.monotonic_ns()
    proc.stdin.write(b"\n")  # release the gate
    proc.stdin.flush()
    proc.stdin.close()
    out = proc.stdout.read()
    proc.stderr.read()
    proc.wait()
    wall_ns = time.monotonic_ns() - release_ns  # real command wall, post-release
    time.sleep(0.25)
    mon.join(timeout=2)
    stop.set()
    poller.join(timeout=2)
    try:
        bpf.ring_buffer_consume()
    except Exception:
        pass
    reserve = bpf["reserve_failures"][ctypes.c_int(0)].value
    perf = bpf["perf_sample_count"][ctypes.c_int(0)].value
    bpf.detach_perf_event(ev_type=PerfType.SOFTWARE, ev_config=PerfSWConfig.CPU_CLOCK)
    bpf.cleanup()
    cidfile.unlink(missing_ok=True)

    with lock:
        ordered = sorted(events, key=lambda r: r["ts_ns"])
    ev = [e for e in ordered if e["cgroup_id"] == cg["id"]]

    run = C.RawRun(
        cgroup_id=cg["id"] or 0,
        quota_cores=cg["quota"] or 0.0,
        status=proc.returncode,
        wall_ns=wall_ns,
        usage_usec=0,
        reserve_failures=reserve,
        perf_sample_count=perf,
        oracle_peak_rss_kb=0,
        oracle_samples=0,
        marker=True,
        events=ev,
    )
    metrics, gaps = C.analyze(run)
    fork_parent = {
        e["child_host_pid"]: e["host_pid"]
        for e in ev
        if e["type"] == "fork" and e["child_host_pid"]
    }
    exec_boundaries = [e for e in ev if e["type"] == "exec_boundary"]
    # prefer the authoritative container-init pid; fall back to earliest exec
    entry_pid = cg.get("init_pid") or (
        min(exec_boundaries, key=lambda e: e["ts_ns"])["host_pid"]
        if exec_boundaries
        else None
    )
    # Classify coverage gaps. A gap is STRUCTURAL if it is container-startup
    # infrastructure (pid with no fork-lineage to the command tree) OR it is the
    # entry/gate shell BEFORE release (the gate phase). It is RELEVANT if it is a
    # command-tree process (descends from entry_pid), INCLUDING the entry shell
    # AFTER release — post-release entry activity is command execution, not gate
    # machinery, and must not be blanket-excused. (bpf ktime and monotonic_ns
    # are both CLOCK_MONOTONIC, so ts_ns and release_ns are comparable.)
    def _descends_from_entry(pid: int) -> bool:
        cur = pid
        seen: set[int] = set()
        while cur is not None and cur not in seen:
            if cur == entry_pid:
                return True
            seen.add(cur)
            cur = fork_parent.get(cur)
        return False

    def _relevant(g: dict) -> bool:
        pid, ts = g["host_pid"], g["ts_ns"]
        if not _descends_from_entry(pid):
            return False  # container infra outside the command tree
        if pid == entry_pid and ts < release_ns:
            return False  # gate phase (pre-release entry shell)
        return True

    relevant_gaps = [g for g in gaps if _relevant(g)]
    structural_gaps = [g for g in gaps if not _relevant(g)]
    return {
        "command": command,
        "image": image,
        "docker_exit_code": proc.returncode,
        "stdout_tail": out.decode("utf-8", "replace")[-400:],
        "wall_ns": wall_ns,
        "cgroup": cg,
        "reserve_failures": reserve,
        "perf_sample_count": perf,
        "raw_event_count": len(ev),
        "coverage_gap_samples": len(gaps),
        "structural_gap_samples": len(structural_gaps),
        "relevant_gap_samples": len(relevant_gaps),
        "relevant_gap_pids": sorted({g["host_pid"] for g in relevant_gaps}),
        "release_ns": release_ns,
        "entry_pid": entry_pid,
        "fork_parent": {str(k): v for k, v in fork_parent.items()},
        "clause_metrics": [dataclasses.asdict(m) for m in metrics],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--command", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if os.geteuid() != 0:
        raise SystemExit("run as root")
    result = collect_docker(args.command, args.image)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"collected {result['raw_event_count']} events, "
          f"{len(result['clause_metrics'])} exec images -> {args.out}")


if __name__ == "__main__":
    main()
