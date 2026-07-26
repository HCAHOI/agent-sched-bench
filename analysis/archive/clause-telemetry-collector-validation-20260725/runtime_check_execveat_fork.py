#!/usr/bin/python3
"""Root/BPF runtime check for two lifecycle-identity properties (A1, A3).

Replaces a source-substring assertion with a real attach-and-observe check:

A1 (execveat pending transition): a process that reaches its final image via the
    ``execveat(2)`` syscall (not ``execve``) still gets a NEW exec_seq and a
    captured argv for that image. If ``sys_enter_execveat`` were unwired, the
    execveat'd image would carry no fresh seq / argv.

A3 (TGID-safe fork lineage): when a NON-LEADER thread forks a child, the fork's
    recorded parent is the forking task's TGID (the process), not its TID. So the
    child's clause attributes up the lineage to the process, not to a phantom
    thread-id parent.

Run as root on a bcc-capable host:  sudo python3 runtime_check_execveat_fork.py
Exits 0 and prints PASS on success; raises with a diagnostic otherwise.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))

from trace_collect import clause_telemetry as C  # noqa: E402

# --- Scenario A1: reach /bin/true via execveat, not execve -------------------
# ctypes issues execveat(fd, "", argv, envp, AT_EMPTY_PATH) on an fd of /bin/true.
_EXECVEAT_CMD = (
    "python3 -c '"
    "import ctypes, os; "
    "libc = ctypes.CDLL(None, use_errno=True); "
    'fd = os.open("/bin/true", os.O_RDONLY); '
    'argv = (ctypes.c_char_p * 2)(b"/bin/true", None); '
    "envp = (ctypes.c_char_p * 1)(None); "
    'rc = libc.syscall(322, fd, b"", argv, envp, 0x1000); '
    "os._exit(0 if rc == 0 else 1)'"
)

# --- Scenario A3: a NON-LEADER thread forks a child that execs sleep ----------
# subprocess from a worker thread does the fork/exec in that thread (TID != TGID)
# via CPython's fork-safe path — a raw os.fork() in a thread would deadlock the
# child before it could exec.
_THREAD_FORK_CMD = (
    "python3 -c '"
    "import threading, subprocess; "
    'w = lambda: subprocess.run(["sleep", "0.2"]); '
    "t = threading.Thread(target=w); t.start(); t.join()'"
)


def check_execveat_new_seq() -> None:
    run = C.collect_case(_EXECVEAT_CMD, "rtchk_execveat")
    clauses, _ = C._clauses_and_lineage(run.events)
    # The /bin/true image reached via execveat must appear as its own clause with
    # a fresh seq and a captured argv head of "true".
    true_clauses = [c for c in clauses if c.bin == "true"]
    py_clauses = [c for c in clauses if c.bin == "python3"]
    assert py_clauses, f"no python3 clause observed; bins={[c.bin for c in clauses]}"
    assert true_clauses, (
        "execveat image ('true') got no exec_boundary/seq — sys_enter_execveat "
        f"transition dropped; bins={[c.bin for c in clauses]}"
    )
    t = true_clauses[0]
    assert t.argv and t.argv[0].endswith("true"), (
        f"execveat argv not captured: {t.argv!r}"
    )
    # A distinct seq from python3's own image on the same pid proves a real
    # pending->current transition (not a reused seq).
    same_pid_py = [c for c in py_clauses if c.host_pid == t.host_pid]
    assert same_pid_py and all(c.exec_seq != t.exec_seq for c in same_pid_py), (
        f"execveat image did not get a NEW seq (py seqs "
        f"{[c.exec_seq for c in same_pid_py]}, true seq {t.exec_seq})"
    )
    print(f"  A1 OK: execveat 'true' seq={t.exec_seq} argv={t.argv} "
          f"(python3 seqs {[c.exec_seq for c in same_pid_py]})")


def check_thread_fork_tgid_lineage() -> None:
    run = C.collect_case(_THREAD_FORK_CMD, "rtchk_threadfork")
    clauses, fork_parent = C._clauses_and_lineage(run.events)
    py = [c for c in clauses if c.bin == "python3"]
    sleep_c = [c for c in clauses if c.bin == "sleep"]
    assert py, f"no python3 clause; bins={[c.bin for c in clauses]}"
    assert sleep_c, (
        f"forked child never exec'd sleep; bins={[c.bin for c in clauses]}"
    )
    py_tgid = py[0].host_pid  # process TGID
    child_pid = sleep_c[0].host_pid
    # The child was forked by a NON-LEADER thread. TGID-safe lineage means the
    # recorded fork parent is the process TGID, reachable by walking fork_parent.
    cur, seen, reached = child_pid, set(), False
    while cur and cur not in seen:
        if cur == py_tgid:
            reached = True
            break
        seen.add(cur)
        cur = fork_parent.get(cur, 0)
    assert reached, (
        f"child pid {child_pid} does not attribute to python3 TGID {py_tgid} via "
        f"fork lineage {fork_parent} — fork recorded a thread TID, not the TGID"
    )
    print(f"  A3 OK: child {child_pid} -> ... -> python3 TGID {py_tgid} "
          f"via {fork_parent}")


def main() -> None:
    if os.geteuid() != 0:
        raise SystemExit("run as root (bcc attach required)")
    print("runtime check: execveat pending transition + TGID fork lineage")
    check_execveat_new_seq()
    check_thread_fork_tgid_lineage()
    print("PASS")


if __name__ == "__main__":
    main()
