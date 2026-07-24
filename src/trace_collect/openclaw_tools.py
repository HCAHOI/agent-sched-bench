"""Execute tool calls inside Docker/Podman containers via a persistent agent."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import subprocess
import textwrap
from dataclasses import dataclass, field
from typing import Any

from trace_collect.pacct import _ACCT_V3_SIZE, parse_pacct_v3
from trace_collect.resource_timeline import valid_resource_timeline
from trace_collect.runtime.task_container import _CONTAINER_PYTHON_CANDIDATES

logger = logging.getLogger(__name__)

_OPENCLAW_EXEC_DEFAULT_TIMEOUT_S = 300.0
_OPENCLAW_EXEC_MAX_TIMEOUT_S = 600.0
_CONTAINER_LIST_DEFAULT_MAX = 200
# Outer guard for resource-aware exec requests. The in-container watchdog owns
# the modeled deadline; this only prevents an agent protocol deadlock from
# blocking simulate forever.
_RESOURCE_AWARE_AGENT_RESPONSE_TIMEOUT_S = 24 * 60 * 60.0
_AGENT_STOP_GRACE_S = 5.0
_AGENT_KILL_WAIT_S = 5.0
_PYTHON_PROBE_TIMEOUT_S = 30.0
_PYTHON_PROBE_KILL_WAIT_S = 5.0
_CONTAINER_PACCT_FILE = "/tmp/.openclaw-pacct.log"
_SOURCE_RUNTIME_ARTIFACT_MARKERS = (
    (
        "/openclaw-runtime/tool-results/tool-results/",
        "/openclaw-runtime/tool-results",
    ),
    (
        "/runtime/tool-results/tool-results/",
        "/runtime/tool-results",
    ),
)


@dataclass(slots=True)
class ContainerPacctSession:
    """Host-managed process accounting for one replay container."""

    container_id: str
    container_executable: str
    path: str = _CONTAINER_PACCT_FILE
    unavailable: bool = False
    unavailable_reason: str | None = None
    overlap_execs: int = 0
    _next_bracket: int = 0
    _active_brackets: set[int] = field(default_factory=set, repr=False)
    _invalid_brackets: set[int] = field(default_factory=set, repr=False)

    def enter_exec(self) -> tuple[int, bool]:
        """Reserve an attribution interval without serializing replay execution."""
        token = self._next_bracket
        self._next_bracket += 1
        collect = not self._active_brackets
        if not collect:
            newly_invalid = self._active_brackets - self._invalid_brackets
            self.overlap_execs += len(newly_invalid) + 1
            self._invalid_brackets.update(self._active_brackets)
            self._invalid_brackets.add(token)
        self._active_brackets.add(token)
        return token, collect

    def leave_exec(self, token: int) -> bool:
        """Return whether ``token`` remained isolated from other execs."""
        self._active_brackets.remove(token)
        valid = token not in self._invalid_brackets
        self._invalid_brackets.discard(token)
        return valid


def _run_container_pacct_python(
    session: ContainerPacctSession,
    script: str,
    *args: str,
    failure_reason: str,
) -> subprocess.CompletedProcess[str] | None:
    try:
        result = subprocess.run(
            [
                session.container_executable,
                "exec",
                "--user",
                "0",
                session.container_id,
                "python3",
                "-c",
                script,
                *args,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=_PYTHON_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        session.unavailable = True
        session.unavailable_reason = failure_reason
        return None
    if result.returncode == 0:
        return result
    stderr = result.stderr.lower()
    session.unavailable = True
    session.unavailable_reason = (
        "python3_unavailable"
        if result.returncode == 127 or ("python3" in stderr and "not found" in stderr)
        else failure_reason
    )
    return None


def enable_container_pacct(
    container_id: str,
    container_executable: str,
) -> ContainerPacctSession:
    """Enable acct(2) inside a replay container, failing soft when unavailable."""
    session = ContainerPacctSession(container_id, container_executable)
    script = (
        "import ctypes,sys;"
        "p=sys.argv[1];"
        "open(p,'wb').close();"
        "libc=ctypes.CDLL(None,use_errno=True);"
        "libc.acct.argtypes=[ctypes.c_char_p];"
        "libc.acct.restype=ctypes.c_int;"
        "rc=libc.acct(p.encode());"
        "rc==0 or (_ for _ in ()).throw(OSError(ctypes.get_errno(),'acct failed'))"
    )
    _run_container_pacct_python(
        session,
        script,
        session.path,
        failure_reason="acct_enable_failed",
    )
    return session


def container_pacct_begin(session: ContainerPacctSession) -> int | None:
    """Return the current acct-file byte offset for one exec bracket."""
    if session.unavailable:
        return None
    # The short-lived python3 stat process is itself appended after it prints,
    # so advance past that fixed-size record as well as the measured file end.
    result = _run_container_pacct_python(
        session,
        f"import os,sys;print(os.path.getsize(sys.argv[1])+{_ACCT_V3_SIZE})",
        session.path,
        failure_reason="acct_offset_read_failed",
    )
    if result is None:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        session.unavailable = True
        session.unavailable_reason = "acct_offset_invalid"
        return None


def decode_container_pacct_delta(payload: str) -> list[dict[str, Any]]:
    """Decode one base64 acct-file delta using the canonical host parser."""
    data = base64.b64decode(payload.encode("ascii"), validate=True)
    return [
        {**record.to_row(), "attribution": "offset_delta"}
        for record in parse_pacct_v3(data)
    ]


def container_pacct_finish(
    session: ContainerPacctSession,
    offset: int | None,
) -> list[dict[str, Any]] | None:
    """Read and decode records appended since ``offset`` without PID filtering."""
    if session.unavailable or offset is None:
        return None
    script = (
        "import base64,sys;"
        "f=open(sys.argv[1],'rb');"
        "f.seek(int(sys.argv[2]));"
        "sys.stdout.write(base64.b64encode(f.read()).decode('ascii'));"
        "f.close()"
    )
    result = _run_container_pacct_python(
        session,
        script,
        session.path,
        str(offset),
        failure_reason="acct_delta_read_failed",
    )
    if result is None:
        return None
    try:
        rows = decode_container_pacct_delta(result.stdout.strip())
    except (UnicodeEncodeError, ValueError):
        session.unavailable = True
        session.unavailable_reason = "acct_delta_invalid"
        return None
    # ponytail: descendants killed at timeout may exit after this read and be missed.
    return rows or None


def _unwrap_tool_args(
    *,
    tool_name: str | None,
    tool_args_json: str,
) -> tuple[str | None, dict[str, Any]]:
    """Return (resolved_tool_name, params)."""
    parsed = json.loads(tool_args_json or "{}")
    if not isinstance(parsed, dict):
        return tool_name, {}

    if tool_name and isinstance(parsed.get(tool_name), dict):
        return tool_name, parsed[tool_name]

    if len(parsed) == 1:
        only_name, only_value = next(iter(parsed.items()))
        if isinstance(only_value, dict):
            return (tool_name or only_name), only_value

    return tool_name, parsed


def _resolve_exec_timeout_s(
    params: dict[str, Any],
    *,
    default_timeout_s: float = _OPENCLAW_EXEC_DEFAULT_TIMEOUT_S,
) -> float:
    """Resolve replay exec timeout; source value wins, otherwise use simulate fallback."""
    return min(
        float(params.get("timeout", default_timeout_s)),
        _OPENCLAW_EXEC_MAX_TIMEOUT_S,
    )


def _is_source_runtime_artifact_path(path: str) -> bool:
    return any(
        marker in path for marker, _root_suffix in _SOURCE_RUNTIME_ARTIFACT_MARKERS
    )


def source_runtime_artifact_root_from_path(path: str) -> str | None:
    for marker, root_suffix in _SOURCE_RUNTIME_ARTIFACT_MARKERS:
        marker_index = path.find(marker)
        if marker_index >= 0:
            return path[:marker_index] + root_suffix
    return None


def source_runtime_artifact_path_from_tool_call(
    *,
    tool_name: str | None,
    tool_args_json: str,
) -> str | None:
    """Return an OpenClaw source-runtime artifact path referenced by a tool call.

    These paths point to files produced by the original collection run, not to
    files in the task repository image. A fresh replay container cannot execute
    them faithfully unless the source artifact tree is explicitly restored.
    """
    resolved_name, params = _unwrap_tool_args(
        tool_name=tool_name,
        tool_args_json=tool_args_json,
    )
    if resolved_name != "read_file":
        return None
    path = params.get("path")
    if isinstance(path, str) and _is_source_runtime_artifact_path(path):
        return path
    return None


def remap_source_runtime_artifact_tool_args(
    *,
    tool_name: str | None,
    tool_args_json: str,
    runtime_root_map: dict[str, str],
) -> tuple[str, str | None, str | None]:
    """Map a source OpenClaw artifact path into the simulator runtime tree."""
    resolved_name, params = _unwrap_tool_args(
        tool_name=tool_name,
        tool_args_json=tool_args_json,
    )
    if resolved_name != "read_file":
        return tool_args_json, None, None
    path = params.get("path")
    if not isinstance(path, str):
        return tool_args_json, None, None
    source_root = source_runtime_artifact_root_from_path(path)
    if source_root is None:
        return tool_args_json, None, None
    mapped_root = runtime_root_map.get(source_root)
    if mapped_root is None:
        return tool_args_json, path, None
    mapped_path = mapped_root + path[len(source_root) :]
    remapped = dict(params)
    remapped["path"] = mapped_path
    return json.dumps(remapped, ensure_ascii=False), path, mapped_path


# Persistent Python agent script injected into Docker containers; reads JSON-line
# requests from stdin and writes JSON-line responses to stdout.
_REPLAY_AGENT_SCRIPT = textwrap.dedent(r"""
import json, os, sys, subprocess, difflib, signal, time, re, shutil, tempfile, struct
WORKDIR = os.environ.get("OPENCLAW_CONTAINER_WORKDIR", "/testbed") or "/testbed"

def _find_match(content, old_text):
    if old_text in content:
        return old_text, content.count(old_text)
    old_lines = old_text.splitlines()
    if not old_lines:
        return None, 0
    stripped_old = [line.strip() for line in old_lines]
    content_lines = content.splitlines()
    candidates = []
    for i in range(len(content_lines) - len(stripped_old) + 1):
        window = content_lines[i : i + len(stripped_old)]
        if [line.strip() for line in window] == stripped_old:
            candidates.append("\n".join(window))
    if candidates:
        return candidates[0], len(candidates)
    return None, 0

def _not_found_msg(old_text, content, path):
    lines = content.splitlines(keepends=True)
    old_lines = old_text.splitlines(keepends=True)
    window = len(old_lines)
    best_ratio, best_start = 0.0, 0
    for i in range(max(1, len(lines) - window + 1)):
        ratio = difflib.SequenceMatcher(None, old_lines, lines[i:i+window]).ratio()
        if ratio > best_ratio:
            best_ratio, best_start = ratio, i
    if best_ratio > 0.5:
        diff = "\n".join(difflib.unified_diff(
            old_lines, lines[best_start:best_start+window],
            fromfile="old_text (provided)",
            tofile=f"{path} (actual, line {best_start+1})", lineterm=""))
        return f"Error: old_text not found in {path}.\nBest match ({best_ratio:.0%}) at line {best_start+1}:\n{diff}"
    return f"Error: old_text not found in {path}. No similar text found."

_MAX_OUTPUT = 10_000

def _truncate_output(text, limit=_MAX_OUTPUT):
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n\n... ({len(text) - limit} chars truncated) ...\n\n" + text[-half:]

# --- Per-binary process accounting (BSD acct v3) -------------------------
# When OPENCLAW_PACCT=1 the container enables kernel process accounting via
# acct(2) (needs --cap-add SYS_PACCT on the container). The kernel then appends
# one 64-byte record per process *exit*, so we attribute a compound command's
# CPU/memory to the individual binaries it ran (python vs head/wc). This inline
# decoder is a minimal twin of trace_collect/pacct.py (the canonical, tested
# copy) — the replay server is a standalone `python -c` with no repo import, so
# keep the two struct layouts in sync. AHZ is a fixed kernel constant of 100.
_PACCT_ENABLED = os.environ.get("OPENCLAW_PACCT") == "1"
_PACCT_FILE = "/tmp/.openclaw-pacct.log"
_PACCT_FMT = "<bbHIIIIIIfHHHHHHHH16s"
_PACCT_SIZE = struct.calcsize(_PACCT_FMT)  # 64

def _pacct_enable():
    global _PACCT_ENABLED
    if not _PACCT_ENABLED:
        return
    try:
        import ctypes, ctypes.util
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        open(_PACCT_FILE, "wb").close()
        if libc.acct(_PACCT_FILE.encode()) != 0:
            raise OSError(ctypes.get_errno(), "acct(2) failed")
    except Exception as exc:
        _PACCT_ENABLED = False
        sys.stderr.write("[pacct] disabled: %s\n" % exc)
        sys.stderr.flush()

def _pacct_decomp(v):
    return (v & 0x1FFF) << (3 * ((v >> 13) & 0x7))

def _pacct_parse(data):
    usable = len(data) - (len(data) % _PACCT_SIZE)
    rows = []
    for o in range(0, usable, _PACCT_SIZE):
        f = struct.unpack(_PACCT_FMT, data[o:o + _PACCT_SIZE])
        if f[1] != 3:
            continue
        rows.append({
            "comm": f[18].split(b"\0", 1)[0].decode("latin1"),
            "pid": f[6], "ppid": f[7], "exitcode": f[3],
            "utime_s": round(_pacct_decomp(f[10]) / 100.0, 3),
            "stime_s": round(_pacct_decomp(f[11]) / 100.0, 3),
            "avg_mem_kb": _pacct_decomp(f[12]),
        })
    return rows

def _pacct_filter_subtree(rows, root_pid):
    # Keep root_pid's subtree (itself + descendants) resolved within this exit
    # batch, so a lingering `&` job from an earlier command (its shell absent
    # here) is dropped. ponytail: fixpoint over one small per-exec batch.
    kept = {root_pid}
    changed = True
    while changed:
        changed = False
        for r in rows:
            if r["pid"] not in kept and r["ppid"] in kept:
                kept.add(r["pid"]); changed = True
    return [r for r in rows if r["pid"] in kept]

def _pacct_begin():
    # Byte offset of the current end-of-file: records before it belong to prior
    # (or between-exec) work and are skipped, not misattributed to this command.
    if not _PACCT_ENABLED:
        return None
    try:
        return os.path.getsize(_PACCT_FILE)
    except OSError:
        return None

def _pacct_finish(off0, root_pid=None):
    if off0 is None:
        return None
    try:
        with open(_PACCT_FILE, "rb") as fh:
            fh.seek(off0)
            data = fh.read()
    except OSError:
        return None
    rows = _pacct_parse(data)
    if root_pid is not None:
        rows = _pacct_filter_subtree(rows, root_pid)
    return rows or None

# --- Per-segment (atom) command timing telemetry (v2) --------------------
# Chained commands ("cd X && make && pytest") arrive as one exec call. When
# enabled we re-run the *unmodified* command under `bash -x -c` with xtrace
# redirected to a dedicated fd (BASH_XTRACEFD), so per-top-level-segment start
# timestamps land in a file the command itself never sees. PS4 embeds
# $EPOCHREALTIME; bash replicates PS4's first char per nesting level, so lines
# with exactly one leading '+' are the top-level segments. Best-effort: absent
# bash or the flag, the untouched /bin/sh path runs and telemetry is recorded
# as absent -- replay never fails because timing hiccupped.
# ponytail: nested `bash -x` inside a replayed command inherits PS4/BASH_XTRACEFD
# and could add spurious depth-1 lines; rare, caught at extraction via
# raw_total_ms reconciliation. Upgrade path: per-invocation trace-fd tagging.
_SEGMENT_TIMELINE_VERSION = 2
_SEGMENT_TIMELINE_ENABLED = os.environ.get("OPENCLAW_SEGMENT_TIMELINE") == "1"
_SEGMENT_BASH_PATH = shutil.which("bash") if _SEGMENT_TIMELINE_ENABLED else None
_SEGMENT_TRACE_DIR = os.environ.get("TMPDIR") or "/tmp"
_SEGMENT_LINE_RE = re.compile(r"^(\++)(\d+[.,]\d+)\s(.*)$")


def _segment_absent(reason):
    return {
        "version": _SEGMENT_TIMELINE_VERSION,
        "telemetry_absent": True,
        "reason": reason,
    }


def _segments_from_xtrace(text, start_wall, end_wall):
    # ponytail: sequential-operator ceiling -- pipeline members (a | b) both
    # emit depth-1 lines but run CONCURRENTLY, so their derived durations are
    # fictitious (first ~0ms, last gets the span); loop headers re-emit per
    # iteration. Durations are trustworthy only for && / ; / newline chains;
    # downstream analyses must filter on the parent command (see CLAUDE.md).
    events = []
    for line in text.splitlines():
        match = _SEGMENT_LINE_RE.match(line)
        if match is None or len(match.group(1)) != 1:
            continue
        events.append((float(match.group(2).replace(",", ".")), match.group(3)))
    if not events:
        return _segment_absent("no_segments_traced")
    segments = []
    for index, (epoch, command_text) in enumerate(events):
        t_start_ms = (epoch - start_wall) * 1000.0
        next_epoch = events[index + 1][0] if index + 1 < len(events) else end_wall
        t_end_ms = (next_epoch - start_wall) * 1000.0
        segments.append({
            "segment_index": index,
            "command_text": command_text,
            "t_start_ms": round(t_start_ms, 3),
            "t_end_ms": round(t_end_ms, 3),
        })
    return {
        "version": _SEGMENT_TIMELINE_VERSION,
        "source": "bash_xtrace_epochrealtime",
        "segments": segments,
        "segment_count": len(segments),
        "raw_total_ms": round((end_wall - start_wall) * 1000.0, 3),
    }


def _open_segment_trace():
    if not _SEGMENT_TIMELINE_ENABLED or not _SEGMENT_BASH_PATH:
        return None
    fd, path = tempfile.mkstemp(prefix=".openclaw-segtrace.", dir=_SEGMENT_TRACE_DIR)
    return (fd, path)


def _shell_launch(cmd, env, seg):
    # seg is None -> identical to the historical /bin/sh -c path (shell=True).
    if seg is None:
        return cmd, {"shell": True, "env": env}
    fd, _path = seg
    traced_env = dict(env)
    traced_env["BASH_XTRACEFD"] = str(fd)
    # PS4 must be a bash-internal assignment, not an inherited env var: on the
    # bash builds observed in real task containers, an env-inherited PS4 is
    # captured once at shell startup (before EPOCHREALTIME is live) and never
    # re-expanded per xtrace line, silently freezing every timestamp empty.
    # A PS4 assignment in the script body, before `set -x`, gets bash's normal
    # per-line re-expansion. Verified live in a real task container (bash
    # 5.2.15): env-set PS4 -> "+ cmd" (no timestamp); script-body PS4 ->
    # "+<epoch> cmd" (real timestamps).
    preamble = "PS4='+$EPOCHREALTIME '\nset -x\n"
    return [_SEGMENT_BASH_PATH, "-c", preamble + cmd], {
        "env": traced_env,
        "pass_fds": (fd,),
    }


def _finish_segment_trace(seg, start_wall, end_wall):
    fd, path = seg
    try:
        os.close(fd)
    except OSError:
        pass
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            result = _segments_from_xtrace(handle.read(), start_wall, end_wall)
    except OSError as exc:
        result = _segment_absent("trace_read_error: %s" % exc)
    try:
        os.unlink(path)
    except OSError:
        pass
    return result


_RESOURCE_CPU_RATE_EPS_CORE = 0.05
_RESOURCE_NET_RATE_EPS_BPS = 1024.0
_RESOURCE_PROGRESS_EPS_S = 1e-6
_RESOURCE_SAMPLE_INTERVAL_S = 0.5
_RESOURCE_STALL_MIN_S = 5.0
_RESOURCE_STALL_MAX_S = 60.0


def _nonnegative_float(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number < 0:
        return default
    return number


def _resource_source_samples(timeline):
    if not isinstance(timeline, dict) or timeline.get("version") != 1:
        return []
    raw_samples = timeline.get("samples")
    if not isinstance(raw_samples, list):
        return []
    samples = []
    offset_s = 0.0
    for raw in raw_samples:
        if not isinstance(raw, dict):
            continue
        dt_s = _nonnegative_float(raw.get("dt_s"))
        if dt_s <= 0:
            continue
        cpu_core_s = _nonnegative_float(raw.get("cpu_core_s"))
        rx_bytes = _nonnegative_float(raw.get("net_rx_bytes"))
        tx_bytes = _nonnegative_float(raw.get("net_tx_bytes"))
        sample = {
            "start_s": offset_s,
            "end_s": offset_s + dt_s,
            "cpu_rate_core": cpu_core_s / dt_s,
            "rx_rate_bps": rx_bytes / dt_s,
            "tx_rate_bps": tx_bytes / dt_s,
        }
        samples.append(sample)
        offset_s += dt_s
    return samples


def _resource_sample_at(samples, virtual_time_s):
    if not samples:
        return None
    for sample in samples:
        if sample["start_s"] <= virtual_time_s < sample["end_s"]:
            return sample
    return samples[-1]


def _read_cgroup_cpu_usage_s():
    try:
        with open("/sys/fs/cgroup/cpu.stat", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) == 2 and parts[0] == "usage_usec":
                    return int(parts[1]) / 1_000_000.0
    except Exception:
        return None
    return None


def _read_proc_net_bytes():
    rx_total = 0
    tx_total = 0
    found = False
    try:
        with open("/proc/net/dev", encoding="utf-8") as fh:
            lines = fh.readlines()[2:]
    except Exception:
        return None, None
    for line in lines:
        if ":" not in line:
            continue
        iface, rest = line.split(":", 1)
        if iface.strip() == "lo":
            continue
        fields = rest.split()
        if len(fields) < 16:
            continue
        try:
            rx_total += int(fields[0])
            tx_total += int(fields[8])
        except ValueError:
            continue
        found = True
    if not found:
        return None, None
    return rx_total, tx_total


def _read_resource_counters():
    rx_bytes, tx_bytes = _read_proc_net_bytes()
    return {
        "time_s": time.monotonic(),
        "cpu_usage_s": _read_cgroup_cpu_usage_s(),
        "rx_bytes": rx_bytes,
        "tx_bytes": tx_bytes,
    }


def _counter_delta(previous, current, key):
    left = previous.get(key)
    right = current.get(key)
    if left is None or right is None:
        return None
    return max(0.0, float(right) - float(left))


def _resource_progress_increment(samples, virtual_time_s, wall_dt_s, deltas):
    if wall_dt_s <= 0:
        return 0.0
    if not samples:
        return wall_dt_s
    source_end_s = samples[-1]["end_s"]
    if virtual_time_s >= source_end_s:
        return wall_dt_s
    sample = _resource_sample_at(samples, virtual_time_s)
    if sample is None:
        return wall_dt_s
    candidates = []
    cpu_rate = sample["cpu_rate_core"]
    if cpu_rate >= _RESOURCE_CPU_RATE_EPS_CORE and deltas.get("cpu_core_s") is not None:
        candidates.append(float(deltas["cpu_core_s"]) / cpu_rate)
    rx_rate = sample["rx_rate_bps"]
    if rx_rate >= _RESOURCE_NET_RATE_EPS_BPS and deltas.get("rx_bytes") is not None:
        candidates.append(float(deltas["rx_bytes"]) / rx_rate)
    tx_rate = sample["tx_rate_bps"]
    if tx_rate >= _RESOURCE_NET_RATE_EPS_BPS and deltas.get("tx_bytes") is not None:
        candidates.append(float(deltas["tx_bytes"]) / tx_rate)
    if not candidates:
        progress_s = wall_dt_s
    else:
        progress_s = max(0.0, min(candidates))
    return min(progress_s, max(0.0, sample["end_s"] - virtual_time_s))


def _resource_has_active_demand(samples, virtual_time_s):
    sample = _resource_sample_at(samples, virtual_time_s)
    if sample is None:
        return False
    return (
        sample["cpu_rate_core"] >= _RESOURCE_CPU_RATE_EPS_CORE
        or sample["rx_rate_bps"] >= _RESOURCE_NET_RATE_EPS_BPS
        or sample["tx_rate_bps"] >= _RESOURCE_NET_RATE_EPS_BPS
    )


def _kill_process_group(process):
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        else:
            process.kill()
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def _run_shell_command_with_resource_timeout(cmd, timeout, env, source_resource_timeline):
    samples = _resource_source_samples(source_resource_timeline)
    if not samples:
        return None
    timeout_s = float(timeout)
    stall_timeout_s = max(
        _RESOURCE_STALL_MIN_S,
        min(_RESOURCE_STALL_MAX_S, timeout_s),
    )
    start_new_session = hasattr(os, "setsid")
    seg = _open_segment_trace()

    def _finalize(resp, start_wall):
        if seg is not None:
            resp["segment_timeline"] = _finish_segment_trace(
                seg, start_wall, time.time()
            )
        per_process = _pacct_finish(pacct_off0, process.pid)
        if per_process is not None:
            resp["per_process"] = per_process
        return resp

    try:
        launch_args, launch_kwargs = _shell_launch(cmd, env, seg)
        start_wall = time.time()
        pacct_off0 = _pacct_begin()
        process = subprocess.Popen(
            launch_args,
            cwd=WORKDIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            start_new_session=start_new_session,
            **launch_kwargs,
        )
        virtual_time_s = 0.0
        last_counters = _read_resource_counters()
        last_progress_wall_s = last_counters["time_s"]
        while True:
            try:
                stdout, stderr = process.communicate(timeout=_RESOURCE_SAMPLE_INTERVAL_S)
                output = (stdout or "") + (stderr or "")
                resp = {
                    "ok": True,
                    "result": _truncate_output(output),
                    "returncode": process.returncode,
                    "resource_timeout_policy": "resource_integrated",
                    "resource_virtual_time_s": round(virtual_time_s, 6),
                }
                _finalize(resp, start_wall)
                seg = None
                return resp
            except subprocess.TimeoutExpired:
                current_counters = _read_resource_counters()
                wall_dt_s = max(0.0, current_counters["time_s"] - last_counters["time_s"])
                deltas = {
                    "cpu_core_s": _counter_delta(last_counters, current_counters, "cpu_usage_s"),
                    "rx_bytes": _counter_delta(last_counters, current_counters, "rx_bytes"),
                    "tx_bytes": _counter_delta(last_counters, current_counters, "tx_bytes"),
                }
                progress_s = _resource_progress_increment(
                    samples,
                    virtual_time_s,
                    wall_dt_s,
                    deltas,
                )
                virtual_time_s += progress_s
                if progress_s > _RESOURCE_PROGRESS_EPS_S:
                    last_progress_wall_s = current_counters["time_s"]
                if virtual_time_s >= timeout_s:
                    _kill_process_group(process)
                    stdout, stderr = process.communicate()
                    output = (stdout or "") + (stderr or "")
                    if output:
                        output = _truncate_output(output) + "\n[resource_timeout]"
                    else:
                        output = "[resource_timeout]"
                    resp = {
                        "ok": False,
                        "result": output,
                        "returncode": 124,
                        "resource_timeout_policy": "resource_integrated",
                        "resource_virtual_time_s": round(virtual_time_s, 6),
                    }
                    _finalize(resp, start_wall)
                    seg = None
                    return resp
                stalled_s = current_counters["time_s"] - last_progress_wall_s
                if stalled_s >= stall_timeout_s and _resource_has_active_demand(
                    samples,
                    virtual_time_s,
                ):
                    _kill_process_group(process)
                    stdout, stderr = process.communicate()
                    output = (stdout or "") + (stderr or "")
                    marker = "[resource_stall_timeout]"
                    if output:
                        output = _truncate_output(output) + "\n" + marker
                    else:
                        output = marker
                    resp = {
                        "ok": False,
                        "result": output,
                        "returncode": 124,
                        "resource_timeout_policy": "resource_integrated",
                        "resource_virtual_time_s": round(virtual_time_s, 6),
                        "resource_stall_s": round(stalled_s, 6),
                    }
                    _finalize(resp, start_wall)
                    seg = None
                    return resp
                last_counters = current_counters
    finally:
        if seg is not None:
            _finish_segment_trace(seg, 0.0, 0.0)


def handle_exec(args):
    cmd = args.get("command", "")
    timeout = args.get("timeout", 600)
    env = {**os.environ, "PAGER": "cat", "MANPAGER": "cat", "LESS": "-R"}
    resource_response = _run_shell_command_with_resource_timeout(
        cmd,
        timeout,
        env,
        args.get("source_resource_timeline"),
    )
    if resource_response is not None:
        return resource_response
    seg = _open_segment_trace()
    try:
        launch_args, launch_kwargs = _shell_launch(cmd, env, seg)
        start_wall = time.time()
        pacct_off0 = _pacct_begin()
        try:
            r = subprocess.run(launch_args, cwd=WORKDIR,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True, timeout=timeout, **launch_kwargs)
            end_wall = time.time()
            output = (r.stdout or "") + (r.stderr or "")
            resp = {"ok": True, "result": _truncate_output(output), "returncode": r.returncode}
        except subprocess.TimeoutExpired:
            end_wall = time.time()
            resp = {"ok": False, "result": "[timeout]", "returncode": 124}
        # subprocess.run hides the child pid, so this fallback path (only taken
        # when the source trace has no resource_timeline) attributes by exit
        # window without ppid filtering. ponytail: a lingering `&` job from a
        # prior command could leak here; the resource-integrated path above does
        # the full subtree filter.
        # SIGKILLed descendants may exit after this acct read and be missed.
        per_process = _pacct_finish(pacct_off0)
        if per_process is not None:
            resp["per_process"] = per_process
        if seg is not None:
            resp["segment_timeline"] = _finish_segment_trace(seg, start_wall, end_wall)
            seg = None
        return resp
    finally:
        if seg is not None:
            _finish_segment_trace(seg, 0.0, 0.0)

def handle_commands(args):
    cmds = args.get("commands", [])
    timeout = args.get("timeout", 600)
    env = {**os.environ, "PAGER": "cat", "MANPAGER": "cat", "LESS": "-R"}
    all_output = []
    last_rc = 0
    first_failed_rc = 0
    any_timeout = False
    for i, cmd in enumerate(cmds):
        try:
            r = subprocess.run(cmd, shell=True, cwd=WORKDIR,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True, timeout=timeout, env=env)
            all_output.append((r.stdout or "") + (r.stderr or ""))
            last_rc = r.returncode
            if r.returncode != 0 and first_failed_rc == 0:
                first_failed_rc = r.returncode
        except subprocess.TimeoutExpired:
            all_output.append("[timeout]")
            last_rc = 124
            any_timeout = True
    if len(cmds) > 1:
        combined = "\n".join(f"[call {k}]\n{out}" for k, out in enumerate(all_output))
    else:
        combined = all_output[0] if all_output else ""
    returncode = 124 if any_timeout else (first_failed_rc or last_rc)
    return {"ok": not any_timeout, "result": combined, "returncode": returncode}

_READ_MAX_CHARS = 128_000
_READ_DEFAULT_LIMIT = 2000

def handle_read_file(args):
    path = args.get("path", "")
    offset = int(args.get("offset", 0))
    limit = int(args.get("limit", _READ_DEFAULT_LIMIT))
    try:
        content = open(path).read()
        if not content:
            return {"ok": True, "result": f"(Empty file: {path})"}
        lines = content.splitlines()
        selected = lines[offset:offset + limit]
        numbered = "\n".join(f"{offset + i + 1}| {ln}" for i, ln in enumerate(selected))
        if len(numbered) > _READ_MAX_CHARS:
            numbered = numbered[:_READ_MAX_CHARS] + f"\n\n... (truncated at {_READ_MAX_CHARS} chars)"
        return {"ok": True, "result": numbered}
    except Exception as e:
        return {"ok": False, "result": f"Error: {e}"}

def handle_write_file(args):
    path = args.get("path", "")
    content = args.get("content", "")
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return {"ok": True, "result": f"Successfully wrote {path}"}
    except Exception as e:
        return {"ok": False, "result": f"Error: {e}"}

def handle_edit_file(args):
    path = args.get("path", "")
    old_text = args.get("old_text", "")
    new_text = args.get("new_text", "")
    replace_all = args.get("replace_all", False)
    try:
        raw = open(path, "rb").read()
        uses_crlf = b"\r\n" in raw
        content = raw.decode("utf-8").replace("\r\n", "\n")
        match, count = _find_match(content, old_text.replace("\r\n", "\n"))
        if match is None:
            return {"ok": False, "result": _not_found_msg(old_text, content, path)}
        if count > 1 and not replace_all:
            return {"ok": False, "result": f"Warning: old_text appears {count} times. Provide more context or set replace_all=true."}
        norm_new = new_text.replace("\r\n", "\n")
        new_content = content.replace(match, norm_new) if replace_all else content.replace(match, norm_new, 1)
        if uses_crlf:
            new_content = new_content.replace("\n", "\r\n")
        open(path, "wb").write(new_content.encode("utf-8"))
        return {"ok": True, "result": f"Successfully edited {path}"}
    except Exception as e:
        return {"ok": False, "result": f"Error editing file: {e}"}

_LIST_IGNORE = {".git", "node_modules", "__pycache__", ".venv", ".tox", ".mypy_cache", ".pytest_cache"}
_LIST_MAX = 200

def handle_list_dir(args):
    path = args.get("path", ".")
    recursive = bool(args.get("recursive", False))
    max_entries = int(args.get("max_entries") or _LIST_MAX)
    try:
        if not os.path.exists(path):
            return {"ok": False, "result": f"Error: Directory not found: {path}"}
        if not os.path.isdir(path):
            return {"ok": False, "result": f"Error: Not a directory: {path}"}
        items = []
        total = 0
        if recursive:
            for root, dirs, files in os.walk(path):
                dirs[:] = sorted(d for d in dirs if d not in _LIST_IGNORE)
                for name in sorted(dirs) + sorted(files):
                    if name in _LIST_IGNORE:
                        continue
                    full_path = os.path.join(root, name)
                    total += 1
                    if len(items) < max_entries:
                        rel = os.path.relpath(full_path, path)
                        items.append(f"{rel}/" if os.path.isdir(full_path) else rel)
        else:
            for name in sorted(e for e in os.listdir(path) if e not in _LIST_IGNORE):
                full_path = os.path.join(path, name)
                total += 1
                if len(items) < max_entries:
                    prefix = "📁 " if os.path.isdir(full_path) else "📄 "
                    items.append(f"{prefix}{name}")
        if not items and total == 0:
            return {"ok": True, "result": f"Directory {path} is empty"}
        result = "\n".join(items)
        if total > max_entries:
            result += f"\n\n(truncated, showing first {max_entries} of {total} entries)"
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "result": f"Error: {e}"}

def handle_extract_patch(args):
    base_commit = args.get("base_commit") or "HEAD"
    exclude_pathspecs = args.get("exclude_pathspecs") or []
    if not isinstance(exclude_pathspecs, list):
        return {"ok": False, "result": "Error: exclude_pathspecs must be a list", "returncode": 2}
    patch_path = "patch.txt"
    try:
        if os.path.exists(patch_path):
            with open(patch_path, encoding="utf-8") as fh:
                text = fh.read().strip()
            if text.lstrip().startswith("diff --git"):
                return {"ok": True, "result": text, "returncode": 0}
        subprocess.run(
            ["git", "config", "--add", "safe.directory", WORKDIR],
            cwd=WORKDIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        subprocess.run(
            ["git", "add", "-A", "--", ".", *[str(spec) for spec in exclude_pathspecs]],
            cwd=WORKDIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        result = subprocess.run(
            ["git", "diff", str(base_commit), "--", ".", *[str(spec) for spec in exclude_pathspecs]],
            cwd=WORKDIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
        )
        output = (result.stdout or "") + (result.stderr or "")
        return {
            "ok": result.returncode == 0,
            "result": output.strip(),
            "returncode": result.returncode,
        }
    except Exception as e:
        return {"ok": False, "result": f"Error extracting patch: {e}", "returncode": 1}



HANDLERS = {
    "exec": handle_exec,
    "commands": handle_commands,
    "read_file": handle_read_file,
    "write_file": handle_write_file,
    "edit_file": handle_edit_file,
    "list_dir": handle_list_dir,
    "extract_patch": handle_extract_patch,
}

signal.signal(signal.SIGTERM, lambda *_: os._exit(0))

_pacct_enable()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        tool = req.get("tool", "")
        args = req.get("args", {})
        handler = HANDLERS.get(tool)
        if handler:
            t0 = time.monotonic()
            resp = handler(args)
            resp["inner_duration_ms"] = (time.monotonic() - t0) * 1000
        else:
            resp = {"ok": False, "result": f"Error: Unsupported tool {tool!r}"}
    except Exception as e:
        resp = {"ok": False, "result": f"Error: agent dispatch failed: {e}"}
    sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
    sys.stdout.flush()
""").strip()


# Idempotent tools safe to retry after agent restart.
_IDEMPOTENT_TOOLS = frozenset({"read_file", "list_dir"})
_CONTROL_PLANE_NOOP_RESULTS = {
    "message": "Message replayed as no-op",
    "spawn": "Subagent spawn replayed as no-op",
    "sessions_yield": "Session yield replayed as no-op",
}


async def _readline_with_timeout(
    stream: asyncio.StreamReader,
    timeout_s: float | None,
) -> bytes:
    if timeout_s is None:
        return await stream.readline()
    return await asyncio.wait_for(stream.readline(), timeout=timeout_s + 5.0)


async def _kill_and_drain_python_probe_process(
    proc: asyncio.subprocess.Process,
    *,
    candidate: str,
    container_id: str,
) -> None:
    """Terminate and reap a timed-out Python probe process.

    A probe that cannot be reaped is a hard failure: continuing would leave a
    live ``docker exec``/``podman exec`` process around and make replay state
    host-dependent.
    """
    if proc.returncode is None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        except Exception as exc:
            raise RuntimeError(
                "ContainerAgent python probe cleanup failed to kill "
                f"candidate {candidate!r} in container {container_id[:12]}: {exc}"
            ) from exc
    try:
        await asyncio.wait_for(
            proc.communicate(),
            timeout=_PYTHON_PROBE_KILL_WAIT_S,
        )
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            "ContainerAgent python probe cleanup timed out after killing "
            f"candidate {candidate!r} in container {container_id[:12]}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "ContainerAgent python probe cleanup failed after killing "
            f"candidate {candidate!r} in container {container_id[:12]}: {exc}"
        ) from exc


class ContainerAgent:
    # The container bridge only needs the stdlib replay script, so Python 3.6+
    # is sufficient.  Full in-container project entrypoints still use
    # trace_collect.runtime.task_container's stricter Python >=3.11 contract.
    _PYTHON_CANDIDATES: tuple[str, ...] = _CONTAINER_PYTHON_CANDIDATES

    def __init__(
        self,
        container_id: str,
        container_executable: str,
        *,
        pythonpath: str | None = None,
        python_runtime: str | None = None,
        workdir: str = "/testbed",
        forward_pacct: bool = True,
    ) -> None:
        self._container_id = container_id
        self._executable = container_executable
        self._process: asyncio.subprocess.Process | None = None
        self._explicit_python_runtime = python_runtime
        self._python_runtime: str = python_runtime or "python3"
        self._pythonpath: str | None = pythonpath
        self._workdir = workdir or "/testbed"
        self._forward_pacct = forward_pacct
        self._lock = asyncio.Lock()

    async def _probe_python(self) -> str:
        """Find a working Python >=3.6 interpreter inside the container."""
        probe_script = (
            "import sys; raise SystemExit(0 if sys.version_info >= (3, 6) else 1)"
        )
        for cand in self._PYTHON_CANDIDATES:
            proc: asyncio.subprocess.Process | None = None
            try:
                proc = await asyncio.create_subprocess_exec(
                    self._executable,
                    "exec",
                    "-i",
                    "--user",
                    "0",
                    "-w",
                    self._workdir,
                    self._container_id,
                    cand,
                    "-c",
                    probe_script,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(
                    proc.communicate(),
                    timeout=_PYTHON_PROBE_TIMEOUT_S,
                )
                if proc.returncode == 0:
                    logger.info(
                        "ContainerAgent python probe: %s (cid=%s)",
                        cand,
                        self._container_id[:12],
                    )
                    return cand
            except asyncio.TimeoutError:
                if proc is not None:
                    await _kill_and_drain_python_probe_process(
                        proc,
                        candidate=cand,
                        container_id=self._container_id,
                    )
                continue
            except asyncio.CancelledError:
                if proc is not None:
                    await _kill_and_drain_python_probe_process(
                        proc,
                        candidate=cand,
                        container_id=self._container_id,
                    )
                raise
            except OSError as exc:
                if proc is not None and proc.returncode is None:
                    await _kill_and_drain_python_probe_process(
                        proc,
                        candidate=cand,
                        container_id=self._container_id,
                    )
                raise RuntimeError(
                    "ContainerAgent python probe failed to execute: "
                    f"{self._executable!r} for container {self._container_id[:12]}"
                ) from exc
        raise RuntimeError(
            "ContainerAgent: no Python >=3.6 found in container "
            f"{self._container_id[:12]}.  Tried: " + ", ".join(self._PYTHON_CANDIDATES)
        )

    async def start(self) -> None:
        if self._explicit_python_runtime is not None:
            self._python_runtime = self._explicit_python_runtime
        else:
            self._python_runtime = await self._probe_python()
        cmd: list[str] = [
            self._executable,
            "exec",
            "-i",
            "--user",
            "0",
            "-w",
            self._workdir,
        ]
        # Propagate PYTHONPATH so replayed subprocesses (e.g. pytest)
        # can find packages installed by bootstrap_task_container_python.
        if self._pythonpath:
            cmd.extend(["-e", f"PYTHONPATH={self._pythonpath}"])
        cmd.extend(["-e", f"OPENCLAW_CONTAINER_WORKDIR={self._workdir}"])
        # Forward the simulate-only segment-timeline toggle into the container.
        # The collect CLI never sets it, so the collect exec path is unchanged.
        segment_flag = os.environ.get("OPENCLAW_SEGMENT_TIMELINE")
        if segment_flag is not None:
            cmd.extend(["-e", f"OPENCLAW_SEGMENT_TIMELINE={segment_flag}"])
        # Forward the per-binary process-accounting toggle (needs --cap-add
        # SYS_PACCT on the container, added by the simulate driver).
        pacct_flag = os.environ.get("OPENCLAW_PACCT")
        if self._forward_pacct and pacct_flag is not None:
            cmd.extend(["-e", f"OPENCLAW_PACCT={pacct_flag}"])
        cmd.extend(
            [
                self._container_id,
                self._python_runtime,
                "-u",
                "-c",
                _REPLAY_AGENT_SCRIPT,
            ]
        )
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=1024 * 1024,  # 1MB — agent responses can exceed default 64KB
        )
        logger.info(
            "ContainerAgent started: cid=%s pid=%s runtime=%s",
            self._container_id[:12],
            self._process.pid,
            self._python_runtime,
        )

    async def stop(self) -> None:
        process = self._process
        if process is None:
            return
        self._process = None

        if process.stdin and not process.stdin.is_closing():
            try:
                process.stdin.close()
            except (BrokenPipeError, ConnectionResetError, ProcessLookupError):
                pass

        wait_task = asyncio.create_task(process.wait())
        try:
            await asyncio.wait_for(
                asyncio.shield(wait_task), timeout=_AGENT_STOP_GRACE_S
            )
            return
        except ProcessLookupError:
            return
        except asyncio.TimeoutError:
            pass

        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass

        try:
            await asyncio.wait_for(
                asyncio.shield(wait_task), timeout=_AGENT_KILL_WAIT_S
            )
        except ProcessLookupError:
            return
        except asyncio.TimeoutError as exc:
            wait_task.cancel()
            raise RuntimeError(
                f"ContainerAgent process did not exit after kill: pid={process.pid}"
            ) from exc

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def _restart(self) -> None:
        logger.warning("ContainerAgent restarting: cid=%s", self._container_id[:12])
        await self.stop()
        await self.start()

    async def execute(
        self,
        request: dict[str, Any],
        *,
        timeout_s: float | None = 600.0,
    ) -> dict[str, Any]:
        """Send a request and return the response. Restarts on crash."""
        tool_name = request.get("tool", "")
        for attempt in range(2):
            if not self.alive:
                if attempt == 0:
                    await self._restart()
                else:
                    return {"ok": False, "result": "Error: agent process dead"}

            proc = self._process
            assert (
                proc is not None and proc.stdin is not None and proc.stdout is not None
            )

            line = json.dumps(request, ensure_ascii=False) + "\n"
            try:
                async with self._lock:
                    proc.stdin.write(line.encode())
                    await proc.stdin.drain()
                    raw = await _readline_with_timeout(
                        proc.stdout,
                        timeout_s,
                    )
            except (asyncio.TimeoutError, BrokenPipeError, ConnectionResetError):
                await self._restart()
                if tool_name in _IDEMPOTENT_TOOLS:
                    continue
                return {"ok": False, "result": "[timeout]", "returncode": 124}

            if not raw:
                # EOF — agent crashed
                await self._restart()
                if tool_name in _IDEMPOTENT_TOOLS:
                    continue
                return {"ok": False, "result": "Error: agent process crashed"}

            # Skip stray non-JSON lines (e.g. Python warnings, sitecustomize output)
            decoded = raw.decode(errors="replace").strip()
            for _skip in range(50):
                if decoded.startswith("{"):
                    break
                logger.debug("Skipping non-JSON agent output: %s", decoded[:120])
                try:
                    raw = await _readline_with_timeout(proc.stdout, timeout_s)
                    decoded = raw.decode(errors="replace").strip()
                except (asyncio.TimeoutError, BrokenPipeError):
                    return {"ok": False, "result": "[timeout]", "returncode": 124}
            else:
                return {"ok": False, "result": "Error: agent emitted no JSON response"}

            try:
                return json.loads(decoded)
            except json.JSONDecodeError:
                return {
                    "ok": False,
                    "result": f"Error: invalid agent response: {decoded[:200]}",
                }

        return {"ok": False, "result": "Error: agent restart failed"}


def _resource_timed_exec_request(
    *,
    command: str,
    timeout_s: float,
    source_resource_timeline: dict[str, Any] | None,
) -> tuple[dict[str, Any], float | None]:
    request = {"tool": "exec", "args": {"command": command, "timeout": timeout_s}}
    resource_timeline = valid_resource_timeline(source_resource_timeline)
    if resource_timeline is None:
        return request, timeout_s
    request["args"]["source_resource_timeline"] = resource_timeline
    return request, _RESOURCE_AWARE_AGENT_RESPONSE_TIMEOUT_S


def _resolve_tool_request(
    tool_name: str | None,
    params: dict[str, Any],
    command_timeout_s: float,
    source_exec_timeout_s: float | None = None,
    source_resource_timeline: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, float | None]:
    """Build a JSON-line request plus the outer response timeout."""

    exec_fallback_timeout_s = (
        source_exec_timeout_s
        if source_exec_timeout_s is not None
        else command_timeout_s
    )

    # Shell commands
    if "command" in params:
        timeout_s = _resolve_exec_timeout_s(
            params,
            default_timeout_s=exec_fallback_timeout_s,
        )
        return _resource_timed_exec_request(
            command=params["command"],
            timeout_s=timeout_s,
            source_resource_timeline=source_resource_timeline,
        )
    if "commands" in params:
        timeout_s = _resolve_exec_timeout_s(
            params,
            default_timeout_s=exec_fallback_timeout_s,
        )
        return (
            {
                "tool": "commands",
                "args": {"commands": list(params["commands"]), "timeout": timeout_s},
            },
            timeout_s,
        )

    if tool_name == "exec":
        command = params.get("command")
        commands = params.get("commands")
        if command:
            timeout_s = _resolve_exec_timeout_s(
                params,
                default_timeout_s=exec_fallback_timeout_s,
            )
            return _resource_timed_exec_request(
                command=command,
                timeout_s=timeout_s,
                source_resource_timeline=source_resource_timeline,
            )
        if commands:
            timeout_s = _resolve_exec_timeout_s(
                params,
                default_timeout_s=exec_fallback_timeout_s,
            )
            return (
                {
                    "tool": "commands",
                    "args": {"commands": list(commands), "timeout": timeout_s},
                },
                timeout_s,
            )
        return None, command_timeout_s  # missing command/commands

    if tool_name == "read_file":
        return {
            "tool": "read_file",
            "args": {"path": params.get("path", "")},
        }, command_timeout_s

    if tool_name == "write_file":
        return (
            {
                "tool": "write_file",
                "args": {
                    "path": params.get("path", ""),
                    "content": params.get("content", ""),
                },
            },
            command_timeout_s,
        )

    if tool_name == "edit_file":
        return (
            {
                "tool": "edit_file",
                "args": {
                    "path": params.get("path", ""),
                    "old_text": params.get("old_text", ""),
                    "new_text": params.get("new_text", ""),
                    "replace_all": bool(params.get("replace_all", False)),
                },
            },
            command_timeout_s,
        )

    if tool_name == "list_dir":
        return {
            "tool": "list_dir",
            "args": {
                "path": params.get("path", "."),
                "recursive": bool(params.get("recursive", False)),
                "max_entries": int(params.get("max_entries") or _CONTAINER_LIST_DEFAULT_MAX),
            },
        }, command_timeout_s

    return None, command_timeout_s  # unsupported tool


def _trace_tool_response_metadata(resp: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in (
        "resource_timeout_policy",
        "resource_virtual_time_s",
        "resource_stall_s",
        "segment_timeline",
        "per_process",
    ):
        if key in resp:
            metadata[key] = resp[key]
    return metadata


async def execute_trace_tool_detailed(
    *,
    agent: ContainerAgent,
    tool_name: str | None,
    tool_args_json: str,
    command_timeout_s: float,
    source_exec_timeout_s: float | None = None,
    allow_source_runtime_artifacts: bool = False,
    source_resource_timeline: dict[str, Any] | None = None,
) -> tuple[str, bool, float | None, dict[str, Any]]:
    """Execute one trace tool call and return replay metadata."""

    resolved_name, params = _unwrap_tool_args(
        tool_name=tool_name,
        tool_args_json=tool_args_json,
    )

    request, request_timeout_s = _resolve_tool_request(
        resolved_name,
        params,
        command_timeout_s,
        source_exec_timeout_s,
        source_resource_timeline,
    )

    if resolved_name in _CONTROL_PLANE_NOOP_RESULTS:
        return _CONTROL_PLANE_NOOP_RESULTS[resolved_name], True, 0.0, {}

    artifact_path = source_runtime_artifact_path_from_tool_call(
        tool_name=resolved_name,
        tool_args_json=json.dumps(params, ensure_ascii=False),
    )
    if artifact_path is not None and not allow_source_runtime_artifacts:
        return (
            "Error: source trace references an OpenClaw runtime artifact "
            f"that is unavailable in a fresh replay container: {artifact_path}",
            False,
            0.0,
            {"replay_failure_kind": "source_runtime_artifact_unavailable"},
        )

    if request is None:
        return (
            f"Error: Unsupported replay tool {resolved_name!r}",
            False,
            None,
            {"replay_failure_kind": "unsupported_replay_tool"},
        )

    resp = await agent.execute(request, timeout_s=request_timeout_s)
    result = resp.get("result", "")
    ok = resp.get("ok", False)
    inner_duration_ms = resp.get("inner_duration_ms")
    metadata = _trace_tool_response_metadata(resp)

    # Append exit code for exec-style commands
    if request["tool"] in ("exec", "commands"):
        rc = resp.get("returncode")
        if not isinstance(rc, int) or isinstance(rc, bool):
            result = f"{result}\n\nExit code: <missing>".strip()
            metadata = {
                **metadata,
                "replay_failure_kind": "malformed_replay_exec_response",
            }
            return result, False, inner_duration_ms, metadata
        result = f"{result}\n\nExit code: {rc}".strip()
        ok = bool(ok)

    return result, ok, inner_duration_ms, metadata


async def execute_trace_tool(
    *,
    agent: ContainerAgent,
    tool_name: str | None,
    tool_args_json: str,
    command_timeout_s: float,
    source_exec_timeout_s: float | None = None,
    allow_source_runtime_artifacts: bool = False,
    source_resource_timeline: dict[str, Any] | None = None,
) -> tuple[str, bool, float | None]:
    """Execute one trace tool call via the persistent in-container agent."""

    result, ok, inner_duration_ms, _metadata = await execute_trace_tool_detailed(
        agent=agent,
        tool_name=tool_name,
        tool_args_json=tool_args_json,
        command_timeout_s=command_timeout_s,
        source_exec_timeout_s=source_exec_timeout_s,
        allow_source_runtime_artifacts=allow_source_runtime_artifacts,
        source_resource_timeline=source_resource_timeline,
    )
    return result, ok, inner_duration_ms
