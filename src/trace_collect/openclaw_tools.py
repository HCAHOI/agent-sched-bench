"""Execute tool calls inside Docker/Podman containers via a persistent agent."""

from __future__ import annotations

import asyncio
import json
import logging
import textwrap
from typing import Any

from trace_collect.resource_timeline import valid_resource_timeline
from trace_collect.runtime.task_container import _CONTAINER_PYTHON_CANDIDATES

logger = logging.getLogger(__name__)

_OPENCLAW_EXEC_DEFAULT_TIMEOUT_S = 300.0
_OPENCLAW_EXEC_MAX_TIMEOUT_S = 600.0
# Outer guard for resource-aware exec requests. The in-container watchdog owns
# the modeled deadline; this only prevents an agent protocol deadlock from
# blocking simulate forever.
_RESOURCE_AWARE_AGENT_RESPONSE_TIMEOUT_S = 24 * 60 * 60.0
_AGENT_STOP_GRACE_S = 5.0
_AGENT_KILL_WAIT_S = 5.0
_PYTHON_PROBE_TIMEOUT_S = 30.0
_PYTHON_PROBE_KILL_WAIT_S = 5.0
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


# Persistent python3 agent script injected into Docker container; reads JSON-line requests
# from stdin, writes JSON-line responses to stdout (subprocess.run uses capture_output=True).
_REPLAY_AGENT_SCRIPT = textwrap.dedent(r"""
import json, os, sys, subprocess, difflib, signal, time

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
    process = subprocess.Popen(
        cmd,
        shell=True,
        cwd="/testbed",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=start_new_session,
    )
    virtual_time_s = 0.0
    last_counters = _read_resource_counters()
    last_progress_wall_s = last_counters["time_s"]
    while True:
        try:
            stdout, stderr = process.communicate(timeout=_RESOURCE_SAMPLE_INTERVAL_S)
            output = (stdout or "") + (stderr or "")
            return {
                "ok": True,
                "result": _truncate_output(output),
                "returncode": process.returncode,
                "resource_timeout_policy": "resource_integrated",
                "resource_virtual_time_s": round(virtual_time_s, 6),
            }
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
                return {
                    "ok": False,
                    "result": output,
                    "returncode": 124,
                    "resource_timeout_policy": "resource_integrated",
                    "resource_virtual_time_s": round(virtual_time_s, 6),
                }
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
                return {
                    "ok": False,
                    "result": output,
                    "returncode": 124,
                    "resource_timeout_policy": "resource_integrated",
                    "resource_virtual_time_s": round(virtual_time_s, 6),
                    "resource_stall_s": round(stalled_s, 6),
                }
            last_counters = current_counters


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
    try:
        r = subprocess.run(cmd, shell=True, cwd="/testbed",
                           capture_output=True, text=True, timeout=timeout, env=env)
        output = (r.stdout or "") + (r.stderr or "")
        return {"ok": True, "result": _truncate_output(output), "returncode": r.returncode}
    except subprocess.TimeoutExpired:
        return {"ok": False, "result": "[timeout]", "returncode": 124}

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
            r = subprocess.run(cmd, shell=True, cwd="/testbed",
                               capture_output=True, text=True, timeout=timeout, env=env)
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
    try:
        entries = sorted(e for e in os.listdir(path) if e not in _LIST_IGNORE)
        if len(entries) > _LIST_MAX:
            entries = entries[:_LIST_MAX]
            entries.append(f"... ({len(os.listdir(path)) - _LIST_MAX} more entries)")
        return {"ok": True, "result": "\n".join(entries)}
    except Exception as e:
        return {"ok": False, "result": f"Error: {e}"}

HANDLERS = {
    "exec": handle_exec,
    "commands": handle_commands,
    "read_file": handle_read_file,
    "write_file": handle_write_file,
    "edit_file": handle_edit_file,
    "list_dir": handle_list_dir,
}

signal.signal(signal.SIGTERM, lambda *_: os._exit(0))

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
    # Container Python interpreter candidates — MUST match
    # _CONTAINER_PYTHON_CANDIDATES in trace_collect.runtime.task_container
    # so that collect (resolve_running_container_exec_config) and simulate
    # (ContainerAgent) select the same interpreter for the same image.
    _PYTHON_CANDIDATES: tuple[str, ...] = _CONTAINER_PYTHON_CANDIDATES

    def __init__(
        self,
        container_id: str,
        container_executable: str,
        *,
        pythonpath: str | None = None,
    ) -> None:
        self._container_id = container_id
        self._executable = container_executable
        self._process: asyncio.subprocess.Process | None = None
        self._python_runtime: str = "python3"  # fallback, overwritten in start()
        self._pythonpath: str | None = pythonpath

    async def _probe_python(self) -> str:
        """Find a working Python >=3.11 interpreter inside the container."""
        probe_script = (
            "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
        )
        for cand in self._PYTHON_CANDIDATES:
            proc: asyncio.subprocess.Process | None = None
            try:
                proc = await asyncio.create_subprocess_exec(
                    self._executable,
                    "exec",
                    "-i",
                    "-w",
                    "/testbed",
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
            "ContainerAgent: no Python >=3.11 found in container "
            f"{self._container_id[:12]}.  Tried: " + ", ".join(self._PYTHON_CANDIDATES)
        )

    async def start(self) -> None:
        self._python_runtime = await self._probe_python()
        cmd: list[str] = [
            self._executable,
            "exec",
            "-i",
            "-w",
            "/testbed",
        ]
        # Propagate PYTHONPATH so replayed subprocesses (e.g. pytest)
        # can find packages installed by bootstrap_task_container_python.
        if self._pythonpath:
            cmd.extend(["-e", f"PYTHONPATH={self._pythonpath}"])
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
            "args": {"path": params.get("path", ".")},
        }, command_timeout_s

    return None, command_timeout_s  # unsupported tool


def _trace_tool_response_metadata(resp: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in (
        "resource_timeout_policy",
        "resource_virtual_time_s",
        "resource_stall_s",
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

    if resolved_name == "message":
        return "Message replayed as no-op", True, 0.0, {}

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
            {},
        )

    if request is None:
        return f"Error: Unsupported replay tool {resolved_name!r}", False, None, {}

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
