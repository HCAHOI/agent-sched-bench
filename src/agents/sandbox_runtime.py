from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import posixpath
import socket
import subprocess
import stat
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.openclaw._checkpoint_container import (
    _probe_container_python,
    _write_container_manifest,
    run_container_checkpoint,
)
from agents.openclaw._session_runner import (
    CheckpointHashCacheEntry,
    CheckpointSnapshotEntry,
    _any_file_newer_than,
    _compute_checkpoint_changed_paths,
    _compute_checkpoint_deleted_paths,
    _checkpoint_entry_type,
    _checkpoint_relpath,
    _checkpoint_relpath_is_skipped,
    _snapshot_checkpoint_entries,
    _write_cas_manifest,
)

logger = logging.getLogger(__name__)

_CHECKPOINT_SKIP_DIRS = frozenset({".git"})


@dataclass
class SandboxSnapshot:
    """Atomically-paired process and disk checkpoint.

    ``process_state`` and ``disk_state`` are opaque backend-specific payloads.
    Docker uses ``process_state=None`` and CAS manifest data in ``disk_state``;
    future FC backends can carry memory/vCPU state in ``process_state`` and
    overlay/delta metadata in ``disk_state``.
    """

    process_state: dict[str, Any] | None
    disk_state: dict[str, Any]
    root: str
    timestamp_ns: int


@dataclass
class AgentTransportRequest:
    tool: str
    args: dict[str, Any]


@dataclass
class AgentTransportResponse:
    result: str
    ok: bool
    inner_duration_ms: float | None = None
    returncode: int | None = None
    timed_out: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SandboxBackend(ABC):
    @abstractmethod
    async def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def stop(self) -> None:
        raise NotImplementedError

    async def execute(
        self,
        request: AgentTransportRequest,
        *,
        timeout_s: float | None = 600.0,
    ) -> AgentTransportResponse:
        raise NotImplementedError

    @abstractmethod
    async def capture_snapshot(
        self,
        *,
        incremental_since: SandboxSnapshot | None = None,
    ) -> SandboxSnapshot:
        raise NotImplementedError

    @abstractmethod
    async def restore_snapshot(self, snapshot: SandboxSnapshot) -> None:
        raise NotImplementedError

    @abstractmethod
    async def probe_changes_since(self, marker_ns: int) -> bool:
        raise NotImplementedError


def agent_response_dict_from_transport(
    response: AgentTransportResponse,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "result": response.result,
        "ok": response.ok,
    }
    if response.inner_duration_ms is not None:
        payload["inner_duration_ms"] = response.inner_duration_ms
    if response.returncode is not None:
        payload["returncode"] = response.returncode
    if response.timed_out is not None:
        payload["timed_out"] = response.timed_out
    payload.update(response.metadata)
    return payload


def transport_response_from_agent_dict(
    response: Mapping[str, Any],
) -> AgentTransportResponse:
    result = response.get("result", "")
    if not isinstance(result, str):
        result = str(result)

    returncode = response.get("returncode")
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        returncode = None

    timed_out = response.get("timed_out")
    if not isinstance(timed_out, bool):
        timed_out = None

    inner_duration_ms = response.get("inner_duration_ms")
    if isinstance(inner_duration_ms, bool) or not isinstance(
        inner_duration_ms,
        int | float,
    ):
        inner_duration_ms = None

    reserved = {"result", "ok", "returncode", "timed_out", "inner_duration_ms"}
    metadata = {key: value for key, value in response.items() if key not in reserved}
    return AgentTransportResponse(
        result=result,
        ok=bool(response.get("ok", False)),
        inner_duration_ms=(
            float(inner_duration_ms) if inner_duration_ms is not None else None
        ),
        returncode=returncode,
        timed_out=timed_out,
        metadata=metadata,
    )


def _coerce_fake_returncode(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        raise ValueError(f"fake returncode must be int, got {value!r}")
    return int(value)


class FakeBackend(SandboxBackend):
    """In-memory sandbox for unit tests and CI without container/KVM access."""

    def __init__(self, *, root: str = "/testbed") -> None:
        self.root = posixpath.normpath(root)
        self._files: dict[str, str] = {}
        self._mtimes: dict[str, int] = {}
        self._clock = 0
        self._exec_counter = 0

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def execute(
        self,
        request: AgentTransportRequest,
        *,
        timeout_s: float | None = 600.0,
    ) -> AgentTransportResponse:
        started = time.monotonic()
        tool = request.tool
        args = request.args
        if tool == "exec":
            response = self._execute_exec(args)
        elif tool == "commands":
            response = self._execute_commands(args)
        elif tool == "read_file":
            response = self._execute_read_file(args)
        elif tool == "write_file":
            response = self._execute_write_file(args)
        elif tool == "edit_file":
            response = self._execute_edit_file(args)
        elif tool == "list_dir":
            response = self._execute_list_dir(args)
        elif tool in {"spawn", "message"}:
            response = AgentTransportResponse(result="", ok=True)
        else:
            response = AgentTransportResponse(result="", ok=True)
        response.inner_duration_ms = (time.monotonic() - started) * 1000.0
        return response

    async def capture_snapshot(
        self,
        *,
        incremental_since: SandboxSnapshot | None = None,
    ) -> SandboxSnapshot:
        entries: dict[str, dict[str, Any]] = {}
        for path, content in sorted(self._files.items()):
            relpath = self._relpath(path)
            content_bytes = content.encode("utf-8")
            entries[relpath] = {
                "hash": hashlib.sha256(content_bytes).hexdigest(),
                "content": content,
                "size": len(content_bytes),
                "mtime_ns": self._mtimes[path],
            }
        return SandboxSnapshot(
            process_state=None,
            disk_state={"entries": entries},
            root=self.root,
            timestamp_ns=self._clock,
        )

    async def restore_snapshot(self, snapshot: SandboxSnapshot) -> None:
        entries = snapshot.disk_state.get("entries")
        if not isinstance(entries, dict):
            raise ValueError("fake snapshot disk_state.entries must be a dict")

        restored_files: dict[str, str] = {}
        restored_mtimes: dict[str, int] = {}
        for relpath, entry in entries.items():
            if not isinstance(relpath, str):
                raise ValueError(f"fake snapshot path must be a string: {relpath!r}")
            if not isinstance(entry, dict):
                raise ValueError(f"fake snapshot entry must be a dict: {relpath}")
            content = entry.get("content")
            if not isinstance(content, str):
                raise ValueError(f"fake snapshot entry missing content: {relpath}")
            path = self._normalize_path(relpath)
            restored_files[path] = content
            raw_mtime = entry.get("mtime_ns", snapshot.timestamp_ns)
            if isinstance(raw_mtime, bool) or not isinstance(raw_mtime, int):
                raise ValueError(f"fake snapshot entry has invalid mtime: {relpath}")
            restored_mtimes[path] = raw_mtime

        self._files = restored_files
        self._mtimes = restored_mtimes
        self._clock = max([snapshot.timestamp_ns, *restored_mtimes.values()], default=0)

    async def probe_changes_since(self, marker_ns: int) -> bool:
        return any(mtime_ns > marker_ns for mtime_ns in self._mtimes.values())

    def _execute_exec(self, args: Mapping[str, Any]) -> AgentTransportResponse:
        command = args.get("command", "")
        if not isinstance(command, str):
            command = str(command)
        exec_counter = self._exec_counter
        self._exec_counter += 1
        returncode = _coerce_fake_returncode(args.get("returncode"))
        return AgentTransportResponse(
            result=f"simulated exec: {command}\n\nExit code: {returncode}",
            ok=True,
            returncode=returncode,
            timed_out=False,
            metadata={"fake_exec_counter": exec_counter},
        )

    def _execute_commands(self, args: Mapping[str, Any]) -> AgentTransportResponse:
        commands = args.get("commands", [])
        if not isinstance(commands, list):
            raise ValueError("fake commands args.commands must be a list")
        outputs: list[str] = []
        returncode = 0
        last_exec_counter = 0
        for index, raw_command in enumerate(commands):
            command = raw_command if isinstance(raw_command, str) else str(raw_command)
            last_exec_counter = self._exec_counter
            self._exec_counter += 1
            returncode = 0
            outputs.append(
                f"[call {index}]\nsimulated exec: {command}\n\nExit code: {returncode}"
            )
        return AgentTransportResponse(
            result="\n".join(outputs),
            ok=True,
            returncode=returncode,
            timed_out=False,
            metadata={"fake_exec_counter": last_exec_counter},
        )

    def _execute_read_file(self, args: Mapping[str, Any]) -> AgentTransportResponse:
        path = self._normalize_path(str(args.get("path", "")))
        content = self._files.get(path)
        if content is None:
            return AgentTransportResponse(result=f"Error: No such file: {path}", ok=False)
        if not content:
            return AgentTransportResponse(result=f"(Empty file: {path})", ok=True)
        lines = content.splitlines()
        numbered = "\n".join(f"{index + 1}| {line}" for index, line in enumerate(lines))
        return AgentTransportResponse(result=numbered, ok=True)

    def _execute_write_file(self, args: Mapping[str, Any]) -> AgentTransportResponse:
        path = self._normalize_path(str(args.get("path", "")))
        content = args.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        self._write(path, content)
        return AgentTransportResponse(result=f"Successfully wrote {path}", ok=True)

    def _execute_edit_file(self, args: Mapping[str, Any]) -> AgentTransportResponse:
        path = self._normalize_path(str(args.get("path", "")))
        content = self._files.get(path)
        if content is None:
            return AgentTransportResponse(result=f"Error editing file: {path}", ok=False)
        old_text = str(args.get("old_text", ""))
        new_text = str(args.get("new_text", ""))
        replace_all = bool(args.get("replace_all", False))
        count = content.count(old_text)
        if old_text == "" or count == 0:
            return AgentTransportResponse(
                result=f"Error: old_text not found in {path}",
                ok=False,
            )
        if count > 1 and not replace_all:
            return AgentTransportResponse(
                result=(
                    f"Warning: old_text appears {count} times. "
                    "Provide more context or set replace_all=true."
                ),
                ok=False,
            )
        self._write(
            path,
            content.replace(old_text, new_text)
            if replace_all
            else content.replace(old_text, new_text, 1),
        )
        return AgentTransportResponse(result=f"Successfully edited {path}", ok=True)

    def _execute_list_dir(self, args: Mapping[str, Any]) -> AgentTransportResponse:
        path = self._normalize_path(str(args.get("path", ".")))
        prefix = path if path.endswith("/") else f"{path}/"
        children: set[str] = set()
        for file_path in self._files:
            if file_path == path:
                continue
            if not file_path.startswith(prefix):
                continue
            child = file_path[len(prefix) :].split("/", 1)[0]
            if child:
                children.add(child)
        return AgentTransportResponse(result="\n".join(sorted(children)), ok=True)

    def _write(self, path: str, content: str) -> None:
        self._clock += 1
        self._files[path] = content
        self._mtimes[path] = self._clock

    def _normalize_path(self, path: str) -> str:
        if not path:
            raise ValueError("fake sandbox path must be non-empty")
        candidate = path if path.startswith("/") else f"{self.root}/{path}"
        normalized = posixpath.normpath(candidate)
        if normalized != self.root and not normalized.startswith(f"{self.root}/"):
            raise ValueError(f"fake sandbox path outside root: {path}")
        return normalized

    def _relpath(self, path: str) -> str:
        if path == self.root:
            return "."
        return path.removeprefix(f"{self.root}/")


_CHECKPOINT_BACKENDS = frozenset({"walk", "overlay", "verify"})


def validate_checkpoint_backend(value: str | None) -> str:
    if value is None:
        return "walk"
    if value not in _CHECKPOINT_BACKENDS:
        choices = ", ".join(sorted(_CHECKPOINT_BACKENDS))
        raise ValueError(f"checkpoint_backend must be one of {choices}, got {value!r}")
    return value


def _probe_container_changes_since(
    *,
    container_runtime: dict[str, str],
    root: str,
    marker_ns: int,
) -> bool:
    script = r"""
import os, sys
root = os.environ["CHECKPOINT_ROOT"]
marker = int(os.environ["CHECKPOINT_MARKER_NS"])
skip_dirs = {".git"}

def newer(path):
    try:
        return os.lstat(path).st_mtime_ns > marker
    except OSError:
        return True

if newer(root):
    sys.exit(0)
for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
    dirnames[:] = [name for name in dirnames if name not in skip_dirs]
    for name in list(dirnames) + list(filenames):
        if newer(os.path.join(dirpath, name)):
            sys.exit(0)
sys.exit(1)
"""
    python_path = _probe_container_python(container_runtime)
    result = subprocess.run(
        [
            container_runtime["executable"],
            "exec",
            "-e",
            f"CHECKPOINT_ROOT={root}",
            "-e",
            f"CHECKPOINT_MARKER_NS={marker_ns}",
            container_runtime["id"],
            python_path,
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    detail = result.stderr.strip() or result.stdout.strip()
    raise RuntimeError(
        "container checkpoint probe failed "
        f"({result.returncode}): {detail}"
    )


def _is_overlay_whiteout(name: str, st: os.stat_result) -> bool:
    return (
        name.startswith(".wh.")
        and stat.S_ISCHR(st.st_mode)
        and os.major(st.st_rdev) == 0
        and os.minor(st.st_rdev) == 0
    )


def _whiteout_deleted_path(relpath: str) -> str:
    path = Path(relpath)
    name = path.name
    if name == ".wh..wh..opq":
        raise RuntimeError(
            "overlay opaque directory whiteouts are unsupported by CAS manifests"
        )
    if not name.startswith(".wh.") or len(name) <= len(".wh."):
        raise RuntimeError(f"invalid overlay whiteout path: {relpath}")
    deleted_name = name.removeprefix(".wh.")
    parent = path.parent.as_posix()
    if parent == ".":
        return deleted_name
    return f"{parent}/{deleted_name}"


def _any_overlay_entry_newer_than(root: Path, marker_ns: int) -> bool:
    walk_errors: list[OSError] = []

    def record_walk_error(exc: OSError) -> None:
        walk_errors.append(exc)

    def entry_is_newer(path: str | Path) -> bool:
        try:
            return os.lstat(path).st_mtime_ns > marker_ns
        except OSError:
            return True

    if entry_is_newer(root):
        return True
    for dirpath, dirnames, filenames in os.walk(
        root,
        topdown=True,
        onerror=record_walk_error,
        followlinks=False,
    ):
        dirnames[:] = [name for name in dirnames if name != ".git"]
        for name in (*dirnames, *filenames):
            if entry_is_newer(os.path.join(dirpath, name)):
                return True
    return bool(walk_errors)


def _normalized_manifest_entries(snapshot: SandboxSnapshot) -> dict[str, Any]:
    entries = snapshot.disk_state.get("entries")
    if not isinstance(entries, dict):
        raise ValueError("snapshot disk_state.entries must be a dict")
    normalized: dict[str, Any] = {}
    for path, entry in entries.items():
        if not isinstance(path, str):
            raise ValueError(f"snapshot manifest path must be a string: {path!r}")
        if not isinstance(entry, dict):
            normalized[path] = entry
            continue
        entry_type = entry.get("type", "file")
        if entry_type == "symlink":
            normalized[path] = {
                "type": "symlink",
                "target": entry.get("target"),
            }
            continue
        normalized[path] = {
            "hash": entry.get("hash"),
            "mode": (
                stat.S_IMODE(entry["mode"])
                if isinstance(entry.get("mode"), int)
                and not isinstance(entry.get("mode"), bool)
                else None
            ),
        }
    return normalized


def _normalized_deleted_paths(snapshot: SandboxSnapshot) -> list[str]:
    deleted_paths = snapshot.disk_state.get("deleted_paths", [])
    if not isinstance(deleted_paths, list):
        raise ValueError("snapshot disk_state.deleted_paths must be a list")
    return sorted(str(path) for path in deleted_paths)


def checkpoint_manifest_differences(
    walk_snapshot: SandboxSnapshot,
    overlay_snapshot: SandboxSnapshot,
) -> dict[str, Any]:
    walk_entries = _normalized_manifest_entries(walk_snapshot)
    overlay_entries = _normalized_manifest_entries(overlay_snapshot)
    differences: dict[str, Any] = {}
    if walk_entries != overlay_entries:
        walk_paths = set(walk_entries)
        overlay_paths = set(overlay_entries)
        mismatched = sorted(
            path
            for path in walk_paths & overlay_paths
            if walk_entries[path] != overlay_entries[path]
        )
        differences["entries"] = {
            "only_walk": sorted(walk_paths - overlay_paths),
            "only_overlay": sorted(overlay_paths - walk_paths),
            "mismatched": mismatched,
        }
    walk_deleted = _normalized_deleted_paths(walk_snapshot)
    overlay_deleted = _normalized_deleted_paths(overlay_snapshot)
    if walk_deleted != overlay_deleted:
        differences["deleted_paths"] = {
            "walk": walk_deleted,
            "overlay": overlay_deleted,
        }
    return differences


async def _capture_backends_for_verification(
    walk_backend: WalkBackend,
    overlay_backend: OverlayBackend,
    *,
    incremental_since: SandboxSnapshot | None,
) -> tuple[SandboxSnapshot, SandboxSnapshot]:
    walk_snapshot, overlay_snapshot = await asyncio.gather(
        walk_backend.capture_snapshot(incremental_since=incremental_since),
        overlay_backend.capture_snapshot(incremental_since=incremental_since),
    )
    return walk_snapshot, overlay_snapshot


async def verify_checkpoint_backends(
    *,
    container_id: str,
    marker_ns: int,
    container_executable: str = "docker",
    root: str = "/testbed",
    checkpoint_dir: Path,
) -> SandboxSnapshot:
    marker_snapshot = SandboxSnapshot(
        process_state=None,
        disk_state={},
        root=root,
        timestamp_ns=marker_ns,
    )
    walk_snapshot, overlay_snapshot = await _capture_backends_for_verification(
        WalkBackend(
            root=root,
            checkpoint_dir=checkpoint_dir,
            container_runtime={"id": container_id, "executable": container_executable},
        ),
        OverlayBackend(
            root=root,
            checkpoint_dir=checkpoint_dir,
            container_id=container_id,
            container_executable=container_executable,
        ),
        incremental_since=marker_snapshot,
    )
    differences = checkpoint_manifest_differences(walk_snapshot, overlay_snapshot)
    if differences:
        raise RuntimeError(
            "checkpoint backend verification failed: "
            f"{json.dumps(differences, sort_keys=True)}"
        )
    return walk_snapshot


class WalkBackend(SandboxBackend):
    """CAS checkpoint backend that walks the sandbox filesystem."""

    def __init__(
        self,
        *,
        root: str = "/testbed",
        checkpoint_dir: Path,
        container_runtime: dict[str, str] | None = None,
    ) -> None:
        self.root = root
        self.checkpoint_dir = checkpoint_dir
        self.container_runtime = container_runtime
        self._checkpoint_snapshot_entries: dict[str, CheckpointSnapshotEntry] | None = (
            None
        )
        self._checkpoint_hash_cache: dict[str, CheckpointHashCacheEntry] = {}

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def execute(
        self,
        request: AgentTransportRequest,
        *,
        timeout_s: float | None = 600.0,
    ) -> AgentTransportResponse:
        raise NotImplementedError("walk checkpoint backend does not execute tools")

    async def capture_snapshot(
        self,
        *,
        incremental_since: SandboxSnapshot | None = None,
    ) -> SandboxSnapshot:
        return self._capture_snapshot_sync(incremental_since)

    async def restore_snapshot(self, snapshot: SandboxSnapshot) -> None:
        raise RuntimeError("walk checkpoint backend does not restore snapshots directly")

    async def probe_changes_since(self, marker_ns: int) -> bool:
        return self._probe_changes_since_sync(marker_ns)

    def _capture_snapshot_sync(
        self,
        incremental_since: SandboxSnapshot | None,
    ) -> SandboxSnapshot:
        if self.container_runtime is not None:
            return self._capture_container_snapshot(incremental_since)
        return self._capture_host_snapshot(incremental_since)

    def _capture_container_snapshot(
        self,
        incremental_since: SandboxSnapshot | None,
    ) -> SandboxSnapshot:
        manifest_path = self.checkpoint_dir / f"snapshot-{time.time_ns()}-manifest.json"
        result = run_container_checkpoint(
            container_runtime=self.container_runtime or {},
            manifest_path=manifest_path,
            incremental_since_ns=(
                None if incremental_since is None else incremental_since.timestamp_ns
            ),
            prev_snapshot_entries=self._checkpoint_snapshot_entries,
            force_full=incremental_since is None,
        )
        if "skipped" in result:
            if incremental_since is None:
                raise RuntimeError("container checkpoint skipped without a base snapshot")
            return incremental_since
        if "error" in result:
            raise RuntimeError(str(result["error"]))

        state = result.pop("_state", {})
        snapshot_entries = state.get("current_snapshot")
        if isinstance(snapshot_entries, dict):
            self._checkpoint_snapshot_entries = snapshot_entries
        timestamp_ns = state.get("now_ns")
        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int):
            timestamp_ns = time.time_ns()

        manifest = _read_manifest(manifest_path)
        return SandboxSnapshot(
            process_state=None,
            disk_state={
                "manifest_path": str(manifest_path),
                "entries": manifest["entries"],
                "deleted_paths": manifest["deleted_paths"],
                "kind": result.get("kind"),
                "incremental": result.get("incremental"),
                "incremental_since_ns": (
                    None if incremental_since is None else incremental_since.timestamp_ns
                ),
                "chain_bytes": result.get("chain_bytes"),
                "size_bytes": manifest_path.stat().st_size,
                "_checkpoint_state": state,
            },
            root=str(result.get("root") or self.root),
            timestamp_ns=timestamp_ns,
        )

    def _capture_host_snapshot(
        self,
        incremental_since: SandboxSnapshot | None,
    ) -> SandboxSnapshot:
        root = Path(self.root)
        marker_before_ns = time.time_ns()
        current_snapshot = _snapshot_checkpoint_entries(root)
        deleted_paths: list[str] = []
        changed_paths: set[str] | None = None
        previous_entries = self._checkpoint_snapshot_entries
        if incremental_since is not None and previous_entries is not None:
            deleted_paths = _compute_checkpoint_deleted_paths(
                previous=previous_entries,
                current=current_snapshot,
            )
            changed_paths = _compute_checkpoint_changed_paths(
                previous=previous_entries,
                current=current_snapshot,
            )
            if not deleted_paths and not changed_paths:
                return incremental_since

        manifest_path = self.checkpoint_dir / f"snapshot-{marker_before_ns}-manifest.json"
        chain_bytes = _write_cas_manifest(
            root=root,
            manifest_path=manifest_path,
            incremental_since_ns=(
                None if incremental_since is None else incremental_since.timestamp_ns
            ),
            deleted_paths=deleted_paths,
            incremental_changed_paths=changed_paths,
            hash_cache=self._checkpoint_hash_cache,
        )
        if _any_file_newer_than(root, marker_before_ns):
            manifest_path.unlink(missing_ok=True)
            raise RuntimeError("checkpoint failed: filesystem changed during checkpoint")
        self._checkpoint_snapshot_entries = current_snapshot
        manifest = _read_manifest(manifest_path)
        return SandboxSnapshot(
            process_state=None,
            disk_state={
                "manifest_path": str(manifest_path),
                "entries": manifest["entries"],
                "deleted_paths": manifest["deleted_paths"],
                "kind": (
                    "cas_manifest_full"
                    if incremental_since is None
                    else "cas_manifest_incremental"
                ),
                "incremental": incremental_since is not None,
                "incremental_since_ns": (
                    None if incremental_since is None else incremental_since.timestamp_ns
                ),
                "chain_bytes": chain_bytes,
                "size_bytes": manifest_path.stat().st_size,
            },
            root=str(root),
            timestamp_ns=marker_before_ns,
        )

    def _probe_changes_since_sync(self, marker_ns: int) -> bool:
        if self.container_runtime is not None:
            return _probe_container_changes_since(
                container_runtime=self.container_runtime,
                root=self.root,
                marker_ns=marker_ns,
            )
        return _any_file_newer_than(Path(self.root), marker_ns)


class OverlayBackend(SandboxBackend):
    """Docker overlay2 upperdir checkpoint backend for net-change manifests."""

    def __init__(
        self,
        *,
        root: str = "/testbed",
        checkpoint_dir: Path,
        container_id: str | None = None,
        container_executable: str = "docker",
        upperdir: Path | None = None,
    ) -> None:
        self.root = root
        self.checkpoint_dir = checkpoint_dir
        self.container_id = container_id
        self.container_executable = container_executable
        self._upperdir = upperdir

    async def start(self) -> None:
        self._resolve_upperdir()

    async def stop(self) -> None:
        return None

    async def execute(
        self,
        request: AgentTransportRequest,
        *,
        timeout_s: float | None = 600.0,
    ) -> AgentTransportResponse:
        raise NotImplementedError("overlay checkpoint backend does not execute tools")

    async def capture_snapshot(
        self,
        *,
        incremental_since: SandboxSnapshot | None = None,
    ) -> SandboxSnapshot:
        return self._capture_snapshot_sync(incremental_since)

    async def restore_snapshot(self, snapshot: SandboxSnapshot) -> None:
        raise RuntimeError("overlay checkpoint backend does not restore snapshots directly")

    async def probe_changes_since(self, marker_ns: int) -> bool:
        return self._probe_changes_since_sync(marker_ns)

    def _capture_snapshot_sync(
        self,
        incremental_since: SandboxSnapshot | None,
    ) -> SandboxSnapshot:
        marker_before_ns = time.time_ns()
        upper_root = self._upper_root()
        changed, deleted_paths, verified_blobs = self._scan_upperdir(
            upper_root=upper_root,
            since_ns=(
                None if incremental_since is None else incremental_since.timestamp_ns
            ),
        )
        manifest_path = self.checkpoint_dir / f"snapshot-{marker_before_ns}-manifest.json"
        chain_bytes = _write_container_manifest(
            manifest_path=manifest_path,
            changed=changed,
            deleted_paths=deleted_paths,
            verified_blobs=verified_blobs,
        )
        if self._probe_changes_since_sync(marker_before_ns):
            manifest_path.unlink(missing_ok=True)
            raise RuntimeError(
                "checkpoint failed: overlay upperdir changed during checkpoint"
            )
        manifest = _read_manifest(manifest_path)
        return SandboxSnapshot(
            process_state=None,
            disk_state={
                "manifest_path": str(manifest_path),
                "entries": manifest["entries"],
                "deleted_paths": manifest["deleted_paths"],
                "kind": (
                    "cas_manifest_full"
                    if incremental_since is None
                    else "cas_manifest_incremental"
                ),
                "incremental": incremental_since is not None,
                "incremental_since_ns": (
                    None if incremental_since is None else incremental_since.timestamp_ns
                ),
                "chain_bytes": chain_bytes,
                "size_bytes": manifest_path.stat().st_size,
            },
            root=self.root,
            timestamp_ns=marker_before_ns,
        )

    def _scan_upperdir(
        self,
        *,
        upper_root: Path,
        since_ns: int | None,
    ) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, bytes]]:
        if not upper_root.exists():
            return {}, [], {}
        changed: dict[str, dict[str, Any]] = {}
        deleted_paths: list[str] = []
        verified_blobs: dict[str, bytes] = {}
        walk_errors: list[OSError] = []

        def record_walk_error(exc: OSError) -> None:
            walk_errors.append(exc)

        for dirpath, dirnames, filenames in os.walk(
            upper_root,
            topdown=True,
            onerror=record_walk_error,
            followlinks=False,
        ):
            dirnames[:] = [name for name in dirnames if name != ".git"]
            dirnames.sort()
            filenames.sort()
            for name in dirnames:
                fpath = Path(dirpath) / name
                if not fpath.is_symlink():
                    continue
                try:
                    st = os.lstat(fpath)
                except OSError as exc:
                    raise OSError(
                        f"failed to stat overlay checkpoint entry {fpath}: {exc}"
                    ) from exc
                if since_ns is not None and st.st_mtime_ns <= since_ns:
                    continue
                rel = _checkpoint_relpath(str(fpath), upper_root)
                if _checkpoint_relpath_is_skipped(rel):
                    continue
                entry_type = _checkpoint_entry_type(fpath, st.st_mode)
                if (
                    not isinstance(entry_type, dict)
                    or entry_type.get("type") != "symlink"
                ):
                    raise OSError(
                        f"unsupported overlay checkpoint entry type: {fpath}"
                    )
                changed[rel] = dict(entry_type)
            for name in filenames:
                fpath = Path(dirpath) / name
                try:
                    st = os.lstat(fpath)
                except OSError as exc:
                    raise OSError(
                        f"failed to stat overlay checkpoint entry {fpath}: {exc}"
                    ) from exc
                if since_ns is not None and st.st_mtime_ns <= since_ns:
                    continue
                rel = _checkpoint_relpath(str(fpath), upper_root)
                if _checkpoint_relpath_is_skipped(rel):
                    continue
                if _is_overlay_whiteout(name, st):
                    deleted_paths.append(_whiteout_deleted_path(rel))
                    continue
                entry_type = _checkpoint_entry_type(fpath, st.st_mode)
                if isinstance(entry_type, dict):
                    if entry_type.get("type") != "symlink":
                        raise OSError(
                            f"unsupported overlay checkpoint entry type: {fpath}"
                        )
                    changed[rel] = dict(entry_type)
                    continue
                if entry_type != "file":
                    raise OSError(
                        f"unsupported overlay checkpoint entry type: {fpath}"
                    )
                try:
                    file_bytes = fpath.read_bytes()
                except OSError as exc:
                    raise OSError(
                        f"failed to read overlay checkpoint entry {fpath}: {exc}"
                    ) from exc
                digest = hashlib.sha256(file_bytes).hexdigest()
                changed[rel] = {
                    "hash": digest,
                    "mode": stat.S_IMODE(st.st_mode),
                    "size": st.st_size,
                    "mtime_ns": st.st_mtime_ns,
                }
                verified_blobs[rel] = file_bytes
        if walk_errors:
            raise OSError(
                f"failed to walk overlay checkpoint root {upper_root}: {walk_errors[0]}"
            )
        return changed, sorted(deleted_paths), verified_blobs

    def _probe_changes_since_sync(self, marker_ns: int) -> bool:
        upper_root = self._upper_root()
        if not upper_root.exists():
            raise RuntimeError(
                "OverlayBackend: upperdir not resolved — container may not exist"
            )
        return _any_overlay_entry_newer_than(upper_root, marker_ns)

    def _upper_root(self) -> Path:
        root_rel = self.root.lstrip("/")
        return self._resolve_upperdir() / root_rel

    def _resolve_upperdir(self) -> Path:
        if self._upperdir is not None:
            return self._upperdir
        if self.container_id is None:
            raise RuntimeError("overlay checkpoint backend requires a container_id")
        result = subprocess.run(
            [
                self.container_executable,
                "inspect",
                self.container_id,
                "--format",
                "{{json .GraphDriver}}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(
                "failed to inspect container graph driver "
                f"for overlay checkpoint: {detail}"
            )
        graph_driver = json.loads(result.stdout)
        name = graph_driver.get("Name")
        data = graph_driver.get("Data")
        if name != "overlay2" or not isinstance(data, dict):
            raise RuntimeError(
                "overlay checkpoint backend requires Docker overlay2; "
                f"container graph driver is {name!r}"
            )
        upperdir = data.get("UpperDir")
        if not isinstance(upperdir, str) or not upperdir:
            raise RuntimeError("overlay checkpoint backend could not resolve UpperDir")
        self._upperdir = Path(upperdir)
        return self._upperdir


class DockerBackend(SandboxBackend):
    """Docker/Podman-backed sandbox using the existing ContainerAgent protocol."""

    def __init__(
        self,
        *,
        source_image: str,
        fixed_image_name: str | None,
        agent_id: str,
        source_agent_id: str,
        manifest_index: int,
        task_output_dir: Path,
        container_executable: str,
        network_mode: str = "host",
        root: str = "/testbed",
        checkpoint_dir: Path | None = None,
        fixed_images_by_source: Mapping[str, str] | None = None,
        bootstrap_mount_args: Sequence[str] = (),
        agent_env_kwargs: Mapping[str, str] | None = None,
        startup_recorder: Any | None = None,
        ensure_fixed_image_fn: Callable[..., tuple[str, float]],
        start_task_container_fn: Callable[..., str],
        configure_apt_mirror_fn: Callable[..., dict[str, str] | None],
        stop_task_container_fn: Callable[..., Any],
        remove_image_fn: Callable[..., bool],
        copy_checkpoint_archive_to_container_fn: Callable[..., None] | None = None,
        restore_cas_manifest_in_container_fn: Callable[..., None] | None = None,
        checkpoint_backend: str | None = None,
    ) -> None:
        self.source_image = source_image
        self.fixed_image_name = fixed_image_name
        self.agent_id = agent_id
        self.source_agent_id = source_agent_id
        self.manifest_index = manifest_index
        self.task_output_dir = task_output_dir
        self.container_executable = container_executable
        self.network_mode = network_mode
        self.root = root
        self.checkpoint_dir = checkpoint_dir or task_output_dir / "checkpoints"
        self.fixed_images_by_source = fixed_images_by_source or {}
        self.bootstrap_mount_args = list(bootstrap_mount_args)
        self.agent_env_kwargs = dict(agent_env_kwargs or {})
        self.startup_recorder = startup_recorder
        self.ensure_fixed_image_fn = ensure_fixed_image_fn
        self.start_task_container_fn = start_task_container_fn
        self.configure_apt_mirror_fn = configure_apt_mirror_fn
        self.stop_task_container_fn = stop_task_container_fn
        self.remove_image_fn = remove_image_fn
        self.copy_checkpoint_archive_to_container_fn = (
            copy_checkpoint_archive_to_container_fn
        )
        self.restore_cas_manifest_in_container_fn = restore_cas_manifest_in_container_fn
        self.checkpoint_backend = validate_checkpoint_backend(checkpoint_backend)

        self._container_id: str | None = None
        self._agent: Any | None = None
        self._fixed_image: str | None = None
        self._cleanup_fixed_image = True
        self._checkpoint_snapshot_entries: dict[str, CheckpointSnapshotEntry] | None = (
            None
        )
        self._checkpoint_hash_cache: dict[str, CheckpointHashCacheEntry] = {}
        self._walk_checkpoint_backend: WalkBackend | None = None
        self._overlay_checkpoint_backend: OverlayBackend | None = None
        self._checkpoint_start_marker_ns: int | None = None

    @property
    def container_id(self) -> str:
        if self._container_id is None:
            raise RuntimeError("docker backend has no running container")
        return self._container_id

    @property
    def agent(self) -> Any:
        if self._agent is None:
            raise RuntimeError("docker backend agent has not started")
        return self._agent

    @property
    def fixed_image(self) -> str | None:
        return self._fixed_image

    @property
    def cleanup_fixed_image(self) -> bool:
        return self._cleanup_fixed_image

    async def start(self) -> None:
        from trace_collect.openclaw_tools import ContainerAgent

        try:
            fixed_name = await self._ensure_fixed_image()
            await self._start_container(fixed_name)
            await self._configure_apt_mirror()
            self._agent = ContainerAgent(
                self.container_id,
                self.container_executable,
                **self.agent_env_kwargs,
            )
            phase = self._start_phase("container_agent_start")
            try:
                await self._agent.start()
                self._finish_phase(phase)
            except (Exception, asyncio.CancelledError) as exc:
                self._finish_phase(phase, status="failed", error=exc)
                raise
            self._write_startup(status="success")
        except (Exception, asyncio.CancelledError) as exc:
            await self._cleanup_failed_start(exc)
            raise

    async def stop(self) -> None:
        agent_stop_error: BaseException | None = None
        container_stop_error: BaseException | None = None
        fixed_image_cleanup_error: BaseException | None = None
        container_stopped = False

        agent = self._agent
        self._agent = None
        if agent is not None:
            try:
                await agent.stop()
            except (Exception, asyncio.CancelledError) as exc:
                agent_stop_error = exc

        container_id = self._container_id
        if container_id is not None:
            try:
                await asyncio.to_thread(
                    self.stop_task_container_fn,
                    container_id,
                    executable=self.container_executable,
                )
                container_stopped = True
            except (Exception, asyncio.CancelledError) as exc:
                container_stop_error = exc
            if container_stopped:
                self._container_id = None
                self._checkpoint_start_marker_ns = None

        if container_stopped and self._fixed_image and self._cleanup_fixed_image:
            try:
                await asyncio.to_thread(
                    self.remove_image_fn,
                    self._fixed_image,
                    container_executable=self.container_executable,
                )
            except (Exception, asyncio.CancelledError) as exc:
                fixed_image_cleanup_error = exc

        for cleanup_error in (
            container_stop_error,
            agent_stop_error,
            fixed_image_cleanup_error,
        ):
            if cleanup_error is not None:
                raise cleanup_error

    async def execute(
        self,
        request: AgentTransportRequest,
        *,
        timeout_s: float | None = 600.0,
    ) -> AgentTransportResponse:
        raw_response = await self.agent.execute(
            {"tool": request.tool, "args": request.args},
            timeout_s=timeout_s,
        )
        if not isinstance(raw_response, dict):
            raise RuntimeError(f"container agent returned non-dict response: {raw_response!r}")
        return transport_response_from_agent_dict(raw_response)

    async def capture_snapshot(
        self,
        *,
        incremental_since: SandboxSnapshot | None = None,
    ) -> SandboxSnapshot:
        if self.checkpoint_backend == "verify":
            return await self._capture_verified_snapshot(incremental_since)
        return await self._active_checkpoint_backend().capture_snapshot(
            incremental_since=incremental_since,
        )

    async def restore_snapshot(self, snapshot: SandboxSnapshot) -> None:
        manifest_path = snapshot.disk_state.get("manifest_path")
        if not isinstance(manifest_path, str) or not manifest_path:
            raise ValueError("docker snapshot disk_state.manifest_path must be set")
        if self.copy_checkpoint_archive_to_container_fn is None:
            raise RuntimeError("docker backend restore has no manifest copy helper")
        if self.restore_cas_manifest_in_container_fn is None:
            raise RuntimeError("docker backend restore has no CAS restore helper")
        container_manifest_path = f"/tmp/agent_sched_manifest_{uuid.uuid4().hex}.json"
        await asyncio.to_thread(
            self.copy_checkpoint_archive_to_container_fn,
            checkpoint_path=Path(manifest_path),
            container_id=self.container_id,
            container_executable=self.container_executable,
            container_archive_path=container_manifest_path,
        )
        await asyncio.to_thread(
            self.restore_cas_manifest_in_container_fn,
            container_id=self.container_id,
            container_executable=self.container_executable,
            container_manifest_path=container_manifest_path,
            restore_root=snapshot.root,
        )

    async def probe_changes_since(self, marker_ns: int) -> bool:
        return await self._active_checkpoint_backend().probe_changes_since(marker_ns)

    def _start_phase(self, name: str) -> Any:
        if self.startup_recorder is None:
            return None
        return self.startup_recorder.start_phase(name)

    def _finish_phase(
        self,
        phase: Any,
        *,
        status: str = "success",
        error: BaseException | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if self.startup_recorder is None:
            return
        self.startup_recorder.finish_phase(
            phase,
            status=status,
            error=error,
            extra=extra,
        )

    def _write_startup(
        self,
        *,
        status: str,
        error: BaseException | None = None,
    ) -> None:
        if self.startup_recorder is None:
            return
        if error is None:
            self.startup_recorder.write(status=status)
        else:
            self.startup_recorder.write(status=status, error=error)

    async def _ensure_fixed_image(self) -> str:
        phase = self._start_phase("ensure_fixed_image")
        try:
            fixed_name = self.fixed_images_by_source.get(self.source_image)
            if fixed_name is not None:
                self._cleanup_fixed_image = False
                extra = {
                    "fixed_image": fixed_name,
                    "reported_elapsed_s": 0.0,
                    "prebuilt": True,
                }
            else:
                if self.fixed_image_name is None:
                    raise ValueError("fixed_image_name is required without prebuild")
                fixed_name, elapsed_s = await asyncio.to_thread(
                    self.ensure_fixed_image_fn,
                    self.source_image,
                    container_executable=self.container_executable,
                    fixed_image_name=self.fixed_image_name,
                    rebuild=True,
                )
                extra = {
                    "fixed_image": fixed_name,
                    "reported_elapsed_s": elapsed_s,
                    "prebuilt": False,
                }
            self._fixed_image = fixed_name
            if self.startup_recorder is not None:
                self.startup_recorder.fixed_image = fixed_name
            self._finish_phase(phase, extra=extra)
            return fixed_name
        except (Exception, asyncio.CancelledError) as exc:
            self._finish_phase(phase, status="failed", error=exc)
            raise

    async def _start_container(self, fixed_name: str) -> None:
        phase = self._start_phase("start_task_container")
        try:
            extra_args = [
                "--label",
                "agent-sched-bench.component=simulate-replay",
                "--label",
                f"agent-sched-bench.run_instance_id={self.agent_id}",
                "--label",
                f"agent-sched-bench.source_agent_id={self.source_agent_id}",
                "--label",
                f"agent-sched-bench.manifest_index={self.manifest_index}",
                "--label",
                f"agent-sched-bench.output_dir={self.task_output_dir}",
                "-v",
                f"{_checkpoint_cas_root()}:{_checkpoint_cas_root()}",
                *self.bootstrap_mount_args,
            ]
            self._container_id = await asyncio.to_thread(
                self.start_task_container_fn,
                fixed_name,
                executable=self.container_executable,
                extra_args=extra_args,
                network_mode=self.network_mode,
            )
            self._checkpoint_start_marker_ns = time.time_ns()
            if self.startup_recorder is not None:
                self.startup_recorder.container_id = self._container_id
            self._finish_phase(phase, extra={"container_id": self._container_id})
        except (Exception, asyncio.CancelledError) as exc:
            self._finish_phase(phase, status="failed", error=exc)
            raise

    async def _configure_apt_mirror(self) -> None:
        phase = self._start_phase("configure_apt_mirror")
        try:
            mirror_info = await asyncio.to_thread(
                self.configure_apt_mirror_fn,
                self.container_id,
                executable=self.container_executable,
            )
            mirror_status = (
                "skipped"
                if mirror_info is None or mirror_info.get("configured") == "false"
                else "success"
            )
            self._finish_phase(
                phase,
                status=mirror_status,
                extra=mirror_info or {"reason": "TASK_CONTAINER_APT_MIRROR unset"},
            )
        except (Exception, asyncio.CancelledError) as exc:
            self._finish_phase(phase, status="failed", error=exc)
            raise

    async def _cleanup_failed_start(self, original: BaseException) -> None:
        cleanup_errors: list[BaseException] = []
        try:
            self._write_startup(status="failed", error=original)
        except (Exception, asyncio.CancelledError):
            pass
        if self._agent is not None:
            try:
                await self._agent.stop()
            except (Exception, asyncio.CancelledError):
                pass
            self._agent = None
        container_stopped = False
        if self._container_id is not None:
            try:
                await asyncio.to_thread(
                    self.stop_task_container_fn,
                    self._container_id,
                    executable=self.container_executable,
                )
                container_stopped = True
            except (Exception, asyncio.CancelledError) as exc:
                cleanup_errors.append(exc)
        if container_stopped:
            self._container_id = None
            self._checkpoint_start_marker_ns = None
        if container_stopped and self._fixed_image and self._cleanup_fixed_image:
            try:
                await asyncio.to_thread(
                    self.remove_image_fn,
                    self._fixed_image,
                    container_executable=self.container_executable,
                )
            except (Exception, asyncio.CancelledError) as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            cleanup_error = cleanup_errors[0]
            if cleanup_error is not original:
                cleanup_error.__context__ = original
            raise cleanup_error

    def _container_runtime(self) -> dict[str, str]:
        return {"id": self.container_id, "executable": self.container_executable}

    def _active_checkpoint_backend(self) -> SandboxBackend:
        if self.checkpoint_backend == "walk":
            return self._walk_backend()
        if self.checkpoint_backend == "overlay":
            return self._overlay_backend()
        raise RuntimeError(f"unsupported active checkpoint backend: {self.checkpoint_backend}")

    def _walk_backend(self) -> WalkBackend:
        container_runtime = self._container_runtime() if self._container_id is not None else None
        if (
            self._walk_checkpoint_backend is None
            or self._walk_checkpoint_backend.container_runtime != container_runtime
        ):
            self._walk_checkpoint_backend = WalkBackend(
                root=self.root,
                checkpoint_dir=self.checkpoint_dir,
                container_runtime=container_runtime,
            )
        return self._walk_checkpoint_backend

    def _overlay_backend(self) -> OverlayBackend:
        if self._container_id is None:
            raise RuntimeError("overlay checkpoint backend requires a running container")
        if (
            self._overlay_checkpoint_backend is None
            or self._overlay_checkpoint_backend.container_id != self.container_id
        ):
            self._overlay_checkpoint_backend = OverlayBackend(
                root=self.root,
                checkpoint_dir=self.checkpoint_dir,
                container_id=self.container_id,
                container_executable=self.container_executable,
            )
        return self._overlay_checkpoint_backend

    async def _capture_verified_snapshot(
        self,
        incremental_since: SandboxSnapshot | None,
    ) -> SandboxSnapshot:
        effective_incremental_since = incremental_since
        if effective_incremental_since is None:
            marker_ns = self._checkpoint_start_marker_ns
            if marker_ns is None:
                raise RuntimeError(
                    "verify checkpoint backend requires a container start marker"
                )
            effective_incremental_since = SandboxSnapshot(
                process_state=None,
                disk_state={},
                root=self.root,
                timestamp_ns=marker_ns,
            )
        walk_snapshot, overlay_snapshot = await _capture_backends_for_verification(
            self._walk_backend(),
            self._overlay_backend(),
            incremental_since=effective_incremental_since,
        )
        differences = checkpoint_manifest_differences(walk_snapshot, overlay_snapshot)
        if differences:
            raise RuntimeError(
                "checkpoint backend verification failed: "
                f"{json.dumps(differences, sort_keys=True)}"
            )
        return walk_snapshot


def _read_manifest_entries(manifest_path: Path) -> dict[str, Any]:
    return _read_manifest(manifest_path)["entries"]


def _read_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("entries")
    if not isinstance(entries, dict):
        raise RuntimeError(f"checkpoint manifest missing entries: {manifest_path}")
    deleted_paths = manifest.get("deleted_paths", [])
    if not isinstance(deleted_paths, list):
        raise RuntimeError(f"checkpoint manifest deleted_paths must be a list: {manifest_path}")
    return {"entries": entries, "deleted_paths": deleted_paths}


def _checkpoint_cas_root() -> Path:
    return Path.home() / ".cache" / "agent-checkpoint-cas"


# Number of host network slots for auto-allocated FC instances.  Each slot
# is a /24 inside 172.16.0.0/12 (172.16.x.y .. 172.31.x.y), giving 4096
# concurrent VMs per host before allocation fails loudly.
_FC_NET_SLOTS = 4096


class FCBackend(SandboxBackend):
    """Firecracker microVM sandbox with vsock agent transport.

    Lifecycle
    ---------
    ``start`` builds a rootfs from a Docker image, sets up a TAP device
    with NAT, launches a Firecracker process, configures the VM via the
    FC API socket, and polls the vsock agent until it is ready.

    ``stop`` kills the Firecracker process, tears down the TAP device
    and iptables rules, and removes the per-instance working rootfs.
    The guest serial console is kept in a per-instance log file
    (``/tmp/fc-<id>-console.log``) for post-mortem debugging.

    Networking
    ----------
    When ``tap_dev``/``host_ip``/``guest_ip`` are omitted, ``start``
    claims a collision-free host network slot (a /24 inside
    172.16.0.0/12) using TAP-device creation as the atomic host-wide
    lock, supporting up to 4096 concurrent VMs per host.  The slot stays
    claimed across snapshot restores and is released by ``stop``.

    Agent transport
    ---------------
    Every ``execute`` call opens a UDS connection to the Firecracker
    vsock host socket, performs the ``CONNECT <port>`` handshake, sends
    one JSON-line request, and reads one JSON-line response.  The guest
    agent speaks the same protocol as the container-based
    ``ContainerAgent`` used by ``DockerBackend``.

    Snapshots
    ---------
    Atomic paired (memory, disk) snapshots capture VM state at the same
    pause point.  Memory is captured via the Firecracker snapshot API
    (always ``Full``; Diff+rebase via snapshot-editor is future work, so
    every memory snapshot currently costs the full guest memory size on
    disk).  Disk is a copy of the backing rootfs file — copy-on-write via
    ``cp --reflink=always`` on XFS/btrfs, with a logged fallback to a
    full sparse copy on filesystems without reflink support (e.g. ext4).
    The guest is quiesced with ``sync`` before pause to flush journals
    and page cache.

    Cadence
    -------
    ``paired`` captures both memory and disk on every checkpoint.
    ``disk_always`` skips the memory snapshot (much faster, but restore
    is a cold boot from the disk snapshot).  ``disk_memory_mixed``
    captures memory every *memory_snapshot_interval* checkpoints and disk
    on every checkpoint.

    Restore
    -------
    Recovery is VM re-instantiation: a new FC process is launched with
    the paired disk snapshot as its root drive loaded via the FC snapshot
    API.  Measured end-to-end restore latency is ~1 s for a 1 GiB-memory
    VM on local NVMe (dominated by the disk copy and memory-file load).

    Probe
    -----
    ``probe_changes_since`` compares the current per-drive write-byte
    counters (from the FC metrics endpoint) against the value recorded at
    the snapshot timestamp — an O(1) signal that avoids walking the
    filesystem.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        source_image: str,
        kernel_path: Path,
        checkpoint_dir: Path | None = None,
        instance_id: str | None = None,
        api_sock: str | None = None,
        vsock_sock: str | None = None,
        vsock_port: int = 5678,
        vcpu_count: int = 2,
        mem_size_mib: int = 1024,
        tap_dev: str | None = None,
        host_ip: str | None = None,
        guest_ip: str | None = None,
        netmask: int = 24,
        root: str = "/testbed",
        fc_binary: str = "firecracker",
        container_executable: str = "docker",
        checkpoint_cadence: str = "paired",
        memory_snapshot_interval: int = 10,
    ) -> None:
        _VALID_CADENCES = frozenset({"paired", "disk_always", "disk_memory_mixed"})
        if checkpoint_cadence not in _VALID_CADENCES:
            raise ValueError(
                f"checkpoint_cadence must be one of {sorted(_VALID_CADENCES)}, "
                f"got {checkpoint_cadence!r}"
            )

        self._source_image = source_image
        self._kernel_path = kernel_path
        self._checkpoint_dir = checkpoint_dir or Path("/tmp/fc-checkpoints")

        # Per-instance uniqueness derived from instance_id.
        self._instance_id = instance_id or uuid.uuid4().hex[:8]
        hex_id = self._instance_id[:8]

        self._api_sock = api_sock or f"/tmp/fc-{self._instance_id}.sock"
        self._vsock_sock = vsock_sock or f"/tmp/fc-{self._instance_id}-vsock.sock"
        self._metrics_path = f"/tmp/fc-{self._instance_id}-metrics.json"
        self._console_log_path = f"/tmp/fc-{self._instance_id}-console.log"
        self._vsock_port = vsock_port
        self._vcpu_count = vcpu_count
        self._mem_size_mib = mem_size_mib
        # Networking: explicit values pin the tap/subnet (caller owns
        # uniqueness); otherwise a collision-free slot is claimed in
        # _setup_tap_and_nat using tap-device creation as the lock.  The
        # hash of the instance id only seeds the probe order.
        if (tap_dev is None) != (host_ip is None) or (host_ip is None) != (
            guest_ip is None
        ):
            raise ValueError(
                "tap_dev, host_ip, and guest_ip must be given together "
                "(or all omitted for automatic allocation)"
            )
        self._tap_dev = tap_dev
        self._host_ip = host_ip
        self._guest_ip = guest_ip
        self._net_index_hint = int(hex_id, 16) % _FC_NET_SLOTS
        self._guest_cid = 3 + (int(hex_id, 16) % 65533)
        self._guest_mac = (
            f"AA:FC:{hex_id[0:2]}:{hex_id[2:4]}:{hex_id[4:6]}:{hex_id[6:8]}"
        )
        self._netmask = netmask
        self.root = root
        self._fc_binary = fc_binary
        self._container_executable = container_executable
        self._checkpoint_cadence = checkpoint_cadence
        self._memory_snapshot_interval = max(1, memory_snapshot_interval)

        self._process: subprocess.Popen[bytes] | None = None
        self._rootfs_path: Path | None = None
        self._snapshot_counter: int = 0
        self._mem_version: int = 0
        self._disk_version: int = 0
        self._last_write_bytes: int | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if not Path("/dev/kvm").exists():
            raise RuntimeError(
                "FCBackend requires /dev/kvm; this host has no KVM support"
            )
        if not Path("/dev/kvm").is_char_device():
            raise RuntimeError("/dev/kvm exists but is not a character device")

        from harness.fc_rootfs_builder import build_fc_rootfs

        rootfs_cache = await asyncio.to_thread(
            build_fc_rootfs,
            docker_image=self._source_image,
            container_executable=self._container_executable,
        )

        # Per-instance working copy — the cache is read-only source material.
        self._rootfs_path = Path(f"/tmp/fc-rootfs-{self._instance_id}.ext4")
        await asyncio.to_thread(
            _cow_copy,
            str(rootfs_cache),
            str(self._rootfs_path),
        )

        try:
            await asyncio.to_thread(self._setup_tap_and_nat)
            await asyncio.to_thread(self._launch_fc)
            await asyncio.to_thread(self._configure_vm)

            await asyncio.to_thread(
                self._api_put, "/actions", {"action_type": "InstanceStart"},
            )

            ready = await asyncio.to_thread(
                self._poll_vsock_ready,
                timeout_s=60.0,
            )
            if not ready:
                raise RuntimeError(
                    "FCBackend: vsock agent did not become ready within 60s"
                )
        except BaseException:
            # Release the VM process, sockets, TAP, and NAT rules so a
            # failed start never leaks host resources.
            with contextlib.suppress(OSError, RuntimeError):
                await self.stop()
            raise

        self._last_write_bytes = await asyncio.to_thread(self._read_write_bytes)

    async def stop(self) -> None:
        error: BaseException | None = None

        # 1. Halt the VM.
        process = self._process
        if process is not None and process.poll() is None:
            try:
                await asyncio.to_thread(self._stop_vm)
            except (OSError, RuntimeError) as exc:
                error = exc
        self._process = None

        # 2. Clean up API socket.
        if os.path.exists(self._api_sock):
            try:
                os.unlink(self._api_sock)
            except OSError:
                pass

        # 3. Tear down TAP + iptables.
        try:
            await asyncio.to_thread(self._teardown_tap_and_nat)
        except (OSError, RuntimeError) as exc:
            if error is None:
                error = exc

        # 4. Remove the per-instance working rootfs copy — restore
        # recreates it from the snapshot disk, and a final stop must not
        # leave multi-GB images behind.
        if self._rootfs_path is not None:
            with contextlib.suppress(OSError):
                self._rootfs_path.unlink()
        self._rootfs_path = None
        self._last_write_bytes = None

        if error is not None:
            raise error

    # ------------------------------------------------------------------
    # Agent transport
    # ------------------------------------------------------------------

    async def execute(
        self,
        request: AgentTransportRequest,
        *,
        timeout_s: float | None = 600.0,
    ) -> AgentTransportResponse:
        raw_response = await asyncio.to_thread(
            self._execute_vsock,
            {"tool": request.tool, "args": request.args},
            timeout_s,
        )
        if not isinstance(raw_response, dict):
            raise RuntimeError(
                f"FC agent returned non-dict response: {raw_response!r}"
            )
        return transport_response_from_agent_dict(raw_response)

    # ------------------------------------------------------------------
    # Snapshots — atomic paired (memory, disk) capture / restore
    # ------------------------------------------------------------------

    async def capture_snapshot(
        self,
        *,
        incremental_since: SandboxSnapshot | None = None,
    ) -> SandboxSnapshot:
        """Atomic paired snapshot at a single VM pause point.

        1. Quiesce guest (``sync`` via vsock).
        2. Pause the VM.
        3. If cadence calls for memory: capture via FC snapshot API
           (Full only — Diff+rebase is future work via snapshot-editor).
        4. Disk: CoW copy of the backing rootfs file.
        5. Resume the VM.

        Returns a ``SandboxSnapshot`` with ``process_state`` holding the
        memory-snapshot metadata (or ``None`` for disk-only cadences) and
        ``disk_state`` holding the disk-snapshot path and version counters.
        """
        del incremental_since  # FC paired snapshots are always full-state
        timestamp_ns = time.time_ns()
        snap_index = self._snapshot_counter
        self._snapshot_counter += 1

        should_capture_mem = self._mem_cadence_enabled(snap_index)

        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 1. Quiesce guest — flush journals and page cache.
        await self._quiesce_guest()

        # 2. Pause the VM so memory and disk are captured at the same point.
        await asyncio.to_thread(self._api_patch, "/vm", {"state": "Paused"})

        try:
            vmstate_path: str | None = None
            mem_path: str | None = None
            disk_snap_path: str | None = None

            # 3. Memory snapshot (always Full; Diff+rebase is future work).
            if should_capture_mem:
                vmstate_path = str(
                    self._checkpoint_dir / f"snap-{snap_index:04d}-vmstate.snap"
                )
                mem_path = str(
                    self._checkpoint_dir / f"snap-{snap_index:04d}-mem.snap"
                )
                await asyncio.to_thread(
                    self._api_put,
                    "/snapshot/create",
                    {
                        "snapshot_type": "Full",
                        "snapshot_path": vmstate_path,
                        "mem_file_path": mem_path,
                    },
                )
                self._mem_version += 1

            # 4. Disk snapshot — CoW copy of the backing rootfs file
            # (full sparse copy with a logged warning on non-reflink
            # filesystems; see _cow_copy).
            assert self._rootfs_path is not None
            disk_snap_path = str(
                self._checkpoint_dir / f"snap-{snap_index:04d}-disk.img"
            )
            await asyncio.to_thread(
                _cow_copy,
                str(self._rootfs_path),
                disk_snap_path,
            )
            self._disk_version += 1

        finally:
            # 5. Resume.
            await asyncio.to_thread(
                self._api_patch, "/vm", {"state": "Resumed"},
            )

        write_bytes = await asyncio.to_thread(self._read_write_bytes)
        self._last_write_bytes = write_bytes

        process_state: dict[str, Any] | None = None
        if mem_path is not None and vmstate_path is not None:
            process_state = {
                "vmstate_path": vmstate_path,
                "mem_path": mem_path,
                "snapshot_type": "Full",
                "mem_version": self._mem_version,
            }

        return SandboxSnapshot(
            process_state=process_state,
            disk_state={
                "disk_path": disk_snap_path,
                "backing_path": str(self._rootfs_path),
                "write_bytes": write_bytes,
                "kind": "fc_paired_snapshot",
                "disk_version": self._disk_version,
                "mem_version": (
                    self._mem_version if should_capture_mem else None
                ),
                "checkpoint_cadence": self._checkpoint_cadence,
            },
            root=self.root,
            timestamp_ns=timestamp_ns,
        )

    async def restore_snapshot(self, snapshot: SandboxSnapshot) -> bool:
        """Restore from an atomic paired snapshot.

        Design:

        **With memory snapshot**:
        1. Halt the current VM process (the TAP/NAT slot stays claimed).
        2. Copy snapshot disk OVER the working rootfs (so the vmstate's
           drive backing path resolves).
        3. Launch a fresh FC process.
        4. ``PUT /snapshot/load`` immediately — NO other config calls
           (FC forbids configure-then-load).
        5. Poll vsock, fix clock, configure metrics.

        **Without memory (cold boot from disk snapshot)**:
        1. Halt the current VM process (the TAP/NAT slot stays claimed).
        2. Copy snapshot disk over the working rootfs.
        3. Launch fresh FC process.
        4. Full ``_configure_vm`` + ``InstanceStart`` (cold boot).
        5. Poll vsock, fix clock.

        After restore ``self._rootfs_path`` still points to the working
        copy (now overwritten with snapshot content), so a subsequent
        ``capture_snapshot`` will work.
        """
        disk_path = snapshot.disk_state.get("disk_path")
        if not isinstance(disk_path, str) or not disk_path:
            raise ValueError(
                "FC snapshot disk_state.disk_path must be a non-empty string"
            )
        if not Path(disk_path).exists():
            raise FileNotFoundError(
                f"FC snapshot disk image not found: {disk_path}"
            )

        mem_path: str | None = None
        vmstate_path: str | None = None
        if snapshot.process_state is not None:
            maybe_mem = snapshot.process_state.get("mem_path")
            if isinstance(maybe_mem, str) and maybe_mem and Path(maybe_mem).exists():
                mem_path = maybe_mem
            maybe_vmstate = snapshot.process_state.get("vmstate_path")
            if isinstance(maybe_vmstate, str) and maybe_vmstate and Path(maybe_vmstate).exists():
                vmstate_path = maybe_vmstate

        # 1. Halt the current VM only — the TAP/NAT slot stays claimed by
        # this instance, so no other instance can steal it mid-restore.
        process = self._process
        if process is not None and process.poll() is None:
            await asyncio.to_thread(self._stop_vm)
        self._process = None
        if os.path.exists(self._api_sock):
            with contextlib.suppress(OSError):
                os.unlink(self._api_sock)

        # 2. Ensure working rootfs exists, then overwrite with snapshot disk.
        if self._rootfs_path is None:
            self._rootfs_path = Path(f"/tmp/fc-rootfs-{self._instance_id}.ext4")
        await asyncio.to_thread(
            _cow_copy,
            disk_path,
            str(self._rootfs_path),
        )

        try:
            # 3. Set up networking (idempotent for a held slot) and launch
            # a fresh FC process.
            await asyncio.to_thread(self._setup_tap_and_nat)
            await asyncio.to_thread(self._launch_fc)

            if mem_path is not None and vmstate_path is not None:
                # Restore with memory — configure metrics before load
                # (FC forbids /metrics after the VM starts).
                Path(self._metrics_path).parent.mkdir(parents=True, exist_ok=True)
                Path(self._metrics_path).touch()
                await asyncio.to_thread(
                    self._api_put,
                    "/metrics",
                    {"metrics_path": self._metrics_path},
                )
                await asyncio.to_thread(
                    self._api_put,
                    "/snapshot/load",
                    {
                        "snapshot_path": vmstate_path,
                        "mem_file_path": mem_path,
                        "resume_vm": True,
                    },
                )
            else:
                # Cold boot — configure VM and start fresh.
                await asyncio.to_thread(self._configure_vm)
                await asyncio.to_thread(
                    self._api_put, "/actions", {"action_type": "InstanceStart"},
                )

            # 5. Wait for vsock agent.
            ready = await asyncio.to_thread(
                self._poll_vsock_ready,
                timeout_s=60.0,
            )
            if not ready:
                raise RuntimeError(
                    "FCBackend: vsock agent did not become ready after "
                    "snapshot restore"
                )
        except BaseException:
            # A half-restored VM is unusable — release everything rather
            # than leak the FC process, sockets, and network slot.
            with contextlib.suppress(OSError, RuntimeError):
                await self.stop()
            raise

        # 6. Fix guest clock.
        await self._fix_guest_clock()

        # 7. Record baseline write bytes for change detection.
        self._last_write_bytes = await asyncio.to_thread(self._read_write_bytes)
        return True

    # ------------------------------------------------------------------
    # Change probe
    # ------------------------------------------------------------------

    async def probe_changes_since(self, marker_ns: int) -> bool:
        del marker_ns  # not used; we compare write counters
        if self._last_write_bytes is None:
            return True  # no baseline → assume changed
        current = await asyncio.to_thread(self._read_write_bytes)
        return current != self._last_write_bytes

    # ------------------------------------------------------------------
    # Internal — TAP and networking
    # ------------------------------------------------------------------

    def _allocate_net_slot(self) -> None:
        """Claim a collision-free tap/subnet slot for this instance.

        Uses TAP-device creation as the host-wide lock: ``ip tuntap add``
        fails when the device already exists, so a successful add claims
        the slot atomically.  The instance-id hash only seeds the probe
        order; slots are probed linearly from there.
        """
        for offset in range(_FC_NET_SLOTS):
            idx = (self._net_index_hint + offset) % _FC_NET_SLOTS
            tap_name = f"fc-tap{idx}"
            result = _checked_run(
                ["sudo", "ip", "tuntap", "add", tap_name, "mode", "tap"],
                check=False,
                timeout=30,
            )
            if result.returncode == 0:
                self._tap_dev = tap_name
                self._host_ip = f"172.{16 + idx // 256}.{idx % 256}.1"
                self._guest_ip = f"172.{16 + idx // 256}.{idx % 256}.2"
                return
            # `ip link show` needs no privileges, so a non-zero exit here
            # means the device genuinely does not exist — the add failed
            # for a non-collision reason (permissions, kernel limits).
            exists = _checked_run(
                ["ip", "link", "show", tap_name],
                check=False,
                timeout=30,
            )
            if exists.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(
                    f"failed to create TAP device {tap_name}: {detail}"
                )
            # Device exists (another instance owns the slot) — probe on.
        raise RuntimeError(
            f"no free FC network slot among {_FC_NET_SLOTS} candidates"
        )

    def _setup_tap_and_nat(self) -> None:
        """Create TAP device, enable NAT + forwarding."""
        if self._tap_dev is None:
            self._allocate_net_slot()
        else:
            # Explicit or previously-allocated slot: (re)create the TAP.
            # check=False keeps re-entry idempotent when the device is
            # still present (e.g. explicit tap reused across runs).
            _checked_run(
                ["sudo", "ip", "tuntap", "add", self._tap_dev, "mode", "tap"],
                check=False,
                timeout=30,
            )
        # Delete-then-add makes the address assignment idempotent for a
        # held slot (restore path) while still failing fast on real
        # errors (missing device, permission loss).
        _checked_run(
            [
                "sudo", "ip", "addr", "del",
                f"{self._host_ip}/{self._netmask}",
                "dev", self._tap_dev,
            ],
            check=False,
            timeout=30,
        )
        _checked_run(
            [
                "sudo", "ip", "addr", "add",
                f"{self._host_ip}/{self._netmask}",
                "dev", self._tap_dev,
            ],
            check=True,
            timeout=30,
        )
        _checked_run(
            ["sudo", "ip", "link", "set", self._tap_dev, "up"],
            check=True,
            timeout=30,
        )
        _checked_run(
            ["sudo", "sysctl", "-w", "net.ipv4.ip_forward=1"],
            check=False,
            timeout=30,
        )
        # Delete stale MASQUERADE rule, then add.
        _checked_run(
            [
                "sudo", "iptables", "-t", "nat", "-D", "POSTROUTING",
                "-s", f"{self._guest_ip}/32", "-j", "MASQUERADE",
            ],
            check=False,
            timeout=30,
        )
        _checked_run(
            [
                "sudo", "iptables", "-t", "nat", "-A", "POSTROUTING",
                "-s", f"{self._guest_ip}/32", "-j", "MASQUERADE",
            ],
            check=False,
            timeout=30,
        )

    def _teardown_tap_and_nat(self) -> None:
        """Remove NAT rules and TAP device (releases the network slot)."""
        if self._tap_dev is None:
            return
        _checked_run(
            [
                "sudo", "iptables", "-t", "nat", "-D", "POSTROUTING",
                "-s", f"{self._guest_ip}/32", "-j", "MASQUERADE",
            ],
            check=False,
            timeout=30,
        )
        _checked_run(
            ["sudo", "ip", "link", "delete", self._tap_dev],
            check=False,
            timeout=30,
        )

    # ------------------------------------------------------------------
    # Internal — Firecracker process
    # ------------------------------------------------------------------

    def _launch_fc(self) -> None:
        """Start the firecracker binary and wait for the API socket."""
        if os.path.exists(self._api_sock):
            os.unlink(self._api_sock)
        # Clean up stale vsock UDS from a prior VM instance to avoid
        # FC snapshot-load errors ("Error binding to …").
        if os.path.exists(self._vsock_sock):
            os.unlink(self._vsock_sock)
        # Truncate stale metrics from prior VM instance.
        Path(self._metrics_path).write_text("")

        # FC stdout carries the guest serial console (console=ttyS0) —
        # keep it in a per-instance log for post-mortem debugging.  The
        # child inherits its own duplicated fd, so closing the parent's
        # handle immediately is safe as long as nothing ever calls
        # communicate() on this process (only poll/kill/wait are used).
        console_log = open(self._console_log_path, "ab")
        try:
            self._process = subprocess.Popen(
                [self._fc_binary, "--api-sock", self._api_sock],
                stdout=console_log,
                stderr=console_log,
            )
        finally:
            console_log.close()

        deadline = time.monotonic() + 10.0
        while not os.path.exists(self._api_sock):
            if time.monotonic() > deadline:
                self._process.kill()
                self._process.wait()
                self._process = None
                raise TimeoutError(
                    "Firecracker API socket did not appear within 10s"
                )
            if self._process.poll() is not None:
                rc = self._process.returncode
                self._process = None
                raise RuntimeError(
                    f"Firecracker exited early (rc={rc})"
                )
            time.sleep(0.1)

    def _configure_vm(self) -> None:
        """Push machine-config, boot-source, root drive, net iface, vsock, and metrics."""
        assert self._rootfs_path is not None
        self._api_put(
            "/machine-config",
            {
                "vcpu_count": self._vcpu_count,
                "mem_size_mib": self._mem_size_mib,
                "track_dirty_pages": True,
            },
        )
        self._api_put(
            "/boot-source",
            {
                "kernel_image_path": str(self._kernel_path),
                "boot_args": (
                    "console=ttyS0 reboot=k panic=1 pci=off "
                    "root=/dev/vda rw quiet "
                    f"guest_ip={self._guest_ip} "
                    f"host_ip={self._host_ip} "
                    f"netmask_len={self._netmask}"
                ),
            },
        )
        self._api_put(
            "/drives/rootfs",
            {
                "drive_id": "rootfs",
                "path_on_host": str(self._rootfs_path),
                "is_root_device": True,
                "is_read_only": False,
            },
        )
        self._api_put(
            "/network-interfaces/eth0",
            {
                "iface_id": "eth0",
                "guest_mac": self._guest_mac,
                "host_dev_name": self._tap_dev,
            },
        )
        self._api_put(
            "/vsock",
            {
                "vsock_id": "vsock0",
                "guest_cid": self._guest_cid,
                "uds_path": self._vsock_sock,
            },
        )
        # Firecracker requires the metrics file to exist before PUT /metrics.
        Path(self._metrics_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self._metrics_path).touch()
        self._api_put(
            "/metrics",
            {"metrics_path": self._metrics_path},
        )

    def _stop_vm(self) -> None:
        """Kill the Firecracker process.

        The custom init script has no Ctrl+Alt+Del handler, so a
        graceful-shutdown signal can never work — the VMM is killed
        directly.  Disk consistency does not depend on this path:
        snapshots quiesce + pause before capture, and the working rootfs
        is discarded (stop) or overwritten from a snapshot (restore)."""
        if self._process is None or self._process.poll() is not None:
            return
        self._process.kill()
        self._process.wait(timeout=5.0)

    # ------------------------------------------------------------------
    # Internal — snapshot helpers
    # ------------------------------------------------------------------

    def _mem_cadence_enabled(self, snap_index: int) -> bool:
        """Return ``True`` when a memory snapshot should be taken at
        *snap_index*."""
        cadence = self._checkpoint_cadence
        if cadence == "paired":
            return True
        if cadence == "disk_always":
            return False
        # disk_memory_mixed — memory every N turns.
        return snap_index % self._memory_snapshot_interval == 0

    async def _quiesce_guest(self) -> None:
        """Send ``sync`` to the guest via vsock before pausing the VM."""
        try:
            await asyncio.to_thread(
                self._execute_vsock,
                {"tool": "exec", "args": {"command": "sync"}},
                10.0,
            )
        except (ConnectionError, RuntimeError):
            # Best-effort — the guest may not have /bin/sync, or the
            # vsock agent may not support exec.  The snapshot is still
            # consistent because the VM is paused; we just lose the
            # journal/page-cache flush.
            pass

    async def _fix_guest_clock(self) -> None:
        """Advance the guest clock after a snapshot restore.

        When a VM is paused and later resumed, the guest clock is behind
        by the pause duration.  We set it to the current host time via
        ``date -s``.
        """
        current_utc = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        try:
            await asyncio.to_thread(
                self._execute_vsock,
                {
                    "tool": "exec",
                    "args": {"command": f"date -s '{current_utc}'"},
                },
                10.0,
            )
        except (ConnectionError, RuntimeError):
            # Best-effort — clock skew is tolerable for most workloads.
            pass

    # ------------------------------------------------------------------
    # Internal — FC API over Unix socket
    # ------------------------------------------------------------------

    def _api_request(
        self, method: str, path: str, body: str | None = None,
    ) -> tuple[int, str]:
        """Send an HTTP request over the FC API socket.

        Returns ``(status_code, response_body)``.

        The FC API server keeps connections alive and ignores
        ``Connection: close``, so the response end is determined from the
        status line and ``Content-Length`` header — never by waiting for
        the server to close the socket (which would block until the
        socket timeout on every call).
        """
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(10)
        try:
            sock.connect(self._api_sock)
            headers = f"{method} {path} HTTP/1.1\r\nHost: localhost\r\n"
            if body is not None:
                body_bytes = body.encode("utf-8")
                headers += (
                    f"Content-Type: application/json\r\n"
                    f"Content-Length: {len(body_bytes)}\r\n"
                )
            else:
                body_bytes = b""
            headers += "\r\n"
            sock.sendall(headers.encode("utf-8") + body_bytes)

            raw = b""
            while b"\r\n\r\n" not in raw:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                raw += chunk
            header_bytes, _, body_rest = raw.partition(b"\r\n\r\n")

            content_length: int | None = None
            for header_line in header_bytes.split(b"\r\n")[1:]:
                name, _, value = header_line.partition(b":")
                if name.strip().lower() == b"content-length":
                    try:
                        content_length = int(value.strip())
                    except ValueError:
                        content_length = None
                    break

            status_line = header_bytes.split(b"\r\n")[0].decode(
                "utf-8", errors="replace",
            )
            try:
                status_code = int(status_line.split(" ")[1])
            except (IndexError, ValueError):
                status_code = 0

            # The FC API contract: every response carries Content-Length
            # except bodyless statuses.  Fail fast on anything else
            # rather than guessing the body length.
            if content_length is None:
                if status_code in (204, 304):
                    content_length = 0
                else:
                    raise RuntimeError(
                        f"FC API {method} {path}: response status "
                        f"{status_code} without Content-Length header"
                    )

            while len(body_rest) < content_length:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                body_rest += chunk
            if len(body_rest) < content_length:
                raise RuntimeError(
                    f"FC API {method} {path}: truncated response body "
                    f"({len(body_rest)}/{content_length} bytes)"
                )
        finally:
            sock.close()

        return status_code, body_rest.decode("utf-8", errors="replace")

    def _api_put(self, path: str, data: dict[str, Any]) -> None:
        body = json.dumps(data)
        status, resp = self._api_request("PUT", path, body)
        if status not in (200, 204):
            raise RuntimeError(
                f"FC API PUT {path} returned {status}: {resp[:200]}"
            )

    def _api_patch(self, path: str, data: dict[str, Any]) -> None:
        body = json.dumps(data)
        status, resp = self._api_request("PATCH", path, body)
        if status not in (200, 204):
            raise RuntimeError(
                f"FC API PATCH {path} returned {status}: {resp[:200]}"
            )

    # ------------------------------------------------------------------
    # Internal — vsock agent communication
    # ------------------------------------------------------------------

    def _execute_vsock(
        self,
        request: dict[str, Any],
        timeout_s: float | None,
    ) -> dict[str, Any]:
        """Send one JSON-line request via the FC vsock UDS and return the
        response."""
        from harness.fc_vsock_transport import VsockTransport

        transport = VsockTransport(
            vsock_sock_path=self._vsock_sock,
            port=self._vsock_port,
            connect_timeout_s=5.0,
            response_timeout_s=(
                timeout_s if timeout_s is not None else 600.0
            ),
        )
        try:
            transport.connect()
            return transport.send_request(request)
        finally:
            transport.close()

    # ------------------------------------------------------------------
    # Internal — readiness
    # ------------------------------------------------------------------

    def _poll_vsock_ready(
        self, timeout_s: float = 60.0, interval_s: float = 0.2,
    ) -> bool:
        """Block until the vsock agent accepts a connection."""
        from harness.fc_vsock_transport import poll_vsock_ready

        return poll_vsock_ready(
            vsock_sock_path=self._vsock_sock,
            port=self._vsock_port,
            timeout_s=timeout_s,
            interval_s=interval_s,
        )

    # ------------------------------------------------------------------
    # Internal — FC metrics
    # ------------------------------------------------------------------

    def _read_write_bytes(self) -> int:
        """Parse total write bytes from the FC JSON metrics file.

        Triggers a FlushMetrics action so FC appends one JSON object to
        the per-instance metrics file, then parses the last JSON line for
        the ``block.<drive_id>.write_bytes`` counter.
        """
        self._api_put("/actions", {"action_type": "FlushMetrics"})

        try:
            content = Path(self._metrics_path).read_text()
        except FileNotFoundError:
            raise RuntimeError(
                f"FC metrics file not found: {self._metrics_path}"
            )

        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        if not lines:
            return 0  # No metrics flushed yet — no writes have happened.

        # FC appends one JSON object per flush — parse from the END
        # backward to find the last complete JSON object.  Fragmented
        # lines (partial writes from concurrent flushes) are skipped.
        for line in reversed(lines):
            try:
                metrics = json.loads(line)
            except json.JSONDecodeError:
                continue
            # FC >=1.10 nests per-drive metrics under "block_<drive_id>".
            # "block" alone holds aggregate virtio-blk counters.
            rootfs_metrics = metrics.get("block_rootfs", {})
            write_bytes = rootfs_metrics.get("write_bytes")
            if write_bytes is not None:
                return int(write_bytes)
            # Found JSON but no rootfs drive — try the next line.
        return 0  # No write_bytes found in any line.


# One-shot warning flag for the reflink fallback.  The check-then-set is
# not thread-safe; a concurrent race only yields a duplicate log line, so
# no lock is taken.  Nothing may branch on this flag besides the warning.
_reflink_fallback_warned = False


def _cow_copy(src: str, dst: str, *, timeout: int = 60) -> None:
    """Copy *src* to *dst*, copy-on-write when the filesystem supports it.

    Tries ``cp --reflink=always`` first (O(1) CoW clone on XFS/btrfs).
    On filesystems without reflink support (e.g. ext4) it falls back to a
    sparse byte copy and logs a one-time warning, because the fallback
    turns every snapshot into a full-data copy — a real cost that must
    not be silent.
    """
    global _reflink_fallback_warned
    # No --sparse flag with reflink: coreutils rejects the combination
    # (--reflink requires --sparse=auto), and a reflink clone shares
    # extents with the source, preserving sparseness by construction.
    result = _checked_run(
        ["cp", "--reflink=always", src, dst],
        check=False,
        timeout=timeout,
    )
    if result.returncode == 0:
        return
    if not _reflink_fallback_warned:
        _reflink_fallback_warned = True
        logger.warning(
            "cp --reflink=always failed for %s (%s); falling back to full "
            "sparse copies. Disk snapshots on this filesystem are full-data "
            "copies — use XFS/btrfs storage for CoW snapshot costs.",
            dst,
            (result.stderr or result.stdout).strip().splitlines()[0]
            if (result.stderr or result.stdout).strip()
            else "no error output",
        )
    _checked_run(
        ["cp", "--sparse=always", src, dst],
        check=True,
        timeout=timeout,
    )


def _checked_run(
    cmd: list[str],
    *,
    check: bool = False,
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command, return ``CompletedProcess``, or raise on
    ``check=True`` and a non-zero exit."""
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"command failed (rc={result.returncode}): {' '.join(cmd)}\n{detail}"
        )
    return result


BACKENDS: dict[str, type[SandboxBackend]] = {
    "fake": FakeBackend,
    "docker": DockerBackend,
    "fc": FCBackend,
}


def get_sandbox_backend_class(name: str) -> type[SandboxBackend]:
    try:
        return BACKENDS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(BACKENDS))
        raise ValueError(f"unknown sandbox_backend {name!r}; expected one of {choices}") from exc


def create_sandbox_backend(name: str, **kwargs: Any) -> SandboxBackend:
    backend_cls = get_sandbox_backend_class(name)
    return backend_cls(**kwargs)
