from __future__ import annotations

import asyncio
import hashlib
import json
import os
import posixpath
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
                workdir=self.root,
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
                    "workspace_root": self.root,
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
                    workspace_root=self.root,
                )
                extra = {
                    "fixed_image": fixed_name,
                    "reported_elapsed_s": elapsed_s,
                    "prebuilt": False,
                    "workspace_root": self.root,
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
            start_task = asyncio.create_task(
                asyncio.to_thread(
                    self.start_task_container_fn,
                    fixed_name,
                    executable=self.container_executable,
                    extra_args=extra_args,
                    network_mode=self.network_mode,
                    workdir=self.root,
                )
            )
            try:
                self._container_id = await asyncio.shield(start_task)
            except asyncio.CancelledError:
                self._container_id = await start_task
                raise
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


BACKENDS: dict[str, type[SandboxBackend]] = {
    "fake": FakeBackend,
    "docker": DockerBackend,
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
