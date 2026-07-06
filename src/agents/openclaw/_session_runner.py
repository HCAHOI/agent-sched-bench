"""Shared bus-based session runner for CLI and evaluation flows."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from loguru import logger

from agents.openclaw._hook import AgentHook, AgentHookContext
from agents.openclaw._loop import AgentLoop
from agents.openclaw.config.schema import ExecToolConfig
from agents.openclaw.bus.events import InboundMessage
from agents.openclaw.bus.queue import MessageBus
from agents.openclaw.eval.collector import ResultCollector
from agents.base import TraceAction
from agents.openclaw.eval.types import (
    LLM,
    MCP,
    SUBAGENT,
    TOOL,
    EvalTraceEvent,
    EvalTraceSummary,
)
from llm_call.provider_base import LLMProvider
from agents.openclaw.session.manager import SessionManager
from trace_collect.latency_metrics import summarize_llm_latencies
from agents.openclaw._checkpoint_container import run_container_checkpoint

_MESSAGE_RECORDING_MODES = frozenset({"full", "delta"})
_CHECKPOINT_SCHEDULING_MODES = frozenset({"sync", "deferred"})
_COLLECTION_CHECKPOINT_BACKENDS = frozenset({"walk", "overlay"})
CheckpointEntryType = str | dict[str, str]
CheckpointSnapshotEntry = (
    CheckpointEntryType | tuple[CheckpointEntryType, int, int]
)
CheckpointHashCacheEntry = tuple[int, int, str]


def _validate_message_recording_mode(value: str) -> str:
    if value not in _MESSAGE_RECORDING_MODES:
        choices = ", ".join(sorted(_MESSAGE_RECORDING_MODES))
        raise ValueError(
            f"message_recording_mode must be one of {choices}, got {value!r}"
        )
    return value


def _validate_checkpoint_scheduling(value: str) -> str:
    if value not in _CHECKPOINT_SCHEDULING_MODES:
        choices = ", ".join(sorted(_CHECKPOINT_SCHEDULING_MODES))
        raise ValueError(
            f"checkpoint_scheduling must be one of {choices}, got {value!r}"
        )
    return value


def _validate_collection_checkpoint_backend(value: str) -> str:
    if value not in _COLLECTION_CHECKPOINT_BACKENDS:
        choices = ", ".join(sorted(_COLLECTION_CHECKPOINT_BACKENDS))
        raise ValueError(f"checkpoint_backend must be one of {choices}, got {value!r}")
    return value


@dataclass
class _PendingCheckpointCapture:
    task: asyncio.Task[dict[str, Any]]
    action_data: dict[str, Any]
    source_concurrent_execs: bool


def _checkpoint_relpath_is_skipped(relpath: str) -> bool:
    from agents.sandbox_runtime import _CHECKPOINT_SKIP_DIRS
    return any(part in _CHECKPOINT_SKIP_DIRS for part in relpath.split("/"))


def _trace_has_llm_error(trace_file: Path | None) -> bool:
    if trace_file is None or not trace_file.exists():
        return False
    with trace_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            if record.get("type") == "event" and record.get("event") == "llm_error":
                return True
    return False


def _resolve_run_outcome(
    *,
    outcome: dict[str, Any],
    content: str | None,
    trace_file: Path | None,
) -> tuple[str, str | None]:
    stop_reason = str(outcome.get("stop_reason") or "completed")
    error = outcome.get("error")
    if (
        stop_reason == "completed"
        and error is None
        and _trace_has_llm_error(trace_file)
    ):
        return "error", content or "LLM returned error."
    if error is None and stop_reason != "completed" and content:
        error = content
    return stop_reason, error


def _sanitize_checkpoint_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return safe[:96] or "tool"


def _single_exec_command_args(tool_name: str, args_json: str) -> dict[str, Any] | None:
    if tool_name != "exec":
        return None
    try:
        parsed = json.loads(args_json or "{}")
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    payload = parsed.get("exec") if isinstance(parsed.get("exec"), dict) else parsed
    if not isinstance(payload, dict):
        return None
    if "command" not in payload or "commands" in payload:
        return None
    return payload


def _iter_checkpoint_entries(root: Path) -> Iterator[str]:
    from agents.sandbox_runtime import _CHECKPOINT_SKIP_DIRS
    walk_errors: list[OSError] = []

    def record_walk_error(exc: OSError) -> None:
        walk_errors.append(exc)

    for dirpath, dirnames, filenames in os.walk(
        root,
        topdown=True,
        onerror=record_walk_error,
        followlinks=False,
    ):
        dirnames[:] = [name for name in dirnames if name not in _CHECKPOINT_SKIP_DIRS]
        dirnames.sort()
        filenames.sort()
        for dname in dirnames:
            full = os.path.join(dirpath, dname)
            if os.path.islink(full):
                yield full
        if Path(dirpath) != root:
            yield str(dirpath)
        for fname in filenames:
            yield os.path.join(dirpath, fname)

    if walk_errors:
        raise OSError(f"failed to walk checkpoint root {root}: {walk_errors[0]}")


def _checkpoint_entry_type(path: str | Path, mode: int) -> CheckpointEntryType:
    if stat.S_ISDIR(mode):
        return "dir"
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISLNK(mode):
        try:
            target = os.readlink(path)
        except OSError as exc:
            raise OSError(f"failed to read checkpoint symlink {path}: {exc}") from exc
        return {"type": "symlink", "target": target}
    return f"other:{stat.S_IFMT(mode)}"


def _checkpoint_snapshot_entry(
    entry_type: CheckpointEntryType,
    st: os.stat_result,
) -> CheckpointSnapshotEntry:
    return (entry_type, st.st_size, st.st_mtime_ns)


def _checkpoint_snapshot_entry_type(
    entry: CheckpointSnapshotEntry,
) -> CheckpointEntryType:
    if isinstance(entry, tuple):
        return entry[0]
    return entry


def _checkpoint_snapshot_entry_size_mtime(
    entry: CheckpointSnapshotEntry,
) -> tuple[int, int] | None:
    if isinstance(entry, tuple):
        return (entry[1], entry[2])
    return None


def _checkpoint_entry_requires_manifest(
    *,
    previous: CheckpointSnapshotEntry | None,
    current: CheckpointSnapshotEntry,
) -> bool:
    if previous is None:
        return True
    if _checkpoint_snapshot_entry_type(previous) != _checkpoint_snapshot_entry_type(
        current,
    ):
        return True
    previous_size_mtime = _checkpoint_snapshot_entry_size_mtime(previous)
    current_size_mtime = _checkpoint_snapshot_entry_size_mtime(current)
    if previous_size_mtime is None or current_size_mtime is None:
        return True
    return previous_size_mtime != current_size_mtime


def _compute_checkpoint_changed_paths(
    *,
    previous: dict[str, CheckpointSnapshotEntry],
    current: dict[str, CheckpointSnapshotEntry],
) -> set[str]:
    return {
        path
        for path, current_entry in current.items()
        if _checkpoint_snapshot_entry_type(current_entry) != "dir"
        and _checkpoint_entry_requires_manifest(
            previous=previous.get(path),
            current=current_entry,
        )
    }


def _compute_checkpoint_deleted_paths(
    *,
    previous: dict[str, CheckpointSnapshotEntry] | None,
    current: dict[str, CheckpointSnapshotEntry],
) -> list[str]:
    if previous is None:
        return []
    return sorted(
        path
        for path, previous_entry in previous.items()
        if (
            path not in current
            or _checkpoint_snapshot_entry_type(current[path])
            != _checkpoint_snapshot_entry_type(previous_entry)
        )
        and not _checkpoint_relpath_is_skipped(path)
    )


def _checkpoint_relpath(path: str, root: Path) -> str:
    return Path(os.path.relpath(path, root)).as_posix()


def _any_file_newer_than(root: Path, marker_mtime_ns: int) -> bool:
    # TODO: remove after full migration to WalkBackend.
    """Return True if any filesystem entry has mtime_ns > marker_mtime_ns.

    Uses os.walk + lstat (stat-only, no content reads). A file or directory
    modified after the marker timestamp indicates a write event occurred and a
    checkpoint is needed.
    """
    from agents.sandbox_runtime import _CHECKPOINT_SKIP_DIRS
    root = root.resolve()
    walk_errors: list[OSError] = []

    def record_walk_error(exc: OSError) -> None:
        walk_errors.append(exc)

    def entry_is_newer(path: str | Path) -> bool:
        try:
            return os.lstat(path).st_mtime_ns > marker_mtime_ns
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
        dirnames[:] = [name for name in dirnames if name not in _CHECKPOINT_SKIP_DIRS]
        for name in (*dirnames, *filenames):
            if entry_is_newer(os.path.join(dirpath, name)):
                return True
    return bool(walk_errors)


def _snapshot_checkpoint_entries(root: Path) -> dict[str, CheckpointSnapshotEntry]:
    # TODO: remove after full migration to WalkBackend.
    root = root.resolve()
    entries: dict[str, CheckpointSnapshotEntry] = {}
    for fpath in _iter_checkpoint_entries(root):
        try:
            st = os.lstat(fpath)
        except OSError as exc:
            raise OSError(f"failed to stat checkpoint entry {fpath}: {exc}") from exc
        entry_type = _checkpoint_entry_type(fpath, st.st_mode)
        if not isinstance(entry_type, dict) and entry_type not in {"dir", "file"}:
            raise OSError(f"unsupported checkpoint entry type: {fpath}")
        entries[_checkpoint_relpath(fpath, root)] = _checkpoint_snapshot_entry(
            entry_type,
            st,
        )
    return entries


def _tree_contains_symlink(root: Path) -> bool:
    for path in root.rglob("*"):
        if path.is_symlink():
            return True
    return False


def _relative_to_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


_CHECKPOINT_CAS_ROOT = Path.home() / ".cache" / "agent-checkpoint-cas"


def _unique_blob_tmp_path(blob_path: Path) -> Path:
    return blob_path.parent / f".tmp.{os.getpid()}.{uuid.uuid4().hex}"


def _write_cas_manifest(
    *,
    root: Path,
    manifest_path: Path,
    incremental_since_ns: int | None = None,
    deleted_paths: list[str] | None = None,
    incremental_changed_paths: set[str] | None = None,
    hash_cache: dict[str, CheckpointHashCacheEntry] | None = None,
) -> int:
    # TODO: remove after full migration to WalkBackend.
    """Walk /testbed, write content-addressed blobs to CAS, emit a JSON manifest.

    Args:
        root: The directory to snapshot (always /testbed).
        manifest_path: Where to write the manifest JSON.
        incremental_since_ns: If set, skip files whose mtime_ns <= this unless
            incremental_changed_paths is provided.
        deleted_paths: Paths to record as deleted (incremental).
        incremental_changed_paths: If set, only manifest these relative paths.
        hash_cache: Optional path-to-(size, mtime_ns, hexdigest) cache. Updated
            in place. Files whose path, size, and mtime match skip re-hashing.

    Returns:
        Total size in bytes of all unique blobs written for this manifest.

    Raises:
        OSError: on stat/read/write failure.
    """
    root = root.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    cas_root = _CHECKPOINT_CAS_ROOT
    entries: dict[str, dict[str, Any]] = {}
    total_unique_bytes = 0

    for fpath in _iter_checkpoint_entries(root):
        try:
            st = os.lstat(fpath)
        except OSError as exc:
            raise OSError(f"failed to stat checkpoint entry {fpath}: {exc}") from exc

        entry_type = _checkpoint_entry_type(fpath, st.st_mode)
        if not isinstance(entry_type, dict) and entry_type not in {"dir", "file"}:
            raise OSError(f"unsupported checkpoint entry type: {fpath}")
        rel = _checkpoint_relpath(fpath, root)

        if entry_type == "dir":
            continue

        if isinstance(entry_type, dict):
            if entry_type.get("type") != "symlink":
                raise OSError(f"unsupported checkpoint entry type: {fpath}")
            if incremental_changed_paths is not None and rel not in incremental_changed_paths:
                continue
            if (
                incremental_changed_paths is None
                and incremental_since_ns is not None
                and st.st_mtime_ns <= incremental_since_ns
            ):
                continue
            entries[rel] = dict(entry_type)
            continue

        if incremental_changed_paths is not None and rel not in incremental_changed_paths:
            continue
        if (
            incremental_changed_paths is None
            and incremental_since_ns is not None
            and st.st_mtime_ns <= incremental_since_ns
        ):
            continue

        cached = hash_cache.get(rel) if hash_cache else None
        file_bytes: bytes | None = None
        if (
            cached is not None
            and cached[0] == st.st_size
            and cached[1] == st.st_mtime_ns
        ):
            digest = cached[2]
        else:
            try:
                with open(fpath, "rb") as fh:
                    file_bytes = fh.read()
            except OSError as exc:
                raise OSError(f"failed to read checkpoint entry {fpath}: {exc}") from exc
            digest = hashlib.sha256(file_bytes).hexdigest()
            if hash_cache is not None:
                hash_cache[rel] = (st.st_size, st.st_mtime_ns, digest)

        blob_path = cas_root / "blobs" / digest[:2] / digest[2:]
        if not blob_path.exists():
            blob_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = _unique_blob_tmp_path(blob_path)
            try:
                if file_bytes is None:
                    with open(fpath, "rb") as fh:
                        file_bytes = fh.read()
                tmp.write_bytes(file_bytes)
                tmp.rename(blob_path)
            except OSError as exc:
                raise OSError(f"failed to write blob {blob_path}: {exc}") from exc
            total_unique_bytes += st.st_size

        entries[rel] = {
            "hash": digest,
            "mode": stat.S_IMODE(st.st_mode),
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
        }

    manifest: dict[str, Any] = {
        "entries": entries,
        "deleted_paths": sorted(
            path
            for path in (deleted_paths or [])
            if not _checkpoint_relpath_is_skipped(path)
        ),
    }

    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp.rename(manifest_path)
    return total_unique_bytes


class TraceCollectorHook(AgentHook):
    """Collect per-iteration actions, events, and summaries as JSONL."""

    def __init__(
        self,
        trace_file: Path,
        instance_id: str,
        *,
        agent_id: str | None = None,
        task_id: str | None = None,
        checkpoint_root: Path | None = None,
        checkpoint_dir: Path | None = None,
        checkpoint_root_label: str = "/testbed",
        checkpoint_rebaseline_bytes: int | None = None,
        container_runtime: dict[str, str] | None = None,
        record_sink: Callable[[dict[str, Any]], None] | None = None,
        hook_type: str = "parent",
        message_recording_mode: str = "full",
        checkpoint_scheduling: str = "sync",
        checkpoint_backend: str = "walk",
    ) -> None:
        self.trace_file = trace_file
        self.instance_id = instance_id
        self.agent_id = agent_id or instance_id
        self.program_id = self.agent_id
        self.task_id = task_id or instance_id
        self.hook_type = hook_type
        self.message_recording_mode = _validate_message_recording_mode(
            message_recording_mode
        )
        self._checkpoint_scheduling = _validate_checkpoint_scheduling(
            checkpoint_scheduling
        )
        self._checkpoint_backend_type = _validate_collection_checkpoint_backend(
            checkpoint_backend
        )
        self.trace_file.parent.mkdir(parents=True, exist_ok=True)
        self._wall_start = time.monotonic()
        self._total_tokens = 0
        self._n_iterations = 0
        self._tool_times: dict[str, float] = {}
        self._tool_timeouts: dict[str, int] = {}
        self._tool_start_ts: dict[str, float] = {}
        self._iter_start_wall: float = 0.0
        self._iter_messages_snapshot: list[dict[str, Any]] | None = None
        self._iter_messages_delta: list[dict[str, Any]] | None = None
        self._message_snapshot_hashes: list[str] = []
        self._before_exec_wall: float = 0.0
        self._records: list[dict[str, Any]] = []
        self._actions: list[dict[str, Any]] = []
        self._pending_llm_records: list[dict[str, Any]] = []
        self._checkpoint_root = Path(checkpoint_root) if checkpoint_root else None
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._checkpoint_root_label = checkpoint_root_label
        self._container_runtime = container_runtime
        self._last_full_checkpoint_ns: int | None = None
        self._last_incremental_checkpoint_ns: int | None = None
        self._checkpoint_snapshot_entries: dict[str, CheckpointSnapshotEntry] | None = None
        self._rebaseline_bytes = checkpoint_rebaseline_bytes
        self._checkpoint_chain_bytes_since_full: int = 0
        self._checkpoint_hash_cache: dict[str, CheckpointHashCacheEntry] = {}
        self._checkpoint_backend: Any | None = None
        self._checkpoint_backend_started = False
        self._last_deferred_checkpoint_snapshot: Any | None = None
        self._pending_checkpoint_captures: list[_PendingCheckpointCapture] = []
        self._flushed = False
        self._record_sink = record_sink
        self._fh: TextIO | None = (
            None
            if record_sink is not None
            else open(trace_file, "w", encoding="utf-8")  # noqa: SIM115
        )

    def _checkpoint_after_tool(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        tool_args_json: str,
    ) -> dict[str, Any] | None:
        # TODO: remove after full migration to WalkBackend.
        if self._checkpoint_root is None or self._checkpoint_dir is None:
            return None
        if _single_exec_command_args(tool_name, tool_args_json) is None:
            return None
        if self._container_runtime is not None:
            return self._checkpoint_container_after_tool(tool_call_id=tool_call_id)
        started = time.monotonic()
        root = self._checkpoint_root.resolve()
        if not root.is_dir():
            return {
                "error": f"checkpoint root is unavailable: {root}",
                "overhead_excluded": True,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            }
        if self._checkpoint_root_label != "/testbed":
            return {
                "error": f"unsupported checkpoint root: {self._checkpoint_root_label}",
                "overhead_excluded": True,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            }
        marker_before_ns = time.time_ns()
        try:
            current_snapshot = _snapshot_checkpoint_entries(root)
        except OSError as exc:
            return {
                "error": f"checkpoint failed: {exc}",
                "overhead_excluded": True,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            }

        precomputed_deleted_paths: list[str] | None = None
        precomputed_changed_paths: set[str] | None = None
        if (
            self._last_incremental_checkpoint_ns is not None
            and self._checkpoint_snapshot_entries is not None
        ):
            precomputed_deleted_paths = _compute_checkpoint_deleted_paths(
                previous=self._checkpoint_snapshot_entries,
                current=current_snapshot,
            )
            precomputed_changed_paths = _compute_checkpoint_changed_paths(
                previous=self._checkpoint_snapshot_entries,
                current=current_snapshot,
            )
            if not precomputed_changed_paths and not precomputed_deleted_paths:
                return {
                    "skipped": "no filesystem changes since last checkpoint",
                    "overhead_excluded": True,
                    "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
                }

        is_first = self._last_full_checkpoint_ns is None

        # Re-baseline: if the incremental CAS chain exceeds the threshold,
        # promote this manifest to full regardless of is_first.
        force_full = (
            not is_first
            and self._rebaseline_bytes is not None
            and self._checkpoint_chain_bytes_since_full >= self._rebaseline_bytes
        )

        incremental_since = None if is_first or force_full else self._last_incremental_checkpoint_ns
        deleted_paths: list[str] = []
        incremental_changed_paths: set[str] | None = None
        if (
            not is_first
            and not force_full
            and self._checkpoint_snapshot_entries is not None
        ):
            if precomputed_deleted_paths is None or precomputed_changed_paths is None:
                precomputed_deleted_paths = _compute_checkpoint_deleted_paths(
                    previous=self._checkpoint_snapshot_entries,
                    current=current_snapshot,
                )
                precomputed_changed_paths = _compute_checkpoint_changed_paths(
                    previous=self._checkpoint_snapshot_entries,
                    current=current_snapshot,
                )
            deleted_paths = precomputed_deleted_paths
            incremental_changed_paths = precomputed_changed_paths

        manifest_path = (
            self._checkpoint_dir
            / f"{_sanitize_checkpoint_name(tool_call_id)}-manifest.json"
        )
        try:
            chain_bytes = _write_cas_manifest(
                root=root,
                manifest_path=manifest_path,
                incremental_since_ns=incremental_since,
                deleted_paths=deleted_paths,
                incremental_changed_paths=incremental_changed_paths,
                hash_cache=self._checkpoint_hash_cache,
            )
        except OSError as exc:
            return {
                "error": f"checkpoint failed: {exc}",
                "overhead_excluded": True,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            }

        try:
            changed_during_write = _any_file_newer_than(root, marker_before_ns)
        except OSError:
            changed_during_write = True
        if changed_during_write:
            try:
                manifest_path.unlink(missing_ok=True)
            except OSError:
                pass
            # Blobs are content-addressed, so tmp+rename prevents ghost reads.
            # Only the manifest is rolled back on concurrent-write detection.
            return {
                "error": "checkpoint failed: filesystem changed during checkpoint",
                "overhead_excluded": True,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            }

        elapsed_ms = (time.monotonic() - started) * 1000
        size_bytes = manifest_path.stat().st_size
        is_full = is_first or force_full
        if is_full:
            self._last_full_checkpoint_ns = marker_before_ns
            self._checkpoint_chain_bytes_since_full = 0
        else:
            self._checkpoint_chain_bytes_since_full += chain_bytes
        self._last_incremental_checkpoint_ns = marker_before_ns
        self._checkpoint_snapshot_entries = current_snapshot
        return {
            "path": _relative_to_or_absolute(manifest_path, self.trace_file.parent),
            "kind": (
                "cas_manifest_full" if is_full else "cas_manifest_incremental"
            ),
            "root": self._checkpoint_root_label,
            "incremental": not is_full,
            "incremental_since_ns": incremental_since,
            "elapsed_ms": round(elapsed_ms, 3),
            "size_bytes": size_bytes,
            "overhead_excluded": True,
            "rebaseline": force_full or None,
            "chain_bytes": chain_bytes,
        }

    def _checkpoint_container_after_tool(
        self,
        *,
        tool_call_id: str,
    ) -> dict[str, Any] | None:
        assert self._container_runtime is not None
        started = time.monotonic()

        is_first = self._last_full_checkpoint_ns is None
        force_full = (
            not is_first
            and self._rebaseline_bytes is not None
            and self._checkpoint_chain_bytes_since_full >= self._rebaseline_bytes
        )
        incremental_since = (
            None if is_first or force_full else self._last_incremental_checkpoint_ns
        )

        manifest_path = (
            self._checkpoint_dir
            / f"{_sanitize_checkpoint_name(tool_call_id)}-manifest.json"
        )
        assert self._checkpoint_dir is not None

        result = run_container_checkpoint(
            container_runtime=self._container_runtime,
            manifest_path=manifest_path,
            incremental_since_ns=incremental_since,
            prev_snapshot_entries=self._checkpoint_snapshot_entries,
            force_full=force_full,
        )

        if "error" in result:
            if result.get("overhead_excluded") is None:
                result["overhead_excluded"] = True
            if "elapsed_ms" not in result:
                result["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
            return result

        if "skipped" in result:
            result.pop("_state", None)
            return result

        state = result.pop("_state", {})
        now_ns: int | None = state.get("now_ns")
        chain_bytes: int = state.get("chain_bytes", 0)
        is_full_state: bool = state.get("is_full", is_first or force_full)
        current_snapshot: dict[str, CheckpointSnapshotEntry] = state.get(
            "current_snapshot",
            {},
        )

        result["path"] = _relative_to_or_absolute(manifest_path, self.trace_file.parent)

        if is_full_state:
            self._last_full_checkpoint_ns = now_ns
            self._checkpoint_chain_bytes_since_full = 0
        else:
            self._checkpoint_chain_bytes_since_full += chain_bytes
        self._last_incremental_checkpoint_ns = now_ns
        self._checkpoint_snapshot_entries = current_snapshot

        return result

    async def _checkpoint_after_tool_deferred(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        tool_args_json: str,
        action_data: dict[str, Any],
        source_concurrent_execs: bool,
    ) -> dict[str, Any] | None:
        if self._checkpoint_root is None or self._checkpoint_dir is None:
            return None
        if _single_exec_command_args(tool_name, tool_args_json) is None:
            return None

        await self._await_pending_checkpoint_captures()
        previous_snapshot = self._last_deferred_checkpoint_snapshot
        force_full = (
            previous_snapshot is not None
            and self._rebaseline_bytes is not None
            and self._checkpoint_chain_bytes_since_full >= self._rebaseline_bytes
        )
        incremental_since = None if force_full else previous_snapshot
        if previous_snapshot is None:
            self._schedule_checkpoint_capture(
                incremental_since=None,
                action_data=action_data,
                source_concurrent_execs=source_concurrent_execs,
                probe_result="initial",
                force_full=False,
            )
            return None

        # Run the filesystem probe (formerly gated by a syntatic command
        # classifier — removed per ADR: the probe dominates the signal,
        # and the classifier was unreliable for unbounded CLI commands).
        probe_started = time.monotonic()
        try:
            backend = await self._ensure_checkpoint_backend()
            probe_changed = await backend.probe_changes_since(
                previous_snapshot.timestamp_ns
            )
        except Exception as exc:
            return {
                "error": f"checkpoint probe failed: {exc}",
                "overhead_excluded": True,
                "elapsed_ms": round((time.monotonic() - probe_started) * 1000, 3),
            }

        probe_elapsed_ms = round((time.monotonic() - probe_started) * 1000, 3)
        if not probe_changed:
            return {
                "skipped": "no filesystem changes since last checkpoint",
                "overhead_excluded": True,
                "elapsed_ms": probe_elapsed_ms,
                "probe_result": "unchanged",
                "checkpoint_decision": "probe_unchanged",
            }

        self._schedule_checkpoint_capture(
            incremental_since=incremental_since,
            action_data=action_data,
            source_concurrent_execs=source_concurrent_execs,
            probe_result="changed",
            force_full=force_full,
        )
        return None

    async def _ensure_checkpoint_backend(self) -> Any:
        if self._checkpoint_dir is None or self._checkpoint_root is None:
            raise RuntimeError("checkpoint backend requires checkpoint root and dir")
        if self._checkpoint_backend is None:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            from agents.sandbox_runtime import OverlayBackend, WalkBackend

            if self._container_runtime is None:
                if self._checkpoint_backend_type != "walk":
                    raise RuntimeError(
                        "overlay checkpoint backend requires container runtime"
                    )
                self._checkpoint_backend = WalkBackend(
                    root=str(self._checkpoint_root.resolve()),
                    checkpoint_dir=self._checkpoint_dir,
                )
            elif self._checkpoint_backend_type == "walk":
                self._checkpoint_backend = WalkBackend(
                    root=self._checkpoint_root_label,
                    checkpoint_dir=self._checkpoint_dir,
                    container_runtime=self._container_runtime,
                )
            elif self._checkpoint_backend_type == "overlay":
                container_id = self._container_runtime.get("id")
                executable = self._container_runtime.get("executable")
                if not container_id or not executable:
                    raise RuntimeError(
                        "overlay checkpoint backend requires container id and executable"
                    )
                self._checkpoint_backend = OverlayBackend(
                    root=self._checkpoint_root_label,
                    checkpoint_dir=self._checkpoint_dir,
                    container_id=container_id,
                    container_executable=executable,
                )
            else:
                raise RuntimeError(
                    f"unknown checkpoint backend: {self._checkpoint_backend_type!r}"
                )
        if not self._checkpoint_backend_started:
            await self._checkpoint_backend.start()
            self._checkpoint_backend_started = True
        return self._checkpoint_backend

    def _schedule_checkpoint_capture(
        self,
        *,
        incremental_since: Any | None,
        action_data: dict[str, Any],
        source_concurrent_execs: bool,
        probe_result: str,
        force_full: bool,
    ) -> None:
        task = asyncio.create_task(
            self._capture_checkpoint_deferred(
                incremental_since=incremental_since,
                probe_result=probe_result,
                force_full=force_full,
            )
        )
        self._pending_checkpoint_captures.append(
            _PendingCheckpointCapture(
                task=task,
                action_data=action_data,
                source_concurrent_execs=source_concurrent_execs,
            )
        )

    async def _capture_checkpoint_deferred(
        self,
        *,
        incremental_since: Any | None,
        probe_result: str,
        force_full: bool,
    ) -> dict[str, Any]:
        started = time.monotonic()
        try:
            backend = await self._ensure_checkpoint_backend()
            snapshot = await backend.capture_snapshot(incremental_since=incremental_since)
        except Exception as exc:
            return {
                "error": f"checkpoint failed: {exc}",
                "overhead_excluded": True,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
                "probe_result": probe_result,
            }

        elapsed_ms = round((time.monotonic() - started) * 1000, 3)
        if incremental_since is not None and snapshot is incremental_since:
            return {
                "skipped": "no filesystem changes since last checkpoint",
                "overhead_excluded": True,
                "elapsed_ms": elapsed_ms,
                "probe_result": probe_result,
                "_snapshot": snapshot,
            }
        result = self._checkpoint_after_from_snapshot(
            snapshot=snapshot,
            incremental_since=incremental_since,
            elapsed_ms=elapsed_ms,
            probe_result=probe_result,
            force_full=force_full,
        )
        result["_snapshot"] = snapshot
        return result

    def _checkpoint_after_from_snapshot(
        self,
        *,
        snapshot: Any,
        incremental_since: Any | None,
        elapsed_ms: float,
        probe_result: str,
        force_full: bool,
    ) -> dict[str, Any]:
        disk_state = snapshot.disk_state
        manifest_path_value = disk_state.get("manifest_path")
        if not isinstance(manifest_path_value, str) or not manifest_path_value:
            raise ValueError("checkpoint snapshot missing disk_state.manifest_path")
        manifest_path = Path(manifest_path_value)
        kind = disk_state.get("kind")
        if not isinstance(kind, str) or not kind:
            kind = (
                "cas_manifest_full"
                if incremental_since is None
                else "cas_manifest_incremental"
            )
        incremental = bool(disk_state.get("incremental", incremental_since is not None))
        size_bytes = disk_state.get("size_bytes")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int):
            size_bytes = manifest_path.stat().st_size
        chain_bytes = disk_state.get("chain_bytes")
        result: dict[str, Any] = {
            "path": _relative_to_or_absolute(manifest_path, self.trace_file.parent),
            "kind": kind,
            "root": self._checkpoint_root_label,
            "incremental": incremental,
            "incremental_since_ns": (
                None if incremental_since is None else incremental_since.timestamp_ns
            ),
            "elapsed_ms": elapsed_ms,
            "size_bytes": size_bytes,
            "overhead_excluded": True,
            "probe_result": probe_result,
        }
        if force_full:
            result["rebaseline"] = True
        if isinstance(chain_bytes, int) and not isinstance(chain_bytes, bool):
            result["chain_bytes"] = chain_bytes
        return result

    async def _await_pending_checkpoint_captures(self) -> None:
        pending_captures = self._pending_checkpoint_captures
        if not pending_captures:
            return
        self._pending_checkpoint_captures = []
        for pending in pending_captures:
            wait_started = time.monotonic()
            if pending.task.done():
                checkpoint_after = pending.task.result()
                exposed_ms = 0.0
            else:
                checkpoint_after = await pending.task
                exposed_ms = (time.monotonic() - wait_started) * 1000
            self._apply_deferred_checkpoint_result(
                pending=pending,
                checkpoint_after=checkpoint_after,
                exposed_ms=round(exposed_ms, 3),
            )

    def _apply_deferred_checkpoint_result(
        self,
        *,
        pending: _PendingCheckpointCapture,
        checkpoint_after: dict[str, Any],
        exposed_ms: float,
    ) -> None:
        snapshot = checkpoint_after.pop("_snapshot", None)
        if snapshot is not None and "error" not in checkpoint_after:
            self._record_deferred_checkpoint_snapshot(snapshot, checkpoint_after)
        pending.action_data["checkpoint_exposed_ms"] = exposed_ms
        if "error" in checkpoint_after:
            pending.action_data["checkpoint_after_error"] = checkpoint_after
        else:
            pending.action_data["checkpoint_after"] = checkpoint_after
        if pending.source_concurrent_execs:
            pending.action_data["smeared_checkpoint"] = True

    def _record_deferred_checkpoint_snapshot(
        self,
        snapshot: Any,
        checkpoint_after: dict[str, Any],
    ) -> None:
        if checkpoint_after.get("skipped") is not None:
            return
        chain_bytes = checkpoint_after.get("chain_bytes", 0)
        if isinstance(chain_bytes, bool) or not isinstance(chain_bytes, int):
            chain_bytes = 0
        is_full = checkpoint_after.get("incremental") is not True
        if is_full:
            self._last_full_checkpoint_ns = snapshot.timestamp_ns
            self._checkpoint_chain_bytes_since_full = 0
        else:
            self._checkpoint_chain_bytes_since_full += chain_bytes
        self._last_incremental_checkpoint_ns = snapshot.timestamp_ns
        self._last_deferred_checkpoint_snapshot = snapshot

    def close(self) -> None:
        if self._flushed:
            return
        if self._record_sink is not None:
            self._flushed = True
            return
        assert self._fh is not None
        if not self._fh.closed:
            self._fh.close()
        tmp_trace_file = self.trace_file.with_suffix(f"{self.trace_file.suffix}.tmp")
        tmp_trace_file.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in self._records
            ),
            encoding="utf-8",
        )
        tmp_trace_file.replace(self.trace_file)
        self._flushed = True

    def add_record(self, record: dict[str, Any]) -> None:
        self._records.append(record)
        if self._record_sink is not None:
            self._record_sink(record)
            return
        assert self._fh is not None
        if self._fh.closed:
            with self.trace_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            return
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def emit_event(
        self,
        category: str,
        event: str,
        data: dict[str, Any],
        *,
        iteration: int = 0,
    ) -> None:
        entry = EvalTraceEvent(
            agent_id=self.agent_id,
            program_id=self.program_id,
            instance_id=self.instance_id,
            event=event,
            category=category,
            data=data,
            ts=time.time(),
            iteration=iteration,
        )
        entry.data["hook_type"] = self.hook_type
        self.add_record(entry.to_dict())

    async def before_iteration(self, context: AgentHookContext) -> None:
        self._iter_start_wall = time.time()
        if self.message_recording_mode == "full":
            self._iter_messages_snapshot = self._clone_messages(context.messages)
            messages_data = {"messages_in": self._iter_messages_snapshot}
        else:
            self._iter_messages_snapshot = None
            self._iter_messages_delta = self._messages_delta_since_previous_snapshot(
                context.messages
            )
            messages_data = {
                "messages_delta": self._iter_messages_delta,
                "is_delta": True,
            }
        self.emit_event(
            LLM,
            "llm_call_start",
            messages_data,
            iteration=context.iteration,
        )

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        await self._await_pending_checkpoint_captures()
        self._before_exec_wall = time.time()
        if context.tool_calls:
            for tc in context.tool_calls:
                self._tool_start_ts[tc.id] = time.monotonic()
                is_mcp = tc.name.startswith("mcp_")
                self.emit_event(
                    MCP if is_mcp else TOOL,
                    "tool_exec_start",
                    {
                        "tool_name": tc.name,
                        "args_preview": json.dumps(tc.arguments, ensure_ascii=False)[
                            :200
                        ],
                    },
                    iteration=context.iteration,
                )

    async def after_iteration(self, context: AgentHookContext) -> None:
        ts_now = time.time()
        self._n_iterations += 1

        usage = context.usage or {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        self._total_tokens += prompt_tokens + completion_tokens

        llm_ts_end = (
            self._resolve_llm_ts_end(context.response)
            or self._before_exec_wall
            or ts_now
        )
        llm_wall_latency_ms = max(0.0, (llm_ts_end - self._iter_start_wall) * 1000)
        llm_call_time_ms = self._resolve_llm_call_time_ms(
            context.response,
            llm_wall_latency_ms,
        )
        llm_timing_source = self._resolve_llm_timing_source(context.response)
        resp_dict = self._build_raw_response(
            context=context,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        llm_event_data = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "llm_latency_ms": round(llm_call_time_ms, 2),
            "llm_call_time_ms": round(llm_call_time_ms, 2),
            "llm_wall_latency_ms": round(llm_wall_latency_ms, 2),
            "llm_timing_source": llm_timing_source,
            "finish_reason": context.response.finish_reason
            if context.response
            else None,
            "is_malformed_retry": context.malformed_retry_count > 0,
        }
        llm_action_data: dict[str, Any] = {
            **self._iter_messages_payload(),
            "raw_response": resp_dict,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "llm_latency_ms": round(llm_call_time_ms, 2),
            "llm_call_time_ms": round(llm_call_time_ms, 2),
            "llm_wall_latency_ms": round(llm_wall_latency_ms, 2),
            "llm_timing_source": llm_timing_source,
            "malformed_retry_count": context.malformed_retry_count,
            "is_malformed_retry": context.malformed_retry_count > 0,
            "hook_type": self.hook_type,
        }
        trace_llm_fields = self._extract_trace_llm_fields(context.response)
        if trace_llm_fields:
            llm_event_data.update(
                {
                    key: value
                    for key, value in trace_llm_fields.items()
                    if key != "openrouter_metadata"
                }
            )
            llm_action_data.update(trace_llm_fields)
        self.emit_event(
            LLM,
            "llm_call_end",
            llm_event_data,
            iteration=context.iteration,
        )
        llm_action = TraceAction(
            action_type="llm_call",
            action_id=f"llm_{context.iteration}",
            agent_id=self.agent_id,
            program_id=self.program_id,
            instance_id=self.instance_id,
            iteration=context.iteration,
            ts_start=self._iter_start_wall,
            ts_end=llm_ts_end,
            data=llm_action_data,
        )
        self._write_action(llm_action)
        if context.response and getattr(context.response, "extra", None):
            if context.response.extra.get("_openrouter_metadata_task") is not None:
                self._pending_llm_records.append(
                    {
                        "response": context.response,
                        "event_data": llm_event_data,
                        "action_data": llm_action_data,
                        "raw_response": resp_dict,
                    }
                )
        self._before_exec_wall = 0.0
        self._iter_messages_snapshot = None
        self._iter_messages_delta = None

        tool_results_from_messages = self._extract_tool_results(context.messages)
        if tool_results_from_messages:
            tool_args_by_id: dict[str, str] = {}
            tool_name_by_id: dict[str, str] = {}
            if context.tool_calls:
                for tc in context.tool_calls:
                    tool_args_by_id[tc.id] = json.dumps(
                        tc.arguments, ensure_ascii=False
                    )
                    tool_name_by_id[tc.id] = tc.name
            concurrent_exec_tool_calls = [
                tc for tc in context.tool_calls if tc.name == "exec"
            ]
            source_concurrent_execs = len(concurrent_exec_tool_calls) > 1

            traceable_tool_results = [
                result
                for result in tool_results_from_messages
                if result[1] != "_invalid_tool_call"
                and not (result[0] and result[0].startswith("malformed_retry_"))
            ]
            for tc_id, tool_name, tool_content, tool_ok in traceable_tool_results:
                tool_start_mono = self._tool_start_ts.pop(tc_id, None)
                duration_ms = (
                    (time.monotonic() - tool_start_mono) * 1000
                    if tool_start_mono
                    else 0.0
                )
                self._tool_times[tool_name] = (
                    self._tool_times.get(tool_name, 0.0) + duration_ms
                )
                structured_results = getattr(context, "tool_structured_results", {})
                structured_result = structured_results.get(tc_id)
                success_source = "text_heuristic"
                if structured_result is not None:
                    structured_fields = self._structured_tool_result_fields(
                        structured_result
                    )
                    tool_ok = not structured_fields["timed_out"]
                    success_source = "structured"
                else:
                    structured_fields = {}
                if not tool_ok:
                    self._tool_timeouts[tool_name] = (
                        self._tool_timeouts.get(tool_name, 0) + 1
                    )

                is_mcp = tool_name.startswith("mcp_")
                self.emit_event(
                    MCP if is_mcp else TOOL,
                    "tool_exec_end",
                    {
                        "tool_name": tool_name,
                        "success": tool_ok,
                        "duration_ms": round(duration_ms, 1),
                        "result_preview": tool_content[:200],
                    },
                    iteration=context.iteration,
                )
                if tool_name == "spawn":
                    self.emit_event(
                        SUBAGENT,
                        "subagent_complete",
                        {"task_preview": tool_content[:200]},
                        iteration=context.iteration,
                    )

                tool_ts_end = time.time()
                tool_ts_start = (
                    tool_ts_end - duration_ms / 1000 if duration_ms else tool_ts_end
                )
                action_id_suffix = tc_id if tc_id else tool_name
                tool_action_data: dict[str, Any] = {
                    "tool_name": tool_name,
                    "tool_call_id": tc_id,
                    "tool_args": tool_args_by_id.get(tc_id, ""),
                    "tool_result": tool_content,
                    "duration_ms": round(duration_ms, 1),
                    "success": tool_ok,
                    "success_source": success_source,
                    "hook_type": self.hook_type,
                    **structured_fields,
                }
                if source_concurrent_execs:
                    tool_action_data["source_concurrent_execs"] = True
                resource_timelines = getattr(context, "tool_resource_timelines", {})
                resource_timeline = resource_timelines.get(tc_id)
                if resource_timeline is not None:
                    tool_action_data["resource_timeline"] = resource_timeline
                if self._checkpoint_scheduling == "deferred":
                    checkpoint_after = await self._checkpoint_after_tool_deferred(
                        tool_call_id=tc_id or action_id_suffix,
                        tool_name=tool_name,
                        tool_args_json=tool_action_data["tool_args"],
                        action_data=tool_action_data,
                        source_concurrent_execs=source_concurrent_execs,
                    )
                else:
                    checkpoint_after = await asyncio.to_thread(
                        self._checkpoint_after_tool,
                        tool_call_id=tc_id or action_id_suffix,
                        tool_name=tool_name,
                        tool_args_json=tool_action_data["tool_args"],
                    )
                if checkpoint_after is not None:
                    if "error" in checkpoint_after:
                        tool_action_data["checkpoint_after_error"] = checkpoint_after
                    else:
                        tool_action_data["checkpoint_after"] = checkpoint_after
                    if source_concurrent_execs:
                        tool_action_data["smeared_checkpoint"] = True
                tool_action = TraceAction(
                    action_type="tool_exec",
                    action_id=f"tool_{context.iteration}_{action_id_suffix}",
                    agent_id=self.agent_id,
                    program_id=self.program_id,
                    instance_id=self.instance_id,
                    iteration=context.iteration,
                    ts_start=tool_ts_start,
                    ts_end=tool_ts_end,
                    data=tool_action_data,
                )
                self._write_action(tool_action)

        if context.response:
            finish_reason = context.response.finish_reason
            if finish_reason == "error":
                error_data: dict[str, Any] = {
                    "error_message": context.response.content[:500]
                    if context.response.content
                    else "",
                    "finish_reason": finish_reason,
                }
                if context.response.extra:
                    error_data.update(
                        {
                            key: value
                            for key, value in context.response.extra.items()
                            if key != "llm_wall_ts_end"
                            and not key.startswith("_")
                        }
                    )
                    error_data.update(self._extract_trace_llm_fields(context.response))
                self.emit_event(
                    LLM,
                    "llm_error",
                    error_data,
                    iteration=context.iteration,
                )
            elif finish_reason == "max_iterations":
                self.emit_event(
                    LLM,
                    "max_iterations",
                    {"total_tokens": self._total_tokens},
                    iteration=context.iteration,
                )

    @staticmethod
    def _extract_tool_results(
        messages: list[dict],
    ) -> list[tuple[str, str, str, bool]]:
        """Extract (tool_call_id, tool_name, content, ok) from trailing tool messages."""
        results: list[tuple[str, str, str, bool]] = []
        i = len(messages) - 1
        while i >= 0 and messages[i].get("role") == "tool":
            m = messages[i]
            tool_call_id = m.get("tool_call_id", "")
            name = m.get("name", "unknown")
            content = str(m.get("content", ""))
            ok = not content.startswith("Error")
            results.append((tool_call_id, name, content, ok))
            i -= 1
        results.reverse()
        return results

    @staticmethod
    def _structured_tool_result_fields(
        structured_result: dict[str, Any],
    ) -> dict[str, Any]:
        returncode = structured_result.get("returncode")
        if isinstance(returncode, bool) or not isinstance(returncode, int):
            raise ValueError(
                f"structured returncode must be int, got {returncode!r}"
            )
        timed_out = structured_result.get("timed_out")
        if not isinstance(timed_out, bool):
            raise ValueError(
                f"structured timed_out must be boolean, got {timed_out!r}"
            )
        return {"returncode": returncode, "timed_out": timed_out}

    def _write_action(self, action: TraceAction) -> None:
        d = action.to_dict()
        self._actions.append(d)
        self.add_record(d)

    async def _resolve_pending_llm_records(self) -> None:
        for pending in self._pending_llm_records:
            response = pending["response"]
            if response is None or not getattr(response, "extra", None):
                continue
            task = response.extra.get("_openrouter_metadata_task")
            if task is not None:
                try:
                    await task
                except Exception as exc:
                    logger.warning("OpenRouter metadata task failed: {}", exc)
            await self._refresh_unavailable_openrouter_metadata(response)
            response.extra.pop("_openrouter_metadata_task", None)
            response.extra.pop("_openrouter_metadata_refetcher", None)
            response.extra["openrouter_metadata_task_pending"] = False
            llm_wall_latency_ms = float(pending["action_data"]["llm_wall_latency_ms"])
            llm_call_time_ms = self._resolve_llm_call_time_ms(
                response,
                llm_wall_latency_ms,
            )
            llm_timing_source = self._resolve_llm_timing_source(response)
            pending["event_data"]["llm_latency_ms"] = round(llm_call_time_ms, 2)
            pending["event_data"]["llm_call_time_ms"] = round(llm_call_time_ms, 2)
            pending["event_data"]["llm_timing_source"] = llm_timing_source
            pending["action_data"]["llm_latency_ms"] = round(llm_call_time_ms, 2)
            pending["action_data"]["llm_call_time_ms"] = round(llm_call_time_ms, 2)
            pending["action_data"]["llm_timing_source"] = llm_timing_source
            trace_llm_fields = self._extract_trace_llm_fields(response)
            pending["event_data"].update(
                {
                    key: value
                    for key, value in trace_llm_fields.items()
                    if key != "openrouter_metadata"
                }
            )
            pending["action_data"].update(trace_llm_fields)
            raw_response = pending["raw_response"]
            openrouter_metadata = response.extra.get("openrouter_metadata")
            if openrouter_metadata is not None:
                raw_response["openrouter_metadata"] = openrouter_metadata
            generation_id = response.extra.get("openrouter_generation_id")
            if generation_id is not None:
                raw_response["openrouter_generation_id"] = generation_id
        self._pending_llm_records.clear()

    @staticmethod
    async def _refresh_unavailable_openrouter_metadata(response: Any) -> None:
        if response is None or not getattr(response, "extra", None):
            return
        extra = response.extra
        if extra.get("openrouter_metadata_fetch_status") == "success":
            return
        generation_id = extra.get("openrouter_generation_id")
        refetcher = extra.get("_openrouter_metadata_refetcher")
        if not generation_id or not callable(refetcher):
            return

        initial_status = extra.get("openrouter_metadata_fetch_status")
        initial_fetch_ms = extra.get("openrouter_metadata_fetch_ms")
        try:
            refreshed = await refetcher()
        except Exception as exc:  # pragma: no cover - defensive guard
            extra["openrouter_metadata_refetch_attempted"] = True
            extra["openrouter_metadata_refetch_error"] = str(exc)
            return
        if not isinstance(refreshed, dict):
            return

        extra.update(refreshed)
        extra["openrouter_metadata_refetch_attempted"] = True
        if initial_status is not None:
            extra["openrouter_metadata_initial_fetch_status"] = initial_status
        if initial_fetch_ms is not None:
            extra["openrouter_metadata_initial_fetch_ms"] = initial_fetch_ms

    @staticmethod
    def _clone_messages(
        messages: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | None:
        if not messages:
            return None
        return json.loads(json.dumps(messages, ensure_ascii=False, default=str))

    @staticmethod
    def _clone_message_delta(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return TraceCollectorHook._clone_messages(messages) or []

    @staticmethod
    def _message_hash(message: dict[str, Any]) -> str:
        encoded = json.dumps(
            message,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _messages_delta_since_previous_snapshot(
        self,
        messages: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        current_messages = list(messages or [])
        current_hashes = [self._message_hash(message) for message in current_messages]
        previous_len = len(self._message_snapshot_hashes)
        if (
            len(current_hashes) < previous_len
            or current_hashes[:previous_len] != self._message_snapshot_hashes
        ):
            raise ValueError(
                "message history is not append-only; cannot record messages_delta"
            )
        delta_messages = current_messages[previous_len:]
        self._message_snapshot_hashes = current_hashes
        return self._clone_message_delta(delta_messages)

    def _iter_messages_payload(self) -> dict[str, Any]:
        if self.message_recording_mode == "full":
            return {"messages_in": self._iter_messages_snapshot}
        if self._iter_messages_delta is None:
            raise RuntimeError(
                "before_iteration must run before after_iteration in delta mode"
            )
        return {"messages_delta": self._iter_messages_delta, "is_delta": True}

    @staticmethod
    def _resolve_llm_ts_end(response: Any | None) -> float | None:
        if response is None or not getattr(response, "extra", None):
            return None
        value = response.extra.get("llm_wall_ts_end")
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _resolve_llm_call_time_ms(
        response: Any | None,
        llm_wall_latency_ms: float,
    ) -> float:
        if response is None or not getattr(response, "extra", None):
            return llm_wall_latency_ms
        for key in (
            "llm_call_time_ms",
            "openrouter_generation_time_ms",
            "llm_latency_ms",
        ):
            value = response.extra.get(key)
            try:
                if value is not None:
                    return float(value)
            except (TypeError, ValueError):
                continue
        return llm_wall_latency_ms

    @staticmethod
    def _resolve_llm_timing_source(response: Any | None) -> str:
        if response is None or not getattr(response, "extra", None):
            return "wall_clock_ms"
        source = response.extra.get("llm_timing_source")
        if isinstance(source, str) and source:
            return source
        if response.extra.get("openrouter_generation_time_ms") is not None:
            return "openrouter_generation_time_ms"
        return "wall_clock_ms"

    @staticmethod
    def _extract_trace_llm_fields(response: Any | None) -> dict[str, Any]:
        if response is None or not getattr(response, "extra", None):
            return {}
        extra = response.extra
        result = {}
        for key in (
            "llm_call_time_ms",
            "llm_timing_source",
            "openrouter_generation_id",
            "openrouter_request_id",
            "openrouter_latency_ms",
            "openrouter_generation_time_ms",
            "openrouter_moderation_latency_ms",
            "openrouter_provider_latency_ms",
            "openrouter_provider_name",
            "openrouter_upstream_id",
            "openrouter_created_at",
            "openrouter_api_type",
            "openrouter_metadata_capture_enabled",
            "openrouter_metadata_task_pending",
            "openrouter_metadata_retry_delays_s",
            "openrouter_metadata_timeout_s",
            "openrouter_metadata_fetch_ms",
            "openrouter_metadata_fetch_status",
            "openrouter_metadata_fetch_attempt_count",
            "openrouter_metadata_fetch_status_codes",
            "openrouter_metadata_fetch_last_status_code",
            "openrouter_metadata_fetch_last_reason",
            "openrouter_metadata_fetch_last_error_type",
            "openrouter_metadata_refetch_attempted",
            "openrouter_metadata_refetch_error",
            "openrouter_metadata_initial_fetch_status",
            "openrouter_metadata_initial_fetch_ms",
            "openrouter_metadata",
        ):
            if key in extra:
                result[key] = extra[key]
        return result

    def _build_raw_response(
        self,
        *,
        context: AgentHookContext,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> dict[str, Any]:
        resp = context.response
        message: dict[str, Any] = {
            "role": "assistant",
            "content": resp.content if resp else "",
        }
        if resp and resp.reasoning_content:
            message["reasoning_content"] = resp.reasoning_content
        if context.tool_calls:
            tool_calls = []
            for idx, tc in enumerate(context.tool_calls):
                arguments = tc.arguments
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                tool_calls.append(
                    {
                        "id": f"call_{context.iteration}_{idx}",
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": arguments,
                        },
                    }
                )
            message["tool_calls"] = tool_calls
        raw_response = {
            "choices": [
                {
                    "message": message,
                    "finish_reason": resp.finish_reason if resp else None,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }
        if resp and resp.extra:
            openrouter_metadata = resp.extra.get("openrouter_metadata")
            if openrouter_metadata is not None:
                raw_response["openrouter_metadata"] = openrouter_metadata
            generation_id = resp.extra.get("openrouter_generation_id")
            if generation_id is not None:
                raw_response["openrouter_generation_id"] = generation_id
        return raw_response

    async def write_summary(
        self,
        *,
        success: bool | None = None,
        elapsed_s: float = 0.0,
        prepare_ms: float | None = None,
    ) -> None:
        await self._await_pending_checkpoint_captures()
        await self._resolve_pending_llm_records()
        llm_summary = summarize_llm_latencies(
            a.get("data") for a in self._actions if a.get("action_type") == "llm_call"
        )
        summary = EvalTraceSummary(
            agent_id=self.agent_id,
            program_id=self.program_id,
            task_id=self.task_id,
            instance_id=self.instance_id,
            n_iterations=self._n_iterations,
            total_llm_ms=float(llm_summary["total_llm_ms"]),
            total_llm_wall_ms=float(llm_summary["total_llm_wall_ms"]),
            total_llm_call_time_ms=float(llm_summary["total_llm_call_time_ms"]),
            llm_call_time_count=int(llm_summary["llm_call_time_count"]),
            llm_timing_source=str(llm_summary["llm_timing_source"]),
            total_tool_ms=sum(self._tool_times.values()),
            total_tokens=self._total_tokens,
            tool_ms_by_name=self._tool_times,
            tool_timeouts=self._tool_timeouts,
            success=success,
            elapsed_s=elapsed_s,
            prepare_ms=prepare_ms,
        )
        summary_dict = summary.to_dict()
        summary_dict["hook_type"] = self.hook_type
        self.add_record(summary_dict)
        self.close()


def inject_event_callbacks(agent: AgentLoop, hook: TraceCollectorHook) -> None:
    def emit(category: str, event: str, data: dict, iteration: int = 0) -> None:
        hook.emit_event(category, event, data, iteration=iteration)

    agent.memory_consolidator._event_callback = lambda cat, evt, d, si=0: emit(
        cat, evt, d, si
    )
    agent.context.skills._event_callback = lambda cat, evt, d, si=0: emit(
        cat, evt, d, si
    )
    agent.sessions._event_callback = lambda cat, evt, d: emit(cat, evt, d)
    agent._event_callback = lambda cat, evt, d: emit(cat, evt, d)


@dataclass
class SessionRunResult:
    content: str | None
    elapsed_s: float
    trace_file: Path | None = None
    session_key: str = ""
    session_manager: SessionManager | None = None
    stop_reason: str = "completed"
    error: str | None = None


class SessionRunner:
    def __init__(
        self,
        provider: LLMProvider,
        *,
        model: str | None = None,
        max_iterations: int | None = None,
        context_window_tokens: int | None = None,
        max_tool_result_chars: int | None = None,
        mcp_servers: dict | None = None,
        extra_hooks: list[AgentHook] | None = None,
        exec_config: ExecToolConfig | None = None,
        malformed_retry_budget: int | None = None,
        container_runtime: dict | None = None,
        message_recording_mode: str = "full",
        checkpoint_scheduling: str = "sync",
        checkpoint_backend: str = "walk",
    ) -> None:
        self.provider = provider
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens or 65536
        self.max_tool_result_chars = max_tool_result_chars
        self.mcp_servers = mcp_servers or {}
        self.extra_hooks = extra_hooks or []
        self.exec_config = exec_config or ExecToolConfig()
        self.malformed_retry_budget = malformed_retry_budget
        self.container_runtime = container_runtime
        self.message_recording_mode = _validate_message_recording_mode(
            message_recording_mode
        )
        self.checkpoint_scheduling = _validate_checkpoint_scheduling(
            checkpoint_scheduling
        )
        self.checkpoint_backend = _validate_collection_checkpoint_backend(
            checkpoint_backend
        )

    @staticmethod
    def _scaffold_tools() -> list[str]:
        return [
            "read_file",
            "write_file",
            "edit_file",
            "list_dir",
            "exec",
            "web_search",
            "web_fetch",
            "message",
            "spawn",
        ]

    async def run(
        self,
        prompt: str,
        workspace: Path,
        *,
        tool_workspace: Path | None = None,
        project_workspace: Path | None = None,
        session_key: str,
        trace_file: Path,
        runtime_dir: Path | None = None,
        instance_id: str | None = None,
        channel: str = "cli",
        prepare_ms: float | None = None,
    ) -> SessionRunResult:
        workspace.mkdir(parents=True, exist_ok=True)
        trace_file = Path(trace_file)
        # Canonical runtime dir: explicit > trace-adjacent. All runtime state
        # (sessions, memory, skills, tool-results) lives under it, outside the
        # task/tool workspace, so it never contaminates a git-tracked target repo.
        effective_runtime_dir = (
            Path(runtime_dir)
            if runtime_dir is not None
            else trace_file.parent / "runtime"
        )
        effective_session_dir = effective_runtime_dir / "sessions"
        effective_memory_dir = effective_runtime_dir / "memory"
        effective_skills_dir = effective_runtime_dir / "skills"
        effective_tool_results_dir = effective_runtime_dir / "tool-results"
        effective_tool_workspace = tool_workspace or workspace
        effective_project_workspace = project_workspace or effective_tool_workspace
        iid = instance_id or session_key

        checkpoint_root: Path | None
        if self.container_runtime is not None:
            checkpoint_root = Path("/testbed")
        elif effective_tool_workspace.resolve() == Path("/testbed"):
            checkpoint_root = effective_tool_workspace
        else:
            checkpoint_root = None
        trace_hook = TraceCollectorHook(
            trace_file,
            iid,
            agent_id=iid,
            task_id=iid,
            checkpoint_root=checkpoint_root,
            checkpoint_dir=effective_runtime_dir / "checkpoints"
            if checkpoint_root is not None
            else None,
            container_runtime=self.container_runtime,
            message_recording_mode=self.message_recording_mode,
            checkpoint_scheduling=self.checkpoint_scheduling,
            checkpoint_backend=self.checkpoint_backend,
        )

        def subagent_trace_hook_factory(subagent_task_id: str) -> AgentHook:
            return TraceCollectorHook(
                trace_file,
                iid,
                agent_id=f"{iid}/{subagent_task_id}",
                task_id=subagent_task_id,
                checkpoint_root=checkpoint_root,
                checkpoint_dir=(
                    effective_runtime_dir / "checkpoints" / "subagents" / subagent_task_id
                    if checkpoint_root is not None
                    else None
                ),
                container_runtime=self.container_runtime,
                record_sink=trace_hook.add_record,
                hook_type="parent_subagent",
                message_recording_mode=self.message_recording_mode,
                checkpoint_scheduling=self.checkpoint_scheduling,
                checkpoint_backend=self.checkpoint_backend,
            )

        metadata: dict[str, Any] = {
            "type": "trace_metadata",
            "scaffold": "openclaw",
            "trace_format_version": 5,
            "mode": "collect",
            "model": self.model,
            "instance_id": iid,
            "session_key": session_key,
            "runtime_dir": str(effective_runtime_dir),
            "session_dir": str(effective_session_dir),
            "memory_dir": str(effective_memory_dir),
            "skills_dir": str(effective_skills_dir),
            "tool_results_dir": str(effective_tool_results_dir),
            "max_iterations": self.max_iterations,
            "scaffold_capabilities": {
                "tools": self._scaffold_tools(),
                "memory": True,
                "skills": True,
                "file_ops": "structured",
            },
        }
        if self.message_recording_mode != "full":
            metadata["run_config"] = {
                **dict(metadata.get("run_config") or {}),
                "message_recording_mode": self.message_recording_mode,
            }
        metadata["run_config"] = {
            **dict(metadata.get("run_config") or {}),
            "checkpoint_scheduling": self.checkpoint_scheduling,
            "checkpoint_backend": self.checkpoint_backend,
        }
        trace_hook.add_record(metadata)

        bus = MessageBus()
        collector = ResultCollector(bus)
        session_manager = SessionManager(workspace, storage_dir=effective_session_dir)

        all_hooks: list[AgentHook] = [trace_hook, *self.extra_hooks]
        agent = AgentLoop(
            bus=bus,
            provider=self.provider,
            workspace=workspace,
            tool_workspace=effective_tool_workspace,
            project_workspace=effective_project_workspace,
            model=self.model,
            max_iterations=self.max_iterations,
            context_window_tokens=self.context_window_tokens,
            max_tool_result_chars=self.max_tool_result_chars,
            exec_config=self.exec_config,
            mcp_servers=self.mcp_servers,
            session_manager=session_manager,
            session_dir=effective_session_dir,
            memory_dir=effective_memory_dir,
            skills_dir=effective_skills_dir,
            tool_results_dir=effective_tool_results_dir,
            hooks=all_hooks,
            subagent_trace_hook_factory=subagent_trace_hook_factory,
            malformed_retry_budget=self.malformed_retry_budget,
            container_runtime=self.container_runtime,
        )

        inject_event_callbacks(agent, trace_hook)

        wall_start = time.monotonic()

        chat_id = session_key.split(":", 1)[-1] if ":" in session_key else session_key
        result_key = f"{channel}:{chat_id}"

        async with AsyncExitStack() as stack:
            await collector.start()
            stack.callback(collector.stop)

            agent_task = asyncio.create_task(agent.run())
            stack.callback(agent.stop)

            msg = InboundMessage(
                channel="system",
                sender_id="user",
                chat_id=f"{channel}:{chat_id}",
                content=prompt,
                session_key_override=session_key,
            )
            await bus.publish_inbound(msg)

            content = await collector.wait_for_result(result_key)

        elapsed_s = time.monotonic() - wall_start

        try:
            await asyncio.wait_for(agent_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

        outcomes = getattr(agent, "_last_run_outcomes", {})
        outcome = outcomes.get(session_key, {})
        stop_reason, error = _resolve_run_outcome(
            outcome=outcome,
            content=content,
            trace_file=trace_file,
        )

        await trace_hook.write_summary(
            success=stop_reason == "completed",
            elapsed_s=elapsed_s,
            prepare_ms=prepare_ms,
        )

        return SessionRunResult(
            content=content,
            elapsed_s=elapsed_s,
            trace_file=trace_file,
            session_key=session_key,
            session_manager=session_manager,
            stop_reason=stop_reason,
            error=error,
        )


def __getattr__(name: str) -> object:
    if name == "_CHECKPOINT_SKIP_DIRS":
        from agents.sandbox_runtime import _CHECKPOINT_SKIP_DIRS
        return _CHECKPOINT_SKIP_DIRS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
