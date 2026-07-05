"""Container-side CAS checkpoint for OpenClaw host-mode runs.

Snapshot /testbed inside a running container via ``docker exec``, fetch
missing blobs via streaming tar, and produce a manifest whose structure
is byte-for-byte compatible with the host-side checkpoint in
``_session_runner._write_cas_manifest``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tarfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any


_CONTAINER_PYTHON_CANDIDATES = (
    "/usr/bin/python3",
    "/usr/bin/python",
    "/opt/miniconda3/bin/python3",
    "/opt/miniconda3/bin/python",
    "/opt/conda/bin/python3",
    "/opt/conda/bin/python",
    "python3",
    "python",
)

_CHECKPOINT_CAS_ROOT = Path.home() / ".cache" / "agent-checkpoint-cas"
_CHECKPOINT_SKIP_DIRS = frozenset({".git"})
CheckpointEntryType = str | dict[str, str]
CheckpointSnapshotEntry = (
    CheckpointEntryType | tuple[CheckpointEntryType, int, int]
)


def _unique_blob_tmp_path(blob_path: Path) -> Path:
    return blob_path.parent / f".tmp.{os.getpid()}.{uuid.uuid4().hex}"


def _checkpoint_relpath_is_skipped(relpath: str) -> bool:
    return any(part in _CHECKPOINT_SKIP_DIRS for part in relpath.split("/"))


def _checkpoint_snapshot_entry_type(
    entry: CheckpointSnapshotEntry,
) -> CheckpointEntryType:
    if isinstance(entry, tuple):
        return entry[0]
    return entry


def _normalize_checkpoint_entry_type(value: Any) -> CheckpointEntryType:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in value.items()
    ):
        return value
    raise RuntimeError(f"invalid checkpoint entry type: {value!r}")


def _normalize_checkpoint_snapshot_entry(value: Any) -> CheckpointSnapshotEntry:
    if isinstance(value, str) or isinstance(value, dict):
        return _normalize_checkpoint_entry_type(value)
    if isinstance(value, list | tuple) and len(value) == 3:
        entry_type = _normalize_checkpoint_entry_type(value[0])
        size = value[1]
        mtime_ns = value[2]
        if isinstance(size, bool) or not isinstance(size, int):
            raise RuntimeError(f"invalid checkpoint entry size: {value!r}")
        if isinstance(mtime_ns, bool) or not isinstance(mtime_ns, int):
            raise RuntimeError(f"invalid checkpoint entry mtime_ns: {value!r}")
        return (entry_type, size, mtime_ns)
    raise RuntimeError(f"invalid checkpoint snapshot entry: {value!r}")


def _normalize_checkpoint_snapshot_entries(
    entries: dict[str, Any],
) -> dict[str, CheckpointSnapshotEntry]:
    return {
        path: _normalize_checkpoint_snapshot_entry(entry)
        for path, entry in entries.items()
    }


_CONTAINER_SNAPSHOT_SCRIPT = (
    r"""
import json, os, stat, hashlib, sys, time

root = "/testbed"
since_ns_str = os.environ.get("CHECKPOINT_SINCE_NS", "")
since_ns = int(since_ns_str) if since_ns_str else 0
prev_entries_raw = sys.stdin.read()
prev_entries = json.loads(prev_entries_raw) if prev_entries_raw else {}
now_ns = time.time_ns()
skip_dirs = set(
"""
    f"{tuple(sorted(_CHECKPOINT_SKIP_DIRS))!r}"
    r"""
)

entries = {}
changed = {}
symlinks = []
changed_during_walk = False

def snapshot_entry(entry_type, st):
    return [entry_type, st.st_size, st.st_mtime_ns]

def entry_type(entry):
    if isinstance(entry, list) and len(entry) == 3:
        return entry[0]
    return entry

def entry_size_mtime(entry):
    if isinstance(entry, list) and len(entry) == 3:
        return (entry[1], entry[2])
    return None

def entry_changed(rel, entry):
    previous = prev_entries.get(rel)
    if previous is None:
        return True
    if entry_type(previous) != entry_type(entry):
        return True
    previous_size_mtime = entry_size_mtime(previous)
    current_size_mtime = entry_size_mtime(entry)
    if previous_size_mtime is None or current_size_mtime is None:
        return True
    return previous_size_mtime != current_size_mtime

def record_symlink(full, rel, st):
    try:
        target = os.readlink(full)
    except OSError:
        return
    entry_type_value = {"type": "symlink", "target": target}
    entry = snapshot_entry(entry_type_value, st)
    entries[rel] = entry
    if entry_changed(rel, entry):
        changed[rel] = entry_type_value

for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
    dirnames[:] = [name for name in dirnames if name not in skip_dirs]
    dirnames.sort()
    filenames.sort()

    for dname in dirnames:
        full = os.path.join(dirpath, dname)
        rel = os.path.relpath(full, root)
        try:
            dst = os.lstat(full)
        except OSError:
            continue
        if stat.S_ISLNK(dst.st_mode):
            record_symlink(full, rel, dst)

    if dirpath != root:
        rel = os.path.relpath(dirpath, root)
        try:
            dst = os.lstat(dirpath)
        except OSError:
            continue
        if stat.S_ISDIR(dst.st_mode):
            entries[rel] = snapshot_entry("dir", dst)
        elif stat.S_ISLNK(dst.st_mode):
            record_symlink(dirpath, rel, dst)

    for fname in filenames:
        full = os.path.join(dirpath, fname)
        rel = os.path.relpath(full, root)
        try:
            fst = os.lstat(full)
        except OSError:
            continue

        if stat.S_ISDIR(fst.st_mode):
            entries[rel] = snapshot_entry("dir", fst)
        elif stat.S_ISREG(fst.st_mode):
            entry = snapshot_entry("file", fst)
            entries[rel] = entry
            if not entry_changed(rel, entry):
                continue
            try:
                with open(full, "rb") as fh:
                    digest = hashlib.sha256(fh.read()).hexdigest()
            except OSError:
                continue
            changed[rel] = {
                "hash": digest,
                "mode": stat.S_IMODE(fst.st_mode),
                "size": fst.st_size,
                "mtime_ns": fst.st_mtime_ns,
            }
        elif stat.S_ISLNK(fst.st_mode):
            record_symlink(full, rel, fst)

for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
    dirnames[:] = [name for name in dirnames if name not in skip_dirs]
    for name in (*dirnames, *filenames):
        full = os.path.join(dirpath, name)
        try:
            fst = os.lstat(full)
        except OSError:
            changed_during_walk = True
            break
        if fst.st_mtime_ns > now_ns:
            changed_during_walk = True
            break
    if changed_during_walk:
        break

json.dump({
    "now_ns": now_ns,
    "entries": entries,
    "changed": changed,
    "symlinks": symlinks,
    "changed_during_walk": changed_during_walk,
}, sys.stdout)
sys.stdout.write("\n")
"""
)

_CONTAINER_BLOB_FETCH_SCRIPT = r"""
import json, os, sys, tarfile

paths = json.loads(sys.stdin.read())
tf = tarfile.open(fileobj=sys.stdout.buffer, mode="w|")
for rel in paths:
    full = os.path.join("/testbed", rel)
    try:
        tf.add(full, arcname=rel, recursive=False)
    except OSError:
        pass
tf.close()
"""


def _build_exec_cmd(
    container_runtime: dict[str, str],
    python_path: str,
    script: str,
    *,
    extra_env: dict[str, str] | None = None,
) -> list[str]:
    executable = container_runtime["executable"]
    container_id = container_runtime["id"]
    cmd = [
        executable,
        "exec",
        "-i",
        "-w",
        "/testbed",
    ]
    if extra_env:
        for key, value in extra_env.items():
            cmd.extend(["-e", f"{key}={value}"])
    cmd.extend([container_id, python_path, "-c", script])
    return cmd


def _probe_container_python(container_runtime: dict[str, str]) -> str:
    """Find a usable Python interpreter inside the container.

    Results are memoised on the container id so we probe once per run.
    """
    container_id = container_runtime["id"]
    cache_attr = f"_probe_cache_{container_id}"
    cached = getattr(_probe_container_python, cache_attr, None)
    if cached is not None:
        return cached

    executable = container_runtime["executable"]
    cid = container_runtime["id"]

    result = subprocess.run(
        [
            executable, "exec", "-i", "-w", "/testbed", cid,
            "/bin/sh", "-c",
            "for c in " + " ".join(_CONTAINER_PYTHON_CANDIDATES) + "; do "
            "if [ -x \"$c\" ] || command -v \"$c\" >/dev/null 2>&1; then "
            "if \"$c\" -c 'import sys; sys.exit(0 if sys.version_info >= (3,7) else 1)' >/dev/null 2>&1; then "
            "echo \"$c\"; exit 0; fi; fi; "
            "done; exit 1",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            "no usable python in container: "
            + (detail if detail else "no Python >=3.7 found")
        )

    python_path = result.stdout.strip()
    if not python_path:
        raise RuntimeError("no usable python in container: empty probe result")

    setattr(_probe_container_python, cache_attr, python_path)
    return python_path


def _run_snapshot(
    container_runtime: dict[str, str],
    python_path: str,
    since_ns: int | None,
    prev_entries: dict[str, CheckpointSnapshotEntry] | None = None,
    *,
    timeout: int = 600,
) -> dict[str, Any]:
    # TODO: remove after full migration to WalkBackend.
    """Run the snapshot script in the container and return parsed JSON."""
    extra_env: dict[str, str] = {}
    if since_ns is not None:
        extra_env["CHECKPOINT_SINCE_NS"] = str(since_ns)

    cmd = _build_exec_cmd(container_runtime, python_path, _CONTAINER_SNAPSHOT_SCRIPT, extra_env=extra_env)
    result = subprocess.run(
        cmd,
        input=json.dumps(prev_entries or {}),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"checkpoint snapshot failed: {detail}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"checkpoint snapshot returned invalid JSON: {exc}") from exc


def _fetch_blobs(
    container_runtime: dict[str, str],
    python_path: str,
    paths: list[str],
    *,
    timeout: int = 600,
) -> dict[str, bytes]:
    # TODO: remove after full migration to WalkBackend.
    """Fetch file contents via streaming tar, verify digests."""
    if not paths:
        return {}

    cmd = _build_exec_cmd(container_runtime, python_path, _CONTAINER_BLOB_FETCH_SCRIPT)
    stdin_data = json.dumps(paths).encode("utf-8")
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None

    stderr_chunks: list[bytes] = []

    def drain_stderr() -> None:
        stderr_chunks.append(process.stderr.read())

    timed_out = False

    def kill_on_timeout() -> None:
        nonlocal timed_out
        timed_out = True
        process.kill()

    stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
    timer = threading.Timer(timeout, kill_on_timeout)
    stderr_thread.start()
    timer.start()

    contents: dict[str, bytes] = {}
    stream_error: Exception | None = None
    returncode: int | None = None
    try:
        process.stdin.write(stdin_data)
        process.stdin.close()
        with tarfile.open(fileobj=process.stdout, mode="r|") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                contents[member.name] = fh.read()
        returncode = process.wait()
    except Exception as exc:
        stream_error = exc
    finally:
        timer.cancel()
        if process.poll() is None:
            process.kill()
            returncode = process.wait()
        elif returncode is None:
            returncode = process.returncode
        stderr_thread.join()

    stderr = b"".join(stderr_chunks)
    if timed_out:
        raise subprocess.TimeoutExpired(cmd, timeout, stderr=stderr)
    if returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"checkpoint blob fetch failed: {detail}")
    if stream_error is not None:
        raise stream_error
    return contents


def _compute_deleted_paths(
    prev_entries: dict[str, CheckpointSnapshotEntry] | None,
    current_entries: dict[str, CheckpointSnapshotEntry],
) -> list[str]:
    if prev_entries is None:
        return []
    return sorted(
        path
        for path, entry_type in prev_entries.items()
        if (
            (
                path not in current_entries
                or _checkpoint_snapshot_entry_type(current_entries[path])
                != _checkpoint_snapshot_entry_type(entry_type)
            )
            and not _checkpoint_relpath_is_skipped(path)
        )
    )


def _write_container_manifest(
    *,
    manifest_path: Path,
    changed: dict[str, dict[str, Any]],
    deleted_paths: list[str],
    verified_blobs: dict[str, bytes] | None = None,
) -> int:
    """Write manifest JSON and CAS blobs from container snapshot output.

    Returns total_unique_bytes for new blobs.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    cas_root = _CHECKPOINT_CAS_ROOT
    total_unique_bytes = 0

    entries: dict[str, dict[str, Any]] = {}
    for rel, info in sorted(changed.items()):
        if _checkpoint_relpath_is_skipped(rel):
            continue
        entry_type = info.get("type", "file")
        if entry_type == "symlink":
            target = info.get("target")
            if not isinstance(target, str):
                raise RuntimeError(f"checkpoint symlink target missing for {rel}")
            entries[rel] = {"type": "symlink", "target": target}
            continue
        if entry_type != "file":
            raise RuntimeError(f"unsupported checkpoint entry type for {rel}: {entry_type}")
        digest = info["hash"]
        blob_path = cas_root / "blobs" / digest[:2] / digest[2:]

        if not blob_path.exists():
            # Blob not in CAS yet — must have been fetched and verified
            # by the caller via verified_blobs.
            verified = (verified_blobs or {}).get(rel)
            if verified is None:
                raise RuntimeError(
                    f"checkpoint blob for {rel} ({digest}) not in CAS and not pre-verified"
                )
            expected_digest = hashlib.sha256(verified).hexdigest()
            if expected_digest != digest:
                raise RuntimeError(
                    f"checkpoint blob digest mismatch for {rel}: "
                    f"expected {digest}, got {expected_digest}"
                )
            blob_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = _unique_blob_tmp_path(blob_path)
            tmp.write_bytes(verified)
            tmp.rename(blob_path)
            total_unique_bytes += info["size"]

        entries[rel] = {
            "hash": digest,
            "mode": info["mode"],
            "size": info["size"],
            "mtime_ns": info["mtime_ns"],
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


def run_container_checkpoint(
    *,
    container_runtime: dict[str, str],
    manifest_path: Path,
    incremental_since_ns: int | None,
    prev_snapshot_entries: dict[str, CheckpointSnapshotEntry] | None,
    force_full: bool = False,
) -> dict[str, Any]:
    # TODO: remove after full migration to WalkBackend.
    """Execute a container-side checkpoint and return a result dict.

    The result has the same shape as the host-side ``_checkpoint_after_tool``
    return value: ok dict with path/kind/root/incremental/..., skipped dict,
    or error dict.

    When *force_full* is True the manifest is written as a full snapshot even
    when *prev_snapshot_entries* is not None (rebaseline).
    """
    started = time.monotonic()
    normalized_prev_entries = (
        None
        if prev_snapshot_entries is None
        else _normalize_checkpoint_snapshot_entries(prev_snapshot_entries)
    )

    try:
        python_path = _probe_container_python(container_runtime)
    except RuntimeError as exc:
        return {
            "error": f"checkpoint failed: {exc}",
            "overhead_excluded": True,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        }

    try:
        snapshot_prev_entries = None if force_full else normalized_prev_entries
        snapshot = _run_snapshot(
            container_runtime,
            python_path,
            incremental_since_ns,
            snapshot_prev_entries,
        )
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        return {
            "error": f"checkpoint snapshot failed: {exc}",
            "overhead_excluded": True,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        }

    if snapshot.get("changed_during_walk"):
        return {
            "error": "checkpoint failed: filesystem changed during checkpoint",
            "overhead_excluded": True,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        }

    current_entries = _normalize_checkpoint_snapshot_entries(snapshot["entries"])
    changed: dict[str, dict[str, Any]] = snapshot["changed"]
    now_ns: int = snapshot["now_ns"]

    deleted_paths = _compute_deleted_paths(normalized_prev_entries, current_entries)

    is_first = normalized_prev_entries is None
    has_changes = bool(changed or deleted_paths)

    if not is_first and not force_full and not has_changes:
        elapsed_ms = (time.monotonic() - started) * 1000
        return {
            "skipped": "no filesystem changes since last checkpoint",
            "overhead_excluded": True,
            "elapsed_ms": round(elapsed_ms, 3),
        }

    # Collect paths whose digests are missing from host CAS.
    missing_paths: list[str] = []
    cas_root = _CHECKPOINT_CAS_ROOT
    for rel, info in changed.items():
        if info.get("type", "file") != "file":
            continue
        digest = info["hash"]
        blob_path = cas_root / "blobs" / digest[:2] / digest[2:]
        if not blob_path.exists():
            missing_paths.append(rel)

    if missing_paths:
        try:
            blob_contents = _fetch_blobs(container_runtime, python_path, missing_paths)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            return {
                "error": f"checkpoint blob fetch failed: {exc}",
                "overhead_excluded": True,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            }

        # Verify digests before writing any blobs.
        for rel in missing_paths:
            content = blob_contents.get(rel)
            if content is None:
                return {
                    "error": f"checkpoint blob missing for {rel}",
                    "overhead_excluded": True,
                    "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
                }
            expected = changed[rel]["hash"]
            actual = hashlib.sha256(content).hexdigest()
            if actual != expected:
                return {
                    "error": f"checkpoint blob digest mismatch for {rel}: expected {expected}, got {actual}",
                    "overhead_excluded": True,
                    "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
                }

        # All digests verified — write manifest.
        verified_blobs = blob_contents
    else:
        verified_blobs = {}

    try:
        chain_bytes = _write_container_manifest(
            manifest_path=manifest_path,
            changed=changed,
            deleted_paths=deleted_paths,
            verified_blobs=verified_blobs,
        )
    except RuntimeError as exc:
        return {
            "error": f"checkpoint failed: {exc}",
            "overhead_excluded": True,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        }

    elapsed_ms = (time.monotonic() - started) * 1000
    size_bytes = manifest_path.stat().st_size

    is_full_for_kind = is_first or force_full
    return {
        "path": str(manifest_path),
        "kind": "cas_manifest_full" if is_full_for_kind else "cas_manifest_incremental",
        "root": "/testbed",
        "incremental": not is_full_for_kind,
        "incremental_since_ns": incremental_since_ns,
        "elapsed_ms": round(elapsed_ms, 3),
        "size_bytes": size_bytes,
        "overhead_excluded": True,
        "rebaseline": True if force_full else None,
        "chain_bytes": chain_bytes,
        "_state": {
            "now_ns": now_ns,
            "current_snapshot": current_entries,
            "chain_bytes": chain_bytes,
            "is_full": is_full_for_kind,
        },
    }
