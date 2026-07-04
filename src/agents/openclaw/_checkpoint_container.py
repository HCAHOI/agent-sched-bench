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
import stat
import subprocess
import tarfile
import time
from io import BytesIO
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

_CONTAINER_SNAPSHOT_SCRIPT = r"""
import json, os, stat, hashlib, sys, time

root = "/testbed"
since_ns_str = os.environ.get("CHECKPOINT_SINCE_NS", "")
since_ns = int(since_ns_str) if since_ns_str else 0
now_ns = time.time_ns()

entries = {}
changed = {}
symlinks = []
changed_during_walk = False

for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
    dirnames.sort()
    filenames.sort()

    if dirpath != root:
        rel = os.path.relpath(dirpath, root)
        try:
            dst = os.lstat(dirpath)
        except OSError:
            continue
        if stat.S_ISDIR(dst.st_mode):
            entries[rel] = "dir"
        elif stat.S_ISLNK(dst.st_mode):
            symlinks.append(rel)

    for fname in filenames:
        full = os.path.join(dirpath, fname)
        rel = os.path.relpath(full, root)
        try:
            fst = os.lstat(full)
        except OSError:
            continue

        if stat.S_ISDIR(fst.st_mode):
            entries[rel] = "dir"
        elif stat.S_ISREG(fst.st_mode):
            entries[rel] = "file"
            if since_ns and fst.st_mtime_ns <= since_ns:
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
            symlinks.append(rel)

for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
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
    *,
    timeout: int = 600,
) -> dict[str, Any]:
    """Run the snapshot script in the container and return parsed JSON."""
    extra_env: dict[str, str] = {}
    if since_ns is not None:
        extra_env["CHECKPOINT_SINCE_NS"] = str(since_ns)

    cmd = _build_exec_cmd(container_runtime, python_path, _CONTAINER_SNAPSHOT_SCRIPT, extra_env=extra_env)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
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
    """Fetch file contents via streaming tar, verify digests."""
    if not paths:
        return {}

    cmd = _build_exec_cmd(container_runtime, python_path, _CONTAINER_BLOB_FETCH_SCRIPT)
    stdin_data = json.dumps(paths).encode("utf-8")
    result = subprocess.run(
        cmd,
        input=stdin_data,
        capture_output=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip() if result.stderr else ""
        raise RuntimeError(f"checkpoint blob fetch failed: {detail}")

    contents: dict[str, bytes] = {}
    with tarfile.open(fileobj=BytesIO(result.stdout), mode="r|") as tf:
        for member in tf:
            if not member.isfile():
                continue
            fh = tf.extractfile(member)
            if fh is None:
                continue
            contents[member.name] = fh.read()
    return contents


def _compute_deleted_paths(
    prev_entries: dict[str, str] | None,
    current_entries: dict[str, str],
) -> list[str]:
    if prev_entries is None:
        return []
    return sorted(
        path
        for path, entry_type in prev_entries.items()
        if current_entries.get(path) != entry_type
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
            tmp = blob_path.with_suffix(".tmp")
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
        "deleted_paths": sorted(deleted_paths) if deleted_paths else [],
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
    prev_snapshot_entries: dict[str, str] | None,
    force_full: bool = False,
) -> dict[str, Any]:
    """Execute a container-side checkpoint and return a result dict.

    The result has the same shape as the host-side ``_checkpoint_after_tool``
    return value: ok dict with path/kind/root/incremental/..., skipped dict,
    or error dict.

    When *force_full* is True the manifest is written as a full snapshot even
    when *prev_snapshot_entries* is not None (rebaseline).
    """
    started = time.monotonic()

    try:
        python_path = _probe_container_python(container_runtime)
    except RuntimeError as exc:
        return {
            "error": f"checkpoint failed: {exc}",
            "overhead_excluded": True,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        }

    try:
        snapshot = _run_snapshot(container_runtime, python_path, incremental_since_ns)
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        return {
            "error": f"checkpoint snapshot failed: {exc}",
            "overhead_excluded": True,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        }

    if snapshot.get("symlinks"):
        return {
            "error": "checkpoint skipped: symlinks under /testbed are unsupported",
            "overhead_excluded": True,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        }

    if snapshot.get("changed_during_walk"):
        return {
            "error": "checkpoint failed: filesystem changed during checkpoint",
            "overhead_excluded": True,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        }

    current_entries: dict[str, str] = snapshot["entries"]
    changed: dict[str, dict[str, Any]] = snapshot["changed"]
    now_ns: int = snapshot["now_ns"]

    prev = prev_snapshot_entries or {}
    deleted_paths = _compute_deleted_paths(prev_snapshot_entries, current_entries)

    is_first = prev_snapshot_entries is None
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
