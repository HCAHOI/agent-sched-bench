#!/usr/bin/env python3
"""End-to-end CAS checkpoint roundtrip verification.

Tests:
  1. CAS write (_write_cas_manifest) produces correct blobs + manifest
  2. CAS restore in a Docker container with --mount CAS only (no home mount)
  3. Incremental checkpoint chain (full -> incremental -> restore)
  4. Cleanup
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Add project root to path so we can import _write_cas_manifest
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.agents.openclaw._session_runner import _write_cas_manifest  # noqa: E402

_CAS_ROOT = Path.home() / ".cache" / "agent-checkpoint-cas"
_CONTAINER_PYTHON = "python3"
_FAILURES: list[str] = []


def check(condition: bool, msg: str) -> None:
    if not condition:
        _FAILURES.append(msg)
        print(f"  FAIL: {msg}")
    else:
        print(f"  PASS: {msg}")


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command, print it, and return the result."""
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, text=True, capture_output=True, **kwargs)


def assert_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run and assert success."""
    result = run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr}")
        print(f"  STDOUT: {result.stdout}")
        raise RuntimeError(f"Command failed (rc={result.returncode}): {' '.join(cmd)}")
    return result


# ============================================================================
# Test 1: CAS Write
# ============================================================================

def test_cas_write() -> tuple[Path, dict]:
    """Create test files, run _write_cas_manifest, verify blobs and manifest."""
    print("\n===== Test 1: CAS Write =====")

    # Create a temp directory with test files
    tmpdir = Path(tempfile.mkdtemp(prefix="cas_test_"))
    print(f"  Test root: {tmpdir}")

    # Create a few test files with known content
    test_files = {
        "hello.txt": b"Hello, CAS world!\n",
        "data/config.json": json.dumps(
            {"version": 1, "name": "test-config"}, indent=2
        ).encode(),
        "src/main.py": b"print('hello from test')\n",
        "empty.txt": b"",
    }

    for relpath, content in test_files.items():
        fpath = tmpdir / relpath
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_bytes(content)
        if relpath == "src/main.py":
            fpath.chmod(0o755)

    # Also create subdirs that _write_cas_manifest will skip (dirs are skipped)
    (tmpdir / "subdir").mkdir(exist_ok=True)

    # Write manifest
    manifest_path = tmpdir / "manifest.json"
    total_bytes = _write_cas_manifest(
        root=tmpdir,
        manifest_path=manifest_path,
        incremental_since_ns=None,
        deleted_paths=[],
        hash_cache={},
    )

    print(f"  Total unique new blob bytes: {total_bytes}")

    # Check manifest exists
    check(manifest_path.exists(), "manifest JSON written")
    manifest = json.loads(manifest_path.read_text())
    check(isinstance(manifest, dict), "manifest is a dict")
    check("entries" in manifest, "manifest has entries")
    check("deleted_paths" in manifest, "manifest has deleted_paths")
    check(manifest["deleted_paths"] == [], "deleted_paths is empty for full snapshot")

    entries = manifest["entries"]
    check(len(entries) == 4, f"4 file entries (got {len(entries)})")

    # Verify each entry has correct hash
    print("\n  Verifying blob storage and content hashes:")
    for relpath, info in entries.items():
        expected_content = test_files.get(relpath)
        check(expected_content is not None, f"entry for known file: {relpath}")
        if expected_content is None:
            continue

        expected_hash = hashlib.sha256(expected_content).hexdigest()
        check(info["hash"] == expected_hash, f"hash match for {relpath}")

        blob_path = _CAS_ROOT / "blobs" / expected_hash[:2] / expected_hash[2:]
        check(blob_path.exists(), f"blob exists: {relpath}")
        blob_content = blob_path.read_bytes()
        check(blob_content == expected_content, f"blob content match: {relpath}")
        check(info["mode"] == stat.S_IMODE(Path(tmpdir / relpath).stat().st_mode),
              f"mode match: {relpath}")
        check(info["size"] == len(expected_content), f"size match: {relpath}")

    print(f"\n  Manifest JSON (first 500 chars):")
    print(json.dumps(manifest, indent=2)[:500])

    return tmpdir, manifest


# ============================================================================
# Test 2: CAS Restore in Docker Container
# ============================================================================

def test_cas_restore_in_container(manifest: dict, tmpdir: Path) -> tuple[str, str]:
    """Start a Docker container with CAS mount only (no home), restore files."""
    print("\n===== Test 2: CAS Restore in Docker Container =====")

    # Find a Python image or pull one
    result = run(["docker", "images", "-q", "python:3.13-slim"])
    if not result.stdout.strip():
        print("  Pulling python:3.13-slim ...")
        assert_run(["docker", "pull", "python:3.13-slim"], timeout=120)

    # Write manifest to a known location for docker cp
    manifest_copy = tmpdir / "manifest_for_container.json"
    manifest_copy.write_text(json.dumps(manifest, indent=2))

    # Start container: no home mount, only CAS mount
    container_name = "cas_verify_test"
    # Kill any stale container
    run(["docker", "rm", "-f", container_name])

    print("\n  Starting container (--mount home=False, CAS bind-mounted)...")
    result = run([
        "docker", "run", "-d", "--rm",
        "--network=host",
        "-v", f"{_CAS_ROOT}:{_CAS_ROOT}",
        "--name", container_name,
        "python:3.13-slim",
        "sleep", "infinity",
    ])
    check(result.returncode == 0, f"container started (rc={result.returncode})")
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr}")
        raise RuntimeError("Failed to start Docker container")

    container_id = result.stdout.strip()
    print(f"  Container ID: {container_id}")

    # Verify container is running
    result = run(["docker", "inspect", "-f", "{{.State.Running}}", container_name])
    check(result.stdout.strip() == "true", "container is running")

    # Verify /home does NOT contain the host user's home directories
    result = run(["docker", "exec", container_name, "ls", "/home"])
    home_contents = result.stdout.strip().splitlines()
    # Host-specific markers that must NOT appear in container /home
    host_markers = {".cache", ".claude", ".codex", ".config", ".ssh", "agent-sched-bench"}
    leaked = set(home_contents) & host_markers
    check(not leaked, f"no host home dirs leaked into /home (found: {leaked})")

    # Verify /root is NOT the host home
    result = run(["docker", "exec", container_name, "ls", "/root"])
    # It should be a fresh container /root
    host_home_contents = set(os.listdir(os.path.expanduser("~")))
    container_root = result.stdout.strip().splitlines()
    overlap = set(container_root) & host_home_contents
    check(not overlap, f"container /root does NOT mirror host home (overlap: {overlap})")

    # Copy manifest into container
    container_manifest = "/tmp/manifest.json"
    print(f"\n  Copying manifest into container...")
    result = run(["docker", "cp", str(manifest_copy), f"{container_name}:{container_manifest}"])
    check(result.returncode == 0, "docker cp manifest success")

    # Set CAS_ROOT env and run restore script
    restore_target = "/testbed"
    restore_script = r'''
import json, os, shutil, stat
manifest_path = os.environ["CAS_MANIFEST_PATH"]
cas_root = os.environ["CAS_ROOT"]
root = os.path.abspath(os.environ["CHECKPOINT_ROOT"])
clear_root = os.environ.get("CHECKPOINT_CLEAR_ROOT") == "1"
if os.path.lexists(root):
    if os.path.islink(root):
        os.unlink(root)
        os.makedirs(root, exist_ok=True)
    elif not os.path.isdir(root):
        os.unlink(root)
        os.makedirs(root, exist_ok=True)
else:
    os.makedirs(root, exist_ok=True)
root_real = os.path.realpath(root)
if root_real != root:
    raise RuntimeError(f"checkpoint root symlinks are unsupported: {root}")

with open(manifest_path, "r") as f:
    manifest = json.load(f)

entries = manifest.get("entries", {})
deleted = manifest.get("deleted_paths", [])
if not isinstance(entries, dict):
    raise RuntimeError("checkpoint manifest missing 'entries' dict")
if not isinstance(deleted, list):
    raise RuntimeError("checkpoint manifest has invalid 'deleted_paths'")

def safe_target(relpath):
    if not isinstance(relpath, str) or relpath == "":
        raise RuntimeError(f"unsafe checkpoint path: {relpath}")
    if os.path.isabs(relpath) or ".." in relpath.split(os.sep):
        raise RuntimeError(f"unsafe checkpoint path: {relpath}")
    target = os.path.abspath(os.path.join(root, relpath))
    if target == root or not target.startswith(root + os.sep):
        raise RuntimeError(f"unsafe checkpoint path: {relpath}")
    return target

def ensure_parent_dir(target):
    parent = os.path.dirname(target)
    rel_parent = os.path.relpath(parent, root)
    current = root
    if rel_parent == ".":
        return
    for part in rel_parent.split(os.sep):
        current = os.path.join(current, part)
        if os.path.lexists(current):
            if os.path.islink(current) or not os.path.isdir(current):
                raise RuntimeError(f"unsafe checkpoint parent path: {current}")
        else:
            os.mkdir(current)

for relpath in list(entries) + deleted:
    safe_target(relpath)

if clear_root:
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.unlink(path)
else:
    for del_path in sorted(deleted, key=lambda p: p.count(os.sep), reverse=True):
        target = safe_target(del_path)
        if os.path.lexists(target):
            if os.path.isdir(target) and not os.path.islink(target):
                shutil.rmtree(target)
            else:
                os.unlink(target)

for relpath, entry in entries.items():
    if not isinstance(entry, dict):
        raise RuntimeError(f"invalid checkpoint manifest entry: {relpath}")
    hash_val = entry["hash"]
    blob_path = os.path.join(cas_root, "blobs", hash_val[:2], hash_val[2:])
    with open(blob_path, "rb") as f:
        content = f.read()
    target = safe_target(relpath)
    ensure_parent_dir(target)
    if os.path.lexists(target):
        if os.path.islink(target):
            raise RuntimeError(f"checkpoint target symlinks are unsupported: {relpath}")
        if os.path.isdir(target):
            shutil.rmtree(target)
        elif not stat.S_ISREG(os.stat(target).st_mode):
            raise RuntimeError(f"checkpoint special targets are unsupported: {relpath}")
    with open(target, "wb") as f:
        f.write(content)
    os.chmod(target, entry.get("mode", 0o644))
    mtime_ns = entry.get("mtime_ns")
    if mtime_ns is not None:
        try:
            os.utime(target, ns=(mtime_ns, mtime_ns))
        except OSError:
            pass
if not os.path.isdir(root):
    raise RuntimeError(f"checkpoint root missing after restore: {root}")
if os.path.exists(manifest_path):
    os.unlink(manifest_path)
print("RESTORE_OK")
'''

    print("\n  Running restore script in container...")
    result = run([
        "docker", "exec",
        "-e", f"CAS_MANIFEST_PATH={container_manifest}",
        "-e", f"CAS_ROOT={_CAS_ROOT}",
        "-e", f"CHECKPOINT_ROOT={restore_target}",
        "-e", "CHECKPOINT_CLEAR_ROOT=1",
        container_name,
        _CONTAINER_PYTHON, "-c", restore_script,
    ])
    check(result.returncode == 0, f"restore script exit 0 (rc={result.returncode})")
    check("RESTORE_OK" in result.stdout, "restore script printed RESTORE_OK")
    if result.returncode != 0:
        print(f"  STDOUT: {result.stdout}")
        print(f"  STDERR: {result.stderr}")
        raise RuntimeError("Restore script failed")

    # Verify restored content via docker exec
    print("\n  Verifying restored files in container:")
    for relpath, info in manifest["entries"].items():
        result = run(["docker", "exec", container_name, "cat", f"{restore_target}/{relpath}"])
        restored_content = result.stdout.encode()
        blob = (_CAS_ROOT / "blobs" / info["hash"][:2] / info["hash"][2:]).read_bytes()
        check(restored_content == blob, f"restored content matches: {relpath}")

        # Check mode
        result = run(["docker", "exec", container_name, "stat", "-c", "%a",
                       f"{restore_target}/{relpath}"])
        restored_mode_str = result.stdout.strip()
        expected_mode_str = format(info["mode"], "o")
        check(restored_mode_str == expected_mode_str,
              f"mode match: {relpath} (got {restored_mode_str}, expected {expected_mode_str})")

    # Verify no host home leakage
    result = run(["docker", "exec", container_name, "ls", "/root"])
    root_contents = result.stdout.strip().splitlines()
    check(".cache" not in root_contents and ".claude" not in root_contents,
          "no host home files leaked into container /root")

    print("\n  CAS restore in container: PASSED")

    # Leave container running for Test 3
    return container_name, container_id


# ============================================================================
# Test 3: Incremental Checkpoint Chain
# ============================================================================

def test_incremental_chain(tmpdir: Path, container_name: str) -> list[Path]:
    """Create full + incremental checkpoints, restore chain, verify state."""
    print("\n===== Test 3: Incremental Checkpoint Chain =====")

    # Use a fresh temp dir (not the original tmpdir)
    workdir = Path(tempfile.mkdtemp(prefix="cas_incr_"))
    print(f"  Incremental test root: {workdir}")

    # Step A: Create full checkpoint with initial files
    initial_files = {
        "README.md": b"# My Project\n\nThis is the initial version.\n",
        "src/main.py": b"print('v1')\n",
        "data/scores.json": b'{"a": 1, "b": 2}\n',
    }
    for relpath, content in initial_files.items():
        fpath = workdir / relpath
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_bytes(content)

    full_manifest_path = workdir / "full_manifest.json"
    hash_cache: dict[str, tuple[int, str]] = {}
    full_bytes = _write_cas_manifest(
        root=workdir,
        manifest_path=full_manifest_path,
        incremental_since_ns=None,
        deleted_paths=[],
        hash_cache=hash_cache,
    )
    print(f"  Full manifest: {full_bytes} unique blob bytes")
    full = json.loads(full_manifest_path.read_text())
    check(len(full["entries"]) == 3, f"full manifest has 3 entries (got {len(full['entries'])})")
    snapshot_ns = time.time_ns()

    # Step B: Modify one file, add one file, delete one file
    time.sleep(0.5)  # Ensure mtime changes are detectable
    (workdir / "src/main.py").write_text("print('v2 - modified')\n")
    (workdir / "new_feature.py").write_text("def do_new_thing():\n    return 42\n")
    (workdir / "README.md").unlink()

    # Step C: Create incremental checkpoint
    incr_manifest_path = workdir / "incr_manifest.json"
    incr_bytes = _write_cas_manifest(
        root=workdir,
        manifest_path=incr_manifest_path,
        incremental_since_ns=snapshot_ns,
        deleted_paths=["README.md"],
        hash_cache=hash_cache,
    )
    print(f"  Incremental manifest: {incr_bytes} unique blob bytes")
    incr = json.loads(incr_manifest_path.read_text())
    print(f"  Incremental entries: {list(incr['entries'].keys())}")
    print(f"  Deleted paths: {incr['deleted_paths']}")
    check(len(incr["entries"]) == 2, f"incremental has 2 entries (got {len(incr['entries'])})")
    check("src/main.py" in incr["entries"], "src/main.py in incremental")
    check("new_feature.py" in incr["entries"], "new_feature.py in incremental")
    check(incr["deleted_paths"] == ["README.md"], "README.md in deleted_paths")

    # Step D: Restore chain in the Docker container
    # First restore full
    restore_target = "/testbed_incr"
    container_full_manifest = "/tmp/full_manifest.json"
    container_incr_manifest = "/tmp/incr_manifest.json"

    # Copy both manifests
    result = run(["docker", "cp", str(full_manifest_path), f"{container_name}:{container_full_manifest}"])
    check(result.returncode == 0, "docker cp full manifest")
    result = run(["docker", "cp", str(incr_manifest_path), f"{container_name}:{container_incr_manifest}"])
    check(result.returncode == 0, "docker cp incremental manifest")

    # Restore full (clear_root=True)
    restore_script_full = r'''
import json, os, shutil, stat
manifest_path = os.environ["CAS_MANIFEST_PATH"]
cas_root = os.environ["CAS_ROOT"]
root = os.path.abspath(os.environ["CHECKPOINT_ROOT"])
with open(manifest_path, "r") as f:
    manifest = json.load(f)
entries = manifest["entries"]
# Clear root
if os.path.lexists(root):
    if os.path.islink(root):
        os.unlink(root)
        os.makedirs(root, exist_ok=True)
    elif not os.path.isdir(root):
        os.unlink(root)
        os.makedirs(root, exist_ok=True)
else:
    os.makedirs(root, exist_ok=True)
for name in os.listdir(root):
    path = os.path.join(root, name)
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    else:
        os.unlink(path)
for relpath, entry in entries.items():
    target = os.path.join(root, relpath)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    hash_val = entry["hash"]
    blob = os.path.join(cas_root, "blobs", hash_val[:2], hash_val[2:])
    with open(blob, "rb") as f:
        content = f.read()
    with open(target, "wb") as f:
        f.write(content)
    os.chmod(target, entry["mode"])
print("FULL_RESTORE_OK")
'''

    print("\n  Restoring full checkpoint...")
    result = run([
        "docker", "exec",
        "-e", f"CAS_MANIFEST_PATH={container_full_manifest}",
        "-e", f"CAS_ROOT={_CAS_ROOT}",
        "-e", f"CHECKPOINT_ROOT={restore_target}",
        container_name,
        _CONTAINER_PYTHON, "-c", restore_script_full,
    ])
    check(result.returncode == 0, "full restore exit 0")
    check("FULL_RESTORE_OK" in result.stdout, "full restore OK")

    # Verify full restore state
    result = run(["docker", "exec", container_name, "cat", f"{restore_target}/src/main.py"])
    check(result.stdout.strip() == "print('v1')", "full restore: main.py is v1")
    result = run(["docker", "exec", container_name, "cat", f"{restore_target}/README.md"])
    check("initial version" in result.stdout, "full restore: README.md present")
    result = run(["docker", "exec", container_name, "test", "-f", f"{restore_target}/new_feature.py"])
    check(result.returncode != 0, "full restore: new_feature.py does NOT exist")

    # Now restore incremental on top (clear_root=False)
    restore_script_incr = r'''
import json, os, shutil, stat
manifest_path = os.environ["CAS_MANIFEST_PATH"]
cas_root = os.environ["CAS_ROOT"]
root = os.path.abspath(os.environ["CHECKPOINT_ROOT"])
with open(manifest_path, "r") as f:
    manifest = json.load(f)
entries = manifest["entries"]
deleted = manifest.get("deleted_paths", [])
for del_path in sorted(deleted, key=lambda p: p.count(os.sep), reverse=True):
    target = os.path.join(root, del_path)
    if os.path.lexists(target):
        if os.path.isdir(target) and not os.path.islink(target):
            shutil.rmtree(target)
        else:
            os.unlink(target)
for relpath, entry in entries.items():
    target = os.path.join(root, relpath)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    hash_val = entry["hash"]
    blob = os.path.join(cas_root, "blobs", hash_val[:2], hash_val[2:])
    with open(blob, "rb") as f:
        content = f.read()
    if os.path.lexists(target):
        if os.path.isdir(target):
            shutil.rmtree(target)
    with open(target, "wb") as f:
        f.write(content)
    os.chmod(target, entry["mode"])
print("INCR_RESTORE_OK")
'''

    print("\n  Restoring incremental checkpoint...")
    result = run([
        "docker", "exec",
        "-e", f"CAS_MANIFEST_PATH={container_incr_manifest}",
        "-e", f"CAS_ROOT={_CAS_ROOT}",
        "-e", f"CHECKPOINT_ROOT={restore_target}",
        container_name,
        _CONTAINER_PYTHON, "-c", restore_script_incr,
    ])
    check(result.returncode == 0, "incremental restore exit 0")
    check("INCR_RESTORE_OK" in result.stdout, "incremental restore OK")
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr}")
        return

    # Verify final state: modified file, new file present, deleted file gone
    result = run(["docker", "exec", container_name, "cat", f"{restore_target}/src/main.py"])
    check(result.stdout.strip() == "print('v2 - modified')", "incr restore: main.py is v2-modified")

    result = run(["docker", "exec", container_name, "cat", f"{restore_target}/new_feature.py"])
    check("def do_new_thing" in result.stdout, "incr restore: new_feature.py present")

    result = run(["docker", "exec", container_name, "test", "-f", f"{restore_target}/README.md"])
    check(result.returncode != 0, "incr restore: README.md deleted")

    result = run(["docker", "exec", container_name, "test", "-f", f"{restore_target}/data/scores.json"])
    check(result.returncode == 0, "incr restore: scores.json still present (carried over)")

    print("\n  Incremental checkpoint chain: PASSED")
    return [workdir]


# ============================================================================
# Test 4: Cleanup
# ============================================================================

def test_cleanup(container_name: str, tmpdirs: list[Path]) -> None:
    """Remove test containers, temp dirs, and CAS test blobs (optional)."""
    print("\n===== Test 4: Cleanup =====")
    result = run(["docker", "rm", "-f", container_name])
    print(f"  Container removed (rc={result.returncode})")
    for d in tmpdirs:
        shutil.rmtree(d, ignore_errors=True)
        print(f"  Removed temp dir: {d}")
    print("  CAS blobs left in place (content-addressed, dedup-safe)")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 70)
    print("CAS CHECKPOINT ROUNDTRIP VERIFICATION")
    print(f"Project root: {_PROJECT_ROOT}")
    print(f"CAS root:     {_CAS_ROOT}")
    print(f"Python:       {sys.version}")
    print("=" * 70)

    tmpdirs: list[Path] = []
    container_name = ""

    try:
        # Test 1
        tmpdir, manifest = test_cas_write()
        tmpdirs.append(tmpdir)

        # Test 2
        container_name, container_id = test_cas_restore_in_container(manifest, tmpdir)

        # Test 3
        incr_dirs = test_incremental_chain(tmpdir, container_name)
        tmpdirs.extend(incr_dirs)

        # Test 4
        test_cleanup(container_name, tmpdirs)

    except Exception as exc:
        _FAILURES.append(f"EXCEPTION: {exc}")
        print(f"\n  EXCEPTION: {exc}")
        import traceback
        traceback.print_exc()

        # Best-effort cleanup
        if container_name:
            run(["docker", "rm", "-f", container_name])
        for d in tmpdirs:
            shutil.rmtree(d, ignore_errors=True)

    print("\n" + "=" * 70)
    if _FAILURES:
        print(f"VERIFICATION FAILED - {len(_FAILURES)} failure(s):")
        for f in _FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
