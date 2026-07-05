#!/usr/bin/env python3
"""FCBackend end-to-end smoke test — run with nohup on KVM host.

Usage (on KVM host):
    FC_KERNEL_PATH=/tmp/fc-cache/vmlinux-5.10.225 \
    PYTHONPATH=src nohup python3 scripts/smoke_test_fc.py \
    > /tmp/fc-smoke-$(date +%s).log 2>&1 &
"""

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from agents.sandbox_runtime import AgentTransportRequest, FCBackend  # noqa: E402
from harness.fc_rootfs_builder import build_fc_rootfs  # noqa: E402


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


async def main() -> bool:
    if not Path("/dev/kvm").exists():
        log("SKIP: no /dev/kvm")
        return True

    kernel_path = Path(os.environ.get("FC_KERNEL_PATH", "/tmp/fc-cache/vmlinux-5.10.225"))
    if not kernel_path.exists():
        log(f"SKIP: kernel not found at {kernel_path}")
        return True

    tmpdir = Path(tempfile.mkdtemp(prefix="fc-smoke-"))
    log(f"tmpdir={tmpdir}")

    # --- 1. Build rootfs ---
    log("[1/8] Building rootfs from python:3.12-slim ...")
    cache = build_fc_rootfs("python:3.12-slim")
    log(f"      cached at {cache} ({cache.stat().st_size / 1024 / 1024:.0f} MiB)")

    # Verify init script is present in the cached rootfs
    import subprocess
    mnt = tmpdir / "mnt"
    mnt.mkdir()
    subprocess.run(["sudo", "mount", "-o", "loop,ro", str(cache), str(mnt)],
                   capture_output=True, check=True, timeout=30)
    init_ok = (mnt / "sbin" / "init").exists()
    python_ok = (mnt / "usr/local/bin/python3").exists()
    env_json = (mnt / "etc" / "agent-env.json").exists()
    subprocess.run(["sudo", "umount", str(mnt)], capture_output=True, check=False, timeout=30)
    log(f"      init={init_ok}, python={python_ok}, agent_env={env_json}")
    assert init_ok, "Rootfs missing /sbin/init — builder cache may be stale"
    assert python_ok, "Rootfs missing python3"

    # --- 2. Start FC ---
    def req(tool: str, **kw: object) -> AgentTransportRequest:
        return AgentTransportRequest(tool=tool, args=kw)

    log("[2/8] Starting FCBackend ...")
    fc0 = FCBackend(
        source_image="python:3.12-slim",
        kernel_path=kernel_path,
        checkpoint_dir=tmpdir / "checkpoints",
        instance_id="00000000",
        api_sock=str(tmpdir / "fc.sock"),
        vsock_sock=str(tmpdir / "fc-vsock.sock"),
        tap_dev="fc-smoke-tap",
        host_ip="192.168.241.1",
        guest_ip="192.168.241.2",
        container_executable="docker",
    )
    await fc0.start()
    log(f"      started (tap={fc0._tap_dev})")

    # --- 3. Execute a command ---
    log("[3/8] echo hello > /testbed/f1.txt ...")
    r = await fc0.execute(req("exec", command="echo hello > /testbed/f1.txt; cat /testbed/f1.txt"))
    assert r.ok and "hello" in (r.result or ""), f"exec failed: {r}"
    log(f"      rc={r.returncode}, result={r.result.strip()!r}")

    # --- 4. Capture snapshot ---
    log("[4/8] capture_snapshot (paired, index 0) ...")
    snap0 = await fc0.capture_snapshot()
    disk0 = snap0.disk_state.get("disk_path", "?")
    mem0 = snap0.process_state is not None
    log(f"      disk={disk0}, mem={mem0}")

    # --- 5. Write file after snapshot ---
    log("[5/8] echo world > /testbed/f2.txt ...")
    r = await fc0.execute(req("exec", command="echo world > /testbed/f2.txt; cat /testbed/f1.txt; cat /testbed/f2.txt"))
    assert r.ok, f"post-snapshot exec failed: {r}"
    log(f"      rc={r.returncode}, has f1='hello': {'hello' in r.result}, has f2='world': {'world' in r.result}")

    # --- 6. Restore snapshot ---
    log("[6/8] restore_snapshot ...")
    restored = await fc0.restore_snapshot(snap0)
    log(f"      restored={restored}")

    # --- 7. Verify: f2 gone, f1 restored ---
    log("[7/8] Verifying f1 present, f2 absent ...")
    r1 = await fc0.execute(req("exec", command="cat /testbed/f1.txt"))
    r2 = await fc0.execute(req("exec", command="cat /testbed/f2.txt"))
    f1_ok = r1.ok and "hello" in (r1.result or "")
    f2_gone = not r2.ok or "No such file" in (r2.result or "") or "cannot open" in (r2.result or "").lower()
    log(f"      f1: {f1_ok} ({r1.result.strip()!r})")
    log(f"      f2: {f2_gone} ({r2.result.strip()!r})")
    assert f1_ok, f"f1.txt missing: {r1}"
    # Rollback is the core semantic under test: a paired (memory) snapshot
    # restore MUST roll back post-snapshot writes.  Only a disk-only
    # cold-boot cadence (process_state is None) may legitimately differ.
    if snap0.process_state is not None:
        assert f2_gone, (
            f"f2.txt survived a paired-snapshot restore — rollback failed: {r2}"
        )
    elif not f2_gone:
        log("      NOTE: disk-only cadence — rollback not asserted")

    # --- 8. Capture snapshot after restore (regression: _rootfs_path=None) ---
    log("[8/8] capture_snapshot after restore (regression test) ...")
    try:
        snap1 = await fc0.capture_snapshot()
        log(f"      OK: disk={snap1.disk_state.get('disk_path', '?')}")
    except Exception as e:
        log(f"      FAIL: {e}")
        return False

    log("\n=== PASS ===")
    await fc0.stop()
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
