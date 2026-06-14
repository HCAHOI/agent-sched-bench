#!/usr/bin/env python3
"""Recompress recorded ``*.npz`` under a trace dir, OFF the GPU critical path.

Recording writes npz uncompressed (``np.savez``) so the per-iter flush never
blocks the agent loop (zlib deflate on large fp16 arrays could stall a step for
minutes). This utility rewrites each npz with ``np.savez_compressed`` in
parallel across all cores AFTER a run finishes — when the GPU is idle — so that
upload/download transfers the small (compressed) form while recording stayed
fast.

Lossless by construction: an npz is a zip of ``.npy`` members. This operates at
the ZIP level — it copies each member's raw ``.npy`` bytes verbatim from the
(stored) source into a new (deflated) archive, WITHOUT deserializing the arrays.
So ``np.load`` returns byte-identical arrays for every dtype, including object
arrays (ragged data) and float arrays containing NaN, whose exact bit patterns
are preserved. Each rewrite is VERIFIED by re-reading the temp archive and
comparing every member's raw bytes against the source before an atomic replace;
a mismatch aborts that file without touching the original. Files already fully
deflated are skipped (idempotent re-runs).

Usage:
    python scripts/campaign/recompress_run.py <trace_dir> [-j N]

<trace_dir> is recursed for ``*.npz`` (e.g. a run-timestamp dir under
traces/<benchmark>/<model>/<ts>/). Safe to re-run (already-compressed files are
simply recompressed, a no-op in content). Do NOT point it at a run that is
still being written.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def recompress_one(path_str: str) -> tuple[str, int, int, str]:
    """Recompress one npz in place at the zip level, verifying byte-identity.
    Returns (path, size_before, size_after, status); status is 'ok', 'skip'
    (already fully deflated), or 'FAIL: ...'."""
    path = Path(path_str)
    tmp = path.with_name(path.name + ".recompress.tmp")
    try:
        before = path.stat().st_size
        with zipfile.ZipFile(path, "r") as zin:
            infos = zin.infolist()
            # Idempotent: skip if every member is already deflated.
            if infos and all(i.compress_type == zipfile.ZIP_DEFLATED for i in infos):
                return (path_str, before, before, "skip")
            members = [(i.filename, zin.read(i.filename)) for i in infos]

        # Copy each .npy member's raw bytes verbatim into a deflated archive.
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for name, data in members:
                zout.writestr(name, data)

        # Verify member names and raw bytes are unchanged (=> np.load identical).
        with zipfile.ZipFile(tmp, "r") as zchk:
            if zchk.namelist() != [n for n, _ in members]:
                raise ValueError("member list changed after recompress")
            for name, data in members:
                if zchk.read(name) != data:
                    raise ValueError(f"member '{name}' bytes differ after recompress")

        after = tmp.stat().st_size
        os.replace(tmp, path)
        return (path_str, before, after, "ok")
    except Exception as e:  # noqa: BLE001 — report any failure, never silently drop
        try:
            tmp.unlink()
        except OSError:
            pass
        return (path_str, 0, 0, f"FAIL: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", help="directory to recurse for *.npz")
    ap.add_argument(
        "-j", "--jobs", type=int, default=os.cpu_count() or 1,
        help="parallel workers (default: all cores)",
    )
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 2

    files = [str(p) for p in root.rglob("*.npz")]
    if not files:
        print(f"no .npz files under {root}")
        return 0

    print(f"recompressing {len(files)} npz under {root} with {args.jobs} workers ...")
    t0 = time.time()
    tot_before = tot_after = 0
    skipped = 0
    fails: list[tuple[str, str]] = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(recompress_one, f) for f in files]
        for fut in as_completed(futs):
            path, before, after, status = fut.result()
            done += 1
            if status == "ok":
                tot_before += before
                tot_after += after
            elif status == "skip":
                skipped += 1
            else:
                fails.append((path, status))
            if done % 50 == 0 or done == len(files):
                print(f"  {done}/{len(files)} done", flush=True)

    dt = time.time() - t0
    if tot_before:
        print(
            f"done in {dt:.1f}s: recompressed {tot_before / 1e9:.2f} GB -> "
            f"{tot_after / 1e9:.2f} GB ({tot_after / tot_before * 100:.0f}% of original, "
            f"{tot_before / tot_after:.1f}x shrink); {skipped} already-deflated skipped"
        )
    else:
        print(f"done in {dt:.1f}s; {skipped} already-deflated skipped")

    if fails:
        print(f"\n{len(fails)} FAILURE(S) — originals left untouched:", file=sys.stderr)
        for p, s in fails[:20]:
            print(f"  {p}: {s}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
