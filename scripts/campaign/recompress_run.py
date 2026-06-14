#!/usr/bin/env python3
"""Recompress recorded ``*.npz`` under a trace dir, OFF the GPU critical path.

Recording writes npz uncompressed (``np.savez``) so the per-iter flush never
blocks the agent loop (zlib deflate on large fp16 arrays could stall a step for
minutes). This utility rewrites each npz with ``np.savez_compressed`` in
parallel across all cores AFTER a run finishes — when the GPU is idle — so that
upload/download transfers the small (compressed) form while recording stayed
fast.

Lossless by construction: ``np.load`` returns byte-identical arrays for the
compressed and uncompressed npz container. Each rewrite is additionally
VERIFIED by reloading the temp file and comparing every array (dtype, shape,
values) against the source before an atomic replace; a mismatch aborts that
file without touching the original.

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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def recompress_one(path_str: str) -> tuple[str, int, int, str]:
    """Recompress one npz in place, verifying byte-identity. Returns
    (path, size_before, size_after, status). status == 'ok' on success."""
    path = Path(path_str)
    tmp = path.with_name(path.name + ".recompress.tmp")
    try:
        with np.load(path, allow_pickle=False) as src:
            arrays = {k: src[k] for k in src.files}
        before = path.stat().st_size

        # Write via an explicit file handle so numpy does not append ".npz".
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **arrays)

        # Verify the rewrite is byte-identical before replacing the original.
        with np.load(tmp, allow_pickle=False) as chk:
            if set(chk.files) != set(arrays):
                raise ValueError(
                    f"key mismatch after recompress: {set(chk.files)} vs {set(arrays)}"
                )
            for k, a in arrays.items():
                b = chk[k]
                if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
                    raise ValueError(f"array '{k}' differs after recompress")

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
    fails: list[tuple[str, str]] = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(recompress_one, f) for f in files]
        for fut in as_completed(futs):
            path, before, after, status = fut.result()
            done += 1
            if status != "ok":
                fails.append((path, status))
            else:
                tot_before += before
                tot_after += after
            if done % 50 == 0 or done == len(files):
                print(f"  {done}/{len(files)} done", flush=True)

    dt = time.time() - t0
    if tot_before:
        print(
            f"done in {dt:.1f}s: {tot_before / 1e9:.2f} GB -> {tot_after / 1e9:.2f} GB "
            f"({tot_after / tot_before * 100:.0f}% of original, "
            f"{tot_before / tot_after:.1f}x shrink)"
        )
    else:
        print(f"done in {dt:.1f}s")

    if fails:
        print(f"\n{len(fails)} FAILURE(S) — originals left untouched:", file=sys.stderr)
        for p, s in fails[:20]:
            print(f"  {p}: {s}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
