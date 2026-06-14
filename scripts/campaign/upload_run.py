#!/usr/bin/env python3
"""Recompress a run's npz in parallel, THEN upload it to the HF dataset repo —
one command, no manual compress step.

For each given trace dir it (1) runs recompress_run.py (parallel, all cores) to
shrink the uncompressed npz back down, then (2) uploads the dir to the dataset
repo, preserving the ``terminal-bench/<model>/<ts>/...`` layout (the repo path is
derived by stripping everything up to and including the local ``traces/``
component). Recompression is lossless (zip-level verbatim member copy); see
recompress_run.py.

Usage:
    python scripts/campaign/upload_run.py <trace_dir> [<trace_dir> ...] \
        [--repo HCAHOI/agent-sched-bench] [-j N] [--no-recompress] [--dry-run]

Token: picked up from the HF_TOKEN env var or the cached HF login (same as the
rest of the box). For faster transfer, export HF_HUB_ENABLE_HF_TRANSFER=1.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def repo_path_for(local: Path) -> str:
    """Map a local trace dir to its repo path (strip up to and incl 'traces/')."""
    parts = local.parts
    if "traces" in parts:
        return "/".join(parts[parts.index("traces") + 1:])
    return local.name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+", help="run trace dir(s), e.g. traces/.../<ts>")
    ap.add_argument("--repo", default="HCAHOI/agent-sched-bench")
    ap.add_argument("--repo-type", default="dataset")
    ap.add_argument("-j", "--jobs", type=int, default=None, help="recompress workers")
    ap.add_argument("--no-recompress", action="store_true", help="skip recompress, upload as-is")
    ap.add_argument("--path-in-repo", default=None,
                    help="override derived repo path (only with a single dir)")
    ap.add_argument("--dry-run", action="store_true",
                    help="recompress + print the upload plan, but do NOT upload")
    args = ap.parse_args()

    if args.path_in_repo and len(args.dirs) != 1:
        print("error: --path-in-repo requires exactly one dir", file=sys.stderr)
        return 2

    from huggingface_hub import HfApi

    api = HfApi()
    try:
        who = api.whoami()  # fail fast on a missing/expired token
    except Exception as e:  # noqa: BLE001
        print(f"error: HF auth failed ({e}); set HF_TOKEN or hf login", file=sys.stderr)
        return 2
    print(f"HF user: {who.get('name')}  repo: {args.repo} ({args.repo_type})")

    recompress = Path(__file__).resolve().with_name("recompress_run.py")
    rc = 0
    for raw in args.dirs:
        d = Path(raw).resolve()
        if not d.is_dir():
            print(f"skip (not a dir): {d}", file=sys.stderr)
            rc = 1
            continue

        if not args.no_recompress:
            cmd = [sys.executable, str(recompress), str(d)]
            if args.jobs:
                cmd += ["-j", str(args.jobs)]
            print(f"\n== recompress {d} ==")
            if subprocess.run(cmd).returncode != 0:
                print(f"recompress FAILED for {d}; NOT uploading", file=sys.stderr)
                rc = 1
                continue

        path_in_repo = args.path_in_repo or repo_path_for(d)
        print(f"== upload {d} -> {args.repo}:{path_in_repo} ==")
        if args.dry_run:
            print("  (dry-run: skipping upload)")
            continue
        api.upload_folder(
            folder_path=str(d),
            path_in_repo=path_in_repo,
            repo_id=args.repo,
            repo_type=args.repo_type,
        )
        print(f"  uploaded {d}")

    return rc


if __name__ == "__main__":
    sys.exit(main())
