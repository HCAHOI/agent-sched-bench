#!/usr/bin/env python3
"""Prune unreferenced blobs from the checkpoint CAS.

Usage:
  .venv/bin/python -m scripts.prune_checkpoint_cas --simulate-output-dir <dir> [--dry-run]
  .venv/bin/python -m scripts.prune_checkpoint_cas --checkpoint-dirs <dir> [<dir> ...] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_CHECKPOINT_CAS_ROOT = Path.home() / ".cache" / "agent-checkpoint-cas"


def _discover_checkpoint_dirs(simulate_output_dir: Path) -> list[Path]:
    if not simulate_output_dir.is_dir():
        raise SystemExit(f"not a directory: {simulate_output_dir}")
    return sorted(
        path for path in simulate_output_dir.rglob("checkpoints") if path.is_dir()
    )


def _manifest_paths(checkpoint_dirs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for checkpoint_dir in checkpoint_dirs:
        if not checkpoint_dir.is_dir():
            raise SystemExit(f"not a directory: {checkpoint_dir}")
        paths.extend(
            path
            for path in checkpoint_dir.rglob("*manifest*.json")
            if path.is_file()
        )
    return sorted(set(paths))


def _entry_digest(entry: Any, *, manifest_path: Path, relpath: str) -> str | None:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        digest = entry.get("hash")
        if digest is None:
            return None
        if not isinstance(digest, str):
            raise SystemExit(
                f"invalid digest for {relpath} in {manifest_path}: {digest!r}"
            )
        return digest
    raise SystemExit(f"invalid manifest entry for {relpath} in {manifest_path}")


def _referenced_digests(manifest_paths: list[Path]) -> set[str]:
    referenced: set[str] = set()
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("entries")
        if entries is None:
            continue
        if not isinstance(entries, dict):
            raise SystemExit(f"manifest entries must be a dict: {manifest_path}")
        for relpath, entry in entries.items():
            if not isinstance(relpath, str):
                raise SystemExit(f"manifest relpath must be a string: {manifest_path}")
            digest = _entry_digest(
                entry,
                manifest_path=manifest_path,
                relpath=relpath,
            )
            if digest is not None:
                referenced.add(digest)
    return referenced


def _digest_for_blob_path(blob_path: Path, blobs_root: Path) -> str:
    rel = blob_path.relative_to(blobs_root)
    parts = rel.parts
    if len(parts) != 2:
        return ""
    return parts[0] + parts[1]


def _prune_blobs(
    *,
    cas_root: Path,
    referenced: set[str],
    dry_run: bool,
) -> tuple[int, int, int]:
    blobs_root = cas_root / "blobs"
    if not blobs_root.exists():
        return (0, 0, 0)
    if not blobs_root.is_dir():
        raise SystemExit(f"CAS blobs path is not a directory: {blobs_root}")

    total_blobs = 0
    deleted_blobs = 0
    bytes_freed = 0
    for blob_path in sorted(path for path in blobs_root.rglob("*") if path.is_file()):
        total_blobs += 1
        digest = _digest_for_blob_path(blob_path, blobs_root)
        if digest in referenced:
            continue
        size = blob_path.stat().st_size
        bytes_freed += size
        deleted_blobs += 1
        if not dry_run:
            blob_path.unlink()
    return (total_blobs, deleted_blobs, bytes_freed)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prune unreferenced blobs from ~/.cache/agent-checkpoint-cas.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--checkpoint-dirs",
        nargs="+",
        type=Path,
        help="Checkpoint directories to scan for *manifest*.json files.",
    )
    source.add_argument(
        "--simulate-output-dir",
        type=Path,
        help="Simulate output directory; checkpoint dirs are discovered below it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report deletions without removing blobs.",
    )
    parser.add_argument(
        "--cas-root",
        type=Path,
        default=_CHECKPOINT_CAS_ROOT,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.simulate_output_dir is not None:
        checkpoint_dirs = _discover_checkpoint_dirs(args.simulate_output_dir)
    else:
        checkpoint_dirs = [path.resolve() for path in args.checkpoint_dirs]

    if not checkpoint_dirs:
        raise SystemExit("no checkpoint directories found")

    manifests = _manifest_paths(checkpoint_dirs)
    if not manifests:
        raise SystemExit(
            "no *manifest*.json files found under: "
            + ", ".join(str(path) for path in checkpoint_dirs)
        )

    referenced = _referenced_digests(manifests)
    total_blobs, deleted_blobs, bytes_freed = _prune_blobs(
        cas_root=args.cas_root,
        referenced=referenced,
        dry_run=args.dry_run,
    )

    mode = "DRY RUN " if args.dry_run else ""
    print(f"{mode}checkpoint CAS prune summary")
    print(f"  checkpoint_dirs: {len(checkpoint_dirs)}")
    print(f"  manifests:       {len(manifests)}")
    print(f"  total_blobs:     {total_blobs}")
    print(f"  referenced:      {len(referenced)}")
    print(f"  deleted:         {deleted_blobs}")
    print(f"  bytes_freed:     {bytes_freed}")


if __name__ == "__main__":
    main()
