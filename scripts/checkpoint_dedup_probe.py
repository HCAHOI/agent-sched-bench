#!/usr/bin/env python3
"""Measure file-level content-addressed dedup potential across checkpoint tars.

Usage:
  .venv/bin/python scripts/checkpoint_dedup_probe.py <run_dir> [<run_dir> ...]
  .venv/bin/python scripts/checkpoint_dedup_probe.py <run_dir> --json <path>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path


def _iter_tar_paths(run_dir: Path) -> list[Path]:
    """Discover *-after.tar files under run_dir/checkpoints/."""
    return sorted(run_dir.rglob("checkpoints/*-after.tar"))


def _scan_tar(tar_path: Path) -> dict:
    """Stream one tar and return stats.

    Returns dict with keys: tar_path, tar_bytes, total_file_bytes,
    file_hashes (Counter of hexdigest -> count), hash_to_example_path
    (hexdigest -> first path seen), hash_to_size (hexdigest -> size).
    """
    tar_bytes = tar_path.stat().st_size
    total_file_bytes = 0
    file_hashes: Counter[str] = Counter()
    hash_to_example_path: dict[str, str] = {}
    hash_to_size: dict[str, int] = {}

    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            hasher = hashlib.sha256()
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                hasher.update(chunk)
            digest = hasher.hexdigest()
            total_file_bytes += member.size
            file_hashes[digest] += 1
            if digest not in hash_to_example_path:
                hash_to_example_path[digest] = (
                    f"{tar_path.name}:{member.name}"
                )
                hash_to_size[digest] = member.size

    return {
        "tar_path": str(tar_path),
        "tar_bytes": tar_bytes,
        "total_file_bytes": total_file_bytes,
        "file_hashes": dict(file_hashes),
        "hash_to_example_path": hash_to_example_path,
        "hash_to_size": hash_to_size,
    }


def _merge_stats(results: list[dict]) -> dict:
    """Merge per-tar results into aggregate."""
    n_tars = len(results)
    total_tar_bytes = sum(r["tar_bytes"] for r in results)
    total_file_bytes = sum(r["total_file_bytes"] for r in results)

    global_hash_counts: Counter[str] = Counter()
    global_hash_example: dict[str, str] = {}
    global_hash_size: dict[str, int] = {}

    for r in results:
        for digest, count in r["file_hashes"].items():
            global_hash_counts[digest] += count
        for digest, path in r["hash_to_example_path"].items():
            if digest not in global_hash_example:
                global_hash_example[digest] = path
        for digest, size in r["hash_to_size"].items():
            if digest not in global_hash_size:
                global_hash_size[digest] = size

    unique_bytes = sum(
        global_hash_size[d] for d in global_hash_counts
    )

    dup_ratio = total_file_bytes / unique_bytes if unique_bytes else 0.0

    # Top 20 largest duplicated blobs (appearing in 2+ files)
    duplicated = [
        {
            "hash": d,
            "size": global_hash_size.get(d, 0),
            "count": c,
            "example_path": global_hash_example.get(d, ""),
        }
        for d, c in global_hash_counts.items()
        if c >= 2
    ]
    duplicated.sort(key=lambda x: -x["size"] * x["count"])
    top_20 = duplicated[:20]

    return {
        "n_tars": n_tars,
        "total_tar_bytes": total_tar_bytes,
        "total_file_bytes": total_file_bytes,
        "unique_bytes": unique_bytes,
        "dedup_ratio": round(dup_ratio, 4),
        "global_unique_hashes": len(global_hash_counts),
        "global_total_hashes": sum(global_hash_counts.values()),
        "top_duplicated": top_20,
    }


def _report_text(overall: dict, per_run: dict[str, dict]) -> str:
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("CHECKPOINT DEDUP PROBE REPORT")
    lines.append("=" * 60)

    lines.append(f"\n{'Run dir':<50} {'Tars':>5} {'Tar MB':>9} {'File MB':>9} {'Unique MB':>9} {'Ratio':>7}")
    lines.append("-" * 90)

    for run_dir, stats in sorted(per_run.items()):
        tar_mb = stats["total_tar_bytes"] / 1_000_000
        file_mb = stats["total_file_bytes"] / 1_000_000
        unique_mb = stats["unique_bytes"] / 1_000_000
        ratio = f"{stats['dedup_ratio']:.2f}" if stats["unique_bytes"] else "-"
        short_dir = run_dir[:48] + ".." if len(run_dir) > 50 else run_dir
        lines.append(
            f"  {short_dir:<48} {stats['n_tars']:>5} {tar_mb:>9.2f} {file_mb:>9.2f} "
            f"{unique_mb:>9.2f} {ratio:>7}"
        )

    lines.append("-" * 90)
    tar_mb = overall["total_tar_bytes"] / 1_000_000
    file_mb = overall["total_file_bytes"] / 1_000_000
    unique_mb = overall["unique_bytes"] / 1_000_000
    lines.append(
        f"  {'TOTAL':<48} {overall['n_tars']:>5} {tar_mb:>9.2f} {file_mb:>9.2f} "
        f"{unique_mb:>9.2f} {overall['dedup_ratio']:>7.2f}"
    )
    lines.append("")

    lines.append(f"\nNaive storage:  {overall['total_file_bytes'] / 1_000_000:.2f} MB")
    lines.append(f"Unique content: {overall['unique_bytes'] / 1_000_000:.2f} MB")
    lines.append(f"Dedup ratio:    {overall['dedup_ratio']:.4f}x")
    lines.append(f"Total hashes:   {overall['global_total_hashes']}")
    lines.append(f"Unique hashes:  {overall['global_unique_hashes']}")
    lines.append("")

    if overall["top_duplicated"]:
        lines.append("Top 20 largest duplicated blobs:")
        lines.append(f"  {'Hash':<20} {'Size':>10} {'Count':>6} {'Example path':<60}")
        lines.append("  " + "-" * 96)
        for entry in overall["top_duplicated"]:
            h = entry["hash"][:18]
            size_mb = entry["size"] / 1_000_000
            lines.append(
                f"  {h:<20} {size_mb:>10.2f} MB {entry['count']:>6} {entry['example_path']:<60}"
            )
        lines.append("")

    # Within-run vs cross-run dedup analysis
    within_unique = 0
    for stats in per_run.values():
        within_unique += stats["unique_bytes"]
    cross_unique = overall["unique_bytes"]
    lines.append("Within-run vs cross-run dedup:")
    lines.append(f"  Sum of per-run unique bytes: {within_unique / 1_000_000:.2f} MB")
    lines.append(f"  Global unique bytes:         {cross_unique / 1_000_000:.2f} MB")
    if within_unique > 0:
        cross_dedup = within_unique / cross_unique if cross_unique else 0
        lines.append(f"  Cross-run dedup multiplier:  {cross_dedup:.4f}x")

    lines.append("\nNote: Kind classification (full vs incremental) is not implemented.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure content-addressed dedup potential across checkpoint tars."
    )
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Run directories (recursively scanned for *-after.tar)")
    parser.add_argument("--json", type=Path, default=None, help="Write machine-readable results to JSON file")
    args = parser.parse_args()

    per_run: dict[str, dict] = {}
    all_results: list[dict] = []

    for run_dir in args.run_dirs:
        if not run_dir.is_dir():
            raise SystemExit(f"not a directory: {run_dir}")
        tar_paths = _iter_tar_paths(run_dir)
        results = [_scan_tar(p) for p in tar_paths]
        all_results.extend(results)
        per_run[str(run_dir)] = _merge_stats(results)

    if not all_results:
        raise SystemExit(
            "no checkpoint tars (**/checkpoints/*-after.tar) found under: "
            + ", ".join(str(d) for d in args.run_dirs)
        )

    overall = _merge_stats(all_results)
    print(_report_text(overall, per_run))

    if args.json:
        output = {
            "overall": overall,
            "per_run": per_run,
            "per_tar": all_results,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(output, indent=2, default=str), encoding="utf-8"
        )
        print(f"Machine-readable results written to: {args.json}")


if __name__ == "__main__":
    main()
