#!/usr/bin/env python3
"""Analyze and plot a frozen utility-clock threshold sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_collect.tool_latency_threshold_sweep import (
    analyze_threshold_sweep_manifest,
    plot_threshold_sweep,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze utility-clock threshold sweep decisions."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--figures-dir", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = analyze_threshold_sweep_manifest(args.manifest)
    figure_paths = plot_threshold_sweep(result, args.figures_dir)
    result["figure_paths"] = [str(path.resolve()) for path in figure_paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Analyzed {len(result['corpora'])} corpora and wrote "
        f"{len(figure_paths)} figures -> {args.output}"
    )


if __name__ == "__main__":
    main()
