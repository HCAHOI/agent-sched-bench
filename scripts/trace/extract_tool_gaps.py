#!/usr/bin/env python3
"""Extract observed tool-gap windows from canonical trace JSONL files.

Usage:
  uv run python scripts/trace/extract_tool_gaps.py traces/run_or_trace.jsonl \
    --output tool_gaps.jsonl

The output is a label/evaluation dataset. Do not use observed current-iteration
fields such as available_gap_ms as online prediction features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trace_collect.tool_gap_extractor import (
    discover_trace_files,
    extract_many_tool_gap_windows,
    write_tool_gap_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract observed tool-gap windows from trace JSONL files."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Trace files or directories recursively containing trace.jsonl files",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL path")
    parser.add_argument(
        "--agent-filter",
        default=None,
        help="Optional substring filter for trace agent_id values",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    trace_paths = discover_trace_files(args.paths)
    windows = extract_many_tool_gap_windows(
        trace_paths,
        agent_filter=args.agent_filter,
    )
    count = write_tool_gap_jsonl(windows, args.output)
    print(f"Wrote {count} tool-gap windows from {len(trace_paths)} trace(s) -> {args.output}")


if __name__ == "__main__":
    main()
