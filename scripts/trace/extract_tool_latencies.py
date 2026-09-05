#!/usr/bin/env python3
"""Extract observed per-tool latency labels from canonical trace JSONL files.

The output is an offline evaluation dataset. Do not feed ``latency_ms`` back as
an online prediction feature.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trace_collect.tool_latency_dataset import (
    discover_trace_files,
    extract_many_tool_latency_samples,
    write_tool_latency_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract observed tool latencies from trace JSONL files."
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
    samples = extract_many_tool_latency_samples(
        trace_paths,
        agent_filter=args.agent_filter,
    )
    count = write_tool_latency_jsonl(samples, args.output)
    print(f"Wrote {count} tool latency samples from {len(trace_paths)} trace(s) -> {args.output}")


if __name__ == "__main__":
    main()
