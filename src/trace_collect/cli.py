"""CLI entry point for trace collection.

Usage:
    DASHSCOPE_API_KEY=sk-xxx python -m trace_collect.cli \\
        --model qwen-plus-latest \\
        --max-steps 40 \\
        --sample 5
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect SWE-Bench agent traces using an external LLM API.",
    )
    parser.add_argument(
        "--api-base",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="OpenAI-compatible API base URL (default: DashScope).",
    )
    parser.add_argument(
        "--model",
        default="qwen-plus-latest",
        help="Model name to use for inference.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=40,
        help="Maximum agent steps per task.",
    )
    parser.add_argument(
        "--command-timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds per bash command.",
    )
    parser.add_argument(
        "--task-timeout",
        type=float,
        default=1200.0,
        help="Timeout in seconds per task overall.",
    )
    parser.add_argument(
        "--task-source",
        default="data/swebench_verified/tasks.json",
        help="Path to tasks JSON file.",
    )
    parser.add_argument(
        "--repos-root",
        default="data/swebench_repos",
        help="Path to pre-cloned repos directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="traces/swebench",
        help="Output directory for trace files.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only run the first N tasks (for testing).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set DASHSCOPE_API_KEY or OPENAI_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    # Import here to avoid loading heavy deps at parse time
    from trace_collect.collector import collect_traces

    trace_file = asyncio.run(
        collect_traces(
            api_base=args.api_base,
            api_key=api_key,
            model=args.model,
            task_source=args.task_source,
            repos_root=args.repos_root,
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            command_timeout_s=args.command_timeout,
            task_timeout_s=args.task_timeout,
            sample=args.sample,
        )
    )
    print(f"Traces written to: {trace_file}")


if __name__ == "__main__":
    main()
