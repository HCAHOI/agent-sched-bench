"""CLI entry point for trace collection, replay, and simulation.

Usage (collect):
    DASHSCOPE_API_KEY=sk-xxx python -m trace_collect.cli \\
        --model qwen3.5-plus \\
        --max-steps 50 \\
        --sample 5

Usage (replay):
    DASHSCOPE_API_KEY=sk-xxx python -m trace_collect.cli replay \\
        --trace traces/swebench/run.jsonl \\
        --agent-id django__django-11734 \\
        --from-step 45 \\
        --max-steps 80

Usage (simulate):
    python -m trace_collect.cli simulate \\
        --source-trace traces/swebench/qwen-plus/.../task.jsonl \\
        --api-base http://localhost:8000/v1 \\
        --model Qwen/Qwen2.5-Coder-7B-Instruct
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path


def parse_collect_args(argv: list[str] | None = None) -> argparse.Namespace:
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
        default="qwen3.5-plus",
        help="Model name to use for inference.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=60,
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
        default="traces",
        help="Root output directory (traces/{benchmark}/{model}/{ts}/).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only run the first N tasks (for testing).",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=256_000,
        help="Sliding window token budget for context management.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Resume an interrupted run by providing its existing run ID.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args(argv)


def parse_replay_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a trace from a specific step with new LLM parameters.",
    )
    parser.add_argument(
        "--trace",
        required=True,
        help="Path to the original JSONL trace file.",
    )
    parser.add_argument(
        "--agent-id",
        required=True,
        help="Agent/instance ID to replay.",
    )
    parser.add_argument(
        "--from-step",
        type=int,
        required=True,
        help="Step index to resume from (0-indexed).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=60,
        help="New maximum steps (total, not additional).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model for new steps (default: same as original trace).",
    )
    parser.add_argument(
        "--api-base",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="OpenAI-compatible API base URL.",
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
        help="Output directory for the replay trace.",
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
        help="Timeout for the entire resumed run.",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=256_000,
        help="Sliding window token budget for context management.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args(argv)


def parse_simulate_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate a trace with local model timing (TTFT/TPOT).",
    )
    parser.add_argument(
        "--source-trace",
        required=True,
        help="Path to the source API trace JSONL file.",
    )
    parser.add_argument(
        "--api-base",
        default="http://localhost:8000/v1",
        help="Local model OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Local model name (e.g. Qwen/Qwen2.5-Coder-7B-Instruct).",
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
        default="traces/simulate",
        help="Output directory for the simulate trace.",
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
        help="Timeout for the entire simulation.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args(argv)


def main() -> None:
    # Keyword detection: subcommand as first arg routes to the right parser.
    if len(sys.argv) > 1 and sys.argv[1] == "replay":
        args = parse_replay_args(sys.argv[2:])
        _run_replay(args)
    elif len(sys.argv) > 1 and sys.argv[1] == "simulate":
        args = parse_simulate_args(sys.argv[2:])
        _run_simulate(args)
    else:
        args = parse_collect_args()
        _run_collect(args)


def _run_collect(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set DASHSCOPE_API_KEY or OPENAI_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    from trace_collect.collector import collect_traces

    run_dir = asyncio.run(
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
            run_id=args.run_id,
            max_context_tokens=args.max_context_tokens,
        )
    )
    print(f"Traces written to: {run_dir}/")


def _run_replay(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set DASHSCOPE_API_KEY or OPENAI_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    from trace_collect.replayer import replay

    trace_file = asyncio.run(
        replay(
            trace_path=Path(args.trace),
            agent_id=args.agent_id,
            from_step=args.from_step,
            task_source=Path(args.task_source),
            repos_root=Path(args.repos_root),
            output_dir=Path(args.output_dir),
            max_steps=args.max_steps,
            api_base=args.api_base,
            api_key=api_key,
            model=args.model,
            command_timeout_s=args.command_timeout,
            task_timeout_s=args.task_timeout,
            max_context_tokens=args.max_context_tokens,
        )
    )
    print(f"Replay trace written to: {trace_file}")


def _run_simulate(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("VLLM_API_KEY", "EMPTY")

    from trace_collect.simulator import simulate

    trace_file = asyncio.run(
        simulate(
            source_trace=Path(args.source_trace),
            task_source=Path(args.task_source),
            repos_root=Path(args.repos_root),
            output_dir=Path(args.output_dir),
            api_base=args.api_base,
            api_key=api_key,
            model=args.model,
            command_timeout_s=args.command_timeout,
            task_timeout_s=args.task_timeout,
        )
    )
    print(f"Simulate trace written to: {trace_file}")


if __name__ == "__main__":
    main()
