#!/usr/bin/env python3
"""Run nanobot eval directly without going through nanobot's CLI.

Imports nanobot's SWEBenchRunner and calls it programmatically.
Avoids conda/pip installation issues by using nanobot source directly.

Usage:
    PYTHONPATH=/Users/chiyuh/Workspace/nanobot:src \
    python scripts/run_nanobot_eval.py \
        --instances /tmp/pilot_instances.jsonl \
        --output /tmp/nanobot-pilot/results.jsonl \
        --workspace /tmp/nanobot-pilot \
        --model openrouter/qwen/qwen3.6-plus:free \
        --max-steps 80
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run nanobot eval directly")
    parser.add_argument("--instances", required=True, help="Path to instances JSONL")
    parser.add_argument(
        "--output",
        default="/tmp/nanobot-eval/results.jsonl",
        help="Output results JSONL",
    )
    parser.add_argument(
        "--workspace", default="/tmp/nanobot-eval", help="Workspace base directory"
    )
    parser.add_argument(
        "--model", default="openrouter/qwen/qwen3.6-plus:free", help="Model name"
    )
    parser.add_argument(
        "--max-steps", type=int, default=80, help="Max agent loop iterations"
    )
    parser.add_argument("--repos-root", default=None, help="Local git repos mirror")
    parser.add_argument("--parallel", type=int, default=1, help="Max concurrent tasks")
    parser.add_argument("--config", default=None, help="Path to nanobot config file")
    args = parser.parse_args()

    # Import nanobot components
    from nanobot.eval.runner import SWEBenchRunner
    from nanobot.eval.types import EvalTask

    # Create provider directly — bypass nanobot's routing to avoid
    # the openrouter/ prefix being sent to the API
    import os
    from nanobot.providers.openai_compat_provider import OpenAICompatProvider

    provider = OpenAICompatProvider(
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        api_base="https://openrouter.ai/api/v1",
        default_model=args.model,
    )

    # Load tasks from JSONL
    tasks: list[EvalTask] = []
    ws_base = Path(args.workspace).expanduser().resolve()
    with open(args.instances, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            tasks.append(EvalTask.from_swebench_instance(data, ws_base))

    print(f"Loaded {len(tasks)} tasks")
    for t in tasks:
        print(f"  {t.instance_id}  repo={t.repo}")

    # Create output directory
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create runner — pass model_for_routing so nanobot's agent loop uses
    # the correct model string for API calls
    repos_root = (
        Path(args.repos_root).expanduser().resolve() if args.repos_root else None
    )
    runner = SWEBenchRunner(
        provider=provider,
        workspace_base=ws_base,
        max_iterations=args.max_steps,
        context_window_tokens=256_000,
        model=args.model,
        repos_root=repos_root,
    )

    # Run
    results = asyncio.run(runner.run_batch(tasks, max_concurrent=args.parallel))

    # Write results
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    # Write predictions
    preds = {}
    model_name = args.model.split("/")[-1] if "/" in args.model else args.model
    for result in results:
        if result.model_patch:
            preds[result.instance_id] = {
                "instance_id": result.instance_id,
                "model_name_or_path": model_name,
                "model_patch": result.model_patch,
            }
    preds_path = output_path.with_name("preds.json")
    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\nResults: {output_path}")
    print(f"Predictions: {preds_path} ({len(preds)} patches)")
    for r in results:
        print(f"  {r.instance_id}: patch={r.patch_generated} stop={r.stop_reason}")
        if r.trace_file:
            print(f"    trace: {r.trace_file}")


if __name__ == "__main__":
    main()
