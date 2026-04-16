"""Phase E smoke: run TongyiDeepResearchRunner against a real cloud backend.

Goal: prove the adapter works end-to-end (not just via mocks) and produces v5
TraceAction records with non-None ttft_ms/tpot_ms on a real streaming endpoint.

Usage:
    PYTHONPATH=src conda run -n ML python scripts/smoke_tongyi_deepresearch.py \\
        --provider dashscope --model qwen-plus-latest

The script exits non-zero if any of the AC#4 streaming-metric invariants fail.
Prints a summary table for copy-paste into VENDOR_NOTES.md "Phase E smoke log".
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Make src/ importable when the script is run from the repo root.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from agents.tongyi_deepresearch import TongyiDeepResearchRunner  # noqa: E402
from llm_call.providers import PROVIDERS  # noqa: E402
from trace_collect.attempt_pipeline import AttemptContext  # noqa: E402


def _resolve_backend(provider: str, api_key: str | None) -> tuple[str, str]:
    if provider not in PROVIDERS:
        raise SystemExit(f"unknown provider {provider!r}; known: {list(PROVIDERS)}")
    pdef = PROVIDERS[provider]
    api_base = pdef.api_base
    key = api_key or os.environ.get(pdef.env_key)
    if not key:
        raise SystemExit(
            f"missing API key: set {pdef.env_key} or pass --api-key"
        )
    return api_base, key


async def _run_smoke(
    *,
    provider: str,
    model: str,
    api_key: str | None,
    max_iterations: int,
) -> int:
    api_base, api_key_val = _resolve_backend(provider, api_key)

    runner = TongyiDeepResearchRunner(
        model=model,
        api_base=api_base,
        api_key=api_key_val,
        max_iterations=max_iterations,
        benchmark_slug="deep-research-bench",
    )

    task = {
        "instance_id": "smoke-e2e-001",
        "problem_statement": (
            "Briefly: What is the capital of France? "
            "Output <think>your thinking</think><answer>the capital</answer>."
        ),
    }

    with tempfile.TemporaryDirectory(prefix="tongyi_smoke_") as tmp_dir:
        ctx = AttemptContext(
            run_dir=Path(tmp_dir),
            instance_id="smoke-e2e-001",
            attempt=1,
            task=task,
            model=model,
            scaffold="tongyi-deepresearch",
            source_image=None,
            prompt_template="default",
            agent_runtime_mode="host_controller",
            execution_environment="host",
        )
        ctx.attempt_dir.mkdir(parents=True, exist_ok=True)

        print(f"[smoke] invoking TongyiDeepResearchRunner against {provider}/{model} ...")
        result = await runner.run_task(
            task, attempt_ctx=ctx, prompt_template="default",
        )

        trace_path = result.trace_path
        records = [
            json.loads(ln)
            for ln in trace_path.read_text().splitlines()
            if ln.strip()
        ]

    llm_calls = [
        r for r in records
        if r.get("action_type") == "llm_call" and not r.get("data", {}).get("transport_retry")
    ]
    tool_execs = [r for r in records if r.get("action_type") == "tool_exec"]
    retry_spans = [
        r for r in records
        if r.get("action_type") == "llm_call" and r.get("data", {}).get("transport_retry")
    ]

    # Invariants
    failures: list[str] = []
    if not llm_calls:
        failures.append("zero llm_call TraceActions captured")
    for call in llm_calls:
        data = call.get("data", {})
        if data.get("ttft_ms") is None:
            failures.append(f"llm_call {call.get('action_id')} has ttft_ms=None")
        if data.get("tpot_ms") is None and data.get("completion_tokens", 0) > 1:
            failures.append(f"llm_call {call.get('action_id')} has tpot_ms=None")

    # Report
    print()
    print("=" * 60)
    print(f"Phase E smoke log  (UTC {datetime.now(timezone.utc).isoformat(timespec='seconds')})")
    print("=" * 60)
    print(f"provider:                 {provider}")
    print(f"model:                    {model}")
    print(f"api_base:                 {api_base}")
    print(f"exit_status:              {result.exit_status}")
    print(f"vendor_termination:       {result.runtime_proof.get('vendor_termination')}")
    print(f"n_turns (summary):        {result.summary.get('n_turns')}")
    print(f"total_llm_ms:             {result.summary.get('total_llm_ms'):.1f}")
    print(f"total_tool_ms:            {result.summary.get('total_tool_ms'):.1f}")
    print(f"total_tokens:             {result.summary.get('total_tokens')}")
    print(f"transport_retry_count:    {result.summary.get('transport_retry_count')}")
    print(f"llm_call actions:         {len(llm_calls)}")
    print(f"tool_exec actions:        {len(tool_execs)}")
    print(f"transport_retry spans:    {len(retry_spans)}")
    print(f"final_answer (first 120): {result.summary.get('final_answer', '')[:120]!r}")
    print("=" * 60)

    if failures:
        print()
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print()
    print("OK: Phase E smoke passed all AC#4 invariants.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="dashscope")
    ap.add_argument("--model", default="qwen-plus-latest")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--max-iterations", type=int, default=8)
    args = ap.parse_args()

    return asyncio.run(_run_smoke(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        max_iterations=args.max_iterations,
    ))


if __name__ == "__main__":
    raise SystemExit(main())
