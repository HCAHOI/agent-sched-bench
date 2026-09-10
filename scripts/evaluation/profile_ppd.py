"""Measure PPD's public calibration grid using our audited chat replay proxy.

Reuses the public grid, arrival generator, aggregation, and result schema.
Both arms use the same seeded arrivals and new user prompts; each carries its
own real first-turn output into the second turn. All raw turn results are saved;
these synthetic hardware calibration requests are not mixed56 results.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import aiohttp


async def profile(args: Any) -> None:
    from scripts.benchmark import comprehensive_benchmark as public

    original_contexts = public.T1_CONFIGS
    context_tokens = getattr(args, "context_tokens", None)
    if context_tokens is not None:
        assert 4096 < context_tokens < 120000
        # Extend only historical context; retain the public T1 output length,
        # T2 grid, arrival process, timeouts, and second-turn scoring.
        public.T1_CONFIGS = {"huge": {"input": context_tokens - 1024, "output": 1024}}
    config = "1P_1D" if args.mode == "pd" else "1P_1pD"
    output = args.output / config
    output.mkdir(parents=True, exist_ok=False)
    fatal: list[str] = []
    point_key = "warmup"
    raw = (output / "turns.jsonl").open("x", buffering=1)
    (output / "profile-config.json").write_text(json.dumps(dict(
        context_tokens=context_tokens, t1_configs=public.T1_CONFIGS,
        t2_configs=public.T2_CONFIGS, qps_points=public.QPS_POINTS,
        arrival_window_s=public.DURATION_PER_POINT_SEC,
        request_timeout_s=public.REQUEST_TIMEOUT_SEC, seed=args.seed,
        start_point=args.start_point, tool_delay_s=0,
    ), indent=2))
    original_conversation = public.run_conversation
    original_turn = public.run_single_turn

    async def release(job: str) -> None:
        # Cleanup must not queue behind inference in the public 50-connection pool.
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as control:
            async with control.post(args.proxy + "/programs/release",
                                    json={"program_id": job}) as response:
                response.raise_for_status()

    async def turn(client: aiohttp.ClientSession, conv_id: str, turn_num: int,
                   input_tokens: int, output_tokens: int, history: Any):
        index = conv_id.rsplit("_", 2)[-2]
        content = public.generate_prompt(input_tokens, prefix=f"calibration-{point_key}-{index}: ",
                                         seed=f"{args.seed}:{point_key}:{index}:{turn_num}")
        messages = list(history) if history else []
        messages.append(dict(role="user", content=content))
        began = time.perf_counter()
        first = last = None
        text = ""
        token_ids: list[int] = []
        usage = None
        request_id = None
        done = False
        try:
            async with client.post(args.proxy + "/v1/chat/completions", json=dict(
                model=args.model, program_id=conv_id, messages=messages, stream=True,
                stream_options={"include_usage": True}, return_token_ids=True,
                max_tokens=output_tokens, ignore_eos=True, seed=0, temperature=0),
                headers={"Connection": "close"},
                timeout=aiohttp.ClientTimeout(total=public.REQUEST_TIMEOUT_SEC)) as response:
                response.raise_for_status()
                async for line in response.content:
                    if not line.startswith(b"data:"):
                        continue
                    if line[5:].strip() == b"[DONE]":
                        done = True
                        continue
                    data = json.loads(line[5:])
                    if "error" in data:
                        raise RuntimeError(data["error"])
                    request_id = data.get("id", request_id)
                    if data.get("usage"):
                        usage = data["usage"]
                    for choice in data.get("choices", []):
                        ids = choice.get("token_ids") or choice.get("delta", {}).get("token_ids") or []
                        if ids:
                            last = time.perf_counter()
                            if first is None:
                                first = last
                            token_ids.extend(ids)
                        text += choice.get("delta", {}).get("content") or ""
            assert done and usage and first is not None and last is not None and request_id
            assert len(token_ids) == usage["completion_tokens"] == output_tokens
            # Exclude trailing usage/DONE transport time from TPOT.
            result = public.TurnResult(turn=turn_num, input_tokens=input_tokens, output_tokens=output_tokens,
                ttft_ms=(first - began) * 1000, tpot_ms=(last - first) * 1000 / (output_tokens - 1),
                e2e_ms=(time.perf_counter() - began) * 1000, success=True, completion_tokens=len(token_ids))
            messages.append(dict(role="assistant", content=text))
        except TimeoutError:
            result = public.TurnResult(turn=turn_num, input_tokens=input_tokens, output_tokens=output_tokens,
                ttft_ms=0, tpot_ms=0, e2e_ms=(time.perf_counter() - began) * 1000,
                success=False, error="Timeout")
        except Exception as exc:
            fatal.append(repr(exc))
            raise
        raw.write(json.dumps(dict(config=config, point=point_key, job_id=conv_id, request_id=request_id,
                                  usage=usage, output_token_ids=token_ids, **asdict(result))) + "\n")
        return result, messages

    async def conversation(client: aiohttp.ClientSession, conv_id: str, *rest: Any):
        try:
            return await original_conversation(client, conv_id, *rest)
        finally:
            try:
                await release(conv_id)
            except Exception as exc:
                fatal.append(repr(exc))
                raise

    public.run_single_turn = turn
    public.run_conversation = conversation
    try:
        async with aiohttp.ClientSession() as client:
            for i in range(public.WARMUP_REQUESTS):
                job = f"warmup_{i}_0"
                result, _ = await turn(client, job, 1, 128, 10, "")
                await release(job)
                assert result.success, result
        workloads = public.build_workload_list()
        assert 1 <= args.start_point <= len(workloads) * len(public.QPS_POINTS)
        print(f"Calibrating {len(workloads) * len(public.QPS_POINTS)} points for {config}; "
              f"{public.DURATION_PER_POINT_SEC}s arrival window per point, then full drain.", flush=True)
        for index, (workload, qps) in enumerate((w, q) for w in workloads for q in public.QPS_POINTS):
            if index + 1 < args.start_point:
                continue
            context, kind = workload.split("_", 1)
            point_key = f"{workload}-{qps}"
            random.seed(args.seed + index)
            result = await public.run_benchmark_point_inner(config, workload, qps,
                        public.T1_CONFIGS[context], public.T2_CONFIGS[kind])
            assert not fatal, fatal
            assert result.total_requests > 0, result
            public.save_result(result, output)
            print(f"{index + 1}/{len(workloads) * len(public.QPS_POINTS)} {point_key}: "
                  f"{result.duration_sec:.1f}s, success={result.success_rate:.1f}%", flush=True)
    finally:
        public.T1_CONFIGS = original_contexts
        public.run_single_turn = original_turn
        public.run_conversation = original_conversation
        raw.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["pd", "static-x1"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--proxy", default="http://127.0.0.1:9000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-tokens", type=int,
                        help="Additional huge-context grid; approximate T1 input plus output tokens")
    parser.add_argument("--start-point", type=int, default=1,
                        help="One-based grid index; retain original indices and seeds when continuing a failed run")
    asyncio.run(profile(parser.parse_args()))
