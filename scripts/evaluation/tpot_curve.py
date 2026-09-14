"""Per-stream decode speed (TPOT) and aggregate output throughput versus concurrency on real agent prompts.

Sends recorded agent-step prompts (replay prefixes: full chat message lists) to an OpenAI-compatible vLLM server with
streaming, at each concurrency level in turn, and records per request: prompt tokens, completion tokens, TTFT and
decode TPOT = (last token time - first token time) / (completion tokens - 1). Aggregates per level: mean/median TPOT,
per-stream tok/s (1/TPOT), aggregate output tok/s over the level's wall time, mean TTFT, and the server's
speculative-decoding counters (drafts, draft tokens, accepted tokens) read from /metrics before and after the level.

Usage: tpot_curve.py --api-base http://127.0.0.1:8300 --model M --prefixes prefixes.jsonl --out curve.json
                     [--concurrency 1 4 8 16 32] [--prompts-per-level 16 32 64 64 64] [--max-tokens 512] [--seed 0]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import statistics
import time
from pathlib import Path

import httpx

SPEC_RE = re.compile(r"^vllm:spec_decode_(num_drafts|num_draft_tokens|num_accepted_tokens)_total\{[^}]*\}\s+([\d.eE+]+)", re.M)


def spec_counters(metrics_text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, value in SPEC_RE.findall(metrics_text):
        out[name] = out.get(name, 0.0) + float(value)
    return out


async def one_request(client: httpx.AsyncClient, api_base: str, model: str, sample: dict, max_tokens: int) -> dict:
    payload = {"model": model, "messages": sample["messages"], "max_tokens": max_tokens, "temperature": 0,
               "stream": True, "stream_options": {"include_usage": True}}
    t_send = time.perf_counter()
    t_first = t_last = None
    n_chunks = 0
    usage = None
    async with client.stream("POST", f"{api_base}/v1/chat/completions", json=payload, timeout=1800) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if choices and (choices[0].get("delta") or {}).get("content") is not None:
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
                n_chunks += 1
    assert usage is not None, "server did not return usage (needs stream_options.include_usage)"
    comp = usage["completion_tokens"]
    tpot = (t_last - t_first) / (comp - 1) if comp > 1 and t_first is not None else None
    return {"sample_id": sample["sample_id"], "prompt_tokens": usage["prompt_tokens"], "completion_tokens": comp,
            "ttft_s": (t_first - t_send) if t_first is not None else None, "tpot_s": tpot, "chunks": n_chunks,
            "e2e_s": time.perf_counter() - t_send}


async def run_level(api_base: str, model: str, samples: list[dict], concurrency: int, max_tokens: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        before = spec_counters((await client.get(f"{api_base}/metrics")).text)

        async def guarded(s):
            async with sem:
                return await one_request(client, api_base, model, s, max_tokens)

        t0 = time.perf_counter()
        results = await asyncio.gather(*(guarded(s) for s in samples))
        wall = time.perf_counter() - t0
        after = spec_counters((await client.get(f"{api_base}/metrics")).text)
    tpots = [r["tpot_s"] for r in results if r["tpot_s"]]
    out_tokens = sum(r["completion_tokens"] for r in results)
    spec = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in set(before) | set(after)}
    drafts = spec.get("num_drafts", 0.0)
    return {"concurrency": concurrency, "requests": len(results), "wall_s": wall, "output_tokens": out_tokens,
            "aggregate_output_tok_s": out_tokens / wall, "tpot_mean_ms": statistics.mean(tpots) * 1e3,
            "tpot_median_ms": statistics.median(tpots) * 1e3, "per_stream_tok_s_median": 1 / statistics.median(tpots),
            "ttft_mean_s": statistics.mean(r["ttft_s"] for r in results if r["ttft_s"] is not None),
            "prompt_tokens_mean": statistics.mean(r["prompt_tokens"] for r in results),
            "completion_tokens_mean": statistics.mean(r["completion_tokens"] for r in results),
            "spec_decode": spec, "accepted_per_draft": (spec.get("num_accepted_tokens", 0.0) / drafts) if drafts else None,
            "per_request": results}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api-base", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--prefixes", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    p.add_argument("--prompts-per-level", type=int, nargs="+", default=[16, 32, 64, 64, 64])
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prewarm", action="store_true", help="prefill every prompt of the level first (max_tokens=1, one at a time) so the level measures decode over cached prefixes: KV read without prefill interference")
    p.add_argument("--max-prompt-chars", type=int, help="keep only samples whose message text totals at most this many characters (short-prompt control)")
    a = p.parse_args()
    assert len(a.prompts_per_level) == len(a.concurrency)
    pool = [json.loads(line) for line in a.prefixes.open()]
    if a.max_prompt_chars:
        pool = [s for s in pool if sum(len(str(m.get("content") or "")) for m in s["messages"]) <= a.max_prompt_chars]
        assert pool, "no samples under --max-prompt-chars"
    rng = random.Random(a.seed)
    rng.shuffle(pool)
    levels = []
    for conc, n in zip(a.concurrency, a.prompts_per_level):
        samples = pool[:n]  # the same leading samples at every level, so levels differ only in concurrency
        if a.prewarm:
            asyncio.run(run_level(a.api_base, a.model, samples, 1, 1))
        level = asyncio.run(run_level(a.api_base, a.model, samples, conc, a.max_tokens))
        levels.append(level)
        print(f"c={conc:<3} n={n:<3} TPOT median {level['tpot_median_ms']:.1f} ms  per-stream {level['per_stream_tok_s_median']:.0f} tok/s"
              f"  aggregate {level['aggregate_output_tok_s']:.0f} tok/s  TTFT {level['ttft_mean_s']:.2f} s"
              f"  prompt {level['prompt_tokens_mean']:.0f} tok  accepted/draft {level['accepted_per_draft']}", flush=True)
        a.out.write_text(json.dumps({"model": a.model, "max_tokens": a.max_tokens, "seed": a.seed, "prewarm": a.prewarm, "max_prompt_chars": a.max_prompt_chars, "levels": levels}, indent=1))


if __name__ == "__main__":
    main()
