"""Paired two-turn PD/local profiling at declared H/U/O lengths.

Uses the existing audited chat proxy, real generation, and public PPD prompt and
Poisson-arrival construction. The plan's numeric table is the input manifest.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any

import aiohttp


PRELIMINARY = ("P02", "P43", "P45", "P47", "P54", "P55", "P56")


def read_points(plan: Path) -> dict[str, tuple[int, int, int]]:
    rows = re.findall(r"^\| (P\d+) \| (\d+) \| (\d+) \| (\d+) \|", plan.read_text(), re.M)
    points = {name: (int(h), int(u), int(o)) for name, h, u, o in rows}
    assert points and len(points) == len(rows), "Missing or duplicate length points"
    assert all(h > 32 and u > 0 and o > 1 and h + u + o <= 131072
               for h, u, o in points.values())
    return points


def schedule(points: dict[str, tuple[int, int, int]], stage: str) -> list[tuple[str, float, int, str]]:
    preliminary = [(p, 0.5, 42, path) for p in PRELIMINARY for path in ("pd", "local")]
    if stage == "preliminary":
        return preliminary
    full = [(p, rate, seed, path) for p in points for rate in (0.5, 1.0)
            for seed in (42, 43) for path in (("pd", "local") if seed == 42 else ("local", "pd"))]
    return preliminary + [g for g in full if g not in preliminary]


def arrival_offsets(rate: float, seed: int, count: int = 8) -> list[float]:
    """Public expovariate inter-arrivals, with a fixed conversation count."""
    rng = random.Random(seed)
    arrivals = []
    total = 0.0
    for _ in range(count):
        total += rng.expovariate(rate)
        arrivals.append(total)
    return arrivals


def prompt_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    return tokenizer.apply_chat_template(messages, tokenize=True,
                                         add_generation_prompt=True, return_dict=False)


def fit_user(tokenizer: Any, history: list[dict[str, str]], target: int,
             content: str) -> tuple[list[dict[str, str]], list[int]]:
    """Trim diverse public text, then correct length with a few single-token words."""
    overhead = len(prompt_tokens(tokenizer, history + [dict(role="user", content="")]))
    budget = target - overhead
    assert budget > 0, (target, overhead)
    ids = tokenizer.encode(content, add_special_tokens=False)
    content = tokenizer.decode(ids[:budget], skip_special_tokens=False)
    for _ in range(12):
        messages = history + [dict(role="user", content=content)]
        tokens = prompt_tokens(tokenizer, messages)
        delta = target - len(tokens)
        if delta == 0:
            return messages, tokens
        if delta > 0:
            content += " x" * delta
        else:
            ids = tokenizer.encode(content, add_special_tokens=False)
            assert len(ids) + delta > 0
            content = tokenizer.decode(ids[:len(ids) + delta], skip_special_tokens=False)
    raise AssertionError(f"Cannot construct {target} prompt tokens; got {len(tokens)}")


def validate_lengths(target: tuple[int, int, int], actual: tuple[int, int, int]) -> None:
    h, u, o = target
    ah, au, ao = actual
    # Generated text can retokenize at its final boundary. Freeze tolerance
    # before sampling; never silently relabel a point or cross a history bin.
    bounds = (4096, 16384, 32768, 65536)
    assert abs(ah - h) <= 8 and sum(ah > b for b in bounds) == sum(h > b for b in bounds), (target, actual)
    assert (au, ao) == (u, o), (target, actual)


def prepare_return(tokenizer: Any, messages: list[dict[str, str]], tokens: list[int],
                   result: dict[str, Any], content: str,
                   target: tuple[int, int, int]) -> tuple[list[dict[str, str]], list[int], int]:
    """Construct and validate T2 off the SSE reader's event loop."""
    previous = tokens + result["output_token_ids"]
    history = messages + [dict(role="assistant", content=result["output_text"])]
    probe = prompt_tokens(tokenizer, history + [dict(role="user", content="")])
    actual_h = next((j for j, (a, b) in enumerate(zip(previous, probe)) if a != b),
                    min(len(previous), len(probe)))
    messages, tokens = fit_user(tokenizer, history, actual_h + target[1], content)
    validate_lengths(target, (actual_h, len(tokens) - actual_h, target[2]))
    return messages, tokens, actual_h


async def stream_turn(client: aiohttp.ClientSession, proxy: str, payload: dict[str, Any],
                      timeout_s: float, row: dict[str, Any], raw: Any) -> dict[str, Any]:
    began = time.perf_counter()
    row.update(started_unix_s=time.time(), output_token_ids=[], token_chunks=[],
               output_text="", usage=None, request_id=None, success=False)
    done = False
    try:
        async with client.post(proxy + "/v1/chat/completions", json=payload,
                               headers={"Connection": "close"},
                               timeout=aiohttp.ClientTimeout(total=timeout_s)) as response:
            response.raise_for_status()
            async for line in response.content:
                if not line.startswith(b"data:"):
                    continue
                if line[5:].strip() == b"[DONE]":
                    done = True
                    continue
                data = json.loads(line[5:])
                assert "error" not in data, data
                row["request_id"] = data.get("id", row["request_id"])
                if data.get("usage"):
                    row["usage"] = data["usage"]
                for choice in data.get("choices", []):
                    ids = choice.get("token_ids") or choice.get("delta", {}).get("token_ids") or []
                    if ids:
                        row["token_chunks"].append([time.perf_counter() - began, len(ids)])
                        row["output_token_ids"].extend(ids)
                    row["output_text"] += choice.get("delta", {}).get("content") or ""
        assert done and row["usage"] and row["request_id"] and row["token_chunks"], row["usage"]
        assert len(row["output_token_ids"]) == row["usage"]["completion_tokens"] == payload["max_tokens"]
        assert row["usage"]["prompt_tokens"] == row["prompt_tokens"]
        first, last = row["token_chunks"][0][0], row["token_chunks"][-1][0]
        row.update(success=True, ttft_s=first, decode_tpot_s=(last - first) / (payload["max_tokens"] - 1))
        return row
    except BaseException as exc:
        row["error"] = repr(exc)
        raise
    finally:
        row.update(e2e_s=time.perf_counter() - began, finished_unix_s=time.time(), done=done)
        raw.write(json.dumps(row) + "\n")


async def clean_group(client: aiohttp.ClientSession, args: Any, key: str) -> float:
    began = time.perf_counter()
    async with client.get(args.proxy + "/health") as response:
        response.raise_for_status()
        state = await response.json()
        assert state["outstanding"] == state["sessions"] == 0, state
    # Existing release-audit procedure: wake idle P so pending transfer-release
    # notifications are consumed. This request is maintenance, never a sample.
    async with client.post(args.backends[0] + "/v1/chat/completions", json=dict(
        model=args.model, messages=[dict(role="user", content="Release audit probe")],
        max_tokens=1, ignore_eos=True, stream=False),
        headers={"x-request-id": "length-profile-maintenance-" + key}) as response:
        response.raise_for_status()
        assert (await response.json())["usage"]["completion_tokens"] == 1
    for backend in args.backends:
        for _ in range(30):
            async with client.post(backend + "/reset_prefix_cache") as response:
                response.raise_for_status()
                if (await response.json())["success"]:
                    break
            await asyncio.sleep(0.2)
        else:
            raise AssertionError(f"Prefix cache still held on {backend}")
    async with client.post(args.proxy + "/profiling/reset") as response:
        response.raise_for_status()
        assert (await response.json())["reset"]
    return time.perf_counter() - began


def verify_group(run: Path, results: list[dict[str, Any]], num_layers: int) -> None:
    """Reconcile client lengths with routing, both engine legs, and real KV transfer."""
    from scripts.evaluation.check_ppd_request_metrics import check
    from scripts.evaluation.check_two_instance_request_metrics import rows
    check(run, num_layers, final=True)
    decisions = {(r["job_id"], r["turn"]): r for r in rows(run / "routing.jsonl", True)
                 if r["event"] == "decision"}
    for result in results:
        decision = decisions[result["job_id"], result["turn"]]
        assert (decision["context_tokens"], decision["input_tokens"]) == (result["actual_h"], result["actual_u"])
        assert decision["engine_request_id"] == result["request_id"]
        assert "cached_tokens" in result["usage"]["prompt_tokens_details"]


async def run_group(args: Any, tokenizer: Any, point: str, target: tuple[int, int, int],
                    rate: float, seed: int, path: str, root: Path) -> dict[str, Any]:
    from scripts.benchmark.comprehensive_benchmark import generate_prompt

    key = f"{point}-rate{rate:g}-seed{seed}-{path}"
    group_started = time.perf_counter()
    h, u, o = target
    prepared = []
    for i in range(8):
        prefix = hashlib.sha256(f"{point}:{seed}:{i}".encode()).hexdigest()[:16] + ": "
        t1 = generate_prompt(h, prefix=prefix, seed=f"{point}:{seed}:{i}:1")
        messages, tokens = fit_user(tokenizer, [], h - 32, t1)
        t2 = generate_prompt(u, prefix=prefix, seed=f"{point}:{seed}:{i}:2")
        prepared.append((messages, tokens, t2))
    # Check independent conversations do not share their first cache block.
    assert len({tuple(p[1][:16]) for p in prepared}) == 8
    prepare_s = time.perf_counter() - group_started
    offsets = arrival_offsets(rate, seed)
    group = dict(key=key, point=point, target_huo=target, conversation_rate=rate,
                 seed=seed, path=path, conversation_count=8, planned_arrivals_s=offsets,
                 cpu_prepare_s=prepare_s, success=False, started_unix_s=time.time())
    raw = (root / (key + "-turns.jsonl")).open("x", buffering=1)
    results = []
    # Releases can be seconds apart: never race the server's idle keep-alive close.
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True),
                                    timeout=aiohttp.ClientTimeout(total=30)) as control:
        group["initial_cleanup_s"] = await clean_group(control, args, key + "-before")
        start = time.perf_counter()
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=50)) as client:
            async def conversation(i: int) -> None:
                job = key + f"-conv{i}"
                await asyncio.sleep(max(0, start + offsets[i] - time.perf_counter()))
                messages, tokens, content = prepared[i]
                try:
                    for turn, output in ((1, 32), (2, o)):
                        construction_started = time.perf_counter()
                        actual_h = 0
                        if turn == 2:
                            messages, tokens, actual_h = await asyncio.to_thread(
                                prepare_return, tokenizer, messages, tokens, result, content, target)
                        row = dict(group=key, job_id=job, turn=turn, actual_h=actual_h,
                            actual_u=len(tokens) - actual_h, target_output=output, prompt_tokens=len(tokens),
                            input_content=messages[-1]["content"],
                            input_construction_s=time.perf_counter() - construction_started,
                            scheduled_t1_offset_s=offsets[i], submitted_offset_s=time.perf_counter() - start)
                        payload = dict(model=args.model, program_id=job, messages=messages,
                            profiling_path="pd" if turn == 1 else path, stream=True,
                            stream_options={"include_usage": True}, return_token_ids=True,
                            max_tokens=output, ignore_eos=True, skip_special_tokens=False, seed=0, temperature=0)
                        result = await stream_turn(client, args.proxy, payload, args.timeout_s, row, raw)
                        results.append(result)
                finally:
                    async with control.post(args.proxy + "/programs/release", json={"program_id": job}) as response:
                        response.raise_for_status()

            tasks = [asyncio.create_task(conversation(i)) for i in range(8)]
            try:
                await asyncio.gather(*tasks)
                group["workload_s"] = time.perf_counter() - start
                group["final_cleanup_s"] = await clean_group(control, args, key + "-after")
                second = [r for r in results if r["turn"] == 2]
                assert len(second) == 8 and len(results) == 16
                verify_group(root.parent, results, args.num_layers)
                group.update(success=True, t1_count=8, t2_count=8,
                    t1_mean_e2e_s=statistics.mean(r["e2e_s"] for r in results if r["turn"] == 1),
                    t2_mean_e2e_s=statistics.mean(r["e2e_s"] for r in second),
                    t2_mean_ttft_s=statistics.mean(r["ttft_s"] for r in second),
                    t2_mean_decode_tpot_s=statistics.mean(r["decode_tpot_s"] for r in second),
                    achieved_llm_requests_per_s=16 / group["workload_s"])
            except BaseException as exc:
                group["error"] = repr(exc)
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            finally:
                raw.close()
                group.update(total_s=time.perf_counter() - group_started, finished_unix_s=time.time())
                (root / (key + ".json")).write_text(json.dumps(group, indent=2))
    return group


def resume_prefix(root: Path, model: str, plan: Path, groups: list) -> list[dict[str, Any]]:
    """Reuse only the contiguous prefix of previously validated groups."""
    config = json.loads((root / "config.json").read_text())
    assert config["model"] == model and config["stage"] == "full"
    assert not config.get("resume_from"), "Resume source must be the original run"
    assert (root / "plan.md").read_text() == plan.read_text(), "Profiling plan changed"
    assert config["groups"] == [list(g) for g in groups], "Group schedule changed"
    completed = []
    gap = False
    for point, rate, seed, path in groups:
        key = f"{point}-rate{rate:g}-seed{seed}-{path}"
        file = root / (key + ".json")
        group = json.loads(file.read_text()) if file.exists() else {}
        if not group.get("success"):
            gap = True
            continue
        assert not gap, "Successful group after a gap; inspect before resuming"
        assert group["key"] == key and group["t1_count"] == group["t2_count"] == 8
        rows = [json.loads(line) for line in (root / (key + "-turns.jsonl")).read_text().splitlines()]
        assert len(rows) == 16 and all(row["success"] for row in rows)
        assert {(row["job_id"], row["turn"]) for row in rows} == {
            (key + f"-conv{i}", turn) for i in range(8) for turn in (1, 2)}
        completed.append(group)
    continuation = json.loads((root / "continue-full.json").read_text())
    assert continuation["continue"] is True
    return completed


async def profile(args: Any) -> None:
    from transformers import AutoConfig, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    args.num_layers = AutoConfig.from_pretrained(args.model, local_files_only=True).num_hidden_layers
    points = read_points(args.plan)
    groups = schedule(points, args.stage)
    completed = resume_prefix(args.resume_from, args.model, args.plan, groups) if args.resume_from else []
    if args.resume_from:
        old = json.loads((args.resume_from / "continue-full.json").read_text())
        assert args.stage == "full" and args.timeout_s == old["request_timeout_s"]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "plan.md").write_text(args.plan.read_text())
    (args.output / "config.json").write_text(json.dumps(dict(vars(args), points=points, groups=groups,
        history_tolerance_tokens=8, input_tolerance_tokens=0, skip_special_tokens=False,
        setup_output_tokens=32, conversations_per_group=8), default=str, indent=2))
    if args.resume_from:
        (args.output / "resumed-groups.json").write_text(json.dumps(completed, indent=2))
        print(f"RESUME: {len(completed)}/{len(groups)} groups validated in {args.resume_from}", flush=True)
    for index, (point, rate, seed, path) in enumerate(groups, 1):
        if index <= len(completed):
            continue
        print(f"START {index}/{len(groups)} {point} H/U/O={points[point]} rate={rate} seed={seed} {path}", flush=True)
        group = await run_group(args, tokenizer, point, points[point], rate, seed, path, args.output)
        completed.append(group)
        print(f"DONE {index}/{len(groups)} {group['key']} {group['total_s']:.1f}s; "
              f"T2 TTFT={group['t2_mean_ttft_s']:.3f}s TPOT={group['t2_mean_decode_tpot_s']*1000:.2f}ms", flush=True)
        if index == 14:
            (args.output / "preliminary-complete.json").write_text(json.dumps(completed, indent=2))
            if args.stage == "full":
                print("Preliminary complete; waiting for measured duration review in continue-full.json.", flush=True)
                while not (args.output / "continue-full.json").exists():
                    await asyncio.sleep(5)
                continuation = json.loads((args.output / "continue-full.json").read_text())
                assert continuation["continue"] is True and 0 < continuation["request_timeout_s"] <= args.timeout_s
                args.timeout_s = continuation["request_timeout_s"]
    (args.output / "complete.json").write_text(json.dumps(dict(groups=len(completed), success=True), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--proxy", default="http://127.0.0.1:9000")
    parser.add_argument("--backends", nargs=2, default=["http://127.0.0.1:8000", "http://127.0.0.1:8001"])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["preliminary", "full"], default="preliminary")
    parser.add_argument("--timeout-s", type=float, default=1800)
    parser.add_argument("--resume-from", type=Path)
    asyncio.run(profile(parser.parse_args()))
