"""Adapt DualMap's public scheduler to closed-loop streaming agent replay.

The public shadow cache remains a routing estimate. LMCache must provide the
actual per-instance CPU KV storage; only engine metrics establish realized hits.
Future replay output lengths are never supplied to the scheduler: its running
block-count tie-break uses the mean of outputs completed so far (initially 128).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from starlette.responses import StreamingResponse
from transformers import AutoTokenizer
from dualmap.scheduler.utils.double_hash_global_scheduler_utils import GlobalRequestQueue

from scripts.baselines.thunderagent_official_launcher import RequestAudit, _request_record


class AgentProgressQueue(GlobalRequestQueue):
    """Trade estimated recomputation savings against observed task waiting.

    Keep the public cache-token heap and accounting intact; select by seconds
    at dispatch and use the same order in migration's queue-delay estimate.
    """

    def __init__(self, num_replicas: int, prefill_tpot: float):
        super().__init__(num_replicas)
        self.prefill_tpot = prefill_tpot

    def priority(self, item: tuple) -> tuple[float, float, int]:
        negative_cached_tokens, arrived, request = item
        return (arrived - request.agent_previous_wait_s
                + negative_cached_tokens * self.prefill_tpot, arrived, request._id)

    def peek(self, replica_id: int) -> Any:
        queue = self.queues[replica_id]
        return min(queue, key=self.priority)[2] if queue else None

    def pop(self, replica_id: int) -> Any:
        request = self.peek(replica_id)
        if request is not None:
            self.del_req(replica_id, request)
        return request

    def pop_schedulable(self, replica_id: int, cur_replica_budget: float,
                        max_pop_num: int | None = None) -> list[Any]:
        selected = []
        # ponytail: linear selection per dispatch; queues are bounded by active
        # agents here. Use a separate priority heap if large queues matter.
        while self.queues[replica_id] and (max_pop_num is None or len(selected) < max_pop_num):
            item = min(self.queues[replica_id], key=self.priority)
            request = item[2]
            cur_replica_budget -= self._get_request_actual_prefill_tokens(request, item[0])
            self.del_req(replica_id, request)
            selected.append(request)
            if cur_replica_budget <= 0:
                break
        return selected

    def get_num_global_actual_waiting_tokens(self, replica_id: int, request: Any,
                                            prefix_cache_hit_len: int) -> int:
        probe = (-prefix_cache_hit_len, request._arrived_at, request)
        # The public caller also queries a request already in this queue.
        probe = next((item for item in self.queues[replica_id] if item[2] is request), probe)
        ahead = [item for item in self.queues[replica_id]
                 if item[2] is not request and self.priority(item) < self.priority(probe)]
        return sum(self._get_request_actual_prefill_tokens(item[2], item[0]) for item in ahead)


def create_app(args: Any) -> RequestAudit:
    from dualmap.entities.request import Request as ScheduledRequest
    from dualmap.scheduler.global_scheduler.double_hash_global_scheduler import DoubleHashGlobalScheduler
    from dualmap.scheduler.utils.lazy_prefix_table import LazyExpansionController, LazyPrefixTable, HotPrefixDetector
    from dualmap.scheduler.utils.shared import SharedState

    assert len(args.backends) >= 2 and args.prefill_tpot > 0 and args.ttft_slo > 0
    assert args.cpu_cache_gib > 0 and args.block_size > 0 and args.kv_bytes_per_token > 0

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    backends = args.backends
    # Public start_up.py uses ttft_slo / 2 for all three token budgets.
    budget = int(args.ttft_slo / (2 * args.prefill_tpot))
    config = SimpleNamespace(
        replicas_ip_port=",".join(url.removeprefix("http://") for url in backends),
        result_path=str(args.output), model_name=args.model, balance_type="dualmap",
        block_size=args.block_size, cache_capacity=int(args.cpu_cache_gib * 2**30),
        kv_cache_size_per_token=args.kv_bytes_per_token, replica_slo_budget=budget,
        dh_first_balance_ttft_thredhold=budget, dh_rebalance_thredhold=budget,
        dh_replica_pending_req_threshold=1, dh_rebalance_waiting_latency_thredhold=3,
        prefill_tpot=args.prefill_tpot, dh_recompute_punish_ratio=1, busy_prefill_interval=2,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "adapter-config.json").write_text(json.dumps(vars(config), indent=2))
    choices: dict[int, asyncio.Future[int]] = {}
    lock = asyncio.Lock()
    counter = itertools.count()
    prefix_table = LazyPrefixTable()
    expansion = LazyExpansionController(prefix_table, HotPrefixDetector())
    completed_lengths: list[int] = []
    agent_wait: dict[str, float] = {}
    visited: dict[str, set[int]] = {}
    fatal: list[str] = []

    class ReplayState(SharedState):
        async def add_posting_request_tasks(self, replica_id: int, request: Any) -> None:
            if args.agent_progress:
                agent_wait[request._native_session_id] = (request.agent_previous_wait_s
                                                          + time.perf_counter() - request._arrived_at)
            choices[request._id].set_result(replica_id)

    state = ReplayState(None, tokenizer, config)
    scheduler = DoubleHashGlobalScheduler(len(backends), 30, "dualmap", state, config)
    if args.agent_progress:
        scheduler.double_hash_util.global_request_queue = AgentProgressQueue(len(backends), args.prefill_tpot)
    (args.output / "agent-progress-config.json").write_text(json.dumps({
        "enabled": args.agent_progress,
        "priority": "arrival_s - previous_task_router_wait_s - cached_tokens * calibrated_prefill_s_per_token",
        "debt_reset": "task release", "tool_duration_used": False,
    }, indent=2))
    queue = scheduler.double_hash_util.global_request_queue

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(timeout=args.timeout_s,
                                     limits=httpx.Limits(max_connections=None, max_keepalive_connections=0),
                                     event_hooks={"request": [audit.dispatch]}) as client:
            app.state.client = client
            yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        if fatal:
            raise HTTPException(503, fatal[-1])
        return {"status": "ok", "outstanding": len(choices)}

    @app.post("/programs/release")
    async def release(request: Request) -> dict[str, Any]:
        job_id = (await request.json())["program_id"]
        agent_wait.pop(job_id, None)
        for i in visited.pop(job_id, set()):
            response = await app.state.client.post(backends[i] + "/continuum/programs/release",
                                                   json={"job_id": job_id})
            response.raise_for_status()
        return {"released": True, "program_id": job_id}

    @app.post("/v1/chat/completions")
    async def completions(request: Request) -> StreamingResponse:
        payload = await request.json()
        job_id = payload["program_id"]
        if payload.get("stream") is not True or payload.get("n", 1) != 1:
            raise HTTPException(400, "Replay requires one streaming completion")
        arrived = time.perf_counter()
        # Use the same chat template and generation prompt as the engine.
        tokens = tokenizer.apply_chat_template(payload["messages"], tokenize=True, add_generation_prompt=True)
        hashes = [int.from_bytes(hashlib.sha256(json.dumps(tokens[i:i+512]).encode()).digest()[:8], "big")
                  for i in range(0, len(tokens), 512)]
        async with lock:
            depth = prefix_table.lookup(hashes)
            expansion.process(hashes)
            rid = next(counter)
            predicted_output = round(sum(completed_lengths) / len(completed_lengths)) if completed_lengths else 128
            item = ScheduledRequest(rid, "agent-replay", job_id, job_id,
                                    "agent@" + "".join(map(str, hashes[:depth])), 0, "", tokens,
                                    len(tokens), len(tokens), predicted_output, False, 1, 0, 1,
                                    predicted_output, True, arrived, 0, depth)
            item.agent_previous_wait_s = agent_wait.get(job_id, 0.0)
            choices[rid] = asyncio.get_running_loop().create_future()

        replica_id: int | None = None
        response: httpx.Response | None = None

        async def cleanup() -> None:
            if response is not None:
                await response.aclose()
            async with lock:
                for i in range(len(backends)):
                    queue.del_req(i, item)
                    await state.replica_budgets[i].abort_request(item)
                choices.pop(rid)
                await scheduler.schedule(None)

        try:
            async with lock:
                await scheduler.schedule(item)
            replica_id = await choices[rid]
            row = _request_record.get()
            row["predicted_output_tokens"] = predicted_output
            row["agent_previous_wait_s"] = item.agent_previous_wait_s
            row["estimated_cached_tokens"] = [max(0, replica.get_num_cached_tokens(tokens))
                                              for replica in state.replica_budgets.values()]
            row["rebalance_count"] = scheduler.double_hash_util.rebalance_cnt
            visited.setdefault(job_id, set()).add(replica_id)
            forwarded = dict(payload, job_id=job_id, this_func_call="", is_last_step=False)
            del forwarded["program_id"]
            response = await app.state.client.send(app.state.client.build_request(
                "POST", backends[replica_id] + "/v1/chat/completions", json=forwarded), stream=True)
            response.raise_for_status()
            # Match the public client's accounting point: successful HTTP headers.
            async with lock:
                assert await state.replica_budgets[replica_id].add_request(item)
        except BaseException as exc:
            if isinstance(exc, Exception) and not isinstance(exc, TimeoutError):
                fatal.append(repr(exc))
            await cleanup()
            raise

        async def stream():
            buffer = b""
            first = True
            text = ""
            usage = None
            try:
                async for chunk in response.aiter_bytes():
                    buffer += chunk
                    while b"\n\n" in buffer:
                        event, buffer = buffer.split(b"\n\n", 1)
                        for line in event.splitlines():
                            if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                                continue
                            data = json.loads(line[5:])
                            if "error" in data:
                                raise RuntimeError(data["error"])
                            if data.get("usage"):
                                usage = data["usage"]
                            for choice in data.get("choices", []):
                                delta = choice.get("delta", {})
                                content = delta.get("content") or ""
                                text += content
                                if first and (content or choice.get("token_ids") or delta.get("token_ids")):
                                    async with lock:
                                        assert await state.replica_budgets[replica_id].complete_request_prefill(
                                            replica_id, item, time.perf_counter() - arrived)
                                    first = False
                    yield chunk
                assert usage is not None, "Missing terminal usage"
                async with lock:
                    state.replica_budgets[replica_id].complete_request_decode(item, text)
                    completed_lengths.append(usage["completion_tokens"])
            except Exception as exc:
                fatal.append(repr(exc))
                raise
            finally:
                await cleanup()

        return StreamingResponse(stream(), media_type="text/event-stream")

    audit = RequestAudit(app, (args.output / "routing.jsonl").open("x", buffering=1), args.timeout_s)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backends", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefill-tpot", type=float, required=True)
    parser.add_argument("--kv-bytes-per-token", type=int, required=True)
    parser.add_argument("--cpu-cache-gib", type=float, required=True)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--ttft-slo", type=float, default=5)
    parser.add_argument("--timeout-s", type=float, default=1800)
    parser.add_argument("--agent-progress", action="store_true")
    args = parser.parse_args()
    uvicorn.run(create_app(args), host="127.0.0.1", port=9000, timeout_graceful_shutdown=2)
