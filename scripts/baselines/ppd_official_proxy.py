"""One P and one D, with the public PPD decision engine and audited replay.

PD always visits P then D. Default PPD calls the unchanged public decision engine.
Explicit flags add measured huge-context scores and a cache/load-aware D guard.
Agent sessions use program IDs; new input is measured against the last completed
token history, and predicted output length uses only completed responses.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from starlette.responses import StreamingResponse
from transformers import AutoTokenizer

from scripts.baselines.thunderagent_official_launcher import RequestAudit, _request_record
from scripts.baselines.ppd_policy import add_huge_context_table, protect_decode


@dataclass
class Session:
    turns: int = 0
    history: list[int] = field(default_factory=list)
    last_access: float = field(default_factory=time.monotonic)

    def context_tokens(self, prompt: list[int]) -> int:
        return next((i for i, (a, b) in enumerate(zip(self.history, prompt)) if a != b),
                    min(len(self.history), len(prompt)))


def create_app(args: Any) -> RequestAudit:
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    engine = None
    extended_context = getattr(args, "extended_context", False)
    state_aware = getattr(args, "state_aware", False)
    assert not state_aware or extended_context, "State-aware comparison uses the same extended table"
    if args.mode == "ppd":
        from ppd.optimizer.ppd_decision_engine import (
            PPDDecisionEngine, QPS_POINTS, T2_WORKLOAD_CONFIGS,
            classify_context_length, find_nearest_qps,
        )
        assert args.benchmark_data is not None, "PPD requires measured calibration data"
        engine = PPDDecisionEngine(str(args.benchmark_data), base_config="1P_1D", w_ttft=1, w_tpot=1)
        assert len(engine.performance_data) == 2 * len(T2_WORKLOAD_CONFIGS) * len(QPS_POINTS), \
            "Incomplete PPD calibration table; refusing default-only routing"
        if extended_context:
            add_huge_context_table(engine, args.benchmark_data)
    assert engine is not None or not (extended_context or state_aware)
    backends = args.backends
    assert len(backends) == 2 and len(set(backends)) == 2 and args.transport == "nixl"
    sessions: dict[str, Session] = {}
    pending: dict[str, asyncio.Task[Any]] = {}
    total_output = completed = arrivals = 0
    started = time.monotonic()
    fatal: list[str] = []
    args.output.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args))
    if engine is not None:
        config["public_ppd"] = dict(
            bypass_threshold=int(os.environ.get("PPD_BYPASS_THRESHOLD", "512")),
            w_ttft=engine.w_ttft, w_tpot=engine.w_tpot,
            output_prediction="mean of completed responses; cold start 128",
            qps_estimate="cumulative arrivals before this request / elapsed seconds",
            extended_context=extended_context, state_aware=state_aware,
            state_guard="actual uncached tokens >= bypass threshold AND D queue full or waiting",
        )
        engine.export_lookup_table(str(args.output / "ppd-lookup-table.json"))
    (args.output / "adapter-config.json").write_text(json.dumps(config, default=str, indent=2))

    class PpdAudit(RequestAudit):
        async def dispatch(self, request: httpx.Request) -> None:
            if request.url.path != "/v1/chat/completions":
                return
            row = _request_record.get()
            request.headers["x-request-id"] = row["wire_request_id"]
            backend = str(request.url.copy_with(path=""))
            if backend == backends[0]:
                self.record("prefill_dispatch", row, backend=backend)
            else:
                assert backend == backends[1]
                row["backend"] = backend
                self.record("dispatch", row)

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
        return {"status": "ok", "outstanding": len(pending), "sessions": len(sessions)}

    @app.post("/programs/release")
    async def release(request: Request) -> dict[str, Any]:
        job_id = (await request.json())["program_id"]
        if task := pending.get(job_id):
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        sessions.pop(job_id, None)
        return {"released": True, "program_id": job_id}

    @app.post("/profiling/reset")
    async def reset_profile() -> dict[str, bool]:
        nonlocal arrivals, total_output, completed, started
        if args.mode != "profile":
            raise HTTPException(404, "Profiling mode is disabled")
        if pending or sessions:
            raise HTTPException(409, "Release and drain every conversation first")
        arrivals = total_output = completed = 0
        started = time.monotonic()
        return {"reset": True}

    @app.post("/v1/chat/completions")
    async def completion(request: Request) -> StreamingResponse:
        nonlocal arrivals, total_output, completed
        payload = await request.json()
        job_id = payload["program_id"]
        if payload.get("stream") is not True or payload.get("n", 1) != 1 or not payload.get("return_token_ids"):
            raise HTTPException(400, "Replay requires one stream with token IDs")
        if job_id in pending:
            raise HTTPException(409, "Concurrent requests for one agent")
        prompt = tokenizer.apply_chat_template(payload["messages"], tokenize=True, add_generation_prompt=True, return_dict=False)
        now = time.monotonic()
        session = sessions.get(job_id)
        if session is None or now - session.last_access > 3600:
            session = sessions[job_id] = Session()
        context_tokens = session.context_tokens(prompt)
        predicted_output = round(total_output / completed) if completed else 128
        # Match the public proxy's cumulative arrival-rate estimate (before this arrival).
        qps = arrivals / max(1, now - started)
        arrivals += 1
        turn = session.turns + 1
        use_local = args.mode == "static-x1" and turn > 1
        decision_details = {}
        forced_path = payload.pop("profiling_path", None)
        if args.mode == "profile":
            if forced_path not in {"pd", "local"} or (turn == 1 and forced_path != "pd"):
                raise HTTPException(400, "Profiling requires PD on turn one and an explicit return path")
            use_local = forced_path == "local"
            decision_details["profiling_path"] = forced_path
        elif forced_path is not None:
            raise HTTPException(400, "Forced paths are only allowed in profiling mode")
        if engine is not None:
            # Read the public engine's counter delta; do not reimplement its branches.
            before = {key: sum(counts.values()) for key, counts in engine.stats.decisions_by_workload.items()}
            use_local = engine.should_use_ppd(turn=turn, input_tokens=len(prompt) - context_tokens,
                                            output_tokens=predicted_output, current_qps=qps,
                                            context_tokens=context_tokens)
            reason = next((key for key, counts in engine.stats.decisions_by_workload.items()
                           if sum(counts.values()) > before.get(key, 0)), "turn1")
            decision_details["ppd_decision_reason"] = reason
            if reason not in {"turn1", "short_input_bypass"}:
                context_class = classify_context_length(context_tokens)
                decision_details["ppd_lookup_key"] = [
                    "large" if context_class == "huge" and not extended_context else context_class,
                    reason, find_nearest_qps(qps),
                ]
        if state_aware:
            query_started = time.perf_counter()
            pending[job_id] = asyncio.current_task()
            try:
                response = await app.state.client.post(backends[1] + "/ppd/cache_state",
                                                       json={"prompt_token_ids": prompt})
                response.raise_for_status()
                state = response.json()
            except BaseException as exc:
                del pending[job_id]
                if isinstance(exc, Exception):
                    fatal.append(repr(exc))
                raise
            decision_details["state_query_ms"] = (time.perf_counter() - query_started) * 1000
            decision_details["decode_state"] = state
            decision_details["calibrated_use_local"] = use_local
            use_local, guard = protect_decode(use_local, len(prompt), state,
                                               int(os.environ.get("PPD_BYPASS_THRESHOLD", "512")))
            decision_details["state_guard_reason"] = guard
        row = _request_record.get()
        wire_id = "ppd_" + row["route_id"]
        row.update(wire_request_id=wire_id, engine_request_id="chatcmpl-" + wire_id,
                   routing_mode="local" if use_local else "pd", turn=turn,
                   input_tokens=len(prompt) - context_tokens, context_tokens=context_tokens,
                   predicted_output_tokens=predicted_output, current_qps=qps,
                   requested_output_tokens=payload["max_tokens"])
        row.update(decision_details)
        audit.record("decision", row)
        session.turns = turn
        session.last_access = now
        pending[job_id] = asyncio.current_task()
        forwarded = dict(payload)
        del forwarded["program_id"]

        async def backend_stream(index: int, data: dict[str, Any], result: dict[str, Any]):
            async with app.state.client.stream("POST", backends[index] + "/v1/chat/completions", json=data) as response:
                response.raise_for_status()
                buffer = b""
                done = False
                async for chunk in response.aiter_bytes():
                    buffer += chunk
                    while b"\n\n" in buffer:
                        event, buffer = buffer.split(b"\n\n", 1)
                        for line in event.splitlines():
                            if not line.startswith(b"data:"):
                                continue
                            if line[5:].strip() == b"[DONE]":
                                done = True
                                continue
                            message = json.loads(line[5:])
                            if "error" in message:
                                raise RuntimeError(message["error"])
                            if message.get("usage"):
                                result["usage"] = message["usage"]
                            for choice in message.get("choices", []):
                                result["tokens"].extend(choice.get("token_ids") or choice.get("delta", {}).get("token_ids") or [])
                    yield chunk
                assert done and result.get("usage"), "Incomplete backend stream"
                assert len(result["tokens"]) == result["usage"]["completion_tokens"] == data["max_tokens"], \
                    "Backend did not preserve forced output length or token IDs"

        try:
            if not use_local:
                # vLLM 0.13 exposes NIXL block metadata only on non-stream responses.
                prefill_payload = dict(forwarded, stream=False, max_tokens=1,
                                       kv_transfer_params={"do_remote_decode": True,
                                                           "do_remote_prefill": False})
                prefill_payload.pop("stream_options", None)
                response = await app.state.client.post(backends[0] + "/v1/chat/completions",
                                                       json=prefill_payload)
                response.raise_for_status()
                prefill_result = response.json()
                assert prefill_result["usage"]["completion_tokens"] == 1, prefill_result
                params = prefill_result["kv_transfer_params"]
                assert params["do_remote_prefill"] and params["remote_block_ids"], params
                assert params["remote_request_id"] == row["engine_request_id"], params
                forwarded["kv_transfer_params"] = params
                audit.record("prefill_finish", row, backend=backends[0], outcome="complete",
                             kv_transfer_params=params)
        except BaseException as exc:
            del pending[job_id]
            if isinstance(exc, Exception) and not isinstance(exc, TimeoutError):
                fatal.append(repr(exc))
            raise

        async def stream():
            nonlocal total_output, completed
            result: dict[str, Any] = {"tokens": []}
            try:
                async for chunk in backend_stream(1, forwarded, result):
                    yield chunk
                session.history = prompt + result["tokens"]
                session.last_access = time.monotonic()
                total_output += len(result["tokens"])
                completed += 1
            except Exception as exc:
                if not isinstance(exc, TimeoutError):
                    fatal.append(repr(exc))
                raise
            finally:
                del pending[job_id]

        return StreamingResponse(stream(), media_type="text/event-stream")

    audit = PpdAudit(app, (args.output / "routing.jsonl").open("x", buffering=1), args.timeout_s)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["pd", "static-x1", "ppd", "profile"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backends", nargs=2, required=True, help="P and D HTTP bases, in that order")
    parser.add_argument("--transport", choices=["nixl"], default="nixl")
    parser.add_argument("--benchmark-data", type=Path)
    parser.add_argument("--extended-context", action="store_true")
    parser.add_argument("--state-aware", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=1800)
    args = parser.parse_args()
    uvicorn.run(create_app(args), host="127.0.0.1", port=9000, timeout_graceful_shutdown=2)
