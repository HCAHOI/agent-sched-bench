"""CPU plumbing regression using the real public scheduling implementation."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("dualmap")
from scripts.baselines import dualmap_official_proxy as proxy


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_progress", [False, True])
async def test_prefill_admission_cancellation_and_stream_feedback(monkeypatch, tmp_path, agent_progress):
    pytest.importorskip("dualmap")

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return list(range(100))

        def encode(self, text):
            return [100] * len(text)

    monkeypatch.setattr(proxy.AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    first_token = asyncio.Event()
    finish = asyncio.Event()
    two_dispatched = asyncio.Event()
    third_dispatched = asyncio.Event()
    received = []

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
            await first_token.wait()
            yield b'data: {"choices":[{"delta":{"content":"x"},"token_ids":[100]}]}\n\n'
            await finish.wait()
            yield b'data: {"choices":[],"usage":{"completion_tokens":1}}\n\ndata: [DONE]\n\n'

    async def backend(request):
        if request.url.path.endswith("/release"):
            return httpx.Response(200, json={"released": True})
        received.append(json.loads(request.content))
        if len(received) == 2:
            two_dispatched.set()
        if len(received) == 3:
            third_dispatched.set()
        return httpx.Response(200, stream=Stream())

    real_client = httpx.AsyncClient

    def client_factory(**kwargs):
        return real_client(transport=httpx.MockTransport(backend), **kwargs)

    monkeypatch.setattr(proxy.httpx, "AsyncClient", client_factory)
    args = SimpleNamespace(model="plumbing-only", backends=["http://engine0", "http://engine1"],
                           output=tmp_path, ttft_slo=5, prefill_tpot=0.05,
                           cpu_cache_gib=0.001, block_size=16, kv_bytes_per_token=16, timeout_s=5,
                           agent_progress=agent_progress)
    audit = proxy.create_app(args)
    async with audit.app.router.lifespan_context(audit.app):
        async with real_client(transport=httpx.ASGITransport(audit), base_url="http://proxy") as client:
            async def send(job):
                return await client.post("/v1/chat/completions", json={"program_id": job,
                    "messages": [{"role": "user", "content": "input"}], "stream": True,
                    "max_tokens": 1, "ignore_eos": True, "return_token_ids": True})

            tasks = [asyncio.create_task(send(str(i))) for i in range(4)]
            await asyncio.wait_for(two_dispatched.wait(), 2)
            await asyncio.sleep(0.02)
            assert len(received) == 2
            tasks[3].cancel()
            with pytest.raises(asyncio.CancelledError):
                await tasks[3]
            first_token.set()
            await asyncio.sleep(0.02)
            assert len(received) == 2  # The public scheduler wakes on arrivals/completions, not first tokens.
            finish.set()
            await asyncio.wait_for(third_dispatched.wait(), 2)
            responses = await asyncio.gather(*tasks[:3])
            assert all(r.content.endswith(b"data: [DONE]\n\n") for r in responses)
            assert (await client.get("/health")).json()["outstanding"] == 0
            assert (await send("0")).status_code == 200
            assert (await client.post("/programs/release", json={"program_id": "0"})).status_code == 200
            assert (await send("0")).status_code == 200
    rows = [json.loads(line) for line in (tmp_path / "routing.jsonl").read_text().splitlines()]
    dispatches = [r for r in rows if r["event"] == "dispatch"]
    assert all(r["predicted_output_tokens"] == 128 for r in dispatches[:3])
    assert dispatches[3]["agent_previous_wait_s"] > 0 if agent_progress else dispatches[3]["agent_previous_wait_s"] == 0
    assert dispatches[4]["agent_previous_wait_s"] == 0
    assert all(p["max_tokens"] == 1 and "program_id" not in p and p["job_id"] for p in received)


def test_agent_wait_vs_cache_savings_and_migration_accounting():
    queue = proxy.AgentProgressQueue(2, 0.01)

    def request(rid, arrival, debt=0):
        return SimpleNamespace(_id=rid, _arrived_at=arrival, agent_previous_wait_s=debt,
                               _num_prefill_tokens=1000, _input_ids=list(range(1000)))

    cold = request(0, 100)
    warm = request(1, 102)
    queue.push(0, cold, 0)
    queue.push(0, warm, 500)  # Saves 5 s: may pass an agent that waited 2 s.
    assert queue.peek(0) is warm
    assert queue.get_num_global_actual_waiting_tokens(0, cold, 0) == 500
    # A later shadow-cache estimate must not change the queued dispatch order.
    assert queue.get_num_global_actual_waiting_tokens(0, warm, 0) == 0
    assert queue.get_num_global_actual_waiting_tokens(0, cold, 1000) == 500
    assert queue.pop_schedulable(0, 100, 0) == []
    assert queue.pop_schedulable(0, 100, 1) == [warm]
    assert queue.get_global_actual_waiting_tokens_count(0) == 1000
    recent = request(2, 106)
    queue.push(0, recent, 500)  # Saving 5 s no longer outweighs the 6 s wait.
    assert queue.peek(0) is cold
    returning = request(3, 108, 10)  # Past task wait survives the tool phase.
    queue.push(0, returning, 0)
    assert queue.peek(0) is returning
    assert queue.get_num_global_actual_waiting_tokens(0, recent, 500) == 2000
    assert queue.del_req(0, returning)
    queue.push(1, returning, 300)  # Migration changes cache estimate, not debt.
    assert queue.get_global_actual_waiting_tokens_count(1) == 700
    assert queue.pop(1) is returning
    assert queue.get_global_actual_waiting_tokens_count(1) == 0
    assert queue.pop_schedulable(0, 2000, 2) == [cold, recent]
    assert queue.get_global_actual_waiting_tokens_count(0) == 0
    assert queue.get_global_input_waiting_tokens_count(0) == 0
