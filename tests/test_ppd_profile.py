"""Reduced CPU plumbing case for paired calibration; not performance evidence."""
import json
from types import SimpleNamespace

import pytest
from aiohttp import ClientSession, web

from scripts.evaluation.profile_ppd import profile


@pytest.mark.asyncio
async def test_paired_prompts_raw_results_and_session_cleanup(monkeypatch, tmp_path):
    from scripts.benchmark import comprehensive_benchmark as public

    monkeypatch.setattr(public, "T1_CONFIGS", {"small": {"input": 16, "output": 4}})
    monkeypatch.setattr(public, "T2_CONFIGS", {"tiny": {"input": 16, "output": 4}})
    monkeypatch.setattr(public, "QPS_POINTS", [20])
    monkeypatch.setattr(public, "DURATION_PER_POINT_SEC", 0.1)
    monkeypatch.setattr(public, "WARMUP_REQUESTS", 1)
    requests = []
    sessions = set()
    inference_clients = set()
    cleanup_clients = set()
    original_request = ClientSession._request

    async def observe_connection_pool(client, method, url, **kwargs):
        if url.endswith("/programs/release"):
            cleanup_clients.add(client)
        else:
            inference_clients.add(client)
        return await original_request(client, method, url, **kwargs)

    monkeypatch.setattr(ClientSession, "_request", observe_connection_pool)

    async def completion(request):
        assert request.headers.get("Connection") == "close"
        data = await request.json()
        requests.append(data)
        sessions.add(data["program_id"])
        count = data["max_tokens"]
        body = "data: " + json.dumps({"id": "chatcmpl-test", "choices": [
            {"delta": {"content": "response"}, "token_ids": list(range(count))}]}) + "\n\n"
        body += "data: " + json.dumps({"id": "chatcmpl-test", "choices": [],
                                      "usage": {"completion_tokens": count, "prompt_tokens": 100}}) + "\n\n"
        body += "data: [DONE]\n\n"
        return web.Response(text=body, content_type="text/event-stream")

    async def release(request):
        sessions.remove((await request.json())["program_id"])
        return web.json_response({"released": True})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completion)
    app.router.add_post("/programs/release", release)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    try:
        for mode in ("pd", "static-x1"):
            await profile(SimpleNamespace(mode=mode, model="plumbing", seed=42, start_point=1,
                proxy=f"http://127.0.0.1:{port}", output=tmp_path / mode))
            assert not sessions
        # Cleanup must remain reachable when every inference connection is busy.
        assert cleanup_clients and cleanup_clients.isdisjoint(inference_clients)
        midpoint = len(requests) // 2
        assert midpoint == 5  # One warmup plus two two-turn conversations.
        assert [r["messages"] for r in requests[:midpoint]] == [r["messages"] for r in requests[midpoint:]]
        for mode, config in (("pd", "1P_1D"), ("static-x1", "1P_1pD")):
            root = tmp_path / mode / config
            rows = [json.loads(s) for s in (root / "turns.jsonl").read_text().splitlines()]
            assert len(rows) == 5 and all(r["success"] for r in rows)
            result = json.loads((root / f"{config}_small_tiny_20.json").read_text())
            assert result["turn2"]["count"] == 2 and result["success_rate"] == 100
        monkeypatch.setattr(public, "QPS_POINTS", [20, 40])
        paired_remaining = []
        for start in (1, 2):
            before = len(requests)
            await profile(SimpleNamespace(mode="pd", model="plumbing", seed=42, start_point=start,
                proxy=f"http://127.0.0.1:{port}", output=tmp_path / f"start-{start}"))
            paired_remaining.append([r["messages"] for r in requests[before:]
                if "small_tiny-40" in r["messages"][0]["content"]])
            rows = [json.loads(s) for s in
                (tmp_path / f"start-{start}/1P_1D/turns.jsonl").read_text().splitlines()]
            points = {r["point"] for r in rows}
            assert ("small_tiny-20" in points) == (start == 1)
            assert "small_tiny-40" in points
            assert not sessions
        assert paired_remaining[0]
        assert sorted(map(json.dumps, paired_remaining[0])) == sorted(map(json.dumps, paired_remaining[1]))
        before_contexts = public.T1_CONFIGS
        before = len(requests)
        await profile(SimpleNamespace(mode="pd", model="plumbing", seed=42, start_point=1,
            context_tokens=32768, proxy=f"http://127.0.0.1:{port}", output=tmp_path / "huge"))
        assert public.T1_CONFIGS is before_contexts
        config = json.loads((tmp_path / "huge/1P_1D/profile-config.json").read_text())
        assert config["t1_configs"] == {"huge": {"input": 31744, "output": 1024}}
        assert config["qps_points"] == [20, 40] and config["tool_delay_s"] == 0
        assert any(r["max_tokens"] == 1024 for r in requests[before:])
        assert (tmp_path / "huge/1P_1D/1P_1D_huge_tiny_20.json").exists()
        assert not sessions
    finally:
        await runner.cleanup()
