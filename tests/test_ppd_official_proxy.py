"""CPU protocol tests; synthetic table values are not performance evidence."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from scripts.baselines import ppd_official_proxy as proxy
from scripts.evaluation.check_ppd_request_metrics import check


@pytest.mark.asyncio
async def test_forced_profile_paths_and_group_reset(monkeypatch, tmp_path):
    monkeypatch.setattr(proxy.AutoTokenizer, "from_pretrained", lambda *a, **kw:
                        SimpleNamespace(apply_chat_template=lambda *a, **kw: list(range(30))))
    received = []

    async def backend(request):
        data = json.loads(request.content)
        received.append((request.url.port, data))
        assert "profiling_path" not in data and "program_id" not in data
        if request.url.port == 8000:
            return httpx.Response(200, json={"usage": {"completion_tokens": 1},
                "kv_transfer_params": {"do_remote_prefill": True, "remote_block_ids": [1],
                    "remote_request_id": "chatcmpl-" + request.headers["x-request-id"]}})
        count = data["max_tokens"]
        return httpx.Response(200, text="data: " + json.dumps({"choices": [
            {"token_ids": list(range(count)), "delta": {"content": "hi"}}],
            "usage": {"completion_tokens": count}}) + "\n\ndata: [DONE]\n\n")

    real_client = httpx.AsyncClient
    monkeypatch.setattr(proxy.httpx, "AsyncClient", lambda **kw:
                        real_client(transport=httpx.MockTransport(backend), **kw))
    args = SimpleNamespace(mode="profile", model="plumbing", benchmark_data=None,
        output=tmp_path, backends=["http://127.0.0.1:8000", "http://127.0.0.1:8001"],
        transport="nixl", timeout_s=10)
    audit = proxy.create_app(args)
    async with audit.app.router.lifespan_context(audit.app):
        async with real_client(transport=httpx.ASGITransport(audit), base_url="http://proxy") as client:
            for path in ("local", "pd"):
                for turn in (1, 2):
                    response = await client.post("/v1/chat/completions", json=dict(
                        program_id=path, messages=[], stream=True, return_token_ids=True,
                        max_tokens=3, profiling_path="pd" if turn == 1 else path))
                    assert response.status_code == 200
                assert (await client.post("/profiling/reset")).status_code == 409
                assert (await client.post("/programs/release", json={"program_id": path})).status_code == 200
                assert (await client.post("/profiling/reset")).json() == {"reset": True}
    decisions = [json.loads(line) for line in (tmp_path / "routing.jsonl").read_text().splitlines()
                 if json.loads(line)["event"] == "decision"]
    assert [r["routing_mode"] for r in decisions] == ["pd", "local", "pd", "pd"]
    assert [p for p, _ in received] == [8000, 8001, 8001, 8000, 8001, 8000, 8001]
    assert decisions[2]["current_qps"] == 0 and decisions[2]["predicted_output_tokens"] == 128


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_phase", ["prefill", "decode"])
@pytest.mark.parametrize("table_favors_local", [False, True])
@pytest.mark.parametrize("state_aware", [False, True])
async def test_pd_legs_causal_routing_and_release_cancellation(monkeypatch, tmp_path, cancel_phase, table_favors_local, state_aware):
    from ppd.optimizer.ppd_decision_engine import QPS_POINTS, T2_WORKLOAD_CONFIGS
    monkeypatch.setenv("PPD_BYPASS_THRESHOLD", "512")

    table = tmp_path / "table"
    for mode, ttft in (("1P_1D", 1), ("1P_1pD", 0.5 if table_favors_local else 2)):
        (table / mode).mkdir(parents=True)
        for context in (("small", "large", "huge") if state_aware else ("small", "large")):
            for workload in T2_WORKLOAD_CONFIGS:
                for qps in QPS_POINTS:
                    (table / mode / f"{mode}_{context}_{workload}_{qps}.json").write_text(json.dumps(
                        {"success_rate": 1, "turn2": {"avg_ttft_ms": ttft, "avg_tpot_ms": 1}}))

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return json.loads(messages[0]["content"])

    monkeypatch.setattr(proxy.AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    received = []
    held = asyncio.Event()
    never_finish = asyncio.Event()

    class Stream(httpx.AsyncByteStream):
        def __init__(self, count, hold):
            self.count, self.hold = count, hold

        async def __aiter__(self):
            if self.hold:
                held.set()
                await never_finish.wait()
            yield ("data: " + json.dumps({"choices": [{"delta": {"content": "x"},
                    "token_ids": list(range(700, 700 + self.count))}]} ) + "\n\n").encode()
            yield ("data: " + json.dumps({"choices": [], "usage": {"completion_tokens": self.count}})
                   + "\n\ndata: [DONE]\n\n").encode()

    async def backend(request):
        data = json.loads(request.content)
        if request.url.path == "/ppd/cache_state":
            assert request.url.port == 8001 and data["prompt_token_ids"]
            return httpx.Response(200, json=dict(cached_tokens=0, running=8, waiting=1,
                                                 max_num_seqs=8, kv_cache_usage=0.9, timestamp_s=0))
        received.append((str(request.url), request.headers["x-request-id"], data))
        if request.url.port == 8000:
            assert data["stream"] is False and data["kv_transfer_params"]["do_remote_decode"]
            if data["seed"] == 99 and cancel_phase == "prefill":
                held.set()
                await never_finish.wait()
            return httpx.Response(200, json={"usage": {"completion_tokens": 1},
                "kv_transfer_params": {"do_remote_prefill": True, "do_remote_decode": False,
                    "remote_block_ids": [3, 4], "remote_engine_id": "P", "remote_host": "127.0.0.1",
                    "remote_port": 14579, "remote_request_id": "chatcmpl-" + request.headers["x-request-id"]}})
        if "kv_transfer_params" in data:
            assert data["kv_transfer_params"]["remote_block_ids"] == [3, 4]
        return httpx.Response(200, stream=Stream(data["max_tokens"], data["seed"] == 99))

    real_client = httpx.AsyncClient
    monkeypatch.setattr(proxy.httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(backend), **kw))
    args = SimpleNamespace(mode="ppd", model="plumbing-only", benchmark_data=table,
                           extended_context=state_aware, state_aware=state_aware,
                           output=tmp_path / "run", backends=["http://127.0.0.1:8000", "http://127.0.0.1:8001"],
                           transport="nixl", timeout_s=10)
    audit = proxy.create_app(args)
    async with audit.app.router.lifespan_context(audit.app):
        async with real_client(transport=httpx.ASGITransport(audit), base_url="http://proxy") as client:
            async def send(prompt, count=3, job="agent", seed=0):
                return await client.post("/v1/chat/completions", json=dict(program_id=job,
                    messages=[dict(role="user", content=json.dumps(prompt))], stream=True,
                    return_token_ids=True, max_tokens=count, ignore_eos=True, seed=seed))

            prompt = list(range(600))
            assert (await send(prompt)).status_code == 200
            prompt += [700, 701, 702] + [900, 901]
            assert (await send(prompt)).status_code == 200  # Public short-input bypass.
            prompt += [700, 701, 702] + list(range(1000, 1600))
            assert (await send(prompt, count=7)).status_code == 200  # Measured-table path.
            request = asyncio.create_task(send([5] * 600, job="cancel", seed=99))
            await asyncio.wait_for(held.wait(), 2)
            assert (await client.post("/programs/release", json={"program_id": "cancel"})).status_code == 200
            with pytest.raises(asyncio.CancelledError):
                await request
            assert (await client.post("/programs/release", json={"program_id": "agent"})).status_code == 200
            assert (await client.get("/health")).json() == {"status": "ok", "outstanding": 0, "sessions": 0}
    rows = [json.loads(line) for line in (args.output / "routing.jsonl").read_text().splitlines()]
    decisions = [row for row in rows if row["event"] == "decision" and row["job_id"] == "agent"]
    second_local = not state_aware
    third_local = table_favors_local and not state_aware
    assert [row["routing_mode"] for row in decisions] == ["pd", "local" if second_local else "pd", "local" if third_local else "pd"]
    if state_aware:
        assert decisions[1]["calibrated_use_local"] is True
        assert decisions[1]["state_guard_reason"] == "uncached_prefill_on_busy_decode"
        assert all(r["state_query_ms"] >= 0 and r["decode_state"]["cached_tokens"] == 0 for r in decisions)
    assert [row["ppd_decision_reason"] for row in decisions] == ["turn1", "short_input_bypass", "huge_paste"]
    assert decisions[2]["ppd_lookup_key"][:2] == ["large", "huge_paste"]
    assert decisions[2]["ppd_lookup_key"][2] in QPS_POINTS
    table_export = json.loads((args.output / "ppd-lookup-table.json").read_text())
    assert len(table_export["entries"]) == (270 if state_aware else 180)
    assert json.loads((args.output / "adapter-config.json").read_text())["public_ppd"]["bypass_threshold"] == 512
    assert [row["input_tokens"] for row in decisions] == [600, 2, 600]
    assert [row["predicted_output_tokens"] for row in decisions] == [128, 3, 3]
    expected_lengths = [1, 3] + ([3] if second_local else [1, 3]) + ([7] if third_local else [1, 7])
    assert [data["max_tokens"] for _, _, data in received[:len(expected_lengths)]] == expected_lengths
    assert received[0][1] == received[1][1]
    if not third_local:
        assert received[len(expected_lengths)-2][1] == received[len(expected_lengths)-1][1]
    assert all("program_id" not in data for _, _, data in received)
    assert [r["outcome"] for r in rows if r["event"] == "finish"] == ["complete", "complete", "complete", "cancelled"]
    # Exercise the integrity gate with the router's actual records. These terminal
    # fixtures validate missing-record detection, not model performance.
    for index in range(2):
        cell = args.output / f"instance-{index}"
        cell.mkdir()
        terminals = []
        terminal_decisions = [row for row in rows if row["event"] == "decision" and
                              (row["job_id"] == "agent" or (index == 0 and cancel_phase == "decode"))]
        for row in terminal_decisions:
            if index == 0 and row["routing_mode"] == "local":
                continue
            terminals.append(dict(request_id=row["engine_request_id"], generation_tokens=1 if index == 0 else row["requested_output_tokens"],
                ttft_s=1, queue_s=0, prefill_s=1, decode_s=1, e2e_s=2, preempted_wait_s=0,
                preemption_timing_complete=True))
        (cell / "vllm-request-telemetry.jsonl").write_text("".join(json.dumps(t) + "\n" for t in terminals))
    (args.output / "instance-0/vllm.log").write_text("".join(
        "INFO PPD_KV_RELEASE " + json.dumps(dict(request_id=row["engine_request_id"],
        reason="consumer_notification")) + "\n" for row in decisions if row["routing_mode"] == "pd"))
    log = args.output / "instance-1/vllm.log"
    allocation_log = "".join("INFO PPD_KV_ALLOCATION " + json.dumps(dict(request_id=row["engine_request_id"], external_tokens=100)) + "\n"
                            for row in decisions if row["routing_mode"] == "pd")
    log.write_text(allocation_log + "".join("INFO PPD_KV_TRANSFER " + json.dumps(dict(request_id=row["engine_request_id"],
        num_layers=36, received_bytes=128, receive_and_inject_span_ms=2)) + "\n"
        for row in decisions if row["routing_mode"] == "pd"))
    check(args.output, 36, final=True)
    prefill_path = args.output / "instance-0/vllm-request-telemetry.jsonl"
    original = prefill_path.read_text()
    prefill_path.write_text("")
    with pytest.raises(AssertionError):
        check(args.output, 36, final=True)
    prefill_path.write_text(original)
    log.write_text(allocation_log)
    with pytest.raises(AssertionError, match="Missing worker KV receive"):
        check(args.output, 36, final=True)
    # An explicitly logged complete local hit on D needs no network receipt.
    log.write_text(allocation_log.replace('"external_tokens": 100', '"external_tokens": 0'))
    check(args.output, 36, final=True)
    decode_path = args.output / "instance-1/vllm-request-telemetry.jsonl"
    decode_path.write_text(decode_path.read_text().replace('"generation_tokens": 7', '"generation_tokens": 3'))
    with pytest.raises(AssertionError):
        check(args.output, 36, final=True)

    next((table / "1P_1D").glob("1P_1D_small_*.json")).unlink()
    with pytest.raises(AssertionError, match="Incomplete PPD calibration table"):
        proxy.create_app(args)


@pytest.mark.asyncio
async def test_arrivals_are_counted_while_cache_queries_overlap(monkeypatch, tmp_path):
    from ppd.optimizer import ppd_decision_engine as public

    qps_seen = []

    class DecisionEngine:
        def __init__(self, *args, **kwargs):
            self.performance_data = dict.fromkeys(range(2 * len(public.T2_WORKLOAD_CONFIGS) * len(public.QPS_POINTS)))
            self.stats = SimpleNamespace(decisions_by_workload={})
            self.w_ttft = self.w_tpot = 1

        def should_use_ppd(self, **kwargs):
            qps_seen.append(kwargs["current_qps"])
            return False

        def export_lookup_table(self, path):
            pass

    monkeypatch.setattr(public, "PPDDecisionEngine", DecisionEngine)
    monkeypatch.setattr(proxy, "add_huge_context_table", lambda *args: None)
    monkeypatch.setattr(proxy.AutoTokenizer, "from_pretrained", lambda *a, **kw:
                        SimpleNamespace(apply_chat_template=lambda *a, **kw: [1, 2, 3]))
    first_query = asyncio.Event()
    both_queries = asyncio.Event()
    query_count = 0

    async def backend(request):
        nonlocal query_count
        assert request.url.path == "/ppd/cache_state"
        query_count += 1
        if query_count == 1:
            first_query.set()
        else:
            both_queries.set()
        await asyncio.Event().wait()  # Release must cancel both pending queries.

    real_client = httpx.AsyncClient
    monkeypatch.setattr(proxy.httpx, "AsyncClient", lambda **kw:
                        real_client(transport=httpx.MockTransport(backend), **kw))
    args = SimpleNamespace(mode="ppd", model="plumbing", benchmark_data=tmp_path,
        extended_context=True, state_aware=True, output=tmp_path / "run",
        backends=["http://127.0.0.1:8000", "http://127.0.0.1:8001"], transport="nixl", timeout_s=10)
    audit = proxy.create_app(args)
    async with audit.app.router.lifespan_context(audit.app):
        async with real_client(transport=httpx.ASGITransport(audit), base_url="http://proxy") as client:
            async def send(job):
                return await client.post("/v1/chat/completions", json=dict(program_id=job,
                    messages=[], stream=True, return_token_ids=True, max_tokens=1))
            first = asyncio.create_task(send("first"))
            await asyncio.wait_for(first_query.wait(), 2)
            second = asyncio.create_task(send("second"))
            await asyncio.wait_for(both_queries.wait(), 2)
            assert qps_seen[0] == 0 and qps_seen[1] > 0
            for job, task in (("first", first), ("second", second)):
                assert (await client.post("/programs/release", json={"program_id": job})).status_code == 200
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert (await client.get("/health")).json()["outstanding"] == 0
