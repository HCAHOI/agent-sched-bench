"""CPU tests for two-sided routing; synthetic snapshots establish semantics only."""
import json
from types import SimpleNamespace

import httpx
import pytest

from scripts.baselines import ppd_official_proxy as proxy
from scripts.baselines.ppd_policy import (KV_TRANSFER_FLOOR_S, KV_TRANSFER_S_PER_TOKEN,
                                          LOCAL_PREFILL_FLOOR_S, LOCAL_PREFILL_S_PER_TOKEN,
                                          PD_HANDOFF_S, P_PREFILL_S_PER_TOKEN, two_sided_estimate)


def snapshot(cached=0, waiting=0, pending=None):
    """pending defaults to one agent-scale 24K prompt per waiting request."""
    return dict(cached_tokens=cached, running=1, waiting=waiting, max_num_seqs=8,
                pending_prefill_tokens=24000 * waiting if pending is None else pending,
                kv_cache_usage=0.1, timestamp_s=0.0)


def test_estimate_charges_measured_costs_and_queue_on_both_sides():
    idle = two_sided_estimate(600, snapshot(), snapshot())
    assert idle["uncached_p"] == idle["uncached_d"] == 600
    assert idle["local_ttft_s"] == pytest.approx(600 * LOCAL_PREFILL_S_PER_TOKEN + LOCAL_PREFILL_FLOOR_S)
    assert idle["pd_ttft_s"] == pytest.approx(600 * P_PREFILL_S_PER_TOKEN + 600 * KV_TRANSFER_S_PER_TOKEN
                                              + KV_TRANSFER_FLOOR_S + PD_HANDOFF_S)
    assert idle["use_local"]  # An idle decode worker never pays the handoff.

    # Pending prefill tokens on a side are charged at that side's per-token cost.
    queued_d = two_sided_estimate(600, snapshot(), snapshot(pending=48000))
    assert queued_d["local_ttft_s"] - idle["local_ttft_s"] == pytest.approx(48000 * LOCAL_PREFILL_S_PER_TOKEN)
    assert not queued_d["use_local"]
    queued_both = two_sided_estimate(600, snapshot(pending=72000), snapshot(pending=48000))
    assert queued_both["pd_ttft_s"] - idle["pd_ttft_s"] == pytest.approx(72000 * P_PREFILL_S_PER_TOKEN)
    assert queued_both["use_local"]  # A busier P outweighs D's own queue.
    # A warm D queue (little pending work) does not repel a warm request.
    assert two_sided_estimate(24000, snapshot(), snapshot(cached=23000, waiting=5, pending=1500))["use_local"]

    # A prefix already resident on D removes the tokens it would have to prefill,
    # which flips a busy decode worker back to local at agent-scale histories.
    assert not two_sided_estimate(24000, snapshot(), snapshot(waiting=9))["use_local"]
    # Nine cold agent-scale prompts queued on D outweigh even a resident prefix (Milestone 3 §3).
    assert not two_sided_estimate(24000, snapshot(), snapshot(cached=23000, waiting=9))["use_local"]
    with pytest.raises(AssertionError):
        two_sided_estimate(600, snapshot(), snapshot(cached=600))


def test_two_sided_is_exclusive_with_the_lookup_table_extensions(monkeypatch, tmp_path):
    monkeypatch.setattr(proxy.AutoTokenizer, "from_pretrained", lambda *a, **kw: SimpleNamespace())
    args = SimpleNamespace(mode="pd", model="plumbing", benchmark_data=None, output=tmp_path,
                           backends=["http://127.0.0.1:8000", "http://127.0.0.1:8001"],
                           transport="nixl", timeout_s=10, two_sided=True,
                           extended_context=True, state_aware=False)
    with pytest.raises(AssertionError, match="replaces the lookup-table extensions"):
        proxy.create_app(args)
    args.extended_context = args.two_sided = False
    args.state_aware = True
    with pytest.raises(AssertionError, match="same extended table"):
        proxy.create_app(args)


@pytest.mark.asyncio
async def test_turn_one_keeps_public_pd_and_later_turns_follow_the_estimate(monkeypatch, tmp_path):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return json.loads(messages[0]["content"])

    monkeypatch.setattr(proxy.AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    decode_waiting = [0]
    calls = []

    async def backend(request):
        data = json.loads(request.content)
        calls.append((request.url.port, request.url.path))
        if request.url.path == "/ppd/cache_state":
            assert data["prompt_token_ids"]
            if request.url.port == 8000:
                return httpx.Response(200, json=snapshot())
            return httpx.Response(200, json=snapshot(waiting=decode_waiting[0]))
        if request.url.port == 8000:
            return httpx.Response(200, json={"usage": {"completion_tokens": 1},
                "kv_transfer_params": {"do_remote_prefill": True, "remote_block_ids": [3],
                    "remote_request_id": "chatcmpl-" + request.headers["x-request-id"]}})
        count = data["max_tokens"]
        return httpx.Response(200, text="data: " + json.dumps({"choices": [
            {"token_ids": list(range(count)), "delta": {"content": "hi"}}],
            "usage": {"completion_tokens": count}}) + "\n\ndata: [DONE]\n\n")

    real_client = httpx.AsyncClient
    monkeypatch.setattr(proxy.httpx, "AsyncClient", lambda **kw:
                        real_client(transport=httpx.MockTransport(backend), **kw))
    args = SimpleNamespace(mode="pd", model="plumbing", benchmark_data=None, output=tmp_path,
                           backends=["http://127.0.0.1:8000", "http://127.0.0.1:8001"],
                           transport="nixl", timeout_s=10, two_sided=True,
                           extended_context=False, state_aware=False)
    audit = proxy.create_app(args)
    async with audit.app.router.lifespan_context(audit.app):
        async with real_client(transport=httpx.ASGITransport(audit), base_url="http://proxy") as client:
            prompt = list(range(600))
            for turn in (1, 2, 3):
                if turn == 3:
                    decode_waiting[0] = 2
                response = await client.post("/v1/chat/completions", json=dict(
                    program_id="agent", messages=[dict(role="user", content=json.dumps(prompt))],
                    stream=True, return_token_ids=True, max_tokens=2, ignore_eos=True))
                assert response.status_code == 200
                prompt = prompt + [0, 1] + list(range(1000 + turn, 1100 + turn))
    decisions = [json.loads(line) for line in (tmp_path / "routing.jsonl").read_text().splitlines()
                 if json.loads(line)["event"] == "decision"]
    assert [row["routing_mode"] for row in decisions] == ["pd", "local", "pd"]
    assert "two_sided" not in decisions[0] and (8000, "/ppd/cache_state") not in calls[:2]
    for row in decisions[1:]:
        assert row["two_sided_reason"] == "two_sided_expected_ttft"
        assert row["prefill_state"]["waiting"] == 0 and row["two_sided"]["uncached_p"] == row["input_tokens"] + row["context_tokens"]
    assert decisions[1]["decode_state"]["waiting"] == 0 and decisions[2]["decode_state"]["waiting"] == 2
    assert decisions[1]["two_sided"]["local_ttft_s"] < decisions[1]["two_sided"]["pd_ttft_s"]
    assert decisions[2]["two_sided"]["local_ttft_s"] > decisions[2]["two_sided"]["pd_ttft_s"]
    assert [row["state_query_ms"] >= 0 for row in decisions[1:]] == [True, True]
    constants = json.loads((tmp_path / "adapter-config.json").read_text())["two_sided_constants"]
    assert constants["PD_HANDOFF_S"] == PD_HANDOFF_S and "BATCH_BUDGET_TOKENS" not in constants
