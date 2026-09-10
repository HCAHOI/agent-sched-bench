"""Protocol checks only; fake backends are never performance evidence."""
import io
import json
import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import ClientSession, web

from scripts.evaluation.profile_serving_lengths import (
    arrival_offsets, fit_user, read_points, schedule, stream_turn, validate_lengths,
)
from scripts.evaluation import profile_serving_lengths as profiler


def test_declared_points_counts_pairing_and_length_contract():
    points = read_points(Path(__file__).resolve().parents[1] / "analysis/development/serving-profiling-plan.md")
    assert list(points) == [f"P{i:02d}" for i in range(1, 57)]
    groups = schedule(points, "full")
    assert len(groups) == len(set(groups)) == 448
    preliminary = schedule(points, "preliminary")
    assert len(preliminary) == 14 and groups[:14] == preliminary
    assert sum(points[g[0]][2] * 8 for g in groups) == 1191936
    assert sum(points[g[0]][2] * 8 for g in preliminary) == 56784
    assert len(arrival_offsets(0.5, 42)) == 8
    assert arrival_offsets(0.5, 42) == arrival_offsets(0.5, 42)
    assert arrival_offsets(0.5, 43) != arrival_offsets(0.5, 42)
    validate_lengths((107482, 1959, 177), (107480, 1959, 177))
    for actual in ((107482, 1959, 64), (107482, 1958, 177), (107400, 1959, 177)):
        with pytest.raises(AssertionError):
            validate_lengths((107482, 1959, 177), actual)
    with pytest.raises(AssertionError):
        validate_lengths((4096, 500, 32), (4097, 500, 32))

    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(text)

        def decode(self, tokens, **kwargs):
            return "".join(tokens)

        def apply_chat_template(self, messages, **kwargs):
            return list("|".join(m["content"] for m in messages) + "!!")

    messages, tokens = fit_user(Tokenizer(), [], 18, "arbitrary diverse input text")
    assert len(tokens) == 18 and messages[-1]["content"] == "arbitrary divers"


@pytest.mark.asyncio
async def test_stream_complete_output_and_preserved_partial_failure():
    async def completion(request):
        data = await request.json()
        count = data["max_tokens"] if data["seed"] == 0 else data["max_tokens"] - 1
        chunks = [{"id": "chatcmpl-test", "choices": [{"delta": {"content": "hello"},
                   "token_ids": [i]}]} for i in range(count)]
        chunks.append({"id": "chatcmpl-test", "usage": {"completion_tokens": count, "prompt_tokens": 20}})
        text = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"
        return web.Response(text=text, content_type="text/event-stream")

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completion)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    try:
        async with ClientSession() as client:
            raw = io.StringIO()
            payload = dict(max_tokens=1536, seed=0)
            result = await stream_turn(client, f"http://127.0.0.1:{runner.addresses[0][1]}",
                payload, 10, dict(prompt_tokens=20), raw)
            assert result["success"] and len(result["output_token_ids"]) == 1536
            assert result["decode_tpot_s"] == (result["token_chunks"][-1][0] - result["token_chunks"][0][0]) / 1535
            payload["seed"] = 1
            with pytest.raises(AssertionError):
                await stream_turn(client, f"http://127.0.0.1:{runner.addresses[0][1]}",
                    payload, 10, dict(prompt_tokens=20), raw)
            failed = json.loads(raw.getvalue().splitlines()[-1])
            assert not failed["success"] and len(failed["output_token_ids"]) == 1535
            assert failed["e2e_s"] > 0 and "error" in failed
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_return_construction_does_not_block_concurrent_streams(monkeypatch, tmp_path):
    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(map(ord, text))

        def decode(self, tokens, **kwargs):
            return "".join(map(chr, tokens))

        def apply_chat_template(self, messages, **kwargs):
            return self.encode("".join(m["role"] + ":" + m["content"] + "|" for m in messages) + "assistant:")

    tok = Tokenizer()
    main_thread = threading.get_ident()
    construction_windows = []
    original = profiler.prepare_return
    async def no_cleanup(control, *args):
        assert control.connector.force_close
        return 0.0

    def delayed_construction(*args):
        assert threading.get_ident() != main_thread
        began = time.perf_counter()
        time.sleep(0.15)
        result = original(*args)
        construction_windows.append((began, time.perf_counter()))
        return result

    monkeypatch.setattr(profiler, "prepare_return", delayed_construction)
    monkeypatch.setattr(profiler, "clean_group", no_cleanup)
    monkeypatch.setattr(profiler, "verify_group", lambda *args: None)
    monkeypatch.setattr(profiler, "arrival_offsets", lambda *args: [0.0] * 8)
    received = []
    reads_during_construction = []
    original_stream = profiler.stream_turn

    async def observe_stream(*args):
        result = await original_stream(*args)
        reads_during_construction.append((time.perf_counter(), result["turn"]))
        return result

    monkeypatch.setattr(profiler, "stream_turn", observe_stream)

    async def completion(request):
        data = await request.json()
        received.append(data)
        # Other T1 streams remain open while the first conversation constructs T2.
        if data["program_id"].endswith("conv0"):
            delay = 0.001
        else:
            delay = 0.003
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for _ in range(data["max_tokens"]):
            await asyncio.sleep(delay)
            await response.write(("data: " + json.dumps({"id": data["program_id"], "choices": [
                {"delta": {"content": "x"}, "token_ids": [ord("x")]}]}) + "\n\n").encode())
        await response.write(("data: " + json.dumps({"usage": {"completion_tokens": data["max_tokens"],
            "prompt_tokens": len(profiler.prompt_tokens(tok, data["messages"]))}}) + "\n\ndata: [DONE]\n\n").encode())
        await response.write_eof()
        return response

    async def release(request):
        return web.json_response({"released": True})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completion)
    app.router.add_post("/programs/release", release)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    try:
        args = SimpleNamespace(proxy=f"http://127.0.0.1:{runner.addresses[0][1]}",
                               model="plumbing", timeout_s=10, num_layers=36)
        group = await profiler.run_group(args, tok, "P00", (200, 80, 33), 0.5, 42, "local", tmp_path)
        assert group["success"] and len(received) == 16
        assert any(turn == 1 and start < at < end for at, turn in reads_during_construction
                   for start, end in construction_windows)
        rows = [json.loads(l) for l in next(tmp_path.glob("*-turns.jsonl")).read_text().splitlines()]
        assert all((r["actual_h"], r["actual_u"]) == (200, 80) for r in rows if r["turn"] == 2)
    finally:
        await runner.cleanup()


def test_resume_keeps_validated_prefix_and_rejects_gaps(tmp_path):
    plan = tmp_path / "plan.md"
    plan.write_text("frozen plan")
    groups = [("P01", 0.5, 42, "pd"), ("P01", 0.5, 42, "local")]
    (tmp_path / "config.json").write_text(json.dumps(dict(model="model", stage="full", groups=groups)))
    (tmp_path / "continue-full.json").write_text(json.dumps({"continue": True}))
    key = "P01-rate0.5-seed42-pd"
    group = dict(key=key, success=True, t1_count=8, t2_count=8)
    (tmp_path / (key + ".json")).write_text(json.dumps(group))
    raw = tmp_path / (key + "-turns.jsonl")
    rows = [dict(job_id=key + f"-conv{i}", turn=turn, success=True)
            for i in range(8) for turn in (1, 2)]
    raw.write_text("\n".join(map(json.dumps, rows)))
    assert profiler.resume_prefix(tmp_path, "model", plan, groups) == [group]
    rows[-1]["success"] = False
    raw.write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(AssertionError):
        profiler.resume_prefix(tmp_path, "model", plan, groups)
    with pytest.raises(AssertionError):
        profiler.resume_prefix(tmp_path, "different model", plan, groups)
    group["success"] = False
    (tmp_path / (key + ".json")).write_text(json.dumps(group))
    (tmp_path / "P01-rate0.5-seed42-local.json").write_text(json.dumps(dict(success=True)))
    with pytest.raises(AssertionError, match="after a gap"):
        profiler.resume_prefix(tmp_path, "model", plan, groups)
