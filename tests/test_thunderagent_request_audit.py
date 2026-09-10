import asyncio
import io
import json

import httpx
import pytest

from scripts.baselines.thunderagent_official_launcher import RequestAudit, set_resume_wait_limit
from scripts.evaluation.check_two_instance_request_metrics import check


@pytest.mark.asyncio
async def test_wait_limit_exercises_upstream_resume_and_cancellation():
    upstream = pytest.importorskip("ThunderAgent.scheduler")
    from ThunderAgent.program import ProgramState, ProgramStatus

    router = upstream.MultiBackendRouter(["http://127.0.0.1:8000"])
    try:
        for invalid in (0, -1, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                set_resume_wait_limit(router, invalid)
        set_resume_wait_limit(router, 0.02)
        state = router.get_or_create_program("returning")
        state.backend_url = "http://127.0.0.1:8000"
        backend = router.backends[state.backend_url]
        backend.register_program(state.program_id, state)
        state.status = ProgramStatus.ACTING
        router._pause_program(state.program_id, state)
        state.status = ProgramStatus.REASONING
        await asyncio.wait_for(router._wait_for_resume(state.program_id, state), 1)
        assert state.state == ProgramState.ACTIVE
        assert state.waiting_event is None
        assert state.backend_url == backend.url
        assert not router.global_waiting_queue

        state.status = ProgramStatus.ACTING
        router._pause_program(state.program_id, state)
        waiter = asyncio.create_task(router._wait_for_resume(state.program_id, state))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert state.state == ProgramState.PAUSED
        await router.release_program(state.program_id)
        assert not router.global_waiting_queue
    finally:
        await router.stop()


@pytest.mark.asyncio
async def test_request_mapping_and_deadline_cover_queue_and_stream():
    events = io.StringIO()
    forwarded = []

    async def app(scope, receive, send):
        body = bytearray()
        while True:
            message = await receive()
            body.extend(message.get("body", b""))
            if not message.get("more_body", False):
                break
        payload = json.loads(body)
        if payload.get("queue"):
            await asyncio.sleep(1)
        request = httpx.Request("POST", "http://127.0.0.1:8001/v1/chat/completions")
        await audit.dispatch(request)
        forwarded.append(request.headers["x-request-id"])
        await send(dict(type="http.response.start", status=200, headers=[]))
        await send(dict(type="http.response.body", body=b"token", more_body=True))
        if payload.get("slow_stream"):
            await asyncio.sleep(1)
        await send(dict(type="http.response.body", body=b"done"))

    audit = RequestAudit(app, events, 0.03)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(audit), base_url="http://proxy") as client:
        assert (await client.post("/v1/chat/completions", json={"program_id": "ok"})).content == b"tokendone"
        assert (await client.post("/v1/chat/completions", json={"program_id": "queued", "queue": True})).status_code == 504
        with pytest.raises(TimeoutError):
            await client.post("/v1/chat/completions", json={"program_id": "stream", "slow_stream": True})
    rows = [json.loads(line) for line in events.getvalue().splitlines()]
    finishes = [r for r in rows if r["event"] == "finish"]
    assert [r["outcome"] for r in finishes] == ["complete", "timeout", "timeout"]
    dispatches = [r for r in rows if r["event"] == "dispatch"]
    assert [r["route_id"] for r in dispatches] == forwarded
    assert [r["job_id"] for r in dispatches] == ["ok", "stream"]
    assert all(r["engine_request_id"] == "chatcmpl-" + r["route_id"] for r in rows)


def test_missing_terminal_metrics_stop_completed_request(tmp_path):
    row = dict(route_id="a", engine_request_id="chatcmpl-a", backend="http://127.0.0.1:8000", timestamp_s=0)
    (tmp_path / "routing.jsonl").write_text("".join(json.dumps(dict(row, event=event, outcome="complete")) + "\n"
                                                  for event in ("arrival", "dispatch", "finish")))
    with pytest.raises(AssertionError, match="Missing terminal telemetry"):
        check(tmp_path)
    cell = tmp_path / "instance-0"
    cell.mkdir()
    metrics = dict(request_id="chatcmpl-a", ttft_s=1, queue_s=0, prefill_s=1, decode_s=2,
                   e2e_s=3, preempted_wait_s=0, preemption_timing_complete=True)
    (cell / "vllm-request-telemetry.jsonl").write_text(json.dumps(metrics) + "\n")
    check(tmp_path, final=True)
