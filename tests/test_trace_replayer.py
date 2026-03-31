from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from harness.trace_replayer import TraceReplayer


def write_demo_trace(path: Path) -> None:
    entries = [
        {
            "type": "step",
            "program_id": "agent-1",
            "step_idx": 0,
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "tool_duration_ms": 5.0,
            "ts_start": 1.0,
        },
        {
            "type": "step",
            "program_id": "agent-1",
            "step_idx": 1,
            "prompt_tokens": 8,
            "completion_tokens": 3,
            "tool_duration_ms": 0.0,
            "ts_start": 2.0,
        },
    ]
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")


class ReplayHandler(BaseHTTPRequestHandler):
    call_count = 0
    arrivals: list[float] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        assert payload["program_id"] == "agent-1"
        self.__class__.call_count += 1
        self.__class__.arrivals.append(time.monotonic())
        body = json.dumps(
            {
                "id": f"resp-{self.__class__.call_count}",
                "object": "chat.completion",
                "created": 0,
                "model": "mock",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


def test_trace_replayer_replays_program_sequence(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    write_demo_trace(trace_path)
    ReplayHandler.call_count = 0
    ReplayHandler.arrivals = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), ReplayHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        replayer = TraceReplayer(
            api_base=f"http://127.0.0.1:{server.server_address[1]}/v1",
            model="mock",
            request_timeout_s=10.0,
        )
        results = asyncio.run(replayer.replay(trace_path, concurrency=1))
        assert len(results) == 1
        assert results[0].program_id == "agent-1"
        assert results[0].replayed_steps == 2
        assert ReplayHandler.call_count == 2
        assert ReplayHandler.arrivals[1] - ReplayHandler.arrivals[0] >= 0.004
    finally:
        server.shutdown()
        thread.join(timeout=2)


class MultiReplayHandler(BaseHTTPRequestHandler):
    arrivals: list[tuple[str, float]] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        self.__class__.arrivals.append((payload["program_id"], time.monotonic()))
        body = json.dumps(
            {
                "id": "resp",
                "object": "chat.completion",
                "created": 0,
                "model": "mock",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


def test_trace_replayer_preserves_inter_program_offset(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace_multi.jsonl"
    entries = [
        {"type": "step", "program_id": "agent-1", "step_idx": 0, "prompt_tokens": 1, "completion_tokens": 1, "tool_duration_ms": 0.0, "ts_start": 1.0},
        {"type": "step", "program_id": "agent-2", "step_idx": 0, "prompt_tokens": 1, "completion_tokens": 1, "tool_duration_ms": 0.0, "ts_start": 1.2},
    ]
    trace_path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")
    MultiReplayHandler.arrivals = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), MultiReplayHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        replayer = TraceReplayer(
            api_base=f"http://127.0.0.1:{server.server_address[1]}/v1",
            model="mock",
            request_timeout_s=10.0,
        )
        asyncio.run(replayer.replay(trace_path, concurrency=2))
        arrivals = MultiReplayHandler.arrivals
        arrivals.sort(key=lambda item: item[1])
        assert arrivals[1][1] - arrivals[0][1] >= 0.10
    finally:
        server.shutdown()
        thread.join(timeout=2)
