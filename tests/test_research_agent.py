from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agents.research_agent import ResearchAgent


class ResearchHandler(BaseHTTPRequestHandler):
    llm_responses = [
        {
            "id": "resp-1",
            "object": "chat.completion",
            "created": 0,
            "model": "mock",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "web_search(test topic)"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        },
        {
            "id": "resp-2",
            "object": "chat.completion",
            "created": 0,
            "model": "mock",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "page_read(http://127.0.0.1:0/page)"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
        },
        {
            "id": "resp-3",
            "object": "chat.completion",
            "created": 0,
            "model": "mock",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Final synthesized answer."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
        },
    ]
    call_count = 0

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/search"):
            body = (
                '<html><body>'
                '<a class="result__a" href="http://127.0.0.1:0/page">Local Article</a>'
                "</body></html>"
            ).replace(":0/", f":{self.server.server_address[1]}/")
        elif self.path.startswith("/page"):
            body = "<html><body><article><p>Useful research content.</p></article></body></html>"
        else:
            body = "{}"
        payload = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers["Content-Length"])
        self.rfile.read(length)
        response = self.llm_responses[self.__class__.call_count]
        if self.__class__.call_count == 1:
            response = json.loads(json.dumps(response).replace("http://127.0.0.1:0/page", f"http://127.0.0.1:{self.server.server_address[1]}/page"))
        self.__class__.call_count += 1
        payload = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


def test_research_agent_search_and_page_read(tmp_path: Path) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), ResearchHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    api_base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    search_base_url = f"http://127.0.0.1:{server.server_address[1]}/search"
    agent = ResearchAgent(
        agent_id="research-1",
        api_base=api_base,
        model="mock",
        search_base_url=search_base_url,
        search_rate_limit_qps=1000.0,
    )
    try:
        success = asyncio.run(
            agent.run(
                {
                    "task_id": "r-1",
                    "question": "What is the test topic?",
                    "reference_answer": "Useful research content.",
                }
            )
        )
        assert success is True
        assert len(agent.trace) == 3
        assert agent.trace[0].tool_name == "web_search"
        assert agent.trace[1].tool_name == "page_read"
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_research_agent_parses_search_results() -> None:
    agent = ResearchAgent(agent_id="research-2", api_base="http://localhost:8000/v1", model="mock")
    parsed = agent._parse_search_results(
        '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdoc">Example Doc</a>'
    )
    assert "Example Doc" in parsed
    assert "https://example.com/doc" in parsed
