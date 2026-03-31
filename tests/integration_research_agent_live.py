from __future__ import annotations

import asyncio
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agents.research_agent import ResearchAgent


class LiveResearchLLMHandler(BaseHTTPRequestHandler):
    call_count = 0

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        latest_user_message = (payload.get("messages") or [{}])[-1].get("content", "")
        match = re.search(r"https://\S+", latest_user_message)
        if self.__class__.call_count in {1, 3} and match is None:
            raise AssertionError("Expected a normalized https:// URL in the latest tool output")
        extracted_url = match.group(0) if match else ""

        if self.__class__.call_count == 0:
            content = "web_search(openai research)"
        elif self.__class__.call_count == 1:
            content = f"page_read({extracted_url})"
        elif self.__class__.call_count == 2:
            content = "web_search(openai deep research)"
        elif self.__class__.call_count == 3:
            content = f"page_read({extracted_url})"
        else:
            content = "Final synthesized answer from live web data."

        response = {
            "id": f"resp-{self.__class__.call_count + 1}",
            "object": "chat.completion",
            "created": 0,
            "model": "mock",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        }
        self.__class__.call_count += 1
        body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


def test_research_agent_live_web_path() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), LiveResearchLLMHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    api_base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    agent = ResearchAgent(
        agent_id="research-live",
        api_base=api_base,
        model="mock",
        search_rate_limit_qps=1000.0,
    )
    try:
        success = asyncio.run(
            agent.run(
                {
                    "task_id": "r-live",
                    "question": "Summarize recent OpenAI research pages.",
                    "reference_answer": "offline only",
                }
            )
        )
        assert success is True
        assert len(agent.trace) >= 5
        web_search_records = [record for record in agent.trace if record.tool_name == "web_search"]
        page_read_records = [record for record in agent.trace if record.tool_name == "page_read"]
        assert len(web_search_records) >= 2
        assert all(record.tool_success is True for record in web_search_records)
        assert all("https://" in (record.tool_result or "") for record in web_search_records)
        assert len(page_read_records) >= 2
    finally:
        server.shutdown()
        thread.join(timeout=2)
