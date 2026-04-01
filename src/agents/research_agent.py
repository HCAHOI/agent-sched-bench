from __future__ import annotations

import asyncio
import html
import json
import re
import textwrap
import time
import urllib.parse
from typing import Any

import httpx

from agents.base import AgentBase


SYSTEM_PROMPT = """You are a research assistant. Given a question, search
the web for information and synthesize a comprehensive answer.

Use multiple searches when needed. When you are ready to finish, call
synthesize with your complete answer."""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web and return the top results with titles and URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": "Read the full text content of a web page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the page to read.",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "synthesize",
            "description": "Submit your final synthesized answer and finish the research task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The complete synthesized answer.",
                    }
                },
                "required": ["answer"],
            },
        },
    },
]


RESULT_LINK_PATTERN = re.compile(
    r'<a[^>]*class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    flags=re.IGNORECASE | re.DOTALL,
)
TAG_PATTERN = re.compile(r"<[^>]+>")


class ResearchAgent(AgentBase):
    """Multi-step search + synthesis agent with DuckDuckGo-style search tools."""

    def __init__(
        self,
        agent_id: str,
        api_base: str,
        model: str,
        *,
        max_steps: int = 30,
        request_timeout_s: float = 30.0,
        search_rate_limit_qps: float = 1.0,
        search_base_url: str = "https://html.duckduckgo.com/html/",
        max_tool_output_chars: int = 8000,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            api_base=api_base,
            model=model,
            request_timeout_s=request_timeout_s,
        )
        self.max_steps = max_steps
        self.request_timeout_s = request_timeout_s
        self.search_rate_limit_qps = search_rate_limit_qps
        self.search_base_url = search_base_url
        self.max_tool_output_chars = max_tool_output_chars
        self._last_search_ts = 0.0

    def _format_task(self, task: dict[str, Any]) -> str:
        return textwrap.dedent(
            f"""
            Research question:
            {task['question']}
            """
        ).strip()

    def _normalize_result_url(self, raw_url: str) -> str:
        if raw_url.startswith("//"):
            raw_url = f"https:{raw_url}"
        parsed = urllib.parse.urlparse(raw_url)
        query = urllib.parse.parse_qs(parsed.query)
        if "uddg" in query:
            return urllib.parse.unquote(query["uddg"][0])
        return raw_url

    async def _respect_rate_limit(self) -> None:
        if self.search_rate_limit_qps <= 0:
            return
        min_interval = 1.0 / self.search_rate_limit_qps
        elapsed = time.monotonic() - self._last_search_ts
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        self._last_search_ts = time.monotonic()

    def _parse_search_results(self, html_text: str) -> str:
        results = []
        for match in RESULT_LINK_PATTERN.finditer(html_text):
            title = re.sub(r"<.*?>", "", match.group("title"))
            href = self._normalize_result_url(html.unescape(match.group("href").strip()))
            results.append(f"- {html.unescape(title.strip())}: {href}")
            if len(results) == 5:
                break
        return "\n".join(results) or "ERROR: no search results parsed"

    def _extract_page_text(self, html_text: str) -> str:
        try:
            import trafilatura  # type: ignore

            extracted = trafilatura.extract(
                html_text,
                include_comments=False,
                include_tables=False,
            )
            if extracted:
                return extracted
        except ImportError:
            pass
        return TAG_PATTERN.sub(" ", html_text).strip()

    async def _web_search(self, query: str) -> str:
        await self._respect_rate_limit()
        async with httpx.AsyncClient(
            timeout=self.request_timeout_s,
            headers={"user-agent": "Mozilla/5.0"},
        ) as client:
            response = await client.get(self.search_base_url, params={"q": query})
            response.raise_for_status()
        return self._parse_search_results(response.text)

    async def _page_read(self, url: str) -> str:
        async with httpx.AsyncClient(timeout=self.request_timeout_s) as client:
            response = await client.get(url)
            response.raise_for_status()
        extracted = self._extract_page_text(response.text)
        return extracted or "ERROR: failed to extract page text"

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_tool_output_chars:
            return text
        half = self.max_tool_output_chars // 2
        return text[:half] + f"\n[... truncated {len(text) - self.max_tool_output_chars} chars ...]\n" + text[-half:]

    async def run(self, task: dict[str, Any]) -> bool:
        self.task_id = str(task.get("task_id", task.get("question", "")))
        self.task_success = False
        self.trace = []
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._format_task(task)},
        ]

        for step_idx in range(self.max_steps):
            ts_start = time.time()
            llm_result = await self._call_llm(messages, tools=TOOLS)
            ts_end = time.time()
            record = self.build_step_record(
                step_idx=step_idx,
                phase="reasoning",
                llm_result=llm_result,
                ts_start=ts_start,
                ts_end=ts_end,
            )

            if not llm_result.tool_calls:
                record.tool_success = bool(llm_result.content.strip())
                self.task_success = bool(llm_result.content.strip())
                self.trace.append(record)
                break

            tc = llm_result.tool_calls[0]
            try:
                args = json.loads(tc.arguments)
            except json.JSONDecodeError:
                args = {}

            record.phase = "acting"
            record.tool_name = tc.name
            record.tool_args = json.dumps(args)
            tool_started = time.monotonic()

            if tc.name == "web_search":
                try:
                    tool_output = await self._web_search(args.get("query", ""))
                except httpx.HTTPError as exc:
                    tool_output = f"ERROR: {exc}"
            elif tc.name == "read_page":
                try:
                    tool_output = await self._page_read(args.get("url", ""))
                except httpx.HTTPError as exc:
                    tool_output = f"ERROR: {exc}"
            elif tc.name == "synthesize":
                tool_output = args.get("answer", "")
            else:
                tool_output = f"ERROR: unknown tool {tc.name}"

            raw_ms = (time.monotonic() - tool_started) * 1000
            record.tool_duration_ms = raw_ms
            record.tool_result = tool_output
            record.tool_success = not tool_output.startswith("ERROR:")
            self.trace.append(record)

            if tc.name == "synthesize":
                self.task_success = bool(tool_output.strip())
                break

            assistant_msg: dict[str, Any] = {"role": "assistant", "content": llm_result.content or ""}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
            ]
            messages.append(assistant_msg)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": self._truncate(tool_output),
            })
        return bool(self.task_success)
