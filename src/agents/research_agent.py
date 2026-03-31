from __future__ import annotations

import asyncio
import html
import re
import textwrap
import time
import urllib.parse
from typing import Any

import httpx

from agents.base import AgentBase
from agents.tool_calling import ToolCall, parse_tool_call


SYSTEM_PROMPT = """You are a research assistant. Given a question, search
the web for information and synthesize a comprehensive answer.

Available tools:
- web_search(query): Search the web and return concise top results
- page_read(url): Read the full content of a web page

Use multiple searches when needed. When you are ready to finish, answer in
plain text without calling another tool."""


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

    def _parse_tool_call(self, response: str) -> ToolCall | None:
        return parse_tool_call(response, {"web_search", "page_read"})

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

    async def _execute_tool(self, tool_call: ToolCall) -> str:
        if tool_call.name == "web_search":
            try:
                return await self._web_search(tool_call.args)
            except httpx.HTTPError as exc:
                return f"ERROR: {exc}"
        if tool_call.name == "page_read":
            try:
                return await self._page_read(tool_call.args)
            except httpx.HTTPError as exc:
                return f"ERROR: {exc}"
        raise ValueError(f"Unsupported tool: {tool_call.name}")

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
            llm_result = await self._call_llm(messages)
            ts_end = time.time()
            record = self.build_step_record(
                step_idx=step_idx,
                phase="reasoning",
                llm_result=llm_result,
                ts_start=ts_start,
                ts_end=ts_end,
            )
            tool_call = self._parse_tool_call(llm_result.content)
            if tool_call is None:
                record.phase = "reasoning"
                record.tool_success = bool(llm_result.content.strip())
                self.task_success = bool(llm_result.content.strip())
                self.trace.append(record)
                break

            record.phase = "acting"
            record.tool_name = tool_call.name
            record.tool_args = tool_call.args
            tool_started = time.monotonic()
            tool_output = await self._execute_tool(tool_call)
            record.tool_duration_ms = (time.monotonic() - tool_started) * 1000
            record.tool_result = tool_output
            record.tool_success = not tool_output.startswith("ERROR:")
            self.trace.append(record)
            messages.append({"role": "assistant", "content": llm_result.content})
            messages.append({"role": "user", "content": f"Tool output:\n{tool_output}"})
        return bool(self.task_success)
