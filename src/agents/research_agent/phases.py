"""Five-phase pipeline for the research-agent scaffold."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from agents.base import TraceAction, _message_content_to_text
from agents.research_agent.evidence import Evidence
from agents.research_agent.tools import TracedWebFetch, TracedWebSearch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dump_obj(obj: Any) -> Any:
    """Convert Pydantic models / dataclass-like objects to dicts."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return obj


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    """getattr / dict.get hybrid."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


async def _call_streaming(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Streaming LLM call that records timing metrics.

    Returns a dict with keys: content, prompt_tokens, completion_tokens,
    llm_wall_latency_ms, ttft_ms, tpot_ms, raw_response, ts_start, ts_end.
    """
    ts_start = time.time()
    mono_start = time.monotonic()
    first_token_mono: float | None = None
    content_parts: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason: str | None = None

    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
    )
    async for chunk in stream:
        if not chunk.choices and hasattr(chunk, "usage") and chunk.usage is not None:
            usage = _dump_obj(chunk.usage)
            continue
        for choice in (chunk.choices or []):
            delta = choice.delta
            if delta is not None:
                text = _message_content_to_text(_get_field(delta, "content"))
                if text:
                    if first_token_mono is None:
                        first_token_mono = time.monotonic()
                    content_parts.append(text)
            if choice.finish_reason:
                finish_reason = choice.finish_reason

    mono_end = time.monotonic()
    ts_end = time.time()
    elapsed_ms = (mono_end - mono_start) * 1000

    prompt_tokens = int(_get_field(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(_get_field(usage, "completion_tokens", 0) or 0)

    ttft_ms: float | None = None
    tpot_ms: float | None = None
    if first_token_mono is not None:
        ttft_ms = (first_token_mono - mono_start) * 1000
        if completion_tokens > 1:
            tpot_ms = max(0.0, (elapsed_ms - ttft_ms) / (completion_tokens - 1))

    return {
        "content": "".join(content_parts),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "llm_wall_latency_ms": elapsed_ms,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "finish_reason": finish_reason,
        "raw_response": {},
        "ts_start": ts_start,
        "ts_end": ts_end,
    }


def _build_llm_action(
    result: dict[str, Any],
    *,
    action_id: str,
    agent_id: str,
    instance_id: str,
    iteration: int,
    messages_in: list[dict[str, Any]],
) -> TraceAction:
    return TraceAction(
        action_type="llm_call",
        action_id=action_id,
        agent_id=agent_id,
        instance_id=instance_id,
        iteration=iteration,
        ts_start=result["ts_start"],
        ts_end=result["ts_end"],
        data={
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "llm_wall_latency_ms": result["llm_wall_latency_ms"],
            "llm_latency_ms": result["llm_wall_latency_ms"],
            "ttft_ms": result["ttft_ms"],
            "tpot_ms": result["tpot_ms"],
            "finish_reason": result["finish_reason"],
            "messages_in": messages_in,
            "content": result["content"],
            "raw_response": result["raw_response"],
        },
    )


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


class PlanPhase:
    """Generate search queries from the problem statement (iteration 0)."""

    ITERATION = 0
    NAME = "plan"

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        agent_id: str = "",
        instance_id: str = "",
        max_queries: int = 5,
    ) -> None:
        self.client = client
        self.model = model
        self.agent_id = agent_id
        self.instance_id = instance_id
        self.max_queries = max_queries

    async def execute(
        self,
        task_prompt: str,
    ) -> tuple[list[str], list[TraceAction]]:
        system_msg = (
            f"Generate at most {self.max_queries} diverse search queries to "
            "research this topic. Output one query per line, no numbering or "
            "bullet points, no extra commentary."
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": task_prompt},
        ]

        result = await _call_streaming(self.client, self.model, messages)
        content = result["content"]
        queries = [q.strip() for q in content.split("\n") if q.strip()]
        # Enforce hard cap to avoid blowing up downstream concurrency
        queries = queries[: self.max_queries]

        action = _build_llm_action(
            result,
            action_id="llm_plan_0",
            agent_id=self.agent_id,
            instance_id=self.instance_id,
            iteration=self.ITERATION,
            messages_in=messages,
        )
        return queries, [action]


class SearchPhase:
    """Run search queries concurrently (iteration 1).

    Concurrency is bounded by ``max_concurrent`` to avoid overwhelming
    search providers like DuckDuckGo which misbehave under high parallelism.
    """

    ITERATION = 1
    NAME = "search"

    def __init__(
        self,
        search_tool: TracedWebSearch,
        *,
        agent_id: str = "",
        instance_id: str = "",
        max_concurrent: int = 1,
    ) -> None:
        self.search_tool = search_tool
        self.agent_id = agent_id
        self.instance_id = instance_id
        # NOTE: default 1 because curl_cffi (used by ddgs) deadlocks on
        # macOS GCD semaphores when multiple DuckDuckGo calls run in parallel.
        self.max_concurrent = max_concurrent

    async def execute(
        self,
        queries: list[str],
    ) -> tuple[list[dict[str, Any]], list[TraceAction]]:
        if not queries:
            return [], []

        sem = asyncio.Semaphore(self.max_concurrent)

        async def _one(idx: int, query: str) -> TraceAction:
            async with sem:
                return await self.search_tool.execute(
                    query,
                    action_id=f"tool_search_{idx}",
                    agent_id=self.agent_id,
                    instance_id=self.instance_id,
                    iteration=self.ITERATION,
                )

        actions = list(
            await asyncio.gather(*[_one(i, q) for i, q in enumerate(queries)])
        )

        search_results: list[dict[str, Any]] = []
        for action in actions:
            result_text = action.data.get("result", "")
            search_results.append({
                "query": action.data.get("args", {}).get("query", ""),
                "result": result_text,
                "error": action.data.get("error"),
            })
        return search_results, actions


class FetchPhase:
    """Fetch top-K URLs from search results (iteration 2)."""

    ITERATION = 2
    NAME = "fetch"

    def __init__(
        self,
        fetch_tool: TracedWebFetch,
        *,
        agent_id: str = "",
        instance_id: str = "",
        top_k: int = 5,
        max_concurrent: int = 3,
    ) -> None:
        self.fetch_tool = fetch_tool
        self.agent_id = agent_id
        self.instance_id = instance_id
        self.top_k = top_k
        self.max_concurrent = max_concurrent

    @staticmethod
    def extract_urls(search_results: list[dict[str, Any]]) -> list[str]:
        """Extract URLs from search result text."""
        import re

        urls: list[str] = []
        seen: set[str] = set()
        for sr in search_results:
            text = sr.get("result", "")
            # Match URLs in format "   http(s)://..."
            for match in re.finditer(r"https?://[^\s)\"']+", text):
                url = match.group(0).rstrip(".,;:")
                if url not in seen:
                    seen.add(url)
                    urls.append(url)
        return urls

    async def execute(
        self,
        urls: list[str],
    ) -> tuple[list[dict[str, Any]], list[TraceAction]]:
        if not urls:
            return [], []

        target_urls = urls[: self.top_k]
        sem = asyncio.Semaphore(self.max_concurrent)

        async def _one(idx: int, url: str) -> TraceAction:
            async with sem:
                return await self.fetch_tool.execute(
                    url,
                    action_id=f"tool_fetch_{idx}",
                    agent_id=self.agent_id,
                    instance_id=self.instance_id,
                    iteration=self.ITERATION,
                )

        actions = list(
            await asyncio.gather(*[_one(i, u) for i, u in enumerate(target_urls)])
        )

        fetched_pages: list[dict[str, Any]] = []
        for i, action in enumerate(actions):
            raw = action.data.get("result", "")
            page: dict[str, Any] = {"url": target_urls[i], "raw": raw}
            # Try to parse JSON result from WebFetchTool
            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                    page["text"] = parsed.get("text", raw)
                    page["status"] = parsed.get("status")
                except (json.JSONDecodeError, AttributeError):
                    page["text"] = raw
            else:
                page["text"] = str(raw)
            page["error"] = action.data.get("error")
            page["fetch_timestamp"] = action.ts_start
            fetched_pages.append(page)
        return fetched_pages, actions


class ExtractPhase:
    """Extract evidence from fetched pages (iteration 3)."""

    ITERATION = 3
    NAME = "extract"

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        agent_id: str = "",
        instance_id: str = "",
    ) -> None:
        self.client = client
        self.model = model
        self.agent_id = agent_id
        self.instance_id = instance_id

    async def execute(
        self,
        task_prompt: str,
        fetched_pages: list[dict[str, Any]],
    ) -> tuple[list[Evidence], list[TraceAction]]:
        if not fetched_pages:
            return [], []

        sources_text = ""
        for i, page in enumerate(fetched_pages):
            text = page.get("text", "")
            url = page.get("url", "unknown")
            # Truncate per-page to keep prompt manageable
            if len(text) > 8000:
                text = text[:8000] + "\n[...truncated...]"
            sources_text += f"\n--- Source {i + 1}: {url} ---\n{text}\n"

        system_msg = "Extract relevant evidence from the provided sources."
        user_content = (
            f"Task:\n{task_prompt}\n\nSources:\n{sources_text}\n\n"
            "For each relevant piece of evidence, output one JSON object per line "
            'with keys: "source_url", "passage", "relevance_note".'
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]

        result = await _call_streaming(self.client, self.model, messages)
        content = result["content"]

        evidence_list: list[Evidence] = []
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                evidence_list.append(Evidence(
                    source_url=str(obj.get("source_url", "")),
                    passage=str(obj.get("passage", "")),
                    relevance_note=str(obj.get("relevance_note", "")),
                    search_query="",
                    fetch_timestamp=0.0,
                ))
            except (json.JSONDecodeError, TypeError):
                continue

        # Enrich evidence with fetch timestamps
        url_to_ts: dict[str, float] = {}
        for page in fetched_pages:
            url_to_ts[page.get("url", "")] = page.get("fetch_timestamp", 0.0)
        for ev in evidence_list:
            ev.fetch_timestamp = url_to_ts.get(ev.source_url, 0.0)

        action = _build_llm_action(
            result,
            action_id="llm_extract_0",
            agent_id=self.agent_id,
            instance_id=self.instance_id,
            iteration=self.ITERATION,
            messages_in=messages,
        )
        return evidence_list, [action]


class SynthesizePhase:
    """Produce a final answer from evidence (iteration 4)."""

    ITERATION = 4
    NAME = "synthesize"

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        agent_id: str = "",
        instance_id: str = "",
    ) -> None:
        self.client = client
        self.model = model
        self.agent_id = agent_id
        self.instance_id = instance_id

    async def execute(
        self,
        task_prompt: str,
        evidence: list[Evidence],
    ) -> tuple[str, list[TraceAction]]:
        system_msg = "Synthesize a final answer based on the evidence."

        evidence_text = ""
        if evidence:
            for i, ev in enumerate(evidence):
                evidence_text += (
                    f"\n[{i + 1}] Source: {ev.source_url}\n"
                    f"    Passage: {ev.passage}\n"
                    f"    Relevance: {ev.relevance_note}\n"
                )
        else:
            evidence_text = "\n(No evidence was collected.)\n"

        user_content = (
            f"Task:\n{task_prompt}\n\n"
            f"Evidence:{evidence_text}\n\n"
            "Provide a concise, well-supported final answer."
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]

        result = await _call_streaming(self.client, self.model, messages)

        action = _build_llm_action(
            result,
            action_id="llm_synthesize_0",
            agent_id=self.agent_id,
            instance_id=self.instance_id,
            iteration=self.ITERATION,
            messages_in=messages,
        )
        return result["content"], [action]
