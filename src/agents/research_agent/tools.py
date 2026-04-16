"""Traced tool wrappers for the research-agent scaffold."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from agents.base import TraceAction
from agents.openclaw.config.schema import WebSearchConfig
from agents.openclaw.tools.web import WebFetchTool, WebSearchTool

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_TIMEOUT_S = 30.0
DEFAULT_FETCH_TIMEOUT_S = 45.0


class TracedWebSearch:
    """Wrap :class:`WebSearchTool` and emit a :class:`TraceAction` per call."""

    def __init__(
        self,
        config: WebSearchConfig | None = None,
        *,
        timeout_s: float = DEFAULT_SEARCH_TIMEOUT_S,
    ) -> None:
        if config is None:
            config = WebSearchConfig(provider="duckduckgo")
        self._tool = WebSearchTool(config=config)
        self._timeout_s = timeout_s

    async def execute(
        self,
        query: str,
        *,
        action_id: str = "tool_search_0",
        agent_id: str = "",
        instance_id: str = "",
        iteration: int = 0,
    ) -> TraceAction:
        ts_start = time.time()
        mono_start = time.monotonic()
        error: str | None = None
        result: str = ""
        try:
            result = await asyncio.wait_for(
                self._tool.execute(query=query),
                timeout=self._timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning("WebSearchTool timed out after %.1fs: %r", self._timeout_s, query)
            error = f"timeout after {self._timeout_s}s"
            result = f"Error: search timeout after {self._timeout_s}s"
        except Exception as exc:
            logger.warning("WebSearchTool raised: %s", exc)
            error = str(exc)
            result = f"Error: {exc}"
        mono_end = time.monotonic()
        ts_end = time.time()
        duration_ms = (mono_end - mono_start) * 1000
        return TraceAction(
            action_type="tool_exec",
            action_id=action_id,
            agent_id=agent_id,
            instance_id=instance_id,
            iteration=iteration,
            ts_start=ts_start,
            ts_end=ts_end,
            data={
                "tool_name": "web_search",
                "args": {"query": query},
                "result": result,
                "duration_ms": duration_ms,
                "error": error,
            },
        )


class TracedWebFetch:
    """Wrap :class:`WebFetchTool` and emit a :class:`TraceAction` per call."""

    def __init__(
        self,
        max_chars: int = 50_000,
        *,
        timeout_s: float = DEFAULT_FETCH_TIMEOUT_S,
    ) -> None:
        self._tool = WebFetchTool(max_chars=max_chars)
        self._timeout_s = timeout_s

    async def execute(
        self,
        url: str,
        *,
        action_id: str = "tool_fetch_0",
        agent_id: str = "",
        instance_id: str = "",
        iteration: int = 0,
    ) -> TraceAction:
        ts_start = time.time()
        mono_start = time.monotonic()
        error: str | None = None
        result: Any = ""
        try:
            result = await asyncio.wait_for(
                self._tool.execute(url=url),
                timeout=self._timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning("WebFetchTool timed out after %.1fs: %s", self._timeout_s, url)
            error = f"timeout after {self._timeout_s}s"
            result = f"Error: fetch timeout after {self._timeout_s}s"
        except Exception as exc:
            logger.warning("WebFetchTool raised: %s", exc)
            error = str(exc)
            result = f"Error: {exc}"
        mono_end = time.monotonic()
        ts_end = time.time()
        duration_ms = (mono_end - mono_start) * 1000
        # Normalise result to string for trace serialisation
        if not isinstance(result, str):
            try:
                result = json.dumps(result, ensure_ascii=False)
            except (TypeError, ValueError):
                result = str(result)
        return TraceAction(
            action_type="tool_exec",
            action_id=action_id,
            agent_id=agent_id,
            instance_id=instance_id,
            iteration=iteration,
            ts_start=ts_start,
            ts_end=ts_end,
            data={
                "tool_name": "web_fetch",
                "args": {"url": url},
                "result": result,
                "duration_ms": duration_ms,
                "error": error,
            },
        )
