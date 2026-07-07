"""Fail-closed web tools for the deep-research scaffold.

These wrappers expose the OpenClaw-compatible ``web_search``/``web_fetch`` tool
schemas while enforcing BrowseComp's configured providers, explicit fallbacks,
and per-attempt budgets. They intentionally do not call OpenClaw's public
``execute`` methods because those methods silently fall back between providers.
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Mapping

import httpx

from agents.openclaw.config.schema import WebSearchConfig
from agents.openclaw.security.network import validate_url_target
from agents.openclaw.tools.base import Tool
from agents.openclaw.tools.web import WebFetchTool, WebSearchTool, _UNTRUSTED_BANNER


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    """AgentRunner metadata for the tool call currently executing."""

    iteration: int
    tool_call_id: str


_TOOL_EXECUTION_CONTEXT: ContextVar[ToolExecutionContext | None] = ContextVar(
    "deep_research_tool_execution_context",
    default=None,
)


class _ExecutionContextMixin:
    def set_execution_context(self, *, iteration: int, tool_call_id: str) -> Token:
        return _TOOL_EXECUTION_CONTEXT.set(
            ToolExecutionContext(iteration=iteration, tool_call_id=tool_call_id)
        )

    def reset_execution_context(self, token: Token) -> None:
        _TOOL_EXECUTION_CONTEXT.reset(token)


@dataclass(slots=True)
class ToolExecutionRecord:
    """One executed web-tool call, including backend provenance."""

    tool_name: str
    tool_args: dict[str, Any]
    tool_result: Any
    requested_provider: str
    actual_provider: str | None
    fallback_used: bool
    error: str | None
    budget_exhausted: bool
    ts_start: float
    ts_end: float
    iteration: int | None = None
    tool_call_id: str | None = None
    artifact_path: str | None = None
    artifact_sha256: str | None = None
    original_size: int | None = None
    preview: str | None = None
    truncated_preview: bool = False

    def to_trace_payload(self) -> dict[str, Any]:
        payload = {
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "iteration": self.iteration,
            "tool_call_id": self.tool_call_id,
            "tool_result": self.tool_result,
            "requested_provider": self.requested_provider,
            "actual_provider": self.actual_provider,
            "fallback_used": self.fallback_used,
            "error": self.error,
            "budget_exhausted": self.budget_exhausted,
        }
        if self.artifact_path is not None:
            payload.update(
                {
                    "artifact_path": self.artifact_path,
                    "artifact_sha256": self.artifact_sha256,
                    "original_size": self.original_size,
                    "preview": self.preview,
                    "truncated_preview": self.truncated_preview,
                }
            )
        return payload


@dataclass(slots=True)
class WebToolRuntimeState:
    """Shared per-attempt counters and tool-call trace records."""

    max_search_calls: int
    max_fetch_calls: int
    records: list[ToolExecutionRecord] = field(default_factory=list)
    search_calls_used: int = 0
    fetch_calls_used: int = 0
    tool_budget_exhausted: bool = False
    budget_exhausted_tool: str | None = None
    tool_backend_failed: bool = False
    backend_failed_tool: str | None = None
    backend_failure_error: str | None = None

    def check_budget(self, tool_name: str) -> bool:
        if tool_name == "web_search":
            if self.search_calls_used >= self.max_search_calls:
                self.tool_budget_exhausted = True
                self.budget_exhausted_tool = tool_name
                return False
            self.search_calls_used += 1
            return True
        if tool_name == "web_fetch":
            if self.fetch_calls_used >= self.max_fetch_calls:
                self.tool_budget_exhausted = True
                self.budget_exhausted_tool = tool_name
                return False
            self.fetch_calls_used += 1
            return True
        raise ValueError(f"unknown web tool: {tool_name}")

    def mark_backend_failed(self, tool_name: str, error: str | None) -> None:
        self.tool_backend_failed = True
        self.backend_failed_tool = tool_name
        self.backend_failure_error = error or f"{tool_name} backend failed"


class BudgetExhaustedResult:
    """Tool result that asks AgentRunner to yield immediately."""

    should_yield = True

    def __init__(self, content: str) -> None:
        self.content = content


class BackendFailureResult:
    """Tool result that stops the attempt when configured backends all fail."""

    should_yield = True

    def __init__(self, content: str) -> None:
        self.content = content


@dataclass(frozen=True, slots=True)
class DeepResearchWebConfig:
    """Provider and budget settings for deep-research web tools."""

    search_provider: str
    search_api_key_env: str | None
    search_fallback_provider: str | None
    fetch_provider: str
    fetch_api_key_env: str | None
    fetch_fallback_provider: str | None
    max_search_calls: int
    max_fetch_calls: int
    search_base_url: str | None = None
    fetch_max_chars: int = 50_000

    @classmethod
    def from_extras(cls, extras: Mapping[str, Any]) -> "DeepResearchWebConfig":
        return cls(
            search_provider=str(extras.get("web_search_provider", "")).strip().lower(),
            search_api_key_env=_optional_env_name(extras.get("web_search_api_key_env")),
            search_fallback_provider=_optional_provider(
                extras.get("web_search_fallback_provider")
            ),
            fetch_provider=str(extras.get("web_fetch_provider", "")).strip().lower(),
            fetch_api_key_env=_optional_env_name(extras.get("web_fetch_api_key_env")),
            fetch_fallback_provider=_optional_provider(
                extras.get("web_fetch_fallback_provider")
            ),
            max_search_calls=int(extras.get("max_search_calls", 0)),
            max_fetch_calls=int(extras.get("max_fetch_calls", 0)),
            search_base_url=_optional_text(extras.get("web_search_base_url")),
            fetch_max_chars=int(extras.get("web_fetch_max_chars", 50_000)),
        )

    def validate(self, environ: Mapping[str, str] | None = None) -> None:
        env = environ if environ is not None else os.environ
        if self.max_search_calls <= 0:
            raise ValueError("max_search_calls must be positive")
        if self.max_fetch_calls <= 0:
            raise ValueError("max_fetch_calls must be positive")
        _validate_search_backend(
            self.search_provider,
            api_key_env=self.search_api_key_env,
            base_url=self.search_base_url,
            environ=env,
            role="web_search_provider",
        )
        if self.search_fallback_provider is not None:
            _validate_search_backend(
                self.search_fallback_provider,
                api_key_env=self.search_api_key_env,
                base_url=self.search_base_url,
                environ=env,
                role="web_search_fallback_provider",
            )
        _validate_fetch_backend(
            self.fetch_provider,
            api_key_env=self.fetch_api_key_env,
            environ=env,
            role="web_fetch_provider",
        )
        if self.fetch_fallback_provider is not None:
            _validate_fetch_backend(
                self.fetch_fallback_provider,
                api_key_env=self.fetch_api_key_env,
                environ=env,
                role="web_fetch_fallback_provider",
            )


class DeepResearchWebSearchTool(_ExecutionContextMixin, Tool):
    """Configured, budgeted web-search wrapper."""

    name = "web_search"
    description = WebSearchTool.description
    parameters = WebSearchTool.parameters

    def __init__(
        self,
        *,
        config: DeepResearchWebConfig,
        state: WebToolRuntimeState,
        environ: Mapping[str, str] | None = None,
        proxy: str | None = None,
    ) -> None:
        self.config = config
        self.state = state
        self.environ = environ if environ is not None else os.environ
        self.proxy = proxy
        self.config.validate(self.environ)

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, query: str, count: int | None = None, **kwargs: Any) -> Any:
        n = min(max(count or 5, 1), 10)
        args = {"query": query, "count": n}
        started = time.time()
        if not self.state.check_budget(self.name):
            content = "Error: web_search tool budget exhausted"
            self._record(
                args,
                content,
                requested_provider=self.config.search_provider,
                actual_provider=None,
                fallback_used=False,
                error="tool_budget_exhausted",
                budget_exhausted=True,
                ts_start=started,
            )
            return BudgetExhaustedResult(content)

        (
            result,
            actual_provider,
            fallback_used,
            error,
            backend_failed,
        ) = await self._run_with_fallback(query, n)
        self._record(
            args,
            result,
            requested_provider=self.config.search_provider,
            actual_provider=actual_provider,
            fallback_used=fallback_used,
            error=error,
            budget_exhausted=False,
            ts_start=started,
        )
        if backend_failed:
            self.state.mark_backend_failed(self.name, error)
            return BackendFailureResult(_result_content(result))
        return result

    async def _run_with_fallback(
        self, query: str, n: int
    ) -> tuple[str, str | None, bool, str | None, bool]:
        result = await self._run_provider(self.config.search_provider, query, n)
        if not _is_error_result(result):
            return result, self.config.search_provider, False, None, False
        if self.config.search_fallback_provider is None:
            return result, self.config.search_provider, False, _error_text(result), True
        fallback_result = await self._run_provider(
            self.config.search_fallback_provider, query, n
        )
        if _is_error_result(fallback_result):
            return (
                fallback_result,
                self.config.search_fallback_provider,
                True,
                _error_text(fallback_result),
                True,
            )
        return fallback_result, self.config.search_fallback_provider, True, None, False

    async def _run_provider(self, provider: str, query: str, n: int) -> str:
        tool = WebSearchTool(
            WebSearchConfig(
                provider=provider,
                api_key=_api_key(self.config.search_api_key_env, self.environ),
                base_url=self.config.search_base_url or "",
                max_results=n,
            ),
            proxy=self.proxy,
        )
        if provider == "brave":
            return await tool._search_brave(query, n)
        if provider == "tavily":
            return await tool._search_tavily(query, n)
        if provider == "searxng":
            return await tool._search_searxng(query, n)
        if provider == "jina":
            return await tool._search_jina(query, n)
        if provider == "duckduckgo":
            return await tool._search_duckduckgo(query, n)
        raise ValueError(f"unsupported search provider: {provider}")

    def _record(
        self,
        args: dict[str, Any],
        result: Any,
        *,
        requested_provider: str,
        actual_provider: str | None,
        fallback_used: bool,
        error: str | None,
        budget_exhausted: bool,
        ts_start: float,
    ) -> None:
        execution_context = _TOOL_EXECUTION_CONTEXT.get()
        self.state.records.append(
            ToolExecutionRecord(
                tool_name=self.name,
                tool_args=args,
                tool_result=result,
                requested_provider=requested_provider,
                actual_provider=actual_provider,
                fallback_used=fallback_used,
                error=error,
                budget_exhausted=budget_exhausted,
                ts_start=ts_start,
                ts_end=time.time(),
                iteration=execution_context.iteration if execution_context else None,
                tool_call_id=execution_context.tool_call_id
                if execution_context
                else None,
            )
        )


class DeepResearchWebFetchTool(_ExecutionContextMixin, Tool):
    """Configured, budgeted web-fetch wrapper."""

    name = "web_fetch"
    description = WebFetchTool.description
    parameters = WebFetchTool.parameters

    def __init__(
        self,
        *,
        config: DeepResearchWebConfig,
        state: WebToolRuntimeState,
        environ: Mapping[str, str] | None = None,
        proxy: str | None = None,
    ) -> None:
        self.config = config
        self.state = state
        self.environ = environ if environ is not None else os.environ
        self.proxy = proxy
        self.config.validate(self.environ)

    @property
    def read_only(self) -> bool:
        return True

    async def execute(
        self,
        url: str,
        extractMode: str = "markdown",
        maxChars: int | None = None,
        **kwargs: Any,
    ) -> Any:
        max_chars = maxChars or self.config.fetch_max_chars
        args = {"url": url, "extractMode": extractMode, "maxChars": max_chars}
        started = time.time()
        if not self.state.check_budget(self.name):
            content = "Error: web_fetch tool budget exhausted"
            self._record(
                args,
                content,
                requested_provider=self.config.fetch_provider,
                actual_provider=None,
                fallback_used=False,
                error="tool_budget_exhausted",
                budget_exhausted=True,
                ts_start=started,
            )
            return BudgetExhaustedResult(content)

        (
            result,
            actual_provider,
            fallback_used,
            error,
            backend_failed,
        ) = await self._run_with_fallback(url, extractMode, max_chars)
        self._record(
            args,
            result,
            requested_provider=self.config.fetch_provider,
            actual_provider=actual_provider,
            fallback_used=fallback_used,
            error=error,
            budget_exhausted=False,
            ts_start=started,
        )
        if backend_failed:
            self.state.mark_backend_failed(self.name, error)
            return BackendFailureResult(_result_content(result))
        return result

    async def _run_with_fallback(
        self, url: str, extract_mode: str, max_chars: int
    ) -> tuple[Any, str | None, bool, str | None, bool]:
        result = await self._run_provider(
            self.config.fetch_provider, url, extract_mode, max_chars
        )
        if _fetch_ok(result):
            return result, self.config.fetch_provider, False, None, False
        error = _error_text(result)
        if self.config.fetch_fallback_provider is None:
            return (
                result,
                self.config.fetch_provider,
                False,
                error,
                _is_fetch_backend_failure(result),
            )
        fallback_result = await self._run_provider(
            self.config.fetch_fallback_provider, url, extract_mode, max_chars
        )
        if not _fetch_ok(fallback_result):
            fallback_error = _error_text(fallback_result)
            return (
                fallback_result,
                self.config.fetch_fallback_provider,
                True,
                fallback_error,
                _is_fetch_backend_failure(fallback_result),
            )
        return fallback_result, self.config.fetch_fallback_provider, True, None, False

    async def _run_provider(
        self, provider: str, url: str, extract_mode: str, max_chars: int
    ) -> Any:
        is_valid, error_msg = validate_url_target(url)
        if not is_valid:
            return json.dumps(
                {"error": f"URL validation failed: {error_msg}", "url": url},
                ensure_ascii=False,
            )
        if provider == "jina":
            return await _fetch_jina_reader(
                url,
                max_chars,
                self.proxy,
                _api_key(self.config.fetch_api_key_env, self.environ),
            )
        if provider == "readability":
            tool = WebFetchTool(max_chars=max_chars, proxy=self.proxy)
            return await tool._fetch_readability(url, extract_mode, max_chars)
        raise ValueError(f"unsupported fetch provider: {provider}")

    def _record(
        self,
        args: dict[str, Any],
        result: Any,
        *,
        requested_provider: str,
        actual_provider: str | None,
        fallback_used: bool,
        error: str | None,
        budget_exhausted: bool,
        ts_start: float,
    ) -> None:
        execution_context = _TOOL_EXECUTION_CONTEXT.get()
        self.state.records.append(
            ToolExecutionRecord(
                tool_name=self.name,
                tool_args=args,
                tool_result=result,
                requested_provider=requested_provider,
                actual_provider=actual_provider,
                fallback_used=fallback_used,
                error=error,
                budget_exhausted=budget_exhausted,
                ts_start=ts_start,
                ts_end=time.time(),
                iteration=execution_context.iteration if execution_context else None,
                tool_call_id=execution_context.tool_call_id
                if execution_context
                else None,
            )
        )


async def _fetch_jina_reader(
    url: str,
    max_chars: int,
    proxy: str | None,
    api_key: str,
) -> str:
    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/537.36",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with httpx.AsyncClient(proxy=proxy, timeout=20.0) as client:
            response = await client.get(f"https://r.jina.ai/{url}", headers=headers)
            response.raise_for_status()
    except Exception as exc:
        return json.dumps(
            {"error": f"Jina Reader failed: {exc}", "url": url},
            ensure_ascii=False,
        )

    try:
        data = response.json().get("data", {})
    except ValueError as exc:
        return json.dumps(
            {"error": f"Jina Reader returned malformed JSON: {exc}", "url": url},
            ensure_ascii=False,
        )
    title = data.get("title", "")
    text = data.get("content", "")
    if not text:
        return json.dumps(
            {"error": "Jina Reader returned no readable content", "url": url},
            ensure_ascii=False,
        )
    if title:
        text = f"# {title}\n\n{text}"
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars]
    text = f"{_UNTRUSTED_BANNER}\n\n{text}"
    return json.dumps(
        {
            "url": url,
            "finalUrl": data.get("url", url),
            "status": response.status_code,
            "extractor": "jina",
            "truncated": truncated,
            "length": len(text),
            "untrusted": True,
            "text": text,
        },
        ensure_ascii=False,
    )


def _optional_text(value: Any) -> str | None:
    text = "" if value is None else str(value).strip()
    return text or None


def _optional_env_name(value: Any) -> str | None:
    return _optional_text(value)


def _optional_provider(value: Any) -> str | None:
    text = _optional_text(value)
    return text.lower() if text else None


def _api_key(env_name: str | None, environ: Mapping[str, str]) -> str:
    return environ.get(env_name, "") if env_name else ""


def _result_content(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


def _require_import(module_name: str, package_name: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise ValueError(
            f"required dependency {package_name!r} is not installed for deep-research web backend"
        )


def _validate_search_backend(
    provider: str,
    *,
    api_key_env: str | None,
    base_url: str | None,
    environ: Mapping[str, str],
    role: str,
) -> None:
    if provider not in {"brave", "tavily", "searxng", "jina", "duckduckgo"}:
        raise ValueError(f"unsupported {role}: {provider!r}")
    if provider == "duckduckgo":
        _require_import("ddgs", "ddgs")
        return
    if provider == "searxng":
        if not base_url:
            raise ValueError(f"{role}=searxng requires web_search_base_url")
        return
    if not api_key_env:
        raise ValueError(f"{role}={provider} requires an API-key env setting")
    if not environ.get(api_key_env):
        raise ValueError(
            f"{role}={provider} requires environment variable {api_key_env}"
        )


def _validate_fetch_backend(
    provider: str,
    *,
    api_key_env: str | None,
    environ: Mapping[str, str],
    role: str,
) -> None:
    if provider not in {"jina", "readability"}:
        raise ValueError(f"unsupported {role}: {provider!r}")
    if provider == "readability":
        _require_import("readability", "readability-lxml")
        return
    if provider == "jina":
        return


def _is_error_result(result: Any) -> bool:
    return isinstance(result, str) and result.strip().lower().startswith("error")


def _fetch_ok(result: Any) -> bool:
    if _is_error_result(result):
        return False
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            return True
        return "error" not in parsed
    return True


def _is_fetch_backend_failure(result: Any) -> bool:
    error = _error_text(result)
    if error is None:
        return False
    lower = error.lower()
    recoverable_markers = (
        "url validation failed",
        "redirect blocked",
        "no readable content",
        "400 bad request",
        "401 unauthorized",
        "403 forbidden",
        "404 not found",
        "410 gone",
        "451",
    )
    if any(marker in lower for marker in recoverable_markers):
        return False
    backend_markers = (
        "jina reader failed",
        "webfetch error",
        "proxy error",
        "timeout",
        "timed out",
        "connect",
        "connection",
        "network",
        "rate limit",
        "429",
        "500",
        "502",
        "503",
        "504",
        "malformed json",
    )
    return any(marker in lower for marker in backend_markers)


def _error_text(result: Any) -> str | None:
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            return result if _is_error_result(result) else None
        error = parsed.get("error") if isinstance(parsed, dict) else None
        return str(error) if error is not None else None
    return None
