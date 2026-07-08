from __future__ import annotations

import asyncio
import json
import sys
import types

import pytest

from agents.deep_research.web_tools import (
    BackendFailureResult,
    BudgetExhaustedResult,
    DeepResearchWebConfig,
    DeepResearchWebFetchTool,
    DeepResearchWebSearchTool,
    WebToolRuntimeState,
)


def _install_failing_ddgs(monkeypatch, error: Exception) -> None:
    ddgs_module = types.ModuleType("ddgs")

    class FakeDDGS:
        def __init__(self, *, timeout: int) -> None:
            self.timeout = timeout

        def text(self, query: str, *, max_results: int):
            del query, max_results
            raise error

    ddgs_module.DDGS = FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", ddgs_module)
    monkeypatch.setattr(
        "agents.deep_research.web_tools.importlib.util.find_spec",
        lambda name: object() if name == "ddgs" else None,
    )


def _config(**overrides) -> DeepResearchWebConfig:
    values = {
        "search_provider": "searxng",
        "search_api_key_env": None,
        "search_fallback_provider": None,
        "fetch_provider": "jina",
        "fetch_api_key_env": None,
        "fetch_fallback_provider": None,
        "max_search_calls": 2,
        "max_fetch_calls": 2,
        "search_base_url": "https://search.example",
        "fetch_max_chars": 50_000,
        **overrides,
    }
    return DeepResearchWebConfig(**values)


class _FakeSearchTool(DeepResearchWebSearchTool):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.provider_calls: list[tuple[str, str, int]] = []

    async def _run_provider(self, provider: str, query: str, n: int) -> str:
        self.provider_calls.append((provider, query, n))
        if provider == "searxng":
            return "Error: primary search unavailable"
        return f"fallback {provider} results for {query} ({n})"


class _FakeFetchTool(DeepResearchWebFetchTool):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.provider_calls: list[tuple[str, str, str, int]] = []

    async def _run_provider(
        self,
        provider: str,
        url: str,
        extract_mode: str,
        max_chars: int,
    ) -> str:
        self.provider_calls.append((provider, url, extract_mode, max_chars))
        return f"fetched {url} via {provider} as {extract_mode} ({max_chars})"


def test_web_config_rejects_unreviewed_backends_and_missing_provider_requirements() -> None:
    with pytest.raises(ValueError, match="unsupported web_search_provider"):
        _config(search_provider="google").validate({})
    with pytest.raises(ValueError, match="unsupported web_fetch_provider"):
        _config(fetch_provider="browserless").validate({})
    with pytest.raises(ValueError, match="web_search_provider=brave requires an API-key env"):
        _config(search_provider="brave", search_api_key_env=None).validate({})
    with pytest.raises(ValueError, match="requires environment variable BRAVE_API_KEY"):
        _config(search_provider="brave", search_api_key_env="BRAVE_API_KEY").validate({})
    with pytest.raises(ValueError, match="web_search_base_url"):
        _config(search_provider="searxng", search_base_url=None).validate({})


def test_jina_fetch_provider_does_not_require_optional_api_key() -> None:
    _config(fetch_provider="jina", fetch_api_key_env=None).validate({})
    _config(fetch_provider="jina", fetch_api_key_env="JINA_API_KEY").validate({})


def test_search_fallback_records_requested_actual_and_error_metadata() -> None:
    state = WebToolRuntimeState(max_search_calls=3, max_fetch_calls=3)
    tool = _FakeSearchTool(
        config=_config(
            search_provider="searxng",
            search_fallback_provider="jina",
            search_api_key_env="JINA_API_KEY",
        ),
        state=state,
        environ={"JINA_API_KEY": "test-key"},
    )

    result = asyncio.run(tool.execute(query="browsecomp", count=99))

    assert result == "fallback jina results for browsecomp (10)"
    assert tool.provider_calls == [
        ("searxng", "browsecomp", 10),
        ("jina", "browsecomp", 10),
    ]
    [record] = state.records
    assert record.requested_provider == "searxng"
    assert record.actual_provider == "jina"
    assert record.fallback_used is True
    assert record.error is None
    assert record.budget_exhausted is False
    assert record.tool_args == {"query": "browsecomp", "count": 10}


def test_search_budget_exhaustion_yields_without_calling_backend_again() -> None:
    state = WebToolRuntimeState(max_search_calls=1, max_fetch_calls=1)
    tool = _FakeSearchTool(
        config=_config(
            max_search_calls=1,
            search_fallback_provider="jina",
            search_api_key_env="JINA_API_KEY",
        ),
        state=state,
        environ={"JINA_API_KEY": "test-key"},
    )

    first = asyncio.run(tool.execute(query="first", count=2))
    second = asyncio.run(tool.execute(query="second", count=2))

    assert first == "fallback jina results for first (2)"
    assert isinstance(second, BudgetExhaustedResult)
    assert second.content == "Error: web_search tool budget exhausted"
    assert tool.provider_calls == [("searxng", "first", 2), ("jina", "first", 2)]
    assert state.search_calls_used == 1
    assert state.tool_budget_exhausted is True
    assert state.budget_exhausted_tool == "web_search"
    assert state.records[-1].actual_provider is None
    assert state.records[-1].error == "tool_budget_exhausted"
    assert state.records[-1].budget_exhausted is True


def test_fetch_budget_exhaustion_yields_without_calling_backend_again() -> None:
    state = WebToolRuntimeState(max_search_calls=1, max_fetch_calls=1)
    tool = _FakeFetchTool(config=_config(max_fetch_calls=1), state=state, environ={})

    first = asyncio.run(
        tool.execute(
            url="https://example.com/page",
            extractMode="text",
            maxChars=1234,
        )
    )
    second = asyncio.run(tool.execute(url="https://example.com/other"))

    assert first == "fetched https://example.com/page via jina as text (1234)"
    assert isinstance(second, BudgetExhaustedResult)
    assert second.content == "Error: web_fetch tool budget exhausted"
    assert tool.provider_calls == [("jina", "https://example.com/page", "text", 1234)]
    assert state.fetch_calls_used == 1
    assert state.tool_budget_exhausted is True
    assert state.budget_exhausted_tool == "web_fetch"
    assert state.records[-1].actual_provider is None
    assert state.records[-1].error == "tool_budget_exhausted"
    assert state.records[-1].budget_exhausted is True


def test_search_backend_failure_yields_and_records_fail_closed_state() -> None:
    state = WebToolRuntimeState(max_search_calls=2, max_fetch_calls=2)
    tool = _FakeSearchTool(config=_config(search_provider="searxng"), state=state, environ={})

    result = asyncio.run(tool.execute(query="outage", count=3))

    assert isinstance(result, BackendFailureResult)
    assert result.content == "Error: primary search unavailable"
    assert state.tool_backend_failed is True
    assert state.backend_failed_tool == "web_search"
    assert state.backend_failure_error == "Error: primary search unavailable"
    assert tool.provider_calls == [("searxng", "outage", 3)]
    [record] = state.records
    assert record.tool_name == "web_search"
    assert record.actual_provider == "searxng"
    assert record.fallback_used is False
    assert record.error == "Error: primary search unavailable"
    assert record.budget_exhausted is False


def test_duckduckgo_no_results_exception_is_recoverable_empty_result(monkeypatch) -> None:
    _install_failing_ddgs(monkeypatch, RuntimeError("No results found."))
    state = WebToolRuntimeState(max_search_calls=2, max_fetch_calls=2)
    tool = DeepResearchWebSearchTool(
        config=_config(search_provider="duckduckgo", search_base_url=None),
        state=state,
        environ={},
    )

    result = asyncio.run(tool.execute(query="site:example.invalid absent", count=4))

    assert result == "No results for: site:example.invalid absent"
    assert not result.startswith("Error:")
    assert state.tool_backend_failed is False
    assert state.backend_failed_tool is None
    [record] = state.records
    assert record.actual_provider == "duckduckgo"
    assert record.error is None
    assert record.tool_result == result
    assert record.budget_exhausted is False


def test_duckduckgo_provider_exception_fails_closed(monkeypatch) -> None:
    _install_failing_ddgs(monkeypatch, RuntimeError("backend exploded"))
    state = WebToolRuntimeState(max_search_calls=2, max_fetch_calls=2)
    tool = DeepResearchWebSearchTool(
        config=_config(search_provider="duckduckgo", search_base_url=None),
        state=state,
        environ={},
    )

    result = asyncio.run(tool.execute(query="provider outage", count=4))

    expected = "Error: DuckDuckGo search failed (backend exploded)"
    assert isinstance(result, BackendFailureResult)
    assert result.content == expected
    assert state.tool_backend_failed is True
    assert state.backend_failed_tool == "web_search"
    assert state.backend_failure_error == expected
    [record] = state.records
    assert record.actual_provider == "duckduckgo"
    assert record.error == expected
    assert record.tool_result == expected
    assert record.budget_exhausted is False


def test_fetch_backend_failure_yields_and_records_fail_closed_state() -> None:
    class BackendFailingFetchTool(DeepResearchWebFetchTool):
        async def _run_provider(
            self,
            provider: str,
            url: str,
            extract_mode: str,
            max_chars: int,
        ) -> str:
            del provider, extract_mode, max_chars
            return json.dumps(
                {"error": "Jina Reader failed: network connection timed out", "url": url},
                ensure_ascii=False,
            )

    state = WebToolRuntimeState(max_search_calls=2, max_fetch_calls=2)
    tool = BackendFailingFetchTool(config=_config(fetch_provider="jina"), state=state, environ={})

    result = asyncio.run(tool.execute(url="https://example.com/outage"))

    assert isinstance(result, BackendFailureResult)
    assert json.loads(result.content) == {
        "error": "Jina Reader failed: network connection timed out",
        "url": "https://example.com/outage",
    }
    assert state.tool_backend_failed is True
    assert state.backend_failed_tool == "web_fetch"
    assert state.backend_failure_error == "Jina Reader failed: network connection timed out"
    [record] = state.records
    assert record.tool_name == "web_fetch"
    assert record.actual_provider == "jina"
    assert record.error == "Jina Reader failed: network connection timed out"
    assert record.budget_exhausted is False


@pytest.mark.parametrize(
    ("case_name", "provider_result", "expected_error"),
    [
        (
            "no-readable-content",
            json.dumps(
                {
                    "error": "Jina Reader returned no readable content",
                    "url": "https://example.com/empty",
                },
                ensure_ascii=False,
            ),
            "Jina Reader returned no readable content",
        ),
        (
            "not-found",
            json.dumps(
                {"error": "404 Not Found", "url": "https://example.com/missing"},
                ensure_ascii=False,
            ),
            "404 Not Found",
        ),
    ],
)
def test_fetch_recoverable_content_failures_do_not_fail_closed(
    case_name: str,
    provider_result: str,
    expected_error: str,
) -> None:
    class RecoverableFetchTool(DeepResearchWebFetchTool):
        async def _run_provider(
            self,
            provider: str,
            url: str,
            extract_mode: str,
            max_chars: int,
        ) -> str:
            del provider, url, extract_mode, max_chars
            return provider_result

    state = WebToolRuntimeState(max_search_calls=2, max_fetch_calls=2)
    tool = RecoverableFetchTool(config=_config(fetch_provider="jina"), state=state, environ={})

    result = asyncio.run(tool.execute(url=f"https://example.com/{case_name}"))

    assert not isinstance(result, BackendFailureResult)
    assert result == provider_result
    assert state.tool_backend_failed is False
    assert state.backend_failed_tool is None
    [record] = state.records
    assert record.error == expected_error
    assert record.budget_exhausted is False


def test_fetch_url_validation_error_is_recoverable_tool_error() -> None:
    state = WebToolRuntimeState(max_search_calls=2, max_fetch_calls=2)
    tool = DeepResearchWebFetchTool(config=_config(fetch_provider="jina"), state=state, environ={})

    result = asyncio.run(tool.execute(url="ftp://example.com/not-http"))

    assert not isinstance(result, BackendFailureResult)
    parsed = json.loads(result)
    assert parsed["error"] == "URL validation failed: Only http/https allowed, got 'ftp'"
    assert parsed["url"] == "ftp://example.com/not-http"
    assert state.tool_backend_failed is False
    assert state.backend_failed_tool is None
    [record] = state.records
    assert record.error == parsed["error"]
    assert record.budget_exhausted is False

def test_web_tool_surface_is_read_only_web_search_and_fetch_schema() -> None:
    state = WebToolRuntimeState(max_search_calls=1, max_fetch_calls=1)
    search = DeepResearchWebSearchTool(config=_config(), state=state, environ={})
    fetch = DeepResearchWebFetchTool(config=_config(), state=state, environ={})

    assert search.name == "web_search"
    assert fetch.name == "web_fetch"
    assert search.read_only is True
    assert fetch.read_only is True
    assert search.parameters["required"] == ["query"]
    assert set(search.parameters["properties"]) == {"query", "count"}
    assert fetch.parameters["required"] == ["url"]
    assert set(fetch.parameters["properties"]) == {"url", "extractMode", "maxChars"}
