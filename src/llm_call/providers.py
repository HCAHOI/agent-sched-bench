"""Registered remote LLM providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_call.provider_base import LLMProvider


@dataclass(frozen=True)
class ProviderDefinition:
    """Static configuration for a supported LLM provider."""

    api_base: str
    env_key: str


PROVIDERS: dict[str, ProviderDefinition] = {
    "openrouter": ProviderDefinition(
        api_base="https://openrouter.ai/api/v1",
        env_key="OPENROUTER_API_KEY",
    ),
    "dashscope": ProviderDefinition(
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        env_key="DASHSCOPE_API_KEY",
    ),
    "openai": ProviderDefinition(
        api_base="https://api.openai.com/v1",
        env_key="OPENAI_API_KEY",
    ),
    "siliconflow": ProviderDefinition(
        api_base="https://api.siliconflow.com/v1",
        env_key="SILICONFLOW_API_KEY",
    ),
    "deepseek": ProviderDefinition(
        api_base="https://api.deepseek.com",
        env_key="DEEPSEEK_API_KEY",
    ),
    "pioneer": ProviderDefinition(
        api_base="https://api.pioneer.ai/v1",
        env_key="PIONEER_API_KEY",
    ),
    "codex": ProviderDefinition(
        api_base="https://chatgpt.com/backend-api/codex",
        env_key="CODEX_ACCESS_TOKEN",
    ),
}


def provider_choices() -> list[str]:
    """Return supported provider names in CLI-friendly order."""

    return list(PROVIDERS.keys())


def create_provider(
    *,
    provider_name: str | None,
    api_key: str | None,
    api_base: str | None,
    default_model: str,
    **kwargs: Any,
) -> LLMProvider:
    """Build the implementation registered for *provider_name*."""

    if provider_name == "codex":
        from llm_call.codex import CodexProvider

        return CodexProvider(api_key, api_base, default_model, **kwargs)

    from llm_call.openclaw import UnifiedProvider

    return UnifiedProvider(api_key, api_base, default_model, **kwargs)
