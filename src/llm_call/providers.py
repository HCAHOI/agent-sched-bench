"""Registered remote LLM providers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderDefinition:
    """Static configuration for a supported LLM provider."""

    api_base: str
    env_key: str
    miniswe_litellm_prefix: str


PROVIDERS: dict[str, ProviderDefinition] = {
    "openrouter": ProviderDefinition(
        api_base="https://openrouter.ai/api/v1",
        env_key="OPENROUTER_API_KEY",
        miniswe_litellm_prefix="openrouter",
    ),
    "dashscope": ProviderDefinition(
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        env_key="DASHSCOPE_API_KEY",
        miniswe_litellm_prefix="openai",
    ),
    "openai": ProviderDefinition(
        api_base="https://api.openai.com/v1",
        env_key="OPENAI_API_KEY",
        miniswe_litellm_prefix="openai",
    ),
}


def provider_choices() -> list[str]:
    """Return supported provider names in CLI-friendly order."""

    return list(PROVIDERS.keys())
