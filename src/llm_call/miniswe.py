"""MiniSWE-specific provider helpers."""

from __future__ import annotations

from llm_call.providers import PROVIDERS


def infer_miniswe_litellm_prefix(
    *,
    provider_name: str | None,
    api_base: str | None,
) -> str:
    """Infer the LiteLLM transport prefix MiniSWE should use."""

    if provider_name is not None and provider_name in PROVIDERS:
        return PROVIDERS[provider_name].miniswe_litellm_prefix
    if api_base and "openrouter" in api_base.lower():
        return "openrouter"
    return "openai"


def build_miniswe_litellm_model_name(
    *,
    model: str,
    provider_name: str | None,
    api_base: str | None,
) -> str:
    """Return the provider-qualified LiteLLM model identifier."""

    prefix = infer_miniswe_litellm_prefix(
        provider_name=provider_name,
        api_base=api_base,
    )
    qualified_prefix = f"{prefix}/"
    if model.startswith(qualified_prefix):
        return model
    return f"{qualified_prefix}{model}"
