"""Shared provider presets for collection-time API resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ProviderPreset:
    """Static defaults for a named provider preset."""

    api_base: str
    env_key: str
    default_model: str
    miniswe_litellm_prefix: str


@dataclass(frozen=True)
class ResolvedProviderConfig:
    """Resolved API settings after applying CLI overrides."""

    name: str
    api_base: str
    api_key: str
    model: str
    env_key: str
    miniswe_litellm_prefix: str


PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    "openrouter": ProviderPreset(
        api_base="https://openrouter.ai/api/v1",
        env_key="OPENROUTER_API_KEY",
        default_model="qwen/qwen3.6-plus:free",
        miniswe_litellm_prefix="openrouter",
    ),
    "dashscope": ProviderPreset(
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        env_key="DASHSCOPE_API_KEY",
        default_model="qwen-plus-latest",
        miniswe_litellm_prefix="openai",
    ),
    "openai": ProviderPreset(
        api_base="https://api.openai.com/v1",
        env_key="OPENAI_API_KEY",
        default_model="gpt-4o",
        miniswe_litellm_prefix="openai",
    ),
}


def provider_choices() -> list[str]:
    """Return the supported provider preset names."""

    return list(PROVIDER_PRESETS.keys())


def get_provider_preset(name: str) -> ProviderPreset:
    """Return the preset definition for ``name``."""

    return PROVIDER_PRESETS[name]


def resolve_provider_config(
    *,
    provider: str,
    api_base: str | None,
    api_key: str | None,
    model: str | None,
    environ: Mapping[str, str] | None = None,
) -> ResolvedProviderConfig:
    """Resolve provider settings from presets plus explicit overrides."""

    env = os.environ if environ is None else environ
    preset = get_provider_preset(provider)
    resolved_api_key = api_key or env.get(preset.env_key, "")
    return ResolvedProviderConfig(
        name=provider,
        api_base=api_base or preset.api_base,
        api_key=resolved_api_key,
        model=model or preset.default_model,
        env_key=preset.env_key,
        miniswe_litellm_prefix=preset.miniswe_litellm_prefix,
    )


def infer_miniswe_litellm_prefix(
    *,
    provider_name: str | None,
    api_base: str | None,
) -> str:
    """Infer the LiteLLM transport prefix MiniSWE should use."""

    if provider_name is not None and provider_name in PROVIDER_PRESETS:
        return PROVIDER_PRESETS[provider_name].miniswe_litellm_prefix
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
