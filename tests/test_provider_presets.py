"""Tests for shared provider preset resolution."""

from __future__ import annotations

from trace_collect.provider_presets import (
    build_miniswe_litellm_model_name,
    resolve_provider_config,
)


def test_resolve_provider_config_uses_openrouter_defaults() -> None:
    resolved = resolve_provider_config(
        provider="openrouter",
        api_base=None,
        api_key=None,
        model=None,
        environ={"OPENROUTER_API_KEY": "test-key"},
    )

    assert resolved.name == "openrouter"
    assert resolved.api_base == "https://openrouter.ai/api/v1"
    assert resolved.api_key == "test-key"
    assert resolved.model == "qwen/qwen3.6-plus:free"
    assert resolved.env_key == "OPENROUTER_API_KEY"


def test_resolve_provider_config_honors_explicit_overrides() -> None:
    resolved = resolve_provider_config(
        provider="dashscope",
        api_base="https://override.example/v1",
        api_key="override-key",
        model="override-model",
        environ={"DASHSCOPE_API_KEY": "ignored"},
    )

    assert resolved.name == "dashscope"
    assert resolved.api_base == "https://override.example/v1"
    assert resolved.api_key == "override-key"
    assert resolved.model == "override-model"


def test_build_miniswe_litellm_model_name_for_openrouter() -> None:
    assert (
        build_miniswe_litellm_model_name(
            model="anthropic/claude-haiku-4.5",
            provider_name="openrouter",
            api_base="https://openrouter.ai/api/v1",
        )
        == "openrouter/anthropic/claude-haiku-4.5"
    )


def test_build_miniswe_litellm_model_name_for_openai_compatible_provider() -> None:
    assert (
        build_miniswe_litellm_model_name(
            model="qwen-plus-latest",
            provider_name="dashscope",
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        == "openai/qwen-plus-latest"
    )
