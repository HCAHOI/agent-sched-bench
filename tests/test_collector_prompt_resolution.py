"""Tests for benchmark-driven prompt template defaults."""

from __future__ import annotations

from types import SimpleNamespace

from trace_collect.cli import parse_collect_args
from trace_collect.collector import _resolve_prompt_template


def test_parse_collect_args_prompt_template_defaults_to_none() -> None:
    args = parse_collect_args([])
    assert args.prompt_template is None


def test_resolve_prompt_template_uses_benchmark_default_when_unset() -> None:
    benchmark = SimpleNamespace(
        config=SimpleNamespace(default_prompt_template="cc_aligned")
    )
    assert (
        _resolve_prompt_template(benchmark=benchmark, prompt_template=None)
        == "cc_aligned"
    )


def test_resolve_prompt_template_respects_explicit_override() -> None:
    benchmark = SimpleNamespace(
        config=SimpleNamespace(default_prompt_template="cc_aligned")
    )
    assert (
        _resolve_prompt_template(benchmark=benchmark, prompt_template="default")
        == "default"
    )
