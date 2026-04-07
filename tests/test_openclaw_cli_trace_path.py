"""Regression tests for OpenClaw CLI trace-output path resolution.

These tests pin down the convention added by US-005:

    Default: <repo_root>/traces/openclaw_cli/<model_slug>/<UTC_TS>/<sid>.jsonl

…so that traces from `python -m agents.openclaw` land in the same
research-friendly location as `trace_collect.cli` outputs, instead of
being hidden inside ``<workspace>/.openclaw/traces/`` where they get lost
when the workspace is cleaned.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from agents.openclaw._cli import (
    _resolve_repo_root,
    _resolve_trace_output,
    _slug_model,
    build_parser,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_args(**overrides) -> argparse.Namespace:
    """Build a Namespace with the parser defaults plus overrides."""
    args = build_parser().parse_args(["--prompt", "x"])
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def test_parser_exposes_trace_output_flag() -> None:
    args = build_parser().parse_args(["--prompt", "x", "--trace-output", "/tmp/foo.jsonl"])
    assert args.trace_output == "/tmp/foo.jsonl"


def test_parser_default_trace_output_is_none() -> None:
    args = build_parser().parse_args(["--prompt", "x"])
    assert args.trace_output is None


def test_default_trace_output_under_repo_root() -> None:
    """No --trace-output → resolves under <repo>/traces/openclaw_cli/..."""
    args = _make_args(trace_output=None)
    out = _resolve_trace_output(
        args, session_id="oc-abc12345", model="qwen/qwen3.6-plus:free"
    )
    assert out.is_absolute()
    assert out.name == "oc-abc12345.jsonl"
    # Path should start with <repo>/traces/openclaw_cli/<slug>/
    repo_traces = REPO_ROOT / "traces" / "openclaw_cli" / "qwen_qwen3.6-plus_free"
    assert str(out).startswith(str(repo_traces))
    # Timestamp parent dir matches YYYYMMDDTHHMMSSZ
    ts_dir = out.parent.name
    assert re.fullmatch(r"\d{8}T\d{6}Z", ts_dir), (
        f"Expected UTC timestamp dir, got {ts_dir!r}"
    )


def test_explicit_trace_output_overrides_default(tmp_path: Path) -> None:
    target = tmp_path / "explicit.jsonl"
    args = _make_args(trace_output=str(target))
    out = _resolve_trace_output(args, session_id="oc-xyz", model="any/model")
    assert out == target.resolve()


def test_explicit_trace_output_expands_user(monkeypatch, tmp_path: Path) -> None:
    """``~`` is expanded so users can pass --trace-output ~/traces/foo.jsonl."""
    monkeypatch.setenv("HOME", str(tmp_path))
    args = _make_args(trace_output="~/foo.jsonl")
    out = _resolve_trace_output(args, "oc-1", "m")
    assert out == (tmp_path / "foo.jsonl").resolve()


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("qwen/qwen3.6-plus:free", "qwen_qwen3.6-plus_free"),
        ("openai/gpt-4o:turbo", "openai_gpt-4o_turbo"),
        ("anthropic/claude-opus-4-6", "anthropic_claude-opus-4-6"),
        ("simple", "simple"),
    ],
)
def test_slug_model_strips_special_chars(raw: str, expected: str) -> None:
    assert _slug_model(raw) == expected


def test_resolve_repo_root_finds_pyproject() -> None:
    root = _resolve_repo_root()
    assert root is not None
    assert (root / "pyproject.toml").exists()
    assert root == REPO_ROOT
