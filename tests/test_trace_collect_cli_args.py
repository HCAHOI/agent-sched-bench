"""Tests for trace_collect CLI argument parsing."""

from __future__ import annotations

import pytest

from trace_collect.cli import parse_collect_args, parse_simulate_args


def test_parse_collect_args_accepts_sample_and_concurrency() -> None:
    args = parse_collect_args([
        "--provider",
        "openrouter",
        "--model",
        "z-ai/glm-5.1",
        "--sample",
        "7",
        "--concurrency",
        "3",
    ])

    assert args.sample == 7
    assert args.concurrency == 3
    assert args.benchmark == "swe-rebench"


def test_parse_collect_args_rejects_skip_argument() -> None:
    with pytest.raises(SystemExit):
        parse_collect_args([
            "--provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.1",
            "--skip",
            "7",
        ])


def test_parse_collect_args_rejects_negative_sample() -> None:
    with pytest.raises(SystemExit):
        parse_collect_args([
            "--provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.1",
            "--sample",
            "-1",
        ])


def test_parse_collect_args_rejects_zero_concurrency() -> None:
    with pytest.raises(SystemExit):
        parse_collect_args([
            "--provider",
            "openrouter",
            "--model",
            "z-ai/glm-5.1",
            "--concurrency",
            "0",
        ])


def test_parse_simulate_args_accepts_ear_policy(tmp_path) -> None:
    policy = tmp_path / "ear.yaml"
    args = parse_simulate_args(
        [
            "--manifest",
            "manifest.yaml",
            "--ear-mode",
            "elastic",
            "--ear-policy",
            str(policy),
        ]
    )

    assert args.ear_mode == "elastic"
    assert args.ear_policy == policy
