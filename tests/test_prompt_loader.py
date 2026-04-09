"""Tests for src/trace_collect/prompt_loader.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trace_collect.prompt_loader import load_prompt_template, render_prompt  # noqa: E402


def test_load_prompt_template_default_contains_placeholder() -> None:
    template = load_prompt_template("default")
    assert "{{task}}" in template
    assert "Submission" in template  # sanity check for content extraction


def test_load_prompt_template_cc_aligned_contains_placeholder_and_test_mandate() -> None:
    template = load_prompt_template("cc_aligned")
    assert "{{task}}" in template
    # The distinguishing feature of cc_aligned is the "run tests until all pass" mandate.
    assert "Run the test suite" in template
    assert "CRITICAL REQUIREMENTS FOR TESTING" in template


def test_load_prompt_template_missing_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError) as exc:
        load_prompt_template("definitely_not_a_real_template_name")
    assert "definitely_not_a_real_template_name" in str(exc.value)
    assert "Available templates" in str(exc.value)


def test_load_prompt_template_without_placeholder_raises_value_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Point the loader at a temp prompts dir holding a bad template.
    bad_dir = tmp_path / "configs" / "prompts" / "swe_rebench"
    bad_dir.mkdir(parents=True)
    (bad_dir / "broken.md").write_text("no placeholder here", encoding="utf-8")
    monkeypatch.setattr(
        "trace_collect.prompt_loader._PROMPTS_DIR", bad_dir
    )
    with pytest.raises(ValueError) as exc:
        load_prompt_template("broken")
    assert "{{task}}" in str(exc.value)


def test_render_prompt_substitutes_placeholder() -> None:
    template = "Fix this: {{task}}\nNow submit."
    result = render_prompt(template, "Null pointer in foo()")
    assert result == "Fix this: Null pointer in foo()\nNow submit."


def test_render_prompt_is_idempotent_without_placeholder() -> None:
    template = "No placeholder here."
    assert render_prompt(template, "anything") == template
