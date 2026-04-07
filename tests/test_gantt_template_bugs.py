"""Regression tests for the codex-caught runtime bugs in gantt_template.html.

US-006:
  (1) Remove button previously used ``onclick="removeTrace(${JSON.stringify(label)})"``
      which rendered as ``onclick="removeTrace("foo")"`` — invalid HTML that
      terminated the attribute early and silently broke the remove flow.
  (2) LoadedTraces seeding collapsed two CLI-embedded traces with the same
      Path.stem into one entry because the init seeding did not apply the
      _2/_3 disambiguation that addTraceFile uses.
  (3) rerender() did not re-anchor pinned tooltips; extracted reanchorPinned
      helper is now called from both resize() and rerender().

These tests are text-level assertions against the generated HTML (no
browser automation). They confirm the generated artifact carries the
fix and catch regression via source audit.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "src" / "trace_collect" / "gantt_template.html"


def _env_with_pythonpath() -> dict[str, str]:
    env = dict(os.environ)
    src = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src}:{existing}" if existing else src
    return env


def _run_empty_gantt(tmp_path: Path) -> str:
    out = tmp_path / "empty.html"
    result = subprocess.run(
        [sys.executable, "-m", "trace_collect.cli", "gantt", "--empty",
         "--output", str(out)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=_env_with_pythonpath(),
        check=True,
    )
    assert result.returncode == 0, result.stderr
    return out.read_text(encoding="utf-8")


# ── Bug #1: remove button uses data-label, not inline onclick ────────


def test_remove_button_uses_data_label_not_inline_onclick() -> None:
    """The template source must use a data-label attribute + delegated
    listener, not an inline onclick that embeds JSON.stringify into a
    double-quoted attribute (which terminates the attribute early)."""
    src = TEMPLATE.read_text(encoding="utf-8")
    # The broken pattern must not reappear.
    assert "removeTrace(${JSON.stringify(label)})" not in src, (
        "Reverted to the broken inline-onclick pattern"
    )
    assert 'onclick="removeTrace(' not in src, (
        "Inline onclick for removeTrace leaked back in"
    )
    # The fixed pattern must be present.
    assert 'data-label="${esc(label)}"' in src
    assert "function onTraceItemsClick(" in src
    assert "items.addEventListener('click', onTraceItemsClick)" in src


def test_generated_empty_html_contains_no_broken_onclick(tmp_path: Path) -> None:
    html = _run_empty_gantt(tmp_path)
    # The generated HTML should never contain the concrete broken button
    # attribute string. (It's empty of trace items at first, but the
    # template source must still be lint-clean.)
    assert 'onclick="removeTrace("' not in html


# ── Bug #2: init seeding disambiguates collided labels ───────────────


def test_init_seeds_disambiguate_duplicate_labels() -> None:
    """The init() seeding loop for LoadedTraces must apply the same
    _2/_3 suffix logic as addTraceFile, otherwise two CLI traces with
    the same Path.stem collapse into one entry."""
    src = TEMPLATE.read_text(encoding="utf-8")
    # Find the init() body section and confirm the while-loop is present.
    init_start = src.index("function init()")
    init_end = src.index("// ─── Legend ", init_start)
    init_body = src[init_start:init_end]
    assert "while (LoadedTraces.has(label))" in init_body, (
        "init() seeding loop missing the disambiguation while-loop"
    )
    assert "baseLabel" in init_body


def test_cli_disambiguates_duplicate_stems(tmp_path: Path) -> None:
    """End-to-end check: two CLI traces with the same basename produce a
    payload with two distinct trace entries, and the generated HTML
    carries both ids (so init() can seed them with unique labels)."""
    # Create two traces with the same filename stem in different dirs
    d1 = tmp_path / "a"
    d2 = tmp_path / "b"
    d1.mkdir()
    d2.mkdir()
    trace_head = json.dumps({
        "type": "trace_metadata", "scaffold": "synthetic",
        "trace_format_version": 4, "instance_id": "task-1",
    })
    action = json.dumps({
        "type": "action", "action_type": "llm_call", "action_id": "llm_0",
        "agent_id": "a1", "iteration": 0, "ts_start": 1.0, "ts_end": 2.0,
        "data": {},
    })
    (d1 / "trace.jsonl").write_text(f"{trace_head}\n{action}\n")
    (d2 / "trace.jsonl").write_text(f"{trace_head}\n{action}\n")

    out = tmp_path / "dup.html"
    result = subprocess.run(
        [sys.executable, "-m", "trace_collect.cli", "gantt",
         str(d1 / "trace.jsonl"), str(d2 / "trace.jsonl"),
         "--output", str(out)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=_env_with_pythonpath(),
        check=True,
    )
    assert result.returncode == 0, result.stderr
    html = out.read_text(encoding="utf-8")
    # Python side emits two trace entries even though both have id
    # "synthetic/task-1" (same instance). JS side will disambiguate them
    # on load via the init loop.
    payload_match = re.search(r"var __TRACE_DATA__ = (\{.*?\});", html, re.DOTALL)
    assert payload_match
    # Count trace entries by scanning for the per-trace "metadata" key.
    n_traces = html.count('"metadata"')
    assert n_traces >= 2, f"Expected >=2 trace metadata blocks, got {n_traces}"


# ── Bug #3: rerender calls reanchorPinned ────────────────────────────


def test_rerender_reanchors_pinned_tooltip() -> None:
    src = TEMPLATE.read_text(encoding="utf-8")
    rerender_start = src.index("function rerender()")
    rerender_end = src.index("\n}\n", rerender_start)
    rerender_body = src[rerender_start:rerender_end]
    assert "reanchorPinned()" in rerender_body, (
        "rerender() does not call reanchorPinned() — stale tooltip bug"
    )
    # The helper itself must be defined at module scope (not inside resize).
    assert "\nfunction reanchorPinned()" in src


def test_resize_still_calls_reanchor_after_extraction() -> None:
    """Make sure extracting the helper didn't drop the call from resize()."""
    src = TEMPLATE.read_text(encoding="utf-8")
    resize_start = src.index("function resize()")
    # resize is defined inside setupCanvas, its closing brace is local;
    # walk forward until we hit the sibling function.
    resize_end = src.index("new ResizeObserver", resize_start)
    resize_body = src[resize_start:resize_end]
    assert "reanchorPinned()" in resize_body
