"""Tests for the ``python -m trace_collect.cli gantt --empty`` flag.

The empty flag produces a self-contained HTML viewer with no embedded
traces. Users open it, then drag-drop ``.jsonl`` files to load traces
client-side via the GanttBuilder JS module spliced into the template.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_with_pythonpath(cwd: Path) -> dict[str, str]:
    env = dict(os.environ)
    src = str(cwd / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src}:{existing}" if existing else src
    return env


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "trace_collect.cli", "gantt", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=_env_with_pythonpath(REPO_ROOT),
        check=False,
    )


def _extract_embedded_payload(html: str) -> dict:
    """Pull the __TRACE_DATA__ literal out of the generated HTML."""
    m = re.search(r"var __TRACE_DATA__ = (\{.*?\});", html, re.DOTALL)
    assert m, "No __TRACE_DATA__ assignment found in HTML"
    raw = m.group(1)
    # Walk forward from the first { to find its matching }. Payload uses
    # compact separators, no </script> embedded, so brace depth works.
    depth = 0
    end = -1
    for i, c in enumerate(raw):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    assert end > 0, "Could not find closing brace of payload"
    return json.loads(raw[:end])


def test_empty_flag_produces_html_with_empty_traces(tmp_path: Path) -> None:
    out = tmp_path / "gantt_empty.html"
    result = _run_cli(["--empty", "--output", str(out)])
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    payload = _extract_embedded_payload(html)
    assert payload["traces"] == []
    # Registries must still be shipped so drag-dropped traces render with
    # the canonical three-color palette.
    assert "spans" in payload["registries"]
    assert "markers" in payload["registries"]
    assert "llm" in payload["registries"]["spans"]
    assert "tool" in payload["registries"]["spans"]
    assert "scheduling" in payload["registries"]["spans"]


def test_empty_flag_embeds_gantt_builder_js(tmp_path: Path) -> None:
    """The empty viewer must still carry GanttBuilder so drag-drop works."""
    out = tmp_path / "gantt_empty.html"
    _run_cli(["--empty", "--output", str(out)])
    html = out.read_text(encoding="utf-8")
    assert "GanttBuilder" in html
    assert "parseJsonl" in html
    assert "buildPayload" in html


def test_empty_conflicts_with_positional_traces() -> None:
    result = _run_cli(["--empty", "nonexistent.jsonl"])
    assert result.returncode != 0
    assert "--empty cannot be combined" in result.stderr


def test_no_traces_and_no_empty_fails() -> None:
    result = _run_cli([])
    assert result.returncode != 0
    assert "pass at least one trace file" in result.stderr
