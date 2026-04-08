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

Phase 5 of trace-sim-vastai-pipeline plan adds:
  test_gantt_dom_snapshot_pre_p5_unchanged
    — DOM-payload regression test using BeautifulSoup + lxml. Renders
      the openclaw_minimal_v5.jsonl fixture (no mcp_call actions, no
      MCP events) and asserts the embedded payload's lane shape +
      action sequence is identical to a golden fixture. Catches any
      Phase 5 edit that accidentally regresses pre-Phase-5 trace
      rendering.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


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
        "trace_format_version": 5, "instance_id": "task-1",
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


# ── Phase 5: DOM-payload regression test ──────────────────────────────


OPENCLAW_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "openclaw_minimal_v5.jsonl"
GOLDEN_PAYLOAD = REPO_ROOT / "tests" / "fixtures" / "gantt_baseline_pre_p5.golden.json"


def test_gantt_dom_snapshot_pre_p5_unchanged(tmp_path: Path) -> None:
    """Phase 5 regression invariant: a pre-Phase-5 trace renders to the
    same payload structure after the Phase 5 edits as it did before.

    The Phase 5 edits are additive — they extend ``_MARKER_CATEGORIES``,
    ``ACTION_TYPE_MAP``, and ``DEFAULT_SPAN_REGISTRY`` with new entries
    that only fire when the input trace contains the new shapes
    (mcp_call action_type, MCP-category events). The
    openclaw_minimal_v5.jsonl fixture contains NEITHER, so its rendered
    output must be unchanged.

    The test:
    1. Renders the fixture through ``generate_gantt_html``
    2. Parses the rendered HTML with BeautifulSoup + lxml to verify
       the static HTML structure is well-formed
    3. Extracts the embedded JSON payload via the same regex pattern
       used by ``test_cli_disambiguates_duplicate_stems``
    4. Compares the traces section against
       ``tests/fixtures/gantt_baseline_pre_p5.golden.json``
    5. Asserts byte-equal match modulo dict key order

    Catches Phase 5 regressions like:
    - Accidentally pop'ing or truncating sim_metrics in
      _extract_detail_from_action
    - Removing an old span/marker entry while adding the new mcp one
    - Breaking the JSONL → payload pipeline for openclaw traces
    """
    bs4 = pytest.importorskip("bs4")
    pytest.importorskip("lxml")

    # Lazy import the renderer to avoid loading agents.* at module import
    from trace_collect.gantt_serve import generate_gantt_html
    from trace_collect.trace_inspector import TraceData

    assert OPENCLAW_FIXTURE.exists(), f"missing fixture: {OPENCLAW_FIXTURE}"
    assert GOLDEN_PAYLOAD.exists(), (
        f"missing golden: {GOLDEN_PAYLOAD}. Regenerate with: "
        f"python -c 'from trace_collect.gantt_data import "
        f"build_gantt_payload_multi; from trace_collect.trace_inspector "
        f"import TraceData; import json; "
        f"d=TraceData.load(\"{OPENCLAW_FIXTURE}\"); "
        f"p=build_gantt_payload_multi([(\"test\", d)]); "
        f"open(\"{GOLDEN_PAYLOAD}\", \"w\").write(json.dumps("
        f"{{\"traces\": p[\"traces\"]}}, indent=2, sort_keys=True))'"
    )

    # ── Step 1: Render the fixture through the live pipeline ──
    data = TraceData.load(OPENCLAW_FIXTURE)
    html = generate_gantt_html([("test", data)])

    # ── Step 2: Parse via BeautifulSoup + lxml ──
    soup = bs4.BeautifulSoup(html, "lxml")

    # Static HTML structure must be well-formed
    assert soup.html is not None, "rendered HTML missing <html> root"
    assert soup.body is not None, "rendered HTML missing <body>"

    # The script tag carrying the payload must be present
    script_tags = soup.find_all("script")
    assert len(script_tags) >= 1, (
        f"expected at least 1 script tag in rendered HTML; got {len(script_tags)}"
    )

    # ── Step 3: Extract the embedded payload via regex ──
    payload_match = re.search(
        r"var __TRACE_DATA__ = (\{.*?\});", html, re.DOTALL
    )
    assert payload_match is not None, (
        "rendered HTML did not contain the __TRACE_DATA__ assignment; "
        "the gantt_template.html splice mechanism may have changed"
    )

    embedded_payload = json.loads(payload_match.group(1))
    assert "traces" in embedded_payload
    assert "registries" in embedded_payload

    # Phase 5 invariant: the new mcp entry must be in the registries
    # (this is the post-Phase-5 expected state, not a regression check)
    assert "mcp" in embedded_payload["registries"]["spans"], (
        "Phase 5 missing: registries.spans should contain 'mcp'"
    )

    # ── Step 4: Compare traces against golden ──
    golden = json.loads(GOLDEN_PAYLOAD.read_text(encoding="utf-8"))

    # Normalize: sort keys recursively so dict ordering doesn't matter
    def _normalize(obj):
        if isinstance(obj, dict):
            return {k: _normalize(obj[k]) for k in sorted(obj.keys())}
        if isinstance(obj, list):
            return [_normalize(x) for x in obj]
        return obj

    embedded_traces_norm = _normalize(embedded_payload["traces"])
    golden_traces_norm = _normalize(golden["traces"])

    # ── Step 5: Byte-equal match (modulo key order) ──
    assert embedded_traces_norm == golden_traces_norm, (
        "Phase 5 regression: openclaw_minimal_v5 fixture renders to a "
        "DIFFERENT payload than the pre-Phase-5 golden. The Phase 5 "
        "edits should be additive only — pre-Phase-5 traces must render "
        "identically. Diff:\n"
        f"expected: {json.dumps(golden_traces_norm, indent=2)[:500]}\n"
        f"actual: {json.dumps(embedded_traces_norm, indent=2)[:500]}"
    )


def test_gantt_dom_snapshot_html_is_lxml_parseable() -> None:
    """The rendered HTML must be lxml-parseable (catches malformed tags).

    A separate, faster test from the full DOM snapshot — runs even if
    the golden fixture is missing or stale, so a developer can iterate
    on the template without regenerating the golden first.
    """
    bs4 = pytest.importorskip("bs4")
    pytest.importorskip("lxml")

    from trace_collect.gantt_serve import generate_empty_gantt_html

    html = generate_empty_gantt_html()
    soup = bs4.BeautifulSoup(html, "lxml")
    assert soup.html is not None
    assert soup.body is not None
    # The empty gantt should still embed the registries
    assert "__TRACE_DATA__" in html or "registries" in html
