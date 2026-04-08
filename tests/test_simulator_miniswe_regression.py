"""Phase 1 mini-swe simulator regression — DEFERRED to US-010 manual smoke.

Pre-mortem B item 1 of the trace-sim-vastai-pipeline plan calls for a
byte-identical regression test that:
- loads a recorded mini-swe v5 fixture
- replays it through the simulator's prepare→replay path
- diffs the produced JSONL against a golden file (timestamps normalized)

This test file is intentionally a documented stub because the byte-
identical regression requires real infrastructure that does NOT exist
in the local Ralph loop:

1. A real recorded mini-swe v5 fixture from a Phase 4 collect run
   against an actual SWE-bench task. The fixture must be Gate-A-clean
   (post-Phase-2 sim_metrics schema) and contain a non-trivial
   sequence of llm_call + tool_exec actions.

2. A real local vLLM server to drive the replay loop's
   _call_local_model_streaming() function (which forces an exact
   token count via the vLLM `min_tokens` extra_body parameter).

3. A real git clone capability (the mini-swe prepare adapter calls
   git clone + git checkout against the source repo at base_commit).

Without these, the test would either:
- silently mock real operations (forbidden by CLAUDE.md "no mocks for
  real workloads")
- pass trivially (defeating the regression-detection purpose)

Compensating coverage in the local Ralph loop:

- `tests/test_scaffold_registry.py` verifies the registry dispatch
  contract: get_prepare("miniswe") returns a callable, the lazy import
  invariant holds, etc.
- `tests/test_openclaw_simulate_adapter.py::test_is_mcp_tool_call_*`
  verifies the MCP-reuse branch logic in isolation.
- `scripts/smoke_full_matrix.sh` cells 1-2 (mini-swe × swe-bench,
  mini-swe × swe-rebench) actually run the collect→evaluate path on
  any host that has cloud LLM keys + podman. The Ralph loop's
  end-to-end smoke run reported these cells PASSING on the dev box.

The byte-identical regression test is documented in
`docs/vastai_setup.md` § "Manual smoke verification (post-Ralph)" as
US-010 deferred work item (a) [Phase 2 vLLM metrics integration smoke,
which subsumes Phase 1 because it requires the same mini-swe-prepare
infrastructure].

This stub exists so:
1. Future maintainers searching for "miniswe_regression" find a clear
   explanation rather than silence.
2. The plan's Phase 1 acceptance criterion ("byte-identical regression
   on recorded fixture") has an auditable record of why it lives in
   US-010 instead of in CI.
3. When the manual smoke produces a recorded fixture, this file is
   the obvious place to land the real test.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(
    reason=(
        "DEFERRED to US-010 manual smoke runbook (item a) — needs real "
        "vLLM A100 + real git clone + recorded mini-swe v5 fixture. "
        "See module docstring for the rationale and compensating coverage."
    )
)
def test_simulator_miniswe_byte_identical_regression() -> None:
    """Placeholder for the deferred byte-identical regression test."""
    raise NotImplementedError(
        "This test is intentionally a stub. See module docstring."
    )
