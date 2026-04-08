"""Deferred mini-swe simulator regression test.

The byte-identical replay check needs three real inputs that are not
available in normal local or CI runs:

- a recorded mini-swe v5 trace fixture,
- a live local vLLM server for exact-token replay,
- and real git clone/checkout during the prepare phase.

Without those inputs, the test would either mock real operations or
degenerate into a no-op. Existing coverage for the surrounding behavior
lives in scaffold-registry tests, MCP-reuse adapter tests, and the
manual smoke runbook referenced from ``docs/vastai_setup.md``.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(
    reason=(
        "Needs a recorded mini-swe v5 fixture, a live local vLLM server, "
        "and real git clone/checkout support. See module docstring."
    )
)
def test_simulator_miniswe_byte_identical_regression() -> None:
    """Placeholder for the deferred byte-identical regression test."""
    raise NotImplementedError(
        "This test is intentionally a stub. See module docstring."
    )
