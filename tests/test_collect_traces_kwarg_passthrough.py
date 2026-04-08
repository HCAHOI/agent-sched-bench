"""Phase 4 [m2]: assert mcp_config arrives at collect_traces as a kwarg.

The CLI must NOT re-read sys.argv inside the collector. The data flow
from --mcp-config → collect_traces(mcp_config=...) must be by-argument,
verifiable by monkeypatching collect_traces and inspecting the kwargs
it received.

This guards against a refactoring regression where someone might be
tempted to add a `os.environ.get(...)` or `sys.argv` lookup inside
collector.py — that would break reproducibility AND the audit trail.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def mock_collect_traces(monkeypatch):
    """Replace collect_traces with a MagicMock that records all calls."""
    import trace_collect.collector as collector_mod

    mock = MagicMock(name="collect_traces", return_value=Path("/tmp/fake_run"))

    async def _mock_collect_traces(**kwargs):
        mock(**kwargs)
        return Path("/tmp/fake_run")

    monkeypatch.setattr(collector_mod, "collect_traces", _mock_collect_traces)
    # Also patch the import in cli.py — _run_collect uses
    # `from trace_collect.collector import collect_traces` AT FUNCTION TIME,
    # so the monkeypatch on collector_mod is enough.
    return mock


def _run_cli_main(monkeypatch, argv: list[str]) -> int:
    """Invoke trace_collect.cli.main with synthetic argv. Returns exit code."""
    import trace_collect.cli as cli_mod

    monkeypatch.setattr(sys, "argv", ["trace_collect.cli"] + argv)

    # Provide a fake API key so the CLI doesn't exit on missing key
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")

    try:
        cli_mod.main()
        return 0
    except SystemExit as e:
        return int(e.code) if e.code is not None else 0


def test_mcp_config_arrives_as_kwarg_for_yaml_path(monkeypatch, mock_collect_traces) -> None:
    """CLI invoked with --mcp-config <yaml path> → kwarg matches."""
    exit_code = _run_cli_main(
        monkeypatch,
        [
            "--benchmark", "swe-rebench",
            "--scaffold", "openclaw",
            "--mcp-config", "configs/mcp/context7.yaml",
            "--sample", "0",
        ],
    )

    assert exit_code == 0, f"CLI exit non-zero: {exit_code}"
    assert mock_collect_traces.call_count == 1
    call_kwargs = mock_collect_traces.call_args.kwargs
    assert call_kwargs.get("mcp_config") == "configs/mcp/context7.yaml", (
        f"collect_traces should receive mcp_config as a kwarg matching the "
        f"CLI value; got {call_kwargs.get('mcp_config')!r}"
    )


def test_mcp_config_arrives_as_none_literal(monkeypatch, mock_collect_traces) -> None:
    """CLI invoked with --mcp-config none → kwarg is the string 'none'."""
    exit_code = _run_cli_main(
        monkeypatch,
        [
            "--benchmark", "swe-rebench",
            "--scaffold", "openclaw",
            "--mcp-config", "none",
            "--sample", "0",
        ],
    )

    assert exit_code == 0
    call_kwargs = mock_collect_traces.call_args.kwargs
    assert call_kwargs.get("mcp_config") == "none"


def test_mini_swe_scaffold_passes_none_mcp_config(monkeypatch, mock_collect_traces) -> None:
    """mini-swe scaffold (no --mcp-config required) → kwarg is None."""
    exit_code = _run_cli_main(
        monkeypatch,
        [
            "--benchmark", "swe-bench-verified",
            "--scaffold", "mini-swe-agent",
            "--sample", "0",
        ],
    )

    assert exit_code == 0
    call_kwargs = mock_collect_traces.call_args.kwargs
    assert call_kwargs.get("mcp_config") is None, (
        "mini-swe scaffold should pass mcp_config=None (the default)"
    )


def test_collect_traces_kwarg_is_not_read_from_environ(monkeypatch, mock_collect_traces) -> None:
    """If MCP_CONFIG env var is set, the CLI must NOT pick it up.

    This guards against a regression where someone adds an os.environ
    fallback inside collector.py — the CLI flag MUST be the only source
    of truth for mcp_config to preserve the audit trail.
    """
    monkeypatch.setenv("MCP_CONFIG", "configs/mcp/from-env.yaml")

    exit_code = _run_cli_main(
        monkeypatch,
        [
            "--benchmark", "swe-bench-verified",
            "--scaffold", "mini-swe-agent",
            "--sample", "0",
        ],
    )

    assert exit_code == 0
    call_kwargs = mock_collect_traces.call_args.kwargs
    # The env var must NOT have leaked into the kwarg
    assert call_kwargs.get("mcp_config") != "configs/mcp/from-env.yaml"
    assert call_kwargs.get("mcp_config") is None
