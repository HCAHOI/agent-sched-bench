"""Phase 4 unit tests: --mcp-config flag enforcement for openclaw scaffold.

Three subtests per the PRD acceptance:
1. Valid YAML path → loads dict + exit code 0 (the run still has to be
   invoked but the validation passes; we test only the validation step).
2. 'none' → empty dict + trace header records mcp_config='none'.
3. Missing flag for openclaw → exit code 2 with the EXACT stderr substring.

Plus coverage for:
- mini-swe scaffold is unaffected (no --mcp-config required).
- _mcp_config_label correctly maps None / 'none' / path → label.
- load_mcp_servers handles all three branches.
- Phase 0 schema audit (a) compatibility: load_mcp_servers returns a
  dict whose values are MCPServerConfig instances, not raw dicts.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTEXT7_YAML = REPO_ROOT / "configs" / "mcp" / "context7.yaml"


# ---------------------------------------------------------------------------
# _mcp_config_label helper
# ---------------------------------------------------------------------------


def test_mcp_config_label_none_returns_none() -> None:
    from trace_collect.collector import _mcp_config_label

    assert _mcp_config_label(None) is None


def test_mcp_config_label_literal_none_returns_none_string() -> None:
    from trace_collect.collector import _mcp_config_label

    assert _mcp_config_label("none") == "none"


def test_mcp_config_label_path_returns_basename() -> None:
    from trace_collect.collector import _mcp_config_label

    assert _mcp_config_label("configs/mcp/context7.yaml") == "context7.yaml"
    assert _mcp_config_label("/abs/path/to/foo.yaml") == "foo.yaml"


# ---------------------------------------------------------------------------
# load_mcp_servers helper
# ---------------------------------------------------------------------------


def test_load_mcp_servers_none_returns_empty_dict() -> None:
    from trace_collect.collector import load_mcp_servers

    assert load_mcp_servers(None) == {}


def test_load_mcp_servers_literal_none_returns_empty_dict() -> None:
    from trace_collect.collector import load_mcp_servers

    assert load_mcp_servers("none") == {}


def test_load_mcp_servers_context7_yaml_loads_mcp_server_config() -> None:
    """Phase 0 schema audit (a): values must be MCPServerConfig, not dicts."""
    from agents.openclaw.config.schema import MCPServerConfig
    from trace_collect.collector import load_mcp_servers

    assert CONTEXT7_YAML.exists(), f"context7 fixture missing: {CONTEXT7_YAML}"

    servers = load_mcp_servers(str(CONTEXT7_YAML))

    assert "context7" in servers
    assert isinstance(servers["context7"], MCPServerConfig), (
        "load_mcp_servers must instantiate MCPServerConfig per Phase 0 audit"
    )
    assert servers["context7"].type == "streamableHttp"
    assert "context7.com" in servers["context7"].url
    assert servers["context7"].tool_timeout == 30


def test_load_mcp_servers_missing_path_raises_file_not_found() -> None:
    from trace_collect.collector import load_mcp_servers

    with pytest.raises(FileNotFoundError):
        load_mcp_servers("/tmp/this-yaml-does-not-exist-anywhere.yaml")


def test_load_mcp_servers_rejects_non_mapping_yaml(tmp_path) -> None:
    from trace_collect.collector import load_mcp_servers

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("- this\n- is\n- a\n- list\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_mcp_servers(str(bad_yaml))
    assert "mapping" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Subtest 3 — missing --mcp-config for openclaw → exit 2 with exact substring
# ---------------------------------------------------------------------------


EXPECTED_ERROR_SUBSTRING = (
    "MCP config is required for openclaw; pass "
    "--mcp-config configs/mcp/context7.yaml or --mcp-config none "
    "to acknowledge running without MCP"
)


def test_openclaw_without_mcp_config_exits_2_with_exact_message() -> None:
    """Subprocess invocation: openclaw + no --mcp-config → exit 2."""
    result = subprocess.run(
        [
            sys.executable,
            "-m", "trace_collect.cli",
            "--benchmark", "swe-rebench",
            "--scaffold", "openclaw",
            "--sample", "1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO_ROOT / "src")},
    )

    # Exit code 2 is the contract — distinguish from generic exit 1 (other errors).
    assert result.returncode == 2, (
        f"expected exit code 2 for openclaw-without-mcp-config; got "
        f"{result.returncode}. stderr: {result.stderr}"
    )

    # Exact substring match per PRD
    assert EXPECTED_ERROR_SUBSTRING in result.stderr, (
        f"expected stderr to contain the exact substring:\n"
        f"  {EXPECTED_ERROR_SUBSTRING}\n"
        f"got stderr:\n{result.stderr}"
    )


def test_miniswe_without_mcp_config_does_not_trigger_validation() -> None:
    """mini-swe scaffold must NOT trigger the openclaw mcp-config validation.

    The CLI call below will eventually fail for OTHER reasons (no API
    key, missing data, etc.) but the failure must NOT be the openclaw
    mcp-config error. We assert by checking the stderr does NOT contain
    the openclaw error string.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m", "trace_collect.cli",
            "--benchmark", "swe-bench-verified",
            "--scaffold", "miniswe",
            "--sample", "1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO_ROOT / "src")},
        timeout=10,
    )

    # Whatever the exit code, the openclaw error must NOT appear
    assert EXPECTED_ERROR_SUBSTRING not in result.stderr, (
        "mini-swe scaffold must not trigger the openclaw mcp-config "
        "validation; got openclaw error in stderr"
    )


# ---------------------------------------------------------------------------
# Subtest 1 + 2 — valid YAML and 'none' both pass validation
# ---------------------------------------------------------------------------


def test_parse_collect_args_accepts_yaml_path() -> None:
    """The CLI parser must accept --mcp-config <yaml path> as a string."""
    from trace_collect.cli import parse_collect_args

    args = parse_collect_args(
        [
            "--scaffold", "openclaw",
            "--mcp-config", "configs/mcp/context7.yaml",
        ]
    )
    assert args.mcp_config == "configs/mcp/context7.yaml"


def test_parse_collect_args_accepts_none_literal() -> None:
    """The CLI parser must accept --mcp-config none as the affirmative opt-out."""
    from trace_collect.cli import parse_collect_args

    args = parse_collect_args(
        [
            "--scaffold", "openclaw",
            "--mcp-config", "none",
        ]
    )
    assert args.mcp_config == "none"


def test_parse_collect_args_default_mcp_config_is_none() -> None:
    """Default is None (the validation step will refuse for openclaw)."""
    from trace_collect.cli import parse_collect_args

    args = parse_collect_args([])
    assert args.mcp_config is None
