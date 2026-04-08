"""OpenClaw scaffold adapter for the trace simulator (Phase 1.5.1).

Wraps `prepare_workspace()` for the scaffold registry and provides
`is_mcp_tool_call()` so the simulator can reuse recorded MCP results
instead of re-dispatching to a live MCP server (Pre-mortem C item 2:
zero context7 egress during replay). Strategy and rationale live in
`.omc/plans/phase1.5-design.md`.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from trace_collect.scaffold_registry import (
    PreparedWorkspace,
    SimulatePrepareConfig,
)


def is_mcp_tool_call(tool_name: str | None) -> bool:
    """True iff `tool_name` is an MCP-routed call (`mcp_<server>_<tool>`)."""
    return tool_name is not None and tool_name.startswith("mcp_")


async def _openclaw_prepare(
    task: dict[str, Any],
    config: SimulatePrepareConfig,
) -> PreparedWorkspace:
    """Clone the openclaw task repo into a fresh tempdir and return a workspace.

    Wraps the module-level `prepare_workspace()` free function from
    `agents.openclaw.eval.prepare` and adapts its signature to the
    scaffold registry contract.
    """
    # prepare_workspace lives inside the heavy openclaw eval package;
    # keep the import lazy so importing this module stays cheap.
    from agents.openclaw.eval.prepare import prepare_workspace

    repo: str = task["repo"]
    base_commit: str = task["base_commit"]
    workspace_dir = Path(tempfile.mkdtemp(prefix="openclaw_sim_"))

    try:
        await prepare_workspace(
            workspace_dir,
            repo=repo,
            base_commit=base_commit,
            repos_root=config.repos_root,
        )
    except Exception:
        shutil.rmtree(workspace_dir, ignore_errors=True)
        raise

    def _cleanup() -> None:
        shutil.rmtree(workspace_dir, ignore_errors=True)

    return PreparedWorkspace(repo_dir=workspace_dir, cleanup=_cleanup)


__all__ = ["_openclaw_prepare", "is_mcp_tool_call"]
