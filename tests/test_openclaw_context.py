"""Focused tests for OpenClaw prompt workspace rendering."""

from __future__ import annotations

from pathlib import Path

from agents.openclaw._context import ContextBuilder


def test_context_builder_uses_project_workspace_for_prompt_identity(
    tmp_path: Path,
) -> None:
    state_workspace = tmp_path / "state"
    state_workspace.mkdir()
    project_workspace = tmp_path / "project"
    project_workspace.mkdir()
    (project_workspace / "AGENTS.md").write_text(
        "project instructions\n",
        encoding="utf-8",
    )

    prompt = ContextBuilder(
        state_workspace,
        project_workspace=project_workspace,
    ).build_system_prompt()

    assert f"Your workspace is at: {project_workspace.resolve()}" in prompt
    assert f"Agent state workspace: {state_workspace.resolve()}" in prompt
    # Runtime memory/skills are managed by the agent runtime; the prompt must
    # not instruct the model to create memory or skill directories in the task
    # workspace (workspace-local runtime state is deprecated).
    assert "memory/MEMORY.md" not in prompt
    assert "memory/HISTORY.md" not in prompt
    assert "Custom skills:" not in prompt
    assert "project instructions" in prompt


def test_context_builder_does_not_instruct_workspace_memory_creation(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    prompt = ContextBuilder(workspace).build_system_prompt()
    # No instructional lines pointing the model at workspace memory/skills paths.
    assert "memory/MEMORY.md" not in prompt
    assert "memory/HISTORY.md" not in prompt
    assert "Custom skills:" not in prompt
    # The prompt should state memory is runtime-managed, not workspace-written.
    assert "runtime" in prompt.lower()
