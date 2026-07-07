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
    # Memory is unused (one fresh run per task): no memory context injection,
    # no memory/history instruction lines, no agent-state-workspace line.
    assert "memory/MEMORY.md" not in prompt
    assert "memory/HISTORY.md" not in prompt
    assert "Custom skills:" not in prompt
    assert "managed automatically by the agent runtime" not in prompt
    assert "Agent state workspace" not in prompt
    assert "project instructions" in prompt


def test_context_builder_uses_runtime_label_override(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()

    prompt = ContextBuilder(
        workspace,
        runtime_label=(
            "Shell/file tools runtime: Linux x86_64\n"
            "Shell/file tools `python3`: Python 3.6.9"
        ),
    ).build_system_prompt()

    assert "Shell/file tools runtime: Linux x86_64" in prompt
    assert "Shell/file tools `python3`: Python 3.6.9" in prompt
    assert "Python 3.12" not in prompt


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
    # No memory/history instruction lines remain in the prompt.
    assert "managed automatically by the agent runtime" not in prompt
    assert "Agent state workspace" not in prompt


def test_context_builder_omits_skills_and_message_attachment_guidance(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    skills_dir = tmp_path / "rt" / "skills"
    (skills_dir / "demo-skill").mkdir(parents=True)
    (skills_dir / "demo-skill" / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: demo\nalways: true\n---\nskill body",
        encoding="utf-8",
    )

    prompt = ContextBuilder(workspace, skills_dir=skills_dir).build_system_prompt()

    assert "# Active Skills" not in prompt
    assert "# Skills" not in prompt
    assert "available=\"false\"" not in prompt
    assert "demo-skill" not in prompt
    assert "skill body" not in prompt
    assert "Only use the 'message' tool" not in prompt
    assert "media" not in prompt
    assert "native image content" not in prompt
