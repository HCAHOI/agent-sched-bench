"""Load externalized agent prompt templates for scaffolds.

Templates live under ``configs/prompts/swe_rebench/<name>.md`` and must contain
the literal ``{{task}}`` placeholder that the caller substitutes with the
SWE-rebench task's ``problem_statement`` at run time. Centralising the loader
keeps prompt iteration out of Python source code.
"""

from __future__ import annotations

from pathlib import Path

_PLACEHOLDER = "{{task}}"
_PROMPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "prompts"
    / "swe_rebench"
)


def load_prompt_template(name: str) -> str:
    """Return the raw template text for the named prompt.

    Raises:
        FileNotFoundError: the template file does not exist.
        ValueError: the template does not contain the required ``{{task}}``
            placeholder.
    """
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt template {name!r} not found at {path}. Available templates: "
            f"{[p.stem for p in sorted(_PROMPTS_DIR.glob('*.md'))]}"
        )
    text = path.read_text(encoding="utf-8")
    if _PLACEHOLDER not in text:
        raise ValueError(
            f"Prompt template {name!r} at {path} is missing the required "
            f"{_PLACEHOLDER!r} placeholder."
        )
    return text


def render_prompt(template: str, problem_statement: str) -> str:
    """Substitute the ``{{task}}`` placeholder with the task's problem statement."""
    return template.replace(_PLACEHOLDER, problem_statement)
