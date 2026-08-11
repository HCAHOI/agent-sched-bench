#!/usr/bin/env python3
"""Run the frozen one-shot plugin-aware pytest ToolSpec successor."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import tiktoken

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from scripts.evaluation.evaluate_doc_tool_semantics import (  # noqa: E402
    _render_generation_prompt,
)
from scripts.evaluation.evaluate_offline_agent_extractor import (  # noqa: E402
    _codex_call,
)
from tool_resource.tool_spec import tool_spec_schema, validate_tool_spec  # noqa: E402


_PROTOCOL_GIT_SHA = "84a0786f809678b65796ee80daeffa0aec27b349"
_SOURCES = _ROOT / "analysis/development/offline-tool-semantics-docs"
_OUTPUT = _ROOT / (
    "analysis/results/pennylane-pytest-xdist-doc-spec-development-v2"
)
_VERSION = "pytest-8.3.5+pytest-xdist-3.8.0"
_FILES = (
    "prompt-template.md",
    "pytest-8.3.5.txt",
    "pytest-xdist-3.8.0.txt",
)
_MAX_INPUT_TOKENS = 64_000
_MAX_RESPONSE_BYTES = 65_536


def _clean_head() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("generation requires a clean committed checkout")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", _PROTOCOL_GIT_SHA, "HEAD"],
        cwd=_ROOT,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _committed_text(name: str) -> str:
    path = _SOURCES / name
    relative = path.relative_to(_ROOT).as_posix()
    expected = subprocess.run(
        ["git", "show", f"{_PROTOCOL_GIT_SHA}:{relative}"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if path.read_text(encoding="utf-8") != expected:
        raise ValueError(f"generation source changed: {name}")
    return expected


def generation_prompt() -> str:
    template, core, plugin = (_committed_text(name) for name in _FILES)
    documentation = (
        "PYTEST 8.3.5 HELP\n\n"
        + core
        + "\n\nPYTEST-XDIST 3.8.0 HELP\n\n"
        + plugin
    )
    return _render_generation_prompt(template, "pytest", _VERSION, documentation)


def _write_attempt(value: Mapping[str, Any]) -> None:
    (_OUTPUT / "attempt.json").write_text(
        json.dumps(value, indent=2) + "\n", encoding="utf-8"
    )


def _reserve_attempt(head: str) -> None:
    _OUTPUT.mkdir(parents=True, exist_ok=False)
    _write_attempt(
        {
            "schema": "pennylane-pytest-xdist-doc-attempt-v2",
            "status": "started",
            "protocol_git_sha": _PROTOCOL_GIT_SHA,
            "generation_git_sha": head,
        }
    )


def _copy_artifacts(source: Path) -> None:
    for path in source.iterdir():
        shutil.copy2(path, _OUTPUT / path.name)


def _response_matches_event() -> bool:
    response = (_OUTPUT / "pytest_xdist.response.json").read_text(encoding="utf-8")
    events = [
        json.loads(line)
        for line in (_OUTPUT / "pytest_xdist.events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    messages = [
        item["text"]
        for event in events
        if event.get("type") == "item.completed"
        and isinstance(item := event.get("item"), dict)
        and item.get("type") == "agent_message"
        and isinstance(item.get("text"), str)
    ]
    return messages == [response]


def _generation_status(
    response: Mapping[str, Any], cost: Mapping[str, Any]
) -> str:
    spec = validate_tool_spec(response)
    usage = cost["usage"]
    return (
        "valid"
        if spec is not None
        and spec.tool == "pytest"
        and spec.documented_version == _VERSION
        and usage["input_tokens"] <= _MAX_INPUT_TOKENS
        and (_OUTPUT / "pytest_xdist.response.json").stat().st_size
        <= _MAX_RESPONSE_BYTES
        and _response_matches_event()
        else "unsupported_structural_failure"
    )


def generate() -> None:
    if _OUTPUT.exists():
        raise ValueError("generation output already exists; the call cannot be repeated")
    head = _clean_head()
    prompt = generation_prompt()
    schema = tool_spec_schema()
    encoding = tiktoken.get_encoding("cl100k_base")
    estimated_tokens = len(encoding.encode(prompt)) + len(
        encoding.encode(json.dumps(schema, sort_keys=True))
    )
    if estimated_tokens > _MAX_INPUT_TOKENS:
        raise ValueError("generation prompt exceeds the frozen 64k-token budget")
    _reserve_attempt(head)
    try:
        directory_context = tempfile.TemporaryDirectory(
            prefix="pytest-xdist-spec-", dir="/tmp"
        )
        with directory_context as directory:
            temporary = Path(directory).resolve()
            if temporary.is_relative_to(_ROOT.resolve()):
                raise ValueError("generation directory is inside the repository")
            try:
                response, cost = _codex_call(
                    prompt, schema, temporary, "pytest_xdist"
                )
            finally:
                _copy_artifacts(temporary)
        status = _generation_status(response, cost)
    except BaseException as error:
        _write_attempt(
            {
                "schema": "pennylane-pytest-xdist-doc-attempt-v2",
                "status": "failed_consumed",
                "protocol_git_sha": _PROTOCOL_GIT_SHA,
                "generation_git_sha": head,
                "error_type": type(error).__name__,
                "error": str(error)[-2_000:],
            }
        )
        raise
    artifact = {
        "schema": "pennylane-pytest-xdist-doc-generation-v2",
        "protocol_git_sha": _PROTOCOL_GIT_SHA,
        "generation_git_sha": head,
        "tool": "pytest",
        "documented_version": _VERSION,
        "sources": [
            {
                "path": str((_SOURCES / name).relative_to(_ROOT)),
                "bytes": len(_committed_text(name).encode()),
            }
            for name in _FILES
        ],
        "model": "gpt-5.6-sol",
        "requested_service_tier": "fast",
        "reasoning_effort": "medium",
        "prediction_time_agent_calls": 0,
        "estimated_input_tokens": estimated_tokens,
        "status": status,
        "cost": cost,
    }
    (_OUTPUT / "generation-artifact.json").write_text(
        json.dumps(artifact, indent=2) + "\n", encoding="utf-8"
    )
    _write_attempt(
        {
            "schema": "pennylane-pytest-xdist-doc-attempt-v2",
            "status": "completed_consumed",
            "generation_status": status,
            "protocol_git_sha": _PROTOCOL_GIT_SHA,
            "generation_git_sha": head,
        }
    )
    print(
        json.dumps({"output": str(_OUTPUT), "status": status, "cost": cost}, indent=2)
    )


if __name__ == "__main__":
    generate()
