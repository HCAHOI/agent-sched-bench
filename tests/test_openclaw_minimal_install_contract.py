from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from agents.openclaw.runtime_deps import (
    OPENCLAW_CONTAINER_RUNTIME_REQUIREMENTS,
    OPENCLAW_MCP_RUNTIME_REQUIREMENTS,
)


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


@pytest.mark.slow
def test_openclaw_minimal_container_requirements_import_runtime_entrypoints(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    wheel_dir = tmp_path / "wheelhouse"
    wheel_dir.mkdir()

    subprocess.run(
        [
            "uv",
            "build",
            "--wheel",
            "--out-dir",
            str(wheel_dir),
            str(repo_root),
        ],
        check=True,
    )
    wheels = sorted(wheel_dir.glob("agent_sched_bench-*.whl"))
    assert len(wheels) == 1

    venv_dir = tmp_path / "venv"
    subprocess.run(["uv", "venv", str(venv_dir)], check=True)
    python = _venv_python(venv_dir)
    requirements = (
        OPENCLAW_CONTAINER_RUNTIME_REQUIREMENTS + OPENCLAW_MCP_RUNTIME_REQUIREMENTS
    )
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--only-binary=:all:",
            *requirements,
        ],
        check=True,
    )
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            str(wheels[0]),
        ],
        check=True,
    )

    probe = "\n".join(
        [
            "import agents.openclaw._cli",
            "import trace_collect.runtime.entrypoint",
            "import agents.openclaw.eval.runner",
            "import agents.benchmarks",
            "import agents.openclaw.tools.mcp",
            "import mcp",
        ]
    )
    subprocess.run([str(python), "-c", probe], check=True)

