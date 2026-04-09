from __future__ import annotations

import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_yaml_configs_load() -> None:
    for path in REPO_ROOT.glob("configs/**/*.yaml"):
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        assert data is not None, f"empty config: {path}"


def test_make_help_runs() -> None:
    result = subprocess.run(
        ["make", "help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "pull" in result.stdout
    assert "test" in result.stdout
    assert "run-smoke" in result.stdout


def test_make_sync_dry_run_documents_repo_local_env() -> None:
    result = subprocess.run(
        ["make", "-n", "sync"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "uv venv .venv" in result.stdout
    assert ".venv/bin/python" in result.stdout


def test_placeholder_scripts_fail_closed() -> None:
    for script in [
        "scripts/run_sweep.sh",
        "scripts/collect_results.sh",
        "scripts/serve_vllm.sh",
    ]:
        result = subprocess.run(
            [script],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, f"{script} unexpectedly succeeded"
