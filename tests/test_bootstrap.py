from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_expected_top_level_paths_exist() -> None:
    expected = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "Makefile",
        REPO_ROOT / "docs" / "CURRENT_PLAN.md",
        REPO_ROOT / "configs" / "sweep.yaml",
        REPO_ROOT / "scripts" / "setup_server.sh",
        REPO_ROOT / "src" / "agents",
        REPO_ROOT / "src" / "harness",
        REPO_ROOT / "src" / "serving",
        REPO_ROOT / "src" / "analysis",
    ]
    for path in expected:
        assert path.exists(), f"missing required path: {path}"


def test_yaml_configs_load() -> None:
    for path in REPO_ROOT.glob("configs/**/*.yaml"):
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        assert data is not None, f"empty config: {path}"


def test_package_directories_import() -> None:
    for package in ["agents", "harness", "serving", "analysis"]:
        module = importlib.import_module(package)
        assert module.__doc__


def test_make_help_runs() -> None:
    result = subprocess.run(
        ["make", "help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "pull" in result.stdout
    assert "verify-bootstrap" in result.stdout
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


def test_primary_workload_configs_default_to_full_dataset() -> None:
    for config_name in ["code_agent"]:
        path = REPO_ROOT / "configs" / "workloads" / f"{config_name}.yaml"
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        assert data["sample_size"] is None
        assert data["sampling_strategy"] == "full_dataset"


def test_primary_workload_configs_use_expected_step_budgets() -> None:
    expected = {
        "code_agent": 80,
    }
    for config_name, max_steps in expected.items():
        path = REPO_ROOT / "configs" / "workloads" / f"{config_name}.yaml"
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        assert data["max_steps"] == max_steps
