from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_env4_scripts_are_valid_shell() -> None:
    for script in [
        "scripts/pull_repo.sh",
        "scripts/run_smoke.sh",
        "scripts/run_sweep.sh",
        "scripts/collect_results.sh",
    ]:
        subprocess.run(["bash", "-n", script], cwd=REPO_ROOT, check=True)


def test_run_sweep_fails_closed_without_harness_runner() -> None:
    result = subprocess.run(
        ["scripts/run_sweep.sh"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--output-root" in result.stderr


def test_collect_results_requires_results_source() -> None:
    result = subprocess.run(
        ["scripts/collect_results.sh"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Set RESULTS_SOURCE" in result.stderr


def test_run_smoke_can_run_an_explicit_subset() -> None:
    result = subprocess.run(
        ["scripts/run_smoke.sh", "tests/test_env1.py", "-q"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "2 passed" in result.stdout
