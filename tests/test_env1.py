from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_setup_server_script_is_valid_shell() -> None:
    subprocess.run(
        ["bash", "-n", "scripts/setup_server.sh"],
        cwd=REPO_ROOT,
        check=True,
    )


def test_report_server_env_writes_json(tmp_path: Path) -> None:
    output_path = tmp_path / "env_report.json"
    subprocess.run(
        [
            "python3",
            "scripts/report_server_env.py",
            "--output",
            str(output_path),
            "--repo-root",
            str(REPO_ROOT),
            "--venv-python",
            "python3",
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["repo_root"] == str(REPO_ROOT.resolve())
    assert report["config"]["expected_gpu_substring"] == "A100-PCIE-40GB"
    assert "commands" in report
    assert "memory_total_bytes" in report
    assert "disk_free_bytes" in report
    assert "venv_runtime" in report
