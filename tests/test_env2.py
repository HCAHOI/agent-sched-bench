from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_download_model_script_is_valid_shell() -> None:
    subprocess.run(
        ["bash", "-n", "scripts/setup/download_model.sh"],
        cwd=REPO_ROOT,
        check=True,
    )


def test_report_model_artifact_help_runs() -> None:
    subprocess.run(
        ["python3", "scripts/report_model_artifact.py", "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_make_help_mentions_verify_env2() -> None:
    result = subprocess.run(
        ["make", "help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "verify-env2" in result.stdout


def test_report_model_artifact_rejects_config_mode_for_acceptance() -> None:
    result = subprocess.run(
        [
            "python3",
            "scripts/report_model_artifact.py",
            "--output",
            "tmp.json",
            "--model-path",
            "missing-model",
            "--backend",
            "huggingface",
            "--model-repo",
            "meta-llama/Llama-3.1-8B-Instruct",
            "--modelscope-model",
            "LLM-Research/Meta-Llama-3.1-8B-Instruct",
            "--verify-load-mode",
            "config",
            "--transformers-spec",
            "transformers>=4.51,<5.0",
            "--hf-hub-spec",
            "huggingface_hub>=0.30,<1.0",
            "--modelscope-spec",
            "modelscope>=1.23,<2.0",
            "--fail-on-mismatch",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--verify-load-mode=full" in result.stderr


def test_download_model_writes_env_after_verification_call() -> None:
    script_text = (REPO_ROOT / "scripts" / "setup" / "download_model.sh").read_text(encoding="utf-8")
    verify_pos = script_text.index("  verify_model_artifact")
    write_env_pos = script_text.index("  write_model_path_env")
    assert verify_pos < write_env_pos
