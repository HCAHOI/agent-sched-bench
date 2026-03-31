from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from serving.health_check import validate_report


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_serve_vllm_script_is_valid_shell() -> None:
    subprocess.run(
        ["bash", "-n", "scripts/serve_vllm.sh"],
        cwd=REPO_ROOT,
        check=True,
    )


def test_engine_launcher_print_only_contract() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        [
            "python3",
            "-m",
            "serving.engine_launcher",
            "--model-path",
            "/data/models/Llama-3.1-8B-Instruct",
            "--enable-chunked-prefill",
            "--print-only",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = json.loads(result.stdout)
    assert payload["config"]["max_model_len"] == 32768
    assert "--enable-chunked-prefill" in payload["command"]


def test_health_check_help_runs() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    subprocess.run(
        ["python3", "-m", "serving.health_check", "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def test_make_help_mentions_verify_env3a() -> None:
    result = subprocess.run(
        ["make", "help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "verify-env3a" in result.stdout


def test_serve_vllm_invokes_fail_closed_health_check() -> None:
    script_text = (REPO_ROOT / "scripts" / "serve_vllm.sh").read_text(encoding="utf-8")
    assert "serving.health_check" in script_text
    assert "--fail-on-mismatch" in script_text


def test_health_check_validate_report_rejects_missing_acceptance_signals() -> None:
    report = {
        "models_response": {"data": []},
        "metrics_available": False,
        "chat_response": {"choices": [{"message": {"content": ""}}]},
    }
    errors = validate_report(report)
    assert "/v1/models returned an empty model list" in errors
    assert "/metrics did not expose any vllm-prefixed metrics" in errors
    assert "chat completion returned empty content" in errors
