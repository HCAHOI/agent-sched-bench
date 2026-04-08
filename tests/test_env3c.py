from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from serving.thunderagent_launcher import build_thunderagent_command, ThunderAgentConfig
from serving.thunderagent_check import validate_report


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_serve_thunderagent_script_is_valid_shell() -> None:
    subprocess.run(
        ["bash", "-n", "scripts/serve_thunderagent.sh"],
        cwd=REPO_ROOT,
        check=True,
    )


def test_thunderagent_launcher_print_only_contract() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        [
            "python3",
            "-m",
            "serving.thunderagent_launcher",
            "--backends",
            "http://127.0.0.1:8000",
            "--profile",
            "--metrics",
            "--print-only",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = json.loads(result.stdout)
    assert "--profile" in payload["command"]
    assert "--metrics" in payload["command"]


def test_build_thunderagent_command_contains_proxy_fields() -> None:
    command = build_thunderagent_command(
        ThunderAgentConfig(backends="http://127.0.0.1:8000", profile=True, metrics=True)
    )
    assert "--backend-type" in command
    assert "--backends" in command


def test_thunderagent_validate_report_requires_program_tracking_and_metrics() -> None:
    report = {
        "models_response": {"data": [{"id": "model"}]},
        "program_id": "pid",
        "pre_programs_response": {"pid": {"step_count": 1}},
        "programs_response": {},
        "profile_response": {},
        "metrics_response": {"metrics_enabled": False, "backends": {}},
        "chat_responses": [{"choices": [{"message": {"content": "ok"}}]}],
    }
    errors = validate_report(report)
    assert "program_id was not tracked in /programs" in errors
    assert "profile endpoint returned an empty payload" in errors
    assert "program step_count did not increase by at least 2 during this run" in errors
    assert "ThunderAgent /metrics reported metrics disabled" in errors


def test_serve_thunderagent_requires_immutable_ref() -> None:
    script_text = (REPO_ROOT / "scripts" / "serve_thunderagent.sh").read_text(
        encoding="utf-8"
    )
    assert "refs/tags/" in script_text
    assert "^[0-9a-f]{40}$" in script_text
