from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from serving.continuum_launcher import build_continuum_command, ContinuumServerConfig
from serving.health_check import build_chat_payload, validate_report


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_serve_continuum_script_is_valid_shell() -> None:
    subprocess.run(
        ["bash", "-n", "scripts/serve_continuum.sh"],
        cwd=REPO_ROOT,
        check=True,
    )


def test_continuum_launcher_print_only_contract() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        [
            "python3",
            "-m",
            "serving.continuum_launcher",
            "--model-path",
            "/data/models/Llama-3.1-8B-Instruct",
            "--print-only",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = json.loads(result.stdout)
    assert "--scheduling-policy" in payload["command"]
    assert "continuum" in payload["command"]


def test_build_continuum_command_includes_kv_transfer_config_when_enabled() -> None:
    config = ContinuumServerConfig(
        model_path="/data/models/Llama-3.1-8B-Instruct",
        enable_cpu_offload=True,
    )
    command = build_continuum_command(config)
    assert "--kv-transfer-config" in command


def test_health_payload_carries_program_id() -> None:
    payload = build_chat_payload(
        model="auto",
        messages=[{"role": "user", "content": "Reply with CONTINUUM."}],
        program_id="continuum-smoke",
    )
    assert payload["program_id"] == "continuum-smoke"


def test_make_help_mentions_verify_env3b() -> None:
    result = subprocess.run(
        ["make", "help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "verify-env3b" in result.stdout


def test_continuum_validate_report_requires_prefix_cache_hit_when_requested() -> None:
    report = {
        "models_response": {"data": [{"id": "model"}]},
        "metrics_available": True,
        "require_prefix_cache_hit": True,
        "pre_prefix_cache_hit_rates": {"gpu_prefix_cache_hit_rate": 0.0},
        "post_prefix_cache_hit_rates": {"gpu_prefix_cache_hit_rate": 0.0},
        "chat_responses": [{"choices": [{"message": {"content": "ok"}}]}],
    }
    errors = validate_report(report)
    assert "prefix cache hit rate did not increase during this run" in errors


def test_serve_continuum_requires_pinned_ref_and_prefix_cache_hit() -> None:
    script_text = (REPO_ROOT / "scripts" / "serve_continuum.sh").read_text(encoding="utf-8")
    assert "require_pinned_ref" in script_text
    assert "--require-prefix-cache-hit" in script_text
