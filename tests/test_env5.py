from __future__ import annotations

import json
import os
import subprocess
import types
from pathlib import Path

from harness.scheduler_hooks import (
    apply_scheduler_hook,
    build_report,
    parse_eviction_events,
    parse_prometheus_metrics,
    scheduler_log_snippet,
)
from harness.vllm_entrypoint_with_hooks import normalize_forwarded_args


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_engine_launcher_print_only_includes_preemption_flags() -> None:
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
            "--preemption-mode",
            "recompute",
            "--max-num-seqs",
            "256",
            "--print-only",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = json.loads(result.stdout)
    assert "--preemption-mode" in payload["command"]
    assert "--max-num-seqs" in payload["command"]


def test_engine_launcher_print_only_includes_scheduler_hook_flags() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        [
            "python3",
            "-m",
            "serving.engine_launcher",
            "--model-path",
            "/data/models/Llama-3.1-8B-Instruct",
            "--enable-scheduler-hook",
            "--scheduler-hook-report-path",
            "hook.json",
            "--print-only",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = json.loads(result.stdout)
    assert "harness.vllm_entrypoint_with_hooks" in " ".join(payload["command"])
    assert "--" in payload["command"]


def test_apply_scheduler_hook_fails_closed_when_symbol_missing(monkeypatch) -> None:
    monkeypatch.setattr("importlib.metadata.version", lambda _name: "0.8.1")

    def fake_import(_name: str) -> object:
        return types.SimpleNamespace()

    monkeypatch.setattr("importlib.import_module", fake_import)
    try:
        apply_scheduler_hook()
    except RuntimeError as exc:
        assert "Scheduler target not found" in str(exc)
    else:
        raise AssertionError("expected missing scheduler symbol to fail")


def test_parse_prometheus_metrics_extracts_preemption_fields() -> None:
    metrics_payload = "\n".join(
        [
            "vllm:num_preemptions_total 3",
            "vllm:gpu_cache_usage_perc 75.0",
            "vllm:cpu_cache_usage_perc 0.0",
            "vllm:gpu_prefix_cache_hit_rate 0.5",
            "vllm:cpu_prefix_cache_hit_rate 0.0",
        ]
    )
    snapshot = parse_prometheus_metrics(metrics_payload)
    assert snapshot.num_preemptions_total == 3.0
    assert snapshot.gpu_cache_usage_perc == 75.0
    assert snapshot.gpu_prefix_cache_hit_rate == 0.5


def test_parse_eviction_events_extracts_structured_records() -> None:
    events = parse_eviction_events(
        'INFO EVICT seq_id=req-1 tokens=128 reason=pressure gpu_usage=0.92\n'
    )
    assert len(events) == 1
    assert events[0].seq_id == "req-1"
    assert events[0].tokens == 128


def test_scheduler_log_snippet_mentions_eviction_fields() -> None:
    snippet = scheduler_log_snippet()
    assert "EVICT seq_id=" in snippet
    assert "gpu_usage" in snippet


def test_build_report_marks_evidence_scope_and_runtime_confirmation() -> None:
    report = build_report(
        metrics_payload="vllm:num_preemptions_total 1\n",
        log_text="INFO EVICT seq_id=req-1 tokens=128 reason=pressure gpu_usage=0.92\n",
    )
    assert report["metrics_fetch_ok"] is True
    assert report["scheduler_log_provided"] is True
    assert report["scheduler_hook_runtime_confirmed"] is True
    assert report["evidence_scope"] == "current_run_log"


def test_serve_vllm_wires_preemption_report_generation() -> None:
    script_text = (REPO_ROOT / "scripts" / "serve_vllm.sh").read_text(encoding="utf-8")
    assert "harness.scheduler_hooks" in script_text
    assert "VLLM_PREEMPTION_REPORT_PATH" in script_text
    assert "VLLM_SCHEDULER_HOOK_REPORT_PATH" in script_text


def test_vllm_hook_wrapper_strips_leading_sentinel() -> None:
    assert normalize_forwarded_args(["--", "--model", "demo"]) == ["--model", "demo"]
