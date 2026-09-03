from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from trace_collect.openclaw_host_runtime import (
    OpenClawReplayProvider,
    ShadowGenerationConfig,
    _build_trace_replay_tools,
    shadow_generation_from_payload,
    shadow_generation_payload,
)
from trace_collect.simulate_openclaw import replay_trace_tools_enabled
from scripts.baselines.thunderagent_official_launcher import (
    _disable_backend_keepalive,
)


class _StreamResponse:
    def raise_for_status(self) -> None:
        pass

    async def aiter_lines(self):
        yield 'data: {"id":"shadow-1","prompt_token_ids":[11,12],"choices":[{"delta":{"token_ids":[101]},"finish_reason":"length"}]}'
        yield 'data: {"id":"shadow-1","choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}'
        yield "data: [DONE]"


class _Stream:
    async def __aenter__(self) -> _StreamResponse:
        return _StreamResponse()

    async def __aexit__(self, *_args: object) -> bool:
        return False


class _ReleaseResponse:
    def __init__(self, released: bool) -> None:
        self._released = released

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, bool]:
        return {"released": self._released}


class _Client:
    def __init__(self, *, released: bool = True, **_kwargs: object) -> None:
        self.released = released
        self.requests: list[tuple[str, str, dict[str, Any]]] = []
        self.release_calls: list[tuple[str, dict[str, Any]]] = []

    def stream(self, method: str, url: str, *, json: dict[str, Any]) -> _Stream:
        self.requests.append((method, url, json))
        return _Stream()

    async def post(self, url: str, *, json: dict[str, Any]) -> _ReleaseResponse:
        self.release_calls.append((url, json))
        return _ReleaseResponse(self.released)

    async def aclose(self) -> None:
        pass


def _source_action() -> dict[str, Any]:
    messages = [{"role": "user", "content": "unchanged workload"}]
    return {
        "action_type": "llm_call",
        "action_id": "llm-1",
        "_source_action_index": 0,
        "data": {
            "messages_in": messages,
            "completion_tokens": 1,
            "raw_response": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "source response"},
                    }
                ]
            },
        },
    }


def _provider(client: _Client) -> OpenClawReplayProvider:
    provider = OpenClawReplayProvider(
        llm_actions=[_source_action()],
        replay_speed=1.0,
        timing_mode="source_scaled",
        shadow_generation=ShadowGenerationConfig(
            api_base="http://127.0.0.1:9000/v1",
            model="test-model",
            timeout_s=12.0,
            seed=0,
            mode="thunderagent",
        ),
        program_id="task-a",
    )
    assert provider._shadow_client is not None
    asyncio.run(provider._shadow_client.aclose())
    provider._shadow_client = client
    return provider


def test_thunderagent_adapter_tags_request_and_releases_once() -> None:
    client = _Client()
    provider = _provider(client)

    asyncio.run(provider.chat([{"role": "user", "content": "ignored replay input"}]))
    asyncio.run(provider.aclose())
    asyncio.run(provider.aclose())

    method, url, payload = client.requests[0]
    assert (method, url) == (
        "POST",
        "http://127.0.0.1:9000/v1/chat/completions",
    )
    assert payload["program_id"] == "task-a"
    assert payload["messages"] == _source_action()["data"]["messages_in"]
    assert client.release_calls == [
        (
            "http://127.0.0.1:9000/programs/release",
            {"program_id": "task-a"},
        )
    ]


def test_thunderagent_release_failure_is_fatal() -> None:
    provider = _provider(_Client(released=False))

    asyncio.run(provider.chat([{"role": "user", "content": "ignored"}]))
    with pytest.raises(RuntimeError, match="did not release program"):
        asyncio.run(provider.aclose())


def test_thunderagent_backend_connections_are_not_reused() -> None:
    class Router:
        client = httpx.AsyncClient()

    router = Router()
    old_client = router.client
    asyncio.run(_disable_backend_keepalive(router))

    assert old_client.is_closed
    assert router.client._transport._pool._max_keepalive_connections == 0
    asyncio.run(router.client.aclose())


def test_thunderagent_mode_is_recorded_without_changing_plain_metadata() -> None:
    kwargs = {
        "api_base": "http://127.0.0.1:9000/v1",
        "model": "test-model",
        "timeout_s": 12.0,
        "seed": 0,
    }

    assert "thunderagent" not in shadow_generation_payload(
        ShadowGenerationConfig(**kwargs)
    )
    assert (
        shadow_generation_payload(
            ShadowGenerationConfig(**kwargs, mode="thunderagent")
        )["thunderagent"]
        is True
    )


def test_native_priority_marks_initial_then_return_requests() -> None:
    first = _source_action()
    second = json.loads(json.dumps(first))
    second["action_id"] = "llm-2"
    second["_source_action_index"] = 2
    client = _Client()
    config = ShadowGenerationConfig(
        api_base="http://127.0.0.1:8000/v1",
        model="test-model",
        timeout_s=12.0,
        seed=0,
        mode="native_priority",
    )
    provider = OpenClawReplayProvider(
        llm_actions=[first, second],
        replay_speed=1.0,
        timing_mode="source_scaled",
        shadow_generation=config,
    )
    assert provider._shadow_client is not None
    asyncio.run(provider._shadow_client.aclose())
    provider._shadow_client = client

    asyncio.run(provider.chat([]))
    asyncio.run(provider.chat([]))

    assert [request[2]["priority"] for request in client.requests] == [1, 0]
    assert shadow_generation_from_payload(shadow_generation_payload(config)) == config


def test_trace_tool_replay_checks_call_and_returns_recorded_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(
        "trace_collect.openclaw_host_runtime.asyncio.sleep", record_sleep
    )
    actions = [
        {
            "action_type": "tool_exec",
            "data": {
                "tool_name": "exec",
                "tool_call_id": "call-1",
                "tool_args": '{"command":"pytest -q"}',
                "tool_result": "recorded output",
                "duration_ms": 4000,
            },
        }
    ]
    tools, state = _build_trace_replay_tools(actions, replay_speed=4.0)
    tool = tools[0]
    tool.set_tool_call_context("call-1", {"command": "pytest -q"})

    assert asyncio.run(tool.execute(command="pytest -q")) == "recorded output"
    assert sleeps == [1.0]
    assert state.complete
    assert state.summary()["completed_calls"] == 1
    assert state.summary()["replay_speed"] == 4.0

    bad_tools, _ = _build_trace_replay_tools(actions, replay_speed=1.0)
    bad_tools[0].set_tool_call_context("call-1", {"command": "pytest tests/unit"})
    with pytest.raises(RuntimeError, match="arguments changed"):
        asyncio.run(bad_tools[0].execute(command="pytest tests/unit"))

    monkeypatch.setenv("OPENCLAW_REPLAY_TRACE_TOOLS", "1")
    assert replay_trace_tools_enabled() is True
    monkeypatch.setenv("OPENCLAW_REPLAY_TRACE_TOOLS", "invalid")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        replay_trace_tools_enabled()


def test_paper_baseline_runner_has_required_policy_checks() -> None:
    root = Path(__file__).parents[1]
    runner = root / "scripts/evaluation/run_paper_baseline.sh"
    subprocess.run(["bash", "-n", runner], check=True)
    runner_text = runner.read_text()
    assert 'export PATH="$HOME/.local/bin:$PATH"' in runner_text
    assert "\"$python\" -c 'import sklearn'" in runner_text
    assert 'cachewise_reproduction.sh" verify-installed' in runner_text
    assert 'summary["completed_traces"] == summary["attempted_traces"]' in runner_text
    assert 'summary["failed_traces"] == 0' in runner_text
    assert 'set +e\n    run_cell "$cell"\n    cell_rc=$?' in runner_text
    assert (
        'continuum_reproduction.sh" serve "$model" --dtype bfloat16 --kv-cache-dtype auto'
        in runner_text
    )
    assert (
        'continuum_reproduction.sh" serve-oracle-length-aging "$model" '
        "--dtype bfloat16 --kv-cache-dtype auto" in runner_text
    )
    assert "continuum_reproduction_oracle_length_aging" in runner_text
    assert 'GPU_MEMORY_UTILIZATION="$gpu_memory_utilization"' in runner_text
    assert "--shadow-llm-cachewise-predictor-checkout" in runner_text
    assert "cachewise-oracle-length" in runner_text
    assert "--shadow-llm-oracle-output-priority" in runner_text
    assert "CACHEWISE_ORACLE_LENGTH_EVENT_LOG" in runner_text
    assert "CACHEWISE_ORACLE_PREFILL_MS_PER_TOKEN" in runner_text
    assert "CACHEWISE_ORACLE_DECODE_MS_PER_TOKEN" in runner_text
    assert 'saga_reproduction.sh" verify' in runner_text
    assert 'saga_reproduction.sh" serve-backend' in runner_text
    assert 'native_priority_aging.sh" verify' in runner_text
    assert 'native_priority_aging.sh" serve' in runner_text
    assert 'cachewise_reproduction.sh" serve-disabled' in runner_text
    assert "shadow_mode=native-priority" in runner_text
    assert '--shadow-llm-saga-profile "$saga_profile"' in runner_text
    assert "--queue-upper-bounds 0.25,1,4,16" in runner_text
    assert 'proxy_launch+=(taskset -c "$vllm_cpuset")' in runner_text
    assert '>"$cell/proxy.argv"' in runner_text
    assert 'OPENCLAW_REPLAY_TRACE_TOOLS="$trace_tool_replay"' in runner_text
    assert 'OPENCLAW_REPLAY_TRACE_TOOL_SPEED="$trace_tool_replay_speed"' in runner_text
    assert '"tool_execution": (' in runner_text
    assert "shadow_llm_timeout_s=${SHADOW_LLM_TIMEOUT_S:-300}" in runner_text
    assert '--shadow-llm-timeout-s "$shadow_llm_timeout_s"' in runner_text
    assert '"shadow_llm_timeout_s": float(' in runner_text
    assert "expected_gpu_name=${EXPECTED_GPU_NAME:-A100}" in runner_text
    assert "min_gpu_memory_mib=${MIN_GPU_MEMORY_MIB:-80000}" in runner_text
    assert "gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.90}" in runner_text
    assert "memory_activity_pct" in runner_text
    assert "utilization.gpu,utilization.memory" in runner_text
    assert 'fail "GPU telemetry stopped early"' in runner_text
    assert 'raise SystemExit("GPU telemetry row is incomplete")' in runner_text
    assert "stage_all_before_replay=${STAGE_ALL_BEFORE_REPLAY:-1}" in runner_text
    assert "replacement_delay_mean_s=${REPLACEMENT_DELAY_MEAN_S:-}" in runner_text
    assert '--replacement-delay-mean-s "$replacement_delay_mean_s"' in runner_text
    assert '--replacement-seed "$replacement_seed"' in runner_text
    assert '"replacement_source": "same trace in a fresh container"' in runner_text
    assert "cleanup_images=${CLEANUP_IMAGES:-0}" in runner_text
    assert "simulate+=(--cleanup-images)" in runner_text
    assert "resource_monitoring=${RESOURCE_MONITORING:-off}" in runner_text
    assert '--resource-monitoring "$resource_monitoring"' in runner_text
    assert "serving_metrics=${SERVING_METRICS:-on}" in runner_text
    assert "--enable-prompt-tokens-details" in runner_text
    assert "--kv-events-config" in runner_text
    assert "--replay-endpoint tcp://127.0.0.1:5558" in runner_text
    assert "collect_vllm_kv_events.py" in runner_text
    assert "scripts.evaluation.cupti_dram_worker.CuptiDramWorker" in runner_text
    assert "CUPTI_DRAM_CSV" in runner_text
    assert '--ambient-caps="+$perfmon_cap"' in runner_text
    assert '--dram-bandwidth-csv "$cell/dram-bandwidth.csv"' in runner_text
    assert "collect_dcgm_metrics.py" not in runner_text
    assert "summarize_serving_metrics.py" in runner_text
    assert 'if os.environ["SERVING_METRICS"] == "on"' in runner_text
    assert '"mean_task_jct_min", "p95_task_jct_min", "tasks_per_hour"' in runner_text
    assert '"minimum_ratio_to_fcfs": 1.20' in runner_text


def test_gpu_setup_installs_cupti_without_dcgm() -> None:
    setup = (
        Path(__file__).parents[1] / "scripts/setup/benchmark_server.sh"
    ).read_text()

    assert "systemctl disable --now nvidia-dcgm" in setup
    assert "cupti-python==13.3.1" in setup
    assert "nvidia-cuda-cupti==13.3.75" in setup
    assert "cuda-bindings==13.3.1" in setup
    assert 'uv pip install --target "$CUPTI_DRAM_OVERLAY" --no-deps' in setup
    assert "systemctl --now enable nvidia-dcgm" not in setup


def test_cachewise_disabled_keeps_fork_config_without_policy_flags() -> None:
    source = (
        Path(__file__).parents[1] / "scripts/baselines/cachewise_reproduction.sh"
    ).read_text()
    body = source.split("serve_disabled() {", 1)[1].split("\n}\n", 1)[0]

    assert "--enable-prefix-caching" in body
    assert "--enable-chunked-prefill" in body
    assert "--max-num-batched-tokens 512" in body
    assert "--enable-cachewise-free-heap" not in body
    assert "--prioritize-waiting-by-prefix-cache" not in body


def test_baseline_runner_rejects_server_and_task_failures(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "scripts/evaluation/run_paper_baseline.sh"
    repo = tmp_path / "repo"
    runner = repo / "scripts/evaluation" / source.name
    runner.parent.mkdir(parents=True)
    runner.write_text(source.read_text())

    def executable(path: Path, body: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
        path.chmod(0o755)

    real_python = sys.executable
    executable(
        repo / ".venv/bin/python",
        f"""[[ -z ${{FAKE_PYTHON_CALLS:-}} ]] || printf '%s\\n' "$*" >>"$FAKE_PYTHON_CALLS"
if [[ ${{FAKE_NO_SKLEARN:-0}} == 1 && ${{1:-}} == -c && ${{2:-}} == 'import sklearn' ]]; then
  exit 42
fi
if [[ ${{1:-}} == -m ]]; then
  out=
  while (( $# )); do
    [[ $1 == --output-dir ]] && {{ out=$2; break; }}
    shift
  done
  mkdir -p "$out"
  failed=0
  [[ $out == *"/${{FAKE_FAILED_CELL:-never}}/output" ]] && failed=1
  "{real_python}" - "$out/throughput_summary.json" "$failed" <<'PY'
import json, sys
failed = int(sys.argv[2])
json.dump({{"attempted_traces": 1, "completed_traces": 1 - failed,
           "failed_traces": failed}}, open(sys.argv[1], "w"))
PY
  exit 0
fi
exec "{real_python}" "$@"
""",
    )
    executable(
        repo / ".venv/bin/vllm",
        """[[ ${FAKE_SERVER_FAIL:-0} == 0 ]] || exit 17
touch "$FAKE_READY"
trap 'rm -f "$FAKE_READY"; exit 0' TERM INT
while true; do sleep 1; done
""",
    )
    fake_bin = tmp_path / "bin"
    executable(fake_bin / "docker", "exit 0\n")
    executable(
        fake_bin / "curl",
        """if [[ $* == *:9000/* ]]; then
  [[ -f "$FAKE_PROXY_READY" ]]
else
  [[ -f "$FAKE_READY" ]]
fi
""",
    )
    executable(fake_bin / "timeout", "exit 1\n")
    executable(
        fake_bin / "git",
        """case "$*" in
  *"status --porcelain"*) exit 0 ;;
  *"rev-parse HEAD"*) echo deadbeef; exit 0 ;;
esac
exit 1
""",
    )
    executable(
        fake_bin / "nvidia-smi",
        """if [[ $* == *name,memory.total* ]]; then
  echo "${FAKE_GPU_ROW:-NVIDIA A100 80GB PCIe, 81920}"
else
  echo "${FAKE_GPU_VALUES:-40, 0, 0, 0}"
fi
""",
    )
    manifest = repo / "manifest.yaml"
    manifest.write_text("tasks: []\n")
    base_env = {
        **os.environ,
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "MODEL": "fake/model",
        "MANIFEST": str(manifest),
        "CONCURRENCY": "1",
        "CONTAINER_CPUS": "1",
        "CONTAINER_CPUSET": "",
        "VLLM_CPUSET": "",
        "TRACE_TOOL_REPLAY": "1",
        "TRACE_TOOL_REPLAY_SPEED": "4",
        "SERVING_METRICS": "off",
        "FAKE_READY": str(tmp_path / "ready"),
        "FAKE_PROXY_READY": str(tmp_path / "proxy-ready"),
    }

    summary_root = tmp_path / "summary-failure"
    summary_run = subprocess.run(
        ["bash", runner, "--run"],
        env={
            **base_env,
            "RUN_ROOT": str(summary_root),
            "CELLS": "fcfs-r1 fcfs-r2",
            "FAKE_FAILED_CELL": "fcfs-r1",
        },
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert summary_run.returncode == 1
    assert (summary_root / "fcfs-r1/cell-exit-code").read_text().strip() == "1"
    assert (summary_root / "fcfs-r2/cell-exit-code").read_text().strip() == "0"
    workload = json.loads((summary_root / "protocol.json").read_text())["workload"]
    assert "replacement_load" not in workload
    assert workload["tool_replay_speed"] == 4.0

    server_root = tmp_path / "server-failure"
    server_run = subprocess.run(
        ["bash", runner, "--run"],
        env={
            **base_env,
            "RUN_ROOT": str(server_root),
            "CELLS": "fcfs-r1",
            "FAKE_SERVER_FAIL": "1",
        },
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert server_run.returncode == 1
    assert (server_root / "fcfs-r1/cell-exit-code").read_text().strip() == "1"
    assert not (server_root / "fcfs-r1/simulate-exit-code").exists()

    telemetry_root = tmp_path / "telemetry-failure"
    telemetry_run = subprocess.run(
        ["bash", runner, "--run"],
        env={
            **base_env,
            "RUN_ROOT": str(telemetry_root),
            "CELLS": "fcfs-r1",
            "FAKE_GPU_VALUES": "40, 0, 0",
        },
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert telemetry_run.returncode == 1
    assert (telemetry_root / "fcfs-r1/simulate-exit-code").read_text().strip() == "0"
    assert (telemetry_root / "fcfs-r1/cell-exit-code").read_text().strip() == "1"
    assert "GPU telemetry row is incomplete" in telemetry_run.stderr

    thunder = repo / "scripts/baselines/thunderagent_official.sh"
    executable(
        thunder,
        """[[ $1 == verify ]] && exit 0
[[ $1 == serve ]] && exit 23
exit 2
""",
    )
    proxy_root = tmp_path / "proxy-failure"
    proxy_run = subprocess.run(
        ["bash", runner, "--run"],
        env={
            **base_env,
            "RUN_ROOT": str(proxy_root),
            "CELLS": "thunderagent-r1",
        },
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert proxy_run.returncode == 1
    assert (proxy_root / "thunderagent-r1/cell-exit-code").read_text().strip() == "1"
    assert not (proxy_root / "thunderagent-r1/simulate-exit-code").exists()

    models = tmp_path / "models"
    models.mkdir()
    (models / "all_models.pkl").touch()
    cachewise_run = subprocess.run(
        ["bash", runner, "--preflight"],
        env={
            **base_env,
            "RUN_ROOT": str(tmp_path / "cachewise-preflight"),
            "CELLS": "cachewise-r1",
            "CACHEWISE_MODELS_DIR": str(models),
            "FAKE_NO_SKLEARN": "1",
        },
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert cachewise_run.returncode == 1
    assert "CacheWise requires: uv sync --extra serving-spike" in cachewise_run.stderr

    oracle_run = subprocess.run(
        ["bash", runner, "--preflight"],
        env={
            **base_env,
            "RUN_ROOT": str(tmp_path / "oracle-preflight"),
            "CELLS": "cachewise-oracle-length-r1",
            "CACHEWISE_MODELS_DIR": str(models),
            "CACHEWISE_ORACLE_PREFILL_MS_PER_TOKEN": "",
            "CACHEWISE_ORACLE_DECODE_MS_PER_TOKEN": "",
        },
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert oracle_run.returncode == 1
    assert "CacheWise Oracle service coefficients must be positive" in oracle_run.stderr

    saga = repo / "scripts/baselines/saga_reproduction.sh"
    executable(
        saga,
        """printf '%s\\n' "$*" >>"$FAKE_SAGA_CALLS"
[[ $1 == verify ]] && exit 0
[[ $1 == serve-backend ]] && exec "$(dirname "$0")/../../.venv/bin/vllm"
exit 2
""",
    )
    missing_profile = subprocess.run(
        ["bash", runner, "--preflight"],
        env={
            **base_env,
            "RUN_ROOT": str(tmp_path / "saga-missing-profile"),
            "CELLS": "saga-r1",
            "SAGA_PROFILE": str(tmp_path / "missing-profile.json"),
            "FAKE_SAGA_CALLS": str(tmp_path / "saga-missing.calls"),
        },
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert missing_profile.returncode == 1
    assert "missing SAGA causal profile" in missing_profile.stderr

    profile = tmp_path / "saga-profile.json"
    profile.write_text("{}\n")
    saga_root = tmp_path / "saga-success"
    saga_calls = tmp_path / "saga.calls"
    python_calls = tmp_path / "python.calls"
    saga_run = subprocess.run(
        ["bash", runner, "--run"],
        env={
            **base_env,
            "RUN_ROOT": str(saga_root),
            "CELLS": "saga-r1",
            "SAGA_PROFILE": str(profile),
            "FAKE_SAGA_CALLS": str(saga_calls),
            "FAKE_PYTHON_CALLS": str(python_calls),
            "EXPECTED_GPU_NAME": "L40S",
            "MIN_GPU_MEMORY_MIB": "46000",
            "GPU_MEMORY_UTILIZATION": "0.95",
            "STAGE_ALL_BEFORE_REPLAY": "0",
            "REPLACEMENT_DELAY_MEAN_S": "50",
            "REPLACEMENT_SEED": "7",
            "FAKE_GPU_ROW": "NVIDIA L40S, 46068",
        },
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert saga_run.returncode == 0, saga_run.stderr
    assert saga_calls.read_text().splitlines() == [
        "verify",
        "serve-backend fake/model --host 127.0.0.1 --port 8000 "
        "--tensor-parallel-size 1 --gpu-memory-utilization 0.95 "
        "--max-model-len 131072 --max-num-seqs 8 --enable-prefix-caching "
        "--kv-cache-dtype auto --enforce-eager",
    ]
    simulate_call = next(
        line
        for line in python_calls.read_text().splitlines()
        if "trace_collect.cli" in line
    )
    assert "--shadow-llm-api-base http://127.0.0.1:8000/v1" in simulate_call
    assert "--shadow-llm-mode saga" in simulate_call
    assert f"--shadow-llm-saga-profile {profile}" in simulate_call
    assert "--replacement-delay-mean-s 50" in simulate_call
    assert "--replacement-seed 7" in simulate_call
    protocol = json.loads((saga_root / "protocol.json").read_text())
    assert protocol["saga_profile"] == str(profile)
    assert protocol["workload"]["expected_gpu_name"] == "L40S"
    assert protocol["workload"]["min_gpu_memory_mib"] == 46000
    assert protocol["workload"]["gpu_memory_utilization"] == 0.95
    assert protocol["workload"]["serving_metrics"] is False
    assert protocol["workload"]["replacement_load"]["delay_mean_s"] == 50.0
    assert protocol["workload"]["replacement_load"]["seed"] == 7
