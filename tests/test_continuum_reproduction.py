import json
import math
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.baselines.continuum_reproduction import (
    CONTINUUM_COMMIT,
    ORACLE_TTFT_DEADLINE_SECONDS,
    PAPER_HISTORY_THRESHOLD,
    PreemptionTracker,
    write_request_telemetry,
    CacheMissProfile,
    RuntimeBinding,
    ToolCallEstimator,
    apply_to_checkout,
    cold_start_ttl_seconds,
    empirical_ttl_seconds,
    inference_manifest,
    install_trace_adapter,
    memoryfulness,
    select_oracle_deadline_request,
    validate_runtime_binding,
    verify_checkout,
    verify_trace_adapter,
)


SCRIPT_DIR = Path(__file__).parents[1] / "scripts/baselines"


def _runtime_binding_payload(model: str = "m") -> dict[str, object]:
    return {
        "model": model,
        "model_dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "gpu": {
            "name": "NVIDIA A100 80GB PCIe",
            "memory_mib": 81920,
            "count": 1,
        },
        "kv_layout": {
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "dtype": "bfloat16",
            "dtype_size": 2,
            "bytes_per_token_per_gpu": 131072,
        },
    }


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _Tokenizer:
    def decode(self, ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        return "tool" if ids else ""


class _Parser:
    def parse(self, text: str) -> str | None:
        return "pytest" if text == "tool" else None


def _request(job: str, *, last: bool, tokens: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        job_id=job,
        is_last_step=last,
        last_func_call=None,
        this_func_call=None,
        output_token_ids=[1],
        num_tokens=tokens,
    )


def test_published_ttl_equations_and_history_boundaries() -> None:
    assert cold_start_ttl_seconds(3.0) == pytest.approx(math.log(3.0))
    assert cold_start_ttl_seconds(1.0) == 0.0
    assert empirical_ttl_seconds([1.0, 4.0], 3.0) == 1.0
    assert PAPER_HISTORY_THRESHOLD == 100
    assert memoryfulness([(1, 2), (2, 1), (3, 0)]) == pytest.approx(1.0)


def test_causal_duration_queue_and_turn_updates_feed_eq2() -> None:
    clock = _Clock()
    estimator = ToolCallEstimator(
        tokenizer=_Tokenizer(),
        parser=_Parser(),
        cache_miss_profile=CacheMissProfile(
            mode="prefill",
            model="model",
            max_context_tokens=10,
            quadratic_seconds=(0.0, 0.0, 3.0),
        ),
        clock=clock,
    )
    first = _request("job", last=False)
    estimator.request_arrives(first)
    clock.now = 1.0
    estimator.request_finished(first)
    assert estimator.set_up_pin(first) == pytest.approx(math.log(3.0))

    clock.now = 5.0
    final = _request("job", last=True)
    estimator.request_arrives(final)
    assert estimator._global_durations == [4.0]
    clock.now = 5.5
    estimator.request_queue_paused(final)
    clock.now = 8.0  # Remote reload/FSM/LoRA stall: deliberately excluded.
    estimator.request_queue_resumed(final)
    clock.now = 8.5
    estimator.request_admitted(final, retained=False)
    assert estimator.average_evicted_queue_seconds == 1.0
    clock.now = 9.0
    estimator.request_finished(final)
    assert estimator.eta == pytest.approx(1.0)


def test_history_threshold_selects_global_then_per_tool_cdf() -> None:
    estimator = ToolCallEstimator(
        cache_miss_profile=CacheMissProfile(
            mode="prefill",
            model="model",
            max_context_tokens=10,
            quadratic_seconds=(0.0, 0.0, 3.0),
        )
    )
    request = _request("job", last=False)
    request.this_func_call = "pytest"
    estimator._global_durations = [2.0] * 101
    estimator._durations_by_tool["pytest"] = [1.0] * 100
    assert estimator.set_up_pin(request) == 2.0
    estimator._durations_by_tool["pytest"].append(1.0)
    assert estimator.set_up_pin(request) == 1.0


def test_replay_tool_signal_and_early_release_are_causal() -> None:
    class _UnexpectedParser:
        def parse(self, _text: str) -> str:
            raise AssertionError("source replay tool should bypass output parsing")

    estimator = ToolCallEstimator(tokenizer=_Tokenizer(), parser=_UnexpectedParser())
    request = _request("job", last=False)
    request.this_func_call = "pytest"
    estimator.request_arrives(request)
    estimator.request_finished(request)

    assert estimator._programs["job"].last_tool == "pytest"
    assert estimator.program_finished("job") is True
    assert estimator.program_finished("job") is False
    assert estimator._turn_pairs == [(1, 0)]
    assert "job" not in estimator._programs

    reused = _request("job", last=False)
    reused.this_func_call = "pytest"
    estimator.request_arrives(reused)
    estimator.request_finished(reused)
    assert estimator.program_finished("job") is True


def test_measured_prefill_and_reload_profiles(tmp_path: Path) -> None:
    prefill = tmp_path / "prefill.json"
    prefill.write_text(
        json.dumps(
            {
                "measurement": "prefill_recompute_cost",
                "model": "m",
                "runtime_binding": _runtime_binding_payload(),
                "quadratic_fit": {
                    "coefficients": [1.0, 2.0, 3.0],
                    "coefficient_order": [
                        "context_tokens^2",
                        "context_tokens",
                        "intercept",
                    ],
                },
                "points": [{"context_tokens": 10}],
            }
        )
    )
    assert CacheMissProfile.from_json(prefill, "prefill").estimate_seconds(
        2
    ) == pytest.approx(0.011)

    reload = tmp_path / "reload.json"
    reload.write_text(
        json.dumps(
            {
                "model": "m",
                "runtime_binding": _runtime_binding_payload(),
                "bytes_per_token": 131072,
                "measurements": [
                    {"tokens": 10, "bytes": 1310720, "swap_in_ms": 1000},
                    {"tokens": 20, "bytes": 2621440, "swap_in_ms": 1000},
                ],
            }
        )
    )
    assert CacheMissProfile.from_json(reload, "reload").estimate_seconds(
        10
    ) == pytest.approx(2 / 3)


def test_profile_requires_complete_runtime_binding(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"model": "m"}))
    with pytest.raises(ValueError, match="runtime_binding"):
        CacheMissProfile.from_json(profile, "prefill")


def test_runtime_binding_matches_model_gpu_tp_dtype_and_layout() -> None:
    binding = RuntimeBinding.from_payload(_runtime_binding_payload(), "profile")
    profile = CacheMissProfile(
        mode="prefill",
        model="m",
        max_context_tokens=10,
        quadratic_seconds=(0.0, 0.0, 1.0),
        runtime_binding=binding,
    )
    config = SimpleNamespace(
        torch_dtype="bfloat16",
        num_hidden_layers=32,
        num_key_value_heads=8,
        num_attention_heads=32,
        hidden_size=4096,
    )
    actual = {
        "model": "m",
        "model_dtype": "bfloat16",
        "kv_cache_dtype": "auto",
        "tensor_parallel_size": 1,
        "gpu_rows": [("NVIDIA A100 80GB PCIe", 81920)],
        "model_config": config,
    }
    validate_runtime_binding(profile, **actual)

    mutations = [
        {"model": "other"},
        {"model_dtype": "float16"},
        {"kv_cache_dtype": "fp8"},
        {"tensor_parallel_size": 2, "gpu_rows": [("NVIDIA A100 80GB PCIe", 81920)] * 2},
        {"gpu_rows": [("NVIDIA H100 80GB HBM3", 81920)]},
        {"model_config": replace(binding, num_hidden_layers=31)},
    ]
    for mutation in mutations:
        candidate = actual | mutation
        if isinstance(candidate["model_config"], RuntimeBinding):
            candidate["model_config"] = SimpleNamespace(
                torch_dtype="bfloat16",
                num_hidden_layers=candidate["model_config"].num_hidden_layers,
                num_key_value_heads=8,
                num_attention_heads=32,
                hidden_size=4096,
            )
        with pytest.raises(ValueError):
            validate_runtime_binding(profile, **candidate)


def test_inferred_choices_are_machine_readable() -> None:
    manifest = inference_manifest()
    assert manifest["paper"] == "arXiv:2511.02230v6"
    assert manifest["published"]["history_threshold"] == 100
    assert set(manifest["inferred_not_tuned"]) == {
        "queue_window",
        "queue_sample",
        "ttl_tie_break",
        "memoryfulness_samples",
        "cold_start_benefit",
        "reload_average",
    }


def _oracle_request(
    request_id: str,
    *,
    arrival: float,
    prompt: int,
    max_tokens: int,
    produced: int = 0,
    computed: int = 0,
    job_id: str | None = None,
) -> SimpleNamespace:
    item = SimpleNamespace(
        request_id=request_id,
        job_id=job_id or request_id,
        max_tokens=max_tokens,
        output_token_ids=[0] * produced,
        arrival_time=arrival,
        num_prompt_tokens=prompt,
        num_computed_tokens=computed,
        num_tokens=prompt + produced,
    )
    item.block_hashes = item
    return item


def _oracle_manager(cached: dict[str, int]) -> SimpleNamespace:
    return SimpleNamespace(
        coordinator=SimpleNamespace(
            find_longest_cache_hit=lambda request, _limit: (
                None,
                cached.get(request.request_id, 0),
            )
        )
    )


def _oracle_profile() -> CacheMissProfile:
    return CacheMissProfile(
        mode="prefill",
        model="model",
        max_context_tokens=500,
        quadratic_seconds=(0.0, 1.0, 0.0),
    )


def test_oracle_deadline_allows_only_predicted_safe_shortest() -> None:
    oldest = _oracle_request("oldest", arrival=1000.0, prompt=10, max_tokens=100)
    shortest = _oracle_request(
        "unpinned-shortest", arrival=1001.0, prompt=1, max_tokens=1
    )
    manager = _oracle_manager({})

    assert (
        select_oracle_deadline_request(
            [oldest, shortest],
            {oldest.job_id: 1000.0, shortest.job_id: 1001.0},
            manager,
            [],
            1,
            _oracle_profile(),
            1.0,
            now=1100.0,
        )
        is shortest
    )

    boundary_now = oldest.arrival_time + ORACLE_TTFT_DEADLINE_SECONDS - 13.0
    assert (
        select_oracle_deadline_request(
            [oldest, shortest],
            {oldest.job_id: 1000.0, shortest.job_id: 1001.0},
            manager,
            [],
            1,
            _oracle_profile(),
            1.0,
            now=boundary_now,
        )
        is oldest
    )
    assert (
        select_oracle_deadline_request(
            [oldest, shortest],
            {oldest.job_id: 1000.0, shortest.job_id: 1001.0},
            manager,
            [],
            1,
            _oracle_profile(),
            1.0,
            now=oldest.arrival_time + ORACLE_TTFT_DEADLINE_SECONDS + 1.0,
        )
        is oldest
    )

    running = _oracle_request(
        "running", arrival=999.0, prompt=1, max_tokens=1, computed=1
    )
    assert (
        select_oracle_deadline_request(
            [oldest, shortest],
            {oldest.job_id: 1000.0, shortest.job_id: 1001.0},
            manager,
            [running],
            3,
            _oracle_profile(),
            1.0,
            now=1299.0,
        )
        is shortest
    )


def test_oracle_deadline_uses_earliest_running_release() -> None:
    oldest = _oracle_request("oldest", arrival=1000.0, prompt=10, max_tokens=200)
    shortest = _oracle_request("shortest", arrival=1001.0, prompt=1, max_tokens=99)
    running = _oracle_request(
        "running", arrival=999.0, prompt=1, max_tokens=1, computed=1
    )
    manager = _oracle_manager({})

    assert (
        select_oracle_deadline_request(
            [oldest, shortest],
            {oldest.job_id: 1000.0, shortest.job_id: 1001.0},
            manager,
            [running],
            2,
            _oracle_profile(),
            1.0,
            now=1280.0,
        )
        is shortest
    )
    assert (
        select_oracle_deadline_request(
            [oldest, shortest],
            {oldest.job_id: 1000.0, shortest.job_id: 1001.0},
            manager,
            [],
            1,
            _oracle_profile(),
            1.0,
            now=1280.0,
        )
        is oldest
    )


def test_oracle_deadline_uses_remaining_output_and_stable_ties() -> None:
    progress = _oracle_request(
        "progress", arrival=3.0, prompt=10, max_tokens=10, produced=9
    )
    more_left = _oracle_request(
        "more-left", arrival=2.0, prompt=10, max_tokens=3, produced=1
    )
    tied_earlier_job = _oracle_request(
        "tied", arrival=4.0, prompt=10, max_tokens=4, produced=3
    )
    manager = _oracle_manager({"progress": 9, "more-left": 1, "tied": 3})

    assert (
        select_oracle_deadline_request(
            [progress, more_left],
            {progress.job_id: 1.0, more_left.job_id: 0.0},
            manager,
            [],
            1,
            _oracle_profile(),
            1.0,
            now=5.0,
        )
        is progress
    )
    assert (
        select_oracle_deadline_request(
            [progress, tied_earlier_job],
            {progress.job_id: 2.0, tied_earlier_job.job_id: 1.0},
            manager,
            [],
            1,
            _oracle_profile(),
            1.0,
            now=5.0,
        )
        is tied_earlier_job
    )

    invalid = _oracle_request(
        "invalid", arrival=5.0, prompt=10, max_tokens=1, produced=2
    )
    with pytest.raises(ValueError, match="output exceeds max_tokens"):
        select_oracle_deadline_request(
            [invalid],
            {invalid.job_id: 5.0},
            _oracle_manager({}),
            [],
            1,
            _oracle_profile(),
            1.0,
            now=5.0,
        )
    with pytest.raises(ValueError, match="output exceeds max_tokens"):
        select_oracle_deadline_request(
            [progress],
            {progress.job_id: 5.0},
            manager,
            [invalid],
            2,
            _oracle_profile(),
            1.0,
            now=5.0,
        )


def test_wheel_overlay_scripts_are_isolated_and_non_editable(tmp_path: Path) -> None:
    public = (SCRIPT_DIR / "continuum_public.sh").read_text()
    reproduction = (SCRIPT_DIR / "continuum_reproduction.sh").read_text()
    assert "--editable" not in public + reproduction
    assert "vllm-continuum-public-" in public
    assert "vllm-continuum-reproduction-v6-" in reproduction
    assert "venvs/continuum-public-" in public
    assert "venvs/continuum-reproduction-v6-" in reproduction
    assert 'continuum_python="${CONTINUUM_PYTHON:-$repo/.venv/bin/python}"' in public
    shared = tmp_path / "shared"
    rejected = subprocess.run(
        [SCRIPT_DIR / "continuum_reproduction.sh", "apply"],
        env={
            "HOME": str(Path.home()),
            "PATH": os.environ["PATH"],
            "CONTINUUM_CHECKOUT": str(shared),
            "CONTINUUM_REPRODUCTION_CHECKOUT": str(shared),
        },
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "must be distinct" in rejected.stderr


def test_reproduction_serve_rejects_config_overrides() -> None:
    script = SCRIPT_DIR / "continuum_reproduction.sh"

    def validate(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; shift; validate_serve_args "$@"; '
                'printf "%s|%s\\n" "$validated_model_dtype" "$validated_kv_dtype"',
                "bash",
                str(script),
                *arguments,
            ],
            capture_output=True,
            text=True,
        )

    valid = validate(
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype=auto",
        "--enable-prompt-tokens-details",
        "--worker-cls",
        "scripts.evaluation.cupti_dram_worker.CuptiDramWorker",
        "--kv-events-config",
        '{"enable_kv_cache_events":true}',
    )
    assert valid.returncode == 0
    assert valid.stdout == "bfloat16|auto\n"
    built = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; venv=/isolated; GPU_MEMORY_UTILIZATION=0.95; '
            "validated_observability_args=(--enable-prompt-tokens-details \
--worker-cls scripts.evaluation.cupti_dram_worker.CuptiDramWorker \
--kv-events-config '{\"enable_kv_cache_events\":true}'); "
            "build_serve_command model bfloat16 auto 1 32768 prefill; "
            'printf "%s\\n" "${serve_command[@]}"',
            "bash",
            str(script),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert built[:3] == ["/isolated/bin/vllm", "serve", "model"]
    assert built[built.index("--gpu-memory-utilization") + 1] == "0.95"
    assert built[built.index("--max-model-len") + 1] == "32768"
    assert built[built.index("--max-num-seqs") + 1] == "8"
    assert "--enable-prefix-caching" in built
    assert "--enforce-eager" in built
    assert built[built.index("--dtype") + 1] == "bfloat16"
    assert built[built.index("--kv-cache-dtype") + 1] == "auto"
    assert "--enable-prompt-tokens-details" in built
    assert built[built.index("--worker-cls") + 1] == (
        "scripts.evaluation.cupti_dram_worker.CuptiDramWorker"
    )
    assert built[built.index("--kv-events-config") + 1] == (
        '{"enable_kv_cache_events":true}'
    )
    for forbidden in ("--revision", "--quantization", "--hf-config-path"):
        invalid = validate(
            "--dtype",
            "bfloat16",
            "--kv-cache-dtype",
            "auto",
            forbidden,
            "override",
        )
        assert invalid.returncode != 0
        assert f"unsupported vLLM argument: {forbidden}" in invalid.stderr


@pytest.mark.slow
def test_overlay_manifest_is_exact_official_fork_python_delta() -> None:
    upstream = Path("/tmp/vllm-upstream-v0.10.2")
    fork = (
        Path.home()
        / ".cache/agent-sched-bench"
        / f"vllm-continuum-public-{CONTINUUM_COMMIT}"
    )
    if not upstream.is_dir() or not fork.is_dir():
        pytest.skip("upstream and pinned fork checkouts are not cached")
    subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; continuum_verify_fork_delta "$2" "$3"',
            "bash",
            str(SCRIPT_DIR / "continuum_public.sh"),
            str(upstream),
            str(fork),
        ],
        check=True,
    )


@pytest.mark.slow
def test_patch_applies_to_clean_pinned_checkout(tmp_path: Path) -> None:
    source = (
        Path.home()
        / ".cache/agent-sched-bench"
        / f"vllm-continuum-public-{CONTINUUM_COMMIT}"
    )
    if not source.is_dir():
        pytest.skip("pinned Continuum checkout is not cached")
    checkout = tmp_path / "continuum"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", source, checkout], check=True
    )
    assert (
        subprocess.run(
            ["git", "-C", checkout, "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == ""
    )
    apply_to_checkout(checkout)
    verify_checkout(checkout)
    assert (checkout / "vllm/v1/core/continuum_reproduction.py").is_file()
    assert (
        "continuum_reproduction"
        in (checkout / "vllm/v1/core/sched/scheduler.py").read_text()
    )
    scheduler = (checkout / "vllm/v1/core/sched/scheduler.py").read_text()
    assert "self.connector, self.running," in scheduler
    assert scheduler.count("self.waiting.remove_request(request)") >= 4


@pytest.mark.slow
def test_trace_replay_adapter_applies_exactly_to_public_overlay(tmp_path: Path) -> None:
    source = (
        Path.home()
        / ".cache/agent-sched-bench"
        / f"vllm-continuum-public-{CONTINUUM_COMMIT}"
        / "vllm"
    )
    if not source.is_dir():
        pytest.skip("pinned Continuum checkout is not cached")
    package = tmp_path / "vllm"
    for relative in (
        "entrypoints/openai/api_server.py",
        "v1/core/sched/scheduler.py",
        "v1/engine/core.py",
        "v1/core/estimate_with_func.py",
        "v1/metrics/stats.py",
        "v1/engine/output_processor.py",
    ):
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)

    install_trace_adapter(source, package, "public")
    verify_trace_adapter(source, package, "public")
    assert (
        '@router.post("/continuum/programs/release")'
        in (package / "entrypoints/openai/api_server.py").read_text()
    )
    assert (
        "this_func_call = request.this_func_call"
        in (package / "v1/core/estimate_with_func.py").read_text()
    )
    assert (
        "self.job_to_history.pop(job_id, None)"
        in (package / "v1/core/estimate_with_func.py").read_text()
    )

    scheduler = package / "v1/core/sched/scheduler.py"
    scheduler.write_text(scheduler.read_text() + "\n# pollution\n")
    with pytest.raises(ValueError, match="differs from expected"):
        verify_trace_adapter(source, package, "public")


@pytest.mark.slow
def test_trace_replay_adapter_adds_reproduction_oracle_deadline(
    tmp_path: Path,
) -> None:
    source = (
        Path.home()
        / ".cache/agent-sched-bench"
        / f"vllm-continuum-public-{CONTINUUM_COMMIT}"
        / "vllm"
    )
    if not source.is_dir():
        pytest.skip("pinned Continuum reproduction checkout is not cached")
    package = tmp_path / "vllm"
    for relative in (
        "entrypoints/openai/api_server.py",
        "v1/core/sched/scheduler.py",
        "v1/core/sched/request_queue.py",
        "v1/engine/core.py",
    ):
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)

    install_trace_adapter(source, package, "reproduction")
    verify_trace_adapter(source, package, "reproduction")

    request_queue = (package / "v1/core/sched/request_queue.py").read_text()
    assert 'os.environ.get("CONTINUUM_ORACLE_DEADLINE") == "1"' in request_queue
    assert "select_oracle_deadline_request" in request_queue
    assert "running_requests: list[Request]" in request_queue
    assert "max_num_running_reqs: int" in request_queue


@pytest.mark.slow
def test_patch_rejects_polluted_or_staged_checkout(tmp_path: Path) -> None:
    source = (
        Path.home()
        / ".cache/agent-sched-bench"
        / f"vllm-continuum-public-{CONTINUUM_COMMIT}"
    )
    if not source.is_dir():
        pytest.skip("pinned Continuum checkout is not cached")
    polluted = tmp_path / "polluted"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", source, polluted], check=True
    )
    (polluted / "unrelated.txt").write_text("untracked")
    with pytest.raises(ValueError, match="completely clean"):
        apply_to_checkout(polluted)

    staged = tmp_path / "staged"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", source, staged], check=True
    )
    readme = staged / "README.md"
    readme.write_text(readme.read_text() + "\nstaged pollution\n")
    subprocess.run(["git", "-C", staged, "add", "README.md"], check=True)
    with pytest.raises(ValueError, match="completely clean"):
        apply_to_checkout(staged)

    clean = tmp_path / "clean"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", source, clean], check=True
    )
    apply_to_checkout(clean)
    (clean / "extra.txt").write_text("unexpected")
    with pytest.raises(ValueError, match=r"expected only scheduler\+sidecar"):
        verify_checkout(clean)


def test_records_per_request_preemption_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = PreemptionTracker()
    tracker.scheduled(1.0)
    tracker.preempted(2.0)
    tracker.scheduled(5.0)
    tracker.preempted(7.0)
    tracker.scheduled(8.5)
    assert (tracker.count, tracker.total_wait_s, tracker.max_wait_s) == (2, 4.5, 3.0)

    path = tmp_path / "requests.jsonl"
    monkeypatch.setenv("VLLM_REQUEST_TELEMETRY_PATH", str(path))
    write_request_telemetry(
        "request-1", "length", 100, 8, 2.0, 1.0, 1.0, 3.0, 4.0, 5.5, tracker
    )
    assert json.loads(path.read_text()) == {
        "schema_version": 1,
        "request_id": "request-1",
        "finish_reason": "length",
        "prompt_tokens": 100,
        "generation_tokens": 8,
        "ttft_s": 2.0,
        "queue_s": 1.0,
        "prefill_s": 1.0,
        "decode_s": 3.0,
        "inference_s": 4.0,
        "e2e_s": 5.5,
        "preemption_count": 2,
        "preempted_wait_s": 4.5,
        "max_preempted_wait_s": 3.0,
        "preemption_timing_complete": True,
    }

    tracker.preempted(10.0)
    tracker.scheduled(9.0)
    assert tracker.timing_complete is False
