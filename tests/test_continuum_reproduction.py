import json
import math
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.baselines.continuum_reproduction import (
    CONTINUUM_COMMIT,
    PAPER_HISTORY_THRESHOLD,
    CacheMissProfile,
    RuntimeBinding,
    ToolCallEstimator,
    apply_to_checkout,
    cold_start_ttl_seconds,
    empirical_ttl_seconds,
    inference_manifest,
    memoryfulness,
    validate_runtime_binding,
    verify_checkout,
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
    assert CacheMissProfile.from_json(prefill, "prefill").estimate_seconds(2) == pytest.approx(0.011)

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


def test_wheel_overlay_scripts_are_isolated_and_non_editable(tmp_path: Path) -> None:
    public = (SCRIPT_DIR / "continuum_public.sh").read_text()
    reproduction = (SCRIPT_DIR / "continuum_reproduction.sh").read_text()
    assert "--editable" not in public + reproduction
    assert "vllm-continuum-public-" in public
    assert "vllm-continuum-reproduction-" in reproduction
    assert "venvs/continuum-public-" in public
    assert "venvs/continuum-reproduction-" in reproduction
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

    valid = validate("--dtype", "bfloat16", "--kv-cache-dtype=auto")
    assert valid.returncode == 0
    assert valid.stdout == "bfloat16|auto\n"
    built = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; venv=/isolated; '
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
    assert built[built.index("--max-model-len") + 1] == "32768"
    assert built[built.index("--dtype") + 1] == "bfloat16"
    assert built[built.index("--kv-cache-dtype") + 1] == "auto"
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
        / f"vllm-continuum-{CONTINUUM_COMMIT}"
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
        / f"vllm-continuum-{CONTINUUM_COMMIT}"
    )
    if not source.is_dir():
        pytest.skip("pinned Continuum checkout is not cached")
    checkout = tmp_path / "continuum"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", source, checkout], check=True
    )
    assert subprocess.run(
        ["git", "-C", checkout, "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout == ""
    apply_to_checkout(checkout)
    verify_checkout(checkout)
    assert (checkout / "vllm/v1/core/continuum_reproduction.py").is_file()
    assert "continuum_reproduction" in (
        checkout / "vllm/v1/core/sched/scheduler.py"
    ).read_text()


@pytest.mark.slow
def test_patch_rejects_polluted_or_staged_checkout(tmp_path: Path) -> None:
    source = (
        Path.home()
        / ".cache/agent-sched-bench"
        / f"vllm-continuum-{CONTINUUM_COMMIT}"
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
