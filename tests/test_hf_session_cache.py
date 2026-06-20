"""Session-shared KV cache regression tests for `HFRecordingProvider`.

The provider keeps a single `BaseEvictionCache` alive for its lifetime so
H2O score buffers, streaming-LLM windows, and eviction state accumulate
across the agent's chat() loop. Three contract tests cover:

1. Legacy parity — one chat() call still produces the same output IDs the
   pre-session-cache path would have, given identical inputs.
2. Cross-call KV reuse — after two chat() calls, the cache contains the
   cumulative token sequence and the second call's delta is short.
3. H2O score accumulation — the score buffer grows monotonically across
   calls; no per-call reset.

We avoid spinning up a real HF model (LayerCapturer hard-wires Qwen3
q_norm/k_norm) and instead exercise the changed seams directly:
`_prepare_session_cache`, `_extend_session_tokens`, `notify_new_call`, plus
the H2OCache's bus observe() path.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from serving.kv_policies.base import EvictionPolicyConfig
from serving.kv_policies.h2o import H2OCache
from serving.recording.attention_bus import AttentionBus
from serving.recording.backend_hf import (
    HFRecordingProvider,
    HFRecordingServer,
    _generation_metadata,
    _generation_seed,
    _longest_common_prefix,
    _looks_like_malformed_tool_output,
    _synchronize_cuda_devices,
)
from serving.recording.recording import RecordingConfig
from agents.openclaw.providers.base import LLMResponse
from llm_call.openclaw import UnifiedProvider


# ---------------------------------------------------------------------------
# Fixtures: build an HFRecordingProvider without loading transformers.
# ---------------------------------------------------------------------------


class _StubModelConfig:
    num_hidden_layers = 2
    max_position_embeddings = 64


class _StubModel:
    def __init__(self) -> None:
        self.config = _StubModelConfig()


class _StubTokenizer:
    """Identity tokenizer: decode then encode round-trips token IDs unchanged."""

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        # Encode each id as a space-separated decimal string.
        return " ".join(str(i) for i in ids)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        if not text.strip():
            return []
        return [int(tok) for tok in text.split()]


class _StubCapturer:
    def __init__(self) -> None:
        self.started: list[Path] = []
        self.finish_calls: list[Path | None] = []
        self.closed = False

    def start_attempt(self, recordings_dir: Path) -> None:
        self.started.append(recordings_dir)

    def set_attempt_extra_meta(self, meta: dict) -> None:
        self.extra_meta = dict(meta)

    def finish_attempt(self, trace_path: Path | None = None) -> None:
        self.finish_calls.append(trace_path)

    def close(self) -> None:
        self.closed = True


def _build_provider(
    eviction_config: EvictionPolicyConfig | None,
    *,
    record_artifacts: bool = True,
) -> HFRecordingProvider:
    """Hand-construct a provider so we don't load HF weights."""
    provider = HFRecordingProvider.__new__(HFRecordingProvider)
    provider.config = RecordingConfig(record_artifacts=record_artifacts)
    provider._eviction_config = eviction_config
    provider._sparse_attention_config = None
    provider._sparse_attention = None
    provider._session_cache = None
    provider._session_token_ids = None
    provider._session_history = []
    provider._last_session_event = None
    provider._message_first_seen = []
    provider._attention_bus = AttentionBus()
    provider.model = _StubModel()
    provider.tokenizer = _StubTokenizer()
    provider._torch = torch
    provider.capturer = _StubCapturer()
    return provider


# ---------------------------------------------------------------------------
# LCP primitive.
# ---------------------------------------------------------------------------


def test_lcp_strict_prefix() -> None:
    a = torch.tensor([10, 20, 30], dtype=torch.long)
    b = torch.tensor([10, 20, 30, 40, 50], dtype=torch.long)
    assert _longest_common_prefix(a, b) == 3


def test_generation_seed_is_stable_per_call() -> None:
    assert _generation_seed(100, 0) == 100
    assert _generation_seed(100, 7) == 107


def test_synchronize_cuda_devices_targets_each_visible_device() -> None:
    class _FakeCuda:
        def __init__(self) -> None:
            self.synced: list[int] = []

        def device_count(self) -> int:
            return 2

        def synchronize(self, device_idx: int) -> None:
            self.synced.append(device_idx)

    fake_torch = SimpleNamespace(cuda=_FakeCuda())

    _synchronize_cuda_devices(fake_torch)

    assert fake_torch.cuda.synced == [0, 1]


def test_malformed_tool_output_detector_is_telemetry_only() -> None:
    assert _looks_like_malformed_tool_output("<function=read_file>\n<parameter=path>", [])
    assert not _looks_like_malformed_tool_output("Task complete.", [])
    assert not _looks_like_malformed_tool_output(
        "<function=read_file>", [SimpleNamespace(name="read_file")]
    )


def test_hf_recording_server_serializes_allowlisted_telemetry() -> None:
    class FakeProvider:
        default_model = "stub-model"

        async def chat(self, **kwargs) -> LLMResponse:
            return LLMResponse(
                content="ok",
                finish_reason="stop",
                usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                extra={
                    "hf_call_idx": 4,
                    "hf_cache_lcp": 123,
                    "hf_generation": {"output_tokens": 2},
                    "hf_unreviewed_future_field": "must-not-leak",
                    "llm_call_time_ms": 7.0,
                },
            )

    with HFRecordingServer(FakeProvider(), bind_host="127.0.0.1") as server:
        request = urllib.request.Request(
            f"{server.api_base}/chat/completions",
            data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode(
                "utf-8"
            ),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            body = json.loads(response.read().decode("utf-8"))

    assert body["choices"][0]["message"]["content"] == "ok"
    assert body["hf_telemetry"]["hf_call_idx"] == 4
    assert body["hf_telemetry"]["hf_cache_lcp"] == 123
    assert body["hf_telemetry"]["hf_generation"] == {"output_tokens": 2}
    assert "hf_unreviewed_future_field" not in body["hf_telemetry"]
    assert "llm_call_time_ms" not in body["hf_telemetry"]


def test_local_hf_server_round_trips_telemetry_through_unified_provider() -> None:
    class FakeProvider:
        default_model = "stub-model"

        async def chat(self, **kwargs) -> LLMResponse:
            return LLMResponse(
                content="ok",
                finish_reason="stop",
                usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                extra={
                    "hf_call_idx": 5,
                    "hf_cache_lcp": 456,
                    "hf_generation": {"output_tokens": 2},
                    "hf_unreviewed_future_field": "must-not-leak",
                },
            )

    async def drive() -> None:
        with HFRecordingServer(FakeProvider(), bind_host="127.0.0.1") as server:
            provider = UnifiedProvider(
                api_key="dummy",
                api_base=server.api_base,
                default_model="stub-model",
            )
            response = await provider.chat(
                [{"role": "user", "content": "hi"}],
                max_tokens=8,
            )

        assert response.content == "ok"
        assert response.extra["hf_call_idx"] == 5
        assert response.extra["hf_cache_lcp"] == 456
        assert response.extra["hf_generation"] == {"output_tokens": 2}
        assert "hf_unreviewed_future_field" not in response.extra

    asyncio.run(drive())


def test_generation_metadata_records_resolved_sampling_params() -> None:
    meta = _generation_metadata(
        seed=123,
        temperature=0.1,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.05,
    )

    assert meta == {
        "seed": 123,
        "do_sample": True,
        "temperature": 0.1,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.05,
    }


def test_lean_full_kv_provider_skips_layer_capturer_hooks(monkeypatch) -> None:
    """Plain local-HF full-KV should not install recording hooks."""
    from transformers import DynamicCache

    class FakeTokenizer:
        pass

    class FakeModel:
        config = _StubModelConfig()

        def eval(self) -> None:
            pass

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )

    def fail_layer_capturer(*args, **kwargs):
        raise AssertionError("LayerCapturer should not be built for lean full-KV")

    monkeypatch.setattr(
        "serving.recording.backend_hf.LayerCapturer",
        fail_layer_capturer,
    )

    provider = HFRecordingProvider(
        default_model="stub-model",
        config=RecordingConfig(record_artifacts=False),
        eviction_config=None,
        sparse_attention_config=None,
    )

    assert provider.capturer is None
    assert provider._attention_bus is None
    assert isinstance(provider._build_session_cache(), DynamicCache)


def test_recording_provider_builds_layer_capturer_when_recording_enabled(
    monkeypatch,
) -> None:
    """Recording-enabled local HF should still install LayerCapturer hooks."""
    seen: dict[str, object] = {}

    class FakeTokenizer:
        pass

    class FakeModel:
        config = _StubModelConfig()

        def eval(self) -> None:
            pass

    class FakeCapturer:
        def __init__(self, model, **kwargs) -> None:
            seen["capturer_model"] = model
            seen["capturer_kwargs"] = dict(kwargs)

        def set_kv_policy_meta(self, meta) -> None:
            seen["kv_policy_meta"] = meta

        def set_sparse_attention_meta(self, meta) -> None:
            seen["sparse_attention_meta"] = meta

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )
    monkeypatch.setattr(
        "serving.recording.backend_hf.LayerCapturer",
        FakeCapturer,
    )

    provider = HFRecordingProvider(
        default_model="stub-model",
        config=RecordingConfig(record_artifacts=True),
        eviction_config=None,
        sparse_attention_config=None,
    )

    assert isinstance(provider.capturer, FakeCapturer)
    assert provider._attention_bus is not None
    assert seen["capturer_kwargs"]["attention_bus"] is provider._attention_bus
    assert seen["capturer_kwargs"]["sparse_attention"] is None
    assert seen["kv_policy_meta"] is None
    assert seen["sparse_attention_meta"] is None


def test_model_summary_records_runtime_versions() -> None:
    provider = _build_provider(None)
    provider.default_model = "stub-model"
    provider.config = RecordingConfig()
    provider._captures_router_logits = False

    summary = provider._model_summary()

    assert summary["name"] == "stub-model"
    assert "torch_version" in summary
    assert "transformers_version" in summary
    assert "accelerate_version" in summary
    assert "torch_cuda_version" in summary
    assert "cuda_available" in summary
    assert "nvidia_driver_version" in summary
    assert "hf_model_commit_hash" in summary


def test_lcp_divergent() -> None:
    a = torch.tensor([10, 20, 99, 40], dtype=torch.long)
    b = torch.tensor([10, 20, 30, 40], dtype=torch.long)
    assert _longest_common_prefix(a, b) == 2


def test_lcp_empty() -> None:
    a = torch.zeros(0, dtype=torch.long)
    b = torch.tensor([1, 2, 3], dtype=torch.long)
    assert _longest_common_prefix(a, b) == 0


# ---------------------------------------------------------------------------
# Test 1: legacy parity — first call passes the full prompt to generate.
# ---------------------------------------------------------------------------


def test_single_chat_equivalent_to_legacy() -> None:
    """First call: prepare returns (full_prompt, True), cache freshly built.

    Pre-session-cache behavior was to build a fresh cache and pass the full
    prompt; the new path matches that on the first call. Regression gate
    against accidentally treating an empty cache as "has prefix" and
    truncating.
    """
    cfg = EvictionPolicyConfig(name="random", budget=8, seed=0)
    provider = _build_provider(cfg)
    prompt = torch.tensor([[5, 6, 7, 8, 9]], dtype=torch.long)
    delta, used = provider._prepare_session_cache(prompt_ids=prompt, call_idx=0)

    assert used is True
    torch.testing.assert_close(delta, prompt)
    # Cache built, token state mirrors the prompt verbatim.
    assert provider._session_cache is not None
    assert provider._session_token_ids is not None
    torch.testing.assert_close(provider._session_token_ids, prompt)
    assert provider._session_history == [
        {
            "call_idx": 0,
            "used_session_cache": True,
            "lcp": 0,
            "cached_len_before": 0,
            "new_len": 5,
            "delta_len": 5,
            "diverged": False,
        }
    ]
    assert provider._last_session_event is not None
    assert provider._last_session_event["cache_state_after"]["cache_type"] == "RandomEvictCache"
    assert provider._last_session_event["cache_state_after"]["logical_token_ids_len"] == 5


def test_no_eviction_still_builds_session_cache() -> None:
    """`eviction_config is None` now also gets a (plain DynamicCache) session
    cache so consecutive chat() calls can resume past_key_values via LCP delta
    prefill. This is the post-`perf(backend_hf)` decoupling — sparse and
    baseline runs benefit from the same cross-call KV reuse as eviction runs.
    """
    from transformers import DynamicCache

    from serving.kv_policies.base import BaseEvictionCache

    provider = _build_provider(eviction_config=None)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    delta, used = provider._prepare_session_cache(prompt_ids=prompt, call_idx=0)
    assert used is True
    torch.testing.assert_close(delta, prompt)
    assert isinstance(provider._session_cache, DynamicCache)
    # Plain DynamicCache — not an eviction subclass.
    assert not isinstance(provider._session_cache, BaseEvictionCache)
    assert provider._session_history == [
        {
            "call_idx": 0,
            "used_session_cache": True,
            "lcp": 0,
            "cached_len_before": 0,
            "new_len": 3,
            "delta_len": 3,
            "diverged": False,
        }
    ]


def test_env_var_disables_session_cache(monkeypatch) -> None:
    """OMC_DISABLE_SESSION_CACHE=1 short-circuits to the legacy path even with
    an eviction policy configured. This is the byte-equality validation escape
    hatch — default is enabled.
    """
    monkeypatch.setenv("OMC_DISABLE_SESSION_CACHE", "1")
    cfg = EvictionPolicyConfig(name="random", budget=8, seed=0)
    provider = _build_provider(cfg)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    delta, used = provider._prepare_session_cache(prompt_ids=prompt, call_idx=0)
    assert used is False
    torch.testing.assert_close(delta, prompt)
    assert provider._session_cache is None
    assert provider._session_history[-1]["used_session_cache"] is False


# ---------------------------------------------------------------------------
# Test 2: consecutive chats share KV state — delta on the second call.
# ---------------------------------------------------------------------------


def test_consecutive_chats_share_kv_state() -> None:
    """Two prepare()s. After call 1 ingests (prompt_1 + generated_1) the
    second call must pass only the delta beyond that cumulative sequence.
    """
    cfg = EvictionPolicyConfig(name="streaming", budget=16, sink_size=2, recent_window=14)
    provider = _build_provider(cfg)

    prompt_1 = torch.tensor([[1, 2, 3]], dtype=torch.long)
    delta_1, used_1 = provider._prepare_session_cache(prompt_ids=prompt_1, call_idx=0)
    assert used_1 is True
    torch.testing.assert_close(delta_1, prompt_1)

    # Simulate post-generate side effect from `_chat_locked`.
    generated_1 = [10, 11]
    provider._extend_session_tokens(prompt_ids=prompt_1, output_ids=generated_1)
    expected_state = torch.tensor([[1, 2, 3, 10, 11]], dtype=torch.long)
    torch.testing.assert_close(provider._session_token_ids, expected_state)

    # Second call: the agent re-includes prior assistant turn + new user turn,
    # so prompt_2 = prompt_1 + generated_1 + new tokens.
    prompt_2 = torch.tensor([[1, 2, 3, 10, 11, 20, 21, 22]], dtype=torch.long)
    delta_2, used_2 = provider._prepare_session_cache(prompt_ids=prompt_2, call_idx=1)
    assert used_2 is True
    # delta = prompt_2[lcp:] where lcp = len(expected_state) = 5
    torch.testing.assert_close(
        delta_2, torch.tensor([[20, 21, 22]], dtype=torch.long)
    )
    assert provider._session_history[-1] == {
        "call_idx": 1,
        "used_session_cache": True,
        "lcp": 5,
        "cached_len_before": 5,
        "new_len": 8,
        "delta_len": 3,
        "diverged": False,
        "resume_len": 5,
        "replayed_last_token": False,
    }
    # Cache instance is the SAME object — not rebuilt.
    assert provider._session_cache is not None
    cache_after = provider._session_cache
    delta_2_again, _ = provider._prepare_session_cache(
        prompt_ids=prompt_2, call_idx=1
    )
    assert provider._session_cache is cache_after
    torch.testing.assert_close(delta_2_again, delta_2)


def test_divergent_prompt_crops_and_prefills_suffix() -> None:
    """When LCP < cached_len, keep the valid prefix and prefill only suffix."""
    cfg = EvictionPolicyConfig(name="random", budget=8, seed=0)
    provider = _build_provider(cfg)
    prompt_1 = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    provider._prepare_session_cache(prompt_ids=prompt_1, call_idx=0)
    cache_before = provider._session_cache
    assert cache_before is not None

    # Divergence at position 2.
    prompt_2 = torch.tensor([[1, 2, 99, 5]], dtype=torch.long)
    delta, used = provider._prepare_session_cache(prompt_ids=prompt_2, call_idx=1)

    assert used is True
    torch.testing.assert_close(delta, torch.tensor([[99, 5]], dtype=torch.long))
    assert provider._session_cache is cache_before
    torch.testing.assert_close(
        provider._session_token_ids, torch.tensor([[1, 2]], dtype=torch.long)
    )
    assert provider._session_history[-1] == {
        "call_idx": 1,
        "used_session_cache": True,
        "lcp": 2,
        "cached_len_before": 4,
        "new_len": 4,
        "delta_len": 2,
        "diverged": True,
        "resume_len": 2,
        "replayed_last_token": False,
    }


def test_exact_match_replays_last_token_without_rebuild() -> None:
    cfg = EvictionPolicyConfig(name="random", budget=8, seed=0)
    provider = _build_provider(cfg)
    prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    provider._prepare_session_cache(prompt_ids=prompt, call_idx=0)
    cache_before = provider._session_cache

    delta, used = provider._prepare_session_cache(prompt_ids=prompt, call_idx=1)

    assert used is True
    torch.testing.assert_close(delta, torch.tensor([[4]], dtype=torch.long))
    assert provider._session_cache is cache_before
    torch.testing.assert_close(
        provider._session_token_ids, torch.tensor([[1, 2, 3]], dtype=torch.long)
    )
    assert provider._session_history[-1] == {
        "call_idx": 1,
        "used_session_cache": True,
        "lcp": 4,
        "cached_len_before": 4,
        "new_len": 4,
        "delta_len": 1,
        "diverged": False,
        "resume_len": 3,
        "replayed_last_token": True,
    }


def test_exact_match_sparse_eviction_cache_replays_last_logical_token() -> None:
    cfg = EvictionPolicyConfig(name="random", budget=8, seed=0)
    provider = _build_provider(cfg)
    prompt = torch.arange(100, dtype=torch.long).unsqueeze(0)
    cache = provider._build_session_cache()
    assert cache is not None
    keys = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4, 1)
    values = keys + 100
    cache.update(keys, values, layer_idx=0)
    cache._logical_indices_by_layer[0] = [0, 1, 97, 99]
    cache._next_logical_by_layer[0] = 100
    provider._session_cache = cache
    provider._session_token_ids = prompt.clone()

    delta, used = provider._prepare_session_cache(prompt_ids=prompt, call_idx=1)

    assert used is True
    torch.testing.assert_close(delta, torch.tensor([[99]], dtype=torch.long))
    assert provider._session_cache is cache
    assert int(cache.get_seq_length(0)) == 3
    assert cache._logical_indices_by_layer[0] == [0, 1, 97]
    torch.testing.assert_close(provider._session_token_ids, prompt[:, :99])
    assert provider._session_history[-1] == {
        "call_idx": 1,
        "used_session_cache": True,
        "lcp": 100,
        "cached_len_before": 100,
        "new_len": 100,
        "delta_len": 1,
        "diverged": False,
        "resume_len": 99,
        "replayed_last_token": True,
    }


def test_plain_dynamic_cache_partial_divergence_crops_physical_cache() -> None:
    provider = _build_provider(eviction_config=None)
    prompt_1 = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    provider._prepare_session_cache(prompt_ids=prompt_1, call_idx=0)
    assert provider._session_cache is not None
    keys = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4, 1)
    values = keys + 100
    provider._session_cache.update(keys, values, 0)
    provider._extend_session_tokens(prompt_ids=prompt_1, output_ids=[])

    prompt_2 = torch.tensor([[1, 2, 99, 5]], dtype=torch.long)
    delta, used = provider._prepare_session_cache(prompt_ids=prompt_2, call_idx=1)

    assert used is True
    torch.testing.assert_close(delta, torch.tensor([[99, 5]], dtype=torch.long))
    assert int(provider._session_cache.get_seq_length(0)) == 2
    torch.testing.assert_close(
        provider._session_token_ids, torch.tensor([[1, 2]], dtype=torch.long)
    )


def test_start_attempt_resets_session_cache_between_attempts(tmp_path: Path) -> None:
    """A provider can span tasks, but KV state must not."""
    cfg = EvictionPolicyConfig(name="streaming", budget=8, sink_size=1, recent_window=7)
    provider = _build_provider(cfg)
    provider.capturer = _StubCapturer()

    provider.start_attempt(tmp_path / "attempt_1" / "recordings")
    prompt_1 = torch.tensor([[1, 2, 3]], dtype=torch.long)
    provider._prepare_session_cache(prompt_ids=prompt_1, call_idx=0)
    provider._extend_session_tokens(prompt_ids=prompt_1, output_ids=[10])
    assert provider._session_cache is not None
    assert provider._session_token_ids is not None

    provider.start_attempt(tmp_path / "attempt_2" / "recordings")
    assert provider._session_cache is None
    assert provider._session_token_ids is None
    assert provider._session_history == []
    assert provider._last_session_event is None
    assert provider._message_first_seen == []

    prompt_2 = torch.tensor([[1, 2, 3, 10, 11]], dtype=torch.long)
    delta, used = provider._prepare_session_cache(prompt_ids=prompt_2, call_idx=0)

    assert used is True
    torch.testing.assert_close(delta, prompt_2)
    assert provider._session_history == [
        {
            "call_idx": 0,
            "used_session_cache": True,
            "lcp": 0,
            "cached_len_before": 0,
            "new_len": 5,
            "delta_len": 5,
            "diverged": False,
        }
    ]


def test_record_artifacts_false_suppresses_attempt_recording_tree(
    tmp_path: Path,
) -> None:
    """No-internals KV runs must not leave recordings/meta.json behind."""
    cfg = EvictionPolicyConfig(name="metadata", budget=8, sink_size=1, recent_window=7)
    provider = _build_provider(cfg, record_artifacts=False)

    class _ArtifactCapturer(_StubCapturer):
        def __init__(self) -> None:
            super().__init__()
            self.recordings_dir: Path | None = None

        def start_attempt(self, recordings_dir: Path) -> None:
            super().start_attempt(recordings_dir)
            self.recordings_dir = recordings_dir
            recordings_dir.mkdir(parents=True, exist_ok=True)

        def finish_attempt(self, trace_path: Path | None = None) -> None:
            super().finish_attempt(trace_path)
            assert self.recordings_dir is not None
            (self.recordings_dir / "meta.json").write_text("{}", encoding="utf-8")

    provider.capturer = _ArtifactCapturer()
    recordings_dir = tmp_path / "attempt" / "recordings"

    provider.start_attempt(recordings_dir)
    provider.finish_attempt()

    assert provider.capturer.started == []
    assert provider.capturer.finish_calls == []
    assert not recordings_dir.exists()


def test_provider_exit_defensively_finishes_attempt() -> None:
    provider = _build_provider(None)
    capturer = provider.capturer

    provider.__exit__(None, None, None)

    assert capturer.finish_calls == [None]
    assert capturer.closed is True


def test_hf_recording_provider_rejects_concurrent_chat() -> None:
    async def drive() -> None:
        provider = HFRecordingProvider.__new__(HFRecordingProvider)
        provider._chat_lock = threading.Lock()
        provider.generation = SimpleNamespace(
            top_p=None,
            top_k=None,
            repetition_penalty=None,
        )
        entered = asyncio.Event()
        release = asyncio.Event()

        async def fake_chat_locked(**_kwargs):
            entered.set()
            await release.wait()
            return LLMResponse(content="ok", finish_reason="stop")

        provider._chat_locked = fake_chat_locked

        first = asyncio.create_task(provider.chat(messages=[]))
        await entered.wait()
        second = await provider.chat(messages=[])
        release.set()
        first_response = await first

        assert first_response.content == "ok"
        assert second.finish_reason == "error"
        assert second.extra["error_type"] == "concurrent_request"

    asyncio.run(drive())


def test_hf_recording_provider_wait_until_idle_timeout() -> None:
    provider = HFRecordingProvider.__new__(HFRecordingProvider)
    provider._chat_lock = threading.Lock()
    provider._chat_lock.acquire()

    try:
        with pytest.raises(
            TimeoutError,
            match="timed out waiting for HF recording provider to become idle",
        ):
            provider.wait_until_idle(timeout_s=0.001)
    finally:
        provider._chat_lock.release()


def test_hf_recording_provider_wait_until_idle_blocks_until_release() -> None:
    class _ObservedLock:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self.acquire_entered = threading.Event()

        def hold(self) -> None:
            self._lock.acquire()

        def acquire(self, *args, **kwargs) -> bool:
            self.acquire_entered.set()
            return self._lock.acquire(*args, **kwargs)

        def release(self) -> None:
            self._lock.release()

    provider = HFRecordingProvider.__new__(HFRecordingProvider)
    observed_lock = _ObservedLock()
    observed_lock.hold()
    provider._chat_lock = observed_lock
    done = threading.Event()
    errors: list[BaseException] = []

    def wait_for_idle() -> None:
        try:
            provider.wait_until_idle(timeout_s=1.0)
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    waiter = threading.Thread(target=wait_for_idle)
    waiter.start()
    assert observed_lock.acquire_entered.wait(timeout=1.0)
    assert not done.is_set()
    time.sleep(0.01)
    assert not done.is_set()

    observed_lock.release()
    waiter.join(timeout=1.0)

    assert done.is_set()
    assert errors == []


def test_message_first_seen_tracks_appends_and_replacements() -> None:
    """Message provenance survives appended turns and resets changed turns."""
    provider = _build_provider(eviction_config=None)

    first = provider._first_seen_calls_for_messages(
        [{"role": "user", "content": "a"}],
        call_idx=0,
    )
    second = provider._first_seen_calls_for_messages(
        [
            {"role": "user", "content": "a"},
            {"role": "tool", "content": "result"},
        ],
        call_idx=1,
    )
    replaced = provider._first_seen_calls_for_messages(
        [{"role": "user", "content": "changed"}],
        call_idx=2,
    )

    assert first == {0: 0}
    assert second == {0: 0, 1: 1}
    assert replaced == {0: 2}


def test_message_first_seen_survives_context_window_snip() -> None:
    provider = _build_provider(eviction_config=None)

    provider._first_seen_calls_for_messages(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "a"},
            {"role": "tool", "content": "result"},
        ],
        call_idx=1,
    )
    snipped = provider._first_seen_calls_for_messages(
        [
            {"role": "system", "content": "sys"},
            {"role": "tool", "content": "result"},
        ],
        call_idx=3,
    )

    assert snipped == {0: 1, 1: 1}


def test_message_first_seen_distinguishes_repeated_messages() -> None:
    provider = _build_provider(eviction_config=None)

    provider._first_seen_calls_for_messages(
        [{"role": "user", "content": "continue"}],
        call_idx=0,
    )
    repeated = provider._first_seen_calls_for_messages(
        [
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "continue"},
        ],
        call_idx=5,
    )
    snipped_latest = provider._first_seen_calls_for_messages(
        [{"role": "user", "content": "continue"}],
        call_idx=6,
    )

    assert repeated == {0: 0, 1: 5, 2: 5}
    assert snipped_latest == {0: 5}


def test_message_first_seen_uses_openclaw_message_ids_for_duplicate_snips() -> None:
    provider = _build_provider(eviction_config=None)

    provider._first_seen_calls_for_messages(
        [{"role": "user", "content": "continue", "_openclaw_message_id": "m0"}],
        call_idx=0,
    )
    full = provider._first_seen_calls_for_messages(
        [
            {"role": "user", "content": "continue", "_openclaw_message_id": "m0"},
            {"role": "assistant", "content": "ok", "_openclaw_message_id": "m1"},
            {"role": "user", "content": "continue", "_openclaw_message_id": "m2"},
            {"role": "assistant", "content": "hmm", "_openclaw_message_id": "m3"},
            {"role": "user", "content": "continue", "_openclaw_message_id": "m4"},
        ],
        call_idx=5,
    )
    snipped = provider._first_seen_calls_for_messages(
        [
            {"role": "assistant", "content": "ok", "_openclaw_message_id": "m1"},
            {"role": "user", "content": "continue", "_openclaw_message_id": "m2"},
        ],
        call_idx=6,
    )

    assert full == {0: 0, 1: 5, 2: 5, 3: 5, 4: 5}
    assert snipped == {0: 5, 1: 5}


def test_context_manager_releases_provider_refs() -> None:
    provider = _build_provider(eviction_config=None)

    with provider as active:
        assert active is provider

    assert provider.model is None
    assert provider.tokenizer is None


# ---------------------------------------------------------------------------
# Test 3: H2O score buffer accumulates across calls.
# ---------------------------------------------------------------------------


def _attn_with_peak(*, key_len: int, peak_pos: int, value: float = 1.0) -> torch.Tensor:
    """(B=1, H=2, Q=1, K=key_len) attn tensor with mass concentrated at peak."""
    attn = torch.zeros(1, 2, 1, key_len, dtype=torch.float32)
    attn[..., peak_pos] = value
    return attn


def test_h2o_crop_to_logical_prefix_compacts_score_buffer() -> None:
    cfg = EvictionPolicyConfig(
        name="h2o",
        budget=8,
        sink_size=1,
        recent_window=2,
        aggregate="sum",
        prefill_mode="sampled",
    )
    bus = AttentionBus()
    cache = H2OCache(
        cfg,
        num_layers=1,
        attention_bus=bus,
        max_position_embeddings=16,
    )
    keys = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4, 1)
    values = keys + 100
    cache.update(keys, values, layer_idx=0)
    cache._logical_indices_by_layer[0] = [0, 2, 4, 6]
    cache._next_logical_by_layer[0] = 7
    score_buffer = torch.zeros(16, dtype=torch.float32)
    score_buffer[:4] = torch.tensor([10.0, 20.0, 30.0, 40.0])
    cache._scores[0] = score_buffer
    cache._score_lengths[0] = 4

    cache.crop_to_logical_length(5)

    assert int(cache.get_seq_length(0)) == 3
    assert cache._logical_indices_by_layer[0] == [0, 2, 4]
    assert cache._next_logical_by_layer[0] == 5
    assert cache._score_lengths[0] == 3
    torch.testing.assert_close(
        cache._scores[0][:6],
        torch.tensor([10.0, 20.0, 30.0, 0.0, 0.0, 0.0]),
    )


def test_h2o_score_buffer_accumulates_across_calls() -> None:
    """Two simulated chat() calls observe attention into the same H2OCache.

    `notify_new_call` resets per-call step counters but the score buffer
    persists. After call 2, every observed key position has score ≥ its
    snapshot from end-of-call-1.
    """
    cfg = EvictionPolicyConfig(
        name="h2o",
        budget=8,
        sink_size=1,
        recent_window=2,
        aggregate="sum",
        prefill_mode="sampled",
    )
    bus = AttentionBus()
    cache = H2OCache(
        cfg,
        num_layers=1,
        attention_bus=bus,
        max_position_embeddings=64,
    )

    # ---- Call 1: prefill across positions 0..3, then decode at 4..5 -------
    cache.notify_new_call(call_idx=0)
    for pos in range(4):
        bus.publish(
            layer=0,
            attn=_attn_with_peak(key_len=pos + 1, peak_pos=pos),
            query_positions=torch.tensor([pos], dtype=torch.long),
            key_len=pos + 1,
            phase="prefill",
            suspended=False,
        )
    snapshot_1 = cache._scores[0].clone()
    assert snapshot_1 is not None
    # Each of positions 0..3 saw mass=1 once; 4..63 still zero.
    assert float(snapshot_1[0]) > 0
    assert float(snapshot_1[3]) > 0
    assert float(snapshot_1[4]) == 0.0

    # ---- Call 2: notify boundary, then more observations -----------------
    cache.notify_new_call(call_idx=1)
    # Counter state cleared, score buffer untouched.
    assert cache._step_counter == {}
    assert cache._seen_prefill == set()
    # Observe new positions 4..5 with strong peaks.
    for pos in range(4, 6):
        bus.publish(
            layer=0,
            attn=_attn_with_peak(key_len=pos + 1, peak_pos=pos, value=5.0),
            query_positions=torch.tensor([pos], dtype=torch.long),
            key_len=pos + 1,
            phase="prefill",
            suspended=False,
        )
    snapshot_2 = cache._scores[0].clone()
    # Live prefix at least covers up to position 5.
    assert cache._score_lengths[0] >= 6
    # Monotone non-decrease vs snapshot_1 across positions 0..5.
    for pos in range(6):
        assert float(snapshot_2[pos]) >= float(snapshot_1[pos]), (
            f"score at pos {pos} regressed: "
            f"{float(snapshot_2[pos])} < {float(snapshot_1[pos])}"
        )
    # Strict growth at the newly-observed positions.
    assert float(snapshot_2[4]) > float(snapshot_1[4])
    assert float(snapshot_2[5]) > float(snapshot_1[5])


def test_h2o_notify_new_call_does_not_unsubscribe() -> None:
    """notify_new_call leaves the bus subscription intact (subscription is
    a provider-lifetime concern, not a per-call one).
    """
    cfg = EvictionPolicyConfig(name="h2o", budget=4, sink_size=1, recent_window=1)
    bus = AttentionBus()
    cache = H2OCache(cfg, num_layers=1, attention_bus=bus, max_position_embeddings=16)
    assert bus.n_consumers() == 1
    cache.notify_new_call(call_idx=42)
    assert bus.n_consumers() == 1


# ---------------------------------------------------------------------------
# Provider lifecycle: close() / __del__ drop the bus subscription.
# ---------------------------------------------------------------------------


def test_close_unsubscribes_h2o_session_cache() -> None:
    cfg = EvictionPolicyConfig(name="h2o", budget=4, sink_size=1, recent_window=1)
    provider = _build_provider(cfg)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    provider._prepare_session_cache(prompt_ids=prompt, call_idx=0)
    assert provider._attention_bus.n_consumers() == 1

    provider.close()
    assert provider._attention_bus.n_consumers() == 0
    assert provider._session_cache is None
    assert provider._session_token_ids is None


def test_close_idempotent() -> None:
    cfg = EvictionPolicyConfig(name="random", budget=4, seed=0)
    provider = _build_provider(cfg)
    provider.close()  # no cache built yet — must not raise.
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    provider._prepare_session_cache(prompt_ids=prompt, call_idx=0)
    provider.close()
    provider.close()  # second close is a no-op.


# ---------------------------------------------------------------------------
# _extend_session_tokens stores raw output_ids; LCP resume handles re-render drift.
# ---------------------------------------------------------------------------


def test_session_extends_with_raw_output_ids() -> None:
    """_extend_session_tokens stores prompt_ids + raw output_ids.

    The raw output_ids are stored directly (no decode→re-encode round-trip).
    For models with thinking tokens or parsed tool calls, the stored state may
    diverge from the next call's canonical prompt rendering; _prepare_session_cache
    now crops to the LCP and prefills only the changed suffix.
    """
    cfg = EvictionPolicyConfig(name="random", budget=8, seed=0)
    provider = _build_provider(cfg)

    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    provider._prepare_session_cache(prompt_ids=prompt, call_idx=0)

    output_ids = [40, 41, 42]
    provider._extend_session_tokens(prompt_ids=prompt, output_ids=output_ids)

    assert provider._session_token_ids is not None
    expected = torch.tensor([[1, 2, 3, 40, 41, 42]], dtype=torch.long)
    torch.testing.assert_close(provider._session_token_ids, expected)
