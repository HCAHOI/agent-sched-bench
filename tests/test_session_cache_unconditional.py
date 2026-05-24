"""Session KV cache must be built for all configs, not just eviction.

Regression gate for the `perf(backend_hf)` decoupling that default-enabled the
session cache. Three configs all need to hit the strict-prefix LCP delta path
on call 2:

  1. sparse-attention configured (no eviction policy)
  2. bare baseline (no sparse, no eviction)
  3. eviction policy configured (must not regress the existing path)

We hand-construct an `HFRecordingProvider` via `__new__` (same trick as the
existing `tests/test_hf_session_cache.py`) so we don't pay HF model load cost.
The seam under test is `_prepare_session_cache` + `_extend_session_tokens`.
"""

from __future__ import annotations

import torch
from transformers import DynamicCache

from serving.kv_policies.base import BaseEvictionCache, EvictionPolicyConfig
from serving.recording.attention_bus import AttentionBus
from serving.recording.backend_hf import HFRecordingProvider
from serving.sparse_attention.config import SparseAttentionConfig


class _StubModelConfig:
    num_hidden_layers = 2
    max_position_embeddings = 64


class _StubModel:
    def __init__(self) -> None:
        self.config = _StubModelConfig()


class _StubTokenizer:
    def decode(self, ids, *, skip_special_tokens: bool = True) -> str:
        return " ".join(str(i) for i in ids)


class _StubCapturer:
    def start_attempt(self, *_args, **_kwargs) -> None:  # pragma: no cover
        pass

    def set_attempt_extra_meta(self, *_args, **_kwargs) -> None:  # pragma: no cover
        pass

    def finish_attempt(self, *_args, **_kwargs) -> None:  # pragma: no cover
        pass

    def close(self) -> None:  # pragma: no cover
        pass


def _build_provider(
    *,
    eviction_config: EvictionPolicyConfig | None,
    sparse_attention_config: SparseAttentionConfig | None,
) -> HFRecordingProvider:
    provider = HFRecordingProvider.__new__(HFRecordingProvider)
    provider._eviction_config = eviction_config
    provider._sparse_attention_config = sparse_attention_config
    provider._sparse_attention = None
    provider._session_cache = None
    provider._session_token_ids = None
    provider._session_history = []
    provider._message_first_seen = []
    provider._attention_bus = AttentionBus()
    provider.model = _StubModel()
    provider.tokenizer = _StubTokenizer()
    provider._torch = torch
    provider.capturer = _StubCapturer()
    return provider


def _drive_two_calls(provider: HFRecordingProvider) -> None:
    """Simulate two chat() calls with a shared prefix."""
    prompt_1 = torch.tensor([[1, 2, 3]], dtype=torch.long)
    delta_1, used_1 = provider._prepare_session_cache(
        prompt_ids=prompt_1, call_idx=0
    )
    assert used_1 is True, "call 1 must build the session cache eagerly"
    torch.testing.assert_close(delta_1, prompt_1)
    # Simulate post-generate token tracking.
    provider._extend_session_tokens(prompt_ids=prompt_1, output_ids=[10, 11])

    # Call 2: same prefix + new user tokens.
    prompt_2 = torch.tensor([[1, 2, 3, 10, 11, 20, 21, 22]], dtype=torch.long)
    delta_2, used_2 = provider._prepare_session_cache(
        prompt_ids=prompt_2, call_idx=1
    )
    assert used_2 is True


def _assert_strict_prefix_on_call_2(provider: HFRecordingProvider) -> None:
    h2 = provider._session_history[-1]
    new_len = h2["new_len"]
    assert h2["used_session_cache"] is True
    assert h2["lcp"] > 0, "call 2 must reuse the cumulative prefix"
    assert h2["lcp"] == h2["cached_len_before"], "call 2 must be a strict prefix"
    assert h2["delta_len"] < new_len, (
        f"call 2 delta_len={h2['delta_len']} not smaller than new_len={new_len}; "
        "no tokens were reused from the session cache"
    )
    assert h2["diverged"] is False


def test_sparse_attention_only_reuses_session_cache() -> None:
    """Sparse-attention-only run benefits from cross-call KV reuse."""
    sparse_cfg = SparseAttentionConfig(
        name="sliding", sink_size=2, recent_window=8, record=False
    )
    provider = _build_provider(
        eviction_config=None, sparse_attention_config=sparse_cfg
    )
    _drive_two_calls(provider)
    # Plain DynamicCache (no eviction subclass) — sparse runs need cross-call
    # KV reuse but not the eviction-policy machinery.
    assert isinstance(provider._session_cache, DynamicCache)
    assert not isinstance(provider._session_cache, BaseEvictionCache)
    _assert_strict_prefix_on_call_2(provider)


def test_bare_baseline_reuses_session_cache() -> None:
    """Baseline run (no sparse, no eviction) must also reuse KV across calls.

    This is the workload that previously paid full re-prefill every chat() call
    on multi-turn agent loops.
    """
    provider = _build_provider(
        eviction_config=None, sparse_attention_config=None
    )
    _drive_two_calls(provider)
    assert isinstance(provider._session_cache, DynamicCache)
    assert not isinstance(provider._session_cache, BaseEvictionCache)
    _assert_strict_prefix_on_call_2(provider)


def test_eviction_path_unchanged() -> None:
    """Existing eviction-policy path keeps building a BaseEvictionCache."""
    cfg = EvictionPolicyConfig(name="random", budget=8, seed=0)
    provider = _build_provider(
        eviction_config=cfg, sparse_attention_config=None
    )
    _drive_two_calls(provider)
    assert isinstance(provider._session_cache, BaseEvictionCache)
    _assert_strict_prefix_on_call_2(provider)


def test_env_var_disables_session_cache_for_all_configs(monkeypatch) -> None:
    """OMC_DISABLE_SESSION_CACHE=1 reverts to per-call full prefill for any
    config — used to A/B verify byte equality between session-cache ON/OFF.
    """
    monkeypatch.setenv("OMC_DISABLE_SESSION_CACHE", "1")
    for eviction_config, sparse_config in [
        (None, None),
        (None, SparseAttentionConfig(name="sliding", sink_size=2, recent_window=8)),
        (EvictionPolicyConfig(name="random", budget=8, seed=0), None),
    ]:
        provider = _build_provider(
            eviction_config=eviction_config,
            sparse_attention_config=sparse_config,
        )
        prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
        delta, used = provider._prepare_session_cache(
            prompt_ids=prompt, call_idx=0
        )
        assert used is False
        torch.testing.assert_close(delta, prompt)
        assert provider._session_cache is None
        assert provider._session_history[-1]["used_session_cache"] is False
