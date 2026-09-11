"""Declared extensions to public PPD: long-context data and a decode-load guard."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def add_huge_context_table(engine: Any, path: Path) -> None:
    """Reuse public scoring, replacing extrapolation with measured 32K results."""
    from ppd.optimizer.ppd_decision_engine import PPDDecisionEngine, QPS_POINTS, T2_WORKLOAD_CONFIGS

    class HugeContextData(PPDDecisionEngine):
        def _load_benchmark_result(self, config: str, workload: str, qps: float):
            if workload.startswith("large_"):
                workload = workload.replace("large_", "huge_", 1)
            return super()._load_benchmark_result(config, workload, qps)

    measured = HugeContextData(str(path), base_config=engine.base_config,
                               w_ttft=engine.w_ttft, w_tpot=engine.w_tpot)
    for workload in T2_WORKLOAD_CONFIGS:
        for qps in QPS_POINTS:
            key = ("large", workload, qps)
            assert key in measured.performance_data, f"Missing measured huge-context point: {key}"
            target = ("huge", workload, qps)
            engine.lookup_table[target] = measured.lookup_table[key]
            engine.performance_data[target] = measured.performance_data[key]


# Cost model for two-sided routing, measured on this host (2xL40S, Qwen3-4B FP8,
# vLLM 0.28.0) from the serving-length profiling matrix in
# results/serving-length-profile-vast-20260908-complete. Declared before any
# replay under this mode; never fitted on replay results.
LOCAL_PREFILL_S_PER_TOKEN = 0.184e-3
LOCAL_PREFILL_FLOOR_S = 0.027
P_PREFILL_S_PER_TOKEN = 0.295e-3
KV_TRANSFER_S_PER_TOKEN = 0.0102e-3
KV_TRANSFER_FLOOR_S = 0.022
PD_HANDOFF_S = 0.4
# Queueing is charged on the engine's actual pending prefill tokens (prompt tokens
# not yet computed over waiting and running requests), read by cache_snapshot.
# Amendment 2026-09-11, before any two-sided replay: the earlier proxy of one
# 2,048-token batch per waiting request under-priced bursts of agent-scale
# prompts by an order of magnitude in results/ppd-load-profile-20260911-r3.


def two_sided_constants() -> dict[str, float]:
    """The declared cost model, for the run's adapter config."""
    return {name: value for name, value in globals().items()
            if name.isupper() and isinstance(value, (int, float))}


def two_sided_estimate(prompt_tokens: int, snap_p: dict[str, Any],
                       snap_d: dict[str, Any]) -> dict[str, Any]:
    """Expected time to first token on each side; prefill locally iff D is no slower.

    Each side's queue is its pending prefill work (tokens still to compute for
    waiting and running requests) at that side's per-token cost; this request's
    own uncached tokens are added on top.
    """
    assert prompt_tokens > 0
    assert 0 <= snap_p["cached_tokens"] < prompt_tokens and snap_p["pending_prefill_tokens"] >= 0
    assert 0 <= snap_d["cached_tokens"] < prompt_tokens and snap_d["pending_prefill_tokens"] >= 0
    uncached_p = prompt_tokens - snap_p["cached_tokens"]
    uncached_d = prompt_tokens - snap_d["cached_tokens"]
    local_ttft_s = ((snap_d["pending_prefill_tokens"] + uncached_d) * LOCAL_PREFILL_S_PER_TOKEN
                    + LOCAL_PREFILL_FLOOR_S)
    pd_ttft_s = ((snap_p["pending_prefill_tokens"] + uncached_p) * P_PREFILL_S_PER_TOKEN
                 + prompt_tokens * KV_TRANSFER_S_PER_TOKEN + KV_TRANSFER_FLOOR_S + PD_HANDOFF_S)
    return dict(uncached_p=uncached_p, uncached_d=uncached_d, local_ttft_s=local_ttft_s,
                pd_ttft_s=pd_ttft_s, use_local=local_ttft_s <= pd_ttft_s)


def protect_decode(use_local: bool, prompt_tokens: int, state: dict[str, Any],
                   bypass_threshold: int) -> tuple[bool, str]:
    """Keep public routing unless substantial local prefill meets a full D queue.

    The threshold is PPD's existing short-input threshold; the load boundary is
    the engine's actual sequence capacity. Neither is fitted on replay results.
    """
    cached = state["cached_tokens"]
    assert 0 <= cached < prompt_tokens and bypass_threshold >= 0
    assert 0 <= state["running"] <= state["max_num_seqs"] and state["waiting"] >= 0
    missing = prompt_tokens - cached
    saturated = state["waiting"] > 0 or state["running"] >= state["max_num_seqs"]
    if use_local and missing >= bypass_threshold and saturated:
        return False, "uncached_prefill_on_busy_decode"
    return use_local, "keep_calibrated_decision"


def cache_snapshot(engine: Any, prompt_token_ids: list[int]) -> dict[str, Any]:
    """Read the scheduler's local cache index without allocating or touching blocks.

    Runs inside EngineCore via its utility RPC. The snapshot is an estimate at
    query time, not a reservation: other requests can evict blocks before admission.
    """
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    assert prompt_token_ids and len(prompt_token_ids) <= engine.vllm_config.model_config.max_model_len
    assert all(type(token) is int and token >= 0 for token in prompt_token_ids)
    assert engine.request_block_hasher is not None
    scheduler = engine.scheduler
    manager = scheduler.kv_cache_manager
    assert manager.enable_caching and len(manager.kv_cache_config.kv_cache_groups) == 1
    request = Request("ppd-cache-query", prompt_token_ids, SamplingParams(max_tokens=1),
                      None, block_hasher=engine.request_block_hasher)
    _, cached, _ = manager.coordinator.find_longest_cache_hit(
        request.block_hashes, len(prompt_token_ids) - 1)
    queued = list(scheduler.waiting) + list(scheduler.skipped_waiting) + list(scheduler.running)
    pending = sum(max(0, r.num_prompt_tokens - r.num_computed_tokens) for r in queued)
    return dict(cached_tokens=cached, running=len(scheduler.running),
                waiting=len(scheduler.waiting) + len(scheduler.skipped_waiting),
                pending_prefill_tokens=pending,
                max_num_seqs=scheduler.max_num_running_reqs,
                kv_cache_usage=scheduler.get_kv_cache_usage(), timestamp_s=time.time())
