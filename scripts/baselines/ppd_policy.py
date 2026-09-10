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
    return dict(cached_tokens=cached, running=len(scheduler.running),
                waiting=len(scheduler.waiting) + len(scheduler.skipped_waiting),
                max_num_seqs=scheduler.max_num_running_reqs,
                kv_cache_usage=scheduler.get_kv_cache_usage(), timestamp_s=time.time())
