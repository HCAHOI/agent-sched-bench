from __future__ import annotations

import importlib
import importlib.metadata
from dataclasses import dataclass
from typing import Any

import httpx

from harness.prometheus import parse_prometheus_metric_values

@dataclass(slots=True)
class GpuMemoryBreakdown:

    gpu_index: int
    pid: int
    total_pid_mib: float
    weights_mib: float | None
    kv_cache_used_mib: float | None
    kv_cache_total_mib: float | None
    activations_mib: float | None
    ts: float


@dataclass(slots=True)
class GpuBaseline:

    weights_mib: float
    kv_cache_total_mib: float
    model: str
    dtype: str
    tensor_parallel_size: int


@dataclass(slots=True)
class GpuComponentBreakdown:

    step_index: int
    attn_mib: float
    mlp_mib: float
    other_activations_mib: float
    per_module: list[dict]
    measurement_kind: str


@dataclass(slots=True)
class PreemptionSnapshot:

    num_preemptions_total: float | None
    gpu_cache_usage_perc: float | None
    cpu_cache_usage_perc: float | None
    gpu_prefix_cache_hit_rate: float | None
    cpu_prefix_cache_hit_rate: float | None
    gpu_memory_breakdown: GpuMemoryBreakdown | None = None

@dataclass(slots=True)
class SchedulerHookStatus:

    runtime_hook_enabled: bool
    vllm_version: str | None
    target: str
    reason: str | None = None

def parse_prometheus_metrics(metrics_payload: str) -> PreemptionSnapshot:
    # vLLM 0.10+ renamed `gpu_cache_usage_perc` → `kv_cache_usage_perc` and
    # dropped `gpu_prefix_cache_hit_rate`. Both old and new gauge names map
    # to the same alias so either vLLM version populates it.
    wanted = {
        "vllm:num_preemptions_total": "num_preemptions_total",
        "vllm:gpu_cache_usage_perc": "gpu_cache_usage_perc",
        "vllm:kv_cache_usage_perc": "gpu_cache_usage_perc",
        "vllm:cpu_cache_usage_perc": "cpu_cache_usage_perc",
        "vllm:gpu_prefix_cache_hit_rate": "gpu_prefix_cache_hit_rate",
        "vllm:cpu_prefix_cache_hit_rate": "cpu_prefix_cache_hit_rate",
    }
    values = parse_prometheus_metric_values(
        metrics_payload,
        wanted,
        include_missing=True,
    )
    return PreemptionSnapshot(**values, gpu_memory_breakdown=None)

def empty_snapshot() -> PreemptionSnapshot:
    """All-None PreemptionSnapshot for callers with no metrics_url."""
    return PreemptionSnapshot(
        num_preemptions_total=None,
        gpu_cache_usage_perc=None,
        cpu_cache_usage_perc=None,
        gpu_prefix_cache_hit_rate=None,
        cpu_prefix_cache_hit_rate=None,
        gpu_memory_breakdown=None,
    )

def get_snapshot(
    metrics_url: str | None = None,
    *,
    timeout_s: float = 5.0,
) -> PreemptionSnapshot:
    """Fetch a PreemptionSnapshot from a vLLM /metrics endpoint; returns empty_snapshot() if url is unset."""
    if not metrics_url:
        return empty_snapshot()
    response = httpx.get(metrics_url, timeout=timeout_s)
    response.raise_for_status()
    return parse_prometheus_metrics(response.text)

def _safe_vllm_version() -> str | None:
    try:
        return importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        return None

def apply_scheduler_hook() -> SchedulerHookStatus:
    version = _safe_vllm_version()
    target = "vllm.v1.core.sched.scheduler.Scheduler.schedule"
    if version is None:
        raise RuntimeError("vllm is not installed")
    if not version.startswith("0.8."):
        raise RuntimeError(f"Unsupported vllm version for scheduler hook: {version}")

    scheduler_module = importlib.import_module("vllm.v1.core.sched.scheduler")
    scheduler_cls = getattr(scheduler_module, "Scheduler", None)
    if scheduler_cls is None or not hasattr(scheduler_cls, "schedule"):
        raise RuntimeError(f"Scheduler target not found: {target}")
    if getattr(scheduler_cls.schedule, "_agent_benchmark_hooked", False):
        return SchedulerHookStatus(
            runtime_hook_enabled=True,
            vllm_version=version,
            target=target,
        )

    original_schedule = scheduler_cls.schedule
    logger = getattr(scheduler_module, "logger")

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        before_preempted = {
            req_id
            for req_id, req in getattr(self, "requests", {}).items()
            if getattr(getattr(req, "status", None), "name", "") == "PREEMPTED"
        }
        result = original_schedule(self, *args, **kwargs)
        for req_id, req in getattr(self, "requests", {}).items():
            status_name = getattr(getattr(req, "status", None), "name", "")
            if status_name == "PREEMPTED" and req_id not in before_preempted:
                logger.info(
                    "EVICT seq_id=%s tokens=%s reason=preempted gpu_usage=%s",
                    req_id,
                    int(getattr(req, "num_tokens_with_spec", 0) or 0),
                    float(
                        getattr(getattr(self, "kv_cache_manager", None), "usage", 0.0)
                        or 0.0
                    ),
                )
        return result

    wrapped._agent_benchmark_hooked = True  # type: ignore[attr-defined]
    scheduler_cls.schedule = wrapped  # type: ignore[assignment]
    return SchedulerHookStatus(
        runtime_hook_enabled=True,
        vllm_version=version,
        target=target,
    )
