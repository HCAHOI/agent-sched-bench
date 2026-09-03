"""Paper-faithful TTL estimator sidecar for Continuum's public vLLM fork.

The public fork contains program-level FCFS and KV pinning but replaces the
paper's estimator with a fixed two-second threshold.  This module supplies the
missing estimator and can patch a clean checkout at the paper authors' pinned
commit without copying their repository.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import statistics
import subprocess
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


CONTINUUM_COMMIT = "316a58794a6ff86b216e579b74fd56ed0c5a911f"
PAPER_HISTORY_THRESHOLD = 100  # Continuum v6 section 4.2.

# The paper calls T a sliding-window average but does not publish its length.
# This value is inferred from the only published history threshold (K=100),
# and is intentionally not tuned on agent-sched-bench traces.
INFERRED_QUEUE_WINDOW = 100
# The paper does not say how equal-reward TTLs are resolved.  Choosing the
# smallest maximizer consumes no extra GPU memory.
INFERRED_TTL_TIE_BREAK = "smallest"
# The paper defines eta over (k, N-k), but not which k values enter online.
# We add k=1..N only after a program completes, so N is never used in advance.
INFERRED_MEMORYFULNESS_SAMPLES = "completed programs, k=1..N"

logger = logging.getLogger(__name__)


def select_oracle_length_request(
    requests: Any,
    pinned_job_ids: set[str],
    job_first_entry_time: dict[str, float],
) -> Any:
    """Keep Continuum's pinned tier, then prefer less remaining decode work."""
    eligible = [request for request in requests if request.job_id in pinned_job_ids]
    if not eligible:
        eligible = list(requests)

    baseline = min(
        eligible,
        key=lambda request: job_first_entry_time.get(
            request.job_id, request.arrival_time
        ),
    )

    def remaining_tokens(request: Any) -> int:
        remaining = request.max_tokens - len(request.output_token_ids)
        if remaining < 0:
            raise ValueError("request output exceeds max_tokens")
        return remaining

    selected = min(
        eligible,
        key=lambda request: (
            remaining_tokens(request),
            job_first_entry_time.get(request.job_id, request.arrival_time),
            request.arrival_time,
            request.request_id,
        ),
    )
    if selected is not baseline:
        logger.info(
            "Continuum oracle-length changed selection from %s (%d tokens) "
            "to %s (%d tokens) among %d requests",
            baseline.request_id,
            remaining_tokens(baseline),
            selected.request_id,
            remaining_tokens(selected),
            len(eligible),
        )
    return selected


@dataclass(frozen=True)
class RuntimeBinding:
    """Exact serving configuration for which a measured profile is valid."""

    model: str
    model_dtype: str
    kv_cache_dtype: str
    tensor_parallel_size: int
    gpu_name: str
    gpu_memory_mib: int
    gpu_count: int
    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int
    kv_dtype: str
    kv_dtype_size: int
    bytes_per_token_per_gpu: int

    @classmethod
    def from_payload(cls, payload: Any, path: str | Path) -> RuntimeBinding:
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: missing runtime_binding")
        gpu = payload.get("gpu")
        layout = payload.get("kv_layout")
        if not isinstance(gpu, dict) or not isinstance(layout, dict):
            raise ValueError(f"{path}: incomplete runtime_binding")
        values = {
            "model": payload.get("model"),
            "model_dtype": payload.get("model_dtype"),
            "kv_cache_dtype": payload.get("kv_cache_dtype"),
            "tensor_parallel_size": payload.get("tensor_parallel_size"),
            "gpu_name": gpu.get("name"),
            "gpu_memory_mib": gpu.get("memory_mib"),
            "gpu_count": gpu.get("count"),
            "num_hidden_layers": layout.get("num_hidden_layers"),
            "num_key_value_heads": layout.get("num_key_value_heads"),
            "head_dim": layout.get("head_dim"),
            "kv_dtype": layout.get("dtype"),
            "kv_dtype_size": layout.get("dtype_size"),
            "bytes_per_token_per_gpu": layout.get("bytes_per_token_per_gpu"),
        }
        for name in ("model", "model_dtype", "kv_cache_dtype", "gpu_name", "kv_dtype"):
            if not isinstance(values[name], str) or not values[name]:
                raise ValueError(f"{path}: invalid runtime_binding.{name}")
        for name in (
            "tensor_parallel_size",
            "gpu_memory_mib",
            "gpu_count",
            "num_hidden_layers",
            "num_key_value_heads",
            "head_dim",
            "kv_dtype_size",
            "bytes_per_token_per_gpu",
        ):
            if (
                not isinstance(values[name], int)
                or isinstance(values[name], bool)
                or values[name] <= 0
            ):
                raise ValueError(f"{path}: invalid runtime_binding.{name}")
        binding = cls(**values)
        if binding.gpu_count != binding.tensor_parallel_size:
            raise ValueError(f"{path}: GPU count must equal tensor parallel size")
        return binding


@dataclass(frozen=True)
class CacheMissProfile:
    """Measured Prefill-Reload(r) profile for one model/hardware pair."""

    mode: str
    model: str
    max_context_tokens: int
    quadratic_seconds: tuple[float, float, float] | None = None
    bytes_per_token: float | None = None
    reload_bytes_per_second: float | None = None
    runtime_binding: RuntimeBinding | None = None

    @classmethod
    def from_json(cls, path: str | Path, mode: str) -> CacheMissProfile:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = payload.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError(f"{path}: missing model")
        binding = RuntimeBinding.from_payload(payload.get("runtime_binding"), path)
        if binding.model != model:
            raise ValueError(f"{path}: profile and runtime-binding models differ")
        measured_layout = payload.get("kv_layout")
        if isinstance(measured_layout, dict):
            expected_layout = {
                "num_hidden_layers": binding.num_hidden_layers,
                "num_key_value_heads": binding.num_key_value_heads,
                "head_dim": binding.head_dim,
                "dtype": binding.kv_dtype,
                "dtype_size": binding.kv_dtype_size,
                "bytes_per_token": binding.bytes_per_token_per_gpu,
            }
            for name, expected in expected_layout.items():
                if name in measured_layout and measured_layout[name] != expected:
                    raise ValueError(f"{path}: measured KV layout disagrees on {name}")
        for device_key in ("device", "device_info"):
            device = payload.get(device_key)
            if (
                isinstance(device, dict)
                and "device" in device
                and device["device"] != binding.gpu_name
            ):
                raise ValueError(f"{path}: measured GPU disagrees with runtime binding")
        if mode == "prefill":
            coefficients = payload.get("quadratic_fit", {}).get("coefficients")
            coefficient_order = payload.get("quadratic_fit", {}).get(
                "coefficient_order"
            )
            points = payload.get("points")
            if (
                payload.get("measurement") != "prefill_recompute_cost"
                or not isinstance(coefficients, list)
                or len(coefficients) != 3
                or not all(_finite_number(value) for value in coefficients)
                or coefficient_order
                != ["context_tokens^2", "context_tokens", "intercept"]
                or not isinstance(points, list)
                or not points
            ):
                raise ValueError(f"{path}: invalid quadratic prefill profile")
            max_context = max(int(point["context_tokens"]) for point in points)
            return cls(
                mode=mode,
                model=model,
                max_context_tokens=max_context,
                quadratic_seconds=tuple(
                    float(value) / 1000.0 for value in coefficients
                ),
                runtime_binding=binding,
            )
        if mode == "reload":
            bytes_per_token = payload.get("bytes_per_token")
            measurements = payload.get("measurements")
            if (
                not _positive_number(bytes_per_token)
                or not isinstance(measurements, list)
                or not measurements
            ):
                raise ValueError(f"{path}: invalid reload profile")
            if not math.isclose(
                float(bytes_per_token),
                binding.bytes_per_token_per_gpu,
                rel_tol=1e-9,
            ):
                raise ValueError(f"{path}: reload bytes/token disagrees with binding")
            total_bytes = 0.0
            total_seconds = 0.0
            max_context = 0
            for row in measurements:
                byte_count = row.get("bytes")
                reload_ms = row.get("swap_in_ms")
                tokens = row.get("tokens")
                if not all(
                    _positive_number(value) for value in (byte_count, reload_ms, tokens)
                ):
                    raise ValueError(f"{path}: invalid reload measurement")
                if not math.isclose(
                    float(byte_count) / float(tokens),
                    float(bytes_per_token),
                    rel_tol=1e-9,
                ):
                    raise ValueError(f"{path}: reload KV byte layout is inconsistent")
                total_bytes += float(byte_count)
                total_seconds += float(reload_ms) / 1000.0
                max_context = max(max_context, int(tokens))
            # The paper says average offload throughput, but not how repeated
            # measurements are aggregated.  This is the aggregate byte rate.
            return cls(
                mode=mode,
                model=model,
                max_context_tokens=max_context,
                bytes_per_token=float(bytes_per_token),
                reload_bytes_per_second=total_bytes / total_seconds,
                runtime_binding=binding,
            )
        raise ValueError("CONTINUUM_REPRODUCTION_MODE must be prefill or reload")

    def estimate_seconds(self, context_tokens: int) -> float:
        if context_tokens <= 0 or context_tokens > self.max_context_tokens:
            raise ValueError(
                f"context_tokens must be in [1, {self.max_context_tokens}], "
                f"got {context_tokens}"
            )
        if self.mode == "prefill":
            assert self.quadratic_seconds is not None
            a, b, c = self.quadratic_seconds
            return max(0.0, (a * context_tokens + b) * context_tokens + c)
        assert self.bytes_per_token is not None
        assert self.reload_bytes_per_second is not None
        return context_tokens * self.bytes_per_token / self.reload_bytes_per_second


_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "half": "float16",
    "float16": "float16",
    "fp16": "float16",
    "float": "float32",
    "float32": "float32",
    "fp32": "float32",
    "fp8": "fp8_e4m3",
    "fp8_e4m3": "fp8_e4m3",
    "fp8_e5m2": "fp8_e5m2",
}
_DTYPE_SIZES = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
}


def _resolved_dtype(value: str, *, auto_dtype: str | None = None) -> str:
    if value == "auto":
        if auto_dtype is None:
            raise ValueError("auto dtype cannot be resolved")
        value = auto_dtype
    try:
        return _DTYPE_ALIASES[value.lower().removeprefix("torch.")]
    except KeyError as error:
        raise ValueError(f"unsupported dtype {value!r}") from error


def validate_runtime_binding(
    profile: CacheMissProfile,
    *,
    model: str,
    model_dtype: str,
    kv_cache_dtype: str,
    tensor_parallel_size: int,
    gpu_rows: list[tuple[str, int]],
    model_config: Any,
) -> None:
    """Fail unless measured and actual serving configurations are identical."""
    binding = profile.runtime_binding
    if binding is None:
        raise ValueError("profile lacks a runtime binding")
    config_dtype = str(getattr(model_config, "torch_dtype", "")).removeprefix("torch.")
    resolved_model_dtype = _resolved_dtype(model_dtype, auto_dtype=config_dtype)
    resolved_kv_dtype = _resolved_dtype(kv_cache_dtype, auto_dtype=resolved_model_dtype)
    kv_heads = getattr(
        model_config,
        "num_key_value_heads",
        getattr(model_config, "num_attention_heads", None),
    )
    attention_heads = getattr(model_config, "num_attention_heads", None)
    hidden_size = getattr(model_config, "hidden_size", None)
    head_dim = getattr(model_config, "head_dim", None)
    if (
        head_dim is None
        and isinstance(hidden_size, int)
        and isinstance(attention_heads, int)
    ):
        if hidden_size % attention_heads:
            raise ValueError("model hidden size is not divisible by attention heads")
        head_dim = hidden_size // attention_heads
    layers = getattr(model_config, "num_hidden_layers", None)
    if not all(
        isinstance(value, int) and value > 0 for value in (layers, kv_heads, head_dim)
    ):
        raise ValueError("model config lacks a complete KV layout")
    if kv_heads % tensor_parallel_size:
        raise ValueError("KV heads are not divisible by tensor parallel size")
    bytes_per_token = (
        2
        * layers
        * (kv_heads // tensor_parallel_size)
        * head_dim
        * _DTYPE_SIZES[resolved_kv_dtype]
    )
    actual = RuntimeBinding(
        model=model,
        model_dtype=resolved_model_dtype,
        kv_cache_dtype=resolved_kv_dtype,
        tensor_parallel_size=tensor_parallel_size,
        gpu_name=gpu_rows[0][0] if gpu_rows else "",
        gpu_memory_mib=gpu_rows[0][1] if gpu_rows else 0,
        gpu_count=len(gpu_rows),
        num_hidden_layers=layers,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        kv_dtype=resolved_kv_dtype,
        kv_dtype_size=_DTYPE_SIZES[resolved_kv_dtype],
        bytes_per_token_per_gpu=bytes_per_token,
    )
    if any(row != (actual.gpu_name, actual.gpu_memory_mib) for row in gpu_rows):
        raise ValueError("serving GPUs are not homogeneous")
    if actual != binding:
        differing = [
            name
            for name in RuntimeBinding.__dataclass_fields__
            if getattr(actual, name) != getattr(binding, name)
        ]
        raise ValueError(
            f"runtime does not match measured profile: {', '.join(differing)}"
        )


@dataclass
class _ProgramState:
    turns: int = 0
    last_finish: float | None = None
    last_tool: str | None = None
    current_arrival: float | None = None
    current_is_followup: bool = False
    queue_segment_start: float | None = None
    queue_wait_seconds: float = 0.0
    queue_recorded: bool = False


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _positive_number(value: Any) -> bool:
    return _finite_number(value) and float(value) > 0


def empirical_ttl_seconds(
    durations: list[float] | tuple[float, ...], benefit_seconds: float
) -> float:
    """Solve Continuum Eq. 2 over unique observed durations and zero."""
    if not math.isfinite(benefit_seconds):
        raise ValueError("benefit_seconds must be finite")
    values = tuple(float(value) for value in durations)
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("durations must be a non-empty finite non-negative sequence")
    best_ttl = 0.0
    best_reward = -math.inf
    for ttl in sorted({0.0, *values}):
        probability = sum(value <= ttl for value in values) / len(values)
        reward = probability * benefit_seconds - ttl
        if reward > best_reward:
            best_ttl = ttl
            best_reward = reward
    return best_ttl


def cold_start_ttl_seconds(benefit_seconds: float) -> float:
    """Analytic optimum for the paper's ToolDuration ~ Exp(1) cold start."""
    if not math.isfinite(benefit_seconds):
        raise ValueError("benefit_seconds must be finite")
    return math.log(benefit_seconds) if benefit_seconds > 1.0 else 0.0


def memoryfulness(turn_pairs: list[tuple[int, int]]) -> float:
    """Compute eta = -Corr(k, N-k), using the paper's cold eta=1 if undefined."""
    if len(turn_pairs) < 2:
        return 1.0
    progress, remaining = zip(*turn_pairs, strict=True)
    if len(set(progress)) < 2 or len(set(remaining)) < 2:
        return 1.0
    return max(-1.0, min(1.0, -statistics.correlation(progress, remaining)))


class ToolCallEstimator:
    """Continuum v6 sections 4.1--5.2 with causal online updates."""

    def __init__(
        self,
        tokenizer: Any | None = None,
        model_name: str | None = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tokenizer_revision: str | None = None,
        parser: Any | None = None,
        cache_miss_profile: CacheMissProfile | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if tokenizer is None and model_name is not None:
            from vllm.transformers_utils.tokenizer import get_tokenizer

            tokenizer = get_tokenizer(
                tokenizer_name=model_name,
                tokenizer_mode=tokenizer_mode,
                trust_remote_code=trust_remote_code,
                revision=tokenizer_revision,
            )
        self.tokenizer = tokenizer
        self.parser = parser
        self._profile = cache_miss_profile
        self._clock = clock
        self._programs: dict[str, _ProgramState] = {}
        self._turn_pairs: list[tuple[int, int]] = []
        self._global_durations: list[float] = []
        self._durations_by_tool: dict[str, list[float]] = defaultdict(list)
        self._queue_delays: deque[float] = deque(maxlen=INFERRED_QUEUE_WINDOW)

    @property
    def eta(self) -> float:
        return memoryfulness(self._turn_pairs)

    @property
    def average_evicted_queue_seconds(self) -> float:
        return statistics.fmean(self._queue_delays) if self._queue_delays else 0.0

    def _cache_miss_profile(self) -> CacheMissProfile:
        if self._profile is None:
            path = os.environ.get("CONTINUUM_REPRODUCTION_PROFILE")
            mode = os.environ.get("CONTINUUM_REPRODUCTION_MODE")
            if not path or not mode:
                raise RuntimeError(
                    "set CONTINUUM_REPRODUCTION_PROFILE and "
                    "CONTINUUM_REPRODUCTION_MODE=prefill|reload"
                )
            self._profile = CacheMissProfile.from_json(path, mode)
            expected_model = os.environ.get("CONTINUUM_REPRODUCTION_MODEL")
            if expected_model and self._profile.model != expected_model:
                raise ValueError(
                    f"profile model {self._profile.model!r} != served model "
                    f"{expected_model!r}"
                )
        return self._profile

    @staticmethod
    def _job_id(request: Any) -> str:
        job_id = getattr(request, "job_id", None)
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("Continuum reproduction requires a non-empty job_id")
        if not isinstance(getattr(request, "is_last_step", None), bool):
            raise ValueError("Continuum reproduction requires boolean is_last_step")
        return job_id

    def request_arrives(self, request: Any) -> None:
        now = self._clock()
        job_id = self._job_id(request)
        state = self._programs.get(job_id)
        if state is None:
            state = self._programs[job_id] = _ProgramState()
        else:
            request.last_func_call = state.last_tool
            if state.last_finish is not None and state.last_tool is not None:
                duration = now - state.last_finish
                if duration < 0:
                    raise ValueError("tool-call duration cannot be negative")
                self._global_durations.append(duration)
                self._durations_by_tool[state.last_tool].append(duration)
        state.current_arrival = now
        state.current_is_followup = state.turns > 0
        state.queue_segment_start = (
            now
            if state.current_is_followup
            and getattr(getattr(request, "status", None), "name", "WAITING")
            == "WAITING"
            else None
        )
        state.queue_wait_seconds = 0.0
        state.queue_recorded = False

    def request_queue_paused(self, request: Any) -> None:
        job_id = self._job_id(request)
        state = self._programs[job_id]
        if state.queue_segment_start is None or state.queue_recorded:
            return
        elapsed = self._clock() - state.queue_segment_start
        if elapsed < 0:
            raise ValueError("queue delay cannot be negative")
        state.queue_wait_seconds += elapsed
        state.queue_segment_start = None

    def request_queue_resumed(self, request: Any) -> None:
        job_id = self._job_id(request)
        state = self._programs[job_id]
        if (
            state.current_is_followup
            and state.queue_segment_start is None
            and not state.queue_recorded
        ):
            state.queue_segment_start = self._clock()

    def request_admitted(self, request: Any, *, retained: bool) -> None:
        """Finish scheduler-only queue timing before prefill or remote reload."""
        job_id = self._job_id(request)
        state = self._programs[job_id]
        if not state.current_is_followup or state.queue_recorded:
            return
        self.request_queue_paused(request)
        if not retained:
            self._queue_delays.append(state.queue_wait_seconds)
        state.queue_recorded = True

    def request_finished(self, request: Any) -> None:
        now = self._clock()
        job_id = self._job_id(request)
        state = self._programs[job_id]
        tool = getattr(request, "this_func_call", None)
        output_ids = getattr(request, "output_token_ids", ())
        if (
            tool is None
            and self.tokenizer is not None
            and self.parser is not None
            and output_ids
        ):
            text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            tool = self.parser.parse(text)
        request.this_func_call = tool or None
        state.turns += 1
        state.last_finish = now
        state.last_tool = request.this_func_call
        if request.is_last_step:
            self.program_finished(job_id)

    def program_finished(self, job_id: str) -> bool:
        state = self._programs.get(job_id)
        if state is None or state.turns < 1:
            return False
        total = state.turns
        self._turn_pairs.extend(
            (served, total - served) for served in range(1, total + 1)
        )
        del self._programs[job_id]
        return True

    def set_up_pin(self, request: Any) -> float:
        self._job_id(request)
        tool = getattr(request, "this_func_call", None)
        if request.is_last_step or not isinstance(tool, str) or not tool:
            return 0.0
        context_tokens = getattr(request, "num_tokens", None)
        if not isinstance(context_tokens, int) or isinstance(context_tokens, bool):
            prompt_tokens = getattr(request, "num_prompt_tokens", None)
            output_ids = getattr(request, "output_token_ids", ())
            if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool):
                raise ValueError("request lacks an integer context token count")
            context_tokens = prompt_tokens + len(output_ids)
        prefill_reload = self._cache_miss_profile().estimate_seconds(context_tokens)
        benefit = self.average_evicted_queue_seconds * self.eta + prefill_reload
        if len(self._global_durations) <= PAPER_HISTORY_THRESHOLD:
            ttl = cold_start_ttl_seconds(benefit)
            source = "exp1_cold_start"
        else:
            tool_history = self._durations_by_tool.get(tool, ())
            if len(tool_history) <= PAPER_HISTORY_THRESHOLD:
                history = self._global_durations
                source = "global_empirical_cdf"
            else:
                history = tool_history
                source = "tool_empirical_cdf"
            ttl = empirical_ttl_seconds(history, benefit)
        logger.info(
            "continuum_ttl job=%s tool=%s ttl_s=%.6f source=%s "
            "prefill_reload_s=%.6f queue_s=%.6f eta=%.6f",
            request.job_id,
            tool,
            ttl,
            source,
            prefill_reload,
            self.average_evicted_queue_seconds,
            self.eta,
        )
        return ttl


_SCHEDULER_REPLACEMENTS = (
    (
        "from vllm.v1.core.estimate_with_func import ToolCallEstimator, Continuum_Recorder",
        "from vllm.v1.core.estimate_with_func import Continuum_Recorder, ToolCallParser\n"
        "from vllm.v1.core.continuum_reproduction import ToolCallEstimator",
    ),
    (
        "            tokenizer_revision=vllm_config.model_config.tokenizer_revision,\n"
        "        )",
        "            tokenizer_revision=vllm_config.model_config.tokenizer_revision,\n"
        "            parser=ToolCallParser(),\n"
        "        )",
    ),
    (
        "                        logger.debug(\n"
        '                            "%s is still in WAITING_FOR_REMOTE_KVS state.",\n'
        "                            request.request_id)",
        "                        logger.debug(\n"
        '                            "%s is still in WAITING_FOR_REMOTE_KVS state.",\n'
        "                            request.request_id)\n"
        "                        self.tool_call_estimator.request_queue_paused(request)",
    ),
    (
        "                    else:\n"
        "                        if self.policy == SchedulingPolicy.CONTINUUM: \n"
        "                            self.waiting.pop_request(self.pinned_requests, self.kv_cache_manager, self.connector)\n"
        "                        else:\n"
        "                            self.waiting.pop_request()\n"
        "                        skipped_waiting_requests.prepend_request(request)\n"
        "                        continue\n\n"
        "                # Check that adding the request still respects the max_loras",
        "                    else:\n"
        "                        self.tool_call_estimator.request_queue_paused(request)\n"
        "                        if self.policy == SchedulingPolicy.CONTINUUM: \n"
        "                            self.waiting.pop_request(self.pinned_requests, self.kv_cache_manager, self.connector)\n"
        "                        else:\n"
        "                            self.waiting.pop_request()\n"
        "                        skipped_waiting_requests.prepend_request(request)\n"
        "                        continue\n\n"
        "                # Check that adding the request still respects the max_loras",
    ),
    (
        "                    # Scheduling would exceed max_loras, skip.\n"
        "                    self.waiting.pop_request()",
        "                    # Scheduling would exceed max_loras, skip.\n"
        "                    self.tool_call_estimator.request_queue_paused(request)\n"
        "                    self.waiting.pop_request()",
    ),
    (
        "                    skipped_waiting_requests.prepend_request(request)\n"
        "                    continue\n\n"
        "                num_external_computed_tokens = 0",
        "                    skipped_waiting_requests.prepend_request(request)\n"
        "                    continue\n\n"
        "                self.tool_call_estimator.request_queue_resumed(request)\n\n"
        "                num_external_computed_tokens = 0",
    ),
    (
        "                            # the number of matched tokens.\n"
        "                            self.waiting.pop_request()",
        "                            # the number of matched tokens.\n"
        "                            self.tool_call_estimator.request_queue_paused(request)\n"
        "                            self.waiting.pop_request()",
    ),
    (
        "                # KVTransfer: the connector uses this info to determine\n"
        "                # if a load is needed. Note that",
        "                self.tool_call_estimator.request_admitted(\n"
        "                    request, retained=self.is_pinned(request))\n\n"
        "                # KVTransfer: the connector uses this info to determine\n"
        "                # if a load is needed. Note that",
    ),
    (
        "            if length_of_pin > 0.01:",
        "            if length_of_pin > 0.0:",
    ),
)


def _git(checkout: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_status(checkout: Path) -> list[str]:
    return subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()


def _validate_checkout(checkout: Path) -> Path:
    if _git(checkout, "rev-parse", "HEAD") != CONTINUUM_COMMIT:
        raise ValueError(f"{checkout}: not pinned at {CONTINUUM_COMMIT}")
    scheduler = checkout / "vllm/v1/core/sched/scheduler.py"
    if not scheduler.is_file():
        raise ValueError(f"{checkout}: missing Continuum scheduler")
    return scheduler


def _patched_scheduler_from_head(checkout: Path) -> str:
    text = _git(checkout, "show", "HEAD:vllm/v1/core/sched/scheduler.py") + "\n"
    for old, new in _SCHEDULER_REPLACEMENTS:
        if text.count(old) != 1:
            raise ValueError(
                f"pinned scheduler patch point occurs {text.count(old)} times"
            )
        text = text.replace(old, new)
    return text


def apply_to_checkout(checkout: str | Path) -> None:
    checkout = Path(checkout).resolve()
    scheduler = _validate_checkout(checkout)
    installed = checkout / "vllm/v1/core/continuum_reproduction.py"
    if installed.exists():
        verify_checkout(checkout)
        return
    if _git_status(checkout):
        raise ValueError(f"{checkout}: checkout is not completely clean")
    text = _patched_scheduler_from_head(checkout)
    shutil.copyfile(Path(__file__).resolve(), installed)
    scheduler.write_text(text, encoding="utf-8")
    verify_checkout(checkout)


def verify_checkout(checkout: str | Path) -> None:
    checkout = Path(checkout).resolve()
    scheduler = _validate_checkout(checkout)
    installed = checkout / "vllm/v1/core/continuum_reproduction.py"
    if not installed.is_file():
        raise ValueError(f"{checkout}: reproduction sidecar is not installed")
    if installed.read_bytes() != Path(__file__).resolve().read_bytes():
        raise ValueError(f"{checkout}: installed reproduction sidecar has drifted")
    if scheduler.read_text(encoding="utf-8") != _patched_scheduler_from_head(checkout):
        raise ValueError(f"{checkout}: scheduler differs from the exact intended patch")
    expected_status = {
        " M vllm/v1/core/sched/scheduler.py",
        "?? vllm/v1/core/continuum_reproduction.py",
    }
    actual_status = set(_git_status(checkout))
    if actual_status != expected_status:
        raise ValueError(
            f"{checkout}: expected only scheduler+sidecar changes, got "
            f"{sorted(actual_status)}"
        )
    compile(installed.read_text(encoding="utf-8"), str(installed), "exec")
    compile(scheduler.read_text(encoding="utf-8"), str(scheduler), "exec")
    subprocess.run(["git", "-C", str(checkout), "diff", "--check"], check=True)


_TRACE_ADAPTER_COMMON_FILES = (
    "entrypoints/openai/api_server.py",
    "v1/core/sched/scheduler.py",
    "v1/engine/core.py",
)


def _replace_once(text: str, old: str, new: str, *, path: Path) -> str:
    if text.count(old) != 1:
        raise ValueError(
            f"{path}: replay adapter anchor occurs {text.count(old)} times"
        )
    return text.replace(old, new)


def _trace_adapter_text(path: Path, relative: str, variant: str) -> str:
    text = path.read_text(encoding="utf-8")
    if relative == "entrypoints/openai/api_server.py":
        anchor = '    @router.post("/reset_prefix_cache")\n'
        addition = """    @router.post("/continuum/programs/release")
    async def release_continuum_program(raw_request: Request):
        body = await raw_request.json()
        job_id = body.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise HTTPException(status_code=400, detail="job_id is required")
        released = await engine_client(
            raw_request).engine_core.call_utility_async(
                "release_continuum_program", job_id)
        return JSONResponse(content={"job_id": job_id, "released": released})


"""
        return _replace_once(text, anchor, addition + anchor, path=path)
    if relative == "v1/engine/core.py":
        anchor = "    def reset_prefix_cache(self):\n"
        addition = """    def release_continuum_program(self, job_id: str) -> bool:
        return self.scheduler.release_continuum_program(job_id)

"""
        return _replace_once(text, anchor, addition + anchor, path=path)
    if relative == "v1/core/sched/scheduler.py":
        anchor = "    def reset_prefix_cache(self) -> bool:\n"
        addition = """    def release_continuum_program(self, job_id: str) -> bool:
        known = job_id in self.running_job_id_first_entry_time
        for request, end_time in list(self.pinned_requests):
            if request.job_id == job_id:
                self.unpin_request(request, end_time)
                known = True
        self.running_job_id_first_entry_time.pop(job_id, None)
        self.waiting.job_id_first_entry_time.pop(job_id, None)
        program_finished = getattr(
            self.tool_call_estimator, "program_finished", None)
        if program_finished is not None:
            known = program_finished(job_id) or known
        return known

"""
        return _replace_once(text, anchor, addition + anchor, path=path)
    if relative == "v1/core/sched/request_queue.py" and variant == "reproduction":
        text = _replace_once(text, "import heapq\n", "import heapq\nimport os\n", path=path)
        anchor = (
            "        pinned_request_job_id_set = "
            "{req.job_id for req, _ in pinned_requests}\n"
        )
        addition = anchor + """
        if os.environ.get("CONTINUUM_ORACLE_OUTPUT_LENGTH") == "1":
            from vllm.v1.core.continuum_reproduction import (
                select_oracle_length_request,
            )
            return select_oracle_length_request(
                self, pinned_request_job_id_set,
                self.job_id_first_entry_time)
"""
        return _replace_once(text, anchor, addition, path=path)
    if relative == "v1/core/estimate_with_func.py" and variant == "public":
        text = _replace_once(
            text,
            "    def set_up_pin(self, request: Request) -> float:\n",
            "    def program_finished(self, job_id: str) -> bool:\n"
            "        return self.job_to_history.pop(job_id, None) is not None\n\n"
            "    def set_up_pin(self, request: Request) -> float:\n",
            path=path,
        )
        text = _replace_once(
            text,
            "        this_func_call = None\n"
            "        if self.tokenizer is not None and len(request.output_token_ids) > 0:\n",
            "        this_func_call = request.this_func_call\n"
            "        if (this_func_call is None and self.tokenizer is not None\n"
            "                and len(request.output_token_ids) > 0):\n",
            path=path,
        )
        return _replace_once(
            text,
            "        request.this_func_call = this_func_call\n",
            "        request.this_func_call = this_func_call or None\n",
            path=path,
        )
    raise ValueError(f"unsupported replay adapter file: {relative} ({variant})")


def _trace_adapter_expected(source_vllm: Path, variant: str) -> dict[str, str]:
    if variant not in {"public", "reproduction"}:
        raise ValueError(f"unsupported Continuum variant: {variant}")
    relatives = list(_TRACE_ADAPTER_COMMON_FILES)
    if variant == "public":
        relatives.append("v1/core/estimate_with_func.py")
    else:
        relatives.append("v1/core/sched/request_queue.py")
    return {
        relative: _trace_adapter_text(source_vllm / relative, relative, variant)
        for relative in relatives
    }


def install_trace_adapter(
    source_vllm: str | Path,
    package_vllm: str | Path,
    variant: str,
) -> None:
    expected = _trace_adapter_expected(Path(source_vllm), variant)
    package = Path(package_vllm)
    for relative, text in expected.items():
        target = package / relative
        target.write_text(text, encoding="utf-8")
        compile(text, str(target), "exec")
    verify_trace_adapter(source_vllm, package_vllm, variant)


def verify_trace_adapter(
    source_vllm: str | Path,
    package_vllm: str | Path,
    variant: str,
) -> None:
    expected = _trace_adapter_expected(Path(source_vllm), variant)
    package = Path(package_vllm)
    for relative, text in expected.items():
        target = package / relative
        if not target.is_file() or target.read_text(encoding="utf-8") != text:
            raise ValueError(f"{target}: replay adapter differs from expected")


def _selected_gpus(tensor_parallel_size: int) -> list[tuple[str, int]]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    rows: dict[int, tuple[str, int]] = {}
    for line in output.splitlines():
        index, name, memory_mib = (part.strip() for part in line.split(",", 2))
        rows[int(index)] = (name, int(float(memory_mib)))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        try:
            indices = [int(value.strip()) for value in visible.split(",")]
        except ValueError as error:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES must use numeric GPU indices for validation"
            ) from error
    else:
        indices = sorted(rows)
    if len(indices) < tensor_parallel_size:
        raise ValueError("fewer visible GPUs than tensor parallel size")
    try:
        return [rows[index] for index in indices[:tensor_parallel_size]]
    except KeyError as error:
        raise ValueError(
            f"CUDA_VISIBLE_DEVICES names unknown GPU {error.args[0]}"
        ) from error


def validate_runtime(
    profile_path: str | Path,
    mode: str,
    model: str,
    model_dtype: str,
    kv_cache_dtype: str,
    tensor_parallel_size: int,
) -> int:
    from transformers import AutoConfig

    profile = CacheMissProfile.from_json(profile_path, mode)
    config = AutoConfig.from_pretrained(model)
    validate_runtime_binding(
        profile,
        model=model,
        model_dtype=model_dtype,
        kv_cache_dtype=kv_cache_dtype,
        tensor_parallel_size=tensor_parallel_size,
        gpu_rows=_selected_gpus(tensor_parallel_size),
        model_config=config,
    )
    return profile.max_context_tokens


def inference_manifest() -> dict[str, Any]:
    return {
        "paper": "arXiv:2511.02230v6",
        "published": {
            "history_threshold": PAPER_HISTORY_THRESHOLD,
            "queue_initial_seconds": 0,
            "cold_duration_distribution": "Exp(rate=1/second)",
            "cold_memoryfulness": 1,
            "ttl_candidates": "zero plus unique observed durations",
        },
        "inferred_not_tuned": {
            "queue_window": INFERRED_QUEUE_WINDOW,
            "queue_sample": (
                "follow-up without a live pin, engine arrival to first schedule"
            ),
            "ttl_tie_break": INFERRED_TTL_TIE_BREAK,
            "memoryfulness_samples": INFERRED_MEMORYFULNESS_SAMPLES,
            "cold_start_benefit": (
                "current queue mean and current request Prefill-Reload in Eq. 2"
            ),
            "reload_average": "aggregate bytes / aggregate seconds",
        },
        "trace_replay_transport": {
            "tool_signal": (
                "recorded source tool signature, consumed only when the shadow "
                "LLM request finishes"
            ),
            "terminal_signal": "causal program release after replay completion",
            "algorithm_change": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "apply",
            "verify",
            "manifest",
            "validate-runtime",
            "install-trace-adapter",
            "verify-trace-adapter",
        ),
    )
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    if args.command == "manifest":
        if args.arguments:
            parser.error("manifest takes no arguments")
        print(json.dumps(inference_manifest(), indent=2, sort_keys=True))
        return
    if args.command == "validate-runtime":
        if len(args.arguments) != 6:
            parser.error(
                "validate-runtime requires PROFILE MODE MODEL MODEL_DTYPE "
                "KV_CACHE_DTYPE TENSOR_PARALLEL_SIZE"
            )
        profile, mode, model, model_dtype, kv_dtype, tp = args.arguments
        max_context_tokens = validate_runtime(
            profile, mode, model, model_dtype, kv_dtype, int(tp)
        )
        print(max_context_tokens)
        return
    if args.command in {"install-trace-adapter", "verify-trace-adapter"}:
        if len(args.arguments) != 3:
            parser.error(f"{args.command} requires SOURCE_VLLM PACKAGE_VLLM VARIANT")
        source_vllm, package_vllm, variant = args.arguments
        if args.command == "install-trace-adapter":
            install_trace_adapter(source_vllm, package_vllm, variant)
        else:
            verify_trace_adapter(source_vllm, package_vllm, variant)
        return
    if len(args.arguments) != 1:
        parser.error("apply/verify require exactly one checkout")
    checkout = args.arguments[0]
    if args.command == "apply":
        apply_to_checkout(checkout)
    else:
        verify_checkout(checkout)


if __name__ == "__main__":
    main()
