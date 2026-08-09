"""CPU-testable policy and trace logic for the W5 multi-tenant harness.

The harness replays recorded LLM prompts, completion lengths, and inter-turn
(tool) gaps.  Policy decisions use only information available when the current
LLM turn finishes; the recorded next-turn arrival time is never consulted.
"""

from __future__ import annotations

import json
import math
import statistics
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Sequence

from spike.trigger_table import TriggerTable, lookup_trigger
from tool_time.command import (
    command_prefix_keys,
    make_row_command_prefix_keys,
    shell_command_heads,
)
from tool_time.prerestore import prerestore_start_ms
from tool_time.prior import (
    LatencyPrior,
    build_latency_prior,
    latency_prior_hierarchy,
)
from trace_collect.tool_latency_dataset import discover_trace_files
from trace_collect.trace_data import TraceData

_CONTINUUM_HISTORY_THRESHOLD = 100  # Continuum §4.2, arXiv:2511.02230v6.
_THUNDERAGENT_BUFFER_TOKENS = 100  # backend/state.py at reference commit below.
_THUNDERAGENT_REFERENCE_COMMIT = "7ddc8610270e56d3b109eed8796b3a4360fc67c9"
_POLICY_NAMES = frozenset({"keep", "deadline", "ours", "continuum", "thunderagent"})
PREFILL_COST_SCHEMA_VERSION = 1


def validate_serving_cell(corpus_role: str, policy: str) -> None:
    """Reject evaluation cells whose deployment policy is not certified."""
    if corpus_role == "heldout_eval" and policy == "ours":
        raise ValueError(
            "held-out ours is blocked: the configured static trigger table is an "
            "uncertified deployment-demo approximation"
        )


@dataclass(frozen=True)
class PrefillCostProfile:
    """Offline quadratic prefill-cost fit for one model/hardware pair."""

    model: str
    kv_cache_dtype: str
    quantization: str | None
    device_name: str
    host_name: str
    max_context_tokens: int
    coefficients: tuple[float, float, float]
    overhead_floor_ms: float

    def estimate_ms(self, context_tokens: int) -> float:
        if context_tokens <= 0 or context_tokens > self.max_context_tokens:
            raise ValueError(
                f"context_tokens must be in [1, {self.max_context_tokens}]"
            )
        a, b, c = self.coefficients
        return max(
            self.overhead_floor_ms, (a * context_tokens + b) * context_tokens + c
        )


def load_prefill_cost_profile(
    path: str | Path,
    *,
    expected_model: str,
    expected_kv_cache_dtype: str,
    expected_quantization: str | None,
) -> PrefillCostProfile:
    """Load a measured Continuum prefill profile, failing on config mismatch."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != PREFILL_COST_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported prefill cost schema")
    if payload.get("measurement") != "prefill_recompute_cost":
        raise ValueError(f"{path}: not a prefill_recompute_cost profile")
    if payload.get("model") != expected_model:
        raise ValueError(
            f"{path}: model {payload.get('model')!r} != {expected_model!r}"
        )
    if payload.get("kv_cache_dtype") != expected_kv_cache_dtype:
        raise ValueError(f"{path}: kv_cache_dtype does not match serving config")
    if payload.get("quantization") != expected_quantization:
        raise ValueError(f"{path}: quantization does not match serving config")
    coefficients = payload.get("quadratic_fit", {}).get("coefficients")
    device_name = payload.get("device", {}).get("device")
    host_name = payload.get("device", {}).get("host")
    if (
        not isinstance(coefficients, list)
        or len(coefficients) != 3
        or not all(isinstance(value, (int, float)) for value in coefficients)
        or not isinstance(device_name, str)
        or not device_name
        or not isinstance(host_name, str)
        or not host_name
    ):
        raise ValueError(f"{path}: incomplete quadratic fit or device metadata")
    points = payload.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError(f"{path}: points must be a non-empty list")
    max_context_tokens = max(int(point["context_tokens"]) for point in points)
    overhead_floor_ms = float(payload["overhead_floor_ms"])
    profile = PrefillCostProfile(
        model=expected_model,
        kv_cache_dtype=expected_kv_cache_dtype,
        quantization=expected_quantization,
        device_name=device_name,
        host_name=host_name,
        max_context_tokens=max_context_tokens,
        coefficients=tuple(float(value) for value in coefficients),
        overhead_floor_ms=overhead_floor_ms,
    )
    if not all(
        math.isfinite(value) for value in (*profile.coefficients, overhead_floor_ms)
    ):
        raise ValueError(f"{path}: fit values must be finite")
    if overhead_floor_ms <= 0:
        raise ValueError(f"{path}: overhead_floor_ms must be > 0")
    return profile


@dataclass(frozen=True)
class ToolSpan:
    """One recorded tool execution relative to the preceding LLM finish."""

    tool_name: str
    command: str
    start_offset_ms: float
    end_offset_ms: float


@dataclass(frozen=True)
class TraceTurn:
    """One recorded LLM request and the real gap before its follow-up."""

    messages: tuple[dict[str, Any], ...]
    recorded_prompt_tokens: int
    completion_tokens: int
    gap_ms: float
    tools: tuple[ToolSpan, ...]

    @property
    def tool_signature(self) -> str:
        names = {
            (
                heads[0]
                if (heads := shell_command_heads(tool.command))
                else tool.tool_name
            )
            for tool in self.tools
        }
        return "+".join(sorted(names)) if names else "__no_tool__"


@dataclass(frozen=True)
class TraceProgram:
    """A complete recorded agent program."""

    task_id: str
    trace_path: str
    turns: tuple[TraceTurn, ...]
    source_llm_call_count: int | None = None
    omitted_terminal_llm_calls: int = 0


def prepare_llama_chat_messages(
    messages: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Adapt OpenAI parallel tool calls to Llama's single-call template."""
    prepared: list[dict[str, Any]] = []
    index = 0
    while index < len(messages):
        message = deepcopy(messages[index])
        calls = message.get("tool_calls")
        if calls is None:
            if message.get("role") == "tool":
                raise ValueError("orphan tool result")
            prepared.append(message)
            index += 1
            continue
        if message.get("role") != "assistant":
            raise ValueError("tool_calls are allowed only on assistant messages")
        if not isinstance(calls, list) or not calls:
            raise ValueError("tool_calls must be a non-empty list")
        normalized: list[dict[str, Any]] = []
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("id"), str):
                raise ValueError("each tool call requires a string id")
            function = call.get("function")
            if not isinstance(function, dict):
                raise ValueError("each tool call requires a function object")
            if not isinstance(function.get("name"), str):
                raise ValueError("each tool call requires a string function name")
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as error:
                    raise ValueError("tool call arguments must be JSON") from error
            if not isinstance(arguments, dict):
                raise ValueError("tool call arguments must decode to an object")
            call["function"] = {**function, "arguments": arguments}
            normalized.append(call)

        results = messages[index + 1 : index + 1 + len(normalized)]
        if len(results) != len(normalized):
            raise ValueError(
                "parallel tool calls require matching adjacent tool results"
            )
        by_id: dict[str, dict[str, Any]] = {}
        for result in results:
            if not isinstance(result, dict):
                raise ValueError(
                    "parallel tool calls require matching adjacent tool results"
                )
            result_id = result.get("tool_call_id")
            if (
                result.get("role") != "tool"
                or not isinstance(result_id, str)
                or result_id in by_id
            ):
                raise ValueError(
                    "parallel tool calls require matching adjacent tool results"
                )
            by_id[result_id] = result
        call_ids = [call["id"] for call in normalized]
        if len(by_id) != len(normalized) or set(by_id) != set(call_ids):
            raise ValueError(
                "parallel tool calls require matching adjacent tool results"
            )
        if len(normalized) == 1:
            message["tool_calls"] = normalized
            prepared.extend((message, deepcopy(by_id[normalized[0]["id"]])))
            index += 2
            continue
        for call in normalized:
            single = deepcopy(message)
            single["tool_calls"] = [call]
            prepared.extend((single, deepcopy(by_id[call["id"]])))
        index += 1 + len(normalized)
    return tuple(prepared)


@dataclass(frozen=True)
class ContinuumProfile:
    """Historical inputs required by Continuum's published TTL equation."""

    global_gap_ms: tuple[float, ...]
    gap_ms_by_tool: dict[str, tuple[float, ...]]
    memoryfulness: float


@dataclass(frozen=True)
class PrerestoreProfile:
    """Latency-prior hierarchy used by the frozen pre-restore optimizer."""

    prior: LatencyPrior
    max_prefix_depth: int
    skip_leading_cd: bool
    min_tool_history: int
    min_profile_tasks: int


@dataclass(frozen=True)
class RetentionPlan:
    """Finished-request action passed to the vLLM connector."""

    policy: str
    action: str
    expire_ms: float | None
    tool_signature: str
    source: str
    prerestore_ms: float | None = None
    prerestore_source: str | None = None


def _parse_command(tool_args: Any, command_field: str) -> str:
    if isinstance(tool_args, str):
        try:
            tool_args = json.loads(tool_args)
        except json.JSONDecodeError:
            return ""
    if not isinstance(tool_args, dict):
        return ""
    command = tool_args.get(command_field)
    return command if isinstance(command, str) else ""


def _required_number(row: dict[str, Any], field: str, path: Path) -> float:
    value = row.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{path}: action lacks numeric {field}")
    return float(value)


def _trace_program(path: Path, *, command_field: str) -> TraceProgram:
    trace = TraceData.load(path)
    task_id = trace.metadata.get("instance_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError(f"{path}: trace metadata lacks a non-empty instance_id")

    llm_calls = [row for row in trace.actions if row.get("action_type") == "llm_call"]
    if not llm_calls:
        raise ValueError(f"{path}: trace has no llm_call actions")
    llm_calls.sort(key=lambda row: _required_number(row, "ts_start", path))
    tools = [row for row in trace.actions if row.get("action_type") == "tool_exec"]

    omitted_terminal_llm_calls = 0
    turns: list[TraceTurn] = []
    for index, call in enumerate(llm_calls):
        data = call.get("data") or {}
        messages = data.get("messages_in")
        prompt_tokens = data.get("prompt_tokens")
        completion_tokens = data.get("completion_tokens")
        if not isinstance(messages, list) or not all(
            isinstance(item, dict) for item in messages
        ):
            raise ValueError(f"{path}: llm_call {index} lacks messages_in objects")
        if (
            not isinstance(prompt_tokens, int)
            or isinstance(prompt_tokens, bool)
            or prompt_tokens < 0
            or not isinstance(completion_tokens, int)
            or isinstance(completion_tokens, bool)
            or completion_tokens < 0
        ):
            raise ValueError(f"{path}: llm_call {index} has invalid token counts")
        if prompt_tokens == 0 or completion_tokens == 0:
            tail = [row.get("data") or {} for row in llm_calls[index:]]
            if any(
                isinstance(row.get("prompt_tokens"), int)
                and row["prompt_tokens"] > 0
                and isinstance(row.get("completion_tokens"), int)
                and row["completion_tokens"] > 0
                for row in tail
            ):
                raise ValueError(
                    f"{path}: zero-token llm_call is not a terminal suffix"
                )
            omitted_terminal_llm_calls = len(tail)
            break

        call_end = _required_number(call, "ts_end", path)
        next_start = (
            _required_number(llm_calls[index + 1], "ts_start", path)
            if index + 1 < len(llm_calls)
            else call_end
        )
        if next_start < call_end:
            raise ValueError(
                f"{path}: llm_call {index + 1} starts before call {index} ends"
            )

        spans: list[ToolSpan] = []
        if index + 1 < len(llm_calls):
            for tool in tools:
                start = _required_number(tool, "ts_start", path)
                if start < call_end or start >= next_start:
                    continue
                end = _required_number(tool, "ts_end", path)
                if end < start:
                    raise ValueError(f"{path}: tool_exec ends before it starts")
                tool_data = tool.get("data") or {}
                tool_name = tool_data.get("tool_name")
                if not isinstance(tool_name, str) or not tool_name:
                    raise ValueError(f"{path}: tool_exec lacks a non-empty tool_name")
                spans.append(
                    ToolSpan(
                        tool_name=tool_name,
                        command=_parse_command(
                            tool_data.get("tool_args"), command_field
                        ),
                        start_offset_ms=(start - call_end) * 1000.0,
                        end_offset_ms=(end - call_end) * 1000.0,
                    )
                )
        spans.sort(
            key=lambda span: (span.start_offset_ms, span.end_offset_ms, span.tool_name)
        )
        turns.append(
            TraceTurn(
                messages=tuple(messages),
                recorded_prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                gap_ms=(next_start - call_end) * 1000.0,
                tools=tuple(spans),
            )
        )
    if not turns:
        raise ValueError(f"{path}: trace has no replayable llm_call actions")
    return TraceProgram(
        task_id=task_id,
        trace_path=str(path),
        turns=tuple(turns),
        source_llm_call_count=len(llm_calls),
        omitted_terminal_llm_calls=omitted_terminal_llm_calls,
    )


def load_trace_programs(
    trace_root: str | Path,
    *,
    task_ids: Sequence[str] | None = None,
    limit: int | None = None,
    seed: int = 0,
    command_field: str = "command",
) -> list[TraceProgram]:
    """Load a deterministic real-program sample, failing on identity drift."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be > 0")
    paths = discover_trace_files([Path(trace_root)])
    by_task: dict[str, Path] = {}
    for path in paths:
        trace = TraceData.load(path)
        task_id = trace.metadata.get("instance_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"{path}: trace metadata lacks instance_id")
        if task_id in by_task:
            raise ValueError(f"duplicate trace for task {task_id!r}")
        by_task[task_id] = path

    if task_ids is not None:
        requested = list(task_ids)
        if len(requested) != len(set(requested)):
            raise ValueError("task_ids contains duplicates")
        missing = sorted(set(requested) - set(by_task))
        if missing:
            raise ValueError(f"trace root misses task IDs: {missing[:5]}")
        selected = requested
    else:
        selected = sorted(by_task)
        Random(seed).shuffle(selected)
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("no trace programs selected")
    return [
        _trace_program(by_task[task_id], command_field=command_field)
        for task_id in selected
    ]


def _memoryfulness(programs: Sequence[TraceProgram]) -> float:
    progress: list[float] = []
    remaining: list[float] = []
    for program in programs:
        total = len(program.turns)
        for served in range(1, total + 1):
            progress.append(float(served))
            remaining.append(float(total - served))
    if len(progress) < 2 or len(set(progress)) < 2 or len(set(remaining)) < 2:
        return 0.0
    return -statistics.correlation(progress, remaining)


def build_continuum_profile(programs: Sequence[TraceProgram]) -> ContinuumProfile:
    """Build Continuum's empirical gap CDF and memoryfulness from profile data."""
    if not programs:
        raise ValueError("Continuum profile requires at least one program")
    by_tool: dict[str, list[float]] = {}
    global_gaps: list[float] = []
    for program in programs:
        gap_turns = len(program.turns) - 1 + bool(program.omitted_terminal_llm_calls)
        for turn in program.turns[:gap_turns]:
            if turn.gap_ms < 0:
                raise ValueError("tool gaps must be non-negative")
            global_gaps.append(turn.gap_ms)
            by_tool.setdefault(turn.tool_signature, []).append(turn.gap_ms)
    if not global_gaps:
        raise ValueError("Continuum profile has no inter-turn gaps")
    return ContinuumProfile(
        global_gap_ms=tuple(global_gaps),
        gap_ms_by_tool={key: tuple(values) for key, values in by_tool.items()},
        memoryfulness=_memoryfulness(programs),
    )


def build_prerestore_profile(
    programs: Sequence[TraceProgram],
    *,
    max_prefix_depth: int,
    skip_leading_cd: bool,
    min_tool_history: int,
    min_profile_tasks: int,
) -> PrerestoreProfile:
    """Fit the shipped hierarchy on a profile corpus, never replay outcomes."""
    rows = [
        {
            "source_trace": program.trace_path,
            "task_id": program.task_id,
            "tool_name": tool.tool_name,
            "latency_ms": tool.end_offset_ms - tool.start_offset_ms,
            "tool_args": {"command": tool.command},
        }
        for program in programs
        for turn in program.turns
        for tool in turn.tools
    ]
    keyer = make_row_command_prefix_keys(
        "command",
        max_depth=max_prefix_depth,
        skip_leading_cd=skip_leading_cd,
    )
    return PrerestoreProfile(
        prior=build_latency_prior(rows, row_group_keys=keyer),
        max_prefix_depth=max_prefix_depth,
        skip_leading_cd=skip_leading_cd,
        min_tool_history=min_tool_history,
        min_profile_tasks=min_profile_tasks,
    )


def _prerestore_for_tool(
    profile: PrerestoreProfile,
    tool: ToolSpan,
    *,
    swap_trigger_ms: float,
    restore_cost_ms: float,
) -> tuple[float | None, str]:
    keys = command_prefix_keys(
        tool.tool_name,
        tool.command,
        max_depth=profile.max_prefix_depth,
        skip_leading_cd=profile.skip_leading_cd,
    )
    node = latency_prior_hierarchy(
        profile.prior,
        tool.tool_name,
        tuple(keys),
        min_tool_history=profile.min_tool_history,
        min_profile_tasks=profile.min_profile_tasks,
    )[-1]
    return (
        prerestore_start_ms(
            node,
            swap_trigger_ms=swap_trigger_ms,
            restore_cost_ms=restore_cost_ms,
        ),
        node.source,
    )


def continuum_ttl_ms(
    profile: ContinuumProfile,
    tool_signature: str,
    *,
    queue_delay_ms: float,
    prefill_reload_ms: float,
) -> tuple[float, str]:
    """Solve Continuum Eq. 2 by enumerating observed durations plus zero."""
    if queue_delay_ms < 0 or prefill_reload_ms < 0:
        raise ValueError("Continuum cost inputs must be non-negative")
    global_history = profile.global_gap_ms
    if len(global_history) <= _CONTINUUM_HISTORY_THRESHOLD:
        reward_seconds = prefill_reload_ms / 1000.0
        ttl_seconds = math.log(reward_seconds) if reward_seconds > 1.0 else 0.0
        return ttl_seconds * 1000.0, "exp1_cold_start"

    tool_history = profile.gap_ms_by_tool.get(tool_signature, ())
    if len(tool_history) <= _CONTINUUM_HISTORY_THRESHOLD:
        history = global_history
        source = "global_cdf"
    else:
        history = tool_history
        source = "tool_cdf"
    benefit = queue_delay_ms * profile.memoryfulness + prefill_reload_ms
    candidates = [0.0, *sorted(set(history))]
    best_ttl = 0.0
    best_reward = 0.0
    for ttl in candidates:
        probability = sum(duration <= ttl for duration in history) / len(history)
        reward = probability * benefit - ttl
        if reward > best_reward:
            best_reward = reward
            best_ttl = ttl
    return best_ttl, source


def build_retention_plan(
    policy: str,
    turn: TraceTurn,
    *,
    deadline_ms: float,
    trigger_table: TriggerTable | None = None,
    continuum_profile: ContinuumProfile | None = None,
    prerestore_profile: PrerestoreProfile | None = None,
    restore_cost_ms: float = 0.0,
    queue_delay_ms: float = 0.0,
    prefill_reload_ms: float = 0.0,
) -> RetentionPlan | None:
    """Choose a finished-turn action without reading the realized gap length."""
    if policy not in _POLICY_NAMES:
        raise ValueError(f"unknown policy {policy!r}")
    if deadline_ms <= 0:
        raise ValueError("deadline_ms must be > 0")
    tools = turn.tools or (ToolSpan("__no_tool__", "", 0.0, 0.0),)

    if policy == "keep":
        return RetentionPlan(
            policy,
            "offload",
            None,
            turn.tool_signature,
            "keep_until_next_turn",
        )
    if policy in {"deadline", "ours"}:
        if policy == "ours":
            if trigger_table is None:
                raise ValueError("ours requires a trigger table")
            if not math.isclose(
                trigger_table.deadline_ms, deadline_ms, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError(
                    "trigger-table deadline differs from workload deadline"
                )
            triggers = [
                lookup_trigger(trigger_table, tool.tool_name, tool.command).trigger_ms
                for tool in tools
            ]
            source = "uncertified_deployment_demo_table"
        else:
            triggers = [deadline_ms] * len(tools)
            source = "deadline"
        expiry = min(
            tool.start_offset_ms + trigger
            for tool, trigger in zip(tools, triggers, strict=True)
        )
        prerestore_candidates: list[tuple[float, str]] = []
        if prerestore_profile is not None:
            if restore_cost_ms <= 0:
                raise ValueError("pre-restore requires restore_cost_ms > 0")
            for tool, trigger in zip(tools, triggers, strict=True):
                start, prior_source = _prerestore_for_tool(
                    prerestore_profile,
                    tool,
                    swap_trigger_ms=trigger,
                    restore_cost_ms=restore_cost_ms,
                )
                if start is not None:
                    prerestore_candidates.append(
                        (tool.start_offset_ms + start, prior_source)
                    )
        prerestore, prerestore_source = (
            min(prerestore_candidates) if prerestore_candidates else (None, None)
        )
        return RetentionPlan(
            policy,
            "offload",
            expiry,
            turn.tool_signature,
            source,
            prerestore,
            prerestore_source,
        )
    if policy == "continuum":
        if continuum_profile is None:
            raise ValueError("continuum requires profile history")
        ttl, source = continuum_ttl_ms(
            continuum_profile,
            turn.tool_signature,
            queue_delay_ms=queue_delay_ms,
            prefill_reload_ms=prefill_reload_ms,
        )
        if ttl <= 0:
            return None
        return RetentionPlan(policy, "release", ttl, turn.tool_signature, source)
    return RetentionPlan(
        policy,
        "pressure",
        None,
        turn.tool_signature,
        f"ThunderAgent@{_THUNDERAGENT_REFERENCE_COMMIT}",
    )


def policy_provenance() -> dict[str, Any]:
    return {
        "prerestore": {
            "optimizer": "exact empirical utility maximizer",
            "source": "tool_time.prerestore.prerestore_start_ms",
            "candidate_rule": "{g} union {L-R>g} union {L>g}; latest positive tie",
        },
        "continuum": {
            "paper": "arXiv:2511.02230v6",
            "history_threshold": _CONTINUUM_HISTORY_THRESHOLD,
            "equation": 2,
            "scheduling_priority": (
                "TTL-hit requests, then program-level arrival order"
            ),
            "pressure_unpin_order": "latest program arrival first",
        },
        "thunderagent": {
            "repository": "https://github.com/ThunderAgent-org/ThunderAgent",
            "commit": _THUNDERAGENT_REFERENCE_COMMIT,
            "buffer_per_program_tokens": _THUNDERAGENT_BUFFER_TOKENS,
            "eviction_order": (
                "ACTING smallest-total-tokens first, then mark REASONING "
                "smallest-total-tokens first"
            ),
            "resume_order": "single-backend best-fit decreasing",
        },
    }


__all__ = [
    "PrefillCostProfile",
    "ContinuumProfile",
    "PrerestoreProfile",
    "RetentionPlan",
    "ToolSpan",
    "TraceProgram",
    "TraceTurn",
    "build_continuum_profile",
    "build_prerestore_profile",
    "build_retention_plan",
    "continuum_ttl_ms",
    "load_prefill_cost_profile",
    "load_trace_programs",
    "policy_provenance",
    "prepare_llama_chat_messages",
    "validate_serving_cell",
]
