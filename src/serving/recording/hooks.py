"""Forward hooks and artifact writer for HF internal recordings."""

from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np

from serving.recording.recording import (
    DecodeRingBuffer,
    RecordingConfig,
    expert_load_per_segment,
    heavy_hitter,
    padded_top_k,
    segment_bucket,
    select_query_positions,
    token_segment_ids,
)

if TYPE_CHECKING:
    # Lazy: avoid forcing transformers import via kv_policies at module load.
    from serving.kv_policies.recorder import KVEvictionRecorder
    from serving.recording.attention_bus import AttentionBus


_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn$")
_GATE_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.gate$")
_LLM_ACTION_ID_RE = re.compile(r"^llm_(\d+)$")


def _layer_index(module_name: str) -> int | None:
    match = _LAYER_RE.search(module_name)
    if match:
        return int(match.group(1))
    if module_name.endswith(".self_attn") or module_name == "self_attn":
        return -1
    return None


def _gate_layer_index(module_name: str) -> int | None:
    match = _GATE_RE.search(module_name)
    return int(match.group(1)) if match else None


def _as_numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().cpu().numpy()


@dataclass(frozen=True)
class _PendingNumpyArray:
    """CPU-staged tensor whose NumPy view is materialized at flush time."""

    tensor: Any
    dtype: np.dtype
    source_device: Any | None
    _synchronized: bool = False

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self.tensor.shape)

    def mark_synchronized(self) -> None:
        object.__setattr__(self, "_synchronized", True)

    def synchronize(self) -> None:
        if self.source_device is None or self._synchronized:
            return
        import torch

        torch.cuda.synchronize(self.source_device)
        self.mark_synchronized()

    def materialize(self) -> np.ndarray:
        self.synchronize()
        array = self.tensor.numpy()
        if array.dtype != self.dtype:
            return array.astype(self.dtype, copy=False)
        return array

    def __array__(self, dtype: Any = None) -> np.ndarray:
        array = self.materialize()
        if dtype is None:
            return array
        return array.astype(dtype, copy=False)

    def astype(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return self.materialize().astype(*args, **kwargs)

    def copy(self) -> np.ndarray:
        return self.materialize().copy()


def _stage_numpy(tensor: Any, dtype: Any) -> _PendingNumpyArray:
    import torch

    np_dtype = np.dtype(dtype)
    torch_dtype = {
        np.dtype(np.float32): torch.float32,
        np.dtype(np.int32): torch.int32,
    }.get(np_dtype)
    staged = tensor.detach()
    if torch_dtype is not None and staged.dtype != torch_dtype:
        staged = staged.to(dtype=torch_dtype)
    source_device = staged.device if staged.device.type == "cuda" else None
    if source_device is not None:
        cpu_tensor = torch.empty_like(
            staged,
            device=torch.device("cpu"),
            pin_memory=True,
        )
        cpu_tensor.copy_(staged, non_blocking=True)
    else:
        cpu_tensor = staged.to(device=torch.device("cpu"))
    return _PendingNumpyArray(
        cpu_tensor,
        np_dtype,
        source_device,
    )


def _materialize_array(value: Any) -> np.ndarray:
    if isinstance(value, _PendingNumpyArray):
        return value.materialize()
    return value


def _synchronize_pending_arrays(records: list[dict[str, Any]]) -> None:
    pending: list[_PendingNumpyArray] = []
    devices: dict[str, Any] = {}
    for record in records:
        for value in record.values():
            if isinstance(value, _PendingNumpyArray) and value.source_device is not None:
                pending.append(value)
                devices[str(value.source_device)] = value.source_device
    if not devices:
        return

    import torch

    for device in devices.values():
        torch.cuda.synchronize(device)
    for value in pending:
        value.mark_synchronized()


def _arg_or_kw(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    index: int,
    name: str,
    *,
    default: Any = None,
) -> Any:
    if len(args) > index:
        return args[index]
    return kwargs.get(name, default)


def _project_query_states(module: Any, hidden_states: Any) -> Any:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    return module.q_norm(module.q_proj(hidden_states).reshape(hidden_shape)).transpose(1, 2)


def _project_key_states(module: Any, hidden_states: Any) -> Any:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    return module.k_norm(module.k_proj(hidden_states).reshape(hidden_shape)).transpose(1, 2)


def _apply_rotary_to_states(states: Any, position_embeddings: tuple[Any, Any]) -> Any:
    import torch

    cos, sin = position_embeddings
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    left = states[..., : states.shape[-1] // 2]
    right = states[..., states.shape[-1] // 2 :]
    rotated = torch.cat((-right, left), dim=-1)
    return (states * cos) + (rotated * sin)


def _select_rotary_positions(
    position_embeddings: tuple[Any, Any],
    row_indices: list[int],
) -> tuple[Any, Any]:
    import torch

    index: Any | None = None
    selected = []
    for tensor in position_embeddings:
        if index is None:
            index = torch.as_tensor(row_indices, dtype=torch.long, device=tensor.device)
        selected.append(tensor.index_select(-2, index))
    return selected[0], selected[1]


def _nonempty_tensor(value: Any) -> Any | None:
    if value is None or not hasattr(value, "numel"):
        return None
    if int(value.numel()) == 0:
        return None
    return value


def _cached_key_states(past_key_values: Any, layer_idx: int) -> Any | None:
    if past_key_values is None:
        return None
    try:
        return _nonempty_tensor(past_key_values[layer_idx][0])
    except (AttributeError, KeyError, IndexError, TypeError):
        pass

    layers = getattr(past_key_values, "layers", None)
    if layers is not None:
        try:
            return _nonempty_tensor(getattr(layers[layer_idx], "keys", None))
        except (IndexError, TypeError):
            pass

    key_cache = getattr(past_key_values, "key_cache", None)
    if key_cache is not None:
        try:
            return _nonempty_tensor(key_cache[layer_idx])
        except (IndexError, TypeError):
            pass
    return None


def _load_trace_llm_calls(trace_path: Path) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "action" and record.get("action_type") == "llm_call":
                calls.append(record)
    return calls


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_trace_indices(values: list[int | None]) -> list[int] | None:
    if not values or any(value is None for value in values):
        return None
    concrete = [int(value) for value in values if value is not None]
    if concrete != sorted(concrete) or len(set(concrete)) != len(concrete):
        return None
    offset = concrete[0]
    if offset not in {0, 1}:
        return None
    return [value - offset for value in concrete]


def _trace_call_indices(trace_calls: list[dict[str, Any]]) -> tuple[list[int], str] | None:
    iteration_indices = _normalize_trace_indices(
        [_int_or_none(call.get("iteration")) for call in trace_calls]
    )
    if iteration_indices is not None:
        return iteration_indices, "iteration"

    action_id_values: list[int | None] = []
    for call in trace_calls:
        action_id = call.get("action_id")
        match = _LLM_ACTION_ID_RE.match(action_id) if isinstance(action_id, str) else None
        action_id_values.append(_int_or_none(match.group(1)) if match else None)
    action_id_indices = _normalize_trace_indices(action_id_values)
    if action_id_indices is not None:
        return action_id_indices, "action_id"
    return None


class LayerCapturer:
    """Capture reduced attention and routing tensors for one HF model."""

    def __init__(
        self,
        model: Any,
        *,
        config: RecordingConfig,
        model_summary: dict[str, Any],
        kv_recorder: "KVEvictionRecorder | None" = None,
        attention_bus: "AttentionBus | None" = None,
    ) -> None:
        self.config = config
        self.model_summary = dict(model_summary)
        self._handles = []
        self._attempt_dir: Path | None = None
        self._session: dict[str, Any] | None = None
        self._prefill_records: list[dict[str, Any]] = []
        self._decode_records = DecodeRingBuffer(config.decode_window)
        self._routing_records: list[dict[str, Any]] = []
        self._meta: dict[str, Any] = {}
        self._attention_suspended = 0
        # KV eviction recorder is per-call; lifecycle is owned by the caller
        # (HFRecordingProvider in step 3+) which swaps it before each
        # `recording_session()`. LayerCapturer only flushes whatever the
        # current recorder buffered when the session ends.
        self._kv_recorder: "KVEvictionRecorder | None" = kv_recorder
        # Bus is per-provider and lives across calls. None preserves the
        # pre-step-5 behaviour byte-for-byte (no publish at all).
        self._attention_bus: "AttentionBus | None" = attention_bus
        # Attempt-level KV policy summary written to meta.json on
        # `finish_attempt`. Provider sets via `set_kv_policy_meta(...)`
        # before `start_attempt`. Default None = `--kv-policy none`.
        self._kv_policy_meta: dict[str, Any] | None = None
        self._attempt_extra_meta: dict[str, Any] = {}
        # Optional manual override for `config.max_prefill_queries`. None = use
        # the frozen RecordingConfig value. H2O full-prefill scoring no longer
        # uses this path; it streams full rows in bounded chunks below so the
        # recording sample cap stays intact.
        self._max_prefill_queries_override: int | None = None

        n_attention_modules = 0
        n_gate_modules = 0
        for name, module in model.named_modules():
            layer = _layer_index(name)
            if layer is not None:
                n_attention_modules += 1
                self._handles.append(
                    module.register_forward_hook(self._hook(layer), with_kwargs=True)
                )
            gate_layer = _gate_layer_index(name)
            if gate_layer is not None:
                n_gate_modules += 1
                self._handles.append(module.register_forward_hook(self._gate_hook(gate_layer)))
        if n_attention_modules == 0:
            raise ValueError("no attention modules matched '.layers.<n>.self_attn'")
        self.model_summary["router_capture_mode"] = (
            "gate_forward_hook" if n_gate_modules else "none"
        )
        self._decode_records = DecodeRingBuffer(
            config.decode_window * max(1, n_attention_modules)
        )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def kv_recorder(self) -> "KVEvictionRecorder | None":
        """Currently-attached KV eviction recorder, or None."""
        return self._kv_recorder

    def set_kv_recorder(self, recorder: "KVEvictionRecorder | None") -> None:
        """Swap the KV recorder. Caller (provider) drives the per-call lifecycle."""
        self._kv_recorder = recorder

    def set_kv_policy_meta(self, meta: dict[str, Any] | None) -> None:
        """Stash the attempt-level KV policy summary for `meta.json`.

        Provider populates this in `start_attempt` (or just before) so that
        `kv_policy.prefill_score_bias` is recorded once per attempt rather
        than per call. Pass None for `--kv-policy none`.
        """
        self._kv_policy_meta = dict(meta) if meta is not None else None

    def set_attempt_extra_meta(self, meta: dict[str, Any]) -> None:
        """Merge provider-owned attempt metadata into the next meta.json write."""
        self._attempt_extra_meta.update(dict(meta))

    def start_attempt(self, recordings_dir: Path) -> None:
        self._attempt_dir = Path(recordings_dir)
        self._attempt_dir.mkdir(parents=True, exist_ok=True)
        self._attempt_extra_meta = {}
        self._meta = {
            "model": self.model_summary,
            "recording_config": asdict(self.config),
            "iters": [],
        }
        if getattr(self, "_kv_policy_meta", None) is not None:
            self._meta["kv_policy"] = dict(self._kv_policy_meta)

    def finish_attempt(self, trace_path: Path | None = None) -> None:
        if self._attempt_dir is None:
            return
        if self._attempt_extra_meta:
            self._meta.update(dict(self._attempt_extra_meta))
        if trace_path is not None and trace_path.exists():
            self._align_meta_to_trace(trace_path)
        (self._attempt_dir / "meta.json").write_text(
            json.dumps(self._meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._attempt_dir = None

    def _align_meta_to_trace(self, trace_path: Path) -> None:
        trace_calls = _load_trace_llm_calls(trace_path)
        trace_indices = _trace_call_indices(trace_calls)
        original_iters = [dict(item) for item in self._meta.get("iters", [])]
        aligned: list[dict[str, Any]] = []
        orphaned: list[dict[str, Any]] = []
        trace_call_by_idx: dict[int, dict[str, Any]] = {}
        alignment_source = "unavailable"
        if trace_indices is not None:
            indices, alignment_source = trace_indices
            trace_call_by_idx = dict(zip(indices, trace_calls))
        for item in original_iters:
            call_idx = int(item.get("call_idx", -1))
            trace_call = trace_call_by_idx.get(call_idx)
            if trace_call is not None:
                item["trace_action_id"] = trace_call.get("action_id")
                item["trace_iteration"] = trace_call.get("iteration")
                aligned.append(item)
            else:
                item["trace_action_id"] = None
                item["trace_iteration"] = None
                item["trace_alignment_status"] = "orphan_no_llm_action"
                orphaned.append(item)
        self._meta["iters"] = aligned
        if orphaned:
            self._meta["orphan_iters"] = orphaned
        self._meta["alignment"] = {
            "trace_path": str(trace_path),
            "trace_llm_actions": len(trace_calls),
            "recording_iters": len(original_iters),
            "aligned_iters": len(aligned),
            "orphan_iters": len(orphaned),
            "alignment_source": alignment_source,
        }

    @contextmanager
    def recording_session(
        self,
        *,
        call_idx: int,
        segments: list[dict[str, Any]],
        input_token_count: int,
    ) -> Iterator[None]:
        if self._attempt_dir is None:
            raise RuntimeError("start_attempt() must be called before recording")
        if self._session is not None:
            raise RuntimeError("nested recording sessions are not supported")

        iter_dir = self._attempt_dir / f"iter_{call_idx:04d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        session_segments = []
        for segment in segments:
            payload = dict(segment)
            if "first_seen_call" not in payload:
                payload["first_seen_call"] = int(call_idx)
                payload["first_seen_call_inferred"] = True
            else:
                payload.setdefault("first_seen_call_inferred", False)
            session_segments.append(payload)
        self._session = {
            "call_idx": call_idx,
            "iter_dir": iter_dir,
            "segments": session_segments,
            "input_token_count": input_token_count,
            "started_at": time.time(),
            "generated_segment_id": len(segments),
            "flushed": False,
        }
        self._prefill_records = []
        self._decode_records.clear()
        self._routing_records = []
        try:
            yield
        except BaseException:
            self._prefill_records = []
            self._decode_records.clear()
            self._routing_records = []
            self._session = None
            raise
        finally:
            if self._session is not None and self._session.get("flushed"):
                self._session = None

    @contextmanager
    def suspend_attention(self) -> Iterator[None]:
        self._attention_suspended += 1
        try:
            yield
        finally:
            self._attention_suspended -= 1

    @contextmanager
    def unbounded_prefill_queries(self) -> Iterator[None]:
        """Temporarily disable the prefill query-row sample cap.

        This is retained for debugging and direct recording experiments. The
        paper-faithful H2O path does not use it because uncapping the recording
        rows can materialize a full QxK tensor on long prompts. Restores the
        previous override on exit; nesting is supported by stashing the prior
        value in a local.
        """
        previous = self._max_prefill_queries_override
        # `2**31 - 1` is what `select_query_positions` treats as "no cap"
        # since `query_len <= max_queries` short-circuits to the identity
        # range. Practical sequence lengths sit comfortably below this.
        self._max_prefill_queries_override = 2_147_483_647
        try:
            yield
        finally:
            self._max_prefill_queries_override = previous

    def _hook(self, layer: int):
        def capture(
            module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            _output: Any,
        ) -> None:
            if self._session is None or self._attention_suspended:
                return
            self._capture_sampled_attention(layer, module, args, kwargs)

        return capture

    def _gate_hook(self, layer: int):
        def capture(module: Any, args: tuple[Any, ...], output: Any) -> None:
            del module, args
            if self._session is None:
                return
            self._record_router_tensor(layer=layer, path=("gate",), tensor=output)

        return capture

    def _token_ids_for_key_len(self, key_len: int):
        assert self._session is not None
        return token_segment_ids(
            key_len,
            self._session["segments"],
            generated_segment_id=self._session["generated_segment_id"],
        )

    def _capture_sampled_attention(
        self,
        layer: int,
        module: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        hidden_states = _arg_or_kw(args, kwargs, 0, "hidden_states")
        position_embeddings = _arg_or_kw(args, kwargs, 1, "position_embeddings")
        attention_mask = _arg_or_kw(args, kwargs, 2, "attention_mask", default=None)
        past_key_values = _arg_or_kw(
            args,
            kwargs,
            3,
            "past_key_values",
            default=kwargs.get("past_key_value"),
        )
        if hidden_states is None or position_embeddings is None:
            raise ValueError("attention hook requires hidden_states and position_embeddings")
        if hidden_states.ndim != 3:
            raise ValueError(f"hidden_states must be rank 3, got {hidden_states.shape}")

        assert self._session is not None
        query_len = int(hidden_states.shape[-2])
        # Manual override path: when `unbounded_prefill_queries()` is active,
        # treat every query row as sampled. Normal H2O full-prefill scoring
        # leaves this cap bounded and streams full rows only to H2O consumers.
        max_queries = (
            self._max_prefill_queries_override
            if self._max_prefill_queries_override is not None
            else self.config.max_prefill_queries
        )
        row_indices = (
            [0]
            if query_len == 1
            else select_query_positions(query_len, max_queries)
        )
        key_states = self._key_states(
            module=module,
            layer=layer,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
        )
        key_len = int(key_states.shape[-2])
        phase = (
            "decode"
            if query_len == 1 and key_len > self._session["input_token_count"]
            else "prefill"
        )
        query_positions = [key_len - query_len + idx for idx in row_indices]
        if (
            phase == "prefill"
            and self._attention_bus is not None
            and self._attention_bus.has_full_prefill_consumers()
        ):
            attn_rows = self._full_prefill_attention_rows(
                module=module,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                key_states=key_states,
                row_indices=row_indices,
                layer=layer,
            )
        else:
            attn_rows = self._sampled_attention_rows(
                module=module,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                key_states=key_states,
                row_indices=row_indices,
                query_positions=query_positions,
                layer=layer,
            )
        token_ids = self._token_ids_for_key_len(key_len).to(device=attn_rows.device)
        n_segments = self._session["generated_segment_id"] + 1

        segment_mass = segment_bucket(attn_rows, token_ids, n_segments)
        top_indices, top_weights = padded_top_k(attn_rows, self.config.attention_top_k)
        hitter_indices, hitter_weights = heavy_hitter(attn_rows, self.config.attention_top_k)
        n_heads = int(attn_rows.shape[0] // len(row_indices))
        head_indices = np.repeat(
            np.arange(n_heads, dtype=np.int32), len(row_indices)
        )
        record = {
            "layer": layer,
            "phase": phase,
            "decode_step": max(0, key_len - self._session["input_token_count"] - 1)
            if phase == "decode"
            else -1,
            "query_heads": head_indices,
            "query_positions": np.tile(
                np.asarray(query_positions, dtype=np.int32), n_heads
            ),
            "segment_mass": _stage_numpy(segment_mass, np.float32),
            "topk_indices": _stage_numpy(top_indices, np.int32),
            "topk_weights": _stage_numpy(top_weights, np.float32),
            "heavy_indices": _stage_numpy(hitter_indices[0], np.int32),
            "heavy_weights": _stage_numpy(hitter_weights[0], np.float32),
        }
        if phase == "decode":
            self._decode_records.append(record)
        else:
            self._prefill_records.append(record)

    def _key_states(
        self,
        *,
        module: Any,
        layer: int,
        hidden_states: Any,
        position_embeddings: tuple[Any, Any],
        past_key_values: Any,
    ) -> Any:
        layer_idx = int(getattr(module, "layer_idx", layer))
        cached = _cached_key_states(past_key_values, layer_idx)
        if cached is not None:
            return cached
        key_states = _project_key_states(module, hidden_states)
        return _apply_rotary_to_states(key_states, position_embeddings)

    def _sampled_attention_rows(
        self,
        *,
        module: Any,
        hidden_states: Any,
        position_embeddings: tuple[Any, Any],
        attention_mask: Any,
        key_states: Any,
        row_indices: list[int],
        query_positions: list[int],
        layer: int,
        full_prefill: bool = False,
    ) -> Any:
        attn = self._attention_tensor(
            module=module,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            key_states=key_states,
            row_indices=row_indices,
            query_positions=query_positions,
        )
        self._publish_attention(
            layer=layer,
            attn=attn,
            query_positions=query_positions,
            key_len=int(key_states.shape[-2]),
            full_prefill=full_prefill,
        )
        return attn[0].reshape(attn.shape[1] * len(row_indices), int(key_states.shape[-2]))

    def _full_prefill_attention_rows(
        self,
        *,
        module: Any,
        hidden_states: Any,
        position_embeddings: tuple[Any, Any],
        attention_mask: Any,
        key_states: Any,
        row_indices: list[int],
        layer: int,
    ) -> Any:
        """Publish all prefill rows to full-prefill consumers in bounded chunks.

        This gives H2O paper-faithful prefill scores without constructing a
        full `(heads, query_len, key_len)` attention tensor at once. We retain
        only the normal sampled rows for `attention.npz`, preserving the bounded
        recording footprint.
        """
        import torch

        query_len = int(hidden_states.shape[-2])
        key_len = int(key_states.shape[-2])
        if query_len <= 0:
            raise ValueError(f"query_len must be positive, got {query_len}")
        chunk_size = max(1, int(self.config.max_prefill_queries))
        selected: dict[int, Any] = {}
        selected_set = set(int(idx) for idx in row_indices)
        base_position = key_len - query_len

        for start in range(0, query_len, chunk_size):
            stop = min(query_len, start + chunk_size)
            chunk_rows = list(range(start, stop))
            chunk_positions = [base_position + idx for idx in chunk_rows]
            attn = self._attention_tensor(
                module=module,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                key_states=key_states,
                row_indices=chunk_rows,
                query_positions=chunk_positions,
            )
            self._publish_attention(
                layer=layer,
                attn=attn,
                query_positions=chunk_positions,
                key_len=key_len,
                full_prefill=True,
            )
            local_selected = [
                (row, row - start) for row in chunk_rows if row in selected_set
            ]
            for row, local_idx in local_selected:
                selected[int(row)] = attn[:, :, local_idx : local_idx + 1, :].detach()

        if set(selected) != selected_set:
            missing = sorted(selected_set.difference(selected))
            raise RuntimeError(f"failed to retain sampled prefill rows: {missing}")
        sampled = torch.cat([selected[int(idx)] for idx in row_indices], dim=2)
        return sampled[0].reshape(sampled.shape[1] * len(row_indices), key_len)

    def _attention_tensor(
        self,
        *,
        module: Any,
        hidden_states: Any,
        position_embeddings: tuple[Any, Any],
        attention_mask: Any,
        key_states: Any,
        row_indices: list[int],
        query_positions: list[int],
    ) -> Any:
        import torch

        q_states = _project_query_states(module, hidden_states[:, row_indices, :])
        q_states = _apply_rotary_to_states(
            q_states,
            _select_rotary_positions(position_embeddings, row_indices),
        )
        key_len = int(key_states.shape[-2])
        n_kv_heads = int(key_states.shape[1])
        n_query_heads = int(q_states.shape[1])
        if n_query_heads % n_kv_heads != 0:
            raise ValueError(
                f"query heads must be divisible by kv heads: {n_query_heads} vs {n_kv_heads}"
            )
        groups = n_query_heads // n_kv_heads
        grouped_q = q_states.reshape(
            q_states.shape[0],
            n_kv_heads,
            groups,
            q_states.shape[-2],
            q_states.shape[-1],
        )
        scores = torch.matmul(grouped_q, key_states.unsqueeze(2).transpose(-1, -2))
        scores = scores.reshape(
            q_states.shape[0],
            n_query_heads,
            q_states.shape[-2],
            key_len,
        )
        scores = scores * float(getattr(module, "scaling", q_states.shape[-1] ** -0.5))
        scores = self._apply_attention_mask(
            scores=scores,
            attention_mask=attention_mask,
            row_indices=row_indices,
            query_positions=query_positions,
            query_len=int(hidden_states.shape[-2]),
            key_len=key_len,
        )
        attn = torch.softmax(scores, dim=-1)
        return attn

    def _publish_attention(
        self,
        *,
        layer: int,
        attn: Any,
        query_positions: list[int],
        key_len: int,
        full_prefill: bool,
    ) -> None:
        import torch

        # Publish post-softmax, pre-reshape: shape is the documented
        # (B, num_q_heads, n_query_rows, key_len) the bus contract requires.
        # H2O (step 6) consumes this exact tensor; the capturer's own reduce
        # below stays unchanged so attention.npz bytes don't move when no
        # subscriber is attached.
        if self._attention_bus is not None:
            phase = (
                "decode"
                if (
                    len(query_positions) == 1
                    and self._session is not None
                    and key_len > int(self._session["input_token_count"])
                )
                else "prefill"
            )
            query_pos_tensor = torch.as_tensor(
                query_positions, dtype=torch.long, device=attn.device
            )
            self._attention_bus.publish(
                layer=layer,
                attn=attn,
                query_positions=query_pos_tensor,
                key_len=key_len,
                phase=phase,
                suspended=self._attention_suspended > 0,
                full_prefill=full_prefill,
            )

    def _apply_attention_mask(
        self,
        *,
        scores: Any,
        attention_mask: Any,
        row_indices: list[int],
        query_positions: list[int],
        query_len: int,
        key_len: int,
    ) -> Any:
        import torch

        if attention_mask is not None:
            mask = attention_mask[:, :, :, :key_len].to(
                device=scores.device, dtype=scores.dtype
            )
            if int(mask.shape[-2]) == query_len:
                index = torch.as_tensor(row_indices, dtype=torch.long, device=mask.device)
                mask = mask.index_select(-2, index)
            elif int(mask.shape[-2]) != len(row_indices):
                raise ValueError(
                    f"unsupported attention mask query dimension: {mask.shape}"
                )
            return scores + mask

        key_positions = torch.arange(key_len, device=scores.device)
        query_pos = torch.as_tensor(
            query_positions, dtype=torch.long, device=scores.device
        )
        blocked = key_positions.view(1, 1, 1, key_len) > query_pos.view(
            1, 1, len(query_positions), 1
        )
        return scores.masked_fill(blocked, torch.finfo(scores.dtype).min)

    def record_router_logits(self, outputs: Any, *, total_tokens: int) -> None:
        router_logits = getattr(outputs, "router_logits", None)
        if router_logits is None and isinstance(outputs, dict):
            router_logits = outputs.get("router_logits")
        if router_logits is None or self._session is None:
            return

        n_segments = self._session["generated_segment_id"] + 1
        top_k_experts = int(
            self.model_summary.get("num_experts_per_tok")
            or self.model_summary.get("num_experts_per_token")
            or 1
        )
        for path, tensor in self._iter_tensors(router_logits):
            if tensor.ndim < 2:
                continue
            logits = tensor.reshape(-1, tensor.shape[-1])
            token_ids = self._routing_token_ids(
                n_tokens=int(logits.shape[0]),
                total_tokens=total_tokens,
                nested_path=path,
            ).to(device=logits.device)
            choices, weights, load = expert_load_per_segment(
                logits,
                token_ids,
                n_segments=n_segments,
                top_k_experts=top_k_experts,
            )
            self._routing_records.append(
                {
                    "path": ".".join(str(part) for part in path),
                    "layer": int(path[-1]) if path else -1,
                    "expert_choice": _as_numpy(choices).astype(np.int32),
                    "expert_weight": _as_numpy(weights).astype(np.float32),
                    "expert_load": _as_numpy(load).astype(np.float32),
                }
            )

    def _record_router_tensor(
        self,
        *,
        layer: int,
        path: tuple[str, ...],
        tensor: Any,
    ) -> None:
        if self._session is None or tensor is None or tensor.ndim < 2:
            return
        n_segments = self._session["generated_segment_id"] + 1
        top_k_experts = int(
            self.model_summary.get("num_experts_per_tok")
            or self.model_summary.get("num_experts_per_token")
            or 1
        )
        logits = tensor.reshape(-1, tensor.shape[-1])
        token_ids = self._gate_token_ids(n_tokens=int(logits.shape[0])).to(
            device=logits.device
        )
        choices, weights, load = expert_load_per_segment(
            logits,
            token_ids,
            n_segments=n_segments,
            top_k_experts=top_k_experts,
        )
        self._routing_records.append(
            {
                "path": ".".join(path),
                "layer": layer,
                "expert_choice": _as_numpy(choices).astype(np.int32),
                "expert_weight": _as_numpy(weights).astype(np.float32),
                "expert_load": _as_numpy(load).astype(np.float32),
            }
        )

    def _gate_token_ids(self, *, n_tokens: int):
        assert self._session is not None
        input_tokens = int(self._session["input_token_count"])
        if n_tokens == input_tokens:
            return token_segment_ids(
                input_tokens,
                self._session["segments"],
                generated_segment_id=None,
            )

        import torch

        return torch.full(
            (n_tokens,),
            self._session["generated_segment_id"],
            dtype=torch.long,
        )

    def _routing_token_ids(self, *, n_tokens: int, total_tokens: int, nested_path: tuple[int, ...]):
        assert self._session is not None
        if n_tokens == total_tokens:
            return token_segment_ids(
                total_tokens,
                self._session["segments"],
                generated_segment_id=self._session["generated_segment_id"],
            )
        if len(nested_path) >= 2 and n_tokens <= 1:
            import torch

            return torch.full(
                (n_tokens,), self._session["generated_segment_id"], dtype=torch.long
            )
        return self._token_ids_for_key_len(n_tokens)

    def _iter_tensors(self, value: Any, path: tuple[int, ...] = ()):
        import torch

        if torch.is_tensor(value):
            yield path, value
            return
        if isinstance(value, (list, tuple)):
            for idx, child in enumerate(value):
                yield from self._iter_tensors(child, (*path, idx))

    def flush(self, *, output_token_ids: list[int]) -> None:
        if self._session is None:
            raise RuntimeError("no active recording session")
        if not self._prefill_records and len(self._decode_records) == 0:
            raise RuntimeError("no attention records captured")
        input_tokens = int(self._session["input_token_count"])
        total_tokens = input_tokens + len(output_token_ids)
        segments = [dict(segment) for segment in self._session["segments"]]
        segments.append(
            {
                "role": "generation",
                "message_index": None,
                "token_start": input_tokens,
                "token_end": total_tokens,
                "has_content": bool(output_token_ids),
                "has_tool_calls": False,
                "first_seen_call": int(self._session["call_idx"]),
                "first_seen_call_inferred": False,
            }
        )
        token_ids = token_segment_ids(
            total_tokens,
            segments,
            generated_segment_id=self._session["generated_segment_id"],
        ).tolist()

        iter_dir: Path = self._session["iter_dir"]
        (iter_dir / "segments.json").write_text(
            json.dumps(
                {
                    "call_idx": self._session["call_idx"],
                    "input_tokens": input_tokens,
                    "output_tokens": len(output_token_ids),
                    "total_tokens": total_tokens,
                    "segments": segments,
                    "token_segment_id": token_ids,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self._write_attention_npz(iter_dir / "attention.npz", len(segments))
        self._write_routing_npz(iter_dir / "routing.npz", len(segments))
        if self._kv_recorder is not None and self._kv_recorder.n_records() > 0:
            # KV eviction npz lives alongside attention.npz / routing.npz so
            # downstream loaders can join on (call_idx, layer, decode_step).
            self._kv_recorder.write(iter_dir / "kv_eviction.npz")
        self._meta["iters"].append(
            {
                "call_idx": self._session["call_idx"],
                "dir": iter_dir.name,
                "input_tokens": input_tokens,
                "output_tokens": len(output_token_ids),
                "total_tokens": total_tokens,
                "n_segments": len(segments),
                "attention_records": len(self._prefill_records)
                + len(self._decode_records),
                "routing_records": len(self._routing_records),
                "elapsed_s": time.time() - float(self._session["started_at"]),
            }
        )
        self._session["flushed"] = True

    def _write_attention_npz(self, path: Path, n_segments: int) -> None:
        records = [*self._prefill_records, *self._decode_records.to_list()]
        _synchronize_pending_arrays(records)
        offsets = [0]
        for record in records:
            offsets.append(offsets[-1] + int(record["segment_mass"].shape[0]))
        n_rows = offsets[-1]
        k = self.config.attention_top_k
        np.savez_compressed(
            path,
            call_idx=np.asarray(self._session["call_idx"], dtype=np.int32),
            n_segments=np.asarray(n_segments, dtype=np.int32),
            top_k=np.asarray(k, dtype=np.int32),
            record_layer=np.asarray([r["layer"] for r in records], dtype=np.int32),
            record_phase=np.asarray([r["phase"] for r in records]),
            record_decode_step=np.asarray(
                [r["decode_step"] for r in records], dtype=np.int32
            ),
            query_row_offsets=np.asarray(offsets, dtype=np.int64),
            query_heads=np.concatenate([r["query_heads"] for r in records])
            if records
            else np.zeros((0,), dtype=np.int32),
            query_positions=np.concatenate(
                [r["query_positions"] for r in records]
            )
            if records
            else np.zeros((0,), dtype=np.int32),
            segment_mass=np.concatenate(
                [_materialize_array(r["segment_mass"]) for r in records],
                axis=0,
            )
            if records
            else np.zeros((0, n_segments), dtype=np.float32),
            topk_indices=np.concatenate(
                [_materialize_array(r["topk_indices"]) for r in records],
                axis=0,
            )
            if records
            else np.zeros((0, k), dtype=np.int32),
            topk_weights=np.concatenate(
                [_materialize_array(r["topk_weights"]) for r in records],
                axis=0,
            )
            if records
            else np.zeros((0, k), dtype=np.float32),
            heavy_indices=np.stack(
                [_materialize_array(r["heavy_indices"]) for r in records],
                axis=0,
            )
            if records
            else np.zeros((0, k), dtype=np.int32),
            heavy_weights=np.stack(
                [_materialize_array(r["heavy_weights"]) for r in records],
                axis=0,
            )
            if records
            else np.zeros((0, k), dtype=np.float32),
            n_query_rows=np.asarray(n_rows, dtype=np.int64),
        )

    def _write_routing_npz(self, path: Path, n_segments: int) -> None:
        records = self._routing_records
        token_offsets = [0]
        n_experts = 0
        top_k = 0
        for record in records:
            choices = record["expert_choice"]
            token_offsets.append(token_offsets[-1] + int(choices.shape[0]))
            n_experts = int(record["expert_load"].shape[1])
            top_k = int(choices.shape[1])
        np.savez_compressed(
            path,
            call_idx=np.asarray(self._session["call_idx"], dtype=np.int32),
            n_segments=np.asarray(n_segments, dtype=np.int32),
            n_experts=np.asarray(n_experts, dtype=np.int32),
            top_k_experts=np.asarray(top_k, dtype=np.int32),
            record_layer=np.asarray([r["layer"] for r in records], dtype=np.int32),
            record_path=np.asarray([r["path"] for r in records]),
            token_row_offsets=np.asarray(token_offsets, dtype=np.int64),
            expert_choice=np.concatenate(
                [r["expert_choice"] for r in records], axis=0
            )
            if records
            else np.zeros((0, 0), dtype=np.int32),
            expert_weight=np.concatenate(
                [r["expert_weight"] for r in records], axis=0
            )
            if records
            else np.zeros((0, 0), dtype=np.float32),
            expert_load=np.stack([r["expert_load"] for r in records], axis=0)
            if records
            else np.zeros((0, n_segments, 0), dtype=np.float32),
        )
