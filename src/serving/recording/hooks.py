"""Forward hooks and artifact writer for HF internal recordings."""

from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

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


_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn$")


def _layer_index(module_name: str) -> int | None:
    match = _LAYER_RE.search(module_name)
    if match:
        return int(match.group(1))
    if module_name.endswith(".self_attn") or module_name == "self_attn":
        return -1
    return None


def _as_numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().cpu().numpy()


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


class LayerCapturer:
    """Capture reduced attention and routing tensors for one HF model."""

    def __init__(
        self,
        model: Any,
        *,
        config: RecordingConfig,
        model_summary: dict[str, Any],
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

        n_attention_modules = 0
        for name, module in model.named_modules():
            layer = _layer_index(name)
            if layer is None:
                continue
            n_attention_modules += 1
            self._handles.append(
                module.register_forward_hook(self._hook(layer), with_kwargs=True)
            )
        if n_attention_modules == 0:
            raise ValueError("no attention modules matched '.layers.<n>.self_attn'")
        self._decode_records = DecodeRingBuffer(
            config.decode_window * max(1, n_attention_modules)
        )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def start_attempt(self, recordings_dir: Path) -> None:
        self._attempt_dir = Path(recordings_dir)
        self._attempt_dir.mkdir(parents=True, exist_ok=True)
        self._meta = {
            "model": self.model_summary,
            "recording_config": asdict(self.config),
            "iters": [],
        }

    def finish_attempt(self) -> None:
        if self._attempt_dir is None:
            return
        (self._attempt_dir / "meta.json").write_text(
            json.dumps(self._meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._attempt_dir = None

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
        self._session = {
            "call_idx": call_idx,
            "iter_dir": iter_dir,
            "segments": [dict(segment) for segment in segments],
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
        row_indices = (
            [0]
            if query_len == 1
            else select_query_positions(query_len, self.config.max_prefill_queries)
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
        attn_rows = self._sampled_attention_rows(
            module=module,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            key_states=key_states,
            row_indices=row_indices,
            query_positions=query_positions,
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
            "segment_mass": _as_numpy(segment_mass).astype(np.float32),
            "topk_indices": _as_numpy(top_indices).astype(np.int32),
            "topk_weights": _as_numpy(top_weights).astype(np.float32),
            "heavy_indices": _as_numpy(hitter_indices[0]).astype(np.int32),
            "heavy_weights": _as_numpy(hitter_weights[0]).astype(np.float32),
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
        return attn[0].reshape(n_query_heads * len(row_indices), key_len)

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
            segment_mass=np.concatenate([r["segment_mass"] for r in records], axis=0)
            if records
            else np.zeros((0, n_segments), dtype=np.float32),
            topk_indices=np.concatenate([r["topk_indices"] for r in records], axis=0)
            if records
            else np.zeros((0, k), dtype=np.int32),
            topk_weights=np.concatenate([r["topk_weights"] for r in records], axis=0)
            if records
            else np.zeros((0, k), dtype=np.float32),
            heavy_indices=np.stack([r["heavy_indices"] for r in records], axis=0)
            if records
            else np.zeros((0, k), dtype=np.int32),
            heavy_weights=np.stack([r["heavy_weights"] for r in records], axis=0)
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
