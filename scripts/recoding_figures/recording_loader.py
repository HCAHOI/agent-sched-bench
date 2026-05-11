"""Load compact distributions from HF recording artifacts.

The loader deliberately keeps only per-iteration, per-layer aggregate
distributions. It does not retain raw query rows or token-level routing arrays,
so it can be used on multi-GB recording runs without materializing the whole run
in memory.

The KV eviction audit (`kv_eviction.npz`) is loaded raw rather than aggregated
since each row is already a small per-(layer, decode_step) record; see
`load_kv_eviction()`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


ROLE_ORDER = [
    "system",
    "user",
    "assistant_message",
    "assistant_call",
    "tool_result",
    "gen_prompt",
    "generation",
    "meta",
    "other",
]


@dataclass(frozen=True)
class IterationRecord:
    """One recorded LLM call with paths and metadata."""

    task: str
    attempt_dir: Path
    recordings_dir: Path
    iter_dir: Path
    call_idx: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    trace_iteration: int | None = None
    is_orphan: bool = False

    @property
    def label(self) -> str:
        """Human-readable short label."""
        return f"{self.task}:c{self.call_idx}"


@dataclass
class LayerDistributionSet:
    """Per-layer distributions for a group of recorded iterations."""

    modality: str
    records: list[IterationRecord]
    layers: list[int]
    axis_labels: list[str]
    distributions: dict[int, np.ndarray] = field(default_factory=dict)
    observation_counts: dict[int, np.ndarray] = field(default_factory=dict)


# Schema fields exposed by `load_kv_eviction`; mirrors `KVEvictionRecorder.write`
# in `src/serving/kv_policies/recorder.py`. Update both together if the writer
# changes.
KV_EVICTION_COLUMNS: tuple[str, ...] = (
    "task",
    "call_idx",
    "iter_dir",
    "policy_name",
    "record_step",
    "record_layer",
    "record_phase",
    "pre_len",
    "post_len",
    "budget",
    "evict_reason",
)


@dataclass
class KVEvictionFrame:
    """Per-row KV eviction audit, flattened across iterations.

    Each row corresponds to one `BaseEvictionCache.update()` call (one
    `(call_idx, layer, decode_step)` tuple). Scalar columns are 1-D numpy
    arrays of length R; CSR data is exposed both as raw flat arrays plus
    offsets and as per-row decoded lists for ergonomic consumption.
    """

    n_rows: int
    # Provenance columns (added by the loader, not in npz):
    task: np.ndarray  # (R,) U
    call_idx: np.ndarray  # (R,) int32
    iter_dir: np.ndarray  # (R,) U  (str(iter_dir) per row)
    # Native npz columns:
    policy_name: np.ndarray  # (R,) U16
    record_step: np.ndarray  # (R,) int32
    record_layer: np.ndarray  # (R,) int32
    record_phase: np.ndarray  # (R,) U7
    pre_len: np.ndarray  # (R,) int32
    post_len: np.ndarray  # (R,) int32
    budget: np.ndarray  # (R,) int32
    evict_reason: np.ndarray  # (R,) U16
    # CSR raw form preserved verbatim for callers who want offsets:
    kept_offsets: np.ndarray  # (R+1,) int64
    kept_indices: np.ndarray  # (sum_kept,) int32
    evicted_offsets: np.ndarray  # (R+1,) int64
    evicted_indices: np.ndarray  # (sum_evicted,) int32
    # Per-row decoded form (one np.ndarray per row), convenient for
    # `for kept, evicted in zip(frame.kept_per_row, frame.evicted_per_row)`:
    kept_per_row: list[np.ndarray] = field(default_factory=list)
    evicted_per_row: list[np.ndarray] = field(default_factory=list)
    # h2o-only diagnostic columns; sentinel-filled (-1 / NaN) for other
    # policies. Width = max heavy-slot count across rows; may be 0.
    score_topk_index: np.ndarray = field(  # (R, k) int32
        default_factory=lambda: np.empty((0, 0), dtype=np.int32)
    )
    score_topk_value: np.ndarray = field(  # (R, k) float32
        default_factory=lambda: np.empty((0, 0), dtype=np.float32)
    )

    @property
    def is_empty(self) -> bool:
        return self.n_rows == 0


def find_attempt_dirs(paths: Sequence[Path]) -> list[Path]:
    """Resolve attempt directories from attempt, task, or run paths."""
    found: list[Path] = []
    for path in paths:
        candidate = path.expanduser().resolve()
        local_found: list[Path] = []
        if (candidate / "recordings").is_dir():
            local_found.append(candidate)
            found.extend(local_found)
            continue
        for meta_path in candidate.rglob("recordings/meta.json"):
            local_found.append(meta_path.parent.parent)
        if not local_found and candidate.is_dir():
            for recordings_dir in candidate.rglob("recordings"):
                if recordings_dir.is_dir():
                    local_found.append(recordings_dir.parent)
        found.extend(local_found)
    unique: list[Path] = []
    seen: set[Path] = set()
    for attempt_dir in found:
        if attempt_dir not in seen:
            unique.append(attempt_dir)
            seen.add(attempt_dir)
    return sorted(unique)


def load_iteration_records(
    paths: Sequence[Path],
    *,
    include_orphans: bool = False,
    max_iters: int | None = None,
) -> list[IterationRecord]:
    """Load iteration records from one or more attempt/run directories."""
    attempt_dirs = find_attempt_dirs(paths)
    if not attempt_dirs:
        raise FileNotFoundError("no attempt directories with recordings were found")

    records: list[IterationRecord] = []
    for attempt_dir in attempt_dirs:
        task = _task_name(attempt_dir)
        recordings_dir = attempt_dir / "recordings"
        meta_path = recordings_dir / "meta.json"
        meta_items: list[tuple[dict, bool]] = []
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta_items.extend((dict(item), False) for item in meta.get("iters", []))
            if include_orphans:
                meta_items.extend(
                    (dict(item), True) for item in meta.get("orphan_iters", [])
                )
        if not meta_items:
            meta_items = [
                ({"dir": path.name, "call_idx": _call_idx_from_iter_dir(path)}, False)
                for path in sorted(recordings_dir.glob("iter_*"))
            ]

        for item, is_orphan in meta_items:
            iter_dir = recordings_dir / str(item.get("dir", ""))
            if not _has_recording_files(iter_dir):
                continue
            records.append(
                IterationRecord(
                    task=task,
                    attempt_dir=attempt_dir,
                    recordings_dir=recordings_dir,
                    iter_dir=iter_dir,
                    call_idx=int(item.get("call_idx", _call_idx_from_iter_dir(iter_dir))),
                    input_tokens=_optional_int(item.get("input_tokens")),
                    output_tokens=_optional_int(item.get("output_tokens")),
                    total_tokens=_optional_int(item.get("total_tokens")),
                    trace_iteration=_optional_int(item.get("trace_iteration")),
                    is_orphan=is_orphan,
                )
            )

    records = sorted(records, key=lambda item: (item.task, item.call_idx, str(item.iter_dir)))
    if max_iters is not None:
        if max_iters <= 0:
            raise ValueError("--max-iters must be positive")
        records = records[:max_iters]
    if not records:
        raise FileNotFoundError("recording directories exist, but no complete iterations were found")
    return records


def collect_role_labels(records: Iterable[IterationRecord]) -> list[str]:
    """Collect normalized segment roles in stable display order."""
    observed: set[str] = set()
    for record in records:
        for role in _segment_roles(record):
            observed.add(role)
    ordered = [role for role in ROLE_ORDER if role in observed]
    ordered.extend(sorted(observed.difference(ordered)))
    return ordered


def segment_role_indices_for_record(
    record: IterationRecord,
    role_labels: Sequence[str],
) -> list[int]:
    """Map each saved segment in a record to a role-label column."""
    role_index = {role: idx for idx, role in enumerate(role_labels)}
    return _segment_role_indices(_segment_roles(record), role_index)


def role_token_counts_for_key_len(
    segments: Sequence[dict],
    role_labels: Sequence[str],
    key_len: int,
) -> np.ndarray:
    """Count visible key tokens by role for a causal query key length."""
    role_index = {role: idx for idx, role in enumerate(role_labels)}
    return _role_token_counts_for_key_len(segments, role_index, key_len)


def load_token_role_distributions(
    records: Sequence[IterationRecord],
    *,
    role_labels: Sequence[str] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Load final per-iteration token-share distributions over normalized roles."""
    labels = list(role_labels or collect_role_labels(records))
    role_index = {role: idx for idx, role in enumerate(labels)}
    matrix = np.zeros((len(records), len(labels)), dtype=np.float64)

    for row_idx, record in enumerate(records):
        payload = json.loads((record.iter_dir / "segments.json").read_text(encoding="utf-8"))
        total_tokens = 0
        for segment in payload.get("segments", []):
            start = int(segment.get("token_start", 0) or 0)
            end = int(segment.get("token_end", start) or start)
            length = max(0, end - start)
            role = _normalize_role(segment)
            col = role_index.get(role, role_index.get("other"))
            if col is None:
                raise ValueError(f"role {role!r} is missing from role labels")
            matrix[row_idx, col] += float(length)
            total_tokens += length
        if total_tokens > 0:
            matrix[row_idx] /= float(total_tokens)

    return labels, matrix


def load_attention_key_role_distributions(
    records: Sequence[IterationRecord],
    *,
    role_labels: Sequence[str] | None = None,
    phase: str = "all",
) -> LayerDistributionSet:
    """Load token-role baselines using only keys visible to attention records."""
    labels = list(role_labels or collect_role_labels(records))
    role_index = {role: idx for idx, role in enumerate(labels)}
    per_layer: dict[int, list[np.ndarray | None]] = {}
    per_layer_counts: dict[int, list[float]] = {}

    for record_index, record in enumerate(records):
        payload = json.loads((record.iter_dir / "segments.json").read_text(encoding="utf-8"))
        segments = list(payload.get("segments", []))
        with np.load(record.iter_dir / "attention.npz") as attention:
            record_layers = attention["record_layer"].astype(np.int64)
            record_phases = attention["record_phase"].astype(str)
            offsets = attention["query_row_offsets"].astype(np.int64)
            query_positions = attention["query_positions"].astype(np.int64)

            layer_sums: dict[int, np.ndarray] = {}
            layer_counts: dict[int, int] = {}
            for idx, layer in enumerate(record_layers):
                if phase != "all" and str(record_phases[idx]) != phase:
                    continue
                start = int(offsets[idx])
                end = int(offsets[idx + 1])
                if end <= start:
                    continue
                positions = query_positions[start:end]
                if positions.size == 0:
                    continue
                counts = np.zeros(len(labels), dtype=np.float64)
                unique_positions, position_counts = np.unique(positions, return_counts=True)
                for position, count in zip(unique_positions, position_counts, strict=True):
                    key_len = int(position) + 1
                    if key_len <= 0:
                        continue
                    counts += _role_token_counts_for_key_len(
                        segments,
                        role_index,
                        key_len,
                    ) * float(count)
                row_count = int(end - start)
                layer_int = int(layer)
                layer_sums.setdefault(layer_int, np.zeros(len(labels), dtype=np.float64))
                layer_sums[layer_int] += counts
                layer_counts[layer_int] = layer_counts.get(layer_int, 0) + row_count

        for layer, values in layer_sums.items():
            _ensure_record_slots(per_layer, layer, record_index)
            _ensure_count_slots(per_layer_counts, layer, record_index)
            per_layer[layer][record_index] = _normalize(values)
            per_layer_counts[layer][record_index] = float(layer_counts[layer])

    distributions = _finalize_distribution_slots(per_layer, len(records), len(labels))
    observation_counts = _finalize_count_slots(per_layer_counts, len(records))
    return LayerDistributionSet(
        modality="attention_key_role",
        records=list(records),
        layers=sorted(distributions),
        axis_labels=labels,
        distributions=distributions,
        observation_counts=observation_counts,
    )


def load_attention_distributions(
    records: Sequence[IterationRecord],
    *,
    role_labels: Sequence[str] | None = None,
    phase: str = "all",
) -> LayerDistributionSet:
    """Load layer x role attention distributions for each iteration."""
    labels = list(role_labels or collect_role_labels(records))
    role_index = {role: idx for idx, role in enumerate(labels)}
    per_layer: dict[int, list[np.ndarray | None]] = {}
    per_layer_counts: dict[int, list[float]] = {}

    for record_index, record in enumerate(records):
        segment_roles = _segment_roles(record)
        segment_role_indices = _segment_role_indices(segment_roles, role_index)
        with np.load(record.iter_dir / "attention.npz") as attention:
            record_layers = attention["record_layer"].astype(np.int64)
            record_phases = attention["record_phase"].astype(str)
            offsets = attention["query_row_offsets"].astype(np.int64)
            segment_mass = attention["segment_mass"].astype(np.float64)

            layer_sums: dict[int, np.ndarray] = {}
            layer_counts: dict[int, int] = {}
            for idx, layer in enumerate(record_layers):
                if phase != "all" and str(record_phases[idx]) != phase:
                    continue
                start = int(offsets[idx])
                end = int(offsets[idx + 1])
                if end <= start:
                    continue
                rows = segment_mass[start:end]
                if np.any(~np.isfinite(rows)):
                    raise ValueError(f"{record.iter_dir}: segment_mass contains non-finite values")
                if rows.shape[1] != len(segment_role_indices):
                    raise ValueError(
                        f"{record.iter_dir}: segment count mismatch "
                        f"{rows.shape[1]} vs {len(segment_role_indices)}"
                    )
                layer_int = int(layer)
                layer_sums.setdefault(layer_int, np.zeros(len(labels), dtype=np.float64))
                segment_totals = rows.sum(axis=0)
                for segment_idx, role_col in enumerate(segment_role_indices):
                    layer_sums[layer_int][role_col] += float(segment_totals[segment_idx])
                layer_counts[layer_int] = layer_counts.get(layer_int, 0) + int(rows.shape[0])

        for layer, values in layer_sums.items():
            _ensure_record_slots(per_layer, layer, record_index)
            _ensure_count_slots(per_layer_counts, layer, record_index)
            count = float(layer_counts[layer])
            per_layer[layer][record_index] = _normalize(values)
            per_layer_counts[layer][record_index] = count

    distributions = _finalize_distribution_slots(per_layer, len(records), len(labels))
    observation_counts = _finalize_count_slots(per_layer_counts, len(records))
    return LayerDistributionSet(
        modality="attention",
        records=list(records),
        layers=sorted(distributions),
        axis_labels=labels,
        distributions=distributions,
        observation_counts=observation_counts,
    )


def load_moe_distributions(
    records: Sequence[IterationRecord],
    *,
    phase: str = "all",
) -> LayerDistributionSet:
    """Load layer x expert-load distributions for each iteration."""
    n_experts = _infer_n_experts(records)
    labels = [str(idx) for idx in range(n_experts)]
    per_layer: dict[int, list[np.ndarray | None]] = {}
    per_layer_counts: dict[int, list[float]] = {}

    for record_index, record in enumerate(records):
        with np.load(record.iter_dir / "routing.npz") as routing:
            record_layers = routing["record_layer"].astype(np.int64)
            expert_load = routing["expert_load"].astype(np.float64)
            if expert_load.ndim != 3:
                raise ValueError(
                    f"{record.iter_dir}: expected expert_load rank 3, got {expert_load.shape}"
                )
            record_phases = (
                derive_moe_record_phases(record, routing, expert_load=expert_load)
                if phase != "all"
                else None
            )
            if expert_load.shape[2] > n_experts:
                raise ValueError(
                    f"{record.iter_dir}: expert dimension {expert_load.shape[2]} > {n_experts}"
                )

            layer_sums: dict[int, np.ndarray] = {}
            layer_counts: dict[int, float] = {}
            for idx, layer in enumerate(record_layers):
                if record_phases is not None and str(record_phases[idx]) != phase:
                    continue
                load = expert_load[idx].sum(axis=0)
                values = np.zeros(n_experts, dtype=np.float64)
                values[: load.shape[0]] = load
                layer_int = int(layer)
                layer_sums.setdefault(layer_int, np.zeros(n_experts, dtype=np.float64))
                layer_sums[layer_int] += values
                layer_counts[layer_int] = layer_counts.get(layer_int, 0.0) + float(values.sum())

        for layer, values in layer_sums.items():
            _ensure_record_slots(per_layer, layer, record_index)
            _ensure_count_slots(per_layer_counts, layer, record_index)
            per_layer[layer][record_index] = _normalize(values)
            per_layer_counts[layer][record_index] = layer_counts[layer]

    distributions = _finalize_distribution_slots(per_layer, len(records), n_experts)
    observation_counts = _finalize_count_slots(per_layer_counts, len(records))
    return LayerDistributionSet(
        modality="moe" if phase == "all" else f"moe_{phase}",
        records=list(records),
        layers=sorted(distributions),
        axis_labels=labels,
        distributions=distributions,
        observation_counts=observation_counts,
    )


def derive_moe_record_phases(
    record: IterationRecord,
    routing: object,
    *,
    expert_load: np.ndarray | None = None,
) -> np.ndarray:
    """Derive per-routing-record prefill/decode labels from saved token spans.

    Prefer an explicit `record_phase` field when present. Older recordings do not
    have that field, but they do preserve token row offsets and per-segment expert
    load. In those artifacts, records covering input tokens with no generation
    load are prefill; records whose load is entirely on the generated segment are
    decode. Any mixed record is left as `mixed` rather than forced into a phase.
    """
    if "record_phase" in routing.files:
        phases = routing["record_phase"].astype(str)
        if expert_load is not None and int(phases.shape[0]) != int(expert_load.shape[0]):
            raise ValueError(
                f"{record.iter_dir}: record_phase length {phases.shape[0]} "
                f"does not match expert_load records {expert_load.shape[0]}"
            )
        if expert_load is None and "expert_load" in routing.files:
            n_records = int(routing["expert_load"].shape[0])
            if int(phases.shape[0]) != n_records:
                raise ValueError(
                    f"{record.iter_dir}: record_phase length {phases.shape[0]} "
                    f"does not match expert_load records {n_records}"
                )
        return phases
    if "token_row_offsets" not in routing.files:
        raise ValueError(f"{record.iter_dir}: routing.npz lacks token_row_offsets")

    load = (
        np.asarray(expert_load, dtype=np.float64)
        if expert_load is not None
        else routing["expert_load"].astype(np.float64)
    )
    if load.ndim != 3:
        raise ValueError(f"{record.iter_dir}: expected expert_load rank 3, got {load.shape}")
    offsets = routing["token_row_offsets"].astype(np.int64)
    if int(offsets.shape[0]) != int(load.shape[0]) + 1:
        raise ValueError(
            f"{record.iter_dir}: token_row_offsets length {offsets.shape[0]} "
            f"does not match expert_load records {load.shape[0]}"
        )
    payload = json.loads((record.iter_dir / "segments.json").read_text(encoding="utf-8"))
    input_tokens = _optional_int(payload.get("input_tokens")) or record.input_tokens
    if input_tokens is None:
        raise ValueError(f"{record.iter_dir}: cannot derive MoE phase without input_tokens")
    segments = list(payload.get("segments", []))
    generation_idx = _generation_segment_index(segments)
    if generation_idx is None or generation_idx >= int(load.shape[1]):
        raise ValueError(f"{record.iter_dir}: cannot identify generation segment")

    eps = 1e-8
    token_rows = np.diff(offsets)
    segment_load = load.sum(axis=2)
    total = segment_load.sum(axis=1)
    generation_mass = segment_load[:, generation_idx]
    non_generation_mass = total - generation_mass

    phases = np.full(int(load.shape[0]), "mixed", dtype="<U7")
    unknown = total <= 0
    prefill = (
        (token_rows == int(input_tokens))
        & ~unknown
        & (generation_mass <= eps * total)
    )
    decode = ~unknown & (non_generation_mass <= eps * total)
    phases[unknown] = "unknown"
    phases[prefill] = "prefill"
    phases[decode] = "decode"
    return phases


def count_moe_record_phases(records: Sequence[IterationRecord]) -> dict[str, object]:
    """Count MoE routing records by explicit or derived phase labels."""
    counts: dict[str, int] = {
        "prefill": 0,
        "decode": 0,
        "mixed": 0,
        "unknown": 0,
    }
    failures: list[dict[str, object]] = []
    n_iteration_records_with_phase = 0
    n_routing_records_with_phase = 0

    for record in records:
        try:
            with np.load(record.iter_dir / "routing.npz") as routing:
                phases = derive_moe_record_phases(record, routing)
        except (KeyError, OSError, ValueError) as exc:
            failures.append(
                {
                    "task": record.task,
                    "call_idx": record.call_idx,
                    "iter_dir": str(record.iter_dir),
                    "error": str(exc),
                }
            )
            continue
        n_iteration_records_with_phase += 1
        n_routing_records_with_phase += int(phases.shape[0])
        for phase in phases.astype(str):
            counts[phase] = counts.get(phase, 0) + 1

    return {
        "counts": counts,
        "n_iteration_records": len(records),
        "n_iteration_records_with_phase": n_iteration_records_with_phase,
        "n_iteration_records_failed": len(failures),
        "n_routing_records_with_phase": n_routing_records_with_phase,
        "failure_examples": failures[:8],
    }


def load_kv_eviction(records: Sequence[IterationRecord]) -> KVEvictionFrame:
    """Load per-row KV eviction audit frames across all iter dirs.

    Each `kv_eviction.npz` is appended along its R axis (one row per
    `(layer, decode_step)` decision). Iterations missing the npz are skipped
    silently — `kv_eviction.npz` is only emitted when an eviction policy is
    configured (`--kv-policy {streaming,h2o,random}`); old recordings and
    `--kv-policy none` runs predate the artifact entirely.

    Returns an empty `KVEvictionFrame` (n_rows=0, all columns 0-length) when
    no records carry the npz. Raises only on malformed npz, never on missing.
    """
    columns: dict[str, list[np.ndarray]] = {
        "task": [],
        "call_idx": [],
        "iter_dir": [],
        "policy_name": [],
        "record_step": [],
        "record_layer": [],
        "record_phase": [],
        "pre_len": [],
        "post_len": [],
        "budget": [],
        "evict_reason": [],
        "kept_indices_per_call": [],
        "kept_offsets_per_call": [],
        "evicted_indices_per_call": [],
        "evicted_offsets_per_call": [],
        "score_index_per_call": [],
        "score_value_per_call": [],
    }

    for record in records:
        npz_path = record.iter_dir / "kv_eviction.npz"
        if not npz_path.is_file():
            continue
        with np.load(npz_path) as data:
            n = int(data["record_step"].shape[0])
            if n == 0:
                continue
            columns["task"].append(np.full(n, record.task, dtype=object))
            columns["call_idx"].append(
                np.full(n, int(record.call_idx), dtype=np.int32)
            )
            columns["iter_dir"].append(np.full(n, str(record.iter_dir), dtype=object))
            columns["policy_name"].append(
                np.full(n, str(data["policy_name"]), dtype="U16")
            )
            columns["record_step"].append(data["record_step"].astype(np.int32))
            columns["record_layer"].append(data["record_layer"].astype(np.int32))
            columns["record_phase"].append(data["record_phase"].astype("U7"))
            columns["pre_len"].append(data["pre_len"].astype(np.int32))
            columns["post_len"].append(data["post_len"].astype(np.int32))
            columns["budget"].append(data["budget"].astype(np.int32))
            columns["evict_reason"].append(data["evict_reason"].astype("U16"))
            columns["kept_offsets_per_call"].append(
                data["kept_offsets"].astype(np.int64)
            )
            columns["kept_indices_per_call"].append(
                data["kept_indices"].astype(np.int32)
            )
            columns["evicted_offsets_per_call"].append(
                data["evicted_offsets"].astype(np.int64)
            )
            columns["evicted_indices_per_call"].append(
                data["evicted_indices"].astype(np.int32)
            )
            columns["score_index_per_call"].append(
                data["score_topk_index"].astype(np.int32)
            )
            columns["score_value_per_call"].append(
                data["score_topk_value"].astype(np.float32)
            )

    if not columns["record_step"]:
        return KVEvictionFrame(
            n_rows=0,
            task=np.empty(0, dtype=object),
            call_idx=np.empty(0, dtype=np.int32),
            iter_dir=np.empty(0, dtype=object),
            policy_name=np.empty(0, dtype="U16"),
            record_step=np.empty(0, dtype=np.int32),
            record_layer=np.empty(0, dtype=np.int32),
            record_phase=np.empty(0, dtype="U7"),
            pre_len=np.empty(0, dtype=np.int32),
            post_len=np.empty(0, dtype=np.int32),
            budget=np.empty(0, dtype=np.int32),
            evict_reason=np.empty(0, dtype="U16"),
            kept_offsets=np.zeros(1, dtype=np.int64),
            kept_indices=np.empty(0, dtype=np.int32),
            evicted_offsets=np.zeros(1, dtype=np.int64),
            evicted_indices=np.empty(0, dtype=np.int32),
            kept_per_row=[],
            evicted_per_row=[],
            score_topk_index=np.empty((0, 0), dtype=np.int32),
            score_topk_value=np.empty((0, 0), dtype=np.float32),
        )

    # Per-call CSR offsets concatenated end-to-end need their `[1:]` slice
    # shifted by the running flat-array length so the global offsets remain
    # monotone. Per-call score widths may differ (different h2o k); pad to
    # max width with sentinels (-1 / NaN) before vstacking.
    kept_per_row: list[np.ndarray] = []
    evicted_per_row: list[np.ndarray] = []
    kept_offsets_global: list[int] = [0]
    kept_flat: list[np.ndarray] = []
    evicted_offsets_global: list[int] = [0]
    evicted_flat: list[np.ndarray] = []

    for k_off, k_idx, e_off, e_idx in zip(
        columns["kept_offsets_per_call"],
        columns["kept_indices_per_call"],
        columns["evicted_offsets_per_call"],
        columns["evicted_indices_per_call"],
        strict=True,
    ):
        n = int(k_off.shape[0]) - 1
        for r in range(n):
            kept_per_row.append(k_idx[int(k_off[r]) : int(k_off[r + 1])].copy())
            evicted_per_row.append(e_idx[int(e_off[r]) : int(e_off[r + 1])].copy())
        base_k = kept_offsets_global[-1]
        for r in range(n):
            kept_offsets_global.append(base_k + int(k_off[r + 1]))
        kept_flat.append(k_idx)
        base_e = evicted_offsets_global[-1]
        for r in range(n):
            evicted_offsets_global.append(base_e + int(e_off[r + 1]))
        evicted_flat.append(e_idx)

    score_widths = [arr.shape[1] for arr in columns["score_index_per_call"]]
    max_k = max(score_widths) if score_widths else 0
    score_index_blocks: list[np.ndarray] = []
    score_value_blocks: list[np.ndarray] = []
    for s_idx, s_val in zip(
        columns["score_index_per_call"],
        columns["score_value_per_call"],
        strict=True,
    ):
        n = int(s_idx.shape[0])
        if max_k == 0:
            score_index_blocks.append(np.empty((n, 0), dtype=np.int32))
            score_value_blocks.append(np.empty((n, 0), dtype=np.float32))
            continue
        idx_padded = np.full((n, max_k), -1, dtype=np.int32)
        val_padded = np.full((n, max_k), np.nan, dtype=np.float32)
        idx_padded[:, : s_idx.shape[1]] = s_idx
        val_padded[:, : s_val.shape[1]] = s_val
        score_index_blocks.append(idx_padded)
        score_value_blocks.append(val_padded)

    return KVEvictionFrame(
        n_rows=sum(int(arr.shape[0]) for arr in columns["record_step"]),
        task=np.concatenate(columns["task"]),
        call_idx=np.concatenate(columns["call_idx"]),
        iter_dir=np.concatenate(columns["iter_dir"]),
        policy_name=np.concatenate(columns["policy_name"]),
        record_step=np.concatenate(columns["record_step"]),
        record_layer=np.concatenate(columns["record_layer"]),
        record_phase=np.concatenate(columns["record_phase"]),
        pre_len=np.concatenate(columns["pre_len"]),
        post_len=np.concatenate(columns["post_len"]),
        budget=np.concatenate(columns["budget"]),
        evict_reason=np.concatenate(columns["evict_reason"]),
        kept_offsets=np.asarray(kept_offsets_global, dtype=np.int64),
        kept_indices=np.concatenate(kept_flat) if kept_flat else np.empty(0, dtype=np.int32),
        evicted_offsets=np.asarray(evicted_offsets_global, dtype=np.int64),
        evicted_indices=(
            np.concatenate(evicted_flat) if evicted_flat else np.empty(0, dtype=np.int32)
        ),
        kept_per_row=kept_per_row,
        evicted_per_row=evicted_per_row,
        score_topk_index=(
            np.vstack(score_index_blocks)
            if score_index_blocks
            else np.empty((0, max_k), dtype=np.int32)
        ),
        score_topk_value=(
            np.vstack(score_value_blocks)
            if score_value_blocks
            else np.empty((0, max_k), dtype=np.float32)
        ),
    )


def average_layer_matrix(
    dataset: LayerDistributionSet,
    *,
    layers: Sequence[int] | None = None,
    equal_iter_weight: bool = True,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Return layer labels, average distributions, and observation counts."""
    selected_layers = list(layers or dataset.layers)
    rows: list[np.ndarray] = []
    counts: list[float] = []
    for layer in selected_layers:
        matrix = dataset.distributions[layer]
        obs = dataset.observation_counts[layer]
        valid = obs > 0
        if not bool(valid.any()):
            rows.append(np.zeros(matrix.shape[1], dtype=np.float64))
            counts.append(0.0)
            continue
        if equal_iter_weight:
            row = matrix[valid].mean(axis=0)
        else:
            weights = obs[valid] / float(obs[valid].sum())
            row = np.sum(matrix[valid] * weights[:, None], axis=0)
        rows.append(_normalize(row))
        counts.append(float(obs[valid].sum()))
    return selected_layers, np.vstack(rows), np.asarray(counts, dtype=np.float64)


def parse_layer_selection(value: str | None, available_layers: Sequence[int]) -> list[int]:
    """Parse comma/range layer selection such as `0,8,16-20`."""
    if not value:
        return list(available_layers)
    selected: list[int] = []
    available = set(available_layers)
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            selected.extend(range(start, end + 1))
        else:
            selected.append(int(item))
    missing = [layer for layer in selected if layer not in available]
    if missing:
        raise ValueError(f"selected layers not present in recordings: {missing}")
    return selected


def task_boundaries(records: Sequence[IterationRecord]) -> list[tuple[int, str]]:
    """Return start indices for task blocks in sorted records."""
    boundaries: list[tuple[int, str]] = []
    last_task: str | None = None
    for idx, record in enumerate(records):
        if record.task != last_task:
            boundaries.append((idx, record.task))
            last_task = record.task
    return boundaries


def _task_name(attempt_dir: Path) -> str:
    if attempt_dir.name.startswith("attempt_"):
        return attempt_dir.parent.name
    return attempt_dir.name


def _has_recording_files(iter_dir: Path) -> bool:
    # `kv_eviction.npz` is intentionally NOT required: it is only emitted when
    # an eviction policy is configured (`--kv-policy {streaming,h2o,random}`),
    # and old recordings predate the artifact entirely. `load_kv_eviction()`
    # returns an empty frame on iter dirs without it.
    return (
        (iter_dir / "attention.npz").is_file()
        and (iter_dir / "routing.npz").is_file()
        and (iter_dir / "segments.json").is_file()
    )


def _call_idx_from_iter_dir(iter_dir: Path) -> int:
    match = re.search(r"iter_(\d+)$", iter_dir.name)
    if not match:
        return -1
    return int(match.group(1))


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _segment_roles(record: IterationRecord) -> list[str]:
    payload = json.loads((record.iter_dir / "segments.json").read_text(encoding="utf-8"))
    return [_normalize_role(segment) for segment in payload.get("segments", [])]


def _normalize_role(segment: dict) -> str:
    role = str(segment.get("role") or "other")
    has_tool_calls = bool(segment.get("has_tool_calls"))
    if role == "assistant" and has_tool_calls:
        return "assistant_call"
    if role == "assistant":
        return "assistant_message"
    if role in {"tool", "tool_result"}:
        return "tool_result"
    if role in ROLE_ORDER:
        return role
    return "other"


def _generation_segment_index(segments: Sequence[dict]) -> int | None:
    for idx, segment in enumerate(segments):
        if _normalize_role(segment) == "generation":
            return idx
    return None


def _segment_role_indices(
    segment_roles: Sequence[str], role_index: dict[str, int]
) -> list[int]:
    indices: list[int] = []
    fallback = role_index.get("other")
    for role in segment_roles:
        col = role_index.get(role, fallback)
        if col is None:
            raise ValueError(f"role {role!r} is missing from role labels")
        indices.append(col)
    return indices


def _role_token_counts_for_key_len(
    segments: Sequence[dict], role_index: dict[str, int], key_len: int
) -> np.ndarray:
    counts = np.zeros(len(role_index), dtype=np.float64)
    if key_len <= 0:
        return counts
    for segment in segments:
        start = int(segment.get("token_start", segment.get("start", 0)) or 0)
        end = int(segment.get("token_end", segment.get("end", start)) or start)
        if end <= 0 or start >= key_len:
            continue
        length = max(0, min(end, key_len) - max(start, 0))
        if length <= 0:
            continue
        role = _normalize_role(segment)
        col = role_index.get(role, role_index.get("other"))
        if col is None:
            raise ValueError(f"role {role!r} is missing from role labels")
        counts[col] += float(length)
    return counts


def _normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if np.any(arr < 0) or np.any(~np.isfinite(arr)):
        raise ValueError("distribution contains negative or non-finite values")
    total = float(arr.sum())
    if total <= 0:
        return np.zeros(arr.shape, dtype=np.float64)
    return arr / total


def _ensure_record_slots(
    mapping: dict[int, list[np.ndarray | None]], layer: int, record_index: int
) -> None:
    slots = mapping.setdefault(layer, [])
    while len(slots) <= record_index:
        slots.append(None)


def _ensure_count_slots(
    mapping: dict[int, list[float]], layer: int, record_index: int
) -> None:
    slots = mapping.setdefault(layer, [])
    while len(slots) <= record_index:
        slots.append(0.0)


def _finalize_distribution_slots(
    mapping: dict[int, list[np.ndarray | None]], n_records: int, width: int
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for layer, slots in mapping.items():
        rows: list[np.ndarray] = []
        for idx in range(n_records):
            value = slots[idx] if idx < len(slots) else None
            if value is None:
                rows.append(np.zeros(width, dtype=np.float64))
            else:
                rows.append(value)
        out[layer] = np.vstack(rows)
    return out


def _finalize_count_slots(
    mapping: dict[int, list[float]], n_records: int
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for layer, slots in mapping.items():
        values = [float(slots[idx]) if idx < len(slots) else 0.0 for idx in range(n_records)]
        out[layer] = np.asarray(values, dtype=np.float64)
    return out


def _infer_n_experts(records: Sequence[IterationRecord]) -> int:
    n_experts = 0
    for record in records:
        with np.load(record.iter_dir / "routing.npz") as routing:
            n_experts = max(n_experts, int(routing["n_experts"]))
            if "expert_load" in routing.files:
                n_experts = max(n_experts, int(routing["expert_load"].shape[2]))
    if n_experts <= 0:
        raise ValueError("could not infer a positive expert count")
    return n_experts
