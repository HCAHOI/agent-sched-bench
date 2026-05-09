"""Load compact distributions from HF recording artifacts.

The loader deliberately keeps only per-iteration, per-layer aggregate
distributions. It does not retain raw query rows or token-level routing arrays,
so it can be used on multi-GB recording runs without materializing the whole run
in memory.
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
                key_len = int(np.max(positions)) + 1
                if key_len <= 0:
                    continue
                counts = _role_token_counts_for_key_len(segments, role_index, key_len)
                row_count = int(end - start)
                layer_int = int(layer)
                layer_sums.setdefault(layer_int, np.zeros(len(labels), dtype=np.float64))
                layer_sums[layer_int] += counts * float(row_count)
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


def load_moe_distributions(records: Sequence[IterationRecord]) -> LayerDistributionSet:
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
            if expert_load.shape[2] > n_experts:
                raise ValueError(
                    f"{record.iter_dir}: expert dimension {expert_load.shape[2]} > {n_experts}"
                )

            layer_sums: dict[int, np.ndarray] = {}
            layer_counts: dict[int, float] = {}
            for idx, layer in enumerate(record_layers):
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
        modality="moe",
        records=list(records),
        layers=sorted(distributions),
        axis_labels=labels,
        distributions=distributions,
        observation_counts=observation_counts,
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
