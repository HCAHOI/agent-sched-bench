"""Evaluate KV-swap profile thresholds against observed tool-gap windows.

This module deliberately separates two measurements:

* KV swap costs come from an explicit profile file. Test fixtures may use toy
  profiles, but experiment inputs should be produced by the serving engine's
  real swap path.
* Tool gaps come from observed traces or trace-derived datasets.

The sweep here is an oracle feasibility check: it labels which observed gaps are
long enough for a profiled KV operation plus guard time. It is not a learned
predictor and must not be reported as online classifier accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

_QUANTILE_FIELDS = frozenset({"p50_ms", "p90_ms", "p95_ms", "p99_ms"})


@dataclass(frozen=True)
class KVSwapProfileEntry:
    """One real measured KV swap latency profile bucket."""

    engine: str
    mechanism: str
    device: str
    kv_size_mb: float
    direction: str
    memory_path: str
    concurrency: str
    samples: int
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, source: str) -> "KVSwapProfileEntry":
        values = {
            "engine": _required_text(raw, "engine", source=source),
            "mechanism": _required_text(raw, "mechanism", source=source),
            "device": _required_text(raw, "device", source=source),
            "kv_size_mb": _required_positive_float(raw, "kv_size_mb", source=source),
            "direction": _required_text(raw, "direction", source=source),
            "memory_path": _required_text(raw, "memory_path", source=source),
            "concurrency": _required_text(raw, "concurrency", source=source),
            "samples": _required_positive_int(raw, "samples", source=source),
            "p50_ms": _required_nonnegative_float(raw, "p50_ms", source=source),
            "p90_ms": _required_nonnegative_float(raw, "p90_ms", source=source),
            "p95_ms": _required_nonnegative_float(raw, "p95_ms", source=source),
            "p99_ms": _required_nonnegative_float(raw, "p99_ms", source=source),
        }
        entry = cls(**values)
        if not (entry.p50_ms <= entry.p90_ms <= entry.p95_ms <= entry.p99_ms):
            raise ValueError(f"{source}: profile quantiles must be monotonic")
        return entry

    def quantile_ms(self, quantile: str) -> float:
        field = f"{quantile}_ms" if not quantile.endswith("_ms") else quantile
        if field not in _QUANTILE_FIELDS:
            allowed = ", ".join(sorted(q.removesuffix("_ms") for q in _QUANTILE_FIELDS))
            raise ValueError(f"unsupported quantile {quantile!r}; choose one of {allowed}")
        return float(getattr(self, field))

    def key(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "mechanism": self.mechanism,
            "device": self.device,
            "kv_size_mb": self.kv_size_mb,
            "direction": self.direction,
            "memory_path": self.memory_path,
            "concurrency": self.concurrency,
            "samples": self.samples,
        }


@dataclass(frozen=True)
class ToolGapSample:
    """One observed window between an LLM response and the next LLM call."""

    sample_id: str
    available_gap_ms: float
    tool_names: tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, source: str) -> "ToolGapSample":
        sample_id = str(raw.get("sample_id") or raw.get("id") or source)
        gap_ms = _required_nonnegative_float(raw, "available_gap_ms", source=source)
        tool_names_raw = raw.get("tool_names") or raw.get("tools") or []
        if not isinstance(tool_names_raw, list):
            raise ValueError(f"{source}: tool_names must be a list")
        tool_names = tuple(str(name).strip() for name in tool_names_raw if str(name).strip())
        return cls(sample_id=sample_id, available_gap_ms=gap_ms, tool_names=tool_names)


def load_profile(path: Path) -> list[KVSwapProfileEntry]:
    entries = [
        KVSwapProfileEntry.from_mapping(raw, source=f"{path}:{line_no}")
        for line_no, raw in _read_jsonl_objects(path)
    ]
    if not entries:
        raise ValueError(f"empty KV profile: {path}")
    return entries


def load_tool_gaps(path: Path) -> list[ToolGapSample]:
    samples = [
        ToolGapSample.from_mapping(raw, source=f"{path}:{line_no}")
        for line_no, raw in _read_jsonl_objects(path)
    ]
    if not samples:
        raise ValueError(f"empty tool-gap dataset: {path}")
    return samples


def filter_profiles(
    entries: Iterable[KVSwapProfileEntry],
    *,
    engine: str | None = None,
    direction: str | None = None,
    kv_size_mb: float | None = None,
) -> list[KVSwapProfileEntry]:
    filtered = []
    for entry in entries:
        if engine is not None and entry.engine != engine:
            continue
        if direction is not None and entry.direction != direction:
            continue
        if kv_size_mb is not None and entry.kv_size_mb != kv_size_mb:
            continue
        filtered.append(entry)
    if not filtered:
        raise ValueError("profile filters matched no entries")
    return filtered


def evaluate_profile_sweep(
    profiles: Iterable[KVSwapProfileEntry],
    gaps: Iterable[ToolGapSample],
    *,
    quantile: str,
    guard_ms_values: Iterable[float],
) -> list[dict[str, Any]]:
    gap_list = list(gaps)
    if not gap_list:
        raise ValueError("cannot evaluate sweep with no tool-gap samples")
    results: list[dict[str, Any]] = []
    for profile in profiles:
        kv_cost_ms = profile.quantile_ms(quantile)
        for guard_ms in guard_ms_values:
            if guard_ms < 0:
                raise ValueError(f"guard_ms must be non-negative, got {guard_ms}")
            threshold_ms = kv_cost_ms + guard_ms
            safe_gaps = [
                sample.available_gap_ms
                for sample in gap_list
                if sample.available_gap_ms >= threshold_ms
            ]
            unsafe_gaps = [
                sample.available_gap_ms
                for sample in gap_list
                if sample.available_gap_ms < threshold_ms
            ]
            safe_count = len(safe_gaps)
            total_count = len(gap_list)
            exposed_if_always_swap_ms = sum(
                max(0.0, kv_cost_ms - sample.available_gap_ms)
                for sample in gap_list
            )
            absorbed_if_oracle_ms = safe_count * kv_cost_ms
            results.append(
                {
                    "profile": profile.key(),
                    "quantile": quantile.removesuffix("_ms"),
                    "kv_cost_ms": kv_cost_ms,
                    "guard_ms": guard_ms,
                    "threshold_ms": threshold_ms,
                    "total_count": total_count,
                    "safe_count": safe_count,
                    "unsafe_count": len(unsafe_gaps),
                    "safe_rate": safe_count / total_count,
                    "min_safe_slack_ms": min(
                        (gap - threshold_ms for gap in safe_gaps),
                        default=None,
                    ),
                    "max_unsafe_deficit_ms": max(
                        (threshold_ms - gap for gap in unsafe_gaps),
                        default=None,
                    ),
                    "absorbed_if_oracle_ms": absorbed_if_oracle_ms,
                    "exposed_if_always_swap_ms": exposed_if_always_swap_ms,
                }
            )
    return results


def _read_jsonl_objects(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append((line_no, raw))
    return rows


def _required_text(raw: Mapping[str, Any], field: str, *, source: str) -> str:
    value = raw.get(field)
    if value is None:
        raise ValueError(f"{source}: missing required field {field!r}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{source}: empty required field {field!r}")
    return text


def _required_nonnegative_float(raw: Mapping[str, Any], field: str, *, source: str) -> float:
    value = raw.get(field)
    if value is None:
        raise ValueError(f"{source}: missing required field {field!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: field {field!r} must be numeric") from exc
    if number < 0:
        raise ValueError(f"{source}: field {field!r} must be non-negative")
    return number


def _required_positive_float(raw: Mapping[str, Any], field: str, *, source: str) -> float:
    number = _required_nonnegative_float(raw, field, source=source)
    if number <= 0:
        raise ValueError(f"{source}: field {field!r} must be positive")
    return number


def _required_positive_int(raw: Mapping[str, Any], field: str, *, source: str) -> int:
    number = _required_positive_float(raw, field, source=source)
    if int(number) != number:
        raise ValueError(f"{source}: field {field!r} must be an integer")
    return int(number)
