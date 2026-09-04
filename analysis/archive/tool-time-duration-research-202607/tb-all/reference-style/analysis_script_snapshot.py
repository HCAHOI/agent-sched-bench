#!/usr/bin/env python3
"""Plot reproducible per-call tool execution latency distributions."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
from importlib.metadata import version as package_version
import json
import math
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import bashlex
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from trace_collect.tool_gap_extractor import discover_trace_files
from trace_collect.tool_latency_dataset import extract_tool_latency_samples


CPU_MIN_CALL_DURATION_S = 4.0
CPU_MIN_INTERIOR_SAMPLES = 2
EXCLUDED_TOOL_NAMES = frozenset({"message"})
MIN_CONTENT_CATEGORY_PLOT_CALLS = 100
EXPECTED_RESOURCE_INTERVAL_S = 2.0
CPU_MAX_RESOURCE_GAP_S = 1.5 * EXPECTED_RESOURCE_INTERVAL_S


@dataclass(frozen=True)
class CohortSpec:
    """One preregistered cohort assembled from one or more trace roots."""

    label: str
    expected_trace_count: int
    roots: tuple[Path, ...]


@dataclass(frozen=True)
class ToolCall:
    """Tool latency observation with optional aligned CPU utilization."""

    cohort: str
    task_id: str
    tool_name: str
    latency_ms: float
    success: bool | None
    ts_start: float
    ts_end: float
    cpu_percent: float | None
    cpu_exclusion_reason: str | None
    exec_command: str | None


@dataclass(frozen=True)
class TraceRecord:
    """Trace identity and provenance retained in the cohort manifest."""

    path: Path
    task_id: str
    benchmark: str
    model: str
    scaffold: str
    tool_execution_environment: str
    tool_runtime: str
    sha256: str
    resources_path: Path | None
    resources_sha256: str | None


@dataclass(frozen=True)
class CohortData:
    """Loaded calls, task identities, and resource coverage for one cohort."""

    spec: CohortSpec
    calls: tuple[ToolCall, ...]
    traces: tuple[TraceRecord, ...]
    resource_trace_count: int
    resource_sample_gaps_s: tuple[float, ...]
    excluded_tool_counts: tuple[tuple[str, int], ...]

    @property
    def contributing_task_count(self) -> int:
        return len({call.task_id for call in self.calls})


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Plot vertically stacked histogram and CDF overviews for one or "
            "more tool-latency cohorts."
        )
    )
    parser.add_argument(
        "--cohort",
        action="append",
        required=True,
        metavar="LABEL=COUNT=PATH",
        help=(
            "English cohort label, expected trace count, and trace root; repeat "
            "with the same label/count to add another root"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New or empty directory for figures and tabular summaries",
    )
    parser.add_argument(
        "--cpu-min-duration-s",
        type=float,
        default=CPU_MIN_CALL_DURATION_S,
        help="CPU alignment threshold; must be at least 4 seconds",
    )
    return parser


def parse_cohort_specs(values: Sequence[str]) -> list[CohortSpec]:
    """Parse repeated ``LABEL=COUNT=PATH`` values, preserving label order."""

    grouped: dict[str, tuple[int, list[Path]]] = {}
    for value in values:
        parts = value.split("=", maxsplit=2)
        if len(parts) != 3:
            raise ValueError(f"invalid cohort {value!r}; expected LABEL=COUNT=PATH")
        label, raw_count, raw_path = (part.strip() for part in parts)
        if not label or not label.isascii() or not raw_path:
            raise ValueError(f"cohort label must be nonempty English/ASCII: {label!r}")
        try:
            expected_count = int(raw_count)
        except ValueError as exc:
            raise ValueError(f"invalid expected trace count {raw_count!r}") from exc
        if expected_count <= 0:
            raise ValueError("expected trace count must be positive")
        if label in grouped:
            prior_count, roots = grouped[label]
            if prior_count != expected_count:
                raise ValueError(f"conflicting expected counts for cohort {label!r}")
            roots.append(Path(raw_path))
        else:
            grouped[label] = (expected_count, [Path(raw_path)])
    return [
        CohortSpec(label=label, expected_trace_count=count, roots=tuple(roots))
        for label, (count, roots) in grouped.items()
    ]


def _parse_cpu_percent(value: object, *, source: str) -> float:
    if isinstance(value, int | float):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip().removesuffix("%"))
        except ValueError as exc:
            raise ValueError(f"{source}: invalid cpu_percent {value!r}") from exc
    else:
        raise ValueError(f"{source}: missing or nonnumeric cpu_percent")
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{source}: invalid cpu_percent {value!r}")
    return number


def load_resource_samples(path: Path) -> list[tuple[float, float]]:
    """Load strictly validated ``(epoch, cpu_percent)`` resource samples."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError(f"{path}: field 'samples' must be a list")
    samples: list[tuple[float, float]] = []
    for index, raw in enumerate(raw_samples):
        source = f"{path}:samples[{index}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{source}: expected an object")
        epoch = raw.get("epoch")
        if not isinstance(epoch, int | float) or not math.isfinite(float(epoch)):
            raise ValueError(f"{source}: missing or invalid epoch")
        samples.append(
            (
                float(epoch),
                _parse_cpu_percent(raw.get("cpu_percent"), source=source),
            )
        )
    samples.sort()
    if any(left[0] >= right[0] for left, right in zip(samples, samples[1:])):
        raise ValueError(f"{path}: resource sample epochs must be unique and increasing")
    return samples


def _interpolate(points: Sequence[tuple[float, float]], timestamp: float) -> float:
    epochs = [point[0] for point in points]
    right_index = bisect.bisect_left(epochs, timestamp)
    if right_index < len(points) and points[right_index][0] == timestamp:
        return points[right_index][1]
    if right_index == 0 or right_index == len(points):
        raise ValueError("timestamp is outside the resource sample range")
    left_time, left_value = points[right_index - 1]
    right_time, right_value = points[right_index]
    fraction = (timestamp - left_time) / (right_time - left_time)
    return left_value + fraction * (right_value - left_value)


def overlap_weighted_cpu(
    samples: Sequence[tuple[float, float]],
    *,
    ts_start: float,
    ts_end: float,
    min_duration_s: float,
) -> tuple[float | None, str | None]:
    """Estimate mean CPU over a long, fully covered call or return a reason."""

    if ts_end - ts_start < min_duration_s:
        return None, "call_shorter_than_cpu_threshold"
    if len(samples) < 2:
        return None, "resource_samples_unavailable"
    if ts_start < samples[0][0] or ts_end > samples[-1][0]:
        return None, "call_outside_resource_coverage"
    epochs = [point[0] for point in samples]
    for boundary in (ts_start, ts_end):
        right_index = bisect.bisect_left(epochs, boundary)
        if right_index < len(samples) and samples[right_index][0] == boundary:
            continue
        if samples[right_index][0] - samples[right_index - 1][0] > CPU_MAX_RESOURCE_GAP_S:
            return None, "resource_boundary_interpolation_gap_exceeds_3_seconds"

    interior = [point for point in samples if ts_start < point[0] < ts_end]
    if len(interior) < CPU_MIN_INTERIOR_SAMPLES:
        return None, "fewer_than_two_interior_resource_samples"

    knots = [(ts_start, _interpolate(samples, ts_start)), *interior]
    knots.append((ts_end, _interpolate(samples, ts_end)))
    if any(
        right[0] - left[0] > CPU_MAX_RESOURCE_GAP_S
        for left, right in zip(knots, knots[1:])
    ):
        return None, "resource_gap_exceeds_3_seconds"
    integral = sum(
        (right_time - left_time) * (left_value + right_value) / 2.0
        for (left_time, left_value), (right_time, right_value) in zip(
            knots, knots[1:]
        )
    )
    return integral / (ts_end - ts_start), None


def _read_trace_metadata(trace_path: Path) -> dict[str, object]:
    with trace_path.open(encoding="utf-8") as handle:
        first_line = handle.readline()
    payload = json.loads(first_line)
    if not isinstance(payload, dict) or payload.get("type") != "trace_metadata":
        raise ValueError(f"{trace_path}: first line is not trace_metadata")
    return payload


def _required_metadata_text(
    metadata: dict[str, object], field: str, *, source: Path
) -> str:
    value = metadata.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: missing metadata field {field!r}")
    return value.strip()


def _required_runtime_text(
    metadata: dict[str, object], field: str, *, source: Path
) -> str:
    value = metadata.get(field)
    if not isinstance(value, str) or not value.strip():
        runtime_proof = metadata.get("runtime_proof")
        if isinstance(runtime_proof, dict):
            value = runtime_proof.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: missing runtime field {field!r}")
    return value.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cohort_sha256(traces: Sequence[TraceRecord]) -> str:
    digest = hashlib.sha256()
    for trace in sorted(traces, key=lambda item: item.task_id):
        fields = (
            trace.benchmark,
            trace.model,
            trace.task_id,
            trace.sha256,
            trace.resources_sha256 or "",
        )
        digest.update("\0".join(fields).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def load_cohort(spec: CohortSpec, *, cpu_min_duration_s: float) -> CohortData:
    """Load and validate one homogeneous, one-trace-per-task cohort."""

    trace_paths = discover_trace_files(spec.roots)
    if len(trace_paths) != spec.expected_trace_count:
        raise ValueError(
            f"cohort {spec.label!r}: expected {spec.expected_trace_count} traces, "
            f"found {len(trace_paths)}"
        )

    calls: list[ToolCall] = []
    traces: list[TraceRecord] = []
    task_ids: set[str] = set()
    resource_trace_count = 0
    resource_sample_gaps: list[float] = []
    excluded_tool_counts: Counter[str] = Counter()
    for trace_path in trace_paths:
        metadata = _read_trace_metadata(trace_path)
        task_id = _required_metadata_text(metadata, "instance_id", source=trace_path)
        if task_id in task_ids:
            raise ValueError(f"cohort {spec.label!r}: duplicate task ID {task_id!r}")
        task_ids.add(task_id)
        resources_path = trace_path.parent / "resources.json"
        resource_samples: list[tuple[float, float]] = []
        resources_resolved: Path | None = None
        resources_sha256: str | None = None
        if resources_path.is_file():
            resources_resolved = resources_path.resolve()
            resources_sha256 = _sha256(resources_path)
            resource_samples = load_resource_samples(resources_path)
            if resource_samples:
                resource_trace_count += 1
                resource_sample_gaps.extend(
                    right[0] - left[0]
                    for left, right in zip(resource_samples, resource_samples[1:])
                )
        trace_record = TraceRecord(
            path=trace_path.resolve(),
            task_id=task_id,
            benchmark=_required_metadata_text(metadata, "benchmark", source=trace_path),
            model=_required_metadata_text(metadata, "model", source=trace_path),
            scaffold=_required_metadata_text(metadata, "scaffold", source=trace_path),
            tool_execution_environment=_required_runtime_text(
                metadata, "tool_execution_environment", source=trace_path
            ),
            tool_runtime=_required_runtime_text(metadata, "tool_runtime", source=trace_path),
            sha256=_sha256(trace_path),
            resources_path=resources_resolved,
            resources_sha256=resources_sha256,
        )
        traces.append(trace_record)
        for sample in extract_tool_latency_samples(trace_path):
            if sample.task_id != task_id:
                raise ValueError(
                    f"{trace_path}: extractor task ID {sample.task_id!r} does not "
                    f"match metadata {task_id!r}"
                )
            if sample.tool_name in EXCLUDED_TOOL_NAMES:
                excluded_tool_counts[sample.tool_name] += 1
                continue
            cpu_percent, exclusion_reason = overlap_weighted_cpu(
                resource_samples,
                ts_start=sample.tool_ts_start,
                ts_end=sample.tool_ts_end,
                min_duration_s=cpu_min_duration_s,
            )
            calls.append(
                ToolCall(
                    cohort=spec.label,
                    task_id=task_id,
                    tool_name=sample.tool_name,
                    latency_ms=sample.latency_ms,
                    success=sample.success,
                    ts_start=sample.tool_ts_start,
                    ts_end=sample.tool_ts_end,
                    cpu_percent=cpu_percent,
                    cpu_exclusion_reason=exclusion_reason,
                    exec_command=(
                        sample.tool_args.get("command")
                        if sample.tool_name == "exec"
                        and isinstance(sample.tool_args, dict)
                        and isinstance(sample.tool_args.get("command"), str)
                        else None
                    ),
                )
            )

    for field in (
        "benchmark",
        "model",
        "scaffold",
        "tool_execution_environment",
        "tool_runtime",
    ):
        values = {getattr(trace, field) for trace in traces}
        if len(values) != 1:
            raise ValueError(f"cohort {spec.label!r} is heterogeneous in {field}: {values}")
    return CohortData(
        spec=spec,
        calls=tuple(calls),
        traces=tuple(traces),
        resource_trace_count=resource_trace_count,
        resource_sample_gaps_s=tuple(resource_sample_gaps),
        excluded_tool_counts=tuple(sorted(excluded_tool_counts.items())),
    )


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated sample percentile."""

    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    left = math.floor(position)
    right = math.ceil(position)
    if left == right:
        return ordered[left]
    fraction = position - left
    return ordered[left] * (1.0 - fraction) + ordered[right] * fraction


def distribution_xy(
    values: Sequence[float],
) -> tuple[list[float], list[float], list[float]]:
    """Return unique x, P(X <= x), and P(X >= x) empirical curves."""

    if not values:
        raise ValueError("cannot build a distribution from no values")
    counts = Counter(values)
    x_values = sorted(counts)
    total = len(values)
    cumulative = 0
    ecdf: list[float] = []
    survival: list[float] = []
    for value in x_values:
        survival.append((total - cumulative) / total)
        cumulative += counts[value]
        ecdf.append(cumulative / total)
    return x_values, ecdf, survival


LATENCY_TICKS_MS = (0.0, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0)


def _save_figure(figure: plt.Figure, output_dir: Path, filename: str) -> None:
    figure.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _set_panel_aspect(axis: plt.Axes) -> None:
    """Fix every subplot to a width:height ratio of 2:1."""

    axis.set_box_aspect(0.5)
    axis.grid(True, which="both", alpha=0.25)


def _histogram_edges(calls: Sequence[ToolCall]) -> np.ndarray:
    transformed = np.log1p([call.latency_ms for call in calls])
    edges = np.histogram_bin_edges(transformed, bins="fd")
    if len(edges) < 2:
        raise ValueError("could not construct histogram bins")
    return edges


def _configure_histogram_x_axis(
    axis: plt.Axes, *, maximum_latency_ms: float
) -> None:
    ticks = [value for value in LATENCY_TICKS_MS if value <= maximum_latency_ms]
    if not ticks or ticks[-1] < maximum_latency_ms:
        ticks.append(maximum_latency_ms)
    axis.set_xticks(np.log1p(ticks), [f"{value:g}" for value in ticks])
    axis.set_xlabel("Tool execution latency (ms; log1p scale)")
    _set_panel_aspect(axis)


def _configure_histogram_axis(axis: plt.Axes, *, maximum_latency_ms: float) -> None:
    _configure_histogram_x_axis(axis, maximum_latency_ms=maximum_latency_ms)
    axis.set_ylabel("Probability per bin")
    axis.set_yscale("log")


def probability_weights(count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("histogram series must contain at least one call")
    return np.full(count, 1.0 / count)


def _plot_probability_histogram(
    axis: plt.Axes,
    values: Sequence[float],
    *,
    edges: np.ndarray,
    label: str,
    filled: bool = False,
) -> None:
    transformed = np.log1p(values)
    weights = probability_weights(len(values))
    axis.hist(
        transformed,
        bins=edges,
        weights=weights,
        histtype="stepfilled" if filled else "step",
        alpha=0.35 if filled else 1.0,
        linewidth=1.6,
        label=label,
    )


def plot_histogram_overview(
    cohorts: Sequence[CohortData],
    *,
    output_dir: Path,
    maximum_latency_ms: float,
) -> np.ndarray:
    """Write one vertically stacked benchmark/tool/total histogram figure."""

    all_calls = [call for cohort in cohorts for call in cohort.calls]
    edges = _histogram_edges(all_calls)
    figure, axes = plt.subplots(3, 1, figsize=(12, 18), constrained_layout=True)

    for cohort in cohorts:
        values = [call.latency_ms for call in cohort.calls]
        _plot_probability_histogram(
            axes[0],
            values,
            edges=edges,
            label=f"{cohort.spec.label} (n={len(values):,})",
        )
    axes[0].set_title("Latency histogram by benchmark")

    tool_names = sorted({call.tool_name for call in all_calls})
    for tool_name in tool_names:
        values = [call.latency_ms for call in all_calls if call.tool_name == tool_name]
        _plot_probability_histogram(
            axes[1], values, edges=edges, label=f"{tool_name} (n={len(values):,})"
        )
    axes[1].set_title("Latency histogram by tool type (benchmarks pooled)")

    total_values = [call.latency_ms for call in all_calls]
    _plot_probability_histogram(
        axes[2],
        total_values,
        edges=edges,
        label=f"All benchmarks and tools (n={len(total_values):,})",
        filled=True,
    )
    axes[2].set_title("Total latency histogram")

    for axis in axes:
        _configure_histogram_axis(axis, maximum_latency_ms=maximum_latency_ms)
        axis.legend()
    figure.suptitle("Tool execution latency histograms", fontsize=16)
    _save_figure(figure, output_dir, "tool_latency_histograms.png")
    return edges


def _configure_cdf_axis(axis: plt.Axes, *, maximum_latency_ms: float) -> None:
    axis.set_xscale("symlog", linthresh=1.0, linscale=0.5)
    axis.set_xlim(left=0.0, right=maximum_latency_ms * 1.05)
    axis.set_ylim(0.0, 1.01)
    axis.set_xlabel("Tool execution latency (ms)")
    axis.set_ylabel("Cumulative probability")
    _set_panel_aspect(axis)


def cdf_plot_xy(
    values: Sequence[float], *, maximum_latency_ms: float
) -> tuple[list[float], list[float]]:
    """Build a complete ECDF from zero through the shared comparison domain."""

    x_values, ecdf, _ = distribution_xy(values)
    plot_x = [0.0, *x_values]
    plot_y = [0.0, *ecdf]
    if x_values[-1] < maximum_latency_ms:
        plot_x.append(maximum_latency_ms)
        plot_y.append(1.0)
    return plot_x, plot_y


def _plot_cdf(
    axis: plt.Axes,
    values: Sequence[float],
    *,
    label: str,
    maximum_latency_ms: float,
) -> None:
    x_values, ecdf = cdf_plot_xy(
        values, maximum_latency_ms=maximum_latency_ms
    )
    axis.step(x_values, ecdf, where="post", linewidth=1.7, label=label)


def plot_cdf_overview(
    cohorts: Sequence[CohortData],
    *,
    output_dir: Path,
    maximum_latency_ms: float,
) -> tuple[tuple[str, ...], Counter[str]]:
    """Write benchmark/tool/exec-structure/total CDFs in one stacked figure."""

    all_calls = [call for cohort in cohorts for call in cohort.calls]
    figure, axes = plt.subplots(4, 1, figsize=(12, 24), constrained_layout=True)

    for cohort in cohorts:
        values = [call.latency_ms for call in cohort.calls]
        _plot_cdf(
            axes[0],
            values,
            label=f"{cohort.spec.label} (n={len(values):,})",
            maximum_latency_ms=maximum_latency_ms,
        )
    axes[0].set_title("CDF by benchmark")

    tool_names = sorted({call.tool_name for call in all_calls})
    for tool_name in tool_names:
        values = [call.latency_ms for call in all_calls if call.tool_name == tool_name]
        _plot_cdf(
            axes[1],
            values,
            label=f"{tool_name} (n={len(values):,})",
            maximum_latency_ms=maximum_latency_ms,
        )
    axes[1].set_title("CDF by tool type (benchmarks pooled)")

    exec_calls = [call for call in all_calls if call.tool_name == "exec"]
    category_counts = Counter(
        exec_content_category(call.exec_command) for call in exec_calls
    )
    displayed_categories = displayed_command_categories(category_counts)
    structure_labels = (
        ("no simple operator", "Single command"),
        ("sequential operators", "Command chain (;, &&, ||)"),
        ("pipeline/background operators", "Pipeline/background (|, &)"),
        ("mixed operators", "Mixed shell operators"),
        ("unclassified", "Complex/unsupported shell syntax"),
    )
    for structure, label in structure_labels:
        values = [
            call.latency_ms
            for call in exec_calls
            if shell_structure(call.exec_command) == structure
        ]
        if values:
            _plot_cdf(
                axes[2],
                values,
                label=f"{label} (n={len(values):,})",
                maximum_latency_ms=maximum_latency_ms,
            )
    axes[2].set_title("Exec CDF by shell command structure")

    total_values = [call.latency_ms for call in all_calls]
    _plot_cdf(
        axes[3],
        total_values,
        label=f"All benchmarks and tools (n={len(total_values):,})",
        maximum_latency_ms=maximum_latency_ms,
    )
    axes[3].set_title("Total CDF")

    for index, axis in enumerate(axes):
        _configure_cdf_axis(axis, maximum_latency_ms=maximum_latency_ms)
        axis.legend(fontsize="x-small", ncol=2 if index == 2 else 1)
    figure.suptitle("Tool execution latency CDFs", fontsize=16)
    _save_figure(figure, output_dir, "tool_latency_cdf.png")
    return displayed_categories, category_counts


@lru_cache(maxsize=None)
def _parse_simple_shell(command: str) -> tuple[object, ...] | None:
    try:
        roots = tuple(bashlex.parse(command))
    except (bashlex.errors.ParsingError, NotImplementedError):
        return None
    simple_kinds = {"command", "list", "pipeline"}
    return roots if roots and all(root.kind in simple_kinds for root in roots) else None


@lru_cache(maxsize=None)
def command_parse_status(command: str | None) -> str:
    if not isinstance(command, str) or not command.strip():
        return "missing command"
    return "parsed" if _parse_simple_shell(command) is not None else "unsupported syntax"


def _top_level_commands(node: object) -> list[object]:
    kind = getattr(node, "kind", None)
    if kind == "command":
        return [node]
    if kind not in {"list", "pipeline"}:
        return []
    commands: list[object] = []
    for part in node.parts:
        if part.kind in {"command", "list", "pipeline"}:
            commands.extend(_top_level_commands(part))
    return commands


def _command_words(node: object) -> list[str]:
    words: list[str] = []
    for part in node.parts:
        if part.kind != "word":
            continue
        word = part.word
        name, separator, _ = word.partition("=")
        if separator and name.isidentifier() and not words:
            continue
        words.append(word)
    return words


def _command_head(node: object) -> str | None:
    words = _command_words(node)
    return words[0].rsplit("/", 1)[-1] if words else None


def _first_effective_command_node(command: str | None) -> object | None:
    if not isinstance(command, str):
        return None
    roots = _parse_simple_shell(command)
    if roots is None:
        return None
    for node in (child for root in roots for child in _top_level_commands(root)):
        if _command_head(node) != "cd":
            return node
    return None


@lru_cache(maxsize=None)
def effective_command_head(command: str | None) -> str:
    """Return the first non-cd top-level command head from the bash AST."""

    node = _first_effective_command_node(command)
    return (_command_head(node) if node is not None else None) or "unclassified"


@lru_cache(maxsize=None)
def exec_content_category(command: str | None) -> str:
    """Return the first-head category, recognizing Python pytest entry points."""

    node = _first_effective_command_node(command)
    if node is None:
        return "unclassified"
    words = _command_words(node)
    if not words:
        return "unclassified"
    head = words[0].rsplit("/", 1)[-1]
    if head == "pytest":
        return "pytest"
    if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", head) and words[1:3] == [
        "-m",
        "pytest",
    ]:
        return "pytest"
    return head


def _is_cd_command(node: object) -> bool:
    if getattr(node, "kind", None) != "command":
        return False
    return _command_head(node) == "cd"


def _contains_only_cd_commands(node: object) -> bool:
    commands = _top_level_commands(node)
    return bool(commands) and all(_is_cd_command(command) for command in commands)


def _operator_flags(roots: Sequence[object]) -> tuple[bool, bool]:
    remaining_roots = list(roots)
    while len(remaining_roots) > 1 and _contains_only_cd_commands(
        remaining_roots[0]
    ):
        remaining_roots.pop(0)
    has_sequential = len(remaining_roots) > 1
    has_pipeline = False

    def visit(node: object) -> None:
        nonlocal has_sequential, has_pipeline
        kind = getattr(node, "kind", None)
        if kind == "pipeline":
            has_pipeline = has_pipeline or any(
                part.kind == "pipe" for part in node.parts
            )
            return
        if kind != "list":
            return
        parts = list(node.parts)
        while (
            len(parts) >= 2
            and _is_cd_command(parts[0])
            and parts[1].kind == "operator"
            and parts[1].op in {"&&", ";"}
        ):
            parts = parts[2:]
        for part in parts:
            if part.kind == "operator":
                if part.op in {"&&", ";", ";;", "||", ";&", ";;&"}:
                    has_sequential = True
                elif part.op in {"&"}:
                    has_pipeline = True
            elif part.kind in {"list", "pipeline"}:
                visit(part)

    for root in remaining_roots:
        visit(root)
    return has_sequential, has_pipeline


@lru_cache(maxsize=None)
def shell_structure(command: str | None) -> str:
    """Classify top-level shell operators with bashlex, excluding nested words."""

    if not isinstance(command, str):
        return "unclassified"
    roots = _parse_simple_shell(command)
    if roots is None:
        return "unclassified"
    has_sequential, has_pipeline = _operator_flags(roots)
    if has_sequential and has_pipeline:
        return "mixed operators"
    if has_sequential:
        return "sequential operators"
    if has_pipeline:
        return "pipeline/background operators"
    return "no simple operator"


def displayed_command_categories(category_counts: Counter[str]) -> tuple[str, ...]:
    """Select separately plotted content categories using the support rule."""

    return tuple(
        sorted(
            category
            for category, count in category_counts.items()
            if count >= MIN_CONTENT_CATEGORY_PLOT_CALLS and category != "unclassified"
        )
    )


def _summary_rows(cohorts: Sequence[CohortData]) -> Iterable[dict[str, object]]:
    quantiles = (("p50_ms", 0.50), ("p90_ms", 0.90), ("p95_ms", 0.95), ("p99_ms", 0.99))
    for cohort in cohorts:
        tools = ["__all__", *sorted({call.tool_name for call in cohort.calls})]
        for tool_name in tools:
            selected = (
                list(cohort.calls)
                if tool_name == "__all__"
                else [call for call in cohort.calls if call.tool_name == tool_name]
            )
            values = [call.latency_ms for call in selected]
            row: dict[str, object] = {
                "cohort": cohort.spec.label,
                "tool": tool_name,
                "call_count": len(selected),
                "task_count": len({call.task_id for call in selected}),
                "success_count": sum(call.success is True for call in selected),
                "failure_count": sum(call.success is False for call in selected),
                "unknown_success_count": sum(call.success is None for call in selected),
                "cpu_aligned_count": sum(call.cpu_percent is not None for call in selected),
                "min_ms": min(values),
                "max_ms": max(values),
            }
            row.update({name: percentile(values, value) for name, value in quantiles})
            yield row


def write_exec_command_summary(
    cohorts: Sequence[CohortData],
    *,
    output_dir: Path,
    displayed_categories: Sequence[str],
) -> None:
    """Write complete command classification and aggregate latency summaries."""

    fieldnames = (
        "cohort",
        "dimension",
        "category",
        "separately_plotted",
        "call_count",
        "task_count",
        "p50_ms",
        "p90_ms",
        "p95_ms",
        "p99_ms",
        "max_ms",
    )
    displayed_set = set(displayed_categories)
    cohort_calls = [
        (cohort.spec.label, [call for call in cohort.calls if call.tool_name == "exec"])
        for cohort in cohorts
    ]
    cohort_calls.append(
        ("All benchmarks", [call for _, calls in cohort_calls for call in calls])
    )
    with (output_dir / "exec_command_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        def write_row(
            cohort_label: str,
            dimension: str,
            category: str,
            selected: Sequence[ToolCall],
            *,
            separately_plotted: bool,
        ) -> None:
            values = [call.latency_ms for call in selected]
            writer.writerow(
                {
                    "cohort": cohort_label,
                    "dimension": dimension,
                    "category": category,
                    "separately_plotted": separately_plotted,
                    "call_count": len(selected),
                    "task_count": len(
                        {(call.cohort, call.task_id) for call in selected}
                    ),
                    "p50_ms": percentile(values, 0.50),
                    "p90_ms": percentile(values, 0.90),
                    "p95_ms": percentile(values, 0.95),
                    "p99_ms": percentile(values, 0.99),
                    "max_ms": max(values),
                }
            )

        for cohort_label, calls in cohort_calls:
            for dimension, classifier in (
                ("first_effective_command_head", effective_command_head),
                ("content_category", exec_content_category),
                ("operator_pattern", shell_structure),
                ("command_parse_status", command_parse_status),
            ):
                grouped: dict[str, list[ToolCall]] = defaultdict(list)
                for call in calls:
                    grouped[classifier(call.exec_command)].append(call)
                for category, selected in sorted(grouped.items()):
                    write_row(
                        cohort_label,
                        dimension,
                        category,
                        selected,
                        separately_plotted=(
                            category in displayed_set or category == "unclassified"
                            if dimension == "content_category"
                            else dimension == "operator_pattern"
                        ),
                    )

            other_calls = [
                call
                for call in calls
                if exec_content_category(call.exec_command) not in displayed_set
            ]
            if other_calls:
                write_row(
                    cohort_label,
                    "content_category_aggregate",
                    "other",
                    other_calls,
                    separately_plotted=True,
                )
            write_row(
                cohort_label,
                "total",
                "all_exec",
                calls,
                separately_plotted=True,
            )


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_status() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.splitlines() if result.returncode == 0 else []


def _write_cpu_exec_audit(cohorts: Sequence[CohortData], output_dir: Path) -> None:
    fieldnames = (
        "cohort",
        "task_id",
        "tool_name",
        "latency_ms",
        "ts_start",
        "ts_end",
        "cpu_percent",
        "cpu_exclusion_reason",
    )
    with (output_dir / "cpu_exec_call_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cohort in cohorts:
            for call in cohort.calls:
                if call.tool_name != "exec":
                    continue
                writer.writerow({field: getattr(call, field) for field in fieldnames})


def write_summaries(
    cohorts: Sequence[CohortData],
    *,
    output_dir: Path,
    cpu_min_duration_s: float,
    histogram_edges_log1p_ms: Sequence[float],
    displayed_content_categories: Sequence[str],
    content_category_counts: Counter[str],
) -> None:
    """Write tabular statistics and a complete cohort/provenance manifest."""

    script_snapshot = output_dir / "analysis_script_snapshot.py"
    shutil.copy2(Path(__file__), script_snapshot)
    script_snapshot_sha256 = _sha256(script_snapshot)
    rows = list(_summary_rows(cohorts))
    with (output_dir / "latency_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    cohort_payloads: list[dict[str, object]] = []
    for cohort in cohorts:
        exclusion_counts = Counter(
            call.cpu_exclusion_reason
            for call in cohort.calls
            if call.cpu_exclusion_reason is not None
        )
        exec_calls = [call for call in cohort.calls if call.tool_name == "exec"]
        exec_exclusion_counts = Counter(
            call.cpu_exclusion_reason
            for call in exec_calls
            if call.cpu_exclusion_reason is not None
        )
        gaps = cohort.resource_sample_gaps_s
        gap_summary = {
            "count": len(gaps),
            "p50": percentile(gaps, 0.50) if gaps else None,
            "p95": percentile(gaps, 0.95) if gaps else None,
            "max": max(gaps) if gaps else None,
        }
        cohort_payloads.append(
            {
                "label": cohort.spec.label,
                "roots": [str(root.resolve()) for root in cohort.spec.roots],
                "expected_trace_count": cohort.spec.expected_trace_count,
                "corpus_sha256": _cohort_sha256(cohort.traces),
                "trace_count": len(cohort.traces),
                "contributing_trace_count": cohort.contributing_task_count,
                "zero_call_trace_count": len(cohort.traces)
                - cohort.contributing_task_count,
                "resource_trace_count": cohort.resource_trace_count,
                "tool_call_count": len(cohort.calls),
                "excluded_tool_counts": dict(cohort.excluded_tool_counts),
                "benchmark": cohort.traces[0].benchmark,
                "model": cohort.traces[0].model,
                "scaffold": cohort.traces[0].scaffold,
                "tool_execution_environment": (
                    cohort.traces[0].tool_execution_environment
                ),
                "tool_runtime": cohort.traces[0].tool_runtime,
                "resource_sample_gap_seconds": gap_summary,
                "cpu_alignment_exclusion_counts_all_tools": dict(
                    sorted(exclusion_counts.items())
                ),
                "cpu_alignment_exclusion_counts_exec": dict(
                    sorted(exec_exclusion_counts.items())
                ),
                "cpu_aligned_call_count": sum(
                    call.cpu_percent is not None for call in cohort.calls
                ),
                "cpu_aligned_exec_call_count": sum(
                    call.cpu_percent is not None for call in exec_calls
                ),
                "traces": [
                    {
                        "path": str(trace.path),
                        "task_id": trace.task_id,
                        "sha256": trace.sha256,
                        "resources_path": (
                            str(trace.resources_path) if trace.resources_path else None
                        ),
                        "resources_sha256": trace.resources_sha256,
                    }
                    for trace in cohort.traces
                ],
            }
        )
    all_exec_calls = [
        call
        for cohort in cohorts
        for call in cohort.calls
        if call.tool_name == "exec"
    ]
    parse_status_counts = Counter(
        command_parse_status(call.exec_command) for call in all_exec_calls
    )
    operator_pattern_counts = Counter(
        shell_structure(call.exec_command) for call in all_exec_calls
    )
    metadata = {
        "cohorts": cohort_payloads,
        "latency_source": "trace.jsonl tool_exec ts_end - ts_start",
        "histogram": {
            "binning": "numpy histogram_bin_edges with method='fd' on log1p milliseconds",
            "edges_log1p_ms": [float(edge) for edge in histogram_edges_log1p_ms],
            "series_normalization": "each displayed series has total weight 1",
            "numpy_version": np.__version__,
        },
        "exec_command_classification": {
            "head_extractor": (
                "first non-cd top-level command word from the bashlex AST; "
                "leading environment assignments are skipped"
            ),
            "attribution": (
                "first-head-only: the full exec-call latency is assigned to the "
                "first effective head; later commands receive no attribution"
            ),
            "pytest_category_rule": (
                "direct pytest executables and first-command Python '-m pytest' "
                "entry points are grouped as pytest"
            ),
            "attribution_limitation": (
                "multi-command latency may be dominated by a later command; use "
                "operator_pattern and full CSV summaries when interpreting heads"
            ),
            "operator_pattern": (
                "no simple operator | sequential operators | "
                "pipeline/background operators | mixed operators | unclassified"
            ),
            "operator_pattern_scope": (
                "bashlex top-level AST classification; leading cd segments are "
                "ignored, and operators nested in words, heredocs, command "
                "substitutions, process substitutions, or escaped arguments do "
                "not affect the outer call pattern; unsupported AST roots are "
                "unclassified"
            ),
            "parser": f"bashlex {package_version('bashlex')}",
            "parse_status_counts": dict(parse_status_counts.most_common()),
            "operator_pattern_counts": dict(operator_pattern_counts.most_common()),
            "plot_minimum_category_calls": MIN_CONTENT_CATEGORY_PLOT_CALLS,
            "plot_threshold_rationale": (
                f"{MIN_CONTENT_CATEGORY_PLOT_CALLS} calls gives "
                f"{1 / MIN_CONTENT_CATEGORY_PLOT_CALLS:.3f} empirical probability "
                "resolution; all categories remain available in "
                "exec_command_summary.csv"
            ),
            "plot_kind": "empirical CDF with a shared latency domain",
            "displayed_categories": list(displayed_content_categories),
            "all_content_category_counts": dict(
                content_category_counts.most_common()
            ),
        },
        "cpu_method": {
            "sample_source": "resources.json samples",
            "minimum_call_duration_seconds": cpu_min_duration_s,
            "minimum_strictly_interior_samples": CPU_MIN_INTERIOR_SAMPLES,
            "expected_sample_interval_seconds": EXPECTED_RESOURCE_INTERVAL_S,
            "maximum_allowed_gap_seconds": CPU_MAX_RESOURCE_GAP_S,
            "aggregation": "linear interpolation and overlap-weighted trapezoid mean",
            "plotting": "not plotted; retained in cpu_exec_call_audit.csv only",
            "interpretation": "exploratory secondary audit data only",
            "sampler_scope": "not encoded in resources.json; collector implementation samples AttemptContext.container_id",
            "scope_limitation": "artifact-level sampler/container identity cannot be independently verified retrospectively",
        },
        "provenance": {
            "command": sys.argv,
            "git_revision": _git_revision(),
            "git_tracked_status_short": _git_status(),
            "analysis_script": str(Path(__file__).resolve()),
            "analysis_script_snapshot": "analysis_script_snapshot.py",
            "analysis_script_sha256": script_snapshot_sha256,
            "python_version": platform.python_version(),
            "matplotlib_version": matplotlib.__version__,
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_cpu_exec_audit(cohorts, output_dir)
    write_exec_command_summary(
        cohorts,
        output_dir=output_dir,
        displayed_categories=displayed_content_categories,
    )


def _prepare_output_dir(path: Path) -> Path:
    if path.exists():
        if any(path.iterdir()):
            raise ValueError(f"output directory must be empty: {path}")
        path.rmdir()
    staging = path.with_name(f".{path.name}.partial")
    if staging.exists():
        raise ValueError(f"staging output already exists: {staging}")
    staging.mkdir(parents=True)
    return staging


def main() -> None:
    """Run extraction, validation, plotting, and summary generation."""

    args = build_parser().parse_args()
    if not math.isfinite(args.cpu_min_duration_s):
        raise ValueError("--cpu-min-duration-s must be finite")
    if args.cpu_min_duration_s < CPU_MIN_CALL_DURATION_S:
        raise ValueError(
            f"--cpu-min-duration-s must be at least {CPU_MIN_CALL_DURATION_S:g}"
        )
    specs = parse_cohort_specs(args.cohort)

    cohorts = [
        load_cohort(spec, cpu_min_duration_s=args.cpu_min_duration_s)
        for spec in specs
    ]
    all_calls = [call for cohort in cohorts for call in cohort.calls]
    if not all_calls:
        raise ValueError("the validated cohorts contain no tool calls")
    maximum_latency_ms = max(call.latency_ms for call in all_calls)
    staging_dir = _prepare_output_dir(args.output_dir)
    try:
        histogram_edges = plot_histogram_overview(
            cohorts,
            output_dir=staging_dir,
            maximum_latency_ms=maximum_latency_ms,
        )
        displayed_categories, content_category_counts = plot_cdf_overview(
            cohorts,
            output_dir=staging_dir,
            maximum_latency_ms=maximum_latency_ms,
        )
        write_summaries(
            cohorts,
            output_dir=staging_dir,
            cpu_min_duration_s=args.cpu_min_duration_s,
            histogram_edges_log1p_ms=histogram_edges,
            displayed_content_categories=displayed_categories,
            content_category_counts=content_category_counts,
        )
        staging_dir.rename(args.output_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    print(
        f"Wrote figures and summaries for {len(cohorts)} validated cohorts to "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
