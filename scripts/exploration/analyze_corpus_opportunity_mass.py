#!/usr/bin/env python3
"""Descriptive sizing: KV-swap opportunity mass per trace corpus.

FREE, existing-data analysis. No collection, no API, no GPU. Compares how much
"swappable" tool-execution latency each corpus offers, to inform whether buying
more of a given benchmark's traces is worthwhile.

Corpora are supplied on the CLI (``--corpus LABEL=ROOT`` [+ ``--task-ids
LABEL=FILE``]); no corpus name or path is hardcoded in the logic. Each corpus is
loaded through the SAME production extractors the latency pipeline uses
(``discover_trace_files`` + ``extract_many_tool_latency_samples``), so the rows
are byte-identical to what the KV-swap machinery sees. Only ``tool_exec`` action
latencies are considered (that is all the extractor emits).

Metrics per corpus (see the task memo for definitions):
  1. call count, task count, total tool-time
  2. fraction of CALLS and of TOOL-TIME above each latency threshold
  3. tail shape: P50/P90/P99/max, and per-task max-call distribution
  4. heavy-verb mix: top verb classes by total time among calls over the heavy
     anchor (default 3500 ms)
  5. swappable mass per task = sum over the task's calls of (latency - T)+, at
     each cost anchor; mean and P90 across tasks
  6. extrapolation: how many tasks of each corpus reproduce another corpus's
     total heavy-call count (rough, clearly labelled)

Verb convention (item 4): the primary verb is the FIRST token of the LAST
sequential segment of the shell command (``shell_command_segments``), i.e. the
verb that actually runs after ``cd``/setup wrappers. Rows with no shell command
(non-``exec`` tools such as ``read_file``) or an untokenizable command are
classed by their ``tool_name`` so nothing is silently dropped.

Percentiles use numpy linear interpolation (``method="linear"``).

EXPLORATORY sizing only, not a paper result.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from trace_collect.command_features import shell_command_segments  # noqa: E402
from trace_collect.tool_gap_extractor import discover_trace_files  # noqa: E402
from trace_collect.tool_latency_dataset import (  # noqa: E402
    ToolLatencySample,
    extract_many_tool_latency_samples,
)

DEFAULT_THRESHOLDS_MS: tuple[float, ...] = (2000.0, 3500.0, 5000.0, 10000.0)
DEFAULT_HEAVY_ANCHOR_MS: float = 3500.0
DEFAULT_MASS_ANCHORS_MS: tuple[float, ...] = (3500.0, 5000.0)
TOP_VERB_CLASSES: int = 15


@dataclass(frozen=True)
class Call:
    """One tool-exec latency observation reduced to what sizing needs."""

    task_id: str
    latency_ms: float
    verb: str


def primary_verb(sample: ToolLatencySample, command_field: str = "command") -> str:
    """First token of the command's last sequential segment, else tool_name.

    ``cd build && make`` -> ``make``; ``pytest -q`` -> ``pytest``. Non-shell
    tools and untokenizable commands fall back to ``tool_name`` so every call
    keeps a class.
    """

    command = None
    if sample.tool_args is not None:
        value = sample.tool_args.get(command_field)
        if isinstance(value, str):
            command = value
    if command:
        segments = shell_command_segments(command)
        if segments and segments[-1]:
            return segments[-1][0]
    return sample.tool_name


def samples_to_calls(samples: list[ToolLatencySample]) -> list[Call]:
    return [
        Call(task_id=s.task_id, latency_ms=s.latency_ms, verb=primary_verb(s))
        for s in samples
    ]


def load_corpus_calls(
    root: Path, task_ids: set[str] | None = None
) -> tuple[list[Call], int, int]:
    """Return (calls, trace_file_count, zero_exec_trace_count).

    ``task_ids`` (when given) restricts to a frozen task set; a declared id with
    no matching trace is a hard error (the frozen manifest must be complete).
    """

    trace_files = discover_trace_files([root])
    samples = extract_many_tool_latency_samples(trace_files)
    if task_ids is not None:
        present = {s.task_id for s in samples}
        missing = task_ids - present
        if missing:
            raise ValueError(
                f"{root}: {len(missing)} declared task_ids absent from traces: "
                f"{sorted(missing)[:5]}"
            )
        samples = [s for s in samples if s.task_id in task_ids]
    calls = samples_to_calls(samples)
    tasks_with_calls = {c.task_id for c in calls}
    zero_exec = len(trace_files) - len(tasks_with_calls)
    return calls, len(trace_files), max(zero_exec, 0)


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q, method="linear"))


def swappable_mass_per_task(calls: list[Call], anchor_ms: float) -> dict[str, float]:
    """Sum of (latency - anchor)+ over each task's calls."""

    mass: dict[str, float] = defaultdict(float)
    for c in calls:
        excess = c.latency_ms - anchor_ms
        if excess > 0.0:
            mass[c.task_id] += excess
    # Tasks with no over-anchor call still exist as workload; include them at 0.
    for c in calls:
        mass.setdefault(c.task_id, 0.0)
    return dict(mass)


@dataclass
class CorpusStats:
    label: str
    root: str
    trace_files: int
    zero_exec_traces: int
    call_count: int
    task_count: int
    total_tool_time_ms: float
    threshold_call_frac: dict[str, float]
    threshold_time_frac: dict[str, float]
    tail: dict[str, float]
    per_task_maxcall: dict[str, float]
    heavy_verbs: list[dict[str, Any]]
    mass: dict[str, dict[str, float]]
    heavy_call_counts: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "root": self.root,
            "trace_files": self.trace_files,
            "zero_exec_traces": self.zero_exec_traces,
            "call_count": self.call_count,
            "task_count": self.task_count,
            "total_tool_time_ms": self.total_tool_time_ms,
            "threshold_call_frac": self.threshold_call_frac,
            "threshold_time_frac": self.threshold_time_frac,
            "tail_ms": self.tail,
            "per_task_maxcall_ms": self.per_task_maxcall,
            "heavy_verbs": self.heavy_verbs,
            "mass_ms": self.mass,
            "heavy_call_counts": self.heavy_call_counts,
        }


def analyze_corpus(
    label: str,
    root: Path,
    calls: list[Call],
    trace_files: int,
    zero_exec: int,
    *,
    thresholds: tuple[float, ...],
    heavy_anchor: float,
    mass_anchors: tuple[float, ...],
) -> CorpusStats:
    latencies = [c.latency_ms for c in calls]
    total_time = float(sum(latencies))
    tasks = sorted({c.task_id for c in calls})
    n = len(calls)

    call_frac: dict[str, float] = {}
    time_frac: dict[str, float] = {}
    for t in thresholds:
        over = [x for x in latencies if x > t]
        call_frac[str(int(t))] = (len(over) / n) if n else 0.0
        time_frac[str(int(t))] = (sum(over) / total_time) if total_time else 0.0

    tail = {
        "p50": _pct(latencies, 50),
        "p90": _pct(latencies, 90),
        "p99": _pct(latencies, 99),
        "max": float(max(latencies)) if latencies else 0.0,
    }

    per_task_max: dict[str, float] = defaultdict(float)
    for c in calls:
        per_task_max[c.task_id] = max(per_task_max[c.task_id], c.latency_ms)
    maxcalls = list(per_task_max.values())
    maxcall_stats = {
        "mean": float(np.mean(maxcalls)) if maxcalls else 0.0,
        "p50": _pct(maxcalls, 50),
        "p90": _pct(maxcalls, 90),
        "max": float(max(maxcalls)) if maxcalls else 0.0,
    }

    # Heavy-verb mix: total time among calls over the heavy anchor.
    verb_time: dict[str, float] = defaultdict(float)
    verb_calls: dict[str, int] = defaultdict(int)
    for c in calls:
        if c.latency_ms > heavy_anchor:
            verb_time[c.verb] += c.latency_ms
            verb_calls[c.verb] += 1
    heavy_verbs = [
        {"verb": v, "total_time_ms": verb_time[v], "n_calls": verb_calls[v]}
        for v in sorted(verb_time, key=lambda k: verb_time[k], reverse=True)
    ][:TOP_VERB_CLASSES]

    mass: dict[str, dict[str, float]] = {}
    for a in mass_anchors:
        per_task = swappable_mass_per_task(calls, a)
        vals = list(per_task.values())
        mass[str(int(a))] = {
            "total_ms": float(sum(vals)),
            "mean_per_task_ms": float(np.mean(vals)) if vals else 0.0,
            "p90_per_task_ms": _pct(vals, 90),
            "tasks_with_mass": int(sum(1 for v in vals if v > 0.0)),
        }

    heavy_call_counts = {
        str(int(t)): int(sum(1 for x in latencies if x > t)) for t in mass_anchors
    }

    return CorpusStats(
        label=label,
        root=str(root),
        trace_files=trace_files,
        zero_exec_traces=zero_exec,
        call_count=n,
        task_count=len(tasks),
        total_tool_time_ms=total_time,
        threshold_call_frac=call_frac,
        threshold_time_frac=time_frac,
        tail=tail,
        per_task_maxcall=maxcall_stats,
        heavy_verbs=heavy_verbs,
        mass=mass,
        heavy_call_counts=heavy_call_counts,
    )


def extrapolate(
    target: CorpusStats, source: CorpusStats, anchor_key: str
) -> dict[str, float]:
    """How many `source` tasks reproduce `target`'s total heavy-call count.

    Rough per-task-rate extrapolation: N = target_heavy_calls / (source
    heavy-calls per task). Assumes source tasks are exchangeable and drawn from
    the same distribution -- a sizing heuristic, not a guarantee.
    """

    src_heavy = source.heavy_call_counts[anchor_key]
    src_tasks = source.task_count
    rate = (src_heavy / src_tasks) if src_tasks else 0.0
    tgt_heavy = target.heavy_call_counts[anchor_key]
    n_needed = (tgt_heavy / rate) if rate else float("inf")
    return {
        "anchor_ms": float(anchor_key),
        "target_heavy_calls": tgt_heavy,
        "source_heavy_calls_per_task": rate,
        "source_tasks_needed": n_needed,
    }


def _ms_to_s(x: float) -> float:
    return x / 1000.0


def render_memo(
    corpora: list[CorpusStats],
    thresholds: tuple[float, ...],
    heavy_anchor: float,
    mass_anchors: tuple[float, ...],
    extraps: list[dict[str, Any]],
    git_sha: str,
) -> str:
    lines: list[str] = []
    lines.append("# Terminal-Bench vs SWE-rebench: KV-swap opportunity sizing")
    lines.append("")
    lines.append(
        "> **EXPLORATORY / sizing only — not a paper result.** Descriptive memo "
        "to inform whether to buy more Terminal-Bench trace collection. No new "
        "collection, no API, no GPU."
    )
    lines.append(">")
    lines.append(f"> git sha: `{git_sha}`")
    lines.append("")

    lines.append("## Corpora")
    lines.append("")
    lines.append("| corpus | root | trace files | tasks w/ exec | zero-exec tasks | exec calls |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for c in corpora:
        lines.append(
            f"| {c.label} | `{c.root}` | {c.trace_files} | {c.task_count} | "
            f"{c.zero_exec_traces} | {c.call_count} |"
        )
    lines.append("")

    # Headline comparison table.
    lines.append("## Headline comparison")
    lines.append("")
    hdr = ["metric"] + [c.label for c in corpora]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "---|" * len(hdr))

    def row(name: str, fn) -> None:
        lines.append("| " + name + " | " + " | ".join(fn(c) for c in corpora) + " |")

    row("exec calls", lambda c: f"{c.call_count}")
    row("tasks with exec", lambda c: f"{c.task_count}")
    row("total tool-time (s)", lambda c: f"{_ms_to_s(c.total_tool_time_ms):.0f}")
    row("calls / task (mean)", lambda c: f"{c.call_count / c.task_count:.1f}")
    row("P50 latency (ms)", lambda c: f"{c.tail['p50']:.0f}")
    row("P90 latency (ms)", lambda c: f"{c.tail['p90']:.0f}")
    row("P99 latency (ms)", lambda c: f"{c.tail['p99']:.0f}")
    row("max latency (ms)", lambda c: f"{c.tail['max']:.0f}")
    for t in thresholds:
        k = str(int(t))
        row(
            f"% calls > {int(t)}ms",
            lambda c, k=k: f"{100 * c.threshold_call_frac[k]:.2f}",
        )
    for t in thresholds:
        k = str(int(t))
        row(
            f"% tool-time > {int(t)}ms",
            lambda c, k=k: f"{100 * c.threshold_time_frac[k]:.1f}",
        )
    for a in mass_anchors:
        k = str(int(a))
        row(
            f"swap mass @ {int(a)}ms: mean/task (s)",
            lambda c, k=k: f"{_ms_to_s(c.mass[k]['mean_per_task_ms']):.2f}",
        )
        row(
            f"swap mass @ {int(a)}ms: P90/task (s)",
            lambda c, k=k: f"{_ms_to_s(c.mass[k]['p90_per_task_ms']):.2f}",
        )
        row(
            f"swap mass @ {int(a)}ms: total (s)",
            lambda c, k=k: f"{_ms_to_s(c.mass[k]['total_ms']):.1f}",
        )
        row(
            f"heavy calls > {int(a)}ms (count)",
            lambda c, k=k: f"{c.heavy_call_counts[k]}",
        )
    lines.append("")

    # Per-task max-call distribution.
    lines.append("## Per-task max-call latency (heaviest single call per task)")
    lines.append("")
    lines.append("| corpus | mean (ms) | P50 (ms) | P90 (ms) | max (ms) |")
    lines.append("|---|---:|---:|---:|---:|")
    for c in corpora:
        m = c.per_task_maxcall
        lines.append(
            f"| {c.label} | {m['mean']:.0f} | {m['p50']:.0f} | {m['p90']:.0f} | "
            f"{m['max']:.0f} |"
        )
    lines.append("")

    # Heavy-verb mix.
    lines.append(
        f"## Heavy-verb mix (top {TOP_VERB_CLASSES} by total time among calls "
        f"> {int(heavy_anchor)}ms)"
    )
    lines.append("")
    lines.append(
        "_Verb = first token of the last sequential command segment "
        "(post-`cd`/setup); non-shell tools classed by tool_name._"
    )
    lines.append("")
    for c in corpora:
        lines.append(f"### {c.label}")
        lines.append("")
        lines.append("| verb | total time (s) | n calls |")
        lines.append("|---|---:|---:|")
        for v in c.heavy_verbs:
            lines.append(
                f"| `{v['verb']}` | {_ms_to_s(v['total_time_ms']):.1f} | "
                f"{v['n_calls']} |"
            )
        lines.append("")

    # Extrapolation.
    lines.append("## Extrapolation: how many tasks buy the same heavy-call mass")
    lines.append("")
    lines.append(
        "_Rough per-task-rate extrapolation, clearly a heuristic: "
        "N = target heavy-call count / (source heavy-calls per task). Assumes "
        "source tasks are exchangeable with those already collected._"
    )
    lines.append("")
    lines.append(
        "| target | source | anchor | target heavy calls | source heavy/task | "
        "source tasks needed |"
    )
    lines.append("|---|---|---:|---:|---:|---:|")
    for e in extraps:
        n = e["source_tasks_needed"]
        n_str = "inf" if n == float("inf") else f"{n:.0f}"
        lines.append(
            f"| {e['target_label']} | {e['source_label']} | "
            f"{int(e['anchor_ms'])}ms | {e['target_heavy_calls']} | "
            f"{e['source_heavy_calls_per_task']:.2f} | {n_str} |"
        )
    lines.append("")

    return "\n".join(lines)


def _parse_kv(items: list[str], flag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"{flag} expects LABEL=VALUE, got {item!r}")
        label, value = item.split("=", 1)
        out[label.strip()] = value.strip()
    return out


def _git_sha() -> str:
    import subprocess

    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2]
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--corpus",
        action="append",
        required=True,
        metavar="LABEL=ROOT",
        help="Corpus label and trace root (repeatable).",
    )
    p.add_argument(
        "--task-ids",
        action="append",
        default=[],
        metavar="LABEL=FILE",
        help="Restrict a labelled corpus to task ids in FILE (repeatable).",
    )
    p.add_argument(
        "--thresholds",
        default=",".join(str(int(t)) for t in DEFAULT_THRESHOLDS_MS),
        help="Comma-separated latency thresholds in ms.",
    )
    p.add_argument("--heavy-anchor", type=float, default=DEFAULT_HEAVY_ANCHOR_MS)
    p.add_argument(
        "--mass-anchors",
        default=",".join(str(int(a)) for a in DEFAULT_MASS_ANCHORS_MS),
    )
    p.add_argument("--out-md", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    args = p.parse_args(argv)

    corpora_roots = _parse_kv(args.corpus, "--corpus")
    task_id_files = _parse_kv(args.task_ids, "--task-ids")
    thresholds = tuple(float(x) for x in args.thresholds.split(","))
    mass_anchors = tuple(float(x) for x in args.mass_anchors.split(","))

    stats: list[CorpusStats] = []
    for label, root in corpora_roots.items():
        task_ids = None
        if label in task_id_files:
            task_ids = {
                line.strip()
                for line in Path(task_id_files[label]).read_text().splitlines()
                if line.strip()
            }
        calls, trace_files, zero_exec = load_corpus_calls(Path(root), task_ids)
        print(
            f"[{label}] {trace_files} traces, {len(calls)} calls, "
            f"{zero_exec} zero-exec",
            file=sys.stderr,
        )
        stats.append(
            analyze_corpus(
                label,
                Path(root),
                calls,
                trace_files,
                zero_exec,
                thresholds=thresholds,
                heavy_anchor=args.heavy_anchor,
                mass_anchors=mass_anchors,
            )
        )

    # Extrapolate each corpus against every other (both directions).
    extraps: list[dict[str, Any]] = []
    for target in stats:
        for source in stats:
            if source.label == target.label:
                continue
            for a in mass_anchors:
                e = extrapolate(target, source, str(int(a)))
                e["target_label"] = target.label
                e["source_label"] = source.label
                extraps.append(e)

    git_sha = _git_sha()
    memo = render_memo(stats, thresholds, args.heavy_anchor, mass_anchors, extraps, git_sha)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(memo)
    payload = {
        "git_sha": git_sha,
        "thresholds_ms": list(thresholds),
        "heavy_anchor_ms": args.heavy_anchor,
        "mass_anchors_ms": list(mass_anchors),
        "corpora": [c.to_json() for c in stats],
        "extrapolation": extraps,
    }
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {args.out_md} and {args.out_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
