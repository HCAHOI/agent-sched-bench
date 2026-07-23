"""Per-call training examples for the BERT tool-resource predictor.

Strict causality: every feature of a call uses only information available at
that call's start. Earlier calls in the same trace have already completed, so
their observed durations and resource labels are legal inputs; the current
call's own window and any later call are never read.

The build here is pure Python (no torch) so causality can be unit-tested
cheaply. Tokenisation and tensorisation live in :mod:`tool_resource.bert_model`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Iterable, Sequence

from tool_resource.labels import load_resource_corpus

DEFAULT_TASKS_JSON = Path("data/swe-rebench/tasks.json")

# task_id -> repo strips the trailing "-<number>" instance suffix.
_REPO_SUFFIX_RE = re.compile(r"-\d+$")

_MAX_COMMAND_CHARS = 200

# All targets are trained in log space so pinball loss optimises geometric
# (relative) error. Eval exponentiates predicted quantiles back to original
# units. ambient_before_mb anchors memory as an INPUT feature, not in the target.
TARGET_NAMES: tuple[str, ...] = (
    "log_cpu_core_seconds",   # log1p(cpu_core_seconds), exec_timeline rows only
    "log_peak_cpu_cores",     # log1p(peak_cpu_cores), eligible rows only
    "log_peak_memory_mb",     # log(peak_memory_mb), rows with a window sample
)

# trained target -> (inverse transform to original units, unit label)
TARGET_INVERSE: dict[str, tuple[str, str]] = {
    "log_cpu_core_seconds": ("expm1", "core_s"),
    "log_peak_cpu_cores": ("expm1", "cores"),
    "log_peak_memory_mb": ("exp", "mb"),
}

# Scalar numeric features, in vector order; the tool one-hot is appended after.
SCALAR_FEATURE_NAMES: tuple[str, ...] = (
    "ambient_before_mb",
    "ambient_before_present",
    "ambient_before_age_s",
    "call_index",
    "elapsed_task_s",
    "running_max_memory_mb",
    "cum_prior_exec_core_s",
)


@dataclass(frozen=True)
class ResourceExample:
    """One training example for one tool call, with causal features only."""

    task_id: str
    repo: str
    source_trace: str
    call_index: int
    tool_name: str
    text: str
    numeric: tuple[float, ...]
    targets: dict[str, float]        # value or NaN where the target is masked
    target_mask: dict[str, bool]


@dataclass
class ResourceDataset:
    """A built corpus plus its tool vocabulary and repo-disjoint fold map."""

    examples: list[ResourceExample]
    tool_vocab: tuple[str, ...]
    feature_names: tuple[str, ...]
    repo_folds: dict[str, int]
    n_folds: int

    @property
    def numeric_dim(self) -> int:
        return len(self.feature_names)

    def fold_split(
        self, fold: int
    ) -> tuple[list[ResourceExample], list[ResourceExample]]:
        """Return (train, val) where val holds exactly the repos of ``fold``."""

        val_repos = {repo for repo, f in self.repo_folds.items() if f == fold}
        train = [ex for ex in self.examples if ex.repo not in val_repos]
        val = [ex for ex in self.examples if ex.repo in val_repos]
        return train, val


def repo_of(task_id: str) -> str:
    """Strip the trailing ``-<digits>`` instance suffix to get the repo key."""

    return _REPO_SUFFIX_RE.sub("", task_id)


def assign_repo_folds(
    repos: Iterable[str], n_folds: int, seed: int
) -> dict[str, int]:
    """Deterministically round-robin unique repos into ``n_folds`` folds."""

    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    unique = sorted(set(repos))
    rng = random.Random(seed)
    rng.shuffle(unique)
    return {repo: index % n_folds for index, repo in enumerate(unique)}


def load_task_prompts(tasks_json: Path = DEFAULT_TASKS_JSON) -> dict[str, str]:
    """Map instance_id -> problem_statement from the swe-rebench task file."""

    rows = json.loads(Path(tasks_json).read_text(encoding="utf-8"))
    return {row["instance_id"]: (row.get("problem_statement") or "") for row in rows}


def _command_text(sample: Any) -> str:
    args = getattr(sample, "tool_args", None)
    if sample.tool_name == "exec" and isinstance(args, dict):
        command = args.get("command")
        if isinstance(command, str) and command.strip():
            return command.strip()[:_MAX_COMMAND_CHARS]
    return sample.tool_name


def _observed_core_seconds(sample: Any) -> float | None:
    """Core-seconds legal to read once the call has completed."""

    if sample.cpu_core_seconds_kind in ("exec_timeline", "non_exec_zero"):
        return sample.cpu_core_seconds
    return None


def _one_hot(tool_name: str, tool_vocab: Sequence[str]) -> list[float]:
    return [1.0 if tool_name == name else 0.0 for name in tool_vocab]


def examples_for_sequence(
    task_id: str,
    source_trace: str,
    calls: Sequence[Any],
    prompt: str,
    tool_vocab: Sequence[str],
    n_prev: int,
) -> list[ResourceExample]:
    """Build examples for one causal unit (one trace), calls in time order.

    All running state (previous-command summaries, running-max memory,
    cumulative exec core-seconds) is updated *after* the current example is
    emitted, so the current call never sees its own window or any later call.
    """

    repo = repo_of(task_id)
    examples: list[ResourceExample] = []
    prev: list[tuple[str, float, float | None]] = []  # (command, dur_s, core_s|None)
    running_max_mem: float | None = None
    cum_exec_core_s = 0.0
    t0 = calls[0].tool_ts_start if calls else 0.0

    for index, sample in enumerate(calls):
        ambient_mb = getattr(sample, "ambient_before_mb", None)
        ambient_age = getattr(sample, "ambient_before_age_s", None)

        numeric = [
            float(ambient_mb) if ambient_mb is not None else 0.0,
            1.0 if ambient_mb is not None else 0.0,
            float(ambient_age) if ambient_age is not None else 0.0,
            float(index),
            float(sample.tool_ts_start - t0),
            running_max_mem if running_max_mem is not None else 0.0,
            cum_exec_core_s,
        ]
        numeric.extend(_one_hot(sample.tool_name, tool_vocab))

        text = _build_text(prompt, prev, _command_text(sample), n_prev)
        targets, masks = _targets(sample)

        examples.append(
            ResourceExample(
                task_id=task_id,
                repo=repo,
                source_trace=source_trace,
                call_index=index,
                tool_name=sample.tool_name,
                text=text,
                numeric=tuple(numeric),
                targets=targets,
                target_mask=masks,
            )
        )

        # --- advance causal state using the now-completed current call ---
        peak_mem = sample.peak_memory_mb
        if peak_mem is not None:
            running_max_mem = (
                peak_mem if running_max_mem is None else max(running_max_mem, peak_mem)
            )
        core_s = _observed_core_seconds(sample)
        if sample.cpu_core_seconds_kind == "exec_timeline" and core_s is not None:
            cum_exec_core_s += core_s
        prev.append(
            (_command_text(sample), sample.tool_ts_end - sample.tool_ts_start, core_s)
        )

    return examples


def _build_text(
    prompt: str,
    prev: Sequence[tuple[str, float, float | None]],
    current_command: str,
    n_prev: int,
) -> str:
    parts = []
    for command, dur_s, core_s in prev[-n_prev:]:
        core = "na" if core_s is None else f"{core_s:.3g}"
        parts.append(f"{command} (dur={dur_s:.1f}s core={core})")
    prev_block = " ; ".join(parts) if parts else "none"
    return f"{prompt}\n[PREV] {prev_block}\n[CUR] {current_command}"


def _targets(sample: Any) -> tuple[dict[str, float], dict[str, bool]]:
    targets = {name: math.nan for name in TARGET_NAMES}
    masks = {name: False for name in TARGET_NAMES}

    if (
        sample.cpu_core_seconds_kind == "exec_timeline"
        and sample.cpu_core_seconds is not None
    ):
        targets["log_cpu_core_seconds"] = math.log1p(sample.cpu_core_seconds)
        masks["log_cpu_core_seconds"] = True

    if sample.peak_cpu_cores_eligible and sample.peak_cpu_cores is not None:
        targets["log_peak_cpu_cores"] = math.log1p(sample.peak_cpu_cores)
        masks["log_peak_cpu_cores"] = True

    if sample.peak_memory_mb is not None and sample.peak_memory_mb > 0.0:
        targets["log_peak_memory_mb"] = math.log(sample.peak_memory_mb)
        masks["log_peak_memory_mb"] = True

    return targets, masks


def _build_examples(
    samples_by_task: dict[str, list[Any]],
    task_ids: Sequence[str],
    prompts: dict[str, str],
    tool_vocab: Sequence[str],
    n_prev: int,
) -> list[ResourceExample]:
    examples: list[ResourceExample] = []
    for task_id in task_ids:
        prompt = prompts.get(task_id, "")
        by_trace: dict[str, list[Any]] = defaultdict(list)
        for sample in samples_by_task[task_id]:
            by_trace[sample.source_trace].append(sample)
        for source_trace, calls in by_trace.items():
            calls = sorted(calls, key=lambda s: s.tool_ts_start)
            examples.extend(
                examples_for_sequence(
                    task_id, source_trace, calls, prompt, tool_vocab, n_prev
                )
            )
    return examples


def build_merged_dataset(
    corpus_pairs: Sequence[tuple[Path, Path]],
    *,
    tasks_json: Path = DEFAULT_TASKS_JSON,
    n_prev: int = 5,
    n_folds: int = 5,
    seed: int = 0,
    limit_tasks: int | None = None,
) -> ResourceDataset:
    """Merge one or more (trace_root, manifest) corpora into one dataset.

    Tool vocabulary and repo-disjoint folds are computed over the union, so a
    repo shared across corpora (36 of them between swe-100 and swe-277) never
    lands in both train and val. Task ids must be disjoint across corpora.
    """

    if not corpus_pairs:
        raise ValueError("corpus_pairs must be non-empty")
    prompts = load_task_prompts(tasks_json)

    merged_samples: dict[str, list[Any]] = {}
    merged_task_ids: list[str] = []
    for trace_root, manifest in corpus_pairs:
        samples_by_task, task_ids = load_resource_corpus(
            Path(trace_root), Path(manifest), limit_tasks=limit_tasks
        )
        duplicates = set(task_ids) & set(merged_task_ids)
        if duplicates:
            raise ValueError(
                f"task_id appears in multiple corpora: {sorted(duplicates)[:5]}"
            )
        merged_samples.update(samples_by_task)
        merged_task_ids.extend(task_ids)

    missing_prompts = [t for t in merged_task_ids if t not in prompts]
    if missing_prompts:
        # Prompts are expected for all swe-rebench tasks; surface the gap loudly
        # rather than silently training on empty text.
        raise ValueError(f"missing problem_statement for tasks: {missing_prompts[:5]}")

    tool_vocab = tuple(
        sorted({s.tool_name for samples in merged_samples.values() for s in samples})
    )
    examples = _build_examples(
        merged_samples, merged_task_ids, prompts, tool_vocab, n_prev
    )
    feature_names = SCALAR_FEATURE_NAMES + tuple(f"tool={t}" for t in tool_vocab)
    repo_folds = assign_repo_folds(
        (repo_of(t) for t in merged_task_ids), n_folds, seed
    )
    return ResourceDataset(examples, tool_vocab, feature_names, repo_folds, n_folds)


def build_dataset(
    trace_root: Path,
    task_manifest: Path,
    *,
    tasks_json: Path = DEFAULT_TASKS_JSON,
    n_prev: int = 5,
    n_folds: int = 5,
    seed: int = 0,
    limit_tasks: int | None = None,
) -> ResourceDataset:
    """Load a single corpus (thin wrapper over :func:`build_merged_dataset`)."""

    return build_merged_dataset(
        [(Path(trace_root), Path(task_manifest))],
        tasks_json=tasks_json,
        n_prev=n_prev,
        n_folds=n_folds,
        seed=seed,
        limit_tasks=limit_tasks,
    )


__all__ = [
    "TARGET_NAMES",
    "TARGET_INVERSE",
    "SCALAR_FEATURE_NAMES",
    "ResourceExample",
    "ResourceDataset",
    "repo_of",
    "assign_repo_folds",
    "load_task_prompts",
    "examples_for_sequence",
    "build_dataset",
    "build_merged_dataset",
]
