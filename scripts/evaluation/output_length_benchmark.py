#!/usr/bin/env python3
"""Build and evaluate next-turn LLM output-length prediction datasets."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Iterator, Sequence

import httpx

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from trace_collect.trace_data import TraceData  # noqa: E402


SCHEMA_VERSION = 1
SPLITS = ("train", "validation", "test")
_RESERVED_REQUEST_OPTIONS = {
    "ignore_eos",
    "max_tokens",
    "max_completion_tokens",
    "messages",
    "min_tokens",
    "model",
    "n",
    "seed",
    "stream",
    "temperature",
    "top_p",
}


def _read_json_object(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected a JSON object in {path}:{lineno}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_natural_labels(
    dataset_dir: Path,
    labels_path: Path,
    *,
    require_stochastic: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    protocol_path = labels_path.parent / "protocol.json"
    if not protocol_path.is_file():
        raise ValueError(f"natural-label protocol is missing: {protocol_path}")
    protocol = _read_json_object(protocol_path)
    dataset = _read_json_object(dataset_dir / "dataset.json")
    draws = protocol.get("draws")
    temperature = protocol.get("temperature")
    splits = protocol.get("splits")
    if (
        protocol.get("schema_version") != SCHEMA_VERSION
        or protocol.get("dataset_protocol") != dataset
        or protocol.get("natural_termination_required") is not True
        or not isinstance(protocol.get("model"), str)
        or not protocol["model"]
        or not isinstance(draws, int)
        or isinstance(draws, bool)
        or draws <= 0
        or not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or not math.isfinite(float(temperature))
        or float(temperature) < 0
        or not isinstance(splits, list)
        or not splits
        or any(split not in SPLITS for split in splits)
    ):
        raise ValueError(
            "label protocol does not match the dataset/natural-label contract"
        )
    if require_stochastic and (draws < 2 or float(temperature) <= 0):
        raise ValueError("TIE requires at least two stochastic natural-label draws")
    rows = _read_jsonl(labels_path)
    if any(
        row.get("label_source") != "target_natural_generation"
        or row.get("finish_reason") not in {"stop", "tool_calls"}
        for row in rows
    ):
        raise ValueError("labels must be uncensored target natural generations")
    target_ids = {
        str(row["sample_id"])
        for row in _read_jsonl(dataset_dir / "prefixes.jsonl")
        if row.get("split") in splits
    }
    keys = [(str(row.get("sample_id")), row.get("draw_id")) for row in rows]
    expected_keys = {
        (sample_id, draw_id) for sample_id in target_ids for draw_id in range(draws)
    }
    if len(keys) != len(set(keys)) or set(keys) != expected_keys:
        raise ValueError("natural-label coverage does not match its protocol")
    return rows, protocol


@contextmanager
def _trace_paths(sources: Sequence[Path]) -> Iterator[list[Path]]:
    """Materialize trace JSONL files from supported local archives."""
    with ExitStack() as stack:
        paths: list[Path] = []
        for source in sources:
            if source.is_dir():
                paths.extend(source.rglob("trace.jsonl"))
                paths.extend(source.rglob("*.trace.jsonl"))
                continue
            if source.suffix == ".jsonl":
                paths.append(source)
                continue
            if source.name.endswith((".tar.zst", ".tar.gz", ".tgz")):
                temp_root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
                command = ["tar"]
                if source.name.endswith(".tar.zst"):
                    command.append("--zstd")
                command.extend(
                    [
                        "-xf",
                        str(source),
                        "-C",
                        str(temp_root),
                        "--wildcards",
                        "*trace.jsonl",
                    ]
                )
                completed = subprocess.run(command, capture_output=True, text=True)
                if completed.returncode:
                    raise RuntimeError(
                        f"failed to extract trace files from {source}: "
                        f"{completed.stderr.strip()}"
                    )
                paths.extend(temp_root.rglob("trace.jsonl"))
                paths.extend(temp_root.rglob("*.trace.jsonl"))
                continue
            raise ValueError(f"unsupported trace source: {source}")

        unique = sorted({path.resolve() for path in paths})
        if not unique:
            raise ValueError("trace sources contain no trace JSONL files")
        yield unique


def _sample_positions(count: int, steps_per_session: int | None) -> list[int]:
    if steps_per_session is not None and steps_per_session < 2:
        raise ValueError("steps_per_session must be at least 2")
    if steps_per_session is None or count <= steps_per_session:
        return list(range(count))
    return sorted(
        {
            round(index * (count - 1) / (steps_per_session - 1))
            for index in range(steps_per_session)
        }
    )


def _source_finish_reason(data: dict[str, Any]) -> str | None:
    response = data.get("raw_response")
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    reason = choices[0].get("finish_reason")
    return str(reason) if reason is not None else None


def _split_by_session(
    session_ids: Sequence[str],
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, str]:
    if train_fraction < 0 or validation_fraction < 0:
        raise ValueError("split fractions must be non-negative")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1")
    shuffled = sorted(set(session_ids))
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_fraction)
    validation_end = train_end + int(len(shuffled) * validation_fraction)
    return {
        session_id: (
            "train"
            if index < train_end
            else "validation"
            if index < validation_end
            else "test"
        )
        for index, session_id in enumerate(shuffled)
    }


def export_dataset(
    sources: Sequence[Path],
    output_dir: Path,
    *,
    request_options_path: Path | None,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    steps_per_session: int | None = None,
    skip_invalid_source_actions: bool = False,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    request_options = _read_json_object(request_options_path)
    forbidden = sorted(_RESERVED_REQUEST_OPTIONS & request_options.keys())
    if forbidden:
        raise ValueError(f"request options contain controlled keys: {forbidden}")

    prefixes: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    excluded_invalid_source_actions = 0
    trajectory_counts: dict[str, int] = {}
    with _trace_paths(sources) as trace_paths:
        for trace_path in trace_paths:
            trace = TraceData.load(trace_path)
            session_id = str(trace.metadata.get("instance_id") or "")
            source_model = str(
                trace.metadata.get("source_model") or trace.metadata.get("model") or ""
            )
            if not session_id or not source_model:
                raise ValueError(
                    f"trace metadata lacks instance_id/model or source_model: {trace_path}"
                )
            benchmark = str(trace.metadata.get("benchmark") or "unknown")
            trajectory_counts[benchmark] = trajectory_counts.get(benchmark, 0) + 1
            llm_actions = [
                (ordinal, action)
                for ordinal, action in enumerate(trace.actions)
                if action.get("action_type") == "llm_call"
            ]
            for position in _sample_positions(len(llm_actions), steps_per_session):
                ordinal, action = llm_actions[position]
                data = action.get("data")
                if not isinstance(data, dict):
                    raise ValueError(f"LLM action has no data object: {trace_path}")
                messages = data.get("messages_in")
                if not isinstance(messages, list):
                    raise ValueError(
                        f"LLM action has no messages_in list: {trace_path}"
                    )
                actual_tokens = data.get("completion_tokens")
                if (
                    not isinstance(actual_tokens, int)
                    or isinstance(actual_tokens, bool)
                    or actual_tokens <= 0
                ):
                    if skip_invalid_source_actions:
                        excluded_invalid_source_actions += 1
                        continue
                    raise ValueError(
                        f"LLM action has invalid completion_tokens in {trace_path}: "
                        f"{actual_tokens!r}"
                    )
                agent_id = str(action.get("agent_id") or session_id)
                action_id = str(action.get("action_id") or f"llm_{ordinal}")
                sample_id = "/".join((source_model, session_id, agent_id, action_id))
                prefixes.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "sample_id": sample_id,
                        "session_id": session_id,
                        "agent_id": agent_id,
                        "action_id": action_id,
                        "step_index": int(action.get("iteration", ordinal)),
                        "source_model": source_model,
                        "benchmark": benchmark,
                        "messages": messages,
                    }
                )
                labels.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "sample_id": sample_id,
                        "draw_id": 0,
                        "actual_tokens": actual_tokens,
                        "finish_reason": _source_finish_reason(data),
                        "label_source": "recorded_trace",
                    }
                )

    sample_ids = [row["sample_id"] for row in prefixes]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("trace sources produce duplicate sample_id values")
    if not prefixes:
        raise ValueError("trace sources contain no LLM actions")

    session_splits = _split_by_session(
        [row["session_id"] for row in prefixes],
        seed=seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    for row in prefixes:
        row["split"] = session_splits[row["session_id"]]
    prefixes.sort(
        key=lambda row: (
            row["session_id"],
            row["agent_id"],
            row["step_index"],
            row["action_id"],
        )
    )
    label_by_id = {row["sample_id"]: row for row in labels}
    labels = [label_by_id[row["sample_id"]] for row in prefixes]

    split_counts = {
        split: sum(row["split"] == split for row in prefixes) for split in SPLITS
    }
    session_counts = {
        split: sum(value == split for value in session_splits.values())
        for split in SPLITS
    }
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "task": "next_turn_output_length_prediction",
        "input_semantics": "full_messages_visible_before_the_next_assistant_turn",
        "source_paths": [str(path.resolve()) for path in sources],
        "source_models": sorted({row["source_model"] for row in prefixes}),
        "sampling": {
            "steps_per_session": steps_per_session,
            "positions": (
                "all_llm_calls"
                if steps_per_session is None
                else "evenly_spaced_including_first_and_last"
            ),
            "invalid_source_actions": (
                "excluded" if skip_invalid_source_actions else "fail"
            ),
            "excluded_invalid_source_actions": excluded_invalid_source_actions,
        },
        "request_options": request_options,
        "prediction_contract": {
            "format": "jsonl",
            "required_fields": ["sample_id", "predicted_tokens"],
            "predicted_tokens": "positive_finite_number",
            "coverage": "exactly_once_for_every_sample_in_the_evaluated_split",
        },
        "split": {
            "unit": "session_id",
            "seed": seed,
            "train_fraction": train_fraction,
            "validation_fraction": validation_fraction,
            "test_fraction": 1 - train_fraction - validation_fraction,
        },
        "counts": {
            "samples": len(prefixes),
            "sessions": len(session_splits),
            "trajectories": sum(trajectory_counts.values()),
            "trajectories_by_benchmark": trajectory_counts,
            "samples_by_benchmark": {
                benchmark: sum(row["benchmark"] == benchmark for row in prefixes)
                for benchmark in sorted({row["benchmark"] for row in prefixes})
            },
            "samples_by_source_model": {
                model: sum(row["source_model"] == model for row in prefixes)
                for model in sorted({row["source_model"] for row in prefixes})
            },
            "samples_by_split": split_counts,
            "sessions_by_split": session_counts,
        },
    }
    output_dir.mkdir(parents=True)
    _write_jsonl(output_dir / "prefixes.jsonl", prefixes)
    _write_jsonl(output_dir / "source_labels.jsonl", labels)
    (output_dir / "dataset.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return protocol


def _parse_splits(value: str) -> tuple[str, ...]:
    splits = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = sorted(set(splits) - set(SPLITS))
    if not splits or invalid:
        raise ValueError(f"invalid splits: {invalid or value!r}")
    return splits


def _existing_label_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, int]] = set()
    for row in _read_jsonl(path):
        key = (str(row.get("sample_id")), int(row.get("draw_id", -1)))
        if key in keys:
            raise ValueError(f"duplicate label row: {key}")
        keys.add(key)
    return keys


def label_dataset(
    dataset_dir: Path,
    output_dir: Path,
    *,
    api_base: str,
    model: str,
    api_key_env: str,
    max_tokens: int,
    temperature: float,
    top_p: float | None,
    seed: int,
    draws: int,
    splits: Sequence[str],
    timeout_s: float,
    concurrency: int = 1,
) -> dict[str, Any]:
    if (
        max_tokens <= 0
        or draws <= 0
        or concurrency <= 0
        or timeout_s <= 0
        or not math.isfinite(temperature)
        or temperature < 0
        or top_p is not None
        and (not math.isfinite(top_p) or not 0 < top_p <= 1)
    ):
        raise ValueError("max_tokens, draws, and timeout_s must be positive")
    dataset = _read_json_object(dataset_dir / "dataset.json")
    prefixes = [
        row
        for row in _read_jsonl(dataset_dir / "prefixes.jsonl")
        if row.get("split") in splits
    ]
    if not prefixes:
        raise ValueError("selected splits contain no prefixes")
    request_options = dataset.get("request_options") or {}
    if not isinstance(request_options, dict):
        raise ValueError("dataset request_options must be an object")
    forbidden = sorted(_RESERVED_REQUEST_OPTIONS & request_options.keys())
    if forbidden:
        raise ValueError(
            f"dataset request options contain controlled keys: {forbidden}"
        )

    protocol = {
        "schema_version": SCHEMA_VERSION,
        "dataset_dir": str(dataset_dir.resolve()),
        "dataset_protocol": dataset,
        "api_base": api_base.rstrip("/"),
        "model": model,
        "api_key_env": api_key_env,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "draws": draws,
        "concurrency": concurrency,
        "splits": list(splits),
        "natural_termination_required": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "protocol.json"
    if protocol_path.exists():
        if _read_json_object(protocol_path) != protocol:
            raise ValueError("existing label protocol does not match requested run")
    else:
        protocol_path.write_text(
            json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    labels_path = output_dir / "labels.jsonl"
    completed_keys = _existing_label_keys(labels_path)
    total = len(prefixes) * draws
    completed_count = len(completed_keys)
    api_key = os.environ.get(api_key_env) if api_key_env else None
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    endpoint = f"{api_base.rstrip('/')}/chat/completions"
    with httpx.Client(timeout=timeout_s, trust_env=False, headers=headers) as client:

        def fetch(prefix: dict[str, Any], draw_id: int) -> dict[str, Any]:
            sample_id = str(prefix["sample_id"])
            body = {
                **request_options,
                "model": model,
                "messages": prefix["messages"],
                "stream": False,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "seed": seed + draw_id,
            }
            if top_p is not None:
                body["top_p"] = top_p
            response = client.post(endpoint, json=body)
            response.raise_for_status()
            payload = response.json()
            choices = payload.get("choices")
            usage = payload.get("usage")
            if (
                not isinstance(choices, list)
                or not choices
                or not isinstance(choices[0], dict)
                or not isinstance(usage, dict)
            ):
                raise ValueError(f"invalid completion response for {sample_id}")
            finish_reason = choices[0].get("finish_reason")
            if finish_reason == "length":
                raise RuntimeError(
                    f"natural label for {sample_id} hit max_tokens={max_tokens}"
                )
            if finish_reason not in {"stop", "tool_calls"}:
                raise RuntimeError(
                    f"unnatural finish_reason for {sample_id}: {finish_reason!r}"
                )
            actual_tokens = usage.get("completion_tokens")
            if (
                not isinstance(actual_tokens, int)
                or isinstance(actual_tokens, bool)
                or actual_tokens <= 0
            ):
                raise ValueError(
                    f"invalid completion_tokens for {sample_id}: {actual_tokens!r}"
                )
            return {
                "schema_version": SCHEMA_VERSION,
                "sample_id": sample_id,
                "draw_id": draw_id,
                "actual_tokens": actual_tokens,
                "prompt_tokens": usage.get("prompt_tokens"),
                "finish_reason": finish_reason,
                "label_source": "target_natural_generation",
                "response_id": payload.get("id"),
                "response_model": payload.get("model"),
                "message": choices[0].get("message"),
            }

        with labels_path.open("a", encoding="utf-8") as output:
            for prefix in prefixes:
                sample_id = str(prefix["sample_id"])
                messages = prefix.get("messages")
                if not isinstance(messages, list):
                    raise ValueError(f"prefix {sample_id} has no messages list")
                pending = [
                    draw_id
                    for draw_id in range(draws)
                    if (sample_id, draw_id) not in completed_keys
                ]
                if not pending:
                    continue
                # Warm the shared prefix before decoding the remaining draws in parallel.
                rows = [fetch(prefix, pending[0])]
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    rows.extend(
                        executor.map(
                            lambda draw_id: fetch(prefix, draw_id), pending[1:]
                        )
                    )
                for row in rows:
                    key = (sample_id, row["draw_id"])
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                    completed_keys.add(key)
                    completed_count += 1
                    if completed_count % 100 == 0 or completed_count == total:
                        print(f"labeled {completed_count}/{total}", flush=True)
    return {**protocol, "completed_labels": len(completed_keys)}


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile of no values")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _prediction_argument(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("prediction must be NAME=PATH")
    return name, Path(raw_path)


def evaluate_predictions(
    dataset_dir: Path,
    labels_path: Path,
    predictions: Sequence[tuple[str, Path]],
    *,
    split: str,
) -> dict[str, Any]:
    if split not in SPLITS:
        raise ValueError(f"invalid split: {split}")
    prefixes = _read_jsonl(dataset_dir / "prefixes.jsonl")
    target_rows = [row for row in prefixes if row.get("split") == split]
    target_ids = {str(row["sample_id"]) for row in target_rows}
    if not target_ids:
        raise ValueError(f"dataset split {split!r} contains no samples")

    label_rows, label_protocol = _read_natural_labels(dataset_dir, labels_path)
    if split not in label_protocol.get("splits", []):
        raise ValueError(f"label protocol does not cover split {split!r}")
    labels_by_id: dict[str, list[int]] = {sample_id: [] for sample_id in target_ids}
    draw_ids_by_id: dict[str, set[int]] = {sample_id: set() for sample_id in target_ids}
    seen_label_keys: set[tuple[str, int]] = set()
    for row in label_rows:
        sample_id = str(row.get("sample_id"))
        if sample_id not in target_ids:
            continue
        draw_id = row.get("draw_id")
        if not isinstance(draw_id, int) or isinstance(draw_id, bool) or draw_id < 0:
            raise ValueError(f"invalid draw_id for {sample_id}: {draw_id!r}")
        key = (sample_id, draw_id)
        if key in seen_label_keys:
            raise ValueError(f"duplicate label row: {key}")
        seen_label_keys.add(key)
        actual = row.get("actual_tokens")
        if not isinstance(actual, int) or isinstance(actual, bool) or actual <= 0:
            raise ValueError(f"invalid actual_tokens for {key}: {actual!r}")
        labels_by_id[sample_id].append(actual)
        draw_ids_by_id[sample_id].add(draw_id)
    missing_labels = sorted(
        sample_id for sample_id, values in labels_by_id.items() if not values
    )
    if missing_labels:
        raise ValueError(f"labels are missing {len(missing_labels)} test samples")
    expected_draw_ids = next(iter(draw_ids_by_id.values()))
    uneven_draws = [
        sample_id
        for sample_id, draw_ids in draw_ids_by_id.items()
        if draw_ids != expected_draw_ids
    ]
    if uneven_draws:
        raise ValueError(
            f"labels have inconsistent draw coverage for {len(uneven_draws)} samples"
        )

    method_names = [name for name, _path in predictions]
    if len(method_names) != len(set(method_names)):
        raise ValueError("prediction method names must be unique")
    methods: dict[str, Any] = {}
    for name, path in predictions:
        predicted_by_id: dict[str, float] = {}
        for row in _read_jsonl(path):
            sample_id = str(row.get("sample_id"))
            if sample_id in predicted_by_id:
                raise ValueError(f"duplicate prediction for {name}: {sample_id}")
            predicted = row.get("predicted_tokens")
            if (
                not isinstance(predicted, (int, float))
                or isinstance(predicted, bool)
                or not math.isfinite(float(predicted))
                or float(predicted) <= 0
            ):
                raise ValueError(
                    f"invalid predicted_tokens for {name}/{sample_id}: {predicted!r}"
                )
            predicted_by_id[sample_id] = float(predicted)
        predicted_ids = set(predicted_by_id)
        if predicted_ids != target_ids:
            raise ValueError(
                f"prediction coverage mismatch for {name}: "
                f"missing={len(target_ids - predicted_ids)}, "
                f"extra={len(predicted_ids - target_ids)}"
            )

        q_errors: list[float] = []
        absolute_errors: list[float] = []
        accuracies: list[float] = []
        underpredictions = 0
        for sample_id in sorted(target_ids):
            predicted = predicted_by_id[sample_id]
            for actual in labels_by_id[sample_id]:
                q_error = max(predicted / actual, actual / predicted)
                q_errors.append(q_error)
                accuracies.append(1 / q_error)
                absolute_errors.append(abs(predicted - actual))
                underpredictions += predicted < actual
        methods[name] = {
            "prediction_file": str(path.resolve()),
            "sample_count": len(target_ids),
            "labeled_draw_count": len(q_errors),
            "q_error": {
                "q50": _percentile(q_errors, 0.50),
                "q90": _percentile(q_errors, 0.90),
                "q95": _percentile(q_errors, 0.95),
                "q99": _percentile(q_errors, 0.99),
                "mean": statistics.fmean(q_errors),
            },
            "mean_accuracy": statistics.fmean(accuracies),
            "mae_tokens": statistics.fmean(absolute_errors),
            "underprediction_rate": underpredictions / len(q_errors),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "task": "next_turn_output_length_prediction",
        "dataset_dir": str(dataset_dir.resolve()),
        "labels_path": str(labels_path.resolve()),
        "split": split,
        "sample_count": len(target_ids),
        "label_semantics": "one observation per sample_id/draw_id",
        "mean_accuracy_semantics": "mean(min(predicted,actual)/max(predicted,actual))",
        "methods": methods,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export", help="export full step prefixes from traces")
    export.add_argument("sources", type=Path, nargs="+")
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--request-options", type=Path)
    export.add_argument("--seed", type=int, default=42)
    export.add_argument("--train-fraction", type=float, default=0.7)
    export.add_argument("--validation-fraction", type=float, default=0.1)
    export.add_argument("--steps-per-session", type=int)
    export.add_argument("--skip-invalid-source-actions", action="store_true")

    label = commands.add_parser("label", help="generate natural target-model labels")
    label.add_argument("--dataset-dir", type=Path, required=True)
    label.add_argument("--output-dir", type=Path, required=True)
    label.add_argument("--api-base", required=True)
    label.add_argument("--model", required=True)
    label.add_argument("--api-key-env", default="OPENAI_API_KEY")
    label.add_argument("--max-tokens", type=int, required=True)
    label.add_argument("--temperature", type=float, default=0.0)
    label.add_argument("--top-p", type=float)
    label.add_argument("--seed", type=int, default=42)
    label.add_argument("--draws", type=int, default=1)
    label.add_argument("--concurrency", type=int, default=1)
    label.add_argument("--splits", default=",".join(SPLITS))
    label.add_argument("--timeout-s", type=float, default=600.0)

    evaluate = commands.add_parser("evaluate", help="evaluate common point predictions")
    evaluate.add_argument("--dataset-dir", type=Path, required=True)
    evaluate.add_argument("--labels", type=Path, required=True)
    evaluate.add_argument(
        "--prediction",
        type=_prediction_argument,
        action="append",
        required=True,
        dest="predictions",
    )
    evaluate.add_argument("--split", choices=SPLITS, default="test")
    evaluate.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "export":
        result = export_dataset(
            args.sources,
            args.output_dir,
            request_options_path=args.request_options,
            seed=args.seed,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            steps_per_session=args.steps_per_session,
            skip_invalid_source_actions=args.skip_invalid_source_actions,
        )
        print(json.dumps(result["counts"], indent=2))
        return
    if args.command == "label":
        result = label_dataset(
            args.dataset_dir,
            args.output_dir,
            api_base=args.api_base,
            model=args.model,
            api_key_env=args.api_key_env,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            draws=args.draws,
            splits=_parse_splits(args.splits),
            timeout_s=args.timeout_s,
            concurrency=args.concurrency,
        )
        print(json.dumps({"completed_labels": result["completed_labels"]}, indent=2))
        return

    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    result = evaluate_predictions(
        args.dataset_dir,
        args.labels,
        args.predictions,
        split=args.split,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["methods"], indent=2))


if __name__ == "__main__":
    main()
