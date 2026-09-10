#!/usr/bin/env python3
"""Train output-length predictors against the shared benchmark contract."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
from typing import Any, Sequence

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from scripts.evaluation.output_length_benchmark import (
    _read_jsonl,
    _read_natural_labels,
    _write_jsonl,
)


SSJF_UPSTREAM_COMMIT = "4b866866a32626677a1841a3b92875b93b1f03ab"
EGTP_UPSTREAM_COMMIT = "170a47893e1351ecc062186aafc686ab3f8be3d1"
OUTLETS_UPSTREAM_COMMIT = "4b53761496da49ad9829a9817cbfa3c7c9047c52"
TIE_UPSTREAM_COMMIT = "ce6ddc4d7abe6a4a0821a03d463ad7af86113850"
TIE_DEGREES_OF_FREEDOM = 3.5

_OUTLETS_WORKER = r"""
import json
from pathlib import Path
import sys

package_dir, checkpoint, base_model, config, device, inputs, output = sys.argv[1:]
sys.path.insert(0, package_dir)
from inference_length import LengthPredictor

predictor = LengthPredictor(
    checkpoint_path=checkpoint,
    base_model_path=base_model,
    config_path=config,
    device=device,
    normalize_length=True,
    adapter=True,
    use_target_model=True,
)
with Path(inputs).open(encoding="utf-8") as source, Path(output).open(
    "w", encoding="utf-8"
) as target:
    for line in source:
        row = json.loads(line)
        result = predictor.predict(row["prompt"])
        target.write(json.dumps({
            "sample_id": row["sample_id"],
            "predicted_tokens": result["predicted_length"],
        }) + "\n")
"""


def _messages_text(messages: object) -> str:
    if not isinstance(messages, list):
        raise ValueError("prefix messages must be a list")
    return json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _point_examples(
    dataset_dir: Path, labels_path: Path
) -> dict[str, list[dict[str, Any]]]:
    prefixes = _read_jsonl(dataset_dir / "prefixes.jsonl")
    prefix_by_id = {str(row.get("sample_id")): row for row in prefixes}
    if len(prefix_by_id) != len(prefixes):
        raise ValueError("dataset contains duplicate sample_id values")

    labels_by_id: dict[str, list[int]] = {sample_id: [] for sample_id in prefix_by_id}
    draw_ids_by_id: dict[str, set[int]] = {
        sample_id: set() for sample_id in prefix_by_id
    }
    seen: set[tuple[str, int]] = set()
    label_rows, _protocol = _read_natural_labels(dataset_dir, labels_path)
    for row in label_rows:
        sample_id = str(row.get("sample_id"))
        if sample_id not in labels_by_id:
            continue
        draw_id = row.get("draw_id")
        actual = row.get("actual_tokens")
        if (
            not isinstance(draw_id, int)
            or isinstance(draw_id, bool)
            or draw_id < 0
            or not isinstance(actual, int)
            or isinstance(actual, bool)
            or actual <= 0
        ):
            raise ValueError(f"invalid label for {sample_id}")
        key = (sample_id, draw_id)
        if key in seen:
            raise ValueError(f"duplicate label row: {key}")
        seen.add(key)
        labels_by_id[sample_id].append(actual)
        draw_ids_by_id[sample_id].add(draw_id)

    missing = [sample_id for sample_id, values in labels_by_id.items() if not values]
    if missing:
        raise ValueError(f"labels are missing {len(missing)} dataset samples")
    expected_draw_ids = next(iter(draw_ids_by_id.values()))
    if any(draw_ids != expected_draw_ids for draw_ids in draw_ids_by_id.values()):
        raise ValueError("labels have inconsistent draw coverage")

    examples = {split: [] for split in ("train", "validation", "test")}
    for sample_id, prefix in prefix_by_id.items():
        split = prefix.get("split")
        if split not in examples:
            raise ValueError(f"invalid split for {sample_id}: {split!r}")
        examples[split].append(
            {
                "sample_id": sample_id,
                "text": _messages_text(prefix.get("messages")),
                "target": statistics.fmean(labels_by_id[sample_id]),
            }
        )
    for rows in examples.values():
        rows.sort(key=lambda row: row["sample_id"])
    if not examples["train"] or not examples["test"]:
        raise ValueError("SSJF-Reg requires non-empty train and test splits")
    return examples


class SSJFRegressor(nn.Module):
    """The released SSJF BERT regression architecture."""

    def __init__(self, encoder: str, hidden_dim: int = 128) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(encoder)
        hidden_size = int(self.bert.config.hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state[:, 0, :]
        return self.head(hidden).squeeze(-1)


class TIERegressor(nn.Module):
    """TIE's fixed-nu dual-head distribution predictor."""

    def __init__(self, encoder: str, hidden_dim: int = 256) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder, local_files_only=True)
        feature_dim = int(self.encoder.config.hidden_size) * 3

        def feature_extractor() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2),
            )

        def predictor() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1),
            )

        self.mu_feature_extractor = feature_extractor()
        self.sigma_feature_extractor = feature_extractor()
        self.mu_predictor = predictor()
        self.sigma_predictor = predictor()
        for module in (
            self.mu_feature_extractor,
            self.sigma_feature_extractor,
            self.mu_predictor,
            self.sigma_predictor,
        ):
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand_as(hidden).float()
        cls = hidden[:, 0, :]
        mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        maximum = (hidden * mask + (1 - mask) * -1e9).max(dim=1).values
        pooled = torch.cat((cls, mean, maximum), dim=-1)
        return (
            self.mu_predictor(self.mu_feature_extractor(pooled)).squeeze(-1),
            self.sigma_predictor(self.sigma_feature_extractor(pooled)).squeeze(-1),
        )


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _batches(
    rows: Sequence[dict[str, Any]], batch_size: int
) -> Sequence[Sequence[dict[str, Any]]]:
    return [
        rows[index : index + batch_size] for index in range(0, len(rows), batch_size)
    ]


def _ssjf_batch(
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    encoded = tokenizer(
        [row["text"] for row in rows],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    inputs = {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }
    targets = torch.tensor(
        [row["target"] for row in rows], dtype=torch.float32, device=device
    )
    return inputs, targets


@torch.no_grad()
def _ssjf_predict(
    model: SSJFRegressor,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    device: torch.device,
) -> list[float]:
    model.eval()
    predictions: list[float] = []
    for batch in _batches(rows, batch_size):
        inputs, _targets = _ssjf_batch(tokenizer, batch, device)
        predictions.extend(model(**inputs).cpu().tolist())
    return [max(1.0, float(value)) for value in predictions]


def run_ssjf_reg(
    dataset_dir: Path,
    labels_path: Path,
    output_dir: Path,
    *,
    encoder: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device_name: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    examples = _point_examples(dataset_dir, labels_path)
    random.seed(seed)
    torch.manual_seed(seed)
    device = _device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(encoder)
    tokenizer.truncation_side = "left"
    model = SSJFRegressor(encoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    training_steps = epochs * math.ceil(len(examples["train"]) / batch_size)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=training_steps,
    )
    losses: list[float] = []
    train_rows = list(examples["train"])
    random.Random(seed).shuffle(train_rows)
    for epoch in range(epochs):
        if epoch == 3:
            for parameter in model.bert.parameters():
                parameter.requires_grad = False
            for group in optimizer.param_groups:
                group["lr"] = 1e-4
        model.train()
        epoch_loss = 0.0
        for batch in _batches(train_rows, batch_size):
            inputs, targets = _ssjf_batch(tokenizer, batch, device)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(**inputs), targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += float(loss.detach())
        losses.append(epoch_loss / math.ceil(len(train_rows) / batch_size))

    test_predictions = _ssjf_predict(
        model,
        tokenizer,
        examples["test"],
        batch_size=batch_size,
        device=device,
    )
    output_dir.mkdir(parents=True)
    predictions_path = output_dir / "predictions.jsonl"
    _write_jsonl(
        predictions_path,
        [
            {
                "sample_id": row["sample_id"],
                "predicted_tokens": predicted,
            }
            for row, predicted in zip(examples["test"], test_predictions, strict=True)
        ],
    )
    torch.save(model.state_dict(), output_dir / "model.pt")
    protocol = {
        "method": "ssjf-reg",
        "upstream_commit": SSJF_UPSTREAM_COMMIT,
        "dataset_dir": str(dataset_dir.resolve()),
        "labels_path": str(labels_path.resolve()),
        "label_reduction": "arithmetic_mean_over_draws",
        "input": "canonical_json_of_full_messages_then_left_truncate_to_512_tokens",
        "prediction_postprocess": "clamp_to_at_least_one_token",
        "encoder": encoder,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "device": str(device),
        "train_loss": losses,
        "counts": {split: len(rows) for split, rows in examples.items()},
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return protocol


def _verify_upstream(
    upstream_dir: Path, commit: str, code_paths: Sequence[str]
) -> None:
    if not (upstream_dir / ".git").exists():
        raise ValueError(f"upstream directory is not a git checkout: {upstream_dir}")
    actual = subprocess.run(
        ["git", "-C", str(upstream_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != commit:
        raise ValueError(f"upstream commit mismatch: expected {commit}, found {actual}")
    status = subprocess.run(
        ["git", "-C", str(upstream_dir), "status", "--porcelain", "--", *code_paths],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    dirty = [
        line
        for line in status
        if "/__pycache__/" not in line and not line.rstrip().endswith(".pyc")
    ]
    if dirty:
        details = "\n".join(dirty)
        raise ValueError(f"upstream implementation is dirty:\n{details}")


def _run_egtp(
    upstream_dir: Path,
    python: Path,
    data_dir: Path,
    output_dir: Path,
    options: dict[str, object],
) -> None:
    upstream_dir = upstream_dir.resolve()
    python = python.resolve()
    data_dir = data_dir.resolve()
    output_dir = output_dir.resolve()
    _verify_upstream(upstream_dir, EGTP_UPSTREAM_COMMIT, ("EGTP/egtp", "EGTP/main.py"))
    entrypoint = upstream_dir / "EGTP" / "main.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"EGTP entrypoint not found: {entrypoint}")
    command = [
        str(python),
        str(entrypoint),
        "--data_dir",
        str(data_dir),
        "--output_dir",
        str(output_dir),
    ]
    for key, value in options.items():
        command.extend((f"--{key}", str(value)))
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(command, check=True, cwd=entrypoint.parent, env=environment)


def _egtp_text(messages: object, tokenizer: Any, tools: object) -> str:
    if not isinstance(messages, list):
        raise ValueError("prefix messages must be a list")
    kwargs = {"tools": tools} if tools else {}
    return str(
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
    )


def _write_egtp_split(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=("user_prompt_content", "target_length")
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "user_prompt_content": row["text"],
                    "target_length": row["target"],
                }
            )


def run_egtp_static(
    dataset_dir: Path,
    labels_path: Path,
    output_dir: Path,
    upstream_dir: Path,
    *,
    upstream_python: Path,
    model_id: str,
    prompt_prefix_k: int,
    num_bins: int,
    lambda_val: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    extractor_batch_size: int,
    torch_dtype: str,
    seed: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if min(prompt_prefix_k, num_bins, epochs, batch_size, extractor_batch_size) <= 0:
        raise ValueError("counts must be positive")
    if not 0 <= lambda_val <= 1 or learning_rate <= 0:
        raise ValueError("lambda_val must be in [0, 1] and learning_rate positive")
    examples = _point_examples(dataset_dir, labels_path)
    dataset_protocol = json.loads((dataset_dir / "dataset.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tools = (dataset_protocol.get("request_options") or {}).get("tools")
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True)
    prefix_by_id = {
        str(prefix["sample_id"]): prefix
        for prefix in _read_jsonl(dataset_dir / "prefixes.jsonl")
    }
    projected = {
        split: [
            {
                **row,
                "text": _egtp_text(
                    prefix_by_id[row["sample_id"]]["messages"], tokenizer, tools
                ),
            }
            for row in rows
        ]
        for split, rows in examples.items()
    }
    _write_egtp_split(data_dir / "train.csv", projected["train"])
    _write_egtp_split(data_dir / "test.csv", projected["test"])
    upstream_output = output_dir / "upstream"
    _run_egtp(
        upstream_dir,
        upstream_python,
        data_dir,
        upstream_output,
        {
            "model_id": model_id,
            "prompt_prefix_k": prompt_prefix_k,
            "num_bins": num_bins,
            "lambda_val": lambda_val,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": learning_rate,
            "extractor_batch_size": extractor_batch_size,
            "torch_dtype": torch_dtype,
            "seed": seed,
        },
    )
    with (upstream_output / "test_predictions.csv").open(
        encoding="utf-8", newline=""
    ) as source:
        predicted = [float(row["pred_length"]) for row in csv.DictReader(source)]
    if len(predicted) != len(examples["test"]):
        raise ValueError("EGTP prediction count does not match the test split")
    if any(not math.isfinite(value) or value <= 0 for value in predicted):
        raise ValueError("EGTP produced a non-positive or non-finite prediction")
    _write_jsonl(
        output_dir / "predictions.jsonl",
        [
            {"sample_id": row["sample_id"], "predicted_tokens": value}
            for row, value in zip(examples["test"], predicted, strict=True)
        ],
    )
    protocol = {
        "method": "egtp-static",
        "upstream_commit": EGTP_UPSTREAM_COMMIT,
        "upstream_dir": str(upstream_dir.resolve()),
        "upstream_python": str(upstream_python.resolve()),
        "dataset_dir": str(dataset_dir.resolve()),
        "labels_path": str(labels_path.resolve()),
        "label_reduction": "arithmetic_mean_over_draws",
        "input": (
            "full_messages_rendered_by_target_tokenizer_chat_template; official extractor "
            "keeps only "
            f"the first {prompt_prefix_k} target-model tokens"
        ),
        "unused_split": "validation",
        "model_id": model_id,
        "prompt_prefix_k": prompt_prefix_k,
        "num_bins": num_bins,
        "lambda_val": lambda_val,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "extractor_batch_size": extractor_batch_size,
        "torch_dtype": torch_dtype,
        "seed": seed,
        "counts": {split: len(rows) for split, rows in examples.items()},
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return protocol


def _run_outlets(
    upstream_dir: Path,
    python: Path,
    inputs_path: Path,
    output_path: Path,
    *,
    checkpoint_path: Path,
    base_model_path: str,
    config_path: Path,
    device: str,
) -> None:
    _verify_upstream(upstream_dir, OUTLETS_UPSTREAM_COMMIT, ("outlets",))
    package_dir = upstream_dir / "outlets"
    if not (package_dir / "inference_length.py").is_file():
        raise FileNotFoundError(f"OUTLETS inference code not found under {package_dir}")
    if not (checkpoint_path / "pytorch_model.bin").is_file():
        raise FileNotFoundError(
            f"OUTLETS checkpoint missing: {checkpoint_path / 'pytorch_model.bin'}"
        )
    if not config_path.is_file():
        raise FileNotFoundError(f"OUTLETS config missing: {config_path}")
    package_dir = package_dir.resolve()
    python = python.resolve()
    checkpoint_path = checkpoint_path.resolve()
    config_path = config_path.resolve()
    inputs_path = inputs_path.resolve()
    output_path = output_path.resolve()
    local_model = Path(base_model_path)
    if local_model.exists():
        base_model_path = str(local_model.resolve())
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [
            str(python),
            "-c",
            _OUTLETS_WORKER,
            str(package_dir),
            str(checkpoint_path),
            base_model_path,
            str(config_path),
            device,
            str(inputs_path),
            str(output_path),
        ],
        check=True,
        cwd=package_dir,
        env=environment,
    )


def run_outlets_static(
    dataset_dir: Path,
    output_dir: Path,
    upstream_dir: Path,
    *,
    upstream_python: Path,
    checkpoint_path: Path,
    base_model_path: str,
    config_path: Path,
    device: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    prefixes = [
        row
        for row in _read_jsonl(dataset_dir / "prefixes.jsonl")
        if row.get("split") == "test"
    ]
    if not prefixes:
        raise ValueError("OUTLETS requires a non-empty test split")
    prefixes.sort(key=lambda row: str(row["sample_id"]))
    output_dir.mkdir(parents=True)
    inputs_path = output_dir / "inputs.jsonl"
    _write_jsonl(
        inputs_path,
        [
            {
                "sample_id": str(row["sample_id"]),
                "prompt": _messages_text(row.get("messages")),
            }
            for row in prefixes
        ],
    )
    raw_predictions = output_dir / "upstream-predictions.jsonl"
    _run_outlets(
        upstream_dir,
        upstream_python,
        inputs_path,
        raw_predictions,
        checkpoint_path=checkpoint_path,
        base_model_path=base_model_path,
        config_path=config_path,
        device=device,
    )
    predictions = _read_jsonl(raw_predictions)
    expected_ids = [str(row["sample_id"]) for row in prefixes]
    actual_ids = [str(row.get("sample_id")) for row in predictions]
    if actual_ids != expected_ids:
        raise ValueError("OUTLETS prediction IDs or order do not match the test split")
    for row in predictions:
        value = row.get("predicted_tokens")
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("OUTLETS produced a non-positive or non-finite prediction")
    _write_jsonl(output_dir / "predictions.jsonl", predictions)
    protocol = {
        "method": "outlets-static",
        "upstream_commit": OUTLETS_UPSTREAM_COMMIT,
        "upstream_dir": str(upstream_dir.resolve()),
        "upstream_python": str(upstream_python.resolve()),
        "dataset_dir": str(dataset_dir.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_provenance": "official_OUTLETS_training_pipeline_required",
        "base_model_path": base_model_path,
        "config_path": str(config_path.resolve()),
        "device": device,
        "input": (
            "canonical_json_of_full_messages passed as the single user prompt accepted "
            "by the official inference API"
        ),
        "normalization": "official_log1p",
        "prediction_postprocess": "official_integer_conversion_and_nonnegative_clamp",
        "counts": {"test": len(prefixes)},
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return protocol


def _tie_examples(
    dataset_dir: Path, labels_path: Path
) -> tuple[dict[str, list[dict[str, Any]]], list[int]]:
    prefixes = _read_jsonl(dataset_dir / "prefixes.jsonl")
    prefix_by_id = {str(row.get("sample_id")): row for row in prefixes}
    if len(prefix_by_id) != len(prefixes):
        raise ValueError("dataset contains duplicate sample_id values")
    labels: dict[str, dict[int, int]] = {sample_id: {} for sample_id in prefix_by_id}
    label_rows, _protocol = _read_natural_labels(
        dataset_dir, labels_path, require_stochastic=True
    )
    for row in label_rows:
        sample_id = str(row.get("sample_id"))
        if sample_id not in labels:
            continue
        draw_id = row.get("draw_id")
        actual = row.get("actual_tokens")
        if (
            not isinstance(draw_id, int)
            or isinstance(draw_id, bool)
            or draw_id < 0
            or not isinstance(actual, int)
            or isinstance(actual, bool)
            or actual <= 0
        ):
            raise ValueError(f"invalid TIE label for {sample_id}")
        if draw_id in labels[sample_id]:
            raise ValueError(f"duplicate TIE label: {(sample_id, draw_id)}")
        labels[sample_id][draw_id] = actual
    draw_sets = {tuple(sorted(values)) for values in labels.values()}
    if len(draw_sets) != 1:
        raise ValueError("TIE labels have inconsistent draw coverage")
    draw_ids = list(next(iter(draw_sets)))
    if len(draw_ids) < 2:
        raise ValueError("TIE requires at least two natural draws per prefix")

    examples = {split: [] for split in ("train", "validation", "test")}
    for sample_id, prefix in prefix_by_id.items():
        split = prefix.get("split")
        if split not in examples:
            raise ValueError(f"invalid split for {sample_id}: {split!r}")
        log_lengths = [math.log(labels[sample_id][draw_id]) for draw_id in draw_ids]
        mu, sigma = _fit_logt_mle(log_lengths)
        examples[split].append(
            {
                "sample_id": sample_id,
                "text": _messages_text(prefix.get("messages")),
                "mu": mu,
                "sigma": max(sigma, 1e-6),
            }
        )
    for rows in examples.values():
        rows.sort(key=lambda row: row["sample_id"])
    if any(not examples[split] for split in examples):
        raise ValueError("TIE requires non-empty train, validation, and test splits")
    return examples, draw_ids


def _fit_logt_mle(log_lengths: Sequence[float]) -> tuple[float, float]:
    if min(log_lengths) == max(log_lengths):
        return float(log_lengths[0]), 1e-6

    from scipy.optimize import minimize
    from scipy.special import gammaln

    nu = TIE_DEGREES_OF_FREEDOM
    initial_mu = statistics.fmean(log_lengths)
    initial_sigma = max(
        math.sqrt(statistics.pvariance(log_lengths) * (nu - 2) / nu),
        1e-3,
    )
    constant = 0.5 * math.log(nu * math.pi) + gammaln(nu / 2) - gammaln((nu + 1) / 2)

    def negative_log_likelihood(parameters: Sequence[float]) -> float:
        mu, sigma = parameters
        return len(log_lengths) * (math.log(sigma) + constant) + ((nu + 1) / 2) * sum(
            math.log1p(((value - mu) / sigma) ** 2 / nu) for value in log_lengths
        )

    result = minimize(
        negative_log_likelihood,
        (initial_mu, initial_sigma),
        method="L-BFGS-B",
        bounds=((None, None), (1e-6, None)),
    )
    if not result.success:
        raise RuntimeError(f"TIE log-t MLE failed: {result.message}")
    mu, sigma = (float(value) for value in result.x)
    if not math.isfinite(mu) or not math.isfinite(sigma) or sigma <= 0:
        raise RuntimeError("TIE log-t MLE returned invalid parameters")
    return mu, sigma


def _tie_normalization(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    mu = [float(row["mu"]) for row in rows]
    sigma_log = [math.log1p(float(row["sigma"])) for row in rows]
    return {
        "mu_mean": statistics.fmean(mu),
        "mu_std": statistics.stdev(mu) if len(mu) > 1 else 0.0,
        "sigma_log_mean": statistics.fmean(sigma_log),
        "sigma_log_std": statistics.stdev(sigma_log) if len(sigma_log) > 1 else 0.0,
    }


def _tie_batch(
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    normalization: dict[str, float],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        [row["text"] for row in rows],
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs = {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }
    mu = torch.tensor(
        [
            (row["mu"] - normalization["mu_mean"]) / (normalization["mu_std"] + 1e-8)
            for row in rows
        ],
        dtype=torch.float32,
        device=device,
    )
    sigma = torch.tensor(
        [
            (math.log1p(row["sigma"]) - normalization["sigma_log_mean"])
            / (normalization["sigma_log_std"] + 1e-8)
            for row in rows
        ],
        dtype=torch.float32,
        device=device,
    )
    return inputs, mu, sigma


@torch.no_grad()
def _tie_loss(
    model: TIERegressor,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    normalization: dict[str, float],
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    for batch in _batches(rows, batch_size):
        inputs, target_mu, target_sigma = _tie_batch(
            tokenizer, batch, normalization, device
        )
        predicted_mu, predicted_sigma = model(**inputs)
        total += float(
            nn.functional.mse_loss(predicted_mu, target_mu)
            + nn.functional.mse_loss(predicted_sigma, target_sigma)
        )
    return total / len(_batches(rows, batch_size))


@torch.no_grad()
def _tie_predict(
    model: TIERegressor,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    normalization: dict[str, float],
    *,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, float]]:
    model.eval()
    predictions: list[dict[str, float]] = []
    for batch in _batches(rows, batch_size):
        inputs, _mu, _sigma = _tie_batch(tokenizer, batch, normalization, device)
        normalized_mu, normalized_sigma = model(**inputs)
        for mu_value, sigma_value in zip(
            normalized_mu.cpu().tolist(), normalized_sigma.cpu().tolist(), strict=True
        ):
            mu = mu_value * normalization["mu_std"] + normalization["mu_mean"]
            sigma_log = (
                sigma_value * normalization["sigma_log_std"]
                + normalization["sigma_log_mean"]
            )
            predictions.append(
                {
                    "predicted_logt_mu": mu,
                    "predicted_logt_sigma": max(math.expm1(sigma_log), 1e-6),
                    "predicted_tokens": math.exp(mu),
                }
            )
    return predictions


def run_tie(
    dataset_dir: Path,
    labels_path: Path,
    output_dir: Path,
    upstream_dir: Path,
    *,
    encoder: str,
    epochs: int,
    encoder_tuning_epochs: int,
    batch_size: int,
    learning_rate: float,
    frozen_learning_rate: float,
    seed: int,
    device_name: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if min(epochs, encoder_tuning_epochs, batch_size) <= 0:
        raise ValueError(
            "epochs, encoder_tuning_epochs, and batch_size must be positive"
        )
    if encoder_tuning_epochs >= epochs:
        raise ValueError("encoder_tuning_epochs must be smaller than epochs")
    if min(learning_rate, frozen_learning_rate) <= 0:
        raise ValueError("learning rates must be positive")
    _verify_upstream(
        upstream_dir,
        TIE_UPSTREAM_COMMIT,
        ("train/model_train.py", "vllm/v1/core/sched/ua_predictor.py"),
    )
    examples, draw_ids = _tie_examples(dataset_dir, labels_path)
    normalization = _tie_normalization(examples["train"])
    random.seed(seed)
    torch.manual_seed(seed)
    device = _device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(encoder, local_files_only=True)
    tokenizer.truncation_side = "right"
    model = TIERegressor(encoder).to(device)
    steps_per_epoch = math.ceil(len(examples["train"]) / batch_size)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01, eps=1e-8
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * epochs * steps_per_epoch),
        num_training_steps=epochs * steps_per_epoch,
    )
    weights = [
        2.0
        if row["mu"] > 6 or row["sigma"] > 1.3
        else 1.5
        if row["mu"] > 5.5 or row["sigma"] > 1
        else 1.0
        for row in examples["train"]
    ]
    generator = torch.Generator().manual_seed(seed)
    output_dir.mkdir(parents=True)
    checkpoint_path = output_dir / "model.pt"
    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_validation = math.inf
    for epoch in range(epochs):
        if epoch == encoder_tuning_epochs:
            for parameter in model.encoder.parameters():
                parameter.requires_grad = False
            optimizer = torch.optim.AdamW(
                filter(lambda parameter: parameter.requires_grad, model.parameters()),
                lr=frozen_learning_rate,
                weight_decay=0.01,
                eps=1e-8,
            )
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=(epochs - epoch) * steps_per_epoch,
            )
        sampled_indices = torch.multinomial(
            torch.tensor(weights),
            len(weights),
            replacement=True,
            generator=generator,
        ).tolist()
        sampled_rows = [examples["train"][index] for index in sampled_indices]
        model.train()
        total_loss = 0.0
        sigma_weight = 3 + epoch / epochs
        for batch in _batches(sampled_rows, batch_size):
            inputs, target_mu, target_sigma = _tie_batch(
                tokenizer, batch, normalization, device
            )
            optimizer.zero_grad()
            predicted_mu, predicted_sigma = model(**inputs)
            loss = nn.functional.mse_loss(
                predicted_mu, target_mu
            ) + sigma_weight * nn.functional.mse_loss(predicted_sigma, target_sigma)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach())
        train_losses.append(total_loss / steps_per_epoch)
        validation = _tie_loss(
            model,
            tokenizer,
            examples["validation"],
            normalization,
            batch_size=batch_size,
            device=device,
        )
        validation_losses.append(validation)
        if validation < best_validation:
            best_validation = validation
            torch.save(model.state_dict(), checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    predictions = _tie_predict(
        model,
        tokenizer,
        examples["test"],
        normalization,
        batch_size=batch_size,
        device=device,
    )
    distribution_rows = [
        {"sample_id": row["sample_id"], **prediction}
        for row, prediction in zip(examples["test"], predictions, strict=True)
    ]
    _write_jsonl(output_dir / "distribution_predictions.jsonl", distribution_rows)
    _write_jsonl(
        output_dir / "predictions.jsonl",
        [
            {
                "sample_id": row["sample_id"],
                "predicted_tokens": row["predicted_tokens"],
            }
            for row in distribution_rows
        ],
    )
    protocol = {
        "method": "tie-fixed-logt",
        "upstream_commit": TIE_UPSTREAM_COMMIT,
        "upstream_dir": str(upstream_dir.resolve()),
        "dataset_dir": str(dataset_dir.resolve()),
        "labels_path": str(labels_path.resolve()),
        "distribution": {"family": "log_student_t", "degrees_of_freedom": 3.5},
        "target_fit": "joint_MLE_via_L-BFGS-B_on_log_token_lengths",
        "draw_ids": draw_ids,
        "point_prediction": "distribution_median_exp_mu",
        "input": "canonical_json_of_full_messages_then_right_truncate_to_512_tokens",
        "encoder": encoder,
        "epochs": epochs,
        "encoder_tuning_epochs": encoder_tuning_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "frozen_learning_rate": frozen_learning_rate,
        "seed": seed,
        "device": str(device),
        "normalization": normalization,
        "train_loss": train_losses,
        "validation_loss": validation_losses,
        "counts": {split: len(rows) for split, rows in examples.items()},
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return protocol


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ssjf = commands.add_parser("ssjf-reg")
    ssjf.add_argument("--dataset-dir", type=Path, required=True)
    ssjf.add_argument("--labels", type=Path, required=True)
    ssjf.add_argument("--output-dir", type=Path, required=True)
    ssjf.add_argument("--encoder", default="bert-base-uncased")
    ssjf.add_argument("--epochs", type=int, default=6)
    ssjf.add_argument("--batch-size", type=int, default=16)
    ssjf.add_argument("--learning-rate", type=float, default=1e-5)
    ssjf.add_argument("--seed", type=int, default=42)
    ssjf.add_argument("--device", default="auto")
    egtp = commands.add_parser("egtp-static")
    egtp.add_argument("--dataset-dir", type=Path, required=True)
    egtp.add_argument("--labels", type=Path, required=True)
    egtp.add_argument("--output-dir", type=Path, required=True)
    egtp.add_argument("--upstream-dir", type=Path, required=True)
    egtp.add_argument("--upstream-python", type=Path, default=Path(sys.executable))
    egtp.add_argument("--model-id", required=True)
    egtp.add_argument("--prompt-prefix-k", type=int, default=4)
    egtp.add_argument("--num-bins", type=int, default=20)
    egtp.add_argument("--lambda-val", type=float, default=0.95)
    egtp.add_argument("--epochs", type=int, default=200)
    egtp.add_argument("--batch-size", type=int, default=256)
    egtp.add_argument("--learning-rate", type=float, default=2e-5)
    egtp.add_argument("--extractor-batch-size", type=int, default=64)
    egtp.add_argument("--torch-dtype", default="bfloat16")
    egtp.add_argument("--seed", type=int, default=42)
    outlets = commands.add_parser("outlets-static")
    outlets.add_argument("--dataset-dir", type=Path, required=True)
    outlets.add_argument("--output-dir", type=Path, required=True)
    outlets.add_argument("--upstream-dir", type=Path, required=True)
    outlets.add_argument("--upstream-python", type=Path, default=Path(sys.executable))
    outlets.add_argument("--checkpoint-path", type=Path, required=True)
    outlets.add_argument("--base-model-path", required=True)
    outlets.add_argument("--config-path", type=Path, required=True)
    outlets.add_argument("--device", default="cuda")
    tie = commands.add_parser("tie")
    tie.add_argument("--dataset-dir", type=Path, required=True)
    tie.add_argument("--labels", type=Path, required=True)
    tie.add_argument("--output-dir", type=Path, required=True)
    tie.add_argument("--upstream-dir", type=Path, required=True)
    tie.add_argument("--encoder", required=True)
    tie.add_argument("--epochs", type=int, default=20)
    tie.add_argument("--encoder-tuning-epochs", type=int, default=12)
    tie.add_argument("--batch-size", type=int, default=32)
    tie.add_argument("--learning-rate", type=float, default=2e-5)
    tie.add_argument("--frozen-learning-rate", type=float, default=5e-5)
    tie.add_argument("--seed", type=int, default=42)
    tie.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "ssjf-reg":
        result = run_ssjf_reg(
            args.dataset_dir,
            args.labels,
            args.output_dir,
            encoder=args.encoder,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device_name=args.device,
        )
        print(json.dumps(result["counts"], indent=2))
    elif args.command == "egtp-static":
        result = run_egtp_static(
            args.dataset_dir,
            args.labels,
            args.output_dir,
            args.upstream_dir,
            upstream_python=args.upstream_python,
            model_id=args.model_id,
            prompt_prefix_k=args.prompt_prefix_k,
            num_bins=args.num_bins,
            lambda_val=args.lambda_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            extractor_batch_size=args.extractor_batch_size,
            torch_dtype=args.torch_dtype,
            seed=args.seed,
        )
        print(json.dumps(result["counts"], indent=2))
    elif args.command == "outlets-static":
        result = run_outlets_static(
            args.dataset_dir,
            args.output_dir,
            args.upstream_dir,
            upstream_python=args.upstream_python,
            checkpoint_path=args.checkpoint_path,
            base_model_path=args.base_model_path,
            config_path=args.config_path,
            device=args.device,
        )
        print(json.dumps(result["counts"], indent=2))
    elif args.command == "tie":
        result = run_tie(
            args.dataset_dir,
            args.labels,
            args.output_dir,
            args.upstream_dir,
            encoder=args.encoder,
            epochs=args.epochs,
            encoder_tuning_epochs=args.encoder_tuning_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            frozen_learning_rate=args.frozen_learning_rate,
            seed=args.seed,
            device_name=args.device,
        )
        print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
