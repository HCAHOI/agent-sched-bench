#!/usr/bin/env python3
"""Evaluate causal tabular quantile and binary resource predictors."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from scripts.evaluation.evaluate_prior_calibration import ecdf_survival  # noqa: E402
from scripts.evaluation.evaluate_resource_prediction import (  # noqa: E402
    build_ambient_residual_ecdfs,
    repo_cluster_key,
)
from scripts.training.train_resource_bert import (  # noqa: E402
    _run_head_epochs,
    set_seed,
)
from tool_resource.bert_model import BertModelConfig, ToolResourceBert  # noqa: E402
from tool_resource.features import TabularDataset, build_tabular_dataset  # noqa: E402
from tool_resource.labels import ResourceCallSample, load_resource_corpus  # noqa: E402
from tool_resource.metrics import ecdf_quantile, pinball_loss  # noqa: E402
from tool_resource.prior import build_resource_prior, resource_prior_hierarchy  # noqa: E402
from tool_time.command import make_row_command_prefix_keys  # noqa: E402
from tool_time.prior import (  # noqa: E402
    LatencyPrior,
    build_latency_prior,
    latency_prior_hierarchy,
)
from tool_time.statistics import resample_task_totals  # noqa: E402


_DEFAULT_FIT = Path("traces/swe-rebench/qwen3.7-max/offline-gated-confirm-100-v2")
_DEFAULT_EVAL = Path("traces/swe-rebench/qwen3.7-max/fresh-seed42-skip150-n200")
_DEFAULT_FIT_MANIFEST = Path("configs/corpora/swe-100.json")
_DEFAULT_EVAL_MANIFEST = Path("configs/corpora/swe-277.json")
_QUANTILES = (0.50, 0.90, 0.95)
_TARGETS = {
    "latency": "latency_ms",
    "cpu_peak": "peak_cpu_cores",
    "mem_peak": "peak_memory_mb",
}
_UNITS = {
    "latency_ms": "ms",
    "peak_cpu_cores": "cores",
    "peak_memory_mb": "mb",
}
_PREFIX_KEYS = make_row_command_prefix_keys("command", max_depth=4)
_HIDDEN_DIM = 256
_EPOCHS = 150
_SMOKE_EPOCHS = 5
_BATCH_SIZE = 256
_LEARNING_RATE = 1e-3
_CLAMP_FACTOR = 10.0
_BOOTSTRAP_REPLICATES = 50_000
_CONFIDENCE_LEVEL = 0.95


def _feature_matrix(dataset: TabularDataset) -> np.ndarray:
    return np.column_stack(
        [dataset.features[name] for name in dataset.feature_names]
    ).astype(np.float32, copy=False)


def _standardize_fit_eval(
    fit_features: np.ndarray,
    eval_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize both splits using statistics from the fit split only."""

    mean = fit_features.mean(axis=0)
    std = np.maximum(fit_features.std(axis=0), 1e-6)
    return (
        (fit_features - mean) / std,
        (eval_features - mean) / std,
        mean,
        std,
    )


def _clamp_prediction_pair(
    model_predictions: np.ndarray,
    baseline_predictions: np.ndarray,
    fit_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same physical prediction range to model and baseline."""

    cap = _CLAMP_FACTOR * fit_max
    return (
        np.clip(model_predictions, 0.0, cap),
        np.clip(baseline_predictions, 0.0, cap),
    )


def quantile_exceedance(
    predictions: np.ndarray,
    thresholds: Sequence[float],
    quantiles: Sequence[float] = _QUANTILES,
) -> np.ndarray:
    """Interpolate ``P(Y > threshold)`` from monotone predicted quantiles."""

    predicted = np.asarray(predictions, dtype=float)
    qs = np.asarray(quantiles, dtype=float)
    cuts = np.asarray(thresholds, dtype=float)
    if predicted.ndim != 2 or predicted.shape[1] != len(qs):
        raise ValueError("predictions must have one column per quantile")
    if np.any(np.diff(qs) <= 0.0):
        raise ValueError("quantiles must be strictly increasing")
    monotone = np.maximum.accumulate(predicted, axis=1)
    result = np.empty((len(monotone), len(cuts)), dtype=float)
    for row_index, row in enumerate(monotone):
        result[row_index] = 1.0 - np.interp(
            cuts,
            row,
            qs,
            left=float(qs[0]),
            right=float(qs[-1]),
        )
    return result


def _sample_row(sample: ResourceCallSample) -> dict[str, Any]:
    row = sample.to_json_obj()
    row["latency_ms"] = (sample.tool_ts_end - sample.tool_ts_start) * 1000.0
    return row


def _flatten(
    samples_by_task: Mapping[str, Sequence[ResourceCallSample]],
    task_ids: Sequence[str],
) -> list[ResourceCallSample]:
    return [sample for task_id in task_ids for sample in samples_by_task[task_id]]


def _samples_in_dataset_order(
    dataset: TabularDataset,
    samples: Sequence[ResourceCallSample],
) -> list[ResourceCallSample]:
    by_id = {sample.sample_id: sample for sample in samples}
    if len(by_id) != len(samples):
        raise ValueError("resource corpus contains duplicate sample IDs")
    return [by_id[str(sample_id)] for sample_id in dataset.sample_ids]


def _load_cost_table(path: Path | None) -> Mapping[str, Mapping[str, float]] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or any(
        not isinstance(key, str) or not isinstance(value, dict)
        for key, value in payload.items()
    ):
        raise ValueError("--cost-table must contain an object of per-binary objects")
    return payload


def _concatenate_datasets(datasets: Sequence[TabularDataset]) -> TabularDataset:
    feature_names = datasets[0].feature_names
    if any(dataset.feature_names != feature_names for dataset in datasets[1:]):
        raise ValueError("out-of-repo fit feature schemas differ")
    repo_folds: dict[str, int] = {}
    for dataset in datasets:
        overlap = set(repo_folds) & set(dataset.repo_folds)
        if overlap:
            raise ValueError(f"duplicate repositories in fit partitions: {overlap}")
        repo_folds.update(dataset.repo_folds)
    return TabularDataset(
        features={
            name: np.concatenate([dataset.features[name] for dataset in datasets])
            for name in feature_names
        },
        targets={
            name: np.concatenate([dataset.targets[name] for dataset in datasets])
            for name in datasets[0].targets
        },
        eligibility_masks={
            name: np.concatenate(
                [dataset.eligibility_masks[name] for dataset in datasets]
            )
            for name in datasets[0].eligibility_masks
        },
        sample_ids=np.concatenate([dataset.sample_ids for dataset in datasets]),
        task_ids=np.concatenate([dataset.task_ids for dataset in datasets]),
        fold_assignment=np.concatenate(
            [dataset.fold_assignment for dataset in datasets]
        ),
        repo_folds=repo_folds,
        feature_names=feature_names,
    )


def _build_out_of_repo_fit_dataset(
    samples_by_task: Mapping[str, Sequence[ResourceCallSample]],
    cost_table: Mapping[str, Mapping[str, float]] | None,
    taus: Sequence[float],
) -> TabularDataset:
    """Build each fit repository's features from priors excluding that repo."""

    tasks_by_repo: dict[str, list[str]] = {}
    for task_id in samples_by_task:
        tasks_by_repo.setdefault(repo_cluster_key(task_id), []).append(task_id)
    if len(tasks_by_repo) < 2:
        raise ValueError("fit features require at least two repositories")

    datasets = []
    for held_out_repo in sorted(tasks_by_repo):
        prior_rows = [
            _sample_row(sample)
            for task_id, samples in samples_by_task.items()
            if repo_cluster_key(task_id) != held_out_repo
            for sample in samples
            if not sample.censored
        ]
        prior = build_latency_prior(prior_rows, row_group_keys=_PREFIX_KEYS)
        held_out_tasks = {
            task_id: samples_by_task[task_id]
            for task_id in tasks_by_repo[held_out_repo]
        }
        datasets.append(build_tabular_dataset(held_out_tasks, prior, cost_table, taus))
    return _concatenate_datasets(datasets)


def _train_quantile_model(
    fit_features: np.ndarray,
    fit_dataset: TabularDataset,
    targets: Sequence[str],
    *,
    seed: int,
    epochs: int,
) -> tuple[ToolResourceBert, torch.device]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = torch.as_tensor(fit_features, dtype=torch.float32, device=device)
    target_tensors: dict[str, torch.Tensor] = {}
    mask_tensors: dict[str, torch.Tensor] = {}
    for target in targets:
        mask = fit_dataset.eligibility_masks[target]
        if not bool(mask.any()):
            raise ValueError(f"fit split has no eligible rows for {target}")
        values = np.where(mask, np.log1p(fit_dataset.targets[target]), 0.0)
        target_tensors[target] = torch.as_tensor(
            values, dtype=torch.float32, device=device
        )
        mask_tensors[target] = torch.as_tensor(mask, dtype=torch.bool, device=device)

    config = BertModelConfig(
        numeric_dim=features.shape[1],
        target_names=tuple(targets),
        quantiles=_QUANTILES,
        hidden_dim=_HIDDEN_DIM,
        numeric_only=True,
    )
    model = ToolResourceBert.heads_only(features.shape[1], config).to(device)
    _run_head_epochs(
        model,
        features,
        target_tensors,
        mask_tensors,
        epochs,
        _LEARNING_RATE,
        _BATCH_SIZE,
    )
    return model, device


@torch.no_grad()
def _predict_log_quantiles(
    model: ToolResourceBert,
    features: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    return {
        name: values.cpu().numpy()
        for name, values in model.heads_forward(tensor).items()
    }


def _inverse_prevalence_weights(labels: np.ndarray) -> np.ndarray:
    """Balance total positive and negative BCE weight."""

    binary = np.asarray(labels, dtype=bool)
    if binary.ndim != 1 or not len(binary):
        raise ValueError("binary labels must be a non-empty vector")
    prevalence = float(binary.mean())
    if prevalence in {0.0, 1.0}:
        return np.ones(len(binary), dtype=np.float32)
    return np.where(
        binary,
        0.5 / prevalence,
        0.5 / (1.0 - prevalence),
    ).astype(np.float32)


def _train_binary_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    *,
    seed: int,
    epochs: int,
) -> torch.nn.Linear:
    """Fit one weighted logistic head with the shared optimizer conventions."""

    set_seed(seed)
    model = torch.nn.Linear(features.shape[1], 1).to(device)
    feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    label_tensor = torch.as_tensor(labels, dtype=torch.float32, device=device)
    weight_tensor = torch.as_tensor(
        _inverse_prevalence_weights(labels), dtype=torch.float32, device=device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=_LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    params = list(model.parameters())
    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(len(features), device=device)
        for start in range(0, len(features), _BATCH_SIZE):
            indices = permutation[start : start + _BATCH_SIZE]
            logits = model(feature_tensor[indices]).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                label_tensor[indices],
                weight=weight_tensor[indices],
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
        scheduler.step()
    return model


@torch.no_grad()
def _predict_binary_probabilities(
    model: torch.nn.Module,
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Return raw sigmoid probabilities without resource-value clamping."""

    model.eval()
    tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    return torch.sigmoid(model(tensor).squeeze(-1)).cpu().numpy()


def _sample_target(sample: ResourceCallSample, target: str) -> float | None:
    if target == "latency_ms":
        return (
            (sample.tool_ts_end - sample.tool_ts_start) * 1000.0
            if not sample.censored
            else None
        )
    if target == "peak_cpu_cores":
        return (
            float(sample.peak_cpu_cores)
            if sample.peak_cpu_cores_eligible and sample.peak_cpu_cores is not None
            else None
        )
    return (
        float(sample.peak_memory_mb)
        if sample.peak_memory_mb_eligible and sample.peak_memory_mb is not None
        else None
    )


def _out_of_repo_baseline_scores(
    target: str,
    all_fit_samples: Sequence[ResourceCallSample],
    scoring_samples: Sequence[ResourceCallSample],
    thresholds: Sequence[float],
) -> np.ndarray:
    """Build leakage-free baseline survival features for fit rows."""

    positions_by_repo: dict[str, list[int]] = {}
    for index, sample in enumerate(scoring_samples):
        positions_by_repo.setdefault(repo_cluster_key(sample.task_id), []).append(index)
    scores = np.empty((len(scoring_samples), len(thresholds)), dtype=float)
    for held_out_repo, positions in sorted(positions_by_repo.items()):
        profile_samples = [
            sample
            for sample in all_fit_samples
            if repo_cluster_key(sample.task_id) != held_out_repo
        ]
        profile_values = [
            value
            for sample in profile_samples
            if (value := _sample_target(sample, target)) is not None
        ]
        if not profile_values:
            raise ValueError(
                f"no {target} profile rows outside repository {held_out_repo}"
            )
        prior = build_latency_prior(
            [_sample_row(sample) for sample in profile_samples if not sample.censored],
            row_group_keys=_PREFIX_KEYS,
        )
        held_out_samples = [scoring_samples[position] for position in positions]
        _, held_out_scores, _ = _baseline_forecasts(
            target,
            profile_samples,
            held_out_samples,
            prior,
            thresholds,
            _CLAMP_FACTOR * max(profile_values),
        )
        scores[positions] = held_out_scores
    return scores


def _baseline_forecasts(
    target: str,
    fit_samples: Sequence[ResourceCallSample],
    eval_samples: Sequence[ResourceCallSample],
    latency_prior: LatencyPrior,
    thresholds: Sequence[float],
    cap: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    cpu_prior = None
    residuals_by_tool = None
    global_residuals = None
    if target == "peak_cpu_cores":
        cpu_prior = build_resource_prior(
            [
                sample.to_json_obj()
                for sample in fit_samples
                if sample.peak_cpu_cores_eligible
            ],
            value_field="peak_cpu_cores",
        )
    elif target == "peak_memory_mb":
        residual_rows = [
            sample.to_json_obj()
            for sample in fit_samples
            if sample.peak_memory_mb_eligible and sample.ambient_before_mb is not None
        ]
        residuals_by_tool, global_residuals = build_ambient_residual_ecdfs(
            residual_rows
        )

    quantile_predictions = np.empty((len(eval_samples), len(_QUANTILES)))
    survival_scores = np.empty((len(eval_samples), len(thresholds)))
    sources: Counter[str] = Counter()
    for index, sample in enumerate(eval_samples):
        offset = 0.0
        if target == "latency_ms":
            node = latency_prior_hierarchy(
                latency_prior,
                sample.tool_name,
                _PREFIX_KEYS(sample.to_json_obj()),
                min_tool_history=1,
                min_profile_tasks=1,
            )[-1]
            values = node.values
            source = node.source
        elif target == "peak_cpu_cores":
            assert cpu_prior is not None
            node = resource_prior_hierarchy(
                cpu_prior,
                sample.tool_name,
                (),
                min_tool_history=1,
                min_profile_tasks=1,
            )[-1]
            values = node.values
            source = node.source
        else:
            assert residuals_by_tool is not None and global_residuals is not None
            selected = residuals_by_tool.get(sample.tool_name)
            values = selected or global_residuals
            offset = float(sample.ambient_before_mb)
            source = "tool_residual" if selected else "global_residual"

        clipped_values = np.clip(np.asarray(values) + offset, 0.0, cap)
        quantile_predictions[index] = [
            ecdf_quantile(clipped_values, quantile) for quantile in _QUANTILES
        ]
        survival_scores[index] = [
            ecdf_survival(clipped_values, threshold) for threshold in thresholds
        ]
        sources[source] += 1
    return quantile_predictions, survival_scores, dict(sorted(sources.items()))


def _pinball_losses(
    observations: np.ndarray,
    predictions: np.ndarray,
    quantile: float,
) -> np.ndarray:
    return np.fromiter(
        (
            pinball_loss(float(observation), float(prediction), quantile)
            for observation, prediction in zip(observations, predictions, strict=True)
        ),
        dtype=float,
        count=len(observations),
    )


def _forecast_metrics(
    observations: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    pinball = {}
    coverage = {}
    for column, quantile in enumerate(_QUANTILES):
        tag = f"p{int(100 * quantile)}"
        pinball[tag] = float(
            _pinball_losses(observations, predictions[:, column], quantile).mean()
        )
        coverage[tag] = float(np.mean(observations <= predictions[:, column]))
    epsilon = 1e-9
    qerror = np.maximum(
        np.maximum(predictions[:, 0], epsilon) / np.maximum(observations, epsilon),
        np.maximum(observations, epsilon) / np.maximum(predictions[:, 0], epsilon),
    )
    return {
        "pinball_original": pinball,
        "qerror_p50": {
            "median": float(np.median(qerror)),
            "p90": float(np.quantile(qerror, 0.9)),
        },
        "coverage": coverage,
    }


def _cluster_contributions(
    task_ids: Sequence[str],
    columns: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    keys = sorted({repo_cluster_key(task_id) for task_id in task_ids})
    positions = {key: index for index, key in enumerate(keys)}
    contributions = np.zeros((len(keys), columns.shape[1]), dtype=float)
    for task_id, row in zip(task_ids, columns, strict=True):
        contributions[positions[repo_cluster_key(task_id)]] += row
    return keys, contributions


def _p90_skill(
    observations: np.ndarray,
    model_predictions: np.ndarray,
    baseline_predictions: np.ndarray,
    task_ids: Sequence[str],
    *,
    seed: int,
) -> dict[str, Any]:
    model_losses = _pinball_losses(observations, model_predictions, 0.9)
    baseline_losses = _pinball_losses(observations, baseline_predictions, 0.9)
    keys, contributions = _cluster_contributions(
        task_ids, np.column_stack([model_losses, baseline_losses])
    )
    totals = contributions.sum(axis=0)
    if totals[1] <= 0.0:
        return {
            "point": None,
            "ci_low": None,
            "ci_high": None,
            "reason": "baseline p90 pinball loss is zero",
            "cluster_count": len(keys),
        }
    draws = resample_task_totals(
        contributions, replicates=_BOOTSTRAP_REPLICATES, seed=seed
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        skills = 1.0 - draws[:, 0] / draws[:, 1]
    finite = skills[np.isfinite(skills)]
    alpha = 1.0 - _CONFIDENCE_LEVEL
    low, high = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "point": float(1.0 - totals[0] / totals[1]),
        "ci_low": float(low),
        "ci_high": float(high),
        "cluster_count": len(keys),
        "confidence_level": _CONFIDENCE_LEVEL,
        "bootstrap_replicates": _BOOTSTRAP_REPLICATES,
        "seed": seed,
    }


def _placement_values(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    positive_scores = np.sort(scores[labels])
    negative_scores = np.sort(scores[~labels])
    placements = np.empty(len(scores), dtype=float)
    pos = scores[labels]
    neg = scores[~labels]
    placements[labels] = (
        np.searchsorted(negative_scores, pos, side="left")
        + 0.5
        * (
            np.searchsorted(negative_scores, pos, side="right")
            - np.searchsorted(negative_scores, pos, side="left")
        )
    ) / len(negative_scores)
    placements[~labels] = (
        len(positive_scores)
        - np.searchsorted(positive_scores, neg, side="right")
        + 0.5
        * (
            np.searchsorted(positive_scores, neg, side="right")
            - np.searchsorted(positive_scores, neg, side="left")
        )
    ) / len(positive_scores)
    return placements


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if not bool(labels.any()) or bool(labels.all()):
        return None
    return float(_placement_values(labels, scores)[labels].mean())


def _binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    fit_prevalence: float,
) -> dict[str, Any]:
    predicted_count = _fit_prevalence_count(fit_prevalence, len(labels))
    metrics = _decision_metrics(labels, scores, predicted_count)
    oracle = _decision_metrics(labels, scores, int(labels.sum()))
    return {
        "auc": _auc(labels, scores),
        "decision_cut_source": "fit_label_prevalence",
        "fit_prevalence": fit_prevalence,
        **metrics,
        "oracle_prevalence_cut": {
            "status": "SECONDARY DIAGNOSTIC - NOT DEPLOYABLE",
            "definition": "cut uses the realized eval positive count",
            "interpretation": "ranking ceiling only; not a deployment policy",
            **oracle,
        },
    }


def _decision_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    predicted_count: int,
) -> dict[str, Any]:
    count = len(labels)
    positive_count = int(labels.sum())
    negative_count = count - positive_count
    predicted_positive, score_threshold, tie_probability = (
        _prevalence_matched_predictions(scores, predicted_count)
    )
    true_positive = float(np.sum(predicted_positive * labels))
    true_negative = float(np.sum((1.0 - predicted_positive) * ~labels))
    balanced_accuracy = (
        0.5 * (true_positive / positive_count + true_negative / negative_count)
        if positive_count and negative_count
        else None
    )
    minority_positive = positive_count <= negative_count
    if minority_positive:
        minority_true = labels
        predicted_minority = predicted_positive
        minority_name = "exceedance"
    else:
        minority_true = ~labels
        predicted_minority = 1.0 - predicted_positive
        minority_name = "non_exceedance"
    minority_correct = float(np.sum(minority_true * predicted_minority))
    minority_count = int(minority_true.sum())
    predicted_minority_count = float(predicted_minority.sum())
    return {
        "balanced_accuracy": (
            float(balanced_accuracy) if balanced_accuracy is not None else None
        ),
        "prevalence_matched_score_threshold": score_threshold,
        "boundary_tie_positive_probability": tie_probability,
        "prevalence_matched_positive_count": predicted_count,
        "minority_class": minority_name,
        "minority_recall": (
            float(minority_correct / minority_count) if minority_count else None
        ),
        "minority_precision": (
            float(minority_correct / predicted_minority_count)
            if predicted_minority_count
            else None
        ),
    }


def _fit_prevalence_count(fit_prevalence: float, eval_count: int) -> int:
    if not math.isfinite(fit_prevalence) or not 0.0 <= fit_prevalence <= 1.0:
        raise ValueError("fit prevalence must be finite and lie in [0, 1]")
    return int(round(fit_prevalence * eval_count))


def _prevalence_matched_predictions(
    scores: np.ndarray,
    positive_count: int,
) -> tuple[np.ndarray, float | None, float]:
    count = len(scores)
    if positive_count == 0:
        score_threshold = None
        tie_probability = 0.0
        predicted_positive = np.zeros(count, dtype=float)
    elif positive_count == count:
        score_threshold = None
        tie_probability = 1.0
        predicted_positive = np.ones(count, dtype=float)
    else:
        score_threshold = float(np.sort(scores)[count - positive_count])
        above = scores > score_threshold
        tied = scores == score_threshold
        tie_probability = (positive_count - int(above.sum())) / int(tied.sum())
        predicted_positive = above.astype(float) + tie_probability * tied
    return predicted_positive, score_threshold, float(tie_probability)


def _auc_difference(
    labels: np.ndarray,
    model_scores: np.ndarray,
    baseline_scores: np.ndarray,
    task_ids: Sequence[str],
    *,
    seed: int,
) -> dict[str, Any]:
    model_auc = _auc(labels, model_scores)
    baseline_auc = _auc(labels, baseline_scores)
    if model_auc is None or baseline_auc is None:
        return {
            "point": None,
            "ci_low": None,
            "ci_high": None,
            "reason": "AUC requires both classes",
        }
    model_placements = _placement_values(labels, model_scores)
    baseline_placements = _placement_values(labels, baseline_scores)
    columns = np.column_stack(
        [
            model_placements * labels,
            baseline_placements * labels,
            labels,
            model_placements * ~labels,
            baseline_placements * ~labels,
            ~labels,
        ]
    )
    keys, contributions = _cluster_contributions(task_ids, columns)
    totals = resample_task_totals(
        contributions, replicates=_BOOTSTRAP_REPLICATES, seed=seed
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        model_draws = (
            totals[:, 0] / totals[:, 2] + totals[:, 3] / totals[:, 5] - model_auc
        )
        baseline_draws = (
            totals[:, 1] / totals[:, 2] + totals[:, 4] / totals[:, 5] - baseline_auc
        )
    differences = model_draws - baseline_draws
    finite = differences[np.isfinite(differences)]
    alpha = 1.0 - _CONFIDENCE_LEVEL
    low, high = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "point": float(model_auc - baseline_auc),
        "ci_low": float(low),
        "ci_high": float(high),
        "cluster_count": len(keys),
        "confidence_level": _CONFIDENCE_LEVEL,
        "bootstrap_replicates": _BOOTSTRAP_REPLICATES,
        "seed": seed,
        "method": "repo-clustered AUC influence bootstrap",
    }


def _balanced_accuracy_uncertainty(
    labels: np.ndarray,
    model_scores: np.ndarray,
    baseline_scores: np.ndarray,
    task_ids: Sequence[str],
    *,
    fit_prevalence: float,
    seed: int,
) -> dict[str, Any]:
    positive_count = int(labels.sum())
    negative_count = len(labels) - positive_count
    predicted_count = _fit_prevalence_count(fit_prevalence, len(labels))
    config = {
        "decision_cut_source": "fit_label_prevalence",
        "fit_prevalence": fit_prevalence,
        "predicted_positive_count": predicted_count,
    }
    if not positive_count or not negative_count:
        undefined = {
            "point": None,
            "ci_low": None,
            "ci_high": None,
            "reason": "balanced accuracy requires both classes",
        }
        return {
            **config,
            "model": undefined,
            "model_vs_baseline_difference": dict(undefined),
        }
    # The deployment cut is fixed once on the full eval score vectors. Bootstrap
    # draws resample these per-repo confusion counts; they never read draw labels
    # to choose a new operating point.
    model_positive, _, _ = _prevalence_matched_predictions(
        model_scores, predicted_count
    )
    baseline_positive, _, _ = _prevalence_matched_predictions(
        baseline_scores, predicted_count
    )
    positive = labels.astype(float)
    negative = (~labels).astype(float)
    columns = np.column_stack(
        [
            model_positive * positive,
            (1.0 - model_positive) * positive,
            (1.0 - model_positive) * negative,
            model_positive * negative,
            baseline_positive * positive,
            (1.0 - baseline_positive) * positive,
            (1.0 - baseline_positive) * negative,
            baseline_positive * negative,
        ]
    )
    keys, contributions = _cluster_contributions(task_ids, columns)
    totals = resample_task_totals(
        contributions, replicates=_BOOTSTRAP_REPLICATES, seed=seed
    )

    def balanced(values: np.ndarray, start: int) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return 0.5 * (
                values[:, start] / (values[:, start] + values[:, start + 1])
                + values[:, start + 2] / (values[:, start + 2] + values[:, start + 3])
            )

    model_draws = balanced(totals, 0)
    baseline_draws = balanced(totals, 4)
    difference_draws = model_draws - baseline_draws
    point_totals = contributions.sum(axis=0, keepdims=True)
    model_point = float(balanced(point_totals, 0)[0])
    baseline_point = float(balanced(point_totals, 4)[0])
    alpha = 1.0 - _CONFIDENCE_LEVEL

    def interval(draws: np.ndarray, point: float) -> dict[str, Any]:
        finite = draws[np.isfinite(draws)]
        low, high = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
        return {
            "point": point,
            "ci_low": float(low),
            "ci_high": float(high),
        }

    return {
        **config,
        "cluster_count": len(keys),
        "confidence_level": _CONFIDENCE_LEVEL,
        "bootstrap_replicates": _BOOTSTRAP_REPLICATES,
        "seed": seed,
        "method": "repo-clustered confusion-count bootstrap",
        "model": interval(model_draws, model_point),
        "model_vs_baseline_difference": interval(
            difference_draws, model_point - baseline_point
        ),
    }


def _thresholds(args: argparse.Namespace, target: str) -> list[float]:
    values = {
        "latency_ms": args.latency_thresholds_ms,
        "peak_memory_mb": args.mem_thresholds_mb,
        "peak_cpu_cores": args.cpu_thresholds_cores,
    }[target]
    normalized = sorted({float(value) for value in values})
    if any(not math.isfinite(value) or value < 0.0 for value in normalized):
        raise ValueError("thresholds must be finite and non-negative")
    return normalized


def _selected_targets(target: str) -> list[str]:
    return list(_TARGETS.values()) if target == "all" else [_TARGETS[target]]


def _scoring_indices(
    target: str,
    mask: np.ndarray,
    samples: Sequence[ResourceCallSample],
) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if target == "peak_memory_mb":
        indices = np.asarray(
            [
                index
                for index in indices
                if samples[index].ambient_before_mb is not None
            ],
            dtype=int,
        )
    return indices


def _target_result(
    target: str,
    fit_dataset: TabularDataset,
    eval_dataset: TabularDataset,
    fit_samples: Sequence[ResourceCallSample],
    eval_samples: Sequence[ResourceCallSample],
    fit_features: np.ndarray,
    eval_features: np.ndarray,
    log_predictions: np.ndarray,
    latency_prior: LatencyPrior,
    device: torch.device,
    epochs: int,
    args: argparse.Namespace,
    dump_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    fit_mask = fit_dataset.eligibility_masks[target]
    eval_mask = eval_dataset.eligibility_masks[target]
    fit_indices = _scoring_indices(target, fit_mask, fit_samples)
    indices = _scoring_indices(target, eval_mask, eval_samples)
    if not len(indices):
        raise ValueError(f"eval split has no baseline-eligible rows for {target}")

    fit_max = float(np.max(fit_dataset.targets[target][fit_mask]))
    if not math.isfinite(fit_max) or fit_max <= 0.0:
        raise ValueError(f"{target} fit maximum must be finite and positive")
    cap = _CLAMP_FACTOR * fit_max
    thresholds = _thresholds(args, target)
    scoring_samples = [eval_samples[index] for index in indices]
    baseline_predictions, baseline_scores, baseline_sources = _baseline_forecasts(
        target,
        fit_samples,
        scoring_samples,
        latency_prior,
        thresholds,
        cap,
    )

    selected_logs = np.maximum.accumulate(log_predictions[indices], axis=1)
    model_predictions = np.expm1(np.clip(selected_logs, 0.0, np.log1p(cap)))
    model_predictions, baseline_predictions = _clamp_prediction_pair(
        model_predictions, baseline_predictions, fit_max
    )
    observations = eval_dataset.targets[target][indices]
    task_ids = [str(eval_dataset.task_ids[index]) for index in indices]
    result: dict[str, Any] = {
        "unit": _UNITS[target],
        "eligible_fit_rows": int(fit_mask.sum()),
        "eligible_eval_rows": len(indices),
        "fit_max": fit_max,
        "prediction_cap": cap,
        "baseline_sources": baseline_sources,
        "model_vs_baseline_p90_pinball_skill": _p90_skill(
            observations,
            model_predictions[:, 1],
            baseline_predictions[:, 1],
            task_ids,
            seed=args.seed,
        ),
    }
    if args.mode in {"quantile", "both"}:
        result["quantile"] = {
            "model": _forecast_metrics(observations, model_predictions),
            "baseline": _forecast_metrics(observations, baseline_predictions),
        }
    if args.mode in {"binary", "both"}:
        interpolation_scores = quantile_exceedance(model_predictions, thresholds)
        fit_scoring_samples = [fit_samples[index] for index in fit_indices]
        fit_baseline_scores = _out_of_repo_baseline_scores(
            target, fit_samples, fit_scoring_samples, thresholds
        )
        binary = {"thresholds": {}}
        dumped_thresholds = (
            [{} for _ in range(len(indices))] if dump_rows is not None else None
        )
        for column, threshold in enumerate(thresholds):
            labels = observations > threshold
            fit_labels = fit_dataset.targets[target][fit_indices] > threshold
            fit_prevalence = float(fit_labels.mean())
            fit_baseline_z, eval_baseline_z, _, _ = _standardize_fit_eval(
                fit_baseline_scores[:, column, None],
                baseline_scores[:, column, None],
            )
            classifier = _train_binary_classifier(
                np.column_stack([fit_features[fit_indices], fit_baseline_z]),
                fit_labels,
                device,
                seed=args.seed + column,
                epochs=epochs,
            )
            classifier_scores = _predict_binary_probabilities(
                classifier,
                np.column_stack([eval_features[indices], eval_baseline_z]),
                device,
            )
            if dumped_thresholds is not None:
                predicted_count = _fit_prevalence_count(fit_prevalence, len(labels))
                model_labels, _, _ = _prevalence_matched_predictions(
                    classifier_scores, predicted_count
                )
                baseline_labels, _, _ = _prevalence_matched_predictions(
                    baseline_scores[:, column], predicted_count
                )
                for row, (
                    label,
                    model_probability,
                    baseline_survival,
                    model_label,
                    baseline_label,
                ) in zip(
                    dumped_thresholds,
                    zip(
                        labels,
                        classifier_scores,
                        baseline_scores[:, column],
                        model_labels,
                        baseline_labels,
                        strict=True,
                    ),
                    strict=True,
                ):
                    row[f"{threshold:g}"] = {
                        "model_probability": float(model_probability),
                        "baseline_survival": float(baseline_survival),
                        "binary_label": bool(label),
                        "model_predicted_label": float(model_label),
                        "baseline_predicted_label": float(baseline_label),
                    }
            binary["thresholds"][f"{threshold:g}"] = {
                "prevalence": float(labels.mean()),
                "fit_prevalence": fit_prevalence,
                "positive_count": int(labels.sum()),
                "row_count": len(labels),
                "model": {
                    "score_method": (
                        "dedicated inverse-prevalence-weighted linear logistic "
                        "head; raw sigmoid probability without value clamping"
                    ),
                    **_binary_metrics(labels, classifier_scores, fit_prevalence),
                },
                "quantile_interpolation": {
                    "score_method": (
                        "secondary piecewise-linear CDF interpolation over "
                        "p50/p90/p95; survival clipped to [0.05, 0.50]"
                    ),
                    **_binary_metrics(
                        labels,
                        interpolation_scores[:, column],
                        fit_prevalence,
                    ),
                },
                "baseline": {
                    "score_method": "fit-built ECDF survival probability",
                    **_binary_metrics(
                        labels, baseline_scores[:, column], fit_prevalence
                    ),
                },
                "model_vs_baseline_auc_difference": _auc_difference(
                    labels,
                    classifier_scores,
                    baseline_scores[:, column],
                    task_ids,
                    seed=args.seed,
                ),
                "balanced_accuracy_uncertainty": (
                    _balanced_accuracy_uncertainty(
                        labels,
                        classifier_scores,
                        baseline_scores[:, column],
                        task_ids,
                        fit_prevalence=fit_prevalence,
                        seed=args.seed,
                    )
                ),
            }
        result["binary"] = binary
        if dump_rows is not None:
            assert dumped_thresholds is not None
            model_p90 = _pinball_losses(observations, model_predictions[:, 1], 0.9)
            baseline_p90 = _pinball_losses(
                observations, baseline_predictions[:, 1], 0.9
            )
            quantile_names = ("p50", "p90", "p95")
            for row, index in enumerate(indices):
                sample = eval_samples[index]
                tool_args = sample.tool_args or {}
                dump_rows.append(
                    {
                        "target": target,
                        "sample_id": sample.sample_id,
                        "task_id": task_ids[row],
                        "repo": repo_cluster_key(task_ids[row]),
                        "tool_name": sample.tool_name,
                        "command": tool_args.get("command"),
                        "observed": float(observations[row]),
                        "model_quantile_predictions": dict(
                            zip(
                                quantile_names,
                                map(float, model_predictions[row]),
                                strict=True,
                            )
                        ),
                        "baseline_quantile_predictions": dict(
                            zip(
                                quantile_names,
                                map(float, baseline_predictions[row]),
                                strict=True,
                            )
                        ),
                        "thresholds": dumped_thresholds[row],
                        "pinball90": {
                            "model": float(model_p90[row]),
                            "baseline": float(baseline_p90[row]),
                        },
                        "features": {
                            name: float(eval_dataset.features[name][index])
                            for name in eval_dataset.feature_names
                        },
                    }
                )
    return result


def _write_dump_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, default=_DEFAULT_FIT)
    parser.add_argument("--eval", type=Path, default=_DEFAULT_EVAL)
    parser.add_argument("--fit-manifest", type=Path, default=_DEFAULT_FIT_MANIFEST)
    parser.add_argument("--eval-manifest", type=Path, default=_DEFAULT_EVAL_MANIFEST)
    parser.add_argument("--cost-table", type=Path)
    parser.add_argument(
        "--mode", choices=("quantile", "binary", "both"), default="both"
    )
    parser.add_argument(
        "--target",
        choices=("latency", "cpu_peak", "mem_peak", "all"),
        default="all",
    )
    parser.add_argument(
        "--latency-thresholds-ms",
        type=float,
        nargs="+",
        default=[3500.0, 5000.0],
    )
    parser.add_argument(
        "--mem-thresholds-mb",
        type=float,
        nargs="+",
        default=[500.0, 1000.0],
    )
    parser.add_argument(
        "--cpu-thresholds-cores",
        type=float,
        nargs="+",
        default=[2.0, 4.0],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dump-rows", type=Path)
    parser.add_argument(
        "--limit-tasks",
        type=int,
        help="Use the first N tasks from each corpus for a non-evidentiary smoke",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.limit_tasks is not None and args.limit_tasks < 1:
        raise ValueError("--limit-tasks must be positive")
    if args.dump_rows is not None and args.mode == "quantile":
        raise ValueError("--dump-rows requires --mode binary or both")
    cost_table = _load_cost_table(args.cost_table)
    fit_by_task, fit_task_ids = load_resource_corpus(
        args.fit, args.fit_manifest, limit_tasks=args.limit_tasks
    )
    eval_by_task, eval_task_ids = load_resource_corpus(
        args.eval, args.eval_manifest, limit_tasks=args.limit_tasks
    )
    overlap = sorted(set(fit_task_ids) & set(eval_task_ids))
    if overlap:
        raise ValueError(f"fit and eval task IDs overlap: {overlap}")
    fit_samples = _flatten(fit_by_task, fit_task_ids)
    eval_samples = _flatten(eval_by_task, eval_task_ids)
    fit_rows = [_sample_row(sample) for sample in fit_samples if not sample.censored]
    latency_prior = build_latency_prior(fit_rows, row_group_keys=_PREFIX_KEYS)
    latency_taus = _thresholds(args, "latency_ms")
    fit_dataset = _build_out_of_repo_fit_dataset(fit_by_task, cost_table, latency_taus)
    eval_dataset = build_tabular_dataset(
        eval_by_task, latency_prior, cost_table, latency_taus
    )
    if fit_dataset.feature_names != eval_dataset.feature_names:
        raise ValueError("fit and eval feature schemas differ")
    fit_samples = _samples_in_dataset_order(fit_dataset, fit_samples)
    eval_samples = _samples_in_dataset_order(eval_dataset, eval_samples)

    fit_features, eval_features, mean, std = _standardize_fit_eval(
        _feature_matrix(fit_dataset), _feature_matrix(eval_dataset)
    )
    targets = _selected_targets(args.target)
    epochs = _SMOKE_EPOCHS if args.limit_tasks is not None else _EPOCHS
    model, device = _train_quantile_model(
        fit_features,
        fit_dataset,
        targets,
        seed=args.seed,
        epochs=epochs,
    )
    log_predictions = _predict_log_quantiles(model, eval_features, device)
    dump_rows: list[dict[str, Any]] | None = [] if args.dump_rows is not None else None
    target_results = {
        alias: _target_result(
            canonical,
            fit_dataset,
            eval_dataset,
            fit_samples,
            eval_samples,
            fit_features,
            eval_features,
            log_predictions[canonical],
            latency_prior,
            device,
            epochs,
            args,
            dump_rows,
        )
        for alias, canonical in _TARGETS.items()
        if canonical in targets
    }
    result = {
        "status": (
            "SMOKE ONLY - limited tasks; metrics are plumbing checks, not findings"
            if args.limit_tasks is not None
            else "UNREGISTERED FULL EVALUATION"
        ),
        "config": {
            "fit": str(args.fit.resolve()),
            "eval": str(args.eval.resolve()),
            "fit_manifest": str(args.fit_manifest.resolve()),
            "eval_manifest": str(args.eval_manifest.resolve()),
            "cost_table": (
                str(args.cost_table.resolve()) if args.cost_table is not None else None
            ),
            "mode": args.mode,
            "target": args.target,
            "limit_tasks": args.limit_tasks,
            "seed": args.seed,
            "device": str(device),
            "quantiles": list(_QUANTILES),
            "epochs": epochs,
            "batch_size": _BATCH_SIZE,
            "hidden_dim": _HIDDEN_DIM,
            "learning_rate": _LEARNING_RATE,
            "prediction_clamp_fit_max_factor": _CLAMP_FACTOR,
            "bootstrap_replicates": _BOOTSTRAP_REPLICATES,
            "repo_cluster_key": "task_id with trailing -<digits> stripped",
            "fit_prior_features": "leave-one-repository-out",
            "binary_classifier": {
                "architecture": "single linear logistic head per target threshold",
                "loss": "inverse-prevalence-weighted BCE",
                "features": (
                    "standardized tabular features plus standardized "
                    "fit-built baseline ECDF survival"
                ),
                "optimizer": "AdamW with cosine annealing and gradient clip 1.0",
                "probability_clamp": None,
            },
        },
        "data": {
            "fit_task_count": len(fit_task_ids),
            "eval_task_count": len(eval_task_ids),
            "fit_row_count": len(fit_dataset.sample_ids),
            "eval_row_count": len(eval_dataset.sample_ids),
            "feature_count": len(fit_dataset.feature_names),
            "feature_names": list(fit_dataset.feature_names),
            "scaler": {
                "fit_only": True,
                "mean": mean.tolist(),
                "std": std.tolist(),
            },
        },
        "targets": target_results,
        "registered_criterion": {
            "definition": (
                "not registered by this harness; register a criterion at run time "
                "before reading full-evaluation metrics"
            ),
            "evaluated": False,
            "passed": None,
        },
    }
    if args.dump_rows is not None:
        assert dump_rows is not None
        _write_dump_rows(args.dump_rows, dump_rows)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
