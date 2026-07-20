"""Shared binary classification metric helpers for latency evaluations."""

from __future__ import annotations

from typing import Iterable


def safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def binary_classification_metrics(
    pairs: Iterable[tuple[bool, bool]],
) -> dict[str, float | int | None]:
    """Confusion counts and ratio metrics for (label, predicted) pairs."""

    tp = fp = tn = fn = 0
    for label, predicted in pairs:
        if predicted and label:
            tp += 1
        elif predicted:
            fp += 1
        elif label:
            fn += 1
        else:
            tn += 1
    total = tp + fp + tn + fn
    return {
        "true_positive_count": tp,
        "false_positive_count": fp,
        "true_negative_count": tn,
        "false_negative_count": fn,
        "accuracy": safe_div(tp + tn, total),
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "false_positive_rate": safe_div(fp, fp + tn),
        "false_negative_rate": safe_div(fn, fn + tp),
    }


__all__ = ["binary_classification_metrics", "safe_div"]
