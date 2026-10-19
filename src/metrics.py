"""Task, stability, and calibration metrics with defined edge-case behaviour."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score

from .model_adapter import PredictionBatch


def classification_metrics(y_true: np.ndarray, predictions: PredictionBatch) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    positive_probability = predictions.probabilities[:, 1]
    return {
        "accuracy": float(accuracy_score(y, predictions.labels)),
        "f1": float(f1_score(y, predictions.labels, zero_division=0)),
        "auroc": float(roc_auc_score(y, positive_probability)) if len(np.unique(y)) == 2 else float("nan"),
        "brier_score": float(brier_score_loss(y, positive_probability)),
    }


def stability_metrics(baseline: PredictionBatch, perturbed: PredictionBatch) -> dict[str, float]:
    if baseline.labels.shape != perturbed.labels.shape:
        raise ValueError("baseline and perturbed predictions must have matching sample counts")
    stable = baseline.labels == perturbed.labels
    baseline_chosen = baseline.probabilities[np.arange(len(stable)), baseline.labels]
    perturbed_for_baseline = perturbed.probabilities[np.arange(len(stable)), baseline.labels]
    return {
        "prediction_stability": float(stable.mean()),
        "prediction_flip_rate": float((~stable).mean()),
        "mean_baseline_class_probability_delta": float((perturbed_for_baseline - baseline_chosen).mean()),
    }


def reliability_curve(confidence: np.ndarray, correctness: np.ndarray, bins: int = 10) -> dict[str, object]:
    confidences = np.asarray(confidence, dtype=float)
    correct = np.asarray(correctness, dtype=float)
    if confidences.shape != correct.shape:
        raise ValueError("confidence and correctness must have matching shapes")
    if bins < 1:
        raise ValueError("bins must be positive")
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, float | int]] = []
    ece = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidences >= lower) & ((confidences < upper) if index < bins - 1 else (confidences <= upper))
        count = int(mask.sum())
        if count == 0:
            continue
        mean_confidence = float(confidences[mask].mean())
        accuracy = float(correct[mask].mean())
        ece += abs(accuracy - mean_confidence) * count / len(confidences)
        rows.append({"lower": float(lower), "upper": float(upper), "count": count, "mean_confidence": mean_confidence, "accuracy": accuracy})
    return {"bins": rows, "expected_calibration_error": float(ece)}
