"""Validation-selected confidence abstention policy."""

from __future__ import annotations

import numpy as np


def selective_metrics(y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray, threshold: float) -> dict[str, float | int]:
    accepted = np.asarray(confidence, dtype=float) >= threshold
    count = int(accepted.sum())
    if count == 0:
        return {"threshold": float(threshold), "accepted_count": 0, "coverage": 0.0, "selective_risk": float("nan")}
    risk = float((np.asarray(y_true)[accepted] != np.asarray(y_pred)[accepted]).mean())
    return {"threshold": float(threshold), "accepted_count": count, "coverage": float(accepted.mean()), "selective_risk": risk}


def select_threshold_on_validation(y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray, risk_budget: float) -> dict[str, float | int]:
    """Return the highest-coverage threshold meeting the risk budget, validation only."""

    if not 0.0 <= risk_budget <= 1.0:
        raise ValueError("risk_budget must be in [0, 1]")
    candidates = np.unique(np.r_[0.0, np.asarray(confidence, dtype=float), 1.0])
    eligible = [selective_metrics(y_true, y_pred, confidence, float(value)) for value in candidates]
    eligible = [result for result in eligible if result["accepted_count"] and result["selective_risk"] <= risk_budget]
    if not eligible:
        return {"threshold": 1.0, "accepted_count": 0, "coverage": 0.0, "selective_risk": float("nan")}
    return max(eligible, key=lambda item: (item["coverage"], -item["threshold"]))
