"""Deterministic tabular stressors with explicit zero-severity behaviour."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _validate_severity(severity: float) -> None:
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must be in [0, 1]")


def gaussian_noise(X: np.ndarray, severity: float, reference_std: np.ndarray, seed: int) -> np.ndarray:
    """Add zero-mean noise scaled to each training feature's standard deviation."""

    _validate_severity(severity)
    values = np.asarray(X, dtype=float)
    scale = np.asarray(reference_std, dtype=float)
    if values.ndim != 2 or scale.shape != (values.shape[1],):
        raise ValueError("reference_std must contain one value per feature")
    rng = np.random.default_rng(seed)
    return values + rng.normal(0.0, scale * severity, size=values.shape)


def attenuation(X: np.ndarray, severity: float, reference_median: np.ndarray) -> np.ndarray:
    """Shrink every feature toward its training median; zero is identity."""

    _validate_severity(severity)
    values = np.asarray(X, dtype=float)
    median = np.asarray(reference_median, dtype=float)
    if values.ndim != 2 or median.shape != (values.shape[1],):
        raise ValueError("reference_median must contain one value per feature")
    return values + severity * (median - values)


def feature_dropout(X: np.ndarray, severity: float, reference_median: np.ndarray, seed: int) -> np.ndarray:
    """Randomly replace features with their training median at a fixed rate."""

    _validate_severity(severity)
    values = np.asarray(X, dtype=float)
    median = np.asarray(reference_median, dtype=float)
    if values.ndim != 2 or median.shape != (values.shape[1],):
        raise ValueError("reference_median must contain one value per feature")
    rng = np.random.default_rng(seed)
    mask = rng.random(values.shape) < severity
    return np.where(mask, median[None, :], values)


PERTURBATIONS: dict[str, Callable[..., np.ndarray]] = {
    "gaussian_noise": gaussian_noise,
    "attenuation": attenuation,
    "feature_dropout": feature_dropout,
}
