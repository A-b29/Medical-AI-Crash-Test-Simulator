"""A fixed baseline estimator behind a small inference interface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class PredictionBatch:
    labels: np.ndarray
    probabilities: np.ndarray
    confidence: np.ndarray


def train_baseline(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> Pipeline:
    """Fit preprocessing on training data only to prevent split leakage."""

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(max_iter=2_000, random_state=seed)),
        ]
    )
    return model.fit(X_train, y_train)


def predict(model: Pipeline, X: np.ndarray) -> PredictionBatch:
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    return PredictionBatch(
        labels=np.asarray(model.predict(X), dtype=int),
        probabilities=probabilities,
        confidence=probabilities.max(axis=1),
    )
