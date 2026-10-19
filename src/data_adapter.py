"""Dataset adapter with a fixed train/validation/test protocol."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    dataset_id: str = "sklearn_breast_cancer"


def load_demo_dataset(seed: int = 42) -> DatasetSplits:
    """Load the scikit-learn educational dataset with stratified fixed splits.

    The dataset is used only to demonstrate an evaluation pipeline; it is not
    appropriate for diagnosis, patient-level decisions, or clinical claims.
    """

    data = load_breast_cancer()
    X_train, X_temp, y_train, y_temp = train_test_split(
        data.data, data.target, test_size=0.4, random_state=seed, stratify=data.target
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )
    return DatasetSplits(
        X_train=np.asarray(X_train, dtype=float),
        y_train=np.asarray(y_train, dtype=int),
        X_val=np.asarray(X_val, dtype=float),
        y_val=np.asarray(y_val, dtype=int),
        X_test=np.asarray(X_test, dtype=float),
        y_test=np.asarray(y_test, dtype=int),
        feature_names=[str(name) for name in data.feature_names],
    )
