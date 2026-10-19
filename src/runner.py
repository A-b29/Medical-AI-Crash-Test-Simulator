"""One reproducible command for the baseline, perturbation, and confidence suite."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import platform
import sys

import numpy as np
import sklearn

from .abstention import select_threshold_on_validation, selective_metrics
from .data_adapter import load_demo_dataset
from .metrics import classification_metrics, reliability_curve, stability_metrics
from .model_adapter import predict, train_baseline
from .perturbations import attenuation, feature_dropout, gaussian_noise


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    severities: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75)
    risk_budget: float = 0.05


def run_experiment(config: ExperimentConfig = ExperimentConfig()) -> dict[str, object]:
    """Run every comparison with a shared fixed test set and record the contract."""

    splits = load_demo_dataset(config.seed)
    model = train_baseline(splits.X_train, splits.y_train, config.seed)
    baseline_test = predict(model, splits.X_test)
    baseline_val = predict(model, splits.X_val)
    training_std = splits.X_train.std(axis=0, ddof=0)
    training_median = np.median(splits.X_train, axis=0)

    rows: list[dict[str, object]] = []
    transforms = {
        "gaussian_noise": lambda severity: gaussian_noise(splits.X_test, severity, training_std, config.seed + 101),
        "attenuation": lambda severity: attenuation(splits.X_test, severity, training_median),
        "feature_dropout": lambda severity: feature_dropout(splits.X_test, severity, training_median, config.seed + 202),
    }
    for name, transform in transforms.items():
        for severity in config.severities:
            perturbed = predict(model, transform(float(severity)))
            row: dict[str, object] = {"perturbation": name, "severity": float(severity)}
            row.update(classification_metrics(splits.y_test, perturbed))
            row.update(stability_metrics(baseline_test, perturbed))
            rows.append(row)

    validation_policy = select_threshold_on_validation(
        splits.y_val, baseline_val.labels, baseline_val.confidence, config.risk_budget
    )
    test_policy = selective_metrics(
        splits.y_test, baseline_test.labels, baseline_test.confidence, float(validation_policy["threshold"])
    )
    calibration = reliability_curve(
        baseline_test.confidence,
        (baseline_test.labels == splits.y_test).astype(int),
    )
    return {
        "project_notice": "Research software demonstration; not a clinical safety certificate or diagnostic tool.",
        "config": asdict(config),
        "manifest": {
            "dataset_id": splits.dataset_id,
            "dataset_samples": {"train": len(splits.y_train), "validation": len(splits.y_val), "test": len(splits.y_test)},
            "model_id": "StandardScaler + LogisticRegression",
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "baseline_metrics": classification_metrics(splits.y_test, baseline_test),
        "perturbation_results": rows,
        "calibration": calibration,
        "abstention": {"selection_split": "validation", "risk_budget": config.risk_budget, "validation_policy": validation_policy, "test_evaluation": test_policy},
    }
