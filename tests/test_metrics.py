import numpy as np
import pytest

from src.abstention import select_threshold_on_validation, selective_metrics
from src.metrics import reliability_curve
from src.model_adapter import PredictionBatch


def test_selective_metrics_handles_empty_acceptance():
    result = selective_metrics(np.array([0, 1]), np.array([0, 0]), np.array([0.2, 0.4]), 0.9)
    assert result["coverage"] == 0.0
    assert np.isnan(result["selective_risk"])


def test_validation_threshold_maximizes_coverage_under_budget():
    selected = select_threshold_on_validation(
        np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]), np.array([0.95, 0.9, 0.4, 0.35]), 0.0
    )
    assert selected["coverage"] == 0.5
    assert selected["selective_risk"] == 0.0


def test_reliability_curve_reports_weighted_ece():
    curve = reliability_curve(np.array([0.8, 0.8]), np.array([1, 0]), bins=2)
    assert curve["expected_calibration_error"] == pytest.approx(0.3)
