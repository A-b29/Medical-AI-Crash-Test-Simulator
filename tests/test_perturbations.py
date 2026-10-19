import numpy as np
import pytest

from src.perturbations import attenuation, feature_dropout, gaussian_noise


@pytest.fixture
def fixture_data():
    return np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([2.0, 5.0]), np.array([2.0, 3.0])


def test_zero_severity_is_identity_for_all_stressors(fixture_data):
    X, std, median = fixture_data
    assert np.array_equal(gaussian_noise(X, 0.0, std, seed=1), X)
    assert np.array_equal(attenuation(X, 0.0, median), X)
    assert np.array_equal(feature_dropout(X, 0.0, median, seed=1), X)


def test_dropout_is_seeded_and_severity_is_valid(fixture_data):
    X, _, median = fixture_data
    assert np.array_equal(feature_dropout(X, 0.5, median, seed=7), feature_dropout(X, 0.5, median, seed=7))
    with pytest.raises(ValueError):
        attenuation(X, 1.1, median)
