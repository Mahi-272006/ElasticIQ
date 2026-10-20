"""Unit tests for the shared forecast metrics."""

import math

import numpy as np
import pytest

from src.evaluation.metrics import mae, rmse, rmsse, wape


def test_wape_matches_hand_computed_value():
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([12.0, 18.0, 33.0])

    # |10-12| + |20-18| + |30-33| = 7; actual sum = 60 -> 7/60 * 100
    assert wape(actual, predicted) == pytest.approx(7 / 60 * 100)


def test_wape_zero_actuals_zero_predicted_is_zero():
    zeros = np.zeros(5)
    assert wape(zeros, zeros) == 0.0


def test_wape_zero_actuals_nonzero_predicted_is_infinite():
    zeros = np.zeros(5)
    assert math.isinf(wape(zeros, np.ones(5)))


def test_wape_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        wape(np.zeros(3), np.zeros(4))


def test_mae_hand_computed():
    actual = np.array([1.0, 2.0, 3.0])
    predicted = np.array([2.0, 2.0, 5.0])
    assert mae(actual, predicted) == pytest.approx((1 + 0 + 2) / 3)


def test_rmse_hand_computed():
    actual = np.array([0.0, 0.0])
    predicted = np.array([3.0, 4.0])
    # sqrt((9 + 16) / 2) = sqrt(12.5)
    assert rmse(actual, predicted) == pytest.approx(math.sqrt(12.5))


def test_rmsse_perfect_model_below_one():
    # Model error (0.5) is much smaller than naive differences (~4).
    actual = np.array([10.0, 10.0, 10.0])
    predicted = np.array([9.5, 10.5, 10.0])
    naive_errors = np.array([4.0, -5.0, 3.0, 6.0])

    assert rmsse(actual, predicted, naive_errors) < 1.0


def test_rmsse_constant_series_edge_case():
    actual = np.array([5.0, 5.0])
    predicted = np.array([5.0, 5.0])
    naive_errors = np.zeros(10)

    assert rmsse(actual, predicted, naive_errors) == 0.0

    assert math.isinf(rmsse(actual, predicted + 1, naive_errors))
