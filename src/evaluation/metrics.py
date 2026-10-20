"""Forecast accuracy metrics shared across the project.

All metrics handle zero denominators explicitly instead of producing silent
NaNs or divide-by-zero warnings.
"""

from __future__ import annotations

import numpy as np


def _as_arrays(actual, predicted) -> tuple[np.ndarray, np.ndarray]:
    actual_arr = np.asarray(actual, dtype=np.float64)
    predicted_arr = np.asarray(predicted, dtype=np.float64)

    if actual_arr.shape != predicted_arr.shape:
        raise ValueError(
            "Shape mismatch: "
            f"actual {actual_arr.shape} vs predicted {predicted_arr.shape}"
        )

    return actual_arr, predicted_arr


def wape(actual, predicted) -> float:
    """Weighted Absolute Percentage Error, expressed as a percentage.

    WAPE = sum(|actual - predicted|) / sum(|actual|) * 100

    If the actuals sum to zero, WAPE is 0.0 when the predictions are also
    zero (perfect match on an all-zero series) and infinity otherwise.
    """

    actual_arr, predicted_arr = _as_arrays(actual, predicted)
    denominator = np.abs(actual_arr).sum()

    if denominator == 0:
        return 0.0 if np.abs(predicted_arr).sum() == 0 else float("inf")

    return float(np.abs(actual_arr - predicted_arr).sum() / denominator * 100)


def mae(actual, predicted) -> float:
    """Mean Absolute Error in units."""

    actual_arr, predicted_arr = _as_arrays(actual, predicted)

    if actual_arr.size == 0:
        return float("nan")

    return float(np.abs(actual_arr - predicted_arr).mean())


def rmse(actual, predicted) -> float:
    """Root Mean Squared Error in units."""

    actual_arr, predicted_arr = _as_arrays(actual, predicted)

    if actual_arr.size == 0:
        return float("nan")

    return float(np.sqrt(((actual_arr - predicted_arr) ** 2).mean()))


def rmsse(actual, predicted, naive_errors: np.ndarray) -> float:
    """Root Mean Squared Scaled Error.

    RMSSE = RMSE(forecast) / RMSE(naive one-step-ahead forecast), where the
    naive errors are the in-sample first differences of the series. A value
    below 1.0 means the model beats the seasonal/random-walk naive benchmark.

    ``naive_errors`` must come from the *training* window only; passing
    validation-period differences would leak information.
    """

    actual_arr, predicted_arr = _as_arrays(actual, predicted)
    naive_arr = np.asarray(naive_errors, dtype=np.float64)

    if naive_arr.size == 0:
        return float("nan")

    denominator = np.sqrt((naive_arr**2).mean())
    if denominator == 0:
        # Series is constant in-sample: any deviation is infinitely worse.
        return 0.0 if rmse(actual_arr, predicted_arr) == 0 else float("inf")

    return rmse(actual_arr, predicted_arr) / denominator
