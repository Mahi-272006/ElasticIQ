"""Baseline forecasters shared by training evaluation and the API.

These are matrix-based implementations (one row per series, one column per
day) used to benchmark ML models on identical series and validation dates.
The historical experiment scripts (``baseline.py``, ``seasonal_baseline.py``)
are kept unchanged for reproducibility of the original recorded results.
"""

from __future__ import annotations

import numpy as np


def mean_baseline(history: np.ndarray, horizon: int, lookback: int = 28) -> np.ndarray:
    """Repeat each series' mean over its last ``lookback`` observed days."""

    recent = history[:, -lookback:]
    return np.repeat(recent.mean(axis=1, keepdims=True), horizon, axis=1)


def last_day_naive(history: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat each series' last observed day across the horizon."""

    return np.repeat(history[:, -1:], horizon, axis=1)


def weekly_seasonal_naive(
    history: np.ndarray, horizon: int, period: int = 7
) -> np.ndarray:
    """Repeat the last ``period`` observed days cyclically across the horizon."""

    last_week = history[:, -period:]
    repeats = (horizon + period - 1) // period
    return np.tile(last_week, (1, repeats))[:, :horizon]
